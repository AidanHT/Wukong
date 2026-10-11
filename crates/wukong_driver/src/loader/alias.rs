//! Type-alias expansion: `type Real = f32;` and `type Mat<M, N> = Tensor[f32, M, N];`.
//!
//! Runs once on the merged program, right after [`super::load_imports`] has renamed every item
//! (an imported alias included) to its program-wide symbol and rewritten every reference to it.
//! The pass
//!
//! 1. collects the `type` items and removes them from the module;
//! 2. expands each alias body in dependency order, reporting a cycle (E0326) and a body that
//!    names no type (E0328);
//! 3. replaces every use in a type position (parameters, returns, `let` annotations, struct and
//!    variant fields, casts, `alloc::<T>` and other turbofish arguments, `size_of::<T>()`, nested
//!    inside other types) with a fresh copy of the expanded body, its parameters substituted by
//!    the use's arguments, reporting a wrong number or kind of argument (E0327).
//!
//! An alias is transparent: after this pass no item, type or annotation mentions one, so sema,
//! shape checking, MIR lowering (`mir_ty_of_ann`, `annotation_storage_ty`) and autodiff see only
//! the types it stands for. Every copy gets fresh `NodeId`s: sema keys its tables by node, and two
//! uses of one alias must never share an array-length expression's entry.

use wukong_ast::{
    Block, Dim, DimKind, Expr, ExprKind, FnDecl, ForIter, GenericParam, GenericParamKind, Ident,
    Item, ItemKind, Module, NodeId, Path, Stmt, StmtKind, TypeArg, TypeExpr, TypeKind, VariantData,
};
use wukong_diag::{Diagnostic, DiagnosticSink};
use wukong_span::{FxHashMap, FxHashSet, Interner, Span, Symbol};
use wukong_types::is_builtin_type_name;

/// One alias, its body expanded in place once [`Expander::resolve`] has visited it.
struct Alias {
    name: Ident,
    params: Vec<Symbol>,
    ty: TypeExpr,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum State {
    Pending,
    Visiting,
    Done,
}

/// Expand every type alias of `module` and remove the alias items. Diagnostics go to `sink`;
/// `next_node` is the program's `NodeId` watermark, advanced past every node the pass creates.
pub(crate) fn expand_type_aliases(
    module: &mut Module,
    interner: &mut Interner,
    sink: &mut DiagnosticSink,
    next_node: &mut u32,
) {
    if !module
        .items
        .iter()
        .any(|it| matches!(it.kind, ItemKind::TypeAlias(_)))
    {
        return;
    }
    let mut types: FxHashSet<Symbol> = FxHashSet::default();
    let mut defined: FxHashMap<Symbol, Span> = FxHashMap::default();
    for item in &module.items {
        match &item.kind {
            ItemKind::Struct(s) => {
                types.insert(s.name.sym);
                defined.entry(s.name.sym).or_insert(s.name.span);
            }
            ItemKind::Enum(e) => {
                types.insert(e.name.sym);
                defined.entry(e.name.sym).or_insert(e.name.span);
            }
            _ => {}
        }
    }
    let mut aliases = Vec::new();
    let mut by_name: FxHashMap<Symbol, usize> = FxHashMap::default();
    let mut rest = Vec::with_capacity(module.items.len());
    for item in std::mem::take(&mut module.items) {
        let ItemKind::TypeAlias(decl) = item.kind else {
            rest.push(item);
            continue;
        };
        let name = decl.name;
        let shown = interner.resolve(name.sym).to_string();
        if let Some(&first) = defined.get(&name.sym) {
            sink.emit(
                Diagnostic::error(format!("the type `{shown}` is defined more than once"))
                    .with_code("E0300")
                    .primary(name.span, "redefined here")
                    .secondary(first, "first defined here"),
            );
            continue;
        }
        // A scalar name or `str`: sema reads the name as the built-in type in every type position.
        // Compared on the name the module WROTE, the last segment: an imported module's alias
        // arrives renamed `<import.path>.<name>` (`resolve.rs`), and `lib.m.str` compared whole
        // is no built-in name, so `type str = i32;` there made every `str` in it an `i32`.
        let written = shown.rsplit('.').next().unwrap_or(&shown);
        if is_builtin_type_name(written) {
            sink.emit(
                Diagnostic::error(format!("a type alias cannot be named `{written}`"))
                    .with_code("E0300")
                    .primary(name.span, format!("`{written}` is a built-in type"))
                    .help("choose another name for the alias"),
            );
            continue;
        }
        defined.insert(name.sym, name.span);
        by_name.insert(name.sym, aliases.len());
        aliases.push(Alias {
            name,
            params: decl.generics.iter().map(generic_sym).collect(),
            ty: decl.ty,
        });
    }
    module.items = rest;

    let mut ex = Expander {
        state: vec![State::Pending; aliases.len()],
        cyclic: vec![false; aliases.len()],
        aliases,
        by_name,
        types,
        interner,
        sink,
        next_node,
        generics: Vec::new(),
        in_alias: false,
        stack: Vec::new(),
    };
    for i in 0..ex.aliases.len() {
        ex.resolve(i);
    }
    for item in &mut module.items {
        ex.item(item);
    }
}

fn generic_sym(g: &GenericParam) -> Symbol {
    match &g.kind {
        GenericParamKind::Type(id) => id.sym,
        GenericParamKind::Const { name, .. } => name.sym,
    }
}

fn unit(span: Span) -> TypeExpr {
    TypeExpr {
        id: NodeId::DUMMY,
        kind: TypeKind::Unit,
        span,
    }
}

struct Expander<'a> {
    aliases: Vec<Alias>,
    by_name: FxHashMap<Symbol, usize>,
    state: Vec<State>,
    /// Aliases already reported as part of a cycle, so one cycle is one diagnostic.
    cyclic: Vec<bool>,
    /// Struct and enum names: the program's own named types.
    types: FxHashSet<Symbol>,
    interner: &'a mut Interner,
    sink: &'a mut DiagnosticSink,
    next_node: &'a mut u32,
    /// The generic parameters in scope; one shadows an alias of the same name.
    generics: Vec<Symbol>,
    /// Walking an alias body, where a name that is no type is E0328 (a program type position
    /// leaves that to sema's E0324, which also sees generics and traits).
    in_alias: bool,
    /// The aliases whose bodies are being expanded, outermost first.
    stack: Vec<usize>,
}

impl Expander<'_> {
    fn fresh(&mut self) -> NodeId {
        let id = NodeId(*self.next_node);
        *self.next_node += 1;
        id
    }

    fn name(&self, s: Symbol) -> String {
        self.interner.resolve(s).to_string()
    }

    /// Expand alias `i`'s body, after every alias it names.
    fn resolve(&mut self, i: usize) {
        if self.state[i] != State::Pending {
            return;
        }
        self.state[i] = State::Visiting;
        self.stack.push(i);
        let saved_generics = std::mem::replace(&mut self.generics, self.aliases[i].params.clone());
        let saved_in_alias = std::mem::replace(&mut self.in_alias, true);
        let placeholder = unit(self.aliases[i].name.span);
        let mut body = std::mem::replace(&mut self.aliases[i].ty, placeholder);
        self.ty(&mut body);
        self.aliases[i].ty = body;
        self.generics = saved_generics;
        self.in_alias = saved_in_alias;
        self.stack.pop();
        self.state[i] = State::Done;
    }

    /// Is `sym` a type an alias body may name?
    fn is_type_name(&self, sym: Symbol) -> bool {
        self.generics.contains(&sym)
            || self.types.contains(&sym)
            || self.by_name.contains_key(&sym)
            || is_builtin_type_name(self.interner.resolve(sym))
    }

    /// The alias a type path names here, if any.
    fn alias_of(&self, p: &Path) -> Option<usize> {
        let sym = p.segments.last()?.sym;
        if p.is_single() && self.generics.contains(&sym) {
            return None;
        }
        if !p.is_single() {
            return None;
        }
        self.by_name.get(&sym).copied()
    }

    fn unknown_target(&mut self, p: &Path, span: Span) {
        let shown = p
            .segments
            .iter()
            .map(|s| self.name(s.sym))
            .collect::<Vec<_>>()
            .join("::");
        let alias = self
            .stack
            .last()
            .map(|&i| self.name(self.aliases[i].name.sym));
        let mut d = Diagnostic::error(format!("cannot find type `{shown}` in this type alias"))
            .with_code("E0328")
            .primary(span, "not a type");
        if let Some(a) = alias {
            d = d.note(format!(
                "the alias `{a}` must stand for a scalar, a struct or enum of the program, \
                 another alias, one of its own parameters, or a type built from those"
            ));
        }
        self.sink.emit(d);
    }

    /// A fresh, expanded copy of alias `j` applied to `args` at `span`.
    fn apply(&mut self, j: usize, args: &[TypeArg], span: Span) -> TypeExpr {
        if self.state[j] == State::Visiting {
            self.cycle(j, span);
            return unit(span);
        }
        self.resolve(j);
        if self.cyclic[j] {
            return unit(span);
        }
        let want = self.aliases[j].params.len();
        if want != args.len() {
            let nm = self.name(self.aliases[j].name.sym);
            let plural = |n: usize| if n == 1 { "argument" } else { "arguments" };
            let d = Diagnostic::error(format!(
                "type alias `{nm}` takes {want} {} but {} {} supplied",
                plural(want),
                args.len(),
                if args.len() == 1 { "was" } else { "were" }
            ))
            .with_code("E0327")
            .primary(span, format!("expected {want} {}", plural(want)))
            .secondary(self.aliases[j].name.span, "alias defined here");
            self.sink.emit(d);
            return unit(span);
        }
        let mut out = self.aliases[j].ty.clone();
        if want > 0 {
            let map: FxHashMap<Symbol, &TypeArg> = self.aliases[j]
                .params
                .iter()
                .copied()
                .zip(args.iter())
                .collect();
            let nm = self.name(self.aliases[j].name.sym);
            let mut sub = Subst {
                map,
                alias: nm,
                interner: &mut *self.interner,
                sink: &mut *self.sink,
            };
            sub.ty(&mut out);
        }
        self.renumber_ty(&mut out);
        out.span = span;
        out
    }

    fn cycle(&mut self, j: usize, use_span: Span) {
        let pos = self.stack.iter().position(|&k| k == j).unwrap_or(0);
        let members: Vec<usize> = self.stack[pos..].to_vec();
        if members.iter().any(|&k| self.cyclic[k]) {
            return;
        }
        for &k in &members {
            self.cyclic[k] = true;
        }
        let mut chain: Vec<String> = members
            .iter()
            .map(|&k| format!("`{}`", self.name(self.aliases[k].name.sym)))
            .collect();
        chain.push(format!("`{}`", self.name(self.aliases[j].name.sym)));
        let nm = self.name(self.aliases[j].name.sym);
        self.sink.emit(
            Diagnostic::error(format!("type alias `{nm}` refers to itself"))
                .with_code("E0326")
                .primary(
                    self.aliases[j].name.span,
                    "this alias is defined in terms of itself",
                )
                .secondary(use_span, "the cycle closes here")
                .note(format!("expanding it never ends: {}", chain.join(" -> ")))
                .help(
                    "an alias is replaced by its definition; to describe a recursive type, use a \
                     struct or enum that holds a pointer or slice to itself",
                ),
        );
    }

    fn renumber_ty(&mut self, t: &mut TypeExpr) {
        t.id = self.fresh();
        match &mut t.kind {
            TypeKind::Path(_) | TypeKind::Int(_) | TypeKind::Unit => {}
            TypeKind::Pointer { pointee, .. } | TypeKind::Ref { pointee, .. } => {
                self.renumber_ty(pointee)
            }
            TypeKind::Slice(elem) | TypeKind::Vector { elem, .. } => self.renumber_ty(elem),
            TypeKind::Array { elem, len, .. } => {
                self.renumber_ty(elem);
                self.renumber_expr(len);
            }
            TypeKind::Tuple(elems) => {
                for e in elems {
                    self.renumber_ty(e);
                }
            }
            TypeKind::Tensor { elem, .. } => self.renumber_ty(elem),
            TypeKind::Fn { params, ret } => {
                for p in params {
                    self.renumber_ty(p);
                }
                if let Some(r) = ret {
                    self.renumber_ty(r);
                }
            }
            TypeKind::Generic { args, .. } => {
                for a in args {
                    if let TypeArg::Type(t) = a {
                        self.renumber_ty(t);
                    }
                }
            }
        }
    }

    /// Fresh ids for an array-length expression copied out of an alias body. A length is a
    /// constant expression; the statement-bearing forms (a block, `if`, `match`, `loop`) are not
    /// constants sema accepts, so only their own node is renumbered.
    fn renumber_expr(&mut self, e: &mut Expr) {
        e.id = self.fresh();
        match &mut e.kind {
            ExprKind::Unary { expr, .. } => self.renumber_expr(expr),
            ExprKind::Binary { lhs, rhs, .. } => {
                self.renumber_expr(lhs);
                self.renumber_expr(rhs);
            }
            ExprKind::Call {
                callee,
                generic_args,
                args,
            } => {
                self.renumber_expr(callee);
                for t in generic_args {
                    self.renumber_ty(t);
                }
                for a in args {
                    self.renumber_expr(a);
                }
            }
            ExprKind::Cast { expr, ty } => {
                self.renumber_expr(expr);
                self.renumber_ty(ty);
            }
            ExprKind::Index { base, indices } => {
                self.renumber_expr(base);
                for i in indices {
                    self.renumber_expr(i);
                }
            }
            ExprKind::Field { base, .. } | ExprKind::TupleField { base, .. } => {
                self.renumber_expr(base)
            }
            ExprKind::ArrayLit(xs) | ExprKind::TupleLit(xs) => {
                for x in xs {
                    self.renumber_expr(x);
                }
            }
            ExprKind::ArrayRepeat { value, count } => {
                self.renumber_expr(value);
                self.renumber_expr(count);
            }
            ExprKind::SizeOf(t) | ExprKind::AlignOf(t) => self.renumber_ty(t),
            _ => {}
        }
    }

    // ---- The program walk ----

    fn item(&mut self, item: &mut Item) {
        match &mut item.kind {
            ItemKind::Fn(f) => self.func(f),
            ItemKind::Struct(s) => {
                self.generics = s.generics.iter().map(generic_sym).collect();
                self.generic_params(&mut s.generics);
                for fld in &mut s.fields {
                    self.ty(&mut fld.ty);
                }
                self.generics.clear();
            }
            ItemKind::Enum(e) => {
                self.generics = e.generics.iter().map(generic_sym).collect();
                self.generic_params(&mut e.generics);
                for v in &mut e.variants {
                    match &mut v.data {
                        VariantData::Unit => {}
                        VariantData::Tuple(tys) => {
                            for t in tys {
                                self.ty(t);
                            }
                        }
                        VariantData::Struct(fields) => {
                            for fld in fields {
                                self.ty(&mut fld.ty);
                            }
                        }
                    }
                    if let Some(d) = &mut v.discriminant {
                        self.expr(d);
                    }
                }
                self.generics.clear();
            }
            ItemKind::Const(c) => {
                self.ty(&mut c.ty);
                self.expr(&mut c.value);
            }
            ItemKind::Extern(blk) => {
                for f in &mut blk.items {
                    self.func(f);
                }
            }
            ItemKind::Trait(t) => {
                for f in &mut t.methods {
                    self.func(f);
                }
            }
            ItemKind::Impl(_) | ItemKind::Import(_) | ItemKind::TypeAlias(_) => {}
        }
    }

    fn generic_params(&mut self, gs: &mut [GenericParam]) {
        for g in gs {
            if let GenericParamKind::Const { ty, .. } = &mut g.kind {
                self.ty(ty);
            }
        }
    }

    fn func(&mut self, f: &mut FnDecl) {
        self.generics = f.generics.iter().map(generic_sym).collect();
        self.generic_params(&mut f.generics);
        for p in &mut f.params {
            self.ty(&mut p.ty);
        }
        if let Some(r) = &mut f.ret {
            self.ty(r);
        }
        for w in &mut f.where_preds {
            self.expr(w);
        }
        if let Some(body) = &mut f.body {
            self.block(body);
        }
        self.generics.clear();
    }

    fn ty(&mut self, t: &mut TypeExpr) {
        match &mut t.kind {
            TypeKind::Path(p) => {
                if let Some(j) = self.alias_of(p) {
                    let span = t.span;
                    *t = self.apply(j, &[], span);
                    return;
                }
                let sym = p.segments.last().map(|s| s.sym);
                if self.in_alias && !(p.is_single() && sym.is_some_and(|s| self.is_type_name(s))) {
                    let p = p.clone();
                    self.unknown_target(&p, t.span);
                }
            }
            TypeKind::Generic { path, args } => {
                for a in args.iter_mut() {
                    if let TypeArg::Type(at) = a {
                        self.ty(at);
                    }
                }
                if let Some(j) = self.alias_of(path) {
                    let args = std::mem::take(args);
                    let span = t.span;
                    *t = self.apply(j, &args, span);
                    return;
                }
                let sym = path.segments.last().unwrap().sym;
                if path.is_single() && self.is_type_name(sym) {
                    let nm = self.name(sym);
                    self.sink.emit(
                        Diagnostic::error(format!("type `{nm}` takes no type arguments"))
                            .with_code("E0327")
                            .primary(t.span, "only a generic type alias takes arguments")
                            .help(format!("write `{nm}` alone")),
                    );
                    t.kind = TypeKind::Unit;
                    return;
                }
                // A name that is no type at all: report it as such, where it is.
                let path: Path = (**path).clone();
                t.kind = TypeKind::Path(path.clone());
                if self.in_alias {
                    self.unknown_target(&path, t.span);
                }
            }
            TypeKind::Int(_) | TypeKind::Unit => {}
            TypeKind::Pointer { pointee, .. } | TypeKind::Ref { pointee, .. } => self.ty(pointee),
            TypeKind::Slice(elem) | TypeKind::Vector { elem, .. } => self.ty(elem),
            TypeKind::Array { elem, len, .. } => {
                self.ty(elem);
                self.expr(len);
            }
            TypeKind::Tuple(elems) => {
                for e in elems {
                    self.ty(e);
                }
            }
            TypeKind::Tensor { elem, .. } => self.ty(elem),
            TypeKind::Fn { params, ret } => {
                for p in params {
                    self.ty(p);
                }
                if let Some(r) = ret {
                    self.ty(r);
                }
            }
        }
    }

    fn block(&mut self, b: &mut Block) {
        for s in &mut b.stmts {
            self.stmt(s);
        }
        if let Some(tail) = &mut b.tail {
            self.expr(tail);
        }
    }

    fn stmt(&mut self, s: &mut Stmt) {
        match &mut s.kind {
            StmtKind::Let { ty, init, .. } => {
                if let Some(t) = ty {
                    self.ty(t);
                }
                if let Some(e) = init {
                    self.expr(e);
                }
            }
            StmtKind::Assign { target, value, .. } => {
                self.expr(target);
                self.expr(value);
            }
            StmtKind::Expr(e) | StmtKind::Defer(e) => self.expr(e),
            StmtKind::Return(e) | StmtKind::Break(_, e) => {
                if let Some(e) = e {
                    self.expr(e);
                }
            }
            StmtKind::Continue(_) => {}
            StmtKind::While { cond, body, .. } => {
                self.expr(cond);
                self.block(body);
            }
            StmtKind::For { iter, body, .. } => {
                match iter {
                    ForIter::Range {
                        start, end, step, ..
                    } => {
                        self.expr(start);
                        if let Some(e) = end {
                            self.expr(e);
                        }
                        if let Some(e) = step {
                            self.expr(e);
                        }
                    }
                    ForIter::Expr(e) => self.expr(e),
                }
                self.block(body);
            }
        }
    }

    fn expr(&mut self, e: &mut Expr) {
        match &mut e.kind {
            ExprKind::Int(_)
            | ExprKind::Float(_)
            | ExprKind::Str(_)
            | ExprKind::Char(_)
            | ExprKind::Bool(_)
            | ExprKind::Path(_) => {}
            ExprKind::Unary { expr, .. } => self.expr(expr),
            ExprKind::Binary { lhs, rhs, .. } => {
                self.expr(lhs);
                self.expr(rhs);
            }
            ExprKind::Call {
                callee,
                generic_args,
                args,
            } => {
                self.expr(callee);
                for t in generic_args {
                    self.ty(t);
                }
                for a in args {
                    self.expr(a);
                }
            }
            ExprKind::Index { base, indices } => {
                self.expr(base);
                for i in indices {
                    self.expr(i);
                }
            }
            ExprKind::Field { base, .. } | ExprKind::TupleField { base, .. } => self.expr(base),
            ExprKind::Cast { expr, ty } => {
                self.expr(expr);
                self.ty(ty);
            }
            ExprKind::StructLit { fields, rest, .. } => {
                for fi in fields {
                    self.expr(&mut fi.value);
                }
                if let Some(r) = rest {
                    self.expr(r);
                }
            }
            ExprKind::ArrayLit(elems) | ExprKind::TupleLit(elems) => {
                for x in elems {
                    self.expr(x);
                }
            }
            ExprKind::ArrayRepeat { value, count } => {
                self.expr(value);
                self.expr(count);
            }
            ExprKind::Block(b) => self.block(b),
            ExprKind::If {
                cond,
                then_branch,
                else_branch,
            } => {
                self.expr(cond);
                self.block(then_branch);
                if let Some(e) = else_branch {
                    self.expr(e);
                }
            }
            ExprKind::Match { scrutinee, arms } => {
                self.expr(scrutinee);
                for arm in arms {
                    if let Some(g) = &mut arm.guard {
                        self.expr(g);
                    }
                    self.expr(&mut arm.body);
                }
            }
            ExprKind::Loop { body, .. } => self.block(body),
            ExprKind::SizeOf(t) | ExprKind::AlignOf(t) => self.ty(t),
        }
    }
}

/// Substitute an alias's parameters by the arguments of one use. A parameter may stand for a
/// type (`Pair<T> = (T, T)`), a dimension (`Tensor[f32, N]`), or an array length (`[f32; N]`);
/// a bare name or an integer argument serves any of those, a type only a type, and a dimension
/// expression (`N*2`, `?`) only a dimension or a length.
struct Subst<'a, 'b> {
    map: FxHashMap<Symbol, &'b TypeArg>,
    alias: String,
    interner: &'a mut Interner,
    sink: &'a mut DiagnosticSink,
}

impl<'b> Subst<'_, 'b> {
    fn kind_error(&mut self, span: Span, msg: String, label: &str) {
        self.sink.emit(
            Diagnostic::error(msg)
                .with_code("E0327")
                .primary(span, label.to_string())
                .note(format!(
                    "an argument of the alias `{}` replaces its parameter wherever the alias uses it",
                    self.alias
                )),
        );
    }

    fn param(&self, p: &Path) -> Option<&'b TypeArg> {
        if p.is_single() {
            self.map.get(&p.first().sym).copied()
        } else {
            None
        }
    }

    fn ty(&mut self, t: &mut TypeExpr) {
        match &mut t.kind {
            TypeKind::Path(p) => {
                let Some(arg) = self.param(p) else {
                    return;
                };
                match arg {
                    TypeArg::Type(at) => {
                        let span = t.span;
                        *t = at.clone();
                        t.span = span;
                    }
                    TypeArg::Dim(d) => match &d.kind {
                        DimKind::Int(v) => {
                            t.kind = TypeKind::Int(self.interner.intern(&v.to_string()))
                        }
                        DimKind::Named(s) => {
                            t.kind = TypeKind::Path(Path {
                                segments: vec![Ident {
                                    sym: *s,
                                    span: d.span,
                                }],
                                span: d.span,
                            })
                        }
                        DimKind::Dynamic | DimKind::Binary { .. } => {
                            let nm = self.interner.resolve(p.first().sym).to_string();
                            self.kind_error(
                                d.span,
                                format!("the alias uses `{nm}` as a type, but this argument is a dimension"),
                                "expected a type",
                            );
                            t.kind = TypeKind::Unit;
                        }
                    },
                }
            }
            TypeKind::Int(_) | TypeKind::Unit => {}
            TypeKind::Pointer { pointee, .. } | TypeKind::Ref { pointee, .. } => self.ty(pointee),
            TypeKind::Slice(elem) | TypeKind::Vector { elem, .. } => self.ty(elem),
            TypeKind::Array { elem, len, .. } => {
                self.ty(elem);
                self.expr(len);
            }
            TypeKind::Tuple(elems) => {
                for e in elems {
                    self.ty(e);
                }
            }
            TypeKind::Tensor { elem, dims, .. } => {
                self.ty(elem);
                for d in dims {
                    self.dim(d);
                }
            }
            TypeKind::Fn { params, ret } => {
                for p in params {
                    self.ty(p);
                }
                if let Some(r) = ret {
                    self.ty(r);
                }
            }
            TypeKind::Generic { args, .. } => {
                for a in args {
                    match a {
                        TypeArg::Type(t) => self.ty(t),
                        TypeArg::Dim(d) => self.dim(d),
                    }
                }
            }
        }
    }

    fn dim(&mut self, d: &mut Dim) {
        match &mut d.kind {
            DimKind::Named(s) => {
                let Some(arg) = self.map.get(s).copied() else {
                    return;
                };
                match arg {
                    TypeArg::Dim(x) => d.kind = x.kind.clone(),
                    TypeArg::Type(t) => match &t.kind {
                        TypeKind::Path(p) if p.is_single() => {
                            d.kind = DimKind::Named(p.first().sym)
                        }
                        TypeKind::Int(v) => match parse_u64(self.interner.resolve(*v)) {
                            Some(n) => d.kind = DimKind::Int(n),
                            None => d.kind = DimKind::Dynamic,
                        },
                        _ => {
                            let nm = self.interner.resolve(*s).to_string();
                            self.kind_error(
                                t.span,
                                format!("the alias uses `{nm}` as a dimension, but this argument is a type"),
                                "expected a dimension (an integer, a name, or arithmetic on those)",
                            );
                            d.kind = DimKind::Dynamic;
                        }
                    },
                }
            }
            DimKind::Binary { lhs, rhs, .. } => {
                self.dim(lhs);
                self.dim(rhs);
            }
            DimKind::Int(_) | DimKind::Dynamic => {}
        }
    }

    /// An array length (`[f32; N]`, `[f32; N * 2]`): a parameter becomes the argument's value.
    fn expr(&mut self, e: &mut Expr) {
        match &mut e.kind {
            ExprKind::Path(p) => {
                let Some(arg) = self.param(p) else {
                    return;
                };
                let nm = self.interner.resolve(p.first().sym).to_string();
                match arg {
                    TypeArg::Type(t) => match &t.kind {
                        TypeKind::Path(q) if q.is_single() => e.kind = ExprKind::Path(q.clone()),
                        TypeKind::Int(v) => e.kind = ExprKind::Int(*v),
                        _ => self.kind_error(
                            t.span,
                            format!("the alias uses `{nm}` as an array length, but this argument is a type"),
                            "expected a length",
                        ),
                    },
                    TypeArg::Dim(d) => match self.dim_expr(d, e.id) {
                        Some(k) => e.kind = k,
                        None => self.kind_error(
                            d.span,
                            format!("the alias uses `{nm}` as an array length, which cannot be `?`"),
                            "expected a length",
                        ),
                    },
                }
            }
            ExprKind::Unary { expr, .. } => self.expr(expr),
            ExprKind::Binary { lhs, rhs, .. } => {
                self.expr(lhs);
                self.expr(rhs);
            }
            ExprKind::Cast { expr, ty } => {
                self.expr(expr);
                self.ty(ty);
            }
            ExprKind::Call { args, .. } => {
                for a in args {
                    self.expr(a);
                }
            }
            _ => {}
        }
    }

    /// A dimension as an array-length expression. Ids are placeholders; the caller renumbers the
    /// whole copy.
    fn dim_expr(&mut self, d: &Dim, id: NodeId) -> Option<ExprKind> {
        Some(match &d.kind {
            DimKind::Int(v) => ExprKind::Int(self.interner.intern(&v.to_string())),
            DimKind::Named(s) => ExprKind::Path(Path {
                segments: vec![Ident {
                    sym: *s,
                    span: d.span,
                }],
                span: d.span,
            }),
            DimKind::Dynamic => return None,
            DimKind::Binary { op, lhs, rhs } => {
                let l = self.dim_expr(lhs, id)?;
                let r = self.dim_expr(rhs, id)?;
                ExprKind::Binary {
                    op: *op,
                    lhs: Box::new(Expr {
                        id,
                        kind: l,
                        span: lhs.span,
                    }),
                    rhs: Box::new(Expr {
                        id,
                        kind: r,
                        span: rhs.span,
                    }),
                }
            }
        })
    }
}

/// The value of an integer type argument as the parser kept it (`4`, `0x10`, `1_024`).
fn parse_u64(text: &str) -> Option<u64> {
    let t: String = text.chars().filter(|&c| c != '_').collect();
    let (digits, radix) = if let Some(h) = t.strip_prefix("0x").or_else(|| t.strip_prefix("0X")) {
        (h, 16)
    } else if let Some(b) = t.strip_prefix("0b").or_else(|| t.strip_prefix("0B")) {
        (b, 2)
    } else if let Some(o) = t.strip_prefix("0o").or_else(|| t.strip_prefix("0O")) {
        (o, 8)
    } else {
        (t.as_str(), 10)
    };
    u64::from_str_radix(digits, radix).ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use wukong_ast::Param;

    /// Parse `src`, expand its aliases, and return the module, its printed form and the codes.
    fn expand_module(src: &str) -> (Module, String, Vec<&'static str>) {
        let mut interner = Interner::new();
        let (mut module, pd) =
            wukong_parser::parse_module(src, wukong_span::SourceId(0), &mut interner);
        assert!(pd.is_empty(), "parse errors: {pd:?}");
        let mut sink = DiagnosticSink::new();
        let mut next = 1_000_000;
        expand_type_aliases(&mut module, &mut interner, &mut sink, &mut next);
        let codes = sink.diagnostics().iter().filter_map(|d| d.code).collect();
        let printed = wukong_ast::print::print_module(&module, &interner);
        (module, printed, codes)
    }

    fn expand(src: &str) -> (String, Vec<&'static str>) {
        let (_, printed, codes) = expand_module(src);
        (printed, codes)
    }

    fn params(m: &Module) -> Vec<&Param> {
        m.items
            .iter()
            .filter_map(|i| match &i.kind {
                ItemKind::Fn(f) => Some(f.params.iter()),
                _ => None,
            })
            .flatten()
            .collect()
    }

    #[test]
    fn scalar_and_generic_aliases_expand_everywhere() {
        let (out, codes) = expand(
            "type Real = f32;\n\
             type Mat<M, N> = Tensor[Real, M, N];\n\
             type Buf<T, N> = [T; N];\n\
             struct S { w: Buf<Real, 3> }\n\
             fn f<K>(a: Mat<K, K * 2>, x: Real) -> Buf<Mat<2, 3>, 4> {\n\
                 let y: Mat<2, 3> = a;\n\
                 let z = x as Real;\n\
                 let h = alloc::<Real>(4);\n\
             }\n",
        );
        assert!(codes.is_empty(), "{codes:?}");
        assert!(
            !out.lines().any(|l| l.starts_with("type ")),
            "alias items survive: {out}"
        );
        assert!(out.contains("field w: [f32; 3]"), "{out}");
        assert!(out.contains("param a: Tensor[f32, K, (K * 2)]"), "{out}");
        assert!(out.contains("param x: f32"), "{out}");
        assert!(out.contains("ret [Tensor[f32, 2, 3]; 4]"), "{out}");
        assert!(!out.contains("Real"), "{out}");
        assert!(!out.contains("Mat"), "{out}");
    }

    /// Two uses of one alias are two trees: no `NodeId` is shared, or sema's per-node tables
    /// (an array length's type, for one) would collide.
    #[test]
    fn every_copy_has_fresh_node_ids() {
        let (m, _, codes) =
            expand_module("type B<N> = [f32; N * 2];\nfn f(a: B<2>, b: B<3>, c: B<2>) {}\n");
        assert!(codes.is_empty(), "{codes:?}");
        let mut ids = Vec::new();
        for p in params(&m) {
            ids.push(p.ty.id);
            let TypeKind::Array { len, .. } = &p.ty.kind else {
                panic!("not an array: {:?}", p.ty.kind);
            };
            ids.push(len.id);
        }
        let unique: FxHashSet<NodeId> = ids.iter().copied().collect();
        assert_eq!(unique.len(), ids.len(), "{ids:?}");
        assert!(ids.iter().all(|id| *id != NodeId::DUMMY));
    }

    #[test]
    fn a_generic_parameter_shadows_an_alias() {
        let (out, codes) =
            expand("type T = i64;\nfn f<T>(x: T) -> T { return x; }\nfn g(y: T) {}\n");
        assert!(codes.is_empty(), "{codes:?}");
        assert!(out.contains("param x: T"), "{out}");
        assert!(out.contains("param y: i64"), "{out}");
    }

    #[test]
    fn alias_errors_carry_their_codes() {
        assert_eq!(expand("type A = A;\nfn f() {}\n").1, vec!["E0326"]);
        // One diagnostic per cycle, however many aliases it runs through or uses it has.
        assert_eq!(
            expand("type A = [B; 2];\ntype B = *C;\ntype C = A;\nfn f(x: A, y: C) {}\n").1,
            vec!["E0326"]
        );
        assert_eq!(
            expand("type M<N> = [f32; N];\nfn f(x: M) {}\n").1,
            vec!["E0327"]
        );
        assert_eq!(
            expand("type M<N> = [f32; N];\nfn f(x: M<1, 2>) {}\n").1,
            vec!["E0327"]
        );
        assert_eq!(expand("type R = f32;\nfn f(x: R<2>) {}\n").1, vec!["E0327"]);
        assert_eq!(
            expand("type R = f32;\nfn f(x: f32<2>) {}\n").1,
            vec!["E0327"]
        );
        assert_eq!(
            expand("type P<T> = *T;\nfn f(x: P<N * 2>) {}\n").1,
            vec!["E0327"]
        );
        assert_eq!(
            expand("type V<N> = Tensor[f32, N];\nfn f(x: V<(i32, i32)>) {}\n").1,
            vec!["E0327"]
        );
        assert_eq!(expand("type W = [Real; 8];\nfn f() {}\n").1, vec!["E0328"]);
        assert_eq!(expand("type f32 = i64;\nfn f() {}\n").1, vec!["E0300"]);
        // `str` is a built-in type as well: it names no alias, and an alias may name it.
        assert_eq!(expand("type str = i32;\nfn f() {}\n").1, vec!["E0300"]);
        let (out, codes) = expand("type Name = str;\nfn f(n: Name) {}\n");
        assert!(codes.is_empty(), "{codes:?}");
        assert!(out.contains("n: str"), "{out}");
        assert_eq!(
            expand("struct S { x: i32 }\ntype S = i32;\nfn f() {}\n").1,
            vec!["E0300"]
        );
    }

    /// This pass runs after `load_imports` has renamed an imported module's items to
    /// `<import.path>.<name>`, so a built-in name must be recognized in the name the module WROTE,
    /// the last segment: `type str = i32;` in `lib/m.wk` arrives as `lib.m.str`, and compared
    /// whole it was no built-in name, so inside that module every `str` silently meant `i32`.
    #[test]
    fn a_builtin_name_is_reserved_for_an_imported_alias_too() {
        for (mangled, reserved) in [
            ("lib.m.str", true),
            ("lib.m.f32", true),
            ("lib.m.strs", false),
            ("lib.str.Name", false),
        ] {
            let mut interner = Interner::new();
            let (mut module, pd) = wukong_parser::parse_module(
                "type Name = i32;\nfn f(x: Name) {}\n",
                wukong_span::SourceId(0),
                &mut interner,
            );
            assert!(pd.is_empty(), "parse errors: {pd:?}");
            let sym = interner.intern(mangled);
            for item in &mut module.items {
                if let ItemKind::TypeAlias(a) = &mut item.kind {
                    a.name.sym = sym;
                }
            }
            let mut sink = DiagnosticSink::new();
            let mut next = 1_000_000;
            expand_type_aliases(&mut module, &mut interner, &mut sink, &mut next);
            let codes: Vec<_> = sink.diagnostics().iter().filter_map(|d| d.code).collect();
            assert_eq!(codes.contains(&"E0300"), reserved, "{mangled}: {codes:?}");
        }
    }
}
