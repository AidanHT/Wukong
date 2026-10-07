//! Module-scoped name resolution for a multi-file program.
//!
//! The loader hands over every file's items, tagged with the file that defines them. This pass
//!
//! 1. builds each module's **export table**: every top-level name it defines, the program-wide
//!    symbol it will carry, and whether it is `pub`. The root file's items keep their own names;
//!    an imported module's items are renamed to `<import.path>.<name>` (`lib.mathlib.triple`), a
//!    spelling no source identifier can produce, so two modules can never collide. An `extern`
//!    function keeps its name: it is the C link symbol;
//! 2. builds each file's **scope** from its own items and its own `import`s (glob, `as` alias, or
//!    `{x, y}` selection), reporting a private (E0312) or missing (E0301) selection, a selection
//!    that collides with a local definition (E0300), and an alias bound twice (E0300);
//! 3. **rewrites** every item of every file: declaration names to their program-wide symbol, and
//!    every reference to the symbol it resolves to in that file. A local binding or a generic
//!    parameter shadows an item exactly as in sema (`resolve_value`: locals, then generics, then
//!    items), so the walk tracks both. `q::name` (parsed as a field access on the path `q`) and
//!    the multi-segment paths of patterns, struct literals and types are module-qualified when
//!    `q` is a module qualifier of the file and not a local, a generic or an item;
//! 4. merges identical `extern` declarations repeated across modules.
//!
//! Everything downstream (sema, MIR, backends) still sees one flat table of unique symbols, which
//! is why no other crate changes. A name this pass leaves alone that sema cannot resolve is sema's
//! ordinary E0301 — e.g. an item of a module the file does not import, which is renamed away.

use wukong_ast::{
    Block, Expr, ExprKind, FnDecl, ForIter, GenericParam, GenericParamKind, Ident, Item, ItemKind,
    PatKind, Path, Pattern, Stmt, StmtKind, TypeExpr, TypeKind, VariantData, VariantPat,
};
use wukong_diag::{Diagnostic, DiagnosticSink};
use wukong_span::{FxHashMap, Interner, Span, Symbol};

use super::FileInfo;

/// One top-level name a module defines.
#[derive(Clone, Copy)]
struct Export {
    /// The program-wide symbol the item carries after resolution.
    sym: Symbol,
    is_pub: bool,
    /// The declaration's name span, for "defined here" labels.
    span: Span,
}

/// How an unqualified name became visible in a file. Local items and explicit `{x}` selections
/// shadow glob imports; only two GLOB imports can disagree silently, and that is reported at use.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Origin {
    Local,
    Explicit,
    Glob,
}

#[derive(Clone, Copy)]
enum Binding {
    Item {
        sym: Symbol,
        origin: Origin,
    },
    /// Two glob imports (of files `a` and `b`) bring different items under one name.
    Ambiguous {
        a: usize,
        b: usize,
    },
}

#[derive(Clone, Copy)]
enum Qualifier {
    Module(usize),
    /// Two plain `import x.m;` / `import y.m;` both bind the implicit qualifier `m`.
    Ambiguous(usize, usize),
}

/// The names visible in one file.
struct Scope {
    file: usize,
    values: FxHashMap<Symbol, Binding>,
    modules: FxHashMap<Symbol, Qualifier>,
}

/// Resolve the whole program in place. `root_items` are the root file's items (file 0, which keep
/// their position at the front of the merged module); `extra` holds every imported item tagged
/// with its file index, in DFS order.
pub(super) fn resolve_program(
    root_items: &mut [Item],
    extra: &mut [(usize, Item)],
    files: &[FileInfo],
    interner: &mut Interner,
    sink: &mut DiagnosticSink,
) {
    let exports = build_exports(root_items, extra, files, interner);
    let interner: &Interner = interner;

    // Root items and extern functions keep their source spelling, so an unqualified reference can
    // reach them without any import — the flat-namespace leak the rewrite must refuse. Remember
    // which file owns each such symbol.
    let mut unmangled: FxHashMap<Symbol, usize> = FxHashMap::default();
    for (file, table) in exports.iter().enumerate() {
        for (&name, e) in table {
            if e.sym == name {
                unmangled.entry(name).or_insert(file);
            }
        }
    }

    let scopes: Vec<Scope> = (0..files.len())
        .map(|f| build_scope(f, files, &exports, interner, sink))
        .collect();

    let mut rw = Rewriter {
        scope: &scopes[0],
        files,
        exports: &exports,
        unmangled: &unmangled,
        interner,
        sink,
        locals: Vec::new(),
        generics: Vec::new(),
    };
    for item in root_items.iter_mut() {
        rw.item(item);
    }
    for (file, item) in extra.iter_mut() {
        rw.scope = &scopes[*file];
        rename_decls(item, &exports[*file]);
        rw.item(item);
    }

    merge_extern_decls(root_items, extra, interner);
}

/// The top-level names an item declares: `(name, is_pub, is_extern)`.
fn decl_names(item: &Item) -> Vec<(Ident, bool, bool)> {
    match &item.kind {
        ItemKind::Fn(f) => vec![(f.name, f.is_pub, false)],
        ItemKind::Struct(s) => vec![(s.name, s.is_pub, false)],
        ItemKind::Enum(e) => vec![(e.name, e.is_pub, false)],
        ItemKind::Const(c) => vec![(c.name, c.is_pub, false)],
        ItemKind::Extern(blk) => blk.items.iter().map(|f| (f.name, f.is_pub, true)).collect(),
        ItemKind::Import(_) => Vec::new(),
    }
}

fn build_exports(
    root_items: &[Item],
    extra: &[(usize, Item)],
    files: &[FileInfo],
    interner: &mut Interner,
) -> Vec<FxHashMap<Symbol, Export>> {
    let mut exports: Vec<FxHashMap<Symbol, Export>> =
        (0..files.len()).map(|_| FxHashMap::default()).collect();
    let tagged = root_items
        .iter()
        .map(|item| (0usize, item))
        .chain(extra.iter().map(|(f, item)| (*f, item)));
    for (file, item) in tagged {
        for (name, is_pub, is_extern) in decl_names(item) {
            if exports[file].contains_key(&name.sym) {
                // A same-file duplicate gets the same symbol and is sema's ordinary E0300.
                continue;
            }
            let sym = if file == 0 || is_extern {
                name.sym
            } else {
                let mangled = format!("{}.{}", files[file].path, interner.resolve(name.sym));
                interner.intern(&mangled)
            };
            exports[file].insert(
                name.sym,
                Export {
                    sym,
                    is_pub,
                    span: name.span,
                },
            );
        }
    }
    exports
}

/// Rename an imported item's declared names to their program-wide symbols.
fn rename_decls(item: &mut Item, table: &FxHashMap<Symbol, Export>) {
    let rename = |name: &mut Ident| {
        if let Some(e) = table.get(&name.sym) {
            name.sym = e.sym;
        }
    };
    match &mut item.kind {
        ItemKind::Fn(f) => rename(&mut f.name),
        ItemKind::Struct(s) => rename(&mut s.name),
        ItemKind::Enum(e) => rename(&mut e.name),
        ItemKind::Const(c) => rename(&mut c.name),
        // Extern functions keep their link name; `Import` declares nothing.
        ItemKind::Extern(_) | ItemKind::Import(_) => {}
    }
}

/// How diagnostics name file `f`'s module.
fn module_name(files: &[FileInfo], f: usize) -> String {
    if f == 0 {
        "the root module".to_string()
    } else {
        format!("module `{}`", files[f].path)
    }
}

fn private_item(
    files: &[FileInfo],
    interner: &Interner,
    name: Ident,
    module: usize,
    def: Span,
) -> Diagnostic {
    let nm = interner.resolve(name.sym);
    Diagnostic::error(format!(
        "`{nm}` is private to {}",
        module_name(files, module)
    ))
    .with_code("E0312")
    .primary(name.span, "private item")
    .secondary(def, "declared here without `pub`")
    .help(
        "declare the item `pub` in its module (`pub fn`, `pub struct`, `pub enum`, `pub const`, \
         or a `pub fn` inside an `extern` block) to use it from another file",
    )
}

fn missing_item(files: &[FileInfo], interner: &Interner, name: Ident, module: usize) -> Diagnostic {
    Diagnostic::error(format!(
        "cannot find `{}` in {}",
        interner.resolve(name.sym),
        module_name(files, module)
    ))
    .with_code("E0301")
    .primary(name.span, "not defined in that module")
}

fn build_scope(
    f: usize,
    files: &[FileInfo],
    exports: &[FxHashMap<Symbol, Export>],
    interner: &Interner,
    sink: &mut DiagnosticSink,
) -> Scope {
    let mut values: FxHashMap<Symbol, Binding> = FxHashMap::default();
    for (&name, e) in &exports[f] {
        values.insert(
            name,
            Binding::Item {
                sym: e.sym,
                origin: Origin::Local,
            },
        );
    }
    let mut modules: FxHashMap<Symbol, Qualifier> = FxHashMap::default();
    let mut globs: Vec<usize> = Vec::new();
    let mut aliases: Vec<Ident> = Vec::new();

    for (imp, target) in &files[f].imports {
        let Some(t) = *target else { continue };
        if let Some(list) = &imp.items {
            select_items(f, t, list, files, exports, interner, sink, &mut values);
        } else if let Some(alias) = imp.alias {
            match modules.get(&alias.sym) {
                Some(Qualifier::Module(m)) if *m == t => {}
                Some(_) => sink.emit(
                    Diagnostic::error(format!(
                        "the module name `{}` is bound more than once",
                        interner.resolve(alias.sym)
                    ))
                    .with_code("E0300")
                    .primary(alias.span, "rebound here")
                    .help("choose a different `as` name for one of the imports"),
                ),
                None => {
                    modules.insert(alias.sym, Qualifier::Module(t));
                }
            }
            aliases.push(alias);
        } else {
            let Some(last) = imp.path.segments.last() else {
                continue;
            };
            match modules.get(&last.sym).copied() {
                None => {
                    modules.insert(last.sym, Qualifier::Module(t));
                }
                Some(Qualifier::Module(m)) if m != t => {
                    modules.insert(last.sym, Qualifier::Ambiguous(m, t));
                }
                Some(_) => {}
            }
            if !globs.contains(&t) {
                globs.push(t);
            }
        }
    }

    for &t in &globs {
        for (&name, e) in &exports[t] {
            if !(e.is_pub || t == f) {
                continue;
            }
            match values.get(&name).copied() {
                None => {
                    values.insert(
                        name,
                        Binding::Item {
                            sym: e.sym,
                            origin: Origin::Glob,
                        },
                    );
                }
                Some(Binding::Item {
                    sym,
                    origin: Origin::Glob,
                }) if sym != e.sym => {
                    let a = exports
                        .iter()
                        .position(|tbl| tbl.get(&name).is_some_and(|x| x.sym == sym))
                        .unwrap_or(t);
                    values.insert(name, Binding::Ambiguous { a, b: t });
                }
                Some(_) => {}
            }
        }
    }

    // An explicit `as` name that is also an item in scope would make `c::x` mean two things.
    for alias in aliases {
        if values.contains_key(&alias.sym) {
            sink.emit(
                Diagnostic::error(format!(
                    "the module name `{}` collides with an item of the same name in scope",
                    interner.resolve(alias.sym)
                ))
                .with_code("E0300")
                .primary(alias.span, "module name bound here")
                .help("choose a different `as` name"),
            );
        }
    }

    Scope {
        file: f,
        values,
        modules,
    }
}

/// `import <t>.{x, y}` in file `f`: bind each selected item, which must exist and be `pub`.
#[allow(clippy::too_many_arguments)]
fn select_items(
    f: usize,
    t: usize,
    list: &[Ident],
    files: &[FileInfo],
    exports: &[FxHashMap<Symbol, Export>],
    interner: &Interner,
    sink: &mut DiagnosticSink,
    values: &mut FxHashMap<Symbol, Binding>,
) {
    for x in list {
        let Some(e) = exports[t].get(&x.sym).copied() else {
            sink.emit(missing_item(files, interner, *x, t));
            continue;
        };
        if !e.is_pub && t != f {
            sink.emit(private_item(files, interner, *x, t, e.span));
            continue;
        }
        match values.get(&x.sym).copied() {
            Some(Binding::Item { sym, origin }) if sym != e.sym => {
                let what = if origin == Origin::Local {
                    "defined in this module"
                } else {
                    "imported from another module"
                };
                sink.emit(
                    Diagnostic::error(format!(
                        "the name `{}` is imported here but is already {what}",
                        interner.resolve(x.sym)
                    ))
                    .with_code("E0300")
                    .primary(x.span, "imported here")
                    .help("refer to one of them through a qualified `module::name` path instead"),
                );
            }
            Some(Binding::Item { .. }) => {}
            _ => {
                values.insert(
                    x.sym,
                    Binding::Item {
                        sym: e.sym,
                        origin: Origin::Explicit,
                    },
                );
            }
        }
    }
}

/// Rewrites one file's items against that file's [`Scope`].
struct Rewriter<'a> {
    scope: &'a Scope,
    files: &'a [FileInfo],
    exports: &'a [FxHashMap<Symbol, Export>],
    unmangled: &'a FxHashMap<Symbol, usize>,
    interner: &'a Interner,
    sink: &'a mut DiagnosticSink,
    /// Local bindings (parameters, `let`, `for` and pattern bindings), innermost scope last.
    locals: Vec<Vec<Symbol>>,
    /// The generic parameters of the item being walked.
    generics: Vec<Symbol>,
}

fn generic_sym(g: &GenericParam) -> Symbol {
    match &g.kind {
        GenericParamKind::Type(id) => id.sym,
        GenericParamKind::Const { name, .. } => name.sym,
    }
}

impl Rewriter<'_> {
    fn shadowed(&self, s: Symbol) -> bool {
        self.generics.contains(&s) || self.locals.iter().any(|scope| scope.contains(&s))
    }

    fn bind(&mut self, s: Symbol) {
        if let Some(scope) = self.locals.last_mut() {
            scope.push(s);
        }
    }

    fn ambiguous(&mut self, name: Ident, a: usize, b: usize, what: &str) {
        let nm = self.interner.resolve(name.sym);
        self.sink.emit(
            Diagnostic::error(format!("`{nm}` is ambiguous"))
                .with_code("E0313")
                .primary(name.span, format!("ambiguous {what}"))
                .note(format!(
                    "`{nm}` is brought in by both {} and {}",
                    module_name(self.files, a),
                    module_name(self.files, b)
                ))
                .help(
                    "qualify the use (`module::name`), import the intended item explicitly with \
                     `import a.b.{name}`, or rename an import with `as`",
                ),
        );
    }

    /// The symbol an unqualified name resolves to in this file, when resolution changes it.
    fn unqualified(&mut self, id: Ident) -> Option<Symbol> {
        if self.shadowed(id.sym) {
            return None;
        }
        match self.scope.values.get(&id.sym).copied() {
            Some(Binding::Item { sym, .. }) => (sym != id.sym).then_some(sym),
            Some(Binding::Ambiguous { a, b }) => {
                self.ambiguous(id, a, b, "name");
                None
            }
            None => {
                // A root item or an extern function keeps its spelling, so sema WOULD find it —
                // but it belongs to a module this file does not see.
                if let Some(&owner) = self.unmangled.get(&id.sym) {
                    if owner != self.scope.file {
                        let nm = self.interner.resolve(id.sym);
                        self.sink.emit(
                            Diagnostic::error(format!("cannot find `{nm}` in this module"))
                                .with_code("E0301")
                                .primary(id.span, "not in scope")
                                .note(format!(
                                    "`{nm}` is defined in {}, which this file does not import",
                                    module_name(self.files, owner)
                                ))
                                .help(
                                    "imports are per file and not transitive: add an `import` \
                                     of the defining module (and declare the item `pub`)",
                                ),
                        );
                    }
                }
                None
            }
        }
    }

    /// Resolve `q::name` when `q` is one of this file's module qualifiers. `None`: `q` is not a
    /// module qualifier here (a local, a generic, an item such as an enum, or a builtin namespace
    /// like `f32x8`) and the path is left alone. `Some(None)`: an error was reported.
    fn qualified(&mut self, q: Ident, name: Ident) -> Option<Option<Symbol>> {
        if self.shadowed(q.sym) || self.scope.values.contains_key(&q.sym) {
            return None;
        }
        let m = match *self.scope.modules.get(&q.sym)? {
            Qualifier::Module(m) => m,
            Qualifier::Ambiguous(a, b) => {
                self.ambiguous(q, a, b, "module name");
                return Some(None);
            }
        };
        let Some(e) = self.exports[m].get(&name.sym).copied() else {
            self.sink
                .emit(missing_item(self.files, self.interner, name, m));
            return Some(None);
        };
        if !e.is_pub && m != self.scope.file {
            self.sink
                .emit(private_item(self.files, self.interner, name, m, e.span));
            return Some(None);
        }
        Some(Some(e.sym))
    }

    /// A multi-segment path (pattern, struct literal, type) or a single name in any position.
    fn path(&mut self, p: &mut Path) {
        let Some(&head) = p.segments.first() else {
            return;
        };
        if p.segments.len() == 1 || self.shadowed(head.sym) {
            if let Some(sym) = self.unqualified(head) {
                p.segments[0].sym = sym;
            }
            return;
        }
        if self.scope.values.contains_key(&head.sym) {
            // `Enum::Variant` / an item head: rename the item.
            if let Some(sym) = self.unqualified(head) {
                p.segments[0].sym = sym;
            }
            return;
        }
        match self.qualified(head, p.segments[1]) {
            Some(Some(sym)) => {
                let span = head.span.to(p.segments[1].span);
                // `[q, item, rest..]` -> `[item', rest..]`.
                p.segments.remove(0);
                p.segments[0] = Ident { sym, span };
            }
            Some(None) => {}
            // Not a module qualifier and not in scope: a builtin namespace is left alone, but a
            // root item or an extern another module owns must not leak in through a pattern or
            // type path (`RootEnum::A`) when the expression spelling of the same path is refused.
            None => {
                self.unqualified(head);
            }
        }
    }

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
            ItemKind::Import(_) => {}
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
        if let Some(body) = &mut f.body {
            self.locals
                .push(f.params.iter().map(|p| p.name.sym).collect());
            self.block(body);
            self.locals.pop();
        }
        self.generics.clear();
    }

    fn ty(&mut self, t: &mut TypeExpr) {
        match &mut t.kind {
            TypeKind::Path(p) => self.path(p),
            TypeKind::Int(_) | TypeKind::Unit => {}
            TypeKind::Pointer { pointee, .. } | TypeKind::Ref { pointee, .. } => self.ty(pointee),
            TypeKind::Slice(elem) => self.ty(elem),
            TypeKind::Array { elem, len, .. } => {
                self.ty(elem);
                self.expr(len);
            }
            TypeKind::Tuple(elems) => {
                for e in elems {
                    self.ty(e);
                }
            }
            TypeKind::Vector { elem, .. } => self.ty(elem),
            TypeKind::Tensor { elem, dims, .. } => {
                self.ty(elem);
                for d in dims {
                    if let wukong_ast::DimKind::Named(sym) = &mut d.kind {
                        let id = Ident {
                            sym: *sym,
                            span: d.span,
                        };
                        if let Some(s) = self.unqualified(id) {
                            *sym = s;
                        }
                    }
                }
            }
        }
    }

    fn block(&mut self, b: &mut Block) {
        self.locals.push(Vec::new());
        for s in &mut b.stmts {
            self.stmt(s);
        }
        if let Some(tail) = &mut b.tail {
            self.expr(tail);
        }
        self.locals.pop();
    }

    fn stmt(&mut self, s: &mut Stmt) {
        match &mut s.kind {
            StmtKind::Let { pat, ty, init, .. } => {
                if let Some(t) = ty {
                    self.ty(t);
                }
                if let Some(e) = init {
                    self.expr(e);
                }
                // The binding is in scope only AFTER its initializer, as in sema.
                self.pattern(pat);
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
            StmtKind::For {
                pat, iter, body, ..
            } => {
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
                self.locals.push(Vec::new());
                self.pattern(pat);
                self.block(body);
                self.locals.pop();
            }
        }
    }

    /// Rewrite the paths inside a pattern and bind the names it introduces in the current scope.
    fn pattern(&mut self, p: &mut Pattern) {
        match &mut p.kind {
            PatKind::Ident(s) => self.bind(*s),
            PatKind::Tuple(subs) | PatKind::Or(subs) => {
                for sub in subs {
                    self.pattern(sub);
                }
            }
            PatKind::Path(path) => self.path(path),
            PatKind::Variant { path, fields } => {
                self.path(path);
                match fields {
                    VariantPat::Tuple(subs) => {
                        for sub in subs {
                            self.pattern(sub);
                        }
                    }
                    VariantPat::Struct(fps) => {
                        for fp in fps {
                            self.pattern(&mut fp.pat);
                        }
                    }
                }
            }
            PatKind::Range { lo, hi, .. } => {
                self.pattern(lo);
                self.pattern(hi);
            }
            PatKind::Wildcard
            | PatKind::Unit
            | PatKind::Int { .. }
            | PatKind::Char(_)
            | PatKind::Bool(_) => {}
        }
    }

    /// `q::name` is parsed as a field access on the single-segment path `q`. Resolve it when `q`
    /// is a module qualifier of this file.
    fn qualified_field(&mut self, e: &Expr) -> Option<Option<Symbol>> {
        let ExprKind::Field { base, name } = &e.kind else {
            return None;
        };
        let ExprKind::Path(p) = &base.kind else {
            return None;
        };
        if !p.is_single() {
            return None;
        }
        self.qualified(p.segments[0], *name)
    }

    fn expr(&mut self, e: &mut Expr) {
        if let Some(resolved) = self.qualified_field(e) {
            if let Some(sym) = resolved {
                let span = e.span;
                e.kind = ExprKind::Path(Path {
                    segments: vec![Ident { sym, span }],
                    span,
                });
            }
            return;
        }
        match &mut e.kind {
            ExprKind::Int(_)
            | ExprKind::Float(_)
            | ExprKind::Str(_)
            | ExprKind::Char(_)
            | ExprKind::Bool(_) => {}
            ExprKind::Path(p) => self.path(p),
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
            ExprKind::StructLit { path, fields, rest } => {
                self.path(path);
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
                    self.locals.push(Vec::new());
                    self.pattern(&mut arm.pat);
                    if let Some(g) = &mut arm.guard {
                        self.expr(g);
                    }
                    self.expr(&mut arm.body);
                    self.locals.pop();
                }
            }
            ExprKind::Loop { body, .. } => self.block(body),
            ExprKind::SizeOf(t) | ExprKind::AlignOf(t) => self.ty(t),
        }
    }
}

/// Drop an `extern` declaration that repeats an earlier one with the same name and the same
/// written signature: the C function is one symbol however many modules declare it. A
/// conflicting redeclaration is kept, so sema reports it as the duplicate it is (E0300).
fn merge_extern_decls(root_items: &mut [Item], extra: &mut [(usize, Item)], interner: &Interner) {
    let mut seen: FxHashMap<Symbol, String> = FxHashMap::default();
    let items = root_items
        .iter_mut()
        .chain(extra.iter_mut().map(|(_, item)| item));
    for item in items {
        let ItemKind::Extern(blk) = &mut item.kind else {
            continue;
        };
        blk.items.retain(|f| {
            let sig = extern_signature(f, interner);
            match seen.get(&f.name.sym) {
                Some(prev) => *prev != sig,
                None => {
                    seen.insert(f.name.sym, sig);
                    true
                }
            }
        });
    }
}

fn extern_signature(f: &FnDecl, interner: &Interner) -> String {
    use wukong_ast::print::print_type;
    let mut s = String::new();
    for g in &f.generics {
        s.push_str(interner.resolve(generic_sym(g)));
        s.push(',');
    }
    s.push('(');
    for p in &f.params {
        s.push_str(&print_type(&p.ty, interner));
        s.push(',');
    }
    s.push(')');
    if let Some(r) = &f.ret {
        s.push_str(&print_type(r, interner));
    }
    s
}
