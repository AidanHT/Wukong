//! Lints: warnings about programs that compile but are probably not what was meant (`W0xxx`).
//!
//! Until this module the compiler produced no warning at all — `Severity::Warning` existed in
//! `wukong_diag` and nothing emitted it — so the mistakes that make a kernel silently compute
//! nothing compiled without a word: `out[i] == a[i] + b[i];` (a comparison written for an
//! assignment), a `mut` output buffer the body never stores to, an accumulator that is updated and
//! never read, code after a `return`, a helper nothing calls.
//!
//! Two entry points, both called by `wukong_driver` after a clean [`crate::check`]:
//!
//! * [`run`] walks the module and returns every lint finding as a `Diagnostic::warning` carrying the
//!   lint's code (`W0001`…). It never errors and never changes [`crate::check`]'s result — the lint
//!   pass is a separate call precisely so every existing consumer of `check` (and every test that
//!   pins its exact diagnostics) is unaffected.
//! * [`apply_levels`] resolves each warning's level — the lint's default, then the command line
//!   (`-A/-W/-D <lint>`, the last one for a lint winning, and `-D warnings` for everything still at
//!   warning level), then the innermost `@allow(..)`/`@warn(..)`/`@deny(..)` attribute whose item,
//!   parameter or statement contains the warning — and drops it, keeps it, or promotes it to an
//!   error. It applies to any diagnostic with a lint code, so the parser's statement-terminator
//!   lints (of every file of the program) and the import loader's `unused_imports` go through the
//!   same levels.
//!
//! Name resolution here is the lint pass's own, mirroring sema's scoping (a block, a `for` and a
//! `match` arm each open a scope; a `let`'s initializer is resolved before its pattern binds), so it
//! reads nothing of sema's beyond the published per-node types in [`SemaResult::types`] and the
//! def map.
//!
//! Every lint errs toward silence. A finding the user cannot act on without an `@allow` is a bug in
//! the lint: `unreachable_code` therefore uses exactly sema's definite-return notion of divergence
//! (a `match` diverges only with a catch-all arm, a diverging `let` initializer does not count), so
//! deleting the reported code can never make sema demand a `return` it did not demand before.

use wukong_ast::*;
use wukong_diag::{Diagnostic, Severity};
use wukong_span::{FxHashSet as HashSet, Interner, Span, Symbol};
use wukong_types::{Scalar, Ty};

use crate::{DefKind, SemaResult};

/// The level a lint is reported at.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum LintLevel {
    /// Not reported.
    Allow,
    /// Reported as a warning; compilation continues.
    Warn,
    /// Reported as an error; compilation fails.
    Deny,
}

/// One lint: the name `-A/-W/-D` and `@allow/@warn/@deny` take, its stable diagnostic code (the
/// catalogue entry `wukongc --explain` prints), and its default level.
#[derive(Clone, Copy, Debug)]
pub struct Lint {
    pub name: &'static str,
    pub code: &'static str,
    pub default: LintLevel,
}

/// Every lint the compiler knows. Codes are permanently stable, like every diagnostic code.
pub const LINTS: &[Lint] = &[
    Lint {
        name: "unused_variables",
        code: "W0001",
        default: LintLevel::Warn,
    },
    Lint {
        name: "unused_mut",
        code: "W0002",
        default: LintLevel::Warn,
    },
    Lint {
        name: "unreachable_code",
        code: "W0003",
        default: LintLevel::Warn,
    },
    Lint {
        name: "unused_result",
        code: "W0004",
        default: LintLevel::Warn,
    },
    Lint {
        name: "dead_code",
        code: "W0005",
        default: LintLevel::Warn,
    },
    Lint {
        name: "unwritten_output",
        code: "W0006",
        default: LintLevel::Warn,
    },
    Lint {
        name: "unknown_lints",
        code: "W0007",
        default: LintLevel::Warn,
    },
    // An `import` no name of its file resolves through. Found by the driver's import loader, whose
    // module-scoped resolution is the only one that knows which import a name came through.
    Lint {
        name: "unused_imports",
        code: "W0008",
        default: LintLevel::Warn,
    },
    // A `match` arm no value can reach, found by the pattern-usefulness analysis that also decides
    // E0405 (`crate::usefulness`).
    Lint {
        name: "unreachable_patterns",
        code: "W0020",
        default: LintLevel::Warn,
    },
    // The parser's statement-terminator lints (`wukong_parser::ParseOutput::lints`); `E0210` is
    // reserved for the day a missing `;` becomes an error.
    Lint {
        name: "missing_semicolon",
        code: "W0030",
        default: LintLevel::Warn,
    },
    Lint {
        name: "ambiguous_continuation",
        code: "W0031",
        default: LintLevel::Warn,
    },
];

/// The three attributes that set a lint level in source (`@allow(unused_variables)`).
pub const LEVEL_ATTRS: &[&str] = &["allow", "warn", "deny"];

/// Whether `name` (an attribute name without the `@`) is a lint-level attribute.
pub fn is_level_attr(name: &str) -> bool {
    LEVEL_ATTRS.contains(&name)
}

fn level_of_attr(name: &str) -> Option<LintLevel> {
    Some(match name {
        "allow" => LintLevel::Allow,
        "warn" => LintLevel::Warn,
        "deny" => LintLevel::Deny,
        _ => return None,
    })
}

/// What a level setting applies to: one lint, or every lint at warning level (`warnings`).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Sel {
    One(usize),
    Warnings,
}

/// Resolve a lint selector: a lint name (`unused_variables`, or `unused-variables` as rustc's
/// command line also accepts), its code (`W0001`, any case), or the group `warnings`.
fn parse_sel(s: &str) -> Option<Sel> {
    let norm = s.replace('-', "_");
    if norm == "warnings" {
        return Some(Sel::Warnings);
    }
    LINTS
        .iter()
        .position(|l| l.name == norm || l.code.eq_ignore_ascii_case(s))
        .map(Sel::One)
}

/// The index in [`LINTS`] of the lint whose code is `code`.
fn lint_of_code(code: &str) -> Option<usize> {
    LINTS.iter().position(|l| l.code == code)
}

/// Lint levels set on the command line, in the order they were given.
#[derive(Clone, Debug, Default)]
pub struct LintLevels {
    cli: Vec<(Sel, LintLevel)>,
}

impl LintLevels {
    /// Record `-A/-W/-D <lint>`. A later setting for the same lint overrides an earlier one, and
    /// `warnings` (the group) applies to every lint whose level is still `Warn`, whatever order it
    /// was given in — so `-D warnings -A dead_code` and `-A dead_code -D warnings` agree.
    pub fn set(&mut self, lint: &str, level: LintLevel) -> Result<(), String> {
        let sel = parse_sel(lint).ok_or_else(|| {
            format!(
                "unknown lint `{lint}` (known: {}, and `warnings` for all of them)",
                LINTS.iter().map(|l| l.name).collect::<Vec<_>>().join(", ")
            )
        })?;
        self.cli.push((sel, level));
        Ok(())
    }
}

/// What the driver knows about how the program is used, which decides `dead_code`'s roots.
#[derive(Clone, Debug, Default)]
pub struct LintOptions {
    /// Functions that are entry points besides `main`: the autodiff root (`--grad-of`, `loss` by
    /// default) under `--emit=grad` / `--train`, which the driver calls by name.
    pub roots: Vec<String>,
    /// Every function is an entry point: the artifact is linked by a host that may call any
    /// exported symbol (`--emit=obj`, `--emit=llvm-ir`). `dead_code` reports nothing.
    pub library: bool,
}

/// Run every lint over `module` (the merged program sema checked) and return the findings as
/// warnings with their lint codes. Levels are NOT applied here; see [`apply_levels`].
pub fn run(
    module: &Module,
    sema: &SemaResult,
    interner: &Interner,
    opts: &LintOptions,
) -> Vec<Diagnostic> {
    let mut diags = Vec::new();
    check_level_attrs(module, interner, &mut diags);
    // The names each function body refers to that resolve to no local, in the order `dead_code`
    // lists the functions (every `fn` with a body, in item order).
    let mut refs: Vec<(Symbol, HashSet<Symbol>)> = Vec::new();
    let mut const_refs: HashSet<Symbol> = HashSet::default();
    for item in &module.items {
        match &item.kind {
            ItemKind::Fn(f) => {
                if let Some(body) = &f.body {
                    let mut cx = FnLint::new(sema, interner);
                    cx.function(f, body);
                    diags.append(&mut cx.diags);
                    refs.push((f.name.sym, cx.globals));
                }
            }
            ItemKind::Const(c) => {
                // A const's initializer is resolved at module scope; what it names stays live.
                let mut cx = FnLint::new(sema, interner);
                cx.expr(&c.value);
                const_refs.extend(cx.globals);
            }
            ItemKind::Enum(e) => {
                for v in &e.variants {
                    if let Some(d) = &v.discriminant {
                        let mut cx = FnLint::new(sema, interner);
                        cx.expr(d);
                        const_refs.extend(cx.globals);
                    }
                }
            }
            // A trait declares signatures only; an `impl` block's methods are separate `Fn` items
            // (`Type.method`) the parser flattened out of it, linted above like any function.
            ItemKind::Struct(_)
            | ItemKind::Import(_)
            | ItemKind::Extern(_)
            | ItemKind::Trait(_)
            | ItemKind::Impl(_)
            | ItemKind::TypeAlias(_) => {}
        }
    }
    dead_code(module, interner, opts, &refs, &const_refs, &mut diags);
    diags
}

// ---- Level attributes -----------------------------------------------------------------------

/// `@allow/@warn/@deny` must name lints: an unknown name (`@allow(unused_varaibles)`) would
/// otherwise silence nothing and say nothing — the attribute-typo failure `E0306` exists for.
fn check_level_attrs(module: &Module, interner: &Interner, diags: &mut Vec<Diagnostic>) {
    let mut scopes = Vec::new();
    collect_scopes(module, interner, &mut scopes, Some(diags));
}

/// One `@allow/@warn/@deny` setting: the extent it governs, the attribute itself, what it selects
/// and the level it sets.
struct Scope {
    extent: Span,
    attr: Span,
    sel: Sel,
    level: LintLevel,
}

/// Collect every lint-level attribute in `module` with the extent of the item, parameter or
/// statement it is written on. With `report`, an argument that names no lint is `W0007`.
fn collect_scopes(
    module: &Module,
    interner: &Interner,
    out: &mut Vec<Scope>,
    mut report: Option<&mut Vec<Diagnostic>>,
) {
    let mut add = |attrs: &[Attr], extent: Span, out: &mut Vec<Scope>| {
        for a in attrs {
            let Some(level) = level_of_attr(interner.resolve(a.name.sym)) else {
                continue;
            };
            if a.args.is_empty() {
                if let Some(d) = report.as_deref_mut() {
                    d.push(
                        Diagnostic::warning(format!(
                            "`@{}` names no lint",
                            interner.resolve(a.name.sym)
                        ))
                        .with_code("W0007")
                        .primary(a.span, "this attribute has no effect")
                        .help(known_lints_help()),
                    );
                }
            }
            for arg in a.args() {
                let (name, span) = match arg {
                    AttrArg::Word(sym, span) => (Some(interner.resolve(*sym)), *span),
                    AttrArg::Int(_, span) | AttrArg::Str(_, span) => (None, *span),
                    AttrArg::KeyValue { key, .. } => (None, key.span),
                };
                match name.and_then(parse_sel) {
                    Some(sel) => out.push(Scope {
                        extent,
                        attr: a.span,
                        sel,
                        level,
                    }),
                    None => {
                        if let Some(d) = report.as_deref_mut() {
                            let msg = match name {
                                Some(n) => format!("unknown lint `{n}`"),
                                None => "expected a lint name".to_string(),
                            };
                            d.push(
                                Diagnostic::warning(msg)
                                    .with_code("W0007")
                                    .primary(span, "this setting has no effect")
                                    .help(known_lints_help()),
                            );
                        }
                    }
                }
            }
        }
    };
    for item in &module.items {
        add(&item.attrs, item.span, out);
        if let ItemKind::Fn(f) = &item.kind {
            for p in &f.params {
                add(&p.attrs, p.span, out);
            }
            if let Some(b) = &f.body {
                walk_block_stmts(b, &mut |s| add(&s.attrs, s.span, out));
            }
        }
    }
}

fn known_lints_help() -> String {
    format!(
        "the lints are: {}; `warnings` selects all of them",
        LINTS.iter().map(|l| l.name).collect::<Vec<_>>().join(", ")
    )
}

/// Call `f` on every statement in `b`, at any depth (statements nest inside blocks, loop bodies,
/// `if`/`match` arms and block expressions).
fn walk_block_stmts(b: &Block, f: &mut dyn FnMut(&Stmt)) {
    for s in &b.stmts {
        f(s);
        match &s.kind {
            StmtKind::Let { init, .. } => {
                if let Some(e) = init {
                    walk_expr_stmts(e, f);
                }
            }
            StmtKind::Assign { target, value, .. } => {
                walk_expr_stmts(target, f);
                walk_expr_stmts(value, f);
            }
            StmtKind::Expr(e) | StmtKind::Defer(e) => walk_expr_stmts(e, f),
            StmtKind::Return(e) | StmtKind::Break(_, e) => {
                if let Some(e) = e {
                    walk_expr_stmts(e, f);
                }
            }
            StmtKind::Continue(_) => {}
            StmtKind::While { cond, body, .. } => {
                walk_expr_stmts(cond, f);
                walk_block_stmts(body, f);
            }
            StmtKind::For { iter, body, .. } => {
                match iter {
                    ForIter::Range {
                        start, end, step, ..
                    } => {
                        walk_expr_stmts(start, f);
                        for e in end.iter().chain(step.iter()) {
                            walk_expr_stmts(e, f);
                        }
                    }
                    ForIter::Expr(e) => walk_expr_stmts(e, f),
                }
                walk_block_stmts(body, f);
            }
        }
    }
    if let Some(t) = &b.tail {
        walk_expr_stmts(t, f);
    }
}

fn walk_expr_stmts(e: &Expr, f: &mut dyn FnMut(&Stmt)) {
    // One callback for both kinds of child: two closures would each need `f` mutably at once.
    for_each_child(e, &mut |c| match c {
        Child::Expr(x) => walk_expr_stmts(x, f),
        Child::Block(b) => walk_block_stmts(b, f),
    });
}

/// A direct child of an expression: a sub-expression or a sub-block.
enum Child<'e> {
    Expr(&'e Expr),
    Block(&'e Block),
}

/// Call `f` on every direct sub-expression and sub-block of `e`, in source order.
fn for_each_child<'e>(e: &'e Expr, f: &mut dyn FnMut(Child<'e>)) {
    match &e.kind {
        ExprKind::Int(_)
        | ExprKind::Float(_)
        | ExprKind::Str(_)
        | ExprKind::Char(_)
        | ExprKind::Bool(_)
        | ExprKind::Path(_)
        | ExprKind::SizeOf(_)
        | ExprKind::AlignOf(_) => {}
        ExprKind::Unary { expr, .. } | ExprKind::Cast { expr, .. } => f(Child::Expr(expr)),
        ExprKind::Field { base, .. } | ExprKind::TupleField { base, .. } => f(Child::Expr(base)),
        ExprKind::Binary { lhs, rhs, .. } => {
            f(Child::Expr(lhs));
            f(Child::Expr(rhs));
        }
        ExprKind::Call { callee, args, .. } => {
            f(Child::Expr(callee));
            for a in args {
                f(Child::Expr(a));
            }
        }
        ExprKind::Index { base, indices } => {
            f(Child::Expr(base));
            for i in indices {
                f(Child::Expr(i));
            }
        }
        ExprKind::StructLit { fields, rest, .. } => {
            for fi in fields {
                f(Child::Expr(&fi.value));
            }
            if let Some(r) = rest {
                f(Child::Expr(r));
            }
        }
        ExprKind::ArrayLit(xs) | ExprKind::TupleLit(xs) => {
            for x in xs {
                f(Child::Expr(x));
            }
        }
        ExprKind::ArrayRepeat { value, count } => {
            f(Child::Expr(value));
            f(Child::Expr(count));
        }
        ExprKind::Block(b) | ExprKind::Loop { body: b, .. } => f(Child::Block(b)),
        ExprKind::If {
            cond,
            then_branch,
            else_branch,
        } => {
            f(Child::Expr(cond));
            f(Child::Block(then_branch));
            if let Some(el) = else_branch {
                f(Child::Expr(el));
            }
        }
        ExprKind::Match { scrutinee, arms } => {
            f(Child::Expr(scrutinee));
            for a in arms {
                if let Some(g) = &a.guard {
                    f(Child::Expr(g));
                }
                f(Child::Expr(&a.body));
            }
        }
        ExprKind::Closure { body, .. } => f(Child::Expr(body)),
    }
}

/// Resolve the level of every lint-coded diagnostic in `diags` and apply it: `Allow` drops the
/// diagnostic, `Warn` keeps it as a warning, `Deny` makes it an error. Diagnostics without a lint
/// code pass through untouched. The result is in source order (file, then position), and the first
/// diagnostic of each lint carries a note saying where its level comes from and how to change it,
/// as rustc's does.
pub fn apply_levels(
    mut diags: Vec<Diagnostic>,
    module: &Module,
    interner: &Interner,
    levels: &LintLevels,
) -> Vec<Diagnostic> {
    // Stable: findings at one position keep the order their pass produced them in.
    diags.sort_by_key(|d| {
        d.primary_span()
            .map_or((u32::MAX, u32::MAX), |s| (s.source.0, s.lo))
    });
    let mut scopes = Vec::new();
    collect_scopes(module, interner, &mut scopes, None);
    let mut explained: Vec<usize> = Vec::new();
    let mut out = Vec::with_capacity(diags.len());
    for mut d in diags {
        let Some(lint) = d.code.and_then(lint_of_code) else {
            out.push(d);
            continue;
        };
        let (level, origin) = resolve_level(lint, d.primary_span(), &scopes, levels);
        let name = LINTS[lint].name;
        match level {
            LintLevel::Allow => continue,
            LintLevel::Warn => {
                d.severity = Severity::Warning;
                if !explained.contains(&lint) {
                    explained.push(lint);
                    d = match origin {
                        Origin::Default => d.note(format!(
                            "`{name}` is on by default; `@allow({name})` on the function or \
                             statement, or `-A {name}` on the command line, silences it"
                        )),
                        Origin::Cli(sel) => {
                            d.note(format!("`-W {sel}` was given on the command line"))
                        }
                        Origin::Attr(at) => d.secondary(at, "the lint level is set here"),
                    };
                }
            }
            LintLevel::Deny => {
                d.severity = Severity::Error;
                d = match origin {
                    Origin::Default => d,
                    Origin::Cli(sel) => d.note(format!("`-D {sel}` was given on the command line")),
                    Origin::Attr(at) => d.secondary(at, "the lint level is set here"),
                };
            }
        }
        out.push(d);
    }
    out
}

/// Where a lint's effective level was decided.
#[derive(Clone, Copy)]
enum Origin {
    Default,
    /// On the command line, by the named lint or by `warnings`.
    Cli(&'static str),
    /// By the attribute at this span.
    Attr(Span),
}

fn resolve_level(
    lint: usize,
    at: Option<Span>,
    scopes: &[Scope],
    levels: &LintLevels,
) -> (LintLevel, Origin) {
    let name = LINTS[lint].name;
    let mut level = LINTS[lint].default;
    let mut origin = Origin::Default;
    for (sel, l) in &levels.cli {
        if *sel == Sel::One(lint) {
            level = *l;
            origin = Origin::Cli(name);
        }
    }
    if let Some(s) = innermost(scopes, at, Sel::One(lint)) {
        level = s.level;
        origin = Origin::Attr(s.attr);
    }
    if level == LintLevel::Warn {
        if let Some(s) = innermost(scopes, at, Sel::Warnings) {
            level = s.level;
            origin = Origin::Attr(s.attr);
        } else if let Some((_, l)) = levels.cli.iter().rev().find(|(s, _)| *s == Sel::Warnings) {
            level = *l;
            origin = Origin::Cli("warnings");
        }
    }
    (level, origin)
}

/// The innermost scope selecting `sel` whose extent contains `at`; among equal extents (two
/// attributes on one item) the later attribute wins.
fn innermost(scopes: &[Scope], at: Option<Span>, sel: Sel) -> Option<&Scope> {
    let at = at?;
    let mut best: Option<&Scope> = None;
    for s in scopes {
        let contains = s.sel == sel
            && s.extent.source == at.source
            && s.extent.lo <= at.lo
            && at.hi <= s.extent.hi;
        if contains && best.is_none_or(|b| s.extent.len() <= b.extent.len()) {
            best = Some(s);
        }
    }
    best
}

// ---- Per-function analysis ------------------------------------------------------------------

/// How an occurrence of a local name uses the binding.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Access {
    /// The name as a whole value — an operand, an argument, an initializer. An aggregate passed or
    /// bound this way may be written by whoever receives it.
    Bare,
    /// Read through a projection (`a[i]`, `s.f`, `*p`) that yields a scalar: a copy, which cannot
    /// alias the binding.
    ScalarRead,
    /// The whole target of `x = …` / `x op= …`.
    Assign,
    /// The root of a projected assignment target (`a[i] = …`, `s.f += …`, `*p = …`); `deref` when
    /// the projection goes through an explicit `*`, which writes the pointee and not the binding.
    StoreThrough { deref: bool },
    /// The root of a `&` / `&mut` operand.
    Borrow,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum BindKind {
    Let,
    Param,
    For,
    Arm,
}

struct Binding {
    name: Symbol,
    span: Span,
    kind: BindKind,
    /// Uses, counting every occurrence except a plain or compound assignment to the whole binding.
    reads: u32,
    /// The whole binding is the target of an assignment.
    assigned: bool,
    /// Something needs the binding to be `mut`: an assignment to it or through a projection of it
    /// (not through an explicit `*`), or a borrow.
    mutated: bool,
    /// The buffer is stored to or escapes (assigned, stored through, passed, bound, borrowed, or
    /// read as a non-scalar).
    written: bool,
    /// Eligible for `unused_mut`: declared `mut`, with a scalar or pointer type — the types whose
    /// only mutation is an assignment to the name itself.
    mut_lint: bool,
    /// Eligible for `unwritten_output`: a `mut` aggregate parameter.
    output: bool,
}

struct FnLint<'a> {
    sema: &'a SemaResult,
    interner: &'a Interner,
    bindings: Vec<Binding>,
    scopes: Vec<Vec<usize>>,
    /// Names resolved to no local: functions, consts, enum and struct names, generics.
    globals: HashSet<Symbol>,
    diags: Vec<Diagnostic>,
}

impl<'a> FnLint<'a> {
    fn new(sema: &'a SemaResult, interner: &'a Interner) -> FnLint<'a> {
        FnLint {
            sema,
            interner,
            bindings: Vec::new(),
            scopes: vec![Vec::new()],
            globals: HashSet::default(),
            diags: Vec::new(),
        }
    }

    fn function(&mut self, f: &FnDecl, body: &Block) {
        for p in &f.params {
            self.type_expr(&p.ty);
            let mut_lint = p.mutable && self.scalar_or_pointer_type(&p.ty);
            let output = self.output_buffer(p);
            self.bind(p.name.sym, p.name.span, BindKind::Param, mut_lint, output);
        }
        if let Some(r) = &f.ret {
            self.type_expr(r);
        }
        self.block(body);
        self.finish();
    }

    // ---- Scopes and bindings ----

    fn bind(&mut self, name: Symbol, span: Span, kind: BindKind, mut_lint: bool, output: bool) {
        let i = self.bindings.len();
        self.bindings.push(Binding {
            name,
            span,
            kind,
            reads: 0,
            assigned: false,
            mutated: false,
            written: false,
            mut_lint,
            output,
        });
        if let Some(scope) = self.scopes.last_mut() {
            scope.push(i);
        }
    }

    fn lookup(&self, name: Symbol) -> Option<usize> {
        self.scopes
            .iter()
            .rev()
            .flat_map(|s| s.iter().rev())
            .copied()
            .find(|&i| self.bindings[i].name == name)
    }

    /// Bind every name `pat` introduces. An or-pattern's alternatives bind the same names, so each
    /// name is bound once. Only a lone identifier can be eligible for `unused_mut`.
    fn bind_pattern(&mut self, pat: &Pattern, kind: BindKind, mut_lint: bool) {
        let mut names = Vec::new();
        pattern_names(pat, &mut names);
        let lone = matches!(pat.kind, PatKind::Ident(_));
        for (sym, span) in names {
            self.bind(sym, span, kind, mut_lint && lone, false);
        }
    }

    fn use_name(&mut self, name: Symbol, access: Access) {
        let Some(i) = self.lookup(name) else {
            self.globals.insert(name);
            return;
        };
        let b = &mut self.bindings[i];
        match access {
            Access::Bare => {
                b.reads += 1;
                b.written = true;
            }
            Access::ScalarRead => b.reads += 1,
            Access::Assign => {
                b.assigned = true;
                b.mutated = true;
                b.written = true;
            }
            Access::StoreThrough { deref } => {
                b.reads += 1;
                b.mutated |= !deref;
                b.written = true;
            }
            Access::Borrow => {
                b.reads += 1;
                b.mutated = true;
                b.written = true;
            }
        }
    }

    fn path(&mut self, p: &Path, access: Access) {
        if p.is_single() {
            self.use_name(p.first().sym, access);
        } else {
            // `Enum::Variant`, `module::item`, `f32x8::load`: never a local; every segment may
            // name an item, which keeps it live for `dead_code`.
            for seg in &p.segments {
                self.globals.insert(seg.sym);
            }
        }
    }

    // ---- Statements ----

    fn block(&mut self, b: &Block) {
        self.scopes.push(Vec::new());
        let mut diverged: Option<Span> = None;
        let mut reported = false;
        for s in &b.stmts {
            if let (Some(at), false) = (diverged, reported) {
                self.unreachable(s.span, at, "statement");
                reported = true;
            }
            self.stmt(s);
            if diverged.is_none() && stmt_diverges(s) {
                diverged = Some(s.span);
            }
        }
        if let Some(t) = &b.tail {
            if let (Some(at), false) = (diverged, reported) {
                self.unreachable(t.span, at, "expression");
            }
            self.expr(t);
        }
        self.scopes.pop();
    }

    fn stmt(&mut self, s: &Stmt) {
        match &s.kind {
            StmtKind::Let {
                pat,
                mutable,
                ty,
                init,
            } => {
                if let Some(t) = ty {
                    self.type_expr(t);
                }
                if let Some(e) = init {
                    self.expr(e);
                }
                let mut_lint = *mutable && self.scalar_or_pointer_let(ty.as_ref(), init.as_ref());
                self.bind_pattern(pat, BindKind::Let, mut_lint);
            }
            StmtKind::Assign { target, value, .. } => {
                self.place(target);
                self.expr(value);
            }
            StmtKind::Expr(e) => {
                self.unused_result(e);
                self.expr(e);
            }
            StmtKind::Return(e) | StmtKind::Break(_, e) => {
                if let Some(e) = e {
                    self.expr(e);
                }
            }
            StmtKind::Continue(_) => {}
            StmtKind::Defer(e) => self.expr(e),
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
                        for e in end.iter().chain(step.iter()) {
                            self.expr(e);
                        }
                    }
                    ForIter::Expr(e) => self.expr(e),
                }
                self.scopes.push(Vec::new());
                self.bind_pattern(pat, BindKind::For, false);
                self.block(body);
                self.scopes.pop();
            }
        }
    }

    /// An assignment target: the whole binding, or the root of a projection chain.
    fn place(&mut self, target: &Expr) {
        match &target.kind {
            ExprKind::Path(p) => self.path(p, Access::Assign),
            _ if is_projection(target) => {
                let deref = chain_has_deref(target);
                self.projection(target, Access::StoreThrough { deref });
            }
            _ => self.expr(target),
        }
    }

    // ---- Expressions ----

    fn expr(&mut self, e: &Expr) {
        match &e.kind {
            ExprKind::Path(p) => self.path(p, Access::Bare),
            _ if is_projection(e) => {
                let scalar = matches!(self.sema.types.get(&e.id), Some(Ty::Scalar(_)));
                let access = if scalar {
                    Access::ScalarRead
                } else {
                    Access::Bare
                };
                self.projection(e, access);
            }
            ExprKind::Unary {
                op: UnOp::Ref | UnOp::RefMut,
                expr,
            } => match &expr.kind {
                ExprKind::Path(p) => self.path(p, Access::Borrow),
                _ if is_projection(expr) => self.projection(expr, Access::Borrow),
                _ => self.expr(expr),
            },
            ExprKind::Call {
                callee,
                generic_args,
                args,
            } => {
                // A method call (`x.m()`, `Type::m()`) names its function only through sema's
                // resolution, so record that function as used for `dead_code`. A call through a
                // trait bound has no single target: every implementation of the method stays live.
                let sema = self.sema;
                if let Some(mc) = sema.methods.calls.get(&e.id) {
                    match mc.target {
                        Some(t) => {
                            self.globals.insert(t);
                        }
                        None => {
                            for ((_, m), f) in &sema.methods.table {
                                if *m == mc.method {
                                    self.globals.insert(*f);
                                }
                            }
                        }
                    }
                }
                // `atomic_add(place, v)` writes its first argument: visit it as an assignment
                // target, or a `mut` buffer written only by atomics reads as never written
                // (`unwritten_output`). Only the builtin -- a local or a user function of that
                // name is an ordinary callee.
                let atomic_place = match &callee.kind {
                    ExprKind::Path(p) if p.is_single() => {
                        let sym = p.first().sym;
                        crate::is_atomic_add(self.interner.resolve(sym))
                            && self.lookup(sym).is_none()
                            && !matches!(
                                self.sema.defs.lookup(sym).map(|d| &d.kind),
                                Some(DefKind::Fn(_))
                            )
                    }
                    _ => false,
                };
                self.expr(callee);
                for g in generic_args {
                    self.type_expr(g);
                }
                for (k, a) in args.iter().enumerate() {
                    if k == 0 && atomic_place {
                        self.place(a);
                    } else {
                        self.expr(a);
                    }
                }
            }
            ExprKind::Cast { expr, ty } => {
                self.expr(expr);
                self.type_expr(ty);
            }
            ExprKind::StructLit { path, fields, rest } => {
                self.path(path, Access::Bare);
                for f in fields {
                    self.expr(&f.value);
                }
                if let Some(r) = rest {
                    self.expr(r);
                }
            }
            ExprKind::SizeOf(t) | ExprKind::AlignOf(t) => self.type_expr(t),
            ExprKind::Block(b) | ExprKind::Loop { body: b, .. } => self.block(b),
            ExprKind::If {
                cond,
                then_branch,
                else_branch,
            } => {
                self.expr(cond);
                self.block(then_branch);
                if let Some(el) = else_branch {
                    self.expr(el);
                }
            }
            ExprKind::Match { scrutinee, arms } => {
                self.expr(scrutinee);
                self.unreachable_arms(scrutinee, arms);
                for arm in arms {
                    self.scopes.push(Vec::new());
                    self.bind_pattern(&arm.pat, BindKind::Arm, false);
                    if let Some(g) = &arm.guard {
                        self.expr(g);
                    }
                    self.expr(&arm.body);
                    self.scopes.pop();
                }
            }
            // A closure body is walked where it is written: a capture is a read of the enclosing
            // binding (so it is never an unused variable), and the closure's parameters are
            // parameters of its own scope. Its `return` leaves the closure only, so nothing here
            // feeds the enclosing block's divergence (`stmt_diverges` does not descend).
            ExprKind::Closure { params, ret, body } => {
                self.scopes.push(Vec::new());
                for p in params {
                    if let Some(t) = &p.ty {
                        self.type_expr(t);
                    }
                    self.bind(p.name.sym, p.name.span, BindKind::Param, false, false);
                }
                if let Some(r) = ret {
                    self.type_expr(r);
                }
                self.expr(body);
                self.scopes.pop();
            }
            ExprKind::Int(_)
            | ExprKind::Float(_)
            | ExprKind::Str(_)
            | ExprKind::Char(_)
            | ExprKind::Bool(_)
            | ExprKind::Unary { .. }
            | ExprKind::Binary { .. }
            | ExprKind::Index { .. }
            | ExprKind::Field { .. }
            | ExprKind::TupleField { .. }
            | ExprKind::ArrayLit(_)
            | ExprKind::ArrayRepeat { .. }
            | ExprKind::TupleLit(_) => {
                for_each_child(e, &mut |c| match c {
                    Child::Expr(x) => self.expr(x),
                    Child::Block(b) => self.block(b),
                });
            }
        }
    }

    /// A projection chain `root[i].f.0` / `*root`: every index is read, and the root (when it is a
    /// name) is used with `access`.
    fn projection(&mut self, e: &Expr, access: Access) {
        let mut cur: &Expr = e;
        loop {
            match &cur.kind {
                ExprKind::Index { base, indices } => {
                    for i in indices {
                        self.expr(i);
                    }
                    cur = &**base;
                }
                ExprKind::Field { base, .. } | ExprKind::TupleField { base, .. } => {
                    cur = &**base;
                }
                ExprKind::Unary {
                    op: UnOp::Deref,
                    expr,
                } => cur = &**expr,
                ExprKind::Path(p) => {
                    self.path(p, access);
                    return;
                }
                _ => {
                    self.expr(cur);
                    return;
                }
            }
        }
    }

    /// Names in a type: a turbofish `f::<n>(..)` may pass a local's value as a dimension, and an
    /// array length is an expression. Over-counting a read here only ever suppresses a warning.
    fn type_expr(&mut self, t: &TypeExpr) {
        match &t.kind {
            TypeKind::Path(p) => {
                if p.is_single() {
                    if let Some(i) = self.lookup(p.first().sym) {
                        self.bindings[i].reads += 1;
                        return;
                    }
                }
                for seg in &p.segments {
                    self.globals.insert(seg.sym);
                }
            }
            TypeKind::Int(_) | TypeKind::Unit => {}
            TypeKind::Pointer { pointee, .. } | TypeKind::Ref { pointee, .. } => {
                self.type_expr(pointee)
            }
            TypeKind::Slice(elem) => self.type_expr(elem),
            TypeKind::Array { elem, len, .. } => {
                self.type_expr(elem);
                self.expr(len);
            }
            TypeKind::Tuple(items) => {
                for it in items {
                    self.type_expr(it);
                }
            }
            TypeKind::Vector { elem, .. } => self.type_expr(elem),
            TypeKind::Tensor { elem, dims, .. } => {
                self.type_expr(elem);
                for d in dims {
                    self.dim(d);
                }
            }
            // Expanded away by the driver before sema; only a direct caller can leave one.
            TypeKind::Generic { args, .. } => {
                for a in args {
                    match a {
                        TypeArg::Type(t) => self.type_expr(t),
                        TypeArg::Dim(d) => self.dim(d),
                    }
                }
            }
            TypeKind::Fn { params, ret } => {
                for p in params {
                    self.type_expr(p);
                }
                if let Some(r) = ret {
                    self.type_expr(r);
                }
            }
        }
    }

    /// The names in one tensor dimension, including every operand of a derived extent (`3*D`).
    fn dim(&mut self, d: &wukong_ast::Dim) {
        match &d.kind {
            DimKind::Named(sym) => match self.lookup(*sym) {
                Some(i) => self.bindings[i].reads += 1,
                None => {
                    self.globals.insert(*sym);
                }
            },
            DimKind::Binary { lhs, rhs, .. } => {
                self.dim(lhs);
                self.dim(rhs);
            }
            DimKind::Int(_) | DimKind::Dynamic => {}
        }
    }

    // ---- Type facts ----

    fn scalar_or_pointer_type(&self, t: &TypeExpr) -> bool {
        match &t.kind {
            TypeKind::Pointer { .. } | TypeKind::Ref { .. } => true,
            TypeKind::Path(p) => {
                p.is_single() && Scalar::from_name(self.interner.resolve(p.first().sym)).is_some()
            }
            _ => false,
        }
    }

    fn scalar_or_pointer_let(&self, ty: Option<&TypeExpr>, init: Option<&Expr>) -> bool {
        match (ty, init) {
            (Some(t), _) => self.scalar_or_pointer_type(t),
            (None, Some(e)) => matches!(
                self.sema.types.get(&e.id),
                Some(Ty::Scalar(_) | Ty::Ptr { .. } | Ty::Ref { .. })
            ),
            (None, None) => false,
        }
    }

    /// A parameter whose job is to be written: `mut` on an aggregate, which is passed by reference
    /// so the caller sees the stores. `mut` is the declaration of that intent — a non-`mut`
    /// aggregate parameter may not be stored through at all (`E0304`). A `*mut T` pointer is not
    /// included: C-style signatures routinely take `*mut` for data they only read.
    fn output_buffer(&self, p: &Param) -> bool {
        if !p.mutable {
            return false;
        }
        match &p.ty.kind {
            TypeKind::Array { .. }
            | TypeKind::Slice(_)
            | TypeKind::Tensor { .. }
            | TypeKind::Tuple(_) => true,
            TypeKind::Path(path) => {
                path.is_single()
                    && matches!(
                        self.sema.defs.lookup(path.first().sym).map(|d| &d.kind),
                        Some(DefKind::Struct(_))
                    )
            }
            TypeKind::Pointer { .. }
            | TypeKind::Int(_)
            | TypeKind::Unit
            | TypeKind::Ref { .. }
            | TypeKind::Vector { .. }
            | TypeKind::Fn { .. }
            | TypeKind::Generic { .. } => false,
        }
    }

    // ---- Findings ----

    fn unreachable(&mut self, span: Span, after: Span, what: &str) {
        self.diags.push(
            Diagnostic::warning(format!("unreachable {what}"))
                .with_code("W0003")
                .primary(span, format!("unreachable {what}"))
                .secondary(after, "any code following this is never executed"),
        );
    }

    /// `unreachable_patterns`: an arm whose every value an earlier unguarded arm already matches.
    /// The verdict is sema's own exhaustiveness analysis, so a match it cannot reason about
    /// reports nothing. An arm the parser synthesized (the fall-through `_` of an `if let`,
    /// `while let` or `let … else`) is never reported: the user wrote no arm there.
    fn unreachable_arms(&mut self, scrutinee: &Expr, arms: &[MatchArm]) {
        let sema = self.sema;
        let Some(ty) = sema.types.get(&scrutinee.id) else {
            return;
        };
        let Some(report) = crate::usefulness::analyze(&sema.defs, self.interner, ty, arms) else {
            return;
        };
        for i in report.unreachable {
            let arm = &arms[i];
            if arm.origin != ArmOrigin::Source {
                continue;
            }
            self.diags.push(
                Diagnostic::warning("unreachable pattern")
                    .with_code("W0020")
                    .primary(
                        arm.pat.span,
                        "every value this matches is matched by an earlier arm",
                    )
                    .help("delete the arm, or move it before the arm that covers it"),
            );
        }
    }

    /// `W0004`: an expression statement whose value is computed and discarded with no effect —
    /// above all `out[i] == a[i] + b[i];`, a comparison where an assignment was meant.
    fn unused_result(&mut self, e: &Expr) {
        let (msg, help): (String, Option<&str>) = match &e.kind {
            ExprKind::Binary { op, lhs, .. } => {
                let g = op.glyph();
                match op {
                    BinOp::Eq if is_place(lhs) => (
                        format!("unused comparison: the result of `{g}` is discarded"),
                        Some("to assign, write `=`"),
                    ),
                    BinOp::Eq | BinOp::Ne | BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge => (
                        format!("unused comparison: the result of `{g}` is discarded"),
                        None,
                    ),
                    BinOp::And | BinOp::Or => (
                        format!("unused logical operation: the result of `{g}` is discarded"),
                        Some(
                            "to evaluate the right-hand side only when the left one decides it, \
                             write an `if`",
                        ),
                    ),
                    _ => (format!("unused result of `{g}`"), None),
                }
            }
            ExprKind::Unary {
                op: op @ (UnOp::Neg | UnOp::Not),
                ..
            } => (format!("unused result of unary `{}`", op.glyph()), None),
            ExprKind::Cast { .. } => ("unused result of a cast".to_string(), None),
            ExprKind::Int(_)
            | ExprKind::Float(_)
            | ExprKind::Str(_)
            | ExprKind::Char(_)
            | ExprKind::Bool(_)
            | ExprKind::Path(_)
            | ExprKind::Field { .. }
            | ExprKind::TupleField { .. }
            | ExprKind::ArrayLit(_)
            | ExprKind::TupleLit(_)
            | ExprKind::ArrayRepeat { .. }
            | ExprKind::StructLit { .. }
            | ExprKind::SizeOf(_)
            | ExprKind::AlignOf(_)
            | ExprKind::Closure { .. } => ("statement has no effect".to_string(), None),
            // A call has effects; an index or a dereference can trap (a bounds or address check
            // is observable); a borrow, a block, a loop and a branch are control flow or effects.
            ExprKind::Unary { .. }
            | ExprKind::Call { .. }
            | ExprKind::Index { .. }
            | ExprKind::Block(_)
            | ExprKind::If { .. }
            | ExprKind::Match { .. }
            | ExprKind::Loop { .. } => return,
        };
        let mut d = Diagnostic::warning(msg)
            .with_code("W0004")
            .primary(e.span, "this value is computed and thrown away");
        if let Some(h) = help {
            d = d.help(h);
        }
        self.diags.push(d);
    }

    /// Report the binding-level lints once the whole function has been walked.
    fn finish(&mut self) {
        for b in &self.bindings {
            let name = self.interner.resolve(b.name);
            // A method's receiver is part of its signature (a trait may require it), not a choice
            // the body can drop.
            if name.starts_with('_') || (name == "self" && b.kind == BindKind::Param) {
                continue;
            }
            let what = match b.kind {
                BindKind::Param => "parameter",
                BindKind::Let | BindKind::For | BindKind::Arm => "variable",
            };
            // An output buffer assigned as a whole has done its job: that write is the use.
            let used = b.reads > 0 || (b.output && b.written);
            if !used {
                let (msg, label) = if b.assigned {
                    (
                        format!("{what} `{name}` is assigned to, but never read"),
                        "every value stored here is overwritten or dropped unread",
                    )
                } else {
                    (format!("unused {what}: `{name}`"), "never used")
                };
                self.diags.push(
                    Diagnostic::warning(msg)
                        .with_code("W0001")
                        .primary(b.span, label)
                        .help(format!(
                            "if this is intentional, prefix it with an underscore: `_{name}`"
                        )),
                );
            } else if b.output && !b.written {
                self.diags.push(
                    Diagnostic::warning(format!("output parameter `{name}` is never written"))
                        .with_code("W0006")
                        .primary(b.span, "declared `mut`, but nothing stores to this buffer")
                        .help(format!(
                            "store the result through it (`{name}[i] = …`), or drop `mut` if the \
                             function only reads it"
                        )),
                );
            }
            if b.mut_lint && !b.mutated {
                self.diags.push(
                    Diagnostic::warning(format!("`{name}` does not need to be mutable"))
                        .with_code("W0002")
                        .primary(b.span, "declared `mut` but never reassigned")
                        .help("remove the `mut`"),
                );
            }
        }
    }
}

// ---- dead_code ------------------------------------------------------------------------------

/// `W0005`: a function of the root file that no entry point can reach. The roots are `main`, the
/// driver's autodiff roots, every `pub` function, every `@export`/`@extern` one, and every function
/// a `const` or an enum discriminant names; a program with no `main`, or one built for a host to
/// link, roots everything. Functions of imported files are not reported — a library is used
/// piecemeal — but their calls still count as uses.
fn dead_code(
    module: &Module,
    interner: &Interner,
    opts: &LintOptions,
    refs: &[(Symbol, HashSet<Symbol>)],
    const_refs: &HashSet<Symbol>,
    diags: &mut Vec<Diagnostic>,
) {
    let fns: Vec<(&FnDecl, &Item)> = module
        .items
        .iter()
        .filter_map(|it| match &it.kind {
            ItemKind::Fn(f) if f.body.is_some() => Some((f, it)),
            _ => None,
        })
        .collect();
    let has_main = fns
        .iter()
        .any(|(f, _)| interner.resolve(f.name.sym) == "main");
    if opts.library || !has_main || fns.len() != refs.len() {
        return;
    }
    // A method of an `impl Trait for Type` block is required by the trait whether or not anything
    // calls it, so it is a root like an exported function.
    let trait_methods: HashSet<Symbol> = module
        .items
        .iter()
        .filter_map(|it| match &it.kind {
            ItemKind::Impl(imp) if imp.trait_ref.is_some() => Some(imp),
            _ => None,
        })
        .flat_map(|imp| imp.methods.iter().map(|m| m.fn_name))
        .collect();
    let mut live: HashSet<Symbol> = HashSet::default();
    let mut work: Vec<usize> = Vec::new();
    for (i, (f, it)) in fns.iter().enumerate() {
        let name = interner.resolve(f.name.sym);
        let exported = it
            .attrs
            .iter()
            .any(|a| matches!(interner.resolve(a.name.sym), "export" | "extern"));
        let root = name == "main"
            || f.is_pub
            || exported
            || opts.roots.iter().any(|r| r == name)
            || const_refs.contains(&f.name.sym)
            || trait_methods.contains(&f.name.sym);
        if root && live.insert(f.name.sym) {
            work.push(i);
        }
    }
    // `refs[i]` holds what `fns[i]`'s body names: both lists are every bodied `fn` in item order.
    while let Some(i) = work.pop() {
        let names = &refs[i].1;
        for (j, (g, _)) in fns.iter().enumerate() {
            if names.contains(&g.name.sym) && live.insert(g.name.sym) {
                work.push(j);
            }
        }
    }
    for (f, it) in &fns {
        if it.span.source == module.span.source && !live.contains(&f.name.sym) {
            diags.push(
                Diagnostic::warning(format!(
                    "function `{}` is never used",
                    interner.resolve(f.name.sym)
                ))
                .with_code("W0005")
                .primary(f.name.span, "no entry point reaches this function"),
            );
        }
    }
}

// ---- Shared helpers -------------------------------------------------------------------------

/// The names a pattern binds, with their spans; an or-pattern contributes each name once.
fn pattern_names(p: &Pattern, out: &mut Vec<(Symbol, Span)>) {
    match &p.kind {
        PatKind::Ident(sym) => {
            if !out.iter().any(|(s, _)| s == sym) {
                out.push((*sym, p.span));
            }
        }
        PatKind::Tuple(subs) | PatKind::Or(subs) => {
            for s in subs {
                pattern_names(s, out);
            }
        }
        PatKind::Variant { fields, .. } => match fields {
            VariantPat::Tuple(subs) => {
                for s in subs {
                    pattern_names(s, out);
                }
            }
            VariantPat::Struct(fps) => {
                for fp in fps {
                    pattern_names(&fp.pat, out);
                }
            }
        },
        PatKind::Struct { fields, .. } => {
            for fp in fields {
                pattern_names(&fp.pat, out);
            }
        }
        PatKind::Binding { name, sub } => {
            if !out.iter().any(|(s, _)| s == name) {
                out.push((*name, p.span));
            }
            pattern_names(sub, out);
        }
        PatKind::Slice {
            prefix,
            rest_binding,
            suffix,
            ..
        } => {
            for s in prefix.iter().chain(rest_binding.as_deref()).chain(suffix) {
                pattern_names(s, out);
            }
        }
        PatKind::Wildcard
        | PatKind::Unit
        | PatKind::Int { .. }
        | PatKind::Char(_)
        | PatKind::Bool(_)
        | PatKind::Path(_)
        | PatKind::Range { .. } => {}
    }
}

fn is_projection(e: &Expr) -> bool {
    matches!(
        e.kind,
        ExprKind::Index { .. }
            | ExprKind::Field { .. }
            | ExprKind::TupleField { .. }
            | ExprKind::Unary {
                op: UnOp::Deref,
                ..
            }
    )
}

fn chain_has_deref(e: &Expr) -> bool {
    match &e.kind {
        ExprKind::Unary {
            op: UnOp::Deref, ..
        } => true,
        ExprKind::Index { base, .. }
        | ExprKind::Field { base, .. }
        | ExprKind::TupleField { base, .. } => chain_has_deref(base),
        _ => false,
    }
}

/// An expression that names a storage location — the left side of an `==` that was probably
/// meant as `=`.
fn is_place(e: &Expr) -> bool {
    matches!(e.kind, ExprKind::Path(_)) || is_projection(e)
}

/// Whether control never continues to the statement after `s`. This is sema's definite-return
/// notion of divergence (`wukong_sema::stmt_diverges`), plus `break` and `continue`, which leave
/// the enclosing block too. Nothing more aggressive is sound for a lint: sema decides whether a
/// function needs a trailing `return` with exactly that rule, so a statement it treats as reachable
/// must never be reported as dead (the user could not delete it).
fn stmt_diverges(s: &Stmt) -> bool {
    match &s.kind {
        StmtKind::Return(_) | StmtKind::Break(..) | StmtKind::Continue(_) => true,
        StmtKind::Expr(e) => expr_diverges(e),
        _ => false,
    }
}

fn expr_diverges(e: &Expr) -> bool {
    match &e.kind {
        ExprKind::Block(b) => block_diverges(b),
        ExprKind::If {
            then_branch,
            else_branch: Some(el),
            ..
        } => block_diverges(then_branch) && expr_diverges(el),
        // Like sema: only a `match` with an unconditional catch-all arm is known to be exhaustive
        // here; a `match` that misses a value at run time falls through.
        ExprKind::Match { arms, .. } => {
            !arms.is_empty()
                && arms.iter().any(|a| {
                    a.guard.is_none() && matches!(a.pat.kind, PatKind::Wildcard | PatKind::Ident(_))
                })
                && arms.iter().all(|a| expr_diverges(&a.body))
        }
        // A `loop` with no `break` anywhere in it never falls through. Any `break` — even one
        // aimed at an inner loop — counts as a way out, which errs toward "reachable".
        ExprKind::Loop { body, .. } => !block_has_break(body),
        _ => false,
    }
}

fn block_diverges(b: &Block) -> bool {
    b.stmts.iter().any(stmt_diverges) || b.tail.as_deref().is_some_and(expr_diverges)
}

fn block_has_break(b: &Block) -> bool {
    let mut found = false;
    walk_block_stmts(b, &mut |s| found |= matches!(s.kind, StmtKind::Break(..)));
    found
}

#[cfg(test)]
mod tests {
    use super::*;
    use wukong_span::SourceId;

    /// Lint `src` (which must check clean) with `levels` and return `(code, severity)` per finding.
    fn lint_with(src: &str, levels: &LintLevels) -> Vec<(&'static str, Severity)> {
        let mut interner = Interner::new();
        let (module, pd) = wukong_parser::parse_module(src, SourceId(0), &mut interner);
        assert!(pd.is_empty(), "parse: {pd:?}");
        let (sema, sd) = crate::check(&module, &interner);
        assert!(sd.iter().all(|d| !d.is_error()), "sema: {sd:?}");
        let found = run(&module, &sema, &interner, &LintOptions::default());
        apply_levels(found, &module, &interner, levels)
            .into_iter()
            .map(|d| (d.code.unwrap(), d.severity))
            .collect()
    }

    fn codes(src: &str) -> Vec<&'static str> {
        lint_with(src, &LintLevels::default())
            .into_iter()
            .map(|(c, _)| c)
            .collect()
    }

    #[test]
    fn clean_kernel_has_no_warnings() {
        let src = "fn add(a: [f32; 4], b: [f32; 4], mut out: [f32; 4]) {
                       for i in 0..4 { out[i] = a[i] + b[i]; }
                   }
                   fn main() -> i32 {
                       let a: [f32; 4] = [1.0; 4];
                       let mut o: [f32; 4] = [0.0; 4];
                       add(a, a, o);
                       let mut s: i32 = 0;
                       for j in 0..3 { s += j; }
                       print(o[0]);
                       return s;
                   }";
        assert_eq!(codes(src), Vec::<&str>::new());
    }

    #[test]
    fn unused_variables() {
        // A never-read `let`, parameter, loop variable and match binding, in source order.
        let src = "fn f(x: i32) -> i32 { let y: i32 = 2; for i in 0..3 { } \
                   return match 4 { n => 1 }; } fn main() -> i32 { return f(1); }";
        assert_eq!(codes(src), vec!["W0001", "W0001", "W0001", "W0001"]);
        // `_`-prefixed names are exempt; a shadowed binding read by its successor is used.
        let src = "fn main() -> i32 { let _x: i32 = 2; let y: i32 = 1; let y: i32 = y + 1; \
                   return y; }";
        assert_eq!(codes(src), Vec::<&str>::new());
        // Updated but never read: the accumulator computes nothing.
        let src = "fn main() -> i32 { let mut acc: i32 = 0; for i in 0..4 { acc += i; } \
                   return 0; }";
        assert_eq!(codes(src), vec!["W0001"]);
    }

    #[test]
    fn unused_mut() {
        let src = "fn main() -> i32 { let mut x: i32 = 1; let mut y: i32 = 1; y = 2; \
                   return x + y; }";
        assert_eq!(codes(src), vec!["W0002"]);
        // A `mut` aggregate `let` may be stored through without `mut` at all, so its `mut` is
        // not judged: never reported.
        let src = "fn main() -> i32 { let mut a: [i32; 2] = [0; 2]; return a[0]; }";
        assert_eq!(codes(src), Vec::<&str>::new());
        // Writing THROUGH a pointer does not need the pointer binding to be `mut`; taking a
        // variable's address does count as needing it.
        let src = "fn main() -> i32 { let mut x: i32 = 0; let mut p: *mut i32 = &mut x; \
                   *p = 3; return x; }";
        assert_eq!(codes(src), vec!["W0002"], "only `p` is needlessly `mut`");
    }

    #[test]
    fn unreachable_code() {
        let src = "fn main() -> i32 { return 1; print(2); return 3; }";
        assert_eq!(codes(src), vec!["W0003"], "one report per block");
        let src = "fn main() -> i32 { let mut i: i32 = 0; loop { i += 1; if i > 3 { break; } } \
                   return i; }";
        assert_eq!(
            codes(src),
            Vec::<&str>::new(),
            "a loop with a break falls through"
        );
        // A `match` without a catch-all arm is not known to diverge — sema demands the trailing
        // `return` after it, so that `return` must not be reported as unreachable.
        let src =
            "fn f(b: bool) -> i32 { match b { true => { return 1; } false => { return 2; } } \
                   return 3; } fn main() -> i32 { return f(true); }";
        assert_eq!(codes(src), Vec::<&str>::new());
        // With a catch-all arm it diverges, and sema agrees.
        let src = "fn f(b: bool) -> i32 { match b { true => { return 1; } _ => { return 2; } } \
                   return 3; } fn main() -> i32 { return f(true); }";
        assert_eq!(codes(src), vec!["W0003"]);
    }

    #[test]
    fn unused_result() {
        // The motivating typo: a comparison written for an assignment. `out` is then never
        // written, which the output-buffer lint reports too.
        let src = "fn k(a: [f32; 4], mut out: [f32; 4]) { for i in 0..4 { out[i] == a[i]; } } \
                   fn main() -> i32 { let a: [f32; 4] = [1.0; 4]; let mut o: [f32; 4] = [0.0; 4]; \
                   k(a, o); print(o[0]); return 0; }";
        let got = codes(src);
        assert!(got.contains(&"W0004"), "{got:?}");
        assert!(got.contains(&"W0006"), "`out` is never written: {got:?}");
        // A call statement and an index (which can trap) are not reported.
        let src = "fn g() -> i32 { return 1; } fn main() -> i32 { let a: [i32; 2] = [0; 2]; \
                   g(); a[1]; return 0; }";
        assert_eq!(codes(src), Vec::<&str>::new());
    }

    #[test]
    fn dead_code() {
        let src = "fn helper() -> i32 { return 1; } fn used() -> i32 { return 2; } \
                   fn rec(n: i32) -> i32 { return rec(n); } \
                   pub fn api() -> i32 { return 3; } fn main() -> i32 { return used(); }";
        assert_eq!(
            codes(src),
            vec!["W0005", "W0005"],
            "helper and the self-recursive rec"
        );
        // No `main`: a library, every function is an entry point.
        assert_eq!(
            codes("fn helper() -> i32 { return 1; }"),
            Vec::<&str>::new()
        );
    }

    #[test]
    fn unwritten_output() {
        // Read but never written: the kernel computes nothing.
        let src = "fn k(a: [f32; 2], mut out: [f32; 2]) -> f32 { return a[0] + out[0]; } \
                   fn main() -> i32 { let a: [f32; 2] = [1.0; 2]; let mut o: [f32; 2] = [0.0; 2]; \
                   print(k(a, o)); return 0; }";
        assert_eq!(codes(src), vec!["W0006"]);
        // Passing the buffer on counts: the callee may write it.
        let src = "fn w(mut o: [f32; 2]) { o[0] = 1.0; } fn k(mut out: [f32; 2]) { w(out); } \
                   fn main() -> i32 { let mut o: [f32; 2] = [0.0; 2]; k(o); print(o[0]); \
                   return 0; }";
        assert_eq!(codes(src), Vec::<&str>::new());
        // A `*mut` pointer parameter is not an output declaration: written through or only read,
        // it is not reported.
        let src = "fn k(p: *mut f32) { *p = 1.0; } fn r(p: *mut f32) -> f32 { return *p; } \
                   fn main() -> i32 { let mut x: f32 = 0.0; k(&mut x); print(r(&mut x)); \
                   return 0; }";
        assert_eq!(codes(src), Vec::<&str>::new());
    }

    #[test]
    fn levels_allow_deny_and_attributes() {
        let src = "fn main() -> i32 { let y: i32 = 2; return 0; }";
        let mut deny = LintLevels::default();
        deny.set("warnings", LintLevel::Deny).unwrap();
        assert_eq!(lint_with(src, &deny), vec![("W0001", Severity::Error)]);
        let mut allow = LintLevels::default();
        allow.set("unused-variables", LintLevel::Allow).unwrap();
        assert_eq!(lint_with(src, &allow), Vec::<(&str, Severity)>::new());
        // `-D warnings` does not resurrect an allowed lint, in either order.
        let mut both = LintLevels::default();
        both.set("W0001", LintLevel::Allow).unwrap();
        both.set("warnings", LintLevel::Deny).unwrap();
        assert_eq!(lint_with(src, &both), Vec::<(&str, Severity)>::new());
        assert!(LintLevels::default()
            .set("unused_varaibles", LintLevel::Allow)
            .is_err());
        // In source: `@allow` on the function wins over the default; `@deny` on a statement.
        let src = "@allow(unused_variables)\nfn main() -> i32 { let y: i32 = 2; return 0; }";
        assert_eq!(codes(src), Vec::<&str>::new());
        let src = "fn main() -> i32 { @deny(unused_variables) let y: i32 = 2; return 0; }";
        assert_eq!(
            lint_with(src, &LintLevels::default()),
            vec![("W0001", Severity::Error)]
        );
        // The innermost attribute wins: `@warn` on the statement inside an `@allow` function.
        let src = "@allow(unused_variables)\nfn main() -> i32 { @warn(unused_variables) \
                   let y: i32 = 2; let z: i32 = 3; return 0; }";
        assert_eq!(codes(src), vec!["W0001"], "only `y`");
        // An unknown lint name is itself a warning.
        let src = "@allow(unused_varaibles)\nfn main() -> i32 { return 0; }";
        assert_eq!(codes(src), vec!["W0007"]);
    }

    /// `atomic_add(place, v)` writes its place: a `mut` buffer parameter updated only by atomics is
    /// written (no `unwritten_output`), and a `mut` local array it targets is needed `mut` (no
    /// `unused_mut`). The control without the atomic still warns.
    #[test]
    fn atomic_add_writes_its_place() {
        let src = "fn count(x: [i32; 8], mut h: [i64; 4]) { \
                       for i in 0..8 { atomic_add(h[x[i] & 3], 1); } } \
                   fn main() -> i32 { let x: [i32; 8] = [1; 8]; let mut h: [i64; 4] = [0; 4]; \
                       count(x, h); let mut c: [i32; 2] = [0; 2]; atomic_add(c[1], 2); \
                       print(h[1]); print(c[1]); return 0; }";
        assert_eq!(codes(src), Vec::<&str>::new());
        let control = "fn count(x: [i32; 8], mut h: [i64; 4]) { \
                           for i in 0..8 { print(h[x[i] & 3]); } } \
                       fn main() -> i32 { let x: [i32; 8] = [1; 8]; \
                           let mut h: [i64; 4] = [0; 4]; count(x, h); return 0; }";
        assert!(codes(control).contains(&"W0006"), "{:?}", codes(control));
    }

    /// Every lint's code is catalogued: `--explain` must describe each one, and the name and the
    /// code select the same lint.
    #[test]
    fn every_lint_is_catalogued() {
        for l in LINTS {
            assert!(
                wukong_diag::explain(l.code).is_some(),
                "{} ({}) has no catalogue entry",
                l.code,
                l.name
            );
            assert_eq!(parse_sel(l.name), parse_sel(l.code));
            assert!(parse_sel(l.name).is_some());
        }
    }
}
