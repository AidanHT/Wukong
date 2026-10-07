//! Attribute legality: the closed v1 `@attribute` set, and `E0306` for everything else.
//!
//! The parser accepts any `@name(args)` and hands it on as an [`Attr`]; before this module nothing
//! ever looked at the name. An attribute the compiler does not implement was therefore *silently
//! discarded* — so `@parellel` (a typo of `@parallel`) compiled to a single-threaded program with no
//! diagnostic at all, which is exactly the failure mode an attribute is supposed to prevent.
//!
//! The set below is closed on purpose. `parallel`, `generic`, `tile`, `extern` and `unchecked` are
//! the ones the language consumes (`@unchecked` removes the slice bounds checks from one function —
//! the explicit opt-out of the safe default, read by `wukong_mir_build`); `inline`, `simd`, `align`
//! and `export` are parsed-and-discarded today and stay in the set so existing programs keep
//! compiling. Anything else is an error, which
//! also means a future attribute cannot be *used* before it is *implemented*.
//!
//! Placement is checked here too, but only where a misplaced attribute would be a silent lie.
//! `@generic` (suppress the kernel recognizers for one function, see `wukong_mir_build`) is read
//! from the **function item**, so accepting it on a statement would promise a per-loop opt-out that
//! nothing implements. `@tile` is read from the function item (into a width queue) or from the
//! `for` statement it sits on, and from nowhere else — so on a `struct`, a parameter, a `let` or a
//! `while` it used to be accepted and dropped, a schedule written on a loop being the sharpest of
//! those. Every other attribute's placement is unchecked in v1 — the ones that are discarded are
//! discarded wherever they appear.
//!
//! This walk is independent of name resolution and types (an attribute names no value), so
//! [`check`] runs before the collection pass and reports every offending attribute in one go.

use wukong_ast::{Attr, Block, Expr, ExprKind, ForIter, Item, ItemKind, Module, Stmt, StmtKind};
use wukong_diag::Diagnostic;
use wukong_span::Interner;

/// Every attribute name the v1 language accepts, in the order the diagnostic lists them: the ones
/// that are (or are being) consumed first, then the four that are parsed and discarded.
pub const KNOWN_ATTRS: &[&str] = &[
    "parallel",
    "generic",
    "tile",
    "extern",
    "unchecked",
    "inline",
    "simd",
    "align",
    "export",
];

/// Attributes that are read from a function item and therefore mean nothing anywhere else.
/// `@unchecked` is per function on purpose: a bounds-check opt-out has to be visible where the
/// unchecked accesses are written, not inherited from a caller.
const FN_ITEM_ONLY: &[&str] = &["generic", "unchecked"];

/// Attributes read from a function item **or** from the `for` statement they sit on, and nowhere
/// else. `@tile` is a loop schedule: `lower_program_with` reads it off the `fn` into a width queue
/// and `lower_for` reads it off the loop. In every other position nothing reads it at all.
const FN_ITEM_OR_FOR: &[&str] = &["tile"];

/// Is `name` (the bare attribute name, without the `@`) part of the v1 known set?
pub fn is_known(name: &str) -> bool {
    KNOWN_ATTRS.contains(&name)
}

/// Where an attribute was written. Only used to phrase the placement diagnostic.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Place {
    /// On a `fn` item — the one position every attribute is legal in.
    FnItem,
    /// On a non-`fn` top-level item (`struct`, `enum`, `const`, `import`, `extern` block).
    Item,
    /// On a function parameter.
    Param,
    /// On a `for` statement inside a body — the one statement `@tile` is read from.
    ForStmt,
    /// On a `while` statement. Split out from [`Place::Stmt`] only so the diagnostic can say so:
    /// `@tile(2) while …` is a schedule written on a loop, and reads as if it applied.
    WhileStmt,
    /// On any other statement inside a body (`@generic let x = …`).
    Stmt,
}

impl Place {
    fn what(self) -> &'static str {
        match self {
            Place::FnItem => "a function",
            Place::Item => "this item",
            Place::Param => "a parameter",
            Place::ForStmt | Place::Stmt => "a statement",
            Place::WhileStmt => "a `while` loop",
        }
    }
}

/// Check every attribute in `module`, pushing one diagnostic per offending attribute onto `diags`.
pub fn check(module: &Module, interner: &Interner, diags: &mut Vec<Diagnostic>) {
    for item in &module.items {
        check_item(item, interner, diags);
    }
}

fn check_item(item: &Item, interner: &Interner, diags: &mut Vec<Diagnostic>) {
    let place = match &item.kind {
        ItemKind::Fn(_) => Place::FnItem,
        _ => Place::Item,
    };
    for a in &item.attrs {
        check_attr(a, place, interner, diags);
    }
    if let ItemKind::Fn(f) = &item.kind {
        for p in &f.params {
            for a in &p.attrs {
                check_attr(a, Place::Param, interner, diags);
            }
        }
        if let Some(body) = &f.body {
            check_block(body, interner, diags);
        }
    }
}

fn check_attr(a: &Attr, place: Place, interner: &Interner, diags: &mut Vec<Diagnostic>) {
    let name = interner.resolve(a.name.sym);
    if !is_known(name) {
        diags.push(
            Diagnostic::error(format!("unknown attribute `@{name}`"))
                .with_code("E0306")
                .primary(a.span, "not a Wukong attribute")
                .help(format!(
                    "the attributes Wukong knows are: @{}",
                    KNOWN_ATTRS.join(", @")
                )),
        );
        return;
    }
    if place != Place::FnItem && FN_ITEM_ONLY.contains(&name) {
        diags.push(
            Diagnostic::error(format!(
                "attribute `@{name}` is not valid on {}",
                place.what()
            ))
            .with_code("E0306")
            .primary(a.span, "only a `fn` item may carry this attribute")
            .help(format!("move `@{name}` onto the enclosing `fn`")),
        );
        return;
    }
    if !matches!(place, Place::FnItem | Place::ForStmt) && FN_ITEM_OR_FOR.contains(&name) {
        diags.push(
            Diagnostic::error(format!(
                "attribute `@{name}` is not valid on {}",
                place.what()
            ))
            .with_code("E0306")
            .primary(a.span, "nothing reads this attribute here")
            .help(format!(
                "write `@{name}` on a `for` statement, or on the enclosing `fn`"
            )),
        );
    }
}

// ---- The body walk. Statements carry attributes, so every block in the function is visited. ----

fn check_block(b: &Block, interner: &Interner, diags: &mut Vec<Diagnostic>) {
    for s in &b.stmts {
        check_stmt(s, interner, diags);
    }
    if let Some(t) = &b.tail {
        check_expr(t, interner, diags);
    }
}

fn check_stmt(s: &Stmt, interner: &Interner, diags: &mut Vec<Diagnostic>) {
    let place = match &s.kind {
        StmtKind::For { .. } => Place::ForStmt,
        StmtKind::While { .. } => Place::WhileStmt,
        _ => Place::Stmt,
    };
    for a in &s.attrs {
        check_attr(a, place, interner, diags);
    }
    match &s.kind {
        StmtKind::Let { init, .. } => {
            if let Some(e) = init {
                check_expr(e, interner, diags);
            }
        }
        StmtKind::Assign { target, value, .. } => {
            check_expr(target, interner, diags);
            check_expr(value, interner, diags);
        }
        StmtKind::Expr(e) | StmtKind::Defer(e) => check_expr(e, interner, diags),
        StmtKind::Return(e) | StmtKind::Break(_, e) => {
            if let Some(e) = e {
                check_expr(e, interner, diags);
            }
        }
        StmtKind::Continue(_) => {}
        StmtKind::While { cond, body, .. } => {
            check_expr(cond, interner, diags);
            check_block(body, interner, diags);
        }
        StmtKind::For { iter, body, .. } => {
            match iter {
                ForIter::Range {
                    start, end, step, ..
                } => {
                    check_expr(start, interner, diags);
                    for e in end.iter().chain(step.iter()) {
                        check_expr(e, interner, diags);
                    }
                }
                ForIter::Expr(e) => check_expr(e, interner, diags),
            }
            check_block(body, interner, diags);
        }
    }
}

/// Recurse through every expression child, because a block — and therefore an attributed statement
/// — can hide anywhere an expression can: a `match` arm, an `if` used as a value, a call argument.
fn check_expr(e: &Expr, interner: &Interner, diags: &mut Vec<Diagnostic>) {
    match &e.kind {
        ExprKind::Int(_)
        | ExprKind::Float(_)
        | ExprKind::Str(_)
        | ExprKind::Char(_)
        | ExprKind::Bool(_)
        | ExprKind::Path(_)
        | ExprKind::SizeOf(_)
        | ExprKind::AlignOf(_) => {}
        ExprKind::Unary { expr, .. } => check_expr(expr, interner, diags),
        ExprKind::Binary { lhs, rhs, .. } => {
            check_expr(lhs, interner, diags);
            check_expr(rhs, interner, diags);
        }
        ExprKind::Call { callee, args, .. } => {
            check_expr(callee, interner, diags);
            for a in args {
                check_expr(a, interner, diags);
            }
        }
        ExprKind::Index { base, indices } => {
            check_expr(base, interner, diags);
            for i in indices {
                check_expr(i, interner, diags);
            }
        }
        ExprKind::Field { base, .. } | ExprKind::TupleField { base, .. } => {
            check_expr(base, interner, diags)
        }
        ExprKind::Cast { expr, .. } => check_expr(expr, interner, diags),
        ExprKind::StructLit { fields, rest, .. } => {
            for f in fields {
                check_expr(&f.value, interner, diags);
            }
            if let Some(r) = rest {
                check_expr(r, interner, diags);
            }
        }
        ExprKind::ArrayLit(items) | ExprKind::TupleLit(items) => {
            for i in items {
                check_expr(i, interner, diags);
            }
        }
        ExprKind::ArrayRepeat { value, count } => {
            check_expr(value, interner, diags);
            check_expr(count, interner, diags);
        }
        ExprKind::Block(b) | ExprKind::Loop { body: b, .. } => check_block(b, interner, diags),
        ExprKind::If {
            cond,
            then_branch,
            else_branch,
        } => {
            check_expr(cond, interner, diags);
            check_block(then_branch, interner, diags);
            if let Some(e) = else_branch {
                check_expr(e, interner, diags);
            }
        }
        ExprKind::Match { scrutinee, arms } => {
            check_expr(scrutinee, interner, diags);
            for arm in arms {
                if let Some(g) = &arm.guard {
                    check_expr(g, interner, diags);
                }
                check_expr(&arm.body, interner, diags);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use wukong_span::SourceId;

    /// Codes reported for `src` by the whole checker (attributes included).
    fn codes(src: &str) -> Vec<&'static str> {
        let mut interner = Interner::new();
        let (module, pdiags) = wukong_parser::parse_module(src, SourceId(0), &mut interner);
        assert!(pdiags.is_empty(), "parse errors: {pdiags:?}");
        let (_res, diags) = crate::check(&module, &interner);
        diags.into_iter().filter_map(|d| d.code).collect()
    }

    #[test]
    fn every_known_attribute_is_accepted_on_a_fn() {
        for name in KNOWN_ATTRS {
            let src = format!("@{name}\nfn f() {{ let x: i32 = 1; }}");
            assert_eq!(codes(&src), Vec::<&str>::new(), "@{name} was rejected");
        }
    }

    #[test]
    fn unknown_attribute_is_rejected() {
        assert_eq!(codes("@bogus\nfn f() {}"), vec!["E0306"]);
    }

    /// The motivating case: a typo of a real attribute must be loud, not silently discarded.
    #[test]
    fn typo_of_a_known_attribute_is_rejected() {
        assert_eq!(codes("@parellel\nfn f() {}"), vec!["E0306"]);
    }

    /// An attribute on a statement, a parameter or a `struct` is walked too — the parser accepts
    /// `@name` in all three positions, so an unknown name must not slip through any of them.
    #[test]
    fn unknown_attribute_is_rejected_in_every_position() {
        assert_eq!(
            codes("fn f() { @bogus let x: i32 = 1; }"),
            vec!["E0306"],
            "statement"
        );
        assert_eq!(codes("fn f(@bogus x: i32) {}"), vec!["E0306"], "parameter");
        assert_eq!(
            codes("@bogus\nstruct S { x: i32 }"),
            vec!["E0306"],
            "struct item"
        );
    }

    /// `@unchecked` (the slice bounds-check opt-out) is read off the function item only; on a loop
    /// or a `let` it would read as a narrower opt-out that nothing implements.
    #[test]
    fn unchecked_is_rejected_off_a_fn_item() {
        assert_eq!(
            codes("fn f(s: []i32) -> i32 { @unchecked let x: i32 = s[0]; return x; }"),
            vec!["E0306"],
            "statement"
        );
        assert_eq!(
            codes(
                "@unchecked
fn f(s: []i32) -> i32 { return s[0]; }"
            ),
            Vec::<&str>::new(),
            "fn item"
        );
    }

    /// `@generic` is read off the function item, so it must be rejected — not ignored — anywhere a
    /// reader would expect it to opt that construct out of kernel recognition.
    #[test]
    fn generic_is_rejected_off_a_fn_item() {
        assert_eq!(
            codes("fn f() { @generic for i in 0..4 { let x: i32 = i; } }"),
            vec!["E0306"],
            "statement"
        );
        assert_eq!(
            codes("@generic\nstruct S { x: i32 }"),
            vec!["E0306"],
            "struct"
        );
    }

    /// `@tile` is read off a function item or off the `for` it sits on. Everywhere else nothing
    /// reads it, so it used to be accepted and dropped -- `@tile(2)` on a `while` being the
    /// sharpest: a schedule written on a loop, silently.
    #[test]
    fn tile_is_rejected_where_nothing_reads_it() {
        assert_eq!(
            codes("fn f(mut o: [i32; 4]) { @tile(2) while true { o[0] = 1; } }"),
            vec!["E0306"],
            "while"
        );
        assert_eq!(
            codes("fn f() { @tile(2) let x: i32 = 1; }"),
            vec!["E0306"],
            "let"
        );
        assert_eq!(
            codes("fn f(@tile(2) x: i32) {}"),
            vec!["E0306"],
            "parameter"
        );
        assert_eq!(
            codes(
                "@tile(64)
struct S { x: i32 }"
            ),
            vec!["E0306"],
            "struct"
        );
    }

    /// ...and stays legal in the two positions that DO read it.
    #[test]
    fn tile_is_accepted_on_a_fn_item_and_on_a_for() {
        assert_eq!(
            codes(
                "@tile(2)
fn f(mut o: [i32; 4]) { for i in 0..4 { o[i] = i; } }"
            ),
            Vec::<&str>::new(),
            "fn item"
        );
        assert_eq!(
            codes("fn f(mut o: [i32; 4]) { @tile(2) for i in 0..4 { o[i] = i; } }"),
            Vec::<&str>::new(),
            "for statement"
        );
    }

    /// A known-and-discarded attribute keeps working in every position it parses in: rejecting
    /// those would be a language change, not a typo check.
    #[test]
    fn discarded_attributes_stay_legal_anywhere() {
        assert_eq!(
            codes("fn f() { @simd @align(32) for i in 0..4 { let x: i32 = i; } }"),
            Vec::<&str>::new()
        );
    }
}
