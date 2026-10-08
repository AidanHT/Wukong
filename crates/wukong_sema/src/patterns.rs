//! Checking the pattern forms beyond literals, tuples and enum variants: struct patterns
//! (`Point { x, y: 0, .. }`), `name @ sub` bindings and array patterns (`[first, .., last]`), and
//! the three statements the parser rewrites into a `match` — `if let`, `while let` and
//! `let … else` (see [`wukong_ast::ArmOrigin`]).
//!
//! * A struct pattern names a declared struct, which must be the scrutinee's type; each listed
//!   field binds through its sub-pattern at the field's declared type, and an unlisted field is
//!   unconstrained.
//! * An array pattern matches a fixed-size array `[T; N]`: without `..` it lists all `N` elements,
//!   with one at most `N`. A `[]T` slice's length is known only at run time, so it is rejected
//!   with a pointer to `s.len()`.
//! * The `else` of a `let … else` must diverge — `return`, `break`, `continue`, or a `loop` with no
//!   way out — since the names the pattern binds do not exist on that path.
//! * A plain `let` pattern must be irrefutable (`E0442`, Rust's E0005): `let Shape::Circle { r } =
//!   s;` would leave `r` unbound for every other shape. The witness names a value it misses.

use wukong_ast::{
    ArmOrigin, Block, Expr, ExprKind, FieldPat, MatchArm, Path, Pattern, Stmt, StmtKind,
};
use wukong_diag::Diagnostic;
use wukong_span::Span;
use wukong_types::Ty;

use crate::{arm_is_catch_all, block_contains_break, usefulness, DefKind, Sema};

impl Sema<'_> {
    /// Type-check and bind a struct pattern `path { fields }` against a value of type `ty`.
    pub(crate) fn check_struct_pattern(
        &mut self,
        path: &Path,
        fields: &[FieldPat],
        ty: &Ty,
        span: Span,
    ) {
        let name = path.segments.last().map(|s| s.sym);
        let decl = name.and_then(|n| match self.defs.lookup(n).map(|d| &d.kind) {
            Some(DefKind::Struct(fs)) => Some(fs.clone()),
            _ => None,
        });
        let (Some(name), Some(decl)) = (name, decl) else {
            let nm = path
                .segments
                .iter()
                .map(|s| self.sym_str(s.sym).to_string())
                .collect::<Vec<_>>()
                .join("::");
            self.error(
                path.span,
                "E0301",
                format!("cannot find struct `{nm}` for this pattern"),
            );
            for fp in fields {
                self.bind_pattern(&fp.pat, &Ty::Unknown);
            }
            return;
        };
        let sname = self.sym_str(name).to_string();
        if !matches!(ty, Ty::Unknown | Ty::Error)
            && *ty != Ty::Named(name)
            && !self.is_generic_ty(ty)
        {
            self.error(
                span,
                "E0401",
                format!(
                    "mismatched pattern: this matches a `{sname}`, but the value is `{}`",
                    ty.display(self.interner)
                ),
            );
        }
        for fp in fields {
            match decl.iter().find(|(f, _)| *f == fp.name) {
                Some((_, t)) => {
                    let t = t.clone();
                    self.bind_pattern(&fp.pat, &t);
                }
                None => {
                    let fname = self.sym_str(fp.name).to_string();
                    self.error(
                        fp.pat.span,
                        "E0301",
                        format!("no field `{fname}` in struct `{sname}`"),
                    );
                    self.bind_pattern(&fp.pat, &Ty::Unknown);
                }
            }
        }
    }

    /// Type-check and bind an array pattern `[prefix.., (..), suffix..]` against `ty`.
    pub(crate) fn check_slice_pattern(
        &mut self,
        prefix: &[Pattern],
        rest: bool,
        suffix: &[Pattern],
        ty: &Ty,
        span: Span,
    ) {
        let listed = (prefix.len() + suffix.len()) as u64;
        let elem = match ty {
            Ty::Array { elem, len, .. } => {
                if (!rest && listed != *len) || (rest && listed > *len) {
                    let what = if rest { "at least " } else { "" };
                    self.error(
                        span,
                        "E0401",
                        format!(
                            "this array pattern matches {what}{listed} element(s), but the array \
                             has {len}"
                        ),
                    );
                }
                (**elem).clone()
            }
            Ty::Unknown | Ty::Error => Ty::Unknown,
            Ty::Slice(_) => {
                self.error(
                    span,
                    "E0401",
                    "an array pattern matches a fixed-size array `[T; N]`; a `[]T` slice's length \
                     is known only at run time — test `s.len()` and index it instead"
                        .to_string(),
                );
                Ty::Unknown
            }
            other => {
                self.error(
                    span,
                    "E0401",
                    format!(
                        "mismatched pattern: an array pattern cannot match a value of type `{}`",
                        other.display(self.interner)
                    ),
                );
                Ty::Unknown
            }
        };
        for p in prefix.iter().chain(suffix) {
            self.bind_pattern(p, &elem);
        }
    }

    /// The `else` of each `let … else` among `arms` must diverge.
    pub(crate) fn check_let_else_arms(&mut self, arms: &[MatchArm]) {
        for arm in arms {
            if arm.origin == ArmOrigin::LetElse && !else_diverges(&arm.body) {
                self.diags.push(
                    Diagnostic::error(
                        "the `else` of a `let … else` must diverge: the names its pattern binds \
                         do not exist on that path",
                    )
                    .with_code("E0401")
                    .primary(arm.body.span, "this block can finish normally")
                    .help("end it with `return`, `break` or `continue`, or a `loop` with no exit"),
                );
            }
        }
    }

    /// `E0442` when the pattern of a plain `let` does not match every value of `ty`.
    pub(crate) fn check_irrefutable_let(&mut self, pat: &Pattern, ty: &Ty) {
        let Some(w) = usefulness::refutable(&self.defs, self.interner, ty, pat) else {
            return;
        };
        let more = if w.more == 0 {
            String::new()
        } else {
            format!(" (and {} more)", w.more)
        };
        self.diags.push(
            Diagnostic::error(format!(
                "refutable pattern in `let`: `{}`{more} not covered",
                w.pattern
            ))
            .with_code("E0442")
            .primary(pat.span, format!("pattern `{}` not covered", w.pattern))
            .help(
                "a `let` pattern must match every value; use `let … else { … }` to leave on the \
                 others, or a `match`",
            ),
        );
    }
}

/// Whether evaluating `e` can never finish normally: a block that reaches a `return`, `break` or
/// `continue`; a `loop` with no `break`; an `if` whose both branches diverge; a `match` with a
/// catch-all whose every arm diverges. Syntactic and conservative (it answers `false` when unsure),
/// like the definite-return check — but a `break`/`continue` counts, because a `let … else` inside
/// a loop may leave it.
fn else_diverges(e: &Expr) -> bool {
    match &e.kind {
        ExprKind::Block(b) => block_diverges(b),
        ExprKind::Loop { body, .. } => !block_contains_break(body),
        ExprKind::If {
            then_branch,
            else_branch: Some(els),
            ..
        } => block_diverges(then_branch) && else_diverges(els),
        ExprKind::Match { arms, .. } => {
            !arms.is_empty()
                && arms.iter().any(arm_is_catch_all)
                && arms.iter().all(|a| else_diverges(&a.body))
        }
        _ => false,
    }
}

fn block_diverges(b: &Block) -> bool {
    b.stmts.iter().any(stmt_diverges) || b.tail.as_deref().is_some_and(else_diverges)
}

fn stmt_diverges(s: &Stmt) -> bool {
    match &s.kind {
        StmtKind::Return(_) | StmtKind::Break(..) | StmtKind::Continue(_) => true,
        StmtKind::Expr(e) => else_diverges(e),
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use wukong_diag::Diagnostic;
    use wukong_span::{Interner, SourceId};

    fn check(src: &str) -> Vec<Diagnostic> {
        let mut interner = Interner::new();
        let (module, pd) = wukong_parser::parse_module(src, SourceId(0), &mut interner);
        assert!(pd.is_empty(), "parse errors: {pd:?}");
        crate::check(&module, &interner).1
    }

    fn codes(src: &str) -> Vec<&'static str> {
        check(src).into_iter().filter_map(|d| d.code).collect()
    }

    #[track_caller]
    fn assert_clean(src: &str) {
        let c = codes(src);
        assert!(c.is_empty(), "unexpected {c:?} for {src:?}");
    }

    #[track_caller]
    fn assert_code(src: &str, code: &str) {
        let c = codes(src);
        assert!(c.contains(&code), "expected {code}, got {c:?} for {src:?}");
    }

    const POINT: &str = "struct Point { x: i32, y: i32 } ";
    const SHAPE: &str = "enum Shape { Circle(i32), Square(i32) } ";

    #[test]
    fn struct_patterns() {
        assert_clean(&format!(
            "{POINT}fn f(p: Point) -> i32 {{ return match p {{ Point {{ x: 0, .. }} => 1, \
             Point {{ x, y }} => x + y }}; }}"
        ));
        assert_clean(&format!(
            "{POINT}fn f(p: Point) -> i32 {{ let Point {{ x: a, y: b }} = p; return a - b; }}"
        ));
        // An unknown field, and a pattern of another type.
        assert_code(
            &format!("{POINT}fn f(p: Point) -> i32 {{ return match p {{ Point {{ z, .. }} => z, _ => 0 }}; }}"),
            "E0301",
        );
        assert_code(
            &format!("{POINT}fn f(t: (i32, i32)) -> i32 {{ return match t {{ Point {{ x, .. }} => x, _ => 0 }}; }}"),
            "E0401",
        );
        // Exhaustiveness sees through the fields and names the uncovered values.
        let d = check(&format!(
            "{POINT}fn f(p: Point) -> i32 {{ return match p {{ Point {{ x: 0, .. }} => 1 }}; }}"
        ));
        assert!(
            d.iter().any(|d| d.code == Some("E0405")
                && d.message
                    .contains("Point { x: i32::MIN..=-1 | 1..=i32::MAX, .. }")),
            "{d:?}"
        );
    }

    #[test]
    fn binding_patterns() {
        assert_clean("fn f(n: i32) -> i32 { return match n { d @ 0..=9 => d, big @ _ => -big }; }");
        assert_clean(
            "fn f(t: (i32, i32)) -> i32 { return match t { whole @ (a, _) => a + whole.1 }; }",
        );
    }

    #[test]
    fn array_patterns() {
        assert_clean(
            "fn f(a: [i32; 4]) -> i32 { return match a { [0, .., 0] => 0, [h, .., t] => h + t }; }",
        );
        assert_clean("fn f(a: [i32; 3]) -> i32 { let [p, q, r] = a; return p + q + r; }");
        // The element count must fit the array; a slice's length is not static.
        assert_code(
            "fn f(a: [i32; 3]) -> i32 { let [p, q] = a; return p + q; }",
            "E0401",
        );
        assert_code(
            "fn f(a: [i32; 2]) -> i32 { return match a { [p, q, r, ..] => p, _ => 0 }; }",
            "E0401",
        );
        assert_code(
            "fn f(s: []i32) -> i32 { return match s { [p, ..] => p, _ => 0 }; }",
            "E0401",
        );
        let d = check(
            "fn f(a: [bool; 2]) -> i32 { return match a { [true, _] => 1, [false, false] => 2 }; }",
        );
        assert!(
            d.iter()
                .any(|d| d.code == Some("E0405") && d.message.contains("[false, true]")),
            "{d:?}"
        );
    }

    #[test]
    fn refutable_let_is_e0442() {
        assert_code(
            &format!("{SHAPE}fn f(s: Shape) -> i32 {{ let Shape::Circle(r) = s; return r; }}"),
            "E0442",
        );
        assert_code(
            "fn f(t: (i32, i32)) -> i32 { let (0, b) = t; return b; }",
            "E0442",
        );
        // A one-variant enum's variant is irrefutable.
        assert_clean("enum W { Only(i32) } fn f(w: W) -> i32 { let W::Only(v) = w; return v; }");
    }

    #[test]
    fn if_let_while_let_and_let_else() {
        assert_clean(&format!(
            "{SHAPE}fn f(s: Shape) -> i32 {{ if let Shape::Circle(r) = s {{ return r; }} \
             else {{ return 0; }} }}"
        ));
        assert_clean(&format!(
            "{SHAPE}fn f(s: Shape) -> i32 {{ let v = if let Shape::Square(w) = s {{ w }} \
             else {{ 1 }}; return v; }}"
        ));
        assert_clean(
            "fn f(n: i32) -> i32 { let mut k = n; while let 1..=9 = k { k = k + 1; } return k; }",
        );
        assert_clean(&format!(
            "{SHAPE}fn f(s: Shape) -> i32 {{ let Shape::Circle(r) = s else {{ return 0; }}; \
             return r; }}"
        ));
        // `continue`/`break` leave a `let … else` inside a loop.
        assert_clean(&format!(
            "{SHAPE}fn f(s: Shape) -> i32 {{ let mut t = 0; for i in 0..3 {{ \
             let Shape::Circle(r) = s else {{ continue; }}; t = t + r + i; }} return t; }}"
        ));
        // An `else` that can finish normally is an error.
        assert_code(
            &format!(
                "{SHAPE}fn f(s: Shape) -> i32 {{ let Shape::Circle(r) = s else {{ print(0); }}; \
                 return r; }}"
            ),
            "E0401",
        );
    }
}
