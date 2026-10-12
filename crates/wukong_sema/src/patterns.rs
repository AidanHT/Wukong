//! Checking the pattern forms beyond literals, tuples and enum variants: struct patterns
//! (`Point { x, y: 0, .. }`), `name @ sub` bindings and array patterns (`[first, .., last]`), and
//! the three statements the parser rewrites into a `match` — `if let`, `while let` and
//! `let … else` (see [`wukong_ast::ArmOrigin`]).
//!
//! * A struct pattern names a declared struct, which must be the scrutinee's type; each listed
//!   field binds through its sub-pattern at the field's declared type, and an unlisted field is
//!   unconstrained.
//! * An array pattern matches a fixed-size array `[T; N]`: without `..` it lists all `N` elements,
//!   with one at most `N`. On a `[]T` slice it is a run-time length test instead, which a `let` or
//!   a `for` head may not make (`E0442`). `rest @ ..` binds the elements the `..` stands for: the
//!   sub-array `[T; N - listed]` of an array, a `[]T` sub-slice of a slice.
//! * The `else` of a `let … else` must diverge — `return`, `break`, `continue`, or a `loop` with no
//!   way out — since the names the pattern binds do not exist on that path.
//! * A plain `let` pattern must be irrefutable (`E0442`, Rust's E0005): `let Shape::Circle { r } =
//!   s;` would leave `r` unbound for every other shape. The witness names a value it misses.
//! * A destructuring `for` head (`for (i, x) in pairs`, `for Point { x, y } in pts`) binds each
//!   element as `let <pat> = a[i];` would, over an array or slice only (`E0401` for a range or any
//!   other iterand), and is never tested, so it must be irrefutable by its form (`E0442`).

use wukong_ast::{
    ArmOrigin, Block, Expr, ExprKind, FieldPat, ForIter, MatchArm, PatKind, Path, Pattern, Stmt,
    StmtKind, VariantPat,
};
use wukong_diag::Diagnostic;
use wukong_span::{Span, Symbol};
use wukong_types::Ty;

use crate::{
    arm_is_catch_all, block_contains_break, usefulness, DefKind, DefMap, Sema, VariantPayload,
};

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

    /// Type-check and bind an array pattern `[prefix.., (rest @ ..), suffix..]` against `ty`, a
    /// fixed-size array `[T; N]` or a `[]T` slice. On an array the element count is checked here;
    /// on a slice it is a run-time length test (`len == listed` without `..`, `len >= listed` with
    /// one). `rest @ ..` binds the elements the `..` stands for: a `[T; N - listed]` sub-array of
    /// an array, a `[]T` sub-slice of a slice — in place, never a copy of the elements.
    pub(crate) fn check_slice_pattern(
        &mut self,
        prefix: &[Pattern],
        rest: bool,
        rest_binding: Option<&Pattern>,
        suffix: &[Pattern],
        ty: &Ty,
        span: Span,
    ) {
        let listed = (prefix.len() + suffix.len()) as u64;
        let (elem, rest_ty) = match ty {
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
                let rest_ty = Ty::Array {
                    elem: elem.clone(),
                    len: len.saturating_sub(listed),
                    shared: false,
                };
                ((**elem).clone(), rest_ty)
            }
            Ty::Slice(elem) => ((**elem).clone(), ty.clone()),
            Ty::Unknown | Ty::Error => (Ty::Unknown, Ty::Unknown),
            other => {
                self.error(
                    span,
                    "E0401",
                    format!(
                        "mismatched pattern: an array pattern cannot match a value of type `{}`",
                        other.display(self.interner)
                    ),
                );
                (Ty::Unknown, Ty::Unknown)
            }
        };
        for p in prefix {
            self.bind_pattern(p, &elem);
        }
        if let Some(b) = rest_binding {
            self.bind_pattern(b, &rest_ty);
        }
        for p in suffix {
            self.bind_pattern(p, &elem);
        }
    }

    /// Type-check and bind an or-pattern `A | B | ..` against `ty`. Each alternative is bound in a
    /// scope of its own, and every alternative must bind the same names (`E0423`) with the same
    /// types (`E0424`): whichever alternative matches, the arm body reads one binding of each
    /// name. The names are then bound once, at the first alternative's types.
    pub(crate) fn check_or_pattern(&mut self, alts: &[Pattern], ty: &Ty) {
        let mut bound: Vec<Vec<(Symbol, Span, Ty)>> = Vec::with_capacity(alts.len());
        for a in alts {
            self.push_scope();
            self.bind_pattern(a, ty);
            let mut names = Vec::new();
            binding_names(a, &mut names);
            let scope = self.scopes.last();
            let typed = names
                .into_iter()
                .map(|(n, sp)| {
                    let t = scope
                        .and_then(|s| s.get(&n))
                        .cloned()
                        .unwrap_or(Ty::Unknown);
                    (n, sp, t)
                })
                .collect();
            self.pop_scope();
            bound.push(typed);
        }
        // Report each name once, against the first alternative that binds it. A name bound at two
        // types is then bound as `Error`, so its uses do not cascade into mismatches of their own.
        let mut reported: Vec<Symbol> = Vec::new();
        let mut clashed: Vec<Symbol> = Vec::new();
        for (i, names) in bound.iter().enumerate() {
            for (n, sp, t) in names {
                if reported.contains(n) {
                    continue;
                }
                let nm = self.sym_str(*n).to_string();
                if let Some(j) = bound.iter().position(|o| !o.iter().any(|(m, ..)| m == n)) {
                    reported.push(*n);
                    self.diags.push(
                        Diagnostic::error(format!(
                            "variable `{nm}` is not bound in all alternatives of this or-pattern"
                        ))
                        .with_code("E0423")
                        .primary(
                            alts[j].span,
                            format!("this alternative does not bind `{nm}`"),
                        )
                        .secondary(*sp, format!("`{nm}` is bound here"))
                        .help(format!(
                            "whichever alternative matches, the arm reads one `{nm}`, so every \
                             alternative must bind it; bind it in each, or match the \
                             alternatives in separate arms"
                        )),
                    );
                    continue;
                }
                let clash = bound[i + 1..].iter().find_map(|o| {
                    o.iter().find(|(m, _, u)| {
                        m == n && !t.is_unknown() && !u.is_unknown() && !same_ty(t, u)
                    })
                });
                if let Some((_, osp, ot)) = clash {
                    reported.push(*n);
                    clashed.push(*n);
                    self.diags.push(
                        Diagnostic::error(format!(
                            "variable `{nm}` has type `{}` in one alternative of this or-pattern \
                             and `{}` in another",
                            t.display(self.interner),
                            ot.display(self.interner)
                        ))
                        .with_code("E0424")
                        .primary(
                            *osp,
                            format!("`{nm}` is `{}` here", ot.display(self.interner)),
                        )
                        .secondary(
                            *sp,
                            format!("`{nm}` is `{}` here", t.display(self.interner)),
                        )
                        .help(
                            "the arm reads one binding of each name, so every alternative must \
                             bind it at the same type; match the alternatives in separate arms",
                        ),
                    );
                }
            }
        }
        let mut done: Vec<Symbol> = Vec::new();
        for names in &bound {
            for (n, _, t) in names {
                if !done.contains(n) {
                    done.push(*n);
                    let t = if clashed.contains(n) {
                        Ty::Error
                    } else {
                        t.clone()
                    };
                    self.bind(*n, t);
                }
            }
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

    /// `E0442` when the pattern of a plain `let` or a `for` head does not match every value of
    /// `ty`, naming a value it misses. Whether it reported: the analysis is silent on a pattern it
    /// cannot place, which a `for` head ([`Sema::check_for_pattern`]) must still reject.
    pub(crate) fn check_irrefutable(&mut self, pat: &Pattern, ty: &Ty, site: PatSite) -> bool {
        let Some(w) = usefulness::refutable(&self.defs, self.interner, ty, pat) else {
            return false;
        };
        let more = if w.more == 0 {
            String::new()
        } else {
            format!(" (and {} more)", w.more)
        };
        self.diags.push(
            Diagnostic::error(format!(
                "refutable pattern in {}: `{}`{more} not covered",
                site.what(),
                w.pattern
            ))
            .with_code("E0442")
            .primary(pat.span, format!("pattern `{}` not covered", w.pattern))
            .help(site.help()),
        );
        true
    }

    /// The destructuring pattern of `for <pat> in <iter>` (anything but a name or `_`), whose
    /// element type is `elem`. It binds as `let <pat> = a[i];` would, at each element's place,
    /// and it is never tested, so it must be irrefutable by its FORM: built only from names, `_`,
    /// tuples, structs, array patterns, `name @ sub` and the variant of a one-variant enum. A
    /// literal, range, path, or-pattern or multi-variant enum variant anywhere inside is `E0442`,
    /// even where the analysis would prove it total (`0..=255` on a `u8`) or cannot place it
    /// (`(0, x)` on an `(f32, i32)`): a rule by form leaves nothing for lowering to test.
    pub(crate) fn check_for_pattern(&mut self, pat: &Pattern, iter: &ForIter, elem: &Ty) {
        let it = match iter {
            ForIter::Range { .. } => {
                self.diags.push(
                    Diagnostic::error(
                        "a range `for` binds one integer and cannot be destructured".to_string(),
                    )
                    .with_code("E0401")
                    .primary(pat.span, "a range yields a single integer")
                    .help("bind a name (`for i in 0..n`) and index with it"),
                );
                return;
            }
            ForIter::Expr(it) => it,
        };
        match self.types.get(&it.id) {
            Some(Ty::Array { .. } | Ty::Slice(_) | Ty::Unknown | Ty::Error) | None => {}
            // A `str` is laid out as a `[]u8`, but its element is a byte: nothing to destructure.
            Some(Ty::Str) => {
                self.diags.push(
                    Diagnostic::error(
                        "a `for` over a `str` yields each byte as a `u8`, which cannot be \
                         destructured"
                            .to_string(),
                    )
                    .with_code("E0401")
                    .primary(pat.span, "this pattern destructures each byte")
                    .help("bind a name (`for b in s`); each `b` is a `u8`"),
                );
                return;
            }
            Some(other) => {
                let shown = other.display(self.interner);
                self.diags.push(
                    Diagnostic::error(format!(
                        "only a `for` over an array or slice can destructure its element; this \
                         iterates a `{shown}`"
                    ))
                    .with_code("E0401")
                    .primary(pat.span, "this pattern destructures each element")
                    .help("bind a name (`for x in …`) and destructure it in the loop body"),
                );
                return;
            }
        }
        if !self.check_pattern_shape(pat, elem) {
            return;
        }
        // A form that tests the element, or an array pattern that lists an element of a `[]T`
        // (it tests the slice's run-time length).
        let (r, what) = match refutable_form(pat, &self.defs) {
            Some(r) => (
                r,
                "a literal, range, path, or-pattern or enum variant tests the element",
            ),
            None => match self.slice_length_test(pat, elem) {
                Some(r) => (r, "an array pattern on a `[]T` tests its length"),
                None => return,
            },
        };
        if !self.check_irrefutable(pat, elem, PatSite::ForHead) {
            self.diags.push(
                Diagnostic::error(
                    "refutable pattern in `for` head: this pattern tests each element",
                )
                .with_code("E0442")
                .primary(r.span, what)
                .help(PatSite::ForHead.help()),
            );
        }
    }

    /// The first array pattern inside `pat` (matched against `ty`) that lists an element of a
    /// `[]T` slice, and so tests the slice's length at run time; `[..]` and `[rest @ ..]` match
    /// every slice. Struct, variant, tuple and array sub-patterns reach their field types.
    fn slice_length_test<'p>(&self, pat: &'p Pattern, ty: &Ty) -> Option<&'p Pattern> {
        match (&pat.kind, ty) {
            (PatKind::Slice { prefix, suffix, .. }, Ty::Slice(_))
                if !prefix.is_empty() || !suffix.is_empty() =>
            {
                Some(pat)
            }
            (PatKind::Slice { prefix, suffix, .. }, Ty::Array { elem, .. }) => prefix
                .iter()
                .chain(suffix)
                .find_map(|p| self.slice_length_test(p, elem)),
            (PatKind::Tuple(subs), Ty::Tuple(tys)) => subs
                .iter()
                .zip(tys)
                .find_map(|(p, t)| self.slice_length_test(p, t)),
            (PatKind::Binding { sub, .. }, _) => self.slice_length_test(sub, ty),
            (PatKind::Struct { fields, .. }, Ty::Named(n)) => {
                let Some(DefKind::Struct(decl)) = self.defs.lookup(*n).map(|d| &d.kind) else {
                    return None;
                };
                fields.iter().find_map(|fp| {
                    let (_, t) = decl.iter().find(|(f, _)| *f == fp.name)?;
                    self.slice_length_test(&fp.pat, t)
                })
            }
            (PatKind::Variant { path, fields }, _) => {
                let payload = variant_payload(path, &self.defs)?;
                match (fields, payload) {
                    (VariantPat::Tuple(subs), VariantPayload::Tuple(tys)) => subs
                        .iter()
                        .zip(&tys)
                        .find_map(|(p, t)| self.slice_length_test(p, t)),
                    (VariantPat::Struct(fps), VariantPayload::Struct(named)) => {
                        fps.iter().find_map(|fp| {
                            let (_, t) = named.iter().find(|(n, _)| *n == fp.name)?;
                            self.slice_length_test(&fp.pat, t)
                        })
                    }
                    _ => None,
                }
            }
            _ => None,
        }
    }

    /// `E0401` for a tuple pattern against a value that is not a tuple of its arity, anywhere
    /// inside `pat` matched against `ty` (struct, variant and array sub-patterns reach their field
    /// types; the other mismatches have their own reports in [`Sema::bind_pattern`]). Whether the
    /// pattern is well-shaped.
    fn check_pattern_shape(&mut self, pat: &Pattern, ty: &Ty) -> bool {
        match &pat.kind {
            PatKind::Tuple(subs) => match ty {
                Ty::Tuple(tys) if tys.len() == subs.len() => {
                    let tys = tys.clone();
                    let mut ok = true;
                    for (p, t) in subs.iter().zip(&tys) {
                        ok &= self.check_pattern_shape(p, t);
                    }
                    ok
                }
                Ty::Unknown | Ty::Error => true,
                other => {
                    let shown = other.display(self.interner);
                    self.diags.push(
                        Diagnostic::error(format!(
                            "mismatched pattern: a tuple pattern cannot match `{shown}`"
                        ))
                        .with_code("E0401")
                        .primary(
                            pat.span,
                            format!("this tuple pattern has {} element(s)", subs.len()),
                        ),
                    );
                    false
                }
            },
            PatKind::Binding { sub, .. } => self.check_pattern_shape(sub, ty),
            PatKind::Slice { prefix, suffix, .. } => {
                let (Ty::Array { elem, .. } | Ty::Slice(elem)) = ty else {
                    return true;
                };
                let elem = (**elem).clone();
                let mut ok = true;
                for p in prefix.iter().chain(suffix) {
                    ok &= self.check_pattern_shape(p, &elem);
                }
                ok
            }
            PatKind::Struct { fields, .. } => {
                let Ty::Named(n) = ty else {
                    return true;
                };
                let Some(DefKind::Struct(decl)) = self.defs.lookup(*n).map(|d| &d.kind) else {
                    return true;
                };
                let decl = decl.clone();
                let mut ok = true;
                for fp in fields {
                    if let Some((_, t)) = decl.iter().find(|(f, _)| *f == fp.name) {
                        ok &= self.check_pattern_shape(&fp.pat, t);
                    }
                }
                ok
            }
            PatKind::Variant { path, fields } => {
                let Some(payload) = variant_payload(path, &self.defs) else {
                    return true;
                };
                let mut ok = true;
                match (fields, payload) {
                    (VariantPat::Tuple(subs), VariantPayload::Tuple(tys)) => {
                        for (p, t) in subs.iter().zip(&tys) {
                            ok &= self.check_pattern_shape(p, t);
                        }
                    }
                    (VariantPat::Struct(fps), VariantPayload::Struct(named)) => {
                        for fp in fps {
                            if let Some((_, t)) = named.iter().find(|(n, _)| *n == fp.name) {
                                ok &= self.check_pattern_shape(&fp.pat, t);
                            }
                        }
                    }
                    _ => {}
                }
                ok
            }
            _ => true,
        }
    }
}

/// Where an irrefutable pattern is required, for E0442's wording.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum PatSite {
    /// A plain `let` (`let … else` is the refutable form).
    Let,
    /// The pattern of `for <pat> in <array or slice>`.
    ForHead,
}

impl PatSite {
    fn what(self) -> &'static str {
        match self {
            PatSite::Let => "`let`",
            PatSite::ForHead => "`for` head",
        }
    }

    fn help(self) -> &'static str {
        match self {
            PatSite::Let => {
                "a `let` pattern must match every value; use `let … else { … }` to leave on the \
                 others, or a `match`"
            }
            PatSite::ForHead => {
                "a `for` pattern must match every element; match inside the loop body instead"
            }
        }
    }
}

/// The names `p` binds, each once, in source order, with the span of the binding (an or-pattern
/// nested inside contributes its first alternative's, which [`Sema::check_or_pattern`] checks
/// against the others).
fn binding_names(p: &Pattern, out: &mut Vec<(Symbol, Span)>) {
    fn push(n: Symbol, sp: Span, out: &mut Vec<(Symbol, Span)>) {
        if !out.iter().any(|(m, _)| *m == n) {
            out.push((n, sp));
        }
    }
    match &p.kind {
        PatKind::Ident(n) => push(*n, p.span, out),
        PatKind::Binding { name, sub } => {
            push(*name, p.span, out);
            binding_names(sub, out);
        }
        PatKind::Tuple(subs) => subs.iter().for_each(|s| binding_names(s, out)),
        PatKind::Or(alts) => {
            if let Some(a) = alts.first() {
                binding_names(a, out);
            }
        }
        PatKind::Variant { fields, .. } => match fields {
            VariantPat::Tuple(subs) => subs.iter().for_each(|s| binding_names(s, out)),
            VariantPat::Struct(fps) => fps.iter().for_each(|fp| binding_names(&fp.pat, out)),
        },
        PatKind::Struct { fields, .. } => fields.iter().for_each(|fp| binding_names(&fp.pat, out)),
        PatKind::Slice {
            prefix,
            rest_binding,
            suffix,
            ..
        } => prefix
            .iter()
            .chain(rest_binding.as_deref())
            .chain(suffix)
            .for_each(|s| binding_names(s, out)),
        PatKind::Wildcard
        | PatKind::Unit
        | PatKind::Int { .. }
        | PatKind::Char(_)
        | PatKind::Bool(_)
        | PatKind::Path(_)
        | PatKind::Range { .. } => {}
    }
}

/// Whether two alternatives' bindings have one type. A literal-typed binding (`Unknown` until
/// inference) never clashes; nor does anything sema reported already (`Error`).
fn same_ty(a: &Ty, b: &Ty) -> bool {
    a == b || matches!(a, Ty::Error) || matches!(b, Ty::Error)
}

/// The payload of the variant `Enum::Variant` names, when it resolves.
fn variant_payload(path: &Path, defs: &DefMap) -> Option<VariantPayload> {
    let (Some(ename), Some(vname)) = (path.segments.first(), path.segments.last()) else {
        return None;
    };
    match defs.lookup(ename.sym).map(|d| &d.kind) {
        Some(DefKind::Enum(vs)) => vs
            .iter()
            .find(|v| v.name == vname.sym)
            .map(|v| v.payload.clone()),
        _ => None,
    }
}

/// The first sub-pattern of `pat` that makes it refutable by its FORM (see
/// [`Sema::check_for_pattern`]): a literal, range, path or or-pattern, or the variant of an enum
/// with more than one variant (or of a name that is not a declared enum).
fn refutable_form<'p>(pat: &'p Pattern, defs: &DefMap) -> Option<&'p Pattern> {
    match &pat.kind {
        PatKind::Wildcard | PatKind::Ident(_) | PatKind::Unit => None,
        PatKind::Int { .. }
        | PatKind::Char(_)
        | PatKind::Bool(_)
        | PatKind::Range { .. }
        | PatKind::Path(_)
        | PatKind::Or(_) => Some(pat),
        PatKind::Tuple(subs) => subs.iter().find_map(|p| refutable_form(p, defs)),
        PatKind::Binding { sub, .. } => refutable_form(sub, defs),
        PatKind::Struct { fields, .. } => {
            fields.iter().find_map(|fp| refutable_form(&fp.pat, defs))
        }
        PatKind::Slice { prefix, suffix, .. } => prefix
            .iter()
            .chain(suffix)
            .find_map(|p| refutable_form(p, defs)),
        PatKind::Variant { path, fields } => {
            let one_variant = path.segments.first().is_some_and(|e| {
                matches!(defs.lookup(e.sym).map(|d| &d.kind), Some(DefKind::Enum(vs)) if vs.len() == 1)
            });
            if !one_variant {
                return Some(pat);
            }
            match fields {
                VariantPat::Tuple(subs) => subs.iter().find_map(|p| refutable_form(p, defs)),
                VariantPat::Struct(fps) => fps.iter().find_map(|fp| refutable_form(&fp.pat, defs)),
            }
        }
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
        // The element count must fit the array.
        assert_code(
            "fn f(a: [i32; 3]) -> i32 { let [p, q] = a; return p + q; }",
            "E0401",
        );
        assert_code(
            "fn f(a: [i32; 2]) -> i32 { return match a { [p, q, r, ..] => p, _ => 0 }; }",
            "E0401",
        );
        // Neither a `str` nor a scalar is an array.
        assert_code(
            "fn f(s: str) -> i32 { return match s { [b, ..] => b as i32, _ => 0 }; }",
            "E0401",
        );
        assert_code(
            "fn f(x: i32) -> i32 { return match x { [b, ..] => b, _ => 0 }; }",
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

    /// An array pattern on a `[]T` tests its length at run time; `rest @ ..` binds the elements
    /// the `..` stands for: a sub-slice of a slice, a `[T; N - listed]` sub-array of an array.
    #[test]
    fn slice_patterns_and_rest_bindings() {
        assert_clean(
            "fn f(s: []i32) -> i32 { return match s { [] => 0, [x] => x, [a, .., z] => a + z }; }",
        );
        assert_clean(
            "fn sum(s: []i32) -> i32 { return match s { [] => 0, [h, t @ ..] => h + sum(t) }; }",
        );
        assert_clean(
            "struct P { x: i32 } fn f(s: []P) -> i32 { return match s { [P { x: 0 }, ..] => 1,              [p, .., q] => p.x + q.x, _ => 2 }; }",
        );
        // A sub-array has the remaining length, `[i32; 2]` out of a `[i32; 4]`, checked like any
        // other array length (a shape error, `E0501`).
        assert_clean(
            "fn g(m: [i32; 2]) -> i32 { return m[0] + m[1]; }              fn f(a: [i32; 4]) -> i32 { let [x, mid @ .., y] = a; return x + g(mid) + y; }",
        );
        assert_code(
            "fn g(m: [i32; 3]) -> i32 { return m[0]; }              fn f(a: [i32; 4]) -> i32 { let [x, mid @ .., y] = a; return x + g(mid) + y; }",
            "E0501",
        );
        // A sub-slice is a `[]T`.
        assert_code(
            "fn f(s: []i32) -> i32 { return match s { [_, t @ ..] => t, _ => 0 }; }",
            "E0401",
        );
        // A refutable slice pattern in `let` or a `for` head names the slice it misses.
        let d = check("fn f(s: []i32) -> i32 { let [a, r @ ..] = s; return a + r[0]; }");
        assert!(
            d.iter()
                .any(|d| d.code == Some("E0442") && d.message.contains("`[]` not covered")),
            "{d:?}"
        );
        assert_code(
            "fn f(ss: [[]i32; 2]) -> i32 { let mut t = 0; for [a, ..] in ss { t = t + a; }              return t; }",
            "E0442",
        );
        // `[..]` and `[all @ ..]` match every slice.
        assert_clean(
            "fn f(ss: [[]i32; 2]) -> i64 { let mut t: i64 = 0; for [all @ ..] in ss {              t = t + all.len(); } return t; }",
        );
        assert_clean("fn f(s: []i32) -> i64 { let [..] = s; let [r @ ..] = s; return r.len(); }");
    }

    /// Every alternative of an or-pattern binds the same names (E0423) at the same types (E0424).
    #[test]
    fn or_pattern_bindings() {
        assert_clean(&format!(
            "{SHAPE}fn f(s: Shape) -> i32 {{ return match s {{ Shape::Circle(r) |              Shape::Square(r) => r }}; }}"
        ));
        assert_clean(
            "fn f(t: (i32, i32)) -> i32 { return match t { (0, v) | (v, 0) => v, (a, b) => a + b              }; }",
        );
        assert_clean("fn f(s: []i32) -> i32 { return match s { [x] | [x, _] => x, _ => 0 }; }");
        let d = check(
            "enum E { A(i32), B } fn f(e: E) -> i32 { return match e { E::A(x) | E::B => x }; }",
        );
        assert!(
            d.iter()
                .any(|d| d.code == Some("E0423") && d.message.contains("`x` is not bound")),
            "{d:?}"
        );
        // A name bound at two types is reported once, and its use does not cascade.
        let d = check(
            "enum N { I(i32), R(f32) } fn f(n: N) -> f32 { return match n { N::I(v) | N::R(v) =>              v }; }",
        );
        let codes: Vec<_> = d.iter().filter_map(|d| d.code).collect();
        assert_eq!(codes, ["E0424"], "{d:?}");
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

    /// A later `let` of an `if let` / `while let` chain binds for the operands after it and the
    /// block, and nowhere else.
    #[test]
    fn let_chains_with_several_lets() {
        assert_clean(&format!(
            "{SHAPE}fn f(a: Shape, b: Shape) -> i32 {{ if let Shape::Circle(r) = a && r > 1 && \
             let Shape::Square(w) = b && w > r {{ return r * 10 + w; }} return 0; }}"
        ));
        assert_clean(
            "fn f(s: []i32, t: (i32, i32)) -> i32 { let mut k = 0; while let [h, ..] = s && \
             let (0, v) = t && k < h { k = k + v; } return k; }",
        );
        // The second `let`'s names do not reach the `else`.
        assert_code(
            &format!(
                "{SHAPE}fn f(a: Shape, b: Shape) -> i32 {{ if let Shape::Circle(r) = a && \
                 let Shape::Square(w) = b {{ return r + w; }} else {{ return w; }} }}"
            ),
            "E0301",
        );
        // A later `let`'s pattern is checked against its own scrutinee.
        assert_code(
            &format!(
                "{SHAPE}{POINT}fn f(a: Shape, p: Point) -> i32 {{ if let Shape::Circle(r) = a && \
                 let Shape::Square(w) = p {{ return r + w; }} return 0; }}"
            ),
            "E0401",
        );
    }

    #[test]
    fn destructuring_for_heads() {
        // Tuples, structs, arrays, `@`, nesting and a one-variant enum, over arrays and slices.
        assert_clean(
            "fn f(ps: [(i32, f32); 2]) -> f32 { let mut t = 0.0; for (i, x) in ps { \
             t = t + x * (i as f32); } return t; }",
        );
        assert_clean(&format!(
            "{POINT}fn f(ps: []Point) -> i32 {{ let mut t = 0; for Point {{ x, y }} in ps {{ \
             t = t + x - y; }} return t; }}"
        ));
        assert_clean(&format!(
            "{POINT}fn f(ps: [Point; 2]) -> i32 {{ let mut t = 0; for Point {{ x, .. }} in ps {{ \
             t = t + x; }} return t; }}"
        ));
        assert_clean(
            "fn f(rs: [[i32; 3]; 2]) -> i32 { let mut t = 0; for [a, .., z] in rs { \
             t = t + a * z; } return t; }",
        );
        assert_clean(
            "fn f(ts: [((i32, i32), i32); 2]) -> i32 { let mut t = 0; for ((a, b), c) in ts { \
             t = t + a + b + c; } return t; }",
        );
        assert_clean(
            "fn f(ts: [(i32, i32); 2]) -> i32 { let mut t = 0; for w @ (a, _) in ts { \
             t = t + a + w.1; } return t; }",
        );
        assert_clean(
            "enum W { Only(i32) } fn f(ws: [W; 2]) -> i32 { let mut t = 0; \
             for W::Only(v) in ws { t = t + v; } return t; }",
        );
    }

    #[test]
    fn refutable_or_misplaced_for_heads() {
        // A literal tests the element, whatever the analysis can say about it.
        assert_code(
            "fn f(ps: [(i32, i32); 2]) -> i32 { let mut t = 0; for (0, x) in ps { t = t + x; } \
             return t; }",
            "E0442",
        );
        assert_code(
            "fn f(ps: [(f32, i32); 2]) -> i32 { let mut t = 0; for (_, 1..=9) in ps { t = t + 1; } \
             return t; }",
            "E0442",
        );
        assert_code(
            &format!(
                "{SHAPE}fn f(ss: [Shape; 2]) -> i32 {{ let mut t = 0; \
                 for Shape::Circle(r) in ss {{ t = t + r; }} return t; }}"
            ),
            "E0442",
        );
        // A range binds one integer; a tensor or any non-array iterand is not destructured.
        assert_code(
            "fn f() -> i32 { let mut t = 0; for (a, b) in 0..4 { t = t + a + b; } return t; }",
            "E0401",
        );
        // A tuple pattern of the wrong arity, or over a non-tuple element.
        assert_code(
            "fn f(ps: [(i32, i32); 2]) -> i32 { let mut t = 0; for (a, b, c) in ps { \
             t = t + a + b + c; } return t; }",
            "E0401",
        );
        assert_code(
            "fn f(ps: [i32; 2]) -> i32 { let mut t = 0; for (a, b) in ps { t = t + a + b; } \
             return t; }",
            "E0401",
        );
        // A `str` is laid out as a `[]u8`, but each element is a `u8` byte, and the message says
        // so rather than suggesting to destructure inside the body.
        let d = check(
            "fn f(s: str) -> i32 { let mut t = 0; for (a, b) in s { t = t + a + b; } \
             return t; }",
        );
        assert!(
            d.iter().any(|d| d.code == Some("E0401")
                && d.message
                    .contains("a `for` over a `str` yields each byte as a `u8`")),
            "{d:?}"
        );
    }

    /// A variant pattern names its enum, which must be the value's type, wherever it appears. It
    /// used to be checked against nothing: `for W::Only(v) in ps` over a struct array read the
    /// struct's second field as the payload, and over an `i32` array lowered a GEP off a loaded
    /// integer (an ICE at MIR verification); `match` and `let` were open the same way.
    #[test]
    fn a_variant_pattern_must_match_the_values_enum() {
        const W: &str = "enum W { Only(i32) } struct P { a: i32, b: i32 } \
                         enum V { One(i32), Two } ";
        let mismatched = |src: &str| {
            let d = check(&format!("{W}{src}"));
            assert!(
                d.iter().any(|d| d.code == Some("E0401")
                    && d.message.contains("this matches a `W`, but the value is")),
                "{src}: {d:?}"
            );
        };
        mismatched(
            "fn f(ps: [P; 2]) -> i32 { let mut t = 0; for W::Only(v) in ps { t = t + v; } \
             return t; }",
        );
        mismatched(
            "fn f(xs: [i32; 3]) -> i32 { let mut t = 0; for W::Only(v) in xs { t = t + v; } \
             return t; }",
        );
        mismatched("fn f(p: P) -> i32 { let W::Only(v) = p; return v; }");
        mismatched("fn f(p: P) -> i32 { return match p { W::Only(v) => v }; }");
        mismatched("fn f(x: V) -> i32 { return match x { W::Only(v) => v, _ => 9 }; }");
        mismatched("fn f(t: (V, i32)) -> i32 { return match t { (W::Only(v), _) => v, _ => 0 }; }");
        // The value's own enum, a generic and an unknown type stay accepted.
        assert_clean(&format!(
            "{W}fn f(x: V) -> i32 {{ return match x {{ V::One(v) => v, V::Two => 0 }}; }}"
        ));
        assert_clean(&format!(
            "{W}fn f(ws: []W) -> i32 {{ let mut t = 0; for W::Only(v) in ws {{ t = t + v; }} \
             return t; }}"
        ));
    }
}
