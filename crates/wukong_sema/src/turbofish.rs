//! **Run-time turbofish dimensions** — `f::<n>(..)` where `n` is a local.
//!
//! A dimension generic can be given a run-time integer at a call (`rowsum::<m, n>(a, out)`, the
//! serving-code story); lowering passes the value through the callee's hidden dimension parameter.
//! The call's result type then mentions that dimension — `fn row<M, N>(..) -> Tensor[f32, N]` makes
//! `row::<2, n>(buf, 0)` a `Tensor[f32, n]` — and so does every shape derived from it.
//!
//! That dimension used to be the local itself, a dimension *named* `n`, and lowering resolved it by
//! reading whatever `n` meant at each later use. Assigning `n` after the call, or shadowing it,
//! changed the extent of a tensor that already existed: `r.dim(0)` printed the new value, a loop to
//! it read past the row (the interpreter trapped, native read whatever lay there), and the results
//! of two `f::<n>` calls taken around an assignment unified as one shape.
//!
//! Now a turbofish names the VALUE `n` holds when the call is made. Each such read is a fresh
//! dimension symbol ([`wukong_span::Interner::fresh`], which still prints as `n`), recorded per call
//! in [`crate::SemaResult::dim_snapshots`]; `wukong_mir_build` evaluates `n` once, before the call's
//! arguments, into a slot of its own that the symbol names for the rest of the function, so nothing
//! that happens to `n` later reaches a shape. Shapes from two reads are the same dimension only when
//! the reads cannot differ: both name one immutable binding — a `let` without `mut` that has an
//! initializer, or a parameter without `mut` — whose name nothing in the function borrows (`&n` or
//! `&mut n` could change it behind the binding's back). Any other two reads are two dimensions, and
//! a shape check between them is the `E0502` it would be between two generic dimensions, with each
//! read labelled.

use wukong_ast::{Block, Expr, ExprKind, ForIter, NodeId, StmtKind, UnOp};
use wukong_diag::Diagnostic;
use wukong_span::{FxHashMap as HashMap, FxHashSet as HashSet, Span, Symbol};
use wukong_types::{Dim, Ty};

use crate::lints::{for_each_child, Child};
use crate::Sema;

/// One run-time dimension of a call: the value a local holds when the call is made
/// (`f::<n>(..)`), published per call in [`crate::SemaResult::dim_snapshots`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DimSnapshot {
    /// The callee's dimension generic the turbofish gives the value.
    pub generic: Symbol,
    /// The dimension the call's types name the value by: a fresh symbol that prints as `source`.
    pub dim: Symbol,
    /// The local read at the call.
    pub source: Symbol,
    /// The local is an unsigned integer, so it zero-extends to the `i64` dimension ABI.
    pub unsigned: bool,
}

/// What one run-time dimension stands for.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum Key {
    /// Every read of one immutable, never-borrowed binding, by the span of its name: the binding
    /// cannot change, so all of its reads are one value.
    Binding(Span),
    /// The read of `Symbol` at one call (the callee's `NodeId`): a binding that may change between
    /// two reads.
    Read(NodeId, Symbol),
}

/// Where a run-time dimension was first read, for its diagnostics.
#[derive(Clone, Copy)]
struct Info {
    at: Span,
    /// One immutable binding's value (`Key::Binding`), rather than a single read.
    shared: bool,
}

/// The run-time dimension state kept across the body pass.
#[derive(Default)]
pub(crate) struct SnapshotState {
    /// The dimension each key was given. Stable across the passes over one body (literal inference
    /// re-checks it, an instantiation re-checks it again), so every pass agrees on the symbols.
    dims: HashMap<Key, Symbol>,
    info: HashMap<Symbol, Info>,
    /// Each call's run-time dimensions, published as [`crate::SemaResult::dim_snapshots`].
    pub(crate) calls: HashMap<NodeId, Vec<DimSnapshot>>,
    /// The names some `&` or `&mut` in the body being checked borrows.
    borrowed: HashSet<Symbol>,
}

impl Sema<'_> {
    /// A function body's check begins: record the names it borrows.
    pub(crate) fn enter_snapshot_body(&mut self, body: &Block) {
        self.snapshots.borrowed.clear();
        borrowed_in_block(body, &mut self.snapshots.borrowed);
    }

    /// The turbofish argument at `at`, at the call whose callee is `call_id`, names the local
    /// `name` of type `ty` for the callee's dimension generic `generic`: the value it gives, or
    /// `None` after reporting a local that is not an integer (`E0401`).
    pub(crate) fn turbofish_local_dim(
        &mut self,
        name: Symbol,
        ty: &Ty,
        generic: Symbol,
        call_id: NodeId,
        at: Span,
    ) -> Option<DimSnapshot> {
        let unsigned = match ty {
            Ty::Scalar(s) if s.is_int() => !s.is_signed(),
            Ty::Unknown | Ty::Error => false,
            other => {
                let (nm, shown) = (self.sym_str(name).to_string(), other.display(self.interner));
                let d = Diagnostic::error(format!(
                    "type mismatch: a dimension must be an integer, but `{nm}` has type `{shown}`"
                ))
                .with_code("E0401")
                .primary(at, format!("`{nm}` is `{shown}`"))
                .help(format!(
                    "a turbofish names a variable, not an expression: convert it into an integer \
                     binding first (`let {nm}_len = {nm} as i64;`) and name that"
                ));
                self.diags.push(d);
                return None;
            }
        };
        let fixed = if self.snapshots.borrowed.contains(&name) {
            None
        } else {
            self.immutable_local_decl(name)
                .map(|d| d.name)
                .or_else(|| self.immutable_param_decl(name))
        };
        let key = match fixed {
            Some(decl) => Key::Binding(decl),
            None => Key::Read(call_id, name),
        };
        let dim = match self.snapshots.dims.get(&key) {
            Some(d) => *d,
            None => {
                let d = self.interner.fresh(name);
                self.snapshots.dims.insert(key, d);
                let shared = fixed.is_some();
                self.snapshots.info.insert(d, Info { at, shared });
                d
            }
        };
        Some(DimSnapshot {
            generic,
            dim,
            source: name,
            unsigned,
        })
    }

    /// Publish the run-time dimensions of the call whose callee is `call_id` (none clears a record
    /// an earlier pass left).
    pub(crate) fn record_dim_snapshots(&mut self, call_id: NodeId, snaps: Vec<DimSnapshot>) {
        if snaps.is_empty() {
            self.snapshots.calls.remove(&call_id);
        } else {
            self.snapshots.calls.insert(call_id, snaps);
        }
    }

    /// Is `s` a run-time turbofish dimension?
    pub(crate) fn is_dim_snapshot(&self, s: Symbol) -> bool {
        self.snapshots.info.contains_key(&s)
    }

    /// The diagnostic just reported is a mismatch between the extents `a` and `b`: label every
    /// run-time turbofish dimension either mentions with the turbofish that read it, and when two
    /// reads disagree, say why they are different dimensions.
    pub(crate) fn explain_dim_snapshots(&mut self, a: &Dim, b: &Dim) {
        let mut snaps: Vec<Symbol> = Vec::new();
        for s in a.vars().into_iter().chain(b.vars()) {
            if self.is_dim_snapshot(s) && !snaps.contains(&s) {
                snaps.push(s);
            }
        }
        if snaps.is_empty() {
            return;
        }
        let Some(mut d) = self.diags.pop() else {
            return;
        };
        let primary = d.primary_span();
        for s in &snaps {
            let at = self.snapshots.info[s].at;
            if !at.is_dummy() && primary != Some(at) {
                let nm = self.sym_str(*s).to_string();
                d = d.secondary(at, format!("`{nm}` is read here, when this call runs"));
            }
        }
        // Two reads of one name are the surprise worth explaining; two names explain themselves.
        let nm = self.sym_str(snaps[0]).to_string();
        if snaps.len() >= 2 && snaps.iter().all(|s| self.sym_str(*s) == nm) {
            let changing = snaps.iter().any(|s| !self.snapshots.info[s].shared);
            d = d.help(if changing {
                format!(
                    "a turbofish reads `{nm}` when its call runs, and a variable that is `mut` or \
                     borrowed can hold another value at the next read, so each read is a \
                     dimension of its own; read it once into an immutable binding \
                     (`let {nm}2 = {nm};`) and name that binding in every turbofish whose shapes \
                     must agree"
                )
            } else {
                format!(
                    "these turbofish read two different bindings named `{nm}`; only the reads of \
                     one immutable binding are the same dimension"
                )
            });
        }
        self.diags.push(d);
    }
}

/// Add to `out` every name a `&` or `&mut` in `b` borrows, at any depth (closure bodies included).
fn borrowed_in_block(b: &Block, out: &mut HashSet<Symbol>) {
    for s in &b.stmts {
        match &s.kind {
            StmtKind::Let { init, .. } => {
                if let Some(e) = init {
                    borrowed_in_expr(e, out);
                }
            }
            StmtKind::Assign { target, value, .. } => {
                borrowed_in_expr(target, out);
                borrowed_in_expr(value, out);
            }
            StmtKind::Expr(e) | StmtKind::Defer(e) => borrowed_in_expr(e, out),
            StmtKind::Return(e) | StmtKind::Break(_, e) => {
                if let Some(e) = e {
                    borrowed_in_expr(e, out);
                }
            }
            StmtKind::Continue(_) => {}
            StmtKind::While { cond, body, .. } => {
                borrowed_in_expr(cond, out);
                borrowed_in_block(body, out);
            }
            StmtKind::For { iter, body, .. } => {
                match iter {
                    ForIter::Range {
                        start, end, step, ..
                    } => {
                        borrowed_in_expr(start, out);
                        for e in end.iter().chain(step.iter()) {
                            borrowed_in_expr(e, out);
                        }
                    }
                    ForIter::Expr(e) => borrowed_in_expr(e, out),
                }
                borrowed_in_block(body, out);
            }
        }
    }
    if let Some(t) = &b.tail {
        borrowed_in_expr(t, out);
    }
}

fn borrowed_in_expr(e: &Expr, out: &mut HashSet<Symbol>) {
    if let ExprKind::Unary {
        op: UnOp::Ref | UnOp::RefMut,
        expr,
    } = &e.kind
    {
        if let Some(root) = place_root(expr) {
            out.insert(root);
        }
    }
    for_each_child(e, &mut |c| match c {
        Child::Expr(x) => borrowed_in_expr(x, out),
        Child::Block(b) => borrowed_in_block(b, out),
    });
}

/// The variable whose storage the place `e` names: the root of an `x`, `x.f`, `x.0`, `x[i]` chain.
fn place_root(e: &Expr) -> Option<Symbol> {
    match &e.kind {
        ExprKind::Path(p) if p.is_single() => Some(p.first().sym),
        ExprKind::Field { base, .. }
        | ExprKind::TupleField { base, .. }
        | ExprKind::Index { base, .. } => place_root(base),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use crate::{check, SemaResult};
    use wukong_diag::Diagnostic;
    use wukong_span::{Interner, SourceId};

    /// `row` returns a view whose shape names the turbofish's dimension; `lanes` makes two shapes
    /// meet.
    const LIB: &str =
        "fn row<M, N>(a: Tensor[i32, M, N], i: i64) -> Tensor[i32, N] { return a[i]; } \
                       fn lanes<N>(x: Tensor[i32, N], y: Tensor[i32, N]) -> i32 { \
                       return x[0] * y[0]; } ";

    fn analyze(src: &str) -> (SemaResult, Vec<Diagnostic>, Interner) {
        let mut interner = Interner::new();
        let (module, pdiags) = wukong_parser::parse_module(src, SourceId(0), &mut interner);
        assert!(pdiags.is_empty(), "parse errors: {pdiags:?}");
        let (result, diags) = check(&module, &interner);
        (result, diags, interner)
    }

    fn codes(src: &str) -> Vec<&'static str> {
        analyze(src).1.into_iter().filter_map(|d| d.code).collect()
    }

    #[track_caller]
    fn ok(src: &str) {
        let d = analyze(src).1;
        assert!(d.is_empty(), "unexpected diagnostics for {src:?}: {d:?}");
    }

    #[track_caller]
    fn rejects(src: &str, code: &str) -> Diagnostic {
        let d = analyze(src).1;
        let found = d.iter().find(|d| d.code == Some(code)).cloned();
        found.unwrap_or_else(|| panic!("expected {code} for {src:?}, got {d:?}"))
    }

    fn with_lib(body: &str) -> String {
        format!("{LIB}fn main() -> i32 {{ let buf: [i32; 6] = [1, 2, 3, 4, 5, 6]; {body} }}")
    }

    fn has_note(d: &Diagnostic, needle: &str) -> bool {
        d.notes.iter().any(|(_, m)| m.contains(needle))
    }

    /// Two reads of a `mut` local are two dimensions (it changes in between here), so the rows
    /// they shape do not meet. Each read is labelled, and the help says how to share one.
    #[test]
    fn two_reads_of_a_changing_local_are_two_dimensions() {
        let d = rejects(
            &with_lib(
                "let mut n = 3; let a = row::<2, n>(buf, 0); n = 2; \
                 let b = row::<3, n>(buf, 0); return lanes(a, b);",
            ),
            "E0502",
        );
        let reads = d
            .labels
            .iter()
            .filter(|l| l.message == "`n` is read here, when this call runs")
            .count();
        assert_eq!(reads, 2, "{d:?}");
        assert!(has_note(&d, "let n2 = n;"), "{d:?}");
        // Unchanged between the reads, a `mut` local is still read twice: nothing proves it equal.
        rejects(
            &with_lib(
                "let mut n = 3; let a = row::<2, n>(buf, 0); let b = row::<2, n>(buf, 1); \
                 n = 0; print(n); return lanes(a, b);",
            ),
            "E0502",
        );
    }

    /// Every read of one immutable binding is one dimension, so its rows meet, even after a later
    /// `let` shadows the name.
    #[test]
    fn reads_of_one_immutable_binding_are_one_dimension() {
        ok(&with_lib(
            "let n = 3; let a = row::<2, n>(buf, 0); let b = row::<2, n>(buf, 1); \
             let n = 1; print(n); return lanes(a, b);",
        ));
        // A parameter without `mut` is such a binding too.
        ok(&format!(
            "{LIB}fn f(buf: [i32; 6], n: i64) -> i32 {{ let a = row::<2, n>(buf, 0); \
             let b = row::<2, n>(buf, 1); return lanes(a, b); }}"
        ));
    }

    /// A borrow could change the binding behind its back (`&mut n`, then `*p = ..`), and two
    /// bindings of one name are two values: neither is one dimension.
    #[test]
    fn a_borrowed_or_shadowed_binding_is_read_afresh() {
        rejects(
            &with_lib(
                "let n = 3; let p = &n; print(*p); let a = row::<2, n>(buf, 0); \
                 let b = row::<2, n>(buf, 1); return lanes(a, b);",
            ),
            "E0502",
        );
        let d = rejects(
            &with_lib(
                "let n = 3; let a = row::<2, n>(buf, 0); let n = 3; \
                 let b = row::<2, n>(buf, 1); return lanes(a, b);",
            ),
            "E0502",
        );
        assert!(has_note(&d, "two different bindings named `n`"), "{d:?}");
    }

    /// The call records what lowering must read: a fresh dimension that prints as the local's name,
    /// shared by the reads of an immutable binding and distinct per read of a `mut` one.
    #[test]
    fn each_call_publishes_the_value_it_reads() {
        let src = with_lib(
            "let n = 3; let a = row::<2, n>(buf, 0); let b = row::<2, n>(buf, 1); \
             let mut m = 3; let c = row::<2, m>(buf, 0); let d = row::<2, m>(buf, 1); \
             m = 0; print(m); return lanes(a, b) + c[0] + d[0];",
        );
        let (result, diags, interner) = analyze(&src);
        assert!(diags.is_empty(), "{diags:?}");
        let n = interner.get("n").expect("`n` is interned");
        let m = interner.get("m").expect("`m` is interned");
        let reads: Vec<_> = result.dim_snapshots.values().flatten().copied().collect();
        assert_eq!(reads.len(), 4, "{reads:?}");
        let of = |src| -> Vec<_> { reads.iter().filter(|s| s.source == src).collect() };
        let (ns, ms) = (of(n), of(m));
        assert_eq!((ns.len(), ms.len()), (2, 2), "{reads:?}");
        assert_eq!(ns[0].dim, ns[1].dim, "one immutable binding, one dimension");
        assert_ne!(
            ms[0].dim, ms[1].dim,
            "two reads of a `mut` local, two dimensions"
        );
        for s in &reads {
            assert!(interner.is_fresh(s.dim) && !s.unsigned, "{s:?}");
            assert_eq!(interner.resolve(s.dim), interner.resolve(s.source));
        }
    }

    /// A run-time value meets a fixed extent at the run-time `?` level (the turbofish is the
    /// caller's claim about the argument), but a FIXED parameter cannot assume it.
    #[test]
    fn a_runtime_dimension_against_a_fixed_extent() {
        ok(
            "fn rowsum<M, N>(a: Tensor[f32, M, N], mut o: Tensor[f32, M]) { o[0] = a[0, 0]; } \
            fn g(n: i64) { let x: [f32; 12] = [0.0; 12]; let mut y: [f32; 3] = [0.0; 3]; \
            let m = 3; rowsum::<m, n>(x, y); }",
        );
        let d = rejects(
            &format!(
                "{LIB}fn three(a: Tensor[i32, 3]) -> i32 {{ return a[0]; }} \
                 fn g(buf: [i32; 6], n: i64) -> i32 {{ return three(row::<2, n>(buf, 0)); }}"
            ),
            "E0502",
        );
        assert!(d.message.contains("the run-time dimension `n`"), "{d:?}");
    }

    /// A turbofish dimension names an integer: a float local is `E0401`, an unsigned one is
    /// fine. A type generic's argument is a type, and stays one.
    #[test]
    fn a_turbofish_dimension_names_an_integer() {
        rejects(
            &with_lib("let x = 2.5; let a = row::<2, x>(buf, 0); return a[0];"),
            "E0401",
        );
        ok("struct P { a: i32 } fn id<T>(x: T) -> T { return x; } \
            fn g() -> i32 { let p = P { a: 1 }; let q = id::<P>(p); return q.a; }");
        assert!(!codes(&with_lib(
            "let u: u32 = 3; let a = row::<2, u>(buf, 0); return a[0];"
        ))
        .contains(&"E0401"));
    }
}
