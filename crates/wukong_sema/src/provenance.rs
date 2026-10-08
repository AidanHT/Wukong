//! **Call-site provenance** — a call's shape and arity errors point at the argument that is wrong,
//! say which argument (or turbofish) fixed the dimension it disagrees with, and show where the
//! callee declares the parameter.
//!
//! The headline diagnostic of a shape-typed language is the call whose shapes disagree, and it used to
//! carry one unlabelled span — the whole call — and a message: `dimension mismatch on `K` (axis 0):
//! expected 512, found 513`. Where 512 came from was gone: `unify_dim` stored the binding but not who
//! made it. Like rustc's E0308 inside a call and Futhark's "size unified with ... at ...", the
//! diagnostic now reads
//!
//! ```text
//! error[E0502]: dimension mismatch on `K` (axis 0): expected 512, found 513
//!   --> wrong_k.wk:22:34
//!    |  matmul::<512, 512, 512>(a, b, c);
//!    |                ---          ^ `K` (axis 0) is 513 in argument 2
//!    |                |
//!    |                `K` = 512 was fixed here, by the turbofish
//!   ::: ... `K` is declared in parameter `b` of `matmul`
//! ```
//!
//! `check_fn_call` opens a [`CallCtx`] for the call, names the argument it is unifying, and
//! `unify_dim` records where each dimension variable is bound ([`Origin`]); a conflict is reported
//! through [`Sema::report_dim_conflict`]. Arity errors (`E0503`) name the callee and label its
//! definition, the extra arguments, or the missing parameters. The headline messages are unchanged
//! (`crates/wukongc/tests/fail.rs` pins them); everything added here is labels and notes.

use wukong_ast::{Expr, FnDecl};
use wukong_diag::Diagnostic;
use wukong_span::{FxHashMap as HashMap, Span, Symbol};
use wukong_types::{Dim, Ty};

use crate::shape::DimSite;
use crate::{FnSig, Sema};

/// Where a call's dimension variable got its value.
#[derive(Clone, Copy)]
enum Origin {
    /// An explicit turbofish argument (`f::<512, ..>`), at this span.
    Turbofish(Span),
    /// The shape of argument `index` (into the call's argument list, receiver first), at `axis`.
    Arg {
        index: usize,
        span: Span,
        axis: usize,
    },
    /// Solved from a derived extent or an element count (`Tensor[f32, 2*N]` fed 8 elements), at the
    /// argument whose equation solved it.
    Solved(Span),
}

/// The spans of one call, for its diagnostics.
pub(crate) struct CallSite {
    /// The function called, when the call names a definition.
    callee: Option<Symbol>,
    /// One span per argument, the method receiver first.
    args: Vec<Span>,
    /// Whether the first argument is a method receiver (`x.m(..)`): user-facing argument numbers
    /// then start after it.
    receiver: bool,
}

impl CallSite {
    /// A call of `callee` (`None` for a function-typed parameter or a trait-bound call) with these
    /// written arguments, after the receiver at `receiver` for a method call.
    pub(crate) fn new(callee: Option<Symbol>, receiver: Option<Span>, args: &[Expr]) -> CallSite {
        let mut spans = Vec::with_capacity(args.len() + 1);
        spans.extend(receiver);
        spans.extend(args.iter().map(|a| a.span));
        CallSite {
            callee,
            args: spans,
            receiver: receiver.is_some(),
        }
    }

    /// The span of argument `i` (receiver first), if it was written.
    pub(crate) fn arg_span(&self, i: usize) -> Option<Span> {
        self.args.get(i).copied()
    }
}

/// The call `check_fn_call` is unifying.
pub(crate) struct CallCtx {
    callee: Option<Symbol>,
    receiver: bool,
    /// The argument being unified: its index (receiver first) and span.
    arg: Option<(usize, Span)>,
    origins: HashMap<Symbol, Origin>,
}

/// Provenance state kept across the body pass.
#[derive(Default)]
pub(crate) struct ProvenanceState {
    /// Each declared function's parameters, by function name: the parameter's name and the span of
    /// its written type.
    params: HashMap<Symbol, Vec<(Symbol, Span)>>,
    /// The call being unified, or `None` (a body check, a kernel builtin).
    cur: Option<CallCtx>,
}

/// `argument 2`, or `the receiver`, for argument `index` of a call (receiver first).
fn arg_desc(index: usize, receiver: bool) -> String {
    match (receiver, index) {
        (true, 0) => "the receiver".to_string(),
        (true, i) => format!("argument {i}"),
        (false, i) => format!("argument {}", i + 1),
    }
}

fn plural(n: usize, one: &str, many: &str) -> String {
    format!("{n} {}", if n == 1 { one } else { many })
}

/// The dimension variables a parameter type's tensor shapes use, in first-seen order, through
/// pointer, reference, slice, array and tuple structure — `mir_build`'s `collect_symbolic_dims`, the
/// set of dimensions lowering passes as hidden arguments.
fn collect_param_dims(t: &Ty, out: &mut Vec<Symbol>) {
    match t {
        Ty::Tensor { shape, .. } => {
            for s in shape.symbolic_dims() {
                if !out.contains(&s) {
                    out.push(s);
                }
            }
        }
        Ty::Ptr { pointee, .. } | Ty::Ref { pointee, .. } | Ty::Slice(pointee) => {
            collect_param_dims(pointee, out)
        }
        Ty::Array { elem, .. } => collect_param_dims(elem, out),
        Ty::Tuple(items) => {
            for it in items {
                collect_param_dims(it, out);
            }
        }
        _ => {}
    }
}

impl Sema<'_> {
    /// Record `f`'s parameter names and type spans, for "declared here" labels at its calls.
    pub(crate) fn record_fn_params(&mut self, f: &FnDecl) {
        let params = f.params.iter().map(|p| (p.name.sym, p.ty.span)).collect();
        self.provenance.params.insert(f.name.sym, params);
    }

    /// Open the provenance context of a call; returns the enclosing one, for [`Self::end_call`].
    pub(crate) fn begin_call(&mut self, site: &CallSite) -> Option<CallCtx> {
        self.provenance.cur.replace(CallCtx {
            callee: site.callee,
            receiver: site.receiver,
            arg: None,
            origins: HashMap::default(),
        })
    }

    pub(crate) fn end_call(&mut self, outer: Option<CallCtx>) {
        self.provenance.cur = outer;
    }

    /// The argument being unified from now on (`None` once the parameters are done).
    pub(crate) fn set_call_arg(&mut self, arg: Option<(usize, Span)>) {
        if let Some(cx) = self.provenance.cur.as_mut() {
            cx.arg = arg;
        }
    }

    /// Dimension `g` was bound by the turbofish argument at `span`.
    pub(crate) fn note_turbofish_dim(&mut self, g: Symbol, span: Span) {
        if let Some(cx) = self.provenance.cur.as_mut() {
            cx.origins.insert(g, Origin::Turbofish(span));
        }
    }

    /// Dimension `v` was bound while unifying the current argument, at `axis` of its shape.
    pub(crate) fn note_dim_bound(&mut self, v: Symbol, axis: usize) {
        if let Some(cx) = self.provenance.cur.as_mut() {
            if let Some((index, span)) = cx.arg {
                cx.origins
                    .entry(v)
                    .or_insert(Origin::Arg { index, span, axis });
            }
        }
    }

    /// Dimension `g` was solved from the derived-extent equation queued at `span`.
    pub(crate) fn note_dim_solved(&mut self, g: Symbol, span: Span) {
        if let Some(cx) = self.provenance.cur.as_mut() {
            cx.origins.entry(g).or_insert(Origin::Solved(span));
        }
    }

    /// An `E0502` dimension conflict at a call: `msg` is the headline (unchanged — it names the
    /// dimension and both extents), `span` the conflicting argument, `site` the axis and its declared
    /// name, `want` the extent the dimension already has and `found` the argument's. Outside a call
    /// (a body check) this is the plain one-span diagnostic.
    pub(crate) fn report_dim_conflict(
        &mut self,
        span: Span,
        msg: String,
        site: DimSite,
        want: &str,
        found: &str,
    ) {
        let at = self.dim_site(site);
        let Some(cx) = self.provenance.cur.as_ref() else {
            self.error(span, "E0502", msg);
            return;
        };
        let (callee, receiver) = (cx.callee, cx.receiver);
        let arg = cx.arg;
        let origin = site.name.and_then(|n| cx.origins.get(&n).copied());
        let primary = match arg {
            Some((i, _)) => format!("{at} is {found} in {}", arg_desc(i, receiver)),
            None => format!("{at} is {found} here"),
        };
        let mut d = Diagnostic::error(msg)
            .with_code("E0502")
            .primary(span, primary);
        if let (Some(name), Some(origin)) = (site.name, origin) {
            let nm = self.sym_str(name).to_string();
            let (at_span, who) = match origin {
                Origin::Turbofish(s) => (s, "by the turbofish".to_string()),
                Origin::Arg {
                    index,
                    span: s,
                    axis,
                } => (
                    s,
                    format!("by axis {axis} of {}", arg_desc(index, receiver)),
                ),
                Origin::Solved(s) => (s, "from the extent of this argument".to_string()),
            };
            if at_span == span {
                // The same argument fixed the dimension at another axis: say so without a second
                // label on the same span.
                d = d.note(format!("`{nm}` = {want} was fixed {who}"));
            } else if !at_span.is_dummy() {
                d = d.secondary(at_span, format!("`{nm}` = {want} was fixed here, {who}"));
            }
        }
        if let (Some(f), Some((i, _))) = (callee, arg) {
            if let Some(&(pname, pspan)) = self.provenance.params.get(&f).and_then(|ps| ps.get(i)) {
                if !pspan.is_dummy() {
                    let (fname, pn) = (self.callee_display(f), self.sym_str(pname).to_string());
                    let label = match site.name {
                        Some(n) => format!(
                            "`{}` is declared in parameter `{pn}` of `{fname}`",
                            self.sym_str(n)
                        ),
                        None => format!("parameter `{pn}` of `{fname}` is declared here"),
                    };
                    d = d.secondary(pspan, label);
                }
            }
        }
        self.diags.push(d);
    }

    /// `E0503` for a call of `site.callee` with `supplied` arguments where its signature takes
    /// `takes` (both counting a method receiver), at the call `span`.
    pub(crate) fn report_call_arity(
        &mut self,
        takes: usize,
        supplied: usize,
        span: Span,
        site: &CallSite,
    ) {
        let off = usize::from(site.receiver);
        let (want, got) = (takes.saturating_sub(off), supplied.saturating_sub(off));
        let what = match site.callee {
            Some(f) if site.receiver => format!("method `{}`", self.callee_display(f)),
            Some(f) => format!("function `{}`", self.callee_display(f)),
            None => "this function".to_string(),
        };
        let msg = format!(
            "{what} takes {} but {} supplied",
            plural(want, "argument", "arguments"),
            if got == 1 {
                "1 was".to_string()
            } else {
                format!("{got} were")
            }
        );
        let mut d = Diagnostic::error(msg).with_code("E0503").primary(
            span,
            format!("expected {}", plural(want, "argument", "arguments")),
        );
        // Each argument past the signature's last parameter.
        for i in takes..supplied {
            if let Some(s) = site.arg_span(i) {
                d = d.secondary(s, "unexpected argument");
            }
        }
        if let Some(f) = site.callee {
            // The parameters no argument reached.
            if supplied < takes {
                let names: Vec<String> = self
                    .provenance
                    .params
                    .get(&f)
                    .map(|ps| {
                        ps.iter()
                            .skip(supplied)
                            .take(takes - supplied)
                            .map(|(n, _)| format!("`{}`", self.sym_str(*n)))
                            .collect()
                    })
                    .unwrap_or_default();
                if !names.is_empty() {
                    d = d.help(format!("missing an argument for {}", names.join(", ")));
                }
            }
            if let Some(def) = self.defs.lookup(f) {
                if !def.span.is_dummy() {
                    let shown = self.callee_display(f);
                    d = d.secondary(def.span, format!("`{shown}` is defined here"));
                }
            }
        }
        self.diags.push(d);
    }

    /// `E0503` for a turbofish with `supplied` generic arguments where the callee declares `takes`.
    pub(crate) fn report_generic_arity(
        &mut self,
        takes: usize,
        supplied: usize,
        span: Span,
        site: &CallSite,
    ) {
        let what = match site.callee {
            Some(f) => format!("function `{}`", self.callee_display(f)),
            None => "this function".to_string(),
        };
        let mut d = Diagnostic::error(format!(
            "{what} takes {} but {supplied} {} supplied",
            plural(takes, "generic argument", "generic arguments"),
            if supplied == 1 { "was" } else { "were" }
        ))
        .with_code("E0503")
        .primary(
            span,
            format!(
                "expected {}",
                plural(takes, "generic argument", "generic arguments")
            ),
        );
        if let Some(def) = site.callee.and_then(|f| self.defs.lookup(f)) {
            if !def.span.is_dummy() {
                d = d.secondary(def.span, "defined here");
            }
        }
        self.diags.push(d);
    }

    /// `E0520`: after a call's parameters are unified and its derived extents solved, a dimension
    /// that the callee's parameter shapes use is still unknown. Lowering passes every such dimension
    /// as a hidden `i64` argument (`mir_build::symbolic_dim_params`) and finds it exactly where this
    /// check looks — a turbofish, an argument's shape, a solved equation, or the run-time extent of
    /// a `?` axis of a tensor argument at the same position — so an unbound one used to surface only
    /// in lowering, as the codegen limit `C0001`. Silent when the call already reported an error
    /// (`errors_before`) or an argument's type is not known, so it never cascades.
    pub(crate) fn check_call_dims_bound(
        &mut self,
        sig: &FnSig,
        arg_tys: &[Ty],
        dims: &HashMap<Symbol, Dim>,
        span: Span,
        site: &CallSite,
        errors_before: usize,
    ) {
        // An argument whose type is not concrete here — not known, or a type parameter of the caller
        // that each instantiation substitutes before lowering — may still bind a dimension then.
        if self.diags[errors_before..].iter().any(|d| d.is_error())
            || arg_tys.iter().any(|t| self.is_opaque_arg(t))
        {
            return;
        }
        let mut used = Vec::new();
        for p in &sig.params {
            collect_param_dims(p, &mut used);
        }
        // A `?` axis of a tensor argument passed where the parameter has a plain `N`: lowering reads
        // the run-time extent (the `?` escape hatch binds nothing at compile time).
        let mut at_runtime: Vec<Symbol> = Vec::new();
        for (p, a) in sig.params.iter().zip(arg_tys) {
            if let (Ty::Tensor { shape: ps, .. }, Ty::Tensor { shape: as_, .. }) = (p, a) {
                if ps.rank() == as_.rank() {
                    for (pd, ad) in ps.0.iter().zip(&as_.0) {
                        if let (Dim::Var(g), Dim::Dynamic) = (pd, ad) {
                            at_runtime.push(*g);
                        }
                    }
                }
            }
        }
        let unbound: Vec<Symbol> = used
            .into_iter()
            .filter(|g| {
                sig.generics.contains(g) && !dims.contains_key(g) && !at_runtime.contains(g)
            })
            .collect();
        let Some(&first) = unbound.first() else {
            return;
        };
        let names: Vec<String> = unbound
            .iter()
            .map(|g| format!("`{}`", self.sym_str(*g)))
            .collect();
        let listed = match names.len() {
            1 => names[0].clone(),
            n => format!("{} and {}", names[..n - 1].join(", "), names[n - 1]),
        };
        let fname = site.callee.map(|f| self.callee_display(f));
        let of = fname
            .as_ref()
            .map(|f| format!(" of `{f}`"))
            .unwrap_or_default();
        let (what, label) = if unbound.len() == 1 {
            ("dimension", format!("{listed} is fixed by no argument"))
        } else {
            ("dimensions", format!("{listed} are fixed by no argument"))
        };
        let mut d = Diagnostic::error(format!("cannot infer {what} {listed}{of}"))
            .with_code("E0520")
            .primary(span, label);
        // The parameter that declares the first one, and why its argument could not fix it.
        if let Some(i) = sig.params.iter().position(|p| {
            let mut v = Vec::new();
            collect_param_dims(p, &mut v);
            v.contains(&first)
        }) {
            if let Some(f) = site.callee {
                if let Some(&(pname, pspan)) =
                    self.provenance.params.get(&f).and_then(|ps| ps.get(i))
                {
                    if !pspan.is_dummy() {
                        let (fs, pn) = (self.callee_display(f), self.sym_str(pname).to_string());
                        d = d.secondary(
                            pspan,
                            format!(
                                "`{}` is declared in parameter `{pn}` of `{fs}`",
                                self.sym_str(first)
                            ),
                        );
                    }
                }
            }
            if let (Some(Ty::Tensor { shape, .. }), Some(Ty::Array { len, .. })) =
                (sig.params.get(i), arg_tys.get(i))
            {
                if shape.rank() >= 2 {
                    d = d.note(format!(
                        "an array argument fixes only the element count of a multi-axis tensor \
                         parameter, and {len} elements can be split over more than one free axis \
                         in more than one way"
                    ));
                }
            }
        }
        if let Some(f) = fname {
            let args: Vec<String> = sig
                .generics
                .iter()
                .map(|g| match dims.get(g) {
                    Some(dim) => self.dim_str(dim),
                    None => self.sym_str(*g).to_string(),
                })
                .collect();
            d = d.help(format!(
                "supply the dimensions with a turbofish: `{f}::<{}>(..)`",
                args.join(", ")
            ));
        }
        self.diags.push(d);
    }

    /// Whether `t` is `Unknown`/`Error` or mentions a type parameter of the enclosing item.
    fn is_opaque_arg(&self, t: &Ty) -> bool {
        match t {
            Ty::Unknown | Ty::Error => true,
            Ty::Named(n) => self.generics.contains(n),
            Ty::Ptr { pointee, .. } | Ty::Ref { pointee, .. } | Ty::Slice(pointee) => {
                self.is_opaque_arg(pointee)
            }
            Ty::Array { elem, .. } => self.is_opaque_arg(elem),
            Ty::Tuple(items) => items.iter().any(|it| self.is_opaque_arg(it)),
            _ => false,
        }
    }

    /// A function's name as the program spells it: a method `Type.m` reads `Type::m`.
    fn callee_display(&self, f: Symbol) -> String {
        let s = self.sym_str(f);
        if let Some((owner, m)) = s.rsplit_once('.') {
            let is_type = self
                .interner
                .get(owner)
                .is_some_and(|o| self.is_declared_named_ty(o));
            if is_type {
                return format!("{owner}::{m}");
            }
        }
        s.to_string()
    }
}

#[cfg(test)]
mod tests {
    use wukong_diag::Diagnostic;
    use wukong_span::{Interner, SourceId};

    /// The first diagnostic carrying `code` for `src` (which must parse).
    #[track_caller]
    fn first(src: &str, code: &str) -> Diagnostic {
        let mut interner = Interner::new();
        let (module, pdiags) = wukong_parser::parse_module(src, SourceId(0), &mut interner);
        assert!(pdiags.is_empty(), "parse errors: {pdiags:?}");
        let ds = crate::check(&module, &interner).1;
        let found = ds.iter().find(|d| d.code == Some(code)).cloned();
        found.unwrap_or_else(|| panic!("no {code} for {src:?}: {ds:?}"))
    }

    fn labels(d: &Diagnostic) -> Vec<(bool, &str)> {
        d.labels
            .iter()
            .map(|l| (l.primary, l.message.as_str()))
            .collect()
    }

    const MATMUL: &str = "fn matmul<M, N, K>(a: Tensor[f32, M, K], b: Tensor[f32, K, N]) {} ";

    #[test]
    fn a_dimension_conflict_names_the_argument_that_fixed_it() {
        let d = first(
            &format!(
                "{MATMUL} fn main() -> i32 {{ let a: Tensor[f32, 512, 512]; \
                 let b: Tensor[f32, 513, 512]; matmul(a, b); return 0; }}"
            ),
            "E0502",
        );
        assert_eq!(
            d.message,
            "dimension mismatch on `K` (axis 0): expected 512, found 513"
        );
        assert_eq!(
            labels(&d),
            vec![
                (true, "`K` (axis 0) is 513 in argument 2"),
                (false, "`K` = 512 was fixed here, by axis 1 of argument 1"),
                (false, "`K` is declared in parameter `b` of `matmul`"),
            ]
        );
    }

    #[test]
    fn a_turbofish_binding_is_named_as_the_origin() {
        let d = first(
            &format!(
                "{MATMUL} fn main() -> i32 {{ let a: Tensor[f32, 512, 512]; \
                 let b: Tensor[f32, 513, 512]; matmul::<512, 512, 512>(a, b); return 0; }}"
            ),
            "E0502",
        );
        assert_eq!(
            labels(&d),
            vec![
                (true, "`K` (axis 0) is 513 in argument 2"),
                (false, "`K` = 512 was fixed here, by the turbofish"),
                (false, "`K` is declared in parameter `b` of `matmul`"),
            ]
        );
    }

    #[test]
    fn a_conflict_within_one_argument_is_a_note() {
        let d = first(
            "fn sq<N>(a: Tensor[f32, N, N]) {} \
             fn main() -> i32 { let a: Tensor[f32, 3, 4]; sq(a); return 0; }",
            "E0502",
        );
        assert_eq!(
            d.message,
            "dimension mismatch on `N` (axis 1): expected 3, found 4"
        );
        assert_eq!(
            labels(&d),
            vec![
                (true, "`N` (axis 1) is 4 in argument 1"),
                (false, "`N` is declared in parameter `a` of `sq`"),
            ]
        );
        assert!(
            d.notes
                .iter()
                .any(|(_, m)| m == "`N` = 3 was fixed by axis 0 of argument 1"),
            "{d:?}"
        );
    }

    #[test]
    fn an_arity_error_names_the_callee_and_the_extra_or_missing_arguments() {
        const ADD: &str = "fn add(a: i32, b: i32) -> i32 { return a + b; } ";
        let extra = first(
            &format!("{ADD} fn main() -> i32 {{ return add(1, 2, 3); }}"),
            "E0503",
        );
        assert_eq!(
            extra.message,
            "function `add` takes 2 arguments but 3 were supplied"
        );
        assert_eq!(
            labels(&extra),
            vec![
                (true, "expected 2 arguments"),
                (false, "unexpected argument"),
                (false, "`add` is defined here"),
            ]
        );
        let missing = first(
            &format!("{ADD} fn main() -> i32 {{ return add(1); }}"),
            "E0503",
        );
        assert_eq!(
            missing.message,
            "function `add` takes 2 arguments but 1 was supplied"
        );
        assert!(
            missing
                .notes
                .iter()
                .any(|(_, m)| m == "missing an argument for `b`"),
            "{missing:?}"
        );
        // A method counts its arguments after the receiver, and is named `Type::method`.
        let method = first(
            "struct V { x: f32 } impl V { fn dot(self, o: V) -> f32 { return self.x * o.x; } } \
             fn f(a: V) -> f32 { return a.dot(a, a); }",
            "E0503",
        );
        assert_eq!(
            method.message,
            "method `V::dot` takes 1 argument but 2 were supplied"
        );
        let generic = first(
            "fn g<N>(a: Tensor[f32, N]) {} \
             fn main() -> i32 { let a: [f32; 2] = [1.0, 2.0]; g::<2, 3>(a); return 0; }",
            "E0503",
        );
        assert_eq!(
            generic.message,
            "function `g` takes 1 generic argument but 2 were supplied"
        );
    }

    fn codes(src: &str) -> Vec<&'static str> {
        let mut interner = Interner::new();
        let (module, pdiags) = wukong_parser::parse_module(src, SourceId(0), &mut interner);
        assert!(pdiags.is_empty(), "parse errors: {pdiags:?}");
        crate::check(&module, &interner)
            .1
            .iter()
            .filter_map(|d| d.code)
            .collect()
    }

    const TOTAL: &str = "fn total<M, N>(a: Tensor[f32, M, N]) -> f32 { return 0.0; } ";

    #[test]
    fn a_dimension_no_argument_fixes_is_reported_at_the_call() {
        let d = first(
            &format!(
                "{TOTAL} fn main() -> i32 {{ let a: [f32; 12] = [1.0; 12]; \
                 return total(a) as i32; }}"
            ),
            "E0520",
        );
        assert_eq!(d.message, "cannot infer dimensions `M` and `N` of `total`");
        assert!(
            d.notes
                .iter()
                .any(|(_, m)| m.contains("`total::<M, N>(..)`")),
            "{d:?}"
        );
        assert_eq!(
            labels(&d),
            vec![
                (true, "`M` and `N` are fixed by no argument"),
                (false, "`M` is declared in parameter `a` of `total`"),
            ]
        );
    }

    #[test]
    fn every_source_of_a_dimension_binds_it() {
        const S: &str = "fn s<N>(a: Tensor[f32, N]) -> f32 { return 0.0; } ";
        for src in [
            // A turbofish.
            format!(
                "{TOTAL} fn main() -> i32 {{ let a: [f32; 12] = [1.0; 12]; \
                 return total::<3, 4>(a) as i32; }}"
            ),
            // Solved from the element count once the other axis is fixed.
            "fn rows<N>(a: Tensor[f32, N, 4]) -> f32 { return 0.0; } \
             fn main() -> i32 { let a: [f32; 12] = [1.0; 12]; return rows(a) as i32; }"
                .to_string(),
            // A `?` axis of a tensor argument: the run-time extent.
            format!("{S} fn f(t: Tensor[f32, ?]) -> f32 {{ return s(t); }}"),
            // A caller's own dimension.
            format!("{S} fn f<M>(t: Tensor[f32, M]) -> f32 {{ return s(t); }}"),
            // A caller's type parameter is substituted before lowering.
            format!("{S} fn f<T>(t: T) -> f32 {{ return s(t); }}"),
            // A `const` turbofish.
            format!(
                "const W: usize = 4; {S} fn f(t: Tensor[f32, ?]) -> f32 {{ return s::<W>(t); }}"
            ),
        ] {
            let cs = codes(&src);
            assert!(!cs.contains(&"E0520"), "{src}: {cs:?}");
        }
    }

    /// A `const` named in a turbofish is its value, so the argument is checked against it.
    #[test]
    fn a_const_turbofish_is_its_value() {
        let src = "const W: usize = 4; fn s<N>(a: Tensor[f32, N]) -> f32 { return 0.0; } \
                   fn main() -> i32 { let a: [f32; 5] = [0.0; 5]; return s::<W>(a) as i32; }";
        assert!(codes(src).contains(&"E0501"), "{:?}", codes(src));
        let fits = src.replace("[f32; 5] = [0.0; 5]", "[f32; 4] = [0.0; 4]");
        assert!(codes(&fits).is_empty(), "{:?}", codes(&fits));
    }
}
