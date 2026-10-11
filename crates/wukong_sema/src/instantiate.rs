//! **Per-instantiation re-checking** of type-generic function bodies.
//!
//! The body pass types a generic body once, with each value-type generic `T` opaque
//! (`Ty::Named(T)`), and every check that asks about an opaque `T` passes: `a + b` on two `T`s,
//! `x.f` on a `T`, `x as f32`. Nothing re-checked an instantiation, so `add(p, q)` with
//! `fn add<T>(a: T, b: T) -> T { return a + b; }` and `p, q: Point` was accepted, and mir_build's
//! monomorphizer then lowered `add$Point` as an arithmetic op on two struct base pointers — the
//! ill-typed MIR the non-generic path rejects up front (E0433), reaching the verifier as an ICE or
//! the two backends as a silent disagreement.
//!
//! Like Zig's per-instantiation analysis and C++'s "required from here", every concrete
//! instantiation is now type-checked again with `T` replaced by its concrete type, and an error
//! found there is reported at the offending expression in the generic body with a secondary label
//! on the call that required it: "required by this instantiation of `add` with T = Point".
//!
//! **Discovery.** `check_fn_call` records, for every call of a function with value-type generics
//! whose substitution is fully concrete, an [`InstCall`]. The body pass records the calls made by
//! concrete code; re-checking an instance records the calls its body makes at the instance's
//! types, so a generic calling a generic is reached transitively (the same closure mir_build's
//! `MonoCollector` computes). Each distinct `(function, substitution)` is checked once, and at
//! most [`MAX_INSTANCES`] are checked in one module, so unbounded polymorphic recursion
//! (`fn f<T>(x: T) { f((x, x)); }`) terminates.
//!
//! **Isolation.** An instance check re-types nodes that lowering reads from the generic pass (the
//! monomorphizer substitutes its own types into the generic record), so every table the pass
//! publishes is restored afterwards: re-checking only ever adds diagnostics. It runs only on a
//! module whose body pass was clean, so an error in a generic body is reported once, not once per
//! instance.

use wukong_ast::{Attr, Block, Expr, FnDecl, ItemKind, Module};
use wukong_span::{FxHashMap as HashMap, FxHashSet as HashSet, Span, Symbol};
use wukong_types::Ty;

use crate::shape::infer_type_generics;
use crate::{FnSig, Sema};

/// The most distinct instantiations re-checked in one module. Real programs stay far below it; it
/// exists so a generic that instantiates itself at an ever-growing type cannot loop forever.
const MAX_INSTANCES: usize = 512;

/// One concrete instantiation of a type-generic function, recorded at the call that requires it.
#[derive(Clone)]
pub(crate) struct InstCall {
    /// The function called.
    callee: Symbol,
    /// Each value-type generic of the callee, in declaration order, with its concrete type.
    subst: Vec<(Symbol, Ty)>,
    /// The call that requires this instantiation.
    span: Span,
}

/// The value-type generics of `sig`: those written as a type somewhere in a parameter or the
/// return type (`Ty::Named(g)`, or the element of a dtype-generic tensor), in declaration order. A
/// dimension generic lives in a shape as a `Dim::Var` and is not one. This is the set mir_build's
/// monomorphizer specializes over (`fn_type_generics`).
pub(crate) fn value_type_generics(sig: &FnSig) -> Vec<Symbol> {
    let mut seen = Vec::new();
    for p in &sig.params {
        collect_named(p, &sig.generics, &mut seen);
    }
    collect_named(&sig.ret, &sig.generics, &mut seen);
    sig.generics
        .iter()
        .copied()
        .filter(|g| seen.contains(g))
        .collect()
}

fn collect_named(ty: &Ty, generics: &[Symbol], out: &mut Vec<Symbol>) {
    match ty {
        Ty::Named(g) => {
            if generics.contains(g) && !out.contains(g) {
                out.push(*g);
            }
        }
        Ty::Ptr { pointee, .. } | Ty::Ref { pointee, .. } | Ty::Slice(pointee) => {
            collect_named(pointee, generics, out)
        }
        Ty::Array { elem, .. } => collect_named(elem, generics, out),
        Ty::Tuple(fields) => {
            for f in fields {
                collect_named(f, generics, out);
            }
        }
        Ty::Fn { params, ret } => {
            for p in params {
                collect_named(p, generics, out);
            }
            collect_named(ret, generics, out);
        }
        Ty::Tensor {
            elem_var: Some(g), ..
        } if generics.contains(g) && !out.contains(g) => out.push(*g),
        _ => {}
    }
}

impl<'a> Sema<'a> {
    /// Record that the call at `span` instantiates `callee` with the value-type bindings `tys`,
    /// when every value-type generic of `callee` is bound to a concrete type. A call made from a
    /// generic body binds the callee's generics to the caller's (still opaque) ones and is not
    /// recorded; the caller's own instances record it once its types are known.
    pub(crate) fn note_instantiation(
        &mut self,
        callee: Symbol,
        sig: &FnSig,
        tys: &HashMap<Symbol, Ty>,
        span: Span,
    ) {
        let generics = value_type_generics(sig);
        if generics.is_empty() {
            return;
        }
        let mut subst = Vec::with_capacity(generics.len());
        for g in generics {
            match tys.get(&g) {
                Some(t) if self.is_concrete_ty(t) => subst.push((g, t.clone())),
                _ => return,
            }
        }
        self.inst_calls.push(InstCall {
            callee,
            subst,
            span,
        });
    }

    /// Whether `t` names no generic type variable and nothing unknown: a type an instance can be
    /// checked at.
    fn is_concrete_ty(&self, t: &Ty) -> bool {
        match t {
            Ty::Named(n) => self.is_declared_named_ty(*n),
            Ty::Unknown | Ty::Error => false,
            Ty::Ptr { pointee, .. } | Ty::Ref { pointee, .. } | Ty::Slice(pointee) => {
                self.is_concrete_ty(pointee)
            }
            Ty::Array { elem, .. } => self.is_concrete_ty(elem),
            Ty::Tuple(fields) => fields.iter().all(|f| self.is_concrete_ty(f)),
            Ty::Fn { params, ret } => {
                params.iter().all(|p| self.is_concrete_ty(p)) && self.is_concrete_ty(ret)
            }
            Ty::Tensor { elem_var, .. } => elem_var.is_none(),
            _ => true,
        }
    }

    /// Adapt an unsuffixed literal argument of a generic call to the type its generic parameter is
    /// bound to by the call's other arguments, exactly as a literal adapts to a concrete scalar
    /// parameter. Inference used to bind from the arguments in order, first binding winning, so
    /// `add(x_i64, 1)` bound `T = i64` and then rejected the `i32` literal (E0401), and
    /// `add(1, x_i64)` bound `T = i32` from the literal. The non-literal arguments now bind first;
    /// a literal whose generic they leave unbound keeps its default type.
    pub(crate) fn adapt_generic_literal_args(
        &mut self,
        sig: &FnSig,
        args: &[Expr],
        arg_tys: &mut [Ty],
    ) {
        if sig.generics.is_empty() {
            return;
        }
        let mut tys: HashMap<Symbol, Ty> = HashMap::default();
        for ((p, a), t) in sig.params.iter().zip(args).zip(arg_tys.iter()) {
            if !self.is_adaptable_num_literal(a) {
                infer_type_generics(p, t, &sig.generics, &mut tys);
            }
        }
        for (i, (p, a)) in sig.params.iter().zip(args).enumerate() {
            let Ty::Named(g) = p else {
                continue;
            };
            let Some(bound) = tys.get(g).filter(|t| matches!(t, Ty::Scalar(_))).cloned() else {
                continue;
            };
            if self.literal_adapts(&bound, a) {
                self.retype_adapted_literal(a, &bound);
                self.range_check_int_literal(a, &bound);
                arg_tys[i] = bound;
            }
        }
    }

    /// `T = Point, U = f32` — how a substitution reads in a diagnostic, and its identity.
    fn subst_desc(&self, subst: &[(Symbol, Ty)]) -> String {
        subst
            .iter()
            .map(|(g, t)| format!("{} = {}", self.sym_str(*g), t.display(self.interner)))
            .collect::<Vec<_>>()
            .join(", ")
    }

    /// Re-check every recorded instantiation (see the module docs). Runs after the body pass.
    pub(crate) fn check_instantiations(&mut self, module: &'a Module) {
        if self.inst_calls.is_empty() || self.diags.iter().any(|d| d.is_error()) {
            self.inst_calls.clear();
            return;
        }
        let mut bodies: HashMap<Symbol, (&'a FnDecl, &'a Block, &'a [Attr])> = HashMap::default();
        for item in &module.items {
            if let ItemKind::Fn(f) = &item.kind {
                if let Some(body) = &f.body {
                    bodies
                        .entry(f.name.sym)
                        .or_insert((f, body, item.attrs.as_slice()));
                }
            }
        }
        // Everything lowering reads stays the generic pass's record.
        let types = self.types.clone();
        let call_dims = self.call_dims.clone();
        let builtin_calls = self.builtin_calls.clone();
        let method_calls = self.methods.resolution.calls.clone();

        let mut seen: HashSet<(Symbol, String)> = HashSet::default();
        let mut reported: HashSet<(Option<&'static str>, Span, String)> = HashSet::default();
        let mut next = 0;
        let mut checked = 0;
        while next < self.inst_calls.len() && checked < MAX_INSTANCES {
            let call = self.inst_calls[next].clone();
            next += 1;
            let with = self.subst_desc(&call.subst);
            if !seen.insert((call.callee, with.clone())) {
                continue;
            }
            let Some(&(f, body, attrs)) = bodies.get(&call.callee) else {
                continue;
            };
            checked += 1;
            let mark = self.diags.len();
            self.inst_subst = call.subst.iter().cloned().collect();
            self.check_fn(f, body, attrs);
            self.inst_subst.clear();
            let found = self.diags.split_off(mark);
            let label = format!(
                "required by this instantiation of `{}` with {with}",
                self.sym_str(call.callee)
            );
            for d in found {
                if !d.is_error() {
                    continue;
                }
                // The same body error under two instantiations, or reached twice, is one report.
                let key = (
                    d.code,
                    d.primary_span().unwrap_or(call.span),
                    d.message.clone(),
                );
                if reported.insert(key) {
                    self.diags.push(d.secondary(call.span, label.clone()));
                }
            }
        }
        self.types = types;
        self.call_dims = call_dims;
        self.builtin_calls = builtin_calls;
        self.methods.resolution.calls = method_calls;
        self.inst_calls.clear();
    }
}

#[cfg(test)]
mod tests {
    //! What a generic body is checked against, called or not (N19/N20). Each bounded generic body
    //! is checked once at an opaque `T` against the bound tables of `methods.rs` — whether or not
    //! anything instantiates it — and each concrete instantiation is checked again at its types.
    //! Operations on an UNbounded opaque `T` (`x + x`) are checked only per instantiation, like a
    //! C++ template: an uncalled body is not rejected for them (a roadmap sharp edge).

    use wukong_diag::Diagnostic;
    use wukong_span::{Interner, SourceId};

    /// The diagnostics `src` checks to (warnings included).
    fn check(src: &str) -> Vec<Diagnostic> {
        let mut interner = Interner::new();
        let (module, pd) = wukong_parser::parse_module(src, SourceId(0), &mut interner);
        assert!(pd.is_empty(), "parse errors: {pd:?}");
        crate::check(&module, &interner).1
    }

    /// The codes of the errors `src` checks to.
    fn error_codes(src: &str) -> Vec<&'static str> {
        check(src)
            .into_iter()
            .filter(|d| d.is_error())
            .filter_map(|d| d.code)
            .collect()
    }

    const SHAPE: &str = "trait Shape { fn area(self) -> f32; } \
                         struct Sq { s: f32 } \
                         impl Shape for Sq { fn area(self) -> f32 { return self.s * self.s; } }";

    #[test]
    fn an_uninstantiated_body_is_checked_against_its_bounds() {
        // Never called, and still a method its bound does not provide is E0315.
        assert_eq!(
            error_codes(&format!(
                "{SHAPE} fn f<T: Shape>(x: T) -> f32 {{ return x.nope(); }} \
                 fn main() -> i32 {{ return 0; }}"
            )),
            vec!["E0315"]
        );
        // The same through a `where` clause: the clause is a bound, not parsed and dropped.
        assert_eq!(
            error_codes(&format!(
                "{SHAPE} fn f<T>(x: T) -> f32 where T: Shape {{ return x.nope(); }} \
                 fn main() -> i32 {{ return 0; }}"
            )),
            vec!["E0315"]
        );
        // A method some type defines, called on an unbounded parameter: the missing bound.
        assert_eq!(
            error_codes(&format!(
                "{SHAPE} fn f<U>(x: U) -> f32 {{ return x.area(); }} \
                 fn main() -> i32 {{ return 0; }}"
            )),
            vec!["E0315"]
        );
        // Forwarding an unbounded parameter to a bounded generic, never called: E0410.
        assert_eq!(
            error_codes(&format!(
                "{SHAPE} fn g<T: Shape>(x: T) -> f32 {{ return x.area(); }} \
                 fn h<U>(y: U) -> f32 {{ return g(y); }} \
                 fn main() -> i32 {{ return 0; }}"
            )),
            vec!["E0410"]
        );
        // The bounded body that uses only its bound checks clean, called or not.
        assert!(error_codes(&format!(
            "{SHAPE} fn f<T: Shape>(x: T) -> f32 {{ return x.area(); }} \
             fn main() -> i32 {{ return 0; }}"
        ))
        .is_empty());
    }

    #[test]
    fn an_unbounded_operation_is_checked_per_instantiation() {
        // Uncalled, `x + x` on an opaque `T` is accepted: nothing says what `T` is.
        let body = "struct Pt { x: f32 } fn twice<T>(x: T) -> T { return x + x; }";
        assert!(error_codes(&format!("{body} fn main() -> i32 {{ return 0; }}")).is_empty());
        // Instantiated at a struct, the body is checked again at `T = Pt` and `+` is E0433 — at
        // the generic body, labelled with the call that required it.
        let called = format!(
            "{body} fn main() -> i32 {{ let p = Pt {{ x: 1.0 }}; let q = twice(p); \
             return q.x as i32; }}"
        );
        let diags = check(&called);
        let d = diags
            .iter()
            .find(|d| d.code == Some("E0433"))
            .unwrap_or_else(|| panic!("{diags:?}"));
        assert!(
            d.labels.iter().any(|l| l
                .message
                .contains("required by this instantiation of `twice`")),
            "{d:?}"
        );
        // Instantiated at a number, it checks clean.
        assert!(error_codes(&format!(
            "{body} fn main() -> i32 {{ return twice(2) as i32; }}"
        ))
        .is_empty());
    }

    #[test]
    fn a_const_generic_is_a_usize_dimension_by_design() {
        // A `const` generic is a compile-time dimension, so it is `usize`; any other type is E0523.
        assert_eq!(
            error_codes("fn f<const N: i32>(x: [f32; N]) -> f32 { return x[0]; }"),
            vec!["E0523"]
        );
        assert!(error_codes(
            "fn f<const N: usize>(x: [f32; N]) -> f32 { return x[0]; } \
             fn main() -> i32 { let a = [1.0, 2.0, 3.0]; return f(a) as i32; }"
        )
        .is_empty());
    }
}
