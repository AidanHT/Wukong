//! Names bound inside an or-pattern: `Shape::Circle(r) | Shape::Square(r) => r`.
//!
//! Sema requires every alternative to bind the same names at the same types (`E0423`, `E0424`),
//! so the arm reads one binding of each name, fed by whichever alternative matched — the FIRST
//! one, left to right, as in Rust. The names are bound only once the arm's pattern test passed
//! (`emit_match_arm_test`), so some alternative is known to match: the alternatives before the
//! last are re-tested in order and the last needs no test. Each alternative binds its names in a
//! scope of its own and branches to a join block, passing a scalar's value or an aggregate's
//! address (the by-pointer convention every aggregate binding follows) as a block parameter; the
//! join binds each name once, a scalar into a fresh slot as an identifier pattern would.
//!
//! A `match` arm with a GUARD is different, in Rust and here ([`FnLowerer::emit_guarded_or_arm`]):
//! the guard is tried with each alternative that matches, in order, and the arm is taken with the
//! first whose bindings pass it — `match (2, 5) { (v, _) | (_, v) if v > 4 => v, _ => 0 }` is 5,
//! not 0. Nested binding or-patterns expand leftmost first, as rustc's candidate expansion does:
//! `(A(x) | B(x), C(y) | D(y))` tries `(A, C)`, `(A, D)`, `(B, C)`, `(B, D)`. An or-pattern that
//! binds nothing is tested as one, since every alternative of it gives the guard the same values;
//! Rust runs the guard once per matching alternative of it too, which differs only in how often a
//! guard with side effects runs. The `&&` operands of an `if let` / `while let` chain are not a
//! guard — Rust binds the first alternative that matches and never retries — so the parser's
//! guard for them is lowered by the first rule.
//!
//! The re-tests read the scrutinee again, which is pure: a pattern test only loads and compares.

use wukong_ast::{Expr, FieldPat, PatKind, Pattern, VariantPat};
use wukong_mir::{BlockId, MirType, Op, ValueId};
use wukong_span::{Span, Symbol};
use wukong_types::Ty;

use crate::FnLowerer;

/// The most candidates a guarded or-pattern arm expands into ([`expand_binding_ors`]): six
/// two-way binding or-patterns in one arm. Past it the arm is a `C0001` rather than a guard
/// lowered hundreds of times.
const MAX_GUARD_CANDIDATES: usize = 64;

/// One name's join parameter: its value (a block parameter), its MIR type, whether it is a `[]T`.
type JoinParam = (ValueId, MirType, bool);

impl FnLowerer<'_> {
    /// Bind the names of the or-pattern `alts` matched against the scrutinee `(scrut, scrut_mir,
    /// scrut_ty)`, which some alternative is known to match (see the module docs).
    pub(crate) fn bind_or_pattern(
        &mut self,
        alts: &[Pattern],
        (scrut, scrut_mir, scrut_ty): (ValueId, &MirType, &Ty),
        span: Span,
    ) {
        let Some(first) = alts.first() else {
            return;
        };
        let mut names = Vec::new();
        binding_order(first, &mut names);
        let join = self.builder.new_block();
        let mut params: Option<Vec<JoinParam>> = None;
        for (k, alt) in alts.iter().enumerate() {
            let mut next = None;
            if k + 1 < alts.len() {
                if let Some(c) = self.pattern_cond(alt, scrut, scrut_mir, scrut_ty, alt.span) {
                    let yes = self.builder.new_block();
                    let no = self.builder.new_block();
                    self.builder.cond_br(c, yes, vec![], no, vec![]);
                    self.builder.switch_to(yes);
                    self.terminated = false;
                    next = Some(no);
                }
            }
            self.push_scope();
            self.bind_match_ident(alt, scrut, scrut_mir, scrut_ty);
            let args = self.join_args(&names, span);
            self.pop_scope();
            if !self.branch_to_join(join, args, &mut params, span) {
                return;
            }
            match next {
                Some(no) => {
                    self.builder.switch_to(no);
                    self.terminated = false;
                }
                // An alternative with no test always matches: the ones after it are dead.
                None => break,
            }
        }
        self.builder.switch_to(join);
        self.terminated = false;
        self.bind_joined(names, params.unwrap_or_default());
    }

    /// End the current (test) block of a guarded `match` arm whose pattern binds names inside an
    /// or-pattern: try each expansion of `pat` ([`expand_binding_ors`]) in order, testing it and,
    /// with its names bound, the guard `guard`; the first that passes branches to `body_bb` with
    /// its names' values, and when none does the arm falls to `next_bb`. `body_bb` becomes the
    /// join: the names are bound once there, in the scope the caller pushed, and the builder is
    /// left there for the body. The guard is lowered once per expansion, as Rust runs it once per
    /// matching alternative.
    pub(crate) fn emit_guarded_or_arm(
        &mut self,
        pat: &Pattern,
        (scrut, scrut_mir, scrut_ty): (ValueId, &MirType, &Ty),
        guard: &Expr,
        body_bb: BlockId,
        next_bb: BlockId,
    ) {
        let mut names = Vec::new();
        binding_order(pat, &mut names);
        let cands = expand_binding_ors(pat, MAX_GUARD_CANDIDATES);
        if cands.is_empty() {
            self.unsupported(
                pat.span,
                "a guarded arm whose or-patterns bind names in more than 64 combinations",
            );
            self.builder.br(next_bb, vec![]);
            self.builder.switch_to(body_bb);
            self.terminated = false;
            // Bound (in a block nothing reaches) only so the body does not cascade.
            self.bind_match_ident(pat, scrut, scrut_mir, scrut_ty);
            return;
        }
        let mut params: Option<Vec<JoinParam>> = None;
        let last = cands.len() - 1;
        for (k, cand) in cands.iter().enumerate() {
            let cont = if k < last {
                self.builder.new_block()
            } else {
                next_bb
            };
            if let Some(c) = self.pattern_cond(cand, scrut, scrut_mir, scrut_ty, cand.span) {
                let yes = self.builder.new_block();
                self.builder.cond_br(c, yes, vec![], cont, vec![]);
                self.builder.switch_to(yes);
                self.terminated = false;
            }
            self.push_scope();
            self.bind_match_ident(cand, scrut, scrut_mir, scrut_ty);
            let pass = self.builder.new_block();
            self.emit_guard(guard, pass, cont);
            self.builder.switch_to(pass);
            self.terminated = false;
            let args = self.join_args(&names, pat.span);
            self.pop_scope();
            if !self.branch_to_join(body_bb, args, &mut params, pat.span) {
                break;
            }
            if k < last {
                self.builder.switch_to(cont);
                self.terminated = false;
            }
        }
        self.builder.switch_to(body_bb);
        self.terminated = false;
        self.bind_joined(names, params.unwrap_or_default());
    }

    /// The values of `names` in the innermost scope, as join arguments: a scalar's value loaded
    /// from its slot, an aggregate's address. `None`, after a diagnostic, when one is unbound.
    fn join_args(&mut self, names: &[Symbol], span: Span) -> Option<Vec<JoinParam>> {
        let got: Vec<Option<(ValueId, MirType)>> = names
            .iter()
            .map(|n| self.scopes.last().and_then(|s| s.get(n)).cloned())
            .collect();
        let mut args = Vec::with_capacity(got.len());
        for g in got {
            let Some((v, mty)) = g else {
                self.unsupported(
                    span,
                    "a name not bound in every alternative of an or-pattern",
                );
                return None;
            };
            let is_slice = self.slice_slots.contains(&v);
            let arg = if matches!(mty, MirType::Array(..)) {
                v
            } else {
                self.builder.build(mty.clone(), Op::Load(v, mty.clone()))
            };
            args.push((arg, mty, is_slice));
        }
        Some(args)
    }

    /// Branch to `join` with `args`. The first branch creates `join`'s parameters from its
    /// arguments' types (an aggregate passes its address, a `Ptr`); every later one must agree.
    /// `false`, after a diagnostic, when there is nothing to branch with or the types disagree.
    fn branch_to_join(
        &mut self,
        join: BlockId,
        args: Option<Vec<JoinParam>>,
        params: &mut Option<Vec<JoinParam>>,
        span: Span,
    ) -> bool {
        let Some(args) = args else {
            return false;
        };
        match params {
            None => {
                let ps = args
                    .iter()
                    .map(|(_, mty, is_slice)| {
                        let pty = if matches!(mty, MirType::Array(..)) {
                            MirType::Ptr
                        } else {
                            mty.clone()
                        };
                        (self.builder.block_param(join, pty), mty.clone(), *is_slice)
                    })
                    .collect();
                *params = Some(ps);
            }
            Some(ps) => {
                if ps.iter().zip(&args).any(|((_, a, _), (_, b, _))| a != b) {
                    self.unsupported(span, "an or-pattern binding a name at two types");
                    return false;
                }
            }
        }
        self.builder
            .br(join, args.into_iter().map(|(v, ..)| v).collect());
        true
    }

    /// Bind each of `names` once, in the current scope, to its join parameter: an aggregate at
    /// its address, a scalar copied into a fresh slot as an identifier pattern binds it.
    fn bind_joined(&mut self, names: Vec<Symbol>, params: Vec<JoinParam>) {
        for (name, (param, mty, is_slice)) in names.into_iter().zip(params) {
            if matches!(mty, MirType::Array(..)) {
                self.bind_slice(name, param, mty, is_slice);
            } else {
                let slot = self.builder.alloca(mty.clone());
                self.builder.build_void(Op::Store {
                    ptr: slot,
                    value: param,
                });
                self.bind_slice(name, slot, mty, false);
            }
        }
    }
}

/// Whether `p` holds an or-pattern that binds a name — what [`expand_binding_ors`] splits.
/// Wildcard-free over `PatKind`, so a new pattern form is a compile error here.
pub(crate) fn has_binding_or(p: &Pattern) -> bool {
    match &p.kind {
        PatKind::Or(alts) => alts.iter().any(crate::pattern_has_binding),
        PatKind::Binding { sub, .. } => has_binding_or(sub),
        PatKind::Tuple(subs) => subs.iter().any(has_binding_or),
        PatKind::Variant { fields, .. } => match fields {
            VariantPat::Tuple(subs) => subs.iter().any(has_binding_or),
            VariantPat::Struct(fps) => fps.iter().any(|fp| has_binding_or(&fp.pat)),
        },
        PatKind::Struct { fields, .. } => fields.iter().any(|fp| has_binding_or(&fp.pat)),
        PatKind::Slice { prefix, suffix, .. } => prefix.iter().chain(suffix).any(has_binding_or),
        PatKind::Ident(_)
        | PatKind::Wildcard
        | PatKind::Unit
        | PatKind::Int { .. }
        | PatKind::Char(_)
        | PatKind::Str(_)
        | PatKind::Bool(_)
        | PatKind::Path(_)
        | PatKind::Range { .. } => false,
    }
}

/// The patterns `p` stands for once every or-pattern in it that binds a name is replaced by one
/// of its alternatives, in the order Rust tries a guard with them: the leftmost such or-pattern
/// varies slowest (`(A(x) | B(x), C(y) | D(y))` gives `(A, C)`, `(A, D)`, `(B, C)`, `(B, D)`). An
/// or-pattern that binds nothing is kept whole. Empty when there would be more than `cap`.
/// Wildcard-free over `PatKind`.
fn expand_binding_ors(p: &Pattern, cap: usize) -> Vec<Pattern> {
    let with = |kind: PatKind| Pattern {
        id: p.id,
        kind,
        span: p.span,
    };
    match &p.kind {
        PatKind::Or(alts) if alts.iter().any(crate::pattern_has_binding) => {
            let mut out = Vec::new();
            for a in alts {
                let e = expand_binding_ors(a, cap);
                if e.is_empty() || out.len() + e.len() > cap {
                    return Vec::new();
                }
                out.extend(e);
            }
            out
        }
        PatKind::Binding { name, sub } => expand_binding_ors(sub, cap)
            .into_iter()
            .map(|s| {
                with(PatKind::Binding {
                    name: *name,
                    sub: Box::new(s),
                })
            })
            .collect(),
        PatKind::Tuple(subs) => product(subs, cap)
            .into_iter()
            .map(|subs| with(PatKind::Tuple(subs)))
            .collect(),
        PatKind::Variant { path, fields } => match fields {
            VariantPat::Tuple(subs) => product(subs, cap)
                .into_iter()
                .map(|subs| {
                    with(PatKind::Variant {
                        path: path.clone(),
                        fields: VariantPat::Tuple(subs),
                    })
                })
                .collect(),
            VariantPat::Struct(fps) => field_product(fps, cap)
                .into_iter()
                .map(|fps| {
                    with(PatKind::Variant {
                        path: path.clone(),
                        fields: VariantPat::Struct(fps),
                    })
                })
                .collect(),
        },
        PatKind::Struct { path, fields, rest } => field_product(fields, cap)
            .into_iter()
            .map(|fields| {
                with(PatKind::Struct {
                    path: path.clone(),
                    fields,
                    rest: *rest,
                })
            })
            .collect(),
        PatKind::Slice {
            prefix,
            rest,
            rest_binding,
            suffix,
        } => {
            let elems: Vec<Pattern> = prefix.iter().chain(suffix).cloned().collect();
            product(&elems, cap)
                .into_iter()
                .map(|mut all| {
                    let suffix = all.split_off(prefix.len());
                    with(PatKind::Slice {
                        prefix: all,
                        rest: *rest,
                        rest_binding: rest_binding.clone(),
                        suffix,
                    })
                })
                .collect()
        }
        PatKind::Or(_)
        | PatKind::Ident(_)
        | PatKind::Wildcard
        | PatKind::Unit
        | PatKind::Int { .. }
        | PatKind::Char(_)
        | PatKind::Str(_)
        | PatKind::Bool(_)
        | PatKind::Path(_)
        | PatKind::Range { .. } => vec![p.clone()],
    }
}

/// Every choice of one expansion per pattern of `ps`, the first varying slowest; empty past `cap`.
fn product(ps: &[Pattern], cap: usize) -> Vec<Vec<Pattern>> {
    let mut out: Vec<Vec<Pattern>> = vec![Vec::new()];
    for p in ps {
        let e = expand_binding_ors(p, cap);
        if e.is_empty() || out.len() * e.len() > cap {
            return Vec::new();
        }
        out = out
            .into_iter()
            .flat_map(|prefix| {
                e.iter().map(move |x| {
                    let mut v = prefix.clone();
                    v.push(x.clone());
                    v
                })
            })
            .collect();
    }
    out
}

/// [`product`] over the sub-patterns of a struct (or struct-variant) pattern's fields.
fn field_product(fps: &[FieldPat], cap: usize) -> Vec<Vec<FieldPat>> {
    let pats: Vec<Pattern> = fps.iter().map(|fp| fp.pat.clone()).collect();
    product(&pats, cap)
        .into_iter()
        .map(|ps| {
            fps.iter()
                .zip(ps)
                .map(|(fp, pat)| FieldPat { name: fp.name, pat })
                .collect()
        })
        .collect()
}

/// The names `p` binds, each once, in source order (a nested or-pattern's first alternative
/// stands for all of them). Wildcard-free over `PatKind`, so a new pattern form is a compile error
/// here rather than a silently dropped binding.
fn binding_order(p: &Pattern, out: &mut Vec<Symbol>) {
    fn push(n: Symbol, out: &mut Vec<Symbol>) {
        if !out.contains(&n) {
            out.push(n);
        }
    }
    match &p.kind {
        PatKind::Ident(n) => push(*n, out),
        PatKind::Binding { name, sub } => {
            push(*name, out);
            binding_order(sub, out);
        }
        PatKind::Tuple(subs) => subs.iter().for_each(|s| binding_order(s, out)),
        PatKind::Or(alts) => {
            if let Some(a) = alts.first() {
                binding_order(a, out);
            }
        }
        PatKind::Variant { fields, .. } => match fields {
            VariantPat::Tuple(subs) => subs.iter().for_each(|s| binding_order(s, out)),
            VariantPat::Struct(fps) => fps.iter().for_each(|fp| binding_order(&fp.pat, out)),
        },
        PatKind::Struct { fields, .. } => fields.iter().for_each(|fp| binding_order(&fp.pat, out)),
        PatKind::Slice {
            prefix,
            rest_binding,
            suffix,
            ..
        } => prefix
            .iter()
            .chain(rest_binding.as_deref())
            .chain(suffix)
            .for_each(|s| binding_order(s, out)),
        PatKind::Wildcard
        | PatKind::Unit
        | PatKind::Int { .. }
        | PatKind::Char(_)
        | PatKind::Str(_)
        | PatKind::Bool(_)
        | PatKind::Path(_)
        | PatKind::Range { .. } => {}
    }
}

#[cfg(test)]
mod tests {
    use wukong_mir::{MirType, Op, Program};
    use wukong_span::{Interner, SourceId};

    fn lowered(src: &str) -> (Program, Interner) {
        let mut interner = Interner::new();
        let (module, pd) = wukong_parser::parse_module(src, SourceId(0), &mut interner);
        assert!(pd.is_empty(), "parse errors: {pd:?}");
        let (sema, sd) = wukong_sema::check(&module, &interner);
        assert!(sd.iter().all(|d| !d.is_error()), "sema errors: {sd:?}");
        let (program, diags) = crate::lower_program(&module, &sema, &mut interner);
        assert!(diags.iter().all(|d| !d.is_error()), "lowering: {diags:?}");
        (program, interner)
    }

    /// The MIR types of the parameters of every block of function `name` that has any.
    fn block_param_tys(src: &str, name: &str) -> Vec<Vec<MirType>> {
        let (p, i) = lowered(src);
        let f = p
            .funcs
            .iter()
            .find(|f| i.resolve(f.name) == name)
            .expect("function");
        f.blocks
            .iter()
            .filter(|b| !b.params.is_empty())
            .map(|b| b.params.iter().map(|v| f.value_type(*v).clone()).collect())
            .collect()
    }

    /// How many calls of `callee` the function `name` makes.
    fn calls(src: &str, name: &str, callee: &str) -> usize {
        let (p, i) = lowered(src);
        let f = p
            .funcs
            .iter()
            .find(|f| i.resolve(f.name) == name)
            .expect("function");
        let callee = i.get(callee).expect("interned");
        f.blocks
            .iter()
            .flat_map(|b| &b.insts)
            .filter(|x| matches!(&x.op, Op::Call { func, .. } if *func == callee))
            .count()
    }

    /// A scalar bound in two alternatives joins as one `i32` parameter; an aggregate as its
    /// address.
    #[test]
    fn alternatives_join_their_bindings() {
        let scalar = block_param_tys(
            "fn f(t: (i32, i32)) -> i32 { return match t { (0, v) | (v, 0) => v, _ => 9 }; }
             fn main() -> i32 { return 0; }",
            "f",
        );
        assert!(scalar.contains(&vec![MirType::I32]), "{scalar:?}");
        let aggregate = block_param_tys(
            "struct P { x: i32 } enum W { A(P), B(P) }
             fn f(w: W) -> i32 { return match w { W::A(p) | W::B(p) => p.x }; }
             fn main() -> i32 { return 0; }",
            "f",
        );
        assert!(aggregate.contains(&vec![MirType::Ptr]), "{aggregate:?}");
    }

    /// A guarded `match` arm tries its guard with each alternative that binds, as Rust does, so
    /// the guard is lowered once per alternative — leftmost or-pattern slowest, four for two
    /// two-way ones — while an `if let` chain's `&&` operands, which are no guard, run once with
    /// the first alternative that matched, and an or-pattern that binds nothing is tested whole.
    #[test]
    fn a_guard_is_tried_with_each_alternative() {
        const G: &str = "fn g(x: i32) -> bool { return x > 4; }";
        let per_alt = format!(
            "{G} fn f(t: (i32, i32)) -> i32 {{ return match t {{ (v, _) | (_, v) if g(v) => v, \
             _ => 0 }}; }} fn main() -> i32 {{ return 0; }}"
        );
        assert_eq!(calls(&per_alt, "f", "g"), 2);
        let nested = format!(
            "{G} fn f(t: ((i32, i32), (i32, i32))) -> i32 {{ return match t {{ \
             ((a, _) | (_, a), (b, _) | (_, b)) if g(a - b) => a * 10 + b, _ => 0 }}; }} \
             fn main() -> i32 {{ return 0; }}"
        );
        assert_eq!(calls(&nested, "f", "g"), 4);
        let chain = format!(
            "{G} fn f(t: (i32, i32)) -> i32 {{ if let (v, _) | (_, v) = t && g(v) {{ return v; }} \
             return 0; }} fn main() -> i32 {{ return 0; }}"
        );
        assert_eq!(calls(&chain, "f", "g"), 1);
        let unbound = format!(
            "{G} fn f(t: (i32, i32)) -> i32 {{ return match t {{ (1 | 2, v) if g(v) => v, \
             _ => 0 }}; }} fn main() -> i32 {{ return 0; }}"
        );
        assert_eq!(calls(&unbound, "f", "g"), 1);
    }

    /// Seven two-way binding or-patterns under one guard would lower the guard 128 times: the arm
    /// is a `C0001` instead, and six (64 expansions) still lower.
    #[test]
    fn a_guard_over_too_many_combinations_is_a_codegen_limit() {
        let arm = |n: usize| {
            let pats: Vec<String> = (0..n).map(|i| format!("(a{i}, _) | (_, a{i})")).collect();
            let pats = pats.join(", ");
            let ty = vec!["(i32, i32)"; n].join(", ");
            format!(
                "fn f(t: ({ty})) -> i32 {{ return match t {{ ({pats}) if a0 > a1 => a0, _ => 0 }}; \
                 }} fn main() -> i32 {{ return 0; }}"
            )
        };
        let codes = |src: &str| {
            let mut interner = Interner::new();
            let (module, pd) = wukong_parser::parse_module(src, SourceId(0), &mut interner);
            assert!(pd.is_empty(), "parse errors: {pd:?}");
            let (sema, _) = wukong_sema::check(&module, &interner);
            let (_, diags) = crate::lower_program(&module, &sema, &mut interner);
            diags.iter().filter_map(|d| d.code).collect::<Vec<_>>()
        };
        assert_eq!(codes(&arm(6)), Vec::<&str>::new());
        assert_eq!(codes(&arm(7)), ["C0001"]);
    }
}
