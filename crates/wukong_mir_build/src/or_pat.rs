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
//! The re-tests read the scrutinee again, which is pure: a pattern test only loads and compares.

use wukong_ast::{PatKind, Pattern, VariantPat};
use wukong_mir::{MirType, Op, ValueId};
use wukong_span::{Span, Symbol};
use wukong_types::Ty;

use crate::FnLowerer;

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
        // Per name: the join's parameter, the binding's MIR type and whether it is a `[]T`.
        let mut params: Option<Vec<(ValueId, MirType, bool)>> = None;
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
            let got: Vec<Option<(ValueId, MirType)>> = names
                .iter()
                .map(|n| self.scopes.last().and_then(|s| s.get(n)).cloned())
                .collect();
            self.pop_scope();
            let mut args = Vec::with_capacity(names.len());
            let mut kinds = Vec::with_capacity(names.len());
            for g in got {
                let Some((v, mty)) = g else {
                    self.unsupported(
                        span,
                        "a name not bound in every alternative of an or-pattern",
                    );
                    return;
                };
                let is_slice = self.slice_slots.contains(&v);
                let arg = if matches!(mty, MirType::Array(..)) {
                    v
                } else {
                    self.builder.build(mty.clone(), Op::Load(v, mty.clone()))
                };
                args.push(arg);
                kinds.push((mty, is_slice));
            }
            match &params {
                None => {
                    let ps = kinds
                        .into_iter()
                        .map(|(mty, is_slice)| {
                            let pty = if matches!(mty, MirType::Array(..)) {
                                MirType::Ptr
                            } else {
                                mty.clone()
                            };
                            (self.builder.block_param(join, pty), mty, is_slice)
                        })
                        .collect();
                    params = Some(ps);
                }
                Some(ps) => {
                    if ps.iter().zip(&kinds).any(|((_, a, _), (b, _))| a != b) {
                        self.unsupported(span, "an or-pattern binding a name at two types");
                        return;
                    }
                }
            }
            self.builder.br(join, args);
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
        for (name, (param, mty, is_slice)) in names.into_iter().zip(params.unwrap_or_default()) {
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
        | PatKind::Bool(_)
        | PatKind::Path(_)
        | PatKind::Range { .. } => {}
    }
}

#[cfg(test)]
mod tests {
    use wukong_mir::{MirType, Program};
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
}
