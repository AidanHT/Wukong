//! The guard of an `if let` / `while let` chain with more than one `let`.
//!
//! The parser makes `if let P = e && c && let Q = f && d { A } else { B }` the match
//! `match e { P if c && (let Q = f) && d => { A }, _ => { B } }`: the chain after the first `let`
//! is the first arm's guard, a left fold of `&&` whose operands may be `ExprKind::Let`. Such a
//! guard is lowered operand by operand rather than as one boolean value: a condition branches to
//! the next operand or to the arm's failure block, and a `let` evaluates its scrutinee, tests the
//! pattern (`pattern_cond`, the same test a `match` arm makes) and, on success, binds the pattern's
//! names in the arm's scope — after the test, as a `match` arm binds (`emit_match_arm_test`) —
//! where the later operands and the arm's body read them. Evaluation is left to right with
//! short-circuit: nothing after a failing operand runs, and `B` runs when any operand fails.
//!
//! A guard with no `let` operand is lowered exactly as before, as one `i1` condition.

use wukong_ast::{BinOp, Expr, ExprKind};
use wukong_mir::{BlockId, MirType, Op};

use crate::FnLowerer;

impl FnLowerer<'_> {
    /// End the current block in the branch the arm guard `g` takes: to `pass` when it holds, to
    /// `fail` otherwise, binding the names of its `let` operands on the way to `pass`.
    pub(crate) fn emit_guard(&mut self, g: &Expr, pass: BlockId, fail: BlockId) {
        if !guard_has_let(g) {
            // Normalized to `i1` like every other boolean condition (`lower_bool_cond`).
            let gv = self.lower_bool_cond(g);
            self.builder.cond_br(gv, pass, vec![], fail, vec![]);
            return;
        }
        match &g.kind {
            ExprKind::Binary {
                op: BinOp::And,
                lhs,
                rhs,
            } => {
                let mid = self.builder.new_block();
                self.emit_guard(lhs, mid, fail);
                self.builder.switch_to(mid);
                self.terminated = false;
                self.emit_guard(rhs, pass, fail);
            }
            ExprKind::Let { pat, scrutinee } => {
                let scrut_mir = self.expr_mir(scrutinee);
                let scrut_ty = self.expr_ty(scrutinee);
                let scrut = self.lower_expr(scrutinee);
                let matched = self.builder.new_block();
                let c = self
                    .pattern_cond(pat, scrut, &scrut_mir, &scrut_ty, pat.span)
                    .unwrap_or_else(|| {
                        self.builder
                            .build(MirType::I1, Op::ConstInt(1, MirType::I1))
                    });
                self.builder.cond_br(c, matched, vec![], fail, vec![]);
                self.builder.switch_to(matched);
                self.terminated = false;
                self.bind_match_ident(pat, scrut, &scrut_mir, &scrut_ty);
                self.builder.br(pass, vec![]);
            }
            // `guard_has_let` admits only the two forms above.
            _ => {
                self.unsupported(g.span, "a `let` inside an `if let` chain operand");
                self.builder.br(fail, vec![]);
            }
        }
    }
}

/// Whether the guard `g` is (or folds with `&&`) a `let` operand.
fn guard_has_let(g: &Expr) -> bool {
    match &g.kind {
        ExprKind::Let { .. } => true,
        ExprKind::Binary {
            op: BinOp::And,
            lhs,
            rhs,
        } => guard_has_let(lhs) || guard_has_let(rhs),
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use wukong_mir::{CmpOp, Function, Op, Program, Terminator};
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

    fn func<'p>(p: &'p Program, interner: &Interner, name: &str) -> &'p Function {
        p.funcs
            .iter()
            .find(|f| interner.resolve(f.name) == name)
            .unwrap_or_else(|| panic!("no function `{name}`"))
    }

    /// The second `let`'s scrutinee (a call) is evaluated only after the first pattern and the
    /// condition between them held: the call sits in a block that only the condition's `true`
    /// edge reaches, and the condition in one only the first pattern's success reaches.
    #[test]
    fn a_later_let_runs_only_after_the_operands_before_it() {
        let (p, i) = lowered(
            "enum E { A(i32), B }
             fn probe(x: i32) -> i32 { return x * 2; }
             fn f(e: E, k: i32) -> i32 {
                 if let E::A(x) = e && x > k && let 2..=8 = probe(x) { return x; }
                 return 0;
             }
             fn main() -> i32 { return 0; }",
        );
        let f = func(&p, &i, "f");
        let probe = i.get("probe").expect("interned");
        let call_id = f
            .blocks
            .iter()
            .find(|b| {
                b.insts
                    .iter()
                    .any(|x| matches!(&x.op, Op::Call { func, .. } if *func == probe))
            })
            .expect("the call is lowered")
            .id;
        let cond = f
            .blocks
            .iter()
            .find(|b| {
                b.insts
                    .iter()
                    .any(|x| matches!(x.op, Op::Cmp(CmpOp::Sgt, ..)))
            })
            .expect("`x > k` is lowered");
        // Exactly one edge enters the call's block: the `true` edge of `x > k`.
        let preds: Vec<_> = f
            .blocks
            .iter()
            .filter(|b| match &b.term {
                Terminator::Br { target, .. } => *target == call_id,
                Terminator::CondBr {
                    then_blk, else_blk, ..
                } => *then_blk == call_id || *else_blk == call_id,
                _ => false,
            })
            .map(|b| b.id)
            .collect();
        assert_eq!(
            preds,
            vec![cond.id],
            "the call's block is entered from `x > k` only"
        );
        assert!(matches!(
            &cond.term,
            Terminator::CondBr { then_blk, .. } if *then_blk == call_id
        ));
        assert_ne!(
            cond.id, f.blocks[0].id,
            "`x > k` is not tested before the pattern matched"
        );
    }
}
