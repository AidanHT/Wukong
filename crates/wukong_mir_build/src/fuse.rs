//! The statement scan behind the AST autovectorizer's legality check ([`FnLowerer::vectorizable`])
//! and the same-range loop-fusion planner built on it ([`FnLowerer::fusable_prefix`]).
//!
//! Fusion asks, for a run of `R` adjacent loops over one range, for the longest prefix whose
//! concatenated body `vectorizable` accepts. Answering that by building and checking each candidate
//! — longest first, then again from the next statement position after a failure — cost O(R^2)
//! legality checks and O(R^3) deep-cloned statements for a run of non-fusable loops. The planner
//! answers it in one incremental pass with no clone, and gives exactly the same answer: every
//! condition `vectorizable` checks is either *prefix-monotone* (a statement that fails given the
//! statements before it, a write/access pair that conflicts — both fail every longer prefix too) or
//! *eventually true* (a lane type has been pinned, a statement exists — both hold for every longer
//! prefix once they hold). So the accepted prefix lengths form an interval ending at the first
//! monotone failure, and the longest accepted prefix is that end point or nothing.

use wukong_ast::{self as ast, ExprKind, Pattern, Stmt, StmtKind};
use wukong_mir::MirType;
use wukong_span::{FxHashSet as HashSet, Symbol};

use crate::{
    affine_stride, exprs_struct_eq, fusable_for, fuse_for_bodies, set_or_check, single_path,
    uses_any, vector_width, FnLowerer,
};

/// One array access the vectorizer recorded: `(base, index expression, unit stride?, is write)`.
pub(crate) type VecAccess<'e> = (Symbol, &'e ast::Expr, bool, bool);

/// The state [`FnLowerer::vectorizable`] threads through a body's statements, in order.
#[derive(Default)]
pub(crate) struct VecScan<'e> {
    /// Inner `let` names: they become vector temps and may not appear inside index expressions.
    pub(crate) locals: HashSet<Symbol>,
    /// Every array access, in statement order.
    pub(crate) acc: Vec<VecAccess<'e>>,
    /// The shared lane type, pinned by the first typed value or store.
    pub(crate) lane: Option<MirType>,
}

impl FnLowerer<'_> {
    /// Feed one statement of a vector body (loop variable `j`) into `scan`. `false` means the
    /// statement cannot be vectorized after the statements already fed — no nested control flow,
    /// calls or returns, and every store unit-stride with an index free of inner temps.
    pub(crate) fn vec_scan_stmt<'e>(&self, scan: &mut VecScan<'e>, s: &'e Stmt, j: Symbol) -> bool {
        match &s.kind {
            StmtKind::Let {
                pat:
                    Pattern {
                        kind: ast::PatKind::Ident(name),
                        ..
                    },
                init: Some(e),
                ..
            } => {
                if !self.vec_check_value(e, j, &scan.locals, &mut scan.lane, &mut scan.acc) {
                    return false;
                }
                scan.locals.insert(*name);
                true
            }
            StmtKind::Assign { target, op, value } => {
                if !self.vec_check_value(value, j, &scan.locals, &mut scan.lane, &mut scan.acc) {
                    return false;
                }
                // A compound assign lowers through `compound_binop` as ONE lane-wise op, so it is
                // vectorizable only where that op has a vector form on every backend: `+ - *`
                // always, `/` on a float lane (x86 has no SIMD integer divide — the same rule
                // `vec_check_value` applies to a written-out `/`), the bitwise ops on an integer
                // lane. `%` (no vector remainder) and the shifts (a per-lane shift amount Cranelift
                // cannot lower) keep the loop scalar instead of reaching the backend as MIR it
                // rejects. The answer depends only on this statement and the lane already pinned,
                // so it stays prefix-monotone for the fusion planner.
                let lane_float = scan.lane.as_ref().is_some_and(|l| l.is_float());
                let op_ok = match op {
                    ast::AssignOp::Assign
                    | ast::AssignOp::Add
                    | ast::AssignOp::Sub
                    | ast::AssignOp::Mul => true,
                    ast::AssignOp::Div => lane_float,
                    ast::AssignOp::BitAnd | ast::AssignOp::BitOr | ast::AssignOp::BitXor => {
                        !lane_float
                    }
                    ast::AssignOp::Rem | ast::AssignOp::Shl | ast::AssignOp::Shr => false,
                };
                if !op_ok {
                    return false;
                }
                match &target.kind {
                    // scalar reassignment of an inner vector temp (e.g. Horner `r = r*v + c`)
                    ExprKind::Path(p) if p.is_single() && scan.locals.contains(&p.first().sym) => {
                        true
                    }
                    // store to an array element at a unit-stride index
                    ExprKind::Index { base, indices } if indices.len() == 1 => {
                        let Some(bsym) = single_path(base) else {
                            return false;
                        };
                        if uses_any(&indices[0], &scan.locals) {
                            return false; // index must not depend on inner temps
                        }
                        if affine_stride(&indices[0], j) != Some(1) {
                            return false; // stores must be unit-stride
                        }
                        let Some(elem) = self.array_elem(base) else {
                            return false;
                        };
                        if set_or_check(&mut scan.lane, &elem).is_none() {
                            return false;
                        }
                        scan.acc.push((bsym, &indices[0], true, true));
                        true
                    }
                    _ => false,
                }
            }
            _ => false, // no nested control flow / calls / returns in a vector body
        }
    }

    /// Would vectorizing put write `w` and access `o` in one vector step unsafely? A vector load
    /// snapshots several scalar iterations before their stores, so every access is checked against
    /// each written buffer, not just accesses with the same spelling: `out[i + 1] = x[i] + 1` is a
    /// recurrence when a caller passes one array as `x` and `out`. Unproved distinct bases can also
    /// be shifted views, so even equal index expressions are insufficient for them.
    pub(crate) fn vec_access_conflict(&self, w: &VecAccess<'_>, o: &VecAccess<'_>) -> bool {
        if w.0 == o.0 {
            !exprs_struct_eq(w.1, o.1)
        } else {
            !self.provably_disjoint(w.0, o.0)
        }
    }

    /// The longest prefix (at least 2) of `loops` — same-range `for` loops over loop variable `j`,
    /// each accepted by `fusable_for` — whose fused body (see `fuse_for_bodies`) `vectorizable`
    /// accepts, or `None`. One pass, no fused body built; see the module docs for why the answer is
    /// identical to checking each prefix longest-first.
    pub(crate) fn fusable_prefix(&self, loops: &[Stmt], j: Symbol) -> Option<usize> {
        let mut scan = VecScan::default();
        // `groups[a]` is the index of the loop that recorded access `a`.
        let mut groups: Vec<usize> = Vec::new();
        let mut lane_at: Option<usize> = None;
        let mut nonempty_at: Option<usize> = None;
        // Every prefix longer than `limit` loops fails a prefix-monotone check.
        let mut limit = loops.len();
        for (k, stmt) in loops.iter().enumerate() {
            let (_, _, body) = fusable_for(stmt)?;
            // A body's trailing expression is appended to the fused body as a statement after its
            // statements, and an expression statement is never vectorizable.
            let ok = body
                .stmts
                .iter()
                .all(|s| self.vec_scan_stmt(&mut scan, s, j))
                && body.tail.is_none();
            groups.resize(scan.acc.len(), k);
            if !ok {
                limit = k;
                break;
            }
            if nonempty_at.is_none() && !body.stmts.is_empty() {
                nonempty_at = Some(k);
            }
            if lane_at.is_none() && scan.lane.is_some() {
                lane_at = Some(k);
            }
        }
        // A conflicting pair fails every prefix that contains both of its loops.
        for (wi, w) in scan.acc.iter().enumerate() {
            if !w.3 || groups[wi] >= limit {
                continue;
            }
            for (oi, o) in scan.acc.iter().enumerate() {
                if groups[oi] < limit && self.vec_access_conflict(w, o) {
                    limit = limit.min(groups[wi].max(groups[oi]));
                }
            }
        }
        let accepted = limit >= 2
            && nonempty_at.is_some_and(|k| k < limit)
            && lane_at.is_some_and(|k| k < limit)
            && scan.lane.as_ref().and_then(vector_width).is_some();
        accepted.then_some(limit)
    }

    /// The fusion search `fusable_prefix` replaces: build each candidate prefix, longest first, and
    /// ask `vectorizable`. Kept as the reference the debug-build cross-check compares against.
    pub(crate) fn fusable_prefix_reference(&self, loops: &[Stmt], j: Symbol) -> Option<usize> {
        (2..=loops.len()).rev().find(|&m| {
            self.vectorizable(&fuse_for_bodies(&loops[..m]), j)
                .is_some()
        })
    }
}

#[cfg(test)]
mod tests {
    use crate::lower_program;
    use wukong_mir::Program;
    use wukong_span::{Interner, SourceId};

    /// Lower `src`, requiring no diagnostics and verifier-clean MIR. In a debug build every fusion
    /// run of at most 16 loops is also cross-checked against the reference search inside
    /// `try_fuse_run`, so lowering these programs IS the equivalence test.
    fn lower(src: &str) -> Program {
        let mut interner = Interner::new();
        let (module, pdiags) = wukong_parser::parse_module(src, SourceId(0), &mut interner);
        assert!(pdiags.is_empty(), "parse: {pdiags:?}");
        let (sema, sdiags) = wukong_sema::check(&module, &interner);
        assert!(sdiags.iter().all(|d| !d.is_error()), "sema: {sdiags:?}");
        let (program, diags) = lower_program(&module, &sema, &mut interner);
        assert!(diags.iter().all(|d| !d.is_error()), "lower: {diags:?}");
        for f in &program.funcs {
            let errors = wukong_mir::verify::verify_function(f);
            assert!(errors.is_empty(), "{errors:?}");
        }
        program
    }

    /// Runs that exercise every planner branch: a fusable pair cut by a shifted write that
    /// conflicts with the first loop, an empty body (fails until a later loop pins a lane), a body
    /// whose trailing `if` makes it unfusable, and a fusable tail pair.
    #[test]
    fn mixed_fusion_runs_match_the_reference_search() {
        lower(
            "fn mix() -> f32 {
                let mut a: [f32; 64] = [1.0; 64];
                let mut b: [f32; 64] = [0.0; 64];
                let mut c: [f32; 64] = [0.0; 64];
                for i in 0..63 { b[i] = a[i] + 1.0; }
                for i in 0..63 { c[i] = b[i] * 2.0; }
                for i in 0..63 { a[i + 1] = c[i]; }
                for i in 0..63 { }
                for i in 0..63 { b[i] = c[i] - a[i]; }
                for i in 0..63 { if a[i] > 2.0 { c[i] = 1.0; } }
                for i in 0..63 { c[i] = c[i] + b[i]; }
                for i in 0..63 { b[i] = c[i] * 0.5; }
                return a[5] + b[6] + c[7];
            }",
        );
        lower(
            "fn empty_first() -> f32 {
                let mut a: [f32; 32] = [1.0; 32];
                let mut b: [f32; 32] = [0.0; 32];
                for i in 0..32 { }
                for i in 0..32 { }
                for i in 0..32 { b[i] = a[i] + 1.0; }
                for i in 0..32 { a[i] = b[i] * 3.0; }
                return a[3] + b[4];
            }",
        );
    }

    /// 64 adjacent same-range loops, none fusable (each is a recurrence on itself). The search this
    /// replaced cloned O(R^3) statements here before emitting any MIR; the planner rejects each
    /// position after scanning its run once.
    #[test]
    fn long_run_of_unfusable_loops_lowers() {
        let body = "for i in 0..63 { a[i + 1] = a[i] * 0.5; }\n".repeat(64);
        let src = format!(
            "fn chain() -> f32 {{ let mut a: [f32; 64] = [1.0; 64]; {body} return a[63]; }}"
        );
        lower(&src);
    }
}
