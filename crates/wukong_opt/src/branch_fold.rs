//! Jump threading over constant edge arguments, and folding a branch into a predecessor that
//! branches to the same place. Both run inside `simplify-cfg`.
//!
//! The two together are what turn a short-circuit condition back into one branch. `a && b` lowers
//! (`mir_build::lower_short_circuit`) to
//!
//! ```text
//!   A:    l = …; f = const.i1 0; cond_br l, R, M(f)
//!   R:    r = …;                 br M(r)
//!   M(p):                        cond_br p, X, Y
//! ```
//!
//! so `if a && b { X } else { Y }` costs a merge block and two branches, and the loop vectorizer —
//! which flattens at most one diamond — refuses every loop guarded that way. Threading retargets
//! `A`'s constant edge straight to `Y`; `M` is then `R`'s alone and `simplify-cfg` merges the two;
//! folding then speculates `R`'s compare into `A`:
//!
//! ```text
//!   A:    l = …; r = …; c = and l, r; cond_br c, X, Y
//! ```
//!
//! # Jump threading ([`thread_edges`]) — LLVM's JumpThreading, restricted to empty blocks
//!
//! A block `M` with no instructions whose terminator branches on one of its own parameters decides
//! its successor from the value an incoming edge passes. When that value is an `i1` constant, the
//! edge is retargeted to the decided successor directly, carrying `M`'s arguments for it with `M`'s
//! parameters replaced by the edge's own arguments. That needs three things:
//!
//! * `M`'s parameters are read **only** by `M`'s own terminator — anything else they reach would
//!   lose its dominating definition on the threaded path;
//! * `M` is **not a loop header** — threading an entering edge past a header would enter the loop
//!   somewhere else (an irreducible loop), so a block that dominates one of its predecessors is
//!   left alone;
//! * the new edge does not duplicate the predecessor's other successor (one edge per target).
//!
//! Every other value `M` passes on is defined outside `M`, so it dominates `M` and therefore every
//! predecessor of `M`: the threaded edge stays in SSA form. Nothing is evaluated differently — the
//! constant decides exactly the branch `M` would have taken.
//!
//! # Branch folding ([`fold_branches`]) — LLVM SimplifyCFG's FoldBranchToCommonDest
//!
//! `A: cond_br l, R, D` where `R` is reached only from `A`, computes only a few pure instructions
//! that cannot trap, and branches `cond_br r, …` with one edge to `D` carrying the same arguments as
//! `A`'s edge to `D`: then control reaches `D` exactly when `A` goes there directly or `R` takes
//! its `D` edge, and `R`'s instructions can run on every path through `A` at no risk. They are
//! moved into `A` and the two branches become one on `and`/`or` (with a `not` when `R` reaches `D`
//! on its true arm). The four cases, with `x` the edge of `R` that does not go to `D`:
//!
//! | `R` is `A`'s | `R` goes to `D` on | new condition | true → | false → |
//! |---|---|---|---|---|
//! | then | false | `l && r` | `x` | `D` |
//! | then | true | `l && !r` | `x` | `D` |
//! | else | true | `l \|\| r` | `D` | `x` |
//! | else | false | `l \|\| !r` | `D` | `x` |
//!
//! Speculation is restricted to instructions that produce a scalar value and cannot fault or have
//! an effect: constants, compares, `not`/`neg`, `select`, every binary operator except the integer
//! divisions and remainders, and casts other than float-to-integer. A load is never speculated: a
//! short-circuit is what keeps `i < n && a[i] > 0` from reading past `a`. Every speculated value is
//! a deterministic function of its operands, so the result is bit-identical on every backend.

use wukong_mir::{BinOp, BlockId, CastKind, Function, Inst, MirType, Op, Terminator, ValueId};

use crate::{each_op_use, each_term_use, CfgAnalyses};

/// The most instructions [`fold_branches`] will move from the folded block into its predecessor,
/// where they run on paths that used to skip them.
const MAX_SPECULATED: usize = 4;

/// The idom of an unreachable block.
const NO_IDOM: u32 = u32::MAX;

/// Does block `a` dominate block `b`? `false` when `b` is unreachable.
fn dominates(a: u32, b: u32, idom: &[u32]) -> bool {
    if idom.get(b as usize).copied().unwrap_or(NO_IDOM) == NO_IDOM {
        return false;
    }
    let mut x = b;
    loop {
        if x == a {
            return true;
        }
        let id = idom[x as usize];
        if id == x || id == NO_IDOM {
            return false;
        }
        x = id;
    }
}

fn fresh(f: &mut Function, ty: MirType) -> ValueId {
    let v = ValueId(f.value_types.len() as u32);
    f.value_types.push(ty);
    v
}

// ---------------------------------------------------------------------------------------------
// Jump threading
// ---------------------------------------------------------------------------------------------

/// Which edge of a predecessor's terminator is retargeted.
#[derive(Clone, Copy)]
enum EdgeSel {
    Br,
    Then,
    Else,
}

struct ThreadEdit {
    from: usize,
    sel: EdgeSel,
    to: BlockId,
    args: Vec<ValueId>,
}

/// Thread every incoming edge that passes a constant into an empty test block straight to the
/// successor that constant selects (see the module docs). Invalidates `cache` when it changes the
/// CFG, and returns whether it did.
pub(crate) fn thread_edges(f: &mut Function, cache: &mut CfgAnalyses) -> bool {
    let n = f.blocks.len();
    let entry = f.entry.0 as usize;
    // Candidates: no instructions, a two-way branch on one of the block's own parameters.
    let mut cond_param: Vec<Option<usize>> = vec![None; n];
    for (m, b) in f.blocks.iter().enumerate() {
        if m == entry || !b.insts.is_empty() {
            continue;
        }
        let Terminator::CondBr {
            cond,
            then_blk,
            else_blk,
            ..
        } = &b.term
        else {
            continue;
        };
        if then_blk != else_blk {
            cond_param[m] = b.params.iter().position(|p| p == cond);
        }
    }
    if cond_param.iter().all(Option::is_none) {
        return false;
    }
    // Every value read somewhere other than a candidate's own terminator reading that candidate's
    // own parameters.
    let mut used = vec![false; f.value_types.len()];
    for (bi, b) in f.blocks.iter().enumerate() {
        for inst in &b.insts {
            each_op_use(&inst.op, &mut |v| {
                if let Some(u) = used.get_mut(v.0 as usize) {
                    *u = true;
                }
            });
        }
        let own: &[ValueId] = if cond_param[bi].is_some() {
            &b.params
        } else {
            &[]
        };
        each_term_use(&b.term, &mut |v| {
            if !own.contains(&v) {
                if let Some(u) = used.get_mut(v.0 as usize) {
                    *u = true;
                }
            }
        });
    }
    let consts = int_consts(f);
    let edits = {
        let (idom, preds) = cache.idoms_and_preds(f);
        plan_threads(f, &cond_param, &used, &consts, idom, preds)
    };
    if edits.is_empty() {
        return false;
    }
    for ThreadEdit {
        from,
        sel,
        to,
        args,
    } in edits
    {
        match (&mut f.blocks[from].term, sel) {
            (Terminator::Br { target, args: a }, EdgeSel::Br) => {
                *target = to;
                *a = args;
            }
            (
                Terminator::CondBr {
                    then_blk,
                    then_args,
                    ..
                },
                EdgeSel::Then,
            ) => {
                *then_blk = to;
                *then_args = args;
            }
            (
                Terminator::CondBr {
                    else_blk,
                    else_args,
                    ..
                },
                EdgeSel::Else,
            ) => {
                *else_blk = to;
                *else_args = args;
            }
            _ => unreachable!("a thread edit is planned against this very terminator"),
        }
    }
    cache.invalidate();
    true
}

/// `ValueId -> integer constant`, dense over the value arena.
fn int_consts(f: &Function) -> Vec<Option<i128>> {
    let mut m = vec![None; f.value_types.len()];
    for b in &f.blocks {
        for inst in &b.insts {
            if let (Some(r), Op::ConstInt(n, _)) = (inst.result, &inst.op) {
                if let Some(slot) = m.get_mut(r.0 as usize) {
                    *slot = Some(*n);
                }
            }
        }
    }
    m
}

fn plan_threads(
    f: &Function,
    cond_param: &[Option<usize>],
    used: &[bool],
    consts: &[Option<i128>],
    idom: &[u32],
    preds: &[Vec<u32>],
) -> Vec<ThreadEdit> {
    let n = f.blocks.len();
    let mut edits = Vec::new();
    // One edit per predecessor per call keeps every plan valid against the unedited terminator.
    let mut edited = vec![false; n];
    for (m, k) in cond_param
        .iter()
        .enumerate()
        .filter_map(|(m, k)| k.map(|k| (m, k)))
    {
        if idom.get(m).copied().unwrap_or(NO_IDOM) == NO_IDOM {
            continue; // unreachable
        }
        let mb = &f.blocks[m];
        if mb
            .params
            .iter()
            .any(|p| used.get(p.0 as usize).copied().unwrap_or(true))
        {
            continue;
        }
        // A loop header dominates one of its predecessors (the back edge's source).
        if preds[m].iter().any(|&p| dominates(m as u32, p, idom)) {
            continue;
        }
        let Terminator::CondBr {
            then_blk,
            then_args,
            else_blk,
            else_args,
            ..
        } = &mb.term
        else {
            continue;
        };
        // The constant an edge passes for the condition parameter, as the branch it decides.
        let decide = |args: &[ValueId]| -> Option<bool> {
            if args.len() != mb.params.len() {
                return None;
            }
            let c = consts.get(args[k].0 as usize).copied().flatten()?;
            Some(c & 1 != 0)
        };
        for &p in &preds[m] {
            let p = p as usize;
            if edited[p] {
                continue;
            }
            // (edge, decided successor, the edge's arguments, the predecessor's other successor)
            let found = match &f.blocks[p].term {
                Terminator::Br { target, args } if target.0 as usize == m => {
                    decide(args.as_slice()).map(|t| (EdgeSel::Br, t, args.as_slice(), None))
                }
                Terminator::CondBr {
                    then_blk: pt,
                    then_args: pta,
                    else_blk: pe,
                    else_args: pea,
                    ..
                } if pt != pe => {
                    if pt.0 as usize == m {
                        decide(pta.as_slice())
                            .map(|t| (EdgeSel::Then, t, pta.as_slice(), Some(*pe)))
                    } else if pe.0 as usize == m {
                        decide(pea.as_slice())
                            .map(|t| (EdgeSel::Else, t, pea.as_slice(), Some(*pt)))
                    } else {
                        None
                    }
                }
                _ => None,
            };
            let Some((sel, taken, edge_args, other)) = found else {
                continue;
            };
            let (to, out) = if taken {
                (*then_blk, then_args)
            } else {
                (*else_blk, else_args)
            };
            if other == Some(to) {
                continue;
            }
            let args = out
                .iter()
                .map(|a| match mb.params.iter().position(|q| q == a) {
                    Some(j) => edge_args[j],
                    None => *a,
                })
                .collect();
            edits.push(ThreadEdit {
                from: p,
                sel,
                to,
                args,
            });
            edited[p] = true;
        }
    }
    edits
}

// ---------------------------------------------------------------------------------------------
// Branch folding
// ---------------------------------------------------------------------------------------------

/// One fold: `r` is merged into its only predecessor `a`.
struct Fold {
    a: usize,
    r: usize,
    /// `r` is `a`'s then-successor (the condition becomes `and`); else-successor (`or`).
    and: bool,
    /// `r` reaches the common destination on its *true* arm, so its condition is negated.
    negate_r: bool,
}

/// May `inst` run on a path that did not execute it before: a scalar result, no effect, no fault?
fn speculatable(f: &Function, inst: &Inst) -> bool {
    let Some(r) = inst.result else {
        return false;
    };
    if f.value_type(r).is_vector() {
        return false;
    }
    match &inst.op {
        Op::ConstInt(..) | Op::ConstFloat(..) => true,
        Op::Cmp(..) | Op::Not(_) | Op::Neg(_) | Op::Select(..) => true,
        Op::Bin(b, ..) => !matches!(b, BinOp::SDiv | BinOp::UDiv | BinOp::SRem | BinOp::URem),
        Op::Cast(k, ..) => !matches!(k, CastKind::FpToSi | CastKind::FpToUi),
        _ => false,
    }
}

/// Fold every branch whose successor re-tests to one of the branch's own destinations into a single
/// branch on the combined condition (see the module docs). Invalidates `cache` when it changes the
/// CFG, and returns whether it did.
pub(crate) fn fold_branches(f: &mut Function, cache: &mut CfgAnalyses) -> bool {
    if f.blocks
        .iter()
        .filter(|b| matches!(b.term, Terminator::CondBr { .. }))
        .count()
        < 2
    {
        return false;
    }
    let folds = {
        let preds = cache.predecessors(f);
        plan_folds(f, preds)
    };
    if folds.is_empty() {
        return false;
    }
    for fold in folds {
        apply_fold(f, fold);
    }
    cache.invalidate();
    true
}

fn plan_folds(f: &Function, preds: &[Vec<u32>]) -> Vec<Fold> {
    let n = f.blocks.len();
    let entry = f.entry.0 as usize;
    // A block takes part in at most one fold per call, so every plan reads unedited blocks.
    let mut touched = vec![false; n];
    let mut folds = Vec::new();
    for a in 0..n {
        if touched[a] {
            continue;
        }
        let Terminator::CondBr {
            then_blk,
            then_args,
            else_blk,
            else_args,
            ..
        } = &f.blocks[a].term
        else {
            continue;
        };
        if then_blk == else_blk {
            continue;
        }
        for r_is_then in [true, false] {
            let (r, d, dargs) = if r_is_then {
                (then_blk.0 as usize, *else_blk, else_args)
            } else {
                (else_blk.0 as usize, *then_blk, then_args)
            };
            if r == a || r == entry || touched[r] || preds[r] != [a as u32] {
                continue;
            }
            let rb = &f.blocks[r];
            if !rb.params.is_empty()
                || rb.insts.len() > MAX_SPECULATED
                || !rb.insts.iter().all(|i| speculatable(f, i))
            {
                continue;
            }
            let Terminator::CondBr {
                then_blk: tr,
                then_args: tra,
                else_blk: fr,
                else_args: fra,
                ..
            } = &rb.term
            else {
                continue;
            };
            if tr == fr
                || [tr.0, fr.0]
                    .iter()
                    .any(|&t| t as usize == a || t as usize == r)
            {
                continue;
            }
            // `and`: the common destination is reached when `R` is false (plain) or true (negated).
            // `or`: when `R` is true (plain) or false (negated).
            let (plain_to_d, negated_to_d) = if r_is_then {
                (*fr == d && fra == dargs, *tr == d && tra == dargs)
            } else {
                (*tr == d && tra == dargs, *fr == d && fra == dargs)
            };
            if !plain_to_d && !negated_to_d {
                continue;
            }
            folds.push(Fold {
                a,
                r,
                and: r_is_then,
                negate_r: !plain_to_d,
            });
            touched[a] = true;
            touched[r] = true;
            break;
        }
    }
    folds
}

fn apply_fold(f: &mut Function, fold: Fold) {
    let mut insts = std::mem::take(&mut f.blocks[fold.r].insts);
    let r_term = std::mem::replace(&mut f.blocks[fold.r].term, Terminator::Unreachable);
    let a_term = std::mem::replace(&mut f.blocks[fold.a].term, Terminator::Unreachable);
    let (
        Terminator::CondBr {
            cond: rc,
            then_blk: tr,
            then_args: tra,
            else_blk: fr,
            else_args: fra,
        },
        Terminator::CondBr {
            cond: l,
            then_blk: ta,
            then_args: taa,
            else_blk: fa,
            else_args: faa,
        },
    ) = (r_term, a_term)
    else {
        unreachable!("plan_folds selects two conditional branches");
    };
    let mut rcond = rc;
    if fold.negate_r {
        let v = fresh(f, MirType::I1);
        insts.push(Inst {
            result: Some(v),
            op: Op::Not(rc),
        });
        rcond = v;
    }
    let c = fresh(f, MirType::I1);
    let op = if fold.and { BinOp::And } else { BinOp::Or };
    insts.push(Inst {
        result: Some(c),
        op: Op::Bin(op, l, rcond),
    });
    // `x`: the edge of `R` that does not go to the common destination.
    let (x, xargs) = if fold.and != fold.negate_r {
        (tr, tra)
    } else {
        (fr, fra)
    };
    let term = if fold.and {
        Terminator::CondBr {
            cond: c,
            then_blk: x,
            then_args: xargs,
            else_blk: fa,
            else_args: faa,
        }
    } else {
        Terminator::CondBr {
            cond: c,
            then_blk: ta,
            then_args: taa,
            else_blk: x,
            else_args: xargs,
        }
    };
    let a = &mut f.blocks[fold.a];
    a.insts.extend(insts);
    a.term = term;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Dce, Pass, SimplifyCfg};
    use wukong_mir::{Builder, CmpOp, Program};
    use wukong_span::Interner;

    /// Call `f(x, y)` from `main` for each input pair, printing every result.
    fn run_xy(f: &Function, it: &mut Interner, inputs: &[(i128, i128)]) -> Vec<u8> {
        let mut m = Builder::new(it.intern("main"), MirType::I32);
        for &(x, y) in inputs {
            let a = m.build(MirType::I32, Op::ConstInt(x, MirType::I32));
            let b = m.build(MirType::I32, Op::ConstInt(y, MirType::I32));
            let r = m.build(
                MirType::I32,
                Op::Call {
                    func: f.name,
                    args: vec![a, b],
                },
            );
            m.build_void(Op::Call {
                func: it.intern("print"),
                args: vec![r],
            });
        }
        let z = m.build(MirType::I32, Op::ConstInt(0, MirType::I32));
        m.ret(Some(z));
        let mut p = Program::new();
        p.funcs = vec![f.clone(), m.finish()];
        let errs = wukong_mir::verify::verify_program(&p);
        assert!(errs.is_empty(), "{errs:?}");
        let main = it.intern("main");
        let (code, out) = wukong_interp::run_with_output(&p, main, it).unwrap();
        assert_eq!(code, 0);
        out
    }

    /// `f(x, y) = if x > 10 <op> y < 20 { 1 } else { 0 }`, shaped exactly as
    /// `lower_short_circuit` + `lower_if` emit it: a merge block whose one parameter is the
    /// short-circuit result and whose terminator branches on it.
    fn short_circuit(it: &mut Interner, is_and: bool) -> Function {
        let mut b = Builder::new(it.intern("f"), MirType::I32);
        let x = b.add_param(MirType::I32);
        let y = b.add_param(MirType::I32);
        let ten = b.build(MirType::I32, Op::ConstInt(10, MirType::I32));
        let l = b.build(MirType::I1, Op::Cmp(CmpOp::Sgt, x, ten));
        let rhs = b.new_block();
        let merge = b.new_block();
        let p = b.block_param(merge, MirType::I1);
        let short = b.build(
            MirType::I1,
            Op::ConstInt(if is_and { 0 } else { 1 }, MirType::I1),
        );
        if is_and {
            b.cond_br(l, rhs, vec![], merge, vec![short]);
        } else {
            b.cond_br(l, merge, vec![short], rhs, vec![]);
        }
        b.switch_to(rhs);
        let twenty = b.build(MirType::I32, Op::ConstInt(20, MirType::I32));
        let r = b.build(MirType::I1, Op::Cmp(CmpOp::Slt, y, twenty));
        b.br(merge, vec![r]);
        b.switch_to(merge);
        let yes = b.new_block();
        let no = b.new_block();
        b.cond_br(p, yes, vec![], no, vec![]);
        b.switch_to(yes);
        let one = b.build(MirType::I32, Op::ConstInt(1, MirType::I32));
        b.ret(Some(one));
        b.switch_to(no);
        let zero = b.build(MirType::I32, Op::ConstInt(0, MirType::I32));
        b.ret(Some(zero));
        b.finish()
    }

    const GRID: [(i128, i128); 8] = [
        (5, 5),
        (15, 5),
        (15, 25),
        (5, 25),
        (11, 19),
        (10, 19),
        (11, 20),
        (-3, -3),
    ];

    fn cond_brs(f: &Function) -> Vec<ValueId> {
        f.blocks
            .iter()
            .filter_map(|b| match &b.term {
                Terminator::CondBr { cond, .. } => Some(*cond),
                _ => None,
            })
            .collect()
    }

    fn def_op(f: &Function, v: ValueId) -> Option<&Op> {
        f.blocks
            .iter()
            .flat_map(|b| &b.insts)
            .find(|i| i.result == Some(v))
            .map(|i| &i.op)
    }

    #[test]
    fn a_short_circuit_and_becomes_one_branch_on_and() {
        let mut it = Interner::new();
        let mut f = short_circuit(&mut it, true);
        let before = run_xy(&f, &mut it, &GRID);
        // x > 10 && y < 20, per pair of GRID.
        assert_eq!(before, b"0\n1\n0\n0\n1\n0\n0\n0\n".to_vec());
        let mut cache = CfgAnalyses::default();
        assert!(SimplifyCfg.run_function(&mut f, &mut cache));
        Dce.run_function(&mut f, &mut cache);
        let conds = cond_brs(&f);
        assert_eq!(
            conds.len(),
            1,
            "{}",
            wukong_mir::print::print_function(&f, &it)
        );
        assert!(matches!(
            def_op(&f, conds[0]),
            Some(Op::Bin(BinOp::And, ..))
        ));
        assert_eq!(
            f.blocks.len(),
            3,
            "entry, then and else are all that is left"
        );
        assert!(wukong_mir::verify::verify_function(&f).is_empty());
        assert_eq!(run_xy(&f, &mut it, &GRID), before);
    }

    #[test]
    fn a_short_circuit_or_becomes_one_branch_on_or() {
        let mut it = Interner::new();
        let mut f = short_circuit(&mut it, false);
        let before = run_xy(&f, &mut it, &GRID);
        // x > 10 || y < 20, per pair of GRID.
        assert_eq!(before, b"1\n1\n1\n0\n1\n1\n1\n1\n".to_vec());
        let mut cache = CfgAnalyses::default();
        assert!(SimplifyCfg.run_function(&mut f, &mut cache));
        Dce.run_function(&mut f, &mut cache);
        let conds = cond_brs(&f);
        assert_eq!(
            conds.len(),
            1,
            "{}",
            wukong_mir::print::print_function(&f, &it)
        );
        assert!(matches!(def_op(&f, conds[0]), Some(Op::Bin(BinOp::Or, ..))));
        assert_eq!(run_xy(&f, &mut it, &GRID), before);
    }

    /// `A: cond_br (x > 10), R, D; R: cond_br (y < 20), D, E` — `R` reaches the common destination
    /// on its true arm, so the fold negates it: `E` exactly when `x > 10 && !(y < 20)`.
    #[test]
    fn a_common_destination_on_the_true_arm_folds_through_not() {
        let mut it = Interner::new();
        let mut b = Builder::new(it.intern("f"), MirType::I32);
        let x = b.add_param(MirType::I32);
        let y = b.add_param(MirType::I32);
        let ten = b.build(MirType::I32, Op::ConstInt(10, MirType::I32));
        let l = b.build(MirType::I1, Op::Cmp(CmpOp::Sgt, x, ten));
        let r_blk = b.new_block();
        let d = b.new_block();
        let e = b.new_block();
        b.cond_br(l, r_blk, vec![], d, vec![]);
        b.switch_to(r_blk);
        let twenty = b.build(MirType::I32, Op::ConstInt(20, MirType::I32));
        let r = b.build(MirType::I1, Op::Cmp(CmpOp::Slt, y, twenty));
        b.cond_br(r, d, vec![], e, vec![]);
        b.switch_to(d);
        let seven = b.build(MirType::I32, Op::ConstInt(7, MirType::I32));
        b.ret(Some(seven));
        b.switch_to(e);
        let nine = b.build(MirType::I32, Op::ConstInt(9, MirType::I32));
        b.ret(Some(nine));
        let mut f = b.finish();
        let before = run_xy(&f, &mut it, &GRID);
        assert_eq!(before, b"7\n7\n9\n7\n7\n7\n9\n7\n".to_vec());
        let mut cache = CfgAnalyses::default();
        assert!(fold_branches(&mut f, &mut cache));
        let conds = cond_brs(&f);
        let reachable = crate::cfg::reachable(&f);
        let live: Vec<ValueId> = f
            .blocks
            .iter()
            .zip(&reachable)
            .filter(|(_, r)| **r)
            .filter_map(|(b, _)| match &b.term {
                Terminator::CondBr { cond, .. } => Some(*cond),
                _ => None,
            })
            .collect();
        assert_eq!(live.len(), 1, "{conds:?}");
        assert!(matches!(def_op(&f, live[0]), Some(Op::Bin(BinOp::And, ..))));
        assert!(f
            .blocks
            .iter()
            .flat_map(|b| &b.insts)
            .any(|i| matches!(i.op, Op::Not(v) if v == r)));
        assert_eq!(run_xy(&f, &mut it, &GRID), before);
    }

    /// A successor that loads or divides is never speculated into its predecessor.
    #[test]
    fn loads_and_divisions_are_not_speculated() {
        for divide in [false, true] {
            let mut it = Interner::new();
            let mut b = Builder::new(it.intern("f"), MirType::I32);
            let x = b.add_param(MirType::I32);
            let y = b.add_param(MirType::I32);
            let slot = b.alloca(MirType::I32);
            b.build_void(Op::Store {
                ptr: slot,
                value: y,
            });
            let zero = b.build(MirType::I32, Op::ConstInt(0, MirType::I32));
            let l = b.build(MirType::I1, Op::Cmp(CmpOp::Ne, x, zero));
            let r_blk = b.new_block();
            let d = b.new_block();
            let e = b.new_block();
            b.cond_br(l, r_blk, vec![], d, vec![]);
            b.switch_to(r_blk);
            let v = if divide {
                b.build(MirType::I32, Op::Bin(BinOp::SDiv, y, x))
            } else {
                b.build(MirType::I32, Op::Load(slot, MirType::I32))
            };
            let r = b.build(MirType::I1, Op::Cmp(CmpOp::Sgt, v, zero));
            b.cond_br(r, e, vec![], d, vec![]);
            b.switch_to(d);
            b.ret(Some(zero));
            b.switch_to(e);
            b.ret(Some(v));
            let mut f = b.finish();
            assert!(!fold_branches(&mut f, &mut CfgAnalyses::default()));
        }
    }

    /// `M(p): cond_br p, Body, Exit` is a loop header (the body branches back to it), so the
    /// constant `true` the entry passes is not threaded past it into the body.
    #[test]
    fn a_loop_header_is_never_threaded() {
        let mut it = Interner::new();
        let mut b = Builder::new(it.intern("f"), MirType::I32);
        let _x = b.add_param(MirType::I32);
        let _y = b.add_param(MirType::I32);
        let yes = b.build(MirType::I1, Op::ConstInt(1, MirType::I1));
        let header = b.new_block();
        let p = b.block_param(header, MirType::I1);
        let body = b.new_block();
        let exit = b.new_block();
        b.br(header, vec![yes]);
        b.switch_to(header);
        b.cond_br(p, body, vec![], exit, vec![]);
        b.switch_to(body);
        let no = b.build(MirType::I1, Op::ConstInt(0, MirType::I1));
        b.br(header, vec![no]);
        b.switch_to(exit);
        let seven = b.build(MirType::I32, Op::ConstInt(7, MirType::I32));
        b.ret(Some(seven));
        let mut f = b.finish();
        assert!(!thread_edges(&mut f, &mut CfgAnalyses::default()));
        assert!(matches!(f.blocks[0].term, Terminator::Br { target, .. } if target == header));
        assert_eq!(run_xy(&f, &mut it, &[(0, 0)]), b"7\n".to_vec());
    }

    /// A test block whose parameter is read past its own terminator keeps every edge: the
    /// threaded path would bypass the parameter's definition.
    #[test]
    fn a_parameter_read_downstream_blocks_threading() {
        let mut it = Interner::new();
        let mut b = Builder::new(it.intern("f"), MirType::I32);
        let x = b.add_param(MirType::I32);
        let _y = b.add_param(MirType::I32);
        let zero = b.build(MirType::I32, Op::ConstInt(0, MirType::I32));
        let l = b.build(MirType::I1, Op::Cmp(CmpOp::Sgt, x, zero));
        let yes = b.build(MirType::I1, Op::ConstInt(1, MirType::I1));
        let other = b.new_block();
        let m = b.new_block();
        let p = b.block_param(m, MirType::I1);
        b.cond_br(l, m, vec![yes], other, vec![]);
        b.switch_to(other);
        let lt = b.build(MirType::I1, Op::Cmp(CmpOp::Slt, x, zero));
        b.br(m, vec![lt]);
        b.switch_to(m);
        let t = b.new_block();
        let e = b.new_block();
        b.cond_br(p, t, vec![], e, vec![]);
        b.switch_to(t);
        let wide = b.build(MirType::I32, Op::Cast(CastKind::ZExt, p, MirType::I32));
        b.ret(Some(wide));
        b.switch_to(e);
        let two = b.build(MirType::I32, Op::ConstInt(2, MirType::I32));
        b.ret(Some(two));
        let mut f = b.finish();
        assert!(!thread_edges(&mut f, &mut CfgAnalyses::default()));
        assert_eq!(
            run_xy(&f, &mut it, &[(5, 0), (-5, 0), (0, 0)]),
            b"1\n1\n2\n".to_vec()
        );
    }

    /// From source: at `-O1` (no inlining) each short-circuit guard is one `cond_br` on an
    /// `and`/`or`, and every level computes the same answers.
    #[test]
    fn source_short_circuits_become_single_branches() {
        use wukong_span::SourceId;
        let src = concat!(
            "fn f(x: i32, y: i32) -> i32 { if x > 10 && y < 20 { return 1; } return 0; }\n",
            "fn g(x: i32, y: i32) -> i32 { if x > 10 || y < 20 { return 1; } return 0; }\n",
            "fn main() -> i32 { print(f(5, 5)); print(f(15, 5)); print(f(15, 25)); ",
            "print(f(11, 19)); print(g(5, 25)); print(g(15, 25)); print(g(5, 5)); ",
            "return 0; }\n",
        );
        let mut interner = Interner::new();
        let (module, parse) = wukong_parser::parse_module(src, SourceId(0), &mut interner);
        assert!(parse.iter().all(|d| !d.is_error()), "{parse:?}");
        let (sema, check) = wukong_sema::check(&module, &interner);
        assert!(check.iter().all(|d| !d.is_error()), "{check:?}");
        let (program, lower) = wukong_mir_build::lower_program(&module, &sema, &mut interner);
        assert!(lower.iter().all(|d| !d.is_error()), "{lower:?}");
        let main = interner.intern("main");
        for level in [0u8, 1, 2] {
            let mut optimized = program.clone();
            crate::optimize(&mut optimized, level);
            assert!(wukong_mir::verify::verify_program(&optimized).is_empty());
            if level == 1 {
                for (name, op) in [("f", BinOp::And), ("g", BinOp::Or)] {
                    let func = optimized.function(interner.intern(name)).unwrap();
                    let conds = cond_brs(func);
                    assert_eq!(
                        conds.len(),
                        1,
                        "{}",
                        wukong_mir::print::print_function(func, &interner)
                    );
                    assert!(
                        matches!(def_op(func, conds[0]), Some(Op::Bin(b, ..)) if *b == op),
                        "{}",
                        wukong_mir::print::print_function(func, &interner)
                    );
                }
            }
            let (exit, stdout) =
                wukong_interp::run_with_output(&optimized, main, &interner).unwrap();
            assert_eq!(exit, 0);
            // f: x > 10 && y < 20 at (5,5) (15,5) (15,25) (11,19); g: x > 10 || y < 20 at
            // (5,25) (15,25) (5,5).
            assert_eq!(stdout, b"0\n1\n0\n1\n0\n1\n1\n".to_vec(), "O{level}");
        }
    }
}
