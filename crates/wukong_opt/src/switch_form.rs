//! Multi-way branches: `switch` formation from compare chains, and folding of constant or
//! degenerate `switch`es.
//!
//! # Switch formation ([`FormSwitches`], once, after every loop transform, `-O2` and above)
//!
//! `match` lowers to one `cond_br (cmp.eq scrutinee, K)` per arm, chained through the *else* edges,
//! so dispatching an N-variant enum costs up to N compares. Formation finds such a chain — each
//! *link* block reached only from the previous test, doing nothing but computing its own compare
//! of the **same** scrutinee against a constant — and replaces the head's branch with one
//! `switch`, which Cranelift lowers to a jump table, bit tests or a balanced search tree. A link's
//! instructions (its constant and compare) are pure, cannot trap and are few, so they are hoisted
//! into the head, which dominates everything the link did; the bypassed links are then unreachable
//! and the cleanup that follows deletes them. Both `cmp.eq` (case on the then-edge) and `cmp.ne`
//! (case on the else-edge) tests are recognized, and a link whose value repeats an earlier case
//! ends the chain (it could never be taken).
//!
//! The semantics are exactly the chain's: a `switch` takes the case equal to the scrutinee, in the
//! interpreter by the same integer comparison `cmp.eq` makes. It runs once and last because a
//! multi-way terminator is the one shape the loop passes do not model — canonicalization,
//! vectorization and unrolling all match two-way branches, so a `switch` formed earlier could only
//! hide a loop from them.
//!
//! # Switch folding ([`fold_switches`], run by `simplify-cfg`)
//!
//! A `switch` on a constant, a `switch` with no cases, and a `switch` whose every edge carries the
//! same target and arguments become a `br`. All three are exact.
//!
//! # What is deliberately *not* folded: edges into `unreachable`
//!
//! LLVM's SimplifyCFG deletes a branch into an `unreachable` block, because its front ends promise
//! such a block is dead. Wukong's front end cannot promise that. Sema's exhaustiveness check
//! (E0405) is deliberately lenient — it fires only on a *certain* miss — so a value `match` over a
//! tuple, a `match` whose arms all carry guards, or one whose enum arms test a payload
//! (`Opt::Some(1)`) is accepted with a fall-through that real inputs reach, and `lower_match` ends
//! that fall-through in `unreachable`. The interpreter, which is the language specification, stops
//! there with "execution reached `unreachable`", and native code traps. Treating the edge as dead
//! would turn that loud stop into a silently wrong arm at `-O1` and above. So a chain whose last
//! test falls into a trap keeps the trap as the `switch`'s default; a jump table makes that default
//! free anyway.

use wukong_mir::{
    switch_case_value, BlockId, CmpOp, Function, Op, SwitchCase, Terminator, ValueId,
};

use crate::{CfgAnalyses, Pass};

/// The fewest distinct case values worth a `switch`. Up to three, Cranelift's own switch lowering
/// emits the same compare-and-branch sequence the chain already is, so forming one would buy
/// nothing — and the `gpu-native` backend declines the terminator.
const MIN_SWITCH_CASES: usize = 4;

/// The most instructions a chain link may carry. Links are hoisted into the head, so this bounds
/// the work executed on paths that would have left the chain earlier.
const MAX_LINK_INSTS: usize = 4;

// ---------------------------------------------------------------------------------------------
// Switch folding
// ---------------------------------------------------------------------------------------------

/// Fold constant and degenerate `switch`es to `br` (see the module docs). Changes successor
/// edges, so the caller must invalidate its CFG analyses when this returns true.
pub(crate) fn fold_switches(f: &mut Function) -> bool {
    if !f
        .blocks
        .iter()
        .any(|b| matches!(b.term, Terminator::Switch { .. }))
    {
        return false;
    }
    let consts = switchable_consts(f);
    let mut changed = false;
    for b in &mut f.blocks {
        let folded = match &b.term {
            Terminator::Switch {
                scrutinee,
                cases,
                default,
                default_args,
            } => fold_switch(*scrutinee, cases, *default, default_args, &consts),
            _ => None,
        };
        if let Some(t) = folded {
            b.term = t;
            changed = true;
        }
    }
    changed
}

/// The `br` a `switch` reduces to, if any.
fn fold_switch(
    scrutinee: ValueId,
    cases: &[SwitchCase],
    default: BlockId,
    default_args: &[ValueId],
    consts: &[Option<i128>],
) -> Option<Terminator> {
    let br = |target: BlockId, args: &[ValueId]| Terminator::Br {
        target,
        args: args.to_vec(),
    };
    // A constant scrutinee takes exactly one edge.
    if let Some(v) = consts.get(scrutinee.0 as usize).copied().flatten() {
        return Some(match cases.iter().find(|c| c.value == v) {
            Some(c) => br(c.target, &c.args),
            None => br(default, default_args),
        });
    }
    // No cases, or every edge to one place with one argument list: unconditional.
    if cases
        .iter()
        .all(|c| c.target == default && c.args.as_slice() == default_args)
    {
        return Some(br(default, default_args));
    }
    None
}

/// `ValueId -> canonical value` of every `i8`..`i64` integer constant, dense over the value arena.
fn switchable_consts(f: &Function) -> Vec<Option<i128>> {
    let mut m = vec![None; f.value_types.len()];
    for b in &f.blocks {
        for inst in &b.insts {
            if let (Some(r), Op::ConstInt(n, ty)) = (inst.result, &inst.op) {
                if let Some(slot) = m.get_mut(r.0 as usize) {
                    *slot = switch_case_value(*n, ty);
                }
            }
        }
    }
    m
}

// ---------------------------------------------------------------------------------------------
// Switch formation
// ---------------------------------------------------------------------------------------------

/// Form `switch` terminators from `cmp.eq`/`cmp.ne` chains on one scrutinee (see the module docs).
pub struct FormSwitches;

impl Pass for FormSwitches {
    fn name(&self) -> &'static str {
        "form-switches"
    }

    fn run_function(&self, f: &mut Function, cache: &mut CfgAnalyses) -> bool {
        let branches = f
            .blocks
            .iter()
            .filter(|b| matches!(b.term, Terminator::CondBr { .. }))
            .count();
        if branches < MIN_SWITCH_CASES {
            return false;
        }
        let chains = {
            let preds = cache.predecessors(f);
            find_chains(f, preds)
        };
        if chains.is_empty() {
            return false;
        }
        for chain in chains {
            apply(f, chain);
        }
        cache.invalidate();
        true
    }
}

/// `scrutinee == value` (or `!=`, when `eq` is false): what one block's branch tests.
#[derive(Clone, Copy)]
struct EqTest {
    scrutinee: ValueId,
    value: i128,
    eq: bool,
}

/// One chain to rewrite: `head` keeps its block and gets the `switch`; each link's instructions move
/// into it.
struct Chain {
    head: usize,
    links: Vec<usize>,
    scrutinee: ValueId,
    cases: Vec<SwitchCase>,
    default: BlockId,
    default_args: Vec<ValueId>,
}

/// The equality test each value computes, when it is `cmp.eq`/`cmp.ne` of a switchable integer
/// value against a constant (either operand order). Both operands constant is a fold for
/// `simplify`, not a switch.
fn eq_tests(f: &Function) -> Vec<Option<EqTest>> {
    let mut raw_const: Vec<Option<i128>> = vec![None; f.value_types.len()];
    for b in &f.blocks {
        for inst in &b.insts {
            if let (Some(r), Op::ConstInt(n, _)) = (inst.result, &inst.op) {
                if let Some(slot) = raw_const.get_mut(r.0 as usize) {
                    *slot = Some(*n);
                }
            }
        }
    }
    let konst = |v: ValueId| raw_const.get(v.0 as usize).copied().flatten();
    let mut tests = vec![None; f.value_types.len()];
    for b in &f.blocks {
        for inst in &b.insts {
            let (Some(r), Op::Cmp(op @ (CmpOp::Eq | CmpOp::Ne), lhs, rhs)) =
                (inst.result, &inst.op)
            else {
                continue;
            };
            let (x, n) = match (konst(*lhs), konst(*rhs)) {
                (None, Some(n)) => (*lhs, n),
                (Some(n), None) => (*rhs, n),
                _ => continue,
            };
            let Some(ty) = f.value_types.get(x.0 as usize) else {
                continue;
            };
            let Some(value) = switch_case_value(n, ty) else {
                continue;
            };
            if let Some(slot) = tests.get_mut(r.0 as usize) {
                *slot = Some(EqTest {
                    scrutinee: x,
                    value,
                    eq: *op == CmpOp::Eq,
                });
            }
        }
    }
    tests
}

/// A test block's two edges as `(case edge, continue edge)`.
type Edges<'a> = ((BlockId, &'a [ValueId]), (BlockId, &'a [ValueId]));

/// The equality test block `b` branches on, with its case and continue edges, if it is a test block:
/// a `cond_br` with two distinct targets on a recognized compare.
fn test_of<'a>(f: &'a Function, tests: &[Option<EqTest>], b: usize) -> Option<(EqTest, Edges<'a>)> {
    let Terminator::CondBr {
        cond,
        then_blk,
        then_args,
        else_blk,
        else_args,
    } = &f.blocks[b].term
    else {
        return None;
    };
    if then_blk == else_blk {
        return None;
    }
    let t = tests.get(cond.0 as usize).copied().flatten()?;
    let then_edge = (*then_blk, then_args.as_slice());
    let else_edge = (*else_blk, else_args.as_slice());
    Some(if t.eq {
        (t, (then_edge, else_edge))
    } else {
        (t, (else_edge, then_edge))
    })
}

/// Can block `b` be hoisted into a chain head: no parameters, and at most [`MAX_LINK_INSTS`]
/// instructions, each a constant or an integer compare (pure, and unable to trap)?
fn hoistable(f: &Function, b: usize) -> bool {
    let blk = &f.blocks[b];
    blk.params.is_empty()
        && blk.insts.len() <= MAX_LINK_INSTS
        && blk.insts.iter().all(|inst| match &inst.op {
            Op::ConstInt(..) => true,
            Op::Cmp(op, a, _) => {
                !op.is_float() && f.value_types.get(a.0 as usize).is_some_and(|t| t.is_int())
            }
            _ => false,
        })
}

fn find_chains(f: &Function, preds: &[Vec<u32>]) -> Vec<Chain> {
    let tests = eq_tests(f);
    let n = f.blocks.len();
    // `is_link[l]`: `l` continues the chain of its only predecessor.
    let mut is_link = vec![false; n];
    for p in 0..n {
        let Some((tp, (_, (cont, cont_args)))) = test_of(f, &tests, p) else {
            continue;
        };
        let l = cont.0 as usize;
        if l == p || l >= n || !cont_args.is_empty() || preds[l] != [p as u32] {
            continue;
        }
        let same = test_of(f, &tests, l).is_some_and(|(tl, _)| tl.scrutinee == tp.scrutinee);
        if same && hoistable(f, l) {
            is_link[l] = true;
        }
    }

    let mut chains = Vec::new();
    for head in 0..n {
        if is_link[head] {
            continue;
        }
        let Some((th, ((case_to, case_args), _))) = test_of(f, &tests, head) else {
            continue;
        };
        let mut cases = vec![SwitchCase {
            value: th.value,
            target: case_to,
            args: case_args.to_vec(),
        }];
        let mut links: Vec<usize> = Vec::new();
        let mut cur = head;
        loop {
            let Some((_, (_, (cont, _)))) = test_of(f, &tests, cur) else {
                break;
            };
            let l = cont.0 as usize;
            if l >= n || !is_link[l] || l == head || links.contains(&l) {
                break;
            }
            let Some((tl, ((to, args), _))) = test_of(f, &tests, l) else {
                break;
            };
            if cases.iter().any(|c| c.value == tl.value) {
                break; // a repeated value can never be taken; the chain stops before it
            }
            cases.push(SwitchCase {
                value: tl.value,
                target: to,
                args: args.to_vec(),
            });
            links.push(l);
            cur = l;
        }
        if cases.len() < MIN_SWITCH_CASES {
            continue;
        }
        let Some((_, (_, (default, default_args)))) = test_of(f, &tests, cur) else {
            continue;
        };
        chains.push(Chain {
            head,
            links,
            scrutinee: th.scrutinee,
            cases,
            default,
            default_args: default_args.to_vec(),
        });
    }
    chains
}

/// Hoist every link's instructions into the head, in chain order (each link's definitions precede
/// its uses in later links), and end the head in the `switch`. The links keep their terminators
/// and are left unreachable for the cleanup to delete.
fn apply(f: &mut Function, chain: Chain) {
    let mut hoisted = Vec::new();
    for &l in &chain.links {
        hoisted.append(&mut f.blocks[l].insts);
    }
    let head = &mut f.blocks[chain.head];
    head.insts.extend(hoisted);
    head.term = Terminator::Switch {
        scrutinee: chain.scrutinee,
        cases: chain.cases,
        default: chain.default,
        default_args: chain.default_args,
    };
}

/// The late stage `optimize` runs once at `-O2` and above, after the pipeline and unrolling: form
/// switches in every function, then — only where one formed — prune the link blocks it bypassed
/// and drop their dead compares. Not a member of `PassManager::standard`: a multi-way branch is the
/// one shape the loop passes do not model, so it may only appear once they are done. Public so a
/// replica of `optimize` (the bench's `compile-work` attribution) can run the same stage.
pub fn form_switches(program: &mut wukong_mir::Program) {
    for f in &mut program.funcs {
        let mut cache = CfgAnalyses::default();
        if !FormSwitches.run_function(f, &mut cache) {
            continue;
        }
        // Pruning exposes the dead compares, and removing them changes nothing structural, so this
        // settles in one or two rounds; the bound only guards a hypothetical oscillation.
        for _ in 0..4 {
            let mut again = crate::SimplifyCfg.run_function(f, &mut cache);
            again |= crate::Dce.run_function(f, &mut cache);
            if !again {
                break;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use wukong_mir::{BinOp, Builder, MirType, Program};
    use wukong_span::{Interner, SourceId};

    /// `f(x: i32) -> i64`: a `cmp.eq` chain on `x` over `values`, each case returning
    /// `100 + index`, the fall-through returning `-1` (or trapping, with `trap_at_end`). The head
    /// computes its own compare; every later test sits in a link block reached only through the
    /// previous test's else edge.
    fn chain_fn_with(
        it: &mut Interner,
        values: &[i128],
        ne_at: Option<usize>,
        trap_at_end: bool,
    ) -> Function {
        let mut b = Builder::new(it.intern("f"), MirType::I64);
        let x = b.add_param(MirType::I32);
        for (i, &v) in values.iter().enumerate() {
            let k = b.build(MirType::I32, Op::ConstInt(v, MirType::I32));
            let ne = ne_at == Some(i);
            let pred = if ne { CmpOp::Ne } else { CmpOp::Eq };
            let c = b.build(MirType::I1, Op::Cmp(pred, x, k));
            let arm = b.new_block();
            let next = b.new_block();
            if ne {
                b.cond_br(c, next, vec![], arm, vec![]);
            } else {
                b.cond_br(c, arm, vec![], next, vec![]);
            }
            b.switch_to(arm);
            let r = b.build(MirType::I64, Op::ConstInt(100 + i as i128, MirType::I64));
            b.ret(Some(r));
            b.switch_to(next);
        }
        // A fresh block is terminated `unreachable`: leaving the last one alone is the trap.
        if !trap_at_end {
            let r = b.build(MirType::I64, Op::ConstInt(-1, MirType::I64));
            b.ret(Some(r));
        }
        b.finish()
    }

    fn chain_fn(it: &mut Interner, values: &[i128], ne_at: Option<usize>) -> Function {
        chain_fn_with(it, values, ne_at, false)
    }

    /// Call `f` on each input from `main`, printing every result; the interpreter's verdict.
    fn try_run(f: &Function, it: &mut Interner, inputs: &[i128]) -> Result<Vec<u8>, String> {
        let mut m = Builder::new(it.intern("main"), MirType::I32);
        for &v in inputs {
            let a = m.build(MirType::I32, Op::ConstInt(v, MirType::I32));
            let r = m.build(
                MirType::I64,
                Op::Call {
                    func: f.name,
                    args: vec![a],
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
        let (code, out) = wukong_interp::run_with_output(&p, main, it)?;
        assert_eq!(code, 0);
        Ok(out)
    }

    fn run(f: &Function, it: &mut Interner, inputs: &[i128]) -> Vec<u8> {
        try_run(f, it, inputs).unwrap()
    }

    fn switches(f: &Function) -> Vec<&Terminator> {
        f.blocks
            .iter()
            .map(|b| &b.term)
            .filter(|t| matches!(t, Terminator::Switch { .. }))
            .collect()
    }

    fn cleanup(f: &mut Function) {
        let mut cache = CfgAnalyses::default();
        crate::SimplifyCfg.run_function(f, &mut cache);
        crate::Dce.run_function(f, &mut cache);
    }

    #[test]
    fn a_four_way_chain_becomes_one_switch_with_the_same_answers() {
        let mut it = Interner::new();
        let values = [3, -7, 0, 2_000_000_000];
        let mut f = chain_fn(&mut it, &values, None);
        let inputs = [3, -7, 0, 2_000_000_000, 1, -1, 4];
        let before = run(&f, &mut it, &inputs);
        // Each case returns 100 + its index; anything else returns -1.
        assert_eq!(before, b"100\n101\n102\n103\n-1\n-1\n-1\n".to_vec());

        assert!(FormSwitches.run_function(&mut f, &mut CfgAnalyses::default()));
        cleanup(&mut f);
        let sw = switches(&f);
        assert_eq!(
            sw.len(),
            1,
            "{}",
            wukong_mir::print::print_function(&f, &it)
        );
        let Terminator::Switch { cases, .. } = sw[0] else {
            unreachable!()
        };
        let got: Vec<i128> = cases.iter().map(|c| c.value).collect();
        assert_eq!(got, values.to_vec());
        // The bypassed link blocks and their compares are gone.
        assert!(!f
            .blocks
            .iter()
            .any(|b| matches!(b.term, Terminator::CondBr { .. })));
        assert_eq!(run(&f, &mut it, &inputs), before);
    }

    #[test]
    fn a_not_equal_test_contributes_its_else_edge() {
        let mut it = Interner::new();
        let values = [10, 20, 30, 40, 50];
        let mut f = chain_fn(&mut it, &values, Some(2));
        let inputs = [10, 20, 30, 40, 50, 60];
        let before = run(&f, &mut it, &inputs);
        assert_eq!(before, b"100\n101\n102\n103\n104\n-1\n".to_vec());
        assert!(FormSwitches.run_function(&mut f, &mut CfgAnalyses::default()));
        cleanup(&mut f);
        assert_eq!(switches(&f).len(), 1);
        assert_eq!(run(&f, &mut it, &inputs), before);
    }

    #[test]
    fn short_chains_and_mixed_scrutinees_are_left_alone() {
        let mut it = Interner::new();
        // Three cases: below MIN_SWITCH_CASES.
        let mut f = chain_fn(&mut it, &[1, 2, 3], None);
        assert!(!FormSwitches.run_function(&mut f, &mut CfgAnalyses::default()));

        // Four tests, but the third compares a different value: no chain reaches four.
        let mut b = Builder::new(it.intern("g"), MirType::I64);
        let x = b.add_param(MirType::I32);
        let y = b.add_param(MirType::I32);
        for (i, s) in [x, x, y, x].into_iter().enumerate() {
            let k = b.build(MirType::I32, Op::ConstInt(i as i128, MirType::I32));
            let c = b.build(MirType::I1, Op::Cmp(CmpOp::Eq, s, k));
            let arm = b.new_block();
            let next = b.new_block();
            b.cond_br(c, arm, vec![], next, vec![]);
            b.switch_to(arm);
            let r = b.build(MirType::I64, Op::ConstInt(i as i128, MirType::I64));
            b.ret(Some(r));
            b.switch_to(next);
        }
        let r = b.build(MirType::I64, Op::ConstInt(-1, MirType::I64));
        b.ret(Some(r));
        let mut g = b.finish();
        assert!(!FormSwitches.run_function(&mut g, &mut CfgAnalyses::default()));
    }

    #[test]
    fn a_link_with_real_work_ends_the_chain() {
        // The second test's block also computes an add: hoisting it would execute work on the
        // first case's path, so the chain may not run through it.
        let mut it = Interner::new();
        let mut b = Builder::new(it.intern("h"), MirType::I64);
        let x = b.add_param(MirType::I32);
        for i in 0..5 {
            let k = b.build(MirType::I32, Op::ConstInt(i, MirType::I32));
            if i == 1 {
                let _ = b.build(MirType::I32, Op::Bin(BinOp::Add, x, k));
            }
            let c = b.build(MirType::I1, Op::Cmp(CmpOp::Eq, x, k));
            let arm = b.new_block();
            let next = b.new_block();
            b.cond_br(c, arm, vec![], next, vec![]);
            b.switch_to(arm);
            let r = b.build(MirType::I64, Op::ConstInt(i, MirType::I64));
            b.ret(Some(r));
            b.switch_to(next);
        }
        let r = b.build(MirType::I64, Op::ConstInt(-1, MirType::I64));
        b.ret(Some(r));
        let mut h = b.finish();
        // The tail from test 1 onward (tests 1..=4) still qualifies: four cases headed at test 1.
        assert!(FormSwitches.run_function(&mut h, &mut CfgAnalyses::default()));
        let sw = switches(&h);
        assert_eq!(sw.len(), 1);
        let Terminator::Switch { cases, .. } = sw[0] else {
            unreachable!()
        };
        assert_eq!(
            cases.iter().map(|c| c.value).collect::<Vec<_>>(),
            vec![1, 2, 3, 4]
        );
        let inputs = [0, 1, 2, 3, 4, 5];
        assert_eq!(run(&h, &mut it, &inputs), b"0\n1\n2\n3\n4\n-1\n".to_vec());
    }

    #[test]
    fn a_trapping_fall_through_stays_the_default() {
        // An exhaustive-looking value match: the last test's else edge is the `unreachable` block
        // `lower_match` emits. Sema cannot always prove that block dead, so it must stay a trap.
        let mut it = Interner::new();
        let mut f = chain_fn_with(&mut it, &[0, 1, 2, 3], None, true);
        let hits = [0, 1, 2, 3];
        assert_eq!(run(&f, &mut it, &hits), b"100\n101\n102\n103\n".to_vec());
        assert!(try_run(&f, &mut it, &[7]).is_err(), "the chain traps on 7");

        assert!(FormSwitches.run_function(&mut f, &mut CfgAnalyses::default()));
        cleanup(&mut f);
        let sw = switches(&f);
        assert_eq!(sw.len(), 1);
        let Terminator::Switch { cases, default, .. } = sw[0] else {
            unreachable!()
        };
        assert_eq!(
            cases.len(),
            4,
            "every test is a case; none was folded into the default"
        );
        let trap = &f.blocks[default.0 as usize];
        assert!(trap.insts.is_empty() && matches!(trap.term, Terminator::Unreachable));
        assert_eq!(run(&f, &mut it, &hits), b"100\n101\n102\n103\n".to_vec());
        assert!(
            try_run(&f, &mut it, &[7]).is_err(),
            "and the switch still traps on 7"
        );
    }

    #[test]
    fn constant_and_degenerate_switches_fold_to_branches() {
        let mut it = Interner::new();
        let case = |value: i128, target: BlockId| SwitchCase {
            value,
            target,
            args: vec![],
        };

        // `0xFF` as an i8 constant is -1: the second case.
        let mut b = Builder::new(it.intern("s"), MirType::I64);
        let minus_one = b.build(MirType::I8, Op::ConstInt(-1, MirType::I8));
        let blocks: Vec<BlockId> = (0..3).map(|_| b.new_block()).collect();
        b.switch(
            minus_one,
            vec![case(0, blocks[0]), case(-1, blocks[1])],
            blocks[2],
            vec![],
        );
        for (i, &blk) in blocks.iter().enumerate() {
            b.switch_to(blk);
            let r = b.build(MirType::I64, Op::ConstInt(i as i128, MirType::I64));
            b.ret(Some(r));
        }
        let mut s = b.finish();
        assert!(fold_switches(&mut s));
        assert!(matches!(&s.blocks[0].term, Terminator::Br { target, .. } if *target == blocks[1]));
        assert!(!fold_switches(&mut s), "nothing left to fold");

        // Every edge to one block with one argument list, and a switch with no cases at all.
        for with_cases in [true, false] {
            let mut t = Builder::new(it.intern("t"), MirType::I64);
            let x = t.add_param(MirType::I16);
            let only = t.new_block();
            let cases = if with_cases {
                vec![case(1, only), case(2, only)]
            } else {
                Vec::new()
            };
            t.switch(x, cases, only, vec![]);
            t.switch_to(only);
            let r = t.build(MirType::I64, Op::ConstInt(5, MirType::I64));
            t.ret(Some(r));
            let mut t = t.finish();
            assert!(fold_switches(&mut t));
            assert!(matches!(&t.blocks[0].term, Terminator::Br { target, .. } if *target == only));
            assert!(wukong_mir::verify::verify_function(&t).is_empty());
        }
    }

    /// The fixture's six-variant `match` becomes exactly one six-case `switch` at `-O2` (its
    /// fall-through stays the trap default), and every variant still selects its own arm.
    #[test]
    fn dense_enum_match_forms_one_switch() {
        let src = include_str!("../../../tests/run/match_dense_enum_switch.wk");
        let mut interner = Interner::new();
        let (module, parse) = wukong_parser::parse_module(src, SourceId(0), &mut interner);
        assert!(parse.iter().all(|d| !d.is_error()), "{parse:?}");
        let (sema, check) = wukong_sema::check(&module, &interner);
        assert!(check.iter().all(|d| !d.is_error()), "{check:?}");
        let (program, lower) = wukong_mir_build::lower_program(&module, &sema, &mut interner);
        assert!(lower.iter().all(|d| !d.is_error()), "{lower:?}");
        let expected = b"55\n18\n2\n6\n23\n4\n28\n30\n".to_vec();
        let main = interner.intern("main");
        for level in [0u8, 2] {
            let mut optimized = program.clone();
            crate::optimize(&mut optimized, level);
            let errs = wukong_mir::verify::verify_program(&optimized);
            assert!(errs.is_empty(), "O{level}: {errs:?}");
            let sw: Vec<(&Function, &Terminator)> = optimized
                .funcs
                .iter()
                .flat_map(|f| f.blocks.iter().map(move |b| (f, &b.term)))
                .filter(|(_, t)| matches!(t, Terminator::Switch { .. }))
                .collect();
            if level == 0 {
                assert!(sw.is_empty(), "-O0 forms no switch");
            } else {
                assert_eq!(sw.len(), 1, "one switch at O{level}");
                let (f, Terminator::Switch { cases, default, .. }) = sw[0] else {
                    unreachable!()
                };
                assert_eq!(cases.len(), 6, "one case per variant");
                assert!(matches!(
                    f.blocks[default.0 as usize].term,
                    Terminator::Unreachable
                ));
            }
            let (exit, stdout) =
                wukong_interp::run_with_output(&optimized, main, &interner).unwrap();
            assert_eq!(exit, 0);
            assert_eq!(stdout, expected, "O{level}");
        }
    }
}
