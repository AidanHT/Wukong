//! Loop canonicalization — put every natural loop into one shape, so that later passes match on
//! structure instead of on the spelling that produced it.
//!
//! The compiler's fast paths today are the `wukong_mir_build` kernel recognizers, which run *before*
//! the optimizer and match an AST shape. A user temporary, a `while` instead of a `for`, or a
//! hoisted bound is enough to miss, and the fallback is a fully scalar loop. This pass is the first
//! half of the answer: canonicalize first, then let a structural pass (the vectorizer) fire on the
//! canonical form.
//!
//! # What it guarantees
//!
//! For every natural loop that it can transform:
//!
//! * **a preheader** — exactly one block outside the loop branches to the header, and it does so
//!   unconditionally. That block is a legal place to evaluate anything that must run once before the
//!   loop (a vector trip count, a hoisted base pointer, an alias guard). `licm` already declines to
//!   hoist out of a loop without one, so this widens LICM too.
//! * **a single latch** — exactly one back edge, so there is one place where the induction variable
//!   is bumped and one place to close a rewritten loop.
//! * **one exit-test polarity** — every conditional branch that leaves the loop takes its *false*
//!   arm out, so the test always reads "keep going". `while i < n { … }` and
//!   `loop { if i >= n { break; } … }` are the same loop written twice; before this they lowered to
//!   `cond_br (i < n), body, exit` and `cond_br (i >= n), exit, body`, which no structural matcher
//!   would recognize as one shape.
//!
//! # What it deliberately does not do
//!
//! * **It does not split exit blocks.** A dedicated exit would immediately be merged back by
//!   `simplify_cfg`'s straight-line merge whenever the exiting block ends in an unconditional
//!   branch, and this pass would then re-split it: the two passes would oscillate for the fixpoint's
//!   full 100 iterations. A consumer that needs a dedicated exit (an epilogue) must split it as part
//!   of its own rewrite, where it controls both ends.
//! * **It does not rotate top-tested loops into guarded do-whiles.** Measured over `tests/run` at
//!   `-O2`: 1065 of the 1079 natural loops test their single exit in the *header* and **none** test
//!   it in a latch, so there is no bottom-tested population to converge — every spelling the front
//!   end emits, `for`, `while` and `loop`+`break` alike, is already top-tested with a preheader.
//!   Rotation would be a codegen change (one fewer branch per iteration), and it would *cost* the
//!   analysis its trip count, because the exit test then compares `i + step` in the latch rather
//!   than `i` in the header, which `loop_info::exit_test` does not model. It is deferred on that
//!   evidence, not on difficulty.
//!
//! Every transform here is a pure CFG edge rewiring: it adds blocks and block parameters, and it
//! never moves, deletes or rewrites an instruction. That is what makes it obviously
//! behaviour-preserving, which the two differential gates then confirm.

use wukong_mir::{BasicBlock, BlockId, CmpOp, Function, MirType, Op, Terminator, ValueId};

use crate::fxhash::FxHashSet;
use crate::loop_info::{self, LoopForest, NaturalLoop};
use crate::{cfg, each_op_use, each_term_use, CfgAnalyses, Pass};

pub struct LoopCanon;

/// A cap on the rewrite loop below. Each transform strictly reduces the number of loops missing a
/// property, so this can only be reached by a bug; it turns a hypothetical hang into a no-op.
const MAX_REWRITES: usize = 256;

impl Pass for LoopCanon {
    fn name(&self) -> &'static str {
        "loop-canon"
    }

    fn run_function(&self, f: &mut Function, cache: &mut CfgAnalyses) -> bool {
        // Dominance is only defined on reachable blocks, and `loop_info` skips the unreachable ones
        // — but a preheader synthesized for a loop whose entering edge is dead would be dead too.
        // Prune first, exactly as `licm` does.
        if !cache.all_reachable(f) {
            cfg::prune_unreachable(f);
            cache.invalidate();
        }
        if f.blocks.len() < 2 {
            return false;
        }

        let mut changed = false;
        // Each rewrite adds a block, so the loop forest has to be rebuilt. Loops are few and the
        // steady state is "nothing to do", where this costs exactly one analysis.
        for _ in 0..MAX_REWRITES {
            let forest = analyze(f, cache);
            let Some(action) = pick(f, &forest) else {
                break;
            };
            match action {
                Action::Preheader(id) => insert_preheader(f, forest.get(id)),
                Action::Latch(id) => insert_latch(f, forest.get(id)),
                Action::FlipExit(blk) => flip_exit_test(f, blk),
            }
            cache.invalidate();
            changed = true;
        }
        changed
    }
}

/// Structure only: this pass rewires edges and never reads an induction variable, so it must not
/// pay for the value analyses — which it would then throw away after each rewrite.
fn analyze(f: &Function, cache: &mut CfgAnalyses) -> LoopForest {
    let (idom, preds) = cache.idoms_and_preds(f);
    loop_info::structure_with(f, idom, preds)
}

enum Action {
    Preheader(loop_info::LoopId),
    Latch(loop_info::LoopId),
    /// Negate the exit test of this exiting block and swap its arms.
    FlipExit(BlockId),
}

/// The first canonicalization the function still needs, outermost loop first. Returning one action
/// at a time (rather than a batch) keeps every mutation applied to a freshly analyzed forest, so no
/// transform ever reads a block set that a previous one invalidated.
fn pick(f: &Function, forest: &LoopForest) -> Option<Action> {
    for l in &forest.loops {
        if l.preheader.is_none() && preheader_is_legal(f, l) {
            return Some(Action::Preheader(l.id));
        }
    }
    for l in &forest.loops {
        if l.latches.len() > 1 && latch_is_legal(f, l) {
            return Some(Action::Latch(l.id));
        }
    }
    // The use-count scan walks the whole function, so it is computed only once and only after a
    // branch has passed the cheap shape test — in the steady state (every loop already canonical)
    // no candidate is found and the scan never runs.
    let mut multi_use: Option<FxHashSet<u32>> = None;
    for l in &forest.loops {
        let inside: FxHashSet<u32> = l.blocks.iter().map(|b| b.0).collect();
        for &(from, _) in &l.exits {
            let Some(cond) = inverted_exit_cond(f, from, &inside) else {
                continue;
            };
            let multi = multi_use.get_or_insert_with(|| values_used_more_than_once(f));
            if !multi.contains(&cond.0) {
                return Some(Action::FlipExit(from));
            }
        }
    }
    None
}

// ---------------------------------------------------------------------------------------------
// Edge rewiring primitives
// ---------------------------------------------------------------------------------------------

/// Mint a value id of type `ty`.
fn fresh_value(f: &mut Function, ty: MirType) -> ValueId {
    let id = ValueId(f.value_types.len() as u32);
    f.value_types.push(ty);
    id
}

/// Append a block. Block ids are positional (`f.blocks[i].id == BlockId(i)` — the verifier checks
/// it, and `cfg`'s DFS indexes by id), so a new block always goes on the end.
fn append_block(f: &mut Function, params: Vec<ValueId>, term: Terminator) -> BlockId {
    let id = BlockId(f.blocks.len() as u32);
    f.blocks.push(BasicBlock {
        id,
        params,
        insts: Vec::new(),
        term,
    });
    id
}

/// A fresh parameter list mirroring `block`'s parameters, and a forwarding block that passes them
/// straight on to `block`. This is the shared body of both transforms: an empty block whose only
/// job is to be the single predecessor of a header.
fn forwarding_block(f: &mut Function, target: BlockId) -> BlockId {
    let tys: Vec<MirType> = f.blocks[target.0 as usize]
        .params
        .iter()
        .map(|p| f.value_type(*p).clone())
        .collect();
    let params: Vec<ValueId> = tys.into_iter().map(|t| fresh_value(f, t)).collect();
    append_block(
        f,
        params.clone(),
        Terminator::Br {
            target,
            args: params,
        },
    )
}

/// Point every edge of `t` that goes to `from` at `to` instead, keeping the arguments.
fn redirect(t: &mut Terminator, from: BlockId, to: BlockId) {
    match t {
        Terminator::Br { target, .. } => {
            if *target == from {
                *target = to;
            }
        }
        Terminator::CondBr {
            then_blk, else_blk, ..
        } => {
            if *then_blk == from {
                *then_blk = to;
            }
            if *else_blk == from {
                *else_blk = to;
            }
        }
        Terminator::Ret(_) | Terminator::Unreachable => {}
    }
}

/// A block whose two `cond_br` arms both reach `target` with *different* argument lists cannot be
/// funnelled through one forwarding block: the single forwarded argument list would have to be both.
/// (Identical argument lists are fine — that edge is one CFG edge, and `simplify_cfg` folds the
/// branch to a `br` anyway.)
fn ambiguous_edge(f: &Function, from: BlockId, target: BlockId) -> bool {
    match &f.blocks[from.0 as usize].term {
        Terminator::CondBr {
            then_blk,
            then_args,
            else_blk,
            else_args,
            ..
        } => *then_blk == target && *else_blk == target && then_args != else_args,
        _ => false,
    }
}

// ---------------------------------------------------------------------------------------------
// Preheader
// ---------------------------------------------------------------------------------------------

fn preheader_is_legal(f: &Function, l: &NaturalLoop) -> bool {
    !l.entering.is_empty() && !l.entering.iter().any(|&e| ambiguous_edge(f, e, l.header))
}

/// Give `l` a preheader: one new empty block that every entering edge is routed through and that
/// branches unconditionally to the header.
///
/// This does not oscillate against `simplify_cfg`'s straight-line merge. That merge fuses `A` into
/// `B` when `A` ends in `br B` and is `B`'s only predecessor. The new block is only created when the
/// loop had *no* preheader, which means either several entering blocks (so it has several
/// predecessors and is not merged) or a single entering block that reaches the header by a
/// `cond_br` (so that block does not end in a `br` and is not merged). And the header itself always
/// keeps a second predecessor — its latch.
fn insert_preheader(f: &mut Function, l: &NaturalLoop) {
    let header = l.header;
    let entering = l.entering.clone();
    let ph = forwarding_block(f, header);
    for e in entering {
        redirect(&mut f.blocks[e.0 as usize].term, header, ph);
    }
}

// ---------------------------------------------------------------------------------------------
// Single latch
// ---------------------------------------------------------------------------------------------

fn latch_is_legal(f: &Function, l: &NaturalLoop) -> bool {
    !l.latches.iter().any(|&e| ambiguous_edge(f, e, l.header))
}

/// Funnel every back edge through one new latch block, so the loop has a single back edge.
///
/// The new block sits *inside* the loop (the header dominates it, and it reaches the header), and it
/// has one predecessor per original latch — two or more, so `simplify_cfg` will not merge it away.
fn insert_latch(f: &mut Function, l: &NaturalLoop) {
    let header = l.header;
    let latches = l.latches.clone();
    let lt = forwarding_block(f, header);
    for e in latches {
        redirect(&mut f.blocks[e.0 as usize].term, header, lt);
    }
}

// ---------------------------------------------------------------------------------------------
// Exit-test polarity
// ---------------------------------------------------------------------------------------------

/// Every value read by more than one instruction/terminator in `f`. A compare with a single use can
/// be negated in place; one with several cannot, because the other readers still want the original
/// sense.
fn values_used_more_than_once(f: &Function) -> FxHashSet<u32> {
    let mut seen = FxHashSet::default();
    let mut multi = FxHashSet::default();
    let mut note = |v: ValueId| {
        if !seen.insert(v.0) {
            multi.insert(v.0);
        }
    };
    for b in &f.blocks {
        for inst in &b.insts {
            each_op_use(&inst.op, &mut note);
        }
        each_term_use(&b.term, &mut note);
    }
    multi
}

/// The negation of an integer comparison predicate.
///
/// Float predicates are refused: every ordered float compare is false on a NaN, so `!(a < b)` is
/// **not** `a >= b` — both are false when either operand is NaN, and flipping the arms on that
/// basis would send a NaN iteration the other way.
fn negate_int_cmp(p: CmpOp) -> Option<CmpOp> {
    use CmpOp::*;
    Some(match p {
        Eq => Ne,
        Ne => Eq,
        Slt => Sge,
        Sle => Sgt,
        Sgt => Sle,
        Sge => Slt,
        Ult => Uge,
        Ule => Ugt,
        Ugt => Ule,
        Uge => Ult,
        Foeq | Fone | Folt | Fole | Fogt | Foge => return None,
    })
}

/// The condition of `from`'s branch when flipping it would canonicalize the loop: its `true` arm
/// currently *leaves* the loop, and the condition is an integer compare defined in this same block.
/// `None` when the branch is already canonical or is not a shape this rewrite handles.
///
/// Requiring the compare to live in `from` keeps the rewrite local: the negation is applied in
/// place, so a definition in another block — which other paths may reach without reaching this
/// branch — is out of scope. The caller still has to confirm the value has no other reader.
fn inverted_exit_cond(f: &Function, from: BlockId, inside: &FxHashSet<u32>) -> Option<ValueId> {
    let blk = &f.blocks[from.0 as usize];
    let Terminator::CondBr {
        cond,
        then_blk,
        else_blk,
        ..
    } = &blk.term
    else {
        return None;
    };
    // The true arm must be the one that leaves, and the false arm the one that stays.
    if then_blk == else_blk || inside.contains(&then_blk.0) || !inside.contains(&else_blk.0) {
        return None;
    }
    let inst = blk.insts.iter().find(|i| i.result == Some(*cond))?;
    let Op::Cmp(pred, ..) = &inst.op else {
        return None;
    };
    // A lane-wise compare has a vector result and can never be a branch condition, but check the
    // declared type rather than assume it.
    (!f.value_type(*cond).is_vector() && negate_int_cmp(*pred).is_some()).then_some(*cond)
}

/// Negate the exit test in place and swap the branch's arms, so the loop continues on `true`.
///
/// Behaviour-preserving by construction: `cond_br !c, B, X` runs `B` on exactly the states where
/// `cond_br c, X, B` ran `B`. The compare is single-use (checked above), so nothing else observes
/// the negated sense.
fn flip_exit_test(f: &mut Function, from: BlockId) {
    let blk = &mut f.blocks[from.0 as usize];
    let Terminator::CondBr {
        cond,
        then_blk,
        then_args,
        else_blk,
        else_args,
    } = &mut blk.term
    else {
        return;
    };
    let c = *cond;
    std::mem::swap(then_blk, else_blk);
    std::mem::swap(then_args, else_args);
    for inst in &mut blk.insts {
        if inst.result == Some(c) {
            if let Op::Cmp(pred, ..) = &mut inst.op {
                *pred = negate_int_cmp(*pred).expect("checked by flip_is_legal");
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fxhash::FxHashSet;
    use crate::PassManager;
    use wukong_mir::{Inst, Op, Program};
    use wukong_span::{Interner, SourceId};

    fn lower(src: &str) -> (Program, Interner) {
        let mut interner = Interner::new();
        let (module, pd) = wukong_parser::parse_module(src, SourceId(0), &mut interner);
        assert!(pd.iter().all(|d| !d.is_error()), "parse: {pd:?}");
        let (sema, sd) = wukong_sema::check(&module, &interner);
        assert!(sd.iter().all(|d| !d.is_error()), "sema: {sd:?}");
        let (program, _) = wukong_mir_build::lower_program(&module, &sema, &mut interner);
        (program, interner)
    }

    fn run_at(src: &str, opt: u8) -> i64 {
        let (mut program, mut interner) = lower(src);
        crate::optimize(&mut program, opt);
        for f in &program.funcs {
            assert!(
                wukong_mir::verify::verify_function(f).is_empty(),
                "verify after -O{opt}"
            );
        }
        let main = interner.intern("main");
        wukong_interp::run(&program, main, &interner).unwrap()
    }

    /// Run the pass by itself on one function until it reports no change.
    fn canon(f: &mut Function) -> bool {
        let mut cache = CfgAnalyses::default();
        LoopCanon.run_function(f, &mut cache)
    }

    /// The `-O2` pipeline with this pass removed, so a test can look at the MIR the optimizer used
    /// to produce and then apply the canonicalization by hand. Keep in step with
    /// [`PassManager::standard`].
    fn pipeline_without_canon() -> PassManager {
        let mut pm = PassManager::new();
        pm.add(Box::new(crate::Mem2Reg));
        pm.add(Box::new(crate::Simplify));
        pm.add(Box::new(crate::SimplifyCfg));
        pm.add(Box::new(crate::SimplifyPhis));
        pm.add(Box::new(crate::Dce));
        pm.add(Box::new(crate::Cse));
        pm.add(Box::new(crate::Dse));
        pm.add(Box::new(crate::Licm));
        pm
    }

    #[test]
    fn a_pass_over_canonical_loops_is_a_no_op() {
        // Every loop the front end emits already has a preheader and one latch, so the pass must
        // report "no change" — otherwise it would re-dirty every other pass on every fixpoint sweep.
        let src = "fn mm(a: [f32; 64], b: [f32; 64], mut c: [f32; 64], n: i32) { \
                     let mut i: i32 = 0; \
                     while i < n { let mut j: i32 = 0; \
                       while j < n { let mut s: f32 = 0.0; let mut k: i32 = 0; \
                         while k < n { s = s + a[i*n+k] * b[k*n+j]; k = k + 1; } \
                         c[i*n+j] = s; j = j + 1; } \
                       i = i + 1; } } \
                   fn main() -> i32 { return 0; }";
        let (mut program, mut interner) = lower(src);
        PassManager::standard(2).run(&mut program);
        let sym = interner.intern("mm");
        let f = program.function(sym).expect("fn mm").clone();
        let mut f2 = f.clone();
        assert!(!canon(&mut f2), "already canonical");
        assert_eq!(f2.blocks.len(), f.blocks.len());
    }

    #[test]
    fn a_conditionally_entered_loop_gets_a_preheader() {
        //   bb0 -> bb1 | bb2 ;  bb1 -> bb1 | bb2   (the header is entered by a conditional branch,
        // so there is nowhere to hoist to and `licm` gives up on the loop entirely).
        let mut f = build(
            vec![MirType::I1],
            vec![
                (
                    vec![],
                    vec![Inst {
                        result: Some(ValueId(0)),
                        op: Op::ConstInt(1, MirType::I1),
                    }],
                    cond_br(0, 1, 2),
                ),
                (vec![], vec![], cond_br(0, 1, 2)),
                (vec![], vec![], Terminator::Ret(None)),
            ],
        );
        assert!(wukong_mir::verify::verify_function(&f).is_empty());
        assert!(canon(&mut f), "a preheader is missing");
        assert!(
            wukong_mir::verify::verify_function(&f).is_empty(),
            "{:?}",
            wukong_mir::verify::verify_function(&f)
        );
        let forest = loop_info::analyze_function(&f);
        assert_eq!(forest.loops.len(), 1);
        let l = &forest.loops[0];
        assert!(l.preheader.is_some(), "preheader after canonicalization");
        assert_eq!(l.header, BlockId(1), "the header did not move");
        // Idempotent.
        assert!(!canon(&mut f), "second pass must be a no-op");
    }

    #[test]
    fn two_back_edges_are_funnelled_through_one_latch() {
        //   bb0 -> bb1 ; bb1 -> bb2 | bb3 ; bb2 -> bb1 ; bb3 -> bb1 | bb4
        let mut f = build(
            vec![MirType::I1],
            vec![
                (
                    vec![],
                    vec![Inst {
                        result: Some(ValueId(0)),
                        op: Op::ConstInt(1, MirType::I1),
                    }],
                    br(1),
                ),
                (vec![], vec![], cond_br(0, 2, 3)),
                (vec![], vec![], br(1)),
                (vec![], vec![], cond_br(0, 1, 4)),
                (vec![], vec![], Terminator::Ret(None)),
            ],
        );
        let before = loop_info::analyze_function(&f);
        assert_eq!(before.loops[0].latches.len(), 2);
        assert!(canon(&mut f));
        assert!(wukong_mir::verify::verify_function(&f).is_empty());
        let after = loop_info::analyze_function(&f);
        assert_eq!(after.loops.len(), 1);
        let l = &after.loops[0];
        assert_eq!(l.latches.len(), 1, "one back edge");
        assert!(l.preheader.is_some());
        // The blocks that used to be latches are still in the loop, and still reach the header.
        let blocks: FxHashSet<u32> = l.blocks.iter().map(|b| b.0).collect();
        assert!(blocks.contains(&2) && blocks.contains(&3));
        assert!(!canon(&mut f), "second pass must be a no-op");
    }

    #[test]
    fn header_parameters_are_forwarded_through_the_new_blocks() {
        // The interesting part of the rewrite: the header carries a loop-carried value, so both the
        // preheader and the funnelled latch have to take it as a parameter and pass it on.
        //   bb0 -> bb1 | bb4 (conditional entry: no preheader)
        //   bb1(p) -> bb2 | bb3 ; bb2 -> bb1(p+1) ; bb3 -> bb1(p+2) | bb4
        let mut f = build(
            vec![
                MirType::I1,  // v0 cond
                MirType::I32, // v1 zero
                MirType::I32, // v2 one
                MirType::I32, // v3 header param
                MirType::I32, // v4 = p + 1
                MirType::I32, // v5 = p + 2
                MirType::I32, // v6 two
            ],
            vec![
                (
                    vec![],
                    vec![
                        Inst {
                            result: Some(ValueId(0)),
                            op: Op::ConstInt(1, MirType::I1),
                        },
                        Inst {
                            result: Some(ValueId(1)),
                            op: Op::ConstInt(0, MirType::I32),
                        },
                        Inst {
                            result: Some(ValueId(2)),
                            op: Op::ConstInt(1, MirType::I32),
                        },
                        Inst {
                            result: Some(ValueId(6)),
                            op: Op::ConstInt(2, MirType::I32),
                        },
                    ],
                    Terminator::CondBr {
                        cond: ValueId(0),
                        then_blk: BlockId(1),
                        then_args: vec![ValueId(1)],
                        else_blk: BlockId(4),
                        else_args: vec![],
                    },
                ),
                (vec![3], vec![], cond_br(0, 2, 3)),
                (
                    vec![],
                    vec![Inst {
                        result: Some(ValueId(4)),
                        op: Op::Bin(wukong_mir::BinOp::Add, ValueId(3), ValueId(2)),
                    }],
                    Terminator::Br {
                        target: BlockId(1),
                        args: vec![ValueId(4)],
                    },
                ),
                (
                    vec![],
                    vec![Inst {
                        result: Some(ValueId(5)),
                        op: Op::Bin(wukong_mir::BinOp::Add, ValueId(3), ValueId(6)),
                    }],
                    Terminator::CondBr {
                        cond: ValueId(0),
                        then_blk: BlockId(1),
                        then_args: vec![ValueId(5)],
                        else_blk: BlockId(4),
                        else_args: vec![],
                    },
                ),
                (vec![], vec![], Terminator::Ret(None)),
            ],
        );
        assert!(
            wukong_mir::verify::verify_function(&f).is_empty(),
            "{:?}",
            wukong_mir::verify::verify_function(&f)
        );
        assert!(canon(&mut f));
        assert!(
            wukong_mir::verify::verify_function(&f).is_empty(),
            "{:?}",
            wukong_mir::verify::verify_function(&f)
        );
        let forest = loop_info::analyze_function(&f);
        assert_eq!(forest.loops.len(), 1);
        let l = &forest.loops[0];
        assert!(l.preheader.is_some());
        assert_eq!(l.latches.len(), 1);
        // The single latch now merges the two different increments, so the header parameter is no
        // longer a basic induction variable — but it is still a *parameter*, correctly typed and
        // fed from both paths, which is what the verifier above checks.
        assert_eq!(f.blocks[l.header.0 as usize].params.len(), 1);
        assert!(!canon(&mut f));
    }

    #[test]
    fn a_break_spelled_loop_converges_on_the_while_spelling() {
        // The same loop written twice. `while i < n { … }` lowers to `cond_br (i < n), body, exit`;
        // `loop { if i >= n { break; } … }` lowers to `cond_br (i >= n), exit, body`. After
        // canonicalization both read `cond_br (i < n), body, exit`.
        let whil = "fn k(mut o: [i32; 64], n: i32) { let mut i: i32 = 0; \
                    while i < n { o[i] = i * 2; i = i + 1; } } \
                    fn main() -> i32 { return 0; }";
        let brk = "fn k(mut o: [i32; 64], n: i32) { let mut i: i32 = 0; \
                   loop { if i >= n { break; } o[i] = i * 2; i = i + 1; } } \
                   fn main() -> i32 { return 0; }";

        let mut shapes = Vec::new();
        for src in [whil, brk] {
            let (mut program, mut interner) = lower(src);
            pipeline_without_canon().run(&mut program);
            let sym = interner.intern("k");
            let mut f = program.function(sym).expect("fn k").clone();
            let before = exit_shape(&f);
            canon(&mut f);
            assert!(
                wukong_mir::verify::verify_function(&f).is_empty(),
                "{:?}",
                wukong_mir::verify::verify_function(&f)
            );
            shapes.push((before, exit_shape(&f)));
            // Idempotent: the flipped test now continues on `true`, so nothing more to do.
            assert!(!canon(&mut f), "second pass must be a no-op");
        }
        let (while_before, while_after) = shapes[0].clone();
        let (brk_before, brk_after) = shapes[1].clone();
        assert_ne!(while_before, brk_before, "the two spellings start apart");
        assert_eq!(
            while_after, brk_after,
            "the two spellings must converge: while {while_after:?} vs break {brk_after:?}"
        );
        assert_eq!(
            while_before, while_after,
            "the `while` form is already canonical"
        );
    }

    /// `(predicate, does the true arm stay in the loop)` for the loop's single exit test.
    fn exit_shape(f: &Function) -> Vec<(String, bool)> {
        let forest = loop_info::analyze_function(f);
        let mut out = Vec::new();
        for l in &forest.loops {
            let inside: FxHashSet<u32> = l.blocks.iter().map(|b| b.0).collect();
            for &(from, _) in &l.exits {
                let blk = &f.blocks[from.0 as usize];
                if let Terminator::CondBr { cond, then_blk, .. } = &blk.term {
                    if let Some(Op::Cmp(p, ..)) = blk
                        .insts
                        .iter()
                        .find(|i| i.result == Some(*cond))
                        .map(|i| &i.op)
                    {
                        out.push((p.name().to_string(), inside.contains(&then_blk.0)));
                    }
                }
            }
        }
        out
    }

    #[test]
    fn a_multiply_used_exit_condition_is_left_alone() {
        // The compare feeds the branch *and* a stored value, so negating it in place would flip the
        // stored bool too. The pass must decline rather than rewrite.
        let src =
            "fn k(mut o: [i32; 64], n: i32) -> i32 { let mut i: i32 = 0; let mut last: i32 = 0; \
                   loop { let done: bool = i >= n; last = done as i32; if done { break; } \
                          o[i] = i; i = i + 1; } return last; } \
                   fn main() -> i32 { return 0; }";
        let (mut program, mut interner) = lower(src);
        pipeline_without_canon().run(&mut program);
        let sym = interner.intern("k");
        let mut f = program.function(sym).expect("fn k").clone();
        let before = f.clone();
        canon(&mut f);
        assert!(wukong_mir::verify::verify_function(&f).is_empty());
        // Whatever else canonicalization did, no compare changed sense.
        let preds = |g: &Function| -> Vec<u8> {
            g.blocks
                .iter()
                .flat_map(|b| &b.insts)
                .filter_map(|i| match &i.op {
                    Op::Cmp(p, ..) => Some(*p as u8),
                    _ => None,
                })
                .collect()
        };
        assert_eq!(
            preds(&before),
            preds(&f),
            "a multiply-used compare must not be negated"
        );
    }

    #[test]
    fn canonicalization_preserves_results() {
        // Behaviour is unchanged at every level, on loop shapes the front end really emits.
        let cases = [
            (
                "fn main() -> i32 { let mut s: i32 = 0; let mut i: i32 = 0; \
                 while i < 100 { s = s + i; i = i + 1; } return s; }",
                4950,
            ),
            (
                "fn main() -> i32 { let mut s: i32 = 0; \
                 for i in 0..50 { if i % 3 == 0 { continue; } s = s + (i as i32); } return s; }",
                817,
            ),
            (
                "fn main() -> i32 { let mut s: i32 = 0; let mut i: i32 = 0; \
                 while i < 40 { let mut j: i32 = 0; \
                   while j < 40 { if j > i { break; } s = s + 1; j = j + 1; } i = i + 1; } \
                 return s; }",
                // The inner loop breaks at j == i + 1, so it adds i + 1; sum over i in 0..39 is
                // 40 * 41 / 2.
                820,
            ),
            (
                "fn f(n: i32) -> i32 { let mut r: i32 = 1; let mut k: i32 = n; \
                 while k > 0 { r = r * k; k = k - 1; } return r; } \
                 fn main() -> i32 { return f(6); }",
                720,
            ),
        ];
        for (src, want) in cases {
            for lvl in [0, 1, 2, 3] {
                assert_eq!(run_at(src, lvl), want, "-O{lvl}: {src}");
            }
        }
    }

    // ---- hand-built CFG helpers ----

    fn build(
        value_types: Vec<MirType>,
        blocks: Vec<(Vec<u32>, Vec<Inst>, Terminator)>,
    ) -> Function {
        let mut interner = Interner::new();
        Function {
            name: interner.intern("t"),
            params: Vec::new(),
            ret: MirType::Void,
            blocks: blocks
                .into_iter()
                .enumerate()
                .map(|(i, (params, insts, term))| BasicBlock {
                    id: BlockId(i as u32),
                    params: params.into_iter().map(ValueId).collect(),
                    insts,
                    term,
                })
                .collect(),
            value_types,
            entry: BlockId(0),
            vec_kernels: Vec::new(),
        }
    }

    fn br(target: u32) -> Terminator {
        Terminator::Br {
            target: BlockId(target),
            args: Vec::new(),
        }
    }

    fn cond_br(cond: u32, t: u32, e: u32) -> Terminator {
        Terminator::CondBr {
            cond: ValueId(cond),
            then_blk: BlockId(t),
            then_args: Vec::new(),
            else_blk: BlockId(e),
            else_args: Vec::new(),
        }
    }
}
