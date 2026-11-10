//! Loop deletion: remove a loop that provably terminates and computes nothing anyone can observe.
//!
//! Dead-store elimination empties loops — the zero-fill of a scratch buffer nothing reads, the
//! store stream of a fused intermediate once CSE has forwarded its value — and DCE then strips the
//! address arithmetic, but the counted loop itself (header test, induction increment, back edge)
//! survives and still runs every iteration. Nothing else in the pipeline removes a loop, so an
//! emptied `[0.0; 4096]` initialization kept costing 4096 compare-and-branch trips per call.
//!
//! A natural loop is deleted when all of the following hold, in which case every execution of it
//! ends by taking its single exit edge with values computed outside it, having done nothing else
//! observable:
//!
//! * it has a preheader (a single entering block that branches unconditionally to the header), so
//!   redirecting that one branch bypasses the loop entirely;
//! * it has exactly one exit edge and no `return`/`unreachable` inside it;
//! * it contains no instruction with a side effect (store, call, vector kernel, barrier). Loads
//!   and trapping arithmetic whose results nothing outside the loop uses are removable by the same
//!   policy DCE already applies to them;
//! * no value defined inside it is used outside it, and the exit edge passes only values defined
//!   outside it (those dominate the preheader, so the preheader can pass them on itself);
//! * it contains no inner loop (inner loops are deleted first, innermost outward, across fixpoint
//!   sweeps), and its trip count is a compile-time constant whose induction variable never wraps
//!   (its last step stays inside the type), or a unit-step strict bound `i < n` / `i > n` that
//!   cannot step over the bound — so it provably terminates, and deleting it cannot turn a program
//!   that hangs into one that returns.
//!
//! The rewrite points the preheader's branch at the exit block with the exit edge's arguments and
//! prunes the now-unreachable loop blocks.

use wukong_mir::{BlockId, CmpOp, Function, MirType, Op, Terminator, ValueId};

use crate::fxhash::FxHashSet;
use crate::loop_info::{self, NaturalLoop, Stages, TripCount};
use crate::{cfg, each_op_use, each_term_use, CfgAnalyses, Pass};

pub struct LoopDelete;

/// A loop chosen for deletion: the branch to rewrite, where it now goes, and the loop's blocks.
struct Doomed {
    preheader: BlockId,
    exit: BlockId,
    args: Vec<ValueId>,
    blocks: FxHashSet<u32>,
}

impl Pass for LoopDelete {
    fn name(&self) -> &'static str {
        "loop-delete"
    }

    fn run_function(&self, f: &mut Function, cache: &mut CfgAnalyses) -> bool {
        let pruned = !cache.all_reachable(f);
        if pruned {
            cfg::prune_unreachable(f);
            cache.invalidate();
        }
        if f.blocks.len() < 2 {
            return pruned;
        }
        let rewrites = {
            let (idom, preds) = cache.idoms_and_preds(f);
            let mut forest = loop_info::structure_with(f, idom, preds);
            let mut picked: Vec<Doomed> = Vec::new();
            for k in 0..forest.loops.len() {
                if !shape_ok(f, &forest.loops[k]) {
                    continue;
                }
                loop_info::analyze_one(f, &mut forest, k, Stages::Ivs);
                let l = &forest.loops[k];
                if !terminates(f, l) {
                    continue;
                }
                let (Some(preheader), Some((exit, args))) = (l.preheader, removable(f, l)) else {
                    continue;
                };
                if exit == preheader {
                    continue;
                }
                picked.push(Doomed {
                    preheader,
                    exit,
                    args,
                    blocks: l.blocks.iter().map(|b| b.0).collect(),
                });
            }
            // Picked loops are disjoint (none contains an inner loop), but one whose preheader sits
            // inside another picked loop is only reachable through it: leave it to that deletion.
            let inside_other = |n: usize| {
                picked
                    .iter()
                    .enumerate()
                    .any(|(m, other)| m != n && other.blocks.contains(&picked[n].preheader.0))
            };
            (0..picked.len())
                .filter(|&n| !inside_other(n))
                .map(|n| {
                    let d = &picked[n];
                    (d.preheader, d.exit, d.args.clone())
                })
                .collect::<Vec<_>>()
        };
        if rewrites.is_empty() {
            return pruned;
        }
        for (preheader, exit, args) in rewrites {
            f.blocks[preheader.0 as usize].term = Terminator::Br { target: exit, args };
        }
        cfg::prune_unreachable(f);
        cache.invalidate();
        true
    }
}

/// The structural preconditions, checked before paying for the induction-variable analysis.
fn shape_ok(f: &Function, l: &NaturalLoop) -> bool {
    l.preheader.is_some()
        && l.exits.len() == 1
        && l.abnormal_exits.is_empty()
        && l.children.is_empty()
        && l.blocks.iter().all(|b| {
            !f.blocks[b.0 as usize]
                .insts
                .iter()
                .any(|i| i.op.has_side_effects())
        })
}

/// Does the loop provably run a finite number of times? A constant trip count that is exact (see
/// [`const_trip_is_exact`]), or an affine one whose induction variable moves one step at a time
/// toward a strict bound (`i < n` with step `+1` reaches `n` without overflow; `i <= n` with `n` at
/// the type's maximum never stops).
fn terminates(f: &Function, l: &NaturalLoop) -> bool {
    match &l.trip {
        TripCount::Const(n) => const_trip_is_exact(f, l, *n),
        TripCount::Affine {
            step, inclusive, ..
        } => !*inclusive && (*step == 1 || *step == -1),
        TripCount::Unknown => false,
    }
}

/// `loop_info` computes a constant trip count in `i64` from the literal endpoints, as though the
/// induction variable could not wrap. Hold it to that before relying on it: both endpoints must be
/// literals of the variable's own type that fit it, and so must the value the variable steps to
/// after the last iteration, `start + n·step`, read with the exit test's signedness. Then every value
/// the variable takes lies between the endpoints and that last step, the masked arithmetic of the
/// backends never wraps, and the loop runs exactly `n` times. Over `i8`, `while i < 127 { i = i + 2 }`
/// has the "trip count" 64, but `126 + 2` wraps to `-128`, which is still `< 127`: the loop never
/// ends, and deleting it would turn a hang into a return.
pub(crate) fn const_trip_is_exact(f: &Function, l: &NaturalLoop, n: u64) -> bool {
    let Some(iv) = l.primary() else {
        return false;
    };
    let Some(step) = iv.step.as_const() else {
        return false;
    };
    let bits: u32 = match iv.ty {
        MirType::I8 => 8,
        MirType::I16 => 16,
        MirType::I32 => 32,
        MirType::I64 => 64,
        _ => return false,
    };
    let def = move |v: ValueId| {
        f.blocks
            .iter()
            .flat_map(|b| &b.insts)
            .find(|i| i.result == Some(v))
            .map(|i| &i.op)
    };
    let Terminator::CondBr { cond, .. } = &f.blocks[l.header.0 as usize].term else {
        return false;
    };
    let Some(Op::Cmp(pred, a, b)) = def(*cond) else {
        return false;
    };
    let bound = if *a == iv.value {
        *b
    } else if *b == iv.value {
        *a
    } else {
        return false;
    };
    let (lo, hi): (i128, i128) = match pred {
        CmpOp::Slt | CmpOp::Sle | CmpOp::Sgt | CmpOp::Sge => {
            (-(1i128 << (bits - 1)), (1i128 << (bits - 1)) - 1)
        }
        CmpOp::Ult | CmpOp::Ule | CmpOp::Ugt | CmpOp::Uge => (0, (1i128 << bits) - 1),
        _ => return false,
    };
    let literal = move |v: ValueId| match def(v) {
        Some(Op::ConstInt(k, ty)) if *ty == iv.ty => Some(*k),
        _ => None,
    };
    let (Some(start), Some(end)) = (iv.start.and_then(literal), literal(bound)) else {
        return false;
    };
    let fits = |x: i128| (lo..=hi).contains(&x);
    let after = i128::from(step)
        .checked_mul(i128::from(n))
        .and_then(|d| start.checked_add(d));
    fits(start) && fits(end) && after.is_some_and(fits)
}

/// The exit block and the arguments the exit edge passes, when nothing defined in the loop is
/// observable outside it.
fn removable(f: &Function, l: &NaturalLoop) -> Option<(BlockId, Vec<ValueId>)> {
    let (from, to) = l.exits[0];
    let inside: FxHashSet<u32> = l.blocks.iter().map(|b| b.0).collect();
    let mut defined: FxHashSet<u32> = FxHashSet::default();
    for b in &l.blocks {
        let blk = &f.blocks[b.0 as usize];
        defined.extend(blk.params.iter().map(|p| p.0));
        defined.extend(blk.insts.iter().filter_map(|i| i.result.map(|r| r.0)));
    }
    let args: Vec<ValueId> = match &f.blocks[from.0 as usize].term {
        Terminator::Br { target, args } if *target == to => args.clone(),
        Terminator::CondBr {
            then_blk,
            then_args,
            else_blk,
            else_args,
            ..
        } => {
            if *then_blk == to && *else_blk != to {
                then_args.clone()
            } else if *else_blk == to && *then_blk != to {
                else_args.clone()
            } else {
                return None;
            }
        }
        _ => return None,
    };
    if args.iter().any(|a| defined.contains(&a.0)) {
        return None;
    }
    let mut escapes = false;
    for blk in f.blocks.iter().filter(|b| !inside.contains(&b.id.0)) {
        for inst in &blk.insts {
            each_op_use(&inst.op, &mut |v| escapes |= defined.contains(&v.0));
        }
        each_term_use(&blk.term, &mut |v| escapes |= defined.contains(&v.0));
        if escapes {
            return None;
        }
    }
    Some((to, args))
}

#[cfg(test)]
mod tests {
    use super::*;
    use wukong_mir::{BinOp, Builder, CmpOp, MirType, Op, Program};
    use wukong_span::Interner;

    fn verify(f: &Function) {
        let errors = wukong_mir::verify::verify_function(f);
        assert!(errors.is_empty(), "{errors:?}");
    }

    fn run(f: &Function, it: &Interner) -> i64 {
        let mut p = Program::new();
        p.funcs.push(f.clone());
        wukong_interp::run(&p, f.name, it).unwrap()
    }

    /// The knobs of [`counted`]'s loop.
    #[derive(Clone, Copy, Default)]
    struct Shape {
        /// Bound the loop by the parameter `n` instead of the constant 8.
        symbolic: bool,
        /// Continue while `i <= bound` instead of `i < bound`.
        inclusive: bool,
        /// Store the induction variable to a stack slot in the body.
        store: bool,
        /// Pass the induction variable (not the bound) to the exit block.
        exit_passes_iv: bool,
        /// Use a value computed in the header after the loop.
        header_value_escapes: bool,
    }

    /// `fn f(n: i64) -> i64 { for i in 0..bound { body } return exit_param (+ header value) }`.
    fn counted(it: &mut Interner, s: Shape) -> Function {
        let mut b = Builder::new(it.intern("f"), MirType::I64);
        let n = b.add_param(MirType::I64);
        let slot = b.alloca(MirType::I64);
        let lo = b.build(MirType::I64, Op::ConstInt(0, MirType::I64));
        let eight = b.build(MirType::I64, Op::ConstInt(8, MirType::I64));
        let one = b.build(MirType::I64, Op::ConstInt(1, MirType::I64));
        let hi = if s.symbolic { n } else { eight };
        let header = b.new_block();
        let body = b.new_block();
        let exit = b.new_block();
        let i = b.block_param(header, MirType::I64);
        let out = b.block_param(exit, MirType::I64);
        b.br(header, vec![lo]);
        b.switch_to(header);
        let doubled = b.build(MirType::I64, Op::Bin(BinOp::Add, i, i));
        let pred = if s.inclusive { CmpOp::Sle } else { CmpOp::Slt };
        let c = b.build(MirType::I1, Op::Cmp(pred, i, hi));
        let exit_arg = if s.exit_passes_iv { i } else { hi };
        b.cond_br(c, body, vec![], exit, vec![exit_arg]);
        b.switch_to(body);
        if s.store {
            b.build_void(Op::Store {
                ptr: slot,
                value: i,
            });
        }
        let next = b.build(MirType::I64, Op::Bin(BinOp::Add, i, one));
        b.br(header, vec![next]);
        b.switch_to(exit);
        let r = if s.header_value_escapes {
            b.build(MirType::I64, Op::Bin(BinOp::Add, out, doubled))
        } else {
            out
        };
        b.ret(Some(r));
        b.finish()
    }

    #[test]
    fn an_emptied_counted_loop_is_deleted() {
        for symbolic in [false, true] {
            let mut it = Interner::new();
            let mut f = counted(
                &mut it,
                Shape {
                    symbolic,
                    ..Shape::default()
                },
            );
            verify(&f);
            let mut cache = CfgAnalyses::default();
            assert!(
                LoopDelete.run_function(&mut f, &mut cache),
                "symbolic={symbolic}"
            );
            verify(&f);
            assert_eq!(f.blocks.len(), 2, "only the entry and the exit remain");
            assert!(loop_info::analyze_function(&f).is_empty());
            assert!(!LoopDelete.run_function(&mut f, &mut cache));
        }
    }

    /// The deleted loop's exit value is still delivered: `main` returns the bound it passed out.
    #[test]
    fn deletion_preserves_the_exit_value() {
        let mut it = Interner::new();
        let mut b = Builder::new(it.intern("main"), MirType::I64);
        let lo = b.build(MirType::I64, Op::ConstInt(0, MirType::I64));
        let hi = b.build(MirType::I64, Op::ConstInt(4096, MirType::I64));
        let one = b.build(MirType::I64, Op::ConstInt(1, MirType::I64));
        let header = b.new_block();
        let body = b.new_block();
        let exit = b.new_block();
        let i = b.block_param(header, MirType::I64);
        let out = b.block_param(exit, MirType::I64);
        b.br(header, vec![lo]);
        b.switch_to(header);
        let c = b.build(MirType::I1, Op::Cmp(CmpOp::Slt, i, hi));
        b.cond_br(c, body, vec![], exit, vec![hi]);
        b.switch_to(body);
        let _dead = b.build(MirType::I64, Op::Bin(BinOp::Mul, i, i));
        let next = b.build(MirType::I64, Op::Bin(BinOp::Add, i, one));
        b.br(header, vec![next]);
        b.switch_to(exit);
        b.ret(Some(out));
        let mut f = b.finish();
        verify(&f);
        assert_eq!(run(&f, &it), 4096);
        assert!(LoopDelete.run_function(&mut f, &mut CfgAnalyses::default()));
        verify(&f);
        assert_eq!(run(&f, &it), 4096);
    }

    /// `while i < 127 { i = i + 2 }` over `i8`: `loop_info` counts 64 trips, but `126 + 2` wraps to
    /// `-128`, which is still `< 127`, so the loop never ends and must not be deleted. Bounded by
    /// `126` instead it really stops, after 63 trips, and goes.
    #[test]
    fn a_constant_count_whose_last_step_wraps_is_not_trusted() {
        for (end, deleted) in [(127, false), (126, true)] {
            let mut it = Interner::new();
            let mut b = Builder::new(it.intern("f"), MirType::I64);
            let lo = b.build(MirType::I8, Op::ConstInt(0, MirType::I8));
            let hi = b.build(MirType::I8, Op::ConstInt(end, MirType::I8));
            let two = b.build(MirType::I8, Op::ConstInt(2, MirType::I8));
            let seven = b.build(MirType::I64, Op::ConstInt(7, MirType::I64));
            let header = b.new_block();
            let body = b.new_block();
            let exit = b.new_block();
            let i = b.block_param(header, MirType::I8);
            b.br(header, vec![lo]);
            b.switch_to(header);
            let c = b.build(MirType::I1, Op::Cmp(CmpOp::Slt, i, hi));
            b.cond_br(c, body, vec![], exit, vec![]);
            b.switch_to(body);
            let next = b.build(MirType::I8, Op::Bin(BinOp::Add, i, two));
            b.br(header, vec![next]);
            b.switch_to(exit);
            b.ret(Some(seven));
            let mut f = b.finish();
            verify(&f);
            assert_eq!(
                LoopDelete.run_function(&mut f, &mut CfgAnalyses::default()),
                deleted,
                "end={end}"
            );
            verify(&f);
        }
    }

    /// A store in the body, an induction value carried out of the loop, a header value used after
    /// it, and a symbolic inclusive bound (`i <= n` never stops when `n` is the largest `i64`) each
    /// keep the loop.
    #[test]
    fn observable_or_possibly_endless_loops_are_kept() {
        let cases = [
            (
                "store",
                Shape {
                    store: true,
                    ..Shape::default()
                },
            ),
            (
                "exit passes iv",
                Shape {
                    exit_passes_iv: true,
                    ..Shape::default()
                },
            ),
            (
                "header value escapes",
                Shape {
                    header_value_escapes: true,
                    ..Shape::default()
                },
            ),
            (
                "symbolic inclusive",
                Shape {
                    symbolic: true,
                    inclusive: true,
                    ..Shape::default()
                },
            ),
        ];
        for (name, shape) in cases {
            let mut it = Interner::new();
            let mut f = counted(&mut it, shape);
            verify(&f);
            let before = f.blocks.len();
            assert!(
                !LoopDelete.run_function(&mut f, &mut CfgAnalyses::default()),
                "{name}"
            );
            assert_eq!(f.blocks.len(), before, "{name}");
        }
    }
}
