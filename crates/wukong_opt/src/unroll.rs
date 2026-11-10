//! Loop unrolling: complete for small known trip counts, partial otherwise.
//!
//! A counted loop pays a compare, a conditional branch and a back-edge branch for every single
//! iteration of its body, and its loads/stores are serialized behind that branch. Unrolling the body
//! `U` times amortizes the loop overhead over `U` iterations and hands the scheduler `U` independent
//! copies of the address arithmetic and the memory traffic to overlap.
//!
//! # What this pass will not do
//!
//! **It never reassociates.** Floating-point addition is not associative, so splitting a float
//! reduction across several accumulators — the textbook way to break the 4-cycle `addss` dependency
//! chain — computes a *different* number. This compiler's entire correctness story is that the
//! interpreter and Cranelift agree bit-for-bit and that `-O0` is observationally identical to
//! `-O1/-O2/-O3`; a reassociated reduction breaks both gates, loudly and correctly. So the unrolled
//! body here is the original body, `U` times, in the original order, computing the original values.
//! The win is loop overhead and instruction-level parallelism between the *independent* parts of
//! successive iterations, not a shorter dependency chain. On a loop whose critical path is a serial
//! float accumulate, that win is the small one — 4%, against 28% on the same loop in integer, see
//! "What it buys, measured" below.
//!
//! # Shape recognized
//!
//! Only the two-block counted loop the front end emits for `for i in a..b`:
//!
//! ```text
//!   P:            ...                       br H(init...)
//!   H(p0..pn):    c = cmp.slt p_k, bound     cond_br c, B, E
//!   B:            <body>                     br H(a0..an)   where a_k = add p_k, <const s>
//!   E:            ...
//! ```
//!
//! with `B` the only latch, `H` and `B` the whole loop, `P` an unconditional single preheader, `s`
//! positive, `bound` loop-invariant, and no call/alloca/vector-kernel in the body. Anything else is
//! left alone. That is deliberately narrow: it is exactly the loop nest kernels are written in, and
//! every extra shape is a new way to be wrong.
//!
//! # Transform
//!
//! ```text
//!   P:            k     = const (U-1)*s
//!                 lim0  = sub bound, k
//!                 ok    = cmp.slt lim0, bound      // false iff that subtraction wrapped
//!                 lim   = select ok, lim0, INT_MIN // a bound no iv can be below
//!                 ...                        br H(init...)
//!   H(p0..pn):    u = cmp.slt p_k, lim       cond_br u, M, R(p0..pn)
//!   M:            <body x U>                 br H(...)
//!   R(q0..qn):    c = cmp.slt q_k, bound     cond_br c, RB, E
//!   RB:           <body>                     br R(...)
//!   E:            ...
//! ```
//!
//! `H` now runs the body `U` times per test, and `R`/`RB` — a verbatim copy of the original loop —
//! finishes the last `trip mod U` iterations one at a time.
//!
//! The `select` is what makes the guard airtight rather than merely usually-right. `bound - (U-1)s`
//! can wrap when `bound` sits near the bottom of its range, and a wrapped limit would be a huge
//! value that admits the unrolled body when fewer than `U` iterations remain. `ok` detects exactly
//! that (subtracting a positive `k` must decrease `bound`), and the wrapped case falls back to a
//! sentinel that sends every iteration to the remainder loop. Both instructions are loop-invariant
//! and live in the preheader, so the steady-state cost is zero. Given `p_k < lim` with no wrap,
//! `p_k + j*s <= bound - 1` for every `j < U`, so none of the `U` induction-variable increments
//! inside `M` can wrap either.
//!
//! # Known trip counts
//!
//! When the induction variable's start and the bound are both constants, the exact trip count `T`
//! is computed in the induction variable's own type — signed for `slt`, unsigned for `ult` — and
//! accepted only when no increment wraps (otherwise the loop's real behavior is not the closed
//! form, and it is treated as unknown). Three things follow:
//!
//!  * **Complete unrolling.** A loop with `T <= 32` whose `T` copies total at most 128 instructions
//!    is replaced by its iterations, in order: the header keeps its own instructions as iteration
//!    0's test, then each body copy feeds the next copy's header, and a final header copy computes
//!    the exit values. The body block is orphaned and the loop is gone — no compare, no branch, and
//!    every copy's induction variable is a constant the cleanup folds into its addresses. This is
//!    the `for k in 0..4` register-tile loop, which the partial unroller used to skip as "mostly
//!    remainder".
//!  * **No remainder loop when `U` divides `T`.** The unrolled header then exits straight to `E`:
//!    after `T / U` passes the induction variable is exactly where the original loop stops, so the
//!    remainder would run zero times. That is the case of every power-of-two kernel extent.
//!  * **A straight-line remainder otherwise.** The last `T mod U` iterations are emitted as that
//!    many copies in one block instead of a loop, since how many there are is known.
//!
//! With `T` known and at least `2U`, `bound - (U-1)s` cannot wrap, so the guard folds to a constant
//! in the cleanup and the passes count exactly `T / U`.
//!
//! # Factor
//!
//! Bodies of up to 16 instructions unroll 4x — the measured configuration below. Bodies of 17 to 32
//! instructions, which the pass used to leave alone, unroll 2x: the loop overhead is a smaller share
//! there, so half the copies buys most of what is left of it at half the growth. That tier is not
//! yet measured on this machine; `WUKONG_NO_UNROLL=1` is the A/B switch.
//!
//! # SSA
//!
//! `E` is reached only through `R` after the transform, so every value the code after the loop reads
//! out of the old header — its block parameters and its instruction results — has to be re-pointed at
//! `R`'s copies. The blocks needing that rewrite are exactly `{X : H dominates X} \ {H, B}`: once
//! control is in `H` the only way to anything else is out through `E`, so every such `X` is
//! dominated by `E`, hence by `R` afterwards. Nothing defined in `B` can be live out (it does not
//! dominate `E`), so the body's own values need no repair.
//!
//! The pass does **not** compute that set. Every block outside `{H, B}` is swept instead, because a
//! block `H` does not dominate cannot mention a value defined in `H` — that is SSA's own dominance
//! rule — so visiting it finds nothing to rename and the result is identical. Building a dominator
//! tree per unrolled loop to learn the same thing cost a third of the optimizer.
//!
//! # Scheduling
//!
//! The pass runs **once**, after the main fixpoint, and only the functions it changed are then run
//! through a reduced cleanup pipeline (simplify, simplify-cfg, simplify-phis, cse, dce — see
//! [`crate::PassManager::unroll_cleanup`]) to fold the guard and the completely unrolled copies'
//! constants and collapse the now four-fold address arithmetic. Nothing here builds a dominator tree or a
//! reachability map, and a linear pre-filter rejects a function with no two-block loop before any
//! analysis is allocated, because this pass is on the compile-time critical path of every `-O2`
//! build and compile speed is one of this compiler's few genuine strengths.
//! `WUKONG_NO_UNROLL=1` turns it off (bisecting a miscompile, or A/B measurement).
//!
//! Autodiff runs after `optimize` in `wukong_driver`, and `wukong_autodiff::grad` refuses any
//! function with more than one basic block. Partial unrolling never changes which functions that
//! admits (the loop is still there). Complete unrolling can: once the cleanup merges the copies into
//! the preheader, a loss function whose only loop was a short constant-trip one is straight-line at
//! `-O2` and is accepted there, while `-O1` (no unrolling) still rejects it. That is a capability
//! gain, not a semantic change — the copies compute the original values in the original order, so
//! the forward value and its derivative are those of the loop.
//!
//! # What it buys, measured
//!
//! Release binary, `--backend=native -O2`, n = 2^20, best-of-8 interleaved `WUKONG_NO_UNROLL=1`
//! versus on, four hand-written kernels that dispatch no recognizer (confirmed with
//! `--emit=mir -O2 | grep wukong_`: only `now_ns`/`rt_alloc`/`rt_free`):
//!
//! | kernel (per iteration)                 | unrolled | round 1 | round 2 |
//! |----------------------------------------|----------|---------|---------|
//! | `s += (a[i]-b[i])*(a[i]-b[i])*c`  f32  | 4x       | 0.874   | 0.959   |
//! | `s += (a[i]-b[i])*(a[i]-b[i])`    i32  | 4x       | 0.719   | 0.702   |
//! | `o[i] = (a[i]-b[i])^2*c`          f32  | 4x       | 0.801   | 0.904   |
//! | `if a[i] > t { k += 1 }`          i64  | no       | 0.988   | 1.011   |
//!
//! The last row is the control: that loop has an `if` in it, so it is not the two-block shape and
//! **neither arm unrolls it** — the 1.2% it moves anyway is this instrument's noise floor, and
//! nothing smaller in the table would mean anything. So, taking the conservative end of each pair:
//! **integer reduction 28% faster, elementwise store loop 10%, float reduction 4%.** That ordering
//! is what a non-reassociating unroller should produce — the float reduction's critical path is the
//! serial `addss` chain, which this pass deliberately does not shorten, so all it can recover there
//! is the loop overhead.
//!
//! Two variants were tried and **refuted**, each by an interleaved two-binary A/B at n = 8 on the
//! same four kernels (against the then-current configuration, whose control band was 3.8%):
//!
//!  * **Flattening the induction variable** — emitting `iv + j*step` off the header parameter for
//!    each copy instead of letting copy `j` inherit copy `j-1`'s `iv + step`, so the copies' address
//!    arithmetic does not serialize. Exact (wrapping integer addition is associative), and the
//!    textbook thing to do. Measured 0.993 / 1.039 / 0.987 / 1.033 — every one inside the control
//!    band, and the kernel unrolling most helps came out nominally *worse* — while costing two extra
//!    instructions per copy. Not kept.
//!  * **Factor 8 for bodies under 12 instructions.** Measured 0.962 / 1.067 / 0.987 / 0.986: the
//!    integer reduction, the clearest winner at 4x, regressed 6.7%, and code size doubles. Not kept.
//!
//! # What it costs
//!
//! Unrolling is `-O2`+ only, so the default `-O0` compile is untouched. At `-O2`, measured
//! in-process over the 331-program `tests/run` corpus (`wukong_bench compile-time`, best-of-4
//! interleaved with `WUKONG_NO_UNROLL=1`), the whole unrolling stage — the scan, the transform and
//! the cleanup sweep — is **22% of optimizer time**, and optimizer time goes from 41.5 ms to
//! 52.7 ms, **+27%**. End to end that mostly disappears into everything else a compile does:
//! `wukong_bench compile-vs`, which times whole `--emit=obj -O2` processes, moved -4.3% / +1.8% /
//! +6.2% on its three kernels — inside its own run-to-run spread — and the gcc/g++/rustc speedup
//! ratios were unchanged at 7.3-11.9x.

use std::sync::OnceLock;

use wukong_mir::{
    BasicBlock, BinOp, BlockId, CmpOp, Function, Inst, MirType, Op, Program, Terminator, ValueId,
};

use crate::parallel::{self, Parallelism};
use crate::{cfg, map_op_uses, map_term_uses};

/// Bodies up to this many instructions unroll 4x. Anything larger is left alone: a big body already
/// amortizes the per-iteration loop overhead, so there is little to win, and every loop unrolled is
/// a function that has to be run through a cleanup sweep at four times its old size — which is what
/// unrolling costs at compile time. The measured wins are all on bodies of seven to ten
/// instructions, so the threshold sits just above that rather than wherever duplication is merely
/// affordable.
const BODY_FOR_4X: usize = 16;
/// Bodies up to this many instructions (and over [`BODY_FOR_4X`]) unroll 2x.
const BODY_FOR_2X: usize = 32;
/// A loop with a known trip count up to this is unrolled completely...
const FULL_UNROLL_MAX_TRIP: u64 = 32;
/// ...when its copies total at most this many instructions.
const FULL_UNROLL_BUDGET: usize = 128;
/// At most this many loops per function, so a pathological function cannot blow up compile time.
const MAX_LOOPS_PER_FN: usize = 16;
/// A function may not grow past this many instructions in total.
const MAX_FN_INSTS: usize = 6000;

/// `WUKONG_NO_UNROLL=1` disables the pass entirely (read once).
fn disabled() -> bool {
    static OFF: OnceLock<bool> = OnceLock::new();
    *OFF.get_or_init(|| std::env::var("WUKONG_NO_UNROLL").is_ok_and(|v| v != "0"))
}

/// Unroll every function in `program`. Returns the indices of the functions that changed, so the
/// caller can re-run the per-function pipeline on just those.
///
/// Each function is first offered to unroll-and-jam (`unroll_jam.rs`), which runs four rows
/// of a nest whose inner loop carries a serial chain side by side; the jammed inner loop is then an
/// ordinary counted loop this pass may still unroll. The two have separate switches
/// (`WUKONG_NO_UNROLL_JAM`, `WUKONG_NO_UNROLL`), and a function either changed is cleaned up.
///
/// One function at a time, on the calling thread; [`crate::optimize`] runs the same stage over
/// several functions at once (`unroll_program_with`), with the same result.
pub fn unroll_program(program: &mut Program) -> Vec<usize> {
    unroll_program_with(program, Parallelism::SEQUENTIAL)
}

/// [`unroll_program`] on up to `par`'s threads. Each function is unrolled from its own body alone,
/// so the functions are independent tasks ([`crate::parallel`]).
pub(crate) fn unroll_program_with(program: &mut Program, par: Parallelism) -> Vec<usize> {
    let all: Vec<usize> = (0..program.funcs.len()).collect();
    let changed = parallel::map(par, parallel::select_mut(&mut program.funcs, &all), |f| {
        let jammed = crate::unroll_jam::jam_function(f);
        let unrolled = !disabled() && unroll_function(f);
        jammed || unrolled
    });
    changed
        .into_iter()
        .enumerate()
        .filter_map(|(i, changed)| changed.then_some(i))
        .collect()
}

/// Unroll the counted loops of one function. Returns whether anything changed.
pub(crate) fn unroll_function(f: &mut Function) -> bool {
    // Cheap structural pre-filter, before any analysis is built. Most functions in a program have no
    // two-block loop at all, and paying a reachability DFS plus a dominator fixpoint per function to
    // discover that is most of what this pass costs at compile time.
    if !has_two_block_loop(f) {
        return false;
    }
    // No up-front `prune_unreachable` and no dominator analysis: nothing below needs either. An
    // unreachable block can only make a loop look *less* like the accepted shape (an extra predecessor edge into
    // the header), which costs an unroll but never correctness, and the live-out rewrite visits
    // every block rather than a dominance-derived subset.

    // `ValueId -> constant`, built once. The induction step is a constant whose defining instruction
    // is usually nowhere near the loop, and re-scanning the whole function for it per candidate was
    // quadratic in a function with many loops.
    let consts = const_index(f);

    // Headers we must not consider: the loops this pass itself produced. The remainder loop `R`/`RB`
    // is a verbatim copy of the original and would otherwise match again, forever.
    let mut banned = vec![false; f.blocks.len()];
    let mut changed = false;
    let mut orphaned = false;
    for _ in 0..MAX_LOOPS_PER_FN {
        if inst_count(f) >= MAX_FN_INSTS {
            break;
        }
        let Some((cand, plan)) = find_candidate(f, &banned, &consts) else {
            break;
        };
        // `apply` rewrites the header and the body in place and only ever *appends* blocks (the
        // remainder loop's `R` and `RB`, or one straight-line remainder block). Nothing already in
        // the function is renumbered, so the ban list stays valid across loops.
        let r_id = f.blocks.len();
        apply(f, &cand, plan);
        banned.resize(f.blocks.len(), false);
        banned[cand.header as usize] = true;
        match plan {
            Plan::Partial {
                tail: Tail::Loop, ..
            } => banned[r_id] = true,
            Plan::Full { .. } => orphaned = true,
            Plan::Partial { .. } => {}
        }
        changed = true;
    }
    // A completely unrolled loop leaves its body block unreachable; drop it (and renumber) only
    // now, after the last candidate scan has used the stable block ids.
    if orphaned {
        cfg::prune_unreachable(f);
    }
    changed
}

fn inst_count(f: &Function) -> usize {
    f.blocks.iter().map(|b| b.insts.len()).sum()
}

/// Is there any block that conditionally enters a block whose only exit branches straight back? That
/// is the skeleton of the shape `candidate_at` accepts, testable in one linear scan with no CFG
/// analysis and no allocation — and it is false for the overwhelming majority of functions, which is
/// what keeps this pass off the compile-time critical path.
fn has_two_block_loop(f: &Function) -> bool {
    f.blocks.iter().any(|hb| {
        let Terminator::CondBr { then_blk, .. } = &hb.term else {
            return false;
        };
        then_blk.0 != hb.id.0
            && matches!(
                &f.blocks[then_blk.0 as usize].term,
                Terminator::Br { target, .. } if target.0 == hb.id.0
            )
    })
}

/// `ValueId -> the integer constant it holds`, for every `ConstInt` in the function. Values that are
/// not integer constants (and every value created after this snapshot) map to `None`.
fn const_index(f: &Function) -> Vec<Option<i128>> {
    let mut idx = vec![None; f.value_types.len()];
    for b in &f.blocks {
        for inst in &b.insts {
            if let (Some(r), Op::ConstInt(k, _)) = (inst.result, &inst.op) {
                idx[r.0 as usize] = Some(*k);
            }
        }
    }
    idx
}

/// A recognized counted loop, with everything `apply` needs already validated.
struct Candidate {
    header: u32,
    body: u32,
    preheader: u32,
    /// Index into the header's block parameters of the induction variable.
    iv_idx: usize,
    /// The induction variable's per-iteration increment; always > 0.
    step: i128,
    /// The loop-invariant value the induction variable is tested against.
    bound: ValueId,
    /// The loop's exit test — `Slt` or `Ult`. Both admit a sentinel limit that no value compares
    /// below, which is what makes the wrap-safe guard cost nothing at run time.
    cmp: CmpOp,
    /// The integer type shared by the induction variable and the bound.
    ity: MirType,
}

/// May this operation appear in a *header* we are about to duplicate into the unrolled body? The
/// header's instructions run once per iteration in the original, and once per body copy here, so
/// they must be repeatable without observable effect. Loads are excluded conservatively: the header
/// of the shape we accept holds the compare and nothing else.
fn pure_header_op(op: &Op) -> bool {
    // No memory access, no write of any kind (`Op::effects`: a call of any kind, an atomic), and
    // no frame slot.
    !op.may_read_memory()
        && !op.may_write_memory()
        && !matches!(
            op,
            Op::Alloca {
                space: wukong_mir::AddrSpace::Local,
                ..
            }
        )
}

/// May this operation appear in a *body* we are about to duplicate? Loads and stores are fine — the
/// copies run in the original order, so the memory trace is unchanged. A call is not: duplicating it
/// is a code-size loss on a body that is already dominated by the call's own cost, and an alloca
/// would multiply the frame. A vector-kernel call already processes many elements per invocation.
fn unrollable_body_op(op: &Op) -> bool {
    // Loads and stores only: no other write (`Op::effects`: a call of any kind, an atomic), and no
    // frame slot.
    (matches!(op, Op::Store { .. }) || !op.may_write_memory())
        && !matches!(
            op,
            Op::Alloca {
                space: wukong_mir::AddrSpace::Local,
                ..
            }
        )
}

/// The largest value the type can hold when read as a *signed* integer. Used to reject an unroll
/// factor whose `(U-1)*step` offset would not fit.
fn signed_max(ty: &MirType) -> i128 {
    match ty {
        MirType::I8 => i8::MAX as i128,
        MirType::I16 => i16::MAX as i128,
        MirType::I32 => i32::MAX as i128,
        MirType::I64 => i64::MAX as i128,
        _ => 0,
    }
}

/// The limit sentinel for a wrapped `bound - k`: a value no induction variable can compare below
/// under this predicate, so the unrolled body is never entered and every iteration goes to the
/// remainder loop.
fn sentinel(cmp: CmpOp, ty: &MirType) -> i128 {
    match cmp {
        // Signed: nothing is `< INT_MIN`.
        CmpOp::Slt => match ty {
            MirType::I8 => i8::MIN as i128,
            MirType::I16 => i16::MIN as i128,
            MirType::I32 => i32::MIN as i128,
            _ => i64::MIN as i128,
        },
        // Unsigned: nothing is `< 0`.
        _ => 0,
    }
}

/// The bit width of an induction-variable type, `None` for anything that is not one.
fn int_width(ty: &MirType) -> Option<u32> {
    Some(match ty {
        MirType::I8 => 8,
        MirType::I16 => 16,
        MirType::I32 => 32,
        MirType::I64 => 64,
        _ => return None,
    })
}

/// `v`'s low `w` bits, sign-extended — the interpreter's canonical form of a `w`-bit integer.
fn sext(v: i128, w: u32) -> i128 {
    let shift = 128 - w;
    (v << shift) >> shift
}

/// `v`'s low `w` bits read as unsigned.
fn zext(v: i128, w: u32) -> i128 {
    (v as u128 & ((1u128 << w) - 1)) as i128
}

/// Exactly how many times the loop runs, when both the induction variable's initial value (the
/// preheader's branch argument) and the bound are compile-time constants. Computed in the
/// induction variable's own type under the loop's own predicate, and only when no increment wraps:
/// an increment past the type's maximum would wrap below the bound and keep the loop running, so
/// the closed form would be wrong. `None` means "not known", which is the interesting case — a
/// runtime bound is exactly the loop worth unrolling partially.
fn exact_trip_count(f: &Function, c: &Candidate, consts: &[Option<i128>]) -> Option<u64> {
    let Terminator::Br { args, .. } = &f.blocks[c.preheader as usize].term else {
        return None;
    };
    let init = (*consts.get(args.get(c.iv_idx)?.0 as usize)?)?;
    let bound = (*consts.get(c.bound.0 as usize)?)?;
    let w = int_width(&c.ity)?;
    let (init, bound, max) = match c.cmp {
        CmpOp::Slt => (sext(init, w), sext(bound, w), (1i128 << (w - 1)) - 1),
        CmpOp::Ult => (zext(init, w), zext(bound, w), (1i128 << w) - 1),
        _ => return None,
    };
    if bound <= init {
        return Some(0);
    }
    // `step` is positive and the predicate is strict, so this ceiling division is exact.
    let trip = (bound - init + c.step - 1) / c.step;
    // The increment that ends the loop must land at or past the bound without wrapping.
    if init.checked_add(trip.checked_mul(c.step)?)? > max {
        return None;
    }
    u64::try_from(trip).ok()
}

/// What to do with one recognized loop.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Plan {
    /// Replace the loop by its `trip` iterations, in order.
    Full { trip: u32 },
    /// Run the body `factor` times per test, then finish with `tail`.
    Partial { factor: u32, tail: Tail },
}

/// How a partially unrolled loop runs the iterations left after its last full pass.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Tail {
    /// The trip count is unknown: a verbatim copy of the original loop.
    Loop,
    /// The trip count is a known multiple of the factor: nothing is left, and the unrolled header
    /// exits straight to the original exit.
    Empty,
    /// The trip count is known and leaves this many iterations: that many straight-line copies.
    Straight(u32),
}

/// Find the lowest-numbered unrollable loop header, and what to do with it. Deterministic: blocks
/// are scanned in id order and nothing here iterates a hash container. No dominator analysis: the
/// only dominance fact the shape check needs is that the preheader dominates the header, and that
/// follows from the predecessor set alone (see `candidate_at`).
fn find_candidate(
    f: &Function,
    banned: &[bool],
    consts: &[Option<i128>],
) -> Option<(Candidate, Plan)> {
    let preds = cfg::predecessors(f);
    let size = inst_count(f);

    for h in 0..f.blocks.len() as u32 {
        if banned.get(h as usize).copied().unwrap_or(false) {
            continue;
        }
        let Some(c) = candidate_at(f, h, &preds, consts) else {
            continue;
        };
        let cost = f.blocks[c.header as usize].insts.len() + f.blocks[c.body as usize].insts.len();
        let trip = exact_trip_count(f, &c, consts);
        if let Some(t) = trip {
            // Few enough iterations, small enough copies: replace the loop by its iterations.
            if t <= FULL_UNROLL_MAX_TRIP && (t as usize) * cost <= FULL_UNROLL_BUDGET {
                if size + (t as usize + 1) * cost <= MAX_FN_INSTS {
                    return Some((c, Plan::Full { trip: t as u32 }));
                }
                continue;
            }
        }
        let mut factor: i128 = if cost <= BODY_FOR_4X {
            4
        } else if cost <= BODY_FOR_2X {
            2
        } else {
            continue;
        };
        // `(U-1)*step` becomes a constant of the induction variable's own type, and the
        // wrap-safety argument needs it to be a genuinely positive value in that type. A step so
        // large it does not fit halves the factor, then gives up.
        while factor > 1 && (factor - 1).saturating_mul(c.step) > signed_max(&c.ity) {
            factor /= 2;
        }
        if factor < 2 {
            continue;
        }
        let factor = factor as u32;
        let tail = match trip {
            // A known trip count smaller than 2*factor spends most or all of itself in the
            // remainder, so unrolling adds `factor` body copies that barely run — pure code growth,
            // and a cleanup sweep over a function that got bigger for nothing.
            Some(t) if t < 2 * u64::from(factor) => continue,
            Some(t) => match (t % u64::from(factor)) as u32 {
                0 => Tail::Empty,
                rest => Tail::Straight(rest),
            },
            None => Tail::Loop,
        };
        // Growth: (factor - 1) extra body copies, the four preheader instructions and the new
        // header compare, plus the remainder: a verbatim loop, or `rest` copies and an exit test.
        let remainder = match tail {
            Tail::Loop => cost,
            Tail::Empty => 0,
            Tail::Straight(rest) => cost * (rest as usize + 1),
        };
        if size + cost * factor as usize + 5 + remainder > MAX_FN_INSTS {
            continue;
        }
        return Some((c, Plan::Partial { factor, tail }));
    }
    None
}

fn candidate_at(
    f: &Function,
    h: u32,
    preds: &[Vec<u32>],
    consts: &[Option<i128>],
) -> Option<Candidate> {
    let hb = &f.blocks[h as usize];
    let Terminator::CondBr {
        cond,
        then_blk,
        then_args,
        else_blk,
        ..
    } = &hb.term
    else {
        return None;
    };
    // The body must be the `then` arm: that fixes the compare's polarity as "true means continue",
    // which is what the limit arithmetic below assumes.
    let body = then_blk.0;
    let exit = else_blk.0;
    if !then_args.is_empty() || body == exit || body == h || exit == h {
        return None;
    }

    // The body is the whole loop and its only latch.
    let bb = &f.blocks[body as usize];
    if !bb.params.is_empty() {
        return None;
    }
    match &bb.term {
        Terminator::Br { target, args } if target.0 == h => {
            if args.len() != hb.params.len() {
                return None;
            }
        }
        _ => return None,
    }
    if preds[body as usize] != [h] {
        return None;
    }

    // Exactly one entry from outside, by an unconditional branch we can append to. That single
    // outside predecessor necessarily *dominates* the header, with no dominator analysis needed:
    // control reaches the header either from it or from the body, and the body is only reachable
    // through the header, so the first arrival at the header is always through this block — hence
    // every path to the header contains it. That is what makes it legal to compute the loop-invariant
    // limit there.
    if preds[h as usize].len() != 2 || !preds[h as usize].contains(&body) {
        return None;
    }
    let preheader = *preds[h as usize].iter().find(|&&p| p != body)?;
    if preheader == exit || preheader == h {
        return None;
    }
    if !matches!(&f.blocks[preheader as usize].term, Terminator::Br { target, .. } if target.0 == h)
    {
        return None;
    }

    if !hb.insts.iter().all(|i| pure_header_op(&i.op)) {
        return None;
    }
    if !bb.insts.iter().all(|i| unrollable_body_op(&i.op)) {
        return None;
    }
    if bb.insts.is_empty() {
        return None;
    }

    // The exit test: `cmp.slt iv, bound` / `cmp.ult iv, bound`, on a header parameter.
    let cmp_inst = hb.insts.iter().find(|i| i.result == Some(*cond))?;
    let (cmp, iv, bound) = match cmp_inst.op {
        Op::Cmp(c @ (CmpOp::Slt | CmpOp::Ult), l, r) => (c, l, r),
        _ => return None,
    };
    let iv_idx = hb.params.iter().position(|&p| p == iv)?;
    let ity = f.value_type(iv).clone();
    if !ity.is_int() || ity == MirType::I1 || f.value_type(bound) != &ity {
        return None;
    }

    // `bound` must be loop-invariant. Because the verifier already guarantees its definition
    // dominates the header, "not defined in H or B" is enough to know it also dominates the
    // preheader — so the limit arithmetic can be appended there.
    if defined_in(hb, bound) || defined_in(bb, bound) {
        return None;
    }

    // The back-edge argument for the induction variable must be `iv + <positive constant>`.
    let Terminator::Br { args, .. } = &bb.term else {
        return None;
    };
    let next = args[iv_idx];
    let next_inst = bb
        .insts
        .iter()
        .chain(hb.insts.iter())
        .find(|i| i.result == Some(next))?;
    let step_val = match next_inst.op {
        Op::Bin(BinOp::Add, l, r) if l == iv => r,
        Op::Bin(BinOp::Add, l, r) if r == iv => l,
        _ => return None,
    };
    // Read the step in the induction variable's own width: a constant whose high bit is set is a
    // negative step, however its raw bits were recorded.
    let step = sext(
        *consts.get(step_val.0 as usize)?.as_ref()?,
        int_width(&ity)?,
    );
    if step <= 0 {
        return None;
    }

    Some(Candidate {
        header: h,
        body,
        preheader,
        iv_idx,
        step,
        bound,
        cmp,
        ity,
    })
}

fn defined_in(b: &BasicBlock, v: ValueId) -> bool {
    b.params.contains(&v) || b.insts.iter().any(|i| i.result == Some(v))
}

/// Allocate a fresh SSA value of the given type.
fn fresh(f: &mut Function, ty: MirType) -> ValueId {
    let v = ValueId(f.value_types.len() as u32);
    f.value_types.push(ty);
    v
}

/// Look `v` up in a renaming table indexed by old value id; values with no entry (defined outside
/// what is being cloned) map to themselves.
fn remap(v: ValueId, map: &[Option<ValueId>]) -> ValueId {
    map.get(v.0 as usize).copied().flatten().unwrap_or(v)
}

/// Clone `src` into `out`, giving every result a fresh value and recording it in `map`.
fn clone_insts(f: &mut Function, src: &[Inst], map: &mut [Option<ValueId>], out: &mut Vec<Inst>) {
    for inst in src {
        let mut op = inst.op.clone();
        map_op_uses(&mut op, |v| remap(v, map));
        let result = inst.result.map(|r| {
            let ty = f.value_types[r.0 as usize].clone();
            let nv = fresh(f, ty);
            // A copy of the loop body is the same source as the original.
            f.copy_value_loc(r, nv);
            map[r.0 as usize] = Some(nv);
            nv
        });
        out.push(Inst { result, op });
    }
}

/// The parts of a recognized loop the transforms copy, read once.
struct LoopParts {
    h_params: Vec<ValueId>,
    h_insts: Vec<Inst>,
    h_term: Terminator,
    b_insts: Vec<Inst>,
    /// The latch's arguments to the header, in parameter order.
    latch_args: Vec<ValueId>,
    /// The header's exit edge.
    exit: BlockId,
    exit_args: Vec<ValueId>,
}

impl LoopParts {
    fn of(f: &Function, c: &Candidate) -> LoopParts {
        let hb = &f.blocks[c.header as usize];
        let bb = &f.blocks[c.body as usize];
        let Terminator::Br {
            args: latch_args, ..
        } = &bb.term
        else {
            unreachable!("candidate_at checked the latch terminator")
        };
        let Terminator::CondBr {
            else_blk,
            else_args,
            ..
        } = &hb.term
        else {
            unreachable!("candidate_at checked the header terminator")
        };
        LoopParts {
            h_params: hb.params.clone(),
            h_insts: hb.insts.clone(),
            h_term: hb.term.clone(),
            b_insts: bb.insts.clone(),
            latch_args: latch_args.clone(),
            exit: *else_blk,
            exit_args: else_args.clone(),
        }
    }
}

fn apply(f: &mut Function, c: &Candidate, plan: Plan) {
    let parts = LoopParts::of(f, c);
    match plan {
        Plan::Full { trip } => apply_full(f, c, &parts, trip),
        Plan::Partial { factor, tail } => apply_partial(f, c, &parts, factor, tail),
    }
}

/// Emit `copies` iterations as straight-line code — each the header then the body, the latch's
/// arguments becoming the next copy's header parameters — followed by the header once more, as the
/// exit test runs it. `map` enters with the first copy's header parameters and leaves mapping every
/// header definition to its value when the loop exits.
fn straight_iterations(
    f: &mut Function,
    parts: &LoopParts,
    map: &mut [Option<ValueId>],
    copies: u32,
    out: &mut Vec<Inst>,
) {
    for _ in 0..copies {
        clone_insts(f, &parts.h_insts, map, out);
        clone_insts(f, &parts.b_insts, map, out);
        let next: Vec<ValueId> = parts.latch_args.iter().map(|a| remap(*a, map)).collect();
        for (p, v) in parts.h_params.iter().zip(next) {
            map[p.0 as usize] = Some(v);
        }
    }
    clone_insts(f, &parts.h_insts, map, out);
}

/// Re-point every use, outside the loop's own two blocks and below `blocks`, of a header definition
/// at its value when the loop exits (`exit_map`). See the module's SSA note for why sweeping every
/// other block is the same as sweeping the blocks the header dominates.
fn repoint_live_outs(
    f: &mut Function,
    c: &Candidate,
    parts: &LoopParts,
    exit_map: &[Option<ValueId>],
    blocks: u32,
) {
    let mut rename: Vec<Option<ValueId>> = vec![None; exit_map.len()];
    let header_defs = parts
        .h_params
        .iter()
        .copied()
        .chain(parts.h_insts.iter().filter_map(|i| i.result));
    for v in header_defs {
        rename[v.0 as usize] = exit_map[v.0 as usize];
    }
    for x in 0..blocks {
        if x == c.header || x == c.body {
            continue;
        }
        let blk = &mut f.blocks[x as usize];
        for inst in &mut blk.insts {
            map_op_uses(&mut inst.op, |v| remap(v, &rename));
        }
        map_term_uses(&mut blk.term, |v| remap(v, &rename));
    }
}

/// Replace the loop by its `trip` iterations. The header keeps its own instructions as iteration
/// 0's test and gains every later copy after them; it then branches straight to the exit. The body
/// block is left empty and unreachable for `unroll_function` to prune.
fn apply_full(f: &mut Function, c: &Candidate, parts: &LoopParts, trip: u32) {
    let nvals = f.value_types.len();
    let n_before = f.blocks.len() as u32;
    let mut map: Vec<Option<ValueId>> = vec![None; nvals];
    let mut out: Vec<Inst> = Vec::new();
    if trip > 0 {
        // Iteration 0's header is the original one, in place, so its body reads the header's own
        // values (an empty map is the identity).
        clone_insts(f, &parts.b_insts, &mut map, &mut out);
        let next: Vec<ValueId> = parts.latch_args.iter().map(|a| remap(*a, &map)).collect();
        for (p, v) in parts.h_params.iter().zip(next) {
            map[p.0 as usize] = Some(v);
        }
        straight_iterations(f, parts, &mut map, trip - 1, &mut out);
    }
    let exit_args = parts.exit_args.iter().map(|a| remap(*a, &map)).collect();
    let hb = &mut f.blocks[c.header as usize];
    hb.insts.extend(out);
    hb.term = Terminator::Br {
        target: parts.exit,
        args: exit_args,
    };
    let bb = &mut f.blocks[c.body as usize];
    bb.insts.clear();
    bb.term = Terminator::Unreachable;
    if trip > 0 {
        repoint_live_outs(f, c, parts, &map, n_before);
    }
}

fn apply_partial(f: &mut Function, c: &Candidate, parts: &LoopParts, factor: u32, tail: Tail) {
    let h = c.header;
    let nvals = f.value_types.len();
    let n_before = f.blocks.len() as u32;
    let iv = parts.h_params[c.iv_idx];

    // The original body block is rewritten in place into the unrolled body: it keeps the header's
    // `then` edge, so nothing is orphaned and the block numbering of everything already in the
    // function stays put (which is what lets `unroll_function` keep a stable "already unrolled" set
    // across loops).
    let m_id = c.body;
    let r_id = n_before;
    let rb_id = r_id + 1;

    // ---- preheader: the wrap-safe limit ----
    let k = (factor as i128 - 1) * c.step;
    let kc = fresh(f, c.ity.clone());
    let lim0 = fresh(f, c.ity.clone());
    let ok = fresh(f, MirType::I1);
    let sent = fresh(f, c.ity.clone());
    let lim = fresh(f, c.ity.clone());
    let ph = &mut f.blocks[c.preheader as usize].insts;
    ph.push(Inst {
        result: Some(kc),
        op: Op::ConstInt(k, c.ity.clone()),
    });
    ph.push(Inst {
        result: Some(lim0),
        op: Op::Bin(BinOp::Sub, c.bound, kc),
    });
    ph.push(Inst {
        result: Some(ok),
        op: Op::Cmp(c.cmp, lim0, c.bound),
    });
    ph.push(Inst {
        result: Some(sent),
        op: Op::ConstInt(sentinel(c.cmp, &c.ity), c.ity.clone()),
    });
    ph.push(Inst {
        result: Some(lim),
        op: Op::Select(ok, lim0, sent),
    });

    // ---- the remainder, and where the unrolled header exits to ----
    let mut appended: Vec<BasicBlock> = Vec::new();
    let mut exit_map: Option<Vec<Option<ValueId>>> = None;
    let (else_blk, else_args) = match tail {
        // A known multiple of the factor: the unrolled loop stops exactly where the original does.
        Tail::Empty => (parts.exit, parts.exit_args.clone()),
        // R/RB: a verbatim copy of the original loop.
        Tail::Loop => {
            let mut rmap: Vec<Option<ValueId>> = vec![None; nvals];
            let mut r_params = Vec::with_capacity(parts.h_params.len());
            for &p in &parts.h_params {
                let ty = f.value_types[p.0 as usize].clone();
                let nv = fresh(f, ty);
                f.copy_value_loc(p, nv);
                rmap[p.0 as usize] = Some(nv);
                r_params.push(nv);
            }
            let mut r_insts = Vec::with_capacity(parts.h_insts.len());
            clone_insts(f, &parts.h_insts, &mut rmap, &mut r_insts);
            // Snapshot before the body is cloned: only the *header's* definitions are live out of
            // the loop.
            exit_map = Some(rmap.clone());
            let mut r_term = parts.h_term.clone();
            map_term_uses(&mut r_term, |v| remap(v, &rmap));
            if let Terminator::CondBr { then_blk, .. } = &mut r_term {
                *then_blk = BlockId(rb_id);
            }
            let mut rb_insts = Vec::with_capacity(parts.b_insts.len());
            clone_insts(f, &parts.b_insts, &mut rmap, &mut rb_insts);
            let rb_term = Terminator::Br {
                target: BlockId(r_id),
                args: parts.latch_args.iter().map(|a| remap(*a, &rmap)).collect(),
            };
            appended.push(BasicBlock {
                id: BlockId(r_id),
                params: r_params,
                insts: r_insts,
                term: r_term,
            });
            appended.push(BasicBlock {
                id: BlockId(rb_id),
                params: Vec::new(),
                insts: rb_insts,
                term: rb_term,
            });
            (BlockId(r_id), parts.h_params.clone())
        }
        // R: the known leftover iterations in a straight line, then the exit.
        Tail::Straight(rest) => {
            let mut rmap: Vec<Option<ValueId>> = vec![None; nvals];
            let mut r_params = Vec::with_capacity(parts.h_params.len());
            for &p in &parts.h_params {
                let ty = f.value_types[p.0 as usize].clone();
                let nv = fresh(f, ty);
                f.copy_value_loc(p, nv);
                rmap[p.0 as usize] = Some(nv);
                r_params.push(nv);
            }
            let mut r_insts = Vec::new();
            straight_iterations(f, parts, &mut rmap, rest, &mut r_insts);
            let args = parts.exit_args.iter().map(|a| remap(*a, &rmap)).collect();
            appended.push(BasicBlock {
                id: BlockId(r_id),
                params: r_params,
                insts: r_insts,
                term: Terminator::Br {
                    target: parts.exit,
                    args,
                },
            });
            exit_map = Some(rmap);
            (BlockId(r_id), parts.h_params.clone())
        }
    };

    // ---- M: the body, `factor` times, in order ----
    let mut mmap: Vec<Option<ValueId>> = vec![None; nvals];
    let mut cur = parts.h_params.clone();
    let mut m_insts =
        Vec::with_capacity((parts.h_insts.len() + parts.b_insts.len()) * factor as usize);
    for _ in 0..factor {
        for (i, &p) in parts.h_params.iter().enumerate() {
            mmap[p.0 as usize] = Some(cur[i]);
        }
        clone_insts(f, &parts.h_insts, &mut mmap, &mut m_insts);
        clone_insts(f, &parts.b_insts, &mut mmap, &mut m_insts);
        cur = parts.latch_args.iter().map(|a| remap(*a, &mmap)).collect();
    }
    let m_term = Terminator::Br {
        target: BlockId(h),
        args: cur,
    };

    // ---- H: test `factor` iterations at a time ----
    let ucond = fresh(f, MirType::I1);
    let hb = &mut f.blocks[h as usize];
    hb.insts.push(Inst {
        result: Some(ucond),
        op: Op::Cmp(c.cmp, iv, lim),
    });
    hb.term = Terminator::CondBr {
        cond: ucond,
        then_blk: BlockId(m_id),
        then_args: Vec::new(),
        else_blk,
        else_args,
    };

    // ---- re-point the loop's live-out values at the remainder's exit values ----
    if let Some(map) = &exit_map {
        repoint_live_outs(f, c, parts, map, n_before);
    }

    let mb = &mut f.blocks[m_id as usize];
    mb.insts = m_insts;
    mb.term = m_term;
    f.blocks.extend(appended);
}

#[cfg(test)]
mod tests {
    use super::*;
    use wukong_mir::{Builder, MirType};
    use wukong_span::Interner;

    /// Build `fn f(n) { s = 0; for i in 0..n { s = s + i } ret s }` in the canonical two-block form,
    /// with `filler` extra instructions in the body.
    fn counted_loop_with(step: i128, filler: usize) -> Function {
        let mut i = Interner::new();
        let mut bld = Builder::new(i.intern("f"), MirType::I64);
        let n = bld.add_param(MirType::I64);
        let zero = bld.build(MirType::I64, Op::ConstInt(0, MirType::I64));
        let one = bld.build(MirType::I64, Op::ConstInt(1, MirType::I64));
        let stepc = bld.build(MirType::I64, Op::ConstInt(step, MirType::I64));
        let header = bld.new_block();
        let body = bld.new_block();
        let exit = bld.new_block();
        bld.br(header, vec![zero, zero]);

        bld.switch_to(header);
        let acc = bld.block_param(header, MirType::I64);
        let iv = bld.block_param(header, MirType::I64);
        let c = bld.build(MirType::I1, Op::Cmp(CmpOp::Slt, iv, n));
        bld.cond_br(c, body, vec![], exit, vec![]);

        bld.switch_to(body);
        let mut acc2 = bld.build(MirType::I64, Op::Bin(BinOp::Add, acc, iv));
        for _ in 0..filler {
            acc2 = bld.build(MirType::I64, Op::Bin(BinOp::Add, acc2, one));
        }
        let iv2 = bld.build(MirType::I64, Op::Bin(BinOp::Add, iv, stepc));
        bld.br(header, vec![acc2, iv2]);

        bld.switch_to(exit);
        bld.ret(Some(acc));
        bld.finish()
    }

    fn counted_loop(step: i128) -> Function {
        counted_loop_with(step, 0)
    }

    #[test]
    fn unrolls_a_canonical_counted_loop() {
        let mut f = counted_loop(1);
        let before = f.blocks.len();
        assert!(unroll_function(&mut f));
        // Two new blocks: the remainder header and the remainder body. The unrolled body reuses the
        // original body block.
        assert_eq!(f.blocks.len(), before + 2);
        assert!(wukong_mir::verify::verify_function(&f).is_empty());
    }

    #[test]
    fn unrolled_body_holds_factor_copies() {
        let mut f = counted_loop(1);
        assert!(unroll_function(&mut f));
        // Block 2 is the body, rewritten in place into the unrolled body. It had two instructions
        // and the header one, so the 4x copy holds four of each.
        let m = &f.blocks[2];
        assert_eq!(m.insts.len(), 4 * 3);
    }

    /// A body over the 4x threshold but within the 2x one is unrolled twice instead of skipped.
    #[test]
    fn a_medium_body_unrolls_twice_and_a_large_one_not_at_all() {
        // Header (1) + body (1 + filler + 1).
        let mut medium = counted_loop_with(1, 15);
        let cost = medium.blocks[1].insts.len() + medium.blocks[2].insts.len();
        assert!(cost > BODY_FOR_4X && cost <= BODY_FOR_2X, "{cost}");
        let before = medium.blocks.len();
        assert!(unroll_function(&mut medium));
        assert_eq!(medium.blocks[2].insts.len(), 2 * cost);
        assert_eq!(medium.blocks.len(), before + 2);
        assert!(wukong_mir::verify::verify_function(&medium).is_empty());

        let mut large = counted_loop_with(1, BODY_FOR_2X);
        assert!(!unroll_function(&mut large));
    }

    #[test]
    fn is_applied_only_once() {
        // A second call must not re-unroll: the remainder loop is a verbatim copy of the original
        // and would otherwise match forever.
        let mut f = counted_loop(1);
        assert!(unroll_function(&mut f));
        let after_first = f.blocks.len();
        assert!(!unroll_function(&mut f));
        assert_eq!(f.blocks.len(), after_first);
    }

    /// `s = 0; for (i = init; i < bound; i += step) s += i; return s` with constant `init` and
    /// `bound`, in the canonical two-block form, at integer type `ty`.
    fn const_loop(ty: MirType, init: i128, bound: i128, step: i128) -> Function {
        let mut i = Interner::new();
        let mut bld = Builder::new(i.intern("f"), ty.clone());
        let zero = bld.build(ty.clone(), Op::ConstInt(0, ty.clone()));
        let start = bld.build(ty.clone(), Op::ConstInt(init, ty.clone()));
        let stepc = bld.build(ty.clone(), Op::ConstInt(step, ty.clone()));
        let nc = bld.build(ty.clone(), Op::ConstInt(bound, ty.clone()));
        let header = bld.new_block();
        let body = bld.new_block();
        let exit = bld.new_block();
        bld.br(header, vec![zero, start]);

        bld.switch_to(header);
        let acc = bld.block_param(header, ty.clone());
        let iv = bld.block_param(header, ty.clone());
        let c = bld.build(MirType::I1, Op::Cmp(CmpOp::Slt, iv, nc));
        bld.cond_br(c, body, vec![], exit, vec![]);

        bld.switch_to(body);
        let acc2 = bld.build(ty.clone(), Op::Bin(BinOp::Add, acc, iv));
        let iv2 = bld.build(ty.clone(), Op::Bin(BinOp::Add, iv, stepc));
        bld.br(header, vec![acc2, iv2]);

        bld.switch_to(exit);
        bld.ret(Some(acc));
        bld.finish()
    }

    fn counted_loop_const_bound(n: i128) -> Function {
        const_loop(MirType::I64, 0, n, 1)
    }

    fn run(f: Function) -> i64 {
        let name = f.name;
        let mut program = wukong_mir::Program::default();
        program.funcs.push(f);
        assert!(wukong_mir::verify::verify_program(&program).is_empty());
        wukong_interp::run(&program, name, &Interner::new()).unwrap()
    }

    fn has_loop(f: &Function) -> bool {
        crate::inline::loop_blocks(f).iter().any(|&b| b)
    }

    /// A known trip count of at most 32 with a small body is replaced by its iterations: the loop
    /// disappears, including a loop that never runs and one whose step is not 1, and the value the
    /// exit block reads from the header is the one the last iteration left.
    #[test]
    fn a_short_constant_trip_loop_is_unrolled_completely() {
        for (init, bound, step, sum) in [
            (5, 3, 1, 0),
            (0, 1, 1, 0),
            (0, 7, 1, 21),
            (0, 8, 1, 28),
            (0, 20, 3, 63),
        ] {
            let mut f = const_loop(MirType::I64, init, bound, step);
            assert!(has_loop(&f));
            assert!(unroll_function(&mut f), "{init}..{bound} by {step}");
            assert!(!has_loop(&f), "{init}..{bound} by {step}");
            assert!(f
                .blocks
                .iter()
                .all(|b| !matches!(b.term, Terminator::CondBr { .. })));
            assert!(wukong_mir::verify::verify_function(&f).is_empty());
            assert_eq!(run(f), sum, "{init}..{bound} by {step}");
        }
    }

    /// 4096 iterations at 4x leave nothing over: the unrolled loop exits straight to the original
    /// exit — no remainder loop is emitted at all — and the cleanup folds the guard to constants.
    #[test]
    fn a_multiple_of_the_factor_needs_no_remainder_and_the_guard_folds() {
        let mut f = counted_loop_const_bound(4096);
        let blocks = f.blocks.len();
        assert!(unroll_function(&mut f));
        assert_eq!(f.blocks.len(), blocks, "no remainder blocks");
        assert!(
            matches!(f.blocks[1].term, Terminator::CondBr { else_blk, .. } if else_blk == BlockId(3))
        );
        assert!(f.blocks[0]
            .insts
            .iter()
            .any(|i| matches!(i.op, Op::Select(..))));
        let mut program = wukong_mir::Program::default();
        program.funcs.push(f);
        crate::PassManager::unroll_cleanup().run_subset(&mut program, &[0]);
        let f = program
            .funcs
            .pop()
            .expect("the function survives its cleanup");
        assert!(f.blocks[0]
            .insts
            .iter()
            .all(|i| matches!(i.op, Op::ConstInt(..))));
        assert!(f.blocks[0]
            .insts
            .iter()
            .any(|i| matches!(i.op, Op::ConstInt(4093, MirType::I64))));
        assert_eq!(run(f), 8_386_560);
    }

    /// 4099 iterations at 4x leave three: they run as three straight-line copies in one block, not
    /// as a remainder loop.
    #[test]
    fn a_known_leftover_runs_as_a_straight_line_remainder() {
        let mut f = counted_loop_const_bound(4099);
        let blocks = f.blocks.len();
        assert!(unroll_function(&mut f));
        assert_eq!(f.blocks.len(), blocks + 1, "one remainder block");
        let remainder = f.blocks.last().unwrap();
        assert!(matches!(remainder.term, Terminator::Br { target, .. } if target == BlockId(3)));
        // Three iterations of header (1) + body (2), then the exit test (1).
        assert_eq!(remainder.insts.len(), 3 * 3 + 1);
        assert!(wukong_mir::verify::verify_function(&f).is_empty());
        assert_eq!(run(f), 8_398_851);
    }

    /// `i8`, from 0 by 2 while `< 127`: the increment after 126 wraps to -128, which is still below
    /// the bound, so the loop never ends and its closed-form trip count (64) is a lie. It must not
    /// be unrolled completely or given a straight-line remainder; the wrap-guarded remainder loop
    /// is the only honest shape.
    #[test]
    fn a_wrapping_constant_loop_is_treated_as_unknown() {
        let mut f = const_loop(MirType::I8, 0, 127, 2);
        let blocks = f.blocks.len();
        assert!(unroll_function(&mut f));
        assert_eq!(f.blocks.len(), blocks + 2, "a remainder loop");
        assert!(has_loop(&f));
        assert!(wukong_mir::verify::verify_function(&f).is_empty());
        // The same bound reached exactly (0..126 by 2) is 63 iterations, all in range: unrolled.
        let mut exact = const_loop(MirType::I8, 0, 126, 2);
        assert!(unroll_function(&mut exact));
        assert!(wukong_mir::verify::verify_function(&exact).is_empty());
    }

    #[test]
    fn refuses_a_negative_step() {
        let mut f = counted_loop(-1);
        assert!(!unroll_function(&mut f));
    }

    #[test]
    fn refuses_a_body_with_a_call() {
        let mut i = Interner::new();
        let mut f = counted_loop(1);
        let call = Inst {
            result: None,
            op: Op::Call {
                func: i.intern("print"),
                args: vec![f.blocks[1].params[1]],
            },
        };
        f.blocks[2].insts.insert(0, call);
        assert!(!unroll_function(&mut f));
    }

    #[test]
    fn limit_is_wrap_guarded() {
        // The preheader must compute `bound - k`, test it against `bound`, and select a sentinel:
        // without that select, a `bound` near INT_MIN would wrap to a huge limit and admit the
        // unrolled body when fewer than `factor` iterations remain.
        let mut f = counted_loop(1);
        assert!(unroll_function(&mut f));
        let ph = &f.blocks[0];
        assert!(ph
            .insts
            .iter()
            .any(|i| matches!(i.op, Op::Bin(BinOp::Sub, ..))));
        assert!(ph.insts.iter().any(|i| matches!(i.op, Op::Select(..))));
        assert!(ph
            .insts
            .iter()
            .any(|i| matches!(i.op, Op::ConstInt(k, _) if k == i64::MIN as i128)));
    }

    #[test]
    fn live_out_header_values_are_repointed() {
        // `ret acc` reads the old header parameter. After the transform the exit is reached from the
        // remainder header, so it must read *that* block's parameter instead.
        let mut f = counted_loop(1);
        let old_acc = f.blocks[1].params[0];
        assert!(unroll_function(&mut f));
        // 0=entry, 1=header, 2=body, 3=exit (the new blocks are appended after it).
        let Terminator::Ret(Some(v)) = f.blocks[3].term else {
            panic!("exit block moved")
        };
        assert_ne!(v, old_acc);
        assert!(wukong_mir::verify::verify_function(&f).is_empty());
    }
}
