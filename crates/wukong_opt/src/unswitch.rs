//! Loop unswitching and induction-variable range splitting: take a per-iteration branch out of a
//! loop by running two copies of the loop, each with the branch decided.
//!
//! Both transforms clone an innermost loop and then *assume* a condition inside each copy —
//! replace every use of it in that copy by the constant it has there (folding the `and`/`or`/`not`
//! that exposes), after which `simplify-cfg` deletes the branch. A loop body that was a diamond with
//! one guarded store becomes one copy with the store unconditional and one without it: straight-line
//! bodies the vectorizer accepts, where it refuses the diamond (`vectorize`'s "the two arms do not
//! store the same number of times").
//!
//! # Induction-variable range splitting (GCC `-fsplit-loops`)
//!
//! A counted loop `for iv in start..end` (unit step, `iv < end` tested in the header, signed or
//! unsigned) whose body branches on `iv ⋈ inv` — `<`, `<=`, `>` or `>=` against a loop-invariant
//! `inv`, with the exit test's signedness — has a guard that is monotone along the iterations: it
//! holds on a prefix of them and fails on the rest, or the reverse. The preheader computes the first
//! iteration `s` where it changes, clamped into `[start, end]`:
//!
//! ```text
//!   iv <  inv, iv >= inv:  s = max(start, min(inv, end))
//!   iv <= inv, iv >  inv:  s = inv < end ? max(start, inv + 1) : end     // inv + 1 cannot wrap
//! ```
//!
//! and the loop runs as two loops back to back: a copy over `[start, s)` with the guard assumed to
//! be its prefix value, exiting into the original loop's header with its whole state, and the
//! original over `[s, end)` with the guard assumed to be the other value. Both halves visit the
//! iterations in the original increasing order and compute every value exactly as before, so the
//! result is bit-identical; the split point only decides which copy runs a given iteration, and
//! every iteration a copy runs is one where its assumption is true. The causal mask
//! `if j > i { s[i*n + j] = -inf }` becomes a loop that never stores and a loop that always does.
//!
//! The loop may also have side exits besides the header's test -- a bounds check's trap on a
//! runtime-sized slice, a `break`, an early `return` -- since each copy reaches such an exit at the
//! iteration the unsplit loop does, with the same state. The one hazard is a block after the loop
//! that reads a loop value directly: leaving from the copy bypasses the original's definition. A
//! read inside an exit block entered only from the loop (the trap block reading the failing index)
//! is rerouted through a new block parameter; any other read blocks the split.
//!
//! ## Derived guards
//!
//! The guard may also test a quantity *derived* from the induction variable, `ext(iv ± k) ⋈ n`
//! with `k` loop-invariant and `ext` an optional sign or zero extension into a wider compare: the
//! stencil edge `i + 1 < n`, the tile edge `jo + j < n` of a strip-mined loop (`jo` the enclosing
//! counter, split at `n - jo`), a counter widened to its bound's type, or a counter compared with
//! the other signedness than its exit test (`(i as u64) < m` on an `i64` counter). Such a guard is
//! monotone only while `iv ± k` does not wrap, so [`plan_derived`] proves that first, from integer
//! ranges ([`Ranges`]: constants, extensions, wrap-free arithmetic and the enclosing loops'
//! counters, which their header tests bound) over the values the loop's counter takes; where the
//! readings differ the values must also be non-negative, so an `i64` counter against a `usize`
//! bound splits only from a start proved non-negative. Anything unproved declines -- a guard that
//! wraps changes twice, and no split point is right for it. The split point itself
//! ([`derived_split_point`]) is formed without an intermediate that can overflow.
//!
//! # Unswitching (LLVM SimpleLoopUnswitch, non-trivial)
//!
//! A branch inside the loop on an `i1` defined outside it — after LICM, any condition that does not
//! change across iterations — is decided once: the preheader branches on it into the original loop
//! (assuming `true`) or into a copy (assuming `false`). Both copies leave through the original exit
//! edges, so this is restricted to loops whose values reach the code after them only through exit
//! edge arguments.
//!
//! # Calls
//!
//! A call in the loop -- a guarded diagnostic `if dbg { print(i) }`, or work done only for a prefix
//! of the iterations, `if i < m { log(i) }` -- keeps the dependence verdict "a call may touch any
//! memory" in every iteration, and with it the whole loop scalar. Cloning such a loop is exact (each
//! iteration still runs in exactly one copy, in order), but a call must never be *duplicated*: two
//! live copies would double the code of every dispatched kernel. So a loop holding a call is
//! rewritten only on a condition that decides, in one of the two copies, to skip an arm holding
//! every call ([`calls_confined_to_one_arm`]): there the arm is dead and simplify-cfg deletes it,
//! leaving a call-free copy the vectorizer can widen, and the calls live on in the other copy only.
//! A call outside such an arm -- unguarded, on both arms, or under an `and`/`or` the condition is
//! only an operand of -- still blocks the rewrite.
//!
//! # Limits
//!
//! Innermost loops with a preheader and no `alloca`, vector-kernel call, barrier or atomic, and no
//! call except under an arm the rewrite kills in one copy (see above), at most [`MAX_LOOP_INSTS`]
//! instructions, in a function under [`MAX_FUNCTION_INSTS`]; a loop fed by an already-vectorized
//! loop (its scalar epilogue) is left alone. Each rewrite removes its condition from both copies, so
//! every loop is rewritten at most once per guard it holds, and the function cap bounds the total.
//! At `-O2`, after LICM (which hoists invariant conditions and bounds into the preheader) and
//! before the vectorizer. Loop fusion runs ahead of both, and asks [`may_decide_a_branch`] before
//! fusing two loops the vectorizer cannot widen, so that it never welds a loop this pass could
//! still turn into widenable copies to work that would keep every copy scalar.

use wukong_mir::{
    BasicBlock, BinOp, BlockId, CastKind, CmpOp, Function, Inst, MirType, Op, Terminator, ValueId,
};

use crate::fxhash::{FxHashMap, FxHashSet};
use crate::loop_info::{self, IvStep, LoopForest, LoopId, NaturalLoop, Stages};
use crate::{each_op_use, each_term_use, map_op_uses, map_term_uses, CfgAnalyses, Pass};

/// The largest loop, in instructions, worth duplicating.
pub(crate) const MAX_LOOP_INSTS: usize = 128;
/// No rewrite may grow a function past this many instructions.
pub(crate) const MAX_FUNCTION_INSTS: usize = 4096;
/// Rewrites one invocation may perform before yielding to the rest of the fixpoint.
const MAX_REWRITES_PER_CALL: usize = 16;

/// Loop unswitching and induction-variable range splitting (see the module docs).
pub struct LoopUnswitch;

impl Pass for LoopUnswitch {
    fn name(&self) -> &'static str {
        "unswitch"
    }

    fn run_function(&self, f: &mut Function, cache: &mut CfgAnalyses) -> bool {
        let mut changed = false;
        for _ in 0..MAX_REWRITES_PER_CALL {
            if f.blocks.len() < 3
                || f.blocks
                    .iter()
                    .filter(|b| matches!(b.term, Terminator::CondBr { .. }))
                    .count()
                    < 2
                || !cache.all_reachable(f)
            {
                break;
            }
            let (forest, idom, preds) = {
                let (idom, preds) = cache.idoms_and_preds(f);
                (
                    loop_info::structure_with(f, idom, preds),
                    idom.to_vec(),
                    preds.to_vec(),
                )
            };
            let Some(plan) = pick(f, forest, &idom, &preds) else {
                break;
            };
            match plan {
                Plan::Split(p) => apply_split(f, p),
                Plan::Unswitch(p) => apply_unswitch(f, p),
            }
            cache.invalidate();
            changed = true;
        }
        changed
    }
}

/// A value the preheader can read: one defined outside the loop, or an integer constant that the
/// loop happens to materialize itself (rematerialized in the preheader).
#[derive(Clone, Copy)]
enum Avail {
    Outside(ValueId),
    Const(i128),
}

/// The guard's predicate, with the induction variable on the left.
#[derive(Clone, Copy, PartialEq, Eq)]
enum GuardKind {
    Lt,
    Le,
    Gt,
    Ge,
}

struct SplitPlan {
    blocks: Vec<BlockId>,
    header: BlockId,
    preheader: BlockId,
    /// The induction variable (a header parameter) and its parameter position.
    iv: ValueId,
    iv_index: usize,
    signed: bool,
    end: Avail,
    guard: ValueId,
    kind: GuardKind,
    /// What the guard compares against. For a [`DerivedGuard`] it has the compare's type.
    inv: Avail,
    /// `Some` when the guard tests a quantity derived from the induction variable rather than the
    /// induction variable itself (see [`DerivedGuard`]).
    derived: Option<DerivedGuard>,
    /// Loop values the side-exit blocks read directly, to be passed as edge arguments instead
    /// (see [`exit_routes`]).
    routes: Vec<(u32, Vec<ValueId>)>,
}

/// A guard `ext(iv ± k) ⋈ n` on a quantity derived from the induction variable: a stencil's
/// `i + 1 < n`, the tile edge `jo + j < n` with `jo` the enclosing loop's counter, a counter widened
/// to the bound's type (`sext(j) < n`), or the counter read with the other signedness than its exit
/// test (`(i as u64) < m`). [`plan_derived`] admits one only after proving that `iv ± k` never wraps
/// on an iteration the loop runs, which is what makes the guard monotone; [`derived_split_point`]
/// then computes where it changes without any intermediate that can overflow.
#[derive(Clone)]
struct DerivedGuard {
    /// The loop-invariant addend, and whether it is subtracted. `None` for the bare counter.
    k: Option<(Avail, bool)>,
    /// The widening into the compare's type, when the compare is wider than the counter.
    ext: Option<CastKind>,
    /// The compare's operand type (the counter's type when `ext` is `None`).
    wty: MirType,
    /// The compare's signedness, which may differ from the exit test's.
    signed: bool,
}

/// An enclosing loop's induction variable, with the facts that bound the values it takes: it starts
/// at `start`, steps by `step > 0`, and the enclosing loop's header leaves unless `value < bound`
/// (`signed` is that compare's signedness). See [`Ranges`].
#[derive(Clone, Copy)]
struct OuterIv {
    value: ValueId,
    start: ValueId,
    step: i64,
    bound: ValueId,
    signed: bool,
}

/// How deep [`Ranges::known`] follows operands.
const RANGE_DEPTH: u32 = 8;

/// The width of an integer type in bits.
fn int_bits(ty: &MirType) -> Option<u32> {
    Some(match ty {
        MirType::I1 => 1,
        MirType::I8 => 8,
        MirType::I16 => 16,
        MirType::I32 => 32,
        MirType::I64 => 64,
        _ => return None,
    })
}

/// The values a `bits`-wide integer can hold under its signed or unsigned reading.
fn type_bounds(bits: u32, signed: bool) -> (i128, i128) {
    if signed {
        (-(1i128 << (bits - 1)), (1i128 << (bits - 1)) - 1)
    } else {
        (0, (1i128 << bits) - 1)
    }
}

/// The integer a constant's low `bits` bits denote under the given reading.
fn interpret(n: i128, bits: u32, signed: bool) -> i128 {
    let m = n & ((1i128 << bits) - 1);
    if signed && m >= (1i128 << (bits - 1)) {
        m - (1i128 << bits)
    } else {
        m
    }
}

/// Integer value ranges, just strong enough to prove that `iv ± k` cannot wrap: constants,
/// sign/zero extension, `add`/`sub`/`mul`/`and` without wrap, and the induction variables of the
/// enclosing loops (whose header test bounds them). Every answer is an exact interval of integers
/// under the requested reading of the value's type, and a sound over-approximation of every value
/// the instruction can produce; an operation that could wrap answers "unknown", never a range.
struct Ranges<'a> {
    f: &'a Function,
    defs: &'a [Option<(usize, usize)>],
    outer: &'a [OuterIv],
}

impl Ranges<'_> {
    /// [`Ranges::known`], or the whole type when nothing better is known.
    fn or_full(&self, v: ValueId, signed: bool, depth: u32) -> Option<(i128, i128)> {
        let bits = int_bits(self.f.value_type(v))?;
        Some(
            self.known(v, signed, depth)
                .unwrap_or_else(|| type_bounds(bits, signed)),
        )
    }

    fn known(&self, v: ValueId, signed: bool, depth: u32) -> Option<(i128, i128)> {
        if depth > RANGE_DEPTH {
            return None;
        }
        let bits = int_bits(self.f.value_type(v))?;
        let (min, max) = type_bounds(bits, signed);
        let fits = |r: (i128, i128)| (r.0 <= r.1 && r.0 >= min && r.1 <= max).then_some(r);
        if let Some(o) = self.outer.iter().find(|o| o.value == v) {
            // Every value is `start` or `p + step` for a `p < bound` (the latch is reached only
            // through the header's staying arm), so nothing wraps while `bound - 1 + step` fits.
            let (lo, hi_start) = self.or_full(o.start, o.signed, depth + 1)?;
            let (_, hi_bound) = self.or_full(o.bound, o.signed, depth + 1)?;
            let top = hi_bound - 1 + i128::from(o.step);
            if top > type_bounds(bits, o.signed).1 {
                return None;
            }
            return fits((lo, hi_start.max(top)));
        }
        let (bi, ii) = (*self.defs.get(v.0 as usize)?)?;
        let d = depth + 1;
        match &self.f.blocks[bi].insts[ii].op {
            Op::ConstInt(n, _) => {
                let x = interpret(*n, bits, signed);
                Some((x, x))
            }
            // A sign extension keeps the signed value; read unsigned it is that value only when it
            // is not negative (`fits` checks `min = 0`).
            Op::Cast(CastKind::SExt, a, _) => fits(self.known(*a, true, d)?),
            // A zero extension keeps the unsigned value, which the wider type holds either way.
            Op::Cast(CastKind::ZExt, a, _) => fits(self.known(*a, false, d)?),
            Op::Bin(BinOp::Add, a, b) => {
                let (x, y) = (self.known(*a, signed, d)?, self.known(*b, signed, d)?);
                fits((x.0 + y.0, x.1 + y.1))
            }
            Op::Bin(BinOp::Sub, a, b) => {
                let (x, y) = (self.known(*a, signed, d)?, self.known(*b, signed, d)?);
                fits((x.0 - y.1, x.1 - y.0))
            }
            Op::Bin(BinOp::Mul, a, b) => {
                let (x, y) = (self.known(*a, signed, d)?, self.known(*b, signed, d)?);
                let c = [
                    x.0.checked_mul(y.0)?,
                    x.0.checked_mul(y.1)?,
                    x.1.checked_mul(y.0)?,
                    x.1.checked_mul(y.1)?,
                ];
                fits((*c.iter().min()?, *c.iter().max()?))
            }
            // `a & b` is at most either operand when that operand is not negative.
            Op::Bin(BinOp::And, a, b) => {
                let nonneg = |r: Option<(i128, i128)>| r.filter(|r| r.0 >= 0).map(|r| r.1);
                let caps = [
                    nonneg(self.known(*a, signed, d)),
                    nonneg(self.known(*b, signed, d)),
                ];
                fits((0, caps.iter().flatten().copied().min()?))
            }
            _ => None,
        }
    }
}

struct UnswitchPlan {
    blocks: Vec<BlockId>,
    header: BlockId,
    preheader: BlockId,
    cond: ValueId,
}

enum Plan {
    Split(SplitPlan),
    Unswitch(UnswitchPlan),
}

fn func_insts(f: &Function) -> usize {
    f.blocks.iter().map(|b| b.insts.len()).sum()
}

fn fresh(f: &mut Function, ty: MirType) -> ValueId {
    let v = ValueId(f.value_types.len() as u32);
    f.value_types.push(ty);
    v
}

/// `ValueId -> (block, instruction index)` of its defining instruction.
fn def_sites(f: &Function) -> Vec<Option<(usize, usize)>> {
    let mut d = vec![None; f.value_types.len()];
    for (bi, b) in f.blocks.iter().enumerate() {
        for (ii, inst) in b.insts.iter().enumerate() {
            if let Some(r) = inst.result {
                if let Some(slot) = d.get_mut(r.0 as usize) {
                    *slot = Some((bi, ii));
                }
            }
        }
    }
    d
}

fn op_of<'a>(f: &'a Function, defs: &[Option<(usize, usize)>], v: ValueId) -> Option<&'a Op> {
    let (b, i) = (*defs.get(v.0 as usize)?)?;
    Some(&f.blocks[b].insts[i].op)
}

/// Does this loop qualify for duplication at all (see the module's limits)? `Some(has_calls)` when
/// it does: a loop holding an [`Op::Call`] may still be cloned, but only by a rewrite that leaves
/// every call in one copy ([`calls_confined_to_one_arm`]).
fn clonable(f: &Function, forest: &LoopForest, l: &NaturalLoop) -> Option<bool> {
    if l.preheader.is_none() || !l.children.is_empty() {
        return None;
    }
    let mut n = 0;
    let mut has_calls = false;
    for b in &l.blocks {
        let blk = f.block(*b);
        if blk.params.iter().any(|p| f.value_type(*p).is_vector())
            || matches!(blk.term, Terminator::Switch { .. })
        {
            return None;
        }
        for inst in &blk.insts {
            n += 1;
            match inst.op {
                Op::Call { .. } => has_calls = true,
                Op::Alloca { .. }
                | Op::VecKernelCall { .. }
                | Op::Barrier
                | Op::AtomicAdd { .. } => return None,
                _ => {}
            }
            if inst.result.is_some_and(|r| f.value_type(r).is_vector()) {
                return None;
            }
        }
    }
    if n > MAX_LOOP_INSTS || func_insts(f) + n > MAX_FUNCTION_INSTS {
        return None;
    }
    // The scalar epilogue of a loop the vectorizer already widened is entered from that loop's
    // exit; it is the remainder, not a loop worth duplicating.
    let ph = l.preheader.expect("checked above");
    let epilogue = forest.loops.iter().any(|v| {
        v.id != l.id
            && v.exits.iter().any(|&(_, to)| to == ph)
            && v.blocks.iter().any(|b| {
                let blk = f.block(*b);
                blk.params.iter().any(|p| f.value_type(*p).is_vector())
                    || blk
                        .insts
                        .iter()
                        .any(|i| i.result.is_some_and(|r| f.value_type(r).is_vector()))
            })
    });
    (!epilogue).then_some(has_calls)
}

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

/// Will deciding `decided` leave every call of the loop in **one** copy? That holds when a body
/// block `bb` branches on exactly `decided` (not on an `and`/`or` it is an operand of, which may stay
/// undecided in one copy) into an arm `A` -- a loop block other than the latch, entered only from
/// `bb` -- that dominates every block holding a call. Each copy assumes `decided` constant, so in one
/// of them `bb` never enters `A`: `A` and everything it dominates, every call included, is dead
/// there, and simplify-cfg deletes it. The other copy keeps the calls, in iteration order, for
/// exactly the iterations it runs, so no call is duplicated, dropped or reordered.
fn calls_confined_to_one_arm(
    f: &Function,
    l: &NaturalLoop,
    idom: &[u32],
    preds: &[Vec<u32>],
    decided: ValueId,
) -> bool {
    let call_blocks: Vec<u32> = l
        .blocks
        .iter()
        .filter(|b| {
            f.block(**b)
                .insts
                .iter()
                .any(|i| matches!(i.op, Op::Call { .. }))
        })
        .map(|b| b.0)
        .collect();
    l.blocks.iter().any(|&bb| {
        if bb == l.header {
            return false;
        }
        let Terminator::CondBr {
            cond,
            then_blk,
            else_blk,
            ..
        } = &f.block(bb).term
        else {
            return false;
        };
        if *cond != decided || then_blk == else_blk {
            return false;
        }
        [*then_blk, *else_blk].into_iter().any(|arm| {
            l.blocks.contains(&arm)
                && !l.latches.contains(&arm)
                && preds
                    .get(arm.0 as usize)
                    .is_some_and(|p| p.as_slice() == [bb.0])
                && call_blocks.iter().all(|&c| dominates(arm.0, c, idom))
        })
    })
}

/// Might this pass, later in the sweep, split or unswitch the innermost loop `l` (counter `iv`) as
/// [`LoopFuse`](crate::LoopFuse) sees it, ahead of LICM? An over-approximation of what
/// [`plan_split`] and [`plan_unswitch`] accept once LICM has hoisted what it can: some body block
/// branches, with two distinct arms, on a condition -- or on one operand of an `and`/`or` condition
/// -- that is
///
/// * *invariant*: defined outside the loop, or computed in it only from such values by operations
///   LICM may hoist (a load included, since one that nothing in the loop writes is hoisted too) --
///   the flag `if dbg`, compared in the loop until LICM moves the compare out; or
/// * a comparison of the counter, or of a quantity derived from it (`ext(iv ± k)`, any cast or
///   any invariant addend), with an invariant -- the split guard `i < m`, `i + 1 < m`.
///
/// Fusion asks this before pairing a branching loop with another loop the vectorizer cannot widen:
/// split or unswitched alone, the branching loop may become copies the vectorizer widens, while
/// fused, every copy would carry the other loop's unwidenable work. It is deliberately wider than
/// the planners -- no size, range or exit-routing proof -- and misses only a condition whose
/// computation inside the loop runs deeper than its walk ([`RANGE_DEPTH`] operations); a `true`
/// only costs fusion a pair. (Every operation a loop fusion accepts is in its hoistable set or has
/// no result; vectors, calls and frame slots never reach it.)
pub(crate) fn may_decide_a_branch(f: &Function, l: &NaturalLoop, iv: ValueId) -> bool {
    let defs = def_sites(f);
    let mut defined: FxHashSet<u32> = FxHashSet::default();
    for b in &l.blocks {
        let blk = f.block(*b);
        defined.extend(blk.params.iter().map(|p| p.0));
        defined.extend(blk.insts.iter().filter_map(|i| i.result.map(|r| r.0)));
    }
    let cx = Hoistable {
        f,
        defs: &defs,
        defined: &defined,
    };
    let decidable = |c: ValueId| match op_of(f, &defs, c) {
        Some(Op::ConstInt(..)) => false,
        Some(Op::Cmp(_, x, y))
            if (cx.derived(iv, *x, RANGE_DEPTH) && cx.invariant(*y, RANGE_DEPTH))
                || (cx.invariant(*x, RANGE_DEPTH) && cx.derived(iv, *y, RANGE_DEPTH)) =>
        {
            true
        }
        _ => *f.value_type(c) == MirType::I1 && cx.invariant(c, RANGE_DEPTH),
    };
    l.blocks.iter().any(|b| {
        if *b == l.header {
            return false;
        }
        let Terminator::CondBr {
            cond,
            then_blk,
            else_blk,
            ..
        } = &f.block(*b).term
        else {
            return false;
        };
        then_blk != else_blk
            && branch_conditions(f, &defs, *cond)
                .into_iter()
                .any(decidable)
    })
}

/// What [`may_decide_a_branch`] asks of a loop's values.
struct Hoistable<'a> {
    f: &'a Function,
    defs: &'a [Option<(usize, usize)>],
    /// The values the loop defines: its blocks' parameters and instruction results.
    defined: &'a FxHashSet<u32>,
}

impl Hoistable<'_> {
    /// Is `v` defined outside the loop, or computed in it only from such values by scalar
    /// operations LICM may hoist? `depth` bounds the walk; running out answers `false`.
    fn invariant(&self, v: ValueId, depth: u32) -> bool {
        if !self.defined.contains(&v.0) {
            return true;
        }
        // A loop block's parameter has no defining instruction: it varies.
        let Some(op) = op_of(self.f, self.defs, v) else {
            return false;
        };
        if depth == 0
            || !matches!(
                op,
                Op::ConstInt(..)
                    | Op::ConstFloat(..)
                    | Op::Bin(..)
                    | Op::Cmp(..)
                    | Op::Neg(_)
                    | Op::Not(_)
                    | Op::Cast(..)
                    | Op::Select(..)
                    | Op::Load(..)
                    | Op::Gep { .. }
                    | Op::FuncAddr(_)
                    | Op::GlobalAddr(_)
                    | Op::Fma(..)
                    | Op::Sqrt(_)
                    | Op::Round(..)
                    | Op::IntUnary(..)
            )
        {
            return false;
        }
        let mut all = true;
        each_op_use(op, &mut |u| all = all && self.invariant(u, depth - 1));
        all
    }

    /// Is `v` the counter `iv`, or the counter cast or offset by an invariant?
    fn derived(&self, iv: ValueId, v: ValueId, depth: u32) -> bool {
        if v == iv {
            return true;
        }
        if depth == 0 {
            return false;
        }
        let d = depth - 1;
        match op_of(self.f, self.defs, v) {
            Some(Op::Cast(_, x, _)) => self.derived(iv, *x, d),
            Some(Op::Bin(BinOp::Add, a, b)) => {
                (self.derived(iv, *a, d) && self.invariant(*b, d))
                    || (self.invariant(*a, d) && self.derived(iv, *b, d))
            }
            Some(Op::Bin(BinOp::Sub, a, b)) => self.derived(iv, *a, d) && self.invariant(*b, d),
            _ => false,
        }
    }
}

/// The conditions a branch on `cond` decides: `cond` itself and, when it is an `i1` `and`/`or`,
/// its two operands (one level — the shape branch folding produces).
fn branch_conditions(f: &Function, defs: &[Option<(usize, usize)>], cond: ValueId) -> Vec<ValueId> {
    let mut out = vec![cond];
    if let Some(Op::Bin(BinOp::And | BinOp::Or, a, b)) = op_of(f, defs, cond) {
        if *f.value_type(cond) == MirType::I1 {
            out.push(*a);
            out.push(*b);
        }
    }
    out
}

/// The first rewrite any innermost loop admits, innermost first. `idom` and `preds` describe `f`.
fn pick(f: &Function, mut forest: LoopForest, idom: &[u32], preds: &[Vec<u32>]) -> Option<Plan> {
    if forest.is_empty() {
        return None;
    }
    let defs = def_sites(f);
    for id in forest.innermost_first() {
        let idx = id.0 as usize;
        let has_calls = {
            let l = forest.get(id);
            let Some(has_calls) = clonable(f, &forest, l) else {
                continue;
            };
            // Something to decide: a two-way branch in the body.
            let any_guard = l.blocks.iter().any(|b| {
                *b != l.header
                    && matches!(&f.block(*b).term,
                        Terminator::CondBr { then_blk, else_blk, .. } if then_blk != else_blk)
            });
            if !any_guard {
                continue;
            }
            has_calls
        };
        loop_info::analyze_one(f, &mut forest, idx, Stages::Ivs);
        let outer = outer_ivs(f, &defs, &mut forest, id);
        let l = forest.get(id);
        let in_loop: FxHashSet<u32> = l.blocks.iter().map(|b| b.0).collect();
        let mut defined: FxHashSet<u32> = FxHashSet::default();
        for b in &l.blocks {
            let blk = f.block(*b);
            defined.extend(blk.params.iter().map(|p| p.0));
            defined.extend(blk.insts.iter().filter_map(|i| i.result.map(|r| r.0)));
        }
        let ranges = Ranges {
            f,
            defs: &defs,
            outer: &outer,
        };
        // A call is cloned only by a rewrite that keeps it in one copy.
        let confined =
            |decided: ValueId| !has_calls || calls_confined_to_one_arm(f, l, idom, preds, decided);
        if let Some(p) = plan_split(f, l, &defs, &in_loop, &defined, &ranges) {
            if confined(p.guard) {
                return Some(Plan::Split(p));
            }
        }
        if let Some(p) = plan_unswitch(f, l, &defs, &in_loop, &defined) {
            if confined(p.cond) {
                return Some(Plan::Unswitch(p));
            }
        }
    }
    None
}

/// The induction variables of the loops enclosing `id` that [`Ranges`] can bound: a positive
/// constant step, a known start, and a header that leaves on the false arm of `iv < bound`.
fn outer_ivs(
    f: &Function,
    defs: &[Option<(usize, usize)>],
    forest: &mut LoopForest,
    id: LoopId,
) -> Vec<OuterIv> {
    let mut out = Vec::new();
    let mut cur = forest.get(id).parent;
    while let Some(p) = cur {
        loop_info::analyze_one(f, forest, p.0 as usize, Stages::Ivs);
        let l = forest.get(p);
        cur = l.parent;
        let Terminator::CondBr {
            cond,
            then_blk,
            else_blk,
            ..
        } = &f.block(l.header).term
        else {
            continue;
        };
        if !l.blocks.contains(then_blk) || l.blocks.contains(else_blk) {
            continue;
        }
        let Some(Op::Cmp(pred @ (CmpOp::Slt | CmpOp::Ult), a, bound)) = op_of(f, defs, *cond)
        else {
            continue;
        };
        for iv in &l.ivs {
            if iv.value != *a {
                continue;
            }
            let (Some(start), IvStep::Const(step)) = (iv.start, iv.step) else {
                continue;
            };
            if step > 0 {
                out.push(OuterIv {
                    value: iv.value,
                    start,
                    step,
                    bound: *bound,
                    signed: *pred == CmpOp::Slt,
                });
            }
        }
    }
    out
}

/// `v` as something the preheader can read (see [`Avail`]).
fn avail(
    f: &Function,
    defs: &[Option<(usize, usize)>],
    defined: &FxHashSet<u32>,
    v: ValueId,
) -> Option<Avail> {
    if !defined.contains(&v.0) {
        return Some(Avail::Outside(v));
    }
    match op_of(f, defs, v)? {
        Op::ConstInt(n, _) => Some(Avail::Const(*n)),
        _ => None,
    }
}

/// `a ⋈ b` with the operands exchanged.
fn swapped(p: CmpOp) -> Option<CmpOp> {
    use CmpOp::*;
    Some(match p {
        Slt => Sgt,
        Sle => Sge,
        Sgt => Slt,
        Sge => Sle,
        Ult => Ugt,
        Ule => Uge,
        Ugt => Ult,
        Uge => Ule,
        _ => return None,
    })
}

fn plan_split(
    f: &Function,
    l: &NaturalLoop,
    defs: &[Option<(usize, usize)>],
    in_loop: &FxHashSet<u32>,
    defined: &FxHashSet<u32>,
    ranges: &Ranges<'_>,
) -> Option<SplitPlan> {
    let iv = l.primary()?;
    if iv.step != IvStep::Const(1) || !iv.ty.is_int() || iv.ty == MirType::I1 {
        return None;
    }
    let preheader = l.preheader?;
    // The header's test `iv < end` exits on its false arm (loop-canon's polarity). Any other exit
    // is a side exit -- a bounds check's trap, a `break`, an early `return` -- taken by the copy
    // or the original at exactly the iteration the unsplit loop takes it, with the same state.
    let mut header_exits = l.exits.iter().filter(|e| e.0 == l.header);
    let (Some(&(_, header_exit)), None) = (header_exits.next(), header_exits.next()) else {
        return None;
    };
    let Terminator::CondBr {
        cond,
        then_blk,
        else_blk,
        ..
    } = &f.block(l.header).term
    else {
        return None;
    };
    if *else_blk != header_exit || !in_loop.contains(&then_blk.0) {
        return None;
    }
    // A side exit from the copy bypasses the original loop's definitions, so code after the loop
    // may then read a loop value only through exit-edge arguments (the original header always
    // runs last before the header exit, so that exit alone needs no such care). A bounds check's
    // trap block reads the failing index directly; that read becomes an edge argument.
    let routes = if l.exits.len() > 1 {
        exit_routes(f, in_loop, defined)?
    } else {
        Vec::new()
    };
    let Some(Op::Cmp(exit_pred, lhs, end)) = op_of(f, defs, *cond) else {
        return None;
    };
    if *lhs != iv.value {
        return None;
    }
    let signed = match exit_pred {
        CmpOp::Slt => true,
        CmpOp::Ult => false,
        _ => return None,
    };
    let end = avail(f, defs, defined, *end)?;
    let Terminator::Br { args, .. } = &f.block(preheader).term else {
        return None;
    };
    let start = *args.get(iv.param_index)?;
    for b in &l.blocks {
        if *b == l.header {
            continue;
        }
        let Terminator::CondBr {
            cond: g,
            then_blk,
            else_blk,
            ..
        } = &f.block(*b).term
        else {
            continue;
        };
        if then_blk == else_blk {
            continue;
        }
        // The guard is assumed only in body blocks (see `apply_split`), so an `and`/`or` the header
        // computes from it would never fold, and the branch would be split again.
        let g_in_header = defs
            .get(g.0 as usize)
            .copied()
            .flatten()
            .is_some_and(|(bi, _)| bi == l.header.0 as usize);
        for v in branch_conditions(f, defs, *g) {
            // The guard varies with the induction variable, so the loop computes it.
            if !defined.contains(&v.0) || (v != *g && g_in_header) {
                continue;
            }
            let found = plain_guard(f, defs, defined, iv.value, signed, v)
                .map(|(kind, inv)| (kind, inv, None))
                .or_else(|| {
                    let cx = DerivedCx {
                        f,
                        defs,
                        defined,
                        ranges,
                        iv: iv.value,
                        ty: &iv.ty,
                        exit_signed: signed,
                        start,
                        end,
                    };
                    plan_derived(&cx, v).map(|(kind, inv, d)| (kind, inv, Some(d)))
                });
            let Some((kind, inv, derived)) = found else {
                continue;
            };
            return Some(SplitPlan {
                blocks: l.blocks.clone(),
                header: l.header,
                preheader,
                iv: iv.value,
                iv_index: iv.param_index,
                signed,
                end,
                guard: v,
                kind,
                inv,
                derived,
                routes,
            });
        }
    }
    None
}

/// `iv ⋈ inv` with the exit test's signedness: the guard on the induction variable itself.
fn plain_guard(
    f: &Function,
    defs: &[Option<(usize, usize)>],
    defined: &FxHashSet<u32>,
    iv: ValueId,
    signed: bool,
    v: ValueId,
) -> Option<(GuardKind, Avail)> {
    let Some(Op::Cmp(p, x, y)) = op_of(f, defs, v) else {
        return None;
    };
    let (p, other) = if *x == iv && *y != iv {
        (*p, *y)
    } else if *y == iv && *x != iv {
        (swapped(*p)?, *x)
    } else {
        return None;
    };
    let kind = match (signed, p) {
        (true, CmpOp::Slt) | (false, CmpOp::Ult) => GuardKind::Lt,
        (true, CmpOp::Sle) | (false, CmpOp::Ule) => GuardKind::Le,
        (true, CmpOp::Sgt) | (false, CmpOp::Ugt) => GuardKind::Gt,
        (true, CmpOp::Sge) | (false, CmpOp::Uge) => GuardKind::Ge,
        _ => return None,
    };
    Some((kind, avail(f, defs, defined, other)?))
}

/// What [`plan_derived`] needs to know about the loop being split.
struct DerivedCx<'a> {
    f: &'a Function,
    defs: &'a [Option<(usize, usize)>],
    defined: &'a FxHashSet<u32>,
    ranges: &'a Ranges<'a>,
    iv: ValueId,
    ty: &'a MirType,
    exit_signed: bool,
    /// The induction variable's start (the preheader's argument) and the exit test's bound.
    start: ValueId,
    end: Avail,
}

/// A guard `ext(iv ± k) ⋈ n` (see [`DerivedGuard`]), when it is monotone along the iterations.
///
/// Iteration `t` of the loop runs with `iv = start + t`, exactly (the header admits it only below
/// `end`, so the increment never wraps). The guard is monotone when `x = iv ± k` never wraps either,
/// in the reading the comparison sees it with: then `x` also grows by exactly one per iteration, and
/// a sign extension (signed reading) or zero extension (unsigned reading) carries that exact value
/// into the compare's type. So the proof obligation is one interval check: over the values `iv`
/// takes while the body runs (`[start, end - 1]` in the exit test's reading) and the values `k` can
/// have, `iv ± k` stays inside the type. Where the two readings differ -- a signed counter compared
/// unsigned, or the reverse; a sign-extended counter compared unsigned -- the values must also be
/// non-negative, where both readings agree. An `i64` counter tested against a `usize` bound
/// (`(i as u64) < m`) therefore splits only when its start is proved non-negative, and a guard
/// whose range is not proved declines: wrap would make it non-monotone, and no split point is
/// right for a guard that changes twice.
fn plan_derived(cx: &DerivedCx<'_>, v: ValueId) -> Option<(GuardKind, Avail, DerivedGuard)> {
    let (f, defs, defined) = (cx.f, cx.defs, cx.defined);
    let Some(Op::Cmp(p, a, b)) = op_of(f, defs, v) else {
        return None;
    };
    // The derived side on the left, the invariant on the right.
    let (p, d, n) = match (avail(f, defs, defined, *a), avail(f, defs, defined, *b)) {
        (None, Some(n)) => (*p, *a, n),
        (Some(n), None) => (swapped(*p)?, *b, n),
        _ => return None,
    };
    let (gsigned, kind) = match p {
        CmpOp::Slt => (true, GuardKind::Lt),
        CmpOp::Sle => (true, GuardKind::Le),
        CmpOp::Sgt => (true, GuardKind::Gt),
        CmpOp::Sge => (true, GuardKind::Ge),
        CmpOp::Ult => (false, GuardKind::Lt),
        CmpOp::Ule => (false, GuardKind::Le),
        CmpOp::Ugt => (false, GuardKind::Gt),
        CmpOp::Uge => (false, GuardKind::Ge),
        _ => return None,
    };
    let wty = f.value_type(d).clone();
    let (x, ext) = match op_of(f, defs, d) {
        Some(Op::Cast(k @ (CastKind::SExt | CastKind::ZExt), x, _)) => (*x, Some(*k)),
        _ => (d, None),
    };
    if f.value_type(x) != cx.ty {
        return None;
    }
    let (tb, wb) = (int_bits(cx.ty)?, int_bits(&wty)?);
    if ext.is_some() && wb <= tb {
        return None;
    }
    let k = if x == cx.iv {
        None
    } else {
        let addend = |k: ValueId| {
            if k == cx.iv {
                None
            } else {
                avail(f, defs, defined, k)
            }
        };
        match op_of(f, defs, x)? {
            Op::Bin(BinOp::Add, a, b) if *a == cx.iv => Some((addend(*b)?, false)),
            Op::Bin(BinOp::Add, a, b) if *b == cx.iv => Some((addend(*a)?, false)),
            Op::Bin(BinOp::Sub, a, b) if *a == cx.iv => Some((addend(*b)?, true)),
            _ => return None,
        }
    };
    // The plain guard on the counter itself is `plain_guard`'s.
    if ext.is_none() && k.is_none() && gsigned == cx.exit_signed {
        return None;
    }

    // The reading `x` is compared in: the extension's, or the compare's own.
    let sx = match ext {
        Some(CastKind::SExt) => true,
        Some(_) => false,
        None => gsigned,
    };
    let lo_iv = cx.ranges.or_full(cx.start, cx.exit_signed, 0)?.0;
    let hi_end = match cx.end {
        Avail::Const(c) => interpret(c, tb, cx.exit_signed),
        Avail::Outside(e) => cx.ranges.or_full(e, cx.exit_signed, 0)?.1,
    };
    let hi_iv = hi_end - 1;
    if hi_iv < lo_iv {
        return None; // the body never runs: nothing to gain
    }
    let smax = type_bounds(tb, true).1;
    if sx != cx.exit_signed && (lo_iv < 0 || hi_iv > smax) {
        return None;
    }
    let (xlo, xhi) = match k {
        None => (lo_iv, hi_iv),
        Some((k, sub)) => {
            let (klo, khi) = match k {
                Avail::Const(c) => {
                    let c = interpret(c, tb, sx);
                    (c, c)
                }
                Avail::Outside(kv) => cx.ranges.or_full(kv, sx, 0)?,
            };
            if sub {
                (lo_iv - khi, hi_iv - klo)
            } else {
                (lo_iv + klo, hi_iv + khi)
            }
        }
    };
    let (min, max) = type_bounds(tb, sx);
    if xlo < min || xhi > max {
        return None; // `iv ± k` may wrap: the guard need not be monotone
    }
    // A sign-extended value compared unsigned is monotone only while it is not negative.
    if ext == Some(CastKind::SExt) && !gsigned && xlo < 0 {
        return None;
    }
    Some((
        kind,
        n,
        DerivedGuard {
            k,
            ext,
            wty,
            signed: gsigned,
        },
    ))
}

fn plan_unswitch(
    f: &Function,
    l: &NaturalLoop,
    defs: &[Option<(usize, usize)>],
    in_loop: &FxHashSet<u32>,
    defined: &FxHashSet<u32>,
) -> Option<UnswitchPlan> {
    let preheader = l.preheader?;
    let mut cond = None;
    'search: for b in &l.blocks {
        if *b == l.header {
            continue;
        }
        let Terminator::CondBr {
            cond: c,
            then_blk,
            else_blk,
            ..
        } = &f.block(*b).term
        else {
            continue;
        };
        if then_blk == else_blk {
            continue;
        }
        for v in branch_conditions(f, defs, *c) {
            if defined.contains(&v.0)
                || *f.value_type(v) != MirType::I1
                || matches!(op_of(f, defs, v), Some(Op::ConstInt(..)))
            {
                continue;
            }
            cond = Some(v);
            break 'search;
        }
    }
    let cond = cond?;
    // Both copies leave through the original exit edges, so nothing after the loop may read a
    // loop value except through those edges' arguments.
    if reads_loop_values_after(f, in_loop, defined) {
        return None;
    }
    Some(UnswitchPlan {
        blocks: l.blocks.clone(),
        header: l.header,
        preheader,
        cond,
    })
}

/// Where code outside the loop reads loop values directly, other than through exit-edge arguments:
/// `None` unless every such read sits in an exit block whose single incoming edge leaves the loop
/// (a bounds check's trap block reading the failing index, say), in which case the values each
/// such block reads, in first-use order. Such a read can be rerouted through a new parameter of
/// that block ([`route_exit_values`]): the value dominates the block, so it dominates the end of its
/// only predecessor.
fn exit_routes(
    f: &Function,
    in_loop: &FxHashSet<u32>,
    defined: &FxHashSet<u32>,
) -> Option<Vec<(u32, Vec<ValueId>)>> {
    let n = f.blocks.len();
    let mut edges_in = vec![0usize; n];
    let mut from_loop = vec![false; n];
    for (bi, b) in f.blocks.iter().enumerate() {
        b.term.for_each_edge(|t, _| {
            if let Some(c) = edges_in.get_mut(t.0 as usize) {
                *c += 1;
            }
            if in_loop.contains(&(bi as u32)) {
                if let Some(fl) = from_loop.get_mut(t.0 as usize) {
                    *fl = true;
                }
            }
        });
    }
    let mut routes = Vec::new();
    for (bi, b) in f.blocks.iter().enumerate() {
        if in_loop.contains(&(bi as u32)) {
            continue;
        }
        let mut used: Vec<ValueId> = Vec::new();
        let mut note = |v: ValueId| {
            if defined.contains(&v.0) && !used.contains(&v) {
                used.push(v);
            }
        };
        for inst in &b.insts {
            each_op_use(&inst.op, &mut note);
        }
        each_term_use(&b.term, &mut note);
        if used.is_empty() {
            continue;
        }
        if bi == 0 || edges_in[bi] != 1 || !from_loop[bi] {
            return None;
        }
        routes.push((bi as u32, used));
    }
    Some(routes)
}

/// Give each exit block of `routes` one new parameter per loop value it reads, passed by its only
/// incoming edge, and read the parameter instead.
fn route_exit_values(f: &mut Function, routes: &[(u32, Vec<ValueId>)]) {
    for (blk, vals) in routes {
        let bi = *blk as usize;
        let mut subst: FxHashMap<u32, ValueId> = FxHashMap::default();
        for v in vals {
            let ty = f.value_type(*v).clone();
            let p = fresh(f, ty);
            f.blocks[bi].params.push(p);
            subst.insert(v.0, p);
        }
        let sub = |u: ValueId| subst.get(&u.0).copied().unwrap_or(u);
        for inst in &mut f.blocks[bi].insts {
            map_op_uses(&mut inst.op, sub);
        }
        map_term_uses(&mut f.blocks[bi].term, sub);
        for b in &mut f.blocks {
            b.term.for_each_edge_mut(|t, args| {
                if t.0 == *blk {
                    args.extend(vals.iter().copied());
                }
            });
        }
    }
}

/// Does any block outside the loop read a value the loop defines, other than through the
/// arguments of an exit edge?
fn reads_loop_values_after(
    f: &Function,
    in_loop: &FxHashSet<u32>,
    defined: &FxHashSet<u32>,
) -> bool {
    f.blocks.iter().enumerate().any(|(bi, b)| {
        if in_loop.contains(&(bi as u32)) {
            return false;
        }
        let mut leaks = false;
        for inst in &b.insts {
            each_op_use(&inst.op, &mut |v| leaks |= defined.contains(&v.0));
        }
        each_term_use(&b.term, &mut |v| leaks |= defined.contains(&v.0));
        leaks
    })
}

/// Append a copy of `blocks` to `f`: fresh block ids and fresh values for everything the blocks
/// define; edges between them stay inside the copy, edges out of them keep their targets. Returns
/// the copy's block indices (in `blocks` order), the block map and the value map.
fn clone_loop(
    f: &mut Function,
    blocks: &[BlockId],
) -> (Vec<usize>, FxHashMap<u32, usize>, FxHashMap<u32, ValueId>) {
    let base = f.blocks.len();
    let bmap: FxHashMap<u32, usize> = blocks
        .iter()
        .enumerate()
        .map(|(i, b)| (b.0, base + i))
        .collect();
    let mut vmap: FxHashMap<u32, ValueId> = FxHashMap::default();
    for b in blocks {
        let defs: Vec<ValueId> = {
            let blk = f.block(*b);
            blk.params
                .iter()
                .copied()
                .chain(blk.insts.iter().filter_map(|i| i.result))
                .collect()
        };
        for v in defs {
            let ty = f.value_type(v).clone();
            let nv = fresh(f, ty);
            vmap.insert(v.0, nv);
        }
    }
    let remap = |v: ValueId| vmap.get(&v.0).copied().unwrap_or(v);
    let mut copies = Vec::with_capacity(blocks.len());
    for (i, b) in blocks.iter().enumerate() {
        let src = f.block(*b);
        let params = src.params.iter().map(|p| remap(*p)).collect();
        let insts = src
            .insts
            .iter()
            .map(|inst| {
                let mut op = inst.op.clone();
                map_op_uses(&mut op, remap);
                Inst {
                    result: inst.result.map(remap),
                    op,
                }
            })
            .collect();
        let mut term = src.term.clone();
        map_term_uses(&mut term, remap);
        term.for_each_edge_mut(|t, _| {
            if let Some(&nb) = bmap.get(&t.0) {
                *t = BlockId(nb as u32);
            }
        });
        copies.push(BasicBlock {
            id: BlockId((base + i) as u32),
            params,
            insts,
            term,
        });
    }
    f.blocks.extend(copies);
    ((base..base + blocks.len()).collect(), bmap, vmap)
}

/// Within `blocks`, replace every use of `v` by the constant `k` (`t`/`fv` are the `i1` constants
/// `true`/`false`, defined where they dominate `blocks`) and fold the `and`/`or`/`not` that exposes.
/// The definitions of the folded values stay, for DCE.
fn assume(f: &mut Function, blocks: &[usize], v: ValueId, k: bool, t: ValueId, fv: ValueId) {
    let mut subst: FxHashMap<u32, ValueId> = FxHashMap::default();
    let mut known: FxHashMap<u32, bool> = FxHashMap::default();
    subst.insert(v.0, if k { t } else { fv });
    known.insert(t.0, true);
    known.insert(fv.0, false);
    let Function {
        blocks: fblocks,
        value_types,
        ..
    } = f;
    // A fold in one block can feed an instruction of another block visited earlier, so the sweep
    // repeats until nothing new is learned (one or two rounds in practice).
    for _ in 0..4 {
        let mut grew = false;
        for &b in blocks {
            for inst in &mut fblocks[b].insts {
                map_op_uses(&mut inst.op, |u| subst.get(&u.0).copied().unwrap_or(u));
                let Some(r) = inst.result else {
                    continue;
                };
                if subst.contains_key(&r.0) || value_types[r.0 as usize] != MirType::I1 {
                    continue;
                }
                let kn = |x: &ValueId| known.get(&x.0).copied();
                let repl = match &inst.op {
                    Op::Bin(BinOp::And, a, c) => match (kn(a), kn(c)) {
                        (Some(false), _) | (_, Some(false)) => Some(fv),
                        (Some(true), _) => Some(*c),
                        (_, Some(true)) => Some(*a),
                        _ => None,
                    },
                    Op::Bin(BinOp::Or, a, c) => match (kn(a), kn(c)) {
                        (Some(true), _) | (_, Some(true)) => Some(t),
                        (Some(false), _) => Some(*c),
                        (_, Some(false)) => Some(*a),
                        _ => None,
                    },
                    Op::Not(a) => kn(a).map(|x| if x { fv } else { t }),
                    _ => None,
                };
                if let Some(rv) = repl {
                    if let Some(kv) = known.get(&rv.0).copied() {
                        known.insert(r.0, kv);
                    }
                    subst.insert(r.0, rv);
                    grew = true;
                }
            }
        }
        if !grew {
            break;
        }
    }
    for &b in blocks {
        map_term_uses(&mut fblocks[b].term, |u| {
            subst.get(&u.0).copied().unwrap_or(u)
        });
    }
}

/// Emit `op` (of type `ty`) into `out`, returning its value.
fn emit(f: &mut Function, out: &mut Vec<Inst>, ty: MirType, op: Op) -> ValueId {
    let v = fresh(f, ty);
    out.push(Inst {
        result: Some(v),
        op,
    });
    v
}

fn materialize(f: &mut Function, out: &mut Vec<Inst>, a: Avail, ty: &MirType) -> ValueId {
    match a {
        Avail::Outside(v) => v,
        Avail::Const(n) => emit(f, out, ty.clone(), Op::ConstInt(n, ty.clone())),
    }
}

/// The preheader values a split point is computed from: the induction variable's start, the exit
/// test's bound, the guard's invariant side, and the exit test's predicate.
#[derive(Clone, Copy)]
struct SplitBounds {
    start: ValueId,
    end: ValueId,
    inv: ValueId,
    lt: CmpOp,
}

/// The first iteration where a [`DerivedGuard`] `x ⋈ n` changes, clamped into `[start, end]`.
///
/// With `x` growing by exactly one per iteration from `base = ext(start ± k)` (what
/// [`plan_derived`] proved), the guard keeps its first value for `n - base` iterations (`<`, `>=`)
/// or `n - base + 1` (`<=`, `>`), and for none when `n` is already past `base`. Every quantity is
/// formed so that it cannot overflow: `n - base` is taken only when `n` is not below `base`, where
/// it is an exact unsigned count of the compare's width, and it is compared with the trip count
/// `end - start` (exact unsigned, since `start < end`) before anything is added to `start`. An empty
/// range splits at `start`.
fn derived_split_point(
    f: &mut Function,
    out: &mut Vec<Inst>,
    d: &DerivedGuard,
    kind: GuardKind,
    b: SplitBounds,
) -> ValueId {
    let ity = f.value_type(b.start).clone();
    let wty = d.wty.clone();
    let x0 = match d.k {
        None => b.start,
        Some((k, sub)) => {
            let kv = materialize(f, out, k, &ity);
            let op = if sub { BinOp::Sub } else { BinOp::Add };
            emit(f, out, ity.clone(), Op::Bin(op, b.start, kv))
        }
    };
    let base = match d.ext {
        None => x0,
        Some(kind) => emit(f, out, wty.clone(), Op::Cast(kind, x0, wty.clone())),
    };
    // `<=` and `>` keep their first value one iteration longer than `<` and `>=`.
    let extra = matches!(kind, GuardKind::Le | GuardKind::Gt);
    let at_once = match (d.signed, extra) {
        (true, false) => CmpOp::Sle,
        (true, true) => CmpOp::Slt,
        (false, false) => CmpOp::Ule,
        (false, true) => CmpOp::Ult,
    };
    // The guard already has its second value on the first iteration.
    let early = emit(f, out, MirType::I1, Op::Cmp(at_once, b.inv, base));
    let dist = emit(f, out, wty.clone(), Op::Bin(BinOp::Sub, b.inv, base));
    let trip = emit(f, out, ity.clone(), Op::Bin(BinOp::Sub, b.end, b.start));
    let trip_w = match d.ext {
        None => trip,
        Some(_) => emit(
            f,
            out,
            wty.clone(),
            Op::Cast(CastKind::ZExt, trip, wty.clone()),
        ),
    };
    let inside = emit(f, out, MirType::I1, Op::Cmp(CmpOp::Ult, dist, trip_w));
    let dist_t = match d.ext {
        None => dist,
        Some(_) => emit(
            f,
            out,
            ity.clone(),
            Op::Cast(CastKind::Trunc, dist, ity.clone()),
        ),
    };
    let mut at = emit(f, out, ity.clone(), Op::Bin(BinOp::Add, b.start, dist_t));
    if extra {
        let one = emit(f, out, ity.clone(), Op::ConstInt(1, ity.clone()));
        at = emit(f, out, ity.clone(), Op::Bin(BinOp::Add, at, one));
    }
    let within = emit(f, out, ity.clone(), Op::Select(inside, at, b.end));
    let split = emit(f, out, ity.clone(), Op::Select(early, b.start, within));
    let runs = emit(f, out, MirType::I1, Op::Cmp(b.lt, b.start, b.end));
    emit(f, out, ity, Op::Select(runs, split, b.start))
}

fn apply_split(f: &mut Function, p: SplitPlan) {
    route_exit_values(f, &p.routes);
    let ph = p.preheader.0 as usize;
    let ity = f.value_type(p.iv).clone();
    let Terminator::Br { args, .. } = &f.blocks[ph].term else {
        unreachable!("a preheader ends in an unconditional branch to the header");
    };
    let start = args[p.iv_index];
    let mut out = Vec::new();
    let end = materialize(f, &mut out, p.end, &ity);
    let inv_ty = p
        .derived
        .as_ref()
        .map_or_else(|| ity.clone(), |d| d.wty.clone());
    let inv = materialize(f, &mut out, p.inv, &inv_ty);
    let lt = if p.signed { CmpOp::Slt } else { CmpOp::Ult };
    let bounds = SplitBounds {
        start,
        end,
        inv,
        lt,
    };
    // The first iteration where the guard changes, clamped into [start, end].
    let split = match (&p.derived, p.kind) {
        (Some(d), kind) => derived_split_point(f, &mut out, d, kind, bounds),
        (None, GuardKind::Lt | GuardKind::Ge) => {
            let inv_lt_end = emit(f, &mut out, MirType::I1, Op::Cmp(lt, inv, end));
            let lo = emit(f, &mut out, ity.clone(), Op::Select(inv_lt_end, inv, end));
            let lo_lt_start = emit(f, &mut out, MirType::I1, Op::Cmp(lt, lo, start));
            emit(f, &mut out, ity.clone(), Op::Select(lo_lt_start, start, lo))
        }
        (None, GuardKind::Le | GuardKind::Gt) => {
            let one = emit(f, &mut out, ity.clone(), Op::ConstInt(1, ity.clone()));
            let next = emit(f, &mut out, ity.clone(), Op::Bin(BinOp::Add, inv, one));
            let inv_lt_end = emit(f, &mut out, MirType::I1, Op::Cmp(lt, inv, end));
            let next_lt_start = emit(f, &mut out, MirType::I1, Op::Cmp(lt, next, start));
            let hi = emit(
                f,
                &mut out,
                ity.clone(),
                Op::Select(next_lt_start, start, next),
            );
            emit(f, &mut out, ity.clone(), Op::Select(inv_lt_end, hi, end))
        }
    };
    let t = emit(f, &mut out, MirType::I1, Op::ConstInt(1, MirType::I1));
    let fv = emit(f, &mut out, MirType::I1, Op::ConstInt(0, MirType::I1));
    f.blocks[ph].insts.extend(out);

    let originals: Vec<usize> = p.blocks.iter().map(|b| b.0 as usize).collect();
    let (copy, bmap, vmap) = clone_loop(f, &p.blocks);
    let ha = bmap[&p.header.0];
    // The copy runs [start, split) and then hands its whole state to the original header.
    let iv_a = vmap[&p.iv.0];
    let ca = fresh(f, MirType::I1);
    f.blocks[ha].insts.push(Inst {
        result: Some(ca),
        op: Op::Cmp(lt, iv_a, split),
    });
    let state = f.blocks[ha].params.clone();
    match &mut f.blocks[ha].term {
        Terminator::CondBr {
            cond,
            else_blk,
            else_args,
            ..
        } => {
            *cond = ca;
            *else_blk = p.header;
            *else_args = state;
        }
        _ => unreachable!("the copied header ends in the exit test"),
    }
    match &mut f.blocks[ph].term {
        Terminator::Br { target, .. } => *target = BlockId(ha as u32),
        _ => unreachable!("a preheader ends in an unconditional branch to the header"),
    }
    let prefix = matches!(p.kind, GuardKind::Lt | GuardKind::Le);
    // The guard holds its assumed value on every iteration a copy's *body* runs, but each header
    // also runs once more, for the exit test, with `iv` already past that copy's range (`split`
    // in the copy, `end` in the original). A header-computed guard there can take the other value
    // and reach the exit arguments or the code after the loop, so headers are never rewritten.
    let header = p.header.0 as usize;
    let copy_body: Vec<usize> = copy.iter().copied().filter(|&b| b != ha).collect();
    let orig_body: Vec<usize> = originals.iter().copied().filter(|&b| b != header).collect();
    assume(f, &copy_body, vmap[&p.guard.0], prefix, t, fv);
    assume(f, &orig_body, p.guard, !prefix, t, fv);
}

fn apply_unswitch(f: &mut Function, p: UnswitchPlan) {
    let ph = p.preheader.0 as usize;
    let mut out = Vec::new();
    let t = emit(f, &mut out, MirType::I1, Op::ConstInt(1, MirType::I1));
    let fv = emit(f, &mut out, MirType::I1, Op::ConstInt(0, MirType::I1));
    f.blocks[ph].insts.extend(out);
    let originals: Vec<usize> = p.blocks.iter().map(|b| b.0 as usize).collect();
    let (copy, bmap, _) = clone_loop(f, &p.blocks);
    let Terminator::Br { target, args } = &f.blocks[ph].term else {
        unreachable!("a preheader ends in an unconditional branch to the header");
    };
    debug_assert_eq!(*target, p.header);
    let args = args.clone();
    f.blocks[ph].term = Terminator::CondBr {
        cond: p.cond,
        then_blk: p.header,
        then_args: args.clone(),
        else_blk: BlockId(bmap[&p.header.0] as u32),
        else_args: args,
    };
    assume(f, &originals, p.cond, true, t, fv);
    assume(f, &copy, p.cond, false, t, fv);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Dce, Simplify, SimplifyCfg};
    use wukong_mir::{Builder, Program};
    use wukong_span::{Interner, SourceId};

    /// `f(start, end, inv) -> i32`, a canonical counted loop whose body guards an
    /// order-dependent update with `guard(j, inv)`:
    ///
    /// ```text
    ///   entry: br H(start, 7)
    ///   H(j, acc):  cond_br (j <exit end), B, X(acc)
    ///   B:          cond_br guard, T, L(acc)
    ///   T:          br L(acc * 3 + j)
    ///   L(p):       br H(j + 1, p)
    ///   X(r):       ret r
    /// ```
    ///
    /// `swap` writes the guard as `inv ⋈' j`.
    fn guarded_loop(it: &mut Interner, exit: CmpOp, pred: CmpOp, swap: bool) -> Function {
        let mut b = Builder::new(it.intern("f"), MirType::I32);
        let start = b.add_param(MirType::I32);
        let end = b.add_param(MirType::I32);
        let inv = b.add_param(MirType::I32);
        let seven = b.build(MirType::I32, Op::ConstInt(7, MirType::I32));
        let h = b.new_block();
        let j = b.block_param(h, MirType::I32);
        let acc = b.block_param(h, MirType::I32);
        let body = b.new_block();
        let then = b.new_block();
        let latch = b.new_block();
        let p = b.block_param(latch, MirType::I32);
        let exit_blk = b.new_block();
        let r = b.block_param(exit_blk, MirType::I32);
        b.br(h, vec![start, seven]);
        b.switch_to(h);
        let c = b.build(MirType::I1, Op::Cmp(exit, j, end));
        b.cond_br(c, body, vec![], exit_blk, vec![acc]);
        b.switch_to(body);
        let g = if swap {
            b.build(MirType::I1, Op::Cmp(pred, inv, j))
        } else {
            b.build(MirType::I1, Op::Cmp(pred, j, inv))
        };
        b.cond_br(g, then, vec![], latch, vec![acc]);
        b.switch_to(then);
        let three = b.build(MirType::I32, Op::ConstInt(3, MirType::I32));
        let m = b.build(MirType::I32, Op::Bin(BinOp::Mul, acc, three));
        let s = b.build(MirType::I32, Op::Bin(BinOp::Add, m, j));
        b.br(latch, vec![s]);
        b.switch_to(latch);
        let one = b.build(MirType::I32, Op::ConstInt(1, MirType::I32));
        let j2 = b.build(MirType::I32, Op::Bin(BinOp::Add, j, one));
        b.br(h, vec![j2, p]);
        b.switch_to(exit_blk);
        b.ret(Some(r));
        b.finish()
    }

    /// Run `f` on every triple from `main`, printing each result.
    fn run3(f: &Function, it: &mut Interner, inputs: &[(i128, i128, i128)]) -> Vec<u8> {
        let mut m = Builder::new(it.intern("main"), MirType::I32);
        for &(a, b, c) in inputs {
            let args = [a, b, c].map(|v| m.build(MirType::I32, Op::ConstInt(v, MirType::I32)));
            let r = m.build(
                MirType::I32,
                Op::Call {
                    func: f.name,
                    args: args.to_vec(),
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

    fn cleanup(f: &mut Function) {
        let mut cache = CfgAnalyses::default();
        for _ in 0..4 {
            let mut c = Simplify.run_function(f, &mut cache);
            c |= SimplifyCfg.run_function(f, &mut cache);
            c |= Dce.run_function(f, &mut cache);
            if !c {
                break;
            }
        }
    }

    /// Every (start, end, inv) combination: empty and reversed ranges, an invariant before, inside
    /// and after the range, and the type's extremes (where `inv + 1` would wrap).
    fn grid(unsigned: bool) -> Vec<(i128, i128, i128)> {
        let starts: &[i128] = if unsigned { &[0, 2, 6] } else { &[-3, 0, 5] };
        let ends: &[i128] = if unsigned { &[0, 4, 9] } else { &[-1, 4, 9] };
        let invs: &[i128] = &[-2147483648, -5, -1, 0, 1, 2, 4, 5, 8, 9, 20, 2147483647];
        let mut g = Vec::new();
        for &s in starts {
            for &e in ends {
                for &i in invs {
                    g.push((s, e, i));
                }
            }
        }
        g
    }

    #[test]
    fn iv_range_split_keeps_every_answer() {
        use CmpOp::*;
        for (exit, preds) in [(Slt, [Slt, Sle, Sgt, Sge]), (Ult, [Ult, Ule, Ugt, Uge])] {
            let inputs = grid(exit == Ult);
            for pred in preds {
                for swap in [false, true] {
                    let mut it = Interner::new();
                    let mut f = guarded_loop(&mut it, exit, pred, swap);
                    let before = run3(&f, &mut it, &inputs);
                    assert!(
                        LoopUnswitch.run_function(&mut f, &mut CfgAnalyses::default()),
                        "{exit:?}/{pred:?}/swap={swap}: the guard is splittable"
                    );
                    assert!(wukong_mir::verify::verify_function(&f).is_empty());
                    assert_eq!(
                        run3(&f, &mut it, &inputs),
                        before,
                        "{exit:?}/{pred:?}/{swap}"
                    );
                    cleanup(&mut f);
                    // Both copies decided the guard, so no branch is left in either body: only
                    // the two exit tests remain.
                    let branches = f
                        .blocks
                        .iter()
                        .filter(|b| matches!(b.term, Terminator::CondBr { .. }))
                        .count();
                    assert_eq!(
                        branches,
                        2,
                        "{}",
                        wukong_mir::print::print_function(&f, &it)
                    );
                    assert_eq!(
                        run3(&f, &mut it, &inputs),
                        before,
                        "{exit:?}/{pred:?}/{swap}"
                    );
                }
            }
        }
    }

    /// A guard the *header* computes also runs at the exit test, where `j` has left the body's
    /// range, and here it reaches the exit arguments: the split assumes it in the bodies only, so the
    /// exit still sees the guard's real final value (`end < inv`).
    ///
    /// ```text
    ///   H(j, acc):  g = j < inv; c = j < end; cond_br c, B, X(acc, g)
    ///   B:          cond_br g, T, L(acc)
    ///   T:          br L(acc * 3 + j)
    ///   L(p):       br H(j + 1, p)
    ///   X(r, q):    ret r + 1000 * zext(q)
    /// ```
    #[test]
    fn a_header_guard_keeps_its_value_at_the_exit_test() {
        let mut it = Interner::new();
        let mut b = Builder::new(it.intern("f"), MirType::I32);
        let start = b.add_param(MirType::I32);
        let end = b.add_param(MirType::I32);
        let inv = b.add_param(MirType::I32);
        let seven = b.build(MirType::I32, Op::ConstInt(7, MirType::I32));
        let h = b.new_block();
        let j = b.block_param(h, MirType::I32);
        let acc = b.block_param(h, MirType::I32);
        let body = b.new_block();
        let then = b.new_block();
        let latch = b.new_block();
        let p = b.block_param(latch, MirType::I32);
        let exit_blk = b.new_block();
        let r = b.block_param(exit_blk, MirType::I32);
        let q = b.block_param(exit_blk, MirType::I1);
        b.br(h, vec![start, seven]);
        b.switch_to(h);
        let g = b.build(MirType::I1, Op::Cmp(CmpOp::Slt, j, inv));
        let c = b.build(MirType::I1, Op::Cmp(CmpOp::Slt, j, end));
        b.cond_br(c, body, vec![], exit_blk, vec![acc, g]);
        b.switch_to(body);
        b.cond_br(g, then, vec![], latch, vec![acc]);
        b.switch_to(then);
        let three = b.build(MirType::I32, Op::ConstInt(3, MirType::I32));
        let m = b.build(MirType::I32, Op::Bin(BinOp::Mul, acc, three));
        let s = b.build(MirType::I32, Op::Bin(BinOp::Add, m, j));
        b.br(latch, vec![s]);
        b.switch_to(latch);
        let one = b.build(MirType::I32, Op::ConstInt(1, MirType::I32));
        let j2 = b.build(MirType::I32, Op::Bin(BinOp::Add, j, one));
        b.br(h, vec![j2, p]);
        b.switch_to(exit_blk);
        let w = b.build(
            MirType::I32,
            Op::Cast(wukong_mir::CastKind::ZExt, q, MirType::I32),
        );
        let k = b.build(MirType::I32, Op::ConstInt(1000, MirType::I32));
        let wk = b.build(MirType::I32, Op::Bin(BinOp::Mul, w, k));
        let out = b.build(MirType::I32, Op::Bin(BinOp::Add, r, wk));
        b.ret(Some(out));
        let mut f = b.finish();

        let inputs = grid(false);
        let before = run3(&f, &mut it, &inputs);
        assert!(LoopUnswitch.run_function(&mut f, &mut CfgAnalyses::default()));
        assert!(wukong_mir::verify::verify_function(&f).is_empty());
        assert_eq!(run3(&f, &mut it, &inputs), before);
        cleanup(&mut f);
        assert_eq!(run3(&f, &mut it, &inputs), before);
    }

    /// How the side exit of [`side_exit_loop`] reads the induction variable.
    #[derive(Clone, Copy, PartialEq, Eq, Debug)]
    enum Leak {
        /// Only through the exit edge's arguments.
        None,
        /// Directly, in the exit block itself (as a bounds check's trap reads the index).
        InExit,
        /// Directly, in a block after the exit block.
        Downstream,
    }

    /// `guarded_loop` with a side exit ahead of the guard -- the shape a bounds check's trap or a
    /// `break` gives -- leaving with the accumulator once it passes 1000:
    ///
    /// ```text
    ///   H(j, acc):  cond_br (j < end), B, X(acc)
    ///   B:          cond_br (acc > 1000), E(acc), G
    ///   G:          cond_br guard, T, L(acc)
    ///   T:          br L(acc * 3 + j)
    ///   L(p):       br H(j + 1, p)
    ///   E(q):       ret q + 5             (InExit: ret q + 5 + j; Downstream: br D(q + 5))
    ///   D(z):       ret z + j             (Downstream only)
    ///   X(r):       ret r
    /// ```
    fn side_exit_loop(it: &mut Interner, pred: CmpOp, leak: Leak) -> Function {
        let mut b = Builder::new(it.intern("f"), MirType::I32);
        let start = b.add_param(MirType::I32);
        let end = b.add_param(MirType::I32);
        let inv = b.add_param(MirType::I32);
        let seven = b.build(MirType::I32, Op::ConstInt(7, MirType::I32));
        let lim = b.build(MirType::I32, Op::ConstInt(1000, MirType::I32));
        let h = b.new_block();
        let j = b.block_param(h, MirType::I32);
        let acc = b.block_param(h, MirType::I32);
        let body = b.new_block();
        let guard_blk = b.new_block();
        let then = b.new_block();
        let latch = b.new_block();
        let p = b.block_param(latch, MirType::I32);
        let side = b.new_block();
        let q = b.block_param(side, MirType::I32);
        let exit_blk = b.new_block();
        let r = b.block_param(exit_blk, MirType::I32);
        b.br(h, vec![start, seven]);
        b.switch_to(h);
        let c = b.build(MirType::I1, Op::Cmp(CmpOp::Slt, j, end));
        b.cond_br(c, body, vec![], exit_blk, vec![acc]);
        b.switch_to(body);
        let big = b.build(MirType::I1, Op::Cmp(CmpOp::Sgt, acc, lim));
        b.cond_br(big, side, vec![acc], guard_blk, vec![]);
        b.switch_to(guard_blk);
        let g = b.build(MirType::I1, Op::Cmp(pred, j, inv));
        b.cond_br(g, then, vec![], latch, vec![acc]);
        b.switch_to(then);
        let three = b.build(MirType::I32, Op::ConstInt(3, MirType::I32));
        let m = b.build(MirType::I32, Op::Bin(BinOp::Mul, acc, three));
        let s = b.build(MirType::I32, Op::Bin(BinOp::Add, m, j));
        b.br(latch, vec![s]);
        b.switch_to(latch);
        let one = b.build(MirType::I32, Op::ConstInt(1, MirType::I32));
        let j2 = b.build(MirType::I32, Op::Bin(BinOp::Add, j, one));
        b.br(h, vec![j2, p]);
        b.switch_to(side);
        let five = b.build(MirType::I32, Op::ConstInt(5, MirType::I32));
        let mut out = b.build(MirType::I32, Op::Bin(BinOp::Add, q, five));
        match leak {
            Leak::None => b.ret(Some(out)),
            Leak::InExit => {
                out = b.build(MirType::I32, Op::Bin(BinOp::Add, out, j));
                b.ret(Some(out));
            }
            Leak::Downstream => {
                let d = b.new_block();
                let z = b.block_param(d, MirType::I32);
                b.br(d, vec![out]);
                b.switch_to(d);
                let w = b.build(MirType::I32, Op::Bin(BinOp::Add, z, j));
                b.ret(Some(w));
            }
        }
        b.switch_to(exit_blk);
        b.ret(Some(r));
        b.finish()
    }

    /// A loop with a side exit still splits on its induction-variable guard: whichever copy runs
    /// the iteration that leaves does so with the same state the unsplit loop had there. A side
    /// exit block that reads the induction variable itself gets it as an edge argument instead.
    #[test]
    fn iv_range_split_with_a_side_exit_keeps_every_answer() {
        use CmpOp::*;
        for leak in [Leak::None, Leak::InExit] {
            let mut inputs = grid(false);
            // Long enough ranges that the accumulator passes 1000 inside either copy.
            for &(s, e, i) in &[
                (0, 12, 3),
                (0, 12, 9),
                (-3, 20, 0),
                (-3, 20, 15),
                (2, 30, 30),
            ] {
                inputs.push((s, e, i));
            }
            for pred in [Slt, Sle, Sgt, Sge] {
                let mut it = Interner::new();
                let mut f = side_exit_loop(&mut it, pred, leak);
                let before = run3(&f, &mut it, &inputs);
                // The side exit is reached in some of these runs, or the test would prove nothing.
                let reached = before
                    .split(|&c| c == b'\n')
                    .filter_map(|l| std::str::from_utf8(l).ok()?.parse::<i64>().ok())
                    .any(|v| v > 1000);
                assert!(reached, "{pred:?}: no run took the side exit");
                assert!(
                    LoopUnswitch.run_function(&mut f, &mut CfgAnalyses::default()),
                    "{pred:?}: the guard is splittable past a side exit"
                );
                assert!(wukong_mir::verify::verify_function(&f).is_empty());
                assert_eq!(run3(&f, &mut it, &inputs), before, "{pred:?}");
                cleanup(&mut f);
                assert!(wukong_mir::verify::verify_function(&f).is_empty());
                // No compare against `inv` survives in either copy (the preheader's split-point
                // arithmetic is all selects).
                let inv = f.blocks[0].params[2];
                let guards = f
                    .blocks
                    .iter()
                    .skip(1)
                    .flat_map(|b| &b.insts)
                    .filter(
                        |i| matches!(i.op, Op::Cmp(p, x, y) if p == pred && (x == inv || y == inv)),
                    )
                    .count();
                assert_eq!(guards, 0, "{}", wukong_mir::print::print_function(&f, &it));
                assert_eq!(run3(&f, &mut it, &inputs), before, "{leak:?}/{pred:?}");
            }
        }
    }

    /// Code after a side exit that reads a loop value directly, beyond the exit block where an edge
    /// argument could carry it, would read the original loop's definition after leaving from the
    /// copy: no split.
    #[test]
    fn a_side_exit_reading_a_loop_value_downstream_blocks_the_split() {
        let mut it = Interner::new();
        let mut f = side_exit_loop(&mut it, CmpOp::Sgt, Leak::Downstream);
        let inputs = [(0, 12, 3), (0, 12, 9), (-3, 4, 0)];
        let before = run3(&f, &mut it, &inputs);
        assert!(!LoopUnswitch.run_function(&mut f, &mut CfgAnalyses::default()));
        assert_eq!(run3(&f, &mut it, &inputs), before);
    }

    /// A body branch on a loop-invariant condition (also as one operand of an `and`) is decided once
    /// in the preheader.
    #[test]
    fn an_invariant_branch_is_unswitched() {
        for with_and in [false, true] {
            let mut it = Interner::new();
            let mut b = Builder::new(it.intern("f"), MirType::I32);
            let n = b.add_param(MirType::I32);
            let x = b.add_param(MirType::I32);
            let y = b.add_param(MirType::I32);
            let zero = b.build(MirType::I32, Op::ConstInt(0, MirType::I32));
            let flag = b.build(MirType::I1, Op::Cmp(CmpOp::Sgt, x, zero));
            let h = b.new_block();
            let j = b.block_param(h, MirType::I32);
            let acc = b.block_param(h, MirType::I32);
            let body = b.new_block();
            let then = b.new_block();
            let latch = b.new_block();
            let p = b.block_param(latch, MirType::I32);
            let exit_blk = b.new_block();
            let r = b.block_param(exit_blk, MirType::I32);
            b.br(h, vec![zero, y]);
            b.switch_to(h);
            let c = b.build(MirType::I1, Op::Cmp(CmpOp::Slt, j, n));
            b.cond_br(c, body, vec![], exit_blk, vec![acc]);
            b.switch_to(body);
            let g = if with_and {
                // `flag && acc is odd`: the second operand varies, so this is not a split.
                let one = b.build(MirType::I32, Op::ConstInt(1, MirType::I32));
                let bit = b.build(MirType::I32, Op::Bin(BinOp::And, acc, one));
                let odd = b.build(MirType::I1, Op::Cmp(CmpOp::Ne, bit, zero));
                b.build(MirType::I1, Op::Bin(BinOp::And, flag, odd))
            } else {
                flag
            };
            b.cond_br(g, then, vec![], latch, vec![acc]);
            b.switch_to(then);
            let three = b.build(MirType::I32, Op::ConstInt(3, MirType::I32));
            let m = b.build(MirType::I32, Op::Bin(BinOp::Mul, acc, three));
            let s = b.build(MirType::I32, Op::Bin(BinOp::Add, m, j));
            b.br(latch, vec![s]);
            b.switch_to(latch);
            let one = b.build(MirType::I32, Op::ConstInt(1, MirType::I32));
            let j2 = b.build(MirType::I32, Op::Bin(BinOp::Add, j, one));
            b.br(h, vec![j2, p]);
            b.switch_to(exit_blk);
            b.ret(Some(r));
            let mut f = b.finish();

            let inputs: Vec<(i128, i128, i128)> = [0, 3, 7]
                .into_iter()
                .flat_map(|n| [-2, 0, 5].map(move |x| (n, x, 5)))
                .collect();
            let before = run3(&f, &mut it, &inputs);
            assert!(LoopUnswitch.run_function(&mut f, &mut CfgAnalyses::default()));
            assert!(wukong_mir::verify::verify_function(&f).is_empty());
            assert!(
                matches!(f.blocks[0].term, Terminator::CondBr { cond, .. } if cond == flag),
                "the preheader decides the flag: {}",
                wukong_mir::print::print_function(&f, &it)
            );
            assert_eq!(run3(&f, &mut it, &inputs), before);
            cleanup(&mut f);
            assert_eq!(run3(&f, &mut it, &inputs), before);
        }
    }

    /// A loop whose value is read after it other than through an exit edge's arguments is not
    /// unswitched: the copy's exit would bypass that value's definition.
    #[test]
    fn a_loop_value_read_after_the_loop_blocks_unswitching() {
        let mut it = Interner::new();
        let mut b = Builder::new(it.intern("f"), MirType::I32);
        let n = b.add_param(MirType::I32);
        let x = b.add_param(MirType::I32);
        let _y = b.add_param(MirType::I32);
        let zero = b.build(MirType::I32, Op::ConstInt(0, MirType::I32));
        let flag = b.build(MirType::I1, Op::Cmp(CmpOp::Sgt, x, zero));
        let h = b.new_block();
        let j = b.block_param(h, MirType::I32);
        let body = b.new_block();
        let then = b.new_block();
        let latch = b.new_block();
        let exit_blk = b.new_block();
        b.br(h, vec![zero]);
        b.switch_to(h);
        let c = b.build(MirType::I1, Op::Cmp(CmpOp::Slt, j, n));
        b.cond_br(c, body, vec![], exit_blk, vec![]);
        b.switch_to(body);
        b.cond_br(flag, then, vec![], latch, vec![]);
        b.switch_to(then);
        b.br(latch, vec![]);
        b.switch_to(latch);
        let one = b.build(MirType::I32, Op::ConstInt(1, MirType::I32));
        let j2 = b.build(MirType::I32, Op::Bin(BinOp::Add, j, one));
        b.br(h, vec![j2]);
        b.switch_to(exit_blk);
        // `j` is read after the loop directly: a leak.
        b.ret(Some(j));
        let mut f = b.finish();
        assert!(!LoopUnswitch.run_function(&mut f, &mut CfgAnalyses::default()));
        assert_eq!(
            run3(&f, &mut it, &[(4, 1, 0), (4, -1, 0)]),
            b"4\n4\n".to_vec()
        );
    }

    /// From source, at every level: a causal (strictly upper) mask, its lower complement and a
    /// `>=` against a parameter, all splittable, compute what the scalar loops say.
    #[test]
    fn source_masks_split_and_agree() {
        let src = concat!(
            "fn upper(mut s: [i32; 16]) { for i in 0..4 { for j in 0..4 { ",
            "if j > i { s[i * 4 + j] = 0 - 1; } } } }\n",
            "fn lower(mut s: [i32; 16]) { for i in 0..4 { for j in 0..4 { ",
            "if j < i { s[i * 4 + j] = 0 - 2; } } } }\n",
            "fn from(mut s: [i32; 16], k: i32) { for j in 0..16 { if j >= k { s[j] = 9; } } }\n",
            "fn main() -> i32 { let mut a: [i32; 16] = [0; 16]; let mut b: [i32; 16] = [0; 16]; ",
            "let mut c: [i32; 16] = [0; 16]; ",
            "for t in 0..16 { a[t] = t as i32; b[t] = t as i32; c[t] = t as i32; } ",
            "upper(a); lower(b); from(c, 11); ",
            "for t in 0..16 { print(a[t] * 100 + b[t] * 10 + c[t]); } return 0; }\n",
        );
        let mut interner = Interner::new();
        let (module, parse) = wukong_parser::parse_module(src, SourceId(0), &mut interner);
        assert!(parse.iter().all(|d| !d.is_error()), "{parse:?}");
        let (sema, check) = wukong_sema::check(&module, &interner);
        assert!(check.iter().all(|d| !d.is_error()), "{check:?}");
        let (program, lower) = wukong_mir_build::lower_program(&module, &sema, &mut interner);
        assert!(lower.iter().all(|d| !d.is_error()), "{lower:?}");
        // Lane t = 4i + j: a = -1 when j > i else t; b = -2 when j < i else t; c = 9 when t >= 11
        // else t. Printed as a*100 + b*10 + c.
        let mut expected = String::new();
        for t in 0..16i64 {
            let (i, j) = (t / 4, t % 4);
            let a = if j > i { -1 } else { t };
            let b = if j < i { -2 } else { t };
            let c = if t >= 11 { 9 } else { t };
            expected.push_str(&format!("{}\n", a * 100 + b * 10 + c));
        }
        let main = interner.intern("main");
        for level in [0u8, 1, 2] {
            let mut optimized = program.clone();
            crate::optimize(&mut optimized, level);
            assert!(wukong_mir::verify::verify_program(&optimized).is_empty());
            let (exit, stdout) =
                wukong_interp::run_with_output(&optimized, main, &interner).unwrap();
            assert_eq!(exit, 0);
            assert_eq!(String::from_utf8(stdout).unwrap(), expected, "O{level}");
        }
    }

    // ---- guards on quantities derived from the induction variable ----

    /// What the guard of [`derived_loop`] tests, `j` being the counter and `n`, `k` parameters.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum Derived {
        /// `j + c ⋈ n`.
        AddConst(i128),
        /// `j - c ⋈ n`.
        SubConst(i128),
        /// `(k & 15) + j ⋈ n` -- a bounded invariant addend, the tile edge's shape.
        AddMasked,
        /// `k + j ⋈ n` with nothing known about `k`.
        AddParam,
        /// `sext(j) ⋈ sext(n)` in `i64`.
        Sext,
        /// `sext(j + 1) ⋈ sext(n)` in `i64`.
        SextAddOne,
        /// `j ⋈ n` with the other signedness than the exit test.
        Mixed,
    }

    /// Where [`derived_loop`]'s counter starts and stops.
    #[derive(Clone, Copy, Debug)]
    enum Bound {
        Const(i128),
        /// The third parameter `e`.
        Param,
    }

    /// `f(n, k, e) -> i32`: `for j in start..end` (`exit` the header's predicate), updating an
    /// order-dependent accumulator when the derived guard holds.
    ///
    /// ```text
    ///   entry: km = k & 15; n64 = sext n; br H(start, 7)
    ///   H(j, acc):  cond_br (j <exit end), B, X(acc)
    ///   B:          x = <shape>; cond_br (x ⋈ n), T, L(acc)
    ///   T:          br L(acc * 3 + j)
    ///   L(p):       br H(j + 1, p)
    ///   X(r):       ret r
    /// ```
    fn derived_loop(
        it: &mut Interner,
        exit: CmpOp,
        pred: CmpOp,
        shape: Derived,
        start: Bound,
        end: Bound,
        swap: bool,
    ) -> Function {
        let mut b = Builder::new(it.intern("f"), MirType::I32);
        let n = b.add_param(MirType::I32);
        let k = b.add_param(MirType::I32);
        let e = b.add_param(MirType::I32);
        let fifteen = b.build(MirType::I32, Op::ConstInt(15, MirType::I32));
        let km = b.build(MirType::I32, Op::Bin(BinOp::And, k, fifteen));
        let n64 = b.build(MirType::I64, Op::Cast(CastKind::SExt, n, MirType::I64));
        let seven = b.build(MirType::I32, Op::ConstInt(7, MirType::I32));
        let bound = |b: &mut Builder, x: Bound| match x {
            Bound::Const(c) => b.build(MirType::I32, Op::ConstInt(c, MirType::I32)),
            Bound::Param => e,
        };
        let s = bound(&mut b, start);
        let end_v = bound(&mut b, end);
        let h = b.new_block();
        let j = b.block_param(h, MirType::I32);
        let acc = b.block_param(h, MirType::I32);
        let body = b.new_block();
        let then = b.new_block();
        let latch = b.new_block();
        let p = b.block_param(latch, MirType::I32);
        let exit_blk = b.new_block();
        let r = b.block_param(exit_blk, MirType::I32);
        b.br(h, vec![s, seven]);
        b.switch_to(h);
        let c = b.build(MirType::I1, Op::Cmp(exit, j, end_v));
        b.cond_br(c, body, vec![], exit_blk, vec![acc]);
        b.switch_to(body);
        let konst = |b: &mut Builder, c: i128| b.build(MirType::I32, Op::ConstInt(c, MirType::I32));
        let (x, rhs) = match shape {
            Derived::AddConst(c) => {
                let cv = konst(&mut b, c);
                (b.build(MirType::I32, Op::Bin(BinOp::Add, j, cv)), n)
            }
            Derived::SubConst(c) => {
                let cv = konst(&mut b, c);
                (b.build(MirType::I32, Op::Bin(BinOp::Sub, j, cv)), n)
            }
            Derived::AddMasked => (b.build(MirType::I32, Op::Bin(BinOp::Add, km, j)), n),
            Derived::AddParam => (b.build(MirType::I32, Op::Bin(BinOp::Add, k, j)), n),
            Derived::Sext => (
                b.build(MirType::I64, Op::Cast(CastKind::SExt, j, MirType::I64)),
                n64,
            ),
            Derived::SextAddOne => {
                let one = konst(&mut b, 1);
                let j1 = b.build(MirType::I32, Op::Bin(BinOp::Add, j, one));
                (
                    b.build(MirType::I64, Op::Cast(CastKind::SExt, j1, MirType::I64)),
                    n64,
                )
            }
            Derived::Mixed => (j, n),
        };
        let g = if swap {
            b.build(MirType::I1, Op::Cmp(swapped(pred).unwrap(), rhs, x))
        } else {
            b.build(MirType::I1, Op::Cmp(pred, x, rhs))
        };
        b.cond_br(g, then, vec![], latch, vec![acc]);
        b.switch_to(then);
        let three = konst(&mut b, 3);
        let m = b.build(MirType::I32, Op::Bin(BinOp::Mul, acc, three));
        let sum = b.build(MirType::I32, Op::Bin(BinOp::Add, m, j));
        b.br(latch, vec![sum]);
        b.switch_to(latch);
        let one = konst(&mut b, 1);
        let j2 = b.build(MirType::I32, Op::Bin(BinOp::Add, j, one));
        b.br(h, vec![j2, p]);
        b.switch_to(exit_blk);
        b.ret(Some(r));
        b.finish()
    }

    /// Every `(n, k, e)`: `n` before, inside and past the range and at the type's extremes, `k`
    /// negative and positive, `e` empty, short and full.
    fn derived_grid() -> Vec<(i128, i128, i128)> {
        let ns: &[i128] = &[
            -2147483648,
            -2147483647,
            -5,
            -1,
            0,
            1,
            3,
            7,
            11,
            12,
            13,
            20,
            2147483646,
            2147483647,
        ];
        let mut g = Vec::new();
        for &n in ns {
            for k in [-3, 0, 5, 37] {
                for e in [0, 9, 12] {
                    g.push((n, k, e));
                }
            }
        }
        g
    }

    fn cond_branches(f: &Function) -> usize {
        f.blocks
            .iter()
            .filter(|b| matches!(b.term, Terminator::CondBr { .. }))
            .count()
    }

    /// The unsigned predicate with the same shape as a signed one, and back.
    fn other_signedness(p: CmpOp) -> CmpOp {
        use CmpOp::*;
        match p {
            Slt => Ult,
            Sle => Ule,
            Sgt => Ugt,
            Sge => Uge,
            Ult => Slt,
            Ule => Sle,
            Ugt => Sgt,
            Uge => Sge,
            other => other,
        }
    }

    /// Guards on `j ± c`, `(k & 15) + j`, `sext(j)`, `sext(j + 1)`, and `j` read with the other
    /// signedness, under both exit signednesses and all four predicates (and with the operands
    /// exchanged): each splits, the copies' bodies lose the guard, and every answer is unchanged --
    /// including invariants at the type's extremes, where a split point computed carelessly
    /// overflows.
    #[test]
    fn derived_guards_split_and_keep_every_answer() {
        use CmpOp::*;
        let inputs = derived_grid();
        let shapes = [
            Derived::AddConst(1),
            Derived::AddConst(3),
            Derived::SubConst(2),
            Derived::AddMasked,
            Derived::Sext,
            Derived::SextAddOne,
            Derived::Mixed,
        ];
        for exit in [Slt, Ult] {
            for shape in shapes {
                // `j - 2` read unsigned wraps below 2: that one is the decline test's.
                if exit == Ult && shape == Derived::SubConst(2) {
                    continue;
                }
                for base in [Slt, Sle, Sgt, Sge] {
                    // The guard's own signedness: the exit test's, except for the mixed shape.
                    let same = if exit == Ult {
                        other_signedness(base)
                    } else {
                        base
                    };
                    let pred = if shape == Derived::Mixed {
                        other_signedness(same)
                    } else if matches!(shape, Derived::Sext | Derived::SextAddOne) {
                        base // compared in i64, signed: the sign extension's own reading
                    } else {
                        same
                    };
                    for swap in [false, true] {
                        let mut it = Interner::new();
                        let mut f = derived_loop(
                            &mut it,
                            exit,
                            pred,
                            shape,
                            Bound::Const(0),
                            Bound::Const(12),
                            swap,
                        );
                        let tag = format!("{exit:?}/{shape:?}/{pred:?}/swap={swap}");
                        let before = run3(&f, &mut it, &inputs);
                        assert!(
                            LoopUnswitch.run_function(&mut f, &mut CfgAnalyses::default()),
                            "{tag}: the derived guard is splittable\n{}",
                            wukong_mir::print::print_function(&f, &it)
                        );
                        assert!(wukong_mir::verify::verify_function(&f).is_empty(), "{tag}");
                        assert_eq!(run3(&f, &mut it, &inputs), before, "{tag}");
                        cleanup(&mut f);
                        assert_eq!(
                            cond_branches(&f),
                            2,
                            "{tag}: only the two exit tests remain\n{}",
                            wukong_mir::print::print_function(&f, &it)
                        );
                        assert_eq!(run3(&f, &mut it, &inputs), before, "{tag}");
                    }
                }
            }
        }
    }

    /// Against a runtime bound `e` nothing caps the counter below the type's top except the exit
    /// test itself, which keeps `j + 1` from wrapping (`j <= e - 1`): the stencil guard `j + 1 ⋈ n`
    /// splits, and the answers hold for every `e`.
    #[test]
    fn a_unit_addend_splits_against_a_runtime_bound() {
        use CmpOp::*;
        let inputs = derived_grid();
        for pred in [Slt, Sle, Sgt, Sge] {
            let mut it = Interner::new();
            let mut f = derived_loop(
                &mut it,
                Slt,
                pred,
                Derived::AddConst(1),
                Bound::Const(0),
                Bound::Param,
                false,
            );
            let before = run3(&f, &mut it, &inputs);
            assert!(
                LoopUnswitch.run_function(&mut f, &mut CfgAnalyses::default()),
                "{pred:?}: `j + 1` cannot wrap below a runtime bound"
            );
            assert!(wukong_mir::verify::verify_function(&f).is_empty());
            assert_eq!(run3(&f, &mut it, &inputs), before, "{pred:?}");
            cleanup(&mut f);
            assert_eq!(cond_branches(&f), 2, "{pred:?}");
            assert_eq!(run3(&f, &mut it, &inputs), before, "{pred:?}");
        }
    }

    /// Guards whose derived quantity may wrap -- so they may change twice along the iterations --
    /// are left alone: `j + 3` below a runtime bound (`j = e - 1` near the top wraps), `j - 2` from
    /// a runtime start (near the bottom), `k + j` with nothing known about `k`, and a signed counter
    /// read unsigned from a start that may be negative (the "`i64` index against a `usize` bound"
    /// case). With the start a non-negative constant the mixed guard does split (the test above).
    #[test]
    fn derived_guards_that_may_wrap_are_not_split() {
        use CmpOp::*;
        let cases = [
            (Slt, Derived::AddConst(3), Bound::Const(0), Bound::Param),
            (Slt, Derived::SubConst(2), Bound::Param, Bound::Const(12)),
            (Slt, Derived::AddParam, Bound::Const(0), Bound::Const(12)),
            (Ult, Derived::Mixed, Bound::Param, Bound::Const(12)),
            (
                Ult,
                Derived::AddConst(-1),
                Bound::Const(0),
                Bound::Const(12),
            ),
        ];
        for (pred, shape, start, end) in cases {
            let mut it = Interner::new();
            // `start = Param` reuses `e`; the test only asks whether the pass fires.
            let mut f = derived_loop(&mut it, Slt, pred, shape, start, end, false);
            assert!(
                !LoopUnswitch.run_function(&mut f, &mut CfgAnalyses::default()),
                "{shape:?}/{pred:?}: a guard that may wrap must not be split\n{}",
                wukong_mir::print::print_function(&f, &it)
            );
        }
    }

    /// From source, the tile-edge guard `jo + j < n` of a strip-mined loop: `jo` is the enclosing
    /// loop's counter (stepping by 8 below 24, so `jo + j` stays small), the inner loop splits at
    /// `n - jo`, and no compare against `n` is left in any innermost loop. Every level prints the
    /// lanes the scalar loop says.
    #[test]
    fn source_tile_edge_guard_splits_on_the_enclosing_counter() {
        let src = concat!(
            "fn tile(mut out: [i32; 24], n: i64) { for jo in 0..24 step 8 { for j in 0..8 { ",
            "if jo + j < n { out[jo + j] = (jo + j) * 2 + 1; } } } }\n",
            "fn main() -> i32 { let mut out: [i32; 24] = [0; 24]; ",
            "let ns: [i64; 6] = [0 - 4, 0, 5, 13, 24, 30]; ",
            "for t in 0..6 { for i in 0..24 { out[i] = 0 - 1; } tile(out, ns[t]); ",
            "for i in 0..24 { print(out[i]); } } return 0; }\n",
        );
        let mut interner = Interner::new();
        let (module, parse) = wukong_parser::parse_module(src, SourceId(0), &mut interner);
        assert!(parse.iter().all(|d| !d.is_error()), "{parse:?}");
        let (sema, check) = wukong_sema::check(&module, &interner);
        assert!(check.iter().all(|d| !d.is_error()), "{check:?}");
        let (program, lower) = wukong_mir_build::lower_program(&module, &sema, &mut interner);
        assert!(lower.iter().all(|d| !d.is_error()), "{lower:?}");
        let mut expected = String::new();
        for n in [-4i64, 0, 5, 13, 24, 30] {
            for i in 0..24i64 {
                let v = if i < n { i * 2 + 1 } else { -1 };
                expected.push_str(&format!("{v}\n"));
            }
        }
        let main = interner.intern("main");
        for level in [0u8, 1, 2] {
            let mut optimized = program.clone();
            crate::optimize(&mut optimized, level);
            assert!(wukong_mir::verify::verify_program(&optimized).is_empty());
            let (exit, stdout) =
                wukong_interp::run_with_output(&optimized, main, &interner).unwrap();
            assert_eq!(exit, 0);
            assert_eq!(String::from_utf8(stdout).unwrap(), expected, "O{level}");
        }
        // The function on its own (no inlining), through the -O2 fixpoint.
        let mut optimized = program.clone();
        crate::PassManager::standard(2).run(&mut optimized);
        let tile = optimized.function(interner.intern("tile")).unwrap();
        let n = tile.blocks[tile.entry.0 as usize].params[1];
        let forest = crate::loop_info::analyze_function(tile);
        let mut innermost = 0;
        let mut inner_blocks = FxHashSet::default();
        for l in forest.loops.iter().filter(|l| l.children.is_empty()) {
            innermost += 1;
            inner_blocks.extend(l.blocks.iter().map(|b| b.0));
            for b in &l.blocks {
                for inst in &tile.block(*b).insts {
                    let reads_n = matches!(inst.op, Op::Cmp(_, x, y) if x == n || y == n);
                    assert!(
                        !reads_n,
                        "an innermost loop still tests the edge:\n{}",
                        wukong_mir::print::print_function(tile, &interner)
                    );
                }
            }
        }
        // The half of the split where the guard is false stores nothing, so loop deletion removes
        // it; what proves the split is the split point's own test of `n` in the inner preheader.
        assert!(innermost >= 1);
        let split_point_reads_n = tile.blocks.iter().any(|b| {
            !inner_blocks.contains(&b.id.0)
                && b.insts
                    .iter()
                    .any(|i| matches!(i.op, Op::Cmp(_, x, y) if x == n || y == n))
        });
        assert!(
            split_point_reads_n,
            "{}",
            wukong_mir::print::print_function(tile, &interner)
        );
    }

    // ---- loops holding calls ----

    /// Lower `src`, optimize it at `-O0` and `-O2` through the whole optimizer, check that both
    /// print the same, and return the `-O2` program.
    fn o0_against_o2(src: &str, interner: &mut Interner) -> Program {
        let (module, parse) = wukong_parser::parse_module(src, SourceId(0), interner);
        assert!(parse.iter().all(|d| !d.is_error()), "{parse:?}");
        let (sema, check) = wukong_sema::check(&module, interner);
        assert!(check.iter().all(|d| !d.is_error()), "{check:?}");
        let (program, lower) = wukong_mir_build::lower_program(&module, &sema, interner);
        assert!(lower.iter().all(|d| !d.is_error()), "{lower:?}");
        let main = interner.intern("main");
        let mut outputs = Vec::new();
        let mut last = None;
        for level in [0u8, 2] {
            let mut optimized = program.clone();
            crate::optimize(&mut optimized, level);
            assert!(wukong_mir::verify::verify_program(&optimized).is_empty());
            let (exit, stdout) =
                wukong_interp::run_with_output(&optimized, main, interner).unwrap();
            assert_eq!(exit, 0);
            outputs.push(String::from_utf8(stdout).unwrap());
            last = Some(optimized);
        }
        assert!(!outputs[0].is_empty());
        assert_eq!(outputs[0], outputs[1]);
        last.unwrap()
    }

    /// How many calls to `callee` `f` makes, and whether it computes anything vector-typed.
    fn calls_and_vectors(f: &Function, callee: wukong_span::Symbol) -> (usize, bool) {
        let calls = f
            .blocks
            .iter()
            .flat_map(|b| &b.insts)
            .filter(|i| matches!(&i.op, Op::Call { func, .. } if *func == callee))
            .count();
        (calls, f.value_types.iter().any(MirType::is_vector))
    }

    /// A diagnostic call under a loop-invariant flag: the loop is unswitched on the flag, the copy
    /// where it is false has no call, and the vectorizer widens that copy. The call itself is not
    /// duplicated -- it lives on in the flag-true copy only -- and both flag values print what the
    /// scalar loop prints, at `-O0` and `-O2`. (`k` and `again` call each other, which keeps `k` out
    /// of line, where its `@noalias` output keeps its promise.)
    #[test]
    fn an_invariant_guarded_call_is_unswitched_and_the_free_copy_widens() {
        let src = concat!(
            "fn k(@noalias mut o: [f32; 37], a: [f32; 37], n: i64, dbg: bool, d: i64) { ",
            "if d > 0 { again(d - 1); return; } ",
            "let mut i: i64 = 0; ",
            "while i < n { if dbg { print(i); } o[i] = a[i] * 2.0 + 1.0; i = i + 1; } }\n",
            "fn again(d: i64) { let a: [f32; 37] = [0.0; 37]; let mut o: [f32; 37] = [0.0; 37]; ",
            "k(o, a, 0, false, d); }\n",
            "fn main() -> i32 { let mut a: [f32; 37] = [0.0; 37]; let mut o: [f32; 37] = [0.0; 37]; ",
            "let mut j: i64 = 0; while j < 37 { a[j] = (j as f32) - 18.0; j = j + 1; } ",
            "k(o, a, 37, true, 0); ",
            "j = 0; while j < 37 { print(o[j] as i64); o[j] = 0.0 - 99.0; j = j + 1; } ",
            "k(o, a, 37, false, 0); ",
            "j = 0; while j < 37 { print(o[j] as i64); j = j + 1; } ",
            "k(o, a, 5, true, 0); ",
            "j = 0; while j < 37 { print(o[j] as i64); j = j + 1; } return 0; }\n",
        );
        let mut interner = Interner::new();
        let optimized = o0_against_o2(src, &mut interner);
        let k = optimized
            .function(interner.intern("k"))
            .expect("kept out of line");
        let (prints, widened) = calls_and_vectors(k, interner.intern("print"));
        assert_eq!(
            (prints, widened),
            (1, true),
            "{}",
            wukong_mir::print::print_function(k, &interner)
        );
    }

    /// A call made only for a prefix of the iterations (`if i < m`, `m` a parameter): the loop is
    /// split at `m`, the `[m, n)` half has no call and widens, and the call is not duplicated.
    #[test]
    fn a_prefix_guarded_call_is_split() {
        let src = concat!(
            "fn k(@noalias mut o: [f32; 37], a: [f32; 37], n: i64, m: i64, d: i64) { ",
            "if d > 0 { again(d - 1); return; } ",
            "let mut i: i64 = 0; ",
            "while i < n { if i < m { print(i); } o[i] = a[i] * 2.0 + 1.0; i = i + 1; } }\n",
            "fn again(d: i64) { let a: [f32; 37] = [0.0; 37]; let mut o: [f32; 37] = [0.0; 37]; ",
            "k(o, a, 0, 0, d); }\n",
            "fn main() -> i32 { let mut a: [f32; 37] = [0.0; 37]; let mut o: [f32; 37] = [0.0; 37]; ",
            "let mut j: i64 = 0; while j < 37 { a[j] = (j as f32) - 18.0; j = j + 1; } ",
            "let ms: [i64; 4] = [0, 5, 37, 50]; let ns: [i64; 4] = [37, 36, 35, 34]; ",
            "for q in 0..4 { j = 0; while j < 37 { o[j] = 0.0 - 99.0; j = j + 1; } ",
            "k(o, a, ns[q], ms[q], 0); ",
            "j = 0; while j < 37 { print(o[j] as i64); j = j + 1; } } return 0; }\n",
        );
        let mut interner = Interner::new();
        let optimized = o0_against_o2(src, &mut interner);
        let k = optimized
            .function(interner.intern("k"))
            .expect("kept out of line");
        let (prints, widened) = calls_and_vectors(k, interner.intern("print"));
        assert_eq!(
            (prints, widened),
            (1, true),
            "{}",
            wukong_mir::print::print_function(k, &interner)
        );
    }

    /// Where the call of [`call_loop`] sits.
    #[derive(Clone, Copy, PartialEq, Eq, Debug)]
    enum CallAt {
        /// In the flag's `then` arm only: confined to one arm.
        Arm,
        /// In the latch, which every iteration runs.
        Unguarded,
        /// In both arms of the flag's branch.
        BothArms,
        /// In the `then` arm, but the branch tests `flag && (acc is odd)`.
        AndOperand,
        /// No call: a barrier in the `then` arm.
        Barrier,
    }

    /// `f(n, x, y) -> i32` over `0..n`, a loop-invariant `flag = x > 0` guarding an
    /// order-dependent update, with a `print` (or a barrier) placed by `at`:
    ///
    /// ```text
    ///   entry: flag = x > 0; br H(0, y)
    ///   H(j, acc):  cond_br (j < n), B, X(acc)
    ///   B:          cond_br flag, T, E          (AndOperand: flag && acc & 1 != 0)
    ///   T:          [print j]; br L(acc * 3 + j)
    ///   E:          [print j + 100]; br L(acc)  (a call only for BothArms)
    ///   L(p):       [print j]; br H(j + 1, p)   (a call only for Unguarded)
    ///   X(r):       ret r
    /// ```
    fn call_loop(it: &mut Interner, at: CallAt) -> Function {
        let print = it.intern("print");
        let mut b = Builder::new(it.intern("f"), MirType::I32);
        let n = b.add_param(MirType::I32);
        let x = b.add_param(MirType::I32);
        let y = b.add_param(MirType::I32);
        let zero = b.build(MirType::I32, Op::ConstInt(0, MirType::I32));
        let flag = b.build(MirType::I1, Op::Cmp(CmpOp::Sgt, x, zero));
        let h = b.new_block();
        let j = b.block_param(h, MirType::I32);
        let acc = b.block_param(h, MirType::I32);
        let body = b.new_block();
        let then = b.new_block();
        let els = b.new_block();
        let latch = b.new_block();
        let p = b.block_param(latch, MirType::I32);
        let exit_blk = b.new_block();
        let r = b.block_param(exit_blk, MirType::I32);
        let say = |b: &mut Builder, v: ValueId| {
            b.build_void(Op::Call {
                func: print,
                args: vec![v],
            });
        };
        b.br(h, vec![zero, y]);
        b.switch_to(h);
        let c = b.build(MirType::I1, Op::Cmp(CmpOp::Slt, j, n));
        b.cond_br(c, body, vec![], exit_blk, vec![acc]);
        b.switch_to(body);
        let g = if at == CallAt::AndOperand {
            let one = b.build(MirType::I32, Op::ConstInt(1, MirType::I32));
            let bit = b.build(MirType::I32, Op::Bin(BinOp::And, acc, one));
            let odd = b.build(MirType::I1, Op::Cmp(CmpOp::Ne, bit, zero));
            b.build(MirType::I1, Op::Bin(BinOp::And, flag, odd))
        } else {
            flag
        };
        b.cond_br(g, then, vec![], els, vec![]);
        b.switch_to(then);
        match at {
            CallAt::Arm | CallAt::BothArms | CallAt::AndOperand => say(&mut b, j),
            CallAt::Barrier => b.build_void(Op::Barrier),
            CallAt::Unguarded => {}
        }
        let three = b.build(MirType::I32, Op::ConstInt(3, MirType::I32));
        let m = b.build(MirType::I32, Op::Bin(BinOp::Mul, acc, three));
        let s = b.build(MirType::I32, Op::Bin(BinOp::Add, m, j));
        b.br(latch, vec![s]);
        b.switch_to(els);
        if at == CallAt::BothArms {
            let hundred = b.build(MirType::I32, Op::ConstInt(100, MirType::I32));
            let jh = b.build(MirType::I32, Op::Bin(BinOp::Add, j, hundred));
            say(&mut b, jh);
        }
        b.br(latch, vec![acc]);
        b.switch_to(latch);
        if at == CallAt::Unguarded {
            say(&mut b, j);
        }
        let one = b.build(MirType::I32, Op::ConstInt(1, MirType::I32));
        let j2 = b.build(MirType::I32, Op::Bin(BinOp::Add, j, one));
        b.br(h, vec![j2, p]);
        b.switch_to(exit_blk);
        b.ret(Some(r));
        b.finish()
    }

    const CALL_INPUTS: [(i128, i128, i128); 6] = [
        (0, 1, 5),
        (4, 1, 5),
        (4, -1, 5),
        (6, 3, 2),
        (6, 0, 2),
        (1, 2, 9),
    ];

    /// The positive control for the refusals below: a call in one arm of an invariant branch is
    /// unswitched, every printed line (the calls' own, in order, and each result) is unchanged, and
    /// after cleanup the function holds the call exactly once.
    #[test]
    fn a_call_confined_to_one_arm_is_unswitched_once() {
        let mut it = Interner::new();
        let mut f = call_loop(&mut it, CallAt::Arm);
        let before = run3(&f, &mut it, &CALL_INPUTS);
        assert!(LoopUnswitch.run_function(&mut f, &mut CfgAnalyses::default()));
        assert!(wukong_mir::verify::verify_function(&f).is_empty());
        assert_eq!(run3(&f, &mut it, &CALL_INPUTS), before);
        cleanup(&mut f);
        let print = it.intern("print");
        assert_eq!(
            calls_and_vectors(&f, print).0,
            1,
            "{}",
            wukong_mir::print::print_function(&f, &it)
        );
        assert_eq!(run3(&f, &mut it, &CALL_INPUTS), before);
    }

    /// A call every iteration makes would be duplicated by any copy: no rewrite.
    #[test]
    fn an_unguarded_call_still_blocks_cloning() {
        let mut it = Interner::new();
        let mut f = call_loop(&mut it, CallAt::Unguarded);
        assert!(!LoopUnswitch.run_function(&mut f, &mut CfgAnalyses::default()));
    }

    /// Calls on both arms of the branch: whichever way the copy decides, a call stays live in it.
    #[test]
    fn calls_on_both_arms_block_cloning() {
        let mut it = Interner::new();
        let mut f = call_loop(&mut it, CallAt::BothArms);
        assert!(!LoopUnswitch.run_function(&mut f, &mut CfgAnalyses::default()));
    }

    /// The branch tests `flag && odd`: unswitching on `flag` decides the branch only in one copy,
    /// so the arm is not proved dead in either -- the condition must be exactly the decided value.
    #[test]
    fn a_call_under_an_and_operand_guard_is_not_cloned() {
        let mut it = Interner::new();
        let mut f = call_loop(&mut it, CallAt::AndOperand);
        assert!(!LoopUnswitch.run_function(&mut f, &mut CfgAnalyses::default()));
        // The same loop branching on the flag alone is unswitched: the `and` is what blocks it.
        let mut g = call_loop(&mut it, CallAt::Arm);
        assert!(LoopUnswitch.run_function(&mut g, &mut CfgAnalyses::default()));
    }

    /// A barrier, unlike a call, is never cloned, even confined to one arm.
    #[test]
    fn a_barrier_still_blocks_cloning() {
        let mut it = Interner::new();
        let mut f = call_loop(&mut it, CallAt::Barrier);
        assert!(f
            .blocks
            .iter()
            .any(|b| b.insts.iter().any(|i| matches!(i.op, Op::Barrier))));
        assert!(!LoopUnswitch.run_function(&mut f, &mut CfgAnalyses::default()));
    }
}
