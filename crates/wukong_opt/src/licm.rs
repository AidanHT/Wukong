//! Loop-invariant code motion.
//!
//! Computations inside a loop whose operands do not change across iterations are hoisted to the
//! loop's preheader so they run once instead of every iteration. For kernels this is a major win:
//! the inner loop of a matmul recomputes row/column base addresses (`a[i*n + k]`) whose `i*n` term
//! is invariant in the `k` loop — LICM lifts exactly that out.
//!
//! Natural loops come from [`crate::loop_info`]. We hoist only into a loop that already has a
//! preheader — a single out-of-loop predecessor that reaches the header by an unconditional branch
//! (and therefore dominates it) — rather than synthesizing one; `loop-canon` gives every loop it can
//! one first. Only side-effect-free, non-trapping operations are moved (see `safe_to_hoist`: no
//! stores, vector kernels, allocas, barriers, integer division/remainder, or calls other than to a
//! callee summarized as speculatable), so hoisting a
//! computation onto a path that would not have executed it can never change observable behavior.
//!
//! # Loads
//!
//! A **load** is hoisted only when three separate things hold, because a load is both a memory read
//! and a potential speculation:
//!
//!  1. its address is loop-invariant (the ordinary operand test);
//!  2. nothing in the loop body can write what it reads — [`crate::alias`] is asked about every
//!     store in the body, and about whether a `call` in the body could reach the location at all
//!     (a loop with a barrier hoists no load: another thread may write between two barriers);
//!  3. moving it to the preheader does not perform a read the program would not have performed.
//!     The preheader runs even when the loop trips zero times, so an address the caller handed us
//!     may be valid only under the loop's own guard, and speculating a read through it would turn a
//!     program that ran into one that faults.
//!
//! Condition 3 is met in one of three ways:
//!
//!  * **Dereferenceable** — [`AliasInfo::is_dereferenceable`]: an address inside an `alloca` of this
//!    function at a known in-bounds offset. A stack slot is live for the whole function, so reading
//!    it early can neither fault nor observe anything new. Every `[]T`/struct/tuple local is such a
//!    slot — the fat-pointer header's `data` field, reread on every subscript, is the classic case.
//!  * **In the header** — the preheader branches unconditionally to the header, so a header load
//!    runs on every entry to the loop, zero-trip entries included: hoisting it speculates nothing.
//!  * **Guaranteed to execute, under a first-iteration guard** — the loop's only exit is its header
//!    test, it has no inner loop and no `return`, and the load's block dominates every latch. Then
//!    every iteration that passes the header test runs the load, so the load happens if and only if
//!    the *first* header test passes. That test is recomputed in the preheader on the values the
//!    preheader passes in (`i < n` becomes `start < n`), and the hoisted load reads through
//!    `select(first_test, addr, dummy)`, where `dummy` is a private stack slot of the load's type
//!    ([`Dummies`]): on a zero-trip entry it reads that slot instead, and the value is never used,
//!    because every use sits in a block only a passing iteration reaches. When the first test is a
//!    comparison of constants that holds (`for k in 0..64`), the body provably runs and the load is
//!    hoisted as is, with no guard. This is LLVM's guaranteed-to-execute rule without the loop
//!    rotation it relies on (`loop-canon` documents why loops here stay top-tested), and it is what
//!    lifts `w[0]` out of a read-only loop such as `s = s + a[i] * w[0]` when `w` is a parameter. (A
//!    loop that also stores through another parameter keeps the load: two parameters may alias, so
//!    that store may write `w[0]`, and only a `noalias` fact says otherwise.) It is not used for an
//!    address inside a local `alloca` (selecting it against the dummy would publish the slot to the
//!    alias analysis); those rely on the first rule.
//!
//! The *legality* argument for "the loop's only exit is the header test" is that once the first test
//! passes, every path from the header either reaches the latch — passing every block that dominates
//! it — or leaves the loop, and leaving is only possible at the header. An inner loop could spin
//! forever before reaching the load, so a loop with one is refused.

use std::sync::Arc;

use crate::alias::{type_bytes, AliasInfo, Prov};
use crate::effects::CallEffects;
use crate::fxhash::{FxHashMap, FxHashSet};
use crate::loop_info::{self, NaturalLoop};

use wukong_mir::{AddrSpace, CmpOp, Function, Inst, MirType, Op, Terminator, ValueId};

use crate::{cfg, each_op_use, CfgAnalyses, Pass};

pub struct Licm;

impl Pass for Licm {
    fn name(&self) -> &'static str {
        "licm"
    }

    fn run_function(&self, f: &mut Function, cache: &mut CfgAnalyses) -> bool {
        // Skip the reachability DFS when the CFG is already fully reachable (the steady state);
        // `all_reachable` is free off the cached rpo `idoms` needs anyway.
        if !cache.all_reachable(f) {
            cfg::prune_unreachable(f);
            cache.invalidate();
        }
        if f.blocks.len() < 2 {
            return false;
        }
        let alias = cache.alias(f);
        let effects = cache.call_effects();
        // `idom`/`preds` borrow the cache; `hoist_from_loop` mutates `f` (a different object), so
        // the two coexist for the whole loop. Hoisting never changes the block graph.
        let (idom, preds) = cache.idoms_and_preds(f);
        let forest = loop_info::structure_with(f, idom, preds);

        let mut changed = false;
        let mut dummies = Dummies::default();
        // Loops come ascending by header id, a deterministic order (M12): the order in which loops
        // are processed can change what one pass hoists, e.g. across nested loops.
        for l in &forest.loops {
            let Some(preheader) = l.preheader else {
                continue;
            };
            let body: FxHashSet<u32> = l.blocks.iter().map(|b| b.0).collect();
            let mem = LoopMem::of(f, &body, effects.clone());
            let spec = Spec::of(f, l, &body, idom);
            changed |= hoist_from_loop(f, &body, preheader.0, &alias, &mem, &spec, &mut dummies);
        }
        dummies.install(f);
        changed
    }
}

/// The zero-trip targets of guarded accesses: one private stack slot per accessed type, shared by
/// every guarded access of that type in one pass run. A guarded access touches its dummy only on an
/// entry where its loop does not run, and nothing uses what it reads there, so one slot per type
/// serves them all — and a function gains one `alloca` per type rather than one per hoisted load
/// (the interpreter keeps every slot an `alloca` reserves, so that is memory per call). A dummy has
/// the accessed type, so the interpreter's default in it is a value of that type, which is what any
/// pure instruction hoisted after the load computes with on a zero-trip entry.
#[derive(Default)]
pub(crate) struct Dummies {
    insts: Vec<Inst>,
    by_ty: FxHashMap<MirType, ValueId>,
}

impl Dummies {
    /// The dummy slot for accesses of type `ty`, minted on first request.
    pub(crate) fn slot(&mut self, f: &mut Function, ty: &MirType) -> ValueId {
        if let Some(&v) = self.by_ty.get(ty) {
            return v;
        }
        let v = fresh_value(f, MirType::Ptr);
        self.insts.push(Inst {
            result: Some(v),
            op: Op::Alloca {
                ty: ty.clone(),
                space: AddrSpace::Local,
                align: 0,
            },
        });
        self.by_ty.insert(ty.clone(), v);
        v
    }

    /// Put the minted slots at the top of the entry block, where they dominate every use.
    pub(crate) fn install(mut self, f: &mut Function) {
        if self.insts.is_empty() {
            return;
        }
        let entry = f.entry.0 as usize;
        self.insts.append(&mut f.blocks[entry].insts);
        f.blocks[entry].insts = self.insts;
    }
}

/// Everything in one loop body that can write memory, summarized once so the hoist test does not
/// rescan the body per candidate. Stores are never hoisted, and the only calls that are (a
/// speculatable callee's) write nothing and were never recorded here, so this stays valid across
/// every round of `hoist_from_loop`.
pub(crate) struct LoopMem {
    /// `(destination, bytes written)` for every store in the body.
    pub(crate) stores: Vec<(ValueId, u32)>,
    /// Does the body contain a call or a vector-kernel call? Either may write through any pointer
    /// it was handed, so only a stack slot that never escaped this function survives one.
    pub(crate) has_call: bool,
    /// Does the body contain a barrier? Another thread may write shared memory between two.
    pub(crate) has_barrier: bool,
    /// Call summaries: a call summarized as side-effect free writes nothing, and a speculatable one
    /// may itself be hoisted (see the `effects` module). `None` treats every call as a writer —
    /// what `promote` passes, because a summarized callee may still *read* the location it keeps
    /// in a register, and the pending register value is invisible to that read.
    effects: Option<Arc<CallEffects>>,
}

impl LoopMem {
    pub(crate) fn of(
        f: &Function,
        body: &FxHashSet<u32>,
        effects: Option<Arc<CallEffects>>,
    ) -> LoopMem {
        let mut mem = LoopMem {
            stores: Vec::new(),
            has_call: false,
            has_barrier: false,
            effects,
        };
        let mut blocks: Vec<u32> = body.iter().copied().collect();
        blocks.sort_unstable();
        for blk in blocks {
            for inst in &f.blocks[blk as usize].insts {
                match &inst.op {
                    Op::Store { ptr, value } => mem
                        .stores
                        .push((*ptr, type_bytes(f.value_type(*value)).unwrap_or(u32::MAX))),
                    Op::Call { func, .. } if mem.writes_nothing(*func) => {}
                    // Any other writer (`Op::effects`): a call of any kind, or an atomic add, which
                    // writes its (escaped) location and may race other threads' adds — opaque.
                    op if op.may_write_memory() => mem.has_call = true,
                    Op::Barrier => mem.has_barrier = true,
                    _ => {}
                }
            }
        }
        mem
    }

    /// Can anything in the loop write the `bytes` bytes at `ptr`?
    pub(crate) fn may_write(&self, alias: &AliasInfo, ptr: ValueId, bytes: u32) -> bool {
        self.has_barrier
            || (self.has_call && !alias.is_private_stack(ptr))
            || self
                .stores
                .iter()
                .any(|&(q, qb)| alias.may_alias_sized(ptr, bytes, q, qb))
    }

    fn writes_nothing(&self, func: wukong_span::Symbol) -> bool {
        self.effects.as_ref().is_some_and(|e| e.get(func).is_some())
    }
}

/// What may be read early in one loop's preheader (condition 3 of the module note).
pub(crate) struct Spec {
    pub(crate) header: u32,
    /// Blocks other than the header that run on every iteration passing the header test. Empty when
    /// the loop's shape does not support the argument (several exits, an inner loop, a `return`).
    pub(crate) every_iteration: FxHashSet<u32>,
    /// The header's continue test recomputed on the values the preheader passes in, when it can be:
    /// `(predicate, lhs, rhs)`, all available at the end of the preheader.
    pub(crate) first_test: Option<(CmpOp, ValueId, ValueId)>,
    /// The first test is a comparison of two integer constants that holds (`for k in 0..64`): the
    /// body provably runs at least once whenever the loop is entered, so an access that runs on
    /// every iteration is not speculative in the preheader and needs no guard.
    pub(crate) runs_once: bool,
}

impl Spec {
    pub(crate) fn of(f: &Function, l: &NaturalLoop, body: &FxHashSet<u32>, idom: &[u32]) -> Spec {
        let header = l.header.0;
        let mut spec = Spec {
            header,
            every_iteration: FxHashSet::default(),
            first_test: None,
            runs_once: false,
        };
        let single_test = l.exits.len() == 1
            && l.exits[0].0 == l.header
            && l.abnormal_exits.is_empty()
            && l.children.is_empty()
            && !l.latches.is_empty();
        let Some(preheader) = l.preheader else {
            return spec;
        };
        if !single_test {
            return spec;
        }
        for b in &l.blocks {
            if b.0 != header && l.latches.iter().all(|lt| dominates(b.0, lt.0, idom)) {
                spec.every_iteration.insert(b.0);
            }
        }
        spec.first_test = first_test(f, l, body, preheader.0);
        spec.runs_once = spec.first_test.is_some_and(|(pred, a, b)| {
            let (Some((x, xt)), Some((y, yt))) = (
                const_before(f, a, preheader.0, idom),
                const_before(f, b, preheader.0, idom),
            ) else {
                return false;
            };
            xt == yt && eval_int_cmp(pred, x, y, xt) == Some(true)
        });
        spec
    }
}

/// The integer constant `v` is, when its definition lies in `from` or a block dominating it (the
/// only places a value available at the end of `from` can be defined).
fn const_before<'f>(
    f: &'f Function,
    v: ValueId,
    from: u32,
    idom: &[u32],
) -> Option<(i128, &'f MirType)> {
    let mut b = from;
    loop {
        for inst in &f.blocks[b as usize].insts {
            if inst.result == Some(v) {
                return match &inst.op {
                    Op::ConstInt(k, ty) => Some((*k, ty)),
                    _ => None,
                };
            }
        }
        let up = *idom.get(b as usize)?;
        if up == b || up == u32::MAX {
            return None;
        }
        b = up;
    }
}

/// An integer comparison of two constants of type `ty`, read the way the backends read them:
/// masked to the width, sign-extended for a signed predicate.
fn eval_int_cmp(pred: CmpOp, a: i128, b: i128, ty: &MirType) -> Option<bool> {
    use CmpOp::*;
    let bits: u32 = match ty {
        MirType::I8 => 8,
        MirType::I16 => 16,
        MirType::I32 => 32,
        MirType::I64 => 64,
        _ => return None,
    };
    let shift = 128 - bits;
    let (sa, sb) = ((a << shift) >> shift, (b << shift) >> shift);
    let mask = (1i128 << bits) - 1;
    let (ua, ub) = (a & mask, b & mask);
    Some(match pred {
        Eq => ua == ub,
        Ne => ua != ub,
        Slt => sa < sb,
        Sle => sa <= sb,
        Sgt => sa > sb,
        Sge => sa >= sb,
        Ult => ua < ub,
        Ule => ua <= ub,
        Ugt => ua > ub,
        Uge => ua >= ub,
        Foeq | Fone | Folt | Fole | Fogt | Foge => return None,
    })
}

/// The header's exit test as it evaluates on the first visit, expressed on values available at the
/// end of the preheader, oriented so that `true` means "the body runs".
fn first_test(
    f: &Function,
    l: &NaturalLoop,
    body: &FxHashSet<u32>,
    preheader: u32,
) -> Option<(CmpOp, ValueId, ValueId)> {
    let hb = &f.blocks[l.header.0 as usize];
    let Terminator::CondBr {
        cond,
        then_blk,
        else_blk,
        ..
    } = &hb.term
    else {
        return None;
    };
    let continue_on_true = match (body.contains(&then_blk.0), body.contains(&else_blk.0)) {
        (true, false) => true,
        (false, true) => false,
        _ => return None,
    };
    let Some(Op::Cmp(pred, a, b)) = hb
        .insts
        .iter()
        .find(|i| i.result == Some(*cond))
        .map(|i| &i.op)
    else {
        return None;
    };
    if f.value_type(*cond).is_vector() {
        return None;
    }
    let Terminator::Br { args, .. } = &f.blocks[preheader as usize].term else {
        return None;
    };
    let mut defined_in_loop: FxHashSet<u32> = FxHashSet::default();
    for blk in body {
        let bb = &f.blocks[*blk as usize];
        defined_in_loop.extend(bb.params.iter().map(|p| p.0));
        defined_in_loop.extend(bb.insts.iter().filter_map(|i| i.result.map(|r| r.0)));
    }
    // A header parameter's first value is what the preheader passes; anything else must already be
    // computed outside the loop.
    let entry_value = |v: ValueId| -> Option<ValueId> {
        if let Some(k) = hb.params.iter().position(|p| *p == v) {
            return args.get(k).copied();
        }
        (!defined_in_loop.contains(&v.0)).then_some(v)
    };
    let (a, b) = (entry_value(*a)?, entry_value(*b)?);
    let pred = if continue_on_true {
        *pred
    } else {
        negate_int(*pred)?
    };
    Some((pred, a, b))
}

/// The negation of an integer predicate. Float predicates have none: every ordered compare is false
/// on a NaN, so `!(a < b)` is not `a >= b`.
fn negate_int(p: CmpOp) -> Option<CmpOp> {
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

/// How a hoisted instruction is moved.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Hoist {
    /// As is.
    Plain,
    /// A load whose address is replaced by `select(first_test, addr, dummy)`.
    Guarded,
}

/// May a `load` of `ty` at `ptr` in block `blk` be lifted out of the loop, and how?
fn load_hoist(
    alias: &AliasInfo,
    ptr: ValueId,
    ty: &MirType,
    blk: u32,
    mem: &LoopMem,
    spec: &Spec,
) -> Option<Hoist> {
    let bytes = type_bytes(ty).unwrap_or(u32::MAX);
    if mem.may_write(alias, ptr, bytes) {
        return None;
    }
    if alias.is_dereferenceable(ptr, bytes)
        || blk == spec.header
        || (spec.runs_once && spec.every_iteration.contains(&blk))
    {
        return Some(Hoist::Plain);
    }
    let guardable = spec.every_iteration.contains(&blk)
        && spec.first_test.is_some()
        && !ty.is_vector()
        && !matches!(alias.prov(ptr), Prov::Alloca(_));
    guardable.then_some(Hoist::Guarded)
}

/// Does `a` dominate `b`? Walks the immediate-dominator chain from `b` up to the entry.
pub(crate) fn dominates(a: u32, b: u32, idom: &[u32]) -> bool {
    let mut x = b;
    loop {
        if x == a {
            return true;
        }
        let Some(&id) = idom.get(x as usize) else {
            return false;
        };
        if id == x || id == u32::MAX {
            return false; // reached the entry (or an unreachable block)
        }
        x = id;
    }
}

/// May this operation be speculatively executed in the preheader (no side effects, cannot trap)?
///
/// `Op::Load` is decided separately by [`load_hoist`], which needs the loop's memory effects and the
/// alias analysis; everything here is judged on the opcode alone, plus the callee's summary for a
/// `call`.
fn safe_to_hoist(op: &Op, mem: &LoopMem) -> bool {
    match op {
        // A speculatable callee always returns, reads nothing and cannot trap: with invariant
        // arguments its result is invariant and computing it early is unobservable.
        Op::Call { func, .. } => mem.effects.as_ref().is_some_and(|e| e.hoistable(*func)),
        // Everything else only when it has no effect at all (`Op::effects`): not a memory access,
        // an allocation, a barrier, any other call, nor an integer division or remainder, which
        // traps on a zero divisor.
        op => op.is_pure(),
    }
}

/// Mint a value of type `ty`.
pub(crate) fn fresh_value(f: &mut Function, ty: MirType) -> ValueId {
    let v = ValueId(f.value_types.len() as u32);
    f.value_types.push(ty);
    v
}

fn hoist_from_loop(
    f: &mut Function,
    body: &FxHashSet<u32>,
    preheader: u32,
    alias: &AliasInfo,
    mem: &LoopMem,
    spec: &Spec,
    dummies: &mut Dummies,
) -> bool {
    // Iterate the loop body in a deterministic (block-id) order. The order in which hoisted
    // instructions are appended to the preheader must not depend on `HashSet` iteration order —
    // that order is randomized per run, so using it directly emits nondeterministic MIR (M12). The
    // hoisted instructions in a round are mutually independent, so any fixed order is equally valid.
    let mut body_blocks: Vec<u32> = body.iter().copied().collect();
    body_blocks.sort_unstable();

    // Values defined inside the loop (block parameters and instruction results).
    let mut defined_in_loop: FxHashSet<u32> = FxHashSet::default();
    for &blk in &body_blocks {
        let b = &f.blocks[blk as usize];
        for p in &b.params {
            defined_in_loop.insert(p.0);
        }
        for inst in &b.insts {
            if let Some(r) = inst.result {
                defined_in_loop.insert(r.0);
            }
        }
    }

    // An operand is available in the preheader if it is not defined in the loop, or it has already
    // been hoisted there.
    let mut hoisted: FxHashSet<u32> = FxHashSet::default();
    let mut moved_ops: Vec<Inst> = Vec::new();
    // The first-iteration test, materialized in the preheader on the first guarded load.
    let mut guard: Option<ValueId> = None;

    loop {
        // Read phase: which still-in-loop instructions are now invariant?
        let mut found: FxHashMap<u32, Hoist> = FxHashMap::default();
        for &blk in &body_blocks {
            for inst in &f.blocks[blk as usize].insts {
                let Some(r) = inst.result else { continue };
                let how = match &inst.op {
                    Op::Load(p, ty) => load_hoist(alias, *p, ty, blk, mem, spec),
                    op => safe_to_hoist(op, mem).then_some(Hoist::Plain),
                };
                let Some(how) = how else { continue };
                let mut invariant = true;
                each_op_use(&inst.op, &mut |v| {
                    if defined_in_loop.contains(&v.0) && !hoisted.contains(&v.0) {
                        invariant = false;
                    }
                });
                if invariant {
                    found.insert(r.0, how);
                }
            }
        }
        if found.is_empty() {
            break;
        }
        hoisted.extend(found.keys().copied());

        // Mutate phase: pull the newly invariant instructions out of their loop blocks. All their
        // operands were available *before* this round, so there are no intra-round dependencies and
        // appending them in (deterministic block-id) discovery order is valid.
        for &blk in &body_blocks {
            let insts = std::mem::take(&mut f.blocks[blk as usize].insts);
            let mut kept = Vec::with_capacity(insts.len());
            for mut inst in insts {
                let how = inst.result.and_then(|r| found.get(&r.0).copied());
                match how {
                    None => kept.push(inst),
                    Some(Hoist::Plain) => moved_ops.push(inst),
                    Some(Hoist::Guarded) => {
                        let Op::Load(p, ty) = &mut inst.op else {
                            unreachable!("only a load is hoisted under a guard");
                        };
                        let cond = match guard {
                            Some(c) => c,
                            None => {
                                let (pred, a, b) = spec
                                    .first_test
                                    .expect("a guarded hoist requires a first-iteration test");
                                let c = fresh_value(f, MirType::I1);
                                moved_ops.push(Inst {
                                    result: Some(c),
                                    op: Op::Cmp(pred, a, b),
                                });
                                guard = Some(c);
                                c
                            }
                        };
                        let dummy = dummies.slot(f, ty);
                        let sel = fresh_value(f, MirType::Ptr);
                        moved_ops.push(Inst {
                            result: Some(sel),
                            op: Op::Select(cond, *p, dummy),
                        });
                        *p = sel;
                        moved_ops.push(inst);
                    }
                }
            }
            f.blocks[blk as usize].insts = kept;
        }
    }

    if moved_ops.is_empty() {
        return false;
    }
    f.blocks[preheader as usize].insts.extend(moved_ops);
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use wukong_mir::{BinOp, Builder, CmpOp, MirType};
    use wukong_span::Interner;

    /// Build `bb0 -> bb1(header, guard) -> bb2(body) -> bb1`, running `body` inside bb2. The
    /// candidate load's address is whatever `addr` returns, built in bb0 (so it is loop-invariant).
    /// Returns the finished function and how many loads sit inside the loop.
    fn loop_fn(
        it: &mut Interner,
        name: &str,
        build: impl FnOnce(&mut Builder) -> ValueId,
        body: impl FnOnce(&mut Builder, ValueId),
    ) -> Function {
        let mut b = Builder::new(it.intern(name), MirType::Void);
        let n = b.add_param(MirType::I64);
        let addr = build(&mut b);
        let zero = b.build(MirType::I64, Op::ConstInt(0, MirType::I64));
        let one = b.build(MirType::I64, Op::ConstInt(1, MirType::I64));
        let header = b.new_block();
        let bodyb = b.new_block();
        let exit = b.new_block();
        let i = b.block_param(header, MirType::I64);
        b.br(header, vec![zero]);
        b.switch_to(header);
        let c = b.build(MirType::I1, Op::Cmp(CmpOp::Slt, i, n));
        b.cond_br(c, bodyb, vec![], exit, vec![]);
        b.switch_to(bodyb);
        body(&mut b, addr);
        let next = b.build(MirType::I64, Op::Bin(BinOp::Add, i, one));
        b.br(header, vec![next]);
        b.switch_to(exit);
        b.ret(None);
        b.finish()
    }

    fn loads_in(f: &Function, blk: usize) -> usize {
        f.blocks[blk]
            .insts
            .iter()
            .filter(|i| matches!(i.op, Op::Load(..)))
            .count()
    }

    /// The `[]T` shape: the fat-pointer header is a stack slot, so re-reading its `data` field every
    /// iteration is both invariant and safe to speculate. The element store goes through the loaded
    /// pointer — unknown provenance — but it cannot reach a slot whose address never escaped.
    #[test]
    fn a_private_stack_load_is_hoisted_out_of_the_loop() {
        let mut it = Interner::new();
        let mut f = loop_fn(
            &mut it,
            "f",
            |b| b.alloca(MirType::Array(Box::new(MirType::I8), 16)),
            |b, hdr| {
                let p = b.build(MirType::Ptr, Op::Load(hdr, MirType::Ptr));
                let x = b.build(MirType::F32, Op::Load(p, MirType::F32));
                b.build_void(Op::Store { ptr: p, value: x });
            },
        );
        assert_eq!(loads_in(&f, 2), 2);
        let mut cache = CfgAnalyses::default();
        assert!(Licm.run_function(&mut f, &mut cache));
        // The header load moved to the preheader; the element load stays (its address varies).
        assert_eq!(
            loads_in(&f, 2),
            1,
            "{}",
            wukong_mir::print::print_function(&f, &it)
        );
        assert_eq!(
            loads_in(&f, 0),
            1,
            "{}",
            wukong_mir::print::print_function(&f, &it)
        );
    }

    /// The identical loop, but the address is a *parameter*. Nothing writes it, so it is invariant,
    /// and the body block runs on every iteration that passes the header test, so each load happens
    /// exactly when the first test `0 < n` passes. Both loads — the parameter's and the one through
    /// the pointer it yields — move to the preheader, each reading through
    /// `select(0 < n, addr, dummy)` so a zero-trip entry reads a private slot instead.
    #[test]
    fn a_parameter_load_that_always_executes_is_hoisted_under_a_first_iteration_guard() {
        let mut it = Interner::new();
        let mut f = loop_fn(
            &mut it,
            "g",
            |b| b.add_param(MirType::Ptr),
            |b, hdr| {
                let p = b.build(MirType::Ptr, Op::Load(hdr, MirType::Ptr));
                let _ = b.build(MirType::F32, Op::Load(p, MirType::F32));
            },
        );
        assert_eq!(loads_in(&f, 2), 2);
        let mut cache = CfgAnalyses::default();
        assert!(Licm.run_function(&mut f, &mut cache));
        let shown = wukong_mir::print::print_function(&f, &it);
        assert!(
            wukong_mir::verify::verify_function(&f).is_empty(),
            "{:?}\n{shown}",
            wukong_mir::verify::verify_function(&f)
        );
        assert_eq!(loads_in(&f, 2), 0, "{shown}");
        assert_eq!(loads_in(&f, 0), 2, "{shown}");
        let pre = &f.blocks[0].insts;
        let selects = pre
            .iter()
            .filter(|i| matches!(i.op, Op::Select(..)))
            .count();
        assert_eq!(selects, 2, "{shown}");
        assert!(
            pre.iter().any(|i| matches!(
                i.op,
                Op::Cmp(CmpOp::Slt, a, b) if a == f.params[0] || b == f.params[0]
            )),
            "the guard must test the first iteration against `n`: {shown}"
        );
    }

    /// The same parameter load in a block only some iterations reach (an `if` in the body) is not
    /// guaranteed to execute, so it stays in the loop.
    #[test]
    fn a_conditionally_executed_parameter_load_stays_in_the_loop() {
        let mut it = Interner::new();
        let mut b = Builder::new(it.intern("h"), MirType::Void);
        let n = b.add_param(MirType::I64);
        let w = b.add_param(MirType::Ptr);
        let zero = b.build(MirType::I64, Op::ConstInt(0, MirType::I64));
        let one = b.build(MirType::I64, Op::ConstInt(1, MirType::I64));
        let header = b.new_block();
        let body = b.new_block();
        let then = b.new_block();
        let latch = b.new_block();
        let exit = b.new_block();
        let i = b.block_param(header, MirType::I64);
        b.br(header, vec![zero]);
        b.switch_to(header);
        let c = b.build(MirType::I1, Op::Cmp(CmpOp::Slt, i, n));
        b.cond_br(c, body, vec![], exit, vec![]);
        b.switch_to(body);
        let odd = b.build(MirType::I64, Op::Bin(BinOp::And, i, one));
        let take = b.build(MirType::I1, Op::Cmp(CmpOp::Eq, odd, one));
        b.cond_br(take, then, vec![], latch, vec![]);
        b.switch_to(then);
        let _ = b.build(MirType::F32, Op::Load(w, MirType::F32));
        b.br(latch, vec![]);
        b.switch_to(latch);
        let next = b.build(MirType::I64, Op::Bin(BinOp::Add, i, one));
        b.br(header, vec![next]);
        b.switch_to(exit);
        b.ret(None);
        let mut f = b.finish();
        assert!(wukong_mir::verify::verify_function(&f).is_empty());
        Licm.run_function(&mut f, &mut CfgAnalyses::default());
        assert_eq!(
            loads_in(&f, then.0 as usize),
            1,
            "{}",
            wukong_mir::print::print_function(&f, &it)
        );
    }

    /// End to end through the interpreter: `f(w, n)` sums `w[0]` over `n` iterations, and `main`
    /// calls it with `n = 3` and `n = 0`. The hoisted, guarded load must give the same answers —
    /// on the zero-trip call it reads the dummy slot and nothing uses the value.
    #[test]
    fn a_guarded_hoist_preserves_results_for_zero_and_nonzero_trips() {
        let mut it = Interner::new();
        let fname = it.intern("f");
        let mut b = Builder::new(fname, MirType::I64);
        let w = b.add_param(MirType::Ptr);
        let n = b.add_param(MirType::I64);
        let zero = b.build(MirType::I64, Op::ConstInt(0, MirType::I64));
        let one = b.build(MirType::I64, Op::ConstInt(1, MirType::I64));
        let header = b.new_block();
        let body = b.new_block();
        let exit = b.new_block();
        let i = b.block_param(header, MirType::I64);
        let acc = b.block_param(header, MirType::I64);
        let out = b.block_param(exit, MirType::I64);
        b.br(header, vec![zero, zero]);
        b.switch_to(header);
        let c = b.build(MirType::I1, Op::Cmp(CmpOp::Slt, i, n));
        b.cond_br(c, body, vec![], exit, vec![acc]);
        b.switch_to(body);
        let w0 = b.build(MirType::I64, Op::Load(w, MirType::I64));
        let acc2 = b.build(MirType::I64, Op::Bin(BinOp::Add, acc, w0));
        let next = b.build(MirType::I64, Op::Bin(BinOp::Add, i, one));
        b.br(header, vec![next, acc2]);
        b.switch_to(exit);
        b.ret(Some(out));
        let mut f = b.finish();

        let mut m = Builder::new(it.intern("main"), MirType::I64);
        let slot = m.alloca(MirType::I64);
        let seven = m.build(MirType::I64, Op::ConstInt(7, MirType::I64));
        m.build_void(Op::Store {
            ptr: slot,
            value: seven,
        });
        let three = m.build(MirType::I64, Op::ConstInt(3, MirType::I64));
        let none = m.build(MirType::I64, Op::ConstInt(0, MirType::I64));
        let r3 = m.build(
            MirType::I64,
            Op::Call {
                func: fname,
                args: vec![slot, three],
            },
        );
        let r0 = m.build(
            MirType::I64,
            Op::Call {
                func: fname,
                args: vec![slot, none],
            },
        );
        let hundred = m.build(MirType::I64, Op::ConstInt(100, MirType::I64));
        let hi = m.build(MirType::I64, Op::Bin(BinOp::Mul, r3, hundred));
        let r = m.build(MirType::I64, Op::Bin(BinOp::Add, hi, r0));
        m.ret(Some(r));
        let main = m.finish();

        let run = |f: &Function| {
            let mut p = wukong_mir::Program::new();
            p.funcs.push(main.clone());
            p.funcs.push(f.clone());
            assert!(wukong_mir::verify::verify_program(&p).is_empty());
            wukong_interp::run(&p, main.name, &it).unwrap()
        };
        // 3 · 7 = 21, then 0 for the zero-trip call: 21·100 + 0.
        assert_eq!(run(&f), 2100);
        assert!(Licm.run_function(&mut f, &mut CfgAnalyses::default()));
        assert_eq!(loads_in(&f, body.0 as usize), 0);
        assert_eq!(run(&f), 2100);
    }

    /// `for i in 0..8 { s = s + w[0] }`: the first test `0 < 8` is a true constant, so the body
    /// provably runs and the parameter load moves to the preheader as it is — no guard.
    #[test]
    fn a_constant_trip_loop_hoists_a_parameter_load_without_a_guard() {
        let mut it = Interner::new();
        let mut b = Builder::new(it.intern("c"), MirType::I64);
        let w = b.add_param(MirType::Ptr);
        let zero = b.build(MirType::I64, Op::ConstInt(0, MirType::I64));
        let eight = b.build(MirType::I64, Op::ConstInt(8, MirType::I64));
        let one = b.build(MirType::I64, Op::ConstInt(1, MirType::I64));
        let header = b.new_block();
        let body = b.new_block();
        let exit = b.new_block();
        let i = b.block_param(header, MirType::I64);
        let s = b.block_param(header, MirType::I64);
        let out = b.block_param(exit, MirType::I64);
        b.br(header, vec![zero, zero]);
        b.switch_to(header);
        let c = b.build(MirType::I1, Op::Cmp(CmpOp::Slt, i, eight));
        b.cond_br(c, body, vec![], exit, vec![s]);
        b.switch_to(body);
        let w0 = b.build(MirType::I64, Op::Load(w, MirType::I64));
        let s2 = b.build(MirType::I64, Op::Bin(BinOp::Add, s, w0));
        let next = b.build(MirType::I64, Op::Bin(BinOp::Add, i, one));
        b.br(header, vec![next, s2]);
        b.switch_to(exit);
        b.ret(Some(out));
        let mut f = b.finish();
        assert!(Licm.run_function(&mut f, &mut CfgAnalyses::default()));
        let shown = wukong_mir::print::print_function(&f, &it);
        assert!(
            wukong_mir::verify::verify_function(&f).is_empty(),
            "{shown}"
        );
        assert_eq!(loads_in(&f, body.0 as usize), 0, "{shown}");
        assert!(
            !f.blocks
                .iter()
                .flat_map(|b| &b.insts)
                .any(|i| matches!(i.op, Op::Select(..) | Op::Alloca { .. })),
            "{shown}"
        );
    }

    /// `s = s + w[0] * v[0]` over two parameters: both loads are hoisted behind the guard, and the
    /// product of the two hoisted values with them. Loads of one type share one zero-trip dummy, so
    /// the function gains a single `alloca`, not one per load.
    #[test]
    fn guarded_loads_of_one_type_share_a_dummy_slot() {
        let mut it = Interner::new();
        let mut b = Builder::new(it.intern("s"), MirType::F32);
        let n = b.add_param(MirType::I64);
        let w = b.add_param(MirType::Ptr);
        let v = b.add_param(MirType::Ptr);
        let zero = b.build(MirType::I64, Op::ConstInt(0, MirType::I64));
        let one = b.build(MirType::I64, Op::ConstInt(1, MirType::I64));
        let fzero = b.build(MirType::F32, Op::ConstFloat(0.0, MirType::F32));
        let header = b.new_block();
        let body = b.new_block();
        let exit = b.new_block();
        let i = b.block_param(header, MirType::I64);
        let s = b.block_param(header, MirType::F32);
        let out = b.block_param(exit, MirType::F32);
        b.br(header, vec![zero, fzero]);
        b.switch_to(header);
        let c = b.build(MirType::I1, Op::Cmp(CmpOp::Slt, i, n));
        b.cond_br(c, body, vec![], exit, vec![s]);
        b.switch_to(body);
        let x = b.build(MirType::F32, Op::Load(w, MirType::F32));
        let y = b.build(MirType::F32, Op::Load(v, MirType::F32));
        let xy = b.build(MirType::F32, Op::Bin(BinOp::FMul, x, y));
        let s2 = b.build(MirType::F32, Op::Bin(BinOp::FAdd, s, xy));
        let next = b.build(MirType::I64, Op::Bin(BinOp::Add, i, one));
        b.br(header, vec![next, s2]);
        b.switch_to(exit);
        b.ret(Some(out));
        let mut f = b.finish();
        assert!(wukong_mir::verify::verify_function(&f).is_empty());
        assert!(Licm.run_function(&mut f, &mut CfgAnalyses::default()));
        let shown = wukong_mir::print::print_function(&f, &it);
        assert!(
            wukong_mir::verify::verify_function(&f).is_empty(),
            "{shown}"
        );
        assert_eq!(loads_in(&f, body.0 as usize), 0, "{shown}");
        let count = |pred: fn(&Op) -> bool| {
            f.blocks
                .iter()
                .flat_map(|b| &b.insts)
                .filter(|i| pred(&i.op))
                .count()
        };
        assert_eq!(count(|op| matches!(op, Op::Select(..))), 2, "{shown}");
        assert_eq!(count(|op| matches!(op, Op::Alloca { .. })), 1, "{shown}");
    }

    /// A store in the loop that may alias the slot keeps its load inside.
    #[test]
    fn a_may_aliasing_store_in_the_loop_pins_the_load() {
        let mut it = Interner::new();
        let mut f = loop_fn(
            &mut it,
            "h",
            |b| b.alloca(MirType::Array(Box::new(MirType::I8), 16)),
            |b, slot| {
                let x = b.build(MirType::I64, Op::Load(slot, MirType::I64));
                b.build_void(Op::Store {
                    ptr: slot,
                    value: x,
                });
            },
        );
        assert_eq!(loads_in(&f, 2), 1);
        let mut cache = CfgAnalyses::default();
        Licm.run_function(&mut f, &mut cache);
        assert_eq!(
            loads_in(&f, 2),
            1,
            "{}",
            wukong_mir::print::print_function(&f, &it)
        );
    }

    /// A call in the loop is opaque, but it still cannot reach a slot this function never published.
    /// The same load behind an escaped slot must stay.
    #[test]
    fn a_call_in_the_loop_pins_only_reachable_loads() {
        let mut it = Interner::new();
        let sink = it.intern("sink");
        let mut f = loop_fn(
            &mut it,
            "k",
            |b| b.alloca(MirType::Array(Box::new(MirType::I8), 16)),
            |b, slot| {
                let _ = b.build(MirType::I64, Op::Load(slot, MirType::I64));
                b.build_void(Op::Call {
                    func: sink,
                    args: vec![],
                });
            },
        );
        let mut cache = CfgAnalyses::default();
        assert!(Licm.run_function(&mut f, &mut cache));
        assert_eq!(
            loads_in(&f, 2),
            0,
            "{}",
            wukong_mir::print::print_function(&f, &it)
        );

        let mut escaped = loop_fn(
            &mut it,
            "k2",
            |b| b.alloca(MirType::Array(Box::new(MirType::I8), 16)),
            |b, slot| {
                let _ = b.build(MirType::I64, Op::Load(slot, MirType::I64));
                b.build_void(Op::Call {
                    func: sink,
                    args: vec![slot],
                });
            },
        );
        let mut cache = CfgAnalyses::default();
        Licm.run_function(&mut escaped, &mut cache);
        assert_eq!(
            loads_in(&escaped, 2),
            1,
            "{}",
            wukong_mir::print::print_function(&escaped, &it)
        );
    }
}
