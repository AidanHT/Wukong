//! Dead-store elimination: overwritten stores, stores no path can observe, and stores into unread
//! private stack slots.
//!
//! The dual of CSE's load forwarding. A store is **dead** when no execution can read the bytes it
//! wrote before something else replaces them or they go out of scope. Three proofs feed one deletion
//! sweep; each shows the store is never the *last* write before any read of its bytes, so deleting
//! all of them together is sound (removing a store that is never last cannot change which store is).
//!
//! 1. **Unread private slots.** A local `alloca` whose address never escapes and that no load
//!    anywhere in the function reads — including through derived pointers — is write-only, so every
//!    store into it is dead. A store is only deleted when it is provably in bounds, because deletion
//!    must not hide a fault: either its offset is a constant inside the slot
//!    ([`AliasInfo::is_dereferenceable`]), or it is indexed by the induction variable of a counted
//!    loop whose whole iteration space stays inside the slot in **both** memory models (native bytes
//!    and the interpreter's leaf slots) — the shape of a fill loop and of a fused loop's intermediate
//!    buffer once CSE has forwarded its in-iteration read.
//! 2. **Overwrite in the same block.** A later store to the same location — same bytes, same type —
//!    with no intervening read that could observe the first. Locations are matched **must-alias**,
//!    not by SSA identity: two `gep`s that land on one constant offset of one base object
//!    ([`AliasInfo::base_and_offset`]) are one location.
//! 3. **Across blocks.** For a store to a *stable* location — a constant, in-bounds offset of a local
//!    slot allocated once in the entry block, or a constant offset of a pointer parameter — every
//!    path forward from the store is searched: it is dead when each path reaches a same-location,
//!    same-type store, or (for a slot no callee can name) the end of the function, before any load
//!    that may alias it, any call that could read it, or a barrier. A path that loops back to the
//!    store itself is an overwrite, which is what retires a loop's private scratch writes at the end
//!    of their lifetime. A parameter's bytes outlive the function, so for one only an overwrite
//!    counts.
//!
//! Which reads observe which stores is decided by [`crate::alias`]: a `load` observes only the
//! stores it `may_alias`; a `call` (or vector-kernel call, which reads through the pointers it was
//! given) observes everything except a stack slot of this function whose address never escaped.
//! Shared storage and functions with barriers are excluded from the lifetime proofs; MIR has no
//! volatile load/store variant. DCE then removes unused addresses and slots, and `loop_delete`
//! removes a loop this pass has emptied.

use crate::alias::{type_bytes, AliasInfo, Prov};
use crate::fxhash::FxHashMap;
use crate::loop_delete::const_trip_is_exact;
use crate::loop_info::{self, AccessKind, AddrForm, Coeff, MemBase, Stages, TripCount};
use crate::sroa::{const_index, slot_count};

use wukong_mir::{AddrSpace, BlockId, Function, MirType, Op, Terminator, ValueId};
use wukong_span::Symbol;

use crate::{cfg, CfgAnalyses, Pass};

pub struct Dse;

/// Most blocks one cross-block search may enter before it gives up and keeps the store.
const MAX_PATH_BLOCKS: u32 = 64;

/// Most cross-block searches per pass run.
const MAX_PATH_QUERIES: usize = 4096;

/// Most instructions all cross-block searches of one pass run may scan together. The pass runs on
/// every fixpoint sweep, so its worst case has to stay linear-ish in the function, not in
/// queries times blocks; an exhausted budget keeps every remaining store.
const PATH_FUEL: u64 = 1 << 18;

/// Where a store writes, canonicalized so two spellings of one address meet.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum Loc {
    /// A constant byte offset into the `alloca` with this result id.
    Slot(u32, i64),
    /// A constant byte offset from pointer parameter `i`.
    Param(u32, i64),
    /// A constant byte offset into a `.rodata` blob.
    Global(Symbol, i64),
    /// Anything else, by its SSA address value.
    Value(u32),
}

fn loc_of(alias: &AliasInfo, p: ValueId) -> Loc {
    match alias.base_and_offset(p) {
        Some((Prov::Alloca(a), off)) => Loc::Slot(a.0, off),
        Some((Prov::Param(i), off)) => Loc::Param(i, off),
        Some((Prov::Global(g), off)) => Loc::Global(g, off),
        _ => Loc::Value(p.0),
    }
}

/// One tracked store destination: the location plus the type stored there. Two stores share a key
/// only when they cover exactly the same bytes, which is what makes the earlier one dead.
#[derive(Clone, PartialEq, Eq, Hash)]
struct StoreKey {
    loc: Loc,
    ty: MirType,
}

/// Indexed by alloca result, not by individual address: even a read from a disjoint offset keeps
/// the entire slot live. Unknown pointers cannot reach a private slot; any publication of its
/// address (including pointer tables passed to VecKernelCall) already disqualifies it in AliasInfo.
fn unread_private_slots(f: &Function, alias: &AliasInfo) -> Vec<bool> {
    let mut unread = vec![false; f.value_types.len()];
    for inst in f.blocks.iter().flat_map(|b| &b.insts) {
        match inst.op {
            Op::Alloca {
                space: AddrSpace::Local,
                ..
            } => {
                if let Some(v) = inst.result {
                    unread[v.0 as usize] = alias.is_private_stack(v);
                }
            }
            Op::Barrier => return vec![false; f.value_types.len()],
            _ => {}
        }
    }
    for inst in f.blocks.iter().flat_map(|b| &b.insts) {
        if let Op::Load(ptr, _) = inst.op {
            if let Prov::Alloca(a) = alias.prov(ptr) {
                unread[a.0 as usize] = false;
            }
        }
    }
    unread
}

/// Per-block deletion bitmaps, allocated only for blocks that lose a store.
struct DeadSet {
    bits: Vec<Vec<u64>>,
    any: bool,
}

impl DeadSet {
    fn new(f: &Function) -> DeadSet {
        DeadSet {
            bits: vec![Vec::new(); f.blocks.len()],
            any: false,
        }
    }

    fn mark(&mut self, f: &Function, b: usize, i: usize) {
        let words = &mut self.bits[b];
        if words.is_empty() {
            words.resize(f.blocks[b].insts.len().div_ceil(64), 0);
        }
        words[i / 64] |= 1u64 << (i % 64);
        self.any = true;
    }

    fn contains(&self, b: usize, i: usize) -> bool {
        self.bits[b]
            .get(i / 64)
            .is_some_and(|w| w & (1u64 << (i % 64)) != 0)
    }
}

impl Pass for Dse {
    fn name(&self) -> &'static str {
        "dse"
    }

    fn run_function(&self, f: &mut Function, cache: &mut CfgAnalyses) -> bool {
        let alias = cache.alias(f);
        let unread = unread_private_slots(f, &alias);
        let mut dead = DeadSet::new(f);

        // 1. Stores into never-read private slots. A dynamically indexed one waits for the loop
        //    bound proof below.
        let mut unproven: Vec<(usize, usize)> = Vec::new();
        for (bi, b) in f.blocks.iter().enumerate() {
            for (i, inst) in b.insts.iter().enumerate() {
                let Op::Store { ptr, value } = &inst.op else {
                    continue;
                };
                if !matches!(alias.prov(*ptr), Prov::Alloca(a) if unread[a.0 as usize]) {
                    continue;
                }
                let in_bounds = type_bytes(f.value_type(*value))
                    .is_some_and(|bytes| alias.is_dereferenceable(*ptr, bytes));
                if in_bounds {
                    dead.mark(f, bi, i);
                } else {
                    unproven.push((bi, i));
                }
            }
        }
        if !unproven.is_empty() && cache.all_reachable(f) {
            for (bi, i) in loop_bounded_stores(f, cache, &alias, &unproven) {
                dead.mark(f, bi, i);
            }
        }

        // 2. Overwrites within one block.
        for (bi, b) in f.blocks.iter().enumerate() {
            // location -> (index, address) of the last store to it that has not yet been read.
            let mut pending: FxHashMap<StoreKey, (usize, ValueId)> = FxHashMap::default();
            for (i, inst) in b.insts.iter().enumerate() {
                match &inst.op {
                    Op::Store { ptr, value } => {
                        if dead.contains(bi, i) {
                            continue;
                        }
                        // A store never *reads*, so it cannot make another store observable; it can
                        // only fully overwrite one, which needs the identical byte range.
                        let key = StoreKey {
                            loc: loc_of(&alias, *ptr),
                            ty: f.value_type(*value).clone(),
                        };
                        if let Some((prev, _)) = pending.insert(key, (i, *ptr)) {
                            dead.mark(f, bi, prev);
                        }
                    }
                    Op::Load(ptr, ty) => {
                        let bytes = type_bytes(ty).unwrap_or(u32::MAX);
                        pending.retain(|k, &mut (_, p)| {
                            !alias.may_alias_sized(
                                p,
                                type_bytes(&k.ty).unwrap_or(u32::MAX),
                                *ptr,
                                bytes,
                            )
                        });
                    }
                    // Any other reader: a call of any kind or an atomic add (`Op::effects`).
                    op if op.may_read_memory() => {
                        // A call may read through any pointer it was given, or any pointer reachable
                        // from one — but never a stack slot of this function that never escaped.
                        // An atomic add reads its own location, which is escaped by construction.
                        pending.retain(|_, &mut (_, p)| alias.is_private_stack(p));
                    }
                    Op::Barrier => pending.clear(),
                    _ => {}
                }
            }
        }

        // 3. Across blocks, for stores to stable locations.
        let barrier = f
            .blocks
            .iter()
            .flat_map(|b| &b.insts)
            .any(|i| matches!(i.op, Op::Barrier));
        if !barrier {
            let once = entry_local_slots(f);
            let mut search = PathSearch {
                stamp: vec![0; f.blocks.len()],
                generation: 0,
                fuel: PATH_FUEL,
            };
            let mut queries = 0usize;
            'blocks: for (bi, b) in f.blocks.iter().enumerate() {
                for (i, inst) in b.insts.iter().enumerate() {
                    let Op::Store { ptr, value } = &inst.op else {
                        continue;
                    };
                    if dead.contains(bi, i) {
                        continue;
                    }
                    let ty = f.value_type(*value);
                    let Some(bytes) = type_bytes(ty) else {
                        continue;
                    };
                    let loc = loc_of(&alias, *ptr);
                    // `scoped`: the bytes end with this call, so the function's end retires them.
                    let scoped = match loc {
                        Loc::Slot(a, _) => {
                            if !once.get(a as usize).copied().unwrap_or(false)
                                || !alias.is_dereferenceable(*ptr, bytes)
                            {
                                continue;
                            }
                            alias.is_private_stack(*ptr)
                        }
                        Loc::Param(..) => false,
                        Loc::Global(..) | Loc::Value(_) => continue,
                    };
                    if queries == MAX_PATH_QUERIES {
                        break 'blocks;
                    }
                    queries += 1;
                    let key = StoreKey {
                        loc,
                        ty: ty.clone(),
                    };
                    if search.dead_on_every_path(f, &alias, (bi, i), &key, *ptr, bytes, scoped) {
                        dead.mark(f, bi, i);
                    }
                }
            }
        }

        if !dead.any {
            return false;
        }
        for (bi, b) in f.blocks.iter_mut().enumerate() {
            let words = &dead.bits[bi];
            if words.is_empty() {
                continue;
            }
            let mut i = 0;
            b.insts.retain(|_| {
                let keep = words[i / 64] & (1u64 << (i % 64)) == 0;
                i += 1;
                keep
            });
        }
        true
    }
}

/// For each value id, whether it is a `Local` `alloca` in the entry block — a slot allocated exactly
/// once per call, so one constant offset into it names the same bytes on every path.
fn entry_local_slots(f: &Function) -> Vec<bool> {
    let mut once = vec![false; f.value_types.len()];
    for inst in &f.blocks[f.entry.0 as usize].insts {
        if let (
            Some(r),
            Op::Alloca {
                space: AddrSpace::Local,
                ..
            },
        ) = (inst.result, &inst.op)
        {
            once[r.0 as usize] = true;
        }
    }
    once
}

/// The forward path search behind proof 3, with a generation-stamped visited set shared by every
/// query of one pass run.
struct PathSearch {
    stamp: Vec<u32>,
    generation: u32,
    /// Instructions left to scan this run (see [`PATH_FUEL`]).
    fuel: u64,
}

impl PathSearch {
    /// Does every path from just after the store at `start` reach an overwrite of `key` (or, when
    /// `scoped`, the end of the function) before anything that may read `bytes` bytes at `ptr`?
    #[allow(clippy::too_many_arguments)]
    fn dead_on_every_path(
        &mut self,
        f: &Function,
        alias: &AliasInfo,
        start: (usize, usize),
        key: &StoreKey,
        ptr: ValueId,
        bytes: u32,
        scoped: bool,
    ) -> bool {
        self.generation += 1;
        let gen = self.generation;
        let mut budget = MAX_PATH_BLOCKS;
        let mut work: Vec<(usize, usize)> = vec![(start.0, start.1 + 1)];
        while let Some((b, from)) = work.pop() {
            let blk = &f.blocks[b];
            let mut overwritten = false;
            for inst in &blk.insts[from.min(blk.insts.len())..] {
                if self.fuel == 0 {
                    return false;
                }
                self.fuel -= 1;
                match &inst.op {
                    Op::Store { ptr: q, value } => {
                        if loc_of(alias, *q) == key.loc && *f.value_type(*value) == key.ty {
                            overwritten = true;
                            break;
                        }
                    }
                    Op::Load(q, ty) => {
                        let qb = type_bytes(ty).unwrap_or(u32::MAX);
                        if alias.may_alias_sized(ptr, bytes, *q, qb) {
                            return false;
                        }
                    }
                    op if op.may_read_memory() => {
                        if !alias.is_private_stack(ptr) {
                            return false;
                        }
                    }
                    Op::Barrier => return false,
                    _ => {}
                }
            }
            if overwritten {
                continue;
            }
            match &blk.term {
                Terminator::Ret(_) | Terminator::Unreachable => {
                    if !scoped {
                        return false;
                    }
                }
                t => {
                    for s in cfg::successors(t) {
                        let s = s.0 as usize;
                        if self.stamp[s] == gen {
                            continue;
                        }
                        if budget == 0 {
                            return false;
                        }
                        budget -= 1;
                        self.stamp[s] = gen;
                        work.push((s, 0));
                    }
                }
            }
        }
        true
    }
}

/// The subset of `cands` — stores into never-read private slots at a dynamic index — that are in
/// bounds on every execution: the address is `slot + (c·iv + k)·elem` for the induction variable of
/// a loop with a constant trip count, the store sits in the loop's body (the header runs once more
/// than the body, with the IV one step past the end), and both extremes of the index land inside the
/// slot in native bytes *and* in the interpreter's leaf slots.
fn loop_bounded_stores(
    f: &Function,
    cache: &mut CfgAnalyses,
    alias: &AliasInfo,
    cands: &[(usize, usize)],
) -> Vec<(usize, usize)> {
    let (idom, preds) = cache.idoms_and_preds(f);
    let mut forest = loop_info::structure_with(f, idom, preds);
    if forest.is_empty() {
        return Vec::new();
    }
    let mut konst: FxHashMap<u32, i64> = FxHashMap::default();
    let mut slot_ty: FxHashMap<u32, &MirType> = FxHashMap::default();
    for inst in f.blocks.iter().flat_map(|b| &b.insts) {
        match (inst.result, &inst.op) {
            (Some(r), Op::ConstInt(k, ty)) => {
                // Only a value whose masked and literal readings agree: that is the reading the
                // trip-count analysis used.
                if let Some(v) = const_index(*k, ty).filter(|v| i128::from(*v) == *k) {
                    konst.insert(r.0, v);
                }
            }
            (Some(r), Op::Alloca { ty, .. }) => {
                slot_ty.insert(r.0, ty);
            }
            _ => {}
        }
    }
    // Per loop, once analyzed: is its trip count a constant the induction variable reaches without
    // wrapping? Only then are the extremes below the whole iteration space.
    let mut exact: Vec<Option<bool>> = vec![None; forest.loops.len()];
    let mut out = Vec::new();
    for &(b, i) in cands {
        let Some(id) = forest.innermost_of(BlockId(b as u32)) else {
            continue;
        };
        let k = id.0 as usize;
        if exact[k].is_none() {
            loop_info::analyze_one(f, &mut forest, k, Stages::Memory);
            let l = &forest.loops[k];
            exact[k] = Some(
                l.trip
                    .as_const()
                    .is_some_and(|n| const_trip_is_exact(f, l, n)),
            );
        }
        if exact[k] != Some(true) {
            continue;
        }
        if store_in_bounds(alias, &forest.loops[k], &konst, &slot_ty, (b, i)).is_some() {
            out.push((b, i));
        }
    }
    out
}

/// See [`loop_bounded_stores`]. `Some(())` when the store is proved in bounds.
fn store_in_bounds(
    alias: &AliasInfo,
    l: &loop_info::NaturalLoop,
    konst: &FxHashMap<u32, i64>,
    slot_ty: &FxHashMap<u32, &MirType>,
    (b, i): (usize, usize),
) -> Option<()> {
    if l.header.0 as usize == b {
        return None;
    }
    let TripCount::Const(n) = l.trip else {
        return None;
    };
    let acc = l
        .accesses
        .iter()
        .find(|a| a.block.0 as usize == b && a.inst == i && a.kind == AccessKind::Store)?;
    if n == 0 {
        return Some(()); // the body never runs, so neither does the store
    }
    let AddrForm::Affine {
        base: MemBase::Outer(base),
        index,
        elem,
    } = &acc.addr
    else {
        return None;
    };
    let c = match index.coeff {
        Coeff::Zero => 0,
        Coeff::Const(c) => c,
        Coeff::Sym(..) => return None,
    };
    if !index.terms.is_empty() {
        return None;
    }
    let iv = l.primary()?;
    if !matches!(iv.ty, MirType::I32 | MirType::I64) {
        return None;
    }
    let step = iv.step.as_const()?;
    let start = konst.get(&iv.start?.0).copied()?;
    // The slot itself, at offset zero, must be the base.
    if !matches!(alias.prov(*base), Prov::Alloca(a) if a == *base) {
        return None;
    }
    let slot: &MirType = slot_ty.get(&base.0)?;

    // The induction variable takes `start + k*step` for k in 0..n in the body; its extremes must fit
    // its own width, and the index extremes an `i32`, so no narrower wraparound can hide in them.
    let n = i64::try_from(n).ok()?;
    let last = start.checked_add(step.checked_mul(n - 1)?)?;
    let fits = |x: i64| iv.ty == MirType::I64 || i32::try_from(x).is_ok();
    if !fits(start) || !fits(last) {
        return None;
    }
    let e0 = c.checked_mul(start)?.checked_add(index.konst)?;
    let e1 = c.checked_mul(last)?.checked_add(index.konst)?;
    let (lo, hi) = (e0.min(e1), e0.max(e1));
    if i32::try_from(lo).is_err() || i32::try_from(hi).is_err() {
        return None;
    }

    // Native bytes.
    let esize = i64::from(type_bytes(elem)?);
    let width = i64::from(type_bytes(&acc.value_ty)?);
    let total = i64::from(type_bytes(slot)?);
    let native = lo.checked_mul(esize)? >= 0 && hi.checked_mul(esize)?.checked_add(width)? <= total;
    // Interpreter leaf slots.
    let eslots = i64::try_from(slot_count(elem)?).ok()?;
    let span = match &acc.value_ty {
        MirType::Vec(_, lanes) => i64::from(*lanes),
        _ => 1,
    };
    let slots = i64::try_from(slot_count(slot)?).ok()?;
    let interp =
        lo.checked_mul(eslots)? >= 0 && hi.checked_mul(eslots)?.checked_add(span)? <= slots;
    (native && interp).then_some(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use wukong_mir::Builder;
    use wukong_span::Interner;

    fn count_stores(f: &Function) -> usize {
        f.blocks
            .iter()
            .flat_map(|b| b.insts.iter())
            .filter(|i| matches!(i.op, Op::Store { .. }))
            .count()
    }

    /// Two writes to one `[]T` element, with a read of a *disjoint* slot in between. The address is
    /// a pointer loaded out of the slice header, so the old alloca-only rule tracked nothing.
    #[test]
    fn a_store_through_a_loaded_pointer_is_killed_by_its_overwrite() {
        let mut it = Interner::new();
        let mut b = Builder::new(it.intern("f"), MirType::Void);
        let hdr = b.add_param(MirType::Ptr);
        let other = b.alloca(MirType::Array(Box::new(MirType::I8), 8));
        let data = b.build(MirType::Ptr, Op::Load(hdr, MirType::Ptr));
        let one = b.build(MirType::F32, Op::ConstFloat(1.0, MirType::F32));
        let two = b.build(MirType::F32, Op::ConstFloat(2.0, MirType::F32));
        b.build_void(Op::Store {
            ptr: data,
            value: one,
        });
        // Reading a private stack slot cannot observe the store above.
        let _ = b.build(MirType::F32, Op::Load(other, MirType::F32));
        b.build_void(Op::Store {
            ptr: data,
            value: two,
        });
        b.ret(None);
        let mut f = b.finish();

        assert_eq!(count_stores(&f), 2);
        let mut cache = CfgAnalyses::default();
        assert!(Dse.run_function(&mut f, &mut cache));
        assert_eq!(
            count_stores(&f),
            1,
            "{}",
            wukong_mir::print::print_function(&f, &it)
        );
    }

    /// The same pair, but the intervening read goes through a pointer that may well be the same
    /// element. The first store is observable and must stay.
    #[test]
    fn an_aliasing_read_keeps_the_first_store() {
        let mut it = Interner::new();
        let mut b = Builder::new(it.intern("g"), MirType::Void);
        let hdr = b.add_param(MirType::Ptr);
        let other = b.add_param(MirType::Ptr);
        let data = b.build(MirType::Ptr, Op::Load(hdr, MirType::Ptr));
        let one = b.build(MirType::F32, Op::ConstFloat(1.0, MirType::F32));
        let two = b.build(MirType::F32, Op::ConstFloat(2.0, MirType::F32));
        b.build_void(Op::Store {
            ptr: data,
            value: one,
        });
        let _ = b.build(MirType::F32, Op::Load(other, MirType::F32));
        b.build_void(Op::Store {
            ptr: data,
            value: two,
        });
        b.ret(None);
        let mut f = b.finish();

        let mut cache = CfgAnalyses::default();
        Dse.run_function(&mut f, &mut cache);
        assert_eq!(
            count_stores(&f),
            2,
            "{}",
            wukong_mir::print::print_function(&f, &it)
        );
    }

    /// A call may read anything it was handed, so it keeps an escaped store alive — but a slot whose
    /// address never left this function is out of its reach.
    #[test]
    fn a_call_keeps_escaped_stores_but_not_private_ones() {
        let mut it = Interner::new();
        let mut b = Builder::new(it.intern("h"), MirType::Void);
        let out = b.add_param(MirType::Ptr);
        let priv_slot = b.alloca(MirType::Array(Box::new(MirType::I8), 8));
        let one = b.build(MirType::F32, Op::ConstFloat(1.0, MirType::F32));
        let two = b.build(MirType::F32, Op::ConstFloat(2.0, MirType::F32));
        b.build_void(Op::Store {
            ptr: out,
            value: one,
        });
        b.build_void(Op::Store {
            ptr: priv_slot,
            value: one,
        });
        b.build_void(Op::Call {
            func: it.intern("sink"),
            args: vec![out],
        });
        b.build_void(Op::Store {
            ptr: out,
            value: two,
        });
        b.build_void(Op::Store {
            ptr: priv_slot,
            value: two,
        });
        b.ret(None);
        let mut f = b.finish();

        assert_eq!(count_stores(&f), 4);
        let mut cache = CfgAnalyses::default();
        assert!(Dse.run_function(&mut f, &mut cache));
        // Both private stores now die: no instruction ever reads the slot. The parameter's
        // first store still survives the call, and its final store remains caller-observable.
        assert_eq!(
            count_stores(&f),
            2,
            "{}",
            wukong_mir::print::print_function(&f, &it)
        );
    }

    fn verify(f: &Function) {
        let errors = wukong_mir::verify::verify_function(f);
        assert!(errors.is_empty(), "{errors:?}");
    }

    fn run(f: &Function, it: &Interner) -> i64 {
        let mut p = wukong_mir::Program::new();
        p.funcs.push(f.clone());
        wukong_interp::run(&p, f.name, it).unwrap()
    }

    #[test]
    fn unread_private_array_stores_die_across_branches_but_read_slot_survives() {
        for take_left in [false, true] {
            let mut it = Interner::new();
            let mut b = Builder::new(it.intern("main"), MirType::I64);
            let dead = b.alloca(MirType::Array(Box::new(MirType::I64), 2));
            let live = b.alloca(MirType::I64);
            let one = b.build(MirType::I64, Op::ConstInt(1, MirType::I64));
            let tail = b.build(
                MirType::Ptr,
                Op::Gep {
                    ptr: dead,
                    index: one,
                    elem: MirType::I64,
                },
            );
            let a = b.build(MirType::I64, Op::ConstInt(23, MirType::I64));
            let c = b.build(MirType::I64, Op::ConstInt(41, MirType::I64));
            let cond = b.build(
                MirType::I1,
                Op::ConstInt(i128::from(take_left), MirType::I1),
            );
            b.build_void(Op::Store {
                ptr: dead,
                value: one,
            });
            let left = b.new_block();
            let right = b.new_block();
            let exit = b.new_block();
            b.cond_br(cond, left, vec![], right, vec![]);
            b.switch_to(left);
            b.build_void(Op::Store {
                ptr: tail,
                value: a,
            });
            b.build_void(Op::Store {
                ptr: live,
                value: a,
            });
            b.br(exit, vec![]);
            b.switch_to(right);
            b.build_void(Op::Store {
                ptr: tail,
                value: c,
            });
            b.build_void(Op::Store {
                ptr: live,
                value: c,
            });
            b.br(exit, vec![]);
            b.switch_to(exit);
            let answer = b.build(MirType::I64, Op::Load(live, MirType::I64));
            b.ret(Some(answer));
            let mut f = b.finish();
            verify(&f);
            assert_eq!(run(&f, &it), if take_left { 23 } else { 41 });
            assert_eq!(count_stores(&f), 5);
            let mut cache = CfgAnalyses::default();
            assert!(Dse.run_function(&mut f, &mut cache));
            assert_eq!(count_stores(&f), 2);
            verify(&f);
            assert_eq!(run(&f, &it), if take_left { 23 } else { 41 });
            assert!(crate::dce::Dce.run_function(&mut f, &mut cache));
            assert!(!f
                .blocks
                .iter()
                .flat_map(|b| &b.insts)
                .any(|i| i.result == Some(dead)));
            verify(&f);
            assert_eq!(run(&f, &it), if take_left { 23 } else { 41 });
        }
    }

    #[test]
    fn successor_read_through_distinct_gep_keeps_the_store() {
        let mut it = Interner::new();
        let mut b = Builder::new(it.intern("main"), MirType::I64);
        let slot = b.alloca(MirType::Array(Box::new(MirType::I64), 2));
        let index = b.build(MirType::I64, Op::ConstInt(1, MirType::I64));
        let p = b.build(
            MirType::Ptr,
            Op::Gep {
                ptr: slot,
                index,
                elem: MirType::I64,
            },
        );
        let value = b.build(MirType::I64, Op::ConstInt(73, MirType::I64));
        b.build_void(Op::Store { ptr: p, value });
        let next = b.new_block();
        b.br(next, vec![]);
        b.switch_to(next);
        let q = b.build(
            MirType::Ptr,
            Op::Gep {
                ptr: slot,
                index,
                elem: MirType::I64,
            },
        );
        let read = b.build(MirType::I64, Op::Load(q, MirType::I64));
        b.ret(Some(read));
        let mut f = b.finish();
        verify(&f);
        assert_eq!(run(&f, &it), 73);
        assert!(!Dse.run_function(&mut f, &mut CfgAnalyses::default()));
        assert_eq!(count_stores(&f), 1);
        verify(&f);
        assert_eq!(run(&f, &it), 73);
    }

    #[test]
    fn published_or_observed_addresses_keep_unread_stores() {
        for observer in ["call", "stored", "ptrtoint", "return", "edge"] {
            let mut it = Interner::new();
            let ret = if observer == "return" {
                MirType::Ptr
            } else {
                MirType::Void
            };
            let mut b = Builder::new(it.intern("f"), ret);
            let out = b.add_param(MirType::Ptr);
            let slot = b.alloca(MirType::I64);
            let value = b.build(MirType::I64, Op::ConstInt(73, MirType::I64));
            b.build_void(Op::Store { ptr: slot, value });
            match observer {
                "call" => b.build_void(Op::Call {
                    func: it.intern("sink"),
                    args: vec![slot],
                }),
                "stored" => b.build_void(Op::Store {
                    ptr: out,
                    value: slot,
                }),
                "ptrtoint" => {
                    let addr = b.build(
                        MirType::I64,
                        Op::Cast(wukong_mir::CastKind::PtrToInt, slot, MirType::I64),
                    );
                    b.build_void(Op::Store {
                        ptr: out,
                        value: addr,
                    });
                }
                "edge" => {
                    let next = b.new_block();
                    let _ = b.block_param(next, MirType::Ptr);
                    b.br(next, vec![slot]);
                    b.switch_to(next);
                }
                "return" => {}
                _ => unreachable!(),
            }
            b.ret((observer == "return").then_some(slot));
            let mut f = b.finish();
            verify(&f);
            let before = count_stores(&f);
            assert!(
                !Dse.run_function(&mut f, &mut CfgAnalyses::default()),
                "{observer}"
            );
            assert_eq!(count_stores(&f), before, "{observer}");
            verify(&f);
        }
    }

    #[test]
    fn vector_kernel_pointer_table_and_scalar_arguments_keep_stores() {
        let mut it = Interner::new();
        let mut b = Builder::new(it.intern("main"), MirType::I64);
        let data = b.alloca(MirType::Array(Box::new(MirType::F32), 8));
        let ptrs = b.alloca(MirType::Ptr);
        let scalars = b.alloca(MirType::F32);
        let value = b.build(MirType::F32, Op::ConstFloat(7.0, MirType::F32));
        // The kernel ABI processes complete eight-lane groups; initialize every lane.
        for lane in 0..8 {
            let index = b.build(MirType::I64, Op::ConstInt(lane, MirType::I64));
            let ptr = b.build(
                MirType::Ptr,
                Op::Gep {
                    ptr: data,
                    index,
                    elem: MirType::F32,
                },
            );
            b.build_void(Op::Store { ptr, value });
        }
        b.build_void(Op::Store {
            ptr: ptrs,
            value: data,
        });
        b.build_void(Op::Store {
            ptr: scalars,
            value,
        });
        let n = b.build(MirType::I64, Op::ConstInt(8, MirType::I64));
        let kernel = b.add_vec_kernel(wukong_mir::VecKernel {
            name: it.intern("kernel"),
            lane: wukong_mir::VecLane::F32,
            streams: 1,
            scalars: 1,
            unroll: 1,
            ops: vec![
                wukong_mir::VecOp::Load { stream: 0 },
                wukong_mir::VecOp::Splat { scalar: 0 },
                wukong_mir::VecOp::Bin {
                    op: wukong_mir::VecBin::Add,
                    a: 0,
                    b: 1,
                },
            ],
            reduce: Some(wukong_mir::VecReduce {
                op: wukong_mir::VecRedOp::Add,
                value: 2,
                fma: None,
            }),
        });
        let result = b.build(
            MirType::F32,
            Op::VecKernelCall {
                kernel,
                ptrs,
                scalars,
                n,
            },
        );
        let answer = b.build(
            MirType::I64,
            Op::Cast(wukong_mir::CastKind::FpToSi, result, MirType::I64),
        );
        b.ret(Some(answer));
        let mut f = b.finish();
        verify(&f);
        assert_eq!(run(&f, &it), 8 * (7 + 7));
        assert!(!Dse.run_function(&mut f, &mut CfgAnalyses::default()));
        assert_eq!(count_stores(&f), 10);
        verify(&f);
        assert_eq!(run(&f, &it), 8 * (7 + 7));
    }

    #[test]
    fn barriers_keep_stores_and_block_local_overwrite_elimination() {
        let mut it = Interner::new();
        let mut b = Builder::new(it.intern("f"), MirType::Void);
        let slot = b.alloca(MirType::I64);
        let value = b.build(MirType::I64, Op::ConstInt(3, MirType::I64));
        b.build_void(Op::Store { ptr: slot, value });
        b.barrier();
        b.build_void(Op::Store { ptr: slot, value });
        b.ret(None);
        let mut f = b.finish();
        verify(&f);
        assert!(!Dse.run_function(&mut f, &mut CfgAnalyses::default()));
        assert_eq!(count_stores(&f), 2);
        verify(&f);
    }

    #[test]
    fn unproven_store_extents_are_retained() {
        for offset in [None, Some(-1), Some(1)] {
            let mut it = Interner::new();
            let mut b = Builder::new(it.intern("f"), MirType::Void);
            let index = match offset {
                None => b.add_param(MirType::I64),
                Some(v) => b.build(MirType::I64, Op::ConstInt(v, MirType::I64)),
            };
            let slot = b.alloca(MirType::I64);
            let p = b.build(
                MirType::Ptr,
                Op::Gep {
                    ptr: slot,
                    index,
                    elem: MirType::I64,
                },
            );
            let value = b.build(MirType::I64, Op::ConstInt(1, MirType::I64));
            b.build_void(Op::Store { ptr: p, value });
            b.ret(None);
            let mut f = b.finish();
            verify(&f);
            assert!(
                !Dse.run_function(&mut f, &mut CfgAnalyses::default()),
                "{offset:?}"
            );
            assert_eq!(count_stores(&f), 1);
            verify(&f);
        }
    }

    fn i64c(b: &mut Builder, k: i128) -> ValueId {
        b.build(MirType::I64, Op::ConstInt(k, MirType::I64))
    }

    /// Emit `for i in start..end { body(i) }` as the front end does (header test, body, latch), and
    /// leave the builder in the exit block.
    fn counted_loop(
        b: &mut Builder,
        start: i128,
        end: i128,
        body: impl FnOnce(&mut Builder, ValueId),
    ) {
        let lo = i64c(b, start);
        let hi = i64c(b, end);
        let one = i64c(b, 1);
        let header = b.new_block();
        let blk = b.new_block();
        let exit = b.new_block();
        let i = b.block_param(header, MirType::I64);
        b.br(header, vec![lo]);
        b.switch_to(header);
        let c = b.build(MirType::I1, Op::Cmp(wukong_mir::CmpOp::Slt, i, hi));
        b.cond_br(c, blk, vec![], exit, vec![]);
        b.switch_to(blk);
        body(b, i);
        let next = b.build(MirType::I64, Op::Bin(wukong_mir::BinOp::Add, i, one));
        b.br(header, vec![next]);
        b.switch_to(exit);
    }

    /// Two `gep`s that are different SSA values but land on the same constant offset of one slot
    /// are one location: the first store is overwritten before anything reads it.
    #[test]
    fn a_location_spelled_by_two_geps_is_matched_must_alias() {
        let mut it = Interner::new();
        let mut b = Builder::new(it.intern("main"), MirType::I64);
        let slot = b.alloca(MirType::Array(Box::new(MirType::I64), 4));
        let one_a = i64c(&mut b, 1);
        let one_b = i64c(&mut b, 1);
        let p = b.build(
            MirType::Ptr,
            Op::Gep {
                ptr: slot,
                index: one_a,
                elem: MirType::I64,
            },
        );
        let q = b.build(
            MirType::Ptr,
            Op::Gep {
                ptr: slot,
                index: one_b,
                elem: MirType::I64,
            },
        );
        let ten = i64c(&mut b, 10);
        let twenty = i64c(&mut b, 20);
        b.build_void(Op::Store { ptr: p, value: ten });
        b.build_void(Op::Store {
            ptr: q,
            value: twenty,
        });
        let v = b.build(MirType::I64, Op::Load(p, MirType::I64));
        b.ret(Some(v));
        let mut f = b.finish();
        verify(&f);
        assert_eq!(run(&f, &it), 20);
        assert!(Dse.run_function(&mut f, &mut CfgAnalyses::default()));
        assert_eq!(count_stores(&f), 1);
        verify(&f);
        assert_eq!(run(&f, &it), 20);
    }

    /// `s = 1; if c { s = 2 } else { s = 3 }; return s` — the first store is overwritten on every
    /// path out of its block. With a read on one path before the overwrite, it must stay.
    #[test]
    fn a_store_overwritten_on_every_path_is_dead_across_blocks() {
        for read_first in [false, true] {
            let mut it = Interner::new();
            let mut b = Builder::new(it.intern("main"), MirType::I64);
            let s = b.alloca(MirType::I64);
            let one = i64c(&mut b, 1);
            let two = i64c(&mut b, 2);
            let three = i64c(&mut b, 3);
            let c = b.build(MirType::I1, Op::ConstInt(1, MirType::I1));
            b.build_void(Op::Store { ptr: s, value: one });
            let left = b.new_block();
            let right = b.new_block();
            let join = b.new_block();
            let carried = b.block_param(join, MirType::I64);
            b.cond_br(c, left, vec![], right, vec![]);
            b.switch_to(left);
            let seen = if read_first {
                b.build(MirType::I64, Op::Load(s, MirType::I64))
            } else {
                one
            };
            b.build_void(Op::Store { ptr: s, value: two });
            b.br(join, vec![seen]);
            b.switch_to(right);
            b.build_void(Op::Store {
                ptr: s,
                value: three,
            });
            b.br(join, vec![one]);
            b.switch_to(join);
            let v = b.build(MirType::I64, Op::Load(s, MirType::I64));
            let r = b.build(MirType::I64, Op::Bin(wukong_mir::BinOp::Add, v, carried));
            b.ret(Some(r));
            let mut f = b.finish();
            verify(&f);
            // The left arm is taken: s = 2, and `carried` is 1 either way (the read sees the 1).
            assert_eq!(run(&f, &it), 3, "read_first={read_first}");
            let changed = Dse.run_function(&mut f, &mut CfgAnalyses::default());
            assert_eq!(changed, !read_first, "read_first={read_first}");
            assert_eq!(count_stores(&f), if read_first { 3 } else { 2 });
            verify(&f);
            assert_eq!(run(&f, &it), 3, "read_first={read_first}");
        }
    }

    /// A private slot read before a loop and only written inside it: every loop store is either
    /// overwritten by the next iteration or retired by the end of the function, so it is dead even
    /// though the slot as a whole is read.
    #[test]
    fn a_private_slot_written_last_inside_a_loop_dies_at_the_end_of_its_lifetime() {
        let mut it = Interner::new();
        let mut b = Builder::new(it.intern("main"), MirType::I64);
        let s = b.alloca(MirType::I64);
        let five = i64c(&mut b, 5);
        b.build_void(Op::Store {
            ptr: s,
            value: five,
        });
        let first = b.build(MirType::I64, Op::Load(s, MirType::I64));
        counted_loop(&mut b, 0, 3, |b, i| {
            b.build_void(Op::Store { ptr: s, value: i });
        });
        b.ret(Some(first));
        let mut f = b.finish();
        verify(&f);
        assert_eq!(run(&f, &it), 5);
        assert!(Dse.run_function(&mut f, &mut CfgAnalyses::default()));
        assert_eq!(
            count_stores(&f),
            1,
            "only the store the load reads survives"
        );
        verify(&f);
        assert_eq!(run(&f, &it), 5);
    }

    /// The fill loop `for i in 0..16 { t[i] = 0.0 }` into a slot nothing reads: the index is the
    /// induction variable of a loop over a constant range inside the slot, so the dynamically
    /// indexed store is provably in bounds and dies. One iteration too many, or a byte stride that
    /// stays inside the slot natively but runs past it in the interpreter, keeps it.
    #[test]
    fn a_loop_bounded_store_into_an_unread_slot_dies_only_when_in_bounds() {
        let f32x16 = MirType::Array(Box::new(MirType::F32), 16);
        let f32x4 = MirType::Array(Box::new(MirType::F32), 4);
        // (trip end, gep element, slot, expect removal)
        let cases = [
            (16, MirType::F32, f32x16.clone(), true),
            (17, MirType::F32, f32x16, false),
            // 12 one-byte stores: bytes 0..12 of 16 natively, but leaf slots 0..12 of 4.
            (12, MirType::I8, f32x4, false),
        ];
        for (end, elem, slot_ty, removed) in cases {
            let mut it = Interner::new();
            let mut b = Builder::new(it.intern("main"), MirType::I64);
            let t = b.alloca(slot_ty);
            let value = if elem == MirType::I8 {
                b.build(MirType::I8, Op::ConstInt(0, MirType::I8))
            } else {
                b.build(MirType::F32, Op::ConstFloat(0.0, MirType::F32))
            };
            let el = elem.clone();
            counted_loop(&mut b, 0, end, move |b, i| {
                let p = b.build(
                    MirType::Ptr,
                    Op::Gep {
                        ptr: t,
                        index: i,
                        elem: el,
                    },
                );
                b.build_void(Op::Store { ptr: p, value });
            });
            let seven = i64c(&mut b, 7);
            b.ret(Some(seven));
            let mut f = b.finish();
            verify(&f);
            let changed = Dse.run_function(&mut f, &mut CfgAnalyses::default());
            assert_eq!(changed, removed, "end={end} elem={elem:?}");
            assert_eq!(count_stores(&f), usize::from(!removed), "end={end}");
            verify(&f);
            if removed {
                assert_eq!(run(&f, &it), 7);
            }
        }
    }

    /// The same store in the loop *header* runs once more than the body, with the induction variable
    /// already one past the end, so the range proof does not cover it.
    #[test]
    fn a_header_store_is_not_covered_by_the_trip_count() {
        let mut it = Interner::new();
        let mut b = Builder::new(it.intern("main"), MirType::I64);
        let t = b.alloca(MirType::Array(Box::new(MirType::F32), 16));
        let zero = b.build(MirType::F32, Op::ConstFloat(0.0, MirType::F32));
        let lo = i64c(&mut b, 0);
        let hi = i64c(&mut b, 16);
        let one = i64c(&mut b, 1);
        let header = b.new_block();
        let body = b.new_block();
        let exit = b.new_block();
        let i = b.block_param(header, MirType::I64);
        b.br(header, vec![lo]);
        b.switch_to(header);
        let p = b.build(
            MirType::Ptr,
            Op::Gep {
                ptr: t,
                index: i,
                elem: MirType::F32,
            },
        );
        b.build_void(Op::Store {
            ptr: p,
            value: zero,
        });
        let c = b.build(MirType::I1, Op::Cmp(wukong_mir::CmpOp::Slt, i, hi));
        b.cond_br(c, body, vec![], exit, vec![]);
        b.switch_to(body);
        let next = b.build(MirType::I64, Op::Bin(wukong_mir::BinOp::Add, i, one));
        b.br(header, vec![next]);
        b.switch_to(exit);
        b.ret(Some(lo));
        let mut f = b.finish();
        verify(&f);
        assert!(!Dse.run_function(&mut f, &mut CfgAnalyses::default()));
        assert_eq!(count_stores(&f), 1);
    }
}
