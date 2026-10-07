//! `sroa` — scalar replacement of aggregates.
//!
//! `mir_build` gives every aggregate local — a fixed array, a struct, a tuple, a `[]T` fat pointer —
//! one `alloca` of an [`MirType::Array`] type and reaches every element through a `gep`.
//! [`crate::Mem2Reg`] only promotes a slot whose address is used *solely* as a load/store address,
//! so a single `gep` keeps the whole aggregate in memory for the life of the function: a
//! register-blocked accumulator `let mut acc: [f32; 4]` pays a load and a store per element per
//! iteration, a struct's fields are re-read from the stack at every use, and a slice local re-loads
//! its data pointer out of its fat pointer at every subscript. Every value-based pass downstream
//! (CSE, LICM, the vectorizer's reduction detection) then works around memory traffic instead of
//! through it.
//!
//! This pass splits such a slot into one scalar `alloca` per **accessed leaf** when every access is
//! at a compile-time-constant position, and leaves the rest to mem2reg, which runs right after it in
//! the same fixpoint sweep and turns each leaf into block-parameter SSA. Leaves nothing reads or
//! writes simply disappear, exactly as in LLVM's SROA.
//!
//! # When a slot is split
//!
//! A local (`AddrSpace::Local`) `alloca` of an `Array` type qualifies when:
//!
//! * every pointer derived from it is a `gep` chain with **constant** indices, and every use of
//!   such a pointer is the address of a `load`/`store` or the base of a further `gep` — anything
//!   else (stored as a value, passed to a call or a vector kernel, branched with, selected, cast) is
//!   an escape and keeps the slot whole;
//! * every access transfers a scalar the split can carry (`i1`..`i64`, `f32`, `f64`, `ptr`). Vector
//!   accesses span several leaves, and `bf16`/`f16` round on their way through memory, which a
//!   register would skip (the open half-float `-O0`/`-O2` defect CSE already refuses to widen);
//! * all accesses at one position use one type, and the native byte ranges of distinct positions
//!   are disjoint and inside the slot — a type-punned or overlapping access (the byte-wise aggregate
//!   copy `emit_copy_bytes`, an `i64` read over two `i32` fields) keeps the slot whole.
//!
//! # The two memory models
//!
//! The interpreter (the language specification) gives every *scalar leaf* of a slot one memory
//! slot and scales a `gep` by `slot_count(elem)`; the native backend uses bytes and scales by
//! `size_of(elem)`. A struct/tuple byte buffer (`[n x i8]`) has one slot per byte, a typed array one
//! slot per element — and the two layouts need not agree: an `[f32; 4]` field of a struct buffer is
//! four consecutive interpreter slots but sixteen native bytes. Every derived pointer therefore
//! carries **both** positions. All accesses at one interpreter slot must agree on the native byte
//! offset and type, distinct slots must occupy disjoint native byte ranges, and both must lie inside
//! the `alloca`; then each leaf is one independent cell in *both* models and the split preserves
//! each separately. A program that addresses one cell with two different strides (and so already
//! means different things on the two backends) is left alone.
//!
//! A new leaf `alloca T` starts at the interpreter's default for `T`, while the leaf it replaces
//! started at the default of the *slot's* leaf type: `Int(0)` for every byte of a struct buffer,
//! `Float(0.0)` for an `f32` field read through it. The two are observably different (`-x` of the
//! former is `0`, of the latter `-0`), so when they differ the pass proves, with a forward
//! must-be-written dataflow from the `alloca`, that no load of that leaf can run before a store to
//! it. The front end initializes every local before reading it, so this costs nothing in practice
//! and only guards the semantics of a hand-built or future read-before-write.
//!
//! The pass never splits a slot that is not an `Array` (a scalar slot is mem2reg's job, and splitting
//! it would report a change every sweep), and caps the leaves at [`MAX_LEAVES`].

use crate::alias::type_bytes;
use crate::fxhash::{FxHashMap, FxHashSet};

use wukong_mir::{AddrSpace, Function, Inst, MirType, Op, ValueId};

use crate::{cfg, each_op_use, each_term_use, CfgAnalyses, Pass};

pub struct Sroa;

/// Most distinct leaves one slot is split into. The must-be-written proof keeps one bit per leaf.
const MAX_LEAVES: usize = 64;

/// Cap on the rounds of the derived-pointer sweep. A `gep` chain whose links are laid out in reverse
/// dominance order needs one round per link; a chain that has not settled by then makes the pass
/// give up on the whole function rather than leave a pointer unclassified.
const MAX_ROUNDS: u32 = 16;

impl Pass for Sroa {
    fn name(&self) -> &'static str {
        "sroa"
    }

    fn run_function(&self, f: &mut Function, _cache: &mut CfgAnalyses) -> bool {
        let (splits, derived) = plan(f);
        if splits.is_empty() {
            return false;
        }
        apply(f, &splits, &derived);
        true
    }
}

/// Where a pointer derived from a candidate slot points: the slot (by its `alloca` result id) and
/// the position in both memory models.
#[derive(Clone, Copy)]
struct Derived {
    root: u32,
    /// Interpreter slot index from the start of the `alloca`.
    slot: i64,
    /// Native byte offset from the start of the `alloca`.
    byte: i64,
}

/// One load or store through a derived pointer.
struct Access {
    root: u32,
    slot: i64,
    byte: i64,
    ty: MirType,
    block: u32,
    inst: u32,
    is_store: bool,
}

/// A split decided by [`plan`]: the slot, and `(interpreter slot, leaf type)` of every accessed leaf,
/// ascending by slot.
struct Split {
    root: u32,
    leaves: Vec<(i64, MirType)>,
}

/// The interpreter's slot footprint of a value of `ty` (`wukong_interp::slot_count`).
pub(crate) fn slot_count(ty: &MirType) -> Option<u64> {
    match ty {
        MirType::Array(elem, n) => slot_count(elem)?.checked_mul(u64::from(*n)),
        _ => Some(1),
    }
}

/// The native byte offset and the leaf type of interpreter slot `s` of a slot of type `ty`, or
/// `None` when `s` is outside it.
fn leaf_at_slot(ty: &MirType, s: u64) -> Option<(u64, &MirType)> {
    match ty {
        MirType::Array(elem, n) => {
            let per = slot_count(elem)?;
            if per == 0 {
                return None;
            }
            let i = s / per;
            if i >= u64::from(*n) {
                return None;
            }
            let (inner, leaf) = leaf_at_slot(elem, s % per)?;
            let off = i.checked_mul(u64::from(type_bytes(elem)?))?;
            Some((off.checked_add(inner)?, leaf))
        }
        MirType::Void => None,
        other => (s == 0).then_some((0, other)),
    }
}

/// The interpreter's zero for a fresh slot of `ty` (`wukong_interp::default_value`): a float, a
/// pointer, or an integer.
fn default_class(ty: &MirType) -> u8 {
    if ty.is_float() {
        1
    } else if matches!(ty, MirType::Ptr) {
        2
    } else {
        0
    }
}

/// Can a leaf carry a value of this type through a scalar `alloca` without changing it?
fn splittable(ty: &MirType) -> bool {
    ty.is_int() || matches!(ty, MirType::F32 | MirType::F64 | MirType::Ptr)
}

/// A `ConstInt` used as a `gep` index, as the interpreter sees it: masked to its declared width
/// (sign-extended, except `i1`, which is its low bit).
pub(crate) fn const_index(k: i128, ty: &MirType) -> Option<i64> {
    let bits: u32 = match ty {
        MirType::I1 => return Some((k & 1) as i64),
        MirType::I8 => 8,
        MirType::I16 => 16,
        MirType::I32 => 32,
        MirType::I64 => 64,
        _ => return None,
    };
    let shift = 128 - bits;
    i64::try_from((k << shift) >> shift).ok()
}

/// Decide which slots to split. Returns the splits (ascending by slot id, so the values `apply`
/// mints are numbered deterministically) and the derived-pointer table `apply` rewrites through.
fn plan(f: &Function) -> (Vec<Split>, FxHashMap<u32, Derived>) {
    // Candidate slots, with their type and position.
    let mut slots: FxHashMap<u32, (MirType, u32, u32)> = FxHashMap::default();
    let mut derived: FxHashMap<u32, Derived> = FxHashMap::default();
    for b in &f.blocks {
        for (i, inst) in b.insts.iter().enumerate() {
            if let (
                Some(r),
                Op::Alloca {
                    ty: ty @ MirType::Array(..),
                    space: AddrSpace::Local,
                },
            ) = (inst.result, &inst.op)
            {
                slots.insert(r.0, (ty.clone(), b.id.0, i as u32));
                derived.insert(
                    r.0,
                    Derived {
                        root: r.0,
                        slot: 0,
                        byte: 0,
                    },
                );
            }
        }
    }
    if slots.is_empty() {
        return (Vec::new(), derived);
    }

    let mut konst: FxHashMap<u32, i64> = FxHashMap::default();
    for inst in f.blocks.iter().flat_map(|b| &b.insts) {
        if let (Some(r), Op::ConstInt(k, ty)) = (inst.result, &inst.op) {
            if let Some(k) = const_index(*k, ty) {
                konst.insert(r.0, k);
            }
        }
    }

    // Slots that must stay whole: a dynamic index, an escape, an unsupported access.
    let mut bad: FxHashSet<u32> = FxHashSet::default();

    // Derived pointers. Dominance, not layout, orders definitions, so sweep until no `gep` is newly
    // classified; give up on the function if that does not settle.
    let mut settled = false;
    for _ in 0..MAX_ROUNDS {
        let mut grew = false;
        for inst in f.blocks.iter().flat_map(|b| &b.insts) {
            let (Some(r), Op::Gep { ptr, index, elem }) = (inst.result, &inst.op) else {
                continue;
            };
            if derived.contains_key(&r.0) {
                continue;
            }
            let Some(&base) = derived.get(&ptr.0) else {
                continue;
            };
            grew = true;
            let step = konst.get(&index.0).copied().and_then(|k| {
                let per_slot = i64::try_from(slot_count(elem)?).ok()?;
                let per_byte = i64::from(type_bytes(elem)?);
                Some((
                    base.slot.checked_add(k.checked_mul(per_slot)?)?,
                    base.byte.checked_add(k.checked_mul(per_byte)?)?,
                ))
            });
            let d = match step {
                Some((slot, byte)) => Derived {
                    root: base.root,
                    slot,
                    byte,
                },
                None => {
                    // A dynamic (or overflowing) index: the leaf is unknown.
                    bad.insert(base.root);
                    Derived {
                        root: base.root,
                        slot: 0,
                        byte: 0,
                    }
                }
            };
            derived.insert(r.0, d);
        }
        if !grew {
            settled = true;
            break;
        }
    }
    if !settled {
        return (Vec::new(), derived);
    }

    // Every use of a derived pointer.
    let mut accesses: Vec<Access> = Vec::new();
    for b in &f.blocks {
        for (i, inst) in b.insts.iter().enumerate() {
            match &inst.op {
                // The base operand was classified above. An address used as the *index* is an
                // integer use of it.
                Op::Gep { index, .. } => {
                    if let Some(d) = derived.get(&index.0) {
                        bad.insert(d.root);
                    }
                }
                Op::Load(p, ty) => {
                    if let Some(d) = derived.get(&p.0) {
                        accesses.push(Access {
                            root: d.root,
                            slot: d.slot,
                            byte: d.byte,
                            ty: ty.clone(),
                            block: b.id.0,
                            inst: i as u32,
                            is_store: false,
                        });
                    }
                }
                Op::Store { ptr, value } => {
                    if let Some(d) = derived.get(&value.0) {
                        bad.insert(d.root); // the address itself is published
                    }
                    if let Some(d) = derived.get(&ptr.0) {
                        accesses.push(Access {
                            root: d.root,
                            slot: d.slot,
                            byte: d.byte,
                            ty: f.value_type(*value).clone(),
                            block: b.id.0,
                            inst: i as u32,
                            is_store: true,
                        });
                    }
                }
                other => each_op_use(other, &mut |v| {
                    if let Some(d) = derived.get(&v.0) {
                        bad.insert(d.root);
                    }
                }),
            }
        }
        each_term_use(&b.term, &mut |v| {
            if let Some(d) = derived.get(&v.0) {
                bad.insert(d.root);
            }
        });
    }

    let mut by_root: FxHashMap<u32, Vec<usize>> = FxHashMap::default();
    for (k, a) in accesses.iter().enumerate() {
        by_root.entry(a.root).or_default().push(k);
    }
    let mut roots: Vec<u32> = slots
        .keys()
        .copied()
        .filter(|r| !bad.contains(r) && by_root.contains_key(r))
        .collect();
    roots.sort_unstable();

    // Predecessors and reverse postorder, computed only if some leaf needs the written-first proof.
    let mut cfg_facts: Option<(Vec<Vec<u32>>, Vec<u32>)> = None;
    let mut splits = Vec::new();
    for root in roots {
        let (ty, home_block, home_inst) = &slots[&root];
        let mine: Vec<&Access> = by_root[&root].iter().map(|&k| &accesses[k]).collect();
        let Some((leaves, mismatch)) = leaves_of(ty, &mine) else {
            continue;
        };
        if mismatch != 0 {
            let facts =
                cfg_facts.get_or_insert_with(|| (cfg::predecessors(f), cfg::reverse_postorder(f)));
            if !written_before_read(
                f,
                (*home_block, *home_inst),
                &leaves,
                &mine,
                mismatch,
                &facts.0,
                &facts.1,
            ) {
                continue;
            }
        }
        splits.push(Split { root, leaves });
    }
    (splits, derived)
}

/// The leaves a slot of type `ty` splits into, given every access to it, or `None` when it must stay
/// whole. The second result has bit `k` set when leaf `k`'s fresh `alloca` would start at a
/// different interpreter default than the slot did (see the module note).
fn leaves_of(ty: &MirType, accesses: &[&Access]) -> Option<(Vec<(i64, MirType)>, u64)> {
    let total = u64::from(type_bytes(ty)?);
    // (slot, byte, access type, the slot's own leaf type)
    let mut leaves: Vec<(i64, i64, MirType, MirType)> = Vec::new();
    for a in accesses {
        if !splittable(&a.ty) {
            return None;
        }
        // Inside the slot in both models: one of its interpreter slots, and a byte range of it.
        let (_, leaf) = leaf_at_slot(ty, u64::try_from(a.slot).ok()?)?;
        let end = u64::try_from(a.byte)
            .ok()?
            .checked_add(u64::from(type_bytes(&a.ty)?))?;
        if end > total {
            return None;
        }
        match leaves.iter().find(|l| l.0 == a.slot) {
            // One interpreter cell must be one native byte range, of one type.
            Some(l) if l.1 != a.byte || l.2 != a.ty => return None,
            Some(_) => {}
            None => {
                if leaves.len() == MAX_LEAVES {
                    return None;
                }
                leaves.push((a.slot, a.byte, a.ty.clone(), leaf.clone()));
            }
        }
    }
    // Distinct interpreter cells must not share native bytes, or separating them would split
    // storage the native backend shares (two cells at one byte offset fail this too).
    leaves.sort_unstable_by_key(|l| l.1);
    for w in leaves.windows(2) {
        let end = w[0].1.checked_add(i64::from(type_bytes(&w[0].2)?))?;
        if end > w[1].1 {
            return None;
        }
    }
    leaves.sort_unstable_by_key(|l| l.0);
    let mut mismatch = 0u64;
    for (k, l) in leaves.iter().enumerate() {
        if default_class(&l.2) != default_class(&l.3) {
            mismatch |= 1u64 << k;
        }
    }
    Some((
        leaves
            .into_iter()
            .map(|(slot, _, t, _)| (slot, t))
            .collect(),
        mismatch,
    ))
}

/// One event of the must-be-written dataflow, at its instruction index within a block.
#[derive(Clone, Copy)]
enum Event {
    /// The slot's `alloca` runs (again): every leaf is fresh.
    Reset,
    Store(u64),
    Load(u64),
}

/// Is every load of a leaf in `mask` preceded, on every path from the slot's `alloca`, by a store to
/// that leaf? A forward must-analysis over the reachable blocks: a block starts with the
/// intersection of what its predecessors end with, the entry block with nothing.
fn written_before_read(
    f: &Function,
    home: (u32, u32),
    leaves: &[(i64, MirType)],
    accesses: &[&Access],
    mask: u64,
    preds: &[Vec<u32>],
    rpo: &[u32],
) -> bool {
    let mut events: Vec<Vec<(u32, Event)>> = vec![Vec::new(); f.blocks.len()];
    events[home.0 as usize].push((home.1, Event::Reset));
    for a in accesses {
        let Some(k) = leaves.iter().position(|l| l.0 == a.slot) else {
            return false;
        };
        let bit = 1u64 << k;
        let e = if a.is_store {
            Event::Store(bit)
        } else {
            Event::Load(bit)
        };
        events[a.block as usize].push((a.inst, e));
    }
    for ev in &mut events {
        ev.sort_unstable_by_key(|e| e.0);
    }
    // `None` when a checked load may read an unwritten leaf.
    let transfer = |b: u32, mut cur: u64, check: bool| -> Option<u64> {
        for &(_, e) in &events[b as usize] {
            match e {
                Event::Reset => cur = 0,
                Event::Store(bit) => cur |= bit,
                Event::Load(bit) => {
                    if check && mask & bit != 0 && cur & bit == 0 {
                        return None;
                    }
                }
            }
        }
        Some(cur)
    };
    let entry_state = |b: u32, out: &[u64]| -> u64 {
        if b == f.entry.0 {
            0
        } else {
            preds[b as usize]
                .iter()
                .fold(u64::MAX, |acc, &p| acc & out[p as usize])
        }
    };
    let mut out = vec![u64::MAX; f.blocks.len()];
    loop {
        let mut changed = false;
        for &b in rpo {
            let inn = entry_state(b, &out);
            let o = transfer(b, inn, false).unwrap_or(0);
            if o != out[b as usize] {
                out[b as usize] = o;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    rpo.iter()
        .all(|&b| transfer(b, entry_state(b, &out), true).is_some())
}

/// Replace each split slot by its leaf `alloca`s (at the slot's own position, in slot order),
/// retarget every access to its leaf, and drop the slot's now-unused `gep`s.
fn apply(f: &mut Function, splits: &[Split], derived: &FxHashMap<u32, Derived>) {
    let mut leaf_ptr: FxHashMap<(u32, i64), ValueId> = FxHashMap::default();
    let mut replacement: FxHashMap<u32, Vec<Inst>> = FxHashMap::default();
    for s in splits {
        let mut allocas = Vec::with_capacity(s.leaves.len());
        for (slot, ty) in &s.leaves {
            let v = ValueId(f.value_types.len() as u32);
            f.value_types.push(MirType::Ptr);
            leaf_ptr.insert((s.root, *slot), v);
            allocas.push(Inst {
                result: Some(v),
                op: Op::Alloca {
                    ty: ty.clone(),
                    space: AddrSpace::Local,
                },
            });
        }
        replacement.insert(s.root, allocas);
    }
    let split_root = |p: ValueId| -> Option<Derived> {
        derived
            .get(&p.0)
            .copied()
            .filter(|d| replacement.contains_key(&d.root))
    };
    let mut retarget: FxHashMap<u32, ValueId> = FxHashMap::default();
    for (&v, d) in derived {
        if replacement.contains_key(&d.root) {
            if let Some(&leaf) = leaf_ptr.get(&(d.root, d.slot)) {
                retarget.insert(v, leaf);
            }
        }
    }

    for b in &mut f.blocks {
        let old = std::mem::take(&mut b.insts);
        let mut kept = Vec::with_capacity(old.len());
        for mut inst in old {
            if let Some(r) = inst.result {
                if let Some(allocas) = replacement.get(&r.0) {
                    if matches!(inst.op, Op::Alloca { .. }) {
                        kept.extend(allocas.iter().cloned());
                        continue;
                    }
                }
                if matches!(inst.op, Op::Gep { .. }) && split_root(r).is_some() {
                    continue;
                }
            }
            match &mut inst.op {
                Op::Load(p, _) => {
                    if let Some(&leaf) = retarget.get(&p.0) {
                        *p = leaf;
                    }
                }
                Op::Store { ptr, .. } => {
                    if let Some(&leaf) = retarget.get(&ptr.0) {
                        *ptr = leaf;
                    }
                }
                _ => {}
            }
            kept.push(inst);
        }
        b.insts = kept;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Mem2Reg;
    use wukong_mir::{BinOp, Builder, CastKind, CmpOp, Program};
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

    fn count(f: &Function, pred: impl Fn(&Op) -> bool) -> usize {
        f.blocks
            .iter()
            .flat_map(|b| &b.insts)
            .filter(|i| pred(&i.op))
            .count()
    }

    fn aggregate_allocas(f: &Function) -> usize {
        count(f, |op| {
            matches!(
                op,
                Op::Alloca {
                    ty: MirType::Array(..),
                    ..
                }
            )
        })
    }

    fn i64c(b: &mut Builder, k: i128) -> ValueId {
        b.build(MirType::I64, Op::ConstInt(k, MirType::I64))
    }

    fn gep(b: &mut Builder, ptr: ValueId, index: ValueId, elem: MirType) -> ValueId {
        b.build(MirType::Ptr, Op::Gep { ptr, index, elem })
    }

    /// The register-blocked accumulator: `acc: [f32; 4]`, zeroed, then `acc[k] += k + 1` for 10
    /// iterations, then `acc[0] + 2·acc[1] + 3·acc[2] + 4·acc[3]` as the exit code.
    fn blocked_accumulator(it: &mut Interner) -> Function {
        let mut b = Builder::new(it.intern("main"), MirType::I64);
        let acc = b.alloca(MirType::Array(Box::new(MirType::F32), 4));
        let zero = b.build(MirType::F32, Op::ConstFloat(0.0, MirType::F32));
        for k in 0..4 {
            let idx = i64c(&mut b, k);
            let p = gep(&mut b, acc, idx, MirType::F32);
            b.build_void(Op::Store {
                ptr: p,
                value: zero,
            });
        }
        let n = i64c(&mut b, 10);
        let start = i64c(&mut b, 0);
        let one = i64c(&mut b, 1);
        let header = b.new_block();
        let body = b.new_block();
        let exit = b.new_block();
        let i = b.block_param(header, MirType::I64);
        b.br(header, vec![start]);
        b.switch_to(header);
        let c = b.build(MirType::I1, Op::Cmp(CmpOp::Slt, i, n));
        b.cond_br(c, body, vec![], exit, vec![]);
        b.switch_to(body);
        for k in 0..4 {
            let idx = i64c(&mut b, k);
            let p = gep(&mut b, acc, idx, MirType::F32);
            let old = b.build(MirType::F32, Op::Load(p, MirType::F32));
            let add = b.build(MirType::F32, Op::ConstFloat(k as f64 + 1.0, MirType::F32));
            let new = b.build(MirType::F32, Op::Bin(BinOp::FAdd, old, add));
            b.build_void(Op::Store { ptr: p, value: new });
        }
        let next = b.build(MirType::I64, Op::Bin(BinOp::Add, i, one));
        b.br(header, vec![next]);
        b.switch_to(exit);
        let mut total = b.build(MirType::F32, Op::ConstFloat(0.0, MirType::F32));
        for k in 0..4 {
            let idx = i64c(&mut b, k);
            let p = gep(&mut b, acc, idx, MirType::F32);
            let v = b.build(MirType::F32, Op::Load(p, MirType::F32));
            let w = b.build(MirType::F32, Op::ConstFloat(k as f64 + 1.0, MirType::F32));
            let t = b.build(MirType::F32, Op::Bin(BinOp::FMul, v, w));
            total = b.build(MirType::F32, Op::Bin(BinOp::FAdd, total, t));
        }
        let r = b.build(
            MirType::I64,
            Op::Cast(CastKind::FpToSi, total, MirType::I64),
        );
        b.ret(Some(r));
        b.finish()
    }

    #[test]
    fn a_small_constant_indexed_array_slot_is_split_and_promoted() {
        let mut it = Interner::new();
        let mut f = blocked_accumulator(&mut it);
        verify(&f);
        // acc[k] = 10·(k+1); Σ (k+1)·acc[k] = 10·(1 + 4 + 9 + 16) = 300.
        assert_eq!(run(&f, &it), 300);
        let mut cache = CfgAnalyses::default();
        assert!(Sroa.run_function(&mut f, &mut cache));
        verify(&f);
        assert_eq!(aggregate_allocas(&f), 0, "the [4 x f32] slot was not split");
        assert_eq!(count(&f, |op| matches!(op, Op::Gep { .. })), 0);
        assert_eq!(
            count(&f, |op| matches!(
                op,
                Op::Alloca {
                    ty: MirType::F32,
                    ..
                }
            )),
            4
        );
        assert_eq!(run(&f, &it), 300);
        // A second run has nothing left to split, so the fixpoint can settle.
        assert!(!Sroa.run_function(&mut f, &mut cache));
        // mem2reg now promotes every leaf: no memory traffic is left at all.
        assert!(Mem2Reg.run_function(&mut f, &mut cache));
        verify(&f);
        assert_eq!(count(&f, |op| matches!(op, Op::Alloca { .. })), 0);
        assert_eq!(
            count(&f, |op| matches!(op, Op::Load(..) | Op::Store { .. })),
            0
        );
        assert_eq!(run(&f, &it), 300);
    }

    /// A struct byte buffer `{ x: f32 @0, n: i32 @4, p: f64 @8 }`. Each field becomes a leaf of its
    /// own type even though the buffer's leaves are bytes; the `f32`/`f64` leaves start at a
    /// different interpreter default than a byte, so they need — and get — the written-first proof.
    #[test]
    fn a_struct_buffer_splits_by_field() {
        let mut it = Interner::new();
        let mut b = Builder::new(it.intern("main"), MirType::I64);
        let s = b.alloca(MirType::Array(Box::new(MirType::I8), 16));
        let o0 = i64c(&mut b, 0);
        let o4 = i64c(&mut b, 4);
        let o8 = i64c(&mut b, 8);
        let px = gep(&mut b, s, o0, MirType::I8);
        let pn = gep(&mut b, s, o4, MirType::I8);
        let pp = gep(&mut b, s, o8, MirType::I8);
        let x = b.build(MirType::F32, Op::ConstFloat(2.5, MirType::F32));
        let n = b.build(MirType::I32, Op::ConstInt(7, MirType::I32));
        let p = b.build(MirType::F64, Op::ConstFloat(0.25, MirType::F64));
        b.build_void(Op::Store { ptr: px, value: x });
        b.build_void(Op::Store { ptr: pn, value: n });
        b.build_void(Op::Store { ptr: pp, value: p });
        let next = b.new_block();
        b.br(next, vec![]);
        b.switch_to(next);
        // Re-derive the field addresses in the successor, as a later statement would.
        let qx = gep(&mut b, s, o0, MirType::I8);
        let qp = gep(&mut b, s, o8, MirType::I8);
        let lx = b.build(MirType::F32, Op::Load(qx, MirType::F32));
        let ln = b.build(MirType::I32, Op::Load(pn, MirType::I32));
        let lp = b.build(MirType::F64, Op::Load(qp, MirType::F64));
        let xd = b.build(MirType::F64, Op::Cast(CastKind::FpExt, lx, MirType::F64));
        let nd = b.build(MirType::F64, Op::Cast(CastKind::SiToFp, ln, MirType::F64));
        let sum = b.build(MirType::F64, Op::Bin(BinOp::FAdd, xd, nd));
        let sum = b.build(MirType::F64, Op::Bin(BinOp::FAdd, sum, lp));
        let four = b.build(MirType::F64, Op::ConstFloat(4.0, MirType::F64));
        let sum = b.build(MirType::F64, Op::Bin(BinOp::FMul, sum, four));
        let r = b.build(MirType::I64, Op::Cast(CastKind::FpToSi, sum, MirType::I64));
        b.ret(Some(r));
        let mut f = b.finish();
        verify(&f);
        // (2.5 + 7 + 0.25) · 4 = 39.
        assert_eq!(run(&f, &it), 39);
        assert!(Sroa.run_function(&mut f, &mut CfgAnalyses::default()));
        verify(&f);
        assert_eq!(aggregate_allocas(&f), 0);
        for ty in [MirType::F32, MirType::I32, MirType::F64] {
            assert_eq!(
                count(&f, |op| matches!(op, Op::Alloca { ty: t, .. } if *t == ty)),
                1,
                "{ty:?}"
            );
        }
        assert_eq!(run(&f, &it), 39);
    }

    /// A float field of a byte buffer that one path reads before any store: the buffer's own leaf is
    /// a byte (`Int(0)` in the interpreter) and a fresh `f32` leaf would be `Float(0.0)`, which `-x`
    /// tells apart. The slot must stay whole. The same shape over an `[1 x f32]` array — whose leaf
    /// already starts at `Float(0.0)` — is split.
    #[test]
    fn a_field_that_may_be_read_unwritten_keeps_a_byte_buffer_whole() {
        for (slot_ty, elem, split) in [
            (MirType::Array(Box::new(MirType::I8), 4), MirType::I8, false),
            (
                MirType::Array(Box::new(MirType::F32), 1),
                MirType::F32,
                true,
            ),
        ] {
            let mut it = Interner::new();
            let mut b = Builder::new(it.intern("main"), MirType::I64);
            let s = b.alloca(slot_ty);
            let z = i64c(&mut b, 0);
            let px = gep(&mut b, s, z, elem);
            let skip = b.build(MirType::I1, Op::ConstInt(1, MirType::I1));
            let write = b.new_block();
            let read = b.new_block();
            b.cond_br(skip, read, vec![], write, vec![]);
            b.switch_to(write);
            let v = b.build(MirType::F32, Op::ConstFloat(1.5, MirType::F32));
            b.build_void(Op::Store { ptr: px, value: v });
            b.br(read, vec![]);
            b.switch_to(read);
            let x = b.build(MirType::F32, Op::Load(px, MirType::F32));
            let nx = b.build(MirType::F32, Op::Neg(x));
            let r = b.build(MirType::I64, Op::Cast(CastKind::FpToSi, nx, MirType::I64));
            b.ret(Some(r));
            let mut f = b.finish();
            verify(&f);
            assert_eq!(run(&f, &it), 0);
            assert_eq!(
                Sroa.run_function(&mut f, &mut CfgAnalyses::default()),
                split
            );
            verify(&f);
            assert_eq!(aggregate_allocas(&f), usize::from(!split));
            assert_eq!(run(&f, &it), 0);
        }
    }

    /// Every way an aggregate slot has to stay in memory.
    #[test]
    fn escaping_dynamic_punned_and_misaligned_slots_stay_whole() {
        let cases = [
            "dynamic",
            "call",
            "stored",
            "edge",
            "punned",
            "overlap",
            "stride",
            "vector",
            "shared",
            "out_of_bounds",
        ];
        for case in cases {
            let mut it = Interner::new();
            let mut b = Builder::new(it.intern("f"), MirType::Void);
            let dynamic = b.add_param(MirType::I64);
            let out = b.add_param(MirType::Ptr);
            let s = if case == "shared" {
                b.alloca_in(MirType::Array(Box::new(MirType::F32), 4), AddrSpace::Shared)
            } else if case == "overlap" || case == "punned" {
                b.alloca(MirType::Array(Box::new(MirType::I8), 8))
            } else {
                b.alloca(MirType::Array(Box::new(MirType::F32), 4))
            };
            let one = i64c(&mut b, 1);
            let four = i64c(&mut b, 4);
            // Element 1 — byte 1 of the byte buffers, so their refusal comes from the type and
            // overlap checks below rather than from a stride mismatch.
            let unit = if case == "overlap" || case == "punned" {
                MirType::I8
            } else {
                MirType::F32
            };
            let p1 = gep(&mut b, s, one, unit);
            let fv = b.build(MirType::F32, Op::ConstFloat(1.0, MirType::F32));
            b.build_void(Op::Store { ptr: p1, value: fv });
            match case {
                "dynamic" => {
                    let pd = gep(&mut b, s, dynamic, MirType::F32);
                    b.build_void(Op::Store { ptr: pd, value: fv });
                }
                "call" => b.build_void(Op::Call {
                    func: it.intern("sink"),
                    args: vec![p1],
                }),
                "stored" => b.build_void(Op::Store { ptr: out, value: s }),
                "edge" => {
                    let next = b.new_block();
                    let _ = b.block_param(next, MirType::Ptr);
                    b.br(next, vec![p1]);
                    b.switch_to(next);
                }
                // An `i32` and an `f32` at one byte offset of a byte buffer (`p1` is byte 1 here,
                // so read it back at the same position with a different type).
                "punned" => {
                    let _ = b.build(MirType::I32, Op::Load(p1, MirType::I32));
                }
                // An `i64` at byte 0 covers the `f32` at byte 1.
                "overlap" => {
                    let z = i64c(&mut b, 0);
                    let p0 = gep(&mut b, s, z, MirType::I8);
                    let _ = b.build(MirType::I64, Op::Load(p0, MirType::I64));
                }
                // Byte 4 of an `[4 x f32]` is element 1 natively but interpreter slot 4 (past the
                // end): the two backends already disagree, so the split must not pick one.
                "stride" => {
                    let pb = gep(&mut b, s, four, MirType::I8);
                    let _ = b.build(MirType::F32, Op::Load(pb, MirType::F32));
                }
                "vector" => {
                    let vt = MirType::Vec(Box::new(MirType::F32), 4);
                    let z = i64c(&mut b, 0);
                    let p0 = gep(&mut b, s, z, MirType::F32);
                    let _ = b.build(vt.clone(), Op::Load(p0, vt));
                }
                "shared" => {}
                "out_of_bounds" => {
                    let p4 = gep(&mut b, s, four, MirType::F32);
                    b.build_void(Op::Store { ptr: p4, value: fv });
                }
                _ => unreachable!(),
            }
            b.ret(None);
            let mut f = b.finish();
            verify(&f);
            assert!(
                !Sroa.run_function(&mut f, &mut CfgAnalyses::default()),
                "{case}"
            );
            assert_eq!(aggregate_allocas(&f), 1, "{case}");
            verify(&f);
        }
    }

    /// A nested array (`[2 x [2 x i32]]`) reached through a two-level `gep` chain: the chain's slot
    /// arithmetic (`slot_count([2 x i32]) = 2`) and byte arithmetic (`8` bytes per row) agree, so each
    /// of the four elements is its own leaf. An array of struct buffers (`[2 x [8 x i8]]`) works the
    /// same way with byte-sized leaves.
    #[test]
    fn a_nested_constant_chain_is_split_per_element() {
        let mut it = Interner::new();
        let mut b = Builder::new(it.intern("main"), MirType::I64);
        let row = MirType::Array(Box::new(MirType::I32), 2);
        let m = b.alloca(MirType::Array(Box::new(row.clone()), 2));
        let mut ptrs = Vec::new();
        for i in 0..2 {
            let ii = i64c(&mut b, i);
            let r = gep(&mut b, m, ii, row.clone());
            for j in 0..2 {
                let jj = i64c(&mut b, j);
                let p = gep(&mut b, r, jj, MirType::I32);
                let v = b.build(MirType::I32, Op::ConstInt(10 * i + j + 1, MirType::I32));
                b.build_void(Op::Store { ptr: p, value: v });
                ptrs.push(p);
            }
        }
        // m = [[1, 2], [11, 12]]; return m[0][0]·1000 + m[0][1]·100 + m[1][0]·10... as a checksum.
        let mut acc = b.build(MirType::I64, Op::ConstInt(0, MirType::I64));
        for (k, p) in ptrs.iter().enumerate() {
            let v = b.build(MirType::I32, Op::Load(*p, MirType::I32));
            let w = b.build(MirType::I64, Op::Cast(CastKind::SExt, v, MirType::I64));
            let scale = i64c(&mut b, [1000, 100, 10, 1][k]);
            let t = b.build(MirType::I64, Op::Bin(BinOp::Mul, w, scale));
            acc = b.build(MirType::I64, Op::Bin(BinOp::Add, acc, t));
        }
        b.ret(Some(acc));
        let mut f = b.finish();
        verify(&f);
        // 1·1000 + 2·100 + 11·10 + 12·1 = 1322.
        assert_eq!(run(&f, &it), 1322);
        assert!(Sroa.run_function(&mut f, &mut CfgAnalyses::default()));
        verify(&f);
        assert_eq!(aggregate_allocas(&f), 0);
        assert_eq!(
            count(&f, |op| matches!(
                op,
                Op::Alloca {
                    ty: MirType::I32,
                    ..
                }
            )),
            4
        );
        assert_eq!(run(&f, &it), 1322);
    }

    /// `struct S { n: i32, v: [f32; 4] }` as the front end lays it out: a 24-byte buffer, `n` at byte
    /// 0 and `v[j]` reached as `gep(gep(s, 8, i8), j, f32)` — interpreter slot `8 + j` but native byte
    /// `8 + 4j`. The two models disagree on where `v[j]` lives, yet each leaf is one cell in both, so
    /// the slot still splits into five leaves.
    #[test]
    fn a_struct_with_an_array_field_splits_per_cell() {
        let mut it = Interner::new();
        let mut b = Builder::new(it.intern("main"), MirType::I64);
        let s = b.alloca(MirType::Array(Box::new(MirType::I8), 24));
        let zero = i64c(&mut b, 0);
        let eight = i64c(&mut b, 8);
        let pn = gep(&mut b, s, zero, MirType::I8);
        let field = gep(&mut b, s, eight, MirType::I8);
        let n = b.build(MirType::I32, Op::ConstInt(3, MirType::I32));
        b.build_void(Op::Store { ptr: pn, value: n });
        let mut cells = Vec::new();
        for j in 0..4 {
            let jj = i64c(&mut b, j);
            let p = gep(&mut b, field, jj, MirType::F32);
            let v = b.build(MirType::F32, Op::ConstFloat(0.5 * j as f64, MirType::F32));
            b.build_void(Op::Store { ptr: p, value: v });
            cells.push(p);
        }
        // n · (v[0] + v[1] + v[2] + v[3]) · 4 = 3 · 3 · 4 = 36.
        let mut sum = b.build(MirType::F32, Op::ConstFloat(0.0, MirType::F32));
        for p in cells {
            let v = b.build(MirType::F32, Op::Load(p, MirType::F32));
            sum = b.build(MirType::F32, Op::Bin(BinOp::FAdd, sum, v));
        }
        let ln = b.build(MirType::I32, Op::Load(pn, MirType::I32));
        let nf = b.build(MirType::F32, Op::Cast(CastKind::SiToFp, ln, MirType::F32));
        let four = b.build(MirType::F32, Op::ConstFloat(4.0, MirType::F32));
        let t = b.build(MirType::F32, Op::Bin(BinOp::FMul, sum, nf));
        let t = b.build(MirType::F32, Op::Bin(BinOp::FMul, t, four));
        let r = b.build(MirType::I64, Op::Cast(CastKind::FpToSi, t, MirType::I64));
        b.ret(Some(r));
        let mut f = b.finish();
        verify(&f);
        assert_eq!(run(&f, &it), 36);
        assert!(Sroa.run_function(&mut f, &mut CfgAnalyses::default()));
        verify(&f);
        assert_eq!(aggregate_allocas(&f), 0);
        assert_eq!(count(&f, |op| matches!(op, Op::Alloca { .. })), 5);
        assert_eq!(run(&f, &it), 36);
    }

    #[test]
    fn const_indices_are_read_at_their_declared_width() {
        assert_eq!(const_index(0xFFFF_FFFF, &MirType::I32), Some(-1));
        assert_eq!(const_index(3, &MirType::I64), Some(3));
        assert_eq!(const_index(3, &MirType::I1), Some(1));
        assert_eq!(const_index(1, &MirType::F32), None);
        let blob = MirType::Array(Box::new(MirType::I8), 16);
        assert_eq!(leaf_at_slot(&blob, 9), Some((9, &MirType::I8)));
        assert_eq!(leaf_at_slot(&blob, 16), None);
        let grid = MirType::Array(Box::new(MirType::Array(Box::new(MirType::F64), 3)), 2);
        assert_eq!(slot_count(&grid), Some(6));
        assert_eq!(leaf_at_slot(&grid, 4), Some((32, &MirType::F64)));
    }
}
