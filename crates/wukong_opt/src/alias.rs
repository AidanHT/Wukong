//! Alias and provenance analysis for MIR pointers.
//!
//! `mir_build` erases every pointer-ish source type — `Ty::Ptr`, `Ty::Ref`, `Ty::Tensor` and
//! `Ty::Slice` all collapse to a bare [`MirType::Ptr`] — so nothing downstream can tell one buffer
//! from another by *type*. What survives is **provenance**: the SSA chain that produced the address.
//! This module recovers that chain and answers the one question every memory transform needs —
//! *can a store through `q` be seen by a load through `p`?*
//!
//! # The lattice
//!
//! Every value of pointer type is classified into a [`Prov`]: the `alloca` it came from, the
//! parameter it came from, the `.rodata` blob it came from, or [`Prov::Unknown`]. A `gep` inherits
//! its base's provenance and, when the index is a constant, tracks the exact byte offset; anything
//! else — a pointer loaded out of memory, a call result, an `inttoptr`, a block parameter — is
//! `Unknown`, which may-aliases everything.
//!
//! # What is actually guaranteed (and what is not)
//!
//! **SOUNDNESS FIRST.** An unsound no-alias answer is a miscompile; a spurious may-alias answer only
//! costs speed. Every fact below is justified, and anything not on this list returns "may alias":
//!
//! * **Two distinct `alloca`s never alias.** Each `Op::Alloca` gets its own stack slot in Cranelift
//!   (`create_sized_stack_slot`) and its own disjoint slot range in the interpreter. True regardless
//!   of escape.
//! * **An `alloca` of *this* function never aliases a *parameter* of this function.** The slot is
//!   created after entry, so its address cannot have been handed to us by the caller — not even
//!   through recursion, where the incoming pointer belongs to a *different* frame's slot.
//! * **An `alloca` never aliases a `.rodata` blob**, and **two distinct blobs never alias** (each
//!   `Op::GlobalAddr` names a separate `StaticData`).
//! * **A non-escaping `alloca` aliases nothing but itself** — including `Unknown` pointers and the
//!   memory a `call` may write. "Non-escaping" is checked literally: the address may appear only as
//!   the pointer operand of a `gep`, `load` or `store`. Any other use (stored *as a value*, passed
//!   to a call, returned, branched with, `ptrtoint`-ed, `select`-ed) marks it escaped.
//! * **Disjoint constant offsets off a common base do not overlap** — `[off_a, off_a+size_a)` vs
//!   `[off_b, off_b+size_b)`, via [`AliasInfo::may_alias_sized`].
//!
//! **Two distinct pointer/slice/tensor parameters MAY alias.** Wukong does not promise otherwise:
//! nothing rejects `f(a, a)`, and there is no `restrict`/`&mut` annotation to carry the promise. So
//! [`AliasInfo::may_alias`] answers `true` for `Param(i)` vs `Param(j)`. (`docs/roadmap.md` notes
//! that the AST loop vectorizer *assumes* distinct array parameters do not alias — that assumption
//! is informal and unchecked, and this module deliberately does not adopt it. Making it real needs a
//! language-level promise, not an analysis.)
//!
//! # Deliberate conservatism
//!
//! Block parameters are `Unknown` rather than the meet of their incoming arguments. A pointer that
//! flows through a block parameter therefore loses its provenance, and any `alloca` whose address
//! reaches a branch argument is marked escaped. This costs precision on loop-carried pointers and
//! buys a single linear pass with no fixpoint — revisit only if a measurement demands it.

use wukong_mir::{Function, MirType, Op, ValueId};
use wukong_span::Symbol;

/// Where a pointer came from. The base object, not the address.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Prov {
    /// The stack slot created by the `Op::Alloca` whose result is this `ValueId`.
    Alloca(ValueId),
    /// Function parameter number `i` (an index into `Function::params`).
    Param(u32),
    /// The read-only `.rodata` blob named by this symbol (`Op::GlobalAddr`).
    Global(Symbol),
    /// Provenance not tracked: loaded from memory, produced by a call, cast from an integer,
    /// carried by a block parameter, or merged from two different bases. May alias anything.
    Unknown,
}

/// Byte size of a MIR type, mirroring `wukong_codegen_cranelift::size_of` and the interpreter's
/// byte layout. `None` when the extent overflows a `u32` (an absurd array length) — callers treat
/// that as "size unknown", i.e. conservatively large.
pub fn type_bytes(t: &MirType) -> Option<u32> {
    byte_size(t)
}

fn byte_size(t: &MirType) -> Option<u32> {
    Some(match t {
        MirType::I1 | MirType::I8 => 1,
        MirType::I16 | MirType::F16 | MirType::BF16 => 2,
        MirType::I32 | MirType::F32 => 4,
        MirType::I64 | MirType::F64 | MirType::Ptr => 8,
        MirType::Vec(e, n) | MirType::Array(e, n) => byte_size(e)?.checked_mul(*n)?,
        MirType::Void => 0,
    })
}

/// Everything the analysis knows about one SSA value, in one flat record.
///
/// The result is a single `Vec<Fact>` indexed by `ValueId` rather than a family of hash maps,
/// because `Cse`, `Dse` and `Licm` each call [`AliasInfo::analyze`] on every invocation of every
/// fixpoint iteration: the analysis sits on the compiler's hot path and must not hash or allocate
/// per value. (Measured: the hash-map form cost ~9% more optimizer time than this one.)
#[derive(Clone, Copy)]
struct Fact {
    /// Base object this value points into.
    prov: Prov,
    /// Constant byte offset from the base, or [`UNKNOWN_OFF`] for "somewhere in that object".
    offset: i64,
    /// The value of this `Op::ConstInt`, or [`UNKNOWN_OFF`] — folds `gep` indices without a second
    /// side table.
    konst: i64,
    /// For an `Op::Alloca` result, the byte size of its slot; `u32::MAX` when this is not an alloca
    /// or its extent is not representable.
    alloca_bytes: u32,
    /// Set on an `Op::Alloca` result whose address is used for anything other than `gep`/`load`/
    /// `store` addressing, and which may therefore be reached through an `Unknown` pointer or by a
    /// callee.
    escaped: bool,
}

/// Sentinel for "no constant / no known offset". A real offset this large is unreachable (it would
/// need an object of 2^63 bytes) and every producer uses checked arithmetic, so it cannot collide.
const UNKNOWN_OFF: i64 = i64::MIN;

impl Default for Fact {
    fn default() -> Fact {
        Fact {
            prov: Prov::Unknown,
            offset: UNKNOWN_OFF,
            konst: UNKNOWN_OFF,
            alloca_bytes: u32::MAX,
            escaped: false,
        }
    }
}

/// The result of running [`AliasInfo::analyze`] over one function: one [`Fact`] per SSA value.
pub struct AliasInfo {
    facts: Vec<Fact>,
}

/// A pointer operand together with how many bytes the access touches. `size` is `None` when the
/// width is not known, which forces every offset-disjointness test to give up.
#[derive(Clone, Copy)]
struct Access {
    v: ValueId,
    size: Option<u32>,
}

impl AliasInfo {
    /// Classify every value in `f`: three linear passes over the instructions, one allocation, no
    /// hashing. No fixpoint (see the module note on block parameters).
    pub fn analyze(f: &Function) -> AliasInfo {
        let n = f.value_types.len();
        let mut info = AliasInfo {
            facts: vec![Fact::default(); n],
        };

        for (i, p) in f.params.iter().enumerate() {
            if let Some(fact) = info.facts.get_mut(p.0 as usize) {
                fact.prov = Prov::Param(i as u32);
                fact.offset = 0;
            }
        }

        // Constant integers, for resolving `gep` indices. Collected first so a `gep` whose index is
        // defined later in layout order (possible after LICM moved the constant) still folds.
        for b in &f.blocks {
            for inst in &b.insts {
                if let (Some(r), Op::ConstInt(k, _)) = (inst.result, &inst.op) {
                    if let (Some(fact), Ok(k)) =
                        (info.facts.get_mut(r.0 as usize), i64::try_from(*k))
                    {
                        fact.konst = k;
                    }
                }
            }
        }

        // Provenance. `alloca` and `global_addr` seed bases; `gep` walks them. Because MIR is SSA
        // and every definition precedes its uses within a block, and blocks are emitted in a order
        // where a `gep` chain's links stay together, a single layout-order pass settles the common
        // shapes; anything it cannot see stays `Unknown`, which is the safe answer.
        for b in &f.blocks {
            for inst in &b.insts {
                let Some(res) = inst.result else { continue };
                let r = res.0 as usize;
                if r >= n {
                    continue;
                }
                match &inst.op {
                    Op::Alloca(ty) => {
                        let fact = &mut info.facts[r];
                        fact.prov = Prov::Alloca(res);
                        fact.offset = 0;
                        fact.alloca_bytes = byte_size(ty).unwrap_or(u32::MAX);
                    }
                    Op::GlobalAddr(sym) => {
                        let fact = &mut info.facts[r];
                        fact.prov = Prov::Global(*sym);
                        fact.offset = 0;
                    }
                    Op::Gep { ptr, index, elem } => {
                        let base = info.prov(*ptr);
                        let base_off = info.offset_of(*ptr);
                        let idx = info
                            .facts
                            .get(index.0 as usize)
                            .map(|fa| fa.konst)
                            .unwrap_or(UNKNOWN_OFF);
                        let off = match (base_off, idx) {
                            (Some(o), k) if k != UNKNOWN_OFF => byte_size(elem)
                                .and_then(|e| k.checked_mul(e as i64))
                                .and_then(|d| o.checked_add(d))
                                .filter(|d| *d != UNKNOWN_OFF),
                            _ => None,
                        };
                        let fact = &mut info.facts[r];
                        fact.prov = base;
                        fact.offset = off.unwrap_or(UNKNOWN_OFF);
                    }
                    _ => {}
                }
            }
        }

        // Escape. Every use of an alloca-derived value that is not "the pointer operand of a gep,
        // load or store" loses track of the address, so the slot must be assumed reachable through
        // an `Unknown` pointer and writable by any callee. A function with no `alloca` has nothing
        // to escape, so the whole scan is skipped — worth checking because it is a third of the
        // analysis and this runs on the compiler's hot path.
        if !info.facts.iter().any(|fa| matches!(fa.prov, Prov::Alloca(_))) {
            return info;
        }
        for b in &f.blocks {
            for inst in &b.insts {
                match &inst.op {
                    // Addressing uses: these keep provenance, so they do not escape the base.
                    // `Gep`'s *index* still escapes (it is an integer use of the address).
                    Op::Gep { ptr: _, index, .. } => info.mark_escaped(*index),
                    Op::Load(_, _) => {}
                    Op::Store { ptr: _, value } => info.mark_escaped(*value),
                    other => crate::each_op_use(other, &mut |v| info.mark_escaped(v)),
                }
            }
            crate::each_term_use(&b.term, &mut |v| info.mark_escaped(v));
        }
        info
    }

    fn mark_escaped(&mut self, v: ValueId) {
        if let Prov::Alloca(a) = self.prov(v) {
            if let Some(fact) = self.facts.get_mut(a.0 as usize) {
                fact.escaped = true;
            }
        }
    }

    /// The base object `v` points into.
    pub fn prov(&self, v: ValueId) -> Prov {
        self.facts
            .get(v.0 as usize)
            .map(|f| f.prov)
            .unwrap_or(Prov::Unknown)
    }

    fn offset_of(&self, v: ValueId) -> Option<i64> {
        self.facts
            .get(v.0 as usize)
            .map(|f| f.offset)
            .filter(|o| *o != UNKNOWN_OFF)
    }

    /// Does the address of this `alloca` leave the set of uses we can see? A non-escaping slot is
    /// unreachable through any `Unknown` pointer and cannot be written by a callee. An id we hold no
    /// fact for answers "escaped" — the conservative direction.
    pub fn alloca_escapes(&self, a: ValueId) -> bool {
        self.facts
            .get(a.0 as usize)
            .map(|f| f.escaped)
            .unwrap_or(true)
    }

    /// Is `v` a pointer into an `alloca` of this function whose address never escapes?
    pub fn is_private_stack(&self, v: ValueId) -> bool {
        matches!(self.prov(v), Prov::Alloca(a) if !self.alloca_escapes(a))
    }

    /// **The query.** Can accesses through `a` and `b` touch the same byte?
    ///
    /// Conservative: `true` whenever the analysis cannot prove otherwise. Access widths are not
    /// known here, so two disjoint constant offsets off a common base still answer `true` — use
    /// [`AliasInfo::may_alias_sized`] when the widths are available.
    pub fn may_alias(&self, a: ValueId, b: ValueId) -> bool {
        self.may_alias_access(
            Access { v: a, size: None },
            Access { v: b, size: None },
        )
    }

    /// [`AliasInfo::may_alias`] refined by the byte width of each access, which lets disjoint
    /// constant offsets off a common base answer `false`.
    pub fn may_alias_sized(&self, a: ValueId, a_bytes: u32, b: ValueId, b_bytes: u32) -> bool {
        self.may_alias_access(
            Access {
                v: a,
                size: Some(a_bytes),
            },
            Access {
                v: b,
                size: Some(b_bytes),
            },
        )
    }

    fn may_alias_access(&self, a: Access, b: Access) -> bool {
        let (pa, pb) = (self.prov(a.v), self.prov(b.v));

        // Same base: overlap unless both offsets and both widths are known and the byte intervals
        // are disjoint.
        if pa == pb && pa != Prov::Unknown {
            let (Some(oa), Some(ob)) = (self.offset_of(a.v), self.offset_of(b.v)) else {
                return true;
            };
            let (Some(sa), Some(sb)) = (a.size, b.size) else {
                return true;
            };
            let (ea, eb) = (oa.saturating_add(sa as i64), ob.saturating_add(sb as i64));
            return oa < eb && ob < ea;
        }

        match (pa, pb) {
            // Distinct stack slots are distinct storage, escape or not.
            (Prov::Alloca(_), Prov::Alloca(_)) => false,
            // A slot created after entry cannot be a pointer the caller passed in — not even under
            // recursion, where the incoming pointer belongs to another frame's slot.
            (Prov::Alloca(_), Prov::Param(_)) | (Prov::Param(_), Prov::Alloca(_)) => false,
            // A stack slot is never a `.rodata` blob.
            (Prov::Alloca(_), Prov::Global(_)) | (Prov::Global(_), Prov::Alloca(_)) => false,
            // Distinct blobs are distinct storage.
            (Prov::Global(_), Prov::Global(_)) => false,
            // A slot whose address never escaped cannot be reached by any pointer we lost track of.
            (Prov::Alloca(a), Prov::Unknown) | (Prov::Unknown, Prov::Alloca(a)) => {
                self.alloca_escapes(a)
            }
            // Wukong does NOT promise that two pointer parameters are distinct — `f(a, a)` is legal
            // — so this stays `true`. It is the single fact a language-level `restrict` would buy.
            (Prov::Param(_), Prov::Param(_)) => true,
            // A `*u8` handed to us could well be a string literal's address.
            (Prov::Param(_), Prov::Global(_)) | (Prov::Global(_), Prov::Param(_)) => true,
            _ => true,
        }
    }

    /// Can executing `op` write memory that an access of `bytes` bytes through `ptr` would read?
    ///
    /// A `call` (or a vector-kernel call, which stores through the pointers handed to it) can write
    /// anything the caller can name — *except* a stack slot of this function whose address never
    /// escaped, which no callee can have an address for.
    pub fn may_clobber(&self, ptr: ValueId, bytes: u32, op: &Op) -> bool {
        match op {
            Op::Store { ptr: q, value: _ } => self.may_alias_sized(ptr, bytes, *q, u32::MAX),
            Op::Call { .. } | Op::VecKernelCall { .. } => !self.is_private_stack(ptr),
            _ => false,
        }
    }

    /// Is an access of `bytes` bytes at `ptr` safe to perform **speculatively** — on a path where
    /// the original program might not have performed it?
    ///
    /// Only one class qualifies today: an address inside an `alloca` of this function at a known
    /// in-bounds offset. A stack slot is live for the whole function, so reading it can neither
    /// fault nor observe anything the program did not already own. Every other pointer — including
    /// a parameter, whose validity is the *caller's* business — answers `false`.
    pub fn is_dereferenceable(&self, ptr: ValueId, bytes: u32) -> bool {
        let Prov::Alloca(a) = self.prov(ptr) else {
            return false;
        };
        let Some(off) = self.offset_of(ptr) else {
            return false;
        };
        let slot = self
            .facts
            .get(a.0 as usize)
            .map(|f| f.alloca_bytes)
            .unwrap_or(u32::MAX);
        if slot == u32::MAX {
            return false; // extent not representable: never claim dereferenceable
        }
        off >= 0 && (off as u128) + (bytes as u128) <= slot as u128
    }

    /// Byte width of the value an `Op::Load`/`Op::Store` moves, when it is representable.
    pub fn access_bytes(f: &Function, op: &Op) -> Option<u32> {
        match op {
            Op::Load(_, ty) => byte_size(ty),
            Op::Store { value, .. } => byte_size(f.value_type(*value)),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use wukong_mir::{BinOp, Builder};
    use wukong_span::Interner;

    fn blob(bytes: u32) -> MirType {
        MirType::Array(Box::new(MirType::I8), bytes)
    }

    fn gep(b: &mut Builder, ptr: ValueId, index: ValueId) -> ValueId {
        b.build(
            MirType::Ptr,
            Op::Gep {
                ptr,
                index,
                elem: MirType::I8,
            },
        )
    }

    #[test]
    fn distinct_allocas_and_disjoint_constant_offsets_do_not_alias() {
        let mut it = Interner::new();
        let mut b = Builder::new(it.intern("f"), MirType::Void);
        let a = b.alloca(blob(32));
        let c = b.alloca(blob(32));
        let z = b.build(MirType::I64, Op::ConstInt(0, MirType::I64));
        let eight = b.build(MirType::I64, Op::ConstInt(8, MirType::I64));
        let a0 = gep(&mut b, a, z);
        let a8 = gep(&mut b, a, eight);
        let c0 = gep(&mut b, c, z);
        let v = b.build(MirType::I64, Op::Load(a0, MirType::I64));
        b.build_void(Op::Store { ptr: c0, value: v });
        b.build_void(Op::Store { ptr: a8, value: v });
        b.ret(None);
        let f = b.finish();
        let info = AliasInfo::analyze(&f);

        assert!(!info.may_alias(a, c));
        assert!(info.may_alias(a, a));
        assert!(!info.may_alias(a0, c0));
        // a+0 and a+8, both 8 bytes wide: adjacent, not overlapping.
        assert!(!info.may_alias_sized(a0, 8, a8, 8));
        // 16 bytes wide at +0 does reach +8.
        assert!(info.may_alias_sized(a0, 16, a8, 8));
        // Without widths the same-base case must give up.
        assert!(info.may_alias(a0, a8));
    }

    #[test]
    fn stored_pointer_escapes_its_slot() {
        let mut it = Interner::new();
        let mut b = Builder::new(it.intern("g"), MirType::Void);
        let a = b.alloca(blob(16));
        let c = b.alloca(blob(16));
        // Storing `a`'s address into `c` publishes it.
        b.build_void(Op::Store { ptr: c, value: a });
        b.ret(None);
        let f = b.finish();
        let info = AliasInfo::analyze(&f);

        assert!(info.alloca_escapes(a));
        assert!(!info.alloca_escapes(c));
        assert!(!info.is_private_stack(a));
        assert!(info.is_private_stack(c));
        // An escaped slot may be reached by a pointer we lost track of; a private one may not.
        let unknown = ValueId(u32::MAX - 1); // out of range -> Prov::Unknown
        assert!(info.may_alias(a, unknown));
        assert!(!info.may_alias(c, unknown));
        // ... but the two slots still never alias each other.
        assert!(!info.may_alias(a, c));
        // Nor can a call reach the private one.
        let call = Op::Call {
            func: it.intern("sink"),
            args: vec![],
        };
        assert!(info.may_clobber(a, 8, &call));
        assert!(!info.may_clobber(c, 8, &call));
    }

    #[test]
    fn params_may_alias_each_other_but_never_a_local_slot() {
        let mut it = Interner::new();
        let mut b = Builder::new(it.intern("h"), MirType::Void);
        let p0 = b.add_param(MirType::Ptr);
        let p1 = b.add_param(MirType::Ptr);
        let a = b.alloca(blob(16));
        let v = b.build(MirType::I64, Op::Load(p0, MirType::I64));
        b.build_void(Op::Store { ptr: a, value: v });
        b.ret(None);
        let f = b.finish();
        let info = AliasInfo::analyze(&f);

        assert_eq!(info.prov(p0), Prov::Param(0));
        assert_eq!(info.prov(p1), Prov::Param(1));
        // The language does not promise `f(x, x)` is impossible.
        assert!(info.may_alias(p0, p1));
        // But a slot this frame created cannot be something the caller handed us.
        assert!(!info.may_alias(a, p0));
        assert!(!info.may_alias(a, p1));
    }

    #[test]
    fn only_in_bounds_stack_offsets_are_dereferenceable() {
        let mut it = Interner::new();
        let mut b = Builder::new(it.intern("k"), MirType::Void);
        let p0 = b.add_param(MirType::Ptr);
        let a = b.alloca(blob(16));
        let eight = b.build(MirType::I64, Op::ConstInt(8, MirType::I64));
        let a8 = gep(&mut b, a, eight);
        let idx = b.build(MirType::I64, Op::Bin(BinOp::Add, eight, eight));
        let adyn = gep(&mut b, a, idx);
        b.ret(None);
        let f = b.finish();
        let info = AliasInfo::analyze(&f);

        assert!(info.is_dereferenceable(a, 16));
        assert!(!info.is_dereferenceable(a, 17));
        assert!(info.is_dereferenceable(a8, 8));
        assert!(!info.is_dereferenceable(a8, 9));
        // A non-constant offset is not provably in bounds ...
        assert!(!info.is_dereferenceable(adyn, 1));
        // ... and a parameter's validity is the caller's business, never ours.
        assert!(!info.is_dereferenceable(p0, 1));
    }

    #[test]
    fn a_branch_argument_escapes_the_slot_it_carries() {
        let mut it = Interner::new();
        let mut b = Builder::new(it.intern("m"), MirType::Void);
        let a = b.alloca(blob(16));
        let next = b.new_block();
        let bp = b.block_param(next, MirType::Ptr);
        b.br(next, vec![a]);
        b.switch_to(next);
        b.ret(None);
        let f = b.finish();
        let info = AliasInfo::analyze(&f);

        // The block parameter loses provenance, so the slot must be treated as published.
        assert_eq!(info.prov(bp), Prov::Unknown);
        assert!(info.alloca_escapes(a));
        assert!(info.may_alias(a, bp));
    }
}
