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
//! **Two distinct pointer/slice/tensor parameters MAY alias** — unless the source promised
//! otherwise. Nothing rejects `f(a, a)` for an ordinary parameter, so [`AliasInfo::may_alias`]
//! answers `true` for `Param(i)` vs `Param(j)` by default. The AST vectorizer separately requires
//! disjointness and recurrence checks before widening source loops; it grants no promise either.
//!
//! # The `@noalias` promise
//!
//! The language contract that changes this is a per-parameter attribute, `fn k(@noalias mut out:
//! []f32, a: []f32)`, carried to MIR as [`wukong_mir::ParamAttrs::noalias`]. It is C's `restrict`
//! and LLVM's `noalias` argument: *for the duration of the call, the storage the parameter points
//! into is accessed only through pointers derived from it.* Sema rejects the call sites where the
//! promise is provably broken (`E0316`); where it cannot tell, the promise is the programmer's, as
//! in C. Every fact below follows from the promise and from what this function itself does:
//!
//! * **A promised parameter is disjoint from every other parameter** — `Param(i)` vs `Param(j)`,
//!   `i != j`, answers `false` when *either* side was promised. One promise is enough: the other
//!   parameter's storage is "something else the caller can name".
//! * **A `@noalias` slice has a data provenance of its own.** A slice parameter points at the
//!   caller's 16-byte fat pointer; a pointer-typed load at offset 0 of it is the data pointer. When
//!   nothing in this function writes that header or lets its address escape, every such load yields
//!   the same pointer, classified [`Prov::ParamData`]. Its elements are disjoint from every other
//!   parameter's storage, from this frame's slots (the caller filled the header before they
//!   existed), and from the slice's own header (a slice never views its own fat pointer). If the
//!   header *is* written or escapes, every `ParamData` fact is demoted back to `Unknown`.
//! * **A promised parameter whose storage never escapes aliases nothing untracked** — the analogue
//!   of a private stack slot. If no value derived from the parameter is stored, passed to a call,
//!   branched with, cast or selected, then no pointer this function lost track of can have been
//!   derived from it, and the promise excludes every pointer not derived from it. So it is
//!   disjoint from `Unknown`, and no call can write it ([`AliasInfo::may_clobber`]).
//!   [`AliasInfo::promised_storage`] exposes exactly this class to the backends, which map it to
//!   disjoint alias regions.
//!
//! None of this fires for a function whose parameters promise nothing: the per-parameter table is
//! not even allocated, so every pre-existing program analyzes exactly as before.
//!
//! # Deliberate conservatism
//!
//! Block parameters are `Unknown` rather than the meet of their incoming arguments. A pointer that
//! flows through a block parameter therefore loses its provenance, and any `alloca` whose address
//! reaches a branch argument is marked escaped. This costs precision on loop-carried pointers and
//! keeps the provenance sweep a simple forward walk — revisit only if a measurement demands it.
//!
//! The provenance sweep *does* iterate, but only when it must. The IR contract is dominance, not
//! block layout, so a `gep`'s base may be defined in a block laid out after it; the sweep repeats
//! until no fact moves. Measured over `tests/run` (2682 `analyze` calls): 2667 settle in one round,
//! 15 take three, and the `MAX_PROV_ROUNDS` cap is never reached.
//!
//! # The query API
//!
//! Build it once per function with [`AliasInfo::analyze`] (a linear provenance sweep, a constant
//! sweep and an escape sweep; one allocation, no hashing), then ask:
//!
//! | question | call |
//! |---|---|
//! | can these two accesses touch the same byte? | [`AliasInfo::may_alias`] |
//! | ... knowing both access widths | [`AliasInfo::may_alias_sized`] |
//! | can this `Op` write what `ptr` reads? | [`AliasInfo::may_clobber`] |
//! | may this access run on a path the program would not have taken? | [`AliasInfo::is_dereferenceable`] |
//! | where did this pointer come from? | [`AliasInfo::prov`] |
//! | is this a stack slot no callee can name? | [`AliasInfo::is_private_stack`] |
//! | is this `@noalias` storage nothing untracked can reach? | [`AliasInfo::promised_storage`] |
//! | can no call in this function read or write this location? | [`AliasInfo::is_call_proof`] |
//! | is this a `@noalias` slice header nothing writes? | [`AliasInfo::is_invariant_slice_header`] |
//! | do two loop bases denote promised-disjoint storage? | [`promised_disjoint_bases`] |
//!
//! A **loop vectorizer** wants the first three. To decide whether two array references in one loop
//! body carry a dependence, ask `may_alias` on their *base* pointers first: `false` settles it for
//! the whole loop with no index reasoning at all (this is what proves a local scratch buffer
//! independent of everything else). Only when the bases may alias do you need a subscript test.
//! `may_clobber` answers the same question against a whole instruction, including calls. Never
//! invert an answer: `true` means "not proven", not "proven to alias".
//!
//! # Cost
//!
//! It is **not free**: building one used to cost ~15-19% of optimizer time over `tests/run`
//! (`CHANGELOG.md` carries the ABBA numbers), with `Cse`, `Dse` and `Licm` each building their own
//! on every invocation. An alias fact that outlives the IR it described is an unsound no-alias
//! answer, so an instance is never carried across a mutation. Within that rule the pass manager's
//! *managed* `CfgAnalyses` shares one instance between those passes until any pass reports a change
//! (`CfgAnalyses::alias` / `note_mutation`): a pass takes the facts once, before it mutates, and the
//! manager drops them the moment it does. A cache built by hand never shares.

use wukong_mir::{Function, MirType, Op, ParamAttrs, ValueId};
use wukong_span::Symbol;

/// Where a pointer came from. The base object, not the address.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Prov {
    /// The stack slot created by the `Op::Alloca` whose result is this `ValueId`.
    Alloca(ValueId),
    /// Function parameter number `i` (an index into `Function::params`).
    Param(u32),
    /// The **elements** of `@noalias` slice parameter number `i`: a pointer-typed load from offset 0
    /// of that parameter's fat-pointer header, in a function that never writes the header nor lets
    /// its address escape — so every such load is the same pointer. Never produced for a parameter
    /// without the promise (see the module note).
    ParamData(u32),
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
    /// Round number in which this value's *definition* was visited by the provenance pass. Compared
    /// against the current round instead of being reset, so "have I seen the definition yet?" costs
    /// no clearing sweep between rounds. `0` means never.
    ///
    /// `u16`, not `usize`: the round never exceeds `MAX_PROV_ROUNDS`, and this way the stamp fits
    /// the padding `alloca_bytes`/`escaped` already leave, keeping `Fact` at 32 bytes. See
    /// `fact_stays_one_cache_friendly_record`.
    seen: u16,
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
            seen: 0,
        }
    }
}

/// Byte size of a `[]T` fat pointer: an 8-byte data pointer at offset 0 and an 8-byte `i64` length
/// at offset 8 (`wukong_mir_build`'s `SLICE_SIZE`).
const SLICE_HEADER_BYTES: u32 = 16;

/// What the analysis learned about one parameter's storage. Only built for a function that carries
/// a `@noalias` promise; [`AliasInfo`]'s table is empty otherwise, so the common case pays nothing.
#[derive(Clone, Copy, Default)]
struct ParamFact {
    /// What the source promised (see [`wukong_mir::ParamAttrs`]).
    attrs: ParamAttrs,
    /// A value with `Prov::Param(i)` provenance was used other than as the address of a `gep`,
    /// `load` or `store` — stored as a value, passed to a call, branched with, cast, selected.
    base_escaped: bool,
    /// The same, for a value with `Prov::ParamData(i)` provenance.
    data_escaped: bool,
    /// A store writes through a `Prov::Param(i)` address. For a slice, that is its header.
    base_written: bool,
    /// A store writes through a `Prov::ParamData(i)` address.
    data_written: bool,
}

/// What a function does to `@noalias` storage that nothing untracked can reach, over its whole
/// body — the answer of [`AliasInfo::promised_storage`].
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum PromisedStorage {
    /// Nothing in the function writes the parameter's storage.
    ReadOnly,
    /// Some store in the function writes it.
    Written,
}

/// Hard cap on provenance rounds. A `gep` chain is a DAG (SSA has no value cycles), so the pass
/// converges in at most its depth; the cap only bounds the pathological case, and exhausting it
/// falls back to "every slot escaped", which is sound.
const MAX_PROV_ROUNDS: u16 = 8;

/// The result of running [`AliasInfo::analyze`] over one function: one `Fact` per SSA value —
/// provenance, constant offset, alloca size and escape bit, in a flat vector indexed by `ValueId`.
pub struct AliasInfo {
    facts: Vec<Fact>,
    /// One record per parameter, but only when some parameter carries a `@noalias` promise; empty
    /// (no allocation) for every other function.
    params: Vec<ParamFact>,
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
        let params = if f.has_noalias_params() {
            (0..f.params.len())
                .map(|i| ParamFact {
                    attrs: f.param_attr(i),
                    ..ParamFact::default()
                })
                .collect()
        } else {
            Vec::new()
        };
        let mut info = AliasInfo {
            facts: vec![Fact::default(); n],
            params,
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

        // Provenance. `alloca` and `global_addr` seed bases; `gep` walks them.
        //
        // The IR contract is **dominance, not layout order**: `wukong_mir`'s verifier walks the
        // dominator tree precisely so a definition may sit in a block laid out *after* a block that
        // uses it. A single layout-order sweep would then read a not-yet-visited base as `Unknown`.
        // For `may_alias` that is merely imprecise, but it is *unsound* for the escape scan below,
        // which would see `call sink(%g)` with `%g` unclassified and fail to mark the slot `%g`
        // points into as escaped — reporting a published slot as private. So sweep until every
        // `gep` has seen its base, tracked with a round stamp so the common (already-ordered) case
        // still costs exactly one pass.
        let mut round = 0u16;
        let stalled = loop {
            round += 1;
            let mut late_base = false;
            let mut changed = false;

            for b in &f.blocks {
                for &p in &b.params {
                    if let Some(fact) = info.facts.get_mut(p.0 as usize) {
                        fact.seen = round;
                    }
                }
                for inst in &b.insts {
                    let Some(res) = inst.result else { continue };
                    let r = res.0 as usize;
                    if r >= n {
                        continue;
                    }
                    match &inst.op {
                        // A base's own classification is a fact that moves too: in round 1 a
                        // `gep` laid out before its base reads the default `Unknown` and so does
                        // not itself change, and only the base moves. Counting just the `gep`s
                        // ended the sweep there, leaving every such `gep` `Unknown` — so a load
                        // through it was not a read of the slot, `dse` saw the slot as never read,
                        // and deleted its constant-offset stores (an inlined call ahead of the
                        // slot's block was enough: `hardsigmoid_nan.wk`, -O2, recognition off).
                        Op::Alloca { ty, .. } => {
                            let fact = &mut info.facts[r];
                            changed |= fact.prov != Prov::Alloca(res) || fact.offset != 0;
                            fact.prov = Prov::Alloca(res);
                            fact.offset = 0;
                            fact.alloca_bytes = byte_size(ty).unwrap_or(u32::MAX);
                        }
                        Op::GlobalAddr(sym) => {
                            let fact = &mut info.facts[r];
                            changed |= fact.prov != Prov::Global(*sym) || fact.offset != 0;
                            fact.prov = Prov::Global(*sym);
                            fact.offset = 0;
                        }
                        Op::Gep { ptr, index, elem } => {
                            // A base whose own definition has not been visited this round is not
                            // yet classified; its `Unknown` is an artefact of layout, not a fact.
                            if info
                                .facts
                                .get(ptr.0 as usize)
                                .is_none_or(|fa| fa.seen != round)
                            {
                                late_base = true;
                            }
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
                            let off = off.unwrap_or(UNKNOWN_OFF);
                            let fact = &mut info.facts[r];
                            changed |= fact.prov != base || fact.offset != off;
                            fact.prov = base;
                            fact.offset = off;
                        }
                        // The data pointer of a `@noalias` slice: a pointer load at offset 0 of its
                        // header. Tentative — demoted after the escape scan if the header turns out
                        // to be written or published (`settle_param_data`). Only ever reached when a
                        // parameter carries the promise.
                        Op::Load(ptr, MirType::Ptr) if !info.params.is_empty() => {
                            if info
                                .facts
                                .get(ptr.0 as usize)
                                .is_none_or(|fa| fa.seen != round)
                            {
                                late_base = true;
                            }
                            let (prov, off) = match (info.prov(*ptr), info.offset_of(*ptr)) {
                                (Prov::Param(i), Some(0)) if info.slice_promised(i) => {
                                    (Prov::ParamData(i), 0)
                                }
                                _ => (Prov::Unknown, UNKNOWN_OFF),
                            };
                            let fact = &mut info.facts[r];
                            changed |= fact.prov != prov || fact.offset != off;
                            fact.prov = prov;
                            fact.offset = off;
                        }
                        _ => {}
                    }
                    info.facts[r].seen = round;
                }
            }

            // Every `gep` saw its base already classified, so this round used final values and
            // there is nothing a further round could add. The overwhelmingly common case: one pass.
            if !late_base {
                break false;
            }
            // Some base was defined in a later-laid-out block. `late_base` says nothing about
            // progress — it is a property of the *layout*, so it recurs identically every round and
            // must never be the termination test. The fixpoint test is whether a fact moved.
            if !changed {
                break false;
            }
            if round >= MAX_PROV_ROUNDS {
                break true;
            }
        };

        // Escape. Every use of an alloca-derived value that is not "the pointer operand of a gep,
        // load or store" loses track of the address, so the slot must be assumed reachable through
        // an `Unknown` pointer and writable by any callee. The same scan records, for a function
        // with a `@noalias` parameter, whether that parameter's storage escapes and whether it is
        // written. A function with neither an `alloca` nor a promise has nothing to learn here, so
        // the whole scan is skipped — worth checking because it is a third of the analysis and
        // this runs on the compiler's hot path.
        if info.params.is_empty()
            && !info
                .facts
                .iter()
                .any(|fa| matches!(fa.prov, Prov::Alloca(_)))
        {
            return info;
        }
        // Provenance never settled (see `MAX_PROV_ROUNDS`). Some `gep` may still be carrying a
        // spurious `Unknown`, so the escape scan below could miss the slot it belongs to. Concede
        // every slot and every parameter instead: "escaped" is always the safe answer, and a
        // conceded header demotes every `ParamData` fact.
        if stalled {
            for fa in info.facts.iter_mut() {
                fa.escaped = true;
            }
            for p in info.params.iter_mut() {
                p.base_escaped = true;
                p.data_escaped = true;
                p.base_written = true;
                p.data_written = true;
            }
            info.settle_param_data();
            return info;
        }
        for b in &f.blocks {
            for inst in &b.insts {
                match &inst.op {
                    // Addressing uses: these keep provenance, so they do not escape the base.
                    // `Gep`'s *index* still escapes (it is an integer use of the address).
                    Op::Gep { ptr: _, index, .. } => info.mark_escaped(*index),
                    Op::Load(ptr, ty) => info.note_header_read(*ptr, ty),
                    Op::Store { ptr, value } => {
                        info.mark_escaped(*value);
                        info.mark_written(*ptr);
                    }
                    other => crate::each_op_use(other, &mut |v| info.mark_escaped(v)),
                }
            }
            crate::each_term_use(&b.term, &mut |v| info.mark_escaped(v));
        }
        info.settle_param_data();
        info
    }

    fn mark_escaped(&mut self, v: ValueId) {
        match self.prov(v) {
            Prov::Alloca(a) => {
                if let Some(fact) = self.facts.get_mut(a.0 as usize) {
                    fact.escaped = true;
                }
            }
            Prov::Param(i) => {
                if let Some(p) = self.params.get_mut(i as usize) {
                    p.base_escaped = true;
                }
            }
            Prov::ParamData(i) => {
                if let Some(p) = self.params.get_mut(i as usize) {
                    p.data_escaped = true;
                }
            }
            Prov::Global(_) | Prov::Unknown => {}
        }
    }

    /// A load from a `@noalias` slice's header that reads any byte of its data-pointer word other
    /// than as the one canonical `ptr` load at offset 0 — an `i64` read that a cast can turn back
    /// into a pointer, a read at an offset the sweep could not fold — copies the data pointer past
    /// the provenance sweep, which only classifies the canonical load as `ParamData`. The elements
    /// could then be reached through a pointer this analysis calls `Unknown`, so they no longer
    /// count as private. (The `i64` length at offset 8 is not a pointer and reads freely.)
    fn note_header_read(&mut self, ptr: ValueId, ty: &MirType) {
        if self.params.is_empty() {
            return;
        }
        let Prov::Param(i) = self.prov(ptr) else {
            return;
        };
        if !self.slice_promised(i) {
            return;
        }
        let off = self.offset_of(ptr);
        if matches!(ty, MirType::Ptr) && off == Some(0) {
            return;
        }
        let bytes = byte_size(ty).unwrap_or(u32::MAX);
        let reads_data_word = match off {
            Some(o) => o < 8 && o.saturating_add(i64::from(bytes)) > 0,
            None => true,
        };
        if reads_data_word {
            if let Some(p) = self.params.get_mut(i as usize) {
                p.data_escaped = true;
            }
        }
    }

    /// Record that a store writes through `ptr`. Only parameters keep this bit; a slot's contents
    /// are not part of any fact here.
    fn mark_written(&mut self, ptr: ValueId) {
        match self.prov(ptr) {
            Prov::Param(i) => {
                if let Some(p) = self.params.get_mut(i as usize) {
                    p.base_written = true;
                }
            }
            Prov::ParamData(i) => {
                if let Some(p) = self.params.get_mut(i as usize) {
                    p.data_written = true;
                }
            }
            Prov::Alloca(_) | Prov::Global(_) | Prov::Unknown => {}
        }
    }

    /// A `@noalias` slice's data provenance is one pointer only while its header holds still. The
    /// provenance sweep assigned `ParamData` tentatively, before anything was known about stores;
    /// now that the escape scan has run, demote every `ParamData(i)` whose header was written or
    /// published — the loads may then see different pointers, so they are `Unknown` like any other
    /// loaded pointer. Demotion only ever weakens a fact, so it cannot invalidate one the escape
    /// scan derived (a `ParamData` value is never an `Alloca`).
    fn settle_param_data(&mut self) {
        if !self
            .params
            .iter()
            .any(|p| p.attrs.noalias && p.attrs.slice && (p.base_written || p.base_escaped))
        {
            return;
        }
        for fa in self.facts.iter_mut() {
            if let Prov::ParamData(i) = fa.prov {
                if self
                    .params
                    .get(i as usize)
                    .is_none_or(|p| p.base_written || p.base_escaped)
                {
                    fa.prov = Prov::Unknown;
                    fa.offset = UNKNOWN_OFF;
                }
            }
        }
    }

    /// Does parameter `i` carry a `@noalias` promise?
    fn promised(&self, i: u32) -> bool {
        self.params.get(i as usize).is_some_and(|p| p.attrs.noalias)
    }

    /// Is parameter `i` a `@noalias` slice — a header whose offset-0 word is a data pointer the
    /// promise covers?
    fn slice_promised(&self, i: u32) -> bool {
        self.params
            .get(i as usize)
            .is_some_and(|p| p.attrs.noalias && p.attrs.slice)
    }

    /// Is parameter `i`'s storage promised AND unreachable from anything this function lost track
    /// of? See the module note: nothing derived from it escaped, so no `Unknown` pointer and no
    /// callee can name it.
    fn promised_private(&self, i: u32) -> bool {
        self.params
            .get(i as usize)
            .is_some_and(|p| p.attrs.noalias && !p.base_escaped && !p.data_escaped)
    }

    /// Distinct parameters `i` and `j` whose storage the source promised apart — one promise
    /// suffices, because the promised side excludes *every* other name the caller holds.
    fn promised_apart(&self, i: u32, j: u32) -> bool {
        i != j && (self.promised(i) || self.promised(j))
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

    /// The base object `v` points into and its constant byte offset from that base's start, when
    /// both are known. Two pointers with equal answers address the same byte — the must-alias
    /// question `Dse` and `Licm` ask to match one location spelled by two different `gep`s. (Like
    /// the rest of this analysis it measures offsets in native bytes, so it shares the assumption
    /// that one object is not addressed with two different element strides.)
    pub fn base_and_offset(&self, v: ValueId) -> Option<(Prov, i64)> {
        match self.prov(v) {
            Prov::Unknown => None,
            p => self.offset_of(v).map(|o| (p, o)),
        }
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

    /// Is `v` a pointer into the storage of a `@noalias` parameter that this function reaches only
    /// through values it tracks — nothing derived from the parameter escapes? If so, returns
    /// whether any store in the function writes that storage.
    ///
    /// This is the parameter analogue of [`AliasInfo::is_private_stack`], and it is what lets a
    /// backend put every access to that storage in an alias class of its own: an access through a
    /// pointer this analysis lost track of cannot reach it (it was not derived from the parameter,
    /// and the promise excludes everything else), and neither can a callee. `None` for every other
    /// pointer — in particular for every pointer of a function without a promise.
    ///
    /// NOT a "dead at exit" fact: unlike a private slot, a parameter's storage is the caller's, so
    /// a store to it is always observable after the return.
    pub fn promised_storage(&self, v: ValueId) -> Option<PromisedStorage> {
        let i = match self.prov(v) {
            Prov::Param(i) | Prov::ParamData(i) => i,
            Prov::Alloca(_) | Prov::Global(_) | Prov::Unknown => return None,
        };
        if !self.promised_private(i) {
            return None;
        }
        let p = self.params.get(i as usize)?;
        Some(if p.base_written || p.data_written {
            PromisedStorage::Written
        } else {
            PromisedStorage::ReadOnly
        })
    }

    /// Can no call in this function read or write the location `v` points into? True for a private
    /// stack slot and for non-escaping `@noalias` storage ([`AliasInfo::promised_storage`]) — the
    /// two classes no callee can hold an address for. A memory transform that forgets everything
    /// at a call may keep facts about exactly these.
    pub fn is_call_proof(&self, v: ValueId) -> bool {
        self.is_private_stack(v) || self.promised_storage(v).is_some()
    }

    /// Is `ptr` inside the fat-pointer header of a `@noalias` slice parameter, at a known offset
    /// such that `bytes` bytes stay within the 16-byte header, in a function that never writes the
    /// header nor lets its address escape? Then a load there reads the same value at every point
    /// of the call — and, the header being a live object for the whole call, the load can neither
    /// fault nor move past anything that changes it. A backend may mark it pure (Cranelift's
    /// `readonly` + `can_move`), which lets a re-read `data`/`len` field leave the loop.
    pub fn is_invariant_slice_header(&self, ptr: ValueId, bytes: u32) -> bool {
        let Prov::Param(i) = self.prov(ptr) else {
            return false;
        };
        let Some(p) = self.params.get(i as usize) else {
            return false;
        };
        p.attrs.noalias
            && p.attrs.slice
            && !p.base_written
            && !p.base_escaped
            && self.within_slice_header(ptr, bytes)
    }

    /// `ptr` sits at a known offset into a parameter's fat-pointer header, and an access of `bytes`
    /// bytes there stays inside its 16 bytes.
    fn within_slice_header(&self, ptr: ValueId, bytes: u32) -> bool {
        let Some(off) = self.offset_of(ptr) else {
            return false;
        };
        off >= 0 && (off as u128) + (bytes as u128) <= SLICE_HEADER_BYTES as u128
    }

    /// **The query.** Can accesses through `a` and `b` touch the same byte?
    ///
    /// Conservative: `true` whenever the analysis cannot prove otherwise. Access widths are not
    /// known here, so two disjoint constant offsets off a common base still answer `true` — use
    /// [`AliasInfo::may_alias_sized`] when the widths are available.
    pub fn may_alias(&self, a: ValueId, b: ValueId) -> bool {
        self.may_alias_access(Access { v: a, size: None }, Access { v: b, size: None })
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
            // Two parameters: `f(a, a)` is legal for ordinary parameters, so they may alias — unless
            // the source promised either one apart (`@noalias`). That promise is the single fact
            // this analysis cannot derive on its own.
            (Prov::Param(i), Prov::Param(j)) => !self.promised_apart(i, j),
            // A `@noalias` slice's elements vs any parameter's storage, its own header included: the
            // promise for a different parameter, and "a slice never views its own fat pointer" for
            // the same one.
            (Prov::ParamData(_), Prov::Param(_)) | (Prov::Param(_), Prov::ParamData(_)) => false,
            // Two different promised slices (one slice against itself took the same-base branch).
            (Prov::ParamData(_), Prov::ParamData(_)) => false,
            // The data pointer came out of a header this function never writes, filled by the
            // caller before any of this frame's slots existed.
            (Prov::Alloca(_), Prov::ParamData(_)) | (Prov::ParamData(_), Prov::Alloca(_)) => false,
            // Promised storage nothing untracked can reach: no pointer we lost track of was derived
            // from it, and the promise excludes every pointer not derived from it.
            (Prov::Param(i) | Prov::ParamData(i), Prov::Unknown)
            | (Prov::Unknown, Prov::Param(i) | Prov::ParamData(i)) => !self.promised_private(i),
            // A `*u8` handed to us could well be a string literal's address.
            (Prov::Param(_) | Prov::ParamData(_), Prov::Global(_))
            | (Prov::Global(_), Prov::Param(_) | Prov::ParamData(_)) => true,
            (Prov::Unknown, Prov::Unknown)
            | (Prov::Unknown, Prov::Global(_))
            | (Prov::Global(_), Prov::Unknown) => true,
        }
    }

    /// Can executing `op` write memory that an access of `bytes` bytes through `ptr` would read?
    ///
    /// A `call` (or a vector-kernel call, which stores through the pointers handed to it) can write
    /// anything the caller can name — *except* a stack slot of this function whose address never
    /// escaped, or `@noalias` storage nothing derived from escaped, neither of which any callee can
    /// have an address for ([`AliasInfo::is_call_proof`]).
    /// The match below is **deliberately exhaustive — never add a `_` arm.** An op that writes
    /// memory and is not classified as a writer here silently answers "cannot clobber", which is an
    /// unsound no-alias answer, i.e. a miscompile. Listing every non-writing variant makes a newly
    /// added `Op` a compile error at this site instead of a silent wrong answer, so the requirement
    /// is enforced by the compiler rather than by `CONTRIBUTING.md`.
    pub fn may_clobber(&self, ptr: ValueId, bytes: u32, op: &Op) -> bool {
        match op {
            Op::Store { ptr: q, value: _ } => self.may_alias_sized(ptr, bytes, *q, u32::MAX),
            Op::Call { .. } | Op::VecKernelCall { .. } => !self.is_call_proof(ptr),
            // Everything below provably writes no memory.
            Op::ConstInt(..)
            | Op::ConstFloat(..)
            | Op::Bin(..)
            | Op::Cmp(..)
            | Op::Neg(..)
            | Op::Not(..)
            | Op::Cast(..)
            | Op::Select(..)
            | Op::Alloca { .. }
            | Op::Load(..)
            | Op::Gep { .. }
            | Op::FuncAddr(..)
            | Op::GlobalAddr(..)
            | Op::Splat(..)
            | Op::ExtractLane(..)
            | Op::Iota(..)
            | Op::Fma(..)
            | Op::Sqrt(..)
            | Op::Round(..)
            | Op::Barrier => false,
        }
    }

    /// Is an access of `bytes` bytes at `ptr` safe to perform **speculatively** — on a path where
    /// the original program might not have performed it?
    ///
    /// Two classes qualify: an address inside an `alloca` of this function at a known in-bounds
    /// offset — a stack slot is live for the whole function, so reading it can neither fault nor
    /// observe anything the program did not already own — and an address inside the 16-byte
    /// fat-pointer header of a `@noalias` slice parameter, which the caller keeps alive for the
    /// whole call. Every other pointer — including any other parameter, whose validity is the
    /// *caller's* business — answers `false`. (The header class is gated on the promise only so
    /// that a program without `@noalias` analyzes exactly as it always did.)
    pub fn is_dereferenceable(&self, ptr: ValueId, bytes: u32) -> bool {
        let a = match self.prov(ptr) {
            Prov::Alloca(a) => a,
            Prov::Param(i) => {
                return self.slice_promised(i) && self.within_slice_header(ptr, bytes);
            }
            Prov::ParamData(_) | Prov::Global(_) | Prov::Unknown => return false,
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

/// Are `a` and `b` — two base pointers a loop indexes off — provably distinct objects, in a
/// function whose parameters carry a `@noalias` promise?
///
/// For a pass that reasons about base pointers structurally and does not otherwise build an
/// [`AliasInfo`] (the loop dependence test, the vectorizer's runtime alias guard). `false` — no
/// claim — for every function without a promise, without analyzing anything, so such a pass behaves
/// exactly as it did before the promise existed. When there is one, the analysis is built once per
/// question; promised functions are the rare case, and the answer is the whole-function one.
///
/// A `false` from [`AliasInfo::may_alias`] on two *base* pointers is a statement about the objects:
/// every address derived from one is disjoint from every address derived from the other, at any
/// offset — which is exactly the fact a dependence test needs to skip a pair.
pub fn promised_disjoint_bases(f: &Function, a: ValueId, b: ValueId) -> bool {
    if a == b || !f.has_noalias_params() {
        return false;
    }
    !AliasInfo::analyze(f).may_alias(a, b)
}

/// [`promised_disjoint_bases`] over two loop-access bases. A [`crate::loop_info::MemBase::ReloadedOuter`]
/// base is the pointer its in-loop load produces — for a `@noalias` slice, a load of the data
/// pointer from a header nothing writes, which the analysis classifies as the slice's elements.
pub fn promised_disjoint_membases(
    f: &Function,
    a: crate::loop_info::MemBase,
    b: crate::loop_info::MemBase,
) -> bool {
    PromisedBases::new(f).disjoint(a, b)
}

/// [`promised_disjoint_membases`] for a pass that asks it about many pairs of one function it does
/// not change in between (the loop dependence test, the vectorizer's runtime alias guard). The
/// analysis is built at most once, on the first question that reaches it — never for a function
/// without a promise — instead of once per pair, which is quadratic in the loop's accesses times
/// the function's size. The borrow of `f` is what makes the cached answer valid: nothing can
/// mutate the function while this exists (see the module note on staleness).
pub(crate) struct PromisedBases<'f> {
    f: &'f Function,
    info: Option<AliasInfo>,
}

impl<'f> PromisedBases<'f> {
    pub(crate) fn new(f: &'f Function) -> PromisedBases<'f> {
        PromisedBases { f, info: None }
    }

    /// See [`promised_disjoint_membases`].
    pub(crate) fn disjoint(
        &mut self,
        a: crate::loop_info::MemBase,
        b: crate::loop_info::MemBase,
    ) -> bool {
        use crate::loop_info::MemBase;
        if a.same_object(b) || !self.f.has_noalias_params() {
            return false;
        }
        let base = |m: MemBase| match m {
            MemBase::Outer(v) => v,
            MemBase::ReloadedOuter { value, .. } => value,
        };
        let (x, y) = (base(a), base(b));
        if x == y {
            return false;
        }
        let f = self.f;
        !self
            .info
            .get_or_insert_with(|| AliasInfo::analyze(f))
            .may_alias(x, y)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use wukong_mir::{AddrSpace, BinOp, Builder, CmpOp};
    use wukong_span::Interner;

    /// `Fact` is allocated one-per-SSA-value on the compiler's hot path, so its size is a real
    /// cost. Pinned here: the round stamp must fit in the padding that `alloca_bytes`/`escaped`
    /// already leave, not push the record into another 8 bytes.
    #[test]
    fn fact_stays_one_cache_friendly_record() {
        assert_eq!(std::mem::size_of::<Fact>(), 32, "Fact grew");
    }

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

    /// The IR contract is **dominance**, not layout order: `wukong_mir`'s verifier walks the
    /// dominator tree precisely so a definition may sit in a block laid out *after* its use. A
    /// provenance pass that trusts layout order would classify the `gep` below as `Unknown` (its
    /// base is not visited yet), and the escape scan would then fail to notice that the slot's
    /// address reached a call — reporting a *published* slot as private. That is an unsound
    /// no-alias answer, i.e. a miscompile, so provenance must settle before the escape scan
    /// regardless of how the blocks happen to be ordered.
    #[test]
    fn provenance_settles_even_when_a_base_is_laid_out_after_its_use() {
        let mut it = Interner::new();
        let mut b = Builder::new(it.intern("late"), MirType::Void);
        let use_blk = b.new_block();
        let def_blk = b.new_block();

        // entry -> def_blk -> use_blk, but the *layout* is entry, use_blk, def_blk.
        b.br(def_blk, vec![]);

        // `Builder::alloca` always lands in the entry block, so the base itself is never late.
        // A `gep` is not so constrained: the *middle* of a gep chain can sit in any block, and
        // here `g1`'s block is laid out after the block that consumes it.
        let a = b.alloca(blob(16));
        let z = b.build(MirType::I64, Op::ConstInt(0, MirType::I64));
        b.br(def_blk, vec![]);

        b.switch_to(def_blk);
        let g1 = gep(&mut b, a, z);
        b.br(use_blk, vec![]);

        b.switch_to(use_blk);
        let g = gep(&mut b, g1, z);
        b.build_void(Op::Call {
            func: it.intern("sink"),
            args: vec![g],
        });
        b.ret(None);
        let f = b.finish();

        // The function really is well-formed: def_blk dominates use_blk.
        assert!(
            wukong_mir::verify::verify_function(&f).is_empty(),
            "{:?}",
            wukong_mir::verify::verify_function(&f)
        );

        let info = AliasInfo::analyze(&f);
        assert_eq!(info.prov(g1), Prov::Alloca(a));
        assert_eq!(info.prov(g), Prov::Alloca(a), "gep lost its base");

        // The consequence that makes this a miscompile rather than a missed optimization: with
        // the base lost, nothing marks the slot escaped, so a callee holding its address is told
        // it cannot write it.
        assert!(
            info.alloca_escapes(a),
            "a slot whose address reached a call was reported private"
        );
        assert!(!info.is_private_stack(g));
        let call = Op::Call {
            func: it.intern("sink"),
            args: vec![],
        };
        assert!(info.may_clobber(a, 8, &call));
    }

    /// The inliner splices a callee's blocks ahead of the caller's remaining code, so a caller's
    /// `alloca` can itself sit in a block laid out AFTER the `gep`s that use it. Round 1 then
    /// classifies the base but leaves each such `gep` at its default `Unknown` — no `gep` fact
    /// moves — and the sweep must not take that for a fixpoint. It did: the loads through those
    /// `gep`s stopped counting as reads of the slot, `dse` judged the slot never read and deleted a
    /// constant-offset store into it (`tests/run/hardsigmoid_nan.wk` at -O2, recognition off).
    #[test]
    fn provenance_settles_when_the_alloca_itself_is_laid_out_after_its_geps() {
        let mut it = Interner::new();
        let mut b = Builder::new(it.intern("late_alloca"), MirType::Void);
        let use_blk = b.new_block();
        let def_blk = b.new_block();

        // entry -> def_blk -> use_blk, laid out entry, use_blk, def_blk.
        let z = b.build(MirType::I64, Op::ConstInt(0, MirType::I64));
        let four = b.build(MirType::I64, Op::ConstInt(4, MirType::I64));
        b.br(def_blk, vec![]);

        b.switch_to(def_blk);
        let a = b.build(
            MirType::Ptr,
            Op::Alloca {
                ty: blob(16),
                space: AddrSpace::Local,
            },
        );
        b.br(use_blk, vec![]);

        b.switch_to(use_blk);
        let g = gep(&mut b, a, four);
        b.build(MirType::I32, Op::Load(g, MirType::I32));
        let g0 = gep(&mut b, a, z);
        b.ret(None);
        let f = b.finish();
        assert!(
            wukong_mir::verify::verify_function(&f).is_empty(),
            "{:?}",
            wukong_mir::verify::verify_function(&f)
        );

        let info = AliasInfo::analyze(&f);
        assert_eq!(info.prov(g), Prov::Alloca(a), "gep lost its late alloca");
        assert_eq!(info.base_and_offset(g), Some((Prov::Alloca(a), 4)));
        assert_eq!(info.base_and_offset(g0), Some((Prov::Alloca(a), 0)));
        assert!(info.may_alias_sized(g, 4, a, 16));
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

    // ---- the `@noalias` parameter promise -------------------------------------------------------

    const NOALIAS: ParamAttrs = ParamAttrs {
        noalias: true,
        slice: false,
    };
    const NOALIAS_SLICE: ParamAttrs = ParamAttrs {
        noalias: true,
        slice: true,
    };
    const PLAIN_SLICE: ParamAttrs = ParamAttrs {
        noalias: false,
        slice: true,
    };

    /// A pointer parameter carrying `attrs`.
    fn param(b: &mut Builder, attrs: ParamAttrs) -> ValueId {
        let p = b.add_param(MirType::Ptr);
        b.set_param_attrs(p, attrs);
        p
    }

    /// One promise separates a parameter from every other parameter; the two unpromised ones keep
    /// the ordinary answer (`params_may_alias_each_other_but_never_a_local_slot`).
    #[test]
    fn a_promised_parameter_is_disjoint_from_every_other_parameter() {
        let mut it = Interner::new();
        let mut b = Builder::new(it.intern("k"), MirType::Void);
        let out = param(&mut b, NOALIAS);
        let x = param(&mut b, ParamAttrs::default());
        let y = param(&mut b, ParamAttrs::default());
        let eight = b.build(MirType::I64, Op::ConstInt(8, MirType::I64));
        let out8 = gep(&mut b, out, eight);
        let v = b.build(MirType::F32, Op::Load(x, MirType::F32));
        let w = b.build(MirType::F32, Op::Load(y, MirType::F32));
        b.build_void(Op::Store {
            ptr: out8,
            value: v,
        });
        b.build_void(Op::Store { ptr: out, value: w });
        b.ret(None);
        let f = b.finish();
        assert!(f.has_noalias_params());
        assert_eq!(f.param_attr(0), NOALIAS);
        assert_eq!(f.param_attr(2), ParamAttrs::default());
        let info = AliasInfo::analyze(&f);

        assert!(!info.may_alias(out, x));
        assert!(!info.may_alias(x, out));
        assert!(!info.may_alias(out8, y));
        // Neither unpromised parameter said anything about the other.
        assert!(info.may_alias(x, y));
        // A parameter still overlaps itself, and offsets still separate it.
        assert!(info.may_alias(out, out8));
        assert!(!info.may_alias_sized(out, 8, out8, 4));
        // So the store through `out` cannot clobber a read of `x`.
        let st = Op::Store {
            ptr: out8,
            value: v,
        };
        assert!(!info.may_clobber(x, 4, &st));
        assert!(info.may_clobber(out, 16, &st));
        assert_eq!(info.promised_storage(out), Some(PromisedStorage::Written));
        assert_eq!(info.promised_storage(x), None);
        assert!(promised_disjoint_bases(&f, out, y));
        assert!(!promised_disjoint_bases(&f, x, y));
    }

    /// A slice parameter points at its fat pointer, and its elements are the pointer loaded from
    /// offset 0. Under the promise, with the header untouched, every such load is one object of its
    /// own: disjoint from every header (its own included), from the frame's slots, and from what an
    /// unpromised slice's data pointer can reach.
    #[test]
    fn a_promised_slice_has_its_own_element_provenance() {
        let mut it = Interner::new();
        let mut b = Builder::new(it.intern("s"), MirType::Void);
        let s = param(&mut b, NOALIAS_SLICE);
        let t = param(&mut b, PLAIN_SLICE);
        let a = b.alloca(blob(16));
        let d1 = b.build(MirType::Ptr, Op::Load(s, MirType::Ptr));
        let d2 = b.build(MirType::Ptr, Op::Load(s, MirType::Ptr));
        let e = b.build(MirType::Ptr, Op::Load(t, MirType::Ptr));
        let eight = b.build(MirType::I64, Op::ConstInt(8, MirType::I64));
        let len = gep(&mut b, s, eight);
        let _n = b.build(MirType::I64, Op::Load(len, MirType::I64));
        let x = b.build(MirType::F32, Op::Load(e, MirType::F32));
        b.build_void(Op::Store { ptr: d1, value: x });
        b.build_void(Op::Store { ptr: a, value: x });
        b.ret(None);
        let f = b.finish();
        let info = AliasInfo::analyze(&f);

        assert_eq!(info.prov(d1), Prov::ParamData(0));
        assert_eq!(info.prov(d2), Prov::ParamData(0));
        // Two loads of an invariant header are one pointer.
        assert!(info.may_alias(d1, d2));
        // The unpromised slice's data pointer is an ordinary loaded pointer ...
        assert_eq!(info.prov(e), Prov::Unknown);
        // ... which cannot reach promised storage that nothing derived from escaped.
        assert!(!info.may_alias(d1, e));
        assert!(!info.may_alias(d1, s));
        assert!(!info.may_alias(d1, len));
        assert!(!info.may_alias(d1, t));
        assert!(!info.may_alias(d1, a));
        // The header is never written: a load there is invariant, and the header is a live
        // 16-byte object, so it may be read speculatively — but not past its end.
        assert!(info.is_invariant_slice_header(s, 8));
        assert!(info.is_invariant_slice_header(len, 8));
        assert!(!info.is_invariant_slice_header(len, 16));
        assert!(info.is_dereferenceable(s, 16));
        assert!(!info.is_dereferenceable(len, 16));
        // An unpromised slice header gets none of this.
        assert!(!info.is_invariant_slice_header(t, 8));
        assert!(!info.is_dereferenceable(t, 8));
        assert_eq!(info.promised_storage(d1), Some(PromisedStorage::Written));
        assert_eq!(info.promised_storage(e), None);
    }

    /// The element provenance rests on a header that holds still. Writing the header, or handing
    /// its address to a call, demotes every element pointer to an ordinary loaded pointer. The
    /// promise between the parameters themselves is unaffected.
    #[test]
    fn a_written_or_published_header_demotes_the_element_provenance() {
        for publish in [false, true] {
            let mut it = Interner::new();
            let sink = it.intern("sink");
            let mut b = Builder::new(it.intern("w"), MirType::Void);
            let s = param(&mut b, NOALIAS_SLICE);
            let other = param(&mut b, ParamAttrs::default());
            let d = b.build(MirType::Ptr, Op::Load(s, MirType::Ptr));
            if publish {
                b.build_void(Op::Call {
                    func: sink,
                    args: vec![s],
                });
            } else {
                b.build_void(Op::Store {
                    ptr: s,
                    value: other,
                });
            }
            b.ret(None);
            let f = b.finish();
            let info = AliasInfo::analyze(&f);
            assert_eq!(info.prov(d), Prov::Unknown, "publish={publish}");
            assert!(!info.is_invariant_slice_header(s, 8), "publish={publish}");
            assert!(!info.may_alias(s, other), "publish={publish}");
        }
    }

    /// Promised storage that escapes is no longer private: a pointer the analysis lost track of may
    /// have been derived from it, and so may anything a callee does with it.
    #[test]
    fn escaped_promised_storage_is_no_longer_call_proof() {
        for escape in [false, true] {
            let mut it = Interner::new();
            let sink = it.intern("sink");
            let mut b = Builder::new(it.intern("e"), MirType::Void);
            let p = param(&mut b, NOALIAS);
            let q = param(&mut b, ParamAttrs::default());
            let unknown = b.build(MirType::Ptr, Op::Load(q, MirType::Ptr));
            let v = b.build(MirType::F32, Op::Load(p, MirType::F32));
            let args = if escape { vec![p] } else { vec![] };
            b.build_void(Op::Call { func: sink, args });
            b.build_void(Op::Store {
                ptr: unknown,
                value: v,
            });
            b.ret(None);
            let f = b.finish();
            let info = AliasInfo::analyze(&f);
            let call = Op::Call {
                func: sink,
                args: vec![],
            };
            assert_eq!(info.prov(unknown), Prov::Unknown);
            assert_eq!(info.may_alias(p, unknown), escape, "escape={escape}");
            assert_eq!(info.may_clobber(p, 4, &call), escape, "escape={escape}");
            assert_eq!(info.is_call_proof(p), !escape, "escape={escape}");
            assert_eq!(
                info.promised_storage(p),
                (!escape).then_some(PromisedStorage::ReadOnly),
                "escape={escape}"
            );
            // The parameter-to-parameter promise does not depend on escape.
            assert!(!info.may_alias(p, q), "escape={escape}");
        }
    }

    /// Reading a promised slice's data word as anything but the canonical pointer load (an `i64`
    /// a cast could turn back into a pointer, or part of the word) copies the element pointer past
    /// the provenance sweep, so the elements stop being private. The length word reads freely.
    #[test]
    fn a_non_canonical_read_of_the_data_word_publishes_the_elements() {
        for (ty, off, private) in [
            (MirType::I64, 0, false),
            (MirType::I32, 4, false),
            (MirType::I64, 8, true),
        ] {
            let mut it = Interner::new();
            let mut b = Builder::new(it.intern("r"), MirType::Void);
            let s = param(&mut b, NOALIAS_SLICE);
            let d = b.build(MirType::Ptr, Op::Load(s, MirType::Ptr));
            let k = b.build(MirType::I64, Op::ConstInt(off, MirType::I64));
            let at = gep(&mut b, s, k);
            let _word = b.build(ty.clone(), Op::Load(at, ty.clone()));
            b.ret(None);
            let f = b.finish();
            let info = AliasInfo::analyze(&f);
            assert_eq!(info.prov(d), Prov::ParamData(0), "{ty:?} at {off}");
            assert_eq!(
                info.promised_storage(d).is_some(),
                private,
                "{ty:?} at {off}"
            );
        }
    }

    /// A function whose parameters promise nothing gets none of the new facts — the per-parameter
    /// table is not even built — so it analyzes exactly as before the promise existed, even when a
    /// parameter is known to be a slice.
    #[test]
    fn a_function_without_a_promise_gets_no_new_facts() {
        let mut it = Interner::new();
        let mut b = Builder::new(it.intern("plain"), MirType::Void);
        let s = param(&mut b, PLAIN_SLICE);
        let t = b.add_param(MirType::Ptr);
        let d = b.build(MirType::Ptr, Op::Load(s, MirType::Ptr));
        b.ret(None);
        let f = b.finish();
        assert!(!f.has_noalias_params());
        let info = AliasInfo::analyze(&f);
        assert!(info.params.is_empty());
        assert_eq!(info.prov(d), Prov::Unknown);
        assert!(info.may_alias(s, t));
        assert!(!info.is_dereferenceable(s, 8));
        assert_eq!(info.promised_storage(s), None);
        assert!(!promised_disjoint_bases(&f, s, t));
    }

    /// A counted loop `for i in 0..n { s[i] = 1.0 }` over slice parameter `s` (parameter 1), the
    /// shape `mir_build` emits: the data pointer is re-read from the header on every iteration.
    /// Returns the function and the loop body's block.
    fn slice_fill_loop(it: &mut Interner, attrs: ParamAttrs) -> (Function, usize) {
        let mut b = Builder::new(it.intern("fill"), MirType::Void);
        let n = b.add_param(MirType::I64);
        let s = param(&mut b, attrs);
        let zero = b.build(MirType::I64, Op::ConstInt(0, MirType::I64));
        let one = b.build(MirType::I64, Op::ConstInt(1, MirType::I64));
        let c1 = b.build(MirType::F32, Op::ConstFloat(1.0, MirType::F32));
        let header = b.new_block();
        let body = b.new_block();
        let exit = b.new_block();
        let i = b.block_param(header, MirType::I64);
        b.br(header, vec![zero]);
        b.switch_to(header);
        let c = b.build(MirType::I1, Op::Cmp(CmpOp::Slt, i, n));
        b.cond_br(c, body, vec![], exit, vec![]);
        b.switch_to(body);
        let data = b.build(MirType::Ptr, Op::Load(s, MirType::Ptr));
        let at = b.build(
            MirType::Ptr,
            Op::Gep {
                ptr: data,
                index: i,
                elem: MirType::F32,
            },
        );
        b.build_void(Op::Store { ptr: at, value: c1 });
        let next = b.build(MirType::I64, Op::Bin(BinOp::Add, i, one));
        b.br(header, vec![next]);
        b.switch_to(exit);
        b.ret(None);
        (b.finish(), body.0 as usize)
    }

    fn loads_in(f: &Function, blk: usize) -> usize {
        f.blocks[blk]
            .insts
            .iter()
            .filter(|i| matches!(i.op, Op::Load(..)))
            .count()
    }

    /// The loop LICM could not touch before: the element store goes through the slice's own data
    /// pointer, which provably cannot land on the header, and the header is live for the whole call
    /// — so the per-iteration re-read of the data pointer leaves the loop. The identical loop over an
    /// unpromised slice keeps it: the store may write the header (`licm::tests::
    /// a_may_aliasing_store_in_the_loop_pins_the_load` is the same refusal for a bare pointer).
    #[test]
    fn licm_hoists_a_promised_slice_header_out_of_its_own_store_loop() {
        use crate::Pass;
        for (attrs, hoisted) in [(PLAIN_SLICE, false), (NOALIAS_SLICE, true)] {
            let mut it = Interner::new();
            let (mut f, body) = slice_fill_loop(&mut it, attrs);
            assert_eq!(loads_in(&f, body), 1);
            let mut cache = crate::CfgAnalyses::default();
            let changed = crate::licm::Licm.run_function(&mut f, &mut cache);
            let dump = wukong_mir::print::print_function(&f, &it);
            assert_eq!(changed, hoisted, "{dump}");
            assert_eq!(loads_in(&f, body), usize::from(!hoisted), "{dump}");
            assert_eq!(loads_in(&f, 0), usize::from(hoisted), "{dump}");
        }
    }

    /// The loop dependence test sees the promise too: the slice's elements and an unpromised
    /// pointer parameter are distinct objects, so no runtime alias check is owed for the pair.
    #[test]
    fn the_loop_dependence_test_needs_no_runtime_check_under_the_promise() {
        use crate::loop_info::MemBase;
        let mut it = Interner::new();
        let (f, body) = slice_fill_loop(&mut it, NOALIAS_SLICE);
        let s = f.params[1];
        let data = f.blocks[body].insts[0]
            .result
            .expect("the data-pointer load");
        let reloaded = MemBase::ReloadedOuter {
            addr: s,
            value: data,
        };
        let n = MemBase::Outer(f.params[0]);
        assert!(promised_disjoint_membases(&f, reloaded, n));
        assert!(!promised_disjoint_membases(&f, reloaded, reloaded));
        let (plain, body) = slice_fill_loop(&mut it, PLAIN_SLICE);
        let data = plain.blocks[body].insts[0]
            .result
            .expect("the data-pointer load");
        let reloaded = MemBase::ReloadedOuter {
            addr: plain.params[1],
            value: data,
        };
        assert!(!promised_disjoint_membases(
            &plain,
            reloaded,
            MemBase::Outer(plain.params[0])
        ));
    }

    /// A re-read of an input across a store to promised output storage is the same value: the
    /// store cannot reach it. CSE forwards the first load to the second only under the promise.
    #[test]
    fn cse_forwards_a_load_across_a_store_to_promised_storage() {
        use crate::Pass;
        for promised in [false, true] {
            let mut it = Interner::new();
            let mut b = Builder::new(it.intern("cp"), MirType::Void);
            let out = param(
                &mut b,
                if promised {
                    NOALIAS
                } else {
                    ParamAttrs::default()
                },
            );
            let x = param(&mut b, ParamAttrs::default());
            let v1 = b.build(MirType::F32, Op::Load(x, MirType::F32));
            b.build_void(Op::Store {
                ptr: out,
                value: v1,
            });
            let v2 = b.build(MirType::F32, Op::Load(x, MirType::F32));
            let four = b.build(MirType::I64, Op::ConstInt(4, MirType::I64));
            let out4 = gep(&mut b, out, four);
            b.build_void(Op::Store {
                ptr: out4,
                value: v2,
            });
            b.ret(None);
            let mut f = b.finish();
            let mut cache = crate::CfgAnalyses::default();
            crate::cse::Cse.run_function(&mut f, &mut cache);
            let last_stored = f.blocks[0]
                .insts
                .iter()
                .rev()
                .find_map(|i| match i.op {
                    Op::Store { value, .. } => Some(value),
                    _ => None,
                })
                .expect("a store");
            let dump = wukong_mir::print::print_function(&f, &it);
            assert_eq!(last_stored, if promised { v1 } else { v2 }, "{dump}");
        }
    }

    /// The printed MIR names the promise on the parameter it covers, and only there.
    #[test]
    fn the_promise_is_visible_in_the_mir_dump() {
        let mut it = Interner::new();
        let (f, _) = slice_fill_loop(&mut it, NOALIAS_SLICE);
        let dump = wukong_mir::print::print_function(&f, &it);
        assert!(
            dump.starts_with("fn fill(v0 i64, v1 ptr noalias) -> void {"),
            "{dump}"
        );
        let (plain, _) = slice_fill_loop(&mut it, PLAIN_SLICE);
        let dump = wukong_mir::print::print_function(&plain, &it);
        assert!(!dump.contains("noalias"), "{dump}");
    }

    /// Lower `src` through the real front end and the `-O1` pipeline, returning `func`.
    fn lowered(src: &str, func: &str, level: u8) -> Function {
        let mut interner = Interner::new();
        let (module, pd) =
            wukong_parser::parse_module(src, wukong_span::SourceId(0), &mut interner);
        assert!(pd.iter().all(|d| !d.is_error()), "parse: {pd:?}");
        let (sema, sd) = wukong_sema::check(&module, &interner);
        assert!(sd.iter().all(|d| !d.is_error()), "sema: {sd:?}");
        let (mut program, _) = wukong_mir_build::lower_program(&module, &sema, &mut interner);
        crate::optimize(&mut program, level);
        let sym = interner.intern(func);
        program
            .function(sym)
            .unwrap_or_else(|| panic!("no fn {func}"))
            .clone()
    }

    /// End to end from the surface syntax: the two-slice loop that owes a runtime alias check
    /// (`loop_info::tests::two_slice_parameters_need_a_runtime_check`) owes none once the output is
    /// promised — the store to `o[i]` provably cannot reach `x`, whatever the caller passes.
    #[test]
    fn a_promised_slice_loop_owes_no_runtime_alias_check() {
        use crate::loop_info::{analyze_function, AddrForm, MemDep};
        let verdict = |params: &str| {
            let src = format!(
                "fn k({params}) {{ let mut i: i64 = 0; \
                 while i < n {{ o[i] = x[i] * 2.0; i = i + 1; }} }} \
                 fn main() -> i32 {{ return 0; }}"
            );
            let f = lowered(&src, "k", 1);
            let forest = analyze_function(&f);
            let l = forest
                .loops
                .iter()
                .max_by_key(|l| {
                    l.accesses
                        .iter()
                        .filter(|a| matches!(a.addr, AddrForm::Affine { .. }))
                        .count()
                })
                .expect("a loop");
            l.dep.memory
        };
        assert_eq!(
            verdict("x: []f32, mut o: []f32, n: i64"),
            MemDep::IndependentIfBasesDisjoint
        );
        let promised = verdict("x: []f32, @noalias mut o: []f32, n: i64");
        assert!(
            matches!(
                promised,
                MemDep::Independent | MemDep::IndependentIfBasesStable
            ),
            "{promised:?}"
        );
    }
}
