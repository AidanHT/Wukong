//! Global value numbering: dominator-tree value numbering, load forwarding across edges and
//! joins, block-parameter numbering, and phi insertion for fully redundant values.
//!
//! Pure operations are value-numbered over the **dominator tree**: a computation in a block is
//! reused by any block it dominates, since the dominating definition reaches all those uses. Walking
//! the tree with a scoped table (entries added on entry to a block, removed on exit) keeps reuse
//! legal without re-checking dominance per candidate. This catches redundancy across blocks, not
//! just within one.
//!
//! Loads are forwarded keyed by the SSA address value and the accessed type: a `store p, v` makes
//! `v` the current contents of `(p, typeof v)`, the first `load p` becomes the current contents of
//! `(p, ty)`, and a later load of the same address and type reuses it. Which entries a store or call
//! invalidates is decided by [`crate::alias`] — a store only forgets the entries it `may_alias`, and
//! a call forgets everything except addresses inside a stack slot whose address never escaped this
//! function (no callee can hold a pointer to one).
//!
//! # Across edges and joins
//!
//! A dominator child whose only predecessor is its parent inherits the parent's final memory table:
//! every execution of the child sees exactly the state the table describes. A **join** is entered
//! with the *intersection* of its predecessors' exit tables. The tree's children are visited in
//! reverse postorder, which guarantees that every forward-edge predecessor of a join has been
//! numbered before the join itself (the predecessor's branch of the dominator tree starts at a
//! sibling of the join that precedes it in RPO); each such predecessor records its exit table on the
//! way past. A location every edge agrees on keeps its value. A location every edge knows, with a
//! different value per edge, becomes a *pending phi*: if a load at or below the join asks for it,
//! the join gains a block parameter carrying each edge's value and the load is replaced by it — the
//! classic store-to-load forwarding across `if c { a[0] = x } else { a[0] = y }`. A loop header
//! has a back-edge predecessor that is not yet numbered, so it starts empty, as before: back-edge
//! writes can never be missed.
//!
//! Expressions get the same treatment. Each predecessor of a join also records the pure expressions
//! computed on its dominator path below the join's immediate dominator (the arms of a diamond); an
//! expression every arm computed is fully redundant at the join, and a recomputation at or below
//! the join is replaced by a phi of the arms' values — partial-redundancy elimination restricted to
//! the fully-available case, which needs no edge splitting and never speculates a computation.
//! Constants and address arithmetic are excluded: they rematerialize for free, and a phi of `gep`s
//! would hide the pointers' provenance from the alias analysis.
//!
//! # Block parameters
//!
//! Before a block's body is numbered its parameters are: a parameter every incoming edge passes the
//! same value (ignoring the parameter itself on a back edge) *is* that value, and two parameters
//! every edge passes identical values are one value. Arguments compare after rewriting, so an edge
//! from an already-numbered predecessor compares by value number; one not yet numbered compares by
//! identity, which is always sound.
//!
//! # Calls
//!
//! A call to a user function summarized as side-effect free (the `effects` module) writes nothing,
//! so it leaves the memory table alone. If it also reads nothing, it is value-numbered like any
//! pure instruction, keyed by callee and arguments. If it reads memory, its result is forwarded like
//! a load: keyed in the memory table, valid until a store that is not to a private stack slot or an
//! opaque call might change what the callee reads. Every other call forgets all but private slots.
//!
//! Forwarding loads to a common value lets the pure value-numbering then collapse the expressions
//! built on top of them. DCE deletes the dead remains, and simplify-phis the parameters made dead.

use crate::alias::{type_bytes, AliasInfo};
use crate::effects::CallEffects;
use crate::fxhash::FxHashMap;

use wukong_mir::{BinOp, CmpOp, Function, MirType, Op, Terminator, ValueId};

use crate::{cfg, map_op_uses, map_term_uses, CfgAnalyses, Pass};

/// Cap on the per-block memory table. A store invalidates by scanning the table, so an unbounded
/// table would make a block with many loads and many stores quadratic. Recognizer-generated blocks
/// can be large; past the cap the pass simply stops recording new locations (already-recorded ones
/// keep working), which only costs optimization, never correctness.
const MAX_TRACKED_LOCATIONS: usize = 256;

/// "No rewrite" in the dense rewrite table.
const NO_REWRITE: u32 = u32::MAX;

/// Cap on the expressions one predecessor offers a join, which bounds the join's pairwise match.
const MAX_JOIN_EXPRESSIONS: usize = 64;

/// Cap on pending phis per run. Past it a join simply offers no new phis.
const MAX_PENDING_PHIS: usize = 4096;

/// Blocks with more parameters than this skip the quadratic duplicate-parameter scan.
const MAX_NUMBERED_PARAMS: usize = 64;

pub struct Cse;

impl Pass for Cse {
    fn name(&self) -> &'static str {
        "cse"
    }

    fn run_function(&self, f: &mut Function, cache: &mut CfgAnalyses) -> bool {
        // Dominance requires a clean CFG. Skip the reachability DFS when nothing is unreachable
        // (the steady state) — `all_reachable` is free off the cached rpo `idoms` needs anyway.
        if !cache.all_reachable(f) {
            cfg::prune_unreachable(f);
            cache.invalidate(); // pruning renumbered blocks
        }
        // Taken before the dominator borrows below: `Rc`s, so they borrow nothing.
        let alias = cache.alias(f);
        let effects = cache.call_effects();
        let (children, preds, idom) = cache.gvn_inputs(f);

        let n = f.blocks.len();
        // How many joins each block feeds: the number of exit records it owes.
        let mut joins_left = vec![0u32; n];
        for (b, blk) in f.blocks.iter().enumerate() {
            for s in cfg::successors(&blk.term) {
                if preds[s.0 as usize].len() >= 2 {
                    joins_left[b] += 1;
                }
            }
        }
        let mut cx = Numbering {
            f,
            children,
            preds,
            idom,
            alias: &alias,
            effects: effects.as_deref(),
            arg_lists: FxHashMap::default(),
            scratch: Vec::new(),
            vn: FxHashMap::default(),
            pre: FxHashMap::default(),
            rewrite: vec![NO_REWRITE; f.value_types.len()],
            rewrites: 0,
            types: TyIds::default(),
            exit_mem: vec![None; n],
            exit_exprs: FxHashMap::default(),
            joins_left,
            entered: vec![false; n],
            pending: Vec::new(),
            created: Vec::new(),
            base: f.value_types.len() as u32,
        };
        cx.visit(f.entry.0);
        let Numbering {
            rewrite,
            rewrites,
            pending,
            created,
            ..
        } = cx;
        if rewrites == 0 {
            return false;
        }

        // Materialize the phis a computation asked for, in creation order: each becomes the next
        // value id (the ids the rewrites already name) and the last parameter of its join, and every
        // edge into the join passes that edge's value.
        let mut changed = !created.is_empty();
        for &k in &created {
            let phi = &pending[k as usize];
            let id = ValueId(f.value_types.len() as u32);
            debug_assert_eq!(phi.param, Some(id.0));
            f.value_types.push(phi.ty.clone());
            f.blocks[phi.block as usize].params.push(id);
            for &(p, v) in &phi.incoming {
                push_edge_arg(&mut f.blocks[p as usize].term, phi.block, ValueId(v));
            }
        }
        for b in &mut f.blocks {
            for inst in &mut b.insts {
                map_op_uses(&mut inst.op, |v| {
                    let r = ValueId(resolve(&rewrite, v.0));
                    changed |= r != v;
                    r
                });
            }
            map_term_uses(&mut b.term, |v| {
                let r = ValueId(resolve(&rewrite, v.0));
                changed |= r != v;
                r
            });
        }
        changed
    }
}

/// Append `v` to every arm of `term` that branches to block `target`.
fn push_edge_arg(term: &mut Terminator, target: u32, v: ValueId) {
    match term {
        Terminator::Br { target: t, args } if t.0 == target => args.push(v),
        Terminator::CondBr {
            then_blk,
            then_args,
            else_blk,
            else_args,
            ..
        } => {
            if then_blk.0 == target {
                then_args.push(v);
            }
            if else_blk.0 == target {
                else_args.push(v);
            }
        }
        _ => {}
    }
}

struct Numbering<'a> {
    f: &'a Function,
    /// Dominator-tree children, each list in reverse postorder.
    children: &'a [Vec<u32>],
    preds: &'a [Vec<u32>],
    idom: &'a [u32],
    /// Provenance facts for `f`. Decides which memory-table entries a store or a call has to forget.
    alias: &'a AliasInfo,
    /// Summaries of the side-effect-free callees, when the pass manager supplied them.
    effects: Option<&'a CallEffects>,
    /// Interned argument lists (operands mapped through rewrites), so call keys stay `Copy`.
    arg_lists: FxHashMap<Vec<u32>, u32>,
    scratch: Vec<u32>,
    /// pure-op key -> canonical value id, scoped to the current dominator-tree path.
    vn: FxHashMap<Key, u32>,
    /// pure-op key -> pending phi of a join on the current path that every arm computed it before.
    /// Scoped like `vn`: an entry lives exactly as long as the join's subtree is being numbered.
    pre: FxHashMap<Key, u32>,
    /// value id -> the value it is replaced by (load forwards and CSE rewrites), dense over the
    /// value arena with [`NO_REWRITE`] for "kept". A target is always a value that was already
    /// visited and kept (or a phi), so it is final when recorded: every chain is one step long.
    rewrite: Vec<u32>,
    rewrites: usize,
    /// Interned key types (see [`TyId`]).
    types: TyIds,
    /// Exit memory table of each numbered block that a join successor has yet to read.
    exit_mem: Vec<Option<MemTable>>,
    /// The expressions each `(join, predecessor)` edge offers, recorded when the predecessor is
    /// numbered and consumed when the join is entered.
    exit_exprs: FxHashMap<(u32, u32), Vec<(Key, u32)>>,
    /// Join successors of each block not yet entered; its exit table is dropped at zero.
    joins_left: Vec<u32>,
    entered: Vec<bool>,
    pending: Vec<PendingPhi>,
    /// Materialized pending phis, in creation order. Phi `i` gets value id `base + i`.
    created: Vec<u32>,
    base: u32,
}

/// What a memory-table entry knows about a location's contents.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Avail {
    /// The location holds this value.
    Value(u32),
    /// The location holds a different known value on each edge into a join: this pending phi.
    Phi(u32),
}

type MemTable = FxHashMap<MemKey, Avail>;

/// A value that is available on every edge into `block`, with a different SSA value per edge. It
/// becomes a block parameter only if something at or below the join reads it.
struct PendingPhi {
    block: u32,
    ty: MirType,
    /// `(predecessor, the value on that edge)`, one per predecessor of `block`.
    incoming: Vec<(u32, u32)>,
    param: Option<u32>,
}

/// One join input: a key every predecessor's exit table holds, and whether they agree.
enum Merged {
    Same(Avail),
    /// The per-predecessor values, in predecessor order.
    Differ(Vec<u32>),
}

/// State suspended while a dominator child's subtree is numbered. Keep pending siblings in
/// their parent frame instead of eagerly copying a memory table for every child.
struct Scope {
    block: u32,
    next_child: usize,
    last_inheriting: Option<usize>,
    mem: MemTable,
    added_start: usize,
    pre_start: usize,
}

/// A [`MirType`] folded into one word, so a key is `Copy` and building or probing one never clones a
/// boxed element type (`Vec`/`Array` types carry a `Box`). Scalars, and vectors or arrays of
/// scalars — every type value numbering meets in practice — encode directly; anything nested
/// deeper is interned per pass run. Equal types always map to equal ids and distinct types to
/// distinct ids, so keys compare exactly as the `MirType`s they replace.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct TyId(u64);

#[derive(Default)]
struct TyIds {
    nested: FxHashMap<MirType, u64>,
}

/// A scalar type's code; `None` for `Vec`/`Array`.
fn scalar_code(ty: &MirType) -> Option<u64> {
    Some(match ty {
        MirType::I1 => 1,
        MirType::I8 => 2,
        MirType::I16 => 3,
        MirType::I32 => 4,
        MirType::I64 => 5,
        MirType::F16 => 6,
        MirType::BF16 => 7,
        MirType::F32 => 8,
        MirType::F64 => 9,
        MirType::Ptr => 10,
        MirType::Void => 11,
        MirType::Vec(..) | MirType::Array(..) => return None,
    })
}

impl TyIds {
    /// Bits 62-63 tag the form (scalar, vector-of-scalar, array-of-scalar, interned); a direct
    /// encoding keeps the element code in bits 32-35 and the length in bits 0-31.
    fn id(&mut self, ty: &MirType) -> TyId {
        if let Some(code) = scalar_code(ty) {
            return TyId(code);
        }
        match ty {
            MirType::Vec(elem, n) => {
                if let Some(code) = scalar_code(elem) {
                    return TyId((1 << 62) | (code << 32) | u64::from(*n));
                }
            }
            MirType::Array(elem, n) => {
                if let Some(code) = scalar_code(elem) {
                    return TyId((2 << 62) | (code << 32) | u64::from(*n));
                }
            }
            _ => {}
        }
        if let Some(&id) = self.nested.get(ty) {
            return TyId((3 << 62) | id);
        }
        let id = self.nested.len() as u64;
        self.nested.insert(ty.clone(), id);
        TyId((3 << 62) | id)
    }
}

/// One fact in the memory table.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum MemKey {
    /// A location: an SSA address value plus the type read/written there. The type is part of the
    /// key because a `store f32` and a `load i32` at one address are not the same value — keying on
    /// the address alone would forward the raw bits of one as the other. `bytes` is the access width
    /// that type implies (a function of `ty`, so it adds no distinction), kept here so an
    /// invalidating store can ask the alias analysis without recovering the `MirType`.
    Loc { addr: u32, ty: TyId, bytes: u32 },
    /// The result of a call to a read-only callee with these interned arguments.
    Call { func: u32, args: u32 },
}

/// May a load or store of `ty` participate in forwarding at all?
///
/// `bf16`/`f16` are excluded: their register form is `f32` and the narrowing happens on the way to
/// (2-byte) storage, so forwarding a stored `f32` register straight into a later load would skip the
/// rounding and change the value. That is the open `-O0` != `-O2` half-float defect (gap-hunt 15D);
/// this pass must not widen it.
fn forwardable(ty: &MirType) -> bool {
    !matches!(ty, MirType::F16 | MirType::BF16)
}

/// A typed value-numbering key for a pure op. One variant per cacheable `Op`, carrying its
/// discriminant, operands mapped through prior rewrites, immediate values, and every type that
/// cannot be recovered from the operands. In particular a splat needs its result vector type.
/// `Eq`/`Hash` are derived and no textual key is built; types are [`TyId`]s, so a key is `Copy`.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum Key {
    ConstInt(i128, TyId),
    /// Float immediate keyed by its raw bits (matches the old `x.to_bits()`), so `-0.0`/`NaN`
    /// payloads stay distinct exactly as before.
    ConstFloat(u64, TyId),
    Bin(u8, u32, u32),
    /// Vector comparisons may have different mask element types for the same operands and lane
    /// count. Keep the result type so a comparison never substitutes a differently typed mask.
    Cmp(u8, u32, u32, TyId),
    Neg(u32),
    Not(u32),
    Cast(u8, u32, TyId),
    Select(u32, u32, u32),
    Gep(u32, u32, TyId),
    FuncAddr(u32),
    GlobalAddr(u32),
    /// The scalar operand does not encode the destination lane count. Two splats of one value
    /// into different vector widths are distinct computations even in the same dominator scope.
    Splat(u32, TyId),
    ExtractLane(u32, u32),
    /// Keyed by the whole vector type: `iota <4 x i32>` and `iota <2 x i64>` are different
    /// constants, and only the type distinguishes them (the op has no operands).
    Iota(TyId),
    Fma(u32, u32, u32),
    Sqrt(u32),
    Round(u8, u32),
    /// A call to a callee that reads nothing the caller can see, by symbol and interned arguments:
    /// its result is a function of the argument values alone.
    PureCall(u32, u32),
}

/// Is a value computed on every arm of a join worth a phi there? Not constants or named addresses,
/// which rematerialize for free, and not address arithmetic: a block parameter's provenance is
/// `Unknown` to the alias analysis, so a phi of `gep`s would cost every later memory transform
/// its knowledge of the base for the sake of one instruction.
fn pre_candidate(key: &Key) -> bool {
    !matches!(
        key,
        Key::ConstInt(..)
            | Key::ConstFloat(..)
            | Key::FuncAddr(..)
            | Key::GlobalAddr(..)
            | Key::Iota(..)
            | Key::Gep(..)
    )
}

/// The keys every exit table holds: `Same` when all edges agree on the contents, `Differ` with the
/// per-table values when each edge holds a known value of its own. Anything else is dropped,
/// including disagreement involving a pending phi (no phi of phis).
fn intersect_tables(tables: &[&MemTable]) -> Vec<(MemKey, Merged)> {
    let Some((first, rest)) = tables.split_first() else {
        return Vec::new();
    };
    let mut out = Vec::new();
    'keys: for (key, &a0) in first.iter() {
        let mut same = true;
        for t in rest {
            match t.get(key) {
                Some(&a) => same &= a == a0,
                None => continue 'keys,
            }
        }
        if same {
            out.push((*key, Merged::Same(a0)));
            continue;
        }
        let mut values = Vec::with_capacity(tables.len());
        for t in tables {
            match t.get(key) {
                Some(&Avail::Value(v)) => values.push(v),
                _ => continue 'keys,
            }
        }
        out.push((*key, Merged::Differ(values)));
    }
    out
}

impl Numbering<'_> {
    fn visit(&mut self, entry: u32) {
        let children = self.children;
        let preds = self.preds;
        // One undo log per scoped table serves the whole DFS. Each scope removes its suffix on
        // exit, so sibling computations never become candidates for one another.
        let mut added: Vec<Key> = Vec::new();
        let mut pre_log: Vec<Key> = Vec::new();
        let first = self.enter(entry, MemTable::default(), &mut added, 0);
        let mut stack = vec![first];
        self.record_exit(&stack, &added);
        while let Some(scope) = stack.last_mut() {
            if let Some(&child) = children[scope.block as usize].get(scope.next_child) {
                let i = scope.next_child;
                scope.next_child += 1;
                // Preserve the walk's child order and move the table into its last inheriting
                // child. Straight-line regions still incur no table copy. A join merges its
                // predecessors' exit tables instead.
                let inherited = if preds[child as usize].len() >= 2 {
                    None
                } else if Some(i) == scope.last_inheriting {
                    Some(std::mem::take(&mut scope.mem))
                } else if preds[child as usize].as_slice() == [scope.block] {
                    Some(scope.mem.clone())
                } else {
                    Some(MemTable::default())
                };
                // The join's pending expressions belong to the join's own scope, so they are logged
                // after this mark and retracted when its subtree is done.
                let pre_start = pre_log.len();
                let mem = match inherited {
                    Some(mem) => mem,
                    None => self.join(child, &mut pre_log),
                };
                let entered = self.enter(child, mem, &mut added, pre_start);
                stack.push(entered);
                self.record_exit(&stack, &added);
            } else {
                for key in added.drain(scope.added_start..) {
                    self.vn.remove(&key);
                }
                for key in pre_log.drain(scope.pre_start..) {
                    self.pre.remove(&key);
                }
                stack.pop();
            }
        }
    }

    fn resolve(&self, v: ValueId) -> u32 {
        resolve(&self.rewrite, v.0)
    }

    fn set_rewrite(&mut self, v: ValueId, to: u32) {
        if let Some(slot) = self.rewrite.get_mut(v.0 as usize) {
            *slot = to;
            self.rewrites += 1;
        }
    }

    fn mem_key(&mut self, addr: u32, ty: &MirType) -> MemKey {
        MemKey::Loc {
            addr,
            ty: self.types.id(ty),
            bytes: type_bytes(ty).unwrap_or(u32::MAX),
        }
    }

    /// Intern a call's argument list, mapped through the rewrites so equal calls compare equal.
    fn arg_list(&mut self, args: &[ValueId]) -> u32 {
        self.scratch.clear();
        for &a in args {
            let r = resolve(&self.rewrite, a.0);
            self.scratch.push(r);
        }
        if let Some(&id) = self.arg_lists.get(self.scratch.as_slice()) {
            return id;
        }
        let id = self.arg_lists.len() as u32;
        self.arg_lists.insert(self.scratch.clone(), id);
        id
    }

    /// Value-number `res` under a pure `key`: reuse a dominating equal value, else a phi of the
    /// equal values every arm of an enclosing join computed, else make `res` the canonical value.
    fn number(&mut self, res: ValueId, res_ty: &MirType, key: Key, added: &mut Vec<Key>) {
        if let Some(&canon) = self.vn.get(&key) {
            self.set_rewrite(res, canon);
            return;
        }
        match self.pre.get(&key).copied() {
            Some(k) if self.pending[k as usize].ty == *res_ty => {
                let phi = self.materialize(k);
                self.set_rewrite(res, phi);
            }
            _ => {
                self.vn.insert(key, res.0);
                added.push(key);
            }
        }
    }

    /// Look `key` up in the memory table on behalf of the read `res` of type `ty`: forward a known
    /// value, or record `res` as the new one.
    fn forward(&mut self, mem: &mut MemTable, key: MemKey, res: ValueId, ty: &MirType) {
        match mem.get(&key).copied() {
            Some(a) => match self.avail_value(a, ty) {
                Some(v) => {
                    self.set_rewrite(res, v);
                    mem.insert(key, Avail::Value(v));
                }
                None => {
                    mem.insert(key, Avail::Value(res.0));
                }
            },
            None if mem.len() < MAX_TRACKED_LOCATIONS => {
                mem.insert(key, Avail::Value(res.0));
            }
            None => {}
        }
    }

    /// The type of a value, including a phi this run created.
    fn ty_of(&self, v: u32) -> Option<&MirType> {
        match v.checked_sub(self.base) {
            None => self.f.value_types.get(v as usize),
            Some(i) => {
                let k = *self.created.get(i as usize)?;
                Some(&self.pending[k as usize].ty)
            }
        }
    }

    /// Register a value that join `j` receives as `values[i]` from `preds[i]`. Pointers qualify
    /// only for memory (`allow_ptr`), where the replaced load had no provenance to lose.
    fn new_pending(
        &mut self,
        j: u32,
        preds: &[u32],
        values: &[u32],
        allow_ptr: bool,
    ) -> Option<u32> {
        if self.pending.len() >= MAX_PENDING_PHIS || values.len() != preds.len() {
            return None;
        }
        let ty = self.ty_of(*values.first()?)?.clone();
        if (!allow_ptr && ty == MirType::Ptr) || values.iter().any(|&v| self.ty_of(v) != Some(&ty))
        {
            return None;
        }
        let k = self.pending.len() as u32;
        self.pending.push(PendingPhi {
            block: j,
            ty,
            incoming: preds.iter().copied().zip(values.iter().copied()).collect(),
            param: None,
        });
        Some(k)
    }

    /// The block parameter for pending phi `k`, created on first use.
    fn materialize(&mut self, k: u32) -> u32 {
        if let Some(p) = self.pending[k as usize].param {
            return p;
        }
        let id = self.base + self.created.len() as u32;
        self.pending[k as usize].param = Some(id);
        self.created.push(k);
        id
    }

    /// The value a memory entry names for a read of type `ty`, materializing a pending phi.
    fn avail_value(&mut self, a: Avail, ty: &MirType) -> Option<u32> {
        match a {
            Avail::Value(v) => Some(v),
            Avail::Phi(k) if self.pending[k as usize].ty == *ty => Some(self.materialize(k)),
            Avail::Phi(_) => None,
        }
    }

    /// Record, for every join this block (the top of `stack`) feeds, what holds on that edge: the
    /// block's exit memory table, and the pure expressions computed on its dominator path below the
    /// join's immediate dominator. A join already entered (this edge is a back edge) gets nothing.
    fn record_exit(&mut self, stack: &[Scope], added: &[Key]) {
        let Some(top) = stack.last() else { return };
        let p = top.block;
        if self.joins_left[p as usize] == 0 {
            return;
        }
        self.exit_mem[p as usize] = Some(top.mem.clone());
        let f = self.f;
        for s in cfg::successors(&f.blocks[p as usize].term) {
            let j = s.0;
            if self.preds[j as usize].len() < 2 || self.entered[j as usize] {
                continue;
            }
            // `idom(j)` dominates every predecessor of `j`, so it is on the stack. When it is this
            // block itself, the edge carries nothing the join's dominator does not already have.
            let d = self.idom[j as usize];
            let Some(at) = stack.iter().rposition(|sc| sc.block == d) else {
                continue;
            };
            let Some(below) = stack.get(at + 1) else {
                continue;
            };
            let snapshot: Vec<(Key, u32)> = added[below.added_start..]
                .iter()
                .filter(|key| pre_candidate(key))
                .filter_map(|key| self.vn.get(key).map(|&v| (*key, v)))
                .take(MAX_JOIN_EXPRESSIONS)
                .collect();
            if !snapshot.is_empty() {
                self.exit_exprs.insert((j, p), snapshot);
            }
        }
    }

    /// The memory table a join is entered with, and the expressions its arms make fully redundant
    /// (registered in `pre`, logged in `pre_log`). Facts survive only when *every* predecessor has
    /// already been numbered; otherwise — `j` is a loop header — the table starts empty.
    fn join(&mut self, j: u32, pre_log: &mut Vec<Key>) -> MemTable {
        let all_preds = self.preds;
        let preds = all_preds[j as usize].as_slice();
        let mut mem = MemTable::default();
        if preds.iter().all(|&p| self.exit_mem[p as usize].is_some()) {
            let merged = {
                let tables: Vec<&MemTable> = preds
                    .iter()
                    .filter_map(|&p| self.exit_mem[p as usize].as_ref())
                    .collect();
                intersect_tables(&tables)
            };
            for (key, m) in merged {
                match m {
                    Merged::Same(a) => {
                        mem.insert(key, a);
                    }
                    Merged::Differ(values) => {
                        if let Some(k) = self.new_pending(j, preds, &values, true) {
                            mem.insert(key, Avail::Phi(k));
                        }
                    }
                }
            }
            let snapshots: Option<Vec<Vec<(Key, u32)>>> = preds
                .iter()
                .map(|&p| self.exit_exprs.remove(&(j, p)))
                .collect();
            if let Some(snapshots) = snapshots {
                if let Some((first, rest)) = snapshots.split_first() {
                    for &(key, v0) in first {
                        if self.vn.contains_key(&key) || self.pre.contains_key(&key) {
                            continue;
                        }
                        let mut values = Vec::with_capacity(snapshots.len());
                        values.push(v0);
                        for snapshot in rest {
                            match snapshot.iter().find(|(k, _)| *k == key) {
                                Some(&(_, v)) => values.push(v),
                                None => break,
                            }
                        }
                        if let Some(k) = self.new_pending(j, preds, &values, false) {
                            self.pre.insert(key, k);
                            pre_log.push(key);
                        }
                    }
                }
            }
        }
        // Every predecessor has now delivered (or, on a back edge, will never deliver) its record
        // for this join.
        for &p in preds {
            self.exit_exprs.remove(&(j, p));
            let left = &mut self.joins_left[p as usize];
            *left = left.saturating_sub(1);
            if *left == 0 {
                self.exit_mem[p as usize] = None;
            }
        }
        mem
    }

    /// Value-number `blk`'s parameters by their incoming arguments (see the module note).
    fn number_params(&mut self, blk: u32) {
        let f = self.f;
        let params = &f.blocks[blk as usize].params;
        if params.is_empty() || blk == f.entry.0 || params.len() > MAX_NUMBERED_PARAMS {
            return;
        }
        let all_preds = self.preds;
        let mut edges: Vec<&[ValueId]> = Vec::new();
        for &p in &all_preds[blk as usize] {
            match &f.blocks[p as usize].term {
                Terminator::Br { target, args } if target.0 == blk => edges.push(args),
                Terminator::CondBr {
                    then_blk,
                    then_args,
                    else_blk,
                    else_args,
                    ..
                } => {
                    if then_blk.0 == blk {
                        edges.push(then_args);
                    }
                    if else_blk.0 == blk {
                        edges.push(else_args);
                    }
                }
                _ => {}
            }
        }
        if edges.is_empty() || edges.iter().any(|args| args.len() != params.len()) {
            return;
        }
        for (k, &p) in params.iter().enumerate() {
            // Trivial: one value on every edge, the parameter itself aside.
            let mut only = None;
            let mut trivial = true;
            for args in &edges {
                let a = self.resolve(args[k]);
                if a == p.0 {
                    continue;
                }
                match only {
                    None => only = Some(a),
                    Some(x) if x == a => {}
                    Some(_) => {
                        trivial = false;
                        break;
                    }
                }
            }
            if let (true, Some(x)) = (trivial, only) {
                self.set_rewrite(p, x);
                continue;
            }
            // Duplicate: an earlier, kept parameter that receives the same value on every edge.
            let twin = params[..k].iter().enumerate().find(|&(k2, &q)| {
                self.resolve(q) == q.0
                    && edges
                        .iter()
                        .all(|args| self.resolve(args[k]) == self.resolve(args[k2]))
            });
            if let Some((_, &q)) = twin {
                self.set_rewrite(p, q.0);
            }
        }
    }

    fn enter(
        &mut self,
        blk: u32,
        mut mem: MemTable,
        added: &mut Vec<Key>,
        pre_start: usize,
    ) -> Scope {
        self.entered[blk as usize] = true;
        self.number_params(blk);
        let added_start = added.len();
        let f = self.f;
        for inst in &f.blocks[blk as usize].insts {
            match &inst.op {
                Op::Alloca {
                    ty: _,
                    space: wukong_mir::AddrSpace::Local,
                } => {}
                Op::Load(ptr, ty) => {
                    let p = self.resolve(*ptr);
                    let Some(res) = inst.result else { continue };
                    if !forwardable(ty) {
                        continue;
                    }
                    let key = self.mem_key(p, ty);
                    self.forward(&mut mem, key, res, ty);
                }
                Op::Store { ptr, value } => {
                    let p = self.resolve(*ptr);
                    let v = self.resolve(*value);
                    let sty = f.value_type(*value);
                    let sbytes = AliasInfo::access_bytes(f, &inst.op).unwrap_or(u32::MAX);
                    // Forget exactly the locations this store may have written, and every read-only
                    // call's result unless no callee can reach the stored-to slot.
                    let alias = self.alias;
                    let private = alias.is_private_stack(ValueId(p));
                    mem.retain(|k, _| match *k {
                        MemKey::Loc { addr, bytes, .. } => {
                            !alias.may_alias_sized(ValueId(addr), bytes, ValueId(p), sbytes)
                        }
                        MemKey::Call { .. } => private,
                    });
                    if forwardable(sty) && mem.len() < MAX_TRACKED_LOCATIONS {
                        let key = self.mem_key(p, sty);
                        mem.insert(key, Avail::Value(v));
                    }
                }
                Op::Call { func, args } if self.effects.and_then(|e| e.get(*func)).is_some() => {
                    // Writes nothing: the memory table survives.
                    let reads = self
                        .effects
                        .and_then(|e| e.get(*func))
                        .is_some_and(|e| e.reads_memory);
                    let Some(res) = inst.result else { continue };
                    let list = self.arg_list(args);
                    let res_ty = f.value_type(res);
                    if reads {
                        let key = MemKey::Call {
                            func: func.0,
                            args: list,
                        };
                        self.forward(&mut mem, key, res, res_ty);
                    } else {
                        self.number(res, res_ty, Key::PureCall(func.0, list), added);
                    }
                }
                Op::Call { .. } | Op::VecKernelCall { .. } => {
                    // A call (or vector kernel) may store through any pointer it was given, or any
                    // pointer reachable from one — but never into a stack slot of *this* function
                    // whose address never escaped. What a read-only callee would read may change.
                    let alias = self.alias;
                    mem.retain(|k, _| match *k {
                        MemKey::Loc { addr, .. } => alias.is_private_stack(ValueId(addr)),
                        MemKey::Call { .. } => false,
                    });
                }
                _ => {
                    let Some(res) = inst.result else { continue };
                    let Some(key) = self.pure_key(&inst.op, f.value_type(res)) else {
                        continue;
                    };
                    self.number(res, f.value_type(res), key, added);
                }
            }
        }

        // At most the actual CFG successors can inherit this table. Clone only when more than
        // one child needs it, then move it into the last such child. Straight-line regions incur
        // no table copy, and the existing location cap bounds the cost at conditional branches.
        let last_inheriting = self.children[blk as usize]
            .iter()
            .rposition(|&c| self.preds[c as usize].as_slice() == [blk]);
        Scope {
            block: blk,
            next_child: 0,
            last_inheriting,
            mem,
            added_start,
            pre_start,
        }
    }

    /// A canonical [`Key`] for a pure op, operands mapped through prior rewrites so equal
    /// computations hash identically. Returns `None` for impure/uncacheable ops (handled
    /// separately). The op sub-kinds (`BinOp`/`CmpOp`/`CastKind`/`RoundMode`) are fieldless C-like
    /// enums; `as u8` is their stable discriminant, which distinguishes the same variants the old
    /// `{:?}` did.
    fn pure_key(&mut self, op: &Op, result_ty: &MirType) -> Option<Key> {
        let rw = &self.rewrite;
        let m = |v: ValueId| -> u32 { resolve(rw, v.0) };
        Some(match op {
            Op::ConstInt(n, ty) => Key::ConstInt(*n, self.types.id(ty)),
            Op::ConstFloat(x, ty) => Key::ConstFloat(x.to_bits(), self.types.id(ty)),
            Op::Bin(o, a, b) => {
                let (mut a, mut b) = (m(*a), m(*b));
                // Wrapping integer addition/multiplication and bitwise operations commute at every
                // scalar and vector lane width. Canonicalize only their lookup key, retaining the
                // first dominating computation's operand order in the emitted MIR. Floating-point
                // arithmetic is deliberately excluded: swapping NaN operands may change payloads.
                if matches!(
                    o,
                    BinOp::Add | BinOp::Mul | BinOp::And | BinOp::Or | BinOp::Xor
                ) && a > b
                {
                    std::mem::swap(&mut a, &mut b);
                }
                Key::Bin(*o as u8, a, b)
            }
            Op::Cmp(o, a, b) => {
                let (mut a, mut b) = (m(*a), m(*b));
                if matches!(o, CmpOp::Eq | CmpOp::Ne) && a > b {
                    std::mem::swap(&mut a, &mut b);
                }
                Key::Cmp(*o as u8, a, b, self.types.id(result_ty))
            }
            Op::Neg(a) => Key::Neg(m(*a)),
            Op::Not(a) => Key::Not(m(*a)),
            Op::Cast(k, a, ty) => {
                let a = m(*a);
                Key::Cast(*k as u8, a, self.types.id(ty))
            }
            Op::Select(c, a, b) => Key::Select(m(*c), m(*a), m(*b)),
            Op::Gep { ptr, index, elem } => {
                let (ptr, index) = (m(*ptr), m(*index));
                Key::Gep(ptr, index, self.types.id(elem))
            }
            Op::FuncAddr(s) => Key::FuncAddr(s.0),
            Op::GlobalAddr(s) => Key::GlobalAddr(s.0),
            Op::Splat(a) => {
                let a = m(*a);
                Key::Splat(a, self.types.id(result_ty))
            }
            Op::ExtractLane(a, k) => Key::ExtractLane(m(*a), *k),
            Op::Iota(ty) => Key::Iota(self.types.id(ty)),
            Op::Fma(a, b, c) => Key::Fma(m(*a), m(*b), m(*c)),
            Op::Sqrt(a) => Key::Sqrt(m(*a)),
            Op::Round(mode, a) => Key::Round(*mode as u8, m(*a)),
            Op::Load(..)
            | Op::Store { .. }
            | Op::Call { .. }
            | Op::VecKernelCall { .. }
            | Op::Alloca { .. }
            | Op::Barrier => return None,
        })
    }
}

/// Follow the rewrite table to the final target. Targets are final when recorded (see
/// [`Numbering::rewrite`]), so this is one lookup; the bound only guards a malformed chain.
fn resolve(rewrite: &[u32], mut v: u32) -> u32 {
    for _ in 0..rewrite.len() {
        match rewrite.get(v as usize) {
            Some(&n) if n != NO_REWRITE && n != v => v = n,
            _ => break,
        }
    }
    v
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CfgAnalyses, Dce, Pass};
    use wukong_mir::Builder;
    use wukong_span::Interner;

    fn count_loads(f: &Function) -> usize {
        f.blocks
            .iter()
            .flat_map(|b| b.insts.iter())
            .filter(|i| matches!(i.op, Op::Load(..)))
            .count()
    }

    fn blob(bytes: u32) -> MirType {
        MirType::Array(Box::new(MirType::I8), bytes)
    }

    /// The shape every `[]T` loop lowers to: a slice fat-pointer header reached through a `gep`,
    /// its `data` field re-loaded per use, with an element store in between. The store writes into
    /// a *different* slot, but before the alias analysis existed this pass could not tell — the
    /// pointer was not literally an `alloca` result, so it cleared its whole table and forwarded
    /// nothing.
    #[test]
    fn header_loads_behind_a_gep_survive_a_store_to_a_disjoint_slot() {
        let mut it = Interner::new();
        let mut b = Builder::new(it.intern("f"), MirType::Void);
        let hdr = b.alloca(blob(16));
        let out = b.alloca(blob(16));
        let z = b.build(MirType::I64, Op::ConstInt(0, MirType::I64));
        let data = b.build(
            MirType::Ptr,
            Op::Gep {
                ptr: hdr,
                index: z,
                elem: MirType::I8,
            },
        );
        let p1 = b.build(MirType::Ptr, Op::Load(data, MirType::Ptr));
        let x = b.build(MirType::F32, Op::Load(p1, MirType::F32));
        b.build_void(Op::Store { ptr: out, value: x });
        let p2 = b.build(MirType::Ptr, Op::Load(data, MirType::Ptr));
        let y = b.build(MirType::F32, Op::Load(p2, MirType::F32));
        b.build_void(Op::Store { ptr: out, value: y });
        b.ret(None);
        let mut f = b.finish();

        assert_eq!(count_loads(&f), 4);
        let mut cache = CfgAnalyses::default();
        assert!(Cse.run_function(&mut f, &mut cache));
        Dce.run_function(&mut f, &mut cache);
        // The second header load forwards to the first, and with it the element load behind it.
        assert_eq!(
            count_loads(&f),
            2,
            "{}",
            wukong_mir::print::print_function(&f, &it)
        );
    }

    /// The same shape, but the intervening store goes through a pointer of unknown provenance —
    /// which may well be the header itself. Nothing may be forwarded.
    #[test]
    fn a_store_through_an_unknown_pointer_still_kills_the_header() {
        let mut it = Interner::new();
        let mut b = Builder::new(it.intern("g"), MirType::Void);
        let hdr = b.add_param(MirType::Ptr);
        let z = b.build(MirType::I64, Op::ConstInt(0, MirType::I64));
        let data = b.build(
            MirType::Ptr,
            Op::Gep {
                ptr: hdr,
                index: z,
                elem: MirType::I8,
            },
        );
        let p1 = b.build(MirType::Ptr, Op::Load(data, MirType::Ptr));
        b.build_void(Op::Store { ptr: p1, value: p1 });
        let p2 = b.build(MirType::Ptr, Op::Load(data, MirType::Ptr));
        b.build_void(Op::Store { ptr: p2, value: p2 });
        b.ret(None);
        let mut f = b.finish();

        assert_eq!(count_loads(&f), 2);
        let mut cache = CfgAnalyses::default();
        Cse.run_function(&mut f, &mut cache);
        assert_eq!(
            count_loads(&f),
            2,
            "{}",
            wukong_mir::print::print_function(&f, &it)
        );
    }

    /// Two loads of one address at different widths are two different values: forwarding the i64
    /// one into the f32 one would hand over raw bits.
    #[test]
    fn a_load_is_keyed_by_its_type_not_only_its_address() {
        let mut it = Interner::new();
        let mut b = Builder::new(it.intern("h"), MirType::Void);
        let slot = b.alloca(blob(8));
        let a = b.build(MirType::I64, Op::Load(slot, MirType::I64));
        let c = b.build(MirType::F32, Op::Load(slot, MirType::F32));
        b.build_void(Op::Store {
            ptr: slot,
            value: a,
        });
        b.build_void(Op::Store {
            ptr: slot,
            value: c,
        });
        b.ret(None);
        let mut f = b.finish();

        let mut cache = CfgAnalyses::default();
        Cse.run_function(&mut f, &mut cache);
        assert_eq!(
            count_loads(&f),
            2,
            "{}",
            wukong_mir::print::print_function(&f, &it)
        );
    }

    use wukong_mir::{BasicBlock, BlockId, Inst, Terminator};

    /// Two loads of *different types* from one slot must not be forwarded to each other.
    ///
    /// The forwarding table used to be keyed by the slot alone, so the second load below was
    /// rewritten to the first one's value and an `f32` operand ended up holding a `<4 x f32>`.
    /// Nothing in the corpus reached it — the front end always addresses a slot through a
    /// `gep slot, 0 : T`, and the two geps are distinct *values*, so the table was never consulted
    /// with two types for one slot. Folding that zero-index gep to the slot itself (a
    /// canonicalization) makes both loads name the slot directly and reaches it immediately.
    #[test]
    fn loads_of_different_types_from_one_slot_are_not_forwarded() {
        let v4 = MirType::Vec(Box::new(MirType::F32), 4);
        let mut interner = Interner::new();
        let mut f = Function {
            name: interner.intern("t"),
            params: Vec::new(),
            ret: MirType::F32,
            blocks: vec![BasicBlock {
                id: BlockId(0),
                params: Vec::new(),
                insts: vec![
                    // v0 = alloca [4 x f32]
                    Inst {
                        result: Some(ValueId(0)),
                        op: Op::Alloca {
                            ty: MirType::Array(Box::new(MirType::F32), 4),
                            space: wukong_mir::AddrSpace::Local,
                        },
                    },
                    // v1 = load <4 x f32> v0   (the vectorizer's lane load)
                    Inst {
                        result: Some(ValueId(1)),
                        op: Op::Load(ValueId(0), v4.clone()),
                    },
                    // v2 = load f32 v0         (the same address, read as one scalar)
                    Inst {
                        result: Some(ValueId(2)),
                        op: Op::Load(ValueId(0), MirType::F32),
                    },
                ],
                term: Terminator::Ret(Some(ValueId(2))),
            }],
            value_types: vec![MirType::Ptr, v4, MirType::F32],
            entry: BlockId(0),
            vec_kernels: Vec::new(),
            param_attrs: Vec::new(),
        };
        assert!(
            wukong_mir::verify::verify_function(&f).is_empty(),
            "the input is well-formed"
        );
        let mut cache = CfgAnalyses::default();
        Cse.run_function(&mut f, &mut cache);
        let errs = wukong_mir::verify::verify_function(&f);
        assert!(errs.is_empty(), "cse produced invalid MIR: {errs:?}");
        assert!(
            matches!(f.blocks[0].term, Terminator::Ret(Some(v)) if v == ValueId(2)),
            "the f32 load must survive as its own value"
        );
    }

    /// A store invalidates *every* view of the slot, not just the one at the stored type: writing
    /// one `f32` lane leaves a previously loaded `<4 x f32>` stale in three lanes.
    #[test]
    fn a_scalar_store_invalidates_a_wider_cached_load() {
        let v4 = MirType::Vec(Box::new(MirType::F32), 4);
        let mut interner = Interner::new();
        let mut f = Function {
            name: interner.intern("t"),
            params: Vec::new(),
            ret: MirType::Void,
            blocks: vec![BasicBlock {
                id: BlockId(0),
                params: Vec::new(),
                insts: vec![
                    Inst {
                        result: Some(ValueId(0)),
                        op: Op::Alloca {
                            ty: MirType::Array(Box::new(MirType::F32), 4),
                            space: wukong_mir::AddrSpace::Local,
                        },
                    },
                    // v1 = load <4 x f32> v0
                    Inst {
                        result: Some(ValueId(1)),
                        op: Op::Load(ValueId(0), v4.clone()),
                    },
                    // v2 = 1.0f32 ; store v2 -> v0
                    Inst {
                        result: Some(ValueId(2)),
                        op: Op::ConstFloat(1.0, MirType::F32),
                    },
                    Inst {
                        result: None,
                        op: Op::Store {
                            ptr: ValueId(0),
                            value: ValueId(2),
                        },
                    },
                    // v3 = load <4 x f32> v0  -- must NOT be forwarded to v1
                    Inst {
                        result: Some(ValueId(3)),
                        op: Op::Load(ValueId(0), v4.clone()),
                    },
                    Inst {
                        result: None,
                        op: Op::Store {
                            ptr: ValueId(0),
                            value: ValueId(3),
                        },
                    },
                ],
                term: Terminator::Ret(None),
            }],
            value_types: vec![MirType::Ptr, v4.clone(), MirType::F32, v4],
            entry: BlockId(0),
            vec_kernels: Vec::new(),
            param_attrs: Vec::new(),
        };
        let mut cache = CfgAnalyses::default();
        Cse.run_function(&mut f, &mut cache);
        assert!(wukong_mir::verify::verify_function(&f).is_empty());
        let reload_survives = f.blocks[0]
            .insts
            .iter()
            .any(|i| matches!(&i.op, Op::Store { value, .. } if *value == ValueId(3)));
        assert!(
            reload_survives,
            "the reload after the scalar store must not be forwarded to the pre-store value"
        );
    }

    /// Keys carry folded type ids instead of `MirType`s. Equal types must still collide and
    /// distinct ones must not — including a vector width the operand cannot encode and an element
    /// type nested too deeply for the direct encoding.
    #[test]
    fn type_keyed_values_merge_only_at_identical_types() {
        let mut it = Interner::new();
        let v4 = MirType::Vec(Box::new(MirType::F32), 4);
        let v8 = MirType::Vec(Box::new(MirType::F32), 8);
        let nested = |n| MirType::Array(Box::new(MirType::Array(Box::new(MirType::I8), 4)), n);
        let mut b = Builder::new(it.intern("t"), MirType::Void);
        let x = b.add_param(MirType::F32);
        let base = b.add_param(MirType::Ptr);
        let i = b.add_param(MirType::I64);
        let s4a = b.build(v4.clone(), Op::Splat(x));
        let s4b = b.build(v4.clone(), Op::Splat(x));
        let s8 = b.build(v8.clone(), Op::Splat(x));
        let gep = |b: &mut Builder, elem: MirType| {
            b.build(
                MirType::Ptr,
                Op::Gep {
                    ptr: base,
                    index: i,
                    elem,
                },
            )
        };
        let g2a = gep(&mut b, nested(2));
        let g2b = gep(&mut b, nested(2));
        let g3 = gep(&mut b, nested(3));
        let sink = it.intern("sink");
        b.build_void(Op::Call {
            func: sink,
            args: vec![s4a, s4b, s8, g2a, g2b, g3],
        });
        b.ret(None);
        let mut f = b.finish();
        assert!(Cse.run_function(&mut f, &mut CfgAnalyses::default()));
        let args = f.blocks[0]
            .insts
            .iter()
            .find_map(|inst| match &inst.op {
                Op::Call { args, .. } => Some(args.clone()),
                _ => None,
            })
            .expect("the call survives");
        assert_eq!(args, vec![s4a, s4a, s8, g2a, g2a, g3]);
        assert!(wukong_mir::verify::verify_function(&f).is_empty());
    }

    // ---- global value numbering: joins, block parameters, fully redundant values ----

    fn count_muls(f: &Function) -> usize {
        f.blocks
            .iter()
            .flat_map(|b| &b.insts)
            .filter(|i| matches!(i.op, Op::Bin(BinOp::Mul, ..)))
            .count()
    }

    /// Run `main() = callee(true) * 10 + callee(false)` for a callee taking one `i1`.
    fn run_both_ways(callee: Function, it: &mut Interner) -> i64 {
        let name = callee.name;
        let mut b = Builder::new(it.intern("main"), MirType::I64);
        let mut results = Vec::new();
        for c in [1, 0] {
            let arg = b.build(MirType::I1, Op::ConstInt(c, MirType::I1));
            results.push(b.build(
                MirType::I64,
                Op::Call {
                    func: name,
                    args: vec![arg],
                },
            ));
        }
        let ten = b.build(MirType::I64, Op::ConstInt(10, MirType::I64));
        let hi = b.build(MirType::I64, Op::Bin(BinOp::Mul, results[0], ten));
        let sum = b.build(MirType::I64, Op::Bin(BinOp::Add, hi, results[1]));
        b.ret(Some(sum));
        let mut p = wukong_mir::Program::new();
        p.funcs = vec![callee, b.finish()];
        let errors = wukong_mir::verify::verify_program(&p);
        assert!(errors.is_empty(), "{errors:?}");
        let main = it.intern("main");
        wukong_interp::run(&p, main, it).unwrap()
    }

    fn cse_then_dce(f: &mut Function) {
        let mut cache = CfgAnalyses::default();
        Cse.run_function(f, &mut cache);
        Dce.run_function(f, &mut cache);
        let errors = wukong_mir::verify::verify_function(f);
        assert!(errors.is_empty(), "{errors:?}");
    }

    /// `if c { *slot = 7 } else { *slot = 9 }; return *slot` — each arm leaves a different known
    /// value in the slot, so the join's load is a phi of the two stored values.
    #[test]
    fn a_load_after_a_diamond_of_stores_becomes_a_phi_of_the_stored_values() {
        let mut it = Interner::new();
        let mut b = Builder::new(it.intern("pick"), MirType::I64);
        let c = b.add_param(MirType::I1);
        let slot = b.alloca(MirType::I64);
        let then_b = b.new_block();
        let else_b = b.new_block();
        let join = b.new_block();
        b.cond_br(c, then_b, vec![], else_b, vec![]);
        for (blk, value) in [(then_b, 7), (else_b, 9)] {
            b.switch_to(blk);
            let v = b.build(MirType::I64, Op::ConstInt(value, MirType::I64));
            b.build_void(Op::Store {
                ptr: slot,
                value: v,
            });
            b.br(join, vec![]);
        }
        b.switch_to(join);
        let x = b.build(MirType::I64, Op::Load(slot, MirType::I64));
        b.ret(Some(x));
        let mut f = b.finish();

        cse_then_dce(&mut f);
        let shown = wukong_mir::print::print_function(&f, &it);
        assert_eq!(count_loads(&f), 0, "{shown}");
        assert_eq!(f.blocks[join.0 as usize].params.len(), 1, "{shown}");
        assert_eq!(run_both_ways(f, &mut it), 79);
    }

    /// The `if` without an `else`: the fall-through edge comes straight from the dominator, whose
    /// own store is still the slot's contents there. With that store the two edges disagree and the
    /// load becomes a phi; without it one edge knows nothing and the load must stay.
    #[test]
    fn a_one_armed_if_merges_with_the_dominators_contents_only_when_it_has_them() {
        for dominator_stores in [true, false] {
            let mut it = Interner::new();
            let mut b = Builder::new(it.intern("half"), MirType::I64);
            let c = b.add_param(MirType::I1);
            let slot = b.alloca(MirType::I64);
            if dominator_stores {
                let five = b.build(MirType::I64, Op::ConstInt(5, MirType::I64));
                b.build_void(Op::Store {
                    ptr: slot,
                    value: five,
                });
            }
            let then_b = b.new_block();
            let join = b.new_block();
            b.cond_br(c, then_b, vec![], join, vec![]);
            b.switch_to(then_b);
            let seven = b.build(MirType::I64, Op::ConstInt(7, MirType::I64));
            b.build_void(Op::Store {
                ptr: slot,
                value: seven,
            });
            b.br(join, vec![]);
            b.switch_to(join);
            let x = b.build(MirType::I64, Op::Load(slot, MirType::I64));
            b.ret(Some(x));
            let mut f = b.finish();

            cse_then_dce(&mut f);
            let shown = wukong_mir::print::print_function(&f, &it);
            if dominator_stores {
                assert_eq!(count_loads(&f), 0, "{shown}");
                assert_eq!(run_both_ways(f, &mut it), 75);
            } else {
                assert_eq!(count_loads(&f), 1, "{shown}");
                // An unwritten stack slot reads as zero on both backends.
                assert_eq!(run_both_ways(f, &mut it), 70);
            }
        }
    }

    /// A loop header is entered before its latch is numbered, so it must start with an empty table:
    /// forwarding the preheader's `0` into the header's load would miss the latch's store.
    #[test]
    fn a_loop_header_never_forwards_a_value_the_latch_may_overwrite() {
        let mut it = Interner::new();
        let mut b = Builder::new(it.intern("count"), MirType::I64);
        let c = b.add_param(MirType::I1);
        let slot = b.alloca(MirType::I64);
        let zero = b.build(MirType::I64, Op::ConstInt(0, MirType::I64));
        let one = b.build(MirType::I64, Op::ConstInt(1, MirType::I64));
        let five = b.build(MirType::I64, Op::ConstInt(5, MirType::I64));
        let three = b.build(MirType::I64, Op::ConstInt(3, MirType::I64));
        let n = b.build(MirType::I64, Op::Select(c, five, three));
        b.build_void(Op::Store {
            ptr: slot,
            value: zero,
        });
        let header = b.new_block();
        let body = b.new_block();
        let exit = b.new_block();
        let i = b.block_param(header, MirType::I64);
        b.br(header, vec![zero]);
        b.switch_to(header);
        let x = b.build(MirType::I64, Op::Load(slot, MirType::I64));
        let k = b.build(MirType::I1, Op::Cmp(CmpOp::Slt, i, n));
        b.cond_br(k, body, vec![], exit, vec![]);
        b.switch_to(body);
        let y = b.build(MirType::I64, Op::Bin(BinOp::Add, x, one));
        b.build_void(Op::Store {
            ptr: slot,
            value: y,
        });
        let i2 = b.build(MirType::I64, Op::Bin(BinOp::Add, i, one));
        b.br(header, vec![i2]);
        b.switch_to(exit);
        b.ret(Some(x));
        let mut f = b.finish();

        cse_then_dce(&mut f);
        assert_eq!(
            count_loads(&f),
            1,
            "{}",
            wukong_mir::print::print_function(&f, &it)
        );
        assert_eq!(run_both_ways(f, &mut it), 53);
    }

    /// `t = x*6` on every arm of a diamond makes the join's `x*6` fully redundant: it becomes a phi
    /// of the arms' products, and only those two multiplies remain. With the product on one arm
    /// only, the join's multiply is partially redundant and stays.
    #[test]
    fn an_expression_computed_on_every_arm_is_fully_redundant_at_the_join() {
        for both_arms in [true, false] {
            let mut it = Interner::new();
            let mut b = Builder::new(it.intern("prod"), MirType::I64);
            let c = b.add_param(MirType::I1);
            let x = b.build(
                MirType::I64,
                Op::Cast(wukong_mir::CastKind::ZExt, c, MirType::I64),
            );
            let six = b.build(MirType::I64, Op::ConstInt(6, MirType::I64));
            let then_b = b.new_block();
            let else_b = b.new_block();
            let join = b.new_block();
            b.cond_br(c, then_b, vec![], else_b, vec![]);
            b.switch_to(then_b);
            b.build(MirType::I64, Op::Bin(BinOp::Mul, x, six));
            b.br(join, vec![]);
            b.switch_to(else_b);
            if both_arms {
                b.build(MirType::I64, Op::Bin(BinOp::Mul, x, six));
            }
            b.br(join, vec![]);
            b.switch_to(join);
            let t = b.build(MirType::I64, Op::Bin(BinOp::Mul, x, six));
            let r = b.build(MirType::I64, Op::Bin(BinOp::Add, t, x));
            b.ret(Some(r));
            let mut f = b.finish();

            let mut cache = CfgAnalyses::default();
            Cse.run_function(&mut f, &mut cache);
            let errors = wukong_mir::verify::verify_function(&f);
            assert!(errors.is_empty(), "{errors:?}");
            let shown = wukong_mir::print::print_function(&f, &it);
            let join_muls = f.blocks[join.0 as usize]
                .insts
                .iter()
                .filter(|i| matches!(i.op, Op::Bin(BinOp::Mul, ..)))
                .count();
            if both_arms {
                assert_eq!(f.blocks[join.0 as usize].params.len(), 1, "{shown}");
                Dce.run_function(&mut f, &mut cache);
                assert_eq!(count_muls(&f), 2, "{shown}");
            } else {
                assert_eq!(f.blocks[join.0 as usize].params.len(), 0, "{shown}");
                assert_eq!(join_muls, 1, "{shown}");
            }
            // prod(true) = 1*6 + 1 = 7, prod(false) = 0*6 + 0 = 0.
            assert_eq!(run_both_ways(f, &mut it), 70);
        }
    }

    /// Two loop-carried parameters that receive the identical value on every edge are one value,
    /// and a parameter that only ever receives one value (or itself) is that value.
    #[test]
    fn congruent_and_trivial_block_parameters_are_numbered_away() {
        let mut it = Interner::new();
        let mut b = Builder::new(it.intern("params"), MirType::I64);
        let c = b.add_param(MirType::I1);
        let x = b.build(
            MirType::I64,
            Op::Cast(wukong_mir::CastKind::ZExt, c, MirType::I64),
        );
        let zero = b.build(MirType::I64, Op::ConstInt(0, MirType::I64));
        let one = b.build(MirType::I64, Op::ConstInt(1, MirType::I64));
        let five = b.build(MirType::I64, Op::ConstInt(5, MirType::I64));
        let three = b.build(MirType::I64, Op::ConstInt(3, MirType::I64));
        let n = b.build(MirType::I64, Op::Select(c, five, three));
        let header = b.new_block();
        let body = b.new_block();
        let exit = b.new_block();
        let i = b.block_param(header, MirType::I64);
        let j = b.block_param(header, MirType::I64);
        let p = b.block_param(header, MirType::I64);
        b.br(header, vec![zero, zero, x]);
        b.switch_to(header);
        let k = b.build(MirType::I1, Op::Cmp(CmpOp::Slt, i, n));
        b.cond_br(k, body, vec![], exit, vec![]);
        b.switch_to(body);
        let i2 = b.build(MirType::I64, Op::Bin(BinOp::Add, i, one));
        b.br(header, vec![i2, i2, p]);
        b.switch_to(exit);
        let s = b.build(MirType::I64, Op::Bin(BinOp::Add, i, j));
        let r = b.build(MirType::I64, Op::Bin(BinOp::Add, s, p));
        b.ret(Some(r));
        let mut f = b.finish();

        let mut cache = CfgAnalyses::default();
        assert!(Cse.run_function(&mut f, &mut cache));
        let errors = wukong_mir::verify::verify_function(&f);
        assert!(errors.is_empty(), "{errors:?}");
        let shown = wukong_mir::print::print_function(&f, &it);
        let exit_ops: Vec<&Op> = f.blocks[exit.0 as usize]
            .insts
            .iter()
            .map(|inst| &inst.op)
            .collect();
        assert!(
            matches!(exit_ops[0], Op::Bin(BinOp::Add, a, b) if *a == i && *b == i),
            "{shown}"
        );
        assert!(
            matches!(exit_ops[1], Op::Bin(BinOp::Add, _, b) if *b == x),
            "{shown}"
        );
        // params(true) = 5 + 5 + 1 = 11, params(false) = 3 + 3 + 0 = 6.
        assert_eq!(run_both_ways(f, &mut it), 116);
    }

    /// The same diamond written in source: the array element both arms write is read back after
    /// the join. Global value numbering turns the read into a phi, and the observable result is
    /// unchanged at every level.
    #[test]
    fn a_source_level_diamond_of_element_stores_forwards_to_the_read_after_it() {
        let src = "fn pick(c: bool) -> i32 { let mut a: [i32; 2] = [0; 2]; \
                   if c { a[0] = 7; } else { a[0] = 9; } return a[0]; } \
                   fn main() -> i32 { return pick(true) * 10 + pick(false); }";
        let mut it = Interner::new();
        let (module, pd) = wukong_parser::parse_module(src, wukong_span::SourceId(0), &mut it);
        assert!(pd.iter().all(|d| !d.is_error()), "{pd:?}");
        let (sema, sd) = wukong_sema::check(&module, &it);
        assert!(sd.iter().all(|d| !d.is_error()), "{sd:?}");
        let (program, ld) = wukong_mir_build::lower_program(&module, &sema, &mut it);
        assert!(ld.iter().all(|d| !d.is_error()), "{ld:?}");
        let main = it.intern("main");
        let pick = it.intern("pick");
        assert_eq!(wukong_interp::run(&program, main, &it).unwrap(), 79);
        let mut optimized = program.clone();
        crate::PassManager::standard(2).run(&mut optimized);
        assert!(wukong_mir::verify::verify_program(&optimized).is_empty());
        let f = optimized
            .function(pick)
            .expect("pick is not inlined by the bare pipeline");
        assert_eq!(
            count_loads(f),
            0,
            "{}",
            wukong_mir::print::print_function(f, &it)
        );
        assert_eq!(wukong_interp::run(&optimized, main, &it).unwrap(), 79);
    }

    // ---- calls to side-effect-free callees ----

    /// `peek(p) = *p + 1` reads memory but writes none. Two calls with no write in between are one
    /// call; a store to the (escaped) slot they read forces the third to run again.
    #[test]
    fn a_read_only_call_is_forwarded_until_a_store_may_change_what_it_reads() {
        let mut it = Interner::new();
        let peek = it.intern("peek");
        let mut b = Builder::new(peek, MirType::I64);
        let p = b.add_param(MirType::Ptr);
        let v = b.build(MirType::I64, Op::Load(p, MirType::I64));
        let one = b.build(MirType::I64, Op::ConstInt(1, MirType::I64));
        let r = b.build(MirType::I64, Op::Bin(BinOp::Add, v, one));
        b.ret(Some(r));
        let peek_fn = b.finish();

        let mut b = Builder::new(it.intern("main"), MirType::I64);
        let slot = b.alloca(MirType::I64);
        let call = |b: &mut Builder| {
            b.build(
                MirType::I64,
                Op::Call {
                    func: peek,
                    args: vec![slot],
                },
            )
        };
        let c41 = b.build(MirType::I64, Op::ConstInt(41, MirType::I64));
        b.build_void(Op::Store {
            ptr: slot,
            value: c41,
        });
        let first = call(&mut b);
        let second = call(&mut b);
        let c1 = b.build(MirType::I64, Op::ConstInt(1, MirType::I64));
        b.build_void(Op::Store {
            ptr: slot,
            value: c1,
        });
        let third = call(&mut b);
        let s = b.build(MirType::I64, Op::Bin(BinOp::Add, first, second));
        let t = b.build(MirType::I64, Op::Bin(BinOp::Add, s, third));
        b.ret(Some(t));
        let mut program = wukong_mir::Program::new();
        program.funcs = vec![peek_fn, b.finish()];
        let main = it.intern("main");
        assert_eq!(wukong_interp::run(&program, main, &it).unwrap(), 86);

        let calls = |f: &Function| {
            f.blocks
                .iter()
                .flat_map(|b| &b.insts)
                .filter(|i| matches!(i.op, Op::Call { .. }))
                .count()
        };
        // Without summaries every call is opaque.
        let mut opaque = program.funcs[1].clone();
        let mut cache = CfgAnalyses::default();
        Cse.run_function(&mut opaque, &mut cache);
        Dce.run_function(&mut opaque, &mut cache);
        assert_eq!(calls(&opaque), 3);

        let table = crate::effects::CallEffects::of(&program);
        let mut cache = CfgAnalyses::managed(Some(std::rc::Rc::new(table)));
        let f = &mut program.funcs[1];
        assert!(Cse.run_function(f, &mut cache));
        Dce.run_function(f, &mut cache);
        assert_eq!(calls(f), 2, "{}", wukong_mir::print::print_function(f, &it));
        assert!(wukong_mir::verify::verify_program(&program).is_empty());
        assert_eq!(wukong_interp::run(&program, main, &it).unwrap(), 86);
    }
}
