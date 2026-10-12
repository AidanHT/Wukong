//! Stack-slot lifetimes: for every `Op::Alloca` of a function, the program points at which its
//! storage may still be read — what a backend needs to let two slots with disjoint lifetimes share
//! one frame range (stack coloring).
//!
//! A large kernel declares many scratch arrays that are never in use at the same time; giving each
//! its own slot makes the frame the *sum* of their sizes, where the *maximum* of every set of
//! simultaneously-live ones would do. This module answers the one question coloring needs — *can
//! these two slots hold a live value at the same moment?* — and nothing else. It never rewrites MIR
//! and needs no lifetime-marker instruction: every MIR consumer keeps its exhaustive `Op` matches,
//! and the interpreter (the oracle) keeps one slot per alloca.
//!
//! # Uses and derived values
//!
//! A slot is reached only through its address, so the analysis follows that address. A value is
//! **derived** from an alloca when it is the alloca itself, a `gep` off a derived value, a `select`
//! either of whose arms is derived, a block parameter some edge passes a derived value to, an
//! `inttoptr`/`bitcast` of one, a value **loaded** from a slot the derived address was stored into
//! (see "contents" below), or the pointer result of a runtime call handed a derived argument. Each
//! value carries the set of allocas it may be derived from (its *roots*). An instruction or
//! terminator **uses** an alloca when any operand has it among its roots — every access to the slot,
//! every call that receives its address, every branch that carries it, every comparison of it.
//!
//! # Liveness
//!
//! A slot is live at a program point when the point is **reachable from a use and can reach a use**
//! of it along control flow. This is exact for straight-line code (first use to last use) and
//! CFG-aware everywhere else: a slot used inside a loop is live around the whole loop (every point
//! of the cycle is both after and before a use), and a slot used before and after a loop is live
//! through it. Both directions are one bitset dataflow over the blocks; inside a block the live set
//! of a slot is then a single interval, so a slot's lifetime is a short sorted list of intervals
//! over a global numbering of program points (block by block, each instruction one point and the
//! terminator one more).
//!
//! Two slots **interfere** when their lifetimes intersect. The argument that sharing storage between
//! two that do not is invisible: every read of a slot reads a value an earlier write of it left
//! (on the path taken), both are uses, so every point between them is live for the slot — and a use
//! of the other slot there would be a point where both are live.
//!
//! # Full overwrites
//!
//! Access-only liveness alone keeps a scratch array that is declared inside a loop — one per head,
//! per row, per step — live around the *whole* loop, because its use in one iteration reaches its
//! use in the next along the back edge; two such arrays then always interfere. But the array is
//! re-initialized at its `let` on every iteration: its contents from the previous iteration are
//! never read. The analysis recognizes that re-initialization when it is a counted loop that
//! writes every byte of the slot before anything reads it (the `[v; N]` fill `mir_build` emits for
//! an array initializer, or any `for i in 0..N { x[i] = .. }` over the whole array — see
//! `full_overwrite_kills` for the exact conditions), and **kills** the slot on that loop's entering
//! edges: walking backward from a later use, the "can reach a use" half of liveness stops there.
//! The argument above still holds — a read observes the full overwrite (or a later write), never a
//! value from before it, so nothing before the overwrite needs to be preserved. What is not
//! recognized keeps the slot live: a 2-D loop nest, a runtime kernel that fills the slot, and at
//! `-O2` the shapes the unroller and loop fusion leave (an induction variable advanced by a chain of
//! adds, a vector body plus a scalar remainder).
//!
//! # Escapes
//!
//! An address the analysis loses track of makes its slot live for the **whole function**: its
//! lifetime can no longer be bounded by the uses it can see. A slot escapes when a derived value is
//!
//! * stored into memory that is not provably one of this function's tracked slots (a parameter's
//!   storage, a pointer loaded from memory, the heap);
//! * returned, cast to an integer (`ptrtoint`), used as a `gep` index, fed to arithmetic, or used by
//!   any op that is not one of the address-preserving ones above;
//! * passed to an `extern "C"` function, to a user function whose parameter may retain or return it
//!   ([`CaptureSummary`]), or to any call that also receives a function address (directly or
//!   through a block parameter) — the runtime then calls *user* code with it, which is how an
//!   outlined `@parallel` region captures its environment — or to a runtime entry point some call
//!   in the program hands a function address, wherever its function argument came from;
//! * stored into a slot that itself escapes, or held in a slot whose address is passed to any call
//!   but a synthesized vector kernel (the callee could read the stored address and do anything with
//!   it).
//!
//! # Contents
//!
//! Storing a derived address into a tracked slot `x` (a slice's fat pointer, a vector kernel's
//! stream table) does not lose it: `x`'s *contents* record the roots, every value loaded from `x` —
//! of any type, since an aggregate copy moves a pointer's bits as an integer — carries them, and a
//! synthesized vector kernel (`Op::VecKernelCall`), which reads its stream addresses out of the
//! table it is handed, uses everything the table holds.
//!
//! # What the runtime is assumed to do
//!
//! A call to a symbol that is neither a user function nor an `extern "C"` declaration is a
//! `wukong_*` runtime entry point or a printing intrinsic. Those run to completion before they
//! return, never keep an argument's address past the call, and never store one argument's address
//! into another argument's memory — they are numeric kernels and formatters. A pointer they return
//! may point into an argument, so it is treated as derived from every argument. The one exception
//! is an entry point that calls user code back (see "Escapes"). A runtime entry point that ever broke
//! this would have to be classified here.

use wukong_mir::{BlockId, CastKind, Function, MirType, Op, Program, Terminator, ValueId};
use wukong_span::Symbol;

use crate::alias::type_bytes;
use crate::fxhash::{FxHashMap, FxHashSet};
use crate::loop_info::{self, AccessKind, AddrForm, Coeff, MemBase, Stages, TripCount};
use crate::{cfg, dom};

/// What a call may do with the pointers it is handed, per user function and parameter position.
///
/// A parameter is **captured** when a call can retain or return a pointer derived from that
/// argument: the callee stores it into memory it does not own, returns it, converts it to an
/// integer, passes it on to a capturing position, and so on — exactly the conditions that make one
/// of the callee's own slots escape, applied to the parameter. Computed bottom-up over the call
/// graph as a greatest fixpoint: a recursive call is first assumed not to capture, and the
/// assumption is checked against every body until nothing changes, so a capture anywhere in a
/// cycle reaches every member that passes the pointer along.
#[derive(Clone, Debug, Default)]
pub struct CaptureSummary {
    /// Per user function: per parameter, may a call retain or return a pointer derived from it?
    user: FxHashMap<Symbol, Vec<bool>>,
    /// The program's `extern "C"` functions, which capture everything they are handed.
    externs: FxHashSet<Symbol>,
    /// Runtime entry points that some call in the program hands a function address — they call
    /// user code back with their other arguments (`wukong_parallel_for` and its environment), so
    /// every call to one captures everything, even where its function argument is no longer a
    /// visible `func_addr`.
    callbacks: FxHashSet<Symbol>,
}

/// How a call treats its pointer arguments.
enum Callee<'a> {
    /// A user function; `true` at a parameter position that captures.
    User(&'a [bool]),
    /// An `extern "C"` function: every pointer argument is captured.
    Extern,
    /// A runtime entry point or intrinsic: nothing is captured (see the module note).
    Runtime,
    /// A runtime entry point that calls user code back: every argument is captured.
    Callback,
}

impl CaptureSummary {
    /// Summarize every user function of `program`.
    pub fn analyze(program: &Program) -> CaptureSummary {
        let mut summary = CaptureSummary {
            user: program
                .funcs
                .iter()
                .map(|f| (f.name, vec![false; f.params.len()]))
                .collect(),
            externs: program.externs.iter().map(|e| e.name).collect(),
            callbacks: FxHashSet::default(),
        };
        for f in &program.funcs {
            let fn_addr = func_addr_values(f);
            for inst in f.blocks.iter().flat_map(|b| &b.insts) {
                if let Op::Call { func, args } = &inst.op {
                    let runtime =
                        !summary.user.contains_key(func) && !summary.externs.contains(func);
                    if runtime
                        && args
                            .iter()
                            .any(|a| fn_addr.get(a.0 as usize).copied().unwrap_or(false))
                    {
                        summary.callbacks.insert(*func);
                    }
                }
            }
        }
        // Reverse call graph: a function whose summary grows sends its callers back to the queue.
        let index: FxHashMap<Symbol, usize> = program
            .funcs
            .iter()
            .enumerate()
            .map(|(i, f)| (f.name, i))
            .collect();
        let mut callers: Vec<Vec<usize>> = vec![Vec::new(); program.funcs.len()];
        for (i, f) in program.funcs.iter().enumerate() {
            for inst in f.blocks.iter().flat_map(|b| &b.insts) {
                if let Op::Call { func, .. } = &inst.op {
                    if let Some(&callee) = index.get(func) {
                        if !callers[callee].contains(&i) {
                            callers[callee].push(i);
                        }
                    }
                }
            }
        }
        let mut queued = vec![true; program.funcs.len()];
        let mut queue: std::collections::VecDeque<usize> = (0..program.funcs.len()).collect();
        while let Some(i) = queue.pop_front() {
            queued[i] = false;
            let f = &program.funcs[i];
            if !f
                .params
                .iter()
                .any(|&p| matches!(f.value_type(p), MirType::Ptr))
            {
                continue;
            }
            let flow = Flow::run(f, &summary);
            let captured: Vec<bool> = f
                .params
                .iter()
                .map(|&p| flow.param_root(p).is_some_and(|r| flow.escaped[r]))
                .collect();
            let slot = summary
                .user
                .get_mut(&f.name)
                .expect("every function summarized");
            if *slot != captured {
                // Monotone: a recomputation only ever adds captures (more captured callees can
                // only make more of this body escape), so the queue drains.
                *slot = captured;
                for &c in &callers[i] {
                    if !queued[c] {
                        queued[c] = true;
                        queue.push_back(c);
                    }
                }
            }
        }
        summary
    }

    /// May a call to `callee` retain or return a pointer derived from argument number `arg`?
    /// Anything this summary does not know as a user function is answered by the module note: an
    /// `extern "C"` function captures, a runtime entry point does not.
    pub fn captures(&self, callee: Symbol, arg: usize) -> bool {
        match self.callee(callee) {
            Callee::User(flags) => flags.get(arg).copied().unwrap_or(true),
            Callee::Extern | Callee::Callback => true,
            Callee::Runtime => false,
        }
    }

    fn callee(&self, sym: Symbol) -> Callee<'_> {
        if let Some(flags) = self.user.get(&sym) {
            Callee::User(flags)
        } else if self.externs.contains(&sym) {
            Callee::Extern
        } else if self.callbacks.contains(&sym) {
            Callee::Callback
        } else {
            Callee::Runtime
        }
    }
}

/// The lifetime of every `Op::Alloca` in one function. See the module note.
#[derive(Clone, Debug)]
pub struct SlotLifetimes {
    slots: Vec<Slot>,
    /// The first program point of each block, indexed by block id.
    block_base: Vec<u32>,
}

/// One alloca's lifetime.
#[derive(Clone, Debug)]
struct Slot {
    value: ValueId,
    /// The address left the analysis: live everywhere.
    escaped: bool,
    /// Sorted, disjoint, non-adjacent inclusive ranges of program points (meaningless when
    /// `escaped`).
    live: Vec<(u32, u32)>,
}

impl SlotLifetimes {
    /// Analyze `f`, resolving its calls through `calls` (built from the program `f` belongs to).
    pub fn analyze(f: &Function, calls: &CaptureSummary) -> SlotLifetimes {
        let mut block_base = Vec::with_capacity(f.blocks.len());
        let mut next = 0u32;
        for b in &f.blocks {
            block_base.push(next);
            next = next.saturating_add(b.insts.len() as u32 + 1);
        }
        let flow = Flow::run(f, calls);
        let na = flow.allocas.len();
        let mut slots: Vec<Slot> = flow
            .allocas
            .iter()
            .enumerate()
            .map(|(i, &value)| Slot {
                value,
                escaped: flow.escaped[i],
                live: Vec::new(),
            })
            .collect();
        // One slot (or none) has nothing to share a frame range with, and an escaped slot is live
        // everywhere: only the remaining ones need the dataflow.
        if slots.iter().filter(|s| !s.escaped).count() >= 2 {
            let kills = full_overwrite_kills(f, &flow);
            let ranges = live_ranges(f, &flow.uses, &kills, na, &block_base);
            for (slot, live) in slots.iter_mut().zip(ranges) {
                slot.live = live;
            }
        }
        SlotLifetimes { slots, block_base }
    }

    /// Every alloca of the function, in layout order (block by block, then instruction order).
    pub fn allocas(&self) -> impl ExactSizeIterator<Item = ValueId> + '_ {
        self.slots.iter().map(|s| s.value)
    }

    /// The number of allocas.
    pub fn len(&self) -> usize {
        self.slots.len()
    }

    /// Does the function have no alloca at all?
    pub fn is_empty(&self) -> bool {
        self.slots.is_empty()
    }

    /// The position of alloca `v` in [`SlotLifetimes::allocas`].
    pub fn index_of(&self, v: ValueId) -> Option<usize> {
        self.slots.iter().position(|s| s.value == v)
    }

    /// Is alloca number `i` live for the whole function — its address escapes the analysis?
    pub fn escapes(&self, i: usize) -> bool {
        self.slots[i].escaped
    }

    /// The program points at which alloca number `i` is live, as sorted inclusive ranges over the
    /// function's point numbering ([`SlotLifetimes::point`]). Empty for a slot that is never used,
    /// and meaningless for one that escapes.
    pub fn live_ranges(&self, i: usize) -> &[(u32, u32)] {
        &self.slots[i].live
    }

    /// The program point of instruction `inst` of block `block` (`inst == insts.len()` names the
    /// terminator).
    pub fn point(&self, block: usize, inst: usize) -> u32 {
        self.block_base[block] + inst as u32
    }

    /// Is alloca number `i` live at instruction `inst` of block `block` (`inst == insts.len()` is
    /// the terminator)?
    pub fn live_at(&self, i: usize, block: usize, inst: usize) -> bool {
        let s = &self.slots[i];
        if s.escaped {
            return true;
        }
        let p = self.point(block, inst);
        s.live.iter().any(|&(lo, hi)| lo <= p && p <= hi)
    }

    /// Can allocas `i` and `j` hold a live value at the same moment? Always `true` when either
    /// escapes, and for a slot against itself.
    pub fn interfere(&self, i: usize, j: usize) -> bool {
        let (a, b) = (&self.slots[i], &self.slots[j]);
        i == j || a.escaped || b.escaped || ranges_intersect(&a.live, &b.live)
    }
}

/// Do two sorted lists of inclusive ranges share a point?
pub fn ranges_intersect(a: &[(u32, u32)], b: &[(u32, u32)]) -> bool {
    let (mut i, mut j) = (0, 0);
    while i < a.len() && j < b.len() {
        let (alo, ahi) = a[i];
        let (blo, bhi) = b[j];
        if alo <= bhi && blo <= ahi {
            return true;
        }
        if ahi < bhi {
            i += 1;
        } else {
            j += 1;
        }
    }
    false
}

/// The union of two sorted lists of inclusive ranges, with touching ranges merged.
pub fn ranges_union(a: &[(u32, u32)], b: &[(u32, u32)]) -> Vec<(u32, u32)> {
    let mut all: Vec<(u32, u32)> = Vec::with_capacity(a.len() + b.len());
    let (mut i, mut j) = (0, 0);
    while i < a.len() || j < b.len() {
        let take_a = j >= b.len() || (i < a.len() && a[i].0 <= b[j].0);
        let r = if take_a {
            i += 1;
            a[i - 1]
        } else {
            j += 1;
            b[j - 1]
        };
        push_range(&mut all, r);
    }
    all
}

/// Append `r` (starting no earlier than the last range) to a sorted range list, merging it into the
/// last range when they overlap or touch.
fn push_range(list: &mut Vec<(u32, u32)>, r: (u32, u32)) {
    if let Some(last) = list.last_mut() {
        if r.0 <= last.1.saturating_add(1) {
            last.1 = last.1.max(r.1);
            return;
        }
    }
    list.push(r);
}

/// The roots of one value: sorted, deduplicated root indices (allocas first, then pointer
/// parameters).
type Roots = Vec<u32>;

/// `dst ∪= src`; whether `dst` grew.
fn union_into(dst: &mut Roots, src: &[u32]) -> bool {
    let mut grew = false;
    for &r in src {
        if let Err(at) = dst.binary_search(&r) {
            dst.insert(at, r);
            grew = true;
        }
    }
    grew
}

/// The address flow of one function: which roots every value may be derived from, what every
/// tracked slot may hold, which roots escape, and where each alloca is used.
struct Flow {
    /// The function's allocas in layout order; root `i < allocas.len()` is `allocas[i]`.
    allocas: Vec<ValueId>,
    /// Each alloca's byte size (`None` when it does not fit the layout).
    alloca_bytes: Vec<Option<u32>>,
    /// Pointer parameters `(position, value)`; root `allocas.len() + k` is `ptr_params[k]`.
    ptr_params: Vec<(usize, ValueId)>,
    /// Per value: the roots it may be derived from.
    roots: Vec<Roots>,
    /// Per value: may it point at memory that is not one of the tracked roots?
    other: Vec<bool>,
    /// Per root: the roots whose addresses may be stored in it.
    contents: Vec<Roots>,
    /// Per root: does its address escape?
    escaped: Vec<bool>,
    /// Per value: may it be a function's address (see [`func_addr_values`])?
    func_addr: Vec<bool>,
    /// Per block: `(alloca, first, last)` instruction positions of its uses there, in alloca order.
    uses: Vec<Vec<(u32, u32, u32)>>,
    /// Per block: `(position, allocas)` of every instruction or terminator that does more with an
    /// alloca's address than compute another address from it (`gep`, `select`, a pointer cast) —
    /// an access, a call, a comparison, a branch carrying it.
    touches: Vec<Vec<(u32, Roots)>>,
}

impl Flow {
    fn run(f: &Function, calls: &CaptureSummary) -> Flow {
        let nv = f.value_types.len();
        let mut allocas = Vec::new();
        let mut alloca_bytes = Vec::new();
        let func_addr = func_addr_values(f);
        // Values whose "may point elsewhere" bit starts clear: it is derived purely by OR-ing their
        // operands' bits, so it only ever moves from clear to set while the sweep iterates.
        let mut derived = vec![false; nv];
        for b in &f.blocks {
            if b.id != f.entry {
                for p in &b.params {
                    if let Some(d) = derived.get_mut(p.0 as usize) {
                        *d = true;
                    }
                }
            }
            for inst in &b.insts {
                let Some(r) = inst.result else { continue };
                let Some(d) = derived.get_mut(r.0 as usize) else {
                    continue;
                };
                match &inst.op {
                    Op::Alloca { ty, .. } => {
                        allocas.push(r);
                        alloca_bytes.push(type_bytes(ty));
                        *d = true;
                    }
                    Op::Gep { .. } | Op::Select(..) => *d = true,
                    Op::Cast(CastKind::Bitcast, _, MirType::Ptr) => *d = true,
                    _ => {}
                }
            }
        }
        let ptr_params: Vec<(usize, ValueId)> = f
            .params
            .iter()
            .enumerate()
            .filter(|(_, &p)| matches!(f.value_type(p), MirType::Ptr))
            .map(|(i, &p)| (i, p))
            .collect();
        let nroots = allocas.len() + ptr_params.len();
        let mut flow = Flow {
            roots: vec![Vec::new(); nv],
            other: derived.iter().map(|d| !d).collect(),
            contents: vec![Vec::new(); nroots],
            escaped: vec![false; nroots],
            func_addr,
            uses: vec![Vec::new(); f.blocks.len()],
            touches: vec![Vec::new(); f.blocks.len()],
            allocas,
            alloca_bytes,
            ptr_params,
        };
        for (i, &a) in flow.allocas.iter().enumerate() {
            flow.roots[a.0 as usize] = vec![i as u32];
            flow.other[a.0 as usize] = false;
        }
        let na = flow.allocas.len();
        for (k, &(_, p)) in flow.ptr_params.iter().enumerate() {
            if let Some(r) = flow.roots.get_mut(p.0 as usize) {
                *r = vec![(na + k) as u32];
            }
        }
        if nroots == 0 {
            return flow;
        }
        // Roots, "points elsewhere" bits and contents grow monotonically to a fixpoint. The IR is
        // ordered by dominance, not layout, and block parameters close cycles, so sweep until
        // nothing moves.
        loop {
            let mut changed = false;
            for b in &f.blocks {
                for inst in &b.insts {
                    changed |= flow.propagate(&inst.op, inst.result, f, calls);
                }
                b.term.for_each_edge(|target, args| {
                    let params = &f.block(target).params;
                    for (&p, &a) in params.iter().zip(args) {
                        changed |= flow.flow_into(p, a);
                    }
                });
            }
            if !changed {
                break;
            }
        }
        // Escapes and uses, from the settled facts.
        for (bi, b) in f.blocks.iter().enumerate() {
            for (ii, inst) in b.insts.iter().enumerate() {
                let used = flow.classify(&inst.op, calls);
                flow.note_uses(bi, ii as u32, &used, !is_address_op(&inst.op));
            }
            let used = flow.classify_term(&b.term);
            flow.note_uses(bi, b.insts.len() as u32, &used, true);
        }
        // Whatever an escaped slot holds is reachable by whoever holds the slot.
        let mut work: Vec<usize> = (0..nroots).filter(|&r| flow.escaped[r]).collect();
        while let Some(r) = work.pop() {
            for &c in &flow.contents[r].clone() {
                if !flow.escaped[c as usize] {
                    flow.escaped[c as usize] = true;
                    work.push(c as usize);
                }
            }
        }
        flow
    }

    /// The root of pointer parameter `p`, if it is one.
    fn param_root(&self, p: ValueId) -> Option<usize> {
        self.ptr_params
            .iter()
            .position(|&(_, v)| v == p)
            .map(|k| self.allocas.len() + k)
    }

    fn roots_of(&self, v: ValueId) -> &[u32] {
        self.roots.get(v.0 as usize).map_or(&[], |r| r)
    }

    fn other_of(&self, v: ValueId) -> bool {
        self.other.get(v.0 as usize).copied().unwrap_or(true)
    }

    /// `dst` takes `src`'s roots and "points elsewhere" bit. Whether anything grew.
    fn flow_into(&mut self, dst: ValueId, src: ValueId) -> bool {
        let (d, s) = (dst.0 as usize, src.0 as usize);
        if d >= self.roots.len() || d == s {
            return false;
        }
        let src_roots = self.roots_of(src).to_vec();
        let src_other = self.other_of(src);
        let mut grew = union_into(&mut self.roots[d], &src_roots);
        if src_other && !self.other[d] {
            self.other[d] = true;
            grew = true;
        }
        grew
    }

    /// `dst` takes `roots` (but is otherwise untouched). Whether anything grew.
    fn add_roots(&mut self, dst: ValueId, roots: &[u32]) -> bool {
        match self.roots.get_mut(dst.0 as usize) {
            Some(d) => union_into(d, roots),
            None => false,
        }
    }

    /// Everything stored in the slots `roots` names, at any depth (not including `roots`
    /// themselves unless a slot holds its own address).
    fn contents_closure(&self, roots: &[u32]) -> Roots {
        let mut out: Roots = Vec::new();
        let mut work: Vec<u32> = roots.to_vec();
        while let Some(r) = work.pop() {
            for &c in &self.contents[r as usize] {
                if let Err(at) = out.binary_search(&c) {
                    out.insert(at, c);
                    work.push(c);
                }
            }
        }
        out
    }

    /// One instruction's contribution to the address flow. Whether anything grew.
    fn propagate(
        &mut self,
        op: &Op,
        result: Option<ValueId>,
        f: &Function,
        calls: &CaptureSummary,
    ) -> bool {
        let mut changed = false;
        match op {
            Op::Gep { ptr, .. } => {
                if let Some(r) = result {
                    changed |= self.flow_into(r, *ptr);
                }
            }
            Op::Select(_, a, b) => {
                if let Some(r) = result {
                    changed |= self.flow_into(r, *a);
                    changed |= self.flow_into(r, *b);
                }
            }
            Op::Cast(CastKind::Bitcast, v, MirType::Ptr) => {
                if let Some(r) = result {
                    changed |= self.flow_into(r, *v);
                }
            }
            // A pointer rebuilt from an integer carries whatever roots that integer's bits came
            // from (a pointer copied through memory as an `i64`); it may point anywhere else too.
            Op::Cast(CastKind::IntToPtr, v, _) => {
                if let Some(r) = result {
                    let roots = self.roots_of(*v).to_vec();
                    changed |= self.add_roots(r, &roots);
                }
            }
            // A value read out of a tracked slot may be any address stored into it.
            Op::Load(p, _) => {
                if let Some(r) = result {
                    let held: Roots = self
                        .roots_of(*p)
                        .iter()
                        .flat_map(|&x| self.contents[x as usize].iter().copied())
                        .collect();
                    let mut held = held;
                    held.sort_unstable();
                    held.dedup();
                    changed |= self.add_roots(r, &held);
                }
            }
            // Storing a derived address into a tracked slot records it as that slot's contents.
            // (Into anything else it escapes; see `classify`.)
            Op::Store { ptr, value } => {
                let stored = self.roots_of(*value).to_vec();
                if !stored.is_empty() && !self.other_of(*ptr) {
                    for x in self.roots_of(*ptr).to_vec() {
                        changed |= union_into(&mut self.contents[x as usize], &stored);
                    }
                }
            }
            // A pointer a runtime entry point returns may point into any argument.
            Op::Call { func, args } => {
                if let (Some(r), Callee::Runtime) = (result, calls.callee(*func)) {
                    if matches!(f.value_type(r), MirType::Ptr) {
                        let mut all: Roots = Vec::new();
                        for a in args {
                            union_into(&mut all, self.roots_of(*a));
                        }
                        changed |= self.add_roots(r, &all);
                    }
                }
            }
            _ => {}
        }
        changed
    }

    /// Mark every root in `roots` escaped.
    fn escape(&mut self, roots: &[u32]) {
        for &r in roots {
            self.escaped[r as usize] = true;
        }
    }

    /// One instruction's escapes, and the roots it uses.
    fn classify(&mut self, op: &Op, calls: &CaptureSummary) -> Roots {
        let mut used: Roots = Vec::new();
        let touch = |flow: &Flow, v: ValueId, used: &mut Roots| {
            union_into(used, flow.roots_of(v));
        };
        match op {
            Op::Alloca { .. }
            | Op::ConstInt(..)
            | Op::ConstFloat(..)
            | Op::FuncAddr(_)
            | Op::GlobalAddr(_)
            | Op::Iota(_)
            | Op::Barrier => {}
            // Address-preserving: the operand's roots flow into the result (`propagate`).
            Op::Gep { ptr, index, .. } => {
                touch(self, *ptr, &mut used);
                // An address used as an index is integer arithmetic on it.
                let idx = self.roots_of(*index).to_vec();
                self.escape(&idx);
                union_into(&mut used, &idx);
            }
            Op::Select(c, a, b) => {
                for v in [c, a, b] {
                    touch(self, *v, &mut used);
                }
            }
            Op::Cast(CastKind::Bitcast, v, MirType::Ptr) | Op::Cast(CastKind::IntToPtr, v, _) => {
                touch(self, *v, &mut used)
            }
            Op::Cmp(_, a, b) => {
                touch(self, *a, &mut used);
                touch(self, *b, &mut used);
            }
            Op::Load(p, _) => touch(self, *p, &mut used),
            Op::Store { ptr, value } => {
                touch(self, *ptr, &mut used);
                touch(self, *value, &mut used);
                let stored = self.roots_of(*value).to_vec();
                if !stored.is_empty() && (self.other_of(*ptr) || self.roots_of(*ptr).is_empty()) {
                    self.escape(&stored);
                }
            }
            Op::Call { func, args } => {
                let callback = args
                    .iter()
                    .any(|a| self.func_addr.get(a.0 as usize).copied().unwrap_or(false));
                let callee = calls.callee(*func);
                for (k, &a) in args.iter().enumerate() {
                    let roots = self.roots_of(a).to_vec();
                    if roots.is_empty() {
                        continue;
                    }
                    union_into(&mut used, &roots);
                    // Whatever the argument's slots hold is handed over with them.
                    let held = self.contents_closure(&roots);
                    self.escape(&held);
                    let captured = callback
                        || match &callee {
                            Callee::User(flags) => flags.get(k).copied().unwrap_or(true),
                            Callee::Extern | Callee::Callback => true,
                            Callee::Runtime => false,
                        };
                    if captured {
                        self.escape(&roots);
                    }
                }
            }
            // A synthesized vector kernel reads its stream addresses out of the `ptrs` table and
            // accesses those streams: it uses everything the table holds.
            Op::VecKernelCall {
                ptrs, scalars, n, ..
            } => {
                touch(self, *ptrs, &mut used);
                let held = self.contents_closure(self.roots_of(*ptrs));
                union_into(&mut used, &held);
                touch(self, *scalars, &mut used);
                let scalars_held = self.contents_closure(self.roots_of(*scalars));
                self.escape(&scalars_held);
                touch(self, *n, &mut used);
            }
            // Every other op computes on its operands' bits: an address that reaches one is lost.
            // Cast(PtrToInt), arithmetic, an atomic's value, lane ops — whatever a later MIR op is.
            other => {
                let mut operands: Vec<ValueId> = Vec::new();
                crate::each_op_use(other, &mut |v| operands.push(v));
                // An atomic's pointer is an ordinary access; only its value is computed on.
                let addressed = match other {
                    Op::AtomicAdd { ptr, .. } => Some(*ptr),
                    _ => None,
                };
                for v in operands {
                    let roots = self.roots_of(v).to_vec();
                    union_into(&mut used, &roots);
                    if Some(v) != addressed {
                        self.escape(&roots);
                    }
                }
            }
        }
        used
    }

    /// A terminator's escapes, and the roots it uses (a branch carrying an address, a condition
    /// computed from a loaded value).
    fn classify_term(&mut self, term: &Terminator) -> Roots {
        let mut used: Roots = Vec::new();
        let mut operands: Vec<ValueId> = Vec::new();
        crate::each_term_use(term, &mut |v| operands.push(v));
        for v in operands {
            union_into(&mut used, self.roots_of(v));
        }
        if let Terminator::Ret(Some(v)) = term {
            let roots = self.roots_of(*v).to_vec();
            self.escape(&roots);
        }
        used
    }

    /// Record that position `at` of block `bi` uses the allocas among `used`.
    fn note_uses(&mut self, bi: usize, at: u32, used: &[u32], touching: bool) {
        let na = self.allocas.len() as u32;
        let list = &mut self.uses[bi];
        let mut slots: Roots = Vec::new();
        for &r in used.iter().filter(|&&r| r < na) {
            slots.push(r);
            match list.iter_mut().find(|u| u.0 == r) {
                Some(u) => u.2 = at,
                None => list.push((r, at, at)),
            }
        }
        if touching && !slots.is_empty() {
            self.touches[bi].push((at, slots));
        }
    }
}

/// Per value of `f`: may it be a function's address — an `Op::FuncAddr` result, or a block
/// parameter, `select` or pointer cast fed by one?
fn func_addr_values(f: &Function) -> Vec<bool> {
    let mut fa = vec![false; f.value_types.len()];
    let set = |fa: &mut Vec<bool>, dst: ValueId, src: ValueId| -> bool {
        let from = fa.get(src.0 as usize).copied().unwrap_or(false);
        match fa.get_mut(dst.0 as usize) {
            Some(d) if from && !*d => {
                *d = true;
                true
            }
            _ => false,
        }
    };
    let mut any = false;
    for inst in f.blocks.iter().flat_map(|b| &b.insts) {
        if let (Some(r), Op::FuncAddr(_)) = (inst.result, &inst.op) {
            if let Some(d) = fa.get_mut(r.0 as usize) {
                *d = true;
                any = true;
            }
        }
    }
    if !any {
        return fa;
    }
    loop {
        let mut changed = false;
        for b in &f.blocks {
            for inst in &b.insts {
                let Some(r) = inst.result else { continue };
                match &inst.op {
                    Op::Select(_, a, c) => {
                        changed |= set(&mut fa, r, *a);
                        changed |= set(&mut fa, r, *c);
                    }
                    Op::Cast(CastKind::Bitcast | CastKind::IntToPtr | CastKind::PtrToInt, v, _) => {
                        changed |= set(&mut fa, r, *v)
                    }
                    _ => {}
                }
            }
            b.term.for_each_edge(|target, args| {
                for (&p, &a) in f.block(target).params.iter().zip(args) {
                    changed |= set(&mut fa, p, a);
                }
            });
        }
        if !changed {
            return fa;
        }
    }
}

/// Does `op` only compute an address from its operands (no access, no comparison, no call)?
fn is_address_op(op: &Op) -> bool {
    matches!(
        op,
        Op::Gep { .. }
            | Op::Select(..)
            | Op::Cast(CastKind::Bitcast, _, MirType::Ptr)
            | Op::Cast(CastKind::IntToPtr, ..)
    )
}

/// The entering edges `(from, header, alloca)` of every counted loop that **overwrites every byte
/// of a slot before anything reads it** — see the module note ("Full overwrites"). Each loop is
/// checked against everything that could make the overwrite partial or observable:
///
/// * inside the loop the slot is touched only by `store`s through pointers derived from that slot
///   alone, of values that are no address of a slot — so nothing in the loop reads the old
///   contents;
/// * every one of those stores is in a block that dominates every latch, so it runs on every
///   iteration, and the loop has a constant trip count `n >= 1` over a primary induction variable
///   with a constant start and a positive constant step `k`, whose values stay far from wrapping;
/// * each store's address is `slot + (iv + c)` elements of one common element size, and the bytes
///   one iteration's stores cover form a single run at least `k` elements long — so the `n`
///   iterations tile `[0, size)` without a gap (one store per iteration, or the `k` copies an
///   unroll by `k` leaves, or a vector store covering `k` lanes);
/// * every way out of the loop is the header's exit test, or an edge into a block that touches no
///   slot and ends in `unreachable` (a bounds-check failure): a path that leaves early never comes
///   back to read the slot.
fn full_overwrite_kills(f: &Function, flow: &Flow) -> Vec<(u32, u32, u32)> {
    // The slot a store may write as part of a full overwrite: through a pointer derived from that
    // slot alone, storing a value that holds no slot's address.
    let store_of = |b: usize, at: u32, slots: &[u32]| -> Option<u32> {
        let [x] = slots else { return None };
        let inst = f.blocks[b].insts.get(at as usize)?;
        let Op::Store { ptr, value } = &inst.op else {
            return None;
        };
        let alone = flow.roots_of(*ptr) == [*x] && !flow.other_of(*ptr);
        let plain = !flow
            .roots_of(*value)
            .iter()
            .any(|&r| (r as usize) < flow.allocas.len());
        (alone && plain).then_some(*x)
    };
    if !flow.touches.iter().enumerate().any(|(b, list)| {
        list.iter()
            .any(|(at, slots)| store_of(b, *at, slots).is_some())
    }) {
        return Vec::new();
    }
    let rpo = cfg::reverse_postorder(f);
    let preds = cfg::predecessors(f);
    let idom = dom::idoms_from(f, &rpo, &preds);
    let mut forest = loop_info::structure_with(f, &idom, &preds);
    let mut consts: FxHashMap<ValueId, i128> = FxHashMap::default();
    for inst in f.blocks.iter().flat_map(|b| &b.insts) {
        if let (Some(r), Op::ConstInt(k, _)) = (inst.result, &inst.op) {
            consts.insert(r, *k);
        }
    }
    let mut kills = Vec::new();
    for li in 0..forest.loops.len() {
        // Per slot touched in the loop: its stores `(block, instruction)`, or `None` once anything
        // else touches it.
        let mut seen: Vec<(u32, Option<Vec<StoreAt>>)> = Vec::new();
        for &b in &forest.loops[li].blocks {
            let bi = b.0 as usize;
            for (at, slots) in &flow.touches[bi] {
                let store = store_of(bi, *at, slots);
                for &x in slots {
                    let at_store = store.map(|_| (b.0, *at));
                    match seen.iter_mut().find(|e| e.0 == x) {
                        Some(e) => match (&mut e.1, at_store) {
                            (Some(list), Some(s)) => list.push(s),
                            (entry, _) => *entry = None,
                        },
                        None => seen.push((x, at_store.map(|s| vec![s]))),
                    }
                }
            }
        }
        let mut cands: Vec<(u32, Vec<StoreAt>)> = seen
            .into_iter()
            .filter(|(x, _)| !flow.escaped[*x as usize])
            .filter_map(|(x, s)| s.map(|s| (x, s)))
            .collect();
        if cands.is_empty() {
            continue;
        }
        cands.sort_unstable();
        loop_info::analyze_one(f, &mut forest, li, Stages::Memory);
        let l = &forest.loops[li];
        let TripCount::Const(n) = l.trip else {
            continue;
        };
        let Some(iv) = l.primary() else { continue };
        let (Some(step), Some(&start)) =
            (iv.step.as_const(), iv.start.and_then(|v| consts.get(&v)))
        else {
            continue;
        };
        let (n, step) = (i128::from(n), i128::from(step));
        // Far from wrapping in any integer type an IV can have (`i32` or wider).
        let small = |v: i128| (0..=i128::from(i32::MAX)).contains(&v);
        if n == 0
            || step <= 0
            || !matches!(iv.ty, MirType::I32 | MirType::I64)
            || !small(start)
            || !small(start + n * step)
        {
            continue;
        }
        let exits_ok = l.abnormal_exits.is_empty()
            && l.exits.iter().all(|&(from, to)| {
                from == l.header
                    || (matches!(f.block(to).term, Terminator::Unreachable)
                        && flow.touches[to.0 as usize].is_empty())
            });
        if !exits_ok {
            continue;
        }
        'slot: for (x, stores) in cands {
            let Some(size) = flow.alloca_bytes[x as usize] else {
                continue;
            };
            // Each store's bytes, relative to `iv * elem`: `[c * elem, c * elem + value)`.
            let mut elem_bytes: Option<i128> = None;
            let mut runs: Vec<(i128, i128)> = Vec::new();
            for (sb, at) in stores {
                if !l.latches.iter().all(|latch| dominates(sb, latch.0, &idom)) {
                    continue 'slot;
                }
                let Some(access) = l.accesses.iter().find(|a| {
                    a.block == BlockId(sb) && a.inst == at as usize && a.kind == AccessKind::Store
                }) else {
                    continue 'slot;
                };
                let AddrForm::Affine {
                    base: MemBase::Outer(base),
                    index,
                    elem,
                } = &access.addr
                else {
                    continue 'slot;
                };
                let (Some(eb), Some(vb)) = (type_bytes(elem), type_bytes(&access.value_ty)) else {
                    continue 'slot;
                };
                let eb = i128::from(eb);
                if *base != flow.allocas[x as usize]
                    || index.coeff != Coeff::Const(1)
                    || !index.terms.is_empty()
                    || eb == 0
                    || elem_bytes.is_some_and(|e| e != eb)
                {
                    continue 'slot;
                }
                elem_bytes = Some(eb);
                let lo = i128::from(index.konst) * eb;
                runs.push((lo, lo + i128::from(vb)));
            }
            let Some(eb) = elem_bytes else { continue };
            // One iteration's stores must form a single run of at least `step` elements.
            runs.sort_unstable();
            let (first, mut end) = runs[0];
            for &(lo, hi) in &runs[1..] {
                if lo > end {
                    continue 'slot;
                }
                end = end.max(hi);
            }
            if end - first < step * eb {
                continue;
            }
            let lo = start * eb + first;
            let hi = (start + (n - 1) * step) * eb + end;
            if lo != 0 || hi < i128::from(size) {
                continue;
            }
            for p in &l.entering {
                kills.push((p.0, l.header.0, x));
            }
        }
    }
    kills
}

/// A store's position: `(block, instruction)`.
type StoreAt = (u32, u32);

/// Does block `a` dominate block `b` (both reachable)?
fn dominates(a: u32, b: u32, idom: &[u32]) -> bool {
    let mut x = b;
    loop {
        if x == a {
            return true;
        }
        match idom.get(x as usize) {
            Some(&up) if up != x && up != u32::MAX => x = up,
            _ => return false,
        }
    }
}

/// Per alloca, its live ranges over the point numbering `block_base` defines — see the module
/// note ("Liveness").
fn live_ranges(
    f: &Function,
    uses: &[Vec<(u32, u32, u32)>],
    kills: &[(u32, u32, u32)],
    na: usize,
    block_base: &[u32],
) -> Vec<Vec<(u32, u32)>> {
    let nb = f.blocks.len();
    let words = na.div_ceil(64);
    let mut use_bits = vec![0u64; nb * words];
    for (b, list) in uses.iter().enumerate() {
        for &(a, _, _) in list {
            use_bits[b * words + a as usize / 64] |= 1 << (a % 64);
        }
    }
    let preds = crate::cfg::predecessors(f);
    let succs: Vec<Vec<u32>> = f
        .blocks
        .iter()
        .map(|b| crate::cfg::successors(&b.term).map(|s| s.0).collect())
        .collect();
    // Per block: the successor edges along which a full overwrite kills some slots, as a mask of
    // the slots that survive the edge.
    let mut survive: Vec<Vec<(u32, Vec<u64>)>> = vec![Vec::new(); nb];
    for &(from, to, x) in kills {
        let list = &mut survive[from as usize];
        let at = match list.iter().position(|e| e.0 == to) {
            Some(at) => at,
            None => {
                list.push((to, vec![u64::MAX; words]));
                list.len() - 1
            }
        };
        list[at].1[x as usize / 64] &= !(1 << (x % 64));
    }
    // Reverse postorder first, then any unreachable block (they are never emitted, but analysing
    // them only adds conservatism).
    let mut order = crate::cfg::reverse_postorder(f);
    let mut seen = vec![false; nb];
    for &b in &order {
        seen[b as usize] = true;
    }
    order.extend((0..nb as u32).filter(|&b| !seen[b as usize]));

    // reach_in[b]: some use of the slot reaches b's entry. reach_out[b]: from b's exit some use of
    // the slot is reachable.
    let mut reach_in = vec![0u64; nb * words];
    let mut reach_out = vec![0u64; nb * words];
    let mut acc = vec![0u64; words];
    loop {
        let mut changed = false;
        for &b in &order {
            let b = b as usize;
            acc.iter_mut().for_each(|w| *w = 0);
            for &p in &preds[b] {
                let p = p as usize;
                for w in 0..words {
                    acc[w] |= reach_in[p * words + w] | use_bits[p * words + w];
                }
            }
            for w in 0..words {
                let cur = &mut reach_in[b * words + w];
                if *cur | acc[w] != *cur {
                    *cur |= acc[w];
                    changed = true;
                }
            }
        }
        if !changed {
            break;
        }
    }
    loop {
        let mut changed = false;
        for &b in order.iter().rev() {
            let b = b as usize;
            acc.iter_mut().for_each(|w| *w = 0);
            for &s in &succs[b] {
                let mask = survive[b].iter().find(|e| e.0 == s).map(|e| &e.1[..]);
                let s = s as usize;
                for w in 0..words {
                    let live = reach_out[s * words + w] | use_bits[s * words + w];
                    acc[w] |= mask.map_or(live, |m| live & m[w]);
                }
            }
            for w in 0..words {
                let cur = &mut reach_out[b * words + w];
                if *cur | acc[w] != *cur {
                    *cur |= acc[w];
                    changed = true;
                }
            }
        }
        if !changed {
            break;
        }
    }

    let mut live: Vec<Vec<(u32, u32)>> = vec![Vec::new(); na];
    let mut first_last: Vec<Option<(u32, u32)>> = vec![None; na];
    for (b, block) in f.blocks.iter().enumerate() {
        let term = block.insts.len() as u32;
        for &(a, first, last) in &uses[b] {
            first_last[a as usize] = Some((first, last));
        }
        let bit = |bits: &[u64], a: usize| bits[b * words + a / 64] >> (a % 64) & 1 == 1;
        for (a, slot_live) in live.iter_mut().enumerate() {
            let used_here = first_last[a];
            let lo = if bit(&reach_in, a) {
                Some(0)
            } else {
                used_here.map(|u| u.0)
            };
            let hi = if bit(&reach_out, a) {
                Some(term)
            } else {
                used_here.map(|u| u.1)
            };
            if let (Some(lo), Some(hi)) = (lo, hi) {
                if lo <= hi {
                    push_range(slot_live, (block_base[b] + lo, block_base[b] + hi));
                }
            }
        }
        for &(a, _, _) in &uses[b] {
            first_last[a as usize] = None;
        }
    }
    live
}

#[cfg(test)]
mod tests;
