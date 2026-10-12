//! The 256-bit form of a loop [`super::Vectorize`] widens: a raw-AVX2 [`VecKernel`] recipe called
//! through `Op::VecKernelCall`, the instruction form `wukong_mir_build` builds for the `for` loops
//! it vectorizes itself.
//!
//! The general vectorizer's own vectors stop at 128 bits because that is the widest vector
//! Cranelift legalizes. The 256-bit path is a separate mechanism — the body as a flat recipe
//! (`wukong_mir::vec_kernel`), assembled into AVX2 machine code by
//! `wukong_codegen_cranelift::avx2` and marshalled lane by lane by the interpreter from the same
//! recipe — and until now only the front end could build one. A loop the front end leaves scalar (a
//! `while` over slices, a body that calls a helper the optimizer inlines, a `let` merge it cannot
//! if-convert) reached this pass and stopped at four lanes. Here such a loop takes the recipe too,
//! when three things hold.
//!
//! * **It already widens at 128 bits.** [`plan`] reads the 128-bit plan ([`VecPlan`]) and only
//!   restates it: every step must have a recipe form, so the legality argument is the 128-bit one
//!   — the dependence verdict, the runtime alias guard, the if-conversion — plus the memory rule
//!   below. A body outside the recipe's coverage (a cast, a gather, a float or non-sum reduction, a
//!   shift, the counter read as a value, a countdown) keeps the 128-bit loop exactly as before.
//! * **Every object the loop stores is touched at one element per iteration.** The kernel runs its
//!   groups unrolled ([`KERNEL_UNROLL`] copies of eight lanes) and its loads may run ahead of the
//!   previous copy's stores, so no dependence distance at all may be crossed — not even one a
//!   128-bit plan could accept at a distance of four or more (`o[i + 4] = o[i] * 2`; today's
//!   dependence test declines that too, and this rule keeps the kernel from relying on it).
//!   Requiring every access to a stored object to be at the stored element itself — `o[i] = o[i] *
//!   2` is fine — removes distance from the question: each lane then reads and writes only its own
//!   element of that object, in program order, whatever the grouping. It is the rule `mir_build`'s
//!   recipe path holds its bodies to.
//! * **The trip is long enough to pay for an out-of-line call.** The call marshals a pointer table,
//!   saves the callee-saved vector registers and runs `vzeroupper` — overhead the inlined 128-bit
//!   loop does not pay. [`KERNEL_MIN_TRIP`] is the crossover `mir_build` measured for its
//!   out-of-line 256-bit reduction kernel, `VEC256_REDUCTION_MIN_TRIP`; a compile-time trip below it
//!   keeps the 128-bit loop, and a runtime trip is tested on entry.
//!
//! A loop whose one carried value besides the counter is an **integer sum** (`s = s + e(i)` over
//! `i32` or `i64`, nothing stored) takes a reduction kernel: the recipe folds `e` into `U`
//! accumulators of eight (or four) lanes and returns their wrapping sum, which `kc` adds to the
//! accumulator's value on entry. Integer addition wraps and is associative, so the kernel's
//! grouping is the scalar loop's answer bit for bit. A float sum is not — `(a + b) + c` and `a +
//! (b + c)` differ, and the scalar loop's order is the program — so it keeps the 128-bit path's
//! in-order fold, and so does every other reduction kind.
//!
//! # The rewrite
//!
//! ```text
//!   preheader ──▶ kg ──▶ kc ──▶ kj ──▶ header ──▶ body ──▶ …   (the ORIGINAL loop, untouched)
//!                  │             ▲
//!                  └─────────────┘   (n = 0: no call)
//! ```
//!
//! `kg` computes `n`, the trip rounded down to a multiple of the lane count — zero below
//! [`KERNEL_MIN_TRIP`], or when a runtime alias check fails — and the counter `start + n` the
//! original loop resumes at. `kc` runs only when `n` is not zero: it rebuilds each stream's address
//! at the loop's first counter and each invariant the body splats, stores them into two stack
//! tables, and calls the kernel over `[start, start + n)` (an integer sum's result is added to the
//! accumulator, which reaches `kj` as a block parameter). The original loop then runs the rest.
//!
//! That remainder is where every trip below the crossover goes, so it must not stay scalar: it is
//! still the 128-bit vectorizer's to widen, and only the 256-bit path is told to leave it alone
//! ([`is_remainder`]) — the same treatment the scalar tail of a `mir_build` kernel gets. A loop
//! that runs a long trip therefore executes the kernel, then at most one 128-bit group, then at
//! most three scalar iterations; a short one runs exactly the 128-bit loop it ran before.
//!
//! # Equivalence
//!
//! Every step is lane-wise: lane `k` of a group runs the operations scalar iteration `i + k` runs,
//! in the same order, on the same operands — a float `fma` stays one rounding (`vfmadd`, `mul_add`),
//! a negation is a sign flip, a compare keeps its ordered/unordered NaN rule (`Fone` is Rust's
//! `!=`, the recipe's unordered `Ne`), and a select picks the same arm. So the kernel computes,
//! element for element, what the loop it replaces computes; the interpreter evaluates the same
//! recipe ([`VecKernel::eval_lane`]) and the differential gate polices the machine code against it.
//!
//! Memory is lane-wise too, by the rule above: each lane touches only its own element of a stored
//! object, so the order in which the kernel's groups and lanes run is unobservable. Two accesses
//! to one element are one stream ([`Recipe::stream`] — a slice's data pointer is re-read from its
//! fat pointer between a store and a load, so the two pointers are different values), and the
//! recipe forwards a store to a later load of its stream. Two *objects* the runtime alias guard
//! admits as possibly the same buffer (`guard_allows_equal`: every access at one offset) are two
//! streams that may share a base; the machine code re-reads memory, and the interpreter forwards a
//! lane's store to a load of any stream with the same base, so a load after such a store reads it
//! on both (`tests/run/vec256_opt_aliased.wk`).
//!
//! `WUKONG_P4_NO_256=1` turns this path off exactly as it turns off `mir_build`'s, and so does the
//! pipeline's own switch ([`crate::OptimizeOptions::vec_kernels`] — off for the PTX backend, whose
//! lowering declines a `VecKernelCall` by design) and the host gate
//! ([`wukong_mir::host_supports_vec_kernels`]). None of them changes what a program computes.

use wukong_mir::{
    BinOp, CastKind, CmpOp, Function, Inst, MirType, Op, Terminator, ValueId, VecBin, VecCmp,
    VecKernel, VecLane, VecOp, VecRedOp, VecReduce,
};

use super::{
    const_of, emit_alias_checks, fits_in, op_of, retarget, Emit, GuardReads, LinInst, Plan,
    TripPlan, VecPlan,
};
use crate::each_op_use;
use crate::fxhash::{FxHashMap, FxHashSet};
use crate::loop_info::{
    AccessKind, AddrForm, AffineExpr, MemBase, NaturalLoop, RedKind, TripCount,
};

/// The shortest trip that takes the kernel: `mir_build`'s `VEC256_REDUCTION_MIN_TRIP`, the
/// same-run A/B crossover of its out-of-line 256-bit kernel against the inlined 128-bit form
/// (N=512 0.52x, N=1024 ~0.9x, N=2048 ~1.35x on a compute-bound body). An elementwise call pays
/// less than a reduction's (no horizontal fold), so this is conservative: below it the 128-bit
/// loop this pass would otherwise emit runs unchanged, and it never loses to it. Re-measuring the
/// elementwise crossover per lane type is owed, with a release binary.
pub(super) const KERNEL_MIN_TRIP: u64 = 2048;

/// The trip gate in force: [`KERNEL_MIN_TRIP`], or `WUKONG_VEC256_MIN_TRIP` when it is set to a
/// number (never below one register group). A **test and measurement knob**, result-identical by
/// construction — it only moves which loops run the kernel and which the 128-bit loop — and the
/// way to put every eligible loop of a small corpus program through the kernel:
/// `WUKONG_VEC256_MIN_TRIP=8 cargo test -p wukongc --test run` runs the differential and
/// opt-invariance gates with the kernel executing on every trip of eight or more.
fn min_trip(lanes: u32) -> u64 {
    std::env::var("WUKONG_VEC256_MIN_TRIP")
        .ok()
        .and_then(|v| v.trim().parse::<u64>().ok())
        .unwrap_or(KERNEL_MIN_TRIP)
        .max(u64::from(lanes))
}

/// Unrolled register groups the assembler is asked for — the same request `mir_build`'s
/// elementwise recipes make (`VEC_UNROLL`); it interleaves fewer when the body does not fit.
const KERNEL_UNROLL: u32 = 4;

/// The most kernels this pass adds to one function. A bound on a bug, not a policy: were the
/// remainder marker ([`is_remainder`]) ever to miss, the same loop could otherwise be handed a
/// kernel once per fixpoint iteration.
const MAX_KERNELS_PER_FUNCTION: usize = 32;

/// Everything [`apply`] needs beyond the 128-bit plan, decided before any mutation.
pub(super) struct KernelPlan {
    /// The recipe, named after the function that owns it (`mir_build`'s convention).
    pub(super) kernel: VecKernel,
    /// Stream `k`'s address in the loop body; [`apply`] rebuilds it at the loop's first counter.
    pub(super) streams: Vec<ValueId>,
    /// Scalar `k`'s value: defined outside the loop, or a loop-invariant value the body computes,
    /// which [`apply`] rebuilds ahead of the call.
    pub(super) scalars: Vec<ValueId>,
    /// The trip at the kernel's own width: a compile-time count rounded down to whole groups of
    /// the recipe's lanes, or the runtime bound. [`super::pick`] hands it to the plan [`apply`]
    /// reads, so the alias guard measures exactly the range the kernel covers.
    pub(super) trip: TripPlan,
    /// The shortest trip the kernel runs: a runtime trip below it is tested on entry and leaves
    /// every iteration to the 128-bit loop ([`min_trip`]).
    pub(super) min_trip: u64,
    /// For an integer-sum kernel, the header parameter index of the accumulator the kernel's
    /// result is added to.
    pub(super) sum: Option<usize>,
}

/// May `f` take another kernel from this pass?
pub(super) fn has_room(f: &Function) -> bool {
    f.vec_kernels.len() < MAX_KERNELS_PER_FUNCTION
}

/// Is `l` the scalar remainder of a 256-bit kernel — this pass's, or `mir_build`'s scalar tail?
///
/// Both start their counter where the kernel stopped: at `start + n`, `n` the count the kernel
/// was called with (or that count before it was widened to `i64` or after it was narrowed to the
/// counter's type). That is a structural marker: it survives block renumbering, the preheader
/// being merged into `kg`, and `start` folding to zero (the counter then begins at `n` itself). A
/// count the kernel was given as a constant matches by value, since `n` and the counter's addend
/// are then two constants of different widths.
///
/// The marker only keeps the 256-bit path off the loop; the 128-bit vectorizer still widens it.
pub(super) fn is_remainder(f: &Function, l: &NaturalLoop) -> bool {
    let Some(start) = l.primary().and_then(|iv| iv.start) else {
        return false;
    };
    let mut counts: Vec<ValueId> = Vec::new();
    for b in &f.blocks {
        for inst in &b.insts {
            if let Op::VecKernelCall { n, .. } = inst.op {
                counts.push(n);
                if let Some(Op::Cast(_, x, _)) = op_of(f, n) {
                    counts.push(*x);
                }
            }
        }
    }
    if counts.is_empty() {
        return false;
    }
    let consts: Vec<i128> = counts.iter().filter_map(|&n| const_of(f, n)).collect();
    let is_count = |v: ValueId| {
        counts.contains(&v)
            || matches!(op_of(f, v), Some(Op::Cast(_, x, _)) if counts.contains(x))
            || const_of(f, v).is_some_and(|c| c > 0 && consts.contains(&c))
    };
    is_count(start)
        || matches!(op_of(f, start), Some(Op::Bin(BinOp::Add, a, b)) if is_count(*a) || is_count(*b))
}

/// Restate the 128-bit plan `p` of loop `l` as a 256-bit recipe, or say why it has none.
///
/// `p` must be [`super::plan_loop`]'s plan for `l`: the legality it carries (dependence, alias
/// guards, if-conversion, speculation) is reused, not re-derived. What is added here is the shape
/// the recipe can express and the memory rule from the module docs.
pub(super) fn plan(f: &Function, l: &NaturalLoop, p: &VecPlan) -> Result<KernelPlan, &'static str> {
    // ---- shapes the recipe has no form for ---------------------------------------------------
    if p.dir != 1 {
        return Err("a countdown");
    }
    if p.early.is_some() {
        return Err("a search loop's early exit");
    }
    if !p.len_checks.is_empty() {
        return Err("a bounds check inside an arm");
    }
    // A reduction kernel regroups its addends (`wukong_mir::vec_kernel`'s canonical tree), which
    // computes the scalar loop's serial fold exactly for one kind only: an integer sum, which wraps
    // and is the same under any grouping. A float sum stays with the 128-bit path's in-order fold.
    let red = match p.reductions.as_slice() {
        [] => None,
        [r] if r.kind == RedKind::Add
            && matches!(r.ty, MirType::I32 | MirType::I64)
            && r.guard.is_none()
            && r.arg.is_none()
            && !r.find_last
            && r.select.is_none()
            && r.fma_factors.is_none() =>
        {
            Some(r)
        }
        _ => return Err("a reduction other than one plain integer sum"),
    };
    if !p.groups.is_empty() {
        return Err("an interleave group");
    }
    if p.ramp_vector || !p.ramp.value_use.is_empty() {
        return Err("the counter is read as a value");
    }

    // ---- one lane type ----------------------------------------------------------------------
    // Every lane-varying value is of the recipe's lane type, except a compare's mask (checked on
    // the compare's operands instead). The first contiguous access names it.
    let lane_ty = p
        .lin
        .iter()
        .zip(&p.plans)
        .find_map(|(s, plan)| match (plan, s) {
            (Plan::VecLoad | Plan::VecStore, LinInst::Real { block, idx }) => {
                match &f.blocks[block.0 as usize].insts[*idx].op {
                    Op::Load(_, ty) => Some(ty.clone()),
                    Op::Store { value, .. } => Some(f.value_type(*value).clone()),
                    _ => None,
                }
            }
            _ => None,
        })
        .ok_or("no contiguous access")?;
    let lane = VecLane::of(&lane_ty).ok_or("a lane type the AVX2 recipe does not cover")?;
    if red.is_some_and(|r| r.ty != lane_ty) {
        return Err("a sum of another lane type than the loads");
    }

    // ---- trip ---------------------------------------------------------------------------------
    // Below the crossover a compile-time trip keeps the 128-bit loop outright; a runtime trip is
    // tested on entry ([`emit_kernel_trip`]).
    let lanes = u64::from(lane.lanes());
    let min = min_trip(lane.lanes());
    // The runtime gate compares the trip against `min` in the counter's own type.
    if !fits_in(&p.iv_ty, min as i64) {
        return Err("a counter too narrow to reach the 256-bit crossover");
    }
    let trip = match (&p.trip, &l.trip) {
        (TripPlan::Const(_), TripCount::Const(n)) if *n >= min => {
            TripPlan::Const(n / lanes * lanes)
        }
        (TripPlan::Const(_), _) => return Err("a compile-time trip below the 256-bit crossover"),
        (TripPlan::Runtime { end, signed }, _) => TripPlan::Runtime {
            end: *end,
            signed: *signed,
        },
        (TripPlan::RuntimeDown { .. }, _) => return Err("a countdown"),
    };

    // ---- memory: a stored object is touched at the stored element only -----------------------
    for s in l.accesses.iter().filter(|a| a.kind == AccessKind::Store) {
        let AddrForm::Affine { base, index, elem } = &s.addr else {
            return Err("a store whose address is not affine");
        };
        for a in &l.accesses {
            let same_place = match &a.addr {
                AddrForm::Affine {
                    base: b,
                    index: i,
                    elem: t,
                } => !b.same_object(*base) || (i == index && t == elem),
                AddrForm::Invariant(b) => !b.same_object(*base),
                AddrForm::Unknown => false,
            };
            if !same_place {
                return Err("a stored object is also accessed at another element");
            }
        }
    }

    // ---- the recipe ---------------------------------------------------------------------------
    let iv = p.ramp.iv;
    let in_loop: FxHashSet<u32> = l
        .blocks
        .iter()
        .flat_map(|b| {
            let blk = &f.blocks[b.0 as usize];
            blk.params
                .iter()
                .map(|v| v.0)
                .chain(blk.insts.iter().filter_map(|i| i.result.map(|r| r.0)))
        })
        .collect();
    // Each access's element, by the pointer value it goes through: two pointers that name one
    // element per iteration (a slice's data pointer re-read from its fat pointer between a store and
    // a load) must be ONE stream, or the store would not reach the load ([`Recipe::stream`]).
    let place: FxHashMap<u32, Place> = l
        .accesses
        .iter()
        .filter_map(|a| match &a.addr {
            AddrForm::Affine { base, index, elem } => {
                Some((a.ptr.0, (*base, index.clone(), elem.clone())))
            }
            _ => None,
        })
        .collect();
    let mut b = Recipe {
        f,
        lane,
        lane_ty: &lane_ty,
        iv,
        def: body_defs(f, p),
        in_loop,
        place,
        places: Vec::new(),
        header: f.blocks[p.header.0 as usize]
            .params
            .iter()
            .map(|v| v.0)
            .collect(),
        ops: Vec::new(),
        val: FxHashMap::default(),
        masks: FxHashSet::default(),
        streams: Vec::new(),
        scalars: Vec::new(),
        splat_of: FxHashMap::default(),
    };
    for (step, plan) in p.lin.iter().zip(&p.plans) {
        match (plan, step) {
            // Address arithmetic, constants and invariants: rebuilt ahead of the call where a
            // stream or a scalar needs them, never part of the recipe.
            (Plan::Scalar | Plan::InvariantPtrLoad, LinInst::Real { block, idx }) => {
                if f.blocks[block.0 as usize].insts[*idx].result.is_none() {
                    return Err("a uniform step with an effect");
                }
            }
            (Plan::VecLoad, LinInst::Real { block, idx }) => {
                let inst = &f.blocks[block.0 as usize].insts[*idx];
                let (Op::Load(ptr, ty), Some(r)) = (&inst.op, inst.result) else {
                    return Err("a contiguous load that is not a load");
                };
                if ty != b.lane_ty {
                    return Err("a load of another lane type");
                }
                let stream = b.stream(*ptr)?;
                let v = b.push(VecOp::Load { stream });
                b.val.insert(r.0, v);
            }
            (Plan::VecStore, LinInst::Real { block, idx }) => {
                let Op::Store { ptr, value } = &f.blocks[block.0 as usize].insts[*idx].op else {
                    return Err("a contiguous store that is not a store");
                };
                b.store(*ptr, *value)?;
            }
            (Plan::Widen, LinInst::Real { block, idx }) => {
                b.widen(&f.blocks[block.0 as usize].insts[*idx])?;
            }
            // `if c { o[i] = x } else { o[i] = y }`: one blend, one store.
            (
                Plan::MaskedStore,
                LinInst::MaskedStore {
                    ptr,
                    cond,
                    then_val,
                    else_val,
                    ..
                },
            ) => {
                if f.value_type(*then_val) != b.lane_ty {
                    return Err("a store of another lane type");
                }
                let v = b.select(*cond, *then_val, *else_val)?;
                let stream = b.stream(*ptr)?;
                b.push(VecOp::Store { stream, val: v });
            }
            // A join parameter the arms merge into: one blend.
            (
                Plan::Merge,
                LinInst::Merge {
                    param,
                    cond,
                    then_arg,
                    else_arg,
                },
            ) => {
                if !p.vector_ty.contains_key(&param.0) {
                    return Err("a merge of uniform values");
                }
                if f.value_type(*param) != b.lane_ty {
                    return Err("a merge of another lane type");
                }
                let v = b.select(*cond, *then_arg, *else_arg)?;
                b.val.insert(param.0, v);
            }
            // The sum's combine: folded by the kernel's tree, not an op of the body.
            (Plan::Reduce, LinInst::Real { .. }) if red.is_some() => {}
            _ => return Err("a step the recipe has no form for"),
        }
    }
    let stores = b.ops.iter().any(|o| matches!(o, VecOp::Store { .. }));
    let reduce = match red {
        // A reduction kernel writes no memory: it returns its fold.
        Some(_) if stores => return Err("a sum beside a store"),
        Some(r) => {
            let (addend, _) = r.addend.ok_or("a sum without an addend")?;
            Some(VecReduce {
                op: VecRedOp::Add,
                value: b.value(addend)?,
                fma: None,
            })
        }
        None if !stores => return Err("no store"),
        None => None,
    };
    let kernel = VecKernel {
        name: f.name,
        lane,
        streams: b.streams.len() as u32,
        scalars: b.scalars.len() as u32,
        ops: b.ops,
        unroll: KERNEL_UNROLL,
        reduce,
    };
    // The single-sourced coverage and register gate the assembler plans with: at most
    // `VEC_MAX_STREAMS` streams, no 64-bit integer multiply, every mask read only by a blend, and
    // one register group plus the hoisted broadcasts (and a reduction's accumulators) in sixteen
    // registers.
    let fits = if kernel.reduce.is_some() {
        kernel.reduction_fits()
    } else {
        kernel.pressure().is_some_and(|q| q.fits())
    };
    if !fits {
        return Err("outside the AVX2 emitter's coverage or register file");
    }
    Ok(KernelPlan {
        kernel,
        streams: b.streams,
        scalars: b.scalars,
        trip,
        min_trip: min,
        sum: red.map(|r| r.param_index),
    })
}

/// The element an affine access touches: its object, its index in the counter, its element type.
type Place = (MemBase, AffineExpr, MirType);

/// Every value the flattened body defines through a step that stays one scalar per group — address
/// arithmetic, a constant, an invariant — and where it is defined.
fn body_defs(f: &Function, p: &VecPlan) -> FxHashMap<u32, (u32, usize)> {
    let mut def = FxHashMap::default();
    for (step, plan) in p.lin.iter().zip(&p.plans) {
        if let (Plan::Scalar | Plan::InvariantPtrLoad, LinInst::Real { block, idx }) = (plan, step)
        {
            if let Some(r) = f.blocks[block.0 as usize].insts[*idx].result {
                def.insert(r.0, (block.0, *idx));
            }
        }
    }
    def
}

/// The recipe under construction.
struct Recipe<'a> {
    f: &'a Function,
    lane: VecLane,
    lane_ty: &'a MirType,
    /// The induction variable: an address may be computed from it, a scalar may not.
    iv: ValueId,
    /// [`body_defs`].
    def: FxHashMap<u32, (u32, usize)>,
    /// Every value a block of the loop defines, parameters included.
    in_loop: FxHashSet<u32>,
    /// An affine access's element, by its pointer value.
    place: FxHashMap<u32, Place>,
    /// Stream `k`'s element, when its first pointer has one.
    places: Vec<Option<Place>>,
    /// The loop header's parameters: the counter, and invariants carried around the loop.
    header: FxHashSet<u32>,
    ops: Vec<VecOp>,
    /// Body value -> the recipe value holding its lanes.
    val: FxHashMap<u32, u32>,
    /// Recipe values that are compare masks, readable only as a blend's mask.
    masks: FxHashSet<u32>,
    streams: Vec<ValueId>,
    scalars: Vec<ValueId>,
    /// Scalar index -> the `Splat` that broadcasts it.
    splat_of: FxHashMap<u32, u32>,
}

impl Recipe<'_> {
    fn push(&mut self, op: VecOp) -> u32 {
        self.ops.push(op);
        (self.ops.len() - 1) as u32
    }

    /// The stream an address belongs to, opened on first use. Two accesses to one element share a
    /// stream — through one pointer value, or through two that name the same element of the same
    /// object — so a load after a store of that element reads it back through the recipe's own
    /// store forwarding, as the scalar loop does through memory.
    fn stream(&mut self, ptr: ValueId) -> Result<u32, &'static str> {
        let place = self.place.get(&ptr.0).cloned();
        let same = |q: &Option<Place>| match (q, &place) {
            (Some((b1, i1, e1)), Some((b2, i2, e2))) => b1.same_object(*b2) && i1 == i2 && e1 == e2,
            _ => false,
        };
        if let Some(k) = self
            .streams
            .iter()
            .zip(&self.places)
            .position(|(&s, q)| s == ptr || same(q))
        {
            return Ok(k as u32);
        }
        if !self.rebuildable(ptr, true, 0) {
            return Err("a stream address that cannot be rebuilt ahead of the loop");
        }
        self.streams.push(ptr);
        self.places.push(place);
        Ok((self.streams.len() - 1) as u32)
    }

    /// The recipe value of a body value read as an arithmetic operand: its lanes when it varies
    /// per lane, otherwise a broadcast of the invariant, pre-loaded as a scalar.
    fn value(&mut self, v: ValueId) -> Result<u32, &'static str> {
        if let Some(&k) = self.val.get(&v.0) {
            if self.masks.contains(&k) {
                return Err("a compare read as a value");
            }
            return Ok(k);
        }
        if self.f.value_type(v) != self.lane_ty {
            return Err("an invariant of another lane type");
        }
        if !self.rebuildable(v, false, 0) {
            return Err("an operand that is neither lane-varying nor invariant");
        }
        let scalar = match self.scalars.iter().position(|&s| s == v) {
            Some(k) => k as u32,
            None => {
                self.scalars.push(v);
                (self.scalars.len() - 1) as u32
            }
        };
        if let Some(&k) = self.splat_of.get(&scalar) {
            return Ok(k);
        }
        let k = self.push(VecOp::Splat { scalar });
        self.splat_of.insert(scalar, k);
        Ok(k)
    }

    /// A blend's mask: the recipe value of a lane compare.
    fn mask(&self, c: ValueId) -> Result<u32, &'static str> {
        match self.val.get(&c.0) {
            Some(k) if self.masks.contains(k) => Ok(*k),
            _ => Err("a blend whose condition is not a lane compare"),
        }
    }

    fn select(&mut self, c: ValueId, a: ValueId, b: ValueId) -> Result<u32, &'static str> {
        let mask = self.mask(c)?;
        let a = self.value(a)?;
        let b = self.value(b)?;
        Ok(self.push(VecOp::Select { mask, a, b }))
    }

    fn store(&mut self, ptr: ValueId, value: ValueId) -> Result<(), &'static str> {
        if self.f.value_type(value) != self.lane_ty {
            return Err("a store of another lane type");
        }
        let val = self.value(value)?;
        let stream = self.stream(ptr)?;
        self.push(VecOp::Store { stream, val });
        Ok(())
    }

    /// One widened instruction as a recipe op, or why it has none.
    fn widen(&mut self, inst: &Inst) -> Result<(), &'static str> {
        let r = inst.result.ok_or("a widened step with no result")?;
        let float = self.lane.is_float();
        let v = match &inst.op {
            Op::Bin(op, x, y) => {
                if self.f.value_type(r) != self.lane_ty {
                    return Err("arithmetic of another lane type");
                }
                let op = match (op, float) {
                    (BinOp::FAdd, true) | (BinOp::Add, false) => VecBin::Add,
                    (BinOp::FSub, true) | (BinOp::Sub, false) => VecBin::Sub,
                    (BinOp::FMul, true) | (BinOp::Mul, false) => VecBin::Mul,
                    (BinOp::FDiv, true) => VecBin::Div,
                    _ => return Err("an operator the recipe has no lane op for"),
                };
                let a = self.value(*x)?;
                let b = self.value(*y)?;
                self.push(VecOp::Bin { op, a, b })
            }
            Op::Fma(x, y, z) if float => {
                let a = self.value(*x)?;
                let b = self.value(*y)?;
                let c = self.value(*z)?;
                self.push(VecOp::Fma { a, b, c })
            }
            Op::Sqrt(x) if float => {
                let a = self.value(*x)?;
                self.push(VecOp::Sqrt { a })
            }
            Op::Neg(x) => {
                let a = self.value(*x)?;
                self.push(VecOp::Neg { a })
            }
            Op::Cmp(pred, x, y) => {
                if self.f.value_type(*x) != self.lane_ty {
                    return Err("a compare of another lane type");
                }
                let pred =
                    cmp_of(*pred, float).ok_or("a compare the recipe has no lane form for")?;
                let a = self.value(*x)?;
                let b = self.value(*y)?;
                let k = self.push(VecOp::Cmp { pred, a, b });
                self.masks.insert(k);
                k
            }
            Op::Select(c, x, y) => {
                if self.f.value_type(r) != self.lane_ty {
                    return Err("a blend of another lane type");
                }
                self.select(*c, *x, *y)?
            }
            _ => return Err("an operation the recipe has no lane form for"),
        };
        if self.f.value_type(r) != self.lane_ty && !self.masks.contains(&v) {
            return Err("a result of another lane type");
        }
        self.val.insert(r.0, v);
        Ok(())
    }

    /// Can `v` be computed ahead of the loop — from values defined outside it, the loop's
    /// invariant header parameters, and (when `at_start`) the counter at its first value? A
    /// pointer re-read from a loop-invariant address counts ([`Plan::InvariantPtrLoad`]): the
    /// 128-bit plan already reads it once per entry. Any other load does not.
    fn rebuildable(&self, v: ValueId, at_start: bool, depth: u32) -> bool {
        if depth > 64 {
            return false;
        }
        if v == self.iv {
            return at_start;
        }
        if self.header.contains(&v.0) {
            return true;
        }
        let Some(&(block, idx)) = self.def.get(&v.0) else {
            // Not one of the body's scalar steps: rebuildable as itself only when the loop does
            // not define it at all (a lane value, a merge or the header's exit test does).
            return !self.in_loop.contains(&v.0);
        };
        let inst = &self.f.blocks[block as usize].insts[idx];
        match &inst.op {
            Op::Load(_, MirType::Ptr) => {}
            Op::Load(..)
            | Op::Store { .. }
            | Op::Call { .. }
            | Op::VecKernelCall { .. }
            | Op::AtomicAdd { .. }
            | Op::Alloca { .. }
            | Op::Barrier => return false,
            _ => {}
        }
        let mut ok = true;
        each_op_use(&inst.op, &mut |u| {
            ok &= self.rebuildable(u, at_start, depth + 1);
        });
        ok
    }
}

/// The recipe compare for a MIR predicate on a lane of this kind, matching each Rust operator on
/// every input: the ordered float compares are false on NaN, `Fone` (Rust's `!=`) is true on it.
/// An unsigned integer compare has no AVX2 form (`vpcmpgt*` is signed) and stays 128-bit.
fn cmp_of(pred: CmpOp, float: bool) -> Option<VecCmp> {
    Some(match (pred, float) {
        (CmpOp::Folt, true) | (CmpOp::Slt, false) => VecCmp::Lt,
        (CmpOp::Fole, true) | (CmpOp::Sle, false) => VecCmp::Le,
        (CmpOp::Fogt, true) | (CmpOp::Sgt, false) => VecCmp::Gt,
        (CmpOp::Foge, true) | (CmpOp::Sge, false) => VecCmp::Ge,
        (CmpOp::Foeq, true) | (CmpOp::Eq, false) => VecCmp::Eq,
        (CmpOp::Fone, true) | (CmpOp::Ne, false) => VecCmp::Ne,
        _ => return None,
    })
}

/// `n`, the element count the kernel owns: the trip rounded down to whole groups, and zero unless
/// the loop runs at least `min` ([`min_trip`]) times. A runtime count wraps in the counter's type,
/// so the order is tested explicitly with the header's own predicate, exactly as
/// [`super::emit_trip`] does for the 128-bit loop (whose doc has the wrapped-`i32` case).
fn emit_kernel_trip(
    e: &mut Emit<'_>,
    p: &VecPlan,
    guard: wukong_mir::BlockId,
    start: ValueId,
    lanes: u32,
    min: u64,
) -> ValueId {
    match &p.trip {
        TripPlan::Const(n) => e.int(guard, &p.iv_ty, *n as i64),
        TripPlan::Runtime { end, signed } => {
            let count = e.push(guard, p.iv_ty.clone(), Op::Bin(BinOp::Sub, *end, start));
            let mask = e.int(guard, &p.iv_ty, !(i64::from(lanes) - 1));
            let masked = e.push(guard, p.iv_ty.clone(), Op::Bin(BinOp::And, count, mask));
            let min = e.int(guard, &p.iv_ty, min as i64);
            let big = e.push(guard, MirType::I1, Op::Cmp(CmpOp::Sge, count, min));
            let lt = if *signed { CmpOp::Slt } else { CmpOp::Ult };
            let runs = e.push(guard, MirType::I1, Op::Cmp(lt, start, *end));
            let ok = e.push(guard, MirType::I1, Op::Bin(BinOp::And, runs, big));
            let zero = e.int(guard, &p.iv_ty, 0);
            e.push(guard, p.iv_ty.clone(), Op::Select(ok, masked, zero))
        }
        TripPlan::RuntimeDown { .. } => unreachable!("`plan` declines a countdown"),
    }
}

/// Splice the kernel in front of loop `p.header` (module docs, "The rewrite"). `p` is the 128-bit
/// plan with its trip and width already set to the kernel's ([`KernelPlan::trip`]).
pub(super) fn apply(f: &mut Function, p: &VecPlan, k: &KernelPlan) {
    let header_params: Vec<ValueId> = f.blocks[p.header.0 as usize].params.clone();
    let header_tys: Vec<MirType> = header_params
        .iter()
        .map(|v| f.value_type(*v).clone())
        .collect();
    let def = body_defs(f, p);
    let lane_ty = k.kernel.lane.mir_type();
    let lanes = k.kernel.lane.lanes();

    let mut e = Emit { f };
    let gp: Vec<ValueId> = header_tys.iter().map(|t| e.value(t.clone())).collect();
    let kg = e.block(gp.clone(), Terminator::Unreachable);
    let kc = e.block(Vec::new(), Terminator::Unreachable);
    // An integer-sum kernel's accumulator arrives at `kj` from both sides: unchanged when the
    // kernel does not run, plus the kernel's sum when it does.
    let sum_in = k.sum.map(|ix| e.value(header_tys[ix].clone()));
    let kj = e.block(sum_in.into_iter().collect(), Terminator::Unreachable);

    // ---- kg: how many elements does the kernel own? -------------------------------------------
    let start = gp[p.iv_index];
    let n_trip = emit_kernel_trip(&mut e, p, kg, start, lanes, k.min_trip);
    let mut reads = GuardReads::new(p, n_trip);
    let n_vec = match emit_alias_checks(&mut e, p, kg, &mut reads) {
        Some(ok) => {
            let zero = e.int(kg, &p.iv_ty, 0);
            e.push(kg, p.iv_ty.clone(), Op::Select(ok, n_trip, zero))
        }
        None => n_trip,
    };
    let resume = e.push(kg, p.iv_ty.clone(), Op::Bin(BinOp::Add, start, n_vec));
    let zero = e.int(kg, &p.iv_ty, 0);
    let runs = e.push(kg, MirType::I1, Op::Cmp(CmpOp::Ne, n_vec, zero));
    e.f.blocks[kg.0 as usize].term = Terminator::CondBr {
        cond: runs,
        then_blk: kc,
        then_args: Vec::new(),
        else_blk: kj,
        else_args: k.sum.map(|ix| gp[ix]).into_iter().collect(),
    };

    // ---- kc: the two tables and the call ------------------------------------------------------
    // Each stream's address and each scalar, rebuilt from the body's own scalar steps with the
    // counter at `start` and every other header parameter at its incoming value.
    let mut memo: FxHashMap<u32, ValueId> = header_params
        .iter()
        .zip(&gp)
        .map(|(h, g)| (h.0, *g))
        .collect();
    let ptr_vals: Vec<ValueId> = k
        .streams
        .iter()
        .map(|&s| rebuild(&mut e, kc, &def, &mut memo, s))
        .collect();
    let scalar_vals: Vec<ValueId> = k
        .scalars
        .iter()
        .map(|&s| rebuild(&mut e, kc, &def, &mut memo, s))
        .collect();
    let ptrs = e.value(MirType::Ptr);
    let scalars = e.value(MirType::Ptr);
    for (i, &v) in ptr_vals.iter().enumerate() {
        let at = e.int(kc, &MirType::I64, i as i64);
        let slot = e.push(
            kc,
            MirType::Ptr,
            Op::Gep {
                ptr: ptrs,
                index: at,
                elem: MirType::Ptr,
            },
        );
        e.push_void(
            kc,
            Op::Store {
                ptr: slot,
                value: v,
            },
        );
    }
    for (i, &v) in scalar_vals.iter().enumerate() {
        let at = e.int(kc, &MirType::I64, i as i64);
        let slot = e.push(
            kc,
            MirType::Ptr,
            Op::Gep {
                ptr: scalars,
                index: at,
                elem: lane_ty.clone(),
            },
        );
        e.push_void(
            kc,
            Op::Store {
                ptr: slot,
                value: v,
            },
        );
    }
    let n = if p.iv_ty == MirType::I64 {
        n_vec
    } else {
        e.push(
            kc,
            MirType::I64,
            Op::Cast(CastKind::SExt, n_vec, MirType::I64),
        )
    };
    let kernel = e.f.vec_kernels.len() as u32;
    e.f.vec_kernels.push(k.kernel.clone());
    let call = Op::VecKernelCall {
        kernel,
        ptrs,
        scalars,
        n,
    };
    let kc_args = match k.sum {
        // The kernel returns the wrapping sum of its addends; added to the accumulator's value on
        // entry it is the accumulator after the first `n` iterations, in any grouping.
        Some(ix) => {
            let r = e.push(kc, lane_ty.clone(), call);
            vec![e.push(kc, lane_ty.clone(), Op::Bin(BinOp::Add, gp[ix], r))]
        }
        None => {
            e.push_void(kc, call);
            Vec::new()
        }
    };
    e.f.blocks[kc.0 as usize].term = Terminator::Br {
        target: kj,
        args: kc_args,
    };

    // ---- kj: the original loop resumes where the kernel stopped --------------------------------
    let mut args = gp.clone();
    args[p.iv_index] = resume;
    if let (Some(ix), Some(acc)) = (k.sum, sum_in) {
        args[ix] = acc;
    }
    e.f.blocks[kj.0 as usize].term = Terminator::Br {
        target: p.header,
        args,
    };

    // ---- the tables live in the entry block, allocated once per call --------------------------
    reads.finish(&mut e);
    let entry = e.f.entry.0 as usize;
    let tables = [
        (
            ptrs,
            MirType::Array(Box::new(MirType::Ptr), ptr_vals.len() as u32),
        ),
        (
            scalars,
            MirType::Array(Box::new(lane_ty), scalar_vals.len().max(1) as u32),
        ),
    ];
    for (slot, ty) in tables.into_iter().rev() {
        e.f.blocks[entry].insts.insert(
            0,
            Inst {
                result: Some(slot),
                op: Op::Alloca {
                    ty,
                    space: wukong_mir::AddrSpace::Local,
                    align: 0,
                },
            },
        );
    }

    // ---- splice: the preheader now enters `kg` ------------------------------------------------
    retarget(&mut e.f.blocks[p.preheader.0 as usize].term, p.header, kg);
}

/// `v` computed in `blk` ahead of the loop: itself when defined outside it, the incoming value of
/// a header parameter (`memo`'s seed), or a copy of the scalar step that defines it with its
/// operands rebuilt the same way. [`Recipe::rebuildable`] has already proved the walk ends.
fn rebuild(
    e: &mut Emit<'_>,
    blk: wukong_mir::BlockId,
    def: &FxHashMap<u32, (u32, usize)>,
    memo: &mut FxHashMap<u32, ValueId>,
    v: ValueId,
) -> ValueId {
    if let Some(&r) = memo.get(&v.0) {
        return r;
    }
    let Some(&(block, idx)) = def.get(&v.0) else {
        return v;
    };
    let mut op = e.f.blocks[block as usize].insts[idx].op.clone();
    let mut uses = Vec::new();
    each_op_use(&op, &mut |u| uses.push(u));
    for u in uses {
        rebuild(e, blk, def, memo, u);
    }
    crate::map_op_uses(&mut op, |u| memo.get(&u.0).copied().unwrap_or(u));
    let ty = e.f.value_type(v).clone();
    let r = e.push(blk, ty, op);
    memo.insert(v.0, r);
    r
}

#[cfg(test)]
mod tests {
    use wukong_mir::{MirType, Op, Program, VecKernel, VecLane, VecOp};
    use wukong_span::{Interner, SourceId};

    use crate::OptimizeOptions;

    /// Can a kernel be built in this process at all? Off a Win64 AVX2+FMA3 host, or with the
    /// kill-switch exported, every test below checks the opposite: that no kernel appears.
    fn host_kernels() -> bool {
        wukong_mir::host_supports_vec_kernels() && std::env::var_os("WUKONG_P4_NO_256").is_none()
    }

    fn lower(src: &str) -> (Program, Interner) {
        let mut interner = Interner::new();
        let (module, pd) = wukong_parser::parse_module(src, SourceId(0), &mut interner);
        assert!(pd.iter().all(|d| !d.is_error()), "parse: {pd:?}");
        let (sema, sd) = wukong_sema::check(&module, &interner);
        assert!(sd.iter().all(|d| !d.is_error()), "sema: {sd:?}");
        let (program, _) = wukong_mir_build::lower_program(&module, &sema, &mut interner);
        (program, interner)
    }

    /// `src` through the full pipeline at `level`, with the 256-bit kernels on or off.
    fn optimized(src: &str, level: u8, kernels: bool) -> (Program, Interner) {
        let (mut program, interner) = lower(src);
        crate::optimize_preserving_with(
            &mut program,
            level,
            &[],
            OptimizeOptions::FULL.with_vec_kernels(kernels),
        );
        for f in &program.funcs {
            let errs = wukong_mir::verify::verify_function(f);
            assert!(errs.is_empty(), "MIR invalid after -O{level}: {errs:?}");
        }
        (program, interner)
    }

    /// The recipes `func` calls, in call order.
    fn kernels_of(program: &Program, interner: &mut Interner, func: &str) -> Vec<VecKernel> {
        let f = program
            .function(interner.intern(func))
            .unwrap_or_else(|| panic!("no fn {func}"));
        f.blocks
            .iter()
            .flat_map(|b| &b.insts)
            .filter_map(|i| match i.op {
                Op::VecKernelCall { kernel, .. } => Some(f.vec_kernels[kernel as usize].clone()),
                _ => None,
            })
            .collect()
    }

    /// Does `func` hold a 128-bit vector value?
    fn has_vectors(program: &Program, interner: &mut Interner, func: &str) -> bool {
        let f = program
            .function(interner.intern(func))
            .unwrap_or_else(|| panic!("no fn {func}"));
        f.value_types.iter().any(|t| t.is_vector())
    }

    fn run(program: &Program, interner: &mut Interner) -> (i64, String) {
        let main = interner.intern("main");
        let (code, bytes) = wukong_interp::run_with_output(program, main, interner)
            .unwrap_or_else(|e| panic!("{e}"));
        (code, String::from_utf8_lossy(&bytes).into_owned())
    }

    /// A body the front end's own vectorizer leaves scalar — it calls a user function — over
    /// runtime-length slices. At `-O2` the helper is inlined, the loop widens, and the body is a
    /// recipe: one fused multiply-add between a load and a store.
    const HELPER: &str = "fn affine(v: f32, s: f32) -> f32 { return v * s + 1.0; } \
        fn k(x: []f32, mut o: []f32, s: f32, n: i64) { let mut i: i64 = 0; \
          while i < n { o[i] = affine(x[i], s); i = i + 1; } } \
        fn main() -> i32 { return 0; }";

    #[test]
    fn a_loop_the_front_end_leaves_scalar_takes_a_256_bit_kernel() {
        let (front, mut fi) = lower(HELPER);
        assert!(
            kernels_of(&front, &mut fi, "k").is_empty(),
            "the front end built this kernel itself, so this case no longer tests the optimizer"
        );
        let (program, mut interner) = optimized(HELPER, 2, true);
        let ks = kernels_of(&program, &mut interner, "k");
        if !host_kernels() {
            assert!(ks.is_empty(), "no host support, yet a kernel: {ks:?}");
            return;
        }
        assert_eq!(ks.len(), 1, "{ks:?}");
        let k = &ks[0];
        assert_eq!(k.lane, VecLane::F32);
        assert_eq!(k.streams, 2, "x and o");
        assert!(k.reduce.is_none());
        assert!(
            k.ops.iter().any(|o| matches!(o, VecOp::Fma { .. })),
            "the contracted multiply-add must stay one fused op: {:?}",
            k.ops
        );
        // The remainder — every trip below the crossover — still widens at 128 bits.
        assert!(has_vectors(&program, &mut interner, "k"));
    }

    /// The pipeline's switch (`OptimizeOptions::vec_kernels`, cleared for gpu-native) keeps every
    /// loop on the 128-bit path.
    #[test]
    fn the_switch_off_keeps_the_128_bit_loop() {
        let (program, mut interner) = optimized(HELPER, 2, false);
        assert!(kernels_of(&program, &mut interner, "k").is_empty());
        assert!(has_vectors(&program, &mut interner, "k"));
    }

    /// `-O1` runs no vectorizer at all.
    #[test]
    fn o1_builds_no_kernel() {
        let (program, mut interner) = optimized(HELPER, 1, true);
        assert!(kernels_of(&program, &mut interner, "k").is_empty());
    }

    /// The kernel's own remainder starts at `start + n` and is the 128-bit loop's, never a second
    /// kernel's — even though it is the same loop, with the same runtime trip, that took one.
    #[test]
    fn the_remainder_is_widened_not_given_a_second_kernel() {
        let (program, mut interner) = optimized(HELPER, 2, true);
        let n = kernels_of(&program, &mut interner, "k").len();
        assert_eq!(n, usize::from(host_kernels()));
    }

    /// Shapes the recipe has no form for keep the 128-bit loop they widened to before.
    #[test]
    fn shapes_outside_the_recipe_keep_the_128_bit_loop() {
        let cases = [
            // A compile-time trip below the crossover.
            (
                "short",
                "fn k(x: []f32, mut o: []f32) { let mut i: i64 = 0; \
                 while i < 37 { o[i] = x[i] * 2.0; i = i + 1; } } \
                 fn main() -> i32 { return 0; }",
            ),
            // A float sum: the kernel's tree would reassociate it.
            (
                "float sum",
                "fn k(x: []f32, n: i64) -> f32 { let mut s: f32 = 0.0; let mut i: i64 = 0; \
                 while i < n { s = s + x[i]; i = i + 1; } return s; } \
                 fn main() -> i32 { return 0; }",
            ),
            // An integer product: the recipe folds sums only.
            (
                "integer product",
                "fn k(x: []i32, n: i64) -> i32 { let mut s: i32 = 1; let mut i: i64 = 0; \
                 while i < n { s = s * (x[i] + 2); i = i + 1; } return s; } \
                 fn main() -> i32 { return 0; }",
            ),
            // An integer sum beside a store: a reduction kernel writes no memory.
            (
                "sum beside a store",
                "fn k(x: []i32, mut o: []i32, n: i64) -> i32 { let mut s: i32 = 0; \
                 let mut i: i64 = 0; \
                 while i < n { o[i] = x[i] * 2; s = s + x[i]; i = i + 1; } return s; } \
                 fn main() -> i32 { return 0; }",
            ),
            // The counter read as a value.
            (
                "counter",
                "fn k(x: []f32, mut o: []f32, n: i64) { let mut i: i64 = 0; \
                 while i < n { o[i] = x[i] + (i as f32); i = i + 1; } } \
                 fn main() -> i32 { return 0; }",
            ),
            // A countdown.
            (
                "countdown",
                "@unchecked fn k(x: []f32, mut o: []f32, n: i64) { let mut i: i64 = n; \
                 while i > 0 { i = i - 1; o[i] = x[i] * 2.0; } } \
                 fn main() -> i32 { return 0; }",
            ),
            // A cast between lane types.
            (
                "cast",
                "fn k(x: []i32, mut o: []f32, n: i64) { let mut i: i64 = 0; \
                 while i < n { o[i] = (x[i] as f32) * 0.5; i = i + 1; } } \
                 fn main() -> i32 { return 0; }",
            ),
            // An unsigned compare: AVX2's integer compares are the signed ones.
            (
                "unsigned",
                "@unchecked fn k(x: []u32, y: []u32, mut o: []u32, n: i64) { let mut i: i64 = 0; \
                 while i < n { o[i] = if x[i] < y[i] { x[i] } else { y[i] }; i = i + 1; } } \
                 fn main() -> i32 { return 0; }",
            ),
            // A 64-bit integer multiply: AVX2 has no `vpmullq`.
            (
                "i64 mul",
                "fn k(x: []i64, mut o: []i64, n: i64) { let mut i: i64 = 0; \
                 while i < n { o[i] = x[i] * 7; i = i + 1; } } \
                 fn main() -> i32 { return 0; }",
            ),
        ];
        for (name, src) in cases {
            let (program, mut interner) = optimized(src, 2, true);
            assert!(
                has_vectors(&program, &mut interner, "k"),
                "{name}: not widened at all, so this case proves nothing about the kernel path"
            );
            let ks = kernels_of(&program, &mut interner, "k");
            assert!(ks.is_empty(), "{name}: took a kernel: {ks:?}");
        }
    }

    /// Every lane of three kernels — f32 with a blend, a negation, a fused multiply-add, a square
    /// root and a division over NaN, both zeros, both infinities and a subnormal; i32 with a
    /// signed compare, a wrapping multiply and a negation; f64 — printed as raw bits, the
    /// interpreter at `-O0` (the scalar loops) against `-O2` (kernel, 128-bit remainder, scalar
    /// tail), at a runtime trip of 2053: 256 kernel groups, then the remainder. Each kernel calls
    /// itself when `d > 0` (never, here), so the inliner — leaf functions only — keeps it out of
    /// `main` and its trip a runtime value.
    const LANES: &str = "\
        fn pick(a: f32, b: f32) -> f32 { if a > b { return -a; } return a * b + 0.5; } \
        fn kf(x: []f32, y: []f32, mut o: []f32, mut p: []f32, n: i64, d: i64) { \
          if d > 0 { kf(x, y, o, p, n, d - 1); return; } let mut i: i64 = 0; \
          while i < n { o[i] = pick(x[i], y[i]); p[i] = sqrt(x[i]) / y[i] - x[i]; i = i + 1; } } \
        fn isel(a: i32, b: i32) -> i32 { if a < b { return a * 3 - b; } return -a; } \
        @unchecked fn ki(a: []i32, b: []i32, mut c: []i32, n: i64, d: i64) { \
          if d > 0 { ki(a, b, c, n, d - 1); return; } let mut i: i64 = 0; \
          while i < n { c[i] = isel(a[i], b[i]); i = i + 1; } } \
        fn kd(a: []f64, mut c: []f64, s: f64, n: i64, d: i64) { \
          if d > 0 { kd(a, c, s, n, d - 1); return; } let mut i: i64 = 0; \
          while i < n { c[i] = a[i] * s - 1.0; i = i + 1; } } \
        fn special(i: i64) -> f32 { let r: i64 = i % 8; \
          if r == 0 { return 0.0 / 0.0; } if r == 1 { return -0.0; } if r == 2 { return 0.0; } \
          if r == 3 { return 1.0 / 0.0; } if r == 4 { return -1.0 / 0.0; } \
          if r == 5 { return (i as f32) * 0.375; } if r == 6 { return -(i as f32) * 1.5; } \
          return f32_from_bits(1 as u32); } \
        fn main() -> i32 { let n: i64 = 2053; \
          let mut x: []f32 = alloc_f32(n); let mut y: []f32 = alloc_f32(n); \
          let mut o: []f32 = alloc_f32(n); let mut p: []f32 = alloc_f32(n); \
          let mut a: []i32 = alloc_i32(n); let mut b: []i32 = alloc_i32(n); \
          let mut c: []i32 = alloc_i32(n); \
          let mut d: []f64 = alloc_f64(n); let mut e: []f64 = alloc_f64(n); \
          let mut i: i64 = 0; \
          while i < n { x[i] = special(i); y[i] = special(i * 7 + 3); \
            a[i] = (i * 2654435761 % 1000 - 500) as i32; b[i] = (i * 40503 % 1000 - 500) as i32; \
            d[i] = (i as f64) * 0.125 - 100.0; i = i + 1; } \
          kf(x, y, o, p, n, 0); ki(a, b, c, n, 0); kd(d, e, 3.0, n, 0); \
          i = 0; \
          while i < n { print(bits(o[i])); print(bits(p[i])); print(c[i]); print(bits(e[i])); \
            i = i + 1; } \
          return 0; }";

    #[test]
    fn every_lane_of_a_kernel_matches_the_scalar_loop() {
        let (p0, mut i0) = optimized(LANES, 0, true);
        let (p2, mut i2) = optimized(LANES, 2, true);
        let lanes: Vec<VecLane> = ["kf", "ki", "kd"]
            .iter()
            .flat_map(|f| kernels_of(&p2, &mut i2, f))
            .map(|k| k.lane)
            .collect();
        if host_kernels() {
            assert_eq!(
                lanes,
                [VecLane::F32, VecLane::I32, VecLane::F64],
                "each kernel function must take exactly one kernel, or this compares less than \
                 it claims"
            );
        } else {
            assert!(lanes.is_empty());
        }
        let (want, got) = (run(&p0, &mut i0), run(&p2, &mut i2));
        assert_eq!(want.0, 0);
        assert_eq!(want.1.lines().count(), 4 * 2053);
        assert_eq!(want, got, "-O0 and -O2 disagree");
    }

    /// The comparison above is only evidence if the kernels really run on those lanes: a trip
    /// below the crossover, a failed alias check or a recipe whose stores reached nothing would
    /// all leave `-O2` equal to `-O0` while proving nothing. So corrupt every recipe — each store
    /// writes the value of its kernel's first load instead — and the output must change.
    #[test]
    fn the_kernels_really_run_on_those_lanes() {
        if !host_kernels() {
            return;
        }
        let (p0, mut i0) = optimized(LANES, 0, true);
        let (mut p2, mut i2) = optimized(LANES, 2, true);
        let mut corrupted = 0;
        for f in &mut p2.funcs {
            for k in &mut f.vec_kernels {
                let Some(first_load) = k.ops.iter().position(|o| matches!(o, VecOp::Load { .. }))
                else {
                    continue;
                };
                for op in &mut k.ops {
                    if let VecOp::Store { val, .. } = op {
                        if *val as usize > first_load {
                            *val = first_load as u32;
                            corrupted += 1;
                        }
                    }
                }
            }
        }
        assert!(corrupted >= 3, "{corrupted} stores corrupted");
        let (want, got) = (run(&p0, &mut i0), run(&p2, &mut i2));
        assert_ne!(
            want, got,
            "corrupting every kernel left the output unchanged: the kernels never ran"
        );
    }

    /// A kernel's operands are rebuilt ahead of the loop with the counter at its first value, so
    /// a stream that starts at a symbolic offset (a row of a matrix) starts at that row.
    #[test]
    fn a_row_offset_stream_starts_at_its_row() {
        let src = "fn k(x: []f32, mut o: []f32, r: i64, c: i64, d: i64) { \
                   if d > 0 { k(x, o, r, c, d - 1); return; } \
                   let mut q: i64 = 0; \
                   while q < r { let b: i64 = q * c; let mut j: i64 = 0; \
                     while j < c { o[b + j] = x[b + j] * 2.0 - 3.0; j = j + 1; } \
                     q = q + 1; } } \
                   fn main() -> i32 { let r: i64 = 3; let c: i64 = 2051; \
                     let mut x: []f32 = alloc_f32(r * c); let mut o: []f32 = alloc_f32(r * c); \
                     let mut i: i64 = 0; \
                     while i < r * c { x[i] = (i % 97) as f32 * 0.25; i = i + 1; } \
                     k(x, o, r, c, 0); i = 0; \
                     while i < r * c { print(bits(o[i])); i = i + 1; } return 0; }";
        let (p0, mut i0) = optimized(src, 0, true);
        let (p2, mut i2) = optimized(src, 2, true);
        assert_eq!(
            kernels_of(&p2, &mut i2, "k").len(),
            usize::from(host_kernels())
        );
        assert_eq!(run(&p0, &mut i0), run(&p2, &mut i2), "-O0 and -O2 disagree");
    }

    /// `o[i] = …; q[i] = o[i] + 1` over slices: the reload of `o[i]` goes through a second copy of
    /// `o`'s data pointer (re-read from the fat pointer after the store), so the two accesses were
    /// two streams, and within a lane the interpreter — which reads every stream before running the
    /// recipe and forwards a store only within its own stream — returned the element's OLD value
    /// where the machine code (which re-reads memory) and the scalar loop return the new one: `1`
    /// for lane 5 at `-O2` against `11`. One element is one stream.
    #[test]
    fn a_store_read_back_through_another_pointer_is_one_stream() {
        let src = "@unchecked fn k(x: []f32, mut o: []f32, mut q: []f32, n: i64, d: i64) { \
                   if d > 0 { k(x, o, q, n, d - 1); return; } let mut i: i64 = 0; \
                   while i < n { o[i] = x[i] * 2.0; q[i] = o[i] + 1.0; i = i + 1; } } \
                   fn main() -> i32 { let n: i64 = 2053; \
                     let mut x: []f32 = alloc_f32(n); let mut o: []f32 = alloc_f32(n); \
                     let mut q: []f32 = alloc_f32(n); let mut i: i64 = 0; \
                     while i < n { x[i] = i as f32; i = i + 1; } \
                     k(x, o, q, n, 0); i = 0; \
                     while i < n { print(bits(o[i])); print(bits(q[i])); i = i + 1; } \
                     return 0; }";
        let (p0, mut i0) = optimized(src, 0, true);
        let (p2, mut i2) = optimized(src, 2, true);
        let ks = kernels_of(&p2, &mut i2, "k");
        assert_eq!(ks.len(), usize::from(host_kernels()));
        if let Some(k) = ks.first() {
            assert_eq!(k.streams, 3, "x, o and q: {:?}", k.ops);
        }
        assert_eq!(run(&p0, &mut i0), run(&p2, &mut i2), "-O0 and -O2 disagree");
    }

    /// The same read-back through two slices the caller passes the SAME buffer as: the runtime
    /// alias guard admits equal bases when every access sits at one offset, and then a load of `x`
    /// after a store to `o` names the element just stored — two streams, one base. The machine code
    /// re-reads memory; the interpreter forwards this lane's store to a load of any stream with the
    /// same base, so it agrees (before it did, it read the element stale).
    #[test]
    fn a_load_after_a_store_to_an_equal_base_reads_the_store() {
        let src = "@unchecked fn k(x: []f32, mut o: []f32, mut q: []f32, n: i64, d: i64) { \
                   if d > 0 { k(x, o, q, n, d - 1); return; } let mut i: i64 = 0; \
                   while i < n { o[i] = x[i] * 2.0; q[i] = x[i] + 1.0; i = i + 1; } } \
                   fn main() -> i32 { let n: i64 = 2053; \
                     let mut a: []f32 = alloc_f32(n); let mut q: []f32 = alloc_f32(n); \
                     let mut i: i64 = 0; \
                     while i < n { a[i] = i as f32; i = i + 1; } \
                     k(a, a, q, n, 0); i = 0; \
                     while i < n { print(bits(a[i])); print(bits(q[i])); i = i + 1; } \
                     return 0; }";
        let (p0, mut i0) = optimized(src, 0, true);
        let (p2, mut i2) = optimized(src, 2, true);
        assert_eq!(
            kernels_of(&p2, &mut i2, "k").len(),
            usize::from(host_kernels())
        );
        assert_eq!(run(&p0, &mut i0), run(&p2, &mut i2), "-O0 and -O2 disagree");
    }

    /// Two integer sums, `i32` and `i64`, whose addends overflow, at every trip from 2040 to 2064:
    /// below the gate the 128-bit loop does everything; from 2048 on the reduction kernel folds the
    /// whole groups (its own grouping — four accumulators of eight or four lanes) and the 128-bit
    /// loop and the scalar tail the remainder, for every remainder length. Integer addition wraps
    /// and is associative, so `-O2` must print the scalar loop's sums exactly — and it must be the
    /// kernel doing it: zeroing its addend changes the answer.
    #[test]
    fn an_integer_sum_takes_a_reduction_kernel_and_wraps_like_the_loop() {
        // `s32`/`s64` and `again` call each other (never, here), so neither is a leaf the inliner
        // would splice into `main`.
        let src = "fn s32(x: []i32, y: []i32, n: i64, d: i64) -> i32 { \
                   if d > 0 { return again(d - 1) as i32; } \
                   let mut s: i32 = 7; let mut i: i64 = 0; \
                   while i < n { s = s + (x[i] * 3 - y[i]); i = i + 1; } return s; } \
                   fn s64(z: []i64, n: i64, d: i64) -> i64 { \
                   if d > 0 { return again(d - 1); } \
                   let mut s: i64 = -5; let mut i: i64 = 0; \
                   while i < n { s = s + (z[i] - 9); i = i + 1; } return s; } \
                   fn again(d: i64) -> i64 { let mut a: []i32 = alloc_i32(1); \
                     let mut b: []i64 = alloc_i64(1); \
                     return (s32(a, a, 1, d) as i64) + s64(b, 1, d); } \
                   fn main() -> i32 { let m: i64 = 2064; \
                     let mut x: []i32 = alloc_i32(m); let mut y: []i32 = alloc_i32(m); \
                     let mut z: []i64 = alloc_i64(m); let mut i: i64 = 0; \
                     while i < m { x[i] = (i * 2654435761 % 2000000000) as i32; \
                       y[i] = (i * 40503 % 1000 - 500) as i32; \
                       z[i] = i * 4611686018427387 + 17; i = i + 1; } \
                     let mut n: i64 = 2040; \
                     while n <= m { print(s32(x, y, n, 0)); print(s64(z, n, 0)); n = n + 1; } \
                     return 0; }";
        let (p0, mut i0) = optimized(src, 0, true);
        let (mut p2, mut i2) = optimized(src, 2, true);
        for (f, lane) in [("s32", VecLane::I32), ("s64", VecLane::I64)] {
            let ks = kernels_of(&p2, &mut i2, f);
            if !host_kernels() {
                assert!(ks.is_empty(), "{f}: {ks:?}");
                continue;
            }
            assert_eq!(ks.len(), 1, "{f}: {ks:?}");
            assert_eq!(ks[0].lane, lane, "{f}");
            assert!(ks[0].reduce.is_some(), "{f}: not a reduction kernel");
        }
        let want = run(&p0, &mut i0);
        assert_eq!(want.1.lines().count(), 50);
        assert_eq!(want, run(&p2, &mut i2), "-O0 and -O2 disagree");
        if !host_kernels() {
            return;
        }
        // Fold a load instead of the addend: the kernel's sum changes, so the output must.
        for f in &mut p2.funcs {
            for k in &mut f.vec_kernels {
                if let (Some(red), Some(load)) = (
                    k.reduce.as_mut(),
                    k.ops.iter().position(|o| matches!(o, VecOp::Load { .. })),
                ) {
                    red.value = load as u32;
                }
            }
        }
        assert_ne!(
            want,
            run(&p2, &mut i2),
            "corrupting the sum kernels changed nothing: they never ran"
        );
    }

    /// The tables are typed: a stream table of pointers, a scalar table of the lane type.
    #[test]
    fn the_tables_are_typed_and_live_in_the_entry_block() {
        if !host_kernels() {
            return;
        }
        let (program, mut interner) = optimized(HELPER, 2, true);
        let f = program.function(interner.intern("k")).expect("fn k");
        let entry = &f.blocks[f.entry.0 as usize];
        let tables: Vec<&MirType> = entry
            .insts
            .iter()
            .filter_map(|i| match &i.op {
                Op::Alloca {
                    ty: MirType::Array(e, _),
                    ..
                } => Some(&**e),
                _ => None,
            })
            .collect();
        assert!(tables.contains(&&MirType::Ptr), "{tables:?}");
        assert!(tables.contains(&&MirType::F32), "{tables:?}");
    }
}
