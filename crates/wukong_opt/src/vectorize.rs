//! Loop vectorization: widen a scalar loop body to SIMD vectors over canonical MIR.
//!
//! This is the general path. It does not pattern-match a source shape the way the `mir_build`
//! kernel recognizers do — it reads [`crate::loop_info`]'s structural answers (induction variable,
//! trip count, affine accesses, loop-carried classification, dependence verdict) and widens
//! whatever passes. A `while`, a `for`, a user temporary, an `i32` counter and an `f64` element
//! type all reach it in the same shape, which is the entire point.
//!
//! # The rewrite
//!
//! The original loop is **not modified**. A vector loop is spliced in front of it and the original
//! becomes its scalar epilogue:
//!
//! ```text
//!   preheader ──▶ guard ──▶ vec.header ──▶ vec.body ─┐   (widened, W lanes per iteration)
//!                             │  ▲                   │
//!                             │  └───────────────────┘
//!                             ▼
//!                          vec.exit ──▶ header ──▶ body ──▶ …   (the ORIGINAL loop, untouched)
//! ```
//!
//! `guard` computes `n_vec`, the largest multiple of `W` that fits in the trip count, and the
//! vector loop runs exactly `n_vec / W` iterations before handing the original loop an induction
//! variable already advanced to `start + n_vec`. The original loop then runs the `0..W-1` remaining
//! elements. Three properties fall out of that shape and are worth stating, because they are what
//! makes the transform auditable:
//!
//! * **No duplicated body.** The scalar epilogue *is* the original code, so there is exactly one
//!   scalar spelling of the loop's semantics in the function, and it is the one the front end
//!   emitted.
//! * **Every runtime bail-out is `n_vec = 0`.** A failed alias check, a trip count below one full
//!   vector, a negative count — all of them set `n_vec` to zero, the vector loop runs no
//!   iterations, and the original loop does all the work. There is no second code path to get
//!   wrong.
//! * **Live-out values need no fix-up.** The loop's only exit is tested in the header, so every
//!   value that escapes is a header block parameter; `vec.exit` passes the vector loop's final
//!   values into exactly those parameters.
//!
//! # Float semantics
//!
//! IEEE addition is not associative, and Wukong's interpreter is the language specification, so a
//! float reduction may **not** be reassociated: `(a+b)+c` and `a+(b+c)` are different programs.
//! What this pass does — what `gcc -O3` does without `-ffast-math` — is widen the *body* that feeds
//! the accumulator and keep the accumulate itself serial: `W` lane values are extracted in index
//! order and folded one at a time, in exactly the order the scalar loop would have. The loads, the
//! subtracts and the multiplies go 4-wide; the `vaddss` chain does not move. See
//! [`emit_reduction_fold`].
//!
//! Everything else is lane-wise and therefore trivially order-preserving: lane `k` of a widened
//! group executes the identical operation sequence on the identical operands that scalar iteration
//! `i + k` would have.
//!
//! # Width
//!
//! `W` is `128 / lane_bits`, i.e. `<4 x f32>`, `<2 x f64>`, `<8 x i16>`, `<16 x i8>`. That is not a
//! guess about what the machine wants — it is the widest vector Cranelift will legalize. A
//! `<8 x f32>` reaches the x64 backend as `Unsupported("Unexpected SSA-value type: f32x8")` and
//! fails the compile; the repo's 256-bit path is a separate mechanism (`VecKernel` +
//! `codegen_cranelift::avx2`, raw assembled AVX2) restricted to f32 elementwise bodies. With AVX
//! enabled Cranelift VEX-encodes the 128-bit ops, and two of them retire per cycle, so a
//! memory-bound loop saturates the same bandwidth a 256-bit one would.
//!
//! # `bf16` / `f16` are deliberately excluded
//!
//! The interpreter rounds a **scalar** `bf16`/`f16` load to the storage grid on read
//! (`wukong_interp`'s `Op::Load` arm) but its **vector** load gathers slots verbatim. Widening a
//! half-precision load would therefore drop a rounding the scalar path performs — a silent
//! interp-vs-native divergence, the worst possible outcome. [`lane_bytes`] refuses them.

use wukong_mir::{
    BasicBlock, BinOp, BlockId, CastKind, CmpOp, Function, Inst, MirType, Op, Terminator, ValueId,
};

use crate::fxhash::{FxHashMap, FxHashSet};
use crate::loop_info::{
    self, AccessKind, AddrForm, Carried, IvStep, LoopForest, MemAccess, MemBase, NaturalLoop,
    RedKind, TripCount,
};
use crate::{each_op_use, CfgAnalyses, Pass};

/// # Termination
///
/// Widening a loop leaves a *second* loop behind — the scalar epilogue — with the same shape the
/// original had, so a pass that only asked "is this vectorizable?" would widen the epilogue, then
/// the epilogue's epilogue, forever. Nothing about the epilogue's own MIR says it has been dealt
/// with; the pass manager re-runs every pass after any mutation, so there is no "one sweep and
/// done" to hide behind either.
///
/// Two independent stops are used, deliberately, because getting this wrong does not produce a
/// wrong answer — it produces a compiler that never finishes:
///
/// * [`is_someones_epilogue`] — structural. The epilogue's only preheader is the exit block of a
///   loop that holds vector values, an arrangement nothing else in the pipeline builds. This
///   survives block renumbering and needs no state.
/// * [`Vectorize::done`] — a record of the header blocks already widened in the function being
///   optimized, reset whenever the pass is handed a different function. A stale entry after
///   `cfg::prune_unreachable` renumbers blocks can only *skip* a loop, never widen one twice.
#[derive(Default)]
pub struct Vectorize {
    /// `(function, headers already widened)` for the function currently being optimized.
    done: std::cell::RefCell<Option<(wukong_span::Symbol, FxHashSet<u32>)>>,
}

/// The SIMD register width the backend can legalize, in bytes. Cranelift's x64 vector ISA tops out
/// at 128 bits — see the module docs for the measurement that established it.
const VEC_BYTES: u32 = 16;

/// Fewest iterations worth widening: one full vector group. Below this the vector loop runs zero
/// times and the guard's arithmetic is pure overhead, so a loop with a *known* smaller trip count
/// is left alone entirely rather than given a guard it can never pass.
const MIN_TRIP_GROUPS: u64 = 1;

/// A cap on how many loops one call will widen, so a bug that fails to mark a loop as already
/// processed degrades into "stopped early" rather than a compile that never terminates.
const MAX_LOOPS_PER_CALL: usize = 64;

impl Pass for Vectorize {
    fn name(&self) -> &'static str {
        "vectorize"
    }

    fn run_function(&self, f: &mut Function, cache: &mut CfgAnalyses) -> bool {
        if f.blocks.len() < 3 {
            return false; // no loop can fit
        }
        // A kill-switch, in the spirit of `WUKONG_P4_NO_256`. Its first job is measurement: the only
        // honest A/B on this machine is same-run and adjacent, and the two arms have to be one
        // binary. Its second is triage — if a body is ever found miscompiled, this turns the whole
        // pass off without a rebuild.
        if std::env::var_os("WUKONG_NO_VECTORIZE").is_some() {
            return false;
        }
        let mut slot = self.done.borrow_mut();
        let done = match slot.as_mut() {
            Some((name, set)) if *name == f.name => set,
            _ => {
                *slot = Some((f.name, FxHashSet::default()));
                &mut slot.as_mut().expect("just set").1
            }
        };
        let mut changed = false;
        for _ in 0..MAX_LOOPS_PER_CALL {
            let (idom, preds) = cache.idoms_and_preds(f);
            let forest = loop_info::analyze_with(f, idom, preds);
            let Some(plan) = pick(f, &forest, done) else {
                break;
            };
            done.insert(plan.header.0);
            apply(f, &plan);
            cache.invalidate();
            changed = true;
        }
        changed
    }
}

// ---------------------------------------------------------------------------------------------
// Widths and lane types
// ---------------------------------------------------------------------------------------------

/// Byte width of a lane type this pass will widen, or `None` for one it refuses.
///
/// `I1` is out because a boolean vector is a *mask*, whose machine representation is one all-ones
/// lane per element of the value it selects, not a one-bit lane. `bf16`/`f16` are out because the
/// interpreter's vector load does not apply the storage-grid rounding its scalar load does (module
/// docs). Pointers and aggregates have no lane-wise arithmetic at all.
fn lane_bytes(t: &MirType) -> Option<u32> {
    Some(match t {
        MirType::I8 => 1,
        MirType::I16 => 2,
        MirType::I32 | MirType::F32 => 4,
        MirType::I64 | MirType::F64 => 8,
        _ => return None,
    })
}

/// Lanes per vector for a lane type.
fn lanes_of(t: &MirType) -> Option<u32> {
    lane_bytes(t).map(|b| VEC_BYTES / b)
}

/// The integer lane type a vector compare over `lane` produces. A lane mask is all-ones or
/// all-zeros *at the lane's own width* — Cranelift's `icmp`/`fcmp` on `f32x4` yields `i32x4` — and
/// `Op::Select` bitcasts it to the value type before blending.
fn mask_lane(lane: &MirType) -> MirType {
    match lane_bytes(lane) {
        Some(1) => MirType::I8,
        Some(2) => MirType::I16,
        Some(8) => MirType::I64,
        _ => MirType::I32,
    }
}

fn vec_of(lane: &MirType, w: u32) -> MirType {
    MirType::Vec(Box::new(lane.clone()), w)
}

// ---------------------------------------------------------------------------------------------
// Planning
// ---------------------------------------------------------------------------------------------

/// One widened value: either kept scalar (the same value in every lane, or an address) or widened
/// to a `W`-lane vector.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Wide {
    /// Loop-invariant, or an address/index that is uniform across the group.
    Uniform(ValueId),
    /// A `Vec(lane, W)` value whose lane `k` holds what scalar iteration `i + k` computed.
    Vector(ValueId),
}

/// How one instruction of the scalar body is reproduced in the vector body.
#[derive(Clone, Debug)]
enum Plan {
    /// Copy the instruction unchanged (address arithmetic, invariant loads, constants).
    Scalar,
    /// Widen it: same opcode, vector-typed result, operands splatted as needed.
    Widen,
    /// A `Load` whose address is affine unit-stride: becomes a `Vec` load.
    VecLoad,
    /// A `Store` whose address is affine unit-stride: becomes a `Vec` store.
    VecStore,
    /// A reduction's combining instruction. Not emitted inline — every accumulate is folded
    /// serially at the end of the vector body by [`emit_reduction_fold`].
    Reduce,
}

/// Everything [`apply`] needs, decided before any mutation.
struct VecPlan {
    header: BlockId,
    body: BlockId,
    preheader: BlockId,
    /// Header parameter index of the primary induction variable.
    iv_index: usize,
    iv_ty: MirType,
    /// How to advance the trip: either a compile-time vector count, or `end` for `end - start`.
    trip: TripPlan,
    /// Lane count.
    w: u32,
    /// One entry per instruction of `body`, in order.
    plans: Vec<Plan>,
    /// Values in `body` that become vectors, and their lane type.
    vector_ty: FxHashMap<u32, MirType>,
    /// Reductions to fold, in `NaturalLoop::reductions` order.
    reductions: Vec<RedPlan>,
    /// Runtime pointer-range checks the vector loop is only legal under.
    guards: Vec<GuardPair>,
    /// Element type the guard ranges are measured in (all guarded accesses share it).
    guard_elem: MirType,
    /// May two guarded bases be *equal* (rather than disjoint)? Only when every guarded access
    /// sits at the same constant offset from its base, so an equal pair is at distance 0.
    guard_allows_equal: bool,
}

#[derive(Clone, Debug)]
enum TripPlan {
    /// The loop runs a known number of times; `vec` is that count rounded down to a multiple of W.
    Const(u64),
    /// `end - start`, rounded down to a multiple of W and clamped at 0.
    Runtime { end: ValueId },
}

#[derive(Clone, Debug)]
struct RedPlan {
    /// Header parameter index of the accumulator.
    param_index: usize,
    kind: RedKind,
    /// `Bin`: the non-accumulator operand, and whether the accumulator was the *left* operand.
    addend: Option<(ValueId, bool)>,
    /// `Fma`: the two multiplicands, re-fused per lane so the rounding is unchanged.
    fma_factors: Option<(ValueId, ValueId)>,
    ty: MirType,
}

/// One pair of bases the vector loop is only legal over if they are equal or disjoint.
#[derive(Clone, Copy, Debug)]
struct GuardPair {
    a: GuardBase,
    b: GuardBase,
}

/// A base pointer, plus the smallest and largest constant element offset the loop applies to it.
#[derive(Clone, Copy, Debug)]
struct GuardBase {
    base: MemBase,
    lo: i64,
    hi: i64,
}

// ---------------------------------------------------------------------------------------------
// Legality
// ---------------------------------------------------------------------------------------------

/// The first loop of `f` this pass can widen, innermost first.
fn pick(f: &Function, forest: &LoopForest, done: &FxHashSet<u32>) -> Option<VecPlan> {
    let trace = std::env::var_os("WUKONG_VEC_TRACE").is_some();
    for id in forest.innermost_first() {
        let l = forest.get(id);
        if done.contains(&l.header.0) || already_vector(f, l) {
            continue;
        }
        if is_someones_epilogue(f, forest, l) {
            continue;
        }
        match plan_loop(f, l) {
            Ok(p) => return Some(p),
            Err(why) if trace => {
                eprintln!("vectorize: declined loop at bb{}: {why}", l.header.0)
            }
            Err(_) => {}
        }
    }
    None
}

/// Does this loop already hold vector values? Then it is either the vector loop this pass just
/// built, or a body the `mir_build` autovectorizer already widened, and re-widening it would be
/// both wrong and pointless.
fn already_vector(f: &Function, l: &NaturalLoop) -> bool {
    l.blocks.iter().any(|b| {
        let blk = &f.blocks[b.0 as usize];
        blk.params.iter().any(|p| f.value_type(*p).is_vector())
            || blk
                .insts
                .iter()
                .any(|i| i.result.is_some_and(|r| f.value_type(r).is_vector()))
    })
}

/// Is this loop the scalar epilogue of a vector loop this pass already built?
///
/// The marker is structural rather than a side table: after [`apply`], the original loop's only
/// preheader is the *exit block of a loop that holds vector values*. Nothing else in the pipeline
/// produces that arrangement, it survives block renumbering (`cfg::prune_unreachable` compacts
/// ids), and it needs no state carried between pass invocations — which matters, because the pass
/// manager re-runs every pass after any pass mutates the function.
fn is_someones_epilogue(f: &Function, forest: &LoopForest, l: &NaturalLoop) -> bool {
    let Some(ph) = l.preheader else { return false };
    forest
        .loops
        .iter()
        .any(|v| v.id != l.id && v.exits.iter().any(|&(_, to)| to == ph) && already_vector(f, v))
}

fn plan_loop(f: &Function, l: &NaturalLoop) -> Result<VecPlan, &'static str> {
    // ---- structure -------------------------------------------------------------------------
    let preheader = l.preheader.ok_or("no preheader")?;
    if l.latches.len() != 1 || !l.abnormal_exits.is_empty() || l.exits.len() != 1 {
        return Err("not one latch / one exit / no abnormal exit");
    }
    let (exit_from, _) = l.exits[0];
    if exit_from != l.header {
        return Err("exit is not tested in the header");
    }
    // Exactly two blocks: the header (which must hold nothing but the exit test) and one body
    // block, which is therefore also the latch. If-conversion, which lifts this, is a later step.
    if l.blocks.len() != 2 {
        return Err("body is not a single block (needs if-conversion)");
    }
    let body = *l
        .blocks
        .iter()
        .find(|&&b| b != l.header)
        .ok_or("no body block")?;
    if l.latches[0] != body {
        return Err("the body block is not the latch");
    }
    if !header_is_test_only(f, l) {
        return Err("the header computes more than the exit test");
    }

    // ---- induction variable and trip count ---------------------------------------------------
    let iv_idx_in_ivs = l.primary_iv.ok_or("no primary induction variable")?;
    let iv = &l.ivs[iv_idx_in_ivs];
    if iv.step != IvStep::Const(1) {
        return Err("induction step is not 1");
    }
    let iv_index = iv.param_index;
    let iv_ty = iv.ty.clone();
    if !iv_ty.is_int() || iv_ty == MirType::I1 {
        return Err("induction variable is not an integer");
    }

    // ---- dependence ---------------------------------------------------------------------------
    if !l.is_vectorizable_shape() {
        return Err("loop-carried dependence");
    }
    if !carried_values_are_handled(l, iv.param_index) {
        return Err("a carried value is neither the primary IV, a reduction nor invariant");
    }
    if !index_values_only_address(f, l, body, iv.value) {
        return Err("the induction variable is used for something other than addressing");
    }
    if !invariant_loads_are_safe(l) {
        return Err("an invariant data load in a loop that also writes memory");
    }

    // ---- width, from the memory the loop touches ----------------------------------------------
    let w = pick_width(l).ok_or("no single lane width for the loop's accesses")?;

    // ---- trip count ---------------------------------------------------------------------------
    let trip = match &l.trip {
        TripCount::Const(n) => {
            let groups = n / w as u64;
            if groups < MIN_TRIP_GROUPS {
                return Err("trip count is below one vector group");
            }
            TripPlan::Const(groups * w as u64)
        }
        TripCount::Affine {
            step: 1,
            inclusive: false,
            end,
            ..
        } => TripPlan::Runtime { end: *end },
        // An inclusive bound would need `end + 1`, which overflows at `end == MAX`; a non-unit step
        // is already excluded above.
        _ => return Err("trip count is not a unit-step half-open range"),
    };

    // ---- per-instruction plan -----------------------------------------------------------------
    let (plans, vector_ty, reductions) =
        plan_body(f, l, body, w).ok_or("an instruction in the body cannot be widened")?;

    // ---- runtime alias guard ------------------------------------------------------------------
    let (guards, guard_elem, guard_allows_equal) =
        plan_guards(f, l).ok_or("the runtime alias check cannot be expressed")?;

    Ok(VecPlan {
        header: l.header,
        body,
        preheader,
        iv_index,
        iv_ty,
        trip,
        w,
        plans,
        vector_ty,
        reductions,
        guards,
        guard_elem,
        guard_allows_equal,
    })
}

/// The header may compute the exit test and nothing else: no memory, no side effects, and no value
/// the body reads. The vector loop replaces the test with its own, so anything else in there would
/// simply be dropped.
fn header_is_test_only(f: &Function, l: &NaturalLoop) -> bool {
    let header = &f.blocks[l.header.0 as usize];
    let defined: FxHashSet<u32> = header
        .insts
        .iter()
        .filter_map(|i| i.result.map(|r| r.0))
        .collect();
    if header.insts.iter().any(|i| {
        matches!(
            i.op,
            Op::Store { .. }
                | Op::Load(..)
                | Op::Call { .. }
                | Op::VecKernelCall { .. }
                | Op::Alloca(..)
        )
    }) {
        return false;
    }
    // Nothing the header computes may be read outside it. (The header's own terminator reads the
    // exit condition, which is fine; every other block of the loop is the body.)
    let mut escapes = false;
    for &b in &l.blocks {
        if b == l.header {
            continue;
        }
        let blk = &f.blocks[b.0 as usize];
        for inst in &blk.insts {
            each_op_use(&inst.op, &mut |u| {
                if defined.contains(&u.0) {
                    escapes = true;
                }
            });
        }
        crate::each_term_use(&blk.term, &mut |u| {
            if defined.contains(&u.0) {
                escapes = true;
            }
        });
    }
    !escapes
}

/// Every header parameter must be one this pass knows how to advance across a whole group: the
/// primary induction variable, a reduction accumulator, or a value carried unchanged.
///
/// A *secondary* induction variable is the case this rules out, and it is worth naming because it
/// is the one that would be silently wrong rather than merely unhandled. `j` is carried, `j = j +
/// 2` in the latch, and the vector body has to advance it by `2·W`, not by `2`. Worse, if `j` is
/// also *read* in the body it does not hold one value across the group at all. Rather than get one
/// of those two right and the other wrong, decline.
fn carried_values_are_handled(l: &NaturalLoop, primary_param: usize) -> bool {
    l.carried.iter().enumerate().all(|(k, c)| match c.kind {
        Carried::Iv(_) => k == primary_param,
        Carried::Reduction(_) | Carried::Invariant => true,
        Carried::Recurrence => false,
    })
}

/// A value that moves with the induction variable may only be used to *address* memory.
///
/// This is the invariant that makes [`Wide::Uniform`] mean what it says. The induction variable is
/// the same in every lane only when it is a `gep` index: the vector load at `base + i` covers
/// elements `i .. i+W-1` on its own, so lane `k` reads exactly what scalar iteration `i+k` read.
/// Used anywhere else — `(i as f32)`, `o[i] = i * 2`, `x[i] + i` — it is *not* uniform, and
/// splatting it writes the group's first value into all `W` lanes. That was a real miscompile
/// during development: an initializer loop `x[i] = (i as f32) * 0.25` produced four copies of every
/// fourth element and interp and native agreed on the wrong answer, so only the `-O0` vs `-O2` gate
/// caught it.
///
/// The set is grown as a transitive closure rather than read off
/// [`crate::loop_info::NaturalLoop::derived`], because `derived` only records *integer* results: a
/// `sitofp` of the induction variable is exactly the dangerous case and would not appear in it.
/// Address arithmetic (`add`/`sub`/`mul`/`shl`/`sext`/`neg`, and `gep` itself) may consume an index
/// value, because its result is then an index value too and is checked in turn; the closure
/// terminates at a `gep` index, a load/store pointer, or the latch's branch arguments (where the
/// only index value is the induction variable's own increment, which [`emit_body`] replaces with a
/// step of `W`).
fn index_values_only_address(f: &Function, l: &NaturalLoop, body: BlockId, iv: ValueId) -> bool {
    let mut index: FxHashSet<u32> = FxHashSet::default();
    index.insert(iv.0);
    // The header parameters other than the IV are not index values: `carried_values_are_handled`
    // has already restricted them to reductions and pass-through invariants.
    let _ = l;
    for inst in &f.blocks[body.0 as usize].insts {
        let mut touches_index = false;
        each_op_use(&inst.op, &mut |u| {
            if index.contains(&u.0) {
                touches_index = true;
            }
        });
        if !touches_index {
            continue;
        }
        // `propagates` is what makes this a closure and not just a filter: an address computed
        // from a lane-varying index is itself lane-varying, but the *value* a lane-varying address
        // loads is not — it is the ordinary vector element, and treating it as an index would
        // reject every loop body that does arithmetic on what it read.
        let (ok, propagates) = match &inst.op {
            Op::Gep { .. } => (true, true),
            Op::Bin(BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Shl, ..)
            | Op::Cast(CastKind::SExt, ..)
            | Op::Neg(_) => (true, true),
            // Only the *pointer* may be lane-varying; a lane-varying stored value would be splatted.
            Op::Load(p, _) => (index.contains(&p.0), false),
            Op::Store { ptr, value } => {
                (index.contains(&ptr.0) && !index.contains(&value.0), false)
            }
            _ => (false, false),
        };
        if !ok {
            return false;
        }
        if propagates {
            if let Some(r) = inst.result {
                index.insert(r.0);
            }
        }
    }
    true
}

/// A load at a loop-invariant address executes once per *group* in the vector body but once per
/// *element* in the scalar loop. That is only the same program when nothing in the loop can change
/// what it reads.
///
/// The `Ptr`-typed case is exempt: it is a `[]T` slice re-reading its own data pointer from the fat
/// pointer, which is exactly the assumption
/// [`crate::loop_info::MemDep::IndependentIfBasesStable`] already states and which the dependence
/// verdict has already been checked against. Any other invariant load in a loop that also *writes*
/// memory is declined, because the write could be the thing it re-reads.
fn invariant_loads_are_safe(l: &NaturalLoop) -> bool {
    let writes = l.accesses.iter().any(|a| a.kind == AccessKind::Store);
    !writes
        || !l.accesses.iter().any(|a| {
            matches!(a.addr, AddrForm::Invariant(_))
                && a.kind == AccessKind::Load
                && a.value_ty != MirType::Ptr
        })
}

/// The lane count, from the element types the loop's affine accesses transfer. Every widened access
/// has to agree on it: a body mixing `f32` and `f64` streams would need two different group sizes.
fn pick_width(l: &NaturalLoop) -> Option<u32> {
    let mut w: Option<u32> = None;
    for a in &l.accesses {
        let AddrForm::Affine { elem, .. } = &a.addr else {
            continue;
        };
        if a.addr.const_stride() != Some(1) {
            return None; // a strided access is a gather, not a vector load
        }
        // The `gep`'s unit and the value transferred must be the same type, or `base + index`
        // does not step one transferred element per lane.
        if *elem != a.value_ty {
            return None;
        }
        let lanes = lanes_of(&a.value_ty)?;
        match w {
            None => w = Some(lanes),
            Some(prev) if prev == lanes => {}
            Some(_) => return None,
        }
    }
    w
}

/// Decide, for every instruction of the body, whether it stays scalar or is widened — and check
/// that every widened one *can* be. Returns `None` the moment one cannot.
fn plan_body(
    f: &Function,
    l: &NaturalLoop,
    body: BlockId,
    w: u32,
) -> Option<(Vec<Plan>, FxHashMap<u32, MirType>, Vec<RedPlan>)> {
    let blk = &f.blocks[body.0 as usize];
    let access_at: FxHashMap<usize, &MemAccess> = l
        .accesses
        .iter()
        .filter(|a| a.block == body)
        .map(|a| (a.inst, a))
        .collect();

    // The reductions, keyed by their combining instruction's result.
    let mut red_of: FxHashSet<u32> = FxHashSet::default();
    let mut reductions = Vec::new();
    for r in &l.reductions {
        let ty = f.value_type(r.param).clone();
        let addend = r.addend.map(|a| {
            // Which side was the accumulator? `combine_kind` accepted either, and for a float
            // `sub`-free `add`/`mul` the two orders are bit-identical — but recording it costs
            // nothing and keeps a future non-commutative reduction honest.
            let left_is_acc = matches!(
                op_of(f, r.combine),
                Some(Op::Bin(_, x, _)) if *x == r.param
            );
            (a, left_is_acc)
        });
        red_of.insert(r.combine.0);
        reductions.push(RedPlan {
            param_index: r.param_index,
            kind: r.kind,
            addend,
            fma_factors: r.fma_factors,
            ty,
        });
    }

    // Every value the vector body may read: a header parameter (which becomes the vector header's
    // parameter) or an instruction already reproduced above. A value defined in the loop that is
    // in neither set has no counterpart in the vector body, and reading the original loop's copy
    // would be a use that its definition does not dominate. That is not a theoretical worry — an
    // inclusive prefix scan (`acc = acc + src[i]; scan[i] = acc;`) is exactly it: the accumulator's
    // only *parameter* use is the combine, so it classifies as a reduction, and then the store
    // reads the combine's result, which the fold does not materialize. `tests/run/prefix_sum.wk`
    // failed the verifier this way, which is the good outcome — the check below turns it into a
    // decline.
    let mut available: FxHashSet<u32> = f.blocks[l.header.0 as usize]
        .params
        .iter()
        .map(|p| p.0)
        .collect();
    let defined_in_loop: FxHashSet<u32> = l
        .blocks
        .iter()
        .flat_map(|b| {
            let blk = &f.blocks[b.0 as usize];
            blk.params
                .iter()
                .map(|p| p.0)
                .chain(blk.insts.iter().filter_map(|i| i.result.map(|r| r.0)))
        })
        .collect();

    // Forward dataflow over the single body block: a value is a vector iff it is a widened load or
    // has a vector operand. Loop-invariant values and the induction variable are uniform.
    let mut vector_ty: FxHashMap<u32, MirType> = FxHashMap::default();
    let mut plans: Vec<Plan> = Vec::with_capacity(blk.insts.len());
    let is_vec = |m: &FxHashMap<u32, MirType>, v: ValueId| m.contains_key(&v.0);

    for (i, inst) in blk.insts.iter().enumerate() {
        let mut readable = true;
        each_op_use(&inst.op, &mut |u| {
            if defined_in_loop.contains(&u.0) && !available.contains(&u.0) {
                readable = false;
            }
        });
        if !readable {
            return None;
        }
        if let Some(r) = inst.result {
            if red_of.contains(&r.0) {
                plans.push(Plan::Reduce);
                continue; // deliberately NOT added to `available`
            }
            available.insert(r.0);
        }
        let mut operand_is_vec = false;
        each_op_use(&inst.op, &mut |u| {
            if is_vec(&vector_ty, u) {
                operand_is_vec = true;
            }
        });
        let plan = match &inst.op {
            Op::Load(_, ty) => match access_at.get(&i).map(|a| &a.addr) {
                Some(AddrForm::Affine { .. }) => {
                    if lanes_of(ty)? != w {
                        return None;
                    }
                    vector_ty.insert(inst.result?.0, ty.clone());
                    Plan::VecLoad
                }
                // An invariant address yields the same value every iteration: keep it scalar and
                // let a consumer splat it. This is how a `[]T` slice's data pointer is read.
                Some(AddrForm::Invariant(_)) => Plan::Scalar,
                _ => return None,
            },
            Op::Store { ptr: _, value } => match access_at.get(&i).map(|a| &a.addr) {
                Some(AddrForm::Affine { .. }) => {
                    let vt = f.value_type(*value);
                    if lanes_of(vt)? != w {
                        return None;
                    }
                    Plan::VecStore
                }
                // Every lane would write the same address; only one wins, and which one is not the
                // scalar loop's answer.
                _ => return None,
            },
            // Address arithmetic stays scalar and must *be* scalar: a vector index is a gather.
            Op::Gep { .. } => {
                if operand_is_vec {
                    return None;
                }
                Plan::Scalar
            }
            Op::ConstInt(..) | Op::ConstFloat(..) => Plan::Scalar,
            Op::Bin(op, ..) => {
                if !operand_is_vec {
                    Plan::Scalar
                } else {
                    let ty = f.value_type(inst.result?).clone();
                    if lanes_of(&ty)? != w || !bin_widenable(*op, &ty) {
                        return None;
                    }
                    vector_ty.insert(inst.result?.0, ty);
                    Plan::Widen
                }
            }
            Op::Cmp(..) => {
                if !operand_is_vec {
                    Plan::Scalar
                } else {
                    // The operand lane type decides the mask width, not the `i1` result type.
                    let mut lane: Option<MirType> = None;
                    each_op_use(&inst.op, &mut |u| {
                        if let Some(t) = vector_ty.get(&u.0) {
                            lane = Some(t.clone());
                        }
                    });
                    let lane = lane?;
                    if lanes_of(&lane)? != w {
                        return None;
                    }
                    vector_ty.insert(inst.result?.0, mask_lane(&lane));
                    Plan::Widen
                }
            }
            Op::Select(c, a, b) => {
                if !operand_is_vec {
                    Plan::Scalar
                } else {
                    // A uniform `i1` condition cannot select between vectors: the mask has to be a
                    // per-lane all-ones/all-zeros vector, which only a vector compare produces.
                    if !is_vec(&vector_ty, *c) || !is_vec(&vector_ty, *a) || !is_vec(&vector_ty, *b)
                    {
                        return None;
                    }
                    let ty = f.value_type(inst.result?).clone();
                    if lanes_of(&ty)? != w {
                        return None;
                    }
                    vector_ty.insert(inst.result?.0, ty);
                    Plan::Widen
                }
            }
            Op::Neg(_) | Op::Not(_) | Op::Fma(..) | Op::Sqrt(_) | Op::Round(..) => {
                if !operand_is_vec {
                    Plan::Scalar
                } else {
                    let ty = f.value_type(inst.result?).clone();
                    if lanes_of(&ty)? != w {
                        return None;
                    }
                    vector_ty.insert(inst.result?.0, ty);
                    Plan::Widen
                }
            }
            Op::Cast(kind, v, to) => {
                if !operand_is_vec {
                    Plan::Scalar
                } else {
                    // A widening or narrowing cast changes the lane count, which would split or
                    // merge groups. Only same-width lane conversions are one-for-one.
                    let from = vector_ty.get(&v.0)?;
                    if lane_bytes(from) != lane_bytes(to)
                        || lanes_of(to)? != w
                        || !cast_widenable(*kind)
                    {
                        return None;
                    }
                    vector_ty.insert(inst.result?.0, to.clone());
                    Plan::Widen
                }
            }
            // A call may do anything; an alloca inside the body would allocate per lane group; a
            // splat/veckernel means someone already vectorized this.
            _ => return None,
        };
        plans.push(plan);
    }

    // Every reduction must actually be foldable: its inputs have to be vectors of this width, or
    // the "extract W lanes and accumulate serially" step has nothing to extract.
    for r in &reductions {
        if lanes_of(&r.ty)? != w {
            return None;
        }
        match (r.addend, r.fma_factors) {
            (Some((a, _)), None) => {
                if !is_vec(&vector_ty, a) {
                    return None;
                }
            }
            (None, Some((x, y))) => {
                if !is_vec(&vector_ty, x) || !is_vec(&vector_ty, y) {
                    return None;
                }
            }
            _ => return None,
        }
    }
    Some((plans, vector_ty, reductions))
}

/// May this binary operator be applied lane-wise at this result type?
///
/// Integer division and remainder are excluded: Cranelift has no vector `sdiv`/`udiv` on x64 and
/// legalization panics rather than declining.
fn bin_widenable(op: BinOp, ty: &MirType) -> bool {
    let lane = ty.lane_type();
    match op {
        BinOp::SDiv | BinOp::UDiv | BinOp::SRem | BinOp::URem | BinOp::FRem => false,
        BinOp::FAdd | BinOp::FSub | BinOp::FMul | BinOp::FDiv => lane.is_float(),
        BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Shl | BinOp::LShr | BinOp::AShr => {
            lane.is_int()
        }
        BinOp::And | BinOp::Or | BinOp::Xor => true,
    }
}

/// Casts that are one-for-one at a fixed lane width.
fn cast_widenable(k: CastKind) -> bool {
    matches!(
        k,
        CastKind::SiToFp | CastKind::UiToFp | CastKind::FpToSi | CastKind::FpToUi | CastKind::Bitcast
    )
}

/// The runtime pointer-range checks this loop needs.
///
/// [`crate::loop_info::MemDep::IndependentIfBasesDisjoint`] states the obligation: every pair of
/// bases the analysis could not relate, where at least one is written, must at runtime be either
/// the same pointer (distance 0) or non-overlapping. A guarded access has to sit at a constant
/// offset from its base (`1·iv + k`) so the range it touches is expressible as two `gep`s.
fn plan_guards(f: &Function, l: &NaturalLoop) -> Option<(Vec<GuardPair>, MirType, bool)> {
    // Group the affine accesses by base, tracking the offset range and whether any is a store.
    let mut bases: Vec<(GuardBase, bool, MirType)> = Vec::new();
    for a in &l.accesses {
        let AddrForm::Affine { base, index, elem } = &a.addr else {
            continue;
        };
        let is_store = a.kind == AccessKind::Store;
        match bases.iter_mut().find(|(g, _, _)| g.base.same_object(*base)) {
            Some((g, st, e)) => {
                if e != elem {
                    return None;
                }
                g.lo = g.lo.min(index.konst);
                g.hi = g.hi.max(index.konst);
                *st |= is_store;
            }
            None => bases.push((
                GuardBase {
                    base: *base,
                    lo: index.konst,
                    hi: index.konst,
                },
                is_store,
                elem.clone(),
            )),
        }
    }
    let elem = bases
        .first()
        .map(|(_, _, e)| e.clone())
        .unwrap_or(MirType::I8);
    let uniform_offset = bases.iter().all(|(g, _, _)| g.lo == g.hi)
        && bases.windows(2).all(|p| p[0].0.lo == p[1].0.lo);

    let mut guards = Vec::new();
    for i in 0..bases.len() {
        for j in i + 1..bases.len() {
            let (ga, sa, _) = &bases[i];
            let (gb, sb, _) = &bases[j];
            if !sa && !sb {
                continue; // read/read is never a dependence
            }
            if distinct_alloca_pair(f, ga.base, gb.base) {
                continue;
            }
            // The whole loop's index must be `1·iv + k` for the touched range to be two `gep`s.
            if !offsets_are_constant(l, ga.base) || !offsets_are_constant(l, gb.base) {
                return None;
            }
            guards.push(GuardPair { a: *ga, b: *gb });
        }
    }
    Some((guards, elem, uniform_offset))
}

/// Does every access on this base use an index of the form `1·iv + constant`?
fn offsets_are_constant(l: &NaturalLoop, base: MemBase) -> bool {
    l.accesses.iter().all(|a| match &a.addr {
        AddrForm::Affine { base: b, index, .. } if b.same_object(base) => {
            index.terms.is_empty() && index.coeff.as_const() == Some(1)
        }
        _ => true,
    })
}

fn distinct_alloca_pair(f: &Function, a: MemBase, b: MemBase) -> bool {
    let is_alloca = |m: MemBase| match m {
        MemBase::Outer(v) => matches!(op_of(f, v), Some(Op::Alloca(..))),
        MemBase::ReloadedOuter { .. } => false,
    };
    is_alloca(a) && is_alloca(b) && !a.same_object(b)
}

fn op_of(f: &Function, v: ValueId) -> Option<&Op> {
    f.blocks
        .iter()
        .flat_map(|b| b.insts.iter())
        .find(|i| i.result == Some(v))
        .map(|i| &i.op)
}

// ---------------------------------------------------------------------------------------------
// Emission
// ---------------------------------------------------------------------------------------------

/// A tiny append-only instruction builder over a `Function`.
struct Emit<'a> {
    f: &'a mut Function,
}

impl<'a> Emit<'a> {
    fn value(&mut self, ty: MirType) -> ValueId {
        let id = ValueId(self.f.value_types.len() as u32);
        self.f.value_types.push(ty);
        id
    }

    fn push(&mut self, b: BlockId, ty: MirType, op: Op) -> ValueId {
        let r = self.value(ty);
        self.f.blocks[b.0 as usize].insts.push(Inst {
            result: Some(r),
            op,
        });
        r
    }

    fn push_void(&mut self, b: BlockId, op: Op) {
        self.f.blocks[b.0 as usize]
            .insts
            .push(Inst { result: None, op });
    }

    /// Insert an instruction just before the entry block's terminator (where an `alloca` belongs:
    /// executed once per call, dominating every block).
    fn push_entry(&mut self, ty: MirType, op: Op) -> ValueId {
        let r = self.value(ty);
        let e = self.f.entry.0 as usize;
        self.f.blocks[e].insts.push(Inst {
            result: Some(r),
            op,
        });
        r
    }

    fn block(&mut self, params: Vec<ValueId>, term: Terminator) -> BlockId {
        let id = BlockId(self.f.blocks.len() as u32);
        self.f.blocks.push(BasicBlock {
            id,
            params,
            insts: Vec::new(),
            term,
        });
        id
    }

    fn int(&mut self, b: BlockId, ty: &MirType, k: i64) -> ValueId {
        self.push(b, ty.clone(), Op::ConstInt(k as i128, ty.clone()))
    }
}

fn apply(f: &mut Function, p: &VecPlan) {
    let header_params: Vec<ValueId> = f.blocks[p.header.0 as usize].params.clone();
    let header_tys: Vec<MirType> = header_params
        .iter()
        .map(|v| f.value_type(*v).clone())
        .collect();

    let mut e = Emit { f };

    // ---- the four new blocks (terminators are patched in once every block exists) --------------
    let guard_params: Vec<ValueId> = header_tys.iter().map(|t| e.value(t.clone())).collect();
    let guard = e.block(guard_params.clone(), Terminator::Unreachable);
    let vh_params: Vec<ValueId> = header_tys.iter().map(|t| e.value(t.clone())).collect();
    let vh = e.block(vh_params.clone(), Terminator::Unreachable);
    let vb = e.block(Vec::new(), Terminator::Unreachable);
    let vx = e.block(Vec::new(), Terminator::Unreachable);

    // ---- guard: how many elements does the vector loop own? ------------------------------------
    let start = guard_params[p.iv_index];
    let n_vec = emit_trip(&mut e, p, guard, start);
    let ok = emit_alias_checks(&mut e, p, guard);
    let n_vec = match ok {
        Some(cond) => {
            let zero = e.int(guard, &p.iv_ty, 0);
            e.push(guard, p.iv_ty.clone(), Op::Select(cond, n_vec, zero))
        }
        None => n_vec,
    };
    let vec_end = e.push(guard, p.iv_ty.clone(), Op::Bin(BinOp::Add, start, n_vec));
    e.f.blocks[guard.0 as usize].term = Terminator::Br {
        target: vh,
        args: guard_params.clone(),
    };

    // ---- vector header: `while iv != vec_end` --------------------------------------------------
    // `!=` rather than `<`: `vec_end - start` is a non-negative multiple of W and the induction
    // variable steps by exactly W, so it lands on `vec_end` exactly. That sidesteps the signed /
    // unsigned question the original exit test answers for itself.
    let cond = e.push(
        vh,
        MirType::I1,
        Op::Cmp(CmpOp::Ne, vh_params[p.iv_index], vec_end),
    );
    e.f.blocks[vh.0 as usize].term = Terminator::CondBr {
        cond,
        then_blk: vb,
        then_args: Vec::new(),
        else_blk: vx,
        else_args: Vec::new(),
    };

    // ---- vector exit: hand the original loop the advanced values -------------------------------
    e.f.blocks[vx.0 as usize].term = Terminator::Br {
        target: p.header,
        args: vh_params.clone(),
    };

    // ---- vector body ---------------------------------------------------------------------------
    let latch_args = emit_body(&mut e, p, vb, &header_params, &vh_params);
    e.f.blocks[vb.0 as usize].term = Terminator::Br {
        target: vh,
        args: latch_args,
    };

    // ---- splice: the preheader now enters the guard ---------------------------------------------
    retarget(
        &mut e.f.blocks[p.preheader.0 as usize].term,
        p.header,
        guard,
    );
}

/// `n_vec`: the element count the vector loop owns — the trip count rounded down to a multiple of
/// `W`, or zero when the loop does not run a full group.
fn emit_trip(e: &mut Emit<'_>, p: &VecPlan, guard: BlockId, start: ValueId) -> ValueId {
    match &p.trip {
        TripPlan::Const(n) => e.int(guard, &p.iv_ty, *n as i64),
        TripPlan::Runtime { end } => {
            let count = e.push(guard, p.iv_ty.clone(), Op::Bin(BinOp::Sub, *end, start));
            // `count & !(W-1)` rounds down for a non-negative count; the compare below is what
            // keeps a negative one (an already-finished loop) from producing a negative `n_vec`
            // and rewinding the scalar loop's induction variable behind its start.
            let mask = e.int(guard, &p.iv_ty, !((p.w as i64) - 1));
            let masked = e.push(guard, p.iv_ty.clone(), Op::Bin(BinOp::And, count, mask));
            let min = e.int(guard, &p.iv_ty, p.w as i64 * MIN_TRIP_GROUPS as i64);
            let big = e.push(guard, MirType::I1, Op::Cmp(CmpOp::Sge, count, min));
            let zero = e.int(guard, &p.iv_ty, 0);
            e.push(guard, p.iv_ty.clone(), Op::Select(big, masked, zero))
        }
    }
}

/// The conjunction of every pair check, or `None` when no pair needed one.
///
/// Each pair contributes `a == b || a_end <= b_start || b_end <= a_start` over the element ranges
/// the *vector part* of the loop touches. Pointers are compared as integers, unsigned: both
/// execution models represent an object as a contiguous range and agree on which ranges overlap
/// (the interpreter's addresses are slot indices into one arena, the native backend's are machine
/// addresses), and the `gep`s that build the range endpoints scale by the same element unit in
/// each. This is an ordering question about two live objects, never a null test — the repo's
/// "interpreter address 0 is a valid address" landmine does not apply.
fn emit_alias_checks(e: &mut Emit<'_>, p: &VecPlan, guard: BlockId) -> Option<ValueId> {
    if p.guards.is_empty() {
        return None;
    }
    // The vector part spans `[start, start + n_vec)`; conservatively use the whole loop, whose
    // endpoints are already in hand, since a range that is too large only rejects more.
    let start = e.f.blocks[guard.0 as usize].params[p.iv_index];
    let end = match &p.trip {
        TripPlan::Const(n) => {
            let k = e.int(guard, &p.iv_ty, *n as i64);
            e.push(guard, p.iv_ty.clone(), Op::Bin(BinOp::Add, start, k))
        }
        TripPlan::Runtime { end } => *end,
    };

    let mut acc: Option<ValueId> = None;
    let mut ptr_cache: FxHashMap<u32, ValueId> = FxHashMap::default();
    for g in &p.guards {
        let (alo, ahi) = range_of(e, p, guard, g.a, start, end, &mut ptr_cache);
        let (blo, bhi) = range_of(e, p, guard, g.b, start, end, &mut ptr_cache);
        // a_end <= b_start
        let a_before = e.push(guard, MirType::I1, Op::Cmp(CmpOp::Ule, ahi, blo));
        let b_before = e.push(guard, MirType::I1, Op::Cmp(CmpOp::Ule, bhi, alo));
        let mut pair = e.push(guard, MirType::I1, Op::Bin(BinOp::Or, a_before, b_before));
        if p.guard_allows_equal {
            let same = e.push(guard, MirType::I1, Op::Cmp(CmpOp::Eq, alo, blo));
            pair = e.push(guard, MirType::I1, Op::Bin(BinOp::Or, pair, same));
        }
        acc = Some(match acc {
            None => pair,
            Some(prev) => e.push(guard, MirType::I1, Op::Bin(BinOp::And, prev, pair)),
        });
    }
    acc
}

/// The half-open address range `[base + start + lo, base + end + hi)` this base touches, as two
/// integers.
fn range_of(
    e: &mut Emit<'_>,
    p: &VecPlan,
    guard: BlockId,
    g: GuardBase,
    start: ValueId,
    end: ValueId,
    cache: &mut FxHashMap<u32, ValueId>,
) -> (ValueId, ValueId) {
    let handle = g.base.outer_handle();
    let base = match g.base {
        MemBase::Outer(v) => v,
        MemBase::ReloadedOuter { addr, .. } => match cache.get(&addr.0) {
            Some(&v) => v,
            None => {
                let v = e.push(guard, MirType::Ptr, Op::Load(addr, MirType::Ptr));
                cache.insert(handle.0, v);
                v
            }
        },
    };
    let off = |e: &mut Emit<'_>, at: ValueId, k: i64| -> ValueId {
        let idx = if k == 0 {
            at
        } else {
            let c = e.int(guard, &p.iv_ty, k);
            e.push(guard, p.iv_ty.clone(), Op::Bin(BinOp::Add, at, c))
        };
        let ptr = e.push(
            guard,
            MirType::Ptr,
            Op::Gep {
                ptr: base,
                index: idx,
                elem: p.guard_elem.clone(),
            },
        );
        e.push(
            guard,
            MirType::I64,
            Op::Cast(CastKind::PtrToInt, ptr, MirType::I64),
        )
    };
    // The induction variable runs over `[start, end)` and each access adds a constant in
    // `[lo, hi]`, so the highest element index touched is `end - 1 + hi` and the half-open upper
    // bound is `end + hi`.
    let lo = off(e, start, g.lo);
    let hi = off(e, end, g.hi);
    (lo, hi)
}

/// Emit the widened body into `vb`, returning the arguments its back edge passes to the vector
/// header.
fn emit_body(
    e: &mut Emit<'_>,
    p: &VecPlan,
    vb: BlockId,
    header_params: &[ValueId],
    vh_params: &[ValueId],
) -> Vec<ValueId> {
    let body_insts = e.f.blocks[p.body.0 as usize].insts.clone();
    let latch_term = e.f.blocks[p.body.0 as usize].term.clone();

    // scalar value -> its counterpart in the vector body.
    let mut map: FxHashMap<u32, Wide> = FxHashMap::default();
    for (i, hp) in header_params.iter().enumerate() {
        map.insert(hp.0, Wide::Uniform(vh_params[i]));
    }

    for (i, inst) in body_insts.iter().enumerate() {
        match &p.plans[i] {
            Plan::Reduce => continue, // folded after the body, see below
            Plan::Scalar => {
                let mut op = inst.op.clone();
                crate::map_op_uses(&mut op, |u| resolve_uniform(&map, u));
                match inst.result {
                    Some(r) => {
                        let ty = e.f.value_type(r).clone();
                        let nv = e.push(vb, ty, op);
                        map.insert(r.0, Wide::Uniform(nv));
                    }
                    None => e.push_void(vb, op),
                }
            }
            Plan::VecLoad => {
                let Op::Load(ptr, ty) = &inst.op else {
                    unreachable!("VecLoad plan on a non-load")
                };
                let ptr = resolve_uniform(&map, *ptr);
                let vty = vec_of(ty, p.w);
                let nv = e.push(vb, vty.clone(), Op::Load(ptr, vty));
                map.insert(inst.result.expect("load has a result").0, Wide::Vector(nv));
            }
            Plan::VecStore => {
                let Op::Store { ptr, value } = &inst.op else {
                    unreachable!("VecStore plan on a non-store")
                };
                let lane = e.f.value_type(*value).clone();
                let ptr = resolve_uniform(&map, *ptr);
                let value = to_vector(e, vb, &map, *value, &lane, p.w);
                e.push_void(vb, Op::Store { ptr, value });
            }
            Plan::Widen => {
                let r = inst.result.expect("a widened op has a result");
                let lane = p.vector_ty[&r.0].clone();
                let vty = vec_of(&lane, p.w);
                let op = widen_op(e, vb, &map, &inst.op, p, &vty);
                let nv = e.push(vb, vty, op);
                map.insert(r.0, Wide::Vector(nv));
            }
        }
    }

    // The serial accumulate, after every load and store of the group has happened. Moving it here
    // is safe: an accumulator is a register value, so no store in the body can change what it
    // reads, and its own operands are all defined above.
    let mut acc_of: FxHashMap<usize, ValueId> = FxHashMap::default();
    for r in &p.reductions {
        let init = vh_params[r.param_index];
        let out = emit_reduction_fold(e, vb, &map, r, init, p.w);
        acc_of.insert(r.param_index, out);
    }

    // Back-edge arguments, one per header parameter.
    let latch_args = match &latch_term {
        Terminator::Br { args, .. } => args.clone(),
        Terminator::CondBr {
            then_blk,
            then_args,
            else_args,
            ..
        } => {
            if *then_blk == p.header {
                then_args.clone()
            } else {
                else_args.clone()
            }
        }
        _ => unreachable!("the latch branches to the header"),
    };
    let mut out = Vec::with_capacity(latch_args.len());
    for (k, a) in latch_args.iter().enumerate() {
        if k == p.iv_index {
            let step = e.int(vb, &p.iv_ty, p.w as i64);
            out.push(e.push(
                vb,
                p.iv_ty.clone(),
                Op::Bin(BinOp::Add, vh_params[k], step),
            ));
        } else if let Some(&acc) = acc_of.get(&k) {
            out.push(acc);
        } else {
            out.push(resolve_uniform(&map, *a));
        }
    }
    out
}

/// Fold `W` lane values into the accumulator **one at a time, in index order**.
///
/// This is the whole float-safety argument of the pass. `acc` starts at the value the scalar loop
/// would have entering iteration `i`, and lane `k` is combined into it exactly where scalar
/// iteration `i + k` would have combined its own value — same operator, same operand order, same
/// intermediate roundings. Nothing is reassociated, so an `f32` sum produces bit-identical output
/// to the scalar loop and the interpreter oracle agrees with the native backend.
///
/// An `fma` accumulate re-fuses its two factors per lane rather than multiplying and adding
/// separately, because `fma(x, y, acc)` rounds once and `acc + x*y` rounds twice.
///
/// Lanes are extracted by storing the vector to a private stack slot and reading the elements back.
/// MIR has no extract-lane operation, and adding one would touch every backend; a stack round-trip
/// costs one store per group and is exact in both execution models (the interpreter scatters lanes
/// to consecutive slots, the native backend writes consecutive elements).
fn emit_reduction_fold(
    e: &mut Emit<'_>,
    vb: BlockId,
    map: &FxHashMap<u32, Wide>,
    r: &RedPlan,
    init: ValueId,
    w: u32,
) -> ValueId {
    let mut acc = init;
    match (r.addend, r.fma_factors) {
        (Some((a, left_is_acc)), None) => {
            let lanes = spill_lanes(e, vb, map, a, &r.ty, w);
            let op = red_binop(r.kind);
            for lane in lanes {
                let (x, y) = if left_is_acc { (acc, lane) } else { (lane, acc) };
                acc = e.push(vb, r.ty.clone(), Op::Bin(op, x, y));
            }
        }
        (None, Some((x, y))) => {
            let xs = spill_lanes(e, vb, map, x, &r.ty, w);
            let ys = spill_lanes(e, vb, map, y, &r.ty, w);
            for k in 0..w as usize {
                acc = e.push(vb, r.ty.clone(), Op::Fma(xs[k], ys[k], acc));
            }
        }
        _ => unreachable!("plan_body rejected any other reduction shape"),
    }
    acc
}

/// Spill a vector to a stack slot and read its lanes back, lowest index first.
fn spill_lanes(
    e: &mut Emit<'_>,
    vb: BlockId,
    map: &FxHashMap<u32, Wide>,
    v: ValueId,
    lane: &MirType,
    w: u32,
) -> Vec<ValueId> {
    let vec = match map.get(&v.0) {
        Some(Wide::Vector(x)) => *x,
        _ => unreachable!("plan_body proved this operand is a vector"),
    };
    let slot = e.push_entry(
        MirType::Ptr,
        Op::Alloca(MirType::Array(Box::new(lane.clone()), w)),
    );
    e.push_void(
        vb,
        Op::Store {
            ptr: slot,
            value: vec,
        },
    );
    (0..w)
        .map(|k| {
            let idx = e.push(vb, MirType::I64, Op::ConstInt(k as i128, MirType::I64));
            let at = e.push(
                vb,
                MirType::Ptr,
                Op::Gep {
                    ptr: slot,
                    index: idx,
                    elem: lane.clone(),
                },
            );
            e.push(vb, lane.clone(), Op::Load(at, lane.clone()))
        })
        .collect()
}

fn red_binop(k: RedKind) -> BinOp {
    match k {
        RedKind::Add => BinOp::Add,
        RedKind::Mul => BinOp::Mul,
        RedKind::And => BinOp::And,
        RedKind::Or => BinOp::Or,
        RedKind::Xor => BinOp::Xor,
        RedKind::FAdd => BinOp::FAdd,
        RedKind::FMul => BinOp::FMul,
        // `plan_body` only ever records the operators above; min/max reductions are spelled with a
        // `select` and are classified `Recurrence` by the analysis, so they never reach here.
        other => unreachable!("{} is not a binary-combine reduction", other.name()),
    }
}

/// Rebuild an operation with every operand widened to a vector.
fn widen_op(
    e: &mut Emit<'_>,
    vb: BlockId,
    map: &FxHashMap<u32, Wide>,
    op: &Op,
    p: &VecPlan,
    vty: &MirType,
) -> Op {
    // For a compare the result type is the mask, but the operands keep their own lane type.
    let operand_lane = |v: ValueId| -> MirType {
        match map.get(&v.0) {
            Some(Wide::Vector(x)) => match e_ty(e, *x) {
                MirType::Vec(l, _) => (*l).clone(),
                other => other,
            },
            _ => vty.lane_type().clone(),
        }
    };
    match op {
        Op::Bin(b, x, y) => {
            let lane = vty.lane_type().clone();
            let x = to_vector(e, vb, map, *x, &lane, p.w);
            let y = to_vector(e, vb, map, *y, &lane, p.w);
            Op::Bin(*b, x, y)
        }
        Op::Cmp(c, x, y) => {
            let lane = operand_lane(*x);
            let lane = if matches!(map.get(&x.0), Some(Wide::Vector(_))) {
                lane
            } else {
                operand_lane(*y)
            };
            let x = to_vector(e, vb, map, *x, &lane, p.w);
            let y = to_vector(e, vb, map, *y, &lane, p.w);
            Op::Cmp(*c, x, y)
        }
        Op::Neg(x) => Op::Neg(to_vector(e, vb, map, *x, vty.lane_type(), p.w)),
        Op::Not(x) => Op::Not(to_vector(e, vb, map, *x, vty.lane_type(), p.w)),
        Op::Sqrt(x) => Op::Sqrt(to_vector(e, vb, map, *x, vty.lane_type(), p.w)),
        Op::Round(m, x) => Op::Round(*m, to_vector(e, vb, map, *x, vty.lane_type(), p.w)),
        Op::Fma(a, b, c) => {
            let lane = vty.lane_type().clone();
            let a = to_vector(e, vb, map, *a, &lane, p.w);
            let b = to_vector(e, vb, map, *b, &lane, p.w);
            let c = to_vector(e, vb, map, *c, &lane, p.w);
            Op::Fma(a, b, c)
        }
        Op::Select(c, a, b) => {
            let lane = vty.lane_type().clone();
            let c = resolve_vector(map, *c);
            let a = to_vector(e, vb, map, *a, &lane, p.w);
            let b = to_vector(e, vb, map, *b, &lane, p.w);
            Op::Select(c, a, b)
        }
        Op::Cast(k, x, to) => {
            let from = operand_lane(*x);
            let x = to_vector(e, vb, map, *x, &from, p.w);
            Op::Cast(*k, x, vec_of(to.lane_type(), p.w))
        }
        other => unreachable!("plan_body accepted an op it cannot widen: {other:?}"),
    }
}

fn e_ty(e: &Emit<'_>, v: ValueId) -> MirType {
    e.f.value_type(v).clone()
}

/// The vector-body counterpart of a value that must stay scalar.
fn resolve_uniform(map: &FxHashMap<u32, Wide>, v: ValueId) -> ValueId {
    match map.get(&v.0) {
        Some(Wide::Uniform(x)) => *x,
        // A value the map does not hold is defined outside the loop and dominates the vector body
        // unchanged (the guard sits between the original preheader and the original header).
        None => v,
        Some(Wide::Vector(_)) => {
            unreachable!("plan_body proved this operand is uniform")
        }
    }
}

fn resolve_vector(map: &FxHashMap<u32, Wide>, v: ValueId) -> ValueId {
    match map.get(&v.0) {
        Some(Wide::Vector(x)) => *x,
        _ => unreachable!("plan_body proved this operand is a vector"),
    }
}

/// A value as a vector: itself if already widened, otherwise broadcast across every lane.
fn to_vector(
    e: &mut Emit<'_>,
    vb: BlockId,
    map: &FxHashMap<u32, Wide>,
    v: ValueId,
    lane: &MirType,
    w: u32,
) -> ValueId {
    match map.get(&v.0) {
        Some(Wide::Vector(x)) => *x,
        Some(Wide::Uniform(x)) => {
            let vt = vec_of(lane, w);
            e.push(vb, vt, Op::Splat(*x))
        }
        None => {
            let vt = vec_of(lane, w);
            e.push(vb, vt, Op::Splat(v))
        }
    }
}

/// Point every edge of `t` that goes to `from` at `to`, keeping its arguments.
fn retarget(t: &mut Terminator, from: BlockId, to: BlockId) {
    match t {
        Terminator::Br { target, .. } => {
            if *target == from {
                *target = to;
            }
        }
        Terminator::CondBr {
            then_blk, else_blk, ..
        } => {
            if *then_blk == from {
                *then_blk = to;
            }
            if *else_blk == from {
                *else_blk = to;
            }
        }
        Terminator::Ret(_) | Terminator::Unreachable => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use wukong_mir::Program;
    use wukong_span::{Interner, SourceId};

    fn lower(src: &str) -> (Program, Interner) {
        let mut interner = Interner::new();
        let (module, pd) = wukong_parser::parse_module(src, SourceId(0), &mut interner);
        assert!(pd.iter().all(|d| !d.is_error()), "parse: {pd:?}");
        let (sema, sd) = wukong_sema::check(&module, &interner);
        assert!(sd.iter().all(|d| !d.is_error()), "sema: {sd:?}");
        let (program, _) = wukong_mir_build::lower_program(&module, &sema, &mut interner);
        (program, interner)
    }

    fn optimized(src: &str, level: u8) -> (Program, Interner) {
        let (mut program, interner) = lower(src);
        crate::optimize(&mut program, level);
        for f in &program.funcs {
            let errs = wukong_mir::verify::verify_function(f);
            assert!(errs.is_empty(), "MIR invalid after -O{level}: {errs:?}");
        }
        (program, interner)
    }

    /// Every vector-typed value in the named function, as `(lane, lanes)` pairs.
    fn vector_values(src: &str, func: &str) -> Vec<(MirType, u32)> {
        let (program, mut interner) = optimized(src, 2);
        let sym = interner.intern(func);
        let f = program.function(sym).unwrap_or_else(|| panic!("no fn {func}"));
        f.value_types
            .iter()
            .filter_map(|t| match t {
                MirType::Vec(l, n) => Some(((**l).clone(), *n)),
                _ => None,
            })
            .collect()
    }

    fn is_widened(src: &str, func: &str) -> bool {
        !vector_values(src, func).is_empty()
    }

    /// `-O0` (the scalar loop the front end emitted) and `-O2` (the vector loop plus its epilogue)
    /// must print the same bytes and exit the same way. That is the whole correctness story of the
    /// pass, run inside the unit tests so a failure names the case rather than a corpus file.
    fn scalar_and_vector_agree(src: &str) {
        let mut outs = Vec::new();
        for level in [0u8, 2u8] {
            let (program, mut interner) = optimized(src, level);
            let main = interner.intern("main");
            let (code, bytes) = wukong_interp::run_with_output(&program, main, &interner)
                .unwrap_or_else(|e| panic!("-O{level}: {e}"));
            outs.push((code, String::from_utf8_lossy(&bytes).to_string()));
        }
        assert_eq!(outs[0], outs[1], "-O0 and -O2 disagree");
    }

    /// A `while`-spelled saxpy over three `[]f32` slices: the case the recognizers miss entirely.
    const SAXPY: &str = "fn k(a: f32, x: []f32, y: []f32, mut o: []f32, n: i64) { \
                         let mut i: i64 = 0; \
                         while i < n { o[i] = a * x[i] + y[i]; i = i + 1; } } \
                         fn main() -> i32 { return 0; }";

    #[test]
    fn an_elementwise_float_loop_is_widened_four_lanes() {
        let vs = vector_values(SAXPY, "k");
        assert!(!vs.is_empty(), "saxpy was not widened");
        assert!(
            vs.iter().all(|(l, n)| *l == MirType::F32 && *n == 4),
            "{vs:?}"
        );
    }

    #[test]
    fn an_f64_loop_is_widened_two_lanes() {
        let src = "fn k(x: []f64, mut o: []f64, n: i64) { let mut i: i64 = 0; \
                   while i < n { o[i] = x[i] * 2.5; i = i + 1; } } \
                   fn main() -> i32 { return 0; }";
        let vs = vector_values(src, "k");
        assert!(!vs.is_empty(), "f64 loop was not widened");
        assert!(
            vs.iter().all(|(l, n)| *l == MirType::F64 && *n == 2),
            "{vs:?}"
        );
    }

    #[test]
    fn an_i32_loop_is_widened_four_lanes() {
        let src = "fn k(x: []i32, mut o: []i32, n: i64) { let mut i: i64 = 0; \
                   while i < n { o[i] = x[i] * 30; i = i + 1; } } \
                   fn main() -> i32 { return 0; }";
        let vs = vector_values(src, "k");
        assert!(!vs.is_empty(), "i32 loop was not widened");
        assert!(
            vs.iter().all(|(l, n)| *l == MirType::I32 && *n == 4),
            "{vs:?}"
        );
    }

    /// `bf16` is refused on purpose: the interpreter rounds a scalar half-precision load to the
    /// storage grid and its vector load does not, so widening one would be a silent
    /// interp-vs-native divergence rather than a missed optimization.
    #[test]
    fn a_bf16_loop_is_refused() {
        let src = "fn k(x: []bf16, mut o: []bf16, n: i64) { let mut i: i64 = 0; \
                   while i < n { o[i] = x[i]; i = i + 1; } } \
                   fn main() -> i32 { return 0; }";
        assert!(!is_widened(src, "k"), "bf16 must not be widened");
    }

    // ---- cases that must be declined -----------------------------------------------------------

    #[test]
    fn a_true_recurrence_is_declined() {
        let src = "fn k(a: []f32, b: []f32, mut o: []f32, n: i64) { let mut h: f32 = 0.0; \
                   let mut i: i64 = 0; \
                   while i < n { h = a[i] * h + b[i]; o[i] = h; i = i + 1; } } \
                   fn main() -> i32 { return 0; }";
        assert!(!is_widened(src, "k"), "h = a*h + b must stay scalar");
    }

    /// An inclusive prefix scan classifies as a reduction — the accumulator's only *parameter* use
    /// is the combine — and then stores the combine's result, which the serial fold does not
    /// materialize. `tests/run/prefix_sum.wk` failed the MIR verifier exactly this way.
    #[test]
    fn a_prefix_scan_is_declined() {
        let src = "fn k(x: []i32, mut o: []i32, n: i64) { let mut acc: i32 = 0; \
                   let mut i: i64 = 0; \
                   while i < n { acc = acc + x[i]; o[i] = acc; i = i + 1; } } \
                   fn main() -> i32 { return 0; }";
        assert!(!is_widened(src, "k"), "a prefix scan must stay scalar");
    }

    /// A gathered index is not affine in the induction variable, so no lane offset is known.
    #[test]
    fn a_gathered_index_is_declined() {
        let src = "fn k(ix: []i32, x: []f32, mut o: []f32, n: i64) { let mut i: i64 = 0; \
                   while i < n { o[i] = x[ix[i] as i64]; i = i + 1; } } \
                   fn main() -> i32 { return 0; }";
        assert!(!is_widened(src, "k"), "a gather must stay scalar");
    }

    /// The induction variable is uniform across a group only as an *address*. Used as a value it
    /// is `i + k` in lane `k`, and splatting it writes the group's first index into all lanes.
    #[test]
    fn an_induction_variable_used_as_a_value_is_declined() {
        let src = "fn k(mut o: []f32, n: i64) { let mut i: i64 = 0; \
                   while i < n { o[i] = (i as f32) * 0.25; i = i + 1; } } \
                   fn main() -> i32 { return 0; }";
        assert!(!is_widened(src, "k"), "`i` as a value must stay scalar");
    }

    /// Stride 2 is a strided access, not a contiguous vector load.
    #[test]
    fn a_non_unit_stride_is_declined() {
        let src = "fn k(x: []f32, mut o: []f32, n: i64) { let mut i: i64 = 0; \
                   while i < n { o[i * 2] = x[i * 2]; i = i + 1; } } \
                   fn main() -> i32 { return 0; }";
        assert!(!is_widened(src, "k"), "stride 2 must stay scalar");
    }

    // ---- termination ---------------------------------------------------------------------------

    /// Widening leaves the original loop behind as a scalar epilogue with exactly the shape that
    /// was just accepted. Without a stop the pass would widen the epilogue, then its epilogue, and
    /// the compile would never finish — so assert directly that a second run adds nothing.
    #[test]
    fn a_second_pass_widens_nothing_further() {
        let (mut program, mut interner) = optimized(SAXPY, 2);
        let sym = interner.intern("k");
        let idx = program.funcs.iter().position(|f| f.name == sym).unwrap();
        let before = program.funcs[idx].blocks.len();
        let pass = Vectorize::default();
        let mut cache = CfgAnalyses::default();
        assert!(
            !pass.run_function(&mut program.funcs[idx], &mut cache),
            "a fresh Vectorize must not widen an already-widened function"
        );
        assert_eq!(program.funcs[idx].blocks.len(), before);
    }

    // ---- the oracle ----------------------------------------------------------------------------

    /// The elementwise case end to end, with a length that is not a multiple of the width so the
    /// scalar epilogue runs.
    #[test]
    fn elementwise_output_matches_the_scalar_loop() {
        let src = "fn k(a: f32, x: []f32, mut o: []f32, n: i64) { let mut i: i64 = 0; \
                   while i < n { o[i] = a * x[i] + 1.0; i = i + 1; } } \
                   fn main() -> i32 { \
                     let mut x: []f32 = alloc_f32(23); let mut o: []f32 = alloc_f32(23); \
                     let mut i: i64 = 0; \
                     while i < 23 { x[i] = (i as f32) * 0.5 - 3.0; o[i] = 0.0; i = i + 1; } \
                     k(1.5, x, o, 23); \
                     i = 0; while i < 23 { print((o[i] * 100.0) as i32); i = i + 1; } \
                     free(o); free(x); return 0; }";
        scalar_and_vector_agree(src);
    }

    /// A float sum over `[2^24, 1, 1, ...]` is exactly `2^24` in source order — every `2^24 + 1`
    /// is a tie that rounds back to even — and strictly larger under any regrouping. If the fold
    /// ever stops being serial, this is the test that says so.
    #[test]
    fn a_float_reduction_is_not_reassociated() {
        // The widening check needs its own copy: `-O2` inlines `k` into the `main` below, and
        // `main` has vector values of its own from the initializer loop, so asserting there would
        // prove nothing about the reduction.
        let alone = "fn k(x: []f32, n: i64) -> f32 { let mut s: f32 = 0.0; let mut i: i64 = 0; \
                     while i < n { s = s + x[i]; i = i + 1; } return s; } \
                     fn main() -> i32 { return 0; }";
        assert!(is_widened(alone, "k"), "the reduction's body must be widened");

        let src = "fn k(x: []f32, n: i64) -> f32 { let mut s: f32 = 0.0; let mut i: i64 = 0; \
                   while i < n { s = s + x[i]; i = i + 1; } return s; } \
                   fn main() -> i32 { \
                     let mut x: []f32 = alloc_f32(23); \
                     let mut i: i64 = 0; while i < 23 { x[i] = 1.0; i = i + 1; } \
                     x[0] = 16777216.0; \
                     print((k(x, 23) - 16777216.0) as i32); \
                     free(x); return 0; }";
        let (program, mut interner) = optimized(src, 2);
        let main = interner.intern("main");
        let (code, out) = wukong_interp::run_with_output(&program, main, &interner).unwrap();
        assert_eq!(
            (code, String::from_utf8_lossy(&out).to_string()),
            (0, "0\n".to_string()),
            "the accumulate was reassociated"
        );
        scalar_and_vector_agree(src);
    }

    /// Two `[]f32` parameters may be the same buffer. The runtime range check must catch it and
    /// leave the whole loop to the scalar epilogue.
    #[test]
    fn an_aliased_call_still_computes_the_scalar_answer() {
        let src = "fn k(a: f32, x: []f32, mut o: []f32, n: i64) { let mut i: i64 = 0; \
                   while i < n { o[i] = a * x[i] + 1.0; i = i + 1; } } \
                   fn main() -> i32 { \
                     let mut x: []f32 = alloc_f32(23); \
                     let mut i: i64 = 0; \
                     while i < 23 { x[i] = (i as f32) * 0.5 - 3.0; i = i + 1; } \
                     k(1.5, x, x, 23); \
                     i = 0; while i < 23 { print((x[i] * 100.0) as i32); i = i + 1; } \
                     free(x); return 0; }";
        let alone = "fn k(a: f32, x: []f32, mut o: []f32, n: i64) { let mut i: i64 = 0; \
                     while i < n { o[i] = a * x[i] + 1.0; i = i + 1; } } \
                     fn main() -> i32 { return 0; }";
        assert!(
            is_widened(alone, "k"),
            "the loop is widened; the guard is what rejects, at runtime"
        );
        scalar_and_vector_agree(src);
    }
}
