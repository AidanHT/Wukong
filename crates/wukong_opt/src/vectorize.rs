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
//! `i + k` would have. A step with no lane-wise vector form — a gather, a scatter, an integer
//! divide, a shift by a per-lane amount, a conversion with no packed route — keeps that property by
//! being **replicated** ([`plan_body`]): it runs once per lane, as the scalar instruction, on that
//! lane's own operands, and its results are packed for the widened steps that read them. A scatter's
//! `W` stores go out in lane order, which is iteration order, so of two lanes that write one element
//! the later iteration lands last — the scalar loop's answer, given that nothing else in the loop
//! touches the scattered object (`loop_info::dependence`). A call to a side-effect-free function is
//! replicated the same way ([`Plan::Call`], [`call_verdict`]).
//!
//! When something else in the loop may touch a gathered or scattered object — a read-modify-write
//! histogram, two scatters into one object, a gather beside a store it may read — no vector order is
//! the scalar loop's, but one order is: lane by lane. Those accesses, and every step reading what
//! they computed, run lane 0's, then lane 1's, ... ([`SerialRegion`]), while the rest of the body
//! stays wide. A body that checks a gathered index on every iteration tests the group's indices at
//! once and hands a failing group to the scalar loop ([`linearize_spine`]). A two-deep nest whose
//! inner loop carries a row's own float sum or recurrence is widened across its rows instead
//! ([`outer`]).
//!
//! # If-conversion
//!
//! A branch inside the body cannot be widened as control flow — lane 3 may take the `then` arm
//! while lane 0 takes the `else`. [`linearize_region`] flattens a diamond or a triangle into one
//! straight line and turns the branch condition into a lane mask, so `if c { o[i] = f } else { o[i]
//! = g }` becomes one `select` and one store. Both arms then run for every lane, which is only
//! sound when running the arm a lane did not take is harmless; the two ways it would not be — an
//! unpaired store (it would write memory the scalar loop leaves alone) and an unpaired load (it
//! would read an address the scalar loop never touches, and Wukong does not bounds-check slice
//! indexing) — are **predicated** instead: each lane accesses its own element where it took the
//! arm and a private scratch slot where it did not ([`Plan::PredStore`], [`Plan::PredLoad`]), the
//! branch-free form of a masked store and load.
//!
//! A merge that carries an accumulator across the back edge is a **guarded reduction**
//! (`if c { s = s + x }`, `loop_info::Reduction::merge`): it is folded with the other reductions,
//! each lane keeping `c ? s ⊕ x : s` — a lane-parallel integer accumulator blends per part, a serial
//! float fold selects per lane, and neither needs an identity element. A branch that only chooses
//! between the accumulator and a value it was compared against (`if x > m { m = x }`) is a min/max
//! and folds as one.
//!
//! # Width
//!
//! Every vector is 128 bits — `<4 x f32>`, `<2 x f64>`, `<8 x i16>`, `<16 x i8>`. That is not a
//! guess about what the machine wants — it is the widest vector Cranelift will legalize. A
//! `<8 x f32>` reaches the x64 backend as `Unsupported("Unexpected SSA-value type: f32x8")` and
//! fails the compile; the repo's 256-bit path is a separate mechanism (`VecKernel` +
//! `codegen_cranelift::avx2`, raw assembled AVX2) that `mir_build` builds for the `for` loops it
//! vectorizes itself — f32/f64/i32/i64 elementwise bodies and reductions, over arrays, slices and
//! tensors — and that this pass builds too, for a loop it widens whose body has a recipe form and
//! whose trip reaches the out-of-line call's crossover ([`vec256`]): the kernel then runs the
//! multiple-of-lanes part and this pass's 128-bit loop the remainder. With AVX enabled Cranelift
//! VEX-encodes the 128-bit ops, and two of them retire per cycle, so a memory-bound loop saturates
//! the same bandwidth a 256-bit one would.
//!
//! The **group width** `W` — how many scalar iterations one vector iteration covers — is set by the
//! *narrowest* lane the widened body holds: `W = 16 / narrowest_bytes`, LLVM LoopVectorize's rule.
//! A wider value is then split into **parts**: in a loop that loads `i8` and computes in `f32`,
//! `W = 16`, the bytes are one `<16 x i8>` and every `f32` value is four `<4 x f32>` parts, part `p`
//! holding iterations `4p .. 4p+3`. The lane-count-changing casts between them are `Op::Widen` and
//! `Op::Narrow` chains (`cast_route`), so an `i8` dequantize, a `u8` image normalize, an `f32 -> i8`
//! saturating quantize and an `i8` dot into an `i32` accumulator all widen. A value is never split
//! across more than [`MAX_PARTS`] vectors.
//!
//! # `bf16` and `f16` widen as `f32` lanes
//!
//! Every backend computes a `bf16` or `f16` value in an `f32` register and rounds it to its 16-bit
//! grid at exactly four points: a load (the storage holds 16 bits), a store, a constant, and a cast
//! (`fptrunc .. to bf16`/`f16`, which `mir_build` places after every half operation, and an
//! `fpext` out of one). Arithmetic in between is plain `f32`. So a half lane is an `f32` lane —
//! four to a 128-bit register, [`lane_bytes`] `4` — and `<4 x bf16>` / `<4 x f16>` keep the same
//! four rounding points lane by lane: the interpreter rounds each lane of a vector load as it
//! rounds a scalar one, and Cranelift (`codegen_cranelift::half_lanes`) loads four 2-byte elements
//! into an `f32x4` (`uload16x4`, then a shift for bf16 or an integer f16 decode), rounds and packs
//! four lanes on a store, and rounds each lane of a half `fptrunc`/`fpext`. The bf16 rounding is the
//! scalar path's integer round-to-nearest-even; the f16 one is an SSE2 integer form of the scalar
//! path's runtime shims (the `half` crate), checked against them on every f16 encoding and a dense
//! cover of `f32` inputs. The arithmetic, compares and blends between them are the `f32x4` ones.
//! Every half cast but a bitcast is lane-wise ([`cast_route`]) in the scalar cast's own steps: the
//! grid-rounding casts between `f32`, bf16 and f16 directly, a cast through `f64` as that rounding
//! plus the `f32 <-> f64` widen or narrow, and a cast to or from an integer as the `f32` one (the
//! scalar `sitofp`/`fptosi` rounds nothing at the half grid; `mir_build` rounds an integer
//! converted to a half type with the `fptrunc` it places after it, which widens like any other).
//! Half interleave groups widen like 32-bit ones: their loads and stores round where the scalar
//! ones do, and the shuffles between them move `f32` lanes. Reductions with a half accumulator (it
//! rounds inside its own recurrence) stay scalar.
//!
//! # Reading `WUKONG_VEC_TRACE=1`
//!
//! Setting it makes [`pick`] print one line per decision to stderr. **The stream is not one line
//! per loop.** This pass is a [`Pass`], the pass manager re-runs the whole list until no pass
//! reports a change, and `pick` re-walks *every* loop of the function on each of those iterations —
//! so a loop that is declined is re-declined, with the identical message, once per fixpoint
//! iteration after the one that widened something. A `WIDEN` line appears once (the header goes
//! into the function's [`DoneSets`]); a `declined` line appears as many times as the fixpoint ran.
//!
//! Measured on this tree, the two `wukong_xbench` general programs at their default dimensions:
//!
//! ```text
//!   general_loss_focal.wk (R=8192 C=1024)         general_scan_s6.wk (T=2048 D=256 N=16)
//!   4 lines total, no declines:                   5 lines total:
//!     x1 WIDEN bb13 w=4 lin=42 guards=1 reds=1      x1 WIDEN bb10 w=4 lin=53 guards=3 reds=1
//!     x1 WIDEN bb16 w=4 lin=8  guards=0 reds=0      x1 WIDEN bb1  w=4 lin=3  guards=0 reds=0
//!     x1 WIDEN bb19 w=4 lin=63 guards=2 reds=2      x3 declined bb16: induction step is not 1
//!     x1 WIDEN bb25 w=4 lin=12 guards=1 reds=0
//! ```
//!
//! (Those lines predate two fields the line now ends with: `guards` counts point checks — and
//! an arm's bounds checks — as well as pairs, and `cost=V/S` is the cost model's estimate for one
//! group — vector operations against `W` scalar iterations, [`estimate_cost`]. A loop whose `V`
//! exceeds its `S` is declined ([`cost_declines`]; `WUKONG_VEC_COST=0` turns that off), and the
//! runtime alias guard has its own budget, [`MAX_GUARD_PAIRS`].)
//!
//! `bb19` is the loss's fused forward+backward pass, and it is the one the **lane ramp** unlocked:
//! it reads its counter as a *value* (`if c == target`), which used to be a flat decline. See
//! [`IvRamp`].
//!
//! Quote it **with the repeat counts** or say explicitly that you have collapsed them. A
//! de-duplicated paste reads like a per-loop verdict list, which is a different (and much stronger)
//! claim than what the tool actually reports, and it hides the one thing the repetition does tell
//! you: how many times the fixpoint went round.

use wukong_mir::{
    BasicBlock, BinOp, BlockId, CastKind, CmpOp, Function, Half, Inst, MirType, NarrowKind, Op,
    Terminator, ValueId,
};

use crate::effects::{CallEffect, CallEffects};
use crate::fxhash::{FxHashMap, FxHashSet};
use crate::loop_info::{
    self, AccessKind, AddrForm, Carried, Coeff, IvStep, LoopForest, MemAccess, MemBase, MemDep,
    NaturalLoop, RedKind, TripCount,
};
use crate::{each_op_use, CfgAnalyses, Pass};
use wukong_span::Span;

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
/// * [`DoneSets`] — a record of the header blocks already widened in the function being
///   optimized, kept in that function's [`CfgAnalyses`] (so it lives exactly as long as the
///   function's fixpoint, and the pass itself holds no state a thread could share) and reset if a
///   cache is ever handed a different function. A stale entry after `cfg::prune_unreachable`
///   renumbers blocks can only *skip* a loop, never widen one twice.
///
/// A loop handed to a 256-bit kernel ([`vec256`]) is not widened in place: the original loop stays
/// behind as the kernel's remainder, and the 128-bit path is still allowed to widen *that*. The
/// stops for the kernel path are the same two in kind — [`vec256::is_remainder`] (the counter
/// resumes at the kernel's count) and a second header record — plus a cap on the kernels one
/// function may receive.
pub struct Vectorize {
    /// May a widened loop take the 256-bit raw-AVX2 recipe instead ([`vec256`])? The pipeline's
    /// switch ([`crate::OptimizeOptions::vec_kernels`]); the host gate and `WUKONG_P4_NO_256` are
    /// the other two conjuncts ([`Vectorize::kernels_enabled`]).
    kernels: bool,
}

/// The record [`Vectorize`] keeps for one function, in its [`CfgAnalyses`]: the function, the
/// headers already widened, and the headers already given a kernel.
pub(crate) struct DoneSets {
    func: wukong_span::Symbol,
    widened: FxHashSet<u32>,
    kernels: FxHashSet<u32>,
}

impl Default for Vectorize {
    /// The pass as [`crate::optimize`] runs it: 256-bit kernels allowed wherever the host and the
    /// environment allow them — the same default as `wukong_mir_build::LowerOptions`.
    fn default() -> Vectorize {
        Vectorize::new(true)
    }
}

impl Vectorize {
    /// The pass, with the 256-bit kernel path on or off. Off leaves every loop on the 128-bit
    /// path, which computes the same values ([`vec256`]).
    pub fn new(kernels: bool) -> Vectorize {
        Vectorize { kernels }
    }

    /// The three conjuncts of the 256-bit path, read once per call: this pass's switch, the host
    /// gate the assembler refuses on (`wukong_mir::host_supports_vec_kernels` — Win64 with AVX2 and
    /// FMA3, single-sourced with `mir_build` and the Cranelift assembler), and the
    /// `WUKONG_P4_NO_256` kill-switch both front-end recipe sites read too.
    fn kernels_enabled(&self) -> bool {
        self.kernels
            && wukong_mir::host_supports_vec_kernels()
            && std::env::var_os("WUKONG_P4_NO_256").is_none()
    }
}

/// The 256-bit raw-AVX2 form of a widened loop (`Op::VecKernelCall`).
#[path = "vec256.rs"]
mod vec256;

/// Outer-loop vectorization: a two-deep nest widened across its outer counter.
#[path = "outer_vec.rs"]
mod outer;

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
        let kernels = self.kernels_enabled();
        // Which calls write nothing (`crate::effects`): a loop calling one may still widen.
        let effects = cache.call_effects();
        // Taken out of the cache for the duration (the loop below also asks the cache for
        // dominators) and put back before returning.
        let mut done = match cache.vectorized.take() {
            Some(d) if d.func == f.name => d,
            _ => DoneSets {
                func: f.name,
                widened: FxHashSet::default(),
                kernels: FxHashSet::default(),
            },
        };
        let mut changed = false;
        for _ in 0..MAX_LOOPS_PER_CALL {
            let (idom, preds) = cache.idoms_and_preds(f);
            let mut forest = loop_info::structure_with(f, idom, preds);
            let cfg = (idom, preds);
            match pick(f, &mut forest, &done, kernels, effects.as_deref(), cfg) {
                None => break,
                Some(Picked::Widen(plan)) => {
                    done.widened.insert(plan.header.0);
                    apply(f, &plan);
                }
                // The header is NOT recorded as widened: the loop left behind is the kernel's
                // remainder, which the 128-bit path may still widen (`vec256` module docs).
                Some(Picked::Kernel(plan, kernel)) => {
                    done.kernels.insert(plan.header.0);
                    vec256::apply(f, &plan, &kernel);
                }
                Some(Picked::Outer(plan)) => {
                    done.widened.insert(plan.header.0);
                    outer::apply(f, &plan);
                }
            }
            cache.invalidate();
            changed = true;
        }
        cache.vectorized = Some(done);
        changed
    }
}

// ---------------------------------------------------------------------------------------------
// Widths and lane types
// ---------------------------------------------------------------------------------------------

/// Byte width of a lane type this pass will widen, or `None` for one it refuses.
///
/// `I1` is out because a boolean vector is a *mask*, whose machine representation is one all-ones
/// lane per element of the value it selects, not a one-bit lane. A `bf16` or `f16` lane is its
/// *register* width, an `f32`'s, though it is stored in two bytes (module docs). Pointers and
/// aggregates have no lane-wise arithmetic at all.
fn lane_bytes(t: &MirType) -> Option<u32> {
    Some(match t {
        MirType::I8 => 1,
        MirType::I16 => 2,
        MirType::I32 | MirType::F32 | MirType::BF16 | MirType::F16 => 4,
        MirType::I64 | MirType::F64 => 8,
        _ => return None,
    })
}

/// Is `t` a half-precision float — bf16 or f16, an `f32` lane in a register and two bytes in
/// memory, rounded to its grid at a load, a store, a constant and a cast (module docs)?
fn is_half(t: &MirType) -> bool {
    matches!(t, MirType::BF16 | MirType::F16)
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

/// The most 128-bit vectors one widened value may be split across: the body's widest lane at most
/// four times its narrowest. An `i8` stream beside an `i64` one (eight parts per `i64` value) is
/// declined rather than turned into a body eight times the size of its scalar original.
const MAX_PARTS: u32 = 4;

/// The one full-width vector type a lane type is widened to: `<4 x f32>`, `<16 x i8>`, ...
fn part_ty(lane: &MirType) -> MirType {
    vec_of(
        lane,
        lanes_of(lane).expect("a widened lane type has a vector width"),
    )
}

/// How many [`part_ty`] vectors hold one value of lane type `lane` in a group of `w` iterations.
fn parts_of(lane: &MirType, w: u32) -> usize {
    (w / lanes_of(lane).expect("a widened lane type has a vector width")) as usize
}

/// The integer lane type of a byte width.
fn int_of_bytes(bytes: u32) -> MirType {
    match bytes {
        1 => MirType::I8,
        2 => MirType::I16,
        4 => MirType::I32,
        _ => MirType::I64,
    }
}

// ---------------------------------------------------------------------------------------------
// Planning
// ---------------------------------------------------------------------------------------------

/// One widened value: either kept scalar (the same value in every lane, or an address) or widened
/// to the `W` lanes of a group.
#[derive(Clone, PartialEq, Eq, Debug)]
enum Wide {
    /// Loop-invariant, or an address/index that is uniform across the group.
    Uniform(ValueId),
    /// The [`parts_of`] full-width vectors of one value, lowest iterations first: lane `k` of part
    /// `p` holds what scalar iteration `i + p·L + k` computed, `L` lanes per part.
    Vector(Vec<ValueId>),
    /// A value computed lane by lane ([`Plan::Replicate`], [`Plan::LaneGep`]): `W` scalars, entry
    /// `k` what scalar iteration `i + k` computed. Packed into vectors (`Op::InsertLane`) only
    /// where a widened step reads it; a per-lane address is never packed.
    Lanes(Vec<ValueId>),
}

/// How a lane conversion is spelled once the group's values are split into parts. Decided from the
/// lane types alone by [`cast_route`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CastRoute {
    /// One lane-wise `Cast` of the same kind per part (equal lane widths).
    Same,
    /// An integer cast between two types of one width: the bits do not change, so the parts are
    /// reused as they are.
    Identity,
    /// `SExt`/`ZExt` doublings (`Op::Widen`) up to the target width — then, for an int-to-float
    /// conversion, one same-width `SiToFp`. A zero-extended value is non-negative, so `SiToFp` of it
    /// is exactly the `UiToFp` of the original.
    WidenInt { ext: CastKind, then_fp: bool },
    /// Truncating halvings (`Op::Narrow`) down to the target width.
    TruncInt,
    /// `f32 -> i16/i8`: a saturating `FpToSi` to `i32`, then saturating halvings. Each step clamps
    /// into the next range, and the composition of nested clamps is the clamp to the innermost
    /// range — which is exactly the scalar saturating cast. `unsigned` makes the last step clamp to
    /// `0 ..= 2^w - 1` (a `FpToUi`).
    FloatToNarrowInt { unsigned: bool },
    /// `f32 -> f64`: each part's two halves (`Op::Widen`).
    FpExt,
    /// `f64 -> f32`: pairs of parts joined (`Op::Narrow`).
    FpTrunc,
    /// bf16/f16 `-> f64`: each part rounded to its half grid as an `f32` lane (the lane-wise
    /// `fpext .. to f32`, the rounding the scalar `fpext` applies to its source), then
    /// [`CastRoute::FpExt`].
    HalfToF64,
    /// `f64 ->` bf16/f16: [`CastRoute::FpTrunc`] to `f32` lanes, then each rounded to the half grid
    /// (the lane-wise `fptrunc`) — the scalar cast's demote, then round.
    F64ToHalf,
}

/// How one instruction of the scalar body is reproduced in the vector body.
#[derive(Clone, Debug)]
enum Plan {
    /// Copy the instruction unchanged (address arithmetic, invariant loads, constants).
    Scalar,
    /// A `Load` at a loop-invariant address whose result is a **pointer** — a `[]T` slice
    /// re-reading its own data pointer out of the fat pointer. Emitted once, in the guard block,
    /// instead of once per lane group. See [`emit_body`] for why that is the same program.
    InvariantPtrLoad,
    /// Widen it: same opcode per part, operands splatted as needed. A shift by a uniform amount
    /// keeps the amount scalar (`Op::Bin`'s vector-by-scalar shape); an `i8` multiply, which x86
    /// cannot do packed, goes through `i16` halves and truncates back.
    Widen,
    /// A conversion, by the route [`cast_route`] chose.
    Cast(CastRoute),
    /// Run the scalar op once per lane, on each lane's own operands (see [`plan_body`]).
    Replicate,
    /// A `gep` off a uniform base with a per-lane index: one address per lane.
    LaneGep,
    /// A load through per-lane addresses: one scalar load per lane, packed into parts.
    Gather,
    /// A call to a side-effect-free function with lane-varying arguments: one scalar call per lane,
    /// in lane order, on the lane's own arguments; its results are packed where a widened step
    /// reads them ([`plan_body`], [`call_verdict`]).
    Call,
    /// A store through per-lane addresses: one scalar store per lane, in lane order — which is
    /// iteration order in both directions ([`VecPlan::dir`]), so of two lanes that write one element
    /// the later iteration lands last, as in the scalar loop. Only planned when
    /// `loop_info::dependence` proved nothing else in the loop touches the scattered object.
    Scatter,
    /// A step of the **lane-serial region** ([`SerialRegion`]): run once per lane, every serial
    /// step of lane `k` before any of lane `k + 1`'s, so the region's accesses happen in exactly the
    /// scalar loop's order. `contiguous` marks an affine access, whose lane `k` address is the
    /// group's address plus `k` elements (`dir·k` in a countdown); every other step reads its
    /// operands lane by lane.
    Serial { contiguous: bool },
    /// A `Load` whose address is affine unit-stride: becomes one `Vec` load per part.
    VecLoad,
    /// A `Store` whose address is affine unit-stride: becomes one `Vec` store per part.
    VecStore,
    /// A member of an interleaved load group (`x[k·i + r]`): the first member in program order
    /// loads the group's whole window — `k` contiguous vectors per part — and splits every member
    /// out with lane shuffles; each member then reads its own lanes.
    GroupLoad { group: usize, offset: u32 },
    /// A member of an interleaved store group: each member records its value, and the last one
    /// interleaves all `k` with lane shuffles and stores the window contiguously.
    GroupStore { group: usize, offset: u32 },
    /// Part of a reduction's accumulate — its combining instruction, a min/max's compare, or a
    /// guarded accumulate's merge. Not emitted inline — every accumulate is folded at the end of the
    /// vector body ([`emit_reduction_fold`], or a vector accumulator's per-part combine).
    Reduce,
    /// An if-converted store: one `store` of a lane-wise `select` between the two arms' values.
    MaskedStore,
    /// An if-converted latch parameter: a lane-wise `select` between the two arms' arguments.
    Merge,
    /// A load only one arm makes ([`LinInst::PredLoad`]): one scalar load per lane, from the lane's
    /// own element where the lane took the arm and from a private scratch slot where it did not.
    PredLoad,
    /// A store only one arm makes ([`LinInst::PredStore`]): one scalar store per lane, to the lane's
    /// own element where the lane took the arm and to the scratch slot where it did not.
    PredStore,
    /// The stores of a nested if-converted region, every path storing ([`LinInst::TreeStore`]):
    /// one vector store per part of a tree of lane-wise `select`s.
    TreeStore,
    /// The same with a path that stores nothing: per lane, the tree's value to the lane's own
    /// element where the lane's path stores and to the scratch slot where it does not.
    TreePredStore,
}

/// One step of the loop body after **if-conversion**: the body's control flow is flattened into a
/// single straight line, with the branch's condition becoming a lane mask.
///
/// A conditional body is the third of the four shapes this pass exists to fix, and it cannot be
/// widened as a CFG — lane 3 might take the `then` arm while lane 0 takes the `else`. Flattening
/// executes *both* arms and selects per lane, which is only sound if executing the arm a lane did
/// not take is harmless. Two things could make it harmful, and both are checked in
/// [`linearize_region`]: a store the other arm does not make (it would write memory the scalar loop
/// leaves alone), and a load the other arm does not make (it would read an address the scalar loop
/// never touches, which in a language with unchecked slice indexing can be off the end of a
/// shorter buffer).
#[derive(Clone, Debug)]
enum LinInst {
    /// Reproduce `blocks[block].insts[idx]` unchanged.
    Real { block: BlockId, idx: usize },
    /// Both arms store to this address: one `store(ptr, select(cond, then_val, else_val))`.
    /// `ptr` is the `then` arm's pointer; the `else` arm's names the same address.
    MaskedStore {
        ptr: ValueId,
        cond: ValueId,
        then_val: ValueId,
        else_val: ValueId,
        /// The `then` arm's store, for its `AddrForm`.
        at: (BlockId, usize),
    },
    /// A join block parameter: `param = select(cond, then_arg, else_arg)`.
    Merge {
        param: ValueId,
        cond: ValueId,
        then_arg: ValueId,
        else_arg: ValueId,
    },
    /// `blocks[block].insts[idx]`, a load only the path where every `(cond, polarity)` of `pred`
    /// holds makes, at an address no unconditional access covers — one literal for an arm of a
    /// diamond, one per enclosing branch for a nested arm. Predicated per lane rather than
    /// speculated: a lane whose path does not reach it reads a scratch slot, never the element
    /// ([`Plan::PredLoad`]).
    PredLoad {
        block: BlockId,
        idx: usize,
        pred: Vec<(ValueId, bool)>,
    },
    /// Every store an if-converted *nested* region makes ([`linearize_tree`]), all to one address:
    /// what each lane's path leaves in it, as a tree over the region's branch conditions. Sunk to
    /// the end of the region like the diamond's merged stores ([`Plan::TreeStore`],
    /// [`Plan::TreePredStore`]).
    TreeStore {
        ptr: ValueId,
        tree: StoreTree,
        /// The first of the stores, for its `AddrForm`.
        at: (BlockId, usize),
    },
    /// A store only the arm taken when `cond == on_true` makes (the other arm stores nothing): per
    /// lane, the element is written only where the lane took the arm ([`Plan::PredStore`]).
    PredStore {
        ptr: ValueId,
        value: ValueId,
        cond: ValueId,
        on_true: bool,
        /// The store itself, for its `AddrForm`.
        at: (BlockId, usize),
    },
}

/// What an if-converted region leaves in one memory location, per lane: the value of the last store
/// on the lane's path, or nothing. Built by [`linearize_tree`] from the region's branches, so every
/// `cond` is a branch condition of the loop body and selecting by it reproduces the path.
#[derive(Clone, Debug, PartialEq, Eq)]
enum StoreTree {
    /// No store on this path: the location keeps what it held.
    Keep,
    /// This path's last store writes this value.
    Val(ValueId),
    /// `cond ? then : els`.
    Node {
        cond: ValueId,
        then: Box<StoreTree>,
        els: Box<StoreTree>,
    },
}

impl StoreTree {
    fn node(cond: ValueId, then: StoreTree, els: StoreTree) -> StoreTree {
        if then == StoreTree::Keep && els == StoreTree::Keep {
            return StoreTree::Keep;
        }
        StoreTree::Node {
            cond,
            then: Box::new(then),
            els: Box::new(els),
        }
    }

    /// `self`, then `later` on the same path: `later`'s store wins wherever it stores.
    fn followed_by(self, later: StoreTree) -> StoreTree {
        match later {
            StoreTree::Keep => self,
            StoreTree::Val(v) => StoreTree::Val(v),
            StoreTree::Node { cond, then, els } => StoreTree::node(
                cond,
                self.clone().followed_by(*then),
                self.followed_by(*els),
            ),
        }
    }

    /// Does every path store?
    fn complete(&self) -> bool {
        match self {
            StoreTree::Keep => false,
            StoreTree::Val(_) => true,
            StoreTree::Node { then, els, .. } => then.complete() && els.complete(),
        }
    }

    /// Every condition, then every stored value, in tree order.
    fn operands(&self, conds: &mut Vec<ValueId>, vals: &mut Vec<ValueId>) {
        match self {
            StoreTree::Keep => {}
            StoreTree::Val(v) => vals.push(*v),
            StoreTree::Node { cond, then, els } => {
                conds.push(*cond);
                then.operands(conds, vals);
                els.operands(conds, vals);
            }
        }
    }
}

/// A slice bounds check inside an if-converted arm (`if c { o[i] = v }` over a checked `[]T`):
/// `iv + off <u len` or the program aborts. The vector loop drops the check and runs only when
/// every index of its own iterations passes it ([`emit_len_checks`]), so no lane of it can be the
/// one that would have aborted; otherwise `n_vec = 0` and the scalar loop checks, and aborts, at
/// exactly the iteration it always did.
#[derive(Clone, Copy, Debug)]
struct LenCheck {
    off: i64,
    len: LenSrc,
}

/// Where a [`LenCheck`]'s length comes from, both readable in the guard block.
#[derive(Clone, Copy, Debug)]
enum LenSrc {
    /// A value defined outside the loop.
    Value(ValueId),
    /// `load i64 at`, `at` defined outside the loop — a fat pointer's length field re-read in the
    /// arm.
    Load(ValueId),
    /// The length field of the fat pointer at `hdr` (`load i64 (hdr + 8)`), `hdr` defined outside
    /// the loop and never faulting — the slice a loaded element belongs to ([`spine_verdict`]).
    Header(ValueId),
}

/// The **lane-serial region** of a body: the steps `start ..= end` of the flattened body, from the
/// first to the last access `loop_info::dependence` listed ([`loop_info::DepSummary::serial`]) — a
/// gather or scatter that may touch what another access of the loop touches in another iteration,
/// and that access: a read-modify-write `h[q[i]] += v`, two scatters into one object, a gather
/// beside a store to an object it may read.
///
/// No vector order keeps such a pair apart, but one order is exactly the scalar loop's: lane by
/// lane. Every memory access of the region and every step that reads a value the region computed
/// is [`Plan::Serial`] — run for lane 0, then lane 1, ... — so the listed accesses, and every
/// access between them, happen in iteration order and, within an iteration, in program order. The
/// region's other steps (pure ones that read nothing it computed: address arithmetic, the
/// arithmetic feeding the stored value) keep their plans and run first, ahead of every lane — a
/// pure step reorders nothing. Everything else in the loop is independent of the region's
/// accesses in any other iteration (`dependence` checked every pair with at most one listed
/// access as usual), so widening it around the region keeps the loop's answer. With repeated
/// indices — two lanes of a group hitting one histogram bin — lane `k + 1` reads what lane `k`
/// wrote, exactly as iteration `i + k + 1` reads what iteration `i + k` wrote.
#[derive(Clone, Copy, Debug)]
struct SerialRegion {
    start: usize,
    end: usize,
}

impl SerialRegion {
    fn contains(&self, at: usize) -> bool {
        (self.start..=self.end).contains(&at)
    }
}

/// Everything [`apply`] needs, decided before any mutation.
struct VecPlan {
    header: BlockId,
    /// The block whose terminator carries the back edge — the latch, and the join of an
    /// if-converted region.
    body: BlockId,
    /// The loop body, flattened to a straight line.
    lin: Vec<LinInst>,
    preheader: BlockId,
    /// Header parameter index of the primary induction variable.
    iv_index: usize,
    iv_ty: MirType,
    /// The induction variable's step, `1` or `-1`. Lane `k` of a group is always the group's
    /// `k`-th iteration — counter `i + dir·k` — so in a countdown the lanes run DOWN through
    /// memory: a contiguous access loads the `W` elements ending at lane 0's and reverses them
    /// with one shuffle per part ([`reverse_lanes`]), and every fold, merge and predicated access
    /// sees lanes in iteration order exactly as in an ascending loop.
    dir: i64,
    /// How to advance the trip: either a compile-time vector count, or `end` for `end - start`.
    trip: TripPlan,
    /// The group width: scalar iterations per vector iteration, `16 / narrowest lane bytes`.
    w: u32,
    /// One entry per element of `lin`, in order.
    plans: Vec<Plan>,
    /// Values in `body` that vary per lane — widened or replicated — and their lane type.
    vector_ty: FxHashMap<u32, MirType>,
    /// Reductions to fold, in `NaturalLoop::reductions` order.
    reductions: Vec<RedPlan>,
    /// The interleaved access groups [`Plan::GroupLoad`]/[`Plan::GroupStore`] index.
    groups: Vec<InterleaveGroup>,
    /// Runtime pointer-range checks the vector loop is only legal under.
    guards: Vec<GuardPair>,
    /// Runtime checks that each loop-invariant data load's address lies outside each stored range
    /// (see [`plan_guards`]); conjoined with `guards`.
    point_guards: Vec<PointGuard>,
    /// May two guarded bases be *equal* (rather than disjoint)? Only when every guarded access
    /// sits at the same constant offset from its base and moves in the same element type, so an
    /// equal pair is at distance 0.
    guard_allows_equal: bool,
    /// Where the body reads the induction variable as a **value** rather than as an address.
    ramp: IvRamp,
    /// Is the lane-ramp vector needed? Not when every step reading the counter as a value is
    /// replicated: those take `i + k` per lane instead.
    ramp_vector: bool,
    /// The cost model's estimate for one group, `(vector, scalar)` ([`estimate_cost`]). Reported
    /// by `WUKONG_VEC_TRACE`; a plan only exists when `vector <= scalar` ([`cost_declines`]).
    cost: (u64, u64),
    /// A search loop's early exit, tested once per group ([`EarlyExit`]).
    early: Option<EarlyExit>,
    /// Bounds checks inside if-converted arms, or on `iv + k` in a checked body, discharged once in
    /// the guard ([`LenCheck`]).
    len_checks: Vec<LenCheck>,
    /// A checked body's other checks ([`linearize_spine`]): each tested over the group's lanes at
    /// its place in the body, leaving for the scalar loop when any lane fails.
    spine_exits: Vec<EarlyExit>,
    /// Is the body a checked spine ([`checked_spine`])? Then an iteration may stop at a check
    /// before it reads anything, so the guard reads only what cannot fault ([`GuardReads`]).
    spine: bool,
    /// The lane-serial region, when the dependence verdict needs one ([`SerialRegion`]).
    region: Option<SerialRegion>,
}

/// The early exit of a **search loop**: `while i < n { <test>; if !c { break } <rest>; i = i + 1 }`
/// — one exit tested in the header, and one more from the body block that runs first.
///
/// A group cannot stop half way through, so it does not try to. The body up to the exit test is
/// widened, the test goes lane-wide, and if *any* lane would leave, the vector loop hands the
/// original loop the group's first iteration — the scalar loop then re-runs at most `W` iterations
/// and leaves at exactly the iteration, with exactly the values, it always did (LLVM 19 / GCC 14
/// early-break vectorization, resolved in the scalar epilogue). Only when every lane continues does
/// the rest of the body run, so the group's stores and accumulates happen exactly when the scalar
/// loop's would. Two things make running the test ahead of the scalar loop harmless, and
/// [`plan_loop`] proves both: nothing before the exit stores, and every load before the exit reads
/// memory that exists for the whole trip ([`speculation_safe`]) — lanes past the exiting one read
/// elements the scalar loop never reaches.
#[derive(Clone, Copy, Debug)]
struct EarlyExit {
    /// The branch condition of the exiting block.
    cond: ValueId,
    /// Does the loop keep going when `cond` is true? (`LoopCanon` makes it so, but the other
    /// polarity costs nothing to support.)
    continue_on_true: bool,
    /// The index in [`VecPlan::lin`] of the first step after the test: the exiting block's steps
    /// come first, the latch's after.
    split: usize,
}

/// The induction variable's second, lane-varying face.
///
/// The IV is normally [`Wide::Uniform`]: one scalar for the whole group, because the only thing it
/// does is address memory and a vector load at `base + i` covers `i .. i+W-1` on its own. A body
/// that reads it as a *value* — `if c == target`, `(i as f32)`, `o[i] = i` — needs the opposite:
/// lane `k` must see `i + k`, the counter scalar iteration `i + k` would have had. That vector is
/// `splat(i) + iota` (the step is 1 — or, in a countdown, `-1`, and the ramp `splat(i) - iota` —
/// both enforced by [`iv_verdict`]), and `iota` is exactly why
/// [`wukong_mir::Op::Iota`] exists — no expression over splatted scalars is lane-varying. When the
/// counter is wider than the group's narrowest lane the ramp is split into parts like any other
/// value, part `p` being `splat(i + p·L) + iota`.
///
/// The two faces coexist. The *same* `v533` is a `gep` index (uniform) and a compare operand
/// (ramp) in the same body, so the role is decided per instruction, not per value:
/// [`classify_iv_uses`] records which steps of the flattened body read it as a value, and
/// [`emit_body`] swaps the map entry for exactly those steps. A value-reading step that is
/// replicated ([`Plan::Replicate`]) takes `i + k` as `W` scalars instead, and when every value use
/// is one of those no ramp vector is built at all ([`VecPlan::ramp_vector`]). The address face must
/// never meet a lane-varying operand — [`plan_body`] declines a step where it would.
///
/// Only the induction variable itself gets a ramp. A value *derived* from it (`i + 1`, `i * 2`,
/// `sext i`) used as a value is still declined: its ramp would be the same arithmetic over
/// `ramp(i)` — expressible now for every op the address closure admits, the `sext` through
/// `Op::Widen` and the `shl` through the vector-by-scalar shift — but the closure classifies those
/// values as addresses, and splitting one value into an address role and a value role at once is a
/// separate change from this one, which is worse to get half right than to decline.
#[derive(Clone, Debug)]
struct IvRamp {
    /// The induction variable's `ValueId` in the *original* loop.
    iv: ValueId,
    /// Its integer type; the ramp is `Vec(ty, W)`.
    ty: MirType,
    /// Indices into [`VecPlan::lin`] whose instruction reads `iv` in a value role.
    value_use: FxHashSet<usize>,
}

impl IvRamp {
    /// Does step `at` of the flattened body read `v` as a lane-varying value?
    fn reads_as_value(&self, at: usize, v: ValueId) -> bool {
        v == self.iv && self.value_use.contains(&at)
    }
}

#[derive(Clone, Debug)]
enum TripPlan {
    /// The loop runs a known number of times; `vec` is that count rounded down to a multiple of W.
    Const(u64),
    /// `end - start`, rounded down to a multiple of W, and zero unless `start < end` under the
    /// header test's own signedness (`signed`) — see [`emit_trip`] for why the order test is not
    /// implied by the subtraction.
    Runtime { end: ValueId, signed: bool },
    /// A countdown's `start - end` (`+ 1` when `inclusive`, the header test being `iv >= end`),
    /// rounded down to a multiple of W, and zero unless the loop runs at all.
    RuntimeDown {
        end: ValueId,
        signed: bool,
        inclusive: bool,
    },
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
    /// A min/max: the compare-and-select the scalar loop accumulates with, re-emitted per lane (or
    /// per part, for a vector accumulator) with the accumulator in the same two positions.
    select: Option<loop_info::SelectCombine>,
    /// A guarded accumulate (`if c { s = s + x }`): the if-converted condition — a lane mask — and
    /// whether the update is the value taken when it is true. Each lane keeps its update only where
    /// its own condition chose it: `c ? acc ⊕ x : acc`, exactly the scalar merge.
    guard: Option<(ValueId, bool)>,
    ty: MirType,
    /// May the partials be kept in `W` independent lanes and folded once, after the loop?
    ///
    /// Only for an operator that really is associative — integer `add`/`mul`/bitwise/min/max. A float
    /// `add` is not, and giving it lane-parallel partials would change the program's answer, so it
    /// keeps a scalar accumulator and folds `W` lanes into it inside the body every iteration.
    /// That is a `W`-long dependent chain per group either way, which is exactly what `gcc -O3`
    /// emits without `-ffast-math`.
    vector_acc: bool,
    /// An argmax/argmin (`if x > m { m = x; at = i; }`): the recorded index's header parameter
    /// index and type. Always with [`Self::vector_acc`]: the pair is kept as per-lane `(value,
    /// index)` partials — parts of `ty` for the value, then parts of the index type — and combined
    /// once on the way out by [`emit_arg_combine`].
    arg: Option<(usize, MirType)>,
    /// The argmax's compare is non-strict (`>=`/`<=`), so a tie moves the index forward and the
    /// LAST maximal element wins. Its index partials start at the vector loop's first counter minus
    /// one — "never updated" — rather than at the scalar loop's own index, because with a non-strict
    /// compare an updated lane can tie an un-updated one, and only the sentinel tells them apart
    /// ([`emit_arg_combine`]).
    arg_last: bool,
    /// A FindLastIV (`if c { last = i }`, [`RedKind::FindLast`]): per-lane partials holding the
    /// last counter each lane recorded, started one below the vector loop's first counter so that
    /// "never recorded" is distinguishable from every real one; [`emit_find_last_combine`] folds
    /// them on the way out. Always with [`Self::vector_acc`] and a [`Self::guard`].
    find_last: bool,
}

/// One pair of bases the vector loop is only legal over if they are equal or disjoint.
#[derive(Clone, Debug)]
struct GuardPair {
    a: GuardBase,
    b: GuardBase,
}

/// A loop-invariant data load — one element of type `ty` at the fixed address `at` — that the
/// vector loop is only legal over if no store in `store`'s range can write it. Never satisfied by
/// equality: the point IS one element, so a store range that starts at it overlaps it.
#[derive(Clone, Debug)]
struct PointGuard {
    at: MemBase,
    ty: MirType,
    store: GuardBase,
}

/// A base pointer, plus the element-offset window the loop applies to it: a loop-invariant
/// symbolic part shared by every access in the group, and the smallest and largest constant on top
/// of it.
///
/// The symbolic part is what makes an inner loop over a *row* guardable. `x[r*C + j]` for
/// `j in 0..C` touches `[x + r*C, x + r*C + C)`, and `r*C` is a value, not a constant; a guard that
/// could only express a constant offset had to decline every row-wise inner loop in the corpus.
#[derive(Clone, Debug)]
struct GuardBase {
    base: MemBase,
    /// `(value, multiplier)` addends of the index that do not move with the induction variable —
    /// the row base `r*C` of an `a[r*C + c]` inner loop being the one that matters — exactly as
    /// [`crate::loop_info::AffineExpr::terms`] spells them. Empty for a flat `a[i + k]`. Every
    /// value here is defined **outside** the loop and its multiplier fits the induction variable's
    /// type (both checked in [`plan_guards`]), so the guard block can re-materialize the offset.
    terms: Vec<(ValueId, i64)>,
    lo: i64,
    hi: i64,
    /// The element type every index in the group counts in. Per group, because one loop may read
    /// `i8` and write `f32`: a range measured in the wrong element size would test the wrong bytes.
    elem: MirType,
    /// The constant per-iteration element stride every access in the group shares: `1` for a plain
    /// stream, `k` for an interleaved access group.
    stride: i64,
}

/// A complete **interleaved access group**: `k` accesses to one object at `k·i + c₀ + r`, one for
/// every `r` in `0..k`, all loads or all stores, moving one `elem` each. Together they touch the
/// `k·W` elements from `k·i + c₀` on contiguously — so a group is `k` ordinary vector accesses per
/// part plus lane shuffles, LLVM's InterleavedAccess lowering. RoPE's `(x[2i], x[2i+1])` pairs, a
/// complex multiply's real and imaginary parts and an RGB pixel's three channels are all one.
///
/// A group must be complete. A gap — only `x[2i]` read — would load elements the scalar loop never
/// reads, and the last group's window can run past the end of the buffer.
#[derive(Clone, Debug)]
struct InterleaveGroup {
    /// The stride, which is also the number of members.
    k: u32,
    /// The element type every member moves.
    elem: MirType,
}

/// The widest interleave group this pass forms. A wider one costs `k - 1` shuffles per member per
/// part; past four that is no longer a clear win over the scalar loop it replaces.
const MAX_INTERLEAVE: i64 = 4;

/// An access's position in its block, the key [`NaturalLoop::accesses`] entries are found by.
type AccessKey = (u32, usize);

// ---------------------------------------------------------------------------------------------
// Legality
// ---------------------------------------------------------------------------------------------

/// The first loop of `f` this pass can widen, innermost first.
///
/// `forest` arrives carrying **structure only** ([`loop_info::structure_with`]); the value analyses
/// are run here, one loop at a time and only as far as the next decision needs. That is not a
/// micro-optimization: the full analysis builds an affine expression for every integer instruction
/// of every loop in the function, this pass is re-entered by the optimizer's fixpoint, and over
/// `tests/run` it made `vectorize` 37-42% of the whole optimizer. The staging is safe because
/// [`crate::loop_info::Stages`]'s unrun stages leave *conservative* defaults — `dep` reads
/// `MemDep::Carried`, so a loop whose memory was never analyzed cannot be widened by accident —
/// and because both early filters call the same code `plan_loop` does.
fn pick(
    f: &Function,
    forest: &mut LoopForest,
    done: &DoneSets,
    kernels: bool,
    effects: Option<&CallEffects>,
    (idom, preds): (&[u32], &[Vec<u32>]),
) -> Option<Picked> {
    // `WUKONG_VEC_TRACE=1` prints one line per decision. Note that this whole function re-runs on
    // every fixpoint iteration of the pass manager, so a DECLINE repeats once per iteration while a
    // WIDEN appears once — see the module docs before quoting the stream anywhere.
    let trace = std::env::var_os("WUKONG_VEC_TRACE").is_some();
    let lane_serial = lane_serial_enabled();
    let decline = |header: u32, why: &str| {
        if trace {
            eprintln!("vectorize: declined loop at bb{header}: {why}");
        }
    };
    // A two-deep nest whose inner loop carries a row's own state (a float sum, a recurrence) is
    // widened across its outer counter first, before the inner path widens the inner loop along
    // its own ([`outer`]).
    if outer::enabled() {
        for id in forest.innermost_first() {
            let l = forest.get(id);
            if l.children.len() != 1
                || done.widened.contains(&l.header.0)
                || already_vector(f, l)
                || is_someones_epilogue(f, forest, l)
            {
                continue;
            }
            let header = l.header.0;
            match outer::plan(f, forest, id.0 as usize) {
                Ok(p) => {
                    if trace {
                        eprintln!("vectorize: OUTER bb{header} w={}", p.w);
                    }
                    return Some(Picked::Outer(p));
                }
                Err(why) => {
                    if trace {
                        eprintln!("vectorize: no outer-loop plan for bb{header}: {why}");
                    }
                }
            }
        }
    }
    for id in forest.innermost_first() {
        let idx = id.0 as usize;
        // Structure first: no value analysis has run yet, and none of these checks needs one.
        {
            let l = forest.get(id);
            if done.widened.contains(&l.header.0) || already_vector(f, l) {
                continue;
            }
            if is_someones_epilogue(f, forest, l) {
                continue;
            }
            if let Err(why) = structural_verdict(f, l) {
                decline(l.header.0, why);
                continue;
            }
        }
        // Stage 1+2 only: is it a unit-stride counted loop? This rejects more of the corpus than
        // any other single check, and it costs a fraction of the memory analysis below.
        loop_info::analyze_one(f, forest, idx, loop_info::Stages::Ivs);
        {
            let l = forest.get(id);
            if let Err(why) = iv_verdict(l) {
                decline(l.header.0, why);
                continue;
            }
        }
        // Stage 3+4: accesses, carried values and dependence. Still not the derived-IV list, which
        // nothing here reads. A call the effect summaries prove side-effect free is no memory
        // access ([`plan_body`] replicates it per lane), and a gather or scatter that may meet
        // another access is listed rather than failing the verdict ([`SerialRegion`]) — unless
        // `WUKONG_NO_LANE_SERIAL` asks for the plain verdict, list and header-read exemption
        // both off ([`lane_serial_enabled`]).
        let facts = loop_info::DepFacts {
            calls: effects,
            serial: lane_serial,
        };
        loop_info::analyze_one_with(f, forest, idx, loop_info::Stages::Memory, facts);
        let l = forest.get(id);
        // A body that checks on every iteration is `bce`'s first, when it can version it into a
        // check-free copy; widening it here would leave an epilogue `bce` versions next, and a
        // fast copy of that epilogue nothing marks as one ([`checked_spine`]).
        let plan = plan_loop(f, l, effects).and_then(|p| {
            if p.spine && crate::bce::would_version(f, l, idom, preds, effects) {
                Err("a checked body bce versions first")
            } else {
                Ok(p)
            }
        });
        match plan {
            // The 256-bit form of the same plan, when the body has one ([`vec256`]). Never for a
            // kernel's own remainder, which keeps the 128-bit loop below.
            Ok(mut p)
                if kernels
                    && !done.kernels.contains(&l.header.0)
                    && vec256::has_room(f)
                    && !vec256::is_remainder(f, l) =>
            {
                match vec256::plan(f, l, &p) {
                    Ok(k) => {
                        if trace {
                            eprintln!(
                                "vectorize: KERNEL bb{} lane={:?} streams={} scalars={} ops={} guards={}",
                                p.header.0,
                                k.kernel.lane,
                                k.kernel.streams,
                                k.kernel.scalars,
                                k.kernel.ops.len(),
                                p.guards.len() + p.point_guards.len()
                            );
                        }
                        p.w = k.kernel.lane.lanes();
                        p.trip = k.trip.clone();
                        return Some(Picked::Kernel(p, k));
                    }
                    Err(why) => {
                        if trace {
                            eprintln!("vectorize: no 256-bit kernel for bb{}: {why}", p.header.0);
                        }
                        return Some(Picked::Widen(widen_traced(p, trace)));
                    }
                }
            }
            Ok(p) => return Some(Picked::Widen(widen_traced(p, trace))),
            Err(why) => decline(l.header.0, why),
        }
    }
    None
}

/// What [`pick`] chose to do with a loop.
enum Picked {
    /// Widen it to 128-bit vectors in place ([`apply`]).
    Widen(VecPlan),
    /// Call a 256-bit kernel ahead of it, its trip and width already the kernel's ([`vec256`]).
    Kernel(VecPlan, vec256::KernelPlan),
    /// Widen a two-deep nest across its outer counter ([`outer`]).
    Outer(outer::OuterPlan),
}

/// `p`, after printing its `WUKONG_VEC_TRACE` line.
fn widen_traced(p: VecPlan, trace: bool) -> VecPlan {
    if trace {
        eprintln!(
            "vectorize: WIDEN bb{} w={} lin={} guards={} reds={} cost={}/{}",
            p.header.0,
            p.w,
            p.lin.len(),
            p.guards.len() + p.point_guards.len() + p.len_checks.len(),
            p.reductions.len(),
            p.cost.0,
            p.cost.1
        );
    }
    p
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

/// The half of [`plan_loop`] that reads only the loop's **structure** — block sets, latches, exits
/// and the instructions themselves — and none of the induction-variable, access or dependence
/// analysis.
///
/// It exists so that [`pick`] can decline a loop before paying for those analyses, and it is a
/// function rather than a copy of the checks so the two cannot drift: `plan_loop` calls it, so
/// anything it rejects is exactly something `plan_loop` rejects.
///
/// Returns the loop's preheader and its latch (the body block that carries the back edge).
fn structural_verdict(f: &Function, l: &NaturalLoop) -> Result<(BlockId, BlockId), &'static str> {
    let preheader = l.preheader.ok_or("no preheader")?;
    if l.latches.len() != 1 || !l.abnormal_exits.is_empty() {
        return Err("not one latch / no abnormal exit");
    }
    if l.exits.len() != 1 {
        if early_exit_of(f, l).is_none()
            && !only_dead_end_exits(f, l)
            && checked_spine(f, l).is_none()
        {
            return Err("not one exit, nor a search loop's one early exit");
        }
    } else if l.exits[0].0 != l.header {
        return Err("exit is not tested in the header");
    }
    // The cheap half of the if-converters' precondition: a straight-line body, a diamond, or a
    // small nest of them ([`linearize_tree`]). A loop holding another loop is none of those, and
    // anything larger is not worth flattening.
    if !l.children.is_empty() {
        return Err("an outer loop");
    }
    if l.blocks.len() > MAX_REGION_BLOCKS {
        return Err("body has more blocks than an if-converted region holds");
    }
    if !header_is_test_only(f, l) {
        return Err("the header computes more than the exit test");
    }
    Ok((preheader, l.latches[0]))
}

/// The most blocks a loop may have for [`linearize_tree`] to flatten it, header included: an
/// `if / else if / else` chain of three stores, each behind a slice bounds check, is eleven.
const MAX_REGION_BLOCKS: usize = 12;

/// Does the loop leave only through its header test and dead ends — blocks outside the loop that
/// end in `unreachable`, which is how a failed bounds check stops the program? Such a loop has one
/// real exit; whether each dead end can be proved unreachable for the vector loop's iterations is
/// [`linearize_tree`]'s question ([`LenCheck`]).
///
/// A dead end leaving from the body's unconditional spine — the blocks every iteration runs, from
/// the body's first block through jumps and passed checks up to the first real branch — is refused
/// here, before any value analysis: that check runs on every iteration, which is the shape of the
/// slow copy `bce` keeps of a versioned slice loop, a cold path that widening would only make
/// bigger (see [`early_exit_of`]). Only a check inside an arm is [`linearize_tree`]'s to discharge.
fn only_dead_end_exits(f: &Function, l: &NaturalLoop) -> bool {
    let dead =
        |b: BlockId| !l.blocks.contains(&b) && matches!(f.block(b).term, Terminator::Unreachable);
    let inside: Vec<BlockId> = succs(&f.block(l.header).term)
        .into_iter()
        .filter(|b| l.blocks.contains(b))
        .collect();
    let [entry] = inside.as_slice() else {
        return false;
    };
    let mut spine = vec![*entry];
    let mut cur = *entry;
    while spine.len() <= l.blocks.len() {
        let next = match &f.block(cur).term {
            Terminator::Br { target, .. } if *target != l.header => *target,
            Terminator::CondBr {
                then_blk, else_blk, ..
            } if dead(*else_blk) => *then_blk,
            Terminator::CondBr {
                then_blk, else_blk, ..
            } if dead(*then_blk) => *else_blk,
            _ => break,
        };
        spine.push(next);
        cur = next;
    }
    l.exits.iter().filter(|(b, _)| *b == l.header).count() == 1
        && l.exits
            .iter()
            .all(|&(b, out)| b == l.header || (!spine.contains(&b) && dead(out)))
}

/// Is this a unit-stride counted loop over an integer induction variable?
///
/// Everything here is answered by [`crate::loop_info::Stages::Ivs`], the cheapest analysis stage —
/// and it is the largest single reason the corpus declines (a step that is not 1 accounts for more
/// declines over `tests/run` than any other check). `plan_loop` repeats it, so this can only
/// decline loops `plan_loop` would have declined.
///
/// A step of `-1` — a countdown, `while i > 0 { i = i - 1; .. }` or `while i >= 0 { ..; i = i - 1 }`
/// — is accepted too: the group then walks down, lane `k` still being the group's `k`-th iteration
/// ([`VecPlan::dir`]). `WUKONG_NO_REVERSE_VEC=1` restores the decline.
fn iv_verdict(l: &NaturalLoop) -> Result<(), &'static str> {
    let idx = l.primary_iv.ok_or("no primary induction variable")?;
    let iv = &l.ivs[idx];
    let reverse_ok = std::env::var_os("WUKONG_NO_REVERSE_VEC").is_none();
    if iv.step != IvStep::Const(1) && !(reverse_ok && iv.step == IvStep::Const(-1)) {
        return Err("induction step is not 1");
    }
    if !iv.ty.is_int() || iv.ty == MirType::I1 {
        return Err("induction variable is not an integer");
    }
    Ok(())
}

/// Would this pass widen `l` as it stands? The same verdict [`pick`] reaches for a loop it has not
/// widened yet, for a caller that must not trade a widenable loop away (`fuse`'s profit rule).
/// `l` must be analyzed to at least [`loop_info::Stages::Memory`]; less reads as "no", because the
/// unrun stages' defaults are the conservative answers.
pub(crate) fn would_widen(f: &Function, l: &NaturalLoop) -> bool {
    plan_loop(f, l, None).is_ok()
}

fn plan_loop(
    f: &Function,
    l: &NaturalLoop,
    effects: Option<&CallEffects>,
) -> Result<VecPlan, &'static str> {
    // ---- structure -------------------------------------------------------------------------
    let (preheader, body) = structural_verdict(f, l)?;
    // The body is either one straight-line block (which is then also the latch), or a diamond /
    // triangle that if-conversion flattens into one, or a nest of them ([`linearize_tree`]).
    // Either way the latch carries the back edge. A search loop's body is its exiting block
    // followed by the latch, the test between them.
    iv_verdict(l)?;
    let iv_value = l.ivs[l
        .primary_iv
        .expect("iv_verdict accepted a primary induction variable")]
    .value;
    let mut len_checks: Vec<LenCheck> = Vec::new();
    let mut scalar_path: Option<u64> = None;
    // A body whose every iteration passes checks to dead ends ([`checked_spine`]): the checks on
    // `iv + k` are discharged in the guard, every other one is a group test ([`linearize_spine`]).
    let mut spine_exits: Vec<EarlyExit> = Vec::new();
    let mut spine = false;
    let (lin, early) = match early_exit_of(f, l) {
        Some((exiting, cond, continue_on_true)) => {
            let first = f.blocks[exiting.0 as usize].insts.len();
            let lin: Vec<LinInst> = (0..first)
                .map(|idx| LinInst::Real {
                    block: exiting,
                    idx,
                })
                .chain(
                    (0..f.blocks[body.0 as usize].insts.len())
                        .map(|idx| LinInst::Real { block: body, idx }),
                )
                .collect();
            let early = EarlyExit {
                cond,
                continue_on_true,
                split: first,
            };
            (lin, Some(early))
        }
        None => match checked_spine(f, l) {
            Some(chain) => {
                let s = linearize_spine(f, l, &chain, iv_value);
                len_checks = s.len_checks;
                spine_exits = s.exits;
                spine = true;
                (s.lin, None)
            }
            // The single diamond first, exactly as before; a nest, a chain of `else if`s or an arm
            // holding a bounds check falls through to the general region flattener.
            None => match linearize_region(f, l, body) {
                Ok(lin) if l.exits.len() == 1 => (lin, None),
                _ => {
                    let t = linearize_tree(f, l, body, iv_value)?;
                    len_checks = t.len_checks;
                    scalar_path = Some(t.scalar_path);
                    (t.lin, None)
                }
            },
        },
    };

    // ---- induction variable and trip count ---------------------------------------------------
    let iv = &l.ivs[l
        .primary_iv
        .expect("iv_verdict accepted a primary induction variable")];
    let iv_index = iv.param_index;
    let iv_ty = iv.ty.clone();
    let dir = iv
        .step
        .as_const()
        .expect("iv_verdict accepted a constant step");
    // A countdown widens the plain shapes: contiguous accesses (reversed per part), if-conversion,
    // replication and the order-free and serial reductions. What reads "later" as "a larger
    // counter" — a search's header bound, an arm's bounds check, an argmax's or FindLastIV's
    // iteration order — and the interleave windows stay with ascending loops.
    if dir == -1 {
        if early.is_some() {
            return Err("a countdown search loop");
        }
        if !len_checks.is_empty() {
            return Err("a bounds check in a countdown's arm");
        }
        if spine {
            return Err("a bounds check in a countdown");
        }
        if l.reductions
            .iter()
            .any(|r| r.arg.is_some() || r.kind == RedKind::FindLast)
        {
            return Err("an index-recording reduction in a countdown");
        }
    }

    // ---- dependence ---------------------------------------------------------------------------
    // A gather or scatter that may meet another access is not a decline any more, but a region
    // the vector body runs lane by lane ([`SerialRegion`]); every other pair still has to be
    // independent.
    let region = serial_region(l, &lin)?;
    let shape = l.is_vectorizable_shape()
        || (region.is_some()
            && !matches!(l.dep.memory, MemDep::Carried)
            && l.carried
                .iter()
                .all(|c| !matches!(c.kind, Carried::Recurrence)));
    if !shape {
        return Err("loop-carried dependence");
    }
    if let Some(r) = region {
        if early.is_some() {
            return Err("a lane-serial region in a search loop");
        }
        // The group tests run before anything stores ([`spine_verdict`]), and the region stores.
        if spine_exits.iter().any(|x| x.split > r.start) {
            return Err("a lane test inside a lane-serial region");
        }
    }
    if !carried_values_are_handled(l, iv.param_index) {
        return Err("a carried value is neither the primary IV, a reduction nor invariant");
    }
    // An argmax's index merge records the counter itself; its fold reads the lane ramp directly.
    let arg_merges: FxHashSet<u32> = l
        .reductions
        .iter()
        .filter_map(|r| r.arg)
        .filter(|a| a.index == iv.value)
        .map(|a| a.merge.0)
        .chain(
            l.reductions
                .iter()
                .filter(|r| r.kind == RedKind::FindLast && r.combine == iv.value)
                .filter_map(|r| r.merge.map(|m| m.0)),
        )
        .collect();
    let iv_value_use = classify_iv_uses(f, &lin, iv.value, &arg_merges)
        .ok_or("the induction variable is used for something other than addressing")?;
    let (groups, member) = interleave_groups(l)?;
    if dir == -1 && !groups.is_empty() {
        return Err("an interleave group in a countdown");
    }
    access_shape_verdict(l, &member)?;
    group_order_verdict(l, &lin, groups.len(), &member)?;

    // ---- per-instruction plan -----------------------------------------------------------------
    let ramp = IvRamp {
        iv: iv.value,
        ty: iv_ty.clone(),
        value_use: iv_value_use,
    };
    let BodyPlan {
        plans,
        vector_ty,
        reductions,
        lane_vals,
        packed,
        ramp_vector,
        replicated,
        calls,
    } = plan_body(f, l, &lin, &ramp, &member, effects, region)
        .ok_or("an instruction in the body cannot be widened")?;
    // Steps before a search loop's exit test, or before a checked body's last group test, run for
    // lanes past the one that leaves.
    let speculative = early
        .iter()
        .chain(&spine_exits)
        .map(|x| x.split)
        .max()
        .unwrap_or(0);
    call_verdict(&calls, &plans, speculative)?;

    // ---- replication: only beside at least as much widened work ------------------------------
    // A replicated step keeps all of its scalar work and adds a lane extract per operand and a
    // lane insert to pack its result, so it only pays for itself next to arithmetic that really
    // does run `W` lanes at once. Loads and stores are not counted as that arithmetic: a body that
    // only moves data through one replicated step (`o[i] = x[i] as i64` from `f64`, a bare
    // `o[i] = lut[q[i]]`, or a bare permutation `o[q[i]] = x[i]`) would pay the packing for nothing,
    // and stays scalar exactly as it did before replication. A scatter counts like a gather. A
    // heuristic, not a measurement — it errs toward the scalar loop.
    let widened = plans
        .iter()
        .filter(|p| {
            matches!(
                p,
                Plan::Widen | Plan::Cast(_) | Plan::MaskedStore | Plan::Merge | Plan::TreeStore
            )
        })
        .count()
        + reductions.iter().filter(|r| r.vector_acc).count();
    if replicated > widened {
        return Err("most of the body would run lane by lane");
    }

    // ---- width, from the narrowest lane the widened body holds --------------------------------
    // Every value that becomes a vector (a replicated one only where it is packed), every affine
    // access (a uniform stored value is splatted to its own type), the lane ramp and every
    // reduction. `W` covers one full vector of the narrowest; each wider value is split into parts,
    // at most `MAX_PARTS` of them.
    let mut narrowest = u32::MAX;
    let mut widest = 0;
    let affine_tys = l
        .accesses
        .iter()
        .filter(|a| matches!(a.addr, AddrForm::Affine { .. }))
        .map(|a| &a.value_ty);
    let ramp_ty = ramp_vector.then_some(&ramp.ty);
    for t in vector_ty
        .iter()
        .filter(|(v, _)| !lane_vals.contains(*v))
        .map(|(_, t)| t)
        .chain(packed.iter())
        .chain(affine_tys)
        .chain(ramp_ty)
        .chain(reductions.iter().map(|r| &r.ty))
    {
        let b = lane_bytes(t).ok_or("a widened value has no vector lane type")?;
        narrowest = narrowest.min(b);
        widest = widest.max(b);
    }
    if widest == 0 {
        return Err("nothing in the body becomes a vector");
    }
    if widest / narrowest > MAX_PARTS {
        return Err("the body's lane widths would split one value across more than four vectors");
    }
    let w = VEC_BYTES / narrowest;

    // ---- trip count ---------------------------------------------------------------------------
    let trip = match &l.trip {
        // A search loop has two exits, so the analysis reports no trip count; the header's test
        // still bounds it, and the vector loop runs only whole groups below that bound. The same
        // for a loop whose other exits are bounds checks' dead ends.
        _ if early.is_some() || l.exits.len() != 1 => early_trip(f, l, iv.value, iv.start, w)?,
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
            signed,
            ..
        } => TripPlan::Runtime {
            end: *end,
            signed: *signed,
        },
        // A countdown's inclusive bound (`i >= 0`) is fine: [`emit_trip`] adds the one only when
        // the difference is below the type's maximum.
        TripCount::Affine {
            step: -1,
            inclusive,
            end,
            signed,
            ..
        } => TripPlan::RuntimeDown {
            end: *end,
            signed: *signed,
            inclusive: *inclusive,
        },
        // An ascending inclusive bound would need `end + 1`, which overflows at `end == MAX`; any
        // other step is already excluded above.
        _ => return Err("trip count is not a unit-step half-open range"),
    };

    // ---- runtime alias guard ------------------------------------------------------------------
    let (guards, point_guards, guard_allows_equal) =
        plan_guards(f, l).ok_or("the runtime alias check cannot be expressed")?;

    // ---- cost ---------------------------------------------------------------------------------
    // The guard is evaluated on every entry to the loop — once per row of a row-wise inner loop —
    // and grows with the square of the stream count, so it has a budget (LLVM's
    // `-runtime-memory-check-threshold`). The rest of the model is the per-group estimate below.
    if guards.len() + point_guards.len() > MAX_GUARD_PAIRS {
        return Err("more runtime alias checks than one loop entry amortizes");
    }
    if let Some(x) = &early {
        early_exit_verdict(f, l, &lin, &plans, x, iv.value, iv.start)?;
    }
    if !spine_exits.is_empty() {
        spine_verdict(f, l, &lin, &plans, &spine_exits, &mut len_checks, iv.start)?;
    }
    if early.is_some() || spine {
        // A search loop may leave on its first iteration, before it reaches the access an alias
        // check is about — and a checked body may stop on its first iteration, at a check before
        // it — so the guard may re-read only a fat pointer that cannot fault ([`GuardReads`]).
        let unproven = |b: MemBase| match b {
            MemBase::ReloadedOuter { addr, .. } => !never_faults(f, addr, 8),
            MemBase::Outer(_) => false,
        };
        if guards
            .iter()
            .any(|g| unproven(g.a.base) || unproven(g.b.base))
            || point_guards
                .iter()
                .any(|g| unproven(g.at) || unproven(g.store.base))
        {
            return Err("a search loop's alias check would read a pointer it may never reach");
        }
    }
    let mut cost = estimate_cost(
        f,
        &lin,
        &plans,
        &vector_ty,
        &reductions,
        body,
        w,
        scalar_path,
        dir,
    );
    // Each lane test: a reduction over the mask's lanes, an extract and a branch.
    let tests = u64::from(early.is_some()) + spine_exits.len() as u64;
    cost.0 += tests * (2 + 2 * u64::from(w.trailing_zeros()));
    if cost_declines(cost, cost_model_enforced()) {
        return Err("the vector body costs more than its W scalar iterations");
    }

    Ok(VecPlan {
        header: l.header,
        body,
        lin,
        preheader,
        iv_index,
        iv_ty,
        dir,
        trip,
        w,
        plans,
        vector_ty,
        reductions,
        groups,
        guards,
        point_guards,
        guard_allows_equal,
        ramp,
        ramp_vector,
        cost,
        early,
        len_checks,
        spine_exits,
        spine,
        region,
    })
}

/// Does the dependence analysis run for the lane-serial consumer ([`loop_info::DepFacts::serial`])?
/// Everything but `WUKONG_NO_LANE_SERIAL` says yes. The switch is a strict revert, for a same-run
/// A/B: the analysis then lists nothing (a gather or scatter that may meet another access is
/// `Carried` again, so no region forms) and takes no header-read exemption (a scatter beside a
/// slice's in-loop data-pointer read is `Carried` again), which is the verdict every loop had
/// before either existed.
fn lane_serial_enabled() -> bool {
    std::env::var_os("WUKONG_NO_LANE_SERIAL").is_none()
}

/// The lane-serial region the dependence verdict asks for, if any: the span of the flattened body
/// from the first to the last listed access ([`SerialRegion`]). Every listed access must be a step
/// of its own — one inside an if-converted arm is predicated or merged, and has no single place in
/// iteration order. Nothing is listed under `WUKONG_NO_LANE_SERIAL` ([`lane_serial_enabled`]).
fn serial_region(l: &NaturalLoop, lin: &[LinInst]) -> Result<Option<SerialRegion>, &'static str> {
    if l.dep.serial.is_empty() {
        return Ok(None);
    }
    let mut span: Option<SerialRegion> = None;
    for &k in &l.dep.serial {
        let a = &l.accesses[k];
        let at = lin
            .iter()
            .position(|s| {
                matches!(s, LinInst::Real { block, idx } if *block == a.block && *idx == a.inst)
            })
            .ok_or("an access a lane-serial region must order sits in an if-converted arm")?;
        span = Some(match span {
            None => SerialRegion { start: at, end: at },
            Some(r) => SerialRegion {
                start: r.start.min(at),
                end: r.end.max(at),
            },
        });
    }
    Ok(span)
}

/// The most runtime pointer-range checks (pairs plus point checks) one widened loop may pay on
/// entry — LLVM LoopVectorize's default `-runtime-memory-check-threshold`. Each check is a handful
/// of address computations and two compares, and the count grows with the square of the number of
/// streams; a loop entered once per row of a short row pays all of them every time.
const MAX_GUARD_PAIRS: usize = 8;

/// Does the cost model decline a loop whose one group costs `(vector, scalar)`? When the vector
/// body is estimated to cost more than the `W` scalar iterations it replaces — and the model is
/// on (`WUKONG_VEC_COST=0` turns it off, for a same-run A/B of the rule itself).
fn cost_declines((vector, scalar): (u64, u64), enforced: bool) -> bool {
    enforced && vector > scalar
}

/// Is the cost model's threshold enforced? Everything but `WUKONG_VEC_COST=0` says yes.
fn cost_model_enforced() -> bool {
    std::env::var_os("WUKONG_VEC_COST").is_none_or(|v| v != "0")
}

/// One pass through a loop's control: the counter step, the exit compare and the branch.
const LOOP_OVERHEAD: u64 = 3;

/// The cost model's estimate for one group of `w` iterations, as `(vector, scalar)`.
///
/// One unit per instruction, the default LLVM's TTI assigns the operations this pass emits: a
/// widened step costs one per part, a replicated one a scalar operation plus a lane extract and a
/// lane insert per lane, an interleave-group member a load or store and a shuffle per part, an
/// if-converted store or merge a select (and a store) per part. A vector accumulator costs one per
/// part; a serial fold an extract and a combine per lane, plus a lane test and a select when the
/// accumulate is guarded. The scalar side is `w` passes through the original body, taking the
/// longer arm of an if-converted branch, plus the loop overhead each.
///
/// It is **enforced** at its one obvious threshold ([`cost_declines`]): a group whose vector
/// operations outnumber the `W` scalar iterations it replaces is declined. That is the only cut
/// the table can make without tuning — no weight here has been fitted to a same-run release A/B
/// against `WUKONG_NO_VECTORIZE`, so the model never declines a loop it merely thinks is a small
/// win, and a lower threshold is a later, measured change. Over `tests/run` the rule declined only
/// bodies the nested if-converter had just started taking (a predicated per-lane store beside a
/// replicated remainder); every loop widened before it still widens.
#[allow(clippy::too_many_arguments)]
fn estimate_cost(
    f: &Function,
    lin: &[LinInst],
    plans: &[Plan],
    vector_ty: &FxHashMap<u32, MirType>,
    reductions: &[RedPlan],
    latch: BlockId,
    w: u32,
    scalar_path: Option<u64>,
    dir: i64,
) -> (u64, u64) {
    let w64 = u64::from(w);
    let parts = |t: &MirType| lanes_of(t).map_or(1, |l| u64::from((w / l).max(1)));
    // The lane type a step produces (or stores): the widened type where the plan recorded one —
    // a compare's mask, a merge's parameter — and the MIR type otherwise.
    let ty_of = |v: ValueId| {
        vector_ty
            .get(&v.0)
            .cloned()
            .unwrap_or_else(|| f.value_type(v).clone())
    };
    let step_ty = |s: &LinInst| -> MirType {
        match s {
            LinInst::Real { block, idx } => {
                let inst = &f.blocks[block.0 as usize].insts[*idx];
                match (&inst.op, inst.result) {
                    (Op::Store { value, .. }, _) => ty_of(*value),
                    (_, Some(r)) => ty_of(r),
                    (_, None) => MirType::I1,
                }
            }
            LinInst::MaskedStore { then_val, .. } => ty_of(*then_val),
            LinInst::Merge { param, .. } => ty_of(*param),
            LinInst::PredLoad { block, idx, .. } => {
                match f.blocks[block.0 as usize].insts[*idx].result {
                    Some(r) => ty_of(r),
                    None => MirType::I1,
                }
            }
            LinInst::PredStore { value, .. } => ty_of(*value),
            LinInst::TreeStore { tree, .. } => {
                let (mut conds, mut vals) = (Vec::new(), Vec::new());
                tree.operands(&mut conds, &mut vals);
                vals.first().map_or(MirType::I1, |v| ty_of(*v))
            }
        }
    };
    // A store tree's selects: one per branch it spans.
    let tree_nodes = |s: &LinInst| -> u64 {
        match s {
            LinInst::TreeStore { tree, .. } => {
                let (mut conds, mut vals) = (Vec::new(), Vec::new());
                tree.operands(&mut conds, &mut vals);
                conds.len() as u64
            }
            _ => 0,
        }
    };

    // Scalar: every real step once per iteration, except the shorter of two if-converted arms —
    // or, for a nested region, the instructions on its longest path ([`linearize_tree`]).
    let entry = match lin.first() {
        Some(LinInst::Real { block, .. }) => Some(block.0),
        _ => None,
    };
    let mut per_block: FxHashMap<u32, u64> = FxHashMap::default();
    for s in lin {
        if let LinInst::Real { block, .. } | LinInst::PredLoad { block, .. } = s {
            *per_block.entry(block.0).or_default() += 1;
        }
    }
    let arms: Vec<u64> = per_block
        .iter()
        .filter(|(b, _)| Some(**b) != entry && **b != latch.0)
        .map(|(_, n)| *n)
        .collect();
    let skipped = if arms.len() == 2 {
        arms.iter().copied().min().unwrap_or(0)
    } else {
        0
    };
    let total: u64 = per_block.values().sum();
    let scalar = w64 * (scalar_path.unwrap_or(total - skipped) + LOOP_OVERHEAD);

    let mut vector = LOOP_OVERHEAD;
    for (s, plan) in lin.iter().zip(plans) {
        vector += match plan {
            Plan::Scalar => 1,
            Plan::InvariantPtrLoad | Plan::Reduce => 0,
            Plan::Widen | Plan::Cast(_) | Plan::Merge => parts(&step_ty(s)),
            // A countdown reverses each part of a contiguous access with one shuffle.
            Plan::VecLoad | Plan::VecStore => parts(&step_ty(s)) * if dir < 0 { 2 } else { 1 },
            Plan::MaskedStore | Plan::GroupLoad { .. } | Plan::GroupStore { .. } => {
                2 * parts(&step_ty(s))
            }
            Plan::Replicate | Plan::LaneGep | Plan::Gather | Plan::Call => 3 * w64,
            // Per lane: the scalar step, and — amortized over the region, whose steps pass their
            // values to each other as scalars — an extract or an insert at its edges.
            Plan::Serial { .. } => 2 * w64,
            // Per lane: an extract of the stored value and the scalar store.
            Plan::Scatter => 2 * w64,
            // Per lane: the lane's condition bit, an address select and the scalar access.
            Plan::PredLoad | Plan::PredStore => 4 * w64,
            // A select per branch and a store, per part.
            Plan::TreeStore => (tree_nodes(s) + 1) * parts(&step_ty(s)),
            // Per lane: a select per branch for the bit and the value, then a predicated store.
            Plan::TreePredStore => (2 * tree_nodes(s) + 4) * w64,
        };
    }
    for r in reductions {
        let guarded = u64::from(r.guard.is_some());
        vector += if let Some((_, ity)) = &r.arg {
            // A compare and a select per value part, a select per index part.
            2 * parts(&r.ty) + parts(ity)
        } else if r.vector_acc {
            parts(&r.ty) * (1 + guarded)
        } else {
            w64 * (2 + 2 * guarded)
        };
    }
    (vector, scalar)
}

/// Flatten the loop body into one straight line, if-converting a conditional region.
///
/// Two shapes are accepted. A body of one block (which is then the latch) linearizes to itself. A
/// **diamond or triangle** — `entry` branching on a condition to a `then` and an `else` side that
/// both reach the latch — linearizes to `entry`, then both arms, then one `select` per latch
/// parameter, then the latch's own instructions.
///
/// Executing an arm a lane did not take must be harmless, and the two ways it might not be are
/// handled here:
///
/// * **A store with no counterpart.** If only the `then` arm writes `o[i]`, the flattened body
///   must not write it for every lane. A read-modify-write (`store(p, select(m, v, load(p)))`)
///   would read and rewrite memory the scalar loop never touches — a lost update against another
///   thread writing those elements — so instead each lane stores on its own, to its element where
///   it took the arm and to a private scratch slot where it did not ([`LinInst::PredStore`]): the
///   branch-free form of a masked store. A pair of stores to the same address becomes one masked
///   store; stores in both arms that do not pair are still refused.
/// * **A load with no counterpart.** Wukong does not bounds-check slice indexing, so speculating
///   `w[i]` when only one arm reads it can read off the end of a buffer that is genuinely shorter
///   than the loop's trip count. Such a load is predicated the same way — a lane that did not take
///   the arm reads the scratch slot ([`LinInst::PredLoad`]) — and a load is speculated as a plain
///   vector load only when the *same* base and index is also
///   read or written by the other arm or by the unconditional part of the body, which makes the
///   set of addresses the flattened body touches equal to the set one arm would have touched.
///
/// Integer division is refused for the same reason in a different register: `sdiv` traps on a zero
/// divisor, and a lane that would not have executed it must not fault.
fn linearize_region(
    f: &Function,
    l: &NaturalLoop,
    latch: BlockId,
) -> Result<Vec<LinInst>, &'static str> {
    let straight = |b: BlockId| -> Vec<LinInst> {
        (0..f.blocks[b.0 as usize].insts.len())
            .map(|idx| LinInst::Real { block: b, idx })
            .collect()
    };
    if l.blocks.len() == 2 {
        return Ok(straight(latch));
    }
    if l.blocks.len() > 5 {
        return Err("body has more blocks than one diamond");
    }

    // `entry` is the header's one successor inside the loop, and it must branch on a condition.
    let entry = {
        let inside: Vec<BlockId> = succs(&f.blocks[l.header.0 as usize].term)
            .into_iter()
            .filter(|b| l.blocks.contains(b))
            .collect();
        match inside.as_slice() {
            [b] if *b != latch => *b,
            _ => return Err("the header does not enter the body through one block"),
        }
    };
    let Terminator::CondBr {
        cond,
        then_blk,
        then_args,
        else_blk,
        else_args,
    } = &f.blocks[entry.0 as usize].term
    else {
        return Err("the body's entry does not branch on a condition");
    };
    if then_blk == else_blk {
        return Err("both arms of the body's branch go the same way");
    }

    // Each side is either an arm block that falls through to the latch, or the latch itself (a
    // triangle). `arm_args` is what that side passes to the latch's parameters.
    let side = |blk: BlockId,
                direct: &[ValueId]|
     -> Result<(Option<BlockId>, Vec<ValueId>), &'static str> {
        if blk == latch {
            return Ok((None, direct.to_vec()));
        }
        if !l.blocks.contains(&blk) {
            return Err("an arm of the body leaves the loop");
        }
        match &f.blocks[blk.0 as usize].term {
            Terminator::Br { target, args } if *target == latch => Ok((Some(blk), args.clone())),
            _ => Err("an arm of the body does not fall through to the latch"),
        }
    };
    let (t_blk, t_args) = side(*then_blk, then_args)?;
    let (e_blk, e_args) = side(*else_blk, else_args)?;
    if t_blk.is_none() && e_blk.is_none() {
        return Err("neither arm has a block of its own");
    }

    // ---- speculation safety --------------------------------------------------------------------
    let arm_accesses = |b: Option<BlockId>| -> Vec<&MemAccess> {
        b.map(|b| l.accesses.iter().filter(|a| a.block == b).collect())
            .unwrap_or_default()
    };
    let t_acc = arm_accesses(t_blk);
    let e_acc = arm_accesses(e_blk);
    let unconditional: Vec<&MemAccess> = l
        .accesses
        .iter()
        .filter(|a| a.block == entry || a.block == latch)
        .collect();
    let same_place = |a: &MemAccess, b: &MemAccess| match (&a.addr, &b.addr) {
        (
            AddrForm::Affine {
                base: ba,
                index: ia,
                elem: ea,
            },
            AddrForm::Affine {
                base: bb,
                index: ib,
                elem: eb,
            },
        ) => ba.same_object(*bb) && ia == ib && ea == eb,
        (AddrForm::Invariant(ba), AddrForm::Invariant(bb)) => ba.same_object(*bb),
        _ => false,
    };
    // Loads no other access covers are predicated per lane; that needs the lane's own element,
    // which an affine address gives and a gather's does not.
    let mut predicated: FxHashSet<(u32, usize)> = FxHashSet::default();
    for (mine, theirs) in [(&t_acc, &e_acc), (&e_acc, &t_acc)] {
        for a in mine.iter().filter(|a| a.kind == AccessKind::Load) {
            let covered = speculable_ptr_load(f, a)
                || theirs
                    .iter()
                    .chain(unconditional.iter())
                    .any(|b| same_place(a, b));
            if !covered {
                if !matches!(a.addr, AddrForm::Affine { .. }) {
                    return Err("an arm loads an address the other arm never touches");
                }
                predicated.insert((a.block.0, a.inst));
            }
        }
    }
    for b in [t_blk, e_blk].into_iter().flatten() {
        for inst in &f.blocks[b.0 as usize].insts {
            match &inst.op {
                Op::Bin(BinOp::SDiv | BinOp::UDiv | BinOp::SRem | BinOp::URem, ..) => {
                    return Err("an arm divides, which may fault on a lane that would not run it")
                }
                // A frame slot, or any write but a plain store (`Op::effects`): a call of any
                // kind, an atomic.
                op if unpredicable(op) => {
                    return Err("an arm has an effect that cannot be predicated")
                }
                _ => {}
            }
        }
    }

    // ---- store pairing -------------------------------------------------------------------------
    let stores_of = |b: Option<BlockId>| -> Vec<(BlockId, usize, ValueId, ValueId)> {
        let Some(b) = b else { return Vec::new() };
        f.blocks[b.0 as usize]
            .insts
            .iter()
            .enumerate()
            .filter_map(|(i, inst)| match &inst.op {
                Op::Store { ptr, value } => Some((b, i, *ptr, *value)),
                _ => None,
            })
            .collect()
    };
    let t_stores = stores_of(t_blk);
    let e_stores = stores_of(e_blk);
    let access_at = |b: BlockId, i: usize| l.accesses.iter().find(|a| a.block == b && a.inst == i);
    let mut merged_stores: Vec<LinInst> = Vec::with_capacity(t_stores.len() + e_stores.len());
    // Only one arm stores: each of its stores happens per lane, where the lane took the arm, in the
    // arm's own order.
    if t_stores.is_empty() || e_stores.is_empty() {
        for (stores, on_true) in [(&t_stores, true), (&e_stores, false)] {
            for &(b, i, ptr, value) in stores {
                let Some(a) = access_at(b, i) else {
                    return Err("a store in an arm has no analyzed address");
                };
                if !matches!(a.addr, AddrForm::Affine { .. }) {
                    return Err("an arm stores to an address that does not step with the loop");
                }
                merged_stores.push(LinInst::PredStore {
                    ptr,
                    value,
                    cond: *cond,
                    on_true,
                    at: (b, i),
                });
            }
        }
    } else if t_stores.len() != e_stores.len() {
        return Err("the two arms do not store the same number of times");
    }
    // (With one arm storing nothing, the zip below is empty.)
    for (ts, es) in t_stores.iter().zip(&e_stores) {
        let (Some(ta), Some(ea)) = (access_at(ts.0, ts.1), access_at(es.0, es.1)) else {
            return Err("a store in an arm has no analyzed address");
        };
        if !same_place(ta, ea) {
            return Err("the two arms store to different addresses");
        }
        merged_stores.push(LinInst::MaskedStore {
            ptr: ts.2,
            cond: *cond,
            then_val: ts.3,
            else_val: es.3,
            at: (ts.0, ts.1),
        });
    }

    // A merged store reads *both* arms' values, so it can only be emitted once both arms have run
    // — which means every store sinks past every load. That is order-preserving only if no arm
    // already read back through a store it made, so require loads-before-stores within each arm.
    for b in [t_blk, e_blk].into_iter().flatten() {
        let mut stored = false;
        for inst in &f.blocks[b.0 as usize].insts {
            match &inst.op {
                Op::Store { .. } => stored = true,
                Op::Load(..) if stored => return Err("an arm reads memory after writing it"),
                _ => {}
            }
        }
    }

    // ---- the flattened body --------------------------------------------------------------------
    let mut lin = straight(entry);
    for b in [t_blk, e_blk].into_iter().flatten() {
        for (idx, inst) in f.blocks[b.0 as usize].insts.iter().enumerate() {
            if matches!(inst.op, Op::Store { .. }) {
                continue; // replaced by the merged store below
            }
            if predicated.contains(&(b.0, idx)) {
                lin.push(LinInst::PredLoad {
                    block: b,
                    idx,
                    pred: vec![(*cond, Some(b) == t_blk)],
                });
            } else {
                lin.push(LinInst::Real { block: b, idx });
            }
        }
    }
    lin.extend(merged_stores);
    for (k, &p) in f.blocks[latch.0 as usize].params.iter().enumerate() {
        let (Some(&ta), Some(&ea)) = (t_args.get(k), e_args.get(k)) else {
            return Err("an arm passes the wrong number of arguments to the latch");
        };
        lin.push(LinInst::Merge {
            param: p,
            cond: *cond,
            then_arg: ta,
            else_arg: ea,
        });
    }
    lin.extend(straight(latch));
    Ok(lin)
}

/// May an if-converted arm's load run for every lane although only some lanes take the arm? A
/// `[]T`'s data pointer, re-read from its fat pointer inside the arm that subscripts it, can when
/// that fat pointer is an object [`never_faults`] proves live — a slice parameter's, or a stack slot
/// of this function — and the vector body reads it once, in the guard ([`Plan::InvariantPtrLoad`]),
/// exactly what already happens to the same read outside an arm. A pointer reached any other way
/// may be null where no lane takes the arm: `(*p).w[i]` through a raw `p: *mut Layer` reads `p`
/// itself, and `0 as *mut Layer` is a value the scalar loop survives by never taking the arm.
fn speculable_ptr_load(f: &Function, a: &MemAccess) -> bool {
    let AddrForm::Invariant(MemBase::Outer(p)) = a.addr else {
        return false;
    };
    a.kind == AccessKind::Load && a.value_ty == MirType::Ptr && never_faults(f, p, 8)
}

/// Can `bytes` bytes at `addr` be read on a path the scalar program never takes, without faulting?
/// Only inside an object that is live for the whole call whatever the program does: the 16-byte
/// fat pointer a `[]T` parameter points at (`ParamAttrs::slice`, which `mir_build` records on every
/// slice parameter as a fact of its type), or a local stack slot of this function. `addr` is the
/// object itself or a constant byte offset into it (`gep obj, k : i8`, how a fat pointer's length
/// field is reached), and the read must end inside it. Anything else — a raw `*T` parameter, a
/// struct parameter's field, a pointer loaded from memory — is not known to be dereferenceable.
fn never_faults(f: &Function, addr: ValueId, bytes: u64) -> bool {
    let (obj, off) = match op_of(f, addr) {
        Some(Op::Gep {
            ptr,
            index,
            elem: MirType::I8,
        }) => match const_of(f, *index) {
            Some(k) => (*ptr, k),
            None => return false,
        },
        _ => (addr, 0),
    };
    let size = match f.params.iter().position(|&q| q == obj) {
        Some(i) if f.param_attr(i).slice => 16,
        Some(_) => return false,
        None => match op_of(f, obj) {
            Some(Op::Alloca {
                ty,
                space: wukong_mir::AddrSpace::Local,
                ..
            }) => match slot_bytes(ty) {
                Some(n) => n,
                None => return false,
            },
            _ => return false,
        },
    };
    off >= 0 && off + i128::from(bytes) <= i128::from(size)
}

/// The byte size of a stack slot of type `t`, when it is a plain scalar, pointer or (nested) array
/// of them.
fn slot_bytes(t: &MirType) -> Option<u64> {
    match t {
        MirType::Array(e, n) => slot_bytes(e)?.checked_mul(u64::from(*n)),
        MirType::Ptr => Some(8),
        MirType::I1 => Some(1),
        MirType::F16 | MirType::BF16 => Some(2),
        t => lane_bytes(t).map(u64::from),
    }
}

/// What [`linearize_tree`] hands back.
struct TreeRegion {
    lin: Vec<LinInst>,
    len_checks: Vec<LenCheck>,
    /// Instructions on the body's longest path, for the cost model's scalar side.
    scalar_path: u64,
}

/// How a walk of one structured piece of the region ended ([`TreeWalk::walk`]).
enum WalkEnd {
    /// By a plain branch into its stop block, passing these arguments.
    Args(Vec<ValueId>),
    /// By a conditional whose join is the stop block itself, whose parameters it already merged.
    Joined,
}

/// The state of one [`linearize_tree`] run.
struct TreeWalk<'a> {
    f: &'a Function,
    l: &'a NaturalLoop,
    /// Each block's successors inside the region, a bounds check's dead end left out.
    succ: FxHashMap<u32, Vec<BlockId>>,
    /// How many region edges enter each block.
    preds: FxHashMap<u32, usize>,
    /// The immediate post-dominator of every region block but the latch.
    ipdom: FxHashMap<u32, BlockId>,
    /// Blocks ending in a recognized bounds check: where they continue, and the check.
    asserts: FxHashMap<u32, (BlockId, LenCheck)>,
    /// Instructions a recognized bounds check consumes (its compare, its length load).
    skip: FxHashSet<(u32, usize)>,
    lin: Vec<LinInst>,
    len_checks: Vec<LenCheck>,
    /// The path predicate each visited block runs under.
    pred_of: FxHashMap<u32, Vec<(ValueId, bool)>>,
    /// Has the walk passed the region's first conditional? Stores before it stay in place.
    branched: bool,
    tree: StoreTree,
    /// The first sunk store: its pointer and position (every sunk store must name its address).
    store_at: Option<(ValueId, (BlockId, usize))>,
    sunk_stores: Vec<(BlockId, usize)>,
    loads: Vec<(BlockId, usize)>,
}

impl TreeWalk<'_> {
    /// Emit `b`'s instructions under the path predicate `pred`.
    fn visit(&mut self, b: BlockId, pred: &[(ValueId, bool)]) -> Result<(), &'static str> {
        let f = self.f;
        self.pred_of.insert(b.0, pred.to_vec());
        for (idx, inst) in f.blocks[b.0 as usize].insts.iter().enumerate() {
            if self.skip.contains(&(b.0, idx)) {
                continue;
            }
            match &inst.op {
                Op::Store { ptr, value } if self.branched => {
                    let here = self
                        .access(b, idx)
                        .ok_or("a store in the body has no analyzed address")?;
                    if !matches!(here.addr, AddrForm::Affine { .. }) {
                        return Err("an if-converted store does not step with the loop");
                    }
                    match self.store_at {
                        None => self.store_at = Some((*ptr, (b, idx))),
                        Some((_, (fb, fi))) => {
                            let first = self.access(fb, fi).expect("recorded from an access");
                            if !same_place(first, here) {
                                return Err("an if-converted region stores to more than one place");
                            }
                        }
                    }
                    // On this path the location now holds `value`; elsewhere it is unchanged.
                    let mut on_path = StoreTree::Val(*value);
                    for &(c, pol) in pred.iter().rev() {
                        on_path = if pol {
                            StoreTree::node(c, on_path, StoreTree::Keep)
                        } else {
                            StoreTree::node(c, StoreTree::Keep, on_path)
                        };
                    }
                    let tree = std::mem::replace(&mut self.tree, StoreTree::Keep);
                    self.tree = tree.followed_by(on_path);
                    self.sunk_stores.push((b, idx));
                }
                Op::Load(..) if !pred.is_empty() => {
                    // Resolved once every block's predicate is known: a load some unconditional
                    // access covers runs for every lane, any other is predicated per lane.
                    self.loads.push((b, idx));
                    self.lin.push(LinInst::PredLoad {
                        block: b,
                        idx,
                        pred: pred.to_vec(),
                    });
                }
                Op::Bin(BinOp::SDiv | BinOp::UDiv | BinOp::SRem | BinOp::URem, ..)
                    if !pred.is_empty() =>
                {
                    return Err("an arm divides, which may fault on a lane that would not run it")
                }
                op if unpredicable(op) && !pred.is_empty() => {
                    return Err("an arm has an effect that cannot be predicated")
                }
                op => {
                    if matches!(op, Op::Load(..)) {
                        self.loads.push((b, idx));
                    }
                    self.lin.push(LinInst::Real { block: b, idx });
                }
            }
        }
        Ok(())
    }

    fn access(&self, b: BlockId, idx: usize) -> Option<&MemAccess> {
        self.l
            .accesses
            .iter()
            .find(|a| a.block == b && a.inst == idx)
    }

    /// Walk the structured piece from `b` until it branches into `stop`.
    fn walk(
        &mut self,
        mut b: BlockId,
        stop: BlockId,
        pred: &[(ValueId, bool)],
    ) -> Result<WalkEnd, &'static str> {
        let f = self.f;
        loop {
            self.visit(b, pred)?;
            if let Some(&(cont, chk)) = self.asserts.get(&b.0) {
                // A bounds check only an arm makes is discharged once, in the guard; the same
                // check on every iteration is the shape of `bce`'s cold slow copy.
                if pred.is_empty() {
                    return Err("a bounds check outside an if-converted arm");
                }
                self.len_checks.push(chk);
                if cont == stop {
                    return Ok(WalkEnd::Args(Vec::new()));
                }
                if self.preds.get(&cont.0) != Some(&1) {
                    return Err("an unstructured join in the body");
                }
                b = cont;
                continue;
            }
            match &f.blocks[b.0 as usize].term {
                Terminator::Br { target, args } => {
                    if *target == stop {
                        return Ok(WalkEnd::Args(args.clone()));
                    }
                    if self.preds.get(&target.0) != Some(&1)
                        || !f.blocks[target.0 as usize].params.is_empty()
                    {
                        return Err("an unstructured join in the body");
                    }
                    b = *target;
                }
                Terminator::CondBr {
                    cond,
                    then_blk,
                    then_args,
                    else_blk,
                    else_args,
                } => {
                    let j = *self.ipdom.get(&b.0).ok_or("a branch with no join")?;
                    if self.preds.get(&j.0) != Some(&2) || then_blk == else_blk {
                        return Err("a join of more than two paths");
                    }
                    self.branched = true;
                    let side = |me: &mut Self,
                                blk: BlockId,
                                direct: &[ValueId],
                                pol: bool|
                     -> Result<Vec<ValueId>, &'static str> {
                        if blk == j {
                            return Ok(direct.to_vec());
                        }
                        if !direct.is_empty() {
                            return Err("an arm block takes parameters");
                        }
                        let mut inner = pred.to_vec();
                        inner.push((*cond, pol));
                        match me.walk(blk, j, &inner)? {
                            WalkEnd::Args(a) => Ok(a),
                            WalkEnd::Joined => Err("a join of more than two paths"),
                        }
                    };
                    let t_args = side(self, *then_blk, then_args, true)?;
                    let e_args = side(self, *else_blk, else_args, false)?;
                    let params = &f.blocks[j.0 as usize].params;
                    if t_args.len() != params.len() || e_args.len() != params.len() {
                        return Err("an arm passes the wrong number of arguments to its join");
                    }
                    for (k, &param) in params.iter().enumerate() {
                        self.lin.push(LinInst::Merge {
                            param,
                            cond: *cond,
                            then_arg: t_args[k],
                            else_arg: e_args[k],
                        });
                    }
                    if j == stop {
                        return Ok(WalkEnd::Joined);
                    }
                    b = j;
                }
                _ => return Err("the body ends a block in neither a branch nor a jump"),
            }
        }
    }
}

/// Two affine accesses at the same address — base, index and element type — or two loads of the
/// same invariant location.
fn same_place(a: &MemAccess, b: &MemAccess) -> bool {
    match (&a.addr, &b.addr) {
        (
            AddrForm::Affine {
                base: ba,
                index: ia,
                elem: ea,
            },
            AddrForm::Affine {
                base: bb,
                index: ib,
                elem: eb,
            },
        ) => ba.same_object(*bb) && ia == ib && ea == eb,
        (AddrForm::Invariant(ba), AddrForm::Invariant(bb)) => ba.same_object(*bb),
        _ => false,
    }
}

/// `v` as `iv + off`: the induction variable itself, or it plus or minus a constant.
fn iv_offset(f: &Function, v: ValueId, iv: ValueId) -> Option<i64> {
    if v == iv {
        return Some(0);
    }
    let k = |x: ValueId| const_of(f, x).and_then(|k| i64::try_from(k).ok());
    match op_of(f, v)? {
        Op::Bin(BinOp::Add, a, b) if *a == iv => k(*b),
        Op::Bin(BinOp::Add, a, b) if *b == iv => k(*a),
        Op::Bin(BinOp::Sub, a, b) if *a == iv => k(*b)?.checked_neg(),
        _ => None,
    }
}

/// How many times the loop's instructions and terminators read `v`.
fn reads_in_loop(f: &Function, l: &NaturalLoop, v: ValueId) -> usize {
    let mut n = 0;
    for &b in &l.blocks {
        let blk = &f.blocks[b.0 as usize];
        for inst in &blk.insts {
            each_op_use(&inst.op, &mut |u| n += usize::from(u == v));
        }
        crate::each_term_use(&blk.term, &mut |u| n += usize::from(u == v));
    }
    n
}

/// A slice bounds check ending block `b`: `cond_br (idx <u len), cont, dead`, `dead` a block
/// outside the loop that ends in `unreachable` (the failed check's call and trap), `idx` the
/// induction variable plus a constant, `len` an `i64` readable before the loop. Returns where the
/// block continues, the check, and the instructions only the check reads (its compare, and a
/// length load from an address outside the loop), which the vector body drops.
#[allow(clippy::type_complexity)]
fn bounds_check_of(
    f: &Function,
    l: &NaturalLoop,
    b: BlockId,
    iv: ValueId,
) -> Option<(BlockId, LenCheck, Vec<(u32, usize)>)> {
    let Terminator::CondBr {
        cond,
        then_blk,
        then_args,
        else_blk,
        else_args,
    } = &f.blocks[b.0 as usize].term
    else {
        return None;
    };
    let dead = |x: BlockId| {
        !l.blocks.contains(&x) && matches!(f.blocks[x.0 as usize].term, Terminator::Unreachable)
    };
    let (cont, on_true, args) = if dead(*else_blk) && l.blocks.contains(then_blk) {
        (*then_blk, true, then_args)
    } else if dead(*then_blk) && l.blocks.contains(else_blk) {
        (*else_blk, false, else_args)
    } else {
        return None;
    };
    if !args.is_empty() || !f.blocks[cont.0 as usize].params.is_empty() || cont == l.header {
        return None;
    }
    let Some(Op::Cmp(p, a, c)) = op_of(f, *cond) else {
        return None;
    };
    let (idx, len) = match (p, on_true) {
        (CmpOp::Ult, true) | (CmpOp::Uge, false) => (*a, *c),
        (CmpOp::Ugt, true) | (CmpOp::Ule, false) => (*c, *a),
        _ => return None,
    };
    if *f.value_type(idx) != MirType::I64 || *f.value_type(len) != MirType::I64 {
        return None;
    }
    let off = iv_offset(f, idx, iv)?;
    let in_loop = |v: ValueId| l.blocks.iter().any(|&bb| defined_in_block(f, bb, v));
    let position = |v: ValueId| {
        l.blocks.iter().find_map(|&bb| {
            f.blocks[bb.0 as usize]
                .insts
                .iter()
                .position(|i| i.result == Some(v))
                .map(|i| (bb.0, i))
        })
    };
    let mut skip = Vec::new();
    let len = if !in_loop(len) {
        LenSrc::Value(len)
    } else {
        let Some(Op::Load(at, MirType::I64)) = op_of(f, len) else {
            return None;
        };
        // The guard re-reads the length whether or not any lane takes the arm, so it must be a
        // read that cannot fault there: a slice parameter's or a stack slot's own length field.
        if in_loop(*at) || !never_faults(f, *at, 8) {
            return None;
        }
        if reads_in_loop(f, l, len) == 1 {
            skip.extend(position(len));
        }
        LenSrc::Load(*at)
    };
    if reads_in_loop(f, l, *cond) == 1 {
        skip.extend(position(*cond));
    }
    Some((cont, LenCheck { off, len }, skip))
}

/// Flatten a loop body that is a small **nest** of conditionals — an `if / else if / else`
/// chain, a diamond inside an arm, two diamonds in a row — into one straight line, and see through
/// the bounds checks a checked `[]T` store or load puts inside an arm. [`linearize_region`] keeps
/// the single diamond it always handled; this takes what it declines.
///
/// The region is the loop's blocks minus the header, a DAG from the header's one successor to the
/// latch. A bounds check whose failure leaves for a dead end ([`bounds_check_of`]) is read as a
/// plain jump to its passing side, its check recorded for the guard ([`LenCheck`]) — but only
/// inside an arm: a check on every iteration is `bce`'s cold slow copy. The walk is **structured**:
/// every conditional's two sides meet again at its immediate post-dominator, which no third path
/// enters, and every other block is entered by one edge. So each block runs under a conjunction of
/// branch conditions, its *path predicate*, and each join's parameters are one `select` on the
/// branch that split it ([`LinInst::Merge`], nested merges composing).
///
/// Memory follows the diamond's rules, generalized to paths:
///
/// * **Stores** after the first conditional are sunk to the end of the region and must all name
///   one address. What each lane's path leaves there is a [`StoreTree`] over the branch
///   conditions — the last store on the path, or nothing — built store by store in program order
///   and simplified by the conditions already decided on each path. A tree every path stores
///   through is one vector store of nested selects (`else if` chains); otherwise each lane stores
///   to its own element or to a scratch slot ([`Plan::TreePredStore`]). Stores before the first
///   conditional stay in place, ahead of everything the region sinks.
/// * **Loads** in an arm are speculated when an access on every path (the unconditional blocks,
///   the latch) touches the same place, or when they re-read a data pointer from a fat pointer
///   that cannot fault — a slice parameter's or a stack slot's ([`speculable_ptr_load`]); any
///   other affine load is predicated by its path predicate
///   ([`LinInst::PredLoad`]); anything else declines.
/// * **Order**: sinking a store moves it past every later instruction, so no load may sit on a
///   path after a store it could read — in the same block after it, or in a block reachable from
///   it. Division and calls stay out of arms, as in a diamond.
fn linearize_tree(
    f: &Function,
    l: &NaturalLoop,
    latch: BlockId,
    iv: ValueId,
) -> Result<TreeRegion, &'static str> {
    let entry = {
        let inside: Vec<BlockId> = succs(&f.blocks[l.header.0 as usize].term)
            .into_iter()
            .filter(|b| l.blocks.contains(b))
            .collect();
        match inside.as_slice() {
            [b] if *b != latch => *b,
            _ => return Err("the header does not enter the body through one block"),
        }
    };
    let region: Vec<BlockId> = l
        .blocks
        .iter()
        .copied()
        .filter(|&b| b != l.header)
        .collect();

    // ---- bounds checks and the region graph ---------------------------------------------------
    let mut asserts: FxHashMap<u32, (BlockId, LenCheck)> = FxHashMap::default();
    let mut skip: FxHashSet<(u32, usize)> = FxHashSet::default();
    for &b in &region {
        if let Some((cont, chk, sk)) = bounds_check_of(f, l, b, iv) {
            asserts.insert(b.0, (cont, chk));
            skip.extend(sk);
        }
    }
    let mut succ: FxHashMap<u32, Vec<BlockId>> = FxHashMap::default();
    let mut preds: FxHashMap<u32, usize> = FxHashMap::default();
    for &b in &region {
        let out: Vec<BlockId> = if let Some(&(cont, _)) = asserts.get(&b.0) {
            vec![cont]
        } else if b == latch {
            Vec::new()
        } else {
            let s = succs(&f.blocks[b.0 as usize].term);
            if s.is_empty() || s.iter().any(|t| !region.contains(t)) {
                return Err("a block of the body leaves the loop");
            }
            s
        };
        for t in &out {
            *preds.entry(t.0).or_default() += 1;
        }
        succ.insert(b.0, out);
    }

    // ---- post-dominators, over the region's DAG -----------------------------------------------
    let mut order: Vec<BlockId> = Vec::new(); // postorder: every successor before its block
    let mut seen: FxHashSet<u32> = FxHashSet::default();
    let mut stack: Vec<(BlockId, usize)> = vec![(entry, 0)];
    seen.insert(entry.0);
    while let Some((b, k)) = stack.pop() {
        let out = &succ[&b.0];
        if k < out.len() {
            stack.push((b, k + 1));
            let t = out[k];
            if seen.insert(t.0) {
                stack.push((t, 0));
            }
        } else {
            order.push(b);
        }
    }
    if order.len() != region.len() || !seen.contains(&latch.0) {
        return Err("a block of the body is not on a path to the latch");
    }
    let rank: FxHashMap<u32, usize> = order.iter().enumerate().map(|(i, b)| (b.0, i)).collect();
    // A back edge inside the region would make a successor rank above its block.
    if order
        .iter()
        .any(|b| succ[&b.0].iter().any(|t| rank[&t.0] >= rank[&b.0]))
    {
        return Err("the body has a cycle besides the loop's own");
    }
    let mut pdom: FxHashMap<u32, Vec<u32>> = FxHashMap::default();
    for &b in &order {
        let out = &succ[&b.0];
        let mut set: Vec<u32> = if out.is_empty() {
            if b != latch {
                return Err("a block of the body is not on a path to the latch");
            }
            Vec::new()
        } else {
            let mut it = out.iter().map(|t| pdom[&t.0].clone());
            let first = it.next().expect("non-empty");
            it.fold(first, |acc, s| {
                acc.into_iter().filter(|x| s.contains(x)).collect()
            })
        };
        set.push(b.0);
        pdom.insert(b.0, set);
    }
    let mut ipdom: FxHashMap<u32, BlockId> = FxHashMap::default();
    for &b in &order {
        // The closest strict post-dominator is the one with the most post-dominators of its own.
        if let Some(&x) = pdom[&b.0]
            .iter()
            .filter(|&&x| x != b.0)
            .max_by_key(|&&x| pdom[&x].len())
        {
            ipdom.insert(b.0, BlockId(x));
        }
    }

    // ---- the walk -----------------------------------------------------------------------------
    let mut w = TreeWalk {
        f,
        l,
        succ,
        preds,
        ipdom,
        asserts,
        skip,
        lin: Vec::new(),
        len_checks: Vec::new(),
        pred_of: FxHashMap::default(),
        branched: false,
        tree: StoreTree::Keep,
        store_at: None,
        sunk_stores: Vec::new(),
        loads: Vec::new(),
    };
    match w.walk(entry, latch, &[])? {
        WalkEnd::Joined => {}
        WalkEnd::Args(a) if a.is_empty() && f.blocks[latch.0 as usize].params.is_empty() => {}
        WalkEnd::Args(_) => return Err("the latch is entered by one edge with arguments"),
    }
    if !w.branched {
        return Err("the body has no conditional to flatten");
    }

    // ---- order: no load after a store it could read ----------------------------------------
    let reaches = |from: BlockId, to: BlockId| -> bool {
        let mut todo = vec![from];
        let mut seen: FxHashSet<u32> = FxHashSet::default();
        while let Some(b) = todo.pop() {
            for &t in &w.succ[&b.0] {
                if t == to {
                    return true;
                }
                if seen.insert(t.0) {
                    todo.push(t);
                }
            }
        }
        false
    };
    for &(sb, si) in &w.sunk_stores {
        for &(lb, li) in &w.loads {
            if (lb == sb && li > si) || (lb != sb && reaches(sb, lb)) {
                return Err("an if-converted region reads memory after writing it");
            }
        }
    }

    // ---- loads: speculate the covered ones, predicate the rest ----------------------------------
    let uncond_blocks: FxHashSet<u32> = w
        .pred_of
        .iter()
        .filter(|(_, p)| p.is_empty())
        .map(|(b, _)| *b)
        .chain(std::iter::once(latch.0))
        .collect();
    let unconditional: Vec<&MemAccess> = l
        .accesses
        .iter()
        .filter(|a| uncond_blocks.contains(&a.block.0))
        .collect();
    let mut lin = std::mem::take(&mut w.lin);
    for s in lin.iter_mut() {
        let LinInst::PredLoad { block, idx, .. } = s else {
            continue;
        };
        let a = w
            .access(*block, *idx)
            .ok_or("a load in the body has no analyzed address")?;
        let covered = speculable_ptr_load(f, a) || unconditional.iter().any(|b| same_place(a, b));
        if covered {
            *s = LinInst::Real {
                block: *block,
                idx: *idx,
            };
        } else if !matches!(a.addr, AddrForm::Affine { .. }) {
            return Err("an arm loads an address no access on every path covers");
        }
    }

    // ---- the sunk stores, then the latch --------------------------------------------------------
    let tree = simplify_tree(
        std::mem::replace(&mut w.tree, StoreTree::Keep),
        &mut Vec::new(),
    );
    if let Some((ptr, at)) = w.store_at {
        if tree != StoreTree::Keep {
            lin.push(LinInst::TreeStore { ptr, tree, at });
        }
    }
    lin.extend(
        (0..f.blocks[latch.0 as usize].insts.len()).map(|idx| LinInst::Real { block: latch, idx }),
    );

    // ---- the scalar side of the cost model: the longest path ----------------------------------
    let mut longest: FxHashMap<u32, u64> = FxHashMap::default();
    for &b in &order {
        let own = f.blocks[b.0 as usize].insts.len() as u64;
        let tail = w.succ[&b.0]
            .iter()
            .map(|t| longest[&t.0])
            .max()
            .unwrap_or(0);
        longest.insert(b.0, own + tail);
    }
    Ok(TreeRegion {
        lin,
        len_checks: w.len_checks,
        scalar_path: longest[&entry.0],
    })
}

/// Drop every test of a condition the path to it has already decided.
fn simplify_tree(t: StoreTree, known: &mut Vec<(ValueId, bool)>) -> StoreTree {
    match t {
        StoreTree::Node { cond, then, els } => {
            if let Some(&(_, v)) = known.iter().find(|(c, _)| *c == cond) {
                return simplify_tree(if v { *then } else { *els }, known);
            }
            known.push((cond, true));
            let t2 = simplify_tree(*then, known);
            known.pop();
            known.push((cond, false));
            let e2 = simplify_tree(*els, known);
            known.pop();
            StoreTree::node(cond, t2, e2)
        }
        leaf => leaf,
    }
}

/// The search-loop shape [`EarlyExit`] widens, as `(exiting block, its condition, continue on
/// true?)`: three blocks — the header, which tests the trip and enters the exiting block; the
/// exiting block, which leaves the loop on one arm and falls into the latch on the other; and the
/// straight-line latch.
fn early_exit_of(f: &Function, l: &NaturalLoop) -> Option<(BlockId, ValueId, bool)> {
    if l.exits.len() != 2 || l.blocks.len() != 3 || l.latches.len() != 1 {
        return None;
    }
    let latch = l.latches[0];
    if l.exits.iter().filter(|(b, _)| *b == l.header).count() != 1 {
        return None;
    }
    let &(exiting, out) = l.exits.iter().find(|(b, _)| *b != l.header)?;
    if exiting == latch {
        return None;
    }
    // A dead-end exit — a failed bounds check's `call wukong_rt_bounds_fail; unreachable` in the
    // slow copy `bce` keeps — is a cold path, not a search: widening it buys nothing and costs code
    // and compile time on every versioned slice loop.
    if matches!(f.blocks[out.0 as usize].term, Terminator::Unreachable) {
        return None;
    }
    let inside: Vec<BlockId> = succs(&f.blocks[l.header.0 as usize].term)
        .into_iter()
        .filter(|b| l.blocks.contains(b))
        .collect();
    if inside.as_slice() != [exiting] {
        return None;
    }
    let Terminator::CondBr {
        cond,
        then_blk,
        then_args,
        else_blk,
        else_args,
    } = &f.blocks[exiting.0 as usize].term
    else {
        return None;
    };
    let (continue_on_true, into_latch) = if *then_blk == latch && *else_blk == out {
        (true, then_args)
    } else if *else_blk == latch && *then_blk == out {
        (false, else_args)
    } else {
        return None;
    };
    let latch_blk = &f.blocks[latch.0 as usize];
    if !into_latch.is_empty() || !latch_blk.params.is_empty() {
        return None;
    }
    if !matches!(&latch_blk.term, Terminator::Br { target, .. } if *target == l.header) {
        return None;
    }
    Some((exiting, *cond, continue_on_true))
}

/// A compile-time integer constant, wherever it is defined.
fn const_of(f: &Function, v: ValueId) -> Option<i128> {
    match op_of(f, v)? {
        Op::ConstInt(k, ty) if ty.is_int() => Some(*k),
        _ => None,
    }
}

/// A search loop's header test as `iv <pred> bound` while the loop runs, `pred` `Slt` or `Ult`.
fn header_bound(
    f: &Function,
    l: &NaturalLoop,
    iv: ValueId,
) -> Result<(CmpOp, ValueId), &'static str> {
    let Terminator::CondBr {
        cond,
        then_blk,
        else_blk,
        ..
    } = &f.blocks[l.header.0 as usize].term
    else {
        return Err("the header does not test the trip");
    };
    if !l.blocks.contains(then_blk) || l.blocks.contains(else_blk) {
        return Err("the header's test does not continue on true");
    }
    let Some(Op::Cmp(pred, a, b)) = op_of(f, *cond) else {
        return Err("the header's test is not a compare");
    };
    let (pred, bound) = if *a == iv && *b != iv {
        (*pred, *b)
    } else if *b == iv && *a != iv {
        let swapped = match pred {
            CmpOp::Sgt => CmpOp::Slt,
            CmpOp::Ugt => CmpOp::Ult,
            _ => return Err("the header's test is not `iv < bound`"),
        };
        (swapped, *a)
    } else {
        return Err("the header's test does not read the induction variable");
    };
    if !matches!(pred, CmpOp::Slt | CmpOp::Ult) {
        return Err("the header's test is not `iv < bound`");
    }
    Ok((pred, bound))
}

/// The trip of a search loop, from its header test. The analysis reports none — the loop has two
/// exits — but the vector loop only needs the header's bound: it runs whole groups below it, and
/// the early exit is resolved in the original loop. Constant endpoints give a constant count
/// (rounded down to whole groups of `w`); otherwise the bound must be readable before the loop.
fn early_trip(
    f: &Function,
    l: &NaturalLoop,
    iv: ValueId,
    start: Option<ValueId>,
    w: u32,
) -> Result<TripPlan, &'static str> {
    let (pred, bound) = header_bound(f, l, iv)?;
    if let (Some(s), Some(e)) = (start.and_then(|v| const_of(f, v)), const_of(f, bound)) {
        if pred == CmpOp::Ult && (s < 0 || e < 0) {
            return Err("an unsigned trip over a negative constant");
        }
        let groups = if e > s { (e - s) / i128::from(w) } else { 0 };
        if groups < i128::from(MIN_TRIP_GROUPS) {
            return Err("trip count is below one vector group");
        }
        return u64::try_from(groups * i128::from(w))
            .map(TripPlan::Const)
            .map_err(|_| "trip count does not fit");
    }
    if l.blocks.iter().any(|b| defined_in_block(f, *b, bound)) {
        return Err("the search loop's bound is computed inside the loop");
    }
    Ok(TripPlan::Runtime {
        end: bound,
        signed: pred == CmpOp::Slt,
    })
}

/// What has to hold before a search loop's test may run `W` iterations ahead of the scalar loop:
/// the steps before the exit store nothing and read only memory that exists for the whole trip
/// (contiguous loads proved by [`speculation_safe`]; a gather or an interleave window is refused),
/// and the test itself is a value the vector body computes before the exit.
fn early_exit_verdict(
    f: &Function,
    l: &NaturalLoop,
    lin: &[LinInst],
    plans: &[Plan],
    x: &EarlyExit,
    iv: ValueId,
    start: Option<ValueId>,
) -> Result<(), &'static str> {
    let (_, end) = header_bound(f, l, iv)?;
    let start = start.ok_or("the induction variable has no single start")?;
    let mut defines_cond = false;
    for (step, plan) in lin[..x.split].iter().zip(&plans[..x.split]) {
        let LinInst::Real { block, idx } = step else {
            return Err("a search loop's test is not straight-line code");
        };
        let inst = &f.blocks[block.0 as usize].insts[*idx];
        defines_cond |= inst.result == Some(x.cond);
        match plan {
            Plan::VecStore | Plan::GroupStore { .. } | Plan::MaskedStore | Plan::Scatter => {
                return Err("a search loop stores before its exit")
            }
            Plan::Gather | Plan::LaneGep | Plan::GroupLoad { .. } => {
                return Err("a search loop gathers before its exit")
            }
            Plan::Reduce if inst.result == Some(x.cond) => {
                return Err("a search loop exits on its own accumulator")
            }
            _ => {}
        }
        if matches!(inst.op, Op::Store { .. }) {
            return Err("a search loop stores before its exit");
        }
        if let Op::Load(..) = inst.op {
            let Some(a) = l
                .accesses
                .iter()
                .find(|a| a.block == *block && a.inst == *idx)
            else {
                return Err("a load before the exit has no analyzed address");
            };
            match &a.addr {
                // The same address every iteration: the vector loop runs only when the trip covers
                // a whole group, so the scalar loop reads it too.
                AddrForm::Invariant(_) => {}
                AddrForm::Affine { base, index, elem } => {
                    if index.coeff != Coeff::Const(1)
                        || !index.terms.is_empty()
                        || !speculation_safe(f, *base, elem, index.konst, start, end)
                    {
                        return Err("a load before the exit may read past its object");
                    }
                }
                AddrForm::Unknown => return Err("a load before the exit has no analyzed address"),
            }
        }
    }
    // A condition from outside the body is one answer for the whole group, which is fine; one
    // defined in the latch would be read before it exists.
    if !defines_cond && l.blocks.iter().any(|b| defined_in_block(f, *b, x.cond)) {
        return Err("a search loop's test is not computed before its exit");
    }
    Ok(())
}

/// The chain of blocks a **checked body** runs every iteration, entry first and latch last, or
/// `None`.
///
/// Every block of the loop but the header is on the chain, entered from the one before it, and
/// leaves it either by a plain jump to the next or by a **check** — `cond_br c, next, dead` with
/// `dead` a block outside the loop that ends in `unreachable`: a failed bounds check's `call
/// wukong_rt_bounds_fail; unreachable`, which is how `mir_build` checks a `[]T` access, and the
/// check `bce` cannot prove (a gathered index, `t[q[i]]`) left behind in its fast copy. At least
/// one check, or this is a straight-line body and not this function's business. A check inside an
/// if-converted arm is [`linearize_tree`]'s; a body with both declines.
fn checked_spine(f: &Function, l: &NaturalLoop) -> Option<Vec<BlockId>> {
    if l.latches.len() != 1 || !l.abnormal_exits.is_empty() {
        return None;
    }
    let latch = l.latches[0];
    if l.exits.iter().filter(|(b, _)| *b == l.header).count() != 1 {
        return None;
    }
    let dead =
        |x: BlockId| !l.blocks.contains(&x) && matches!(f.block(x).term, Terminator::Unreachable);
    let inside: Vec<BlockId> = succs(&f.block(l.header).term)
        .into_iter()
        .filter(|b| l.blocks.contains(b))
        .collect();
    let [entry] = inside.as_slice() else {
        return None;
    };
    let mut chain = vec![*entry];
    let mut checks = 0usize;
    let mut cur = *entry;
    while cur != latch {
        if chain.len() >= l.blocks.len() {
            return None;
        }
        let next = match &f.block(cur).term {
            Terminator::Br { target, args } if args.is_empty() => *target,
            Terminator::CondBr {
                cond,
                then_blk,
                then_args,
                else_blk,
                else_args,
            } => {
                // `bce` marks the checks its guard proved, in the original loop it keeps as the
                // slow copy, by rewriting `idx <u len` to `len >u idx`. That copy only runs when a
                // check may fail — a cold path that widening would only make bigger.
                if matches!(op_of(f, *cond), Some(Op::Cmp(CmpOp::Ugt, ..))) {
                    return None;
                }
                // A check on a constant — what `unswitch` leaves in each half of a split range —
                // is a branch `simplify-cfg` is about to fold, not a test to vectorize.
                if matches!(op_of(f, *cond), Some(Op::ConstInt(..))) {
                    return None;
                }
                checks += 1;
                if dead(*else_blk) && then_args.is_empty() {
                    *then_blk
                } else if dead(*then_blk) && else_args.is_empty() {
                    *else_blk
                } else {
                    return None;
                }
            }
            _ => return None,
        };
        if next == l.header || !l.blocks.contains(&next) || !f.block(next).params.is_empty() {
            return None;
        }
        chain.push(next);
        cur = next;
    }
    if !matches!(&f.block(latch).term, Terminator::Br { target, .. } if *target == l.header) {
        return None;
    }
    if checks == 0 || chain.len() + 1 != l.blocks.len() {
        return None;
    }
    if !l
        .exits
        .iter()
        .all(|&(b, out)| b == l.header || (chain.contains(&b) && dead(out)))
    {
        return None;
    }
    Some(chain)
}

/// What [`linearize_spine`] hands back.
struct SpineBody {
    lin: Vec<LinInst>,
    len_checks: Vec<LenCheck>,
    exits: Vec<EarlyExit>,
}

/// Flatten a [checked body](checked_spine) into one straight line, with each check either
/// discharged before the loop or tested per group.
///
/// A check on `iv + k` against a length readable before the loop ([`bounds_check_of`]) holds for
/// every iteration of the vector loop or for none of them, so it is one [`LenCheck`] in the guard,
/// exactly as one inside an arm: the vector loop runs only when no lane of it can fail, and the
/// scalar loop otherwise does all the work and stops where it always did. Every other check — the
/// index of a gather, `t[q[i]]`, is known only once `q[i]` is loaded — is a **group test**
/// ([`VecPlan::spine_exits`]): computed over the group's lanes at its place in the body, and when
/// any lane fails it, the vector loop leaves for the scalar loop at the group's first iteration,
/// which re-runs at most `W` iterations and stops at exactly the check, the iteration and the
/// message it always did — the search loop's early exit ([`EarlyExit`]), aimed at a dead end. What
/// makes running the steps before a test ahead of the scalar loop harmless is [`spine_verdict`]'s.
fn linearize_spine(f: &Function, l: &NaturalLoop, chain: &[BlockId], iv: ValueId) -> SpineBody {
    let mut skip: FxHashSet<(u32, usize)> = FxHashSet::default();
    let mut len_checks = Vec::new();
    let mut discharged: FxHashSet<u32> = FxHashSet::default();
    for &b in chain {
        if let Some((_, chk, sk)) = bounds_check_of(f, l, b, iv) {
            len_checks.push(chk);
            skip.extend(sk);
            discharged.insert(b.0);
        }
    }
    let mut lin = Vec::new();
    let mut exits = Vec::new();
    for &b in chain {
        for idx in 0..f.block(b).insts.len() {
            if !skip.contains(&(b.0, idx)) {
                lin.push(LinInst::Real { block: b, idx });
            }
        }
        if discharged.contains(&b.0) {
            continue;
        }
        if let Terminator::CondBr { cond, then_blk, .. } = &f.block(b).term {
            exits.push(EarlyExit {
                cond: *cond,
                continue_on_true: l.blocks.contains(then_blk),
                split: lin.len(),
            });
        }
    }
    SpineBody {
        lin,
        len_checks,
        exits,
    }
}

/// What has to hold before a checked body's group tests may run `W` iterations ahead of the scalar
/// loop ([`linearize_spine`]). Every step before the last test runs for every lane of a group
/// whose lanes passed the tests before it — including lanes past one that will fail a later test,
/// iterations the scalar loop never reaches. So, before the last test:
///
/// * **nothing stores** — a group that leaves is re-run by the scalar loop, which must find memory
///   as the group found it;
/// * every **load** reads memory that exists: an invariant address (lane 0's iteration reads it
///   too); a contiguous `x[i + k]` over a `[]T` whose fat pointer cannot fault, which adds a
///   [`LenCheck`] so the vector loop runs only when every lane's element exists, or over a
///   fixed-extent object [`speculation_safe`] proves; or a gather whose index an earlier test
///   checked against its own object's length ([`checked_by_test`]) — each lane passed that test,
///   so its element exists;
/// * a call is one the effect summaries let run speculatively ([`call_verdict`]).
fn spine_verdict(
    f: &Function,
    l: &NaturalLoop,
    lin: &[LinInst],
    plans: &[Plan],
    exits: &[EarlyExit],
    len_checks: &mut Vec<LenCheck>,
    start: Option<ValueId>,
) -> Result<(), &'static str> {
    let last = exits.iter().map(|x| x.split).max().unwrap_or(0);
    let iv = l.primary().ok_or("no primary induction variable")?.value;
    let end = header_bound(f, l, iv).ok().map(|(_, end)| end);
    for (at, (step, plan)) in lin[..last].iter().zip(&plans[..last]).enumerate() {
        let LinInst::Real { block, idx } = step else {
            return Err("a checked body's test is not straight-line code");
        };
        let inst = &f.blocks[block.0 as usize].insts[*idx];
        if matches!(
            plan,
            Plan::VecStore | Plan::GroupStore { .. } | Plan::Scatter | Plan::GroupLoad { .. }
        ) || matches!(inst.op, Op::Store { .. })
        {
            return Err("a checked body stores before a lane test");
        }
        let Op::Load(ptr, ty) = &inst.op else {
            continue;
        };
        let a = l
            .accesses
            .iter()
            .find(|a| a.block == *block && a.inst == *idx)
            .ok_or("a load before a lane test has no analyzed address")?;
        match &a.addr {
            AddrForm::Invariant(_) => {}
            AddrForm::Affine { base, index, elem } => {
                if index.coeff != Coeff::Const(1) || !index.terms.is_empty() || elem != ty {
                    return Err("a load before a lane test does not step with the loop");
                }
                match *base {
                    MemBase::ReloadedOuter { addr, .. } if never_faults(f, addr, 16) => {
                        len_checks.push(LenCheck {
                            off: index.konst,
                            len: LenSrc::Header(addr),
                        });
                    }
                    b if start
                        .zip(end)
                        .is_some_and(|(s, e)| speculation_safe(f, b, elem, index.konst, s, e)) => {}
                    _ => return Err("a load before a lane test may read past its object"),
                }
            }
            AddrForm::Unknown => {
                if !checked_by_test(f, *ptr, ty, exits, at) {
                    return Err(
                        "a gather before a lane test reads an index no earlier test checked",
                    );
                }
            }
        }
    }
    Ok(())
}

/// Is the gathered address `ptr` (`gep data, idx`, moving one `ty`) in bounds for every lane that
/// passed the group tests before step `at` — one of them `idx <u len` with `len` the length of the
/// object `data` points into? The pairing is read off the MIR: `data` and `len` re-read from one
/// fat pointer (`load ptr hdr`, `load i64 (hdr + 8)`) — how `mir_build` checks a `[]T` element.
fn checked_by_test(
    f: &Function,
    ptr: ValueId,
    ty: &MirType,
    exits: &[EarlyExit],
    at: usize,
) -> bool {
    let Some(Op::Gep {
        ptr: data,
        index,
        elem,
    }) = op_of(f, ptr)
    else {
        return false;
    };
    if elem != ty {
        return false;
    }
    exits.iter().filter(|x| x.split <= at).any(|x| {
        let Some(Op::Cmp(p, a, b)) = op_of(f, x.cond) else {
            return false;
        };
        let (idx, len) = match (p, x.continue_on_true) {
            (CmpOp::Ult, true) | (CmpOp::Uge, false) => (*a, *b),
            (CmpOp::Ugt, true) | (CmpOp::Ule, false) => (*b, *a),
            _ => return false,
        };
        if idx != *index || *f.value_type(idx) != MirType::I64 {
            return false;
        }
        match (op_of(f, *data), op_of(f, len)) {
            (Some(Op::Load(hdr, MirType::Ptr)), Some(Op::Load(len_at, MirType::I64))) => {
                matches!(
                    op_of(f, *len_at),
                    Some(Op::Gep { ptr: h, index: k, elem: MirType::I8 })
                        if h == hdr && const_of(f, *k) == Some(8)
                )
            }
            _ => false,
        }
    })
}

/// Is `v` a parameter or an instruction result of block `b`?
fn defined_in_block(f: &Function, b: BlockId, v: ValueId) -> bool {
    let blk = &f.blocks[b.0 as usize];
    blk.params.contains(&v) || blk.insts.iter().any(|i| i.result == Some(v))
}

/// May the vector loop read `base[iv + c]` for every `iv` in `[start, end)`, although the scalar
/// loop may leave before it reaches the end? Two proofs, both about the object rather than the
/// loop:
///
/// * a **stack array** of this function, `[T; N]` with `T` the element read: the constant range
///   `[start + c, end + c)` lies inside `[0, N)`;
/// * a **fixed-extent buffer parameter** — a `[T; N]` array or a statically-shaped tensor, whose
///   extent the front end records on the MIR parameter ([`wukong_mir::ParamExtent`], counted in
///   scalar leaves) — under the same constant-range test, when the element read is a scalar of the
///   recorded width. The extent is the declared type's contract, and no view reinterprets a
///   buffer, so the pointer addresses at least `N` leaves for the whole call. The front end checks
///   the contract where the argument's extent is known (another constant length is `E0501`), but a
///   `?` (run-time) extent binds a fixed one unchecked, and a shorter buffer bound that way is
///   outside the defined contract (`docs/language-guide.md`, "Extents") — here `-O2` may read past
///   it where `-O0` happened to stop in time;
/// * a **`[]T` slice whose loop runs to its own length** — `end` is `load i64 (hdr + 8)`, the
///   fat pointer's length field, and the data is re-read from the same `hdr` — with `c <= 0` and a
///   constant `start + c >= 0`: every element below the length exists.
///
/// Anything else — a pointer with no known extent, a symbolic offset or bound — is refused: a
/// group straddling the end of a shorter buffer could fault where the scalar loop stops in time.
fn speculation_safe(
    f: &Function,
    base: MemBase,
    elem: &MirType,
    c: i64,
    start: ValueId,
    end: ValueId,
) -> bool {
    let c = i128::from(c);
    match base {
        MemBase::Outer(v) => {
            let n = match op_of(f, v) {
                Some(Op::Alloca { ty, .. }) => match ty {
                    MirType::Array(e, n) if **e == *elem => i128::from(*n),
                    t if t == elem => 1,
                    _ => return false,
                },
                Some(_) => return false,
                None => match param_extent(f, v) {
                    Some(x) if lane_bytes(elem) == Some(x.elem_bytes) => i128::from(x.elems),
                    _ => return false,
                },
            };
            let (Some(s), Some(e)) = (const_of(f, start), const_of(f, end)) else {
                return false;
            };
            s + c >= 0 && e + c <= n
        }
        MemBase::ReloadedOuter { addr, .. } => {
            let Some(Op::Load(len_at, MirType::I64)) = op_of(f, end) else {
                return false;
            };
            let Some(Op::Gep {
                ptr,
                index,
                elem: MirType::I8,
            }) = op_of(f, *len_at)
            else {
                return false;
            };
            *ptr == addr
                && const_of(f, *index) == Some(8)
                && c <= 0
                && const_of(f, start).is_some_and(|s| s + c >= 0)
        }
    }
}

/// The recorded storage extent of `v` when it is one of `f`'s own parameters.
fn param_extent(f: &Function, v: ValueId) -> Option<wukong_mir::ParamExtent> {
    let i = f.params.iter().position(|&p| p == v)?;
    f.param_attr(i).extent
}

fn succs(t: &Terminator) -> Vec<BlockId> {
    match t {
        Terminator::Br { target, .. } => vec![*target],
        Terminator::CondBr {
            then_blk, else_blk, ..
        } => vec![*then_blk, *else_blk],
        _ => Vec::new(),
    }
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
    // No memory access, no write of any kind (`Op::effects`) and no frame slot.
    if header.insts.iter().any(|i| {
        i.op.may_read_memory()
            || i.op.may_write_memory()
            || matches!(
                i.op,
                Op::Alloca {
                    space: wukong_mir::AddrSpace::Local,
                    ..
                }
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

/// Split every use of a value that moves with the induction variable into an **address** role and a
/// **value** role, and decline the loop unless each one is handled.
///
/// This is the invariant that makes [`Wide::Uniform`] mean what it says. The induction variable is
/// the same in every lane only when it is a `gep` index: the vector load at `base + i` covers
/// elements `i .. i+W-1` on its own, so lane `k` reads exactly what scalar iteration `i+k` read.
/// Used as a *value* it is not uniform, and splatting it writes the group's first counter into all
/// `W` lanes. That was a real miscompile during development: an initializer loop
/// `x[i] = (i as f32) * 0.25` produced four copies of every fourth element and interp and native
/// agreed on the wrong answer, so only the `-O0` vs `-O2` gate caught it.
///
/// The address set is grown as a transitive closure rather than read off
/// [`crate::loop_info::NaturalLoop::derived`], because `derived` only records *integer* results: a
/// `sitofp` of the induction variable is exactly the dangerous case and would not appear in it.
/// Address arithmetic (`add`/`sub`/`mul`/`shl`/`sext`/`neg`, and `gep` itself) may consume an index
/// value, because its result is then an index value too and is checked in turn; the closure
/// terminates at a `gep` index, a load/store pointer, or the latch's branch arguments (where the
/// only index value is the induction variable's own increment, which [`emit_body`] replaces with a
/// step of `W`).
///
/// A value-role use is no longer fatal: it is answered by a lane ramp ([`IvRamp`]), and the returned
/// set names the steps of `lin` that need one. Three things keep that narrow enough to be obviously
/// sound:
///
/// * **Only the induction variable itself.** An instruction whose index-set operands include
///   anything other than `iv` is still declined — a ramp for `i + 1` or `sext i` is a different
///   (and in the `sext` case inexpressible) construction. See [`IvRamp`].
/// * **Never a pointer or a memory address.** `gep`, a load's pointer and a store's pointer are
///   address roles by construction, and the induction variable is an integer, so it can never *be*
///   one of them. The only value-role memory case is a store whose stored *value* is the counter.
/// * **Never a synthesized if-conversion step.** The two `select`s [`linearize_region`] builds keep
///   today's flat rejection; widening them over a ramp is expressible but buys nothing measured, and
///   an unexercised path here is a miscompile waiting for a program to write it.
///
/// `WUKONG_NO_IV_RAMP=1` restores the flat decline, for the same reason
/// [`Vectorize::run_function`]'s `WUKONG_NO_VECTORIZE` exists: the only honest A/B on this machine
/// is same-run and adjacent, so the two arms of "what did the ramp buy" have to be one binary. It
/// is strictly a *narrowing* switch — a loop it turns off is one this pass declined before the ramp
/// existed — so no program can compute a different answer under it.
fn classify_iv_uses(
    f: &Function,
    lin: &[LinInst],
    iv: ValueId,
    arg_merges: &FxHashSet<u32>,
) -> Option<FxHashSet<usize>> {
    let no_ramp = std::env::var_os("WUKONG_NO_IV_RAMP").is_some();
    let mut index: FxHashSet<u32> = FxHashSet::default();
    index.insert(iv.0);
    let mut value_use: FxHashSet<usize> = FxHashSet::default();
    // The header parameters other than the IV are not index values: `carried_values_are_handled`
    // has already restricted them to reductions and pass-through invariants.
    for (at, step) in lin.iter().enumerate() {
        // A synthesized `select` never takes an index value: `linearize_region` only ever merges a
        // latch parameter or two stored *values*, and a lane-varying stored value is rejected
        // separately. Check it rather than assume it.
        let inst = match step {
            LinInst::Real { block, idx } => &f.blocks[block.0 as usize].insts[*idx],
            LinInst::PredLoad { block, idx, pred } => {
                if pred.iter().any(|(c, _)| index.contains(&c.0)) {
                    return None;
                }
                &f.blocks[block.0 as usize].insts[*idx]
            }
            LinInst::TreeStore { tree, .. } => {
                let (mut conds, mut vals) = (Vec::new(), Vec::new());
                tree.operands(&mut conds, &mut vals);
                if conds.iter().chain(&vals).any(|v| index.contains(&v.0)) {
                    return None;
                }
                continue;
            }
            LinInst::PredStore { cond, value, .. } => {
                if index.contains(&cond.0) || index.contains(&value.0) {
                    return None;
                }
                continue;
            }
            LinInst::MaskedStore {
                cond,
                then_val,
                else_val,
                ..
            } => {
                if index.contains(&cond.0)
                    || index.contains(&then_val.0)
                    || index.contains(&else_val.0)
                {
                    return None;
                }
                continue;
            }
            // An argmax's index merge (`at = c ? i : at`) is folded with its reduction, which
            // takes the counter's lane ramp itself ([`RedPlan::arg`]).
            LinInst::Merge { param, .. } if arg_merges.contains(&param.0) => continue,
            LinInst::Merge {
                cond,
                then_arg,
                else_arg,
                ..
            } => {
                if index.contains(&cond.0)
                    || index.contains(&then_arg.0)
                    || index.contains(&else_arg.0)
                {
                    return None;
                }
                continue;
            }
        };
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
        let role = match &inst.op {
            Op::Gep { .. } => Role::Address { propagates: true },
            Op::Bin(BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Shl, ..)
            | Op::Cast(CastKind::SExt, ..)
            | Op::Neg(_) => Role::Address { propagates: true },
            Op::Load(p, _) if index.contains(&p.0) => Role::Address { propagates: false },
            Op::Load(..) => Role::Reject,
            // A store reads its pointer in an address role and its value in a value role, so the
            // two are judged separately: the address must be the loop's index (a store at a
            // uniform address would have every lane write the same place), and the stored value
            // may be the counter itself — `o[i] = i` — through the ramp.
            Op::Store { ptr, value } => {
                if !index.contains(&ptr.0) {
                    Role::Reject
                } else if !index.contains(&value.0) {
                    Role::Address { propagates: false }
                } else if *value == iv {
                    Role::Value
                } else {
                    Role::Reject
                }
            }
            _ => Role::Value,
        };
        match role {
            Role::Reject => return None,
            Role::Address { propagates } => {
                if propagates {
                    if let Some(r) = inst.result {
                        index.insert(r.0);
                    }
                }
            }
            Role::Value => {
                // Every index-set operand has to be the induction variable itself (a `Store`'s
                // pointer excepted — it was just checked as an address), and the result, if any, is
                // deliberately NOT added to `index`: it is lane-varying data from here on, and
                // `plan_body`'s ordinary vector dataflow owns it.
                let store_ptr = match &inst.op {
                    Op::Store { ptr, .. } => Some(*ptr),
                    _ => None,
                };
                let mut only_iv = true;
                each_op_use(&inst.op, &mut |u| {
                    if index.contains(&u.0) && u != iv && Some(u) != store_ptr {
                        only_iv = false;
                    }
                });
                if no_ramp || !only_iv || !ramp_consumable(&inst.op) {
                    return None;
                }
                value_use.insert(at);
            }
        }
    }
    Some(value_use)
}

/// How one instruction reads the values that move with the induction variable.
enum Role {
    /// Purely to address memory; the operands stay scalar. `propagates` marks the ops whose result
    /// is itself an address value and joins the closure.
    Address { propagates: bool },
    /// As a lane-varying value; the induction variable operand becomes the lane ramp.
    Value,
    /// Neither, so the loop cannot be widened.
    Reject,
}

/// May this op not run under a lane mask? A frame slot, or a write other than a plain store
/// (`Op::effects`: a call of any kind, an atomic) — something a lane that would not have run it
/// cannot be kept from doing.
fn unpredicable(op: &Op) -> bool {
    matches!(
        op,
        Op::Alloca {
            space: wukong_mir::AddrSpace::Local,
            ..
        }
    ) || (op.may_write_memory() && !matches!(op, Op::Store { .. }))
}

/// May this operation read the induction variable as one lane-varying vector operand?
///
/// The list is the ops [`plan_body`] knows how to widen, minus the ones whose operands are
/// addresses. It is a filter, not a promise: `plan_body` still decides how each one is done. An
/// integer divide ([`bin_widenable`] refuses it), a shift whose amount is the ramp (that amount
/// varies per lane) and a cast with no exact vector form ([`cast_route`]) are replicated lane by
/// lane, reading `i + k` in lane `k` instead of the ramp vector — so anything this lets through
/// that has no vector form is replicated or comes back as a plain decline, never as bad MIR.
fn ramp_consumable(op: &Op) -> bool {
    matches!(
        op,
        Op::Bin(..)
            | Op::Cmp(..)
            | Op::Select(..)
            | Op::Neg(_)
            | Op::Not(_)
            | Op::Cast(..)
            | Op::Fma(..)
            | Op::Store { .. }
            | Op::Call { .. }
    )
}

/// Every affine access must move one element of a vectorizable lane type per iteration, and step by
/// one element — or be a member of a complete interleave group. The group width is no longer one
/// access's lane count — [`plan_loop`] derives it from the narrowest lane of the whole widened body
/// — so what is left to check here is the shape of each access on its own.
fn access_shape_verdict(
    l: &NaturalLoop,
    member: &FxHashMap<AccessKey, (usize, u32)>,
) -> Result<(), &'static str> {
    for a in &l.accesses {
        let AddrForm::Affine { elem, .. } = &a.addr else {
            continue;
        };
        if a.addr.const_stride() != Some(1) && !member.contains_key(&(a.block.0, a.inst)) {
            return Err("an access is strided outside a complete interleave group");
        }
        // The `gep`'s unit and the value transferred must be the same type, or `base + index`
        // does not step one transferred element per lane.
        if *elem != a.value_ty {
            return Err("an access moves a different type than its index counts in");
        }
        if lane_bytes(&a.value_ty).is_none() {
            return Err("an access moves a type with no vector lane form");
        }
    }
    Ok(())
}

/// Collect the loop's interleave groups: the affine accesses with a constant stride `k` in
/// `2..=MAX_INTERLEAVE`, grouped by object, invariant part, stride, element type and kind. Each
/// group must cover every residue `0..k` exactly from its lowest constant (a load group may read one
/// element twice; a store group may not write one twice), or the loop is declined. Returns the
/// groups and, per member access, `(group, residue)`.
///
/// Any other strided access is left out here and declined by [`access_shape_verdict`] — including
/// a negative stride on an ascending counter (`x[n - 1 - i]`). A countdown's `x[i]` has stride
/// `+1` against its own counter and is widened ([`VecPlan::dir`]); its groups are not formed.
#[allow(clippy::type_complexity)]
fn interleave_groups(
    l: &NaturalLoop,
) -> Result<(Vec<InterleaveGroup>, FxHashMap<AccessKey, (usize, u32)>), &'static str> {
    struct Building<'a> {
        base: MemBase,
        terms: &'a [(ValueId, i64)],
        k: i64,
        elem: &'a MirType,
        kind: AccessKind,
        members: Vec<(&'a MemAccess, i64)>,
    }
    let mut building: Vec<Building<'_>> = Vec::new();
    for a in &l.accesses {
        let AddrForm::Affine { base, index, elem } = &a.addr else {
            continue;
        };
        let Some(k) = index.coeff.as_const() else {
            continue;
        };
        if !(2..=MAX_INTERLEAVE).contains(&k) {
            continue;
        }
        let found = building.iter_mut().find(|g| {
            g.base.same_object(*base)
                && g.terms == index.terms.as_slice()
                && g.k == k
                && g.elem == elem
                && g.kind == a.kind
        });
        match found {
            Some(g) => g.members.push((a, index.konst)),
            None => building.push(Building {
                base: *base,
                terms: &index.terms,
                k,
                elem,
                kind: a.kind,
                members: vec![(a, index.konst)],
            }),
        }
    }
    let mut groups = Vec::with_capacity(building.len());
    let mut member: FxHashMap<AccessKey, (usize, u32)> = FxHashMap::default();
    for g in building {
        let c0 = g
            .members
            .iter()
            .map(|(_, c)| *c)
            .min()
            .expect("a group is built from at least one access");
        let mut seen = vec![false; g.k as usize];
        for (a, c) in &g.members {
            let r = c - c0;
            if r >= g.k {
                return Err("an interleaved access group spans more than one stride");
            }
            if seen[r as usize] && g.kind == AccessKind::Store {
                return Err("an interleaved store group writes one element twice");
            }
            seen[r as usize] = true;
            member.insert((a.block.0, a.inst), (groups.len(), r as u32));
        }
        if seen.iter().any(|s| !s) {
            return Err("an interleaved access group has a gap");
        }
        groups.push(InterleaveGroup {
            k: g.k as u32,
            elem: g.elem.clone(),
        });
    }
    Ok((groups, member))
}

/// A group is emitted at ONE point: a load group at its first member in program order, a store
/// group at its last. Every other member moves there — a load earlier, a store later — so no move
/// may reverse the order of two accesses to the same element, or the iteration would read a value
/// before the store it read, or store under a later access's feet.
///
/// The check is pairwise over **where each access ends up**, not where it started: a member is
/// placed at its group's anchor, everything else stays put, and every pair that may touch one
/// element (at least one of them a store) must keep its program order. Checking each member only
/// against the accesses it passes at their *original* places is not enough, because two groups can
/// move toward each other: `x[2i] = a; t = x[2i + 1]; x[2i + 1] = b; u = x[2i];` moves the first
/// store down to the last one and the last load up to the first, and those two swap although
/// neither crosses the other's starting point.
///
/// Only same-object pairs can conflict: accesses to different objects are already proved disjoint,
/// or guarded disjoint at run time, by the dependence verdict this pass requires. A same-object pair
/// is harmless when the GCD test proves the two never touch one element
/// ([`crate::loop_info::never_the_same_element`]). The groups are also only formed in a straight-line
/// body: an if-converted one would make the emission point conditional.
fn group_order_verdict(
    l: &NaturalLoop,
    lin: &[LinInst],
    groups: usize,
    member: &FxHashMap<AccessKey, (usize, u32)>,
) -> Result<(), &'static str> {
    if groups == 0 {
        return Ok(());
    }
    if lin.iter().any(|s| !matches!(s, LinInst::Real { .. })) {
        return Err("an interleave group in an if-converted body");
    }
    let pos: FxHashMap<AccessKey, usize> = lin
        .iter()
        .enumerate()
        .filter_map(|(at, s)| match s {
            LinInst::Real { block, idx } => Some(((block.0, *idx), at)),
            _ => None,
        })
        .collect();
    let at_of = |a: &MemAccess| pos.get(&(a.block.0, a.inst)).copied();
    let group_of = |a: &MemAccess| member.get(&(a.block.0, a.inst)).map(|m| m.0);
    // Where each group is emitted: a load group at its first member, a store group at its last.
    let mut anchor: Vec<Option<usize>> = vec![None; groups];
    for a in &l.accesses {
        let Some(g) = group_of(a) else {
            continue;
        };
        let p = at_of(a).ok_or("an interleave group member is not in the body")?;
        anchor[g] = Some(match (anchor[g], a.kind) {
            (None, _) => p,
            (Some(q), AccessKind::Load) => q.min(p),
            (Some(q), AccessKind::Store) => q.max(p),
        });
    }
    // Every access as (where it is, where it is emitted).
    let placed: Vec<Option<(usize, usize)>> = l
        .accesses
        .iter()
        .map(|a| {
            let p = at_of(a)?;
            Some((p, group_of(a).and_then(|g| anchor[g]).unwrap_or(p)))
        })
        .collect();
    for (i, a) in l.accesses.iter().enumerate() {
        let Some((pa, ea)) = placed[i] else {
            continue;
        };
        for (j, b) in l.accesses.iter().enumerate().skip(i + 1) {
            let Some((pb, eb)) = placed[j] else {
                continue;
            };
            if a.kind == AccessKind::Load && b.kind == AccessKind::Load {
                continue;
            }
            // One group is one emission: its members never conflict with each other (a store
            // group writes each element once, a load group only reads).
            let ga = group_of(a);
            if ga.is_some() && ga == group_of(b) {
                continue;
            }
            if ea != eb && (pa < pb) == (ea < eb) {
                continue;
            }
            let same_object = match (a.addr.base(), b.addr.base()) {
                (Some(x), Some(y)) => x.same_object(y),
                _ => true,
            };
            if same_object && !crate::loop_info::never_the_same_element(a, b) {
                return Err(
                    "an interleave group member would move across an access to its element",
                );
            }
        }
    }
    Ok(())
}

/// What [`plan_body`] decides for one loop body.
struct BodyPlan {
    /// One entry per element of `lin`, in order.
    plans: Vec<Plan>,
    /// Every lane-varying value and its lane type — widened vectors and replicated lane values
    /// alike, so the dataflow treats both as varying.
    vector_ty: FxHashMap<u32, MirType>,
    reductions: Vec<RedPlan>,
    /// The values computed lane by lane ([`Plan::Replicate`]) rather than as vectors. Their lane
    /// type joins the group width only when something widened reads them (`packed`).
    lane_vals: FxHashSet<u32>,
    /// Lane types of replicated values a widened step reads, which are packed into vectors there.
    packed: Vec<MirType>,
    /// Does any widened (not replicated) step read the induction variable as a value? Only then is
    /// the lane-ramp vector built.
    ramp_vector: bool,
    /// Steps that do real work lane by lane: every gather, and every replicated value something
    /// other than an address reads. An index computed per lane only to feed per-lane addresses is
    /// not counted — the scalar loop computes it too, and nothing packs it.
    replicated: usize,
    /// Every call of the body: its step in `lin`, and what the effect summaries say it does
    /// ([`call_verdict`]).
    calls: Vec<(usize, CallEffect)>,
}

/// Decide, for every instruction of the body, whether it stays scalar, is widened or is replicated
/// lane by lane — and check that every one *can* be. Returns `None` the moment one cannot.
///
/// The decisions do not depend on the group width: whether a value is a vector, and which lane type
/// it has, follows from the dataflow alone, so [`plan_loop`] picks the width from this result.
///
/// **Replication** (LLVM VPlan's REPLICATE recipe) is how a body with a step that has no vector
/// form still widens everything else. The step runs once per lane on the lane's own scalar
/// operands — extracted from a vector, or taken per lane from another replicated value — and its
/// results are packed back into vectors (`Op::InsertLane`) only where a widened step reads them.
/// Four things are replicated:
///
/// * a **gather** — a `gep` whose index varies per lane becomes one address per lane
///   ([`Plan::LaneGep`]), and the load through it one scalar load per lane ([`Plan::Gather`]).
///   Every lane performs exactly the load scalar iteration `i + k` performs, so nothing is
///   speculated (`linearize_region` keeps a gather out of an if-converted arm), and
///   `loop_info::dependence` has already proved no store in the loop can reach the gathered object;
/// * an integer **divide or remainder**, an `f32`/`f64` remainder and a **per-lane shift** — no
///   packed form exists, and MIR's scalar forms never trap (`x / 0` is `0`);
/// * a **cast with no vector route**, and any `Bin`/`Cast` whose only readers are `gep` indices: an
///   index widened to `i64` only ever feeds per-lane addresses, so computing it per lane keeps an
///   `i64` value out of the group width at no cost;
/// * a **call** to a function the effect summaries prove side-effect free ([`Plan::Call`]).
///
/// A **scatter** — a store through a per-lane address — is replicated too ([`Plan::Scatter`]): one
/// scalar store per lane, in lane order. `loop_info::dependence` has proved that nothing else in
/// the loop, load or store, can touch the scattered object, so the stores' order relative to the
/// rest of the body is unobservable and their order among themselves is iteration order. A scatter
/// whose stored value is itself a per-lane address is declined, as is any other use of one. A
/// scatter inside an if-converted arm is declined too: the predicated store forms need an affine
/// address. Whether replicating is *worth* it is [`plan_loop`]'s call, from the `replicated` count
/// this returns.
fn plan_body(
    f: &Function,
    l: &NaturalLoop,
    lin: &[LinInst],
    ramp: &IvRamp,
    member: &FxHashMap<AccessKey, (usize, u32)>,
    effects: Option<&CallEffects>,
    region: Option<SerialRegion>,
) -> Option<BodyPlan> {
    let access_at: FxHashMap<(u32, usize), &MemAccess> = l
        .accesses
        .iter()
        .map(|a| ((a.block.0, a.inst), a))
        .collect();

    // The reductions, keyed by their combining instruction's result.
    let mut red_of: FxHashSet<u32> = FxHashSet::default();
    // A min/max reduction's compare: part of the accumulate, folded with it, never widened alone.
    let mut red_cmp: FxHashSet<u32> = FxHashSet::default();
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
        // A FindLast's "combine" is the counter itself, which is not an instruction of the body.
        if r.kind != RedKind::FindLast {
            red_of.insert(r.combine.0);
        }
        if let Some(s) = r.select {
            red_cmp.insert(s.cmp.0);
        }
        // A guarded accumulate is folded where if-conversion merged it: find that merge, and the
        // side of it the update arrives on. A merge outside the flattened body cannot be widened.
        let guard = match r.merge {
            None => None,
            Some(m) => {
                let (cond, then_arg, else_arg) = lin.iter().find_map(|s| match s {
                    LinInst::Merge {
                        param,
                        cond,
                        then_arg,
                        else_arg,
                    } if *param == m => Some((*cond, *then_arg, *else_arg)),
                    _ => None,
                })?;
                let on_true = if then_arg == r.combine && else_arg == r.param {
                    true
                } else if else_arg == r.combine && then_arg == r.param {
                    false
                } else {
                    return None;
                };
                red_of.insert(m.0);
                Some((cond, on_true))
            }
        };
        // Lane-parallel partials need the combine to exist *as a vector instruction*. It is the
        // same `Op::Bin` the elementwise path emits, so it is the same question `bin_widenable`
        // answers — an `i8` product has no packed form, and a reduction is not exempt from that.
        // Such a reduction still widens; it just takes the serial fold, whose combine is scalar.
        // An argmax/argmin keeps per-lane `(value, index)` partials, which reproduce the scalar
        // loop's answer only under two conditions ([`emit_arg_combine`] has the argument): the
        // update is the `then` arm, so a NaN (every ordered compare false) never updates; and the
        // index is this loop's counter, so a later iteration always records a later index. A strict
        // compare keeps the EARLIEST of tied elements and a non-strict one the LAST
        // ([`RedPlan::arg_last`]); the combine breaks value ties the same way. Anything else stays
        // scalar.
        let mut arg_last = false;
        let arg = match r.arg {
            None => None,
            Some(a) => {
                let s = r.select?;
                arg_last = matches!(
                    s.pred,
                    CmpOp::Sge | CmpOp::Sle | CmpOp::Uge | CmpOp::Ule | CmpOp::Foge | CmpOp::Fole
                );
                let value_ok = match &ty {
                    MirType::F32 | MirType::F64 => true,
                    t => t.is_int() && lane_bytes(t).is_some_and(|b| b <= 4),
                };
                if s.acc_is_then || a.index != ramp.iv || !value_ok {
                    return None;
                }
                red_of.insert(a.merge.0);
                Some((a.param_index, f.value_type(a.param).clone()))
            }
        };
        // FindLastIV records this loop's own counter; any other induction variable stays scalar.
        let find_last = r.kind == RedKind::FindLast;
        if find_last && (r.combine != ramp.iv || guard.is_none()) {
            return None;
        }
        let vector_acc = arg.is_some()
            || find_last
            || (r.reassociable
                && r.addend.is_some()
                && red_identity(r.kind, &ty).is_some()
                && vector_combinable(r.kind, &ty));
        reductions.push(RedPlan {
            param_index: r.param_index,
            kind: r.kind,
            addend,
            fma_factors: r.fma_factors,
            select: r.select,
            guard,
            ty,
            vector_acc,
            arg,
            arg_last,
            find_last,
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

    // Values read only as a `gep` index (every reader is the index operand of a `gep`): computing
    // one of those per lane is free when the `gep` is a gather, and keeps it out of the width.
    let mut index_only: FxHashMap<u32, bool> = FxHashMap::default();
    for step in lin {
        match step {
            LinInst::Real { block, idx } => match &f.blocks[block.0 as usize].insts[*idx].op {
                Op::Gep { ptr, index, .. } => {
                    index_only.entry(index.0).or_insert(true);
                    index_only.insert(ptr.0, false);
                }
                op => each_op_use(op, &mut |u| {
                    index_only.insert(u.0, false);
                }),
            },
            LinInst::MaskedStore {
                ptr,
                cond,
                then_val,
                else_val,
                ..
            } => {
                for u in [*ptr, *cond, *then_val, *else_val] {
                    index_only.insert(u.0, false);
                }
            }
            LinInst::Merge {
                cond,
                then_arg,
                else_arg,
                ..
            } => {
                for u in [*cond, *then_arg, *else_arg] {
                    index_only.insert(u.0, false);
                }
            }
            LinInst::PredLoad { block, idx, pred } => {
                for (c, _) in pred {
                    index_only.insert(c.0, false);
                }
                each_op_use(&f.blocks[block.0 as usize].insts[*idx].op, &mut |u| {
                    index_only.insert(u.0, false);
                });
            }
            LinInst::PredStore {
                ptr, value, cond, ..
            } => {
                for u in [*ptr, *value, *cond] {
                    index_only.insert(u.0, false);
                }
            }
            LinInst::TreeStore { ptr, tree, .. } => {
                let (mut conds, mut vals) = (Vec::new(), Vec::new());
                tree.operands(&mut conds, &mut vals);
                for u in std::iter::once(ptr).chain(&conds).chain(&vals) {
                    index_only.insert(u.0, false);
                }
            }
        }
    }
    for b in &l.blocks {
        crate::each_term_use(&f.blocks[b.0 as usize].term, &mut |u| {
            index_only.insert(u.0, false);
        });
    }
    let only_indexes = |v: ValueId| index_only.get(&v.0).copied().unwrap_or(false);

    // Forward dataflow over the flattened body: a value is lane-varying iff it is a widened load or
    // has a lane-varying operand. Loop-invariant values and the induction variable are uniform.
    let mut vector_ty: FxHashMap<u32, MirType> = FxHashMap::default();
    let mut plans: Vec<Plan> = Vec::with_capacity(lin.len());
    let mut lane_vals: FxHashSet<u32> = FxHashSet::default();
    // Per-lane addresses (`Plan::LaneGep` results): readable only as a gather's address.
    let mut lane_ptrs: FxHashSet<u32> = FxHashSet::default();
    let mut packed: Vec<MirType> = Vec::new();
    let mut ramp_vector = false;
    let mut replicated = 0usize;
    let mut calls: Vec<(usize, CallEffect)> = Vec::new();
    // The values the lane-serial region computes ([`SerialRegion`]), and the steps of the listed
    // accesses.
    let mut serial_vals: FxHashSet<u32> = FxHashSet::default();
    let listed: FxHashSet<usize> = l
        .dep
        .serial
        .iter()
        .filter_map(|&k| {
            let a = &l.accesses[k];
            lin.iter().position(|s| {
                matches!(s, LinInst::Real { block, idx } if *block == a.block && *idx == a.inst)
            })
        })
        .collect();
    // The induction variable and every value the body computes from it while it is still uniform
    // (`i * 2`, `gep(x, i)`) — see the check after the plan below.
    let mut iv_addr: FxHashSet<u32> = FxHashSet::default();
    iv_addr.insert(ramp.iv.0);
    let is_vec = |m: &FxHashMap<u32, MirType>, v: ValueId| m.contains_key(&v.0);
    // The lane type an operand contributes *at this step*, or `None` when it is uniform. The
    // induction variable is the one value whose answer depends on where it is read — uniform as an
    // address, a `W`-lane ramp as a value — so this takes the step index and `is_vec` does not.
    let lane_at = |m: &FxHashMap<u32, MirType>, at: usize, v: ValueId| -> Option<MirType> {
        match m.get(&v.0) {
            Some(t) => Some(t.clone()),
            None if ramp.reads_as_value(at, v) => Some(ramp.ty.clone()),
            None => None,
        }
    };

    for (at, step) in lin.iter().enumerate() {
        // The lane-serial region is straight-line code: an if-converted step has no lane order.
        if region.is_some_and(|r| r.contains(at)) && !matches!(step, LinInst::Real { .. }) {
            return None;
        }
        // A synthesized step (an if-converted store or merge) reads values the arms computed and,
        // for a merge, defines the latch parameter the rest of the body reads.
        let (block, i, inst) = match step {
            LinInst::Real { block, idx } => (
                block.0,
                *idx,
                f.blocks[block.0 as usize].insts[*idx].clone(),
            ),
            LinInst::MaskedStore {
                ptr,
                cond,
                then_val,
                else_val,
                at,
            } => {
                for u in [*ptr, *cond, *then_val, *else_val] {
                    if defined_in_loop.contains(&u.0) && !available.contains(&u.0) {
                        return None;
                    }
                    if lane_ptrs.contains(&u.0) {
                        return None;
                    }
                }
                for u in [*then_val, *else_val] {
                    if lane_vals.contains(&u.0) {
                        packed.push(f.value_type(u).clone());
                    }
                }
                let addr = access_at.get(&(at.0 .0, at.1)).map(|a| &a.addr);
                if !matches!(addr, Some(AddrForm::Affine { .. })) {
                    return None;
                }
                let vt = f.value_type(*then_val).clone();
                if vt != *f.value_type(*else_val) || lane_bytes(&vt).is_none() {
                    return None;
                }
                // A predicated store must be a *lane* selection: with a uniform `i1` condition
                // there is no per-lane mask, and blending two vectors with it is not expressible.
                if !is_vec(&vector_ty, *cond)
                    || (!is_vec(&vector_ty, *then_val) && !is_vec(&vector_ty, *else_val))
                {
                    return None;
                }
                plans.push(Plan::MaskedStore);
                continue;
            }
            LinInst::Merge {
                param,
                cond,
                then_arg,
                else_arg,
            } => {
                // A reduction's merge — the join of a guarded accumulate, or the branch a min/max is
                // written as — is folded with its reduction, after the body. It is never
                // `available`: nothing but the back edge reads it (`loop_info` counted).
                if red_of.contains(&param.0) {
                    plans.push(Plan::Reduce);
                    continue;
                }
                // A min/max's compare is re-emitted by its fold and never widened, so it cannot
                // steer anything else. (`loop_info` only takes a branch as a min/max when the branch
                // chooses nothing else; this keeps the invariant local.)
                if red_cmp.contains(&cond.0) {
                    return None;
                }
                for u in [*cond, *then_arg, *else_arg] {
                    if defined_in_loop.contains(&u.0) && !available.contains(&u.0) {
                        return None;
                    }
                    if lane_ptrs.contains(&u.0) {
                        return None;
                    }
                }
                for u in [*then_arg, *else_arg] {
                    if lane_vals.contains(&u.0) {
                        packed.push(f.value_type(u).clone());
                    }
                }
                let ty = f.value_type(*param).clone();
                // The merged parameter is lane-varying when the mask is, or when either arm is —
                // and in both cases the condition has to be a real lane mask, since a uniform `i1`
                // cannot blend. Two *uniform* arms under a lane mask are still lane-varying
                // (`emit_body` splats each and selects), which is exactly the shape a label-
                // smoothing `if c == t { qt } else { qo }` takes after if-conversion.
                let vec_cond = is_vec(&vector_ty, *cond);
                if vec_cond || is_vec(&vector_ty, *then_arg) || is_vec(&vector_ty, *else_arg) {
                    if !vec_cond || lane_bytes(&ty).is_none() {
                        return None;
                    }
                    vector_ty.insert(param.0, ty);
                }
                available.insert(param.0);
                plans.push(Plan::Merge);
                continue;
            }
            // A load only one arm makes: one scalar load per lane at the lane's own element
            // (`ptr + k`) or the scratch slot, so the address must be the group's one scalar address
            // and the condition a mask (or one answer for the group). Its lanes are packed wherever
            // a widened step reads them, like any replicated value.
            LinInst::PredLoad { block, idx, pred } => {
                let inst = &f.blocks[block.0 as usize].insts[*idx];
                let Op::Load(ptr, ty) = &inst.op else {
                    return None;
                };
                for u in std::iter::once(*ptr).chain(pred.iter().map(|(c, _)| *c)) {
                    if (defined_in_loop.contains(&u.0) && !available.contains(&u.0))
                        || lane_ptrs.contains(&u.0)
                    {
                        return None;
                    }
                }
                if lane_at(&vector_ty, at, *ptr).is_some()
                    || pred.iter().any(|(c, _)| red_cmp.contains(&c.0))
                {
                    return None;
                }
                if !matches!(
                    access_at.get(&(block.0, *idx)).map(|a| &a.addr),
                    Some(AddrForm::Affine { .. })
                ) {
                    return None;
                }
                lane_bytes(ty)?;
                let r = inst.result?;
                vector_ty.insert(r.0, ty.clone());
                lane_vals.insert(r.0);
                available.insert(r.0);
                replicated += 1;
                plans.push(Plan::PredLoad);
                continue;
            }
            // A store only one arm makes: per lane, to the lane's own element or the scratch slot.
            // The stored value is read lane by lane, so it may be a vector, replicated lanes or one
            // value for the group — but never the induction variable's scalar address face.
            LinInst::PredStore {
                ptr,
                value,
                cond,
                at: store_at,
                ..
            } => {
                for u in [*ptr, *value, *cond] {
                    if (defined_in_loop.contains(&u.0) && !available.contains(&u.0))
                        || lane_ptrs.contains(&u.0)
                    {
                        return None;
                    }
                }
                if lane_at(&vector_ty, at, *ptr).is_some()
                    || red_cmp.contains(&cond.0)
                    || iv_addr.contains(&value.0)
                {
                    return None;
                }
                if !matches!(
                    access_at.get(&(store_at.0 .0, store_at.1)).map(|a| &a.addr),
                    Some(AddrForm::Affine { .. })
                ) {
                    return None;
                }
                lane_bytes(f.value_type(*value))?;
                replicated += 1;
                plans.push(Plan::PredStore);
                continue;
            }
            // A nested region's stores, one address: a vector store of nested selects when every
            // path stores (each condition must then be a real lane mask, since a uniform `i1`
            // cannot blend vectors), a per-lane predicated store otherwise.
            LinInst::TreeStore {
                ptr,
                tree,
                at: store_at,
            } => {
                let (mut conds, mut vals) = (Vec::new(), Vec::new());
                tree.operands(&mut conds, &mut vals);
                for &u in std::iter::once(ptr).chain(&conds).chain(&vals) {
                    if (defined_in_loop.contains(&u.0) && !available.contains(&u.0))
                        || lane_ptrs.contains(&u.0)
                    {
                        return None;
                    }
                }
                if lane_at(&vector_ty, at, *ptr).is_some()
                    || conds.iter().any(|c| red_cmp.contains(&c.0))
                    || vals.iter().any(|v| iv_addr.contains(&v.0))
                {
                    return None;
                }
                if !matches!(
                    access_at.get(&(store_at.0 .0, store_at.1)).map(|a| &a.addr),
                    Some(AddrForm::Affine { .. })
                ) {
                    return None;
                }
                let vt = f.value_type(*vals.first()?).clone();
                if vals.iter().any(|v| *f.value_type(*v) != vt) {
                    return None;
                }
                lane_bytes(&vt)?;
                if tree.complete() {
                    if !conds.iter().all(|c| is_vec(&vector_ty, *c)) {
                        return None;
                    }
                    for v in &vals {
                        if lane_vals.contains(&v.0) {
                            packed.push(vt.clone());
                        }
                    }
                    plans.push(Plan::TreeStore);
                } else {
                    replicated += 1;
                    plans.push(Plan::TreePredStore);
                }
                continue;
            }
        };
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
            // A min/max's compare is re-emitted by the fold, beside its select — its only reader
            // (`loop_info::classify_carried` counted). Available, so that select reads as defined.
            if red_cmp.contains(&r.0) {
                plans.push(Plan::Reduce);
                continue;
            }
        }
        let mut operand_is_vec = false;
        let mut reads_lane_ptr = false;
        each_op_use(&inst.op, &mut |u| {
            if lane_at(&vector_ty, at, u).is_some() {
                operand_is_vec = true;
            }
            reads_lane_ptr |= lane_ptrs.contains(&u.0);
        });
        // A per-lane address may only be the address of a gather or of a scatter: anything else —
        // a scatter that stores the address itself included — would need the address as a value.
        if reads_lane_ptr
            && !matches!(&inst.op, Op::Load(p, _) if lane_ptrs.contains(&p.0))
            && !matches!(&inst.op, Op::Store { ptr, value }
                if lane_ptrs.contains(&ptr.0) && !lane_ptrs.contains(&value.0))
        {
            return None;
        }
        let plan = match &inst.op {
            Op::Load(p, ty) if lane_ptrs.contains(&p.0) => {
                lane_bytes(ty)?;
                vector_ty.insert(inst.result?.0, ty.clone());
                replicated += 1;
                Plan::Gather
            }
            // A scatter: one scalar store per lane, in lane order. The stored value is read lane by
            // lane, so it may be a vector, replicated lanes, the counter's ramp or one value for
            // the group.
            Op::Store { ptr, value } if lane_ptrs.contains(&ptr.0) => {
                lane_bytes(f.value_type(*value))?;
                replicated += 1;
                Plan::Scatter
            }
            Op::Load(_, ty) => match access_at.get(&(block, i)).map(|a| &a.addr) {
                Some(AddrForm::Affine { .. }) => {
                    lane_bytes(ty)?;
                    vector_ty.insert(inst.result?.0, ty.clone());
                    match member.get(&(block, i)) {
                        Some(&(group, offset)) => Plan::GroupLoad { group, offset },
                        None => Plan::VecLoad,
                    }
                }
                // An invariant address yields the same value every iteration: keep it scalar and
                // let a consumer splat it. This is how a `[]T` slice's data pointer is read — and
                // when that is what it is, it is hoisted out of the vector body entirely.
                Some(AddrForm::Invariant(_)) => {
                    if *ty == MirType::Ptr {
                        Plan::InvariantPtrLoad
                    } else {
                        Plan::Scalar
                    }
                }
                _ => return None,
            },
            Op::Store { ptr: _, value } => match access_at.get(&(block, i)).map(|a| &a.addr) {
                Some(AddrForm::Affine { .. }) => {
                    lane_bytes(f.value_type(*value))?;
                    match member.get(&(block, i)) {
                        Some(&(group, offset)) => Plan::GroupStore { group, offset },
                        None => Plan::VecStore,
                    }
                }
                // Every lane would write the same address; only one wins, and which one is not the
                // scalar loop's answer.
                _ => return None,
            },
            // Address arithmetic stays scalar, except a gather's: a `gep` off a uniform base whose
            // index varies per lane becomes one address per lane.
            Op::Gep { ptr, index, .. } => {
                if !operand_is_vec {
                    Plan::Scalar
                } else if lane_at(&vector_ty, at, *ptr).is_none()
                    && lane_at(&vector_ty, at, *index).is_some()
                {
                    lane_ptrs.insert(inst.result?.0);
                    Plan::LaneGep
                } else {
                    return None;
                }
            }
            Op::ConstInt(..) | Op::ConstFloat(..) => Plan::Scalar,
            Op::Bin(op, lhs, rhs) => {
                if !operand_is_vec {
                    Plan::Scalar
                } else {
                    let r = inst.result?;
                    let ty = f.value_type(r).clone();
                    lane_bytes(&ty)?;
                    let widenable = match op {
                        // A shift by an amount that is the same in every lane keeps that amount
                        // scalar — `Op::Bin`'s vector-by-scalar shape, which is what every vector
                        // ISA's shift takes. A per-lane amount (`x[i] << y[i]`) has no such form
                        // and is replicated.
                        BinOp::Shl | BinOp::LShr | BinOp::AShr => {
                            lane_at(&vector_ty, at, *lhs).is_some()
                                && lane_at(&vector_ty, at, *rhs).is_none()
                        }
                        // x86 has no packed byte multiply; `emit_bin` goes through `i16` halves.
                        BinOp::Mul if ty == MirType::I8 => true,
                        _ => bin_widenable(*op, &ty),
                    };
                    vector_ty.insert(r.0, ty);
                    if !widenable || only_indexes(r) {
                        lane_vals.insert(r.0);
                        if !only_indexes(r) {
                            replicated += 1;
                        }
                        Plan::Replicate
                    } else {
                        Plan::Widen
                    }
                }
            }
            Op::Cmp(..) => {
                if !operand_is_vec {
                    Plan::Scalar
                } else {
                    // The operand lane type decides the mask width, not the `i1` result type. Both
                    // operands of a `Cmp` share one MIR type, so which vector operand answers is
                    // immaterial — including when it is the induction variable's ramp, which is
                    // how `if c == target` becomes a per-lane mask instead of one scalar bool.
                    let mut lane: Option<MirType> = None;
                    let mut compares_masks = false;
                    each_op_use(&inst.op, &mut |u| {
                        if let Some(t) = lane_at(&vector_ty, at, u) {
                            lane = Some(t);
                        }
                        compares_masks |= *f.value_type(u) == MirType::I1;
                    });
                    // Two masks are not two values: the interpreter's lanes are `0`/`1` and the
                    // native ones `0`/`-1`, which agree under a `select` and nowhere else.
                    if compares_masks {
                        return None;
                    }
                    let lane = lane?;
                    lane_bytes(&lane)?;
                    vector_ty.insert(inst.result?.0, mask_lane(&lane));
                    Plan::Widen
                }
            }
            Op::Select(c, ..) => {
                if !operand_is_vec {
                    Plan::Scalar
                } else {
                    // A uniform `i1` condition cannot select between vectors: the mask has to be a
                    // per-lane all-ones/all-zeros vector, which only a vector compare produces.
                    //
                    // The *arms*, by contrast, need not already be vectors. `select(mask, k1, k2)`
                    // with two loop-invariant operands is a perfectly ordinary lane-varying value —
                    // `emit_widened` splats whichever arm is not already a vector, through
                    // `to_vector`, exactly as it does for a partly-uniform `Bin`/`Fma`/`Cmp` — and
                    // `select(mask, splat(a), splat(b))` computes in each lane what the scalar loop
                    // computed. Requiring both arms to be vectors declined a whole family of
                    // bodies: a `relu` is `fmax(x[i], 0.0)`, i.e. `select(x[i] > 0, x[i], 0.0)`,
                    // whose `else` arm is a literal, and every table-driven transcendental the
                    // front end inlines picks its coefficients with `select(x > hi, hi_const, x)`
                    // and `select(bit, table_a, table_b)` over f32 literals — so the first widened
                    // compare in the body ran straight into an arm that was a constant, and one
                    // such `select` made the entire loop unwidenable.
                    if !is_vec(&vector_ty, *c) {
                        return None;
                    }
                    let ty = f.value_type(inst.result?).clone();
                    lane_bytes(&ty)?;
                    vector_ty.insert(inst.result?.0, ty);
                    Plan::Widen
                }
            }
            Op::Neg(_) | Op::Not(_) | Op::Fma(..) | Op::Sqrt(_) | Op::Round(..) => {
                if !operand_is_vec {
                    Plan::Scalar
                } else {
                    let ty = f.value_type(inst.result?).clone();
                    lane_bytes(&ty)?;
                    vector_ty.insert(inst.result?.0, ty);
                    Plan::Widen
                }
            }
            Op::Cast(kind, v, to) => {
                if !operand_is_vec {
                    Plan::Scalar
                } else {
                    // A mask is not a value (see the `Cmp` arm), so a bool-to-int cast of one
                    // stays declined.
                    if *f.value_type(*v) == MirType::I1 {
                        return None;
                    }
                    let from = lane_at(&vector_ty, at, *v)?;
                    lane_bytes(to)?;
                    let r = inst.result?;
                    vector_ty.insert(r.0, to.clone());
                    match cast_route(*kind, &from, to) {
                        Some(route) if !only_indexes(r) => Plan::Cast(route),
                        // No vector form (an `f64 -> i64` conversion), or a value only addresses
                        // read: one scalar cast per lane.
                        _ if !matches!(kind, CastKind::IntToPtr | CastKind::PtrToInt) => {
                            lane_vals.insert(r.0);
                            if !only_indexes(r) {
                                replicated += 1;
                            }
                            Plan::Replicate
                        }
                        _ => return None,
                    }
                }
            }
            // A call to a function the effect summaries prove side-effect free (`crate::effects`):
            // it writes nothing the caller can see, so running it once per lane — lane order is
            // iteration order — on the lane's own arguments is the scalar loop's sequence of calls.
            // With arguments that are one value for the group it is one call per group: the same
            // function of the same arguments, reading memory no iteration of the loop writes
            // (`loop_info::dependence` refuses a reading call beside any store). Whether it may
            // trap or never return, and so whether it may be moved ahead of a lane's other steps,
            // is [`call_verdict`]'s question. Any other call may do anything.
            Op::Call { func, .. } => {
                let effect = effects.and_then(|e| e.get(*func))?;
                calls.push((at, effect));
                if !operand_is_vec {
                    Plan::Scalar
                } else {
                    // The results are packed where a widened step reads them, so they need a lane
                    // type; a call for its trap alone (no result) packs nothing.
                    if let Some(r) = inst.result {
                        let ty = f.value_type(r).clone();
                        lane_bytes(&ty)?;
                        vector_ty.insert(r.0, ty);
                        lane_vals.insert(r.0);
                    }
                    replicated += 1;
                    Plan::Call
                }
            }
            // An alloca inside the body would allocate per lane group; a splat/veckernel means
            // someone already vectorized this.
            _ => return None,
        };
        // Inside the lane-serial region, every memory access and every step reading what the region
        // computed runs lane by lane ([`SerialRegion`]); the region's other steps are pure and keep
        // their plans, running ahead of the lanes. A slice re-reading its data pointer is neither:
        // the loop writes no slice header (`MemDep::IndependentIfBasesStable`), so that read is the
        // same wherever it runs.
        let plan = match region {
            Some(r) if r.contains(at) => {
                let mut reads_region = false;
                each_op_use(&inst.op, &mut |u| {
                    reads_region |= serial_vals.contains(&u.0)
                });
                let memory = matches!(inst.op, Op::Load(..) | Op::Store { .. })
                    && !matches!(plan, Plan::InvariantPtrLoad);
                if !(memory || reads_region || listed.contains(&at)) {
                    plan
                } else {
                    let serial = match plan {
                        // Not counted as lane-by-lane work when the plan already was.
                        Plan::VecLoad | Plan::VecStore => {
                            replicated += 1;
                            Plan::Serial { contiguous: true }
                        }
                        Plan::Widen | Plan::Cast(_) => {
                            replicated += 1;
                            Plan::Serial { contiguous: false }
                        }
                        Plan::Replicate
                        | Plan::LaneGep
                        | Plan::Gather
                        | Plan::Scatter
                        | Plan::Call => Plan::Serial { contiguous: false },
                        // Folded after the body like any accumulate; a value the region computed
                        // reaches it packed.
                        Plan::Reduce => Plan::Reduce,
                        // An invariant data access, an interleave group member: no lane order.
                        _ => return None,
                    };
                    if let (Some(r), Plan::Serial { .. }) = (inst.result, &serial) {
                        // Lane values of the instruction's own type: a compare's lanes are `i1`s
                        // here, never a mask, and a widened step after the region packs them.
                        vector_ty.insert(r.0, f.value_type(r).clone());
                        lane_vals.insert(r.0);
                        serial_vals.insert(r.0);
                    }
                    serial
                }
            }
            _ => plan,
        };
        // The induction variable's *address* face is one scalar for the whole group, and so is
        // every value computed from it while it stays scalar (`i * 2`, `gep(x, i)`): exactly right
        // where a widened access takes it as its base, wrong anywhere a lane needs its own
        // iteration's value. `classify_iv_uses` keeps such values out of every value role, which
        // leaves one way to mix one into a lane-varying step: an address that is itself gathered,
        // `x[i + q[i]]` or a row `x[i][q[i]]`, where every lane would see the group's first `i`.
        // Only the ramp (`i + k` in lane `k`; `lane_at` answers it) may meet a lane-varying
        // operand.
        if matches!(plan, Plan::Scalar) {
            let mut derived = false;
            each_op_use(&inst.op, &mut |u| derived |= iv_addr.contains(&u.0));
            if derived {
                if let Some(r) = inst.result {
                    iv_addr.insert(r.0);
                }
            }
        } else {
            // A widened access's own address is the one place the scalar face is the right one.
            let address = match &inst.op {
                Op::Load(p, _) => Some(*p),
                Op::Store { ptr, .. } => Some(*ptr),
                _ => None,
            };
            let mut mixes = false;
            each_op_use(&inst.op, &mut |u| {
                mixes |= Some(u) != address
                    && iv_addr.contains(&u.0)
                    && lane_at(&vector_ty, at, u).is_none();
            });
            if mixes {
                return None;
            }
        }
        // A widened step reading a replicated value packs it into vectors, so its lane type joins
        // the group width. A scatter or a call reads its operands lane by lane and packs nothing.
        if !matches!(
            plan,
            Plan::Replicate
                | Plan::LaneGep
                | Plan::Gather
                | Plan::Scatter
                | Plan::Call
                | Plan::Serial { .. }
        ) {
            each_op_use(&inst.op, &mut |u| {
                if lane_vals.contains(&u.0) {
                    packed.push(f.value_type(u).clone());
                }
            });
        }
        // A replicated step, a scatter or a call takes the counter's `i + k` lane by lane, not the
        // ramp.
        if ramp.value_use.contains(&at)
            && !matches!(
                plan,
                Plan::Replicate | Plan::Scatter | Plan::Call | Plan::Serial { .. }
            )
        {
            ramp_vector = true;
        }
        plans.push(plan);
    }

    // Every reduction must actually be foldable: its inputs have to be vectors, or the "extract W
    // lanes and accumulate serially" step has nothing to extract.
    //
    // A guarded accumulate is the exception for its addend, which may be uniform — `count = count
    // + 1` under `if x[i] > t` adds the same value in every lane that took the branch — as long as
    // the vector body has it (defined outside the loop, or reproduced as a scalar). Its condition
    // must be a real lane mask: a uniform `i1` cannot blend vectors.
    for r in &reductions {
        lane_bytes(&r.ty)?;
        // A bf16 or f16 accumulate rounds to the grid inside its own recurrence (`fptrunc` after
        // the combine), which no fold form here reproduces.
        if is_half(&r.ty) {
            return None;
        }
        if let Some((cond, _)) = r.guard {
            if !is_vec(&vector_ty, cond) {
                return None;
            }
        }
        match (r.addend, r.fma_factors) {
            // A FindLast's addend is the counter, which it reads through the lane ramp.
            _ if r.find_last => {}
            (Some((a, _)), None) => {
                // Never the induction variable's address face, nor anything computed from it while
                // it was uniform: those are one scalar for the group but differ per lane.
                let uniform_ok = r.guard.is_some()
                    && (!defined_in_loop.contains(&a.0) || available.contains(&a.0))
                    && !red_cmp.contains(&a.0)
                    && !iv_addr.contains(&a.0);
                if !is_vec(&vector_ty, a) && !uniform_ok {
                    return None;
                }
                if lane_vals.contains(&a.0) {
                    packed.push(r.ty.clone());
                }
            }
            (None, Some((x, y))) => {
                if !is_vec(&vector_ty, x) || !is_vec(&vector_ty, y) {
                    return None;
                }
                if lane_vals.contains(&x.0) || lane_vals.contains(&y.0) {
                    packed.push(r.ty.clone());
                }
            }
            _ => return None,
        }
    }
    // An argmax's index partials are updated from the counter's lane ramp.
    let ramp_vector = ramp_vector || reductions.iter().any(|r| r.arg.is_some() || r.find_last);
    Some(BodyPlan {
        plans,
        vector_ty,
        reductions,
        lane_vals,
        packed,
        ramp_vector,
        replicated,
        calls,
    })
}

/// May the body's calls run in the vector body's order? Every one is side-effect free
/// ([`plan_body`]); what is left is **when** each runs relative to the other lanes' steps.
///
/// A call with lane-varying arguments runs once per lane, all `W` lanes at the call's place in the
/// body ([`Plan::Call`]); a call with one argument value for the group runs once, there
/// ([`Plan::Scalar`]). The scalar loop runs the whole body for one iteration before the next, so a
/// lane's call moves ahead of the earlier lanes' later steps. For a call that always returns
/// (`CallEffect::will_return`) nothing observes that — it writes nothing, and what it reads no
/// iteration writes. One that may **trap or never return** is observed: it must be the only step
/// of the body that can stop the program, and then it stops it at the same iteration the scalar
/// loop would — every earlier lane's call has returned, no other step can have stopped first, and
/// what the vector body has stored by then is never seen again. Two such calls could stop the
/// program in the other order (lane 1's first call before lane 0's second), so they decline.
///
/// A search loop runs the steps before its exit test for lanes past the one that leaves
/// ([`EarlyExit`]), and a checked body the steps before its last group test for lanes past one
/// that fails ([`linearize_spine`]) — everything before `speculative`; a call there with
/// lane-varying arguments runs for iterations the scalar loop never reaches, which only a
/// `speculatable` call may.
fn call_verdict(
    calls: &[(usize, CallEffect)],
    plans: &[Plan],
    speculative: usize,
) -> Result<(), &'static str> {
    if calls.iter().filter(|(_, e)| !e.will_return).count() > 1 {
        return Err("two calls that may trap or never return");
    }
    if calls
        .iter()
        .any(|&(at, e)| at < speculative && matches!(plans[at], Plan::Call) && !e.speculatable)
    {
        return Err("a call it cannot speculate runs before a lane test");
    }
    Ok(())
}

/// May this binary operator be applied lane-wise, vector by vector, at this result type?
///
/// The three exclusions are all about what the x64 backend can actually lower, and none of them
/// degrades gracefully — so this predicate is the only thing standing between a widened body and
/// a failed or crashed compile.
///
/// * **Integer divide and remainder.** There is no packed integer division on x86 at any width.
/// * **Integer multiply at `i8` lanes.** x86 has `pmullw` (16-bit) and `pmulld` (32-bit) but no
///   8-bit packed multiply, and Cranelift does not synthesize one out of two 16-bit halves:
///   `imul.i8x16` reaches the x64 backend as
///   `Unsupported("should be implemented in ISLE: inst = imul.i8x16 ..")`. The body planner widens
///   it anyway, through `i16` halves ([`emit_bin`]); a reduction's vector accumulator, whose
///   combine is this predicate's question, keeps the serial fold.
/// * **Every shift, vector by vector.** This one is not a missing instruction, it is a *shape*
///   mismatch. A lane-wise shift takes each lane of `a` by the matching lane of `b`; Cranelift's
///   `ishl`/`ushr`/`sshr` take a **scalar** amount for every lane, so handing them a splatted vector
///   is a type error:
///
///   ```text
///   arg 1 (v29) with type i32x4 failed to satisfy type set ValueTypeSet { lanes: {0}, .. }
///   ```
///
///   The CLIF verifier says exactly that — but **only in a debug build**. In release it is off, the
///   malformed `ishl.i32x4` goes straight to lowering, and the compile panics inside Cranelift's ISLE
///   (`no rule matched for term bitcast_xmm_to_gpr`): a program that runs at `-O0` and crashes the
///   compiler at `-O2`, in the shipping configuration only. The body planner therefore widens a
///   shift only when its amount is uniform, through MIR's vector-by-*scalar* shift, which is exactly
///   Cranelift's shape; a per-lane amount stays scalar.
///
/// Those are the only three. An 82-case (operator x lane type) matrix over `i8`/`i16`/`i32`/
/// `i64`/`f32`/`f64` — add, sub, mul, and, or, xor, shl, ashr, neg, not, compare+select, the
/// float four, sqrt, and the five reassociable reductions — compiles through `--emit=obj -O2`
/// with a **debug** `wukongc` (so the CLIF verifier runs) for every other pair, `imul.i64x2`
/// included. Re-run it against a debug binary before adding an operator here.
pub(crate) fn bin_widenable(op: BinOp, ty: &MirType) -> bool {
    let lane = ty.lane_type();
    match op {
        BinOp::SDiv | BinOp::UDiv | BinOp::SRem | BinOp::URem | BinOp::FRem => false,
        BinOp::Shl | BinOp::LShr | BinOp::AShr => false,
        BinOp::Mul if *lane == MirType::I8 => false,
        BinOp::FAdd | BinOp::FSub | BinOp::FMul | BinOp::FDiv => lane.is_float(),
        BinOp::Add | BinOp::Sub | BinOp::Mul => lane.is_int(),
        BinOp::And | BinOp::Or | BinOp::Xor => true,
    }
}

/// Casts that are one-for-one at a fixed lane width, and that the x64 backend can lower at that
/// width.
///
/// `to` is the *result* lane type; the caller has already required it to be the same width as the
/// source, so this is a question about one width, not two.
///
/// **Float to integer at 64-bit lanes is refused.** Cranelift lowers `fcvt_to_sint_sat.i32x4`
/// (SSE2 `cvttps2dq` plus the saturation fix-ups) but has no rule for `.i64x2`, which needs
/// AVX512DQ's `vcvttpd2qq`; it comes back as `Unsupported("should be implemented in ISLE")`, the
/// same failed-at-`-O2`-but-runs-at-`-O0` shape as the entries in [`bin_widenable`]. The reverse
/// direction is fine at both widths, and so is `f32 -> i32`. Enumerated the same way, with a debug
/// `wukongc` so the CLIF verifier runs: all four conversion kinds at both lane widths, of which
/// exactly `FpToSi`/`FpToUi` at 8 bytes fail.
fn cast_widenable(k: CastKind, to: &MirType) -> bool {
    match k {
        CastKind::FpToSi | CastKind::FpToUi => lane_bytes(to.lane_type()) != Some(8),
        CastKind::SiToFp | CastKind::UiToFp | CastKind::Bitcast => true,
        _ => false,
    }
}

/// How the scalar cast `kind` from lane type `from` to lane type `to` is spelled over parts, or
/// `None` when it has no exact vector form here.
///
/// Every route is an exact lane map: each lane of the result is what the scalar cast computes for
/// the same lane. The ones that are not offered say why:
///
/// * an integer to a *narrower* float (`i64 -> f32`) would have to round twice;
/// * a float to a 64-bit integer, or `f64` to any integer, needs the packed `cvttpd2qq` /
///   `fcvt_to_sint_sat.i64x2` that [`cast_widenable`] documents Cranelift does not have;
/// * `f32 -> i64` would go through `f64`, which is that same conversion.
///
/// Same-width integer casts (`i32 as u32` lowers to an equal-width `Trunc`) change no bits, so their
/// parts are reused unchanged.
fn cast_route(kind: CastKind, from: &MirType, to: &MirType) -> Option<CastRoute> {
    use CastKind::*;
    let (fb, tb) = (lane_bytes(from)?, lane_bytes(to)?);
    // A bf16 or f16 lane is an `f32` lane in a register, so the casts that only round it to a
    // grid are lane-wise: an `fpext` out of a half type (it rounds the source to its grid first)
    // and an `fptrunc` into one (the rounding `mir_build` places after every half operation), each
    // between two of `f32`, bf16 and f16 — the backends round exactly where the scalar cast does.
    // Through `f64` the cast is that rounding and the `f32 <-> f64` lane-count change, in the
    // scalar cast's order (`HalfToF64`, `F64ToHalf`). To and from an integer, a half lane converts
    // exactly as an `f32` lane does — the scalar cast rounds nothing at the half grid there (an
    // integer becomes an `f32`, and `mir_build` rounds it to the half grid with the same-type
    // `fptrunc` it places right after the conversion, which takes the rounding route above; a half
    // value is truncated from its `f32` register) — so the `f32` route serves, with the half lane
    // type at the half end. A bitcast has no lane-wise form and is replicated.
    if is_half(from) || is_half(to) {
        let float32 = |t: &MirType| *t == MirType::F32 || is_half(t);
        let as_f32 = |t: &MirType| {
            if is_half(t) {
                MirType::F32
            } else {
                t.clone()
            }
        };
        return match kind {
            FpExt if is_half(from) && float32(to) => Some(CastRoute::Same),
            FpTrunc if is_half(to) && float32(from) => Some(CastRoute::Same),
            FpExt if is_half(from) && *to == MirType::F64 => Some(CastRoute::HalfToF64),
            FpTrunc if *from == MirType::F64 && is_half(to) => Some(CastRoute::F64ToHalf),
            SiToFp | UiToFp | FpToSi | FpToUi => cast_route(kind, &as_f32(from), &as_f32(to)),
            _ => None,
        };
    }
    Some(match kind {
        SExt | ZExt | Trunc if from.is_int() && to.is_int() && fb == tb => CastRoute::Identity,
        _ if fb == tb => {
            if !cast_widenable(kind, to) {
                return None;
            }
            CastRoute::Same
        }
        SExt | ZExt if from.is_int() && to.is_int() && fb < tb => CastRoute::WidenInt {
            ext: kind,
            then_fp: false,
        },
        Trunc if from.is_int() && to.is_int() && fb > tb => CastRoute::TruncInt,
        SiToFp | UiToFp if from.is_int() && to.is_float() && fb < tb => CastRoute::WidenInt {
            ext: if kind == SiToFp { SExt } else { ZExt },
            then_fp: true,
        },
        FpToSi | FpToUi if *from == MirType::F32 && to.is_int() && tb < 4 => {
            CastRoute::FloatToNarrowInt {
                unsigned: kind == FpToUi,
            }
        }
        FpExt if *from == MirType::F32 && *to == MirType::F64 => CastRoute::FpExt,
        FpTrunc if *from == MirType::F64 && *to == MirType::F32 => CastRoute::FpTrunc,
        _ => return None,
    })
}

/// The runtime pointer-range checks this loop needs.
///
/// [`crate::loop_info::MemDep::IndependentIfBasesDisjoint`] states the obligation: every pair of
/// bases the analysis could not relate, where at least one is written, must at runtime be either
/// the same pointer (distance 0) or non-overlapping. A guarded access has to sit at an offset from
/// its base that the guard block can *recompute*, so the range it touches is expressible as two
/// `gep`s: `1·iv + Σ invariant + k`. See [`GuardBase`] for why the invariant part matters as much
/// as the constant one.
///
/// The invariant part is the whole reason this is not just "a constant offset". The single most
/// common shape in this language is a 2-D row-major inner loop, `a[r*C + c]` for `c in 0..C`, whose
/// index decomposes to `1·c + (r*C)` — one loop-invariant term, never a constant. Requiring an
/// empty term list declined every such loop, which is to say every row-wise softmax, every row
/// reduction, and both hot loops of a fused per-row loss.
///
/// The unit of grouping is therefore **`(base, invariant part)`, not `base`** — one buffer read at
/// two different row offsets in the same body (`g[i] = w[ga + i]; b[i] = w[gb + i];`, the LayerNorm
/// γ/β gather of a transformer block) contributes two ranges, and each one is a plain `[lo, hi)` the
/// guard can compute. Grouping by base alone forced those two into one range with no expressible
/// offset, and declined the loop.
///
/// Splitting one base across two groups is only sound because two such groups can only ever both be
/// **loads**: `loop_info::dependence` reports [`crate::loop_info::MemDep::Carried`] for any
/// same-object pair whose index expressions differ when one of them writes — except a pair the GCD
/// test proves never meets, and that pair shares its invariant part and stride and so lands in ONE
/// group here (an interleaved `o[2i]`/`o[2i+1]` store pair) — so such a loop never reaches here.
/// That is a distant invariant, so it is also re-checked locally below rather than assumed — the
/// check costs nothing and cannot fire on a loop the old code accepted.
///
/// Given that, the split is not merely safe but strictly *more precise* than grouping by base:
/// each group's range is an exact cover of what that group touches (every member shares `terms` and
/// differs only by a constant in `[lo, hi]`), the union of a base's groups covers everything the
/// base touches, and testing each range separately against a written group accepts exactly the
/// cases where the base really does not overlap it — where the hull `[min, max)` would have had to
/// reject. The only pair a split leaves untested is one *within* a base, and that pair is either
/// read/read (never a dependence at any width) or refused outright three lines below. Nothing here
/// weakens `guard_allows_equal` either: distance-0 acceptance still demands that **every** group
/// have `lo == hi` and share one `lo` and one `terms`, which a split base can never satisfy.
///
/// The end-to-end evidence is `tests/run/vectorized_split_base_guard.wk`, which aliases the read
/// base with the output through a pointer the optimizer cannot fold, calls the loop once with
/// overlapping and once with disjoint windows, and prints every lane of both buffers against
/// hand-derived expectations. Deleting the `Select` on `emit_alias_checks`'s result makes it fail 7
/// of its 96 lines.
fn plan_guards(f: &Function, l: &NaturalLoop) -> Option<(Vec<GuardPair>, Vec<PointGuard>, bool)> {
    // A symbolic addend is only usable in the guard if it is available there, and the guard sits in
    // front of the header. `AffineExpr` deliberately also accepts a value *defined inside* the loop
    // whose operands are all invariant (`loop_info::value_invariant` — that is what keeps the
    // analysis independent of whether LICM has run), so "invariant" is not enough: such a value is
    // defined below the guard and reading it there is a use its definition does not dominate.
    let inside: FxHashSet<u32> = l
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
    // Every index the guard rebuilds is arithmetic on the induction variable, so each symbolic
    // addend has to *be* that type. It usually is; the exception is a `sext`, which the affine
    // decomposition looks through (`Op::Cast(SExt, ..)`), leaving a narrower value behind.
    let iv_ty = l.primary()?.ty.clone();
    // The stride must be a positive constant the induction variable's type can hold: the range
    // below is `stride·iv + invariants + k`, rebuilt in that type. (`access_shape_verdict` has
    // already limited every affine access to a unit stride or a complete interleave group.)
    let usable_terms = |ix: &crate::loop_info::AffineExpr| {
        matches!(ix.coeff.as_const(), Some(s) if s >= 1 && fits_in(&iv_ty, s))
            && ix.terms.iter().all(|(v, m)| {
                !inside.contains(&v.0) && *f.value_type(*v) == iv_ty && fits_in(&iv_ty, *m)
            })
    };

    // Group the affine accesses by `(base, invariant offset, stride)`, tracking the constant offset
    // window and whether any access in the group is a store. `usable` is per group rather than per
    // access because `lo`/`hi` only bound the touched range for accesses that share the symbolic
    // part and the stride — which, with both in the key, every member of a group does. An
    // interleave group's `k` members land in one guard group, `lo..=hi` spanning its residues.
    struct Group {
        g: GuardBase,
        stored: bool,
        elem: MirType,
        usable: bool,
    }
    let mut bases: Vec<Group> = Vec::new();
    for a in &l.accesses {
        let AddrForm::Affine { base, index, elem } = &a.addr else {
            continue;
        };
        let is_store = a.kind == AccessKind::Store;
        let stride = index.coeff.as_const().unwrap_or(0);
        match bases.iter_mut().find(|gr| {
            gr.g.base.same_object(*base) && gr.g.terms == index.terms && gr.g.stride == stride
        }) {
            Some(gr) => {
                if gr.elem != *elem {
                    return None;
                }
                gr.usable &= usable_terms(index);
                gr.g.lo = gr.g.lo.min(index.konst);
                gr.g.hi = gr.g.hi.max(index.konst);
                gr.stored |= is_store;
            }
            None => bases.push(Group {
                usable: usable_terms(index),
                g: GuardBase {
                    base: *base,
                    terms: index.terms.clone(),
                    lo: index.konst,
                    hi: index.konst,
                    elem: elem.clone(),
                    stride,
                },
                stored: is_store,
                elem: elem.clone(),
            }),
        }
    }
    // Conservative, and only ever *adds* an accepted case at runtime: two ranges may be reported
    // "equal, therefore at distance 0" only when every group touches a single offset and all of
    // them share it, symbolic part and element type included. Two equal pointers moving in
    // different element types (`f32` written where `i8` is read) are not at distance 0 — iteration
    // `i` writes bytes `4i .. 4i+4` that iteration `4i` then reads — so a mixed-width loop never
    // takes the equality exit.
    let uniform_offset = bases.iter().all(|gr| gr.g.lo == gr.g.hi)
        && bases.windows(2).all(|p| {
            p[0].g.lo == p[1].g.lo && p[0].g.terms == p[1].g.terms && p[0].elem == p[1].elem
        });

    let mut guards = Vec::new();
    // Built at most once, and only for a function with a `@noalias` parameter (`alias.rs`).
    let mut promised = crate::alias::PromisedBases::new(f);
    for i in 0..bases.len() {
        for j in i + 1..bases.len() {
            let (ga, gb) = (&bases[i], &bases[j]);
            if !ga.stored && !gb.stored {
                continue; // read/read is never a dependence
            }
            // Two groups on ONE object: their symbolic parts differ (equal ones share a group), and
            // `lo`/`hi` say nothing about the distance between two different symbolic offsets, so
            // there is no range test to emit. This is the local restatement of the invariant the
            // doc comment leans on — `loop_info::dependence` already reports `Carried` here, so it
            // is belt and braces rather than a live path.
            if ga.g.base.same_object(gb.g.base) {
                return None;
            }
            // Provably distinct objects need no range test: two slots, or storage a `@noalias`
            // parameter promised apart (`alias.rs`) — which also spares a pair whose index the
            // guard could not have expressed.
            if distinct_alloca_pair(f, ga.g.base, gb.g.base)
                || promised.disjoint(ga.g.base, gb.g.base)
            {
                continue;
            }
            // The whole group's index must be `1·iv + invariants + k` for the touched range to be
            // two `gep`s the guard can build.
            if !ga.usable || !gb.usable {
                return None;
            }
            guards.push(GuardPair {
                a: ga.g.clone(),
                b: gb.g.clone(),
            });
        }
    }

    // Loop-invariant data loads against the stores. A load whose address never moves executes once
    // per GROUP in the vector body but once per ELEMENT in the scalar loop, so the two read the same
    // values only if no store in the loop writes that address — the pairwise question the ranges
    // above already answer, with a one-element range on one side. Until this existed, ONE invariant
    // scalar load (`w[0]` in `o[i] = a[i]*w[0] + w[1]`, the canonical scaled-elementwise kernel)
    // plus ANY store declined the whole loop. Now each such load is checked against each stored
    // group at run time — unless the two are distinct stack slots — and a failing check sets
    // `n_vec = 0` like every other bail-out, so the scalar loop does all the work.
    //
    // A `Ptr`-typed invariant load is exempt as before: it is a `[]T` slice re-reading its own data
    // pointer, the assumption `MemDep::IndependentIfBasesStable` states and the dependence verdict
    // already answers. Two fixed addresses (an invariant load beside an invariant store) are not a
    // shape this guard expresses, so that loop still declines, as do a load whose address is
    // computed inside the loop (the guard runs before it) and a stored group with no rebuildable
    // range.
    let invariant_store = l
        .accesses
        .iter()
        .any(|a| a.kind == AccessKind::Store && matches!(a.addr, AddrForm::Invariant(_)));
    let mut points = Vec::new();
    for a in &l.accesses {
        let AddrForm::Invariant(at) = &a.addr else {
            continue;
        };
        if a.kind != AccessKind::Load || a.value_ty == MirType::Ptr {
            continue;
        }
        if invariant_store || inside.contains(&at.outer_handle().0) {
            return None;
        }
        for gr in bases.iter().filter(|gr| gr.stored) {
            if distinct_alloca_pair(f, *at, gr.g.base) {
                continue;
            }
            if !gr.usable {
                return None;
            }
            points.push(PointGuard {
                at: *at,
                ty: a.value_ty.clone(),
                store: gr.g.clone(),
            });
        }
    }
    Some((guards, points, uniform_offset))
}

/// Is `k` representable in this integer type? The guard materializes a symbolic addend's
/// multiplier as a constant of the induction variable's type, and a multiplier that does not fit
/// would silently become a different number. `AffineExpr::scale` accumulates multipliers in `i64`
/// (`checked_mul`), so a nested `a[(x*K1)*K2 + i]` really can leave one that overflows an `i32`
/// induction variable's type.
fn fits_in(ty: &MirType, k: i64) -> bool {
    match ty {
        MirType::I8 => i8::try_from(k).is_ok(),
        MirType::I16 => i16::try_from(k).is_ok(),
        MirType::I32 => i32::try_from(k).is_ok(),
        MirType::I64 => true,
        _ => false,
    }
}

fn distinct_alloca_pair(f: &Function, a: MemBase, b: MemBase) -> bool {
    let is_alloca = |m: MemBase| match m {
        MemBase::Outer(v) => matches!(
            op_of(f, v),
            Some(Op::Alloca {
                space: wukong_mir::AddrSpace::Local,
                ..
            })
        ),
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
///
/// Like `wukong_mir::Builder` it carries a **current location**: every value it creates is
/// attributed to `loc` (`wukong_mir::SourceLocs`; a dummy span records nothing). A widened loop is
/// still its source loop, so [`apply`] starts at the loop's span ([`loop_loc`]) for the guard, the
/// trip and the exit, and [`emit_body`] moves it to each scalar step's own span ([`step_loc`])
/// while that step is widened — the vector `fadd` of `y[i] = a * x[i] + y[i]` is on the line the
/// scalar `fadd` was.
struct Emit<'a> {
    f: &'a mut Function,
    loc: Span,
}

impl<'a> Emit<'a> {
    fn new(f: &'a mut Function, loc: Span) -> Emit<'a> {
        Emit { f, loc }
    }

    fn value(&mut self, ty: MirType) -> ValueId {
        let id = ValueId(self.f.value_types.len() as u32);
        self.f.value_types.push(ty);
        self.f.set_value_loc(id, self.loc);
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

/// The span of the source loop `p` widens: its header's exit test, else its induction variable,
/// else the first located value of its body. Dummy when none of them is located (a loop the
/// optimizer itself built), so its new values stay unknown rather than borrow another line.
fn loop_loc(f: &Function, p: &VecPlan) -> Span {
    let header = &f.blocks[p.header.0 as usize];
    let exit_test = match &header.term {
        Terminator::CondBr { cond, .. } => f.value_loc(*cond),
        _ => None,
    };
    exit_test
        .or_else(|| f.value_loc(header.params[p.iv_index]))
        .or_else(|| {
            f.blocks[p.body.0 as usize]
                .insts
                .iter()
                .find_map(|i| i.result.and_then(|r| f.value_loc(r)))
        })
        .unwrap_or_else(Span::dummy)
}

/// The span of a scalar instruction: its result's, or for a store (which has no result) its
/// address's, else its stored value's. `None` when it is not located.
fn inst_loc(f: &Function, inst: &Inst) -> Option<Span> {
    match inst {
        Inst {
            result: Some(r), ..
        } => f.value_loc(*r),
        Inst {
            op: Op::Store { ptr, value },
            ..
        } => f.value_loc(*ptr).or_else(|| f.value_loc(*value)),
        _ => None,
    }
}

/// The span of the scalar step a body step widens ([`inst_loc`] of its instruction). `None` when
/// the step is not located.
fn step_loc(f: &Function, step: &LinInst) -> Option<Span> {
    let inst_loc =
        |(block, idx): (BlockId, usize)| inst_loc(f, &f.blocks[block.0 as usize].insts[idx]);
    match step {
        LinInst::Real { block, idx } | LinInst::PredLoad { block, idx, .. } => {
            inst_loc((*block, *idx))
        }
        LinInst::MaskedStore { at, .. }
        | LinInst::TreeStore { at, .. }
        | LinInst::PredStore { at, .. } => inst_loc(*at),
        LinInst::Merge { param, .. } => f.value_loc(*param),
    }
}

/// The span of reduction `r`'s scalar update (its back-edge argument), else of its accumulator,
/// else `fallback`: the combines and folds of its partials are that update, regrouped.
fn red_loc(f: &Function, p: &VecPlan, r: &RedPlan, fallback: Span) -> Span {
    let update = match &f.blocks[p.body.0 as usize].term {
        Terminator::Br { args, .. } => args.get(r.param_index).copied(),
        Terminator::CondBr {
            then_blk,
            then_args,
            else_args,
            ..
        } => {
            let args = if *then_blk == p.header {
                then_args
            } else {
                else_args
            };
            args.get(r.param_index).copied()
        }
        _ => None,
    };
    update
        .and_then(|v| f.value_loc(v))
        .or_else(|| f.value_loc(f.blocks[p.header.0 as usize].params[r.param_index]))
        .unwrap_or(fallback)
}

fn apply(f: &mut Function, p: &VecPlan) {
    let header_params: Vec<ValueId> = f.blocks[p.header.0 as usize].params.clone();
    let header_tys: Vec<MirType> = header_params
        .iter()
        .map(|v| f.value_type(*v).clone())
        .collect();

    let loop_at = loop_loc(f, p);
    let mut e = Emit::new(f, loop_at);

    // Reassociable reductions carry an extra, vector-typed accumulator through the vector loop —
    // `W` independent partials that are folded into the scalar accumulator once, on the way out.
    // The scalar parameter is still carried (unchanged) so the block signatures stay aligned with
    // the original header's.
    let vacc: Vec<&RedPlan> = p.reductions.iter().filter(|r| r.vector_acc).collect();

    // ---- the four new blocks (terminators are patched in once every block exists) --------------
    let guard_params: Vec<ValueId> = header_tys.iter().map(|t| e.value(t.clone())).collect();
    let guard = e.block(guard_params.clone(), Terminator::Unreachable);
    let mut vh_params: Vec<ValueId> = header_tys.iter().map(|t| e.value(t.clone())).collect();
    // One header parameter per part of each vector accumulator, reductions in order — the layout
    // `emit_body` reads them back in.
    for r in &vacc {
        for _ in 0..parts_of(&r.ty, p.w) {
            vh_params.push(e.value(part_ty(&r.ty)));
        }
        if let Some((_, ity)) = &r.arg {
            for _ in 0..parts_of(ity, p.w) {
                vh_params.push(e.value(part_ty(ity)));
            }
        }
    }
    let vh = e.block(vh_params.clone(), Terminator::Unreachable);
    let vb = e.block(Vec::new(), Terminator::Unreachable);
    let vx = e.block(Vec::new(), Terminator::Unreachable);

    // ---- guard: how many elements does the vector loop own? ------------------------------------
    let start = guard_params[p.iv_index];
    let n_vec = emit_trip(&mut e, p, guard, start);
    let mut reads = GuardReads::new(p, n_vec);
    let ok = emit_alias_checks(&mut e, p, guard, &mut reads);
    let ok = match (ok, emit_len_checks(&mut e, p, guard, start, n_vec)) {
        (Some(a), Some(b)) => Some(e.push(guard, MirType::I1, Op::Bin(BinOp::And, a, b))),
        (a, b) => a.or(b),
    };
    let n_vec = match ok {
        Some(cond) => {
            let zero = e.int(guard, &p.iv_ty, 0);
            e.push(guard, p.iv_ty.clone(), Op::Select(cond, n_vec, zero))
        }
        None => n_vec,
    };
    // A countdown's vector loop runs the counter down to `start - n_vec`.
    let toward = if p.dir > 0 { BinOp::Add } else { BinOp::Sub };
    let vec_end = e.push(guard, p.iv_ty.clone(), Op::Bin(toward, start, n_vec));
    let mut guard_args = guard_params.clone();
    for r in &vacc {
        // An argmax's partials need no identity: every lane starts at the scalar loop's own
        // `(m, at)`, which is exactly the state each lane's sub-sequence starts from. A last-tie
        // argmax starts its index lanes one below the first counter instead, so that a lane which
        // never updated is told apart from one that tied `m` ([`emit_arg_combine`]).
        if let Some((ai, ity)) = &r.arg {
            let m0 = e.push(
                guard,
                part_ty(&r.ty),
                Op::Splat(guard_params[r.param_index]),
            );
            for _ in 0..parts_of(&r.ty, p.w) {
                guard_args.push(m0);
            }
            let first = if r.arg_last {
                let one = e.int(guard, ity, 1);
                e.push(guard, ity.clone(), Op::Bin(BinOp::Sub, start, one))
            } else {
                guard_params[*ai]
            };
            let a0 = e.push(guard, part_ty(ity), Op::Splat(first));
            for _ in 0..parts_of(ity, p.w) {
                guard_args.push(a0);
            }
            continue;
        }
        if r.find_last {
            // One below the first counter: no lane can record it, since every recorded counter lies
            // in `start .. start + n_vec` (wrapping arithmetic keeps that true at the type's ends).
            let one = e.int(guard, &r.ty, 1);
            let below = e.push(guard, r.ty.clone(), Op::Bin(BinOp::Sub, start, one));
            let none = e.push(guard, part_ty(&r.ty), Op::Splat(below));
            for _ in 0..parts_of(&r.ty, p.w) {
                guard_args.push(none);
            }
            continue;
        }
        let id = red_identity(r.kind, &r.ty).expect("vector_acc implies an identity");
        let scalar = e.push(guard, r.ty.clone(), Op::ConstInt(id as i128, r.ty.clone()));
        let identity = e.push(guard, part_ty(&r.ty), Op::Splat(scalar));
        for _ in 0..parts_of(&r.ty, p.w) {
            guard_args.push(identity);
        }
    }
    e.f.blocks[guard.0 as usize].term = Terminator::Br {
        target: vh,
        args: guard_args,
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
    // A vector accumulator is folded here, once, rather than in the body — the whole point of
    // keeping `W` independent partials.
    //
    // The operator is associative and commutative (`vector_acc` admits integer `add`/`mul`/bitwise
    // and integer min/max only), so the parts are first combined lane-wise into one vector and its
    // lanes then folded into the scalar accumulator: any order gives the same integer, wraparound
    // included — and a min/max of equal integers is that integer whichever side the select names.
    let mut exit_args = vh_params[..header_tys.len()].to_vec();
    let mut next_param = header_tys.len();
    for r in &vacc {
        e.loc = red_loc(e.f, p, r, loop_at);
        let n = parts_of(&r.ty, p.w);
        let parts = vh_params[next_param..next_param + n].to_vec();
        next_param += n;
        if let Some((ai, ity)) = &r.arg {
            let na = parts_of(ity, p.w);
            let at_parts = vh_params[next_param..next_param + na].to_vec();
            next_param += na;
            let prior = exit_args[*ai];
            let (m, at) = emit_arg_combine(&mut e, vx, r, ity, &parts, &at_parts, start, prior);
            exit_args[r.param_index] = m;
            exit_args[*ai] = at;
            continue;
        }
        if r.find_last {
            let prior = exit_args[r.param_index];
            exit_args[r.param_index] =
                emit_find_last_combine(&mut e, vx, &r.ty, &parts, start, prior);
            continue;
        }
        let mut vec = parts[0];
        for &q in &parts[1..] {
            vec = emit_combine(&mut e, vx, r, true, vec, q, &part_ty(&r.ty));
        }
        let mut acc = exit_args[r.param_index];
        for k in 0..lanes_of(&r.ty).expect("a widened reduction type") {
            let lane = e.push(vx, r.ty.clone(), Op::ExtractLane(vec, k));
            acc = emit_combine(&mut e, vx, r, true, acc, lane, &r.ty);
        }
        exit_args[r.param_index] = acc;
    }
    e.loc = loop_at;
    e.f.blocks[vx.0 as usize].term = Terminator::Br {
        target: p.header,
        args: exit_args,
    };

    // ---- vector body ---------------------------------------------------------------------------
    // A search loop's body is two blocks: the test leaves for `vx` before the rest of the group.
    let (latch_args, last) = emit_body(
        &mut e,
        p,
        vb,
        vx,
        guard,
        &mut reads,
        &header_params,
        &vh_params,
    );
    e.loc = loop_at;
    reads.finish(&mut e);
    e.f.blocks[last.0 as usize].term = Terminator::Br {
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
///
/// A runtime count is `end - start` in the induction variable's own type, which **wraps**. The
/// scalar loop runs zero times whenever `start < end` fails under the header test's signedness,
/// but the difference alone does not say so: an `i32` loop from `2147483600` to `-2147483600` runs
/// no iteration, and `end - start` wraps to `96`. Treated as a count, that sent 24 groups of loads
/// off the end of the buffer (the interpreter stopped with "vector load out of bounds" at `-O2`
/// where `-O0` returned the untouched seed). So the order is tested explicitly, with the same
/// predicate the header uses; once `start < end` holds, the mathematical count is positive and a
/// wrapped difference can only read as negative, which the `>= W` test already turns into zero.
fn emit_trip(e: &mut Emit<'_>, p: &VecPlan, guard: BlockId, start: ValueId) -> ValueId {
    trip_count(e, &p.trip, &p.iv_ty, p.w, guard, start)
}

/// [`emit_trip`] for a counter of type `iv_ty` stepping in groups of `w`.
fn trip_count(
    e: &mut Emit<'_>,
    trip: &TripPlan,
    iv_ty: &MirType,
    w: u32,
    guard: BlockId,
    start: ValueId,
) -> ValueId {
    match trip {
        TripPlan::Const(n) => e.int(guard, iv_ty, *n as i64),
        TripPlan::Runtime { end, signed } => {
            let count = e.push(guard, iv_ty.clone(), Op::Bin(BinOp::Sub, *end, start));
            // `count & !(W-1)` rounds down for a non-negative count; the compare below is what
            // keeps a negative one (an already-finished loop) from producing a negative `n_vec`
            // and rewinding the scalar loop's induction variable behind its start.
            let mask = e.int(guard, iv_ty, !((w as i64) - 1));
            let masked = e.push(guard, iv_ty.clone(), Op::Bin(BinOp::And, count, mask));
            let min = e.int(guard, iv_ty, w as i64 * MIN_TRIP_GROUPS as i64);
            let big = e.push(guard, MirType::I1, Op::Cmp(CmpOp::Sge, count, min));
            let lt = if *signed { CmpOp::Slt } else { CmpOp::Ult };
            let runs = e.push(guard, MirType::I1, Op::Cmp(lt, start, *end));
            let ok = e.push(guard, MirType::I1, Op::Bin(BinOp::And, runs, big));
            let zero = e.int(guard, iv_ty, 0);
            e.push(guard, iv_ty.clone(), Op::Select(ok, masked, zero))
        }
        // A countdown runs `start - end` iterations (one more when the test is `iv >= end`). The
        // same order test comes first — the loop runs at all only if `start > end` (`>=`) under
        // the header's signedness — and the inclusive `+ 1` is taken only below the type's
        // maximum, where it cannot wrap; a larger count reads as negative and runs no group.
        TripPlan::RuntimeDown {
            end,
            signed,
            inclusive,
        } => {
            let diff = e.push(guard, iv_ty.clone(), Op::Bin(BinOp::Sub, start, *end));
            let gt = match (signed, inclusive) {
                (true, false) => CmpOp::Sgt,
                (true, true) => CmpOp::Sge,
                (false, false) => CmpOp::Ugt,
                (false, true) => CmpOp::Uge,
            };
            let runs = e.push(guard, MirType::I1, Op::Cmp(gt, start, *end));
            let (count, fits) = if *inclusive {
                let one = e.int(guard, iv_ty, 1);
                let count = e.push(guard, iv_ty.clone(), Op::Bin(BinOp::Add, diff, one));
                // `diff + 1 > diff` fails exactly at the signed maximum.
                let fits = e.push(guard, MirType::I1, Op::Cmp(CmpOp::Sgt, count, diff));
                (count, Some(fits))
            } else {
                (diff, None)
            };
            let mask = e.int(guard, iv_ty, !((w as i64) - 1));
            let masked = e.push(guard, iv_ty.clone(), Op::Bin(BinOp::And, count, mask));
            let min = e.int(guard, iv_ty, w as i64 * MIN_TRIP_GROUPS as i64);
            let big = e.push(guard, MirType::I1, Op::Cmp(CmpOp::Sge, count, min));
            let mut ok = e.push(guard, MirType::I1, Op::Bin(BinOp::And, runs, big));
            if let Some(fits) = fits {
                ok = e.push(guard, MirType::I1, Op::Bin(BinOp::And, ok, fits));
            }
            let zero = e.int(guard, iv_ty, 0);
            e.push(guard, iv_ty.clone(), Op::Select(ok, masked, zero))
        }
    }
}

/// How the guard reads a pointer out of memory: an alias check's data pointer
/// ([`range_of`], the point checks) or a hoisted [`Plan::InvariantPtrLoad`].
///
/// The guard runs on every entry to the loop, including one that runs no vector group — a zero
/// trip among them. A read of a fat pointer [`never_faults`] proves live costs nothing there. Any
/// other pointer the guard re-reads — a `[]T` field reached through a raw `*T` parameter, say — is
/// read by every iteration of the loop: an arm reads only proven fat pointers
/// ([`speculable_ptr_load`], [`bounds_check_of`]), and a search loop, which may leave before
/// reaching a read, neither hoists nor guards an unproven one ([`plan_loop`], [`emit_body`]). So
/// the scalar loop performs that read as soon as it runs once, and the guard may perform it
/// exactly when a group would run: always under a constant trip (at least one group), and under a
/// runtime trip through `select(n_vec != 0, addr, slot)` — `slot` a private stack slot standing in
/// when no group runs, whose contents then reach nothing but a `select` that discards them. Before
/// this, `k(x, 0 as *mut Layer, 0)` over `(*p).w[i] = x[i]` returned at `-O0` and stopped with
/// "null pointer dereference" at `-O2` (native: SIGSEGV).
struct GuardReads {
    /// `n_vec` from the trip alone (before any runtime check), for a runtime trip; `None` for a
    /// constant one.
    trip: Option<(ValueId, MirType)>,
    /// `trip != 0`, built on first use.
    live: Option<ValueId>,
    /// The stand-in slot, allocated in the entry block by [`GuardReads::finish`] once used.
    slot: Option<ValueId>,
}

impl GuardReads {
    fn new(p: &VecPlan, n_vec: ValueId) -> Self {
        let trip = match p.trip {
            TripPlan::Const(_) => None,
            TripPlan::Runtime { .. } | TripPlan::RuntimeDown { .. } => {
                Some((n_vec, p.iv_ty.clone()))
            }
        };
        GuardReads {
            trip,
            live: None,
            slot: None,
        }
    }

    /// `load ptr addr`, in `guard`, read only when the guard may read it.
    fn load_ptr(&mut self, e: &mut Emit<'_>, guard: BlockId, addr: ValueId) -> ValueId {
        let at = match &self.trip {
            Some((n_vec, ty)) if !never_faults(e.f, addr, 8) => {
                let live = match self.live {
                    Some(v) => v,
                    None => {
                        let zero = e.int(guard, ty, 0);
                        let v = e.push(guard, MirType::I1, Op::Cmp(CmpOp::Ne, *n_vec, zero));
                        self.live = Some(v);
                        v
                    }
                };
                let slot = *self.slot.get_or_insert_with(|| e.value(MirType::Ptr));
                e.push(guard, MirType::Ptr, Op::Select(live, addr, slot))
            }
            _ => addr,
        };
        e.push(guard, MirType::Ptr, Op::Load(at, MirType::Ptr))
    }

    /// Allocate the stand-in slot, if a read used one, in the entry block — inserted last, like
    /// [`emit_body`]'s scratch slots, since nothing after this reads an instruction index.
    fn finish(self, e: &mut Emit<'_>) {
        if let Some(slot) = self.slot {
            let entry = e.f.entry;
            e.f.blocks[entry.0 as usize].insts.insert(
                0,
                Inst {
                    result: Some(slot),
                    op: Op::Alloca {
                        ty: MirType::Ptr,
                        space: wukong_mir::AddrSpace::Local,
                        align: 0,
                    },
                },
            );
        }
    }
}

/// The conjunction of the arms' bounds checks over the vector loop's own iterations, or `None`
/// when there are none ([`LenCheck`]).
///
/// The vector loop runs `iv` over `[start, start + n_vec)`, so a check `iv + off <u len` holds on
/// every one of its iterations exactly when `0 <= start + off` and `start + n_vec + off <= len`
/// (signed, both sides `i64`; a length is never negative). The third test, `start + off <= start
/// + n_vec + off`, refuses the one way that reasoning could break — the upper end wrapping — and
/// costs one compare. The length is read here, once per entry: from the value the arm compared
/// against, or re-loaded from the same address outside the loop. Re-reading it is the scalar
/// loop's own answer only if no store in the loop can change it, which `plan_guards` already
/// proves at run time for an invariant `i64` load the loop also performs.
fn emit_len_checks(
    e: &mut Emit<'_>,
    p: &VecPlan,
    guard: BlockId,
    start: ValueId,
    n_vec: ValueId,
) -> Option<ValueId> {
    if p.len_checks.is_empty() {
        return None;
    }
    let vec_end = e.push(guard, p.iv_ty.clone(), Op::Bin(BinOp::Add, start, n_vec));
    let zero = e.int(guard, &p.iv_ty, 0);
    let mut acc: Option<ValueId> = None;
    for chk in &p.len_checks {
        let len = match chk.len {
            LenSrc::Value(v) => v,
            LenSrc::Load(at) => e.push(guard, MirType::I64, Op::Load(at, MirType::I64)),
            LenSrc::Header(hdr) => {
                let eight = e.int(guard, &MirType::I64, 8);
                let at = e.push(
                    guard,
                    MirType::Ptr,
                    Op::Gep {
                        ptr: hdr,
                        index: eight,
                        elem: MirType::I8,
                    },
                );
                e.push(guard, MirType::I64, Op::Load(at, MirType::I64))
            }
        };
        let k = e.int(guard, &p.iv_ty, chk.off);
        let lo = e.push(guard, p.iv_ty.clone(), Op::Bin(BinOp::Add, start, k));
        let hi = e.push(guard, p.iv_ty.clone(), Op::Bin(BinOp::Add, vec_end, k));
        let from_zero = e.push(guard, MirType::I1, Op::Cmp(CmpOp::Sge, lo, zero));
        let ordered = e.push(guard, MirType::I1, Op::Cmp(CmpOp::Sge, hi, lo));
        let within = e.push(guard, MirType::I1, Op::Cmp(CmpOp::Sle, hi, len));
        let a = e.push(guard, MirType::I1, Op::Bin(BinOp::And, from_zero, ordered));
        let ok = e.push(guard, MirType::I1, Op::Bin(BinOp::And, a, within));
        acc = Some(match acc {
            None => ok,
            Some(prev) => e.push(guard, MirType::I1, Op::Bin(BinOp::And, prev, ok)),
        });
    }
    acc
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
fn emit_alias_checks(
    e: &mut Emit<'_>,
    p: &VecPlan,
    guard: BlockId,
    reads: &mut GuardReads,
) -> Option<ValueId> {
    if p.guards.is_empty() && p.point_guards.is_empty() {
        return None;
    }
    // The vector part spans `[start, start + n_vec)`; conservatively use the whole loop, whose
    // endpoints are already in hand, since a range that is too large only rejects more.
    let start = e.f.blocks[guard.0 as usize].params[p.iv_index];
    // The counters the loop runs through, as a half-open `[start, end)` in ascending order — a
    // countdown's `(end, start]` (or `[end, start]`) turned around.
    let one = e.int(guard, &p.iv_ty, 1);
    let (start, end) = match &p.trip {
        TripPlan::Const(n) if p.dir > 0 => {
            let k = e.int(guard, &p.iv_ty, *n as i64);
            (
                start,
                e.push(guard, p.iv_ty.clone(), Op::Bin(BinOp::Add, start, k)),
            )
        }
        TripPlan::Const(n) => {
            let k = e.int(guard, &p.iv_ty, *n as i64 - 1);
            let lo = e.push(guard, p.iv_ty.clone(), Op::Bin(BinOp::Sub, start, k));
            (
                lo,
                e.push(guard, p.iv_ty.clone(), Op::Bin(BinOp::Add, start, one)),
            )
        }
        TripPlan::Runtime { end, .. } => (start, *end),
        TripPlan::RuntimeDown { end, inclusive, .. } => {
            let lo = if *inclusive {
                *end
            } else {
                e.push(guard, p.iv_ty.clone(), Op::Bin(BinOp::Add, *end, one))
            };
            (
                lo,
                e.push(guard, p.iv_ty.clone(), Op::Bin(BinOp::Add, start, one)),
            )
        }
    };

    let mut acc: Option<ValueId> = None;
    let mut ptr_cache: FxHashMap<u32, ValueId> = FxHashMap::default();
    for g in &p.guards {
        let (alo, ahi) = range_of(e, p, guard, reads, &g.a, start, end, &mut ptr_cache);
        let (blo, bhi) = range_of(e, p, guard, reads, &g.b, start, end, &mut ptr_cache);
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
    // Each invariant data load is the one-element range `[at, at + 1)`, built as a `gep` by one
    // element of its own type so the interpreter (slot-indexed) and native (byte-addressed) agree on
    // what "one past" is. It must lie entirely before or entirely after the stored range; equality
    // is never acceptable here (see [`PointGuard`]).
    for g in &p.point_guards {
        let (slo, shi) = range_of(e, p, guard, reads, &g.store, start, end, &mut ptr_cache);
        let at = match g.at {
            MemBase::Outer(v) => v,
            MemBase::ReloadedOuter { addr, .. } => match ptr_cache.get(&addr.0) {
                Some(&v) => v,
                None => {
                    let v = reads.load_ptr(e, guard, addr);
                    ptr_cache.insert(addr.0, v);
                    v
                }
            },
        };
        let one = e.int(guard, &MirType::I64, 1);
        let past = e.push(
            guard,
            MirType::Ptr,
            Op::Gep {
                ptr: at,
                index: one,
                elem: g.ty.clone(),
            },
        );
        let plo = e.push(
            guard,
            MirType::I64,
            Op::Cast(CastKind::PtrToInt, at, MirType::I64),
        );
        let phi = e.push(
            guard,
            MirType::I64,
            Op::Cast(CastKind::PtrToInt, past, MirType::I64),
        );
        let point_before = e.push(guard, MirType::I1, Op::Cmp(CmpOp::Ule, phi, slo));
        let point_after = e.push(guard, MirType::I1, Op::Cmp(CmpOp::Ule, shi, plo));
        let pair = e.push(
            guard,
            MirType::I1,
            Op::Bin(BinOp::Or, point_before, point_after),
        );
        acc = Some(match acc {
            None => pair,
            Some(prev) => e.push(guard, MirType::I1, Op::Bin(BinOp::And, prev, pair)),
        });
    }
    acc
}

/// The half-open address range `[base + s·start + lo + Σt, base + s·end + hi + Σt)` this base
/// touches, as two integers (`s` the group's stride, `1` for a plain stream).
///
/// `Σt` is the loop-invariant symbolic part of the index — the `r*C` of a row-wise `x[r*C + j]`. It
/// is rebuilt here rather than read out of the loop body because the guard runs *before* the body,
/// and [`plan_guards`] has already proved every value it names is defined outside the loop, is
/// typed like the induction variable, and has a multiplier that fits that type — so it may simply
/// be re-added.
#[allow(clippy::too_many_arguments)]
fn range_of(
    e: &mut Emit<'_>,
    p: &VecPlan,
    guard: BlockId,
    reads: &mut GuardReads,
    g: &GuardBase,
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
                let v = reads.load_ptr(e, guard, addr);
                cache.insert(handle.0, v);
                v
            }
        },
    };
    let off = |e: &mut Emit<'_>, at: ValueId, k: i64| -> ValueId {
        let scaled = if g.stride == 1 {
            at
        } else {
            let s = e.int(guard, &p.iv_ty, g.stride);
            e.push(guard, p.iv_ty.clone(), Op::Bin(BinOp::Mul, at, s))
        };
        let mut idx = if k == 0 {
            scaled
        } else {
            let c = e.int(guard, &p.iv_ty, k);
            e.push(guard, p.iv_ty.clone(), Op::Bin(BinOp::Add, scaled, c))
        };
        for &(v, mult) in &g.terms {
            let addend = if mult == 1 {
                v
            } else {
                let m = e.int(guard, &p.iv_ty, mult);
                e.push(guard, p.iv_ty.clone(), Op::Bin(BinOp::Mul, v, m))
            };
            idx = e.push(guard, p.iv_ty.clone(), Op::Bin(BinOp::Add, idx, addend));
        }
        let ptr = e.push(
            guard,
            MirType::Ptr,
            Op::Gep {
                ptr: base,
                index: idx,
                elem: g.elem.clone(),
            },
        );
        e.push(
            guard,
            MirType::I64,
            Op::Cast(CastKind::PtrToInt, ptr, MirType::I64),
        )
    };
    // The induction variable runs over `[start, end)` and each access adds a constant in
    // `[lo, hi]`, so the highest element index touched is `s·(end - 1) + hi` and `s·end + hi` is a
    // half-open upper bound (exact for `s = 1`, a few elements generous for an interleave group,
    // which can only reject more).
    let lo = off(e, start, g.lo);
    let hi = off(e, end, g.hi);
    (lo, hi)
}

/// Emit the widened body into `vb`, returning the arguments its back edge passes to the vector
/// header.
///
/// `guard` is where work that belongs once per *call* goes rather than once per lane group. Only
/// one thing uses it — [`Plan::InvariantPtrLoad`], the `[]T` slice base reload — and it is worth
/// saying why that is sound rather than merely convenient, because LICM refuses to do it.
///
/// LICM's `safe_to_hoist` excludes `Op::Load`, and rightly so in general: the loop stores, and
/// without alias analysis a store might be the thing the load re-reads. This pass does not need
/// general alias analysis here, because it has already required the answer. A loop only reaches
/// [`apply`] if its dependence verdict is [`crate::loop_info::MemDep::Independent`] or
/// `IndependentIfBasesStable`, and the latter's entire content is "a `load ptr <invariant>`
/// executed inside the loop yields the same pointer on every iteration". `plan_guards`
/// already leans on exactly that to let such a load coexist with a store at all. Doing it once is
/// that same statement, spent.
///
/// The restriction to `Ptr` results is not decoration. The guard runs even when the vector loop
/// runs zero iterations, so hoisting speculates the load; for a slice's fat pointer that is free
/// (the fat pointer is the parameter the loop indexes through, live for the whole call, and
/// `emit_alias_checks` already reads it), but for an arbitrary invariant address it would be a
/// read the scalar loop never performs when the trip count is zero. Even a pointer-typed read is
/// free only through a fat pointer that cannot fault: any other one is taken through
/// [`GuardReads`], which performs it only when a group will run.
///
/// A search loop ([`EarlyExit`]) is emitted in two blocks: the steps up to its exit test in `vb`,
/// which ends in the lane test and leaves for `vx` when any lane would leave the loop, and the rest
/// of the group in a block of its own. Returns the back edge's arguments and the block that carries
/// them.
#[allow(clippy::too_many_arguments)]
fn emit_body(
    e: &mut Emit<'_>,
    p: &VecPlan,
    mut vb: BlockId,
    vx: BlockId,
    guard: BlockId,
    reads: &mut GuardReads,
    header_params: &[ValueId],
    vh_params: &[ValueId],
) -> (Vec<ValueId>, BlockId) {
    let latch_term = e.f.blocks[p.body.0 as usize].term.clone();
    // The loop's own span ([`loop_loc`]), for what no single step owns: the lane ramp, the step of
    // the counter and the back edge. Each step and each reduction moves the cursor to its own.
    let loop_at = e.loc;
    let exit_at = p
        .early
        .and_then(|x| e.f.value_loc(x.cond))
        .unwrap_or(loop_at);

    // scalar value -> its counterpart in the vector body.
    let mut map: FxHashMap<u32, Wide> = FxHashMap::default();
    for (i, hp) in header_params.iter().enumerate() {
        map.insert(hp.0, Wide::Uniform(vh_params[i]));
    }
    // `(group, residue)` -> that member's parts: filled for every member of a load group by its
    // first member, and member by member for a store group until its last member stores them all.
    let mut group_vals: FxHashMap<(usize, u32), Vec<ValueId>> = FxHashMap::default();
    // The scratch slots predicated accesses aim their idle lanes at, one per element type.
    let mut scratch: Vec<(MirType, ValueId)> = Vec::new();

    // The lane ramp, when the body reads the induction variable as a value: `splat(i) + iota`, so
    // lane `k` holds `i + k` — the counter scalar iteration `i + k` would have had. The step is 1
    // (`iv_verdict`; a countdown's `-1` subtracts the same ramp, lane `k` holding `i - k`), and MIR
    // `Add` wraps exactly as the scalar `i + 1` chain does, so the two agree
    // even across an integer overflow. The `iota` is a constant with no operands, so LICM lifts it
    // into the guard and only the `splat` and the `add` remain per group. A counter wider than the
    // group's narrowest lane is split into parts like any other value: part `p` is
    // `splat(i + p·L) + iota`, which is again `i + k` in every lane by the same wrapping argument.
    //
    // Emitted once, before the body, and mapped in and out per instruction below: the *same*
    // induction variable is a `gep` index (uniform) and a compare operand (ramp) in the same block,
    // and the plan says which role each step is.
    let iv_scalar = vh_params[p.iv_index];
    // The counter's per-lane values `i + k`, built the first time a replicated step reads it.
    let mut iv_lanes: Option<Vec<ValueId>> = None;
    let ramp: Option<Vec<ValueId>> = p.ramp_vector.then(|| {
        let ty = p.ramp.ty.clone();
        let vt = part_ty(&ty);
        let per_part = lanes_of(&ty).expect("an integer induction variable has a vector width");
        let iota = e.push(vb, vt.clone(), Op::Iota(vt.clone()));
        (0..parts_of(&ty, p.w))
            .map(|k| {
                // A countdown's lane `k` holds `i - k`: the same ramp, subtracted.
                let toward = if p.dir > 0 { BinOp::Add } else { BinOp::Sub };
                let first = if k == 0 {
                    iv_scalar
                } else {
                    let off = e.int(vb, &ty, (k as u32 * per_part) as i64);
                    e.push(vb, ty.clone(), Op::Bin(toward, iv_scalar, off))
                };
                let base = e.push(vb, vt.clone(), Op::Splat(first));
                e.push(vb, vt.clone(), Op::Bin(toward, base, iota))
            })
            .collect()
    });

    for (i, step) in p.lin.iter().enumerate() {
        if let Some(x) = p.early.filter(|x| x.split == i) {
            e.loc = exit_at;
            vb = emit_exit_test(e, p, vb, vx, &map, x);
        }
        // A checked body's lane test sits on its own bounds check's line.
        for x in p.spine_exits.iter().filter(|x| x.split == i) {
            e.loc = e.f.value_loc(x.cond).unwrap_or(loop_at);
            vb = emit_exit_test(e, p, vb, vx, &map, *x);
        }
        e.loc = step_loc(e.f, step).unwrap_or(loop_at);
        // Give the induction variable its lane-varying face for exactly the steps that read it as a
        // value, and its scalar one everywhere else, so the address arithmetic around them stays
        // scalar. Set at the top of every iteration rather than restored at the bottom: the body
        // below leaves through half a dozen `continue`s and one missed restore would splat the
        // group's first counter into all `W` lanes.
        if let Some(r) = &ramp {
            let face = if p.ramp.value_use.contains(&i) {
                Wide::Vector(r.clone())
            } else {
                Wide::Uniform(iv_scalar)
            };
            map.insert(p.ramp.iv.0, face);
        }
        // A reduction's combine, a min/max's compare and a guarded accumulate's merge are all
        // folded after the body, below — whether the step is an instruction or a merge.
        if matches!(p.plans[i], Plan::Reduce) {
            continue;
        }
        // The lane-serial region runs once its last step is reached, after every step of it that
        // keeps its plan ([`SerialRegion`]).
        if let Plan::Serial { .. } = p.plans[i] {
            if p.region.is_some_and(|r| r.end == i) {
                emit_serial_region(e, p, vb, &mut map, &mut iv_lanes, iv_scalar);
            }
            continue;
        }
        // The two synthesized steps of if-conversion. Both are lane-wise `select`s: one feeds a
        // single store that stands in for the two arms' stores, the other defines the latch
        // parameter the arms were merging into.
        match (&p.plans[i], step) {
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
                let lane = e.f.value_type(*then_val).clone();
                let mask = mask_for(e, vb, &map, *cond, &lane);
                let a = to_vector(e, vb, &map, *then_val, &lane, p.w);
                let b = to_vector(e, vb, &map, *else_val, &lane, p.w);
                let ptr = resolve_uniform(&map, *ptr);
                for (k, ((m, x), y)) in mask.iter().zip(&a).zip(&b).enumerate() {
                    let blended = e.push(vb, part_ty(&lane), Op::Select(*m, *x, *y));
                    store_part(e, vb, ptr, &lane, k, blended, p.dir);
                }
                continue;
            }
            (
                Plan::Merge,
                LinInst::Merge {
                    param,
                    cond,
                    then_arg,
                    else_arg,
                },
            ) => {
                match p.vector_ty.get(&param.0) {
                    Some(lane) => {
                        let lane = lane.clone();
                        let mask = mask_for(e, vb, &map, *cond, &lane);
                        let a = to_vector(e, vb, &map, *then_arg, &lane, p.w);
                        let b = to_vector(e, vb, &map, *else_arg, &lane, p.w);
                        let nv = (0..a.len())
                            .map(|k| e.push(vb, part_ty(&lane), Op::Select(mask[k], a[k], b[k])))
                            .collect();
                        map.insert(param.0, Wide::Vector(nv));
                    }
                    None => {
                        let ty = e.f.value_type(*param).clone();
                        let c = resolve_uniform(&map, *cond);
                        let a = resolve_uniform(&map, *then_arg);
                        let b = resolve_uniform(&map, *else_arg);
                        let nv = e.push(vb, ty, Op::Select(c, a, b));
                        map.insert(param.0, Wide::Uniform(nv));
                    }
                }
                continue;
            }
            // A load one arm makes, lane by lane: each lane reads its own element if it took the
            // arm and the scratch slot if it did not, so no lane touches memory the scalar loop
            // leaves alone. The idle lanes' values are discarded by the arm's merge.
            (Plan::PredLoad, LinInst::PredLoad { block, idx, pred }) => {
                let inst = e.f.blocks[block.0 as usize].insts[*idx].clone();
                let Op::Load(ptr, ty) = &inst.op else {
                    unreachable!("a predicated load is a load")
                };
                let ptr = resolve_uniform(&map, *ptr);
                let bits = pred_lane_bools(e, vb, &map, pred, p.w);
                let slot = scratch_slot(e, &mut scratch, ty);
                let lanes = bits
                    .iter()
                    .enumerate()
                    .map(|(k, &bit)| {
                        let own = elem_ptr(e, vb, ptr, ty, p.dir * k as i64);
                        let at = e.push(vb, MirType::Ptr, Op::Select(bit, own, slot));
                        e.push(vb, ty.clone(), Op::Load(at, ty.clone()))
                    })
                    .collect();
                map.insert(
                    inst.result.expect("a load has a result").0,
                    Wide::Lanes(lanes),
                );
                continue;
            }
            // A store one arm makes, lane by lane, in lane order: to the lane's own element where
            // the lane took the arm, to the scratch slot where it did not — a masked store without
            // a read-modify-write of the elements it skips.
            (
                Plan::PredStore,
                LinInst::PredStore {
                    ptr,
                    value,
                    cond,
                    on_true,
                    ..
                },
            ) => {
                let lane_ty = e.f.value_type(*value).clone();
                let ptr = resolve_uniform(&map, *ptr);
                let vals = lane_values(e, vb, &map, *value, p.w);
                let bits = mask_lane_bools(e, vb, &map, *cond, p.w);
                let slot = scratch_slot(e, &mut scratch, &lane_ty);
                for (k, (&v, &bit)) in vals.iter().zip(&bits).enumerate() {
                    let own = elem_ptr(e, vb, ptr, &lane_ty, p.dir * k as i64);
                    let (a, b) = if *on_true { (own, slot) } else { (slot, own) };
                    let at = e.push(vb, MirType::Ptr, Op::Select(bit, a, b));
                    e.push_void(vb, Op::Store { ptr: at, value: v });
                }
                continue;
            }
            // Every path of a nested region stores: one vector store per part of the select tree.
            (Plan::TreeStore, LinInst::TreeStore { ptr, tree, .. }) => {
                let lane = tree_lane_ty(e, tree);
                let ptr = resolve_uniform(&map, *ptr);
                let parts = tree_vector(e, vb, &map, tree, &lane, p.w);
                for (k, value) in parts.into_iter().enumerate() {
                    store_part(e, vb, ptr, &lane, k, value, p.dir);
                }
                continue;
            }
            // Some path stores nothing: per lane, in lane order, the tree's value to the lane's own
            // element where its path stores and to the scratch slot where it does not.
            (Plan::TreePredStore, LinInst::TreeStore { ptr, tree, .. }) => {
                let lane = tree_lane_ty(e, tree);
                let ptr = resolve_uniform(&map, *ptr);
                let slot = scratch_slot(e, &mut scratch, &lane);
                let (bits, vals) = tree_lanes(e, vb, &map, tree, p.w);
                let vals = vals.expect("a store tree with a store has a value");
                for (k, (&v, &bit)) in vals.iter().zip(&bits).enumerate() {
                    let own = elem_ptr(e, vb, ptr, &lane, p.dir * k as i64);
                    let at = e.push(vb, MirType::Ptr, Op::Select(bit, own, slot));
                    e.push_void(vb, Op::Store { ptr: at, value: v });
                }
                continue;
            }
            _ => {}
        }
        let LinInst::Real { block, idx } = step else {
            unreachable!("a synthesized step is handled above")
        };
        let inst = e.f.blocks[block.0 as usize].insts[*idx].clone();
        let inst = &inst;
        match &p.plans[i] {
            Plan::MaskedStore
            | Plan::Merge
            | Plan::PredLoad
            | Plan::PredStore
            | Plan::TreeStore
            | Plan::TreePredStore
            | Plan::Serial { .. } => {
                unreachable!("handled above")
            }
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
            Plan::InvariantPtrLoad => {
                let Op::Load(ptr, ty) = &inst.op else {
                    unreachable!("InvariantPtrLoad plan on a non-load")
                };
                // Hoist only if the address itself is available where the guard runs. A value the
                // map does not hold is defined outside the loop, so it dominates the preheader and
                // therefore the guard; anything else (a pointer carried in a header parameter) has
                // its counterpart in the vector *header*, which the guard does not dominate. And
                // only if the guard may read it ([`GuardReads`]): a search loop can leave before
                // it reaches the load, so there only a fat pointer that cannot fault is hoisted,
                // and any other read stays where the group reaches it.
                let in_place = map.contains_key(&ptr.0)
                    || ((p.early.is_some() || p.spine) && !never_faults(e.f, *ptr, 8));
                let nv = if in_place {
                    let ptr = resolve_uniform(&map, *ptr);
                    e.push(vb, ty.clone(), Op::Load(ptr, ty.clone()))
                } else {
                    reads.load_ptr(e, guard, *ptr)
                };
                map.insert(
                    inst.result.expect("a load has a result").0,
                    Wide::Uniform(nv),
                );
            }
            Plan::VecLoad => {
                let Op::Load(ptr, ty) = &inst.op else {
                    unreachable!("VecLoad plan on a non-load")
                };
                let ptr = resolve_uniform(&map, *ptr);
                let parts = (0..parts_of(ty, p.w))
                    .map(|k| load_part(e, vb, ptr, ty, k, p.dir))
                    .collect();
                map.insert(
                    inst.result.expect("load has a result").0,
                    Wide::Vector(parts),
                );
            }
            Plan::VecStore => {
                let Op::Store { ptr, value } = &inst.op else {
                    unreachable!("VecStore plan on a non-store")
                };
                let lane = e.f.value_type(*value).clone();
                let ptr = resolve_uniform(&map, *ptr);
                let values = to_vector(e, vb, &map, *value, &lane, p.w);
                for (k, value) in values.into_iter().enumerate() {
                    store_part(e, vb, ptr, &lane, k, value, p.dir);
                }
            }
            Plan::Replicate | Plan::LaneGep | Plan::Call => {
                // A call made only for its trap has no result; every other replicated op has one.
                let r = inst.result;
                let ty = r.map(|r| e.f.value_type(r).clone());
                // Each operand's `W` per-lane values: the counter's `i + k` where this step reads
                // it as a value, otherwise extracted from a vector (a compare's mask as each
                // lane's `bool`), a replicated value's own lanes, or a uniform value repeated.
                let mut uses = Vec::new();
                each_op_use(&inst.op, &mut |u| uses.push(u));
                let mut table: FxHashMap<u32, Vec<ValueId>> = FxHashMap::default();
                for u in uses {
                    if table.contains_key(&u.0) {
                        continue;
                    }
                    let lanes = if p.ramp.reads_as_value(i, u) {
                        iv_lanes
                            .get_or_insert_with(|| {
                                iv_lane_values(e, vb, iv_scalar, &p.iv_ty, p.w, p.dir)
                            })
                            .clone()
                    } else {
                        scalar_lanes(e, vb, &map, u, p.w)
                    };
                    table.insert(u.0, lanes);
                }
                let out: Vec<Option<ValueId>> = (0..p.w as usize)
                    .map(|k| {
                        let mut op = inst.op.clone();
                        crate::map_op_uses(&mut op, |u| table[&u.0][k]);
                        match &ty {
                            Some(ty) => Some(e.push(vb, ty.clone(), op)),
                            None => {
                                e.push_void(vb, op);
                                None
                            }
                        }
                    })
                    .collect();
                if let Some(r) = r {
                    map.insert(r.0, Wide::Lanes(out.into_iter().flatten().collect()));
                }
            }
            Plan::Gather => {
                let Op::Load(ptr, ty) = &inst.op else {
                    unreachable!("Gather plan on a non-load")
                };
                let addrs = match map.get(&ptr.0) {
                    Some(Wide::Lanes(x)) => x.clone(),
                    _ => unreachable!("plan_body proved a gather reads per-lane addresses"),
                };
                let lanes: Vec<ValueId> = addrs
                    .into_iter()
                    .map(|q| e.push(vb, ty.clone(), Op::Load(q, ty.clone())))
                    .collect();
                let parts = pack_lanes(e, vb, &lanes, ty);
                map.insert(
                    inst.result.expect("a load has a result").0,
                    Wide::Vector(parts),
                );
            }
            // One scalar store per lane, lane 0 first: lane `k` is iteration `i + dir·k`, so two
            // lanes writing one element leave the later iteration's value, as the scalar loop does.
            Plan::Scatter => {
                let Op::Store { ptr, value } = &inst.op else {
                    unreachable!("Scatter plan on a non-store")
                };
                let addrs = match map.get(&ptr.0) {
                    Some(Wide::Lanes(x)) => x.clone(),
                    _ => unreachable!("plan_body proved a scatter writes per-lane addresses"),
                };
                let vals = if p.ramp.reads_as_value(i, *value) {
                    iv_lanes
                        .get_or_insert_with(|| {
                            iv_lane_values(e, vb, iv_scalar, &p.iv_ty, p.w, p.dir)
                        })
                        .clone()
                } else {
                    lane_values(e, vb, &map, *value, p.w)
                };
                for (q, v) in addrs.into_iter().zip(vals) {
                    e.push_void(vb, Op::Store { ptr: q, value: v });
                }
            }
            Plan::GroupLoad { group, offset } => {
                let Op::Load(ptr, _) = &inst.op else {
                    unreachable!("GroupLoad plan on a non-load")
                };
                if !group_vals.contains_key(&(*group, *offset)) {
                    let ptr = resolve_uniform(&map, *ptr);
                    emit_group_load(e, vb, ptr, *offset, *group, p, &mut group_vals);
                }
                map.insert(
                    inst.result.expect("load has a result").0,
                    Wide::Vector(group_vals[&(*group, *offset)].clone()),
                );
            }
            Plan::GroupStore { group, offset } => {
                let Op::Store { ptr, value } = &inst.op else {
                    unreachable!("GroupStore plan on a non-store")
                };
                let lane = e.f.value_type(*value).clone();
                let values = to_vector(e, vb, &map, *value, &lane, p.w);
                group_vals.insert((*group, *offset), values);
                let k = p.groups[*group].k;
                if (0..k).all(|r| group_vals.contains_key(&(*group, r))) {
                    let ptr = resolve_uniform(&map, *ptr);
                    emit_group_store(e, vb, ptr, *offset, *group, p, &group_vals);
                }
            }
            Plan::Widen => {
                let r = inst.result.expect("a widened op has a result");
                let lane = p.vector_ty[&r.0].clone();
                let parts = emit_widened(e, vb, &map, &inst.op, p.w, &lane);
                map.insert(r.0, Wide::Vector(parts));
            }
            Plan::Cast(route) => {
                let Op::Cast(kind, x, to) = &inst.op else {
                    unreachable!("Cast plan on a non-cast")
                };
                let r = inst.result.expect("a cast has a result");
                let from = e.f.value_type(*x).clone();
                let xs = to_vector(e, vb, &map, *x, &from, p.w);
                let parts = emit_cast(e, vb, *kind, xs, &from, to, *route);
                map.insert(r.0, Wide::Vector(parts));
            }
        }
    }

    // A search loop whose latch computes nothing but the step tests after every body step.
    if let Some(x) = p.early.filter(|x| x.split == p.lin.len()) {
        e.loc = exit_at;
        vb = emit_exit_test(e, p, vb, vx, &map, x);
    }
    for x in p.spine_exits.iter().filter(|x| x.split == p.lin.len()) {
        e.loc = e.f.value_loc(x.cond).unwrap_or(loop_at);
        vb = emit_exit_test(e, p, vb, vx, &map, *x);
    }

    // Past the body the induction variable is scalar again: the reduction folds and the back-edge
    // arguments below both read it (or values it feeds) through `resolve_uniform`, and the
    // induction variable's own back-edge argument is the `+ W` step.
    if ramp.is_some() {
        map.insert(p.ramp.iv.0, Wide::Uniform(iv_scalar));
    }

    // The accumulates, after every load and store of the group has happened. Moving them here is
    // safe: an accumulator is a register value, so no store in the body can change what it reads,
    // and its own operands are all defined above.
    //
    // A reassociable reduction just combines the whole addend into its vector accumulator — one
    // instruction per part, `W` independent partials. A float one folds `W` lanes into the scalar
    // accumulator in index order, which is a `W`-long dependent chain and is the price of not
    // reassociating.
    let mut acc_of: FxHashMap<usize, ValueId> = FxHashMap::default();
    let mut vacc_next: Vec<ValueId> = Vec::new();
    let header_len = header_params.len();
    for r in &p.reductions {
        e.loc = red_loc(e.f, p, r, loop_at);
        if let Some((_, ity)) = &r.arg {
            // An argmax: per part, the loop's own compare with the partial maximum in the
            // accumulator's position, a select of the value, and a select of the counter's lane
            // ramp into the index partials under the same lane mask — lane `k` runs exactly the
            // scalar loop's update over iterations `k, k + W, k + 2W, ..`.
            let s = r.select.expect("an argmax is a min/max");
            let n = parts_of(&r.ty, p.w);
            let na = parts_of(ity, p.w);
            let first = header_len + vacc_next.len();
            let cur = vh_params[first..first + n].to_vec();
            let cur_at = vh_params[first + n..first + n + na].to_vec();
            let (a, _) = r.addend.expect("an argmax has a compared value");
            let xs = to_vector(e, vb, &map, a, &r.ty, p.w);
            let pty = part_ty(&r.ty);
            let mty = vec_of(
                &mask_lane(&r.ty),
                lanes_of(&r.ty).expect("a widened reduction type"),
            );
            let mut masks = Vec::with_capacity(n);
            for (&acc, &x) in cur.iter().zip(&xs) {
                let (l, rhs) = if s.acc_left_in_cmp {
                    (acc, x)
                } else {
                    (x, acc)
                };
                let c = e.push(vb, mty.clone(), Op::Cmp(s.pred, l, rhs));
                masks.push(c);
                vacc_next.push(e.push(vb, pty.clone(), Op::Select(c, x, acc)));
            }
            let mut mmap: FxHashMap<u32, Wide> = FxHashMap::default();
            mmap.insert(s.cmp.0, Wide::Vector(masks));
            let at_masks = mask_for(e, vb, &mmap, s.cmp, ity);
            let ramp = ramp.as_ref().expect("an argmax builds the lane ramp");
            let ity_part = part_ty(ity);
            for k in 0..na {
                vacc_next.push(e.push(
                    vb,
                    ity_part.clone(),
                    Op::Select(at_masks[k], ramp[k], cur_at[k]),
                ));
            }
            continue;
        }
        if r.find_last {
            // Lane `k` keeps the counter of the last iteration in its sub-sequence whose condition
            // held: the scalar merge, with the counter's lane ramp as the recorded value.
            let n = parts_of(&r.ty, p.w);
            let first = header_len + vacc_next.len();
            let cur = vh_params[first..first + n].to_vec();
            let (cond, on_true) = r.guard.expect("a FindLast is a guarded reduction");
            let mask = mask_for(e, vb, &map, cond, &r.ty);
            let ramp = ramp.as_ref().expect("a FindLast builds the lane ramp");
            let pty = part_ty(&r.ty);
            for k in 0..n {
                let (a, b) = if on_true {
                    (ramp[k], cur[k])
                } else {
                    (cur[k], ramp[k])
                };
                vacc_next.push(e.push(vb, pty.clone(), Op::Select(mask[k], a, b)));
            }
            continue;
        }
        if r.vector_acc {
            let n = parts_of(&r.ty, p.w);
            let first = header_len + vacc_next.len();
            let cur = vh_params[first..first + n].to_vec();
            let (a, _) = r.addend.expect("vector_acc implies a binary combine");
            let addend = to_vector(e, vb, &map, a, &r.ty, p.w);
            let next: Vec<ValueId> = match r.select {
                // An integer min/max: the compare and select, part by part.
                Some(_) => {
                    let pty = part_ty(&r.ty);
                    cur.iter()
                        .zip(&addend)
                        .map(|(&acc, &x)| emit_combine(e, vb, r, true, acc, x, &pty))
                        .collect()
                }
                None => emit_bin(e, vb, red_binop(r.kind), &cur, &addend, &r.ty),
            };
            // A guarded accumulate keeps each lane's update only where that lane's condition chose
            // it: `select(mask, acc ⊕ x, acc)` per part, the scalar merge lane for lane.
            let next = match r.guard {
                None => next,
                Some((cond, on_true)) => {
                    let mask = mask_for(e, vb, &map, cond, &r.ty);
                    let pty = part_ty(&r.ty);
                    (0..next.len())
                        .map(|k| {
                            let (a, b) = if on_true {
                                (next[k], cur[k])
                            } else {
                                (cur[k], next[k])
                            };
                            e.push(vb, pty.clone(), Op::Select(mask[k], a, b))
                        })
                        .collect()
                }
            };
            vacc_next.extend(next);
            continue;
        }
        let init = vh_params[r.param_index];
        let out = emit_reduction_fold(e, vb, &map, r, init, p.w);
        acc_of.insert(r.param_index, out);
    }

    e.loc = loop_at;

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
            let step = e.int(vb, &p.iv_ty, p.dir * p.w as i64);
            out.push(e.push(vb, p.iv_ty.clone(), Op::Bin(BinOp::Add, vh_params[k], step)));
        } else if let Some(&acc) = acc_of.get(&k) {
            out.push(acc);
        } else if p.reductions.iter().any(|r| {
            r.vector_acc && (r.param_index == k || r.arg.as_ref().is_some_and(|a| a.0 == k))
        }) {
            // The scalar accumulator of a vector-accumulated reduction is carried unchanged; the
            // partials live in the extra parameters appended below.
            out.push(vh_params[k]);
        } else {
            out.push(resolve_uniform(&map, *a));
        }
    }
    out.extend(vacc_next);
    // The scratch slots live in the entry block, allocated once per call. Inserted last: nothing
    // after this point reads an instruction index.
    let entry = e.f.entry;
    for (ty, slot) in scratch.into_iter().rev() {
        e.f.blocks[entry.0 as usize].insts.insert(
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
    (out, vb)
}

/// The lane-serial region's steps, lane by lane: for lane `k = 0 .. W`, every [`Plan::Serial`]
/// step in body order, on lane `k`'s operands — its own earlier results, the counter's `i + k`
/// where the step reads it as a value, lane `k` of anything computed before the region — and, for
/// a contiguous access, the group's address plus `k` elements. Then each result is the lanes it
/// took ([`Wide::Lanes`]), packed where a widened step after the region reads it.
fn emit_serial_region(
    e: &mut Emit<'_>,
    p: &VecPlan,
    vb: BlockId,
    map: &mut FxHashMap<u32, Wide>,
    iv_lanes: &mut Option<Vec<ValueId>>,
    iv_scalar: ValueId,
) {
    let r = p.region.expect("a serial step implies a region");
    let steps: Vec<(usize, Inst, bool)> = (r.start..=r.end)
        .filter_map(|i| {
            let Plan::Serial { contiguous } = p.plans[i] else {
                return None;
            };
            let LinInst::Real { block, idx } = p.lin[i] else {
                unreachable!("a lane-serial region is straight-line code")
            };
            Some((
                i,
                e.f.blocks[block.0 as usize].insts[idx].clone(),
                contiguous,
            ))
        })
        .collect();
    let own: FxHashSet<u32> = steps
        .iter()
        .filter_map(|(_, inst, _)| inst.result.map(|r| r.0))
        .collect();
    // A contiguous access's address operand: one scalar for the group, offset per lane.
    let address = |inst: &Inst, contiguous: bool| match (&inst.op, contiguous) {
        (Op::Load(ptr, _), true) | (Op::Store { ptr, .. }, true) => Some(*ptr),
        _ => None,
    };
    // Each step's copies, and the lane values read for it, keep that step's own span ([`inst_loc`]);
    // the caller's cursor, the region's last step, is the fallback and is restored after.
    let region_at = e.loc;
    // Lane values of everything the region reads from outside it, read once, before lane 0 — a
    // compare widened before the region as each lane's `bool` ([`scalar_lanes`]).
    let mut table: FxHashMap<u32, Vec<ValueId>> = FxHashMap::default();
    for (i, inst, contiguous) in &steps {
        e.loc = inst_loc(e.f, inst).unwrap_or(region_at);
        let mut uses = Vec::new();
        each_op_use(&inst.op, &mut |u| uses.push(u));
        for u in uses {
            if Some(u) == address(inst, *contiguous) || own.contains(&u.0) {
                continue;
            }
            if p.ramp.reads_as_value(*i, u) {
                iv_lanes
                    .get_or_insert_with(|| iv_lane_values(e, vb, iv_scalar, &p.iv_ty, p.w, p.dir));
            } else if u == p.ramp.iv {
                table.insert(u.0, vec![iv_scalar; p.w as usize]);
            } else if let std::collections::hash_map::Entry::Vacant(slot) = table.entry(u.0) {
                slot.insert(scalar_lanes(e, vb, map, u, p.w));
            }
        }
    }
    let mut out: FxHashMap<u32, Vec<ValueId>> = FxHashMap::default();
    for k in 0..p.w as usize {
        for (i, inst, contiguous) in &steps {
            e.loc = inst_loc(e.f, inst).unwrap_or(region_at);
            let at = address(inst, *contiguous).map(|ptr| {
                let elem = match &inst.op {
                    Op::Load(_, ty) => ty.clone(),
                    Op::Store { value, .. } => e.f.value_type(*value).clone(),
                    _ => unreachable!("a contiguous step is an access"),
                };
                let base = resolve_uniform(map, ptr);
                elem_ptr(e, vb, base, &elem, p.dir * k as i64)
            });
            let mut op = inst.op.clone();
            let ptr = address(inst, *contiguous);
            crate::map_op_uses(&mut op, |u| {
                if Some(u) == ptr {
                    at.expect("a contiguous access has its lane address")
                } else if let Some(v) = out.get(&u.0) {
                    v[k]
                } else if p.ramp.reads_as_value(*i, u) {
                    iv_lanes.as_ref().expect("built above")[k]
                } else {
                    table[&u.0][k]
                }
            });
            match inst.result {
                Some(r) => {
                    let ty = e.f.value_type(r).clone();
                    let v = e.push(vb, ty, op);
                    out.entry(r.0).or_default().push(v);
                }
                None => e.push_void(vb, op),
            }
        }
    }
    e.loc = region_at;
    for (r, lanes) in out {
        map.insert(r, Wide::Lanes(lanes));
    }
}

/// Each lane's truth of the path predicate `pred` (every `(cond, polarity)` holds), as an `i1`
/// per lane, lowest iteration first. One literal is the lane mask's own bits; a nested arm's is
/// their conjunction, a negated literal read through a `select` rather than a bitwise not (a mask
/// lane is `1` in the interpreter and all-ones natively, and only `select` reads both alike).
fn pred_lane_bools(
    e: &mut Emit<'_>,
    vb: BlockId,
    map: &FxHashMap<u32, Wide>,
    pred: &[(ValueId, bool)],
    w: u32,
) -> Vec<ValueId> {
    let t = e.push(vb, MirType::I1, Op::ConstInt(1, MirType::I1));
    let fa = e.push(vb, MirType::I1, Op::ConstInt(0, MirType::I1));
    let mut acc: Option<Vec<ValueId>> = None;
    for &(c, pol) in pred {
        let bits = mask_lane_bools(e, vb, map, c, w);
        let bits: Vec<ValueId> = if pol {
            bits
        } else {
            bits.into_iter()
                .map(|b| e.push(vb, MirType::I1, Op::Select(b, fa, t)))
                .collect()
        };
        acc = Some(match acc {
            None => bits,
            Some(prev) => prev
                .into_iter()
                .zip(bits)
                .map(|(a, b)| e.push(vb, MirType::I1, Op::Bin(BinOp::And, a, b)))
                .collect(),
        });
    }
    acc.unwrap_or_else(|| vec![t; w as usize])
}

/// The type a store tree stores: its first value's.
fn tree_lane_ty(e: &Emit<'_>, tree: &StoreTree) -> MirType {
    let (mut conds, mut vals) = (Vec::new(), Vec::new());
    tree.operands(&mut conds, &mut vals);
    e_ty(e, vals[0])
}

/// A complete store tree as parts: each node a lane-wise `select` of its two subtrees under its
/// condition's mask, re-expressed at the stored lane's width.
fn tree_vector(
    e: &mut Emit<'_>,
    vb: BlockId,
    map: &FxHashMap<u32, Wide>,
    tree: &StoreTree,
    lane: &MirType,
    w: u32,
) -> Vec<ValueId> {
    match tree {
        StoreTree::Val(v) => to_vector(e, vb, map, *v, lane, w),
        StoreTree::Node { cond, then, els } => {
            let mask = mask_for(e, vb, map, *cond, lane);
            let a = tree_vector(e, vb, map, then, lane, w);
            let b = tree_vector(e, vb, map, els, lane, w);
            (0..a.len())
                .map(|k| e.push(vb, part_ty(lane), Op::Select(mask[k], a[k], b[k])))
                .collect()
        }
        StoreTree::Keep => unreachable!("a complete store tree has no Keep leaf"),
    }
}

/// A store tree lane by lane: whether lane `k`'s path stores (an `i1`), and the value it stores.
/// Where a node has one storing side, that side's value stands for both, since a lane on the other
/// side stores nothing and its value is never written.
fn tree_lanes(
    e: &mut Emit<'_>,
    vb: BlockId,
    map: &FxHashMap<u32, Wide>,
    tree: &StoreTree,
    w: u32,
) -> (Vec<ValueId>, Option<Vec<ValueId>>) {
    match tree {
        StoreTree::Keep => {
            let fa = e.push(vb, MirType::I1, Op::ConstInt(0, MirType::I1));
            (vec![fa; w as usize], None)
        }
        StoreTree::Val(v) => {
            let t = e.push(vb, MirType::I1, Op::ConstInt(1, MirType::I1));
            (vec![t; w as usize], Some(lane_values(e, vb, map, *v, w)))
        }
        StoreTree::Node { cond, then, els } => {
            let c = mask_lane_bools(e, vb, map, *cond, w);
            let (tb, tv) = tree_lanes(e, vb, map, then, w);
            let (eb, ev) = tree_lanes(e, vb, map, els, w);
            let bits = (0..w as usize)
                .map(|k| e.push(vb, MirType::I1, Op::Select(c[k], tb[k], eb[k])))
                .collect();
            let vals = match (tv, ev) {
                (Some(a), Some(b)) => Some(
                    (0..w as usize)
                        .map(|k| {
                            let ty = e_ty(e, a[k]);
                            e.push(vb, ty, Op::Select(c[k], a[k], b[k]))
                        })
                        .collect(),
                ),
                (a, b) => a.or(b),
            };
            (bits, vals)
        }
    }
}

/// The scratch slot a predicated access of `ty` aims its idle lanes at: one private stack slot per
/// element type, whose address escapes nowhere but into those selects, so writing it is invisible
/// and reading it reads a value the arm's merge discards.
fn scratch_slot(e: &mut Emit<'_>, scratch: &mut Vec<(MirType, ValueId)>, ty: &MirType) -> ValueId {
    if let Some(&(_, slot)) = scratch.iter().find(|(t, _)| t == ty) {
        return slot;
    }
    let slot = e.value(MirType::Ptr);
    scratch.push((ty.clone(), slot));
    slot
}

/// End `vb` in a search loop's exit test and continue the group in a new block, returned.
///
/// The group leaves for `vx` — handing the original loop the group's first iteration and the values
/// it started with, since nothing after the test has run — when any of its `W` lanes would leave the
/// loop; the original loop then re-runs at most `W` iterations and stops at exactly the one it
/// always stopped at. Otherwise every lane continues and so does the group.
fn emit_exit_test(
    e: &mut Emit<'_>,
    p: &VecPlan,
    vb: BlockId,
    vx: BlockId,
    map: &FxHashMap<u32, Wide>,
    x: EarlyExit,
) -> BlockId {
    // Continuing on true, every lane must hold; continuing on false, no lane may.
    let (fold, all_stay) = if x.continue_on_true {
        (BinOp::And, true)
    } else {
        (BinOp::Or, false)
    };
    let verdict = fold_lanes(e, vb, map, x.cond, p.w, fold);
    let rest = e.block(Vec::new(), Terminator::Unreachable);
    let (then_blk, else_blk) = if all_stay { (rest, vx) } else { (vx, rest) };
    e.f.blocks[vb.0 as usize].term = Terminator::CondBr {
        cond: verdict,
        then_blk,
        then_args: Vec::new(),
        else_blk,
        else_args: Vec::new(),
    };
    rest
}

/// Combine an argmax's per-lane `(value, index)` partials into the scalar `(m, at)` — once, in the
/// vector exit — so that the result is bit-identical to the scalar loop's.
///
/// Lane `k` ran the scalar update `c ? (x, i) : (m, at)` over iterations `k, k + W, ..` from the
/// loop's own start `(m0, a0)`; `plan_body` admitted only an update on the compare's `then` arm.
/// With a strict compare the scalar answer is then the **first** iteration whose value is
/// strictly better than every earlier one and not beaten later — the earliest member of the best
/// class (`-0.0` and `+0.0` are one class: neither is `>` the other) — or `(m0, a0)` when nothing
/// beat `m0`.
/// Every ordered compare against a NaN is false, so a NaN element never updates anything, and a
/// NaN `m0` updates nothing at all.
///
/// Each lane holds the earliest member of its own best class, so the answer is the lane whose
/// value is best, ties going to the lane whose index is **earlier**. The fold walks the lanes and
/// takes lane `k` over the running `(M, A)` when the loop's own compare says `m_k` beats `M`, or
/// when the two are equal (`==`, so `±0` tie) and `a_k` came first. Three cases close the argument:
///
/// * a lane that never updated still holds `(m0, a0)` bit for bit, and every lane that did holds
///   a value strictly better than `m0` — so an un-updated lane only ever ties with another
///   un-updated lane, and both are the same pair; the index compare never decides between `a0`
///   and a real index;
/// * between updated lanes the indices are distinct counter values from this vector loop, which
///   are ordered by iteration once taken relative to its first counter: `(a - start)` as unsigned
///   is the iteration number, wrap-around included;
/// * a NaN never compares equal or better, so a NaN `m0` (every lane un-updated) leaves lane 0's
///   pair — `(m0, a0)` — exactly as the scalar loop does.
///
/// The scalar epilogue then continues from `(M, A)` over the iterations after the vector loop's.
///
/// # The last-tie form (`>=` / `<=`, [`RedPlan::arg_last`])
///
/// A non-strict compare moves the index on a tie, so the scalar answer is the **last** element of
/// the best class — or `(m0, a0)` when no element reached `m0`. Each lane holds the last member of
/// its own best class, so the answer is the lane whose value is best, ties going to the lane whose
/// index is **later**. What changes is the un-updated lane: an updated lane may now hold a value
/// *equal* to `m0` (it reached `m0` and moved), so the index alone has to say which lanes updated.
/// Every index lane therefore starts at `start - 1`, and the fold compares `q = at - (start - 1)`
/// (wrapping), which is `0` exactly for a lane that never updated and `j + 1` for one whose last
/// update was iteration `start + j`. The fold takes lane `k` over the running `(M, Q)` when the
/// compare's strict form says `m_k` beats `M`, or when the two are equal and `q_k > Q` (unsigned).
/// An un-updated lane (`q = 0`) never displaces anything it ties, and a NaN still compares neither
/// better nor equal, so a NaN `m0` again leaves lane 0's `(m0, 0)`. `Q = 0` at the end means no
/// lane updated, and the index is the scalar loop's own `prior`; the value is lane 0's `m0`.
#[allow(clippy::too_many_arguments)]
fn emit_arg_combine(
    e: &mut Emit<'_>,
    vx: BlockId,
    r: &RedPlan,
    ity: &MirType,
    val_parts: &[ValueId],
    at_parts: &[ValueId],
    start: ValueId,
    prior: ValueId,
) -> (ValueId, ValueId) {
    let s = r.select.expect("an argmax is a min/max");
    let lanes = |e: &mut Emit<'_>, parts: &[ValueId], ty: &MirType| -> Vec<ValueId> {
        let per = lanes_of(ty).expect("a widened lane type");
        let mut out = Vec::new();
        for &q in parts {
            for k in 0..per {
                out.push(e.push(vx, ty.clone(), Op::ExtractLane(q, k)));
            }
        }
        out
    };
    let ms = lanes(e, val_parts, &r.ty);
    let ats = lanes(e, at_parts, ity);
    let eq = if r.ty.is_float() {
        CmpOp::Foeq
    } else {
        CmpOp::Eq
    };
    if r.arg_last {
        let strict = strict_cmp(s.pred);
        let one = e.int(vx, ity, 1);
        let below = e.push(vx, ity.clone(), Op::Bin(BinOp::Sub, start, one));
        let qs: Vec<ValueId> = ats
            .iter()
            .map(|&a| e.push(vx, ity.clone(), Op::Bin(BinOp::Sub, a, below)))
            .collect();
        let (mut m, mut q) = (ms[0], qs[0]);
        for (&mk, &qk) in ms.iter().zip(&qs).skip(1) {
            let (l, rhs) = if s.acc_left_in_cmp { (m, mk) } else { (mk, m) };
            let better = e.push(vx, MirType::I1, Op::Cmp(strict, l, rhs));
            let same = e.push(vx, MirType::I1, Op::Cmp(eq, mk, m));
            let later = e.push(vx, MirType::I1, Op::Cmp(CmpOp::Ugt, qk, q));
            let tie = e.push(vx, MirType::I1, Op::Bin(BinOp::And, same, later));
            let take = e.push(vx, MirType::I1, Op::Bin(BinOp::Or, better, tie));
            m = e.push(vx, r.ty.clone(), Op::Select(take, mk, m));
            q = e.push(vx, ity.clone(), Op::Select(take, qk, q));
        }
        let zero = e.int(vx, ity, 0);
        let found = e.push(vx, MirType::I1, Op::Cmp(CmpOp::Ne, q, zero));
        let last = e.push(vx, ity.clone(), Op::Bin(BinOp::Add, below, q));
        let at = e.push(vx, ity.clone(), Op::Select(found, last, prior));
        return (m, at);
    }
    let (mut m, mut at) = (ms[0], ats[0]);
    for (&mk, &ak) in ms.iter().zip(&ats).skip(1) {
        let (l, rhs) = if s.acc_left_in_cmp { (m, mk) } else { (mk, m) };
        let better = e.push(vx, MirType::I1, Op::Cmp(s.pred, l, rhs));
        let same = e.push(vx, MirType::I1, Op::Cmp(eq, mk, m));
        let ok = e.push(vx, ity.clone(), Op::Bin(BinOp::Sub, ak, start));
        let oa = e.push(vx, ity.clone(), Op::Bin(BinOp::Sub, at, start));
        let earlier = e.push(vx, MirType::I1, Op::Cmp(CmpOp::Ult, ok, oa));
        let tie = e.push(vx, MirType::I1, Op::Bin(BinOp::And, same, earlier));
        let take = e.push(vx, MirType::I1, Op::Bin(BinOp::Or, better, tie));
        m = e.push(vx, r.ty.clone(), Op::Select(take, mk, m));
        at = e.push(vx, ity.clone(), Op::Select(take, ak, at));
    }
    (m, at)
}

/// Combine a FindLastIV's per-lane partials into the scalar `last` — once, in the vector exit.
///
/// Every lane started at `start - 1` and records only counters of the vector loop's own
/// iterations, `start .. start + n_vec`. Shifted by one more, `q = at - (start - 1)` (wrapping) is
/// `0` for a lane that recorded nothing and `j + 1` for one whose last record was iteration
/// `start + j` — so the unsigned maximum of `q` over the lanes is the latest iteration any lane
/// recorded, exactly the scalar loop's last record. `q == 0` everywhere means nothing was
/// recorded, and the scalar value from before the loop (`prior`) stands — which is also what the
/// scalar loop holds. No compare ever sees the start value against a recorded counter.
fn emit_find_last_combine(
    e: &mut Emit<'_>,
    vx: BlockId,
    ty: &MirType,
    parts: &[ValueId],
    start: ValueId,
    prior: ValueId,
) -> ValueId {
    let one = e.int(vx, ty, 1);
    let below = e.push(vx, ty.clone(), Op::Bin(BinOp::Sub, start, one));
    let per = lanes_of(ty).expect("a widened lane type");
    let mut best: Option<ValueId> = None;
    for &q in parts {
        for k in 0..per {
            let at = e.push(vx, ty.clone(), Op::ExtractLane(q, k));
            let off = e.push(vx, ty.clone(), Op::Bin(BinOp::Sub, at, below));
            best = Some(match best {
                None => off,
                Some(b) => {
                    let later = e.push(vx, MirType::I1, Op::Cmp(CmpOp::Ugt, off, b));
                    e.push(vx, ty.clone(), Op::Select(later, off, b))
                }
            });
        }
    }
    let best = best.expect("at least one lane");
    let zero = e.int(vx, ty, 0);
    let found = e.push(vx, MirType::I1, Op::Cmp(CmpOp::Ne, best, zero));
    let last = e.push(vx, ty.clone(), Op::Bin(BinOp::Add, below, best));
    e.push(vx, ty.clone(), Op::Select(found, last, prior))
}

/// `fold` (`And` or `Or`) over every lane of the lane mask `c`, as one `i1`: the parts combined
/// lane-wise, then a halving tree of lane shuffles, then lane 0 against zero — `log2` of the lane
/// count steps instead of one extract per lane. A mask lane is all-ones natively and `1` in the
/// interpreter, zero in both when false, so the bitwise fold and the `!= 0` test read the same truth
/// on every backend. A condition that is uniform across the group is its own answer.
fn fold_lanes(
    e: &mut Emit<'_>,
    vb: BlockId,
    map: &FxHashMap<u32, Wide>,
    c: ValueId,
    w: u32,
    fold: BinOp,
) -> ValueId {
    let parts = match map.get(&c.0) {
        None => return c,
        Some(Wide::Uniform(x)) => return *x,
        Some(Wide::Lanes(_)) => {
            let bits = mask_lane_bools(e, vb, map, c, w);
            let mut acc = bits[0];
            for &b in &bits[1..] {
                acc = e.push(vb, MirType::I1, Op::Bin(fold, acc, b));
            }
            return acc;
        }
        Some(Wide::Vector(parts)) => parts.clone(),
    };
    let vt = e_ty(e, parts[0]);
    let MirType::Vec(lane, n) = vt.clone() else {
        unreachable!("a widened compare is a lane-mask vector")
    };
    let lane = *lane;
    let mut acc = parts[0];
    for &q in &parts[1..] {
        acc = e.push(vb, vt.clone(), Op::Bin(fold, acc, q));
    }
    let mut width = n;
    while width > 1 {
        let half = width / 2;
        // Lane `k` takes lane `half + k mod half`: the upper half folded onto the lower.
        let mask: Vec<u8> = (0..n).map(|k| (half + k % half) as u8).collect();
        let upper = e.push(
            vb,
            vt.clone(),
            Op::Shuffle {
                a: acc,
                b: acc,
                mask,
            },
        );
        acc = e.push(vb, vt.clone(), Op::Bin(fold, acc, upper));
        width = half;
    }
    let first = e.push(vb, lane.clone(), Op::ExtractLane(acc, 0));
    let zero = e.int(vb, &lane, 0);
    e.push(vb, MirType::I1, Op::Cmp(CmpOp::Ne, first, zero))
}

/// Fold `W` lane values into the accumulator **one at a time, in index order**.
///
/// This is the whole float-safety argument of the pass. `acc` starts at the value the scalar loop
/// would have entering iteration `i`, and lane `k` is combined into it exactly where scalar
/// iteration `i + k` would have combined its own value — same operator, same operand order, same
/// intermediate roundings. Nothing is reassociated, so an `f32` sum produces bit-identical output
/// to the scalar loop and the interpreter oracle agrees with the native backend. With the value split
/// into parts, "index order" is part 0's lanes, then part 1's: exactly iterations `i, i+1, ..`.
///
/// An `fma` accumulate re-fuses its two factors per lane rather than multiplying and adding
/// separately, because `fma(x, y, acc)` rounds once and `acc + x*y` rounds twice.
///
/// A min/max (`m = max(m, x[i])`, `fmin`, ...) is the same argument with a compare and a select in
/// place of the operator ([`emit_combine`]): a float max is not associative once NaN and `±0` ties
/// are in play (`loop_info::RedKind::reassociable`), so the body's loads and arithmetic go `W` wide
/// and the running maximum is updated lane by lane, in index order, exactly as the scalar loop did.
///
/// Lanes come out through [`Op::ExtractLane`], which is one shuffle (and free for lane 0). The
/// first version of this pass spilled the vector to a stack slot and loaded the elements back,
/// because MIR had no lane read; that cost a store-forwarding stall per group and left the
/// reduction kernel 1.8x behind `gcc -O3`, which is what motivated adding the operation.
fn emit_reduction_fold(
    e: &mut Emit<'_>,
    vb: BlockId,
    map: &FxHashMap<u32, Wide>,
    r: &RedPlan,
    init: ValueId,
    w: u32,
) -> ValueId {
    // A guarded accumulate: lane `k`'s condition as an `i1`, and which arm of the merge the update
    // arrives on. Lane `k` then becomes `c_k ? acc ⊕ x_k : acc` — the scalar merge itself, so no
    // identity element is involved and nothing is rounded that the scalar loop did not round.
    let keep = r
        .guard
        .map(|(cond, on_true)| (mask_lane_bools(e, vb, map, cond, w), on_true));
    let mut acc = init;
    match (r.addend, r.fma_factors) {
        (Some((a, left_is_acc)), None) => {
            // A guarded addend may be uniform (`count + 1`); `lane_values` repeats it per lane.
            let lanes = if r.guard.is_some() {
                lane_values(e, vb, map, a, w)
            } else {
                extract_lanes(e, vb, map, a, &r.ty)
            };
            for (k, lane) in lanes.into_iter().enumerate() {
                let updated = emit_combine(e, vb, r, left_is_acc, acc, lane, &r.ty);
                acc = keep_lane(e, vb, keep.as_ref(), k, updated, acc, &r.ty);
            }
        }
        (None, Some((x, y))) => {
            let xs = extract_lanes(e, vb, map, x, &r.ty);
            let ys = extract_lanes(e, vb, map, y, &r.ty);
            for (k, (xk, yk)) in xs.into_iter().zip(ys).enumerate() {
                let updated = e.push(vb, r.ty.clone(), Op::Fma(xk, yk, acc));
                acc = keep_lane(e, vb, keep.as_ref(), k, updated, acc, &r.ty);
            }
        }
        _ => unreachable!("plan_body rejected any other reduction shape"),
    }
    acc
}

/// Lane `k` of a guarded fold: `updated` where the lane's condition chose the update, `old` where
/// it chose the untouched accumulator. Unguarded, simply `updated`.
fn keep_lane(
    e: &mut Emit<'_>,
    vb: BlockId,
    keep: Option<&(Vec<ValueId>, bool)>,
    k: usize,
    updated: ValueId,
    old: ValueId,
    ty: &MirType,
) -> ValueId {
    match keep {
        None => updated,
        Some((bits, on_true)) => {
            let (a, b) = if *on_true {
                (updated, old)
            } else {
                (old, updated)
            };
            e.push(vb, ty.clone(), Op::Select(bits[k], a, b))
        }
    }
}

/// Each lane of the lane mask `c` as an `i1`, lowest iteration first. A mask lane is all-ones
/// natively and `1` in the interpreter, and zero in both when false, so `lane != 0` reads the same
/// truth on every backend.
fn mask_lane_bools(
    e: &mut Emit<'_>,
    vb: BlockId,
    map: &FxHashMap<u32, Wide>,
    c: ValueId,
    w: u32,
) -> Vec<ValueId> {
    lane_values(e, vb, map, c, w)
        .into_iter()
        .map(|m| {
            let ty = e_ty(e, m);
            let zero = e.int(vb, &ty, 0);
            e.push(vb, MirType::I1, Op::Cmp(CmpOp::Ne, m, zero))
        })
        .collect()
}

/// Every lane value of a widened value, lowest iteration first (part 0's lanes, then part 1's).
fn extract_lanes(
    e: &mut Emit<'_>,
    vb: BlockId,
    map: &FxHashMap<u32, Wide>,
    v: ValueId,
    lane: &MirType,
) -> Vec<ValueId> {
    if let Some(Wide::Lanes(xs)) = map.get(&v.0) {
        return xs.clone();
    }
    let parts = resolve_vector(map, v);
    let per_part = lanes_of(lane).expect("a widened lane type has a vector width");
    let mut out = Vec::with_capacity(parts.len() * per_part as usize);
    for part in parts {
        for k in 0..per_part {
            out.push(e.push(vb, lane.clone(), Op::ExtractLane(part, k)));
        }
    }
    out
}

/// The identity element of a reassociable reduction over `ty`, as a raw integer constant: the value
/// every lane of a vector accumulator starts at, so that folding the lanes afterwards reproduces the
/// scalar answer exactly. `And` is all-ones, which sign-extends from `-1` at every integer width;
/// so is an unsigned min's identity, the largest unsigned value. A signed max starts at the type's
/// minimum and a signed min at its maximum, so neither can ever win against a real element.
fn red_identity(k: RedKind, ty: &MirType) -> Option<i64> {
    let bits = lane_bytes(ty)? * 8;
    Some(match k {
        RedKind::Add | RedKind::Or | RedKind::Xor | RedKind::UMax => 0,
        RedKind::Mul => 1,
        RedKind::And | RedKind::UMin => -1,
        RedKind::SMax if ty.is_int() => i64::MIN >> (64 - bits),
        RedKind::SMin if ty.is_int() => i64::MAX >> (64 - bits),
        _ => return None,
    })
}

/// Can a reassociable reduction keep lane-parallel partials — does its combine exist as a vector
/// instruction at this lane type?
///
/// A binary combine asks [`bin_widenable`]. An integer min/max is a vector compare and a lane
/// select, which every integer lane width has — but only up to 32-bit lanes here: a 64-bit
/// unsigned compare has no SSE instruction at all (the backend synthesizes it), and the gain of two
/// lanes does not justify the exposure. A 64-bit min/max keeps the serial fold, which is exact.
fn vector_combinable(k: RedKind, ty: &MirType) -> bool {
    match k {
        RedKind::SMax | RedKind::SMin | RedKind::UMax | RedKind::UMin => {
            ty.is_int() && lane_bytes(ty).is_some_and(|b| b <= 4)
        }
        RedKind::FMax | RedKind::FMin => false,
        other => bin_widenable(red_binop(other), ty),
    }
}

/// One accumulate `acc ⊕ x`, spelled exactly as the scalar loop spelled it.
///
/// A binary combine is `op(acc, x)`, or `op(x, acc)` when the accumulator was the right operand
/// (`left_is_acc`). A min/max re-emits the loop's own compare and select with the accumulator in
/// the positions [`loop_info::SelectCombine`] recorded — on a float tie or a NaN the select returns
/// its second arm, so swapping the operands would change which `±0` or NaN payload survives. `ty` is
/// the type of `acc` and `x`: the scalar lane type, or a part's vector type, whose compare then
/// yields a lane mask at the same width.
fn emit_combine(
    e: &mut Emit<'_>,
    b: BlockId,
    r: &RedPlan,
    left_is_acc: bool,
    acc: ValueId,
    x: ValueId,
    ty: &MirType,
) -> ValueId {
    match r.select {
        Some(s) => {
            let (l, rhs) = if s.acc_left_in_cmp {
                (acc, x)
            } else {
                (x, acc)
            };
            let mask_ty = match ty {
                MirType::Vec(lane, n) => vec_of(&mask_lane(lane), *n),
                _ => MirType::I1,
            };
            let c = e.push(b, mask_ty, Op::Cmp(s.pred, l, rhs));
            let (t, f) = if s.acc_is_then { (acc, x) } else { (x, acc) };
            e.push(b, ty.clone(), Op::Select(c, t, f))
        }
        None => {
            let (l, rhs) = if left_is_acc { (acc, x) } else { (x, acc) };
            e.push(b, ty.clone(), Op::Bin(red_binop(r.kind), l, rhs))
        }
    }
}

/// The strict form of an ordering predicate (`>=` to `>`, `<=` to `<`); a strict one is itself.
/// For floats `a >= b` and `a > b || a == b` agree on every input, NaN included (both false), so
/// "strictly better or equal" is exactly the loop's own test.
fn strict_cmp(p: CmpOp) -> CmpOp {
    match p {
        CmpOp::Sge => CmpOp::Sgt,
        CmpOp::Sle => CmpOp::Slt,
        CmpOp::Uge => CmpOp::Ugt,
        CmpOp::Ule => CmpOp::Ult,
        CmpOp::Foge => CmpOp::Fogt,
        CmpOp::Fole => CmpOp::Folt,
        other => other,
    }
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
        // Never reassociable, so it only ever reaches the serial fold, which re-emits the combine
        // with the accumulator on the left exactly as the scalar loop had it.
        RedKind::Sub => BinOp::Sub,
        RedKind::FSub => BinOp::FSub,
        // A min/max is a compare and a select, never a binary operator: every caller asks
        // `RedPlan::select` first (`emit_combine`, `vector_combinable`), so it never reaches here.
        other => unreachable!("{} is not a binary-combine reduction", other.name()),
    }
}

/// The address of part `k` of a widened access at `ptr`: `k` full vectors of `elem` further on.
fn part_ptr(e: &mut Emit<'_>, vb: BlockId, ptr: ValueId, elem: &MirType, k: usize) -> ValueId {
    let per_part = lanes_of(elem).expect("a widened access has a vector width");
    elem_ptr(e, vb, ptr, elem, k as i64 * per_part as i64)
}

/// Part `k` of a contiguous access whose lane 0 is at `ptr`, loaded with its lanes in iteration
/// order. Ascending, that is the `L` elements from `k·L` on; in a countdown it is the `L` elements
/// ENDING `k·L` below `ptr`, loaded as one vector and reversed ([`reverse_lanes`]), so lane `j`
/// holds element `ptr - (k·L + j)` — iteration `k·L + j`'s.
fn load_part(
    e: &mut Emit<'_>,
    vb: BlockId,
    ptr: ValueId,
    elem: &MirType,
    k: usize,
    dir: i64,
) -> ValueId {
    let vty = part_ty(elem);
    if dir > 0 {
        let at = part_ptr(e, vb, ptr, elem, k);
        return e.push(vb, vty.clone(), Op::Load(at, vty));
    }
    let per_part = i64::from(lanes_of(elem).expect("a widened access has a vector width"));
    let at = elem_ptr(e, vb, ptr, elem, -(k as i64 * per_part + per_part - 1));
    let v = e.push(vb, vty.clone(), Op::Load(at, vty));
    reverse_lanes(e, vb, v)
}

/// The store mirror of [`load_part`]: `value`'s lanes are in iteration order, so a countdown
/// reverses them and stores the part where its last lane's element begins.
fn store_part(
    e: &mut Emit<'_>,
    vb: BlockId,
    ptr: ValueId,
    elem: &MirType,
    k: usize,
    value: ValueId,
    dir: i64,
) {
    let (at, value) = if dir > 0 {
        (part_ptr(e, vb, ptr, elem, k), value)
    } else {
        let per_part = i64::from(lanes_of(elem).expect("a widened access has a vector width"));
        let at = elem_ptr(e, vb, ptr, elem, -(k as i64 * per_part + per_part - 1));
        (at, reverse_lanes(e, vb, value))
    };
    e.push_void(vb, Op::Store { ptr: at, value });
}

/// `v` with its lanes in the opposite order: one single-source shuffle. Lane `j` of the result is
/// lane `n - 1 - j` of `v`, bit for bit, so nothing is computed — a float lane's NaN payload and
/// sign survive.
fn reverse_lanes(e: &mut Emit<'_>, vb: BlockId, v: ValueId) -> ValueId {
    let vty = e_ty(e, v);
    let MirType::Vec(_, n) = vty else {
        unreachable!("a widened part is a vector")
    };
    let mask: Vec<u8> = (0..n).rev().map(|j| j as u8).collect();
    e.push(vb, vty, Op::Shuffle { a: v, b: v, mask })
}

/// `ptr` moved by `delta` elements of `elem` (`ptr` itself for zero).
fn elem_ptr(e: &mut Emit<'_>, vb: BlockId, ptr: ValueId, elem: &MirType, delta: i64) -> ValueId {
    if delta == 0 {
        return ptr;
    }
    let index = e.int(vb, &MirType::I64, delta);
    e.push(
        vb,
        MirType::Ptr,
        Op::Gep {
            ptr,
            index,
            elem: elem.clone(),
        },
    )
}

/// Load an interleave group's whole window and split every member out of it.
///
/// `ptr` is the address the first member (residue `offset`) reads in iteration `i`, so the window
/// starts `offset` elements earlier. Part `p` of the group covers iterations `p·L .. p·L + L` and
/// so the `k·L` elements from `p·k·L` on: `k` contiguous vector loads. Lane `l` of member `r` is
/// element `k·l + r` of that span — vector `(k·l + r) / L`, lane `(k·l + r) % L` — gathered with
/// lane shuffles ([`gather_lanes`]). Every element loaded is one some member reads in some iteration
/// of the group, so nothing outside what the scalar loop touches is read.
fn emit_group_load(
    e: &mut Emit<'_>,
    vb: BlockId,
    ptr: ValueId,
    offset: u32,
    gi: usize,
    p: &VecPlan,
    vals: &mut FxHashMap<(usize, u32), Vec<ValueId>>,
) {
    let g = &p.groups[gi];
    let k = g.k as usize;
    let l = lanes_of(&g.elem).expect("a group moves a vector lane type") as usize;
    let vty = part_ty(&g.elem);
    let window = elem_ptr(e, vb, ptr, &g.elem, -i64::from(offset));
    // `part` selects the same split of every source value in the group, not one vector's elements.
    #[allow(clippy::needless_range_loop)]
    for part in 0..parts_of(&g.elem, p.w) {
        let mut vecs = Vec::with_capacity(k);
        for m in 0..k {
            let at = elem_ptr(e, vb, window, &g.elem, ((part * k + m) * l) as i64);
            vecs.push(e.push(vb, vty.clone(), Op::Load(at, vty.clone())));
        }
        for r in 0..k {
            let sources: Vec<(ValueId, u32)> = (0..l)
                .map(|lane| {
                    let el = k * lane + r;
                    (vecs[el / l], (el % l) as u32)
                })
                .collect();
            let v = gather_lanes(e, vb, &sources, &vty);
            vals.entry((gi, r as u32)).or_default().push(v);
        }
    }
}

/// Interleave a store group's `k` member values and store the window contiguously.
///
/// The mirror of [`emit_group_load`]: output vector `m` of part `p` holds elements
/// `p·k·L + m·L .. + L`, and element `m·L + t` of the part's span is lane `(m·L + t) / k` of member
/// `(m·L + t) % k`. Exactly the elements the scalar loop writes are written, each once.
// `part` selects the same split of every source value in the group, not one vector's elements.
#[allow(clippy::needless_range_loop)]
fn emit_group_store(
    e: &mut Emit<'_>,
    vb: BlockId,
    ptr: ValueId,
    offset: u32,
    gi: usize,
    p: &VecPlan,
    vals: &FxHashMap<(usize, u32), Vec<ValueId>>,
) {
    let g = &p.groups[gi];
    let k = g.k as usize;
    let l = lanes_of(&g.elem).expect("a group moves a vector lane type") as usize;
    let vty = part_ty(&g.elem);
    let window = elem_ptr(e, vb, ptr, &g.elem, -i64::from(offset));
    for part in 0..parts_of(&g.elem, p.w) {
        for m in 0..k {
            let sources: Vec<(ValueId, u32)> = (0..l)
                .map(|t| {
                    let el = m * l + t;
                    (vals[&(gi, (el % k) as u32)][part], (el / k) as u32)
                })
                .collect();
            let v = gather_lanes(e, vb, &sources, &vty);
            let at = elem_ptr(e, vb, window, &g.elem, ((part * k + m) * l) as i64);
            e.push_void(vb, Op::Store { ptr: at, value: v });
        }
    }
}

/// A vector whose lane `j` is lane `sources[j].1` of vector `sources[j].0`, built from two-source
/// `Op::Shuffle`s: the first two distinct sources fill their lanes in one shuffle, and each further
/// source is merged into the running vector keeping every lane already filled. One source in
/// identity order is returned as it is.
fn gather_lanes(
    e: &mut Emit<'_>,
    vb: BlockId,
    sources: &[(ValueId, u32)],
    vty: &MirType,
) -> ValueId {
    let n = sources.len() as u32;
    let mut order: Vec<ValueId> = Vec::new();
    for (v, _) in sources {
        if !order.contains(v) {
            order.push(*v);
        }
    }
    let first = order[0];
    if order.len() == 1 {
        if sources
            .iter()
            .enumerate()
            .all(|(j, (_, lane))| *lane == j as u32)
        {
            return first;
        }
        let mask = sources.iter().map(|(_, lane)| *lane as u8).collect();
        return e.push(
            vb,
            vty.clone(),
            Op::Shuffle {
                a: first,
                b: first,
                mask,
            },
        );
    }
    let second = order[1];
    let mask = sources
        .iter()
        .map(|(v, lane)| {
            if *v == first {
                *lane as u8
            } else if *v == second {
                (n + lane) as u8
            } else {
                0
            }
        })
        .collect();
    let mut acc = e.push(
        vb,
        vty.clone(),
        Op::Shuffle {
            a: first,
            b: second,
            mask,
        },
    );
    for &next in &order[2..] {
        let mask = sources
            .iter()
            .enumerate()
            .map(|(j, (v, lane))| {
                if *v == next {
                    (n + lane) as u8
                } else {
                    j as u8
                }
            })
            .collect();
        acc = e.push(
            vb,
            vty.clone(),
            Op::Shuffle {
                a: acc,
                b: next,
                mask,
            },
        );
    }
    acc
}

/// Rebuild one lane-wise operation over parts, every operand widened to the result's parts.
///
/// `lane` is the result's lane type. For a compare that is the mask, while the operands keep their
/// own type (the scalar operand type, which a widened value always has as its lane type).
fn emit_widened(
    e: &mut Emit<'_>,
    vb: BlockId,
    map: &FxHashMap<u32, Wide>,
    op: &Op,
    w: u32,
    lane: &MirType,
) -> Vec<ValueId> {
    let pty = part_ty(lane);
    match op {
        Op::Bin(b, x, y) => {
            let xs = to_vector(e, vb, map, *x, lane, w);
            if matches!(b, BinOp::Shl | BinOp::LShr | BinOp::AShr)
                && matches!(map.get(&y.0), Some(Wide::Uniform(_)) | None)
            {
                // The amount is the same in every lane: keep it scalar (`Op::Bin`'s
                // vector-by-scalar shape), masked to the lane width by every backend exactly as
                // the scalar shift masks it.
                let amount = resolve_uniform(map, *y);
                return xs
                    .into_iter()
                    .map(|x| e.push(vb, pty.clone(), Op::Bin(*b, x, amount)))
                    .collect();
            }
            let ys = to_vector(e, vb, map, *y, lane, w);
            emit_bin(e, vb, *b, &xs, &ys, lane)
        }
        Op::Cmp(c, x, y) => {
            let operand = e_ty(e, *x);
            let xs = to_vector(e, vb, map, *x, &operand, w);
            let ys = to_vector(e, vb, map, *y, &operand, w);
            xs.into_iter()
                .zip(ys)
                .map(|(a, b)| e.push(vb, pty.clone(), Op::Cmp(*c, a, b)))
                .collect()
        }
        Op::Select(c, a, b) => {
            let mask = mask_for(e, vb, map, *c, lane);
            let a = to_vector(e, vb, map, *a, lane, w);
            let b = to_vector(e, vb, map, *b, lane, w);
            (0..a.len())
                .map(|k| e.push(vb, pty.clone(), Op::Select(mask[k], a[k], b[k])))
                .collect()
        }
        Op::Neg(x) | Op::Not(x) | Op::Sqrt(x) | Op::Round(_, x) => {
            let xs = to_vector(e, vb, map, *x, lane, w);
            xs.into_iter()
                .map(|x| {
                    let one = match op {
                        Op::Neg(_) => Op::Neg(x),
                        Op::Not(_) => Op::Not(x),
                        Op::Sqrt(_) => Op::Sqrt(x),
                        Op::Round(m, _) => Op::Round(*m, x),
                        _ => unreachable!("matched above"),
                    };
                    e.push(vb, pty.clone(), one)
                })
                .collect()
        }
        Op::Fma(a, b, c) => {
            let a = to_vector(e, vb, map, *a, lane, w);
            let b = to_vector(e, vb, map, *b, lane, w);
            let c = to_vector(e, vb, map, *c, lane, w);
            (0..a.len())
                .map(|k| e.push(vb, pty.clone(), Op::Fma(a[k], b[k], c[k])))
                .collect()
        }
        other => unreachable!("plan_body accepted an op it cannot widen: {other:?}"),
    }
}

/// `op` applied part by part. An `i8` multiply — which x86 has no packed instruction for, at any
/// width — is done in `i16`: both operands are sign-extended half by half, multiplied, and the
/// products truncated back. The low eight bits of a product do not depend on how the operands were
/// extended, so every lane is exactly the scalar wrapping `i8` product.
fn emit_bin(
    e: &mut Emit<'_>,
    vb: BlockId,
    op: BinOp,
    xs: &[ValueId],
    ys: &[ValueId],
    lane: &MirType,
) -> Vec<ValueId> {
    let pty = part_ty(lane);
    if op == BinOp::Mul && *lane == MirType::I8 {
        let wide = part_ty(&MirType::I16);
        let mut out = Vec::with_capacity(xs.len());
        for (&x, &y) in xs.iter().zip(ys) {
            let mut halves = [x; 2];
            for (slot, half) in halves.iter_mut().zip([Half::Lo, Half::Hi]) {
                let xw = e.push(
                    vb,
                    wide.clone(),
                    Op::Widen {
                        v: x,
                        half,
                        kind: CastKind::SExt,
                    },
                );
                let yw = e.push(
                    vb,
                    wide.clone(),
                    Op::Widen {
                        v: y,
                        half,
                        kind: CastKind::SExt,
                    },
                );
                *slot = e.push(vb, wide.clone(), Op::Bin(BinOp::Mul, xw, yw));
            }
            out.push(e.push(
                vb,
                pty.clone(),
                Op::Narrow {
                    lo: halves[0],
                    hi: halves[1],
                    kind: NarrowKind::Trunc,
                },
            ));
        }
        return out;
    }
    xs.iter()
        .zip(ys)
        .map(|(&x, &y)| e.push(vb, pty.clone(), Op::Bin(op, x, y)))
        .collect()
}

/// Spell a conversion over parts, by the route [`cast_route`] chose. `from` and `to` are the scalar
/// lane types; `xs` are the source's parts.
fn emit_cast(
    e: &mut Emit<'_>,
    vb: BlockId,
    kind: CastKind,
    xs: Vec<ValueId>,
    from: &MirType,
    to: &MirType,
    route: CastRoute,
) -> Vec<ValueId> {
    match route {
        CastRoute::Same => cast_parts(e, vb, kind, xs, to),
        CastRoute::Identity => xs,
        CastRoute::WidenInt { ext, then_fp } => {
            let to_bytes = lane_bytes(to).expect("a widened lane type");
            let ints = widen_chain(e, vb, xs, from, &int_of_bytes(to_bytes), ext);
            if then_fp {
                cast_parts(e, vb, CastKind::SiToFp, ints, to)
            } else {
                ints
            }
        }
        CastRoute::TruncInt => narrow_chain(e, vb, xs, from, to, |_| NarrowKind::Trunc),
        CastRoute::FloatToNarrowInt { unsigned } => {
            let ints = cast_parts(e, vb, CastKind::FpToSi, xs, &MirType::I32);
            narrow_chain(e, vb, ints, &MirType::I32, to, |last| {
                if last && unsigned {
                    NarrowKind::SatUnsigned
                } else {
                    NarrowKind::SatSigned
                }
            })
        }
        CastRoute::FpExt => widen_chain(e, vb, xs, from, to, CastKind::FpExt),
        CastRoute::FpTrunc => narrow_chain(e, vb, xs, from, to, |_| NarrowKind::FpTrunc),
        CastRoute::HalfToF64 => {
            let rounded = cast_parts(e, vb, CastKind::FpExt, xs, &MirType::F32);
            widen_chain(e, vb, rounded, &MirType::F32, to, CastKind::FpExt)
        }
        CastRoute::F64ToHalf => {
            let demoted = narrow_chain(e, vb, xs, from, &MirType::F32, |_| NarrowKind::FpTrunc);
            cast_parts(e, vb, CastKind::FpTrunc, demoted, to)
        }
    }
}

/// One lane-wise `Cast` of kind `k` per part, to `to`'s full-width vector.
fn cast_parts(
    e: &mut Emit<'_>,
    vb: BlockId,
    k: CastKind,
    xs: Vec<ValueId>,
    to: &MirType,
) -> Vec<ValueId> {
    xs.into_iter()
        .map(|x| e.push(vb, part_ty(to), Op::Cast(k, x, part_ty(to))))
        .collect()
}

/// Double the lane width of every part, `kind` (`SExt`/`ZExt`/`FpExt`) at a time, until it is
/// `to`'s: each part becomes its low and its high half, in that order, so the parts stay in
/// iteration order.
fn widen_chain(
    e: &mut Emit<'_>,
    vb: BlockId,
    mut parts: Vec<ValueId>,
    from: &MirType,
    to: &MirType,
    kind: CastKind,
) -> Vec<ValueId> {
    let target = lane_bytes(to).expect("a widened lane type");
    let mut cur = from.clone();
    while lane_bytes(&cur).expect("a widened lane type") < target {
        let next = if kind == CastKind::FpExt {
            MirType::F64
        } else {
            int_of_bytes(lane_bytes(&cur).expect("a widened lane type") * 2)
        };
        let nty = part_ty(&next);
        let mut out = Vec::with_capacity(parts.len() * 2);
        for x in parts {
            for half in [Half::Lo, Half::Hi] {
                out.push(e.push(vb, nty.clone(), Op::Widen { v: x, half, kind }));
            }
        }
        parts = out;
        cur = next;
    }
    parts
}

/// Halve the lane width of the parts, two at a time, until it is `to`'s. `step_kind` picks each
/// step's narrowing; it is told whether the step is the last one (the one that lands on `to`).
fn narrow_chain(
    e: &mut Emit<'_>,
    vb: BlockId,
    mut parts: Vec<ValueId>,
    from: &MirType,
    to: &MirType,
    step_kind: impl Fn(bool) -> NarrowKind,
) -> Vec<ValueId> {
    let target = lane_bytes(to).expect("a widened lane type");
    let mut cur = from.clone();
    while lane_bytes(&cur).expect("a widened lane type") > target {
        let half = lane_bytes(&cur).expect("a widened lane type") / 2;
        let next = if cur == MirType::F64 {
            MirType::F32
        } else {
            int_of_bytes(half)
        };
        let kind = step_kind(half == target);
        let nty = part_ty(&next);
        parts = parts
            .chunks(2)
            .map(|pair| {
                e.push(
                    vb,
                    nty.clone(),
                    Op::Narrow {
                        lo: pair[0],
                        hi: pair[1],
                        kind,
                    },
                )
            })
            .collect();
        cur = next;
    }
    parts
}

/// The parts of the lane mask `c`, re-expressed at the lane width of `value_lane` so they can blend
/// values of that type. A compare's mask has its *operands'* lane width — `x[i] > 0.0` over `f32`
/// is four `<4 x i32>` parts — while the selected values may be narrower or wider. Narrowing
/// truncates and widening sign-extends: both turn an all-ones lane into an all-ones lane and zero
/// into zero (and the interpreter's `1` lanes into `1`), so the selection is unchanged.
fn mask_for(
    e: &mut Emit<'_>,
    vb: BlockId,
    map: &FxHashMap<u32, Wide>,
    c: ValueId,
    value_lane: &MirType,
) -> Vec<ValueId> {
    let parts = resolve_vector(map, c);
    let from = match e_ty(e, parts[0]) {
        MirType::Vec(l, _) => *l,
        other => other,
    };
    let to = mask_lane(value_lane);
    if lane_bytes(&from) > lane_bytes(&to) {
        narrow_chain(e, vb, parts, &from, &to, |_| NarrowKind::Trunc)
    } else {
        widen_chain(e, vb, parts, &from, &to, CastKind::SExt)
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
        Some(Wide::Vector(_) | Wide::Lanes(_)) => {
            unreachable!("plan_body proved this operand is uniform")
        }
    }
}

fn resolve_vector(map: &FxHashMap<u32, Wide>, v: ValueId) -> Vec<ValueId> {
    match map.get(&v.0) {
        Some(Wide::Vector(x)) => x.clone(),
        _ => unreachable!("plan_body proved this operand is a vector"),
    }
}

/// A value as parts: itself if already widened, otherwise one broadcast reused for every part.
fn to_vector(
    e: &mut Emit<'_>,
    vb: BlockId,
    map: &FxHashMap<u32, Wide>,
    v: ValueId,
    lane: &MirType,
    w: u32,
) -> Vec<ValueId> {
    let scalar = match map.get(&v.0) {
        Some(Wide::Vector(x)) => return x.clone(),
        Some(Wide::Lanes(xs)) => {
            let xs = xs.clone();
            return pack_lanes(e, vb, &xs, lane);
        }
        Some(Wide::Uniform(x)) => *x,
        None => v,
    };
    let splat = e.push(vb, part_ty(lane), Op::Splat(scalar));
    vec![splat; parts_of(lane, w)]
}

/// `W` scalar lane values packed into full-width vectors of `lane`: each part starts as a splat of
/// its first lane and takes every further lane through `Op::InsertLane`.
fn pack_lanes(e: &mut Emit<'_>, vb: BlockId, lanes: &[ValueId], lane: &MirType) -> Vec<ValueId> {
    let per_part = lanes_of(lane).expect("a packed lane type has a vector width") as usize;
    let vty = part_ty(lane);
    lanes
        .chunks(per_part)
        .map(|chunk| {
            let mut v = e.push(vb, vty.clone(), Op::Splat(chunk[0]));
            for (k, &x) in chunk.iter().enumerate().skip(1) {
                v = e.push(vb, vty.clone(), Op::InsertLane(v, x, k as u32));
            }
            v
        })
        .collect()
}

/// A value's `w` per-lane scalars, lowest iteration first: a replicated value's own lanes, a
/// vector's lanes extracted part by part, or a uniform value repeated.
fn lane_values(
    e: &mut Emit<'_>,
    vb: BlockId,
    map: &FxHashMap<u32, Wide>,
    v: ValueId,
    w: u32,
) -> Vec<ValueId> {
    match map.get(&v.0) {
        Some(Wide::Lanes(xs)) => xs.clone(),
        Some(Wide::Vector(parts)) => {
            let parts = parts.clone();
            let mut out = Vec::with_capacity(w as usize);
            for part in parts {
                let (lane, n) = match e_ty(e, part) {
                    MirType::Vec(l, n) => (*l, n),
                    other => (other, 1),
                };
                for k in 0..n {
                    out.push(e.push(vb, lane.clone(), Op::ExtractLane(part, k)));
                }
            }
            out
        }
        Some(Wide::Uniform(x)) => vec![*x; w as usize],
        None => vec![v; w as usize],
    }
}

/// A value's `w` per-lane scalars **of its own scalar type**, for a step that runs once per lane
/// on them — a replicated op, a per-lane call, a lane-serial step: [`lane_values`], except that a
/// widened `bool` — a lane mask, whose lanes are integers at its compare's operand width — is read
/// as each lane's truth ([`mask_lane_bools`]), the `i1` the scalar step takes. The raw mask lane
/// is an `i32` (or wider) where the step, a call's `bool` parameter or a `select`'s condition,
/// declares `i1`: the MIR verifier refuses it.
fn scalar_lanes(
    e: &mut Emit<'_>,
    vb: BlockId,
    map: &FxHashMap<u32, Wide>,
    v: ValueId,
    w: u32,
) -> Vec<ValueId> {
    if *e.f.value_type(v) == MirType::I1 && matches!(map.get(&v.0), Some(Wide::Vector(_))) {
        mask_lane_bools(e, vb, map, v, w)
    } else {
        lane_values(e, vb, map, v, w)
    }
}

/// The induction variable's value in each lane of the group: `i + k`, wrapping exactly as the
/// scalar loop's `i + 1` chain does.
fn iv_lane_values(
    e: &mut Emit<'_>,
    vb: BlockId,
    iv: ValueId,
    ty: &MirType,
    w: u32,
    dir: i64,
) -> Vec<ValueId> {
    let toward = if dir > 0 { BinOp::Add } else { BinOp::Sub };
    (0..w)
        .map(|k| {
            if k == 0 {
                iv
            } else {
                let off = e.int(vb, ty, i64::from(k));
                e.push(vb, ty.clone(), Op::Bin(toward, iv, off))
            }
        })
        .collect()
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
        Terminator::Switch { cases, default, .. } => {
            for c in cases.iter_mut() {
                if c.target == from {
                    c.target = to;
                }
            }
            if *default == from {
                *default = to;
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

    /// Every loop under test over `[]T` slices is spelled as a counting `while`. That is the shape
    /// `mir_build`'s own front-end vectorizer leaves to this pass (it normalizes a counting `while`
    /// into its `for` path for arrays, never for slices), so these cases keep exercising THIS pass
    /// however much the front end's coverage grows — a `for` over provably disjoint slices is now
    /// widened there, before this pass ever sees it.
    fn lower(src: &str) -> (Program, Interner) {
        let mut interner = Interner::new();
        let (module, pd) = wukong_parser::parse_module(src, SourceId(0), &mut interner);
        assert!(pd.iter().all(|d| !d.is_error()), "parse: {pd:?}");
        let (sema, sd) = wukong_sema::check(&module, &interner);
        assert!(sd.iter().all(|d| !d.is_error()), "sema: {sd:?}");
        let (program, _) = wukong_mir_build::lower_program(&module, &sema, &mut interner);
        (program, interner)
    }

    /// `src` lowered and optimized at `level`, then verified as a whole program — the check the
    /// driver runs before every backend (`verify_or_ice`). Per function, the verifier cannot hold
    /// a call's arguments against its callee's signature, and a lane value of the wrong type
    /// handed to a per-lane call is exactly what it would miss.
    fn optimized(src: &str, level: u8) -> (Program, Interner) {
        let (mut program, interner) = lower(src);
        crate::optimize(&mut program, level);
        let errs = wukong_mir::verify::verify_program(&program);
        assert!(errs.is_empty(), "MIR invalid after -O{level}: {errs:?}");
        (program, interner)
    }

    /// Every vector-typed value in the named function, as `(lane, lanes)` pairs.
    fn vector_values(src: &str, func: &str) -> Vec<(MirType, u32)> {
        let (program, mut interner) = optimized(src, 2);
        let sym = interner.intern(func);
        let f = program
            .function(sym)
            .unwrap_or_else(|| panic!("no fn {func}"));
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
    ///
    /// It also **refuses a vacuous differential**. A caller writes a `main` that exercises some
    /// shape, but the `-O2` program it actually gets is not the one it wrote: `-O2` inlines first,
    /// and inlining can dissolve the very property the case is about. Passing one buffer twice
    /// (`k(w, w, ..)`) makes the two parameters the *same SSA value* after the splice, so
    /// `loop_info::dependence` sees one object at two different indices, reports `Carried`, and
    /// declines the loop outright — no vector code, no guard, and a comparison of two identical
    /// scalar programs that passes no matter what this pass does. That is exactly what happened to
    /// `two_symbolic_offsets_on_one_base_are_guarded_separately`. Asserting that *something* is
    /// vector-typed at `-O2` is the cheap check that keeps a differential honest; a case that needs
    /// more (which loop widened, how wide) asserts that itself.
    fn scalar_and_vector_agree(src: &str) {
        let mut outs = Vec::new();
        for level in [0u8, 2u8] {
            let (program, mut interner) = optimized(src, level);
            if level == 2 {
                let widened: usize = program
                    .funcs
                    .iter()
                    .map(|f| f.value_types.iter().filter(|t| t.is_vector()).count())
                    .sum();
                assert!(
                    widened > 0,
                    "vacuous differential: nothing is vector-typed at -O2, so this compares two \
                     scalar programs and proves nothing about the vectorizer"
                );
            }
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

    /// A `[]T` slice re-reads its data pointer out of the fat pointer at every indexed access, so
    /// a three-slice saxpy body holds three pointer loads on top of its two data loads and one
    /// store — half its memory operations are the same three addresses. The widened body inherited
    /// all three and did them once per *group*. They are loop-invariant by the same
    /// `IndependentIfBasesStable` verdict the pass already requires, so they belong in the guard.
    ///
    /// Asserted structurally rather than by counting instructions: no block that holds a
    /// vector-typed value may contain a `Ptr`-typed load. That fails if the hoist ever silently
    /// stops firing, and does not care how many streams the kernel has.
    #[test]
    fn a_slice_base_is_read_once_not_once_per_group() {
        let (program, mut interner) = optimized(SAXPY, 2);
        let sym = interner.intern("k");
        let f = program.function(sym).expect("no fn k");
        let mut checked = 0;
        for b in &f.blocks {
            // The vector *body* is the block that transfers vectors to and from memory. The guard
            // is deliberately not it: the guard is where the hoisted loads went, and it also holds
            // the splat of the loop-invariant scalar, so "any vector-typed value" would match it.
            let is_vector_body = b.insts.iter().any(|i| match &i.op {
                Op::Load(_, t) => t.is_vector(),
                Op::Store { value, .. } => f.value_type(*value).is_vector(),
                _ => false,
            });
            if !is_vector_body {
                continue;
            }
            checked += 1;
            for inst in &b.insts {
                assert!(
                    !matches!(&inst.op, Op::Load(_, MirType::Ptr)),
                    "bb{} is a vector body and still reloads a slice base",
                    b.id.0
                );
            }
        }
        assert!(checked > 0, "saxpy was not widened at all");
    }

    /// A temporary declared inside a **nested** body and never read after the loop is still
    /// loop-carried: one alloca per local in the entry block, promoted by mem2reg into a block
    /// parameter at the inner header *and* one at the outer header, because the outer header is in
    /// the inner one's iterated dominance frontier. Those two feed each other — the outer latch
    /// passes the inner parameter out, the inner preheader passes the outer parameter back in — so
    /// a use *count* calls both live and `simplify-phis` removed neither. `loop_info` then
    /// classified the inner one `Carried::Recurrence`, and this pass refused the loop for carrying
    /// a dependence. One dead `let` in a nested body was enough.
    ///
    /// The indexes here are deliberately 1-D, so this case turns on the parameter cycle alone and
    /// not on the alias guard's invariant-offset support.
    #[test]
    fn a_dead_temporary_in_a_nested_body_does_not_block_widening() {
        let src = "fn k(x: []f32, mut o: []f32, r: i64, n: i64) { \
                   let mut a: i64 = 0; \
                   while a < r { \
                     let mut j: i64 = 0; \
                     while j < n { let t: f32 = x[j] * 3.0; o[j] = t + 1.0; j = j + 1; } \
                     a = a + 1; } } \
                   fn main() -> i32 { return 0; }";
        assert!(
            is_widened(src, "k"),
            "a nested body with a dead temporary was not widened"
        );
    }

    /// A 2-D row-major inner loop: `o[b + j]` for `j in 0..c` with the row base `b` computed in the
    /// enclosing loop. Its index is `1·j + b`, an affine expression with one loop-INVARIANT term,
    /// and the runtime alias guard could only express `1·iv + constant` — so this shape, which is
    /// what a row softmax, a row reduction and every fused per-row loss are made of, was declined
    /// with "the runtime alias check cannot be expressed" whatever else it did.
    #[test]
    fn a_two_dimensional_row_loop_is_widened() {
        let src = "fn k(x: []f32, mut o: []f32, r: i64, c: i64) { \
                   let mut i: i64 = 0; \
                   while i < r { \
                     let b: i64 = i * c; \
                     let mut j: i64 = 0; \
                     while j < c { o[b + j] = x[b + j] * 2.0 + 1.0; j = j + 1; } \
                     i = i + 1; } } \
                   fn main() -> i32 { return 0; }";
        assert!(is_widened(src, "k"), "a 2-D row loop was not widened");
    }

    /// One buffer read at two *different* invariant row offsets in one body — the LayerNorm γ/β
    /// gather of a transformer block, `g[i] = w[ga + i]; b[i] = w[gb + i];`. Grouping the alias
    /// guard's ranges by base alone forced those two reads into one range with no expressible
    /// offset; they are two ranges, and both are ordinary.
    #[test]
    fn one_base_read_at_two_row_offsets_is_widened() {
        let src = "fn k(w: []f32, mut g: []f32, mut b: []f32, ga: i64, gb: i64, n: i64) { \
                   let mut i: i64 = 0; \
                   while i < n { g[i] = w[ga + i]; b[i] = w[gb + i]; i = i + 1; } } \
                   fn main() -> i32 { return 0; }";
        assert!(is_widened(src, "k"), "a two-offset gather was not widened");
    }

    /// `select(lane mask, uniform, uniform)`. A relu is the smallest spelling of it —
    /// `fmax(x[i], 0.0)` lowers to `select(x[i] > 0, x[i], 0.0)`, whose `else` arm is a literal —
    /// and the planner used to demand that *both* arms already be vectors, which no literal is.
    /// The same check refused every inlined `exp`/`log`, whose range reduction is a chain of
    /// `select(bit, table_a, table_b)` over f32 constants.
    #[test]
    fn a_select_with_a_uniform_arm_is_widened() {
        let relu = "fn k(x: []f32, mut o: []f32, n: i64) { let mut i: i64 = 0; \
                    while i < n { o[i] = fmax(x[i], 0.0); i = i + 1; } } \
                    fn main() -> i32 { return 0; }";
        assert!(is_widened(relu, "k"), "relu was not widened");
        // Both arms uniform, under a mask derived from the data: the if-converted latch parameter.
        let step = "fn k(x: []f32, mut o: []f32, n: i64) { let mut i: i64 = 0; \
                    while i < n { let g: f32 = if x[i] > 0.0 { 1.5 } else { -0.5 }; \
                      o[i] = g; i = i + 1; } } \
                    fn main() -> i32 { return 0; }";
        assert!(
            is_widened(step, "k"),
            "a two-constant merge was not widened"
        );
        scalar_and_vector_agree(
            "fn k(x: []f32, mut o: []f32, n: i64) { let mut i: i64 = 0; \
             while i < n { let g: f32 = if x[i] > 0.0 { 1.5 } else { -0.5 }; \
               o[i] = fmax(g, x[i]); i = i + 1; } } \
             fn main() -> i32 { let mut a: []f32 = alloc_f32(11); let mut b: []f32 = alloc_f32(11); \
               let mut i: i64 = 0; while i < 11 { a[i] = (i as f32) - 5.0; i = i + 1; } \
               k(a, b, 11); i = 0; \
               while i < 11 { print((b[i] * 2.0) as i32); i = i + 1; } \
               free(a); free(b); return 0; }",
        );
    }

    /// `acc = acc - x[i]` is an accumulate with a FIXED operand order, and it is how a loss is
    /// spelled — `lr = lr - alpha*q*(1-p)^2*log p`. It is never reassociable (`((a-x)-y)` is
    /// `a - (x+y)`, so lane-parallel partials would need a different combining operator and a
    /// different identity), so it takes the serial fold, and the fold has to keep the accumulator
    /// on the LEFT. Putting it on the right turns `-sum` into something unrelated, which the
    /// `-O0` vs `-O2` comparison catches.
    #[test]
    fn a_subtract_accumulate_is_a_reduction() {
        let src = "fn k(x: []f32, n: i64) -> f32 { let mut s: f32 = 0.0; let mut i: i64 = 0; \
                   while i < n { s = s - x[i]; i = i + 1; } return s; } \
                   fn main() -> i32 { return 0; }";
        assert!(
            is_widened(src, "k"),
            "a subtract accumulate was not widened"
        );
        scalar_and_vector_agree(
            "fn k(x: []f32, n: i64) -> f32 { let mut s: f32 = 0.0; let mut i: i64 = 0; \
             while i < n { s = s - x[i] * 2.0; i = i + 1; } return s; } \
             fn main() -> i32 { let mut a: []f32 = alloc_f32(11); let mut i: i64 = 0; \
               while i < 11 { a[i] = (i as f32) * 0.5 + 1.0; i = i + 1; } \
               print((k(a, 11) * 2.0) as i32); free(a); return 0; }",
        );
    }

    /// `acc = x[i] - acc` is NOT an accumulate: it negates the whole history every iteration, so
    /// the iterations cannot be grouped at all. It has to stay a `Carried::Recurrence`.
    #[test]
    fn a_reversed_subtract_is_not_a_reduction() {
        let src = "fn k(x: []f32, n: i64) -> f32 { let mut s: f32 = 0.0; let mut i: i64 = 0; \
                   while i < n { s = x[i] - s; i = i + 1; } return s; } \
                   fn main() -> i32 { return 0; }";
        assert!(
            !is_widened(src, "k"),
            "`x[i] - acc` must not widen as a reduction"
        );
    }

    /// An inline transcendental in the body. `exp` expands to a bit-trick range reduction — a
    /// `bitcast` to the integer lane type, integer `and`/`sub`/`mul`, `cmp` and a `select` table
    /// over f32 literals, then a `bitcast` back — and every one of those has to widen for the loop
    /// to widen at all. This is the shape a custom loss is made of, and it is checked here
    /// structurally (an integer-lane vector must appear, which only the bit trick produces).
    #[test]
    fn an_inline_transcendental_body_is_widened() {
        let src = "fn k(x: []f32, mut o: []f32, n: i64) { let mut i: i64 = 0; \
                   while i < n { o[i] = exp(x[i]); i = i + 1; } } \
                   fn main() -> i32 { return 0; }";
        let vs = vector_values(src, "k");
        assert!(!vs.is_empty(), "an exp loop was not widened");
        assert!(
            vs.iter().any(|(l, n)| *l == MirType::I32 && *n == 4),
            "the exp bit trick did not widen: {vs:?}"
        );
    }

    /// `f32 -> i32` widens (Cranelift lowers `fcvt_to_sint_sat.i32x4`); `f64 -> i64` must not,
    /// because the `.i64x2` form needs AVX512DQ and Cranelift has no rule for it. The reverse
    /// direction widens at both widths, which is what makes this a per-width question rather than
    /// a per-kind one.
    #[test]
    fn a_float_to_int_cast_widens_only_at_32_bit_lanes() {
        let narrow = "fn k(x: []f32, mut o: []i32, n: i64) { let mut i: i64 = 0; \
                      while i < n { o[i] = x[i] as i32; i = i + 1; } } \
                      fn main() -> i32 { return 0; }";
        let wide = "fn k(x: []f64, mut o: []i64, n: i64) { let mut i: i64 = 0; \
                    while i < n { o[i] = x[i] as i64; i = i + 1; } } \
                    fn main() -> i32 { return 0; }";
        let back = "fn k(x: []i64, mut o: []f64, n: i64) { let mut i: i64 = 0; \
                    while i < n { o[i] = x[i] as f64; i = i + 1; } } \
                    fn main() -> i32 { return 0; }";
        assert!(is_widened(narrow, "k"), "f32 -> i32 should widen");
        assert!(!is_widened(wide, "k"), "f64 -> i64 must not widen");
        assert!(is_widened(back, "k"), "i64 -> f64 should widen");
    }

    /// Cranelift's vector shifts take one *scalar* amount for every lane, and so does MIR's
    /// vector-by-scalar `Bin` shape — so a shift whose amount is the same in every lane widens at
    /// every integer width, `>>` as well as `<<`, by a literal or by a loop-invariant value. Each
    /// emitted shift keeps its amount scalar (asserted on the MIR). A per-lane amount (`x[i] <<
    /// y[i]`) never becomes a vector shift — handing Cranelift a vector amount is the CLIF type
    /// error that, in a release build with the verifier off, crashed the compiler inside ISLE. It
    /// is replicated lane by lane instead, and a body with no other arithmetic, like this one, is
    /// not worth that and stays scalar.
    #[test]
    fn a_uniform_shift_widens_at_every_width_and_a_per_lane_one_does_not() {
        for (ty, lit) in [
            ("i8", "(3 as i8)"),
            ("i16", "(3 as i16)"),
            ("i32", "3"),
            ("i64", "3"),
        ] {
            for op in ["<<", ">>"] {
                let src = format!(
                    "fn k(x: [{ty}; 64], mut o: [{ty}; 64], n: i64) {{ let mut i: i64 = 0; \
                     while i < n {{ o[i] = x[i] {op} {lit}; i = i + 1; }} }} \
                     fn main() -> i32 {{ return 0; }}"
                );
                assert!(is_widened(&src, "k"), "{ty} {op} by a literal must widen");
            }
            let variable = format!(
                "fn k(x: [{ty}; 64], mut o: [{ty}; 64], amount: {ty}, n: i64) {{ \
                 let mut i: i64 = 0; while i < n {{ \
                 o[i] = x[i] >> amount; i = i + 1; }} }} \
                 fn main() -> i32 {{ return 0; }}"
            );
            let (program, mut interner) = optimized(&variable, 2);
            let f = program.function(interner.intern("k")).expect("no fn k");
            let shifts: Vec<(bool, bool)> = f
                .blocks
                .iter()
                .flat_map(|b| b.insts.iter())
                .filter_map(|i| match &i.op {
                    Op::Bin(BinOp::AShr | BinOp::LShr | BinOp::Shl, a, b) => {
                        Some((f.value_type(*a).is_vector(), f.value_type(*b).is_vector()))
                    }
                    _ => None,
                })
                .collect();
            assert!(
                shifts.contains(&(true, false)),
                "{ty}: no vector-by-scalar shift in {shifts:?}"
            );
            assert!(
                !shifts.contains(&(true, true)),
                "{ty}: a shift took a vector amount: {shifts:?}"
            );
            let per_lane = format!(
                "fn k(x: [{ty}; 64], y: [{ty}; 64], mut o: [{ty}; 64], n: i64) {{ \
                 let mut i: i64 = 0; while i < n {{ \
                 o[i] = x[i] << y[i]; i = i + 1; }} }} \
                 fn main() -> i32 {{ return 0; }}"
            );
            assert!(
                !is_widened(&per_lane, "k"),
                "{ty} per-lane shift must stay scalar"
            );
        }
    }

    /// Every width, both directions, negative values and an amount past the lane width (masked, as
    /// the scalar shift masks it): `-O2` must print what `-O0` prints.
    #[test]
    fn uniform_shifts_agree_with_the_scalar_loop() {
        for (ty, amount) in [("i8", 11), ("i16", 9), ("i32", 33), ("i64", 61)] {
            let src = format!(
                "fn k(x: []{ty}, mut a: []{ty}, mut b: []{ty}, s: {ty}, n: i64) {{ \
                 let mut i: i64 = 0; while i < n {{ a[i] = x[i] >> s; b[i] = x[i] << s; \
                 i = i + 1; }} }} \
                 fn main() -> i32 {{ let mut x: []{ty} = alloc_{ty}(37); \
                 let mut a: []{ty} = alloc_{ty}(37); let mut b: []{ty} = alloc_{ty}(37); \
                 let mut j: i64 = 0; while j < 37 {{ x[j] = ((j * 7 - 120) as {ty}); j = j + 1; }} \
                 k(x, a, b, ({amount} as {ty}), 37); \
                 j = 0; while j < 37 {{ print(a[j] as i64); print(b[j] as i64); j = j + 1; }} \
                 free(x); free(a); free(b); return 0; }}"
            );
            scalar_and_vector_agree(&src);
        }
    }

    /// x86 has no packed byte multiply, and Cranelift does not synthesize one (`imul.i8x16` is
    /// `Unsupported`), so an `i8` product used to keep its whole loop scalar. It now widens through
    /// `i16`: both operands are sign-extended half by half, multiplied, and truncated back — the low
    /// eight bits of a product do not depend on how the operands were extended.
    #[test]
    fn an_i8_multiply_widens_through_i16() {
        let src = "fn k(x: [i8; 64], mut o: [i8; 64], n: i64) { let mut i: i64 = 0; \
                   while i < n { o[i] = x[i] * (3 as i8); i = i + 1; } } \
                   fn main() -> i32 { return 0; }";
        let vs = vector_values(src, "k");
        assert!(
            vs.contains(&(MirType::I8, 16)) && vs.contains(&(MirType::I16, 8)),
            "expected <16 x i8> data multiplied in <8 x i16> halves: {vs:?}"
        );
        scalar_and_vector_agree(
            "fn k(x: []i8, y: []i8, mut o: []i8, n: i64) { let mut i: i64 = 0; \
             while i < n { o[i] = x[i] * y[i] + (1 as i8); i = i + 1; } } \
             fn main() -> i32 { let mut x: []i8 = alloc_i8(37); \
             let mut y: []i8 = alloc_i8(37); let mut o: []i8 = alloc_i8(37); \
             let mut j: i64 = 0; \
             while j < 37 { x[j] = ((j * 9 - 150) as i8); y[j] = ((j * 5 + 3) as i8); j = j + 1; } \
             k(x, y, o, 37); \
             j = 0; while j < 37 { print(o[j] as i32); j = j + 1; } \
             free(x); free(y); free(o); return 0; }",
        );
    }

    /// The same exclusion reached through the *reduction* path, where the combine is built by
    /// `red_binop` rather than copied from the body. A product reduction over `i8` still widens —
    /// its loads go 16 lanes wide — but it may not take a vector accumulator, because that
    /// accumulator's combine would be the packed byte multiply that does not exist. It falls back
    /// to the serial fold, whose multiplies are scalar.
    #[test]
    fn an_i8_product_reduction_folds_serially() {
        let src = "fn k(x: [i8; 64], n: i64) -> i8 { let mut s: i8 = 1; let mut i: i64 = 0; \
                   while i < n { s = s * x[i]; i = i + 1; } return s; } \
                   fn main() -> i32 { return 0; }";
        let vs = vector_values(src, "k");
        assert!(!vs.is_empty(), "the loads should still be widened");
        assert!(
            vs.iter().all(|(l, n)| *l == MirType::I8 && *n == 16),
            "{vs:?}"
        );
    }

    // ---- mixed lane widths ---------------------------------------------------------------------

    /// The group width follows the NARROWEST lane: `u8` pixels in, `f32` out is sixteen iterations
    /// per group — one `<16 x u8>` load, two zero-extending widen steps, four `<4 x f32>` parts —
    /// and every lane must still be what the scalar `(px as f32)` computed.
    #[test]
    fn a_u8_to_f32_normalize_widens_in_parts() {
        let src = "fn k(px: []u8, mut o: []f32, n: i64) { let mut i: i64 = 0; \
                   while i < n { o[i] = (px[i] as f32) * 0.25 - 3.0; i = i + 1; } } \
                   fn main() -> i32 { return 0; }";
        let vs = vector_values(src, "k");
        for want in [
            (MirType::I8, 16),
            (MirType::I16, 8),
            (MirType::I32, 4),
            (MirType::F32, 4),
        ] {
            assert!(vs.contains(&want), "missing {want:?} in {vs:?}");
        }
        scalar_and_vector_agree(
            "fn k(px: []u8, mut o: []f32, n: i64) { let mut i: i64 = 0; \
             while i < n { o[i] = (px[i] as f32) * 0.25 - 3.0; i = i + 1; } } \
             fn main() -> i32 { let mut px: []u8 = alloc_u8(37); let mut o: []f32 = alloc_f32(37); \
             let mut j: i64 = 0; while j < 37 { px[j] = ((j * 29 + 7) as u8); j = j + 1; } \
             k(px, o, 37); \
             j = 0; while j < 37 { print((o[j] * 4.0) as i32); j = j + 1; } \
             free(px); free(o); return 0; }",
        );
    }

    /// `f32 -> i8` and `f32 -> u8` saturate. Spelled as one saturating `f32 -> i32` and then
    /// saturating packs, the nested clamps compose to exactly the scalar clamp — including at the
    /// edges, which the inputs here straddle (`±200`, `NaN`-free, every lane printed).
    #[test]
    fn a_saturating_float_to_byte_quantize_agrees_with_the_scalar_loop() {
        let src = "fn k(x: []f32, mut q: []i8, mut u: []u8, n: i64) { let mut i: i64 = 0; \
                   while i < n { q[i] = (x[i] * 4.0) as i8; u[i] = (x[i] * 4.0) as u8; \
                   i = i + 1; } } \
                   fn main() -> i32 { return 0; }";
        let vs = vector_values(src, "k");
        assert!(
            vs.contains(&(MirType::I8, 16)) && vs.contains(&(MirType::F32, 4)),
            "the quantize did not widen: {vs:?}"
        );
        scalar_and_vector_agree(
            "fn k(x: []f32, mut q: []i8, mut u: []u8, n: i64) { let mut i: i64 = 0; \
             while i < n { q[i] = (x[i] * 4.0) as i8; u[i] = (x[i] * 4.0) as u8; \
             i = i + 1; } } \
             fn main() -> i32 { let mut x: []f32 = alloc_f32(37); let mut q: []i8 = alloc_i8(37); \
             let mut u: []u8 = alloc_u8(37); \
             let mut j: i64 = 0; while j < 37 { x[j] = ((j as f32) - 18.0) * 2.9; j = j + 1; } \
             k(x, q, u, 37); \
             j = 0; while j < 37 { print(q[j] as i32); print(u[j] as i32); j = j + 1; } \
             free(x); free(q); free(u); return 0; }",
        );
    }

    /// The int8 dot product: `acc_i32 += (a[i] as i32) * (b[i] as i32)`. Sixteen bytes per group,
    /// sign-extended to four `<4 x i32>` parts, multiplied, and summed into a four-part vector
    /// accumulator — integer addition is associative including its wraparound, so the parts and lanes
    /// may be combined in any order and the answer is still the scalar one to the bit.
    #[test]
    fn an_i8_dot_into_an_i32_accumulator_widens() {
        let src =
            "fn k(a: []i8, b: []i8, n: i64) -> i32 { let mut s: i32 = 0; let mut i: i64 = 0; \
                   while i < n { s = s + (a[i] as i32) * (b[i] as i32); i = i + 1; } return s; } \
                   fn main() -> i32 { return 0; }";
        let vs = vector_values(src, "k");
        assert!(
            vs.contains(&(MirType::I8, 16)) && vs.contains(&(MirType::I32, 4)),
            "the i8 dot did not widen: {vs:?}"
        );
        scalar_and_vector_agree(
            "fn k(a: []i8, b: []i8, n: i64) -> i32 { let mut s: i32 = 0; let mut i: i64 = 0; \
             while i < n { s = s + (a[i] as i32) * (b[i] as i32); i = i + 1; } return s; } \
             fn main() -> i32 { let mut a: []i8 = alloc_i8(37); let mut b: []i8 = alloc_i8(37); \
             let mut j: i64 = 0; \
             while j < 37 { a[j] = ((j * 11 - 128) as i8); b[j] = ((127 - j * 13) as i8); j = j + 1; } \
             print(k(a, b, 37)); free(a); free(b); return 0; }",
        );
    }

    /// An `f32` stream accumulated in `f64`: four iterations per group, each `<4 x f32>` widened to
    /// two `<2 x f64>` parts, then folded serially in iteration order — a float sum is never
    /// reassociated, at any width.
    #[test]
    fn an_f32_stream_summed_in_f64_widens_and_folds_in_order() {
        let src = "fn k(x: []f32, n: i64) -> f64 { let mut s: f64 = 0.0; let mut i: i64 = 0; \
                   while i < n { s = s + (x[i] as f64); i = i + 1; } return s; } \
                   fn main() -> i32 { return 0; }";
        let vs = vector_values(src, "k");
        assert!(
            vs.contains(&(MirType::F32, 4)) && vs.contains(&(MirType::F64, 2)),
            "the f32->f64 sum did not widen: {vs:?}"
        );
        scalar_and_vector_agree(
            "fn k(x: []f32, n: i64) -> f64 { let mut s: f64 = 0.0; let mut i: i64 = 0; \
             while i < n { s = s + (x[i] as f64); i = i + 1; } return s; } \
             fn main() -> i32 { let mut x: []f32 = alloc_f32(37); \
             let mut j: i64 = 0; while j < 37 { x[j] = ((j as f32) + 0.1) * 1.7; j = j + 1; } \
             print((k(x, 37) * 1024.0) as i64); free(x); return 0; }",
        );
    }

    /// A narrowing store: `i32` arithmetic truncated into an `i16` stream, eight iterations per group
    /// — two `<4 x i32>` parts joined by one truncating narrow.
    #[test]
    fn an_i32_result_truncated_into_i16_widens() {
        let src = "fn k(x: []i32, mut o: []i16, n: i64) { let mut i: i64 = 0; \
                   while i < n { o[i] = (x[i] * 3 + 7) as i16; i = i + 1; } } \
                   fn main() -> i32 { return 0; }";
        let vs = vector_values(src, "k");
        assert!(
            vs.contains(&(MirType::I16, 8)) && vs.contains(&(MirType::I32, 4)),
            "the truncating store did not widen: {vs:?}"
        );
        scalar_and_vector_agree(
            "fn k(x: []i32, mut o: []i16, n: i64) { let mut i: i64 = 0; \
             while i < n { o[i] = (x[i] * 3 + 7) as i16; i = i + 1; } } \
             fn main() -> i32 { let mut x: []i32 = alloc_i32(37); let mut o: []i16 = alloc_i16(37); \
             let mut j: i64 = 0; while j < 37 { x[j] = ((j - 18) * 4099) as i32; j = j + 1; } \
             k(x, o, 37); \
             j = 0; while j < 37 { print(o[j] as i32); j = j + 1; } \
             free(x); free(o); return 0; }",
        );
    }

    /// A lane mask computed at one width selecting values of another: `x[i] < 0.0` over `f32` is four
    /// `<4 x i32>` mask parts, while the merged value is one `<16 x i8>`. The mask is narrowed
    /// (truncation keeps all-ones and zero lanes exactly) before it blends.
    #[test]
    fn a_wide_mask_selects_narrow_values() {
        let src = "fn k(x: []f32, y: []i8, mut o: []i8, n: i64) { let mut i: i64 = 0; \
                   while i < n { let mut t: i8 = y[i]; if x[i] < 0.0 { t = 0 as i8; } \
                   o[i] = t; i = i + 1; } } \
                   fn main() -> i32 { return 0; }";
        assert!(
            is_widened(src, "k"),
            "the mixed-width triangle did not widen"
        );
        scalar_and_vector_agree(
            "fn k(x: []f32, y: []i8, mut o: []i8, n: i64) { let mut i: i64 = 0; \
             while i < n { let mut t: i8 = y[i]; if x[i] < 0.0 { t = 0 as i8; } \
             o[i] = t; i = i + 1; } } \
             fn main() -> i32 { let mut x: []f32 = alloc_f32(37); let mut y: []i8 = alloc_i8(37); \
             let mut o: []i8 = alloc_i8(37); \
             let mut j: i64 = 0; \
             while j < 37 { x[j] = ((j % 5) as f32) - 2.0; y[j] = ((j * 3 + 1) as i8); j = j + 1; } \
             k(x, y, o, 37); \
             j = 0; while j < 37 { print(o[j] as i32); j = j + 1; } \
             free(x); free(y); free(o); return 0; }",
        );
    }

    /// An `i64` counter compared as a value over `f32` data: the ramp is two `<2 x i64>` parts and
    /// its mask is narrowed to the `f32` value's width before the select. (The previous rule — "the
    /// ramp's lane count must equal the data's" — declined this outright.)
    #[test]
    fn an_i64_counter_compared_over_f32_data_widens() {
        let src = "fn k(x: []f32, mut o: []f32, t: i64, n: i64) { let mut i: i64 = 0; \
                   while i < n { let q: f32 = if i == t { 8.0 } else { 2.0 }; \
                   o[i] = x[i] * q; i = i + 1; } } \
                   fn main() -> i32 { return 0; }";
        let vs = vector_values(src, "k");
        assert!(
            vs.contains(&(MirType::I64, 2)) && vs.contains(&(MirType::F32, 4)),
            "the i64 ramp over f32 data did not widen: {vs:?}"
        );
        scalar_and_vector_agree(
            "fn k(x: []f32, mut o: []f32, t: i64, n: i64) { let mut i: i64 = 0; \
             while i < n { let q: f32 = if i == t { 8.0 } else { 2.0 }; \
             o[i] = x[i] * q; i = i + 1; } } \
             fn main() -> i32 { let mut x: []f32 = alloc_f32(23); let mut o: []f32 = alloc_f32(23); \
             let mut j: i64 = 0; while j < 23 { x[j] = (j as f32) + 1.0; j = j + 1; } \
             k(x, o, 17, 23); \
             j = 0; while j < 23 { print(o[j] as i32); j = j + 1; } \
             free(x); free(o); return 0; }",
        );
    }

    /// A bf16 lane is an `f32` lane in a register (module docs): a bf16 body widens to
    /// `<4 x bf16>` parts — four lanes per 128-bit register, stored as four 2-byte elements — and to
    /// nothing else.
    #[test]
    fn a_bf16_loop_widens_as_four_f32_lanes() {
        let src = "fn k(x: []bf16, y: []bf16, mut o: []bf16, n: i64) { let mut i: i64 = 0; \
                   while i < n { o[i] = x[i] * y[i] + x[i]; i = i + 1; } } \
                   fn main() -> i32 { return 0; }";
        let vs = vector_values(src, "k");
        assert!(!vs.is_empty(), "a bf16 body was not widened");
        assert!(
            vs.iter().all(|(l, n)| *l == MirType::BF16 && *n == 4),
            "{vs:?}"
        );
    }

    /// Every lane of five bf16 bodies, `-O0` (the scalar loops) against `-O2` (`<4 x bf16>` groups
    /// and the scalar epilogue), printed as raw bits: the contracted `x * y + x` and its rounding,
    /// an ordered compare choosing `x` or `-y`, a division (0/0, inf/inf, x/0), `fpext` to `f32`
    /// and `fptrunc` back with ties to even — over NaN payloads, both zeros, both infinities, the
    /// smallest subnormal and the largest finite value. The four rounding points a bf16 value has
    /// (load, store, constant, cast) must land on the same lanes in both programs.
    #[test]
    fn bf16_lanes_agree_with_the_scalar_loop() {
        scalar_and_vector_agree(
            "fn pick(a: bf16, b: bf16) -> bf16 { if a > b { return a; } return -b; } \
             @unchecked fn k(x: []bf16, y: []bf16, mut o1: []bf16, mut o2: []bf16, \
               mut o3: []bf16, mut f: []f32, mut o5: []bf16, n: i64, d: i64) { \
               if d > 0 { k(x, y, o1, o2, o3, f, o5, n, d - 1); return; } \
               let mut i: i64 = 0; \
               while i < n { o1[i] = x[i] * y[i] + x[i]; i = i + 1; } \
               i = 0; while i < n { o2[i] = pick(x[i], y[i]); i = i + 1; } \
               i = 0; while i < n { o3[i] = x[i] / y[i]; i = i + 1; } \
               i = 0; while i < n { f[i] = (x[i] as f32) * 0.5; i = i + 1; } \
               i = 0; while i < n { o5[i] = (f[i] * 3.0) as bf16; i = i + 1; } } \
             fn xs(r: i64) -> u16 { let t: [u16; 16] = [32705, 32768, 0, 32640, 65408, 16320, \
               49168, 1, 32639, 16257, 16457, 49024, 16000, 17152, 16256, 16672]; return t[r]; } \
             fn ys(r: i64) -> u16 { let t: [u16; 16] = [16256, 16384, 0, 32768, 32640, 16257, \
               16255, 49216, 128, 19328, 16128, 32512, 48640, 16800, 16383, 32706]; return t[r]; } \
             fn main() -> i32 { let n: i64 = 37; \
               let mut x: []bf16 = alloc_bf16(n); let mut y: []bf16 = alloc_bf16(n); \
               let mut o1: []bf16 = alloc_bf16(n); let mut o2: []bf16 = alloc_bf16(n); \
               let mut o3: []bf16 = alloc_bf16(n); let mut o5: []bf16 = alloc_bf16(n); \
               let mut f: []f32 = alloc_f32(n); let mut i: i64 = 0; \
               while i < n { x[i] = bf16_from_bits(xs(i % 16)); \
                 y[i] = bf16_from_bits(ys((5 * i + 3) % 16)); i = i + 1; } \
               k(x, y, o1, o2, o3, f, o5, n, 0); i = 0; \
               while i < n { print(bits(o1[i])); print(bits(o2[i])); print(bits(o3[i])); \
                 print(bits(f[i])); print(bits(o5[i])); i = i + 1; } \
               return 0; }",
        );
    }

    /// A bf16 accumulator rounds to the grid inside its own recurrence (`fptrunc` after every
    /// combine), which no fold form reproduces, so the loop stays scalar.
    #[test]
    fn a_bf16_reduction_stays_scalar() {
        let src = "fn k(x: []bf16, n: i64) -> bf16 { let mut s: bf16 = 0.0; let mut i: i64 = 0; \
                   while i < n { s = s + x[i]; i = i + 1; } return s; } \
                   fn main() -> i32 { return 0; }";
        assert!(!is_widened(src, "k"), "a bf16 sum must not be widened");
    }

    /// An f16 lane is an `f32` lane in a register, like a bf16 one: an f16 body widens to
    /// `<4 x f16>` parts and to nothing else.
    #[test]
    fn an_f16_loop_widens_as_four_f32_lanes() {
        let src = "fn k(x: []f16, y: []f16, mut o: []f16, n: i64) { let mut i: i64 = 0; \
                   while i < n { o[i] = x[i] * y[i] + x[i]; i = i + 1; } } \
                   fn main() -> i32 { return 0; }";
        let vs = vector_values(src, "k");
        assert!(!vs.is_empty(), "an f16 body was not widened");
        assert!(
            vs.iter().all(|(l, n)| *l == MirType::F16 && *n == 4),
            "{vs:?}"
        );
    }

    /// The f16 twin of [`bf16_lanes_agree_with_the_scalar_loop`]: every lane of five f16 bodies,
    /// `-O0` against `-O2`, as raw bits — over NaN payloads (quiet and signalling), both zeros, both
    /// infinities, the smallest subnormal and normal, the largest finite value (whose products
    /// overflow) and rounding ties.
    #[test]
    fn f16_lanes_agree_with_the_scalar_loop() {
        scalar_and_vector_agree(
            "fn pick(a: f16, b: f16) -> f16 { if a > b { return a; } return -b; } \
             @unchecked fn k(x: []f16, y: []f16, mut o1: []f16, mut o2: []f16, \
               mut o3: []f16, mut f: []f32, mut o5: []f16, n: i64, d: i64) { \
               if d > 0 { k(x, y, o1, o2, o3, f, o5, n, d - 1); return; } \
               let mut i: i64 = 0; \
               while i < n { o1[i] = x[i] * y[i] + x[i]; i = i + 1; } \
               i = 0; while i < n { o2[i] = pick(x[i], y[i]); i = i + 1; } \
               i = 0; while i < n { o3[i] = x[i] / y[i]; i = i + 1; } \
               i = 0; while i < n { f[i] = (x[i] as f32) * 0.5; i = i + 1; } \
               i = 0; while i < n { o5[i] = (f[i] * 3.0) as f16; i = i + 1; } } \
             fn xs(r: i64) -> u16 { let t: [u16; 16] = [32257, 32000, 32768, 0, 31744, 64512, \
               15872, 49280, 1, 31743, 15361, 16968, 48128, 13312, 22528, 18688]; return t[r]; } \
             fn ys(r: i64) -> u16 { let t: [u16; 16] = [15360, 16384, 0, 32768, 31744, 15361, \
               15359, 49664, 1024, 27648, 14336, 30720, 45056, 19712, 16383, 32258]; return t[r]; } \
             fn main() -> i32 { let n: i64 = 37; \
               let mut x: []f16 = alloc_f16(n); let mut y: []f16 = alloc_f16(n); \
               let mut o1: []f16 = alloc_f16(n); let mut o2: []f16 = alloc_f16(n); \
               let mut o3: []f16 = alloc_f16(n); let mut o5: []f16 = alloc_f16(n); \
               let mut f: []f32 = alloc_f32(n); let mut i: i64 = 0; \
               while i < n { x[i] = f16_from_bits(xs(i % 16)); \
                 y[i] = f16_from_bits(ys((5 * i + 3) % 16)); i = i + 1; } \
               k(x, y, o1, o2, o3, f, o5, n, 0); i = 0; \
               while i < n { print(bits(o1[i])); print(bits(o2[i])); print(bits(o3[i])); \
                 print(bits(f[i])); print(bits(o5[i])); i = i + 1; } \
               return 0; }",
        );
    }

    /// An f16 accumulator rounds to its grid inside its own recurrence, like a bf16 one, so the
    /// loop stays scalar.
    #[test]
    fn an_f16_reduction_stays_scalar() {
        let src = "fn k(x: []f16, n: i64) -> f16 { let mut s: f16 = 0.0; let mut i: i64 = 0; \
                   while i < n { s = s + x[i]; i = i + 1; } return s; } \
                   fn main() -> i32 { return 0; }";
        assert!(!is_widened(src, "k"), "an f16 sum must not be widened");
    }

    /// Does `func` at `-O2` replicate any step — pack a lane with `Op::InsertLane` — or does every
    /// step it widened run on whole vectors?
    fn replicates(src: &str, func: &str) -> bool {
        let (program, mut interner) = optimized(src, 2);
        let f = program
            .function(interner.intern(func))
            .unwrap_or_else(|| panic!("no fn {func}"));
        f.blocks
            .iter()
            .flat_map(|b| &b.insts)
            .any(|i| matches!(i.op, Op::InsertLane(..)))
    }

    /// Half casts through `f64` and to and from integers convert lane-wise — the half end rounds
    /// (or not) exactly where the scalar cast does, the `f32 <-> f64` end is a widen or a narrow —
    /// where each lane used to be extracted, converted by the scalar instruction and packed again.
    /// Each body widens with no lane replicated at all.
    #[test]
    fn half_casts_through_f64_and_integers_convert_lane_wise() {
        for (name, ty, from, to, expr) in [
            ("f16 to f64", "f16", "f16", "f64", "x[i] as f64 * 0.5"),
            ("bf16 to f64", "bf16", "bf16", "f64", "x[i] as f64 - 1.0"),
            ("f64 to f16", "f16", "f64", "f16", "(x[i] * 3.0) as f16"),
            ("f64 to bf16", "bf16", "f64", "bf16", "(x[i] * 3.0) as bf16"),
            ("f16 to i32", "f16", "f16", "i32", "x[i] as i32"),
            ("bf16 to i32", "bf16", "bf16", "i32", "x[i] as i32"),
            ("f16 to u8", "f16", "f16", "u8", "x[i] as u8"),
            ("f16 to u32", "f16", "f16", "u32", "x[i] as u32"),
            ("i32 to f16", "f16", "i32", "f16", "x[i] as f16"),
            ("i32 to bf16", "bf16", "i32", "bf16", "x[i] as bf16"),
            ("i16 to f16", "f16", "i16", "f16", "x[i] as f16"),
            ("u8 to bf16", "bf16", "u8", "bf16", "x[i] as bf16"),
        ] {
            let src = format!(
                "fn k(x: []{from}, mut o: []{to}, n: i64) {{ let mut i: i64 = 0; \
                 while i < n {{ o[i] = {expr}; i = i + 1; }} }} \
                 fn main() -> i32 {{ return 0; }}"
            );
            let vs = vector_values(&src, "k");
            assert!(
                vs.iter().any(|(l, _)| l.display() == ty),
                "{name}: no {ty} vector: {vs:?}"
            );
            assert!(!replicates(&src, "k"), "{name}: a cast is still replicated");
        }
    }

    /// Every lane of those casts, `-O0` against `-O2`, over every class a half holds — NaN
    /// payloads, both zeros and infinities, subnormals, the largest finite values — and integers
    /// at and past the half ranges (f16 saturates to infinity from 65520; bf16 rounds 2^24 + 2^16
    /// + 1 through `f32`). Runtime trips, so the vector loop and the scalar tail both run.
    #[test]
    fn half_cast_lanes_agree_with_the_scalar_loop() {
        scalar_and_vector_agree(
            "@unchecked fn k(x: []f16, b: []bf16, p: []i32, s: []i16, mut d: []f64, \
               mut e: []f64, mut x2: []f16, mut b2: []bf16, mut pi: []i32, mut qi: []i32, \
               mut u: []u8, mut x3: []f16, mut b3: []bf16, mut y3: []f16, n: i64, z: i64) { \
               if z > 0 { k(x, b, p, s, d, e, x2, b2, pi, qi, u, x3, b3, y3, n, z - 1); return; } \
               let mut i: i64 = 0; \
               while i < n { d[i] = x[i] as f64 * 0.5; i = i + 1; } \
               i = 0; while i < n { e[i] = b[i] as f64 - 1.0; i = i + 1; } \
               i = 0; while i < n { x2[i] = (d[i] * 3.0) as f16; i = i + 1; } \
               i = 0; while i < n { b2[i] = (e[i] * 3.0) as bf16; i = i + 1; } \
               i = 0; while i < n { pi[i] = x[i] as i32; i = i + 1; } \
               i = 0; while i < n { qi[i] = b[i] as i32; i = i + 1; } \
               i = 0; while i < n { u[i] = x[i] as u8; i = i + 1; } \
               i = 0; while i < n { x3[i] = p[i] as f16; i = i + 1; } \
               i = 0; while i < n { b3[i] = p[i] as bf16; i = i + 1; } \
               i = 0; while i < n { y3[i] = s[i] as f16; i = i + 1; } } \
             fn main() -> i32 { let n: i64 = 67; \
               let mut x: []f16 = alloc_f16(n); let mut b: []bf16 = alloc_bf16(n); \
               let mut p: []i32 = alloc_i32(n); let mut s: []i16 = alloc_i16(n); \
               let mut d: []f64 = alloc_f64(n); let mut e: []f64 = alloc_f64(n); \
               let mut x2: []f16 = alloc_f16(n); let mut b2: []bf16 = alloc_bf16(n); \
               let mut pi: []i32 = alloc_i32(n); let mut qi: []i32 = alloc_i32(n); \
               let mut u: []u8 = alloc_u8(n); let mut x3: []f16 = alloc_f16(n); \
               let mut b3: []bf16 = alloc_bf16(n); let mut y3: []f16 = alloc_f16(n); \
               let mut i: i64 = 0; \
               while i < n { x[i] = f16_from_bits(((i * 2731 + 1) % 65536) as u16); \
                 b[i] = bf16_from_bits(((i * 4099 + 7) % 65536) as u16); \
                 p[i] = ((i * 1000003) % 33554432 - 16777216) as i32; \
                 s[i] = ((i * 977) % 65536 - 32768) as i16; i = i + 1; } \
               p[3] = 2049; p[4] = 65519; p[5] = 65520; p[6] = -70000; p[7] = 16842753; \
               k(x, b, p, s, d, e, x2, b2, pi, qi, u, x3, b3, y3, n, 0); i = 0; \
               while i < n { print(bits(d[i])); print(bits(e[i])); print(bits(x2[i])); \
                 print(bits(b2[i])); print(pi[i]); print(qi[i]); print(u[i]); \
                 print(bits(x3[i])); print(bits(b3[i])); print(bits(y3[i])); i = i + 1; } \
               return 0; }",
        );
    }

    /// bf16 and f16 interleave groups widen like the 32-bit ones: a complex multiply over
    /// interleaved real and imaginary parts, its stride-2 loads and stores split and merged by
    /// lane shuffles, every lane `-O0` against `-O2` over special values.
    #[test]
    fn half_interleave_groups_widen_and_agree_with_the_scalar_loop() {
        let src = "@unchecked fn kh(a: []f16, c: []f16, mut o: []f16, n: i64, z: i64) { \
               if z > 0 { kh(a, c, o, n, z - 1); return; } let mut i: i64 = 0; \
               while i < n { \
                 o[2 * i] = a[2 * i] * c[2 * i] - a[2 * i + 1] * c[2 * i + 1]; \
                 o[2 * i + 1] = a[2 * i] * c[2 * i + 1] + a[2 * i + 1] * c[2 * i]; \
                 i = i + 1; } } \
             @unchecked fn kb(a: []bf16, c: []bf16, mut o: []bf16, n: i64, z: i64) { \
               if z > 0 { kb(a, c, o, n, z - 1); return; } let mut i: i64 = 0; \
               while i < n { \
                 o[2 * i] = a[2 * i] * c[2 * i] - a[2 * i + 1] * c[2 * i + 1]; \
                 o[2 * i + 1] = a[2 * i] * c[2 * i + 1] + a[2 * i + 1] * c[2 * i]; \
                 i = i + 1; } } \
             fn main() -> i32 { let n: i64 = 37; let m: i64 = 2 * n; \
               let mut a: []f16 = alloc_f16(m); let mut c: []f16 = alloc_f16(m); \
               let mut o: []f16 = alloc_f16(m); let mut a2: []bf16 = alloc_bf16(m); \
               let mut c2: []bf16 = alloc_bf16(m); let mut o2: []bf16 = alloc_bf16(m); \
               let mut i: i64 = 0; \
               while i < m { a[i] = f16_from_bits(((i * 2731 + 1) % 65536) as u16); \
                 c[i] = f16_from_bits(((i * 3517 + 5) % 65536) as u16); \
                 a2[i] = bf16_from_bits(((i * 4099 + 7) % 65536) as u16); \
                 c2[i] = bf16_from_bits(((i * 1543 + 3) % 65536) as u16); i = i + 1; } \
               kh(a, c, o, n, 0); kb(a2, c2, o2, n, 0); i = 0; \
               while i < m { print(bits(o[i])); print(bits(o2[i])); i = i + 1; } \
               return 0; }";
        for (f, lane) in [("kh", MirType::F16), ("kb", MirType::BF16)] {
            let vs = vector_values(src, f);
            assert!(
                vs.iter().any(|(l, _)| *l == lane),
                "{f}: the group was not widened: {vs:?}"
            );
        }
        scalar_and_vector_agree(src);
    }

    // ---- min/max reductions --------------------------------------------------------------------

    /// Does `func` carry a vector-typed block parameter — a lane-parallel accumulator — at `-O2`?
    fn has_vector_accumulator(src: &str, func: &str) -> bool {
        let (program, mut interner) = optimized(src, 2);
        let sym = interner.intern(func);
        let f = program
            .function(sym)
            .unwrap_or_else(|| panic!("no fn {func}"));
        f.blocks
            .iter()
            .any(|b| b.params.iter().any(|p| f.value_type(*p).is_vector()))
    }

    /// `m = max(m, x[i] * s)` used to be a `Recurrence` (the combine is a `select`, which the
    /// reduction analysis did not read) and declined the whole loop. A float max may not be
    /// reassociated — `max(max(1, NaN), 0)` is `0` but `max(1, max(NaN, 0))` is `1` — so the
    /// product goes two lanes wide and the running maximum is folded lane by lane, in index order:
    /// no vector accumulator. The differential feeds it a NaN mid-stream (which the next element
    /// displaces, exactly as in the scalar loop) and `±0` ties (the select keeps its second arm).
    #[test]
    fn a_float_max_reduction_widens_and_folds_in_order() {
        let kernel = "fn k(x: []f64, s: f64, n: i64) -> f64 { let mut m: f64 = -0.0; \
                      let mut i: i64 = 0; \
                      while i < n { m = max(m, x[i] * s); i = i + 1; } return m; }";
        let shape = format!("{kernel} fn main() -> i32 {{ return 0; }}");
        let vs = vector_values(&shape, "k");
        assert!(
            vs.contains(&(MirType::F64, 2)),
            "an f64 max did not widen: {vs:?}"
        );
        assert!(
            !has_vector_accumulator(&shape, "k"),
            "a float max must fold serially, not in lane-parallel partials"
        );
        scalar_and_vector_agree(&format!(
            "{kernel} fn main() -> i32 {{ let mut x: []f64 = alloc_f64(13); \
             let mut j: i64 = 0; \
             while j < 13 {{ x[j] = (((j * 5) % 13) as f64) - 6.0; j = j + 1; }} \
             let mut n: i64 = 0; \
             while n <= 13 {{ let m: f64 = k(x, 0.5, n); print((m * 4.0) as i64); n = n + 1; }} \
             x[5] = 0.0 / 0.0; \
             n = 0; \
             while n <= 13 {{ let m: f64 = k(x, 1.0, n); \
               if m != m {{ print(-77777); }} else {{ print((m * 4.0) as i64); }} n = n + 1; }} \
             j = 0; while j < 13 {{ x[j] = 0.0; if j % 3 == 1 {{ x[j] = -0.0; }} j = j + 1; }} \
             n = 0; \
             while n <= 13 {{ let m: f64 = k(x, 1.0, n); \
               if 1.0 / m > 0.0 {{ print(1); }} else {{ print(0); }} n = n + 1; }} \
             free(x); return 0; }}"
        ));
    }

    /// `min(x[i], m)` over `i32` — the accumulator on the right — is associative, so it keeps four
    /// lane-parallel partial minima seeded with `i32::MAX` and folds them once, after the loop.
    #[test]
    fn an_integer_min_reduction_keeps_vector_accumulators() {
        let kernel = "fn k(x: []i32, n: i64) -> i32 { let mut m: i32 = 2147483647; \
                      let mut i: i64 = 0; \
                      while i < n { m = min(x[i], m); i = i + 1; } return m; }";
        let shape = format!("{kernel} fn main() -> i32 {{ return 0; }}");
        assert!(
            has_vector_accumulator(&shape, "k"),
            "an i32 min must keep lane-parallel partials"
        );
        scalar_and_vector_agree(&format!(
            "{kernel} fn main() -> i32 {{ let mut x: []i32 = alloc_i32(19); \
             let mut j: i64 = 0; \
             while j < 19 {{ x[j] = ((j as i32) * 1103515245 + 12345) % 1000; j = j + 1; }} \
             let mut n: i64 = 0; \
             while n <= 19 {{ print(k(x, n)); n = n + 1; }} \
             free(x); return 0; }}"
        ));
    }

    /// An unsigned byte max: sixteen lanes, seeded with `0`, compared unsigned — `200u8` is larger
    /// than `100u8` although its `i8` bits are negative.
    #[test]
    fn an_unsigned_byte_max_compares_unsigned() {
        let kernel = "fn k(x: []u8, n: i64) -> u8 { let mut m: u8 = 0; let mut i: i64 = 0; \
                      while i < n { m = max(m, x[i]); i = i + 1; } return m; }";
        let shape = format!("{kernel} fn main() -> i32 {{ return 0; }}");
        let vs = vector_values(&shape, "k");
        assert!(
            vs.contains(&(MirType::I8, 16)),
            "a u8 max did not widen: {vs:?}"
        );
        scalar_and_vector_agree(&format!(
            "{kernel} fn main() -> i32 {{ let mut x: []u8 = alloc_u8(37); \
             let mut j: i64 = 0; \
             while j < 37 {{ x[j] = ((j * 37 + 11) % 251) as u8; j = j + 1; }} \
             let mut n: i64 = 0; \
             while n <= 37 {{ print(k(x, n) as i64); n = n + 1; }} \
             free(x); return 0; }}"
        ));
    }

    /// A 64-bit integer max widens its body but folds serially: the vector accumulator is limited
    /// to 32-bit lanes (`vector_combinable`).
    #[test]
    fn an_i64_max_folds_serially_and_agrees() {
        let kernel = "fn k(x: []i64, n: i64) -> i64 { let mut m: i64 = -5000000000; \
                      let mut i: i64 = 0; \
                      while i < n { m = max(m, x[i] * 3); i = i + 1; } return m; }";
        let shape = format!("{kernel} fn main() -> i32 {{ return 0; }}");
        assert!(is_widened(&shape, "k"), "an i64 max did not widen");
        assert!(
            !has_vector_accumulator(&shape, "k"),
            "an i64 max must not keep a vector accumulator"
        );
        scalar_and_vector_agree(&format!(
            "{kernel} fn main() -> i32 {{ let mut x: []i64 = alloc_i64(11); \
             let mut j: i64 = 0; \
             while j < 11 {{ x[j] = (j * 7919) % 23 - 2000000000; j = j + 1; }} \
             let mut n: i64 = 0; \
             while n <= 11 {{ print(k(x, n)); n = n + 1; }} \
             free(x); return 0; }}"
        ));
    }

    // ---- guarded reductions --------------------------------------------------------------------

    /// `if x[i] > t { s = s + x[i]; }` reaches the latch as a merge parameter and used to be a
    /// `Recurrence`. It is a guarded accumulate now: the compare goes four lanes wide and each lane
    /// folds `c ? s + x : s` in index order — the scalar merge itself, so a `-0.0` element, a NaN
    /// (whose compare is false) and the starting `-0.0` all come out as the scalar loop has them.
    #[test]
    fn a_guarded_float_sum_widens_and_agrees() {
        let kernel = "@unchecked fn k(x: []f32, t: f32, n: i64) -> f32 { let mut s: f32 = -0.0; \
                      let mut i: i64 = 0; \
                      while i < n { if x[i] > t { s = s + x[i]; } i = i + 1; } return s; }";
        let shape = format!("{kernel} fn main() -> i32 {{ return 0; }}");
        let vs = vector_values(&shape, "k");
        assert!(
            vs.contains(&(MirType::F32, 4)),
            "a guarded f32 sum did not widen: {vs:?}"
        );
        scalar_and_vector_agree(&format!(
            "{kernel} fn main() -> i32 {{ let mut x: []f32 = alloc_f32(23); \
             let mut j: i64 = 0; \
             while j < 23 {{ x[j] = (((j * 7) % 11) as f32) * 0.375 - 1.5; j = j + 1; }} \
             x[4] = -0.0; x[9] = 0.0 / 0.0; \
             let mut n: i64 = 0; \
             while n <= 23 {{ print((k(x, -0.5, n) * 64.0) as i64); \
               let z: f32 = k(x, 100.0, n); if 1.0 / z > 0.0 {{ print(1); }} else {{ print(0); }} \
               n = n + 1; }} \
             free(x); return 0; }}"
        ));
    }

    /// Two guarded accumulates on the two arms of one branch — `s` takes its update when the
    /// condition holds, the counter `q` when it does not — and the counter's addend is a constant
    /// every lane shares. The integer one keeps lane-parallel partials, blended per part.
    #[test]
    fn guarded_accumulates_on_both_arms_agree() {
        let kernel = "@unchecked fn k(x: []f32, t: f32, n: i64) -> i32 { let mut s: f32 = 0.0; \
                      let mut q: i32 = 0; let mut i: i64 = 0; \
                      while i < n { let v: f32 = x[i]; \
                        if v > t { s = s + v; } else { q = q + 1; } i = i + 1; } \
                      return (s as i32) * 1000 + q; }";
        let shape = format!("{kernel} fn main() -> i32 {{ return 0; }}");
        assert!(
            has_vector_accumulator(&shape, "k"),
            "the guarded i32 counter must keep lane-parallel partials"
        );
        scalar_and_vector_agree(&format!(
            "{kernel} fn main() -> i32 {{ let mut x: []f32 = alloc_f32(29); \
             let mut j: i64 = 0; \
             while j < 29 {{ x[j] = (((j * 5) % 13) as f32) - 6.0; j = j + 1; }} \
             let mut n: i64 = 0; \
             while n <= 29 {{ print(k(x, 0.5, n)); n = n + 1; }} \
             free(x); return 0; }}"
        ));
    }

    /// A max written as a branch, `if x[i] > m { m = x[i]; }`: the branch's compare and merge are
    /// the select, folded in order over `f64`; the `i32` min is associative and keeps partials.
    #[test]
    fn branch_spelled_min_and_max_widen_and_agree() {
        let fmax = "@unchecked fn k(x: []f64, n: i64) -> f64 { let mut m: f64 = -1000.0; \
                    let mut i: i64 = 0; \
                    while i < n { if x[i] > m { m = x[i]; } i = i + 1; } return m; }";
        let shape = format!("{fmax} fn main() -> i32 {{ return 0; }}");
        assert!(
            is_widened(&shape, "k"),
            "a branch-spelled f64 max did not widen"
        );
        assert!(!has_vector_accumulator(&shape, "k"));
        scalar_and_vector_agree(&format!(
            "{fmax} fn main() -> i32 {{ let mut x: []f64 = alloc_f64(13); \
             let mut j: i64 = 0; \
             while j < 13 {{ x[j] = (((j * 5) % 13) as f64) - 6.0; j = j + 1; }} \
             x[6] = 0.0 / 0.0; \
             let mut n: i64 = 0; \
             while n <= 13 {{ let m: f64 = k(x, n); \
               if m != m {{ print(-77777); }} else {{ print(m as i64); }} n = n + 1; }} \
             free(x); return 0; }}"
        ));
        let imin = "@unchecked fn k(x: []i32, n: i64) -> i32 { let mut m: i32 = 1000000; \
                    let mut i: i64 = 0; \
                    while i < n { let v: i32 = x[i]; if v < m { m = v; } i = i + 1; } \
                    return m; }";
        let shape = format!("{imin} fn main() -> i32 {{ return 0; }}");
        assert!(
            has_vector_accumulator(&shape, "k"),
            "a branch-spelled i32 min must keep lane-parallel partials"
        );
        scalar_and_vector_agree(&format!(
            "{imin} fn main() -> i32 {{ let mut x: []i32 = alloc_i32(19); \
             let mut j: i64 = 0; \
             while j < 19 {{ x[j] = (((j * 7919) % 601) as i32) - 300 - (j as i32) * 9; \
               j = j + 1; }} \
             let mut n: i64 = 0; \
             while n <= 19 {{ print(k(x, n)); n = n + 1; }} \
             free(x); return 0; }}"
        ));
    }

    /// The same branch also recording where the maximum was (`at = i`) is an argmax: per-lane
    /// `(value, index)` partials, folded on the way out by value and then by iteration order. The
    /// data puts tied maxima in different lanes (the earliest in a LATER lane), a NaN, a `-0.0`
    /// beside a `+0.0`, and every trip count `0..=13` so the vector/epilogue split moves through
    /// every position; the index and the sign-tagged maximum must match the scalar loop exactly.
    #[test]
    fn a_branch_that_records_an_index_widens_and_agrees() {
        let k =
            "@unchecked fn k(x: []f32, n: i64, mut o: []f32) -> i64 { let mut m: f32 = -100.0; \
                 let mut at: i64 = -1; let mut i: i64 = 0; \
                 while i < n { if x[i] > m { m = x[i]; at = i; } i = i + 1; } \
                 o[0] = m; return at; }";
        let shape = format!("{k} fn main() -> i32 {{ return 0; }}");
        assert!(is_widened(&shape, "k"), "an argmax keeps per-lane partials");
        assert!(has_vector_accumulator(&shape, "k"));
        let data = "x[0] = -3.0; x[1] = 0.0 * -1.0; x[2] = 0.0; x[3] = 0.0 / 0.0; x[4] = 2.0; \
                    x[5] = 7.0; x[6] = 5.0; x[7] = 7.0; x[8] = 7.0; x[9] = 1.0; x[10] = 0.0 / 0.0; \
                    x[11] = 7.0; x[12] = 7.0;";
        scalar_and_vector_agree(&format!(
            "{k} fn main() -> i32 {{ let mut x: []f32 = alloc_f32(13); \
             let mut o: []f32 = alloc_f32(1); {data} \
             let mut n: i64 = 0; \
             while n <= 13 {{ print(k(x, n, o)); let v: f32 = o[0]; \
               if 1.0 / v < 0.0 {{ print(-1); }} else {{ print((v * 4.0) as i64); }} \
               n = n + 1; }} \
             free(x); free(o); return 0; }}"
        ));
    }

    /// FindLastIV (`if c { last = i }`): per-lane "last counter" partials folded by iteration
    /// order. A one-hot hit at every position of a 13-element buffer (every lane of every group,
    /// and the epilogue), no hit, and a start value equal to a real index — on both arms.
    #[test]
    fn a_find_last_index_widens_and_agrees() {
        let k = "@unchecked fn k(x: []f32, n: i64, init: i64) -> i64 { let mut last: i64 = init; \
                 let mut i: i64 = 0; \
                 while i < n { if x[i] > 0.5 { last = i; } i = i + 1; } return last; } \
                 @unchecked fn e(x: []f32, n: i64) -> i64 { let mut last: i64 = -1; \
                 let mut i: i64 = 0; \
                 while i < n { if x[i] > 0.5 { } else { last = i; } i = i + 1; } return last; }";
        let shape = format!("{k} fn main() -> i32 {{ return 0; }}");
        assert!(
            is_widened(&shape, "k"),
            "a FindLast keeps per-lane partials"
        );
        assert!(has_vector_accumulator(&shape, "k"));
        assert!(is_widened(&shape, "e"), "so does its else-arm spelling");
        scalar_and_vector_agree(&format!(
            "{k} fn main() -> i32 {{ let mut x: []f32 = alloc_f32(13); \
             let mut p: i64 = 0; \
             while p <= 13 {{ let mut j: i64 = 0; \
               while j < 13 {{ x[j] = 0.0; j = j + 1; }} \
               if p < 13 {{ x[p] = 1.0; }} \
               let mut n: i64 = 0; \
               while n <= 13 {{ print(k(x, n, -1)); print(k(x, n, 5)); print(e(x, n)); \
                 n = n + 1; }} \
               p = p + 1; }} \
             free(x); return 0; }}"
        ));
    }

    /// An argmin of `i32` with the accumulator on the LEFT of the compare (`m > y[i]`), from a
    /// start nothing beats and from one everything beats: un-updated lanes tie only each other.
    #[test]
    fn an_integer_argmin_with_the_accumulator_left_agrees() {
        let k = "@unchecked fn k(y: []i32, n: i64, m0: i32) -> i64 { let mut m: i32 = m0; \
                 let mut at: i64 = -1; let mut i: i64 = 0; \
                 while i < n { if m > y[i] { m = y[i]; at = i; } i = i + 1; } return at; }";
        let shape = format!("{k} fn main() -> i32 {{ return 0; }}");
        assert!(is_widened(&shape, "k"));
        scalar_and_vector_agree(&format!(
            "{k} fn main() -> i32 {{ let mut y: []i32 = alloc_i32(19); \
             let mut j: i64 = 0; \
             while j < 19 {{ y[j] = (((j * 7) % 5) as i32) - 2; j = j + 1; }} \
             let mut n: i64 = 0; \
             while n <= 19 {{ print(k(y, n, 1000)); print(k(y, n, -2)); print(k(y, n, 0)); \
               n = n + 1; }} \
             free(y); return 0; }}"
        ));
    }

    /// A non-strict argmax (`x[i] >= m`) keeps the LAST of tied maxima, and an argmin written with
    /// the accumulator on the left (`m >= y[i]`) the last of tied minima. Their index partials start
    /// below the first counter so a lane that tied `m0` (and so moved) is told apart from one that
    /// never updated. Repeated maxima land in different lanes at every trip `0..=len`, ties with the
    /// start value, `±0` (one class: `-0 >= +0` and `+0 >= -0`), a NaN element and a NaN start.
    #[test]
    fn a_non_strict_argmax_keeps_the_last_tie_and_agrees() {
        let k = "@unchecked fn k(x: []f32, n: i64, m0: f32, mut o: []f32) -> i64 { \
                   let mut m: f32 = m0; let mut at: i64 = -1; let mut i: i64 = 0; \
                   while i < n { if x[i] >= m { m = x[i]; at = i; } i = i + 1; } \
                   o[0] = m; return at; } \
                 @unchecked fn j(y: []i32, n: i64, m0: i32) -> i64 { let mut m: i32 = m0; \
                   let mut at: i64 = -1; let mut i: i64 = 0; \
                   while i < n { if m >= y[i] { m = y[i]; at = i; } i = i + 1; } return at; }";
        let shape = format!("{k} fn main() -> i32 {{ return 0; }}");
        assert!(is_widened(&shape, "k"), "a `>=` argmax did not widen");
        assert!(has_vector_accumulator(&shape, "k"));
        assert!(is_widened(&shape, "j"), "a `<=` argmin did not widen");
        scalar_and_vector_agree(&format!(
            "{k} fn main() -> i32 {{ let mut x: []f32 = alloc_f32(13); \
             let mut o: []f32 = alloc_f32(1); \
             x[0] = -3.0; x[1] = 7.0; x[2] = 1.0; x[3] = 7.0; x[4] = 9.0; x[5] = 2.0; x[6] = 9.0; \
             x[7] = 0.0; x[8] = 9.0; x[9] = 5.0; x[10] = 4.0; x[11] = 9.0; x[12] = 8.0; \
             let mut n: i64 = 0; \
             while n <= 13 {{ print(k(x, n, -100.0, o)); print(k(x, n, 9.0, o)); \
               print(k(x, n, 10.0, o)); n = n + 1; }} \
             let negz: f32 = 0.0 * -1.0; let nan: f32 = 0.0 / 0.0; \
             x[0] = negz; x[1] = 0.0; x[2] = nan; x[3] = -1.0; x[4] = 0.0; x[5] = nan; \
             x[6] = negz; x[7] = -5.0; x[8] = 0.0; x[9] = negz; x[10] = -9.0; \
             n = 0; \
             while n <= 11 {{ print(k(x, n, -1.0, o)); print(bits(o[0])); \
               print(k(x, n, negz, o)); print(bits(o[0])); \
               print(k(x, n, nan, o)); print(bits(o[0])); n = n + 1; }} \
             let mut y: []i32 = alloc_i32(17); \
             y[0] = 5; y[1] = -2; y[2] = 8; y[3] = -2; y[4] = -7; y[5] = 3; y[6] = -7; y[7] = -7; \
             y[8] = 0; y[9] = 1; y[10] = -7; y[11] = 6; y[12] = -3; y[13] = 2; y[14] = -7; \
             y[15] = 9; y[16] = 4; \
             n = 0; \
             while n <= 17 {{ print(j(y, n, 1000)); print(j(y, n, -7)); print(j(y, n, -100)); \
               n = n + 1; }} \
             free(x); free(o); free(y); return 0; }}"
        ));
    }

    /// The argmax spellings whose per-lane partials would NOT reproduce the scalar answer stay
    /// scalar: the update on the `else` arm (`!(x <= m)` is true for a NaN), and an index that is
    /// not the counter.
    #[test]
    fn argmax_spellings_the_partials_cannot_reproduce_are_declined() {
        let else_arm = "@unchecked fn k(x: []f32, n: i64) -> i64 { let mut m: f32 = 0.0; \
                        let mut at: i64 = 0; let mut i: i64 = 0; \
                        while i < n { if x[i] <= m { } else { m = x[i]; at = i; } i = i + 1; } \
                        return at; } \
                        fn main() -> i32 { return 0; }";
        assert!(
            !is_widened(else_arm, "k"),
            "an else-arm argmax must stay scalar"
        );
        let not_iv = "@unchecked fn k(x: []f32, n: i64) -> i64 { let mut m: f32 = 0.0; \
                      let mut at: i64 = 0; let mut i: i64 = 0; \
                      while i < n { if x[i] > m { m = x[i]; at = n - i; } i = i + 1; } return at; } \
                      fn main() -> i32 { return 0; }";
        assert!(
            !is_widened(not_iv, "k"),
            "an index other than the counter must stay scalar"
        );
    }

    // ---- cost model ----------------------------------------------------------------------------

    /// The runtime alias check grows with the square of the stream count and is paid on every
    /// entry, so it has a budget of eight checks. Eight unrelated input slices beside one output
    /// slice need exactly eight (each input against the output) and still widen; a ninth input
    /// makes it nine, and the same loop stays scalar.
    #[test]
    fn the_runtime_alias_check_budget_is_eight() {
        let params = |k: usize| (0..k).map(|j| format!("x{j}: []f32, ")).collect::<String>();
        let sum = |k: usize| {
            (0..k)
                .map(|j| format!("x{j}[i]"))
                .collect::<Vec<_>>()
                .join(" + ")
        };
        let kernel = |k: usize| {
            format!(
                "fn k({}mut o: []f32, n: i64) {{ let mut i: i64 = 0; \
                 while i < n {{ o[i] = {}; i = i + 1; }} }} \
                 fn main() -> i32 {{ return 0; }}",
                params(k),
                sum(k)
            )
        };
        assert!(
            is_widened(&kernel(8), "k"),
            "eight alias checks are within the budget"
        );
        assert!(
            !is_widened(&kernel(9), "k"),
            "nine alias checks must not be paid on every entry"
        );
    }

    /// The cost model declines a group whose vector body costs more than the `W` scalar iterations
    /// it replaces, and nothing at or below that line; `WUKONG_VEC_COST=0` (here: `enforced =
    /// false`) widens it anyway.
    #[test]
    fn the_cost_model_declines_only_a_group_dearer_than_its_scalar_iterations() {
        assert!(cost_declines((29, 28), true));
        assert!(!cost_declines((28, 28), true));
        assert!(!cost_declines((3, 28), true));
        assert!(!cost_declines((29, 28), false));
        // Two `f64` lanes of `xd[j] = 0.0; if j % 3 == 1 { xd[j] = -0.0 }`: a remainder replicated
        // per lane and a store predicated per lane, estimated at 28 operations a group against 24
        // for the two scalar iterations. With `WUKONG_VEC_COST=0` it widens (`cost=28/24`).
        let src = "fn main() -> i32 { let mut xd: []f64 = alloc_f64(13); let mut j: i64 = 0; \
                   while j < 13 { xd[j] = 0.0; if j % 3 == 1 { xd[j] = -0.0; } j = j + 1; } \
                   j = 0; while j < 13 { print(bits(xd[j])); j = j + 1; } return 0; }";
        assert!(
            !is_widened(src, "main"),
            "a group dearer than its scalar iterations must stay scalar"
        );
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

    /// A gathered index is not affine in the induction variable, so no lane offset is known. `x`
    /// and `o` are two unpromised slices that may be one buffer, so the gather may read what the
    /// loop stores, and the checked `x[ix[i]]` keeps a bounds check `bce` cannot prove. Both used to
    /// keep the loop scalar; now the check is a lane test ([`linearize_spine`]) and the gather and
    /// the store run lane by lane in iteration order ([`SerialRegion`]). The differential passes one
    /// buffer as both, with indices pointing behind and ahead of the store.
    #[test]
    fn a_checked_gather_from_a_buffer_the_loop_stores_runs_lane_by_lane() {
        let k = "fn k(ix: []i32, x: []f32, mut o: []f32, n: i64, d: i64) { \
                 if d > 0 { again(d - 1); return; } let mut i: i64 = 0; \
                 while i < n { o[i] = x[ix[i] as i64] * 2.0 + 1.0; i = i + 1; } } \
                 fn again(d: i64) { let mut t: []f32 = alloc_f32(1); \
                 let mut q: []i32 = alloc_i32(1); k(q, t, t, 1, d); } ";
        assert!(
            is_widened(&format!("{k} fn main() -> i32 {{ return 0; }}"), "k"),
            "a checked gather beside a store was not widened"
        );
        scalar_and_vector_agree(&format!(
            "{k} fn main() -> i32 {{ let mut o: []f32 = alloc_f32(14); \
             let mut ix: []i32 = alloc_i32(14); let mut j: i64 = 0; \
             while j < 14 {{ o[j] = (j as f32) * 0.5; ix[j] = ((j * 9 + 4) % 14) as i32; \
               j = j + 1; }} \
             k(ix, o, o, 14, 0); j = 0; while j < 14 {{ print(o[j]); j = j + 1; }} \
             return 0; }}"
        ));
    }

    /// The induction variable is uniform across a group only as an *address*. Used as a value it is
    /// `i + k` in lane `k`. Converting an `i64` counter to `f32` has no exact packed form (`i64 ->
    /// f64 -> f32` rounds twice), so that one conversion is replicated: each lane converts its own
    /// `i + k` with the scalar cast, and the multiply reading the packed result widens. No lane
    /// ramp is built — the only step reading the counter as a value is the replicated one. The
    /// counter starts just below 2^24, where `f32` stops holding every integer, so a lane given
    /// the wrong `i + k`, or a conversion that rounded twice, changes what is printed.
    #[test]
    fn an_i64_counter_converted_to_f32_is_replicated_lane_by_lane() {
        let kernel = "@unchecked fn k(mut o: []f32, lo: i64, hi: i64) { \
                      let mut i: i64 = lo; \
                      while i < hi { o[i - lo] = (i as f32) * 0.25; i = i + 1; } }";
        let shape = format!("{kernel} fn main() -> i32 {{ return 0; }}");
        let (program, mut interner) = optimized(&shape, 2);
        let f = program.function(interner.intern("k")).expect("no fn k");
        let ops: Vec<&Op> = f
            .blocks
            .iter()
            .flat_map(|b| b.insts.iter())
            .map(|i| &i.op)
            .collect();
        assert!(
            ops.iter().any(|op| matches!(op, Op::InsertLane(..))),
            "the converted lanes were not packed: {ops:?}"
        );
        assert!(
            !ops.iter().any(|op| matches!(op, Op::Iota(_))),
            "no widened step reads the counter, so no ramp should be built"
        );
        scalar_and_vector_agree(&format!(
            "{kernel} fn main() -> i32 {{ let mut o: []f32 = alloc_f32(23); \
             k(o, 16777211, 16777234); let mut j: i64 = 0; \
             while j < 23 {{ print((o[j] * 4.0) as i64); j = j + 1; }} \
             free(o); return 0; }}"
        ));
    }

    /// The ops of one function at `-O2`, flattened.
    fn ops_of(src: &str, func: &str) -> Vec<Op> {
        let (program, mut interner) = optimized(src, 2);
        let f = program
            .function(interner.intern(func))
            .unwrap_or_else(|| panic!("no fn {func}"));
        f.blocks
            .iter()
            .flat_map(|b| b.insts.iter())
            .map(|i| i.op.clone())
            .collect()
    }

    /// `lut[q[i]]` is a gather: its index is data, so no lane offset is known. Every lane can still
    /// do its own scalar load (`Plan::LaneGep` + `Plan::Gather`), and the multiply and add around
    /// it widen. `@noalias` is what lets the gather go: it promises the stored slice is not `lut`,
    /// so `loop_info::dependence` can tell the gathered load apart from every store in the loop.
    /// `@unchecked` drops the bounds check a data-dependent index would otherwise keep.
    #[test]
    fn a_gather_beside_widened_work_is_replicated_lane_by_lane() {
        let src = "@unchecked fn k(@noalias mut o: []f32, lut: []f32, q: []i32, s: []f32, \
                   n: i64) { let mut i: i64 = 0; \
                   while i < n { o[i] = lut[q[i] as i64] * s[i] + 1.0; i = i + 1; } } \
                   fn main() -> i32 { return 0; }";
        let ops = ops_of(src, "k");
        let inserts = ops
            .iter()
            .filter(|op| matches!(op, Op::InsertLane(..)))
            .count();
        assert!(
            inserts >= 3,
            "four gathered f32 lanes take three lane inserts, found {inserts}"
        );
        let f32x4 = MirType::Vec(Box::new(MirType::F32), 4);
        assert!(
            ops.iter()
                .any(|op| matches!(op, Op::Load(_, t) if *t == f32x4)),
            "s[i] was not loaded as a vector"
        );
    }

    /// The same loop without the promise: `lut` and `o` may be one buffer, so the gather may read
    /// what an earlier lane stored, and the loop stays scalar.
    #[test]
    fn a_gather_from_a_slice_the_loop_may_store_is_declined() {
        let src = "@unchecked fn k(mut o: []f32, lut: []f32, q: []i32, s: []f32, n: i64) { \
                   let mut i: i64 = 0; \
                   while i < n { o[i] = lut[q[i] as i64] * s[i] + 1.0; i = i + 1; } } \
                   fn main() -> i32 { return 0; }";
        assert!(
            !is_widened(src, "k"),
            "a possibly-aliased gather must stay scalar"
        );
    }

    /// A local table nothing outside the function can name needs no promise: no pointer the loop
    /// stores through can reach a stack slot whose address never escaped. The gather loop sits in
    /// `main` itself so inlining cannot change what is compared, and it is the only loop there
    /// that can widen (the fill loop carries `v`, the print loop calls), so a lane insert in
    /// `main` is this loop. Every lane is printed, exactly (`lut * s + 1` is a multiple of 1/32).
    #[test]
    fn a_gather_from_a_local_table_agrees_with_the_scalar_loop() {
        let src = "@unchecked fn main() -> i32 { \
                   let lut: [f32; 8] = [0.5, -1.25, 2.0, 3.75, -4.5, 5.25, 6.0, -7.75]; \
                   let mut q: []i32 = alloc_i32(37); let mut s: []f32 = alloc_f32(37); \
                   let mut o: []f32 = alloc_f32(37); let mut j: i64 = 0; let mut v: f32 = -2.0; \
                   while j < 37 { q[j] = ((j * 5) % 8) as i32; s[j] = v; v = v + 0.375; \
                   j = j + 1; } \
                   j = 0; while j < 37 { o[j] = lut[q[j] as i64] * s[j] + 1.0; j = j + 1; } \
                   j = 0; while j < 37 { print((o[j] * 64.0) as i64); j = j + 1; } \
                   free(q); free(s); free(o); return 0; }";
        assert!(
            ops_of(src, "main")
                .iter()
                .any(|op| matches!(op, Op::InsertLane(..))),
            "the gather loop in main was not widened"
        );
        scalar_and_vector_agree(src);
    }

    /// The counter's *address* face is one scalar for the whole group. Mixed into a gathered index
    /// (`x[i + q[i]]`), replicating the add would hand every lane the group's first `i`, so a
    /// lane-varying step that reads an address computed from the counter is declined. Without the
    /// check this loop widens — the gather is promised apart from the store — and lane `k` reads
    /// `x[i + q[i + k]]` instead of `x[i + k + q[i + k]]`.
    #[test]
    fn a_counter_inside_a_gathered_index_is_declined() {
        let src = "@unchecked fn k(@noalias mut o: []f32, x: []f32, q: []i32, n: i64) { \
                   let mut i: i64 = 0; \
                   while i < n { o[i] = x[i + (q[i] as i64)] * 2.0 + 1.0; i = i + 1; } } \
                   fn main() -> i32 { return 0; }";
        assert!(
            !is_widened(src, "k"),
            "a gathered index built from the counter must stay scalar"
        );
    }

    // ---- scatters ------------------------------------------------------------------------------

    /// `o[q[i]] = x[i]*2 + 1` stores through a data-dependent index — a scatter. The arithmetic
    /// widens, and the store runs once per lane, lane 0 first (`Plan::Scatter`): one lane extract
    /// per lane after the first and four scalar `f32` stores in the vector body. `@noalias` is what
    /// lets it go: nothing else the loop touches can be `o`.
    #[test]
    fn a_scatter_beside_widened_work_is_replicated_lane_by_lane() {
        let src = "@unchecked fn k(@noalias mut o: []f32, x: []f32, q: []i32, n: i64) { \
                   let mut i: i64 = 0; \
                   while i < n { o[q[i] as i64] = x[i] * 2.0 + 1.0; i = i + 1; } } \
                   fn main() -> i32 { return 0; }";
        let (program, mut interner) = optimized(src, 2);
        let f = program.function(interner.intern("k")).expect("no fn k");
        let f32x4 = MirType::Vec(Box::new(MirType::F32), 4);
        let ops: Vec<&Op> = f
            .blocks
            .iter()
            .flat_map(|b| b.insts.iter())
            .map(|i| &i.op)
            .collect();
        assert!(
            ops.iter()
                .any(|op| matches!(op, Op::Load(_, t) if *t == f32x4)),
            "x[i] was not loaded as a vector: {ops:?}"
        );
        let extracts = ops
            .iter()
            .filter(|op| matches!(op, Op::ExtractLane(..)))
            .count();
        assert!(
            extracts >= 3,
            "four scattered lanes, found {extracts} extracts"
        );
        let most_f32_stores = f
            .blocks
            .iter()
            .map(|b| {
                b.insts
                    .iter()
                    .filter(|i| {
                        matches!(&i.op, Op::Store { value, .. }
                            if *f.value_type(*value) == MirType::F32)
                    })
                    .count()
            })
            .max()
            .unwrap_or(0);
        assert!(
            most_f32_stores >= 4,
            "the vector body stores each lane on its own: at most {most_f32_stores} in a block"
        );
    }

    /// Duplicate indices are the whole risk of a scatter: two lanes of one group, or two groups,
    /// writing one element. Lane-order stores leave the later iteration's value, exactly as the
    /// scalar loop does. `q[j] = (j / 2) % 8` repeats every index twice inside each group of four
    /// and again five groups later; the table is local to `main` (so nothing outside can name it
    /// and no promise is needed), and every slot is printed.
    #[test]
    fn a_scatter_with_repeated_indices_keeps_the_last_writer() {
        let src = "@unchecked fn main() -> i32 { \
                   let mut t: [f32; 8] = [-9.0; 8]; \
                   let mut q: []i32 = alloc_i32(37); let mut s: []f32 = alloc_f32(37); \
                   let mut j: i64 = 0; let mut v: f32 = -2.0; \
                   while j < 37 { q[j] = ((j / 2) % 8) as i32; s[j] = v; v = v + 0.375; \
                   j = j + 1; } \
                   j = 0; while j < 37 { t[q[j] as i64] = s[j] * 2.0 + 1.0; j = j + 1; } \
                   j = 0; while j < 8 { print((t[j] * 8.0) as i64); j = j + 1; } \
                   free(q); free(s); return 0; }";
        assert!(
            ops_of(src, "main")
                .iter()
                .any(|op| matches!(op, Op::ExtractLane(..))),
            "the scatter loop in main was not widened"
        );
        scalar_and_vector_agree(src);
    }

    /// A histogram, `h[q[i]] += x[i]`: the gather reads what the scatter writes, so lanes of one
    /// group may read a slot before an earlier lane's update lands. That is legal now — the update
    /// runs lane by lane ([`SerialRegion`]) — but this body has one multiply to widen beside three
    /// lane-by-lane steps, and [`plan_loop`]'s replication rule keeps it scalar
    /// (`a_histogram_with_widened_work_runs_lane_by_lane` is one that pays). Spelled as a `while`
    /// so the `scatter_add` recognizer does not take it first.
    #[test]
    fn a_histogram_scatter_is_declined() {
        let src = "@unchecked fn k(@noalias mut h: []f32, x: []f32, q: []i32, n: i64) { \
                   let mut i: i64 = 0; \
                   while i < n { let b: i64 = q[i] as i64; h[b] = h[b] + x[i] * 2.0; i = i + 1; } } \
                   fn main() -> i32 { return 0; }";
        assert!(!is_widened(src, "k"), "a read-modify-write scatter widened");
    }

    /// A scatter beside an affine store to its own object: which one lands last on a shared
    /// element depends on the iteration interleaving, which a vector group changes — unless both
    /// run lane by lane in iteration order ([`SerialRegion`]), which is what they do now.
    #[test]
    fn a_scatter_beside_a_store_to_its_object_runs_lane_by_lane() {
        let k = "@unchecked fn k(@noalias mut o: []f32, x: []f32, q: []i32, n: i64, d: i64) { \
                 if d > 0 { again(d - 1); return; } let mut i: i64 = 0; \
                 while i < n { o[q[i] as i64] = x[i] * 2.0 + 1.0; o[i] = x[i] * 0.5; \
                 i = i + 1; } } \
                 fn again(d: i64) { let mut t: []f32 = alloc_f32(1); let mut u: []f32 = alloc_f32(1); \
                 let mut q: []i32 = alloc_i32(1); k(u, t, q, 1, d); } ";
        assert!(
            is_widened(&format!("{k} fn main() -> i32 {{ return 0; }}"), "k"),
            "a scatter beside a store to its object was not widened"
        );
        scalar_and_vector_agree(&format!(
            "{k} fn main() -> i32 {{ let mut o: []f32 = alloc_f32(13); \
             let mut x: []f32 = alloc_f32(13); let mut q: []i32 = alloc_i32(13); \
             let mut j: i64 = 0; \
             while j < 13 {{ o[j] = -1.0; x[j] = (j as f32) * 0.25; \
               q[j] = ((j * 7 + 2) % 13) as i32; j = j + 1; }} \
             k(o, x, q, 13, 0); j = 0; while j < 13 {{ print(o[j]); j = j + 1; }} \
             return 0; }}"
        ));
    }

    /// Without the promise `o` may be `x` or `q`, so a lane's scatter may change what a later
    /// lane reads: every access of the body would have to run lane by lane ([`SerialRegion`]),
    /// which buys nothing, and the replication rule declines it.
    #[test]
    fn a_scatter_without_a_promise_is_declined() {
        let src = "@unchecked fn k(mut o: []f32, x: []f32, q: []i32, n: i64) { \
                   let mut i: i64 = 0; \
                   while i < n { o[q[i] as i64] = x[i] * 2.0 + 1.0; i = i + 1; } } \
                   fn main() -> i32 { return 0; }";
        assert!(!is_widened(src, "k"), "an unpromised scatter widened");
    }

    /// A predicated scatter needs a per-lane address in the predicated store forms, which take an
    /// affine one only. Residual: it stays scalar.
    #[test]
    fn a_scatter_in_an_arm_is_declined() {
        let src = "@unchecked fn k(@noalias mut o: []f32, x: []f32, q: []i32, n: i64) { \
                   let mut i: i64 = 0; \
                   while i < n { let v: f32 = x[i]; \
                   if v > 0.0 { o[q[i] as i64] = v * 2.0 + 1.0; } i = i + 1; } } \
                   fn main() -> i32 { return 0; }";
        assert!(
            !is_widened(src, "k"),
            "a scatter in an if-converted arm widened"
        );
    }

    /// A countdown's lane `k` is iteration `i - k`, so lane order is still iteration order and the
    /// last writer of a repeated index is still the scalar loop's. Kept out of line so the promise
    /// survives into the compared program: `k` and `k2` call each other, which the inliner never
    /// splices, and `k2` takes no slice from `k` — handing `o` to a call would let its storage
    /// escape and void the promise's facts. Every slot of `o` is printed at each trip `0..=19`,
    /// which moves the vector/epilogue split through every position.
    #[test]
    fn a_countdown_scatter_agrees() {
        let k = "@unchecked fn k(@noalias mut o: []f32, x: []f32, q: []i32, n: i64, d: i64) { \
                   if d > 0 { k2(d - 1); return; } \
                   let mut i: i64 = n; \
                   while i > 0 { i = i - 1; o[q[i] as i64] = x[i] * 2.0 + 1.0; } } \
                 fn k2(d: i64) { let mut e: []f32 = alloc_f32(1); let mut w: []f32 = alloc_f32(1); \
                   let mut z: []i32 = alloc_i32(1); k(e, w, z, 0, d); free(e); free(w); free(z); }";
        let shape = format!("{k} fn main() -> i32 {{ return 0; }}");
        assert!(
            is_widened(&shape, "k"),
            "the countdown scatter did not widen"
        );
        scalar_and_vector_agree(&format!(
            "{k} fn main() -> i32 {{ let mut x: []f32 = alloc_f32(19); \
             let mut q: []i32 = alloc_i32(19); let mut o: []f32 = alloc_f32(6); \
             let mut j: i64 = 0; \
             while j < 19 {{ x[j] = ((j * 5) % 7) as f32 - 2.5; q[j] = ((j * j) % 6) as i32; \
               j = j + 1; }} \
             let mut n: i64 = 0; \
             while n <= 19 {{ j = 0; while j < 6 {{ o[j] = -7.0; j = j + 1; }} \
               k(o, x, q, n, 0); j = 0; while j < 6 {{ print((o[j] * 4.0) as i32); j = j + 1; }} \
               n = n + 1; }} \
             free(x); free(q); free(o); return 0; }}"
        ));
    }

    /// Integer division and remainder, a per-lane shift amount and an `f64 -> i64` conversion have
    /// no packed form on this target. Each runs once per lane on that lane's own operands, its
    /// results are packed, and the arithmetic around it widens. No vector divide and no shift by a
    /// vector amount may reach a backend (Cranelift has neither, and the second is the CLIF type
    /// error that crashed ISLE). The inputs reach every edge the scalar forms define: a zero
    /// divisor, `i32::MIN / -1`, negative shift amounts (masked to the lane width) and conversions
    /// that saturate, then wrap on `+ 1`.
    #[test]
    fn steps_with_no_packed_form_are_replicated_and_agree_with_the_scalar_loop() {
        let kernels = "@unchecked fn div(x: []i32, y: []i32, mut o: []i32, n: i64) { \
                       let mut i: i64 = 0; \
                       while i < n { o[i] = x[i] * 3 + x[i] / y[i] - x[i] % y[i]; i = i + 1; } } \
                       @unchecked fn shl(x: []i32, y: []i32, mut o: []i32, n: i64) { \
                       let mut i: i64 = 0; \
                       while i < n { o[i] = ((x[i] << y[i]) + (x[i] >> y[i])) ^ x[i]; \
                       i = i + 1; } } \
                       @unchecked fn cvt(x: []f64, mut o: []i64, n: i64) { let mut i: i64 = 0; \
                       while i < n { o[i] = (x[i] * 2.5) as i64 + 1; i = i + 1; } }";
        let shape = format!("{kernels} fn main() -> i32 {{ return 0; }}");
        let (program, mut interner) = optimized(&shape, 2);
        for name in ["div", "shl", "cvt"] {
            let f = program
                .function(interner.intern(name))
                .unwrap_or_else(|| panic!("no fn {name}"));
            let mut packed = false;
            for inst in f.blocks.iter().flat_map(|b| b.insts.iter()) {
                match &inst.op {
                    Op::InsertLane(..) => packed = true,
                    Op::Bin(BinOp::SDiv | BinOp::UDiv | BinOp::SRem | BinOp::URem, a, _) => {
                        assert!(!f.value_type(*a).is_vector(), "{name}: a vector divide")
                    }
                    Op::Bin(BinOp::Shl | BinOp::LShr | BinOp::AShr, _, b) => {
                        assert!(
                            !f.value_type(*b).is_vector(),
                            "{name}: a shift by a vector amount"
                        )
                    }
                    Op::Cast(CastKind::FpToSi, v, _) => assert!(
                        !f.value_type(*v).is_vector(),
                        "{name}: a vector f64 -> i64 conversion"
                    ),
                    _ => {}
                }
            }
            assert!(packed, "{name}: no replicated result was packed");
        }
        scalar_and_vector_agree(&format!(
            "{kernels} @unchecked fn main() -> i32 {{ let n: i64 = 29; \
             let mut x: []i32 = alloc_i32(n); let mut y: []i32 = alloc_i32(n); \
             let mut o: []i32 = alloc_i32(n); let mut d: []f64 = alloc_f64(n); \
             let mut l: []i64 = alloc_i64(n); let mut j: i64 = 0; \
             while j < n {{ x[j] = ((j * 977 + 13) % 2001 - 1000) as i32; \
             y[j] = (j % 7 - 3) as i32; d[j] = ((j * 37 - 300) as f64) * 0.375; j = j + 1; }} \
             x[2] = (0 - 2147483648) as i32; \
             let big: f64 = 1000000000000.0 * 1000000000000.0; d[5] = big; d[6] = 0.0 - big; \
             div(x, y, o, n); j = 0; while j < n {{ print(o[j]); j = j + 1; }} \
             shl(x, y, o, n); j = 0; while j < n {{ print(o[j]); j = j + 1; }} \
             cvt(d, l, n); j = 0; while j < n {{ print(l[j]); j = j + 1; }} \
             free(x); free(y); free(o); free(d); free(l); return 0; }}"
        ));
    }

    /// A counter at its data's lane width: an `i32` counter over `f32` data widens, and the
    /// ramp — not a splat — is what makes lane `k` see `i + k`.
    #[test]
    fn a_counter_read_as_a_value_is_widened_through_a_lane_ramp() {
        let src = "fn k(mut o: []f32, n: i32) { let mut i: i32 = 0; \
                   while i < n { o[i] = (i as f32) * 0.25; i = i + 1; } } \
                   fn main() -> i32 { return 0; }";
        let vs = vector_values(src, "k");
        assert!(
            !vs.is_empty(),
            "an i32 counter over f32 data was not widened"
        );
        let (program, mut interner) = optimized(src, 2);
        let sym = interner.intern("k");
        let f = program.function(sym).expect("no fn k");
        let iota: Vec<&MirType> = f
            .blocks
            .iter()
            .flat_map(|b| b.insts.iter())
            .filter_map(|i| match &i.op {
                Op::Iota(t) => Some(t),
                _ => None,
            })
            .collect();
        assert_eq!(
            iota,
            vec![&vec_of(&MirType::I32, 4)],
            "expected exactly one <4 x i32> lane ramp"
        );
    }

    /// An integer compare against the counter has to become a **lane mask**, not one scalar bool:
    /// the `i == target` of a classification loss selects a different value in one lane of the
    /// group. Asserted on the widened `Cmp`'s operand type, which is what decides the mask width.
    #[test]
    fn a_compare_against_the_counter_becomes_a_lane_mask() {
        let src = "fn k(x: []f32, mut o: []f32, t: i32, n: i32) { let mut i: i32 = 0; \
                   while i < n { let q: f32 = if i == t { 8.0 } else { 2.0 }; \
                   o[i] = x[i] * q; i = i + 1; } } \
                   fn main() -> i32 { return 0; }";
        let (program, mut interner) = optimized(src, 2);
        let sym = interner.intern("k");
        let f = program.function(sym).expect("no fn k");
        let masks = f
            .blocks
            .iter()
            .flat_map(|b| b.insts.iter())
            .filter(|i| match &i.op {
                Op::Cmp(_, a, _) => f.value_type(*a) == &vec_of(&MirType::I32, 4),
                _ => false,
            })
            .count();
        assert_eq!(masks, 1, "expected one <4 x i32> lane-mask compare");
        scalar_and_vector_agree(
            "fn k(x: []f32, mut o: []f32, t: i32, n: i32) { let mut i: i32 = 0; \
             while i < n { let q: f32 = if i == t { 8.0 } else { 2.0 }; \
             o[i] = x[i] * q; i = i + 1; } } \
             fn main() -> i32 { let mut x: []f32 = alloc_f32(23); let mut o: []f32 = alloc_f32(23); \
             let mut j: i64 = 0; while j < 23 { x[j] = (j as f32) + 1.0; o[j] = -777.0; j = j + 1; } \
             k(x, o, 17, 23); \
             j = 0; while j < 23 { print(o[j] as i32); j = j + 1; } \
             free(o); free(x); return 0; }",
        );
    }

    /// The counter stored as *data* rather than used as an address: the store's pointer stays
    /// uniform and its value becomes the ramp.
    #[test]
    fn storing_the_counter_itself_is_widened() {
        let src = "fn k(mut o: []i32, n: i32) { let mut i: i32 = 0; \
                   while i < n { o[i] = i; i = i + 1; } } \
                   fn main() -> i32 { return 0; }";
        assert!(is_widened(src, "k"), "`o[i] = i` was not widened");
        scalar_and_vector_agree(
            "fn k(mut o: []i32, n: i32) { let mut i: i32 = 0; \
             while i < n { o[i] = i; i = i + 1; } } \
             fn main() -> i32 { let mut o: []i32 = alloc_i32(23); \
             let mut j: i64 = 0; while j < 23 { o[j] = -777; j = j + 1; } \
             k(o, 23); \
             j = 0; while j < 23 { print(o[j]); j = j + 1; } free(o); return 0; }",
        );
    }

    /// Only the counter itself gets a ramp. A value *derived* from it and then read as a value is
    /// still declined — see [`IvRamp`] for why the derived construction is not attempted.
    #[test]
    fn a_derived_index_read_as_a_value_is_declined() {
        let src = "fn k(mut o: []f32, n: i32) { let mut i: i32 = 0; \
                   while i < n { let j: i32 = i + 1; o[i] = (j as f32) * 0.25; i = i + 1; } } \
                   fn main() -> i32 { return 0; }";
        assert!(
            !is_widened(src, "k"),
            "a derived index used as a value must stay scalar"
        );
    }

    /// A stride-2 access with no partner at `2i + 1` is an interleave group with a gap: widening it
    /// would load (and here store) elements the scalar loop never touches, and the last group's
    /// window can run past the end of the buffer. It stays scalar.
    #[test]
    fn a_non_unit_stride_is_declined() {
        let src = "fn k(x: []f32, mut o: []f32, n: i64) { let mut i: i64 = 0; \
                   while i < n { o[i * 2] = x[i * 2]; i = i + 1; } } \
                   fn main() -> i32 { return 0; }";
        assert!(!is_widened(src, "k"), "stride 2 must stay scalar");
    }

    // ---- interleave groups ---------------------------------------------------------------------

    /// A complex multiply over interleaved `(re, im)` pairs: two complete stride-2 load groups and
    /// one complete stride-2 store group. Each load group is two contiguous `<4 x f32>` loads split
    /// by shuffles; the store group is two shuffles back and two contiguous stores. The two stores
    /// to `o` are one object at two different indices — the GCD test proves `2i` and `2i+1` never
    /// meet, which is what lets the loop through the dependence check at all.
    #[test]
    fn a_complex_multiply_widens_through_interleave_groups() {
        let kernel = "fn k(a: []f32, b: []f32, mut o: []f32, n: i64) { let mut i: i64 = 0; \
                      while i < n { let ar: f32 = a[2 * i]; let ai: f32 = a[2 * i + 1]; \
                      let br: f32 = b[2 * i]; let bi: f32 = b[2 * i + 1]; \
                      o[2 * i] = ar * br - ai * bi; o[2 * i + 1] = ar * bi + ai * br; \
                      i = i + 1; } }";
        let (program, mut interner) =
            optimized(&format!("{kernel} fn main() -> i32 {{ return 0; }}"), 2);
        let f = program.function(interner.intern("k")).expect("no fn k");
        let shuffles = f
            .blocks
            .iter()
            .flat_map(|b| b.insts.iter())
            .filter(|i| matches!(i.op, Op::Shuffle { .. }))
            .count();
        assert!(
            shuffles >= 6,
            "expected the de/interleave shuffles of three groups, got {shuffles}"
        );
        scalar_and_vector_agree(&format!(
            "{kernel} \
             fn main() -> i32 {{ let mut a: []f32 = alloc_f32(46); let mut b: []f32 = alloc_f32(46); \
             let mut o: []f32 = alloc_f32(46); \
             let mut j: i64 = 0; while j < 46 {{ a[j] = ((j % 7) as f32) - 3.0; \
             b[j] = ((j % 5) as f32) + 1.0; o[j] = -99.0; j = j + 1; }} \
             k(a, b, o, 23); \
             j = 0; while j < 46 {{ print(o[j] as i32); j = j + 1; }} \
             free(a); free(b); free(o); return 0; }}"
        ));
    }

    /// RGB pixels to gray: a stride-3 `u8` load group (three `<16 x u8>` loads per group, each
    /// channel gathered from all three by two shuffles) feeding `f32` arithmetic in four parts —
    /// interleave groups and mixed widths together.
    #[test]
    fn an_rgb_to_gray_conversion_widens() {
        let kernel = "fn k(rgb: []u8, mut g: []f32, n: i64) { let mut i: i64 = 0; \
                      while i < n { g[i] = 0.25 * (rgb[3 * i] as f32) + 0.5 * (rgb[3 * i + 1] as f32) \
                      + 0.25 * (rgb[3 * i + 2] as f32); i = i + 1; } }";
        assert!(
            is_widened(&format!("{kernel} fn main() -> i32 {{ return 0; }}"), "k"),
            "the RGB group did not widen"
        );
        scalar_and_vector_agree(&format!(
            "{kernel} \
             fn main() -> i32 {{ let mut rgb: []u8 = alloc_u8(111); let mut g: []f32 = alloc_f32(37); \
             let mut j: i64 = 0; while j < 111 {{ rgb[j] = ((j * 37 + 11) as u8); j = j + 1; }} \
             k(rgb, g, 37); \
             j = 0; while j < 37 {{ print((g[j] * 4.0) as i32); j = j + 1; }} \
             free(rgb); free(g); return 0; }}"
        ));
    }

    /// Interleaved RoPE in place: `x[2i]`, `x[2i+1]` read as one group and written back as another
    /// on the SAME object, beside unit-stride `cos`/`sin` streams. Each iteration reads its pair
    /// before writing it, and the load group is emitted at its first member, the store group at its
    /// last, so nothing is reordered against an access to the same element.
    #[test]
    fn an_in_place_interleaved_rope_widens() {
        let kernel = "fn k(mut x: []f32, c: []f32, s: []f32, n: i64) { let mut i: i64 = 0; \
                      while i < n { let x0: f32 = x[2 * i]; let x1: f32 = x[2 * i + 1]; \
                      x[2 * i] = x0 * c[i] - x1 * s[i]; x[2 * i + 1] = x0 * s[i] + x1 * c[i]; \
                      i = i + 1; } }";
        assert!(
            is_widened(&format!("{kernel} fn main() -> i32 {{ return 0; }}"), "k"),
            "the in-place RoPE did not widen"
        );
        scalar_and_vector_agree(&format!(
            "{kernel} \
             fn main() -> i32 {{ let mut x: []f32 = alloc_f32(42); let mut c: []f32 = alloc_f32(21); \
             let mut s: []f32 = alloc_f32(21); \
             let mut j: i64 = 0; while j < 42 {{ x[j] = ((j % 9) as f32) - 4.0; j = j + 1; }} \
             j = 0; while j < 21 {{ c[j] = ((j % 3) as f32) - 1.0; s[j] = ((j % 4) as f32) - 2.0; \
             j = j + 1; }} \
             k(x, c, s, 21); \
             j = 0; while j < 42 {{ print(x[j] as i32); j = j + 1; }} \
             free(x); free(c); free(s); return 0; }}"
        ));
    }

    /// A stride-4 group, summed: four `<4 x f32>` loads per group and three shuffles per member.
    #[test]
    fn a_stride_four_group_widens() {
        let kernel = "fn k(x: []f32, mut o: []f32, n: i64) { let mut i: i64 = 0; \
                      while i < n { o[i] = x[4 * i] + x[4 * i + 1] * 2.0 + x[4 * i + 2] * 3.0 \
                      + x[4 * i + 3] * 4.0; i = i + 1; } }";
        assert!(
            is_widened(&format!("{kernel} fn main() -> i32 {{ return 0; }}"), "k"),
            "the stride-4 group did not widen"
        );
        scalar_and_vector_agree(&format!(
            "{kernel} \
             fn main() -> i32 {{ let mut x: []f32 = alloc_f32(92); let mut o: []f32 = alloc_f32(23); \
             let mut j: i64 = 0; while j < 92 {{ x[j] = ((j % 11) as f32) - 5.0; j = j + 1; }} \
             k(x, o, 23); \
             j = 0; while j < 23 {{ print(o[j] as i32); j = j + 1; }} \
             free(x); free(o); return 0; }}"
        ));
    }

    /// A load moved to its group's first member must not cross a store to its own element: here the
    /// second read of `x[2i]` happens AFTER `x[2i]` is rewritten and must see the new value. Whether
    /// the optimizer forwards that store (and the group forms) or the read survives (and
    /// `group_order_verdict` declines the loop), `-O2` must print what `-O0` prints — compared
    /// directly, since either outcome is legal and only the answer is pinned.
    #[test]
    fn a_group_member_reading_its_own_store_keeps_the_scalar_answer() {
        let src = "fn k(mut x: []f32, n: i64) { let mut i: i64 = 0; \
                   while i < n { let x0: f32 = x[2 * i]; let x1: f32 = x[2 * i + 1]; \
                   x[2 * i] = x0 + x1; x[2 * i + 1] = x[2 * i] * x1; i = i + 1; } } \
                   fn main() -> i32 { let mut x: []f32 = alloc_f32(42); \
                   let mut j: i64 = 0; while j < 42 { x[j] = ((j % 5) as f32) - 2.0; j = j + 1; } \
                   k(x, 21); \
                   j = 0; while j < 42 { print(x[j] as i32); j = j + 1; } \
                   free(x); return 0; }";
        let mut outs = Vec::new();
        for level in [0u8, 2u8] {
            let (program, mut interner) = optimized(src, level);
            let main = interner.intern("main");
            let run = wukong_interp::run_with_output(&program, main, &interner)
                .unwrap_or_else(|e| panic!("-O{level}: {e}"));
            outs.push((run.0, String::from_utf8_lossy(&run.1).to_string()));
        }
        assert_eq!(outs[0], outs[1], "-O0 and -O2 disagree");
    }

    /// Two groups moving TOWARD each other. `x[2i]` is stored first, and its store group is emitted
    /// at its last member, `x[2i + 1] = ..`; `x[2i]` is read again last, and its load group is
    /// emitted at its first member, `x[2i + 1]`'s read. Neither move passes the other's starting
    /// point, yet the two swap: the re-read would see the element before the store. Every lane
    /// prints `5 + x[2i + 1]`; the swapped order would print `x[2i] + x[2i + 1]` instead.
    #[test]
    fn two_groups_moving_toward_each_other_keep_the_scalar_answer() {
        let src = "fn k(mut x: []f32, mut o: []f32, n: i64) { let mut i: i64 = 0; \
                   while i < n { x[2 * i] = 5.0; let t: f32 = x[2 * i + 1]; \
                   x[2 * i + 1] = t * 3.0; o[i] = x[2 * i] + t; i = i + 1; } } \
                   fn main() -> i32 { let mut x: []f32 = alloc_f32(42); \
                   let mut o: []f32 = alloc_f32(21); let mut j: i64 = 0; \
                   while j < 42 { x[j] = ((j % 7) as f32) * 10.0; j = j + 1; } \
                   k(x, o, 21); \
                   j = 0; while j < 42 { print(x[j] as i32); j = j + 1; } \
                   j = 0; while j < 21 { print(o[j] as i32); j = j + 1; } \
                   free(x); free(o); return 0; }";
        let mut outs = Vec::new();
        for level in [0u8, 2u8] {
            let (program, mut interner) = optimized(src, level);
            let main = interner.intern("main");
            let run = wukong_interp::run_with_output(&program, main, &interner)
                .unwrap_or_else(|e| panic!("-O{level}: {e}"));
            outs.push((run.0, String::from_utf8_lossy(&run.1).to_string()));
        }
        assert_eq!(outs[0], outs[1], "-O0 and -O2 disagree");
    }

    // ---- if-conversion -------------------------------------------------------------------------

    /// A diamond whose two arms store to the same address becomes one masked store.
    ///
    /// `@unchecked`, because the arms' slice accesses are *conditional*: `bce` may not read a
    /// parameter slice's length ahead of a check that does not run on every iteration, so with
    /// bounds checks the check-free copy keeps the arms' checks as extra exits and nothing reaches
    /// if-conversion. The opt-out gives the plain diamond this case is about.
    #[test]
    fn a_diamond_body_is_if_converted() {
        let src =
            "@unchecked fn k(x: []f32, y: []f32, mut o: []f32, n: i64) { let mut i: i64 = 0; \
                   while i < n { let xi: f32 = x[i]; \
                   if xi > 0.0 { o[i] = xi * 2.0 + y[i]; } else { o[i] = y[i] - xi; } \
                   i = i + 1; } } \
                   fn main() -> i32 { return 0; }";
        let vs = vector_values(src, "k");
        assert!(!vs.is_empty(), "a diamond body was not if-converted");
    }

    /// A triangle whose `then` arm is empty: the merge is a `select` on the join parameter.
    #[test]
    fn a_triangle_body_is_if_converted() {
        let src = "fn k(x: []f32, mut o: []f32, n: i64) { let mut i: i64 = 0; \
                   while i < n { let mut t: f32 = x[i]; if t < 0.0 { t = 0.0; } \
                   o[i] = t; i = i + 1; } } \
                   fn main() -> i32 { return 0; }";
        assert!(
            !vector_values(src, "k").is_empty(),
            "a triangle was not if-converted"
        );
    }

    /// Only one arm stores, so the flattened body must not write `o[i]` for lanes the scalar loop
    /// leaves untouched: each lane stores on its own, to `o[i + k]` where it took the arm and to a
    /// scratch slot where it did not. The sentinels the arm skips must survive.
    ///
    /// `o` is `@noalias` because only the arm subscripts it, so only the arm would re-read its data
    /// pointer from the fat pointer: an invariant load no other access covers, which
    /// `linearize_region` refuses to speculate. The promise makes the header dereferenceable, LICM
    /// hoists that read out of the loop, and the arm is left with the element store this case is
    /// about. (Inlined into `main`, the slice is a local and the same hoist needs no promise.)
    #[test]
    fn a_one_armed_store_is_predicated_and_skipped_elements_survive() {
        let kernel = "@unchecked fn k(x: []f32, @noalias mut o: []f32, n: i64) { \
                      let mut i: i64 = 0; \
                      while i < n { let v: f32 = x[i]; \
                      if v > 0.0 { o[i] = v * 2.0 + 1.0; } i = i + 1; } }";
        let shape = format!("{kernel} fn main() -> i32 {{ return 0; }}");
        assert!(
            is_widened(&shape, "k"),
            "a one-armed store was not predicated"
        );
        assert!(
            ops_of(&shape, "k")
                .iter()
                .any(|op| matches!(op, Op::Alloca { .. })),
            "a predicated store needs its scratch slot"
        );
        scalar_and_vector_agree(&format!(
            "{kernel} fn main() -> i32 {{ let mut x: []f32 = alloc_f32(23); \
             let mut o: []f32 = alloc_f32(23); let mut i: i64 = 0; \
             while i < 23 {{ x[i] = (((i * 5) % 7) as f32) - 3.0; o[i] = -7.0; i = i + 1; }} \
             k(x, o, 23); \
             i = 0; while i < 23 {{ print(o[i] as i32); i = i + 1; }} \
             free(o); free(x); return 0; }}"
        ));
    }

    /// Only one arm reads `w[i]`, and `w` is shorter than the trip: the scalar loop reads it only
    /// where `x[i] > 0`. Speculating the load would read past its end; predicating it reads the
    /// scratch slot for every lane that did not take the arm. `w` is `@noalias` for the reason the
    /// one-armed store's `o` is: only the arm reads it, data pointer included.
    #[test]
    fn an_arm_only_load_is_predicated_and_never_reads_past_its_buffer() {
        let kernel = "@unchecked fn k(x: []f32, @noalias w: []f32, mut o: []f32, n: i64) { \
                      let mut i: i64 = 0; \
                      while i < n { let xi: f32 = x[i]; \
                      if xi > 0.0 { o[i] = w[i] * xi; } else { o[i] = 1.0; } i = i + 1; } }";
        let shape = format!("{kernel} fn main() -> i32 {{ return 0; }}");
        assert!(
            is_widened(&shape, "k"),
            "an arm-only load was not predicated"
        );
        scalar_and_vector_agree(&format!(
            "{kernel} fn main() -> i32 {{ let mut x: []f32 = alloc_f32(23); \
             let mut w: []f32 = alloc_f32(10); let mut o: []f32 = alloc_f32(23); \
             let mut i: i64 = 0; \
             while i < 23 {{ x[i] = 9.5 - (i as f32); o[i] = 0.0; i = i + 1; }} \
             i = 0; while i < 10 {{ w[i] = (i as f32) + 0.5; i = i + 1; }} \
             k(x, w, o, 23); \
             i = 0; while i < 23 {{ print((o[i] * 4.0) as i32); i = i + 1; }} \
             free(o); free(w); free(x); return 0; }}"
        ));
    }

    /// Both arms store, to different places: there is no one masked store to merge them into.
    #[test]
    fn stores_to_different_places_in_both_arms_are_declined() {
        let src = "@unchecked fn k(x: []f32, mut o: []f32, mut p: []f32, n: i64) { \
                   let mut i: i64 = 0; \
                   while i < n { if x[i] > 0.0 { o[i] = 1.0; } else { p[i] = 2.0; } i = i + 1; } } \
                   fn main() -> i32 { return 0; }";
        assert!(
            !is_widened(src, "k"),
            "unpaired stores in both arms must stay scalar"
        );
    }

    #[test]
    fn if_converted_output_matches_the_scalar_loop() {
        let src = "fn k(x: []f32, y: []f32, mut o: []f32, n: i64) { let mut i: i64 = 0; \
                   while i < n { let xi: f32 = x[i]; \
                   if xi > 0.0 { o[i] = xi * 2.0 + y[i]; } else { o[i] = y[i] - xi; } \
                   i = i + 1; } } \
                   fn main() -> i32 { \
                     let mut x: []f32 = alloc_f32(23); let mut y: []f32 = alloc_f32(23); \
                     let mut o: []f32 = alloc_f32(23); let mut i: i64 = 0; \
                     while i < 23 { x[i] = (i as f32) * 0.5 - 5.0; y[i] = (i as f32) * 0.25 + 1.0; \
                       o[i] = 0.0; i = i + 1; } \
                     k(x, y, o, 23); \
                     i = 0; while i < 23 { print((o[i] * 100.0) as i32); i = i + 1; } \
                     free(o); free(y); free(x); return 0; }";
        scalar_and_vector_agree(src);
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
        assert!(
            is_widened(alone, "k"),
            "the reduction's body must be widened"
        );

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

    /// Integer addition is associative *including its wraparound*, so `W` lane partials folded at
    /// the end give the identical answer mod 2^32 — even when every partial overflows. A sum of
    /// `i * 400000009` over 23 elements wraps many times over; the vectorized answer must still be
    /// the scalar one to the bit.
    #[test]
    fn a_wrapping_integer_reduction_is_exact() {
        let alone = "fn k(x: []i32, n: i64) -> i32 { let mut s: i32 = 0; let mut i: i64 = 0; \
                     while i < n { s = s + x[i]; i = i + 1; } return s; } \
                     fn main() -> i32 { return 0; }";
        assert!(
            is_widened(alone, "k"),
            "an integer reduction must be widened"
        );

        let src = "fn k(x: []i32, n: i64) -> i32 { let mut s: i32 = 0; let mut i: i64 = 0; \
                   while i < n { s = s + x[i]; i = i + 1; } return s; } \
                   fn main() -> i32 { \
                     let mut x: []i32 = alloc_i32(23); \
                     let mut i: i64 = 0; \
                     while i < 23 { x[i] = (i as i32) * 400000009; i = i + 1; } \
                     print(k(x, 23)); free(x); return 0; }";
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

    // ---- symbolic row offsets in the alias guard ---------------------------------------------

    /// The inner loop of a row-wise 2-D access. `r*c` is a *value*, not a constant, so the range
    /// this loop touches on `x` is `[x + r*c, x + r*c + c)` — and a guard that could only express
    /// a constant offset had to decline the whole shape. That is the single most common inner loop
    /// in the corpus (every `for j in 0..C { .. a[i*C + j] .. }`), so declining it declined most of
    /// the ML code this pass exists for.
    #[test]
    fn an_inner_loop_over_a_row_is_widened() {
        let src = "fn k(x: []f32, mut o: []f32, r: i64, c: i64) { \
                   let b: i64 = r * c; \
                   let mut j: i64 = 0; while j < c { o[b + j] = x[b + j] * 2.0 + 1.0; j = j + 1; } } \
                   fn main() -> i32 { return 0; }";
        let vs = vector_values(src, "k");
        assert!(!vs.is_empty(), "a row-wise inner loop was not widened");
        assert!(
            vs.iter().all(|(l, n)| *l == MirType::F32 && *n == 4),
            "{vs:?}"
        );
    }

    /// …and the guard it gets is a real one: passing the same buffer for `x` and `o` at *different*
    /// rows still has to compute the scalar answer. Every lane is printed, so a partially-correct
    /// buffer cannot pass.
    #[test]
    fn a_row_wise_loop_aliasing_itself_computes_the_scalar_answer() {
        let src = "fn k(x: []f32, mut o: []f32, r: i64, c: i64) { \
                   let b: i64 = r * c; \
                   let mut j: i64 = 0; while j < c { o[b + j] = x[b + j] * 2.0 + 1.0; j = j + 1; } } \
                   fn main() -> i32 { \
                     let mut x: []f32 = alloc_f32(24); \
                     for i in 0..24 { x[i] = (i as f32) * 0.5; } \
                     k(x, x, 1, 11); \
                     for i in 0..24 { print((x[i] * 10.0) as i32); } \
                     free(x); return 0; }";
        scalar_and_vector_agree(src);
    }

    /// A base read at two different *constant* offsets keeps working. This is the shape the guard
    /// already handled (`lo`/`hi` bound the window), and the symbolic generalization must not
    /// disturb it.
    #[test]
    fn two_constant_offsets_on_one_base_are_still_guardable() {
        let src = "fn k(w: []f32, mut g: []f32, mut b: []f32, d: i64) { \
                   let mut i: i64 = 0; \
                   while i < d { g[i] = w[64 + i]; b[i] = w[128 + i]; i = i + 1; } } \
                   fn main() -> i32 { return 0; }";
        assert!(
            is_widened(src, "k"),
            "the two-constant-offset copy was not widened"
        );
    }

    /// Two READS on one base at offsets that differ *symbolically* are two separate ranges, not one
    /// unexpressible one — this is the LayerNorm γ/β gather, `g[i] = w[ga + i]; b[i] = w[gb + i];`.
    /// `lo`/`hi` bound the constant part only, so the two windows genuinely are not one range;
    /// the answer is to make the grouping key `(base, invariant part)` rather than `base`, which
    /// gives each read its own plain `[lo, hi)` and pairs each of them against the two stores.
    ///
    /// Splitting one base is only sound because the two groups can only ever both be *loads*: a
    /// same-object pair whose index expressions differ, where one writes, is reported
    /// `MemDep::Carried` by `loop_info::dependence` and never reaches `plan_guards` — and
    /// `plan_guards` re-checks it locally anyway. So this widens, and the two proofs that it may
    /// are below: the guard is built, and it still computes the scalar answer when the caller
    /// aliases the read base with an output at an overlapping offset.
    ///
    /// (An earlier version of this test asserted the *refusal*, which was this pass's behaviour
    /// before the grouping key changed. Refusing is safe but is a missed optimization, not a
    /// correctness requirement, so the assertion was inverted and backed with a differential run
    /// rather than deleted.)
    ///
    /// WRITING THE DIFFERENTIAL IS THE HARD PART, and the first attempt was **vacuous**: it called
    /// `k(w, w, b, 3, 9, 13)`, and at `-O2` inlining runs first, so the two parameters became one
    /// SSA value, `dependence` reported `Carried` for one object at two indices, the loop was
    /// declined, and the run compared two identical scalar programs. `WUKONG_VEC_TRACE=1` on that
    /// source prints no `WIDEN` at all. Three things are therefore needed at once:
    ///
    /// * the two pointers must be **distinct SSA values that alias at run time** — hence `pick`,
    ///   whose selector comes out of the heap so nothing can fold it;
    /// * the offsets must stay **symbolic** past inlining — hence reading `p`/`q`/`t` out of the
    ///   buffer too, since inlined integer literals fold into one group's constant window and turn
    ///   the case back into `two_constant_offsets_on_one_base_are_still_guardable`;
    /// * the store offset `t` must be **ahead of** the read offset `p`, so that overlap actually
    ///   changes the answer. `g[i] = w[p+i]` with `g == w` and `p > 0` reads ahead of the write and
    ///   the scalar and vector answers coincide, which would make the run undiscriminating.
    ///
    /// The full lane-by-lane version, with hand-derived expectations and both the overlapping and
    /// the disjoint call, is `tests/run/vectorized_split_base_guard.wk`.
    #[test]
    fn two_symbolic_offsets_on_one_base_are_guarded_separately() {
        let src = "fn k(w: []f32, mut g: []f32, mut b: []f32, p: i64, q: i64, d: i64) { \
                   let mut i: i64 = 0; \
                   while i < d { g[i] = w[p + i]; b[i] = w[q + i]; i = i + 1; } } \
                   fn main() -> i32 { return 0; }";
        assert!(
            is_widened(src, "k"),
            "two symbolic read offsets on one base are two ranges and must still be guardable"
        );
        // `w` read at two symbolic offsets, written at a third through an alias the optimizer
        // cannot see through. `t = 2`, `p = 0`, `d = 13`: the write window [w+2, w+15) overlaps the
        // read window [w+0, w+13) — and so do the two the guard actually tests, which cover the
        // vector part only, [w+2, w+14) and [w+0, w+12). So the guard must fail and the scalar loop
        // must do the work —
        // and because the write is AHEAD of the read, the answer it produces (0,1,0,1,… propagated
        // two apart) is one the vector loop cannot produce. Every lane is printed.
        let diff = "fn k(w: []f32, mut g: []f32, mut b: []f32, p: i64, q: i64, t: i64, d: i64) { \
                    let mut i: i64 = 0; \
                    while i < d { g[t + i] = w[p + i]; b[i] = w[q + i]; i = i + 1; } } \
                    fn pick(x: []f32, y: []f32, s: i64) -> []f32 { \
                      if s > 0 { return x; } return y; } \
                    fn fill(mut a: []f32, n: i64) { for i in 0..n { a[i] = (i as f32); } } \
                    fn main() -> i32 { \
                      let mut w: []f32 = alloc_f32(64); \
                      let mut z: []f32 = alloc_f32(64); \
                      let mut b: []f32 = alloc_f32(64); \
                      fill(w, 64); fill(z, 64); fill(b, 64); \
                      let s: i64 = w[1] as i64; \
                      let p: i64 = w[0] as i64; \
                      let q: i64 = w[40] as i64; \
                      let t: i64 = w[2] as i64; \
                      let d: i64 = w[13] as i64; \
                      let mut g: []f32 = pick(w, z, s); \
                      k(w, g, b, p, q, t, d); \
                      for i in 0..64 { print(w[i] as i32); } \
                      for i in 0..64 { print(b[i] as i32); } \
                      free(w); free(z); free(b); return 0; }";
        // `k` is inlined, so the widened loop lives in `main`. Two vector LOADS is the gather's own
        // signature: the three `fill` loops use their induction variable as a value and are
        // declined, and the two print loops contain a call, so nothing else in `main` can widen.
        let (program, mut interner) = optimized(diff, 2);
        let main_sym = interner.intern("main");
        let m = program.function(main_sym).expect("no fn main");
        let vec_loads = m
            .blocks
            .iter()
            .flat_map(|b| b.insts.iter())
            .filter(|i| matches!(&i.op, Op::Load(_, t) if t.is_vector()))
            .count();
        assert_eq!(
            vec_loads, 2,
            "the two-symbolic-offset gather did not widen inside the inlined `main`"
        );
        scalar_and_vector_agree(diff);
    }

    /// The case that is genuinely refused: a base *written* at one symbolic offset and touched at
    /// another. `lo`/`hi` say nothing about the distance between two different symbolic offsets, so
    /// the two windows cannot be collapsed and a store makes them a real dependence. (`loop_info`
    /// already declines this one step earlier, with "two accesses to one object at a non-zero
    /// distance"; this pins the outcome, whichever stage produces it.)
    #[test]
    fn one_base_written_at_two_symbolic_offsets_is_refused() {
        let src = "fn k(mut w: []f32, x: []f32, p: i64, q: i64, d: i64) { \
                   let mut i: i64 = 0; \
                   while i < d { w[p + i] = x[i]; w[q + i] = x[i] * 2.0; i = i + 1; } } \
                   fn main() -> i32 { return 0; }";
        assert!(
            !is_widened(src, "k"),
            "a store at one symbolic offset and a store at another must not be widened"
        );
    }

    /// Two row-wise loops side by side both widen. A single symbolic-offset loop proves the guard
    /// can be *built*; two prove that building one does not disturb the other's.
    #[test]
    fn two_row_wise_loops_are_both_widened() {
        let src = "fn k(x: []f32, mut o: []f32, mut p: []f32, r: i64, c: i64) { \
                   let b: i64 = r * c; \
                   let mut j: i64 = 0; while j < c { o[b + j] = x[b + j] * 2.0 + 1.0; j = j + 1; } \
                   let mut m: i64 = 0; while m < c { p[b + m] = x[b + m] * 3.0 + 1.0; m = m + 1; } } \
                   fn main() -> i32 { return 0; }";
        let (program, mut interner) = optimized(src, 2);
        let sym = interner.intern("k");
        let f = program.function(sym).expect("no fn k");
        let vec_loads = f
            .blocks
            .iter()
            .flat_map(|b| b.insts.iter())
            .filter(|i| matches!(&i.op, Op::Load(_, t) if t.is_vector()))
            .count();
        assert_eq!(
            vec_loads, 2,
            "both loops must be widened, one vector load each"
        );
    }

    // ---- a lane mask selecting between uniform arms -------------------------------------------

    /// `select(mask, a, b)` with both arms loop-invariant is still lane-varying, and `emit_widened`
    /// splats them exactly as it does a `Bin` operand. Requiring both arms to be vectors declined
    /// every table-driven transcendental the front end inlines — `exp` picks its 2^k coefficients
    /// with a chain of exactly such selects, so one of them made the whole loop unwidenable.
    ///
    /// The assertion is on the *shape*, not just on "something was widened": the widened body must
    /// contain a vector `Select` whose arms are `Splat`s, which is the case that used to be
    /// refused.
    #[test]
    fn a_lane_mask_selects_between_two_splatted_uniform_arms() {
        let src = "fn k(x: []f32, mut o: []f32, n: i64) { \
                   let mut i: i64 = 0; while i < n { o[i] = exp(x[i] * 0.25); i = i + 1; } } \
                   fn main() -> i32 { return 0; }";
        let (program, mut interner) = optimized(src, 2);
        let sym = interner.intern("k");
        let f = program.function(sym).expect("no fn k");
        let splats: FxHashSet<u32> = f
            .blocks
            .iter()
            .flat_map(|b| b.insts.iter())
            .filter(|i| matches!(i.op, Op::Splat(_)))
            .filter_map(|i| i.result.map(|r| r.0))
            .collect();
        let uniform_armed = f
            .blocks
            .iter()
            .flat_map(|b| b.insts.iter())
            .filter(|i| i.result.is_some_and(|r| f.value_type(r).is_vector()))
            .any(|i| match &i.op {
                Op::Select(_, a, b) => splats.contains(&a.0) && splats.contains(&b.0),
                _ => false,
            });
        assert!(
            uniform_armed,
            "no vector select over two splatted uniforms: the exp table was not widened"
        );
    }

    /// The transcendental case end to end: `exp` inlines to a masked coefficient table, a bitcast
    /// and integer arithmetic, and every lane must equal what the scalar loop produced.
    #[test]
    fn an_inlined_exp_loop_widens_and_agrees_with_the_scalar_loop() {
        let alone = "fn k(x: []f32, mut o: []f32, n: i64) { \
                     let mut i: i64 = 0; while i < n { o[i] = exp(x[i] * 0.25); i = i + 1; } } \
                     fn main() -> i32 { return 0; }";
        assert!(
            is_widened(alone, "k"),
            "an inlined exp loop was not widened"
        );
        let src = "fn k(x: []f32, mut o: []f32, n: i64) { \
                   let mut i: i64 = 0; while i < n { o[i] = exp(x[i] * 0.25); i = i + 1; } } \
                   fn main() -> i32 { \
                     let mut x: []f32 = alloc_f32(23); \
                     let mut o: []f32 = alloc_f32(23); \
                     for i in 0..23 { x[i] = (i as f32) - 11.0; } \
                     k(x, o, 23); \
                     for i in 0..23 { print((o[i] * 1000.0) as i32); } \
                     free(x); free(o); return 0; }";
        scalar_and_vector_agree(src);
    }

    /// The canonical scaled-elementwise kernel: one loop-invariant scalar load (`w[0]`, `w[1]`)
    /// beside a store. `invariant_loads_are_safe` used to veto the WHOLE loop for that — "an
    /// invariant data load in a loop that also writes memory" — although the only hazard is a
    /// store in the loop rewriting the address the load re-reads, a pairwise question this pass
    /// already answers at run time for its affine streams. Each such load is now a one-element
    /// `PointGuard` against every stored range, so the loop widens behind the guard.
    #[test]
    fn a_scaled_elementwise_loop_with_an_invariant_scalar_load_widens_under_a_guard() {
        let src = "fn k(x: []f32, w: [f32; 16], mut o: []f32, n: i64) { let mut i: i64 = 0; \
                   while i < n { o[i] = x[i] * w[0] + w[1]; i = i + 1; } } \
                   fn main() -> i32 { return 0; }";
        assert!(
            is_widened(src, "k"),
            "an elementwise loop with an invariant scalar load was not widened"
        );
    }

    /// The guard is real: when the caller makes the invariant `w[0]`/`w[1]` the very elements the
    /// loop stores to (`o` is a view of `w`'s storage, hidden behind `pick` so the optimizer cannot
    /// prove it), the point check fails, `n_vec` is 0, and the scalar loop computes the answer.
    ///
    /// Scalar semantics, by hand: `buf = [2, 3, 4, …, 17]`, `x = 1`, `n = 13`, so iteration `i`
    /// stores `buf[0] + buf[1]` as they stand: `buf[0] = 2+3 = 5`, `buf[1] = 5+3 = 8`, then
    /// `buf[2..13] = 5+8 = 13`, and `buf[13..16]` keep 15, 16, 17. A vector loop that read `w[0]`
    /// and `w[1]` once per group would store 5 in every lane of the first group instead.
    #[test]
    fn an_invariant_load_the_loop_overwrites_keeps_the_scalar_answer() {
        let src = "fn k(x: []f32, w: [f32; 16], mut o: []f32, n: i64) { let mut i: i64 = 0; \
                   while i < n { o[i] = x[i] * w[0] + w[1]; i = i + 1; } } \
                   fn pick(a: []f32, b: []f32, s: i64) -> []f32 { \
                     if s > 0 { return a; } return b; } \
                   fn main() -> i32 { \
                     let mut buf: [f32; 16] = [0.0; 16]; \
                     let mut j: i64 = 0; while j < 16 { buf[j] = (j as f32) + 2.0; j = j + 1; } \
                     let mut x: []f32 = alloc_f32(16); \
                     let mut z: []f32 = alloc_f32(16); \
                     let mut m: i64 = 0; while m < 16 { x[m] = 1.0; z[m] = 0.0; m = m + 1; } \
                     let s: i64 = x[3] as i64; \
                     let view: []f32 = buf; \
                     let mut o: []f32 = pick(view, z, s); \
                     k(x, buf, o, 13); \
                     for i in 0..16 { print(buf[i] as i32); } \
                     free(x); free(z); return 0; }";
        let expected: Vec<i32> = [5, 8]
            .into_iter()
            .chain(std::iter::repeat_n(13, 11))
            .chain([15, 16, 17])
            .collect();
        let (program, mut interner) = optimized(src, 2);
        let main = interner.intern("main");
        let (code, bytes) = wukong_interp::run_with_output(&program, main, &interner).unwrap();
        assert_eq!(code, 0);
        let printed: Vec<i32> = String::from_utf8_lossy(&bytes)
            .lines()
            .map(|l| l.trim().parse().expect("an integer line"))
            .collect();
        assert_eq!(printed, expected, "-O2 lost the scalar answer");
        scalar_and_vector_agree(src);
    }

    // ---- search loops (early exits) -----------------------------------------------------------

    /// The first index of an 18-element local table whose element exceeds `t`. The table is a
    /// stack array of this function, so every group may read it whole; 18 is not a multiple of
    /// the group, so the last two positions are found by the scalar tail.
    const TABLE_SEARCH: &str = "fn find(t: f32) -> i64 { \
                                let mut a: [f32; 18] = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, \
                                  4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5]; \
                                let mut i: i64 = 0; \
                                while i < 18 { if a[i] > t { break; } i = i + 1; } \
                                return i; }";

    #[test]
    fn a_search_over_a_local_table_widens_and_stops_where_the_scalar_loop_does() {
        let shape = format!("{TABLE_SEARCH} fn main() -> i32 {{ return 0; }}");
        assert!(
            is_widened(&shape, "find"),
            "a search over a local table did not widen"
        );
        // Every position 0..18 and two thresholds no element exceeds.
        scalar_and_vector_agree(&format!(
            "{TABLE_SEARCH} fn main() -> i32 {{ let mut q: i64 = 0; \
             while q < 20 {{ print(find((q as f32) * 0.5 - 0.25)); q = q + 1; }} \
             return 0; }}"
        ));
    }

    /// A search to a slice's own length (`n = x.len()`) may read every element below it; this one
    /// is `@unchecked`, so the loop has no bounds-check exits of its own.
    const SLICE_SEARCH: &str = "@unchecked fn first_neg(x: []i32) -> i64 { \
                                let n = x.len(); let mut i: i64 = 0; \
                                while i < n { if x[i] < 0 { break; } i = i + 1; } \
                                return i; }";

    #[test]
    fn a_search_to_a_slices_own_length_widens_and_agrees() {
        let shape = format!("{SLICE_SEARCH} fn main() -> i32 {{ return 0; }}");
        assert!(
            is_widened(&shape, "first_neg"),
            "a search to the slice's own length did not widen"
        );
        // The negative element at every position 0..19, then nowhere.
        scalar_and_vector_agree(&format!(
            "{SLICE_SEARCH} fn main() -> i32 {{ let mut s: []i32 = alloc_i32(19); \
             let mut j: i64 = 0; while j < 19 {{ s[j] = 1; j = j + 1; }} \
             let mut q: i64 = 0; \
             while q < 20 {{ if q < 19 {{ s[q] = -1; }} print(first_neg(s)); \
                             if q < 19 {{ s[q] = 1; }} q = q + 1; }} \
             free(s); return 0; }}"
        ));
    }

    /// A parameter array has no extent the loop can see: a group straddling its end could read
    /// past a shorter buffer where the scalar loop stops in time, so the search stays scalar.
    #[test]
    fn a_search_over_a_buffer_of_unknown_extent_is_declined() {
        let src = "fn find(x: [f32; 64], t: f32, n: i64) -> i64 { let mut i: i64 = 0; \
                   while i < n { if x[i] > t { break; } i = i + 1; } return i; } \
                   fn main() -> i32 { return 0; }";
        assert!(
            !is_widened(src, "find"),
            "a search over a buffer of unknown extent must not be widened"
        );
    }

    /// A `[T; N]` or statically-shaped tensor parameter carries its extent into MIR, so a search
    /// whose constant bound keeps every group load inside the parameter widens. The two functions
    /// call each other so neither is inlined into `main` (where the buffer would be a local and the
    /// stack-array proof would answer instead); the hit at 36 is found by the scalar tail.
    #[test]
    fn a_search_over_a_fixed_extent_parameter_widens_and_agrees() {
        let k = "fn find(x: [f32; 37], t: f32, d: i64) -> i64 { \
                   if d > 0 { return again(x, t, d - 1); } \
                   let mut i: i64 = 0; while i < 37 { if x[i] > t { break; } i = i + 1; } \
                   return i; } \
                 fn again(x: [f32; 37], t: f32, d: i64) -> i64 { return find(x, t, d); } \
                 fn tfind(x: Tensor[f32, 37], t: f32) -> i64 { \
                   let mut i: i64 = 0; while i < 36 { if x[i + 1] > t { break; } i = i + 1; } \
                   return i; }";
        let shape = format!("{k} fn main() -> i32 {{ return 0; }}");
        assert!(is_widened(&shape, "find"), "a [f32; 37] parameter search");
        assert!(
            is_widened(&shape, "tfind"),
            "a Tensor[f32, 37] parameter search"
        );
        scalar_and_vector_agree(&format!(
            "{k} fn main() -> i32 {{ let mut a: [f32; 37] = [0.0; 37]; let mut j: i64 = 0; \
             while j < 37 {{ a[j] = j as f32; j = j + 1; }} \
             let mut q: i64 = 0; \
             while q < 38 {{ print(find(a, (q as f32) - 0.5, 1)); q = q + 1; }} \
             return 0; }}"
        ));
    }

    /// A bound past the parameter's extent — or an offset that carries the last group past it —
    /// keeps the search scalar: lanes beyond the exiting one would read past the buffer.
    #[test]
    fn a_search_reaching_past_a_parameters_extent_is_declined() {
        for (bound, off) in [(38, 0), (37, 1)] {
            let src = format!(
                "fn find(x: [f32; 37], t: f32) -> i64 {{ let mut i: i64 = 0; \
                 while i < {bound} {{ if x[i + {off}] > t {{ break; }} i = i + 1; }} return i; }} \
                 fn main() -> i32 {{ return 0; }}"
            );
            assert!(
                !is_widened(&src, "find"),
                "bound {bound} offset {off} reads past a [f32; 37]"
            );
        }
    }

    /// The lane test and the exit: the widened body ends in a branch on the folded mask that
    /// leaves for the original loop.
    #[test]
    fn a_widened_search_folds_its_mask_with_lane_shuffles() {
        let shape = format!("{TABLE_SEARCH} fn main() -> i32 {{ return 0; }}");
        let ops = ops_of(&shape, "find");
        assert!(
            ops.iter().any(|op| matches!(op, Op::Shuffle { .. })),
            "the exit test did not fold the lane mask: {ops:?}"
        );
    }

    // ---- nested if-conversion -------------------------------------------------------------------

    /// The data for the region tests: 29 values cycling through every arm of a three-way chain,
    /// a NaN included (no ordered compare holds for it).
    const CHAIN_DATA: &str = "let mut x: []f32 = alloc_f32(29); let mut o: []f32 = alloc_f32(29); \
                              let mut j: i64 = 0; \
                              while j < 29 { x[j] = (((j * 7) % 9) as f32) * 0.5 - 1.75; \
                                o[j] = -7.0; j = j + 1; } \
                              x[11] = 0.0 / 0.0;";

    fn print_o(n: usize) -> String {
        format!("let mut q: i64 = 0; while q < {n} {{ print((o[q] * 8.0) as i32); q = q + 1; }}")
    }

    /// `if / else if / else`, every arm storing `o[i]`: one vector store of two nested selects.
    #[test]
    fn an_else_if_chain_of_stores_widens_into_nested_selects() {
        let k = "@unchecked fn k(x: []f32, mut o: []f32, n: i64) { let mut i: i64 = 0; \
                 while i < n { let v: f32 = x[i]; \
                   if v > 1.0 { o[i] = v * 2.0; } else if v > 0.0 { o[i] = v + 10.0; } \
                   else { o[i] = 0.0 - v; } i = i + 1; } }";
        let shape = format!("{k} fn main() -> i32 {{ return 0; }}");
        assert!(is_widened(&shape, "k"), "an else-if chain did not widen");
        let ops = ops_of(&shape, "k");
        assert!(
            !ops.iter().any(|op| matches!(op, Op::Alloca { .. })),
            "a complete chain needs no per-lane scratch: {ops:?}"
        );
        // Every trip 0..=29 (the counter is runtime after inlining), the buffer reset each time.
        scalar_and_vector_agree(&format!(
            "{k} fn main() -> i32 {{ {CHAIN_DATA} let mut n: i64 = 0; \
             while n <= 29 {{ j = 0; while j < 29 {{ o[j] = -7.0; j = j + 1; }} k(x, o, n); {} \
               n = n + 1; }} return 0; }}",
            print_o(29)
        ));
    }

    /// The same chain with no final `else`: lanes whose path stores nothing keep the old element,
    /// so the store goes lane by lane to the element or to the scratch slot.
    #[test]
    fn an_else_if_chain_without_an_else_predicates_its_store() {
        let k = "@unchecked fn k(x: []f32, mut o: []f32, n: i64) { let mut i: i64 = 0; \
                 while i < n { let v: f32 = x[i]; \
                   if v > 1.0 { o[i] = v * 2.0; } else if v > 0.0 { o[i] = v + 10.0; } \
                   i = i + 1; } }";
        let shape = format!("{k} fn main() -> i32 {{ return 0; }}");
        assert!(
            is_widened(&shape, "k"),
            "a partial else-if chain did not widen"
        );
        scalar_and_vector_agree(&format!(
            "{k} fn main() -> i32 {{ {CHAIN_DATA} k(x, o, 29); {} return 0; }}",
            print_o(29)
        ));
    }

    /// A value chosen by an `else if` chain and stored once: nested merges at two joins.
    #[test]
    fn an_else_if_value_chain_merges_at_each_join() {
        let k = "@unchecked fn k(x: []f32, mut o: []f32, n: i64) { let mut i: i64 = 0; \
                 while i < n { let v: f32 = x[i]; \
                   let r: f32 = if v > 1.0 { v * 2.0 } else if v > 0.0 { v + 10.0 } \
                     else { 0.0 - v }; \
                   o[i] = r; i = i + 1; } }";
        let shape = format!("{k} fn main() -> i32 {{ return 0; }}");
        assert!(
            is_widened(&shape, "k"),
            "an else-if value chain did not widen"
        );
        scalar_and_vector_agree(&format!(
            "{k} fn main() -> i32 {{ {CHAIN_DATA} k(x, o, 27); {} return 0; }}",
            print_o(29)
        ));
    }

    /// A store only one arm makes, over CHECKED `[]T` parameters: the arm re-reads `o`'s data
    /// pointer and checks `i < o.len()` before it stores. The vector loop drops the check under a
    /// guard that every index of its own iterations passes it, and a buffer shorter than the trip
    /// — whose short tail the arm never reaches — runs the scalar loop instead, with the same
    /// answer.
    #[test]
    fn a_checked_one_armed_store_over_a_slice_parameter_widens_under_a_length_guard() {
        let k = "fn k(x: []f32, mut o: []f32, n: i64) { let mut i: i64 = 0; \
                 while i < n { let v: f32 = x[i]; if v > 0.0 { o[i] = v * 2.0 + 1.0; } \
                   i = i + 1; } }";
        let shape = format!("{k} fn main() -> i32 {{ return 0; }}");
        assert!(
            is_widened(&shape, "k"),
            "a checked one-armed store did not widen"
        );
        scalar_and_vector_agree(&format!(
            "{k} fn main() -> i32 {{ {CHAIN_DATA} k(x, o, 29); {} \
             let mut s: []f32 = alloc_f32(9); let mut z: []f32 = alloc_f32(29); \
             j = 0; while j < 29 {{ z[j] = -1.0; if j == 3 || j == 8 {{ z[j] = 2.5; }} \
               j = j + 1; }} \
             j = 0; while j < 9 {{ s[j] = 0.0; j = j + 1; }} \
             k(z, s, 29); j = 0; while j < 9 {{ print((s[j] * 2.0) as i32); j = j + 1; }} \
             return 0; }}",
            print_o(29)
        ));
    }

    /// The `@unchecked` spelling over plain slice parameters: the arm still re-reads `o`'s data
    /// pointer, a load no access on every path covers. That read is the parameter's own fat
    /// pointer, so it is speculated (and hoisted into the guard) rather than refused.
    #[test]
    fn an_arm_rereading_a_slice_parameters_data_pointer_widens() {
        let k = "@unchecked fn k(x: []f32, mut o: []f32, n: i64) { let mut i: i64 = 0; \
                 while i < n { let v: f32 = x[i]; if v > 0.0 { o[i] = v * 2.0 + 1.0; } \
                   i = i + 1; } }";
        let shape = format!("{k} fn main() -> i32 {{ return 0; }}");
        assert!(is_widened(&shape, "k"), "a plain-slice one-armed store");
    }

    /// The same one-armed store through a RAW pointer's slice field: `(*p).w[i]` re-reads the data
    /// pointer at `p` itself (the field is at offset 0), and `p` may be `0 as *mut Layer` exactly
    /// when no lane takes the arm, since the scalar loop then never dereferences it. Speculating
    /// that read, and hoisting it into the guard, stopped `-O2` with "null pointer dereference"
    /// (native: SIGSEGV) where `-O0` returned; so did the checked spelling's length re-read. Only
    /// a slice parameter's or a stack slot's fat pointer is provably live, so both arms decline.
    /// `k` calls itself through `k2`, so the inliner leaves it out of line, where `p` is still a
    /// parameter.
    #[test]
    fn an_arm_rereading_a_raw_pointers_slice_field_is_not_speculated() {
        for attr in ["@unchecked ", ""] {
            let k = format!(
                "struct Layer {{ w: []f32, }} \
                 {attr}fn k(x: []f32, p: *mut Layer, n: i64, d: i64) {{ \
                   if d > 0 {{ k2(x, p, n, d - 1); return; }} let mut i: i64 = 0; \
                   while i < n {{ let v: f32 = x[i]; if v > 0.0 {{ (*p).w[i] = v; }} \
                     i = i + 1; }} }} \
                 fn k2(x: []f32, p: *mut Layer, n: i64, d: i64) {{ k(x, p, n, d); }}"
            );
            let shape = format!("{k} fn main() -> i32 {{ return 0; }}");
            assert!(
                !is_widened(&shape, "k"),
                "{attr}an arm's read through a raw pointer was speculated"
            );
            scalar_and_vector_agree(&format!(
                "{k} fn main() -> i32 {{ let mut x: []f32 = alloc_f32(16); \
                 let mut j: i64 = 0; while j < 16 {{ x[j] = -1.0; j = j + 1; }} \
                 k(x, 0 as *mut Layer, 16, 1); \
                 let mut wa: []f32 = alloc_f32(16); let mut l: Layer = Layer {{ w: wa }}; \
                 j = 0; while j < 16 {{ x[j] = (j % 3) as f32 - 0.5; wa[j] = 9.0; j = j + 1; }} \
                 k(x, &mut l, 16, 1); \
                 j = 0; while j < 16 {{ print((wa[j] * 2.0) as i32); j = j + 1; }} \
                 return 0; }}"
            ));
        }
    }

    /// A diamond inside one arm of another, the outer `else` storing nothing: three paths, two of
    /// them storing, and an arm-only load (`w[i]`) under a two-literal path predicate.
    #[test]
    fn a_diamond_nested_in_an_arm_widens_and_agrees() {
        let k = "@unchecked fn k(x: []f32, w: []f32, mut o: []f32, n: i64) { \
                 let mut i: i64 = 0; \
                 while i < n { let v: f32 = x[i]; \
                   if v > 0.0 { if v > 1.5 { o[i] = w[i] * v; } else { o[i] = v; } } \
                   i = i + 1; } }";
        let shape = format!("{k} fn main() -> i32 {{ return 0; }}");
        assert!(is_widened(&shape, "k"), "a nested diamond did not widen");
        scalar_and_vector_agree(&format!(
            "{k} fn main() -> i32 {{ {CHAIN_DATA} let mut w: []f32 = alloc_f32(29); \
             j = 0; while j < 29 {{ w[j] = (j as f32) * 0.25; j = j + 1; }} \
             k(x, w, o, 29); {} return 0; }}",
            print_o(29)
        ));
    }

    /// What the region flattener still refuses: stores to two different places, and a load on a
    /// path after a store it could read (sinking the store would move it past the load).
    #[test]
    fn nested_regions_it_cannot_order_are_declined() {
        let two = "@unchecked fn k(x: []f32, mut o: []f32, mut p: []f32, n: i64) { \
                   let mut i: i64 = 0; while i < n { let v: f32 = x[i]; \
                   if v > 1.0 { o[i] = v; } else if v > 0.0 { p[i] = v; } else { o[i] = 1.0; } \
                   i = i + 1; } } fn main() -> i32 { return 0; }";
        assert!(!is_widened(two, "k"), "stores to two places");
        let reread = "@unchecked fn k(x: []f32, mut o: []f32, n: i64) { let mut i: i64 = 0; \
                      while i < n { let v: f32 = x[i]; \
                      if v > 1.0 { o[i] = v; if v > 2.0 { o[i] = o[i] + 1.0; } } \
                      else if v > 0.0 { o[i] = 3.0; } i = i + 1; } } \
                      fn main() -> i32 { return 0; }";
        assert!(!is_widened(reread, "k"), "a load after a store on one path");
    }

    // ---- countdowns ------------------------------------------------------------------------------

    /// `while i > 0 { i = i - 1; .. }` and `while i >= 0 { ..; i = i - 1 }`: the group walks down,
    /// each contiguous access one load or store of the `W` elements ending at lane 0's, reversed
    /// with a shuffle so lane `k` is still iteration `k`. An `f32` sum keeps its serial fold in
    /// that (descending) order — the data makes any other order round differently — and every
    /// trip `0..=23` moves the vector/epilogue split through every position.
    #[test]
    fn a_countdown_widens_with_reversed_lanes_and_agrees() {
        let k = "@unchecked fn down(x: []f32, mut o: []f32, n: i64) { let mut i: i64 = n; \
                   while i > 0 { i = i - 1; o[i] = x[i] * 2.0 + 1.0; } } \
                 @unchecked fn down_ge(x: []f32, mut o: []f32, n: i64) { let mut i: i64 = n - 1; \
                   while i >= 0 { o[i] = x[i] - (i as f32); i = i - 1; } } \
                 @unchecked fn dsum(x: []f32, n: i64) -> f32 { let mut s: f32 = 0.0; \
                   let mut i: i64 = n; while i > 0 { i = i - 1; s = s + x[i]; } return s; } \
                 @unchecked fn isum(y: []i32, n: i64) -> i32 { let mut s: i32 = 7; \
                   let mut i: i64 = n; while i > 0 { i = i - 1; s = s + y[i] * 3; } return s; } \
                 @unchecked fn sel(x: []f32, mut o: []f32, n: i64) { let mut i: i64 = n; \
                   while i > 0 { i = i - 1; if x[i] > 0.0 { o[i] = x[i]; } else { o[i] = -1.0; } } }";
        let shape = format!("{k} fn main() -> i32 {{ return 0; }}");
        for f in ["down", "down_ge", "dsum", "isum", "sel"] {
            assert!(is_widened(&shape, f), "countdown {f} did not widen");
        }
        let ops = ops_of(&shape, "down");
        assert!(
            ops.iter()
                .any(|op| matches!(op, Op::Shuffle { mask, .. } if mask == &vec![3, 2, 1, 0])),
            "a countdown reverses its lanes: {ops:?}"
        );
        scalar_and_vector_agree(&format!(
            "{k} fn main() -> i32 {{ let mut x: []f32 = alloc_f32(23); \
             let mut y: []i32 = alloc_i32(23); let mut o: []f32 = alloc_f32(23); \
             let mut j: i64 = 0; \
             while j < 23 {{ x[j] = (((j * 5) % 7) as f32) - 2.5; y[j] = ((j * 3) % 11) as i32; \
               j = j + 1; }} \
             x[3] = 100000000.0; x[17] = -100000000.0; \
             let mut n: i64 = 0; \
             while n <= 23 {{ j = 0; while j < 23 {{ o[j] = -7.0; j = j + 1; }} \
               down(x, o, n); j = 0; while j < 23 {{ print((o[j] * 4.0) as i32); j = j + 1; }} \
               down_ge(x, o, n); j = 0; while j < 23 {{ print((o[j] * 4.0) as i32); j = j + 1; }} \
               sel(x, o, n); j = 0; while j < 23 {{ print((o[j] * 4.0) as i32); j = j + 1; }} \
               print(bits(dsum(x, n))); print(isum(y, n)); n = n + 1; }} \
             return 0; }}"
        ));
    }

    /// A countdown over bytes widened to floats: sixteen lanes, four `f32` parts, so the parts
    /// themselves must come out in descending order too.
    #[test]
    fn a_countdown_over_mixed_widths_orders_its_parts() {
        let k = "@unchecked fn k(q: []u8, mut o: []f32, n: i64) { let mut i: i64 = n; \
                 while i > 0 { i = i - 1; o[i] = (q[i] as f32) * 0.5; } }";
        let shape = format!("{k} fn main() -> i32 {{ return 0; }}");
        let vs = vector_values(&shape, "k");
        assert!(
            vs.contains(&(MirType::I8, 16)),
            "no 16-lane byte vector: {vs:?}"
        );
        scalar_and_vector_agree(&format!(
            "{k} fn main() -> i32 {{ let mut q: []u8 = alloc_u8(41); let mut o: []f32 = alloc_f32(41); \
             let mut j: i64 = 0; while j < 41 {{ q[j] = ((j * 37 + 11) % 251) as u8; j = j + 1; }} \
             let mut n: i64 = 30; \
             while n <= 41 {{ k(q, o, n); j = 0; \
               while j < 41 {{ print((o[j] * 2.0) as i32); j = j + 1; }} n = n + 1; }} \
             return 0; }}"
        ));
    }

    /// What a countdown keeps scalar: a search, an argmax (its fold reads iteration order as
    /// counter order), and `WUKONG_NO_REVERSE_VEC` is not set here, so the plain body widens.
    #[test]
    fn countdown_shapes_that_read_counter_order_are_declined() {
        let search = "fn find(t: f32) -> i64 { \
                      let mut a: [f32; 18] = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, \
                        4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5]; \
                      let mut i: i64 = 17; \
                      while i >= 0 { if a[i] < t { break; } i = i - 1; } return i; } \
                      fn main() -> i32 { return 0; }";
        assert!(!is_widened(search, "find"), "a countdown search");
        let argmax = "@unchecked fn k(x: []f32, n: i64) -> i64 { let mut m: f32 = -100.0; \
                      let mut at: i64 = -1; let mut i: i64 = n; \
                      while i > 0 { i = i - 1; if x[i] > m { m = x[i]; at = i; } } return at; } \
                      fn main() -> i32 { return 0; }";
        assert!(!is_widened(argmax, "k"), "a countdown argmax");
    }

    // ---- zero-trip and sub-group trips ---------------------------------------------------------

    /// Run `src` at `-O2` in the interpreter and return its printed lines.
    fn printed_at_o2(src: &str) -> Vec<String> {
        let (program, mut interner) = optimized(src, 2);
        let main = interner.intern("main");
        let (code, bytes) = wukong_interp::run_with_output(&program, main, &interner)
            .unwrap_or_else(|e| panic!("-O2: {e}"));
        assert_eq!(code, 0);
        String::from_utf8_lossy(&bytes)
            .lines()
            .map(|l| l.trim().to_string())
            .collect()
    }

    /// A trip below one vector group runs no vector iteration, so whatever the vector path does on
    /// the way out — fold identity partials into the seed, combine an argmax's `(value, index)`
    /// lanes, resolve a FindLastIV's sentinel — must hand back the scalar seed bit for bit. Each
    /// form below widens; each is run at trips `0 .. W-1` from seeds a wrong identity would change
    /// (`-0.0`, a payload NaN and `+inf`; `i32::MIN`, `i32::MAX` and `-1`) over data that also
    /// leaves the seed alone in the scalar loop, so every expected line is the seed itself.
    #[test]
    fn every_fold_form_returns_its_seed_below_one_group() {
        let kernels = "@unchecked fn fs(x: []f32, a: f32, n: i64) -> f32 { let mut acc: f32 = a; \
                         let mut i: i64 = 0; while i < n { acc = acc + x[i]; i = i + 1; } return acc; } \
                       @unchecked fn fm(x: []f32, a: f32, n: i64) -> f32 { let mut m: f32 = a; \
                         let mut i: i64 = 0; while i < n { m = max(x[i], m); i = i + 1; } return m; } \
                       @unchecked fn fa(x: []f32, a: f32, n: i64) -> i64 { let mut m: f32 = a; \
                         let mut at: i64 = -1; let mut i: i64 = 0; \
                         while i < n { if x[i] > m { m = x[i]; at = i; } i = i + 1; } \
                         if bits(m) != bits(a) { return -99; } return at; } \
                       @unchecked fn fl(x: []f32, n: i64) -> i64 { let mut last: i64 = -5; \
                         let mut i: i64 = 0; while i < n { if x[i] > 0.5 { last = i; } i = i + 1; } \
                         return last; } \
                       @unchecked fn is(x: []i32, a: i32, n: i64) -> i32 { let mut acc: i32 = a; \
                         let mut i: i64 = 0; while i < n { acc = acc + x[i]; i = i + 1; } return acc; } \
                       @unchecked fn im(x: []i32, a: i32, n: i64) -> i32 { let mut m: i32 = a; \
                         let mut i: i64 = 0; while i < n { m = max(x[i], m); i = i + 1; } return m; }";
        let shape = format!("{kernels} fn main() -> i32 {{ return 0; }}");
        for k in ["fs", "fm", "fa", "fl", "is", "im"] {
            assert!(is_widened(&shape, k), "{k} did not widen");
        }
        let src = format!(
            "{kernels} fn main() -> i32 {{ let mut z: []f32 = alloc_f32(4); \
             let mut w: []i32 = alloc_i32(4); let mut lo: []i32 = alloc_i32(4); \
             let mut fseed: []f32 = alloc_f32(3); let mut iseed: []i32 = alloc_i32(3); \
             let mut j: i64 = 0; \
             while j < 4 {{ z[j] = f32_from_bits(0x80000000 as u32); w[j] = 0; \
               lo[j] = -2147483647 - 1; j = j + 1; }} \
             fseed[0] = f32_from_bits(0x80000000 as u32); \
             fseed[1] = f32_from_bits(0x7fc00001 as u32); \
             fseed[2] = f32_from_bits(0x7f800000 as u32); \
             iseed[0] = -2147483647 - 1; iseed[1] = 2147483647; iseed[2] = -1; \
             let mut q: i64 = 0; \
             while q < 3 {{ let mut n: i64 = 0; \
               while n < 4 {{ print(bits(fs(z, fseed[q], n))); print(bits(fm(z, fseed[q], n))); \
                 print(fa(z, fseed[q], n)); print(fl(z, n)); \
                 print(is(w, iseed[q], n)); print(im(lo, iseed[q], n)); n = n + 1; }} \
               q = q + 1; }} \
             free(z); free(w); free(lo); free(fseed); free(iseed); return 0; }}"
        );
        let mut expected: Vec<String> = Vec::new();
        for (fbits, ival) in [
            (2147483648u32, i32::MIN),
            (2143289345, i32::MAX),
            (2139095040, -1),
        ] {
            for _ in 0..4 {
                for v in [
                    fbits.to_string(),
                    fbits.to_string(),
                    "-1".to_string(),
                    "-5".to_string(),
                    ival.to_string(),
                    ival.to_string(),
                ] {
                    expected.push(v);
                }
            }
        }
        assert_eq!(printed_at_o2(&src), expected, "-O2 lost a seed");
        scalar_and_vector_agree(&src);
    }

    /// A runtime trip is `end - start` in the counter's own type, and that wraps: an `i32` loop
    /// from `2147483600` to `-2147483600` runs no iteration, but the difference is `96`. Read as a
    /// count, it ran 24 groups of loads past the end of the buffer ("vector load out of bounds" in
    /// the interpreter at `-O2`, the untouched seed at `-O0`). The same for an unsigned `u64`
    /// counter from `2^64 - 16` up to `16`, whose difference is `32`. The guard now also requires
    /// `start < end` under the header's own signedness.
    #[test]
    fn a_runtime_trip_whose_difference_wraps_runs_no_vector_group() {
        let k = "@unchecked fn k(x: []i32, s: i32, e: i32) -> i32 { let mut acc: i32 = 0; \
                   let mut i: i32 = s; while i < e { acc = acc + x[i]; i = i + 1; } \
                   return acc * 1000 + i; } \
                 @unchecked fn u(x: []i32, s: u64, e: u64) -> i32 { let mut acc: i32 = 0; \
                   let mut i: u64 = s; while i < e { acc = acc + x[i]; i = i + 1; } \
                   return acc; }";
        let shape = format!("{k} fn main() -> i32 {{ return 0; }}");
        assert!(
            is_widened(&shape, "k"),
            "the i32-counter reduction did not widen"
        );
        assert!(
            is_widened(&shape, "u"),
            "the u64-counter reduction did not widen"
        );
        let src = format!(
            "{k} fn main() -> i32 {{ let mut x: []i32 = alloc_i32(16); \
             let mut j: i64 = 0; while j < 16 {{ x[j] = (j as i32) + 1; j = j + 1; }} \
             let mut lo: []i64 = alloc_i64(3); let mut hi: []i64 = alloc_i64(3); \
             lo[0] = 2147483600; hi[0] = -2147483600; \
             lo[1] = 3; hi[1] = 15; \
             lo[2] = -16; hi[2] = 16; \
             let mut q: i64 = 0; \
             while q < 2 {{ print(k(x, lo[q] as i32, hi[q] as i32)); q = q + 1; }} \
             q = 1; \
             while q < 3 {{ print(u(x, lo[q] as u64, hi[q] as u64)); q = q + 1; }} \
             free(x); free(lo); free(hi); return 0; }}"
        );
        // k(2147483600, -2147483600): no iteration, `acc = 0`, the counter stays at its start.
        // k(3, 15) and u(3, 15): 4 + 5 + .. + 15 = 114 (the counter leaves at 15). u(2^64-16, 16):
        // no iteration.
        assert_eq!(
            printed_at_o2(&src),
            vec![
                "2147483600".to_string(),
                "114015".to_string(),
                "114".to_string(),
                "0".to_string()
            ]
        );
        scalar_and_vector_agree(&src);
    }

    /// The guard runs on every entry, a zero trip included, and re-reads the fat pointer of every
    /// slice the loop indexes. Through a raw pointer that read is `(*p).w`'s data pointer at `p`
    /// itself, and the scalar loop over `(*p).w[i] = x[i]` with `n = 0` never makes it — so
    /// `k(x, 0 as *mut Layer, 0)` returned at `-O0` and stopped with "null pointer dereference" at
    /// `-O2` (native: SIGSEGV). The read now happens only when a vector group will run
    /// ([`GuardReads`]), and the loop still widens. `k` calls itself through `k2`, so it stays out
    /// of line with `p` a parameter.
    #[test]
    fn a_zero_trip_reads_no_pointer_the_scalar_loop_never_reads() {
        let k = "struct Layer { w: []f32, } \
                 @unchecked fn k(x: []f32, p: *mut Layer, n: i64, d: i64) { \
                   if d > 0 { k2(x, p, n, d - 1); return; } let mut i: i64 = 0; \
                   while i < n { (*p).w[i] = x[i] * 2.0; i = i + 1; } } \
                 fn k2(x: []f32, p: *mut Layer, n: i64, d: i64) { k(x, p, n, d); }";
        let shape = format!("{k} fn main() -> i32 {{ return 0; }}");
        assert!(
            is_widened(&shape, "k"),
            "the loop through a raw pointer no longer widens"
        );
        let src = format!(
            "{k} fn main() -> i32 {{ let mut x: []f32 = alloc_f32(11); \
             let mut wa: []f32 = alloc_f32(11); let mut j: i64 = 0; \
             while j < 11 {{ x[j] = (j as f32) + 0.5; wa[j] = -1.0; j = j + 1; }} \
             k(x, 0 as *mut Layer, 0, 1); print(7); \
             let mut l: Layer = Layer {{ w: wa }}; let lp: *mut Layer = &mut l; \
             k(x, lp, 11, 1); \
             j = 0; while j < 11 {{ print(wa[j] as i32); j = j + 1; }} \
             free(x); free(wa); return 0; }}"
        );
        // The null call returns; then lane j holds (j + 0.5) * 2 = 2j + 1, all 11 of them (two
        // groups and a scalar tail of three).
        let mut expected = vec!["7".to_string()];
        expected.extend((0..11).map(|j| (2 * j + 1).to_string()));
        assert_eq!(printed_at_o2(&src), expected);
        scalar_and_vector_agree(&src);
    }

    /// A search loop may leave on its first iteration, before it reaches the store an alias check
    /// is about, so its guard may not re-read a raw pointer's fat pointer even when a group would
    /// run: `s(x, 0 as *mut Layer, ..)` whose first element already passes the test never touches
    /// `p`. Such a search declines; the same search storing through a `[]f32` parameter widens.
    /// (Each runs to its slice's length read once before the loop, the bound a search over a
    /// `[]T` parameter needs.)
    #[test]
    fn a_search_loop_guard_reads_only_a_fat_pointer_that_cannot_fault() {
        let k = "struct Layer { w: []f32, } \
                 @unchecked fn s(x: []f32, p: *mut Layer, t: f32, d: i64) -> i64 { \
                   if d > 0 { return s2(x, p, t, d - 1); } let n: i64 = x.len(); \
                   let mut i: i64 = 0; \
                   while i < n { if x[i] > t { break; }; (*p).w[i] = x[i]; i = i + 1; } \
                   return i; } \
                 fn s2(x: []f32, p: *mut Layer, t: f32, d: i64) -> i64 { return s(x, p, t, d); } \
                 @unchecked fn o(x: []f32, mut w: []f32, t: f32, d: i64) -> i64 { \
                   if d > 0 { return o2(x, w, t, d - 1); } let n: i64 = x.len(); \
                   let mut i: i64 = 0; \
                   while i < n { if x[i] > t { break; } w[i] = x[i]; i = i + 1; } \
                   return i; } \
                 fn o2(x: []f32, mut w: []f32, t: f32, d: i64) -> i64 { return o(x, w, t, d); }";
        let shape = format!("{k} fn main() -> i32 {{ return 0; }}");
        assert!(
            !is_widened(&shape, "s"),
            "a search loop's guard re-reads a raw pointer"
        );
        assert!(
            is_widened(&shape, "o"),
            "the same search over a slice parameter no longer widens"
        );
        scalar_and_vector_agree(&format!(
            "{k} fn main() -> i32 {{ let mut x: []f32 = alloc_f32(16); \
             let mut wa: []f32 = alloc_f32(16); let mut j: i64 = 0; \
             while j < 16 {{ x[j] = j as f32; wa[j] = -1.0; j = j + 1; }} \
             print(s(x, 0 as *mut Layer, -5.0, 1)); \
             let mut l: Layer = Layer {{ w: wa }}; let lp: *mut Layer = &mut l; \
             print(s(x, lp, 9.5, 1)); print(o(x, wa, 12.5, 1)); \
             j = 0; while j < 16 {{ print(wa[j] as i32); j = j + 1; }} \
             free(x); free(wa); return 0; }}"
        ));
    }

    // ---- calls, checked gathers and lane-serial regions ------------------------------------------

    /// How many calls to `callee` the named function holds after `-O2`.
    fn calls_to(src: &str, func: &str, callee: &str) -> usize {
        let (program, mut interner) = optimized(src, 2);
        let sym = interner.intern(func);
        let callee = interner.intern(callee);
        program
            .function(sym)
            .unwrap_or_else(|| panic!("no fn {func}"))
            .blocks
            .iter()
            .flat_map(|b| &b.insts)
            .filter(|i| matches!(&i.op, Op::Call { func, .. } if *func == callee))
            .count()
    }

    /// A pure function the inliner never splices (it calls itself) and the effect summaries never
    /// call "always returns" (recursion): side-effect free, but it may not return.
    const TRI: &str = "fn tri(x: f32, d: i64) -> f32 { if d <= 0 { return x; } \
                       return tri(x * 0.5 + 1.0, d - 1) * 0.75 + x; } ";

    /// A call to a side-effect-free function used to decline the whole loop (`plan_body` refused
    /// every call, and `loop_info::dependence` called any call a carried dependence). Now each lane
    /// calls it on its own argument, in lane order, and everything around the call stays 4-wide.
    #[test]
    fn a_pure_call_is_replicated_lane_by_lane_beside_widened_work() {
        let k = format!(
            "{TRI} @unchecked fn k(x: []f32, y: []f32, mut o: []f32, n: i64, d: i64) {{ \
             let mut i: i64 = 0; \
             while i < n {{ let a: f32 = x[i] * 2.0 + y[i]; \
               o[i] = tri(a, d) * 0.5 - x[i] * y[i]; i = i + 1; }} }} "
        );
        let shape = format!("{k} fn main() -> i32 {{ return 0; }}");
        let vs = vector_values(&shape, "k");
        assert!(vs.contains(&(MirType::F32, 4)), "not widened: {vs:?}");
        assert!(
            calls_to(&shape, "k", "tri") >= 5,
            "four lane calls in the vector body and one in the epilogue"
        );
        scalar_and_vector_agree(&format!(
            "{k} fn main() -> i32 {{ let mut a: []f32 = alloc_f32(11); \
             let mut b: []f32 = alloc_f32(11); let mut c: []f32 = alloc_f32(11); \
             let mut j: i64 = 0; \
             while j < 11 {{ a[j] = (j as f32) * 0.25; b[j] = 1.0 - (j as f32) * 0.125; j = j + 1; }} \
             k(a, b, c, 11, 3); j = 0; while j < 11 {{ print(c[j]); j = j + 1; }} \
             free(a); free(b); free(c); return 0; }}"
        ));
    }

    /// The counter itself as a call argument: each lane passes its own `i + k`.
    #[test]
    fn a_call_reading_the_counter_takes_each_lanes_own_counter() {
        let k = "@unchecked fn h(i: i64, d: i64) -> i64 { if d <= 0 { return i * 3 + 1; } \
                 return h(i + 1, d - 1) - 1; } \
                 @unchecked fn k(x: []i64, mut o: []i64, n: i64) { let mut i: i64 = 0; \
                 while i < n { o[i] = (x[i] * 5 + 2) * h(i, 2) - x[i]; i = i + 1; } } ";
        let shape = format!("{k} fn main() -> i32 {{ return 0; }}");
        assert!(
            is_widened(&shape, "k"),
            "a call on the counter was not widened"
        );
        scalar_and_vector_agree(&format!(
            "{k} fn main() -> i32 {{ let mut a: []i64 = alloc_i64(9); \
             let mut b: []i64 = alloc_i64(9); let mut j: i64 = 0; \
             while j < 9 {{ a[j] = 7 - j; j = j + 1; }} \
             k(a, b, 9); j = 0; while j < 9 {{ print(b[j]); j = j + 1; }} \
             free(a); free(b); return 0; }}"
        ));
    }

    /// A call that writes memory (here through its slice argument) is no candidate: it stays an
    /// opaque effect, and the loop stays scalar.
    #[test]
    fn a_call_with_an_effect_keeps_the_loop_scalar() {
        let src = "@unchecked fn bump(mut c: []f32, v: f32, d: i64) -> f32 { \
                   if d <= 0 { c[0] = v; return v; } return bump(c, v * 0.5, d - 1) + 1.0; } \
                   @unchecked fn k(x: []f32, mut o: []f32, mut c: []f32, n: i64) { \
                   let mut i: i64 = 0; \
                   while i < n { o[i] = bump(c, x[i] * 3.0 + 1.0, 2) * 0.5 + x[i]; i = i + 1; } } \
                   fn main() -> i32 { return 0; }";
        assert!(!is_widened(src, "k"), "a call with a side effect widened");
    }

    /// Two calls that may each never return: lane 1's first call would run before lane 0's
    /// second, so which one stops the program could change. One such call widens.
    #[test]
    fn two_calls_that_may_not_return_are_declined() {
        let two = format!(
            "{TRI} @unchecked fn k(x: []f32, mut o: []f32, n: i64, d: i64) {{ \
             let mut i: i64 = 0; \
             while i < n {{ let a: f32 = x[i] * 2.0 + 1.0; let b: f32 = a * a - 0.5; \
               o[i] = tri(a, d) + tri(b, d) * a - b * 3.0; i = i + 1; }} }} \
             fn main() -> i32 {{ return 0; }}"
        );
        assert!(
            !is_widened(&two, "k"),
            "two calls that may not return widened"
        );
        let one = format!(
            "{TRI} @unchecked fn k(x: []f32, mut o: []f32, n: i64, d: i64) {{ \
             let mut i: i64 = 0; \
             while i < n {{ let a: f32 = x[i] * 2.0 + 1.0; let b: f32 = a * a - 0.5; \
               o[i] = tri(a, d) + b * a - b * 3.0; i = i + 1; }} }} \
             fn main() -> i32 {{ return 0; }}"
        );
        assert!(
            is_widened(&one, "k"),
            "one call that may not return was declined"
        );
    }

    /// A call that reads memory is a load from anywhere: beside a store it may read what an
    /// earlier lane's store would have changed, so the loop is declined; in a loop that stores
    /// nothing — a sum — memory never changes and the call is replicated.
    #[test]
    fn a_reading_call_widens_only_where_the_loop_stores_nothing() {
        let rd = "@unchecked fn rd(t: []f32, j: i64, d: i64) -> f32 { \
                  if d <= 0 { return t[j]; } return rd(t, j, d - 1) * 0.5; } ";
        let stores = format!(
            "{rd} @unchecked fn k(t: []f32, x: []f32, mut o: []f32, n: i64) {{ \
             let mut i: i64 = 0; \
             while i < n {{ o[i] = rd(t, i, 1) * x[i] + 1.0; i = i + 1; }} }} \
             fn main() -> i32 {{ return 0; }}"
        );
        assert!(
            !is_widened(&stores, "k"),
            "a reading call beside a store widened"
        );
        let sum = format!(
            "{rd} @unchecked fn k(t: []f32, x: []f32, n: i64) -> f32 {{ let mut s: f32 = 0.0; \
             let mut i: i64 = 0; \
             while i < n {{ s = s + (x[i] * 2.0 - 1.0) * rd(t, i, 1); i = i + 1; }} return s; }} "
        );
        assert!(
            is_widened(&format!("{sum} fn main() -> i32 {{ return 0; }}"), "k"),
            "a reading call in a store-free sum was declined"
        );
        scalar_and_vector_agree(&format!(
            "{sum} fn main() -> i32 {{ let mut a: []f32 = alloc_f32(10); \
             let mut b: []f32 = alloc_f32(10); let mut j: i64 = 0; \
             while j < 10 {{ a[j] = (j as f32) * 0.5; b[j] = 2.0 - (j as f32) * 0.25; j = j + 1; }} \
             print(k(a, b, 10)); free(a); free(b); return 0; }}"
        ));
    }

    /// A pure function the inliner never splices, taking a `bool`: what a compare in a widened
    /// body hands a per-lane call.
    const SEL: &str = "fn sel(b: bool, v: f32, d: i64) -> f32 { if d <= 0 { if b { return v; } \
                       return 0.0 - v; } return sel(b, v * 0.5, d - 1) + 1.0; } ";

    /// A compare widened beside a per-lane call is a lane mask — integer lanes at its operands'
    /// width, `1` in the interpreter and all-ones natively — while the call takes a `bool`. Each
    /// lane's argument must be that lane's truth (`lane != 0`), not the raw mask lane: an `i32`
    /// passed where the callee declares `i1` fails the MIR verifier, an internal compiler error on
    /// a program `-O0` runs.
    #[test]
    fn a_compare_reaches_a_per_lane_call_as_each_lanes_bool() {
        let k = format!(
            "{SEL} @unchecked fn k(x: []f32, @noalias mut o: []f32, n: i64) {{ \
             let mut i: i64 = 0; \
             while i < n {{ o[i] = sel(x[i] > 0.0, x[i] * 2.0, 2) * 0.5 + x[i]; i = i + 1; }} }} "
        );
        let shape = format!("{k} fn main() -> i32 {{ return 0; }}");
        assert!(
            is_widened(&shape, "k"),
            "the call on a compare was declined"
        );
        assert!(
            calls_to(&shape, "k", "sel") >= 5,
            "four lane calls in the vector body and one in the epilogue"
        );
        // x[j] = j - 4 for j < 9: negative, zero and positive lanes in each group, and a tail.
        scalar_and_vector_agree(&format!(
            "{k} fn main() -> i32 {{ let mut x: []f32 = alloc_f32(9); \
             let mut o: []f32 = alloc_f32(9); let mut j: i64 = 0; \
             while j < 9 {{ x[j] = (j as f32) - 4.0; j = j + 1; }} \
             k(x, o, 9); j = 0; while j < 9 {{ print(o[j]); j = j + 1; }} \
             free(x); free(o); return 0; }}"
        ));
    }

    /// The same inside a lane-serial region: the call reads the gathered bin (so it runs lane by
    /// lane with the region) and a compare the widened work before the region computed, which
    /// reaches each lane as that lane's `bool`.
    #[test]
    fn a_lane_serial_call_reads_an_earlier_compare_as_each_lanes_bool() {
        // `k` and `again` call each other, so the inliner never splices `k` into `main`, where its
        // `@noalias` promise would be lost and the loop declined.
        let k = format!(
            "{SEL} @unchecked fn k(@noalias mut h: []f32, q: []i32, x: []f32, y: []f32, n: i64, \
               z: i64) {{ \
             if z > 0 {{ again(z - 1); return; }} let mut i: i64 = 0; \
             while i < n {{ let c: f32 = x[i] * 2.0 + y[i] * 0.5 - 1.0; \
               let e: f32 = c * c - y[i]; let pos: bool = c > 4.0; let j: i64 = q[i] as i64; \
               let g: f32 = h[j]; h[j] = sel(pos, g + e, 1); i = i + 1; }} }} \
             fn again(z: i64) {{ let mut t: []f32 = alloc_f32(1); let mut u: []f32 = alloc_f32(1); \
               let mut q: []i32 = alloc_i32(1); k(u, q, t, t, 1, z); }} "
        );
        let shape = format!("{k} fn main() -> i32 {{ return 0; }}");
        assert!(
            is_widened(&shape, "k"),
            "the region around the call was declined"
        );
        assert!(
            calls_to(&shape, "k", "sel") >= 5,
            "four lane calls in the vector body and one in the epilogue"
        );
        // q[j] = j^2 mod 5 for j < 11: each group hits one bin twice (j = 2, 3 and j = 4, 6),
        // and a tail of three.
        scalar_and_vector_agree(&format!(
            "{k} fn main() -> i32 {{ let mut h: []f32 = alloc_f32(5); \
             let mut q: []i32 = alloc_i32(11); let mut x: []f32 = alloc_f32(11); \
             let mut y: []f32 = alloc_f32(11); let mut j: i64 = 0; \
             while j < 11 {{ q[j] = ((j * j) % 5) as i32; x[j] = (j as f32) * 0.5; \
               y[j] = 1.0 - (j as f32) * 0.25; j = j + 1; }} \
             k(h, q, x, y, 11, 0); j = 0; while j < 5 {{ print(h[j]); j = j + 1; }} \
             free(h); free(q); free(x); free(y); return 0; }}"
        ));
    }

    /// A lane-serial region is straight-line code, and only the region is. An if-converted step
    /// after it — a predicated store to another `@noalias` object — touches nothing the region
    /// does, so it widens beside the region. The same step between the gather and the scatter has
    /// no place in lane order, and declines the loop; that same store made unconditional (a step
    /// of its own) widens inside the region, so the `if` is what declines it.
    #[test]
    fn an_if_converted_step_widens_beside_a_lane_serial_region_but_not_inside_it() {
        // `k` and `again` call each other, so the inliner never splices `k` into `main`.
        let k = |body: &str| {
            format!(
                "@unchecked fn k(@noalias mut h: []f32, q: []i32, x: []f32, @noalias mut o: []f32, \
                   n: i64, z: i64) {{ \
                 if z > 0 {{ again(z - 1); return; }} let mut i: i64 = 0; \
                 while i < n {{ let j: i64 = q[i] as i64; let c: f32 = x[i] * 2.0 + 1.0; \
                   let e: f32 = c * c - x[i] * 0.25 + 3.0; {body} i = i + 1; }} }} \
                 fn again(z: i64) {{ let mut t: []f32 = alloc_f32(1); \
                   let mut u: []f32 = alloc_f32(1); let mut w: []f32 = alloc_f32(1); \
                   let mut q: []i32 = alloc_i32(1); k(u, q, t, w, 1, z); }} "
            )
        };
        let shape = |src: &str| format!("{src} fn main() -> i32 {{ return 0; }}");
        let after = k("let g: f32 = h[j]; h[j] = g + e; if c > 0.0 { o[i] = e * 0.5 - c; }");
        let inside = k("let g: f32 = h[j]; if c > 0.0 { o[i] = e * 0.5 - c; } h[j] = g + e;");
        let inside_plain = k("let g: f32 = h[j]; o[i] = e * 0.5 - c; h[j] = g + e;");
        assert!(
            is_widened(&shape(&after), "k"),
            "an if-converted store after the region was declined"
        );
        assert!(
            !is_widened(&shape(&inside), "k"),
            "an if-converted store inside the region widened"
        );
        assert!(
            is_widened(&shape(&inside_plain), "k"),
            "the unconditional control inside the region was declined"
        );
        // q[j] = j^2 mod 5 for j < 11 (bins hit twice inside a group), x[j] = j/2 - 2 so that
        // c = j - 3 (negative, zero and positive lanes), o[j] = -1 before the call so a lane the
        // store skips shows.
        scalar_and_vector_agree(&format!(
            "{after} fn main() -> i32 {{ let mut h: []f32 = alloc_f32(5); \
             let mut q: []i32 = alloc_i32(11); let mut x: []f32 = alloc_f32(11); \
             let mut o: []f32 = alloc_f32(11); let mut j: i64 = 0; \
             while j < 11 {{ q[j] = ((j * j) % 5) as i32; x[j] = (j as f32) * 0.5 - 2.0; \
               o[j] = -1.0; j = j + 1; }} \
             k(h, q, x, o, 11, 0); j = 0; while j < 5 {{ print(h[j]); j = j + 1; }} \
             j = 0; while j < 11 {{ print(o[j]); j = j + 1; }} \
             free(h); free(q); free(x); free(o); return 0; }}"
        ));
    }

    /// A checked gather, `o[i] = t[q[i]] * x[i] + 1.0` over `[]T` slices. `bce` versions the loop
    /// but cannot prove the gathered index, so its fast copy keeps that one check on every
    /// iteration — a second exit the vectorizer used to refuse. Now the check is a lane test: the
    /// group's four indices are compared at once, and a group with any failing lane goes to the
    /// scalar loop, which stops at the same iteration with the same message.
    #[test]
    fn a_checked_gather_widens_behind_a_lane_test() {
        let k = "fn k(t: []f32, q: []i32, x: []f32, @noalias mut o: []f32, n: i64, d: i64) { \
                 if d > 0 { again(d - 1); return; } \
                 let mut i: i64 = 0; \
                 while i < n { o[i] = t[q[i] as i64] * x[i] + 1.0; i = i + 1; } } \
                 fn again(d: i64) { let mut t: []f32 = alloc_f32(1); let mut u: []f32 = alloc_f32(1); \
                 let mut q: []i32 = alloc_i32(1); k(t, q, t, u, 1, d); } ";
        let shape = format!("{k} fn main() -> i32 {{ return 0; }}");
        let vs = vector_values(&shape, "k");
        assert!(
            vs.contains(&(MirType::F32, 4)),
            "a checked gather was not widened: {vs:?}"
        );
        scalar_and_vector_agree(&format!(
            "{k} fn main() -> i32 {{ let mut t: []f32 = alloc_f32(5); \
             let mut q: []i32 = alloc_i32(11); let mut x: []f32 = alloc_f32(11); \
             let mut o: []f32 = alloc_f32(11); let mut j: i64 = 0; \
             while j < 5 {{ t[j] = (j as f32) + 0.5; j = j + 1; }} j = 0; \
             while j < 11 {{ q[j] = ((j * 3) % 5) as i32; x[j] = j as f32; j = j + 1; }} \
             k(t, q, x, o, 11, 0); j = 0; while j < 11 {{ print(o[j]); j = j + 1; }} \
             return 0; }}"
        ));
    }

    /// A lane test runs `W` iterations ahead of the scalar loop, so nothing may store before it:
    /// a group that leaves is re-run by the scalar loop.
    #[test]
    fn a_store_before_a_lane_test_is_declined() {
        let src = "fn k(t: []f32, q: []i32, x: []f32, @noalias mut y: []f32, \
                   @noalias mut o: []f32, n: i64) { let mut i: i64 = 0; \
                   while i < n { y[i] = x[i] * 2.0 + 1.0; o[i] = t[q[i] as i64] * x[i]; \
                   i = i + 1; } } \
                   fn main() -> i32 { return 0; }";
        assert!(!is_widened(src, "k"), "a store before a lane test widened");
    }

    /// A read-modify-write histogram with real arithmetic feeding it: the bin update runs lane by
    /// lane in iteration order (a lane-serial region), so two lanes of one group hitting one bin
    /// add in the scalar loop's order; the value computation stays 4-wide.
    #[test]
    fn a_histogram_with_widened_work_runs_lane_by_lane() {
        let k = "@unchecked fn k(@noalias mut h: []f32, q: []i32, x: []f32, y: []f32, n: i64, \
                 d: i64) { if d > 0 { again(d - 1); return; } \
                 let mut i: i64 = 0; \
                 while i < n { let v: f32 = x[i] * 0.5 + y[i] * 0.25 - 1.0; \
                 h[q[i] as i64] = h[q[i] as i64] + v * v; i = i + 1; } } \
                 fn again(d: i64) { let mut t: []f32 = alloc_f32(1); let mut u: []f32 = alloc_f32(1); \
                 let mut q: []i32 = alloc_i32(1); k(u, q, t, t, 1, d); } ";
        let shape = format!("{k} fn main() -> i32 {{ return 0; }}");
        let vs = vector_values(&shape, "k");
        assert!(
            vs.contains(&(MirType::F32, 4)),
            "the histogram was not widened: {vs:?}"
        );
        scalar_and_vector_agree(&format!(
            "{k} fn main() -> i32 {{ let mut h: []f32 = alloc_f32(6); \
             let mut q: []i32 = alloc_i32(23); let mut x: []f32 = alloc_f32(23); \
             let mut y: []f32 = alloc_f32(23); let mut j: i64 = 0; \
             while j < 23 {{ q[j] = ((j * j) % 6) as i32; x[j] = (j as f32) * 0.5; \
               y[j] = 3.0 - (j as f32); j = j + 1; }} \
             k(h, q, x, y, 23, 0); j = 0; while j < 6 {{ print(h[j]); j = j + 1; }} \
             return 0; }}"
        ));
    }

    /// A gather from a buffer the loop also stores to: `t` may be `o`, so lane `k`'s gather must
    /// see every earlier lane's store and no later one. The store and the gather run lane by lane;
    /// the arithmetic before and after stays 4-wide. The differential passes `o` as `t`, with
    /// indices that point both behind and ahead of the store.
    #[test]
    fn a_gather_beside_a_store_it_may_read_runs_lane_by_lane() {
        let k = "@unchecked fn k(t: []f32, q: []i32, x: []f32, mut o: []f32, @noalias mut y: []f32, \
                 n: i64, d: i64) { if d > 0 { again(d - 1); return; } \
                 let mut i: i64 = 0; \
                 while i < n { let j: i64 = q[i] as i64; let a: f32 = x[i] * 2.0 + 1.0; \
                 let b: f32 = a * a - 3.0; o[i] = b; let g: f32 = t[j]; y[i] = g * 3.0 + a; \
                 i = i + 1; } } \
                 fn again(d: i64) { let mut t: []f32 = alloc_f32(1); let mut u: []f32 = alloc_f32(1); \
                 let mut q: []i32 = alloc_i32(1); k(t, q, t, t, u, 1, d); } ";
        let shape = format!("{k} fn main() -> i32 {{ return 0; }}");
        let vs = vector_values(&shape, "k");
        assert!(vs.contains(&(MirType::F32, 4)), "not widened: {vs:?}");
        scalar_and_vector_agree(&format!(
            "{k} fn main() -> i32 {{ let mut o: []f32 = alloc_f32(16); \
             let mut q: []i32 = alloc_i32(13); let mut x: []f32 = alloc_f32(13); \
             let mut y: []f32 = alloc_f32(13); let mut j: i64 = 0; \
             while j < 16 {{ o[j] = 100.0 + (j as f32); j = j + 1; }} j = 0; \
             while j < 13 {{ q[j] = ((j * 5 + 3) % 16) as i32; x[j] = (j as f32) * 0.25; j = j + 1; }} \
             k(o, q, x, o, y, 13, 0); j = 0; while j < 16 {{ print(o[j]); j = j + 1; }} \
             j = 0; while j < 13 {{ print(y[j]); j = j + 1; }} return 0; }}"
        ));
    }

    /// Two scatters into one object, repeated indices in a group and across groups: the last
    /// writer of each slot is the scalar loop's, because both stores run lane by lane in order.
    #[test]
    fn two_scatters_into_one_object_run_lane_by_lane() {
        let k = "@unchecked fn k(@noalias mut o: []i32, p: []i32, q: []i32, x: []i32, n: i64, \
                 d: i64) { if d > 0 { again(d - 1); return; } \
                 let mut i: i64 = 0; \
                 while i < n { let a: i32 = x[i] * 3 - 7; let b: i32 = a * a + x[i]; \
                 o[p[i] as i64] = a; o[q[i] as i64] = b - a * 2; i = i + 1; } } \
                 fn again(d: i64) { let mut t: []i32 = alloc_i32(1); let mut u: []i32 = alloc_i32(1); \
                 k(u, t, t, t, 1, d); } ";
        let shape = format!("{k} fn main() -> i32 {{ return 0; }}");
        let vs = vector_values(&shape, "k");
        assert!(vs.contains(&(MirType::I32, 4)), "not widened: {vs:?}");
        scalar_and_vector_agree(&format!(
            "{k} fn main() -> i32 {{ let mut o: []i32 = alloc_i32(7); \
             let mut p: []i32 = alloc_i32(19); let mut q: []i32 = alloc_i32(19); \
             let mut x: []i32 = alloc_i32(19); let mut j: i64 = 0; \
             while j < 19 {{ p[j] = ((j * 3) % 7) as i32; q[j] = ((j * 5 + 1) % 7) as i32; \
               x[j] = (j as i32) - 4; j = j + 1; }} \
             k(o, p, q, x, 19, 0); j = 0; while j < 7 {{ print(o[j]); j = j + 1; }} \
             return 0; }}"
        ));
    }

    // ---- outer-loop vectorization ----------------------------------------------------------------

    /// Was a nest of `func` widened across its rows? The mark is the vector nest's shape: a loop
    /// header carrying a vector parameter (each lane a row's own accumulator) that is then stored
    /// whole — the inner path never stores a carried vector, it folds one to a scalar first.
    fn outer_widened(src: &str, func: &str) -> bool {
        let (program, mut interner) = optimized(src, 2);
        let sym = interner.intern(func);
        let f = program
            .function(sym)
            .unwrap_or_else(|| panic!("no fn {func}"));
        let carried: FxHashSet<u32> = f
            .blocks
            .iter()
            .flat_map(|b| b.params.iter())
            .filter(|p| f.value_type(**p).is_vector())
            .map(|p| p.0)
            .collect();
        f.blocks
            .iter()
            .flat_map(|b| &b.insts)
            .any(|i| matches!(&i.op, Op::Store { value, .. } if carried.contains(&value.0)))
    }

    /// The out-of-line driver every outer-loop case shares: `k` and `again` call each other.
    const AGAIN: &str = "fn again(z: i64) { let mut t: []f32 = alloc_f32(1); \
                         let mut u: []f32 = alloc_f32(1); k(t, t, u, 1, 1, z); } ";

    /// A row-major matrix-vector product: per row a float sum the inner path can only fold
    /// serially. Widened across rows, each lane keeps its own row's sum in order.
    #[test]
    fn a_matrix_vector_product_widens_across_its_rows() {
        let k = format!(
            "@unchecked fn k(a: []f32, x: []f32, @noalias mut y: []f32, rows: i64, cols: i64, \
             z: i64) {{ if z > 0 {{ again(z - 1); return; }} let mut r: i64 = 0; \
             while r < rows {{ let mut s: f32 = 0.0; let mut c: i64 = 0; \
               while c < cols {{ s = s + a[r * cols + c] * x[c]; c = c + 1; }} \
               y[r] = s; r = r + 1; }} }} {AGAIN}"
        );
        assert!(
            outer_widened(&format!("{k} fn main() -> i32 {{ return 0; }}"), "k"),
            "the matrix-vector product was not widened across its rows"
        );
        scalar_and_vector_agree(&format!(
            "{k} fn main() -> i32 {{ let mut a: []f32 = alloc_f32(63); \
             let mut x: []f32 = alloc_f32(7); let mut y: []f32 = alloc_f32(9); \
             let mut j: i64 = 0; \
             while j < 63 {{ a[j] = ((j * 7) % 11) as f32 * 0.3 - 1.1; j = j + 1; }} j = 0; \
             while j < 7 {{ x[j] = (j as f32) * 0.7 - 2.0; j = j + 1; }} \
             k(a, x, y, 9, 7, 0); j = 0; while j < 9 {{ print(y[j]); j = j + 1; }} \
             return 0; }}"
        ));
    }

    /// A per-row recurrence, `h = h·t + a[c·rows + r]`: the inner path cannot widen it along `c`
    /// at all. Column-major, so the group's rows are contiguous and each step is one vector load.
    #[test]
    fn a_per_row_recurrence_widens_across_its_rows() {
        let k = format!(
            "@unchecked fn k(a: []f32, x: []f32, @noalias mut y: []f32, rows: i64, cols: i64, \
             z: i64) {{ if z > 0 {{ again(z - 1); return; }} let mut r: i64 = 0; \
             while r < rows {{ let mut h: f32 = x[0]; let mut c: i64 = 0; \
               while c < cols {{ h = h * 0.75 + a[c * rows + r]; c = c + 1; }} \
               y[r] = h; r = r + 1; }} }} {AGAIN}"
        );
        assert!(
            outer_widened(&format!("{k} fn main() -> i32 {{ return 0; }}"), "k"),
            "the per-row recurrence was not widened across its rows"
        );
        scalar_and_vector_agree(&format!(
            "{k} fn main() -> i32 {{ let mut a: []f32 = alloc_f32(70); \
             let mut x: []f32 = alloc_f32(1); let mut y: []f32 = alloc_f32(10); \
             let mut j: i64 = 0; \
             while j < 70 {{ a[j] = ((j * 3) % 17) as f32 * 0.37 - 2.5; j = j + 1; }} \
             x[0] = 0.125; k(a, x, y, 10, 7, 0); j = 0; \
             while j < 10 {{ print(y[j]); j = j + 1; }} return 0; }}"
        ));
    }

    /// The same recurrence over a row-major matrix: each inner step would gather one element per
    /// lane, and `unroll_jam` already runs four rows as four scalar chains without the packing, so
    /// the nest is left to it.
    #[test]
    fn a_recurrence_over_gathered_rows_is_left_to_unroll_and_jam() {
        let src = format!(
            "@unchecked fn k(a: []f32, x: []f32, @noalias mut y: []f32, rows: i64, cols: i64,              z: i64) {{ if z > 0 {{ again(z - 1); return; }} let mut r: i64 = 0;              while r < rows {{ let mut h: f32 = x[0]; let mut c: i64 = 0;                while c < cols {{ h = h * 0.75 + a[r * cols + c]; c = c + 1; }}                y[r] = h; r = r + 1; }} }} {AGAIN} fn main() -> i32 {{ return 0; }}"
        );
        assert!(
            !outer_widened(&src, "k"),
            "a recurrence over gathered rows took the outer path"
        );
    }

    /// The inner trip depends on the row (a triangle): the rows of a group would run different
    /// iterations, so the nest stays with the inner path.
    #[test]
    fn a_triangular_nest_is_not_widened_across_its_rows() {
        let src = format!(
            "@unchecked fn k(a: []f32, x: []f32, @noalias mut y: []f32, rows: i64, cols: i64, \
             z: i64) {{ if z > 0 {{ again(z - 1); return; }} let mut r: i64 = 0; \
             while r < rows {{ let mut s: f32 = 0.0; let mut c: i64 = 0; \
               while c < r {{ s = s + a[r * cols + c] * x[c]; c = c + 1; }} \
               y[r] = s; r = r + 1; }} }} {AGAIN} fn main() -> i32 {{ return 0; }}"
        );
        assert!(
            !outer_widened(&src, "k"),
            "a triangular nest widened across its rows"
        );
    }

    /// A store inside the inner loop would run row `r + 1`'s step `c` before row `r`'s step
    /// `c + 1`; declined.
    #[test]
    fn a_store_inside_the_inner_loop_keeps_the_nest_off_the_outer_path() {
        let src = format!(
            "@unchecked fn k(a: []f32, x: []f32, @noalias mut y: []f32, rows: i64, cols: i64, \
             z: i64) {{ if z > 0 {{ again(z - 1); return; }} let mut r: i64 = 0; \
             while r < rows {{ let mut h: f32 = 0.0; let mut c: i64 = 0; \
               while c < cols {{ h = h * 0.5 + a[r * cols + c]; y[r * cols + c] = h; \
               c = c + 1; }} r = r + 1; }} }} {AGAIN} fn main() -> i32 {{ return 0; }}"
        );
        assert!(
            !outer_widened(&src, "k"),
            "a nest storing in its inner loop widened"
        );
    }

    /// Without the promise `y` may be `a` or `x`: a lane's store could change what another lane
    /// reads.
    #[test]
    fn a_nest_whose_output_may_be_its_input_is_not_widened_across_its_rows() {
        let src = format!(
            "@unchecked fn k(a: []f32, x: []f32, mut y: []f32, rows: i64, cols: i64, z: i64) {{ \
             if z > 0 {{ again(z - 1); return; }} let mut r: i64 = 0; \
             while r < rows {{ let mut s: f32 = 0.0; let mut c: i64 = 0; \
               while c < cols {{ s = s + a[r * cols + c] * x[c]; c = c + 1; }} \
               y[r] = s; r = r + 1; }} }} {AGAIN} fn main() -> i32 {{ return 0; }}"
        );
        assert!(
            !outer_widened(&src, "k"),
            "an unpromised output widened across rows"
        );
    }

    /// An integer row sum is reassociable: the inner path keeps `W` lane partials and folds them
    /// once, which already runs in parallel — the nest is left to it.
    #[test]
    fn an_integer_row_sum_is_left_to_the_inner_path() {
        let src = "@unchecked fn k(a: []i32, @noalias mut y: []i32, rows: i64, cols: i64, \
                   z: i64) { if z > 0 { again(z - 1); return; } let mut r: i64 = 0; \
                   while r < rows { let mut s: i32 = 0; let mut c: i64 = 0; \
                     while c < cols { s = s + a[r * cols + c] * 3; c = c + 1; } \
                     y[r] = s; r = r + 1; } } \
                   fn again(z: i64) { let mut t: []i32 = alloc_i32(1); \
                   let mut u: []i32 = alloc_i32(1); k(t, u, 1, 1, z); } \
                   fn main() -> i32 { return 0; }";
        assert!(
            !outer_widened(src, "k"),
            "an integer row sum took the outer path"
        );
    }

    /// A half-precision row recurrence rounds to its grid inside every row's chain (`mir_build`
    /// places an `fptrunc` after each half operation). The inner path keeps such an accumulate
    /// scalar, and so does the outer one, for `f16` exactly as for `bf16`: `f16` became a vector
    /// lane type with `gap7/f16-vec`, after the outer path's `bf16` rule was written. The shape is
    /// the column-major recurrence that widens across its rows in `f32`.
    #[test]
    fn a_half_precision_row_recurrence_stays_off_the_outer_path() {
        for half in ["f16", "bf16"] {
            let src = format!(
                "@unchecked fn k(a: []{half}, x: []{half}, @noalias mut y: []{half}, rows: i64, \
                 cols: i64, z: i64) {{ if z > 0 {{ again(z - 1); return; }} \
                 let t: {half} = 0.75 as {half}; let mut r: i64 = 0; \
                 while r < rows {{ let mut h: {half} = x[0]; let mut c: i64 = 0; \
                   while c < cols {{ h = h * t + a[c * rows + r]; c = c + 1; }} \
                   y[r] = h; r = r + 1; }} }} \
                 fn again(z: i64) {{ let mut t: []{half} = alloc_{half}(1); \
                   let mut u: []{half} = alloc_{half}(1); k(t, t, u, 1, 1, z); }} \
                 fn main() -> i32 {{ return 0; }}"
            );
            assert!(
                !outer_widened(&src, "k"),
                "a {half} row recurrence took the outer path"
            );
        }
    }
}
