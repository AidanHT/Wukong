//! GPU **op-graph fusion planner + megakernel eligibility analysis** (Phase 3 general / Phase 8).
//!
//! This is **pure MIR analysis** — no `cudarc`, no device, no PTX. It runs on any box (the unit
//! tests need no GPU) and feeds two consumers:
//!   1. [`megakernel`](crate::megakernel) — decides whether an entry function can be compiled into a
//!      *cooperative single-block megakernel* (the SPMD whole-program kernel where recognized ops run
//!      cooperatively across threads), and inventories the recognized cooperative ops it contains.
//!   2. [`lower`](crate::lower) — `lower_call_mega` consumes the same
//!      [`classify_call`](crate::fusion::classify_call) classification to pick each recognized op's
//!      mega-mode form (block-wide tree / chunked-cooperative / `tid==0`-serial).
//!
//! ## The megakernel safety gate (the load-bearing analysis)
//! The cooperative megakernel runs the **whole program SPMD** (every thread executes the scalar glue;
//! recognized ops split work by `threadIdx`; `bar.sync` barriers bracket each cooperative region).
//! That is only correct when **every thread follows the same control flow** — otherwise threads
//! diverge and a `bar.sync` some threads never reach **deadlocks the block**. A thread's control flow
//! diverges from another's only if a branch condition depends on a *per-thread-varying* value, and
//! the only per-thread-varying inputs are memory reads racing with `tid==0`-guarded writes. So the
//! gate is: **no `CondBr` condition (transitively) depends on a `Load`** — i.e. the program's control
//! flow is *data-independent*. [`mem_tainted`] computes the dependence to a fixpoint (through block
//! params, so loops are covered); [`analyze`] rejects any program whose control flow is tainted.
//! Data-independent control flow is exactly the shape of dense tensor kernels (loop bounds are static
//! dims / counters, never loaded values), so the gate admits the ML corpus and rejects only genuinely
//! data-dependent programs (sorts, `gcd`, …) — which fall back to the correct single-thread path.

use std::collections::{HashMap, HashSet};

use wukong_mir::{Function, Op, Program, Terminator, ValueId};
use wukong_span::{Interner, Symbol};

/// The class of a recognized runtime call — the cooperative-op menu the megakernel can accelerate.
/// Mirrors `lower::rt_helper` / the `wukong_interp` `Accelerator` recognizers (one entry per
/// `wukong_*` symbol family). A call not in this menu is either a side effect or general code.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CoopKind {
    /// `wukong_sreduce_f32[_parallel](x, y, n, op) -> f32` — dot/ssd/sum/sumsq/max/min/maxabs plus
    /// sumabs(9)/absdiff(10). (The arg-reductions 7/8 have a `-> i64` ABI and are not in this menu.)
    Reduce,
    /// bf16/f16 reductions: `wukong_{dot,sum,reduce}_{bf16,f16}(...) -> f32`.
    ReduceLowp,
    /// `wukong_sgemm[_parallel]` — `C = A·B (+beta·C)`.
    Gemm,
    /// `wukong_sgemm_nt[_parallel]` — `C = A·Bᵀ`.
    GemmNt,
    /// `wukong_sgemm_nt_epi[_parallel]` — `C = act(A·Bᵀ[+bias])`, the fused FFN/Linear epilogue.
    GemmNtEpi,
    /// `wukong_i8gemm_nt[_parallel]` — int8 (u8×i8→i32) GEMM.
    I8GemmNt,
    /// `wukong_norm_f32[_parallel]` — softmax / LayerNorm / RMSNorm (no affine).
    Norm,
    /// `wukong_norm_affine_f32[_parallel]` — affine LayerNorm/RMSNorm (γ/β).
    NormAffine,
    /// `wukong_vmath_{f32,bf16,f16}(x, out, n, op)` — elementwise activation/transcendental.
    Vmath,
    /// `wukong_vmath2_f32(a, b, out, n, op)` — two-arg pow/atan2/hypot.
    Vmath2,
    /// `wukong_velem_f32[_parallel]` — streaming `act(a·x + b·y + c)` (residual add / fused bias).
    Velem,
    /// `wukong_axpby_{bf16,f16}` — low-precision `y := a·x + b·y`.
    Axpby,
    /// `wukong_parallel_for(n, func_addr F, ctx)` — the generic `@parallel` outliner.
    ParallelFor,
}

impl CoopKind {
    /// Does this op return an f32 scalar (vs. write through an out-pointer)?
    pub fn returns_f32(self) -> bool {
        matches!(self, CoopKind::Reduce | CoopKind::ReduceLowp)
    }
}

/// How a call site is classified for megakernel eligibility.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CallClass {
    /// A recognized cooperative runtime op.
    Coop(CoopKind),
    /// A deterministic side effect replayed from the device record buffer (`print`/`println`/`assert`).
    SideEffect,
    /// Anything else — another user function, or an unrecognized runtime symbol. Its presence in the
    /// entry makes the program ineligible for the (single-function) cooperative megakernel.
    Other,
}

/// Classify a resolved call-target name into its [`CallClass`]. The `_parallel` suffix selects the
/// multicore recognizer arm but the same cooperative kernel, so both map to the same [`CoopKind`].
///
/// **Every spelling `lower::rt_helper` maps must appear here**, or a program whose recognized op came
/// out of an `@parallel fn` is declared ineligible and falls back to the one-thread path — the
/// megakernel never accelerating the very shape the user asked to parallelize. `WUKONG_GPU_EXPLAIN=1`
/// now names the missing spelling in the decline note instead of leaving that fallback silent, but
/// the note is a diagnosis, not a fix: the entry still belongs here. (The converse
/// direction is safe: a name classified here that `rt_helper` does not know now returns an
/// `UNSUPPORTED:` decline rather than panicking.)
pub fn classify_call(name: &str) -> CallClass {
    use CoopKind::*;
    let kind = match name {
        "wukong_sreduce_f32" | "wukong_sreduce_f32_parallel" => Reduce,
        "wukong_dot_bf16"
        | "wukong_dot_bf16_parallel"
        | "wukong_dot_f16"
        | "wukong_dot_f16_parallel"
        | "wukong_sum_bf16"
        | "wukong_sum_bf16_parallel"
        | "wukong_sum_f16"
        | "wukong_sum_f16_parallel"
        | "wukong_reduce_bf16"
        | "wukong_reduce_bf16_parallel"
        | "wukong_reduce_f16"
        | "wukong_reduce_f16_parallel" => ReduceLowp,
        "wukong_sgemm" | "wukong_sgemm_parallel" => Gemm,
        "wukong_sgemm_nt" | "wukong_sgemm_nt_parallel" => GemmNt,
        "wukong_sgemm_nt_epi" | "wukong_sgemm_nt_epi_parallel" => GemmNtEpi,
        "wukong_i8gemm_nt" | "wukong_i8gemm_nt_parallel" => I8GemmNt,
        "wukong_norm_f32" | "wukong_norm_f32_parallel" => Norm,
        "wukong_norm_affine_f32" | "wukong_norm_affine_f32_parallel" => NormAffine,
        "wukong_vmath_f32"
        | "wukong_vmath_f32_parallel"
        | "wukong_vmath_bf16"
        | "wukong_vmath_f16" => Vmath,
        "wukong_vmath2_f32" => Vmath2,
        "wukong_velem_f32" | "wukong_velem_f32_parallel" => Velem,
        "wukong_axpby_bf16" | "wukong_axpby_f16" => Axpby,
        "wukong_parallel_for" => ParallelFor,
        "print" | "println" | "assert" => return CallClass::SideEffect,
        _ => return CallClass::Other,
    };
    CallClass::Coop(kind)
}

/// One recognized cooperative-op call site located in the entry function.
#[derive(Clone, Debug)]
pub struct CoopOp {
    /// Index of the block in `func.blocks` holding the call.
    pub block: usize,
    /// Index of the instruction within that block.
    pub inst: usize,
    /// The cooperative-op classification.
    pub kind: CoopKind,
    /// Resolved symbol name (e.g. `"wukong_sreduce_f32_parallel"`) — distinguishes precision arms.
    pub name: String,
    /// The call argument values, in order.
    pub args: Vec<ValueId>,
    /// The result value, if the op returns one (reductions).
    pub result: Option<ValueId>,
}

/// The plan produced by [`analyze`]: whether the entry is megakernel-eligible, why not if not, and
/// the inventory of cooperative ops a megakernel would accelerate.
#[derive(Clone, Debug)]
pub struct MegaPlan {
    /// True iff the entry can be compiled into the cooperative single-block megakernel.
    pub eligible: bool,
    /// Human-readable reason the program is ineligible (empty when eligible) — for honest reporting.
    ///
    /// **It reaches a user through `WUKONG_GPU_EXPLAIN=1`** (`megakernel::EXPLAIN_ENV`), which makes
    /// `megakernel::decline` print it beside the fallback it took. It has to be non-empty for every
    /// ineligible plan or the note says nothing, which is what
    /// `megakernel::tests::the_megakernel_decline_reason_reaches_the_note` pins over three arms of
    /// [`analyze`]. Until that knob existed the field really was written and never read outside a
    /// `WUKONG_GPU_DUMP_PTX` dump — a silent decline is this project's signature failure.
    pub reason: String,
    /// The recognized cooperative ops in entry order (program order within the block sweep).
    pub coop_ops: Vec<CoopOp>,
}

impl MegaPlan {
    fn ineligible(reason: impl Into<String>) -> Self {
        MegaPlan {
            eligible: false,
            reason: reason.into(),
            coop_ops: Vec::new(),
        }
    }
}

/// The operand values an [`Op`] *reads* (used by the taint fixpoint). Address/value operands of a
/// `Store` are included even though `Store` has no result, so a callers iterating "reads" see them;
/// the taint pass only consults this for value-producing ops.
fn op_operands(op: &Op) -> Vec<ValueId> {
    match op {
        Op::ConstInt(..)
        | Op::ConstFloat(..)
        | Op::Alloca { .. }
        | Op::Barrier
        | Op::FuncAddr(_)
        | Op::GlobalAddr(_)
        | Op::Iota(_) => Vec::new(),
        Op::Bin(_, a, b) | Op::Cmp(_, a, b) => vec![*a, *b],
        Op::Neg(a)
        | Op::Not(a)
        | Op::Sqrt(a)
        | Op::Splat(a)
        | Op::Round(_, a)
        | Op::ExtractLane(a, _) => vec![*a],
        Op::Cast(_, a, _) => vec![*a],
        Op::Select(c, a, b) => vec![*c, *a, *b],
        Op::Fma(a, b, c) => vec![*a, *b, *c],
        Op::Load(p, _) => vec![*p],
        Op::Store { ptr, value } => vec![*ptr, *value],
        Op::Gep { ptr, index, .. } => vec![*ptr, *index],
        Op::Call { args, .. } => args.clone(),
        // A CPU raw-AVX2 microkernel call reads its buffer, scalar, and length operands. `analyze`
        // does not inspect this op at all (it only classifies `Op::Call`), so such a program can be
        // reported *eligible* here and is instead declined during lowering — `lower.rs` emits
        // `UNSUPPORTED: VecKernelCall`, which `megakernel::try_run` turns into `Ok(None)` (fall back
        // to the single-thread path). Reporting its reads keeps the taint fixpoint total.
        Op::VecKernelCall {
            ptrs, scalars, n, ..
        } => vec![*ptrs, *scalars, *n],
    }
}

/// Compute the set of `ValueId`s (by raw index) that transitively depend on a memory read — see the
/// module docs. A value is tainted iff it is the result of a `Load`, the result of a `Call` (every
/// recognized op reads its input buffers; a user call is conservatively assumed to as well), or any
/// op with a tainted operand; block params are tainted iff any incoming branch passes a tainted arg.
/// Iterated to a fixpoint so loop-carried params converge.
pub fn mem_tainted(func: &Function) -> HashSet<u32> {
    let mut tainted: HashSet<u32> = HashSet::new();

    // Reverse edge map: target block -> list of (incoming arg vectors), so block params can pull
    // taint from every predecessor edge.
    let mut incoming: HashMap<u32, Vec<Vec<ValueId>>> = HashMap::new();
    for b in &func.blocks {
        match &b.term {
            Terminator::Br { target, args } => {
                incoming.entry(target.0).or_default().push(args.clone());
            }
            Terminator::CondBr {
                then_blk,
                then_args,
                else_blk,
                else_args,
                ..
            } => {
                incoming
                    .entry(then_blk.0)
                    .or_default()
                    .push(then_args.clone());
                incoming
                    .entry(else_blk.0)
                    .or_default()
                    .push(else_args.clone());
            }
            Terminator::Ret(_) | Terminator::Unreachable => {}
        }
    }

    // Seed: Load and Call results are memory-derived roots.
    for b in &func.blocks {
        for inst in &b.insts {
            if let Some(r) = inst.result {
                if matches!(inst.op, Op::Load(..) | Op::Call { .. }) {
                    tainted.insert(r.0);
                }
            }
        }
    }

    // Fixpoint: propagate through ops and block params until stable.
    loop {
        let mut changed = false;

        for b in &func.blocks {
            // Block params: tainted if any predecessor edge passes a tainted arg in that slot.
            if let Some(edges) = incoming.get(&b.id.0) {
                for (i, p) in b.params.iter().enumerate() {
                    if tainted.contains(&p.0) {
                        continue;
                    }
                    if edges
                        .iter()
                        .any(|args| args.get(i).is_some_and(|a| tainted.contains(&a.0)))
                    {
                        tainted.insert(p.0);
                        changed = true;
                    }
                }
            }
            // Value-producing insts: tainted if any operand is tainted (Load/Call already seeded).
            for inst in &b.insts {
                if let Some(r) = inst.result {
                    if tainted.contains(&r.0) {
                        continue;
                    }
                    if op_operands(&inst.op).iter().any(|v| tainted.contains(&v.0)) {
                        tainted.insert(r.0);
                        changed = true;
                    }
                }
            }
        }

        if !changed {
            break;
        }
    }
    tainted
}

/// Does any `CondBr` in `func` branch on a memory-derived (tainted) condition? If so the program's
/// control flow is data-dependent and unsafe for the SPMD cooperative megakernel.
fn has_data_dependent_control_flow(func: &Function, tainted: &HashSet<u32>) -> bool {
    func.blocks.iter().any(|b| match &b.term {
        Terminator::CondBr { cond, .. } => tainted.contains(&cond.0),
        _ => false,
    })
}

/// Can this `wukong_parallel_for(n, func_addr F, ctx)` call site run as cooperative work?
///
/// **What the op already guarantees.** `wukong_mir_build` only emits it after proving the loop's
/// iterations independent — a cross-iteration dependence, a captured scalar (i.e. a reduction) or
/// overlapping per-iteration write slices all keep the loop serial, and `tests/run/parallel_*.wk`
/// are the fixtures for each decline. So *any* partition of `[0, n)` computes the same result; that
/// is already load-bearing on CPU, where `wukong_runtime::parallel_for` hands out chunks whose
/// boundaries depend on the core count. The GPU's per-thread partition is one more partition.
///
/// **What it does not guarantee, and this checks.** The outlined body is lowered as an ordinary
/// single-thread `.func` and called by *every* thread, so a side effect inside it happens once per
/// thread instead of once. `print`/`println` append to the record buffer through an unguarded
/// `atom.global.add`, which would duplicate and interleave stdout — decline. `assert` is a single
/// idempotent flag store, so a per-thread repeat is the same observable outcome and stays allowed.
///
/// A nested `wukong_parallel_for` inside the body is allowed: within a `.func` it lowers to the
/// sequential chunk `F(0, n)`, which is correct (just not itself parallel).
fn parallel_for_is_cooperative(
    program: &Program,
    interner: &Interner,
    entry: &Function,
    args: &[ValueId],
) -> Result<(), String> {
    if args.len() != 3 {
        return Err(format!(
            "wukong_parallel_for takes 3 args, this call site has {}",
            args.len()
        ));
    }
    // PTX has no portable function pointers, so the target must be a compile-time `func_addr` — the
    // same requirement `lower.rs` has, checked here so eligibility and lowering agree.
    let mut target: Option<Symbol> = None;
    for b in &entry.blocks {
        for inst in &b.insts {
            if let (Some(r), Op::FuncAddr(sym)) = (inst.result, &inst.op) {
                if r == args[1] {
                    target = Some(*sym);
                }
            }
        }
    }
    let Some(sym) = target else {
        return Err("wukong_parallel_for target is not a func_addr".to_string());
    };
    let Some(body) = program.function(sym) else {
        return Err(format!(
            "wukong_parallel_for target `{}` is not a known function",
            interner.resolve(sym)
        ));
    };

    // Walk the body's reachability closure for per-thread-unsafe side effects.
    let mut seen: HashSet<u32> = HashSet::new();
    let mut work = vec![body];
    while let Some(f) = work.pop() {
        if !seen.insert(f.name.0) {
            continue;
        }
        for b in &f.blocks {
            for inst in &b.insts {
                if let Op::Call { func: callee, .. } = &inst.op {
                    let nm = interner.resolve(*callee);
                    if matches!(nm, "print" | "println") {
                        return Err(format!(
                            "outlined @parallel body `{}` prints (one record per thread, not per \
                             iteration)",
                            interner.resolve(sym)
                        ));
                    }
                    if let Some(next) = program.function(*callee) {
                        work.push(next);
                    }
                }
            }
        }
    }
    Ok(())
}

/// Does `func` contain a direct `wukong_parallel_for` call?
pub(crate) fn function_contains_parallel_for(interner: &Interner, func: &Function) -> bool {
    func.blocks.iter().any(|b| {
        b.insts.iter().any(|inst| {
            matches!(
                &inst.op,
                Op::Call { func: callee, .. } if interner.resolve(*callee) == "wukong_parallel_for"
            )
        })
    })
}

/// May the megakernel entry call this user-defined callee?
///
/// Admitted when the callee is a serial helper (no recognized cooperative op) or when it carries
/// the generic `@parallel` outline (`wukong_parallel_for`). A callee that calls any other
/// recognized op — `wukong_sgemm`, `wukong_sreduce`, … — stays on the single-thread path: those
/// ops need the entry's shared frame and the existing cooperative lowering, not an SPMD `.func`.
fn user_callee_is_admitted(
    program: &Program,
    interner: &Interner,
    func: &Function,
) -> Result<(), String> {
    if function_contains_parallel_for(interner, func) {
        return Ok(());
    }
    for b in &func.blocks {
        for inst in &b.insts {
            if let Op::Call { func: callee, .. } = &inst.op {
                if program.function(*callee).is_some() {
                    continue;
                }
                if matches!(classify_call(interner.resolve(*callee)), CallClass::Coop(_)) {
                    return Err(format!(
                        "user callee `{}` calls a recognized op other than wukong_parallel_for",
                        interner.resolve(func.name)
                    ));
                }
            }
        }
    }
    Ok(())
}

/// Scan one function's call sites for cooperative work, recursing into user-defined callees.
///
/// Each function in the call tree must have data-independent control flow. Cooperative-op arguments
/// must be data-independent in the *calling* function's taint set. Mutual recursion is declined.
fn scan_function_calls(
    program: &Program,
    interner: &Interner,
    func: &Function,
    coop_ops: &mut Vec<CoopOp>,
    visiting: &mut HashSet<Symbol>,
) -> Result<(), String> {
    let tainted = mem_tainted(func);
    if has_data_dependent_control_flow(func, &tainted) {
        return Err("data-dependent control flow (a branch reads memory)".to_string());
    }

    for (bi, b) in func.blocks.iter().enumerate() {
        for (ii, inst) in b.insts.iter().enumerate() {
            if let Op::Call { func: callee, args } = &inst.op {
                let name = interner.resolve(*callee).to_string();
                if let Some(callee_fn) = program.function(*callee) {
                    if let Err(why) = user_callee_is_admitted(program, interner, callee_fn) {
                        return Err(format!("call to user function `{name}`: {why}"));
                    }
                    if !visiting.insert(*callee) {
                        return Err(format!("recursive call to user function `{name}`"));
                    }
                    scan_function_calls(program, interner, callee_fn, coop_ops, visiting)?;
                    visiting.remove(callee);
                    if args.iter().any(|a| tainted.contains(&a.0)) {
                        return Err(format!(
                            "call to user function `{name}` has a data-dependent argument"
                        ));
                    }
                    continue;
                }
                match classify_call(&name) {
                    CallClass::Coop(kind) => {
                        if kind == CoopKind::ParallelFor {
                            parallel_for_is_cooperative(program, interner, func, args)?;
                        }
                        if args.iter().any(|a| tainted.contains(&a.0)) {
                            return Err(format!(
                                "recognized op `{name}` has a data-dependent argument"
                            ));
                        }
                        coop_ops.push(CoopOp {
                            block: bi,
                            inst: ii,
                            kind,
                            name,
                            args: args.clone(),
                            result: inst.result,
                        });
                    }
                    CallClass::SideEffect => {}
                    CallClass::Other => {
                        return Err(format!("call to unrecognized `{name}`"));
                    }
                }
            }
        }
    }
    Ok(())
}

/// Analyze `entry` for cooperative-megakernel eligibility and inventory its recognized ops.
///
/// Eligible iff:
///  - the entry and every user-defined callee it reaches (transitively) only call recognized
///    cooperative ops, deterministic side effects (`print`/`assert`), or more such callees. A
///    `wukong_parallel_for` is admitted when [`parallel_for_is_cooperative`] holds; its outlined body
///    is reached through `Op::FuncAddr`, not called directly, and `lower.rs` emits it as a
///    single-thread `.func`;
///  - control flow is data-independent in every function in that call tree;
///  - cooperative-op arguments are themselves data-independent (so every thread agrees on sizes and
///    pointers);
///  - at least one recognized cooperative op is present (else the single-thread path suffices).
pub fn analyze(program: &Program, entry: Symbol, interner: &Interner) -> MegaPlan {
    let Some(func) = program.function(entry) else {
        return MegaPlan::ineligible("no entry function");
    };
    if !func.params.is_empty() {
        return MegaPlan::ineligible("entry takes parameters");
    }

    let mut coop_ops: Vec<CoopOp> = Vec::new();
    let mut visiting = HashSet::new();
    visiting.insert(entry);
    if let Err(why) = scan_function_calls(program, interner, func, &mut coop_ops, &mut visiting) {
        return MegaPlan::ineligible(why);
    }

    if coop_ops.is_empty() {
        return MegaPlan::ineligible("no recognized cooperative op (single-thread path suffices)");
    }

    MegaPlan {
        eligible: true,
        reason: String::new(),
        coop_ops,
    }
}

#[cfg(all(test, feature = "gpu"))]
mod tests {
    use super::*;
    // **The one MIR builder in this crate that used to sit outside the environment lock.** This
    // module had its own private copy of the lex->parse->sema->mir_build->opt pipeline, which meant
    // a `build` here could run concurrently with `hostvec::normalized`'s `setenv(WUKONG_P4_NO_256)`
    // in another test thread. On Windows that is merely serialized by the OS; on glibc `setenv`
    // racing a concurrent `getenv` is a genuine data race (Rust made `std::env::set_var` `unsafe`
    // in edition 2024 for exactly this), and `mir_build` calls `var_os(VAR)` at both 256-bit recipe
    // sites. The seven gates below happen to be switch-insensitive today — proven, not assumed, by
    // `fusion_eligibility_is_host_vectorizer_invariant` at the bottom of this module — so nothing
    // was failing; the hazard was that the *next* sensitive fusion gate would become flaky with a
    // failure that pointed nowhere near its cause. Routing through the shared builder puts every
    // MIR build in this crate under one lock and, on hosts where the switch can matter, in one
    // normalized condition.
    use crate::lower::hostvec::{build, build_in_current_host_mode};

    fn plan(src: &str, opt: u8) -> (MegaPlan, Program, Interner) {
        let (program, mut interner) = build(src, opt).expect("frontend ok");
        let entry = interner.intern("main");
        let p = analyze(&program, entry, &interner);
        (p, program, interner)
    }

    #[test]
    fn classify_symbols() {
        assert_eq!(
            classify_call("wukong_sreduce_f32_parallel"),
            CallClass::Coop(CoopKind::Reduce)
        );
        assert_eq!(
            classify_call("wukong_sgemm_nt_epi"),
            CallClass::Coop(CoopKind::GemmNtEpi)
        );
        assert_eq!(
            classify_call("wukong_vmath_bf16"),
            CallClass::Coop(CoopKind::Vmath)
        );
        assert_eq!(classify_call("print"), CallClass::SideEffect);
        assert_eq!(classify_call("println"), CallClass::SideEffect);
        assert_eq!(classify_call("some_user_fn"), CallClass::Other);
    }

    /// **Every `_parallel` spelling `lower::rt_helper` maps must classify as its serial twin's kind.**
    /// These seven were missing, so a program whose activation or low-precision reduction came out of an
    /// `@parallel fn` was reported ineligible ("entry calls unrecognized `wukong_vmath_f32_parallel`")
    /// and silently fell back to the one-thread path — the megakernel declining exactly the shape the
    /// user asked to parallelize. The serial twins are asserted alongside so the pairing stays visible.
    #[test]
    fn parallel_twins_classify_like_their_serial_form() {
        for (serial, parallel) in [
            ("wukong_vmath_f32", "wukong_vmath_f32_parallel"),
            ("wukong_dot_bf16", "wukong_dot_bf16_parallel"),
            ("wukong_dot_f16", "wukong_dot_f16_parallel"),
            ("wukong_sum_bf16", "wukong_sum_bf16_parallel"),
            ("wukong_sum_f16", "wukong_sum_f16_parallel"),
            ("wukong_reduce_bf16", "wukong_reduce_bf16_parallel"),
            ("wukong_reduce_f16", "wukong_reduce_f16_parallel"),
        ] {
            let (a, b) = (classify_call(serial), classify_call(parallel));
            assert!(
                matches!(a, CallClass::Coop(_)),
                "{serial} must be a cooperative op"
            );
            assert_eq!(a, b, "`{parallel}` must classify exactly like `{serial}`");
        }
    }

    /// The `@parallel` elementwise-activation shape, `N` large enough that `mir_build`'s 256-bit
    /// recipe site is reachable. A function rather than a `const` because it interpolates `N`.
    /// Shared with [`parallel_activation_is_eligible`] and the host-vectorizer invariance gate
    /// below, which needs the *same* program under both switch settings.
    fn src_parallel_activation() -> String {
        const N: usize = 1024;
        format!(
            r#"module t
@parallel
fn act(x: [f32; {N}], mut o: [f32; {N}], mut tail: [f32; 1]) {{
    for i in 0..{N} {{ o[i] = silu(x[i]); }}
    tail[0] = o[{last}];
}}
fn main() -> i32 {{
    let mut x: [f32; {N}] = [0.0; {N}];
    let mut o: [f32; {N}] = [0.0; {N}];
    let mut tail: [f32; 1] = [0.0; 1];
    for i in 0..{N} {{ x[i] = ((i as f32) - 512.0) / 128.0; }}
    act(x, o, tail);
    print((o[3] * 10000.0) as i32);
    return 0;
}}
"#,
            last = N - 1
        )
    }

    /// The `@parallel` recognized-reduction shape (the other `mir_build` 256-bit recipe site).
    const SRC_PARALLEL_REDUCE: &str = r#"module t
@parallel
fn dotp(x: [f32; 256], y: [f32; 256], mut o: [f32; 1]) {
    let mut s: f32 = 0.0;
    for k in 0..256 { s = s + x[k] * y[k]; }
    o[0] = s;
}
fn main() -> i32 {
    let mut x: [f32; 256] = [2.0; 256];
    let mut y: [f32; 256] = [3.0; 256];
    let mut o: [f32; 1] = [0.0; 1];
    dotp(x, y, o); print((o[0]) as i32);
    return 0;
}
"#;

    /// The **`@parallel` activation shape** is megakernel-eligible. In a *multi-statement* `@parallel`
    /// function the region runs with `parallel_fn = true`, so at -O2 mir_build interns
    /// `wukong_vmath_f32_parallel` and inlines it into `main` (verified with `wukongc --emit=mir -O2`;
    /// the same twin `tests/run/vmath_parallel.wk` gates on the CPU). Before the classification fix
    /// that name reached `CallClass::Other`, so `analyze` refused the whole program with "entry calls
    /// unrecognized `wukong_vmath_f32_parallel`" and the megakernel silently declined.
    #[test]
    fn parallel_activation_is_eligible() {
        let src = src_parallel_activation();
        let (p, _prog, _i) = plan(&src, 2);
        assert!(p.eligible, "expected eligible, got: {}", p.reason);
        assert!(
            p.coop_ops
                .iter()
                .any(|c| c.name == "wukong_vmath_f32_parallel"),
            "expected the @parallel vmath twin, got {:?}",
            p.coop_ops
                .iter()
                .map(|c| (c.kind, c.name.clone()))
                .collect::<Vec<_>>()
        );
    }

    /// A recognized-reduction program over statically-sized buffers: data-independent control flow,
    /// recognized cooperative ops, no other user calls -> eligible. (At -O2 the @parallel reductions
    /// become `wukong_sreduce_f32_parallel` and `main` inlines them.)
    #[test]
    fn parallel_reduce_is_eligible() {
        let (p, _prog, _i) = plan(SRC_PARALLEL_REDUCE, 2);
        assert!(p.eligible, "expected eligible, got: {}", p.reason);
        assert!(
            p.coop_ops.iter().any(|c| c.kind == CoopKind::Reduce),
            "expected a Reduce coop op, got {:?}",
            p.coop_ops.iter().map(|c| c.kind).collect::<Vec<_>>()
        );
    }

    /// A recognized small matmul: the loop nest lowers to `wukong_sgemm`; control flow (the GEMM's
    /// internal loops are inside the kernel, `main`'s remaining loops are static) is data-independent.
    #[test]
    fn matmul_is_eligible_or_pure_general() {
        let src = std::fs::read_to_string(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/run/matmul_f32.wk"),
        )
        .unwrap();
        // At -O2 the nest is recognized to a GEMM call -> eligible with a Gemm/GemmNt coop op.
        let (p, _prog, _i) = plan(&src, 2);
        assert!(p.eligible, "matmul_f32 -O2 expected eligible: {}", p.reason);
        assert!(p
            .coop_ops
            .iter()
            .any(|c| matches!(c.kind, CoopKind::Gemm | CoopKind::GemmNt)));
    }

    /// Data-dependent control flow (a sort branches on loaded values) must be rejected — the SPMD
    /// cooperative model would deadlock, so it stays on the single-thread path.
    #[test]
    fn data_dependent_control_flow_is_rejected() {
        let src = r#"module t
fn main() -> i32 {
    let mut a: [i32; 8] = [5, 3, 8, 1, 9, 2, 7, 4];
    let mut i: i32 = 0;
    while i < 8 {
        let mut j: i32 = 0;
        while j < 7 {
            if a[j] > a[j + 1] {
                let t: i32 = a[j];
                a[j] = a[j + 1];
                a[j + 1] = t;
            }
            j = j + 1;
        }
        i = i + 1;
    }
    print(a[0]);
    return 0;
}
"#;
        for opt in [0u8, 2u8, 3u8] {
            let (p, _prog, _i) = plan(src, opt);
            assert!(
                !p.eligible,
                "bubble sort -O{opt} must be ineligible (data-dependent control flow)"
            );
            assert!(p.reason.contains("data-dependent"), "reason: {}", p.reason);
        }
    }

    /// A mid-function `@parallel` region over a body no recognizer claims — the generic elementwise
    /// shape Phase 1 exists for. The `wukong_parallel_for` in `main` used to make the whole program
    /// ineligible ("not yet cooperative"), so the one construct that *says* "these iterations are
    /// independent" was the one construct the cooperative megakernel refused.
    const SRC_GENERIC_PARALLEL: &str = r#"module t
@parallel
fn main() -> i32 {
    let mut x: [f32; 64] = [0.0; 64];
    let mut y: [f32; 64] = [0.0; 64];
    for i in 0..64 { x[i] = (i as f32) * 0.5; }
    for i in 0..64 {
        let v = x[i];
        y[i] = v * v * 0.25 + 3.0;
    }
    print(y[63] as i32);
    return 0;
}
"#;

    /// `-O1` and up only, and that is not this op's doing: at `-O0` the entry's own loop counters
    /// still live in allocas, so `i < 64` branches on a `Load` and the pre-existing
    /// data-independent-control-flow gate rejects the program before `wukong_parallel_for` is even
    /// looked at. `jit_run` then runs it on the (correct) single-thread path, which is why the
    /// corpus's `-O0 == -O3` comparison is unaffected.
    #[test]
    fn generic_parallel_region_is_eligible() {
        for opt in [2u8, 3u8] {
            let (p, _prog, _i) = plan(SRC_GENERIC_PARALLEL, opt);
            assert!(
                p.eligible,
                "generic @parallel -O{opt} expected eligible, got: {}",
                p.reason
            );
            assert!(
                p.coop_ops.iter().any(|c| c.kind == CoopKind::ParallelFor),
                "expected a ParallelFor coop op at -O{opt}, got {:?}",
                p.coop_ops.iter().map(|c| c.kind).collect::<Vec<_>>()
            );
        }
    }

    /// The common `@parallel fn saxpy(..)` called from `main` shape — cooperative work lives in the
    /// callee, not the entry. Until Phase 1.2b this was declined ("entry calls user function").
    const SRC_CALLEE_PARALLEL: &str = r#"module t
@parallel
fn saxpy(mut x: [f32; 64], mut y: [f32; 64]) {
    for i in 0..64 { x[i] = (i as f32) * 2.0; }
    for i in 0..64 {
        let v = x[i];
        y[i] = v * v * 0.25 + 3.0;
    }
}
fn main() -> i32 {
    let mut x: [f32; 64] = [0.0; 64];
    let mut y: [f32; 64] = [0.0; 64];
    saxpy(x, y);
    print(y[63] as i32);
    return 0;
}
"#;

    #[test]
    fn generic_parallel_callee_is_eligible() {
        for opt in [2u8, 3u8] {
            let (p, _prog, _i) = plan(SRC_CALLEE_PARALLEL, opt);
            assert!(
                p.eligible,
                "generic @parallel callee -O{opt} expected eligible, got: {}",
                p.reason
            );
            assert!(
                p.coop_ops.iter().any(|c| c.kind == CoopKind::ParallelFor),
                "expected a ParallelFor coop op at -O{opt}, got {:?}",
                p.coop_ops.iter().map(|c| c.kind).collect::<Vec<_>>()
            );
        }
    }

    /// A user callee that calls `wukong_sgemm` / `wukong_norm` / … directly — not
    /// `wukong_parallel_for` — must keep the program on the single-thread path. `transformer_block`
    /// is the shape.
    #[test]
    fn gemm_callee_keeps_program_ineligible() {
        let src = std::fs::read_to_string(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../tests/run/transformer_block.wk"),
        )
        .expect("read transformer_block.wk");
        for opt in [2u8, 3u8] {
            let (p, _prog, _i) = plan(&src, opt);
            assert!(
                !p.eligible,
                "transformer_block -O{opt} must stay ineligible (gemm callee), got eligible"
            );
            assert!(
                p.reason.contains("wukong_parallel_for")
                    || p.reason.contains("recognized op")
                    || p.reason.contains("user callee"),
                "reason should name the gemm callee: {}",
                p.reason
            );
        }
    }

    /// A hand-built `wukong_parallel_for(64, func_addr body, env)` in `main`, with or without a
    /// `print` in the outlined body.
    ///
    /// Hand-built because the front end will not produce the printing case: `mir_build`'s region
    /// prover already refuses to outline a loop containing a side effect (checked — `--emit=mir -O2`
    /// of such a loop has no `wukong_parallel_for`). That makes the check below a *defensive* one,
    /// and defending it here is the point: this eligibility gate is the only thing standing between
    /// a future outliner that does allow side effects and stdout gaining one line per thread.
    fn parallel_for_program(body_prints: bool) -> (Program, Interner) {
        use wukong_mir::{Builder, MirType};

        let mut interner = Interner::new();
        let body_sym = interner.intern("wukong$par$0");
        let print_sym = interner.intern("print");
        let pfor_sym = interner.intern("wukong_parallel_for");

        let mut fb = Builder::new(body_sym, MirType::Void);
        let lo = fb.add_param(MirType::I64);
        let _hi = fb.add_param(MirType::I64);
        let _env = fb.add_param(MirType::Ptr);
        if body_prints {
            fb.build_void(Op::Call {
                func: print_sym,
                args: vec![lo],
            });
        }
        fb.ret(None);

        let mut mb = Builder::new(interner.intern("main"), MirType::I32);
        let env = mb.alloca(MirType::Array(Box::new(MirType::Ptr), 1));
        let n = mb.build(MirType::I64, Op::ConstInt(64, MirType::I64));
        let f = mb.build(MirType::Ptr, Op::FuncAddr(body_sym));
        mb.build_void(Op::Call {
            func: pfor_sym,
            args: vec![n, f, env],
        });
        let z = mb.build(MirType::I32, Op::ConstInt(0, MirType::I32));
        mb.ret(Some(z));

        let mut program = Program::new();
        program.funcs.push(mb.finish());
        program.funcs.push(fb.finish());
        (program, interner)
    }

    /// A `print` inside the outlined body must keep the program off the cooperative path. The body is
    /// lowered as an ordinary `.func` and called by every thread, and `print` appends to the record
    /// buffer through an unguarded `atom.global.add` — so stdout would gain one line per *thread*
    /// instead of one per iteration. This is the one thing the `@parallel` independence proof does not
    /// cover, because duplicating a side effect is not a data race.
    #[test]
    fn a_printing_parallel_body_is_not_cooperative() {
        let (quiet, mut qi) = parallel_for_program(false);
        let entry = qi.intern("main");
        let p = analyze(&quiet, entry, &qi);
        assert!(
            p.eligible,
            "a side-effect-free @parallel outline is cooperative work: {}",
            p.reason
        );
        assert!(p.coop_ops.iter().any(|c| c.kind == CoopKind::ParallelFor));

        let (loud, mut li) = parallel_for_program(true);
        let entry = li.intern("main");
        let p = analyze(&loud, entry, &li);
        assert!(!p.eligible, "a printing @parallel body must decline");
        assert!(
            p.reason.contains("prints"),
            "the reason should name the print: {}",
            p.reason
        );
    }

    /// A pure-scalar program with no recognized op is (correctly) ineligible: the single-thread path
    /// already suffices and there is nothing to parallelize cooperatively.
    #[test]
    fn pure_scalar_has_no_coop_benefit() {
        let src = r#"module t
fn main() -> i32 {
    let mut s: i32 = 0;
    let mut i: i32 = 0;
    while i < 10 { s = s + i; i = i + 1; }
    print(s);
    return 0;
}
"#;
        let (p, _prog, _i) = plan(src, 2);
        assert!(!p.eligible);
        assert!(
            p.reason.contains("no recognized cooperative op"),
            "reason: {}",
            p.reason
        );
    }

    /// **Every gate in this module is host-vectorizer invariant — proven, not asserted.**
    ///
    /// The gates above now build their MIR through [`crate::lower::hostvec`], which on a host with
    /// the raw-AVX2 256-bit loop vectorizer (Win64 only) forces `WUKONG_P4_NO_256=1` so the corpus
    /// numbers are one number per project rather than one per operating system. That normalization
    /// is *safe* here only if the eligibility analysis cannot see the switch — and until now that
    /// was a claim in a comment in `lower.rs`, backed by a one-off manual run someone did once.
    ///
    /// A comment is not a gate. `mir_build` reads `WUKONG_P4_NO_256` at exactly two sites — the
    /// elementwise and the reduction 256-bit `VecKernel` recipes — and both are on the recognized
    /// `@parallel` paths this module classifies, so a future widening of either recipe (a new
    /// recognized symbol, a different call shape, a lane-width-dependent inline decision) is
    /// precisely the change that would make one of these programs eligible under one setting and
    /// not the other. This sweeps both recipe shapes plus the recognized GEMM under
    /// [`hostvec::normalized`] and [`hostvec::host_native`] and requires the whole [`MegaPlan`] to
    /// agree: eligibility, decline reason, and the cooperative-op inventory in order.
    ///
    /// Where the switch cannot matter (any non-Win64 host — `switch_can_matter()` is false, the
    /// environment is never touched, and both wrappers reduce to calling `f`) the two arms are the
    /// same run by construction, so the invariance half degenerates to running each program twice
    /// and the non-vacuity half is skipped with a note, exactly as `lower.rs`'s corpus gate does.
    ///
    /// **Non-vacuity.** "Both arms agree" is worth nothing if both arms are the same MIR. The
    /// `MIXED` program below exists for that: a recognized `@parallel` dot (so it is *eligible*,
    /// not just trivially rejected) beside a 4096-trip `b[i] = a[i]*c[i] + d[i]` loop, which the
    /// recognizers do **not** claim (`velem` wants `a*x + b*y + c` with scalar coefficients) and
    /// the 256-bit recipe therefore does. Measured on this host: 1 `Op::VecKernelCall` host-native,
    /// 0 normalized — genuinely different MIR into `analyze`, identical `MegaPlan` out.
    #[test]
    fn fusion_eligibility_is_host_vectorizer_invariant() {
        use crate::lower::hostvec;

        /// Eligible **and** a 256-bit-recipe carrier — see the non-vacuity note above.
        const MIXED: &str = r#"module t
@parallel
fn dotp(x: [f32; 256], y: [f32; 256], mut o: [f32; 1]) {
    let mut s: f32 = 0.0;
    for k in 0..256 { s = s + x[k] * y[k]; }
    o[0] = s;
}
fn main() -> i32 {
    let mut a: [f32; 4096] = [1.0; 4096];
    let mut b: [f32; 4096] = [0.0; 4096];
    let mut c: [f32; 4096] = [2.0; 4096];
    let mut d: [f32; 4096] = [3.0; 4096];
    for i in 0..4096 { b[i] = a[i] * c[i] + d[i]; }
    let mut x: [f32; 256] = [2.0; 256];
    let mut y: [f32; 256] = [3.0; 256];
    let mut o: [f32; 1] = [0.0; 1];
    dotp(x, y, o);
    print((o[0] + b[7]) as i32);
    return 0;
}
"#;

        /// `(MegaPlan, VecKernelCall carriers)` for one arm — the plan is what must match, the
        /// count is what proves the two arms were fed different MIR. It takes
        /// [`build_in_current_host_mode`] and never [`build`]: `hostvec`'s `ENV_LOCK` is a plain
        /// non-reentrant `Mutex`, and this runs *inside* a `normalized`/`host_native` window, so
        /// calling `build` here would deadlock against itself (the trap `lower.rs`'s corpus
        /// invariance gate documents at its own sweep).
        fn probe(src: &str, opt: u8) -> (MegaPlan, usize) {
            let (program, mut interner) =
                build_in_current_host_mode(src, opt).expect("frontend ok");
            let vk = program
                .funcs
                .iter()
                .flat_map(|f| f.blocks.iter())
                .flat_map(|b| b.insts.iter())
                .filter(|i| matches!(i.op, Op::VecKernelCall { .. }))
                .count();
            let entry = interner.intern("main");
            (analyze(&program, entry, &interner), vk)
        }

        let act = src_parallel_activation();
        let matmul = std::fs::read_to_string(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/run/matmul_f32.wk"),
        )
        .unwrap();
        let corpus: [(&str, &str); 4] = [
            ("parallel_activation", act.as_str()),
            ("parallel_reduce", SRC_PARALLEL_REDUCE),
            ("matmul_f32", matmul.as_str()),
            ("mixed_reduce_plus_general_loop", MIXED),
        ];
        let ops = |p: &MegaPlan| -> Vec<(CoopKind, String)> {
            p.coop_ops
                .iter()
                .map(|c| (c.kind, c.name.clone()))
                .collect()
        };
        let mut native_carriers = 0usize;
        let mut norm_carriers = 0usize;
        for (name, src) in corpus {
            for opt in [0u8, 2u8] {
                let (norm, nvk) = hostvec::normalized(|| probe(src, opt));
                let (native, hvk) = hostvec::host_native(|| probe(src, opt));
                norm_carriers += nvk;
                native_carriers += hvk;
                assert_eq!(
                    norm.eligible, native.eligible,
                    "{name}@O{opt}: eligibility depends on the host 256-bit vectorizer \
                     (normalized {} vs host-native {}) — every gate in this module is built under \
                     `hostvec::normalized`, so the switch is not allowed to be visible here. Fix \
                     the analysis or stop normalizing; do not delete this gate.",
                    norm.eligible, native.eligible
                );
                assert_eq!(
                    norm.reason, native.reason,
                    "{name}@O{opt}: the decline reason depends on the host 256-bit vectorizer"
                );
                assert_eq!(
                    ops(&norm),
                    ops(&native),
                    "{name}@O{opt}: the cooperative-op inventory depends on the host 256-bit \
                     vectorizer"
                );
            }
        }
        if !wukong_mir::host_supports_vec_kernels() {
            eprintln!(
                "fusion host-vectorizer invariance: {} programs x 2 opt levels agree; non-vacuity \
                 check skipped — `host_supports_vec_kernels()` is false here, so `mir_build` emits \
                 no 256-bit recipe and the two arms are the same build by construction.",
                corpus.len()
            );
            return;
        }
        assert_eq!(
            norm_carriers, 0,
            "the NORMALIZED arm still built {norm_carriers} `Op::VecKernelCall`(s) — \
             `hostvec::normalized` is supposed to suppress the 256-bit recipe outright, so this \
             module's gates are not running in the condition they claim to."
        );
        assert!(
            native_carriers > 0,
            "no program in this gate's corpus produced an `Op::VecKernelCall` on a host where \
             `host_supports_vec_kernels()` is true, so both arms were handed identical MIR and \
             'they agree' proves nothing. `MIXED` is the program that is supposed to carry one; \
             either the recognizers now claim its `a[i]*c[i] + d[i]` loop or the recipe changed. \
             Find a carrier again rather than deleting this assertion."
        );
        eprintln!(
            "fusion host-vectorizer invariance: {} programs x 2 opt levels, MegaPlan identical \
             (eligibility, reason, coop-op inventory) — and NON-VACUOUS: the host-native arm built \
             {native_carriers} `Op::VecKernelCall`(s) the normalized arm did not, so the two arms \
             really did analyze different MIR.",
            corpus.len()
        );
    }
}
