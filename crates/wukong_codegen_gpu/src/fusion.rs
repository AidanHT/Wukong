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
/// out of an `@parallel fn` is declared ineligible and silently falls back to the one-thread path —
/// the megakernel never accelerating the very shape the user asked to parallelize. (The converse
/// direction is safe: a name classified here that `rt_helper` does not know now returns an
/// `UNSUPPORTED:` decline rather than panicking.)
pub fn classify_call(name: &str) -> CallClass {
    use CoopKind::*;
    let kind = match name {
        "wukong_sreduce_f32" | "wukong_sreduce_f32_parallel" => Reduce,
        "wukong_dot_bf16" | "wukong_dot_bf16_parallel" | "wukong_dot_f16"
        | "wukong_dot_f16_parallel" | "wukong_sum_bf16" | "wukong_sum_bf16_parallel"
        | "wukong_sum_f16" | "wukong_sum_f16_parallel" | "wukong_reduce_bf16"
        | "wukong_reduce_bf16_parallel" | "wukong_reduce_f16" | "wukong_reduce_f16_parallel" => {
            ReduceLowp
        }
        "wukong_sgemm" | "wukong_sgemm_parallel" => Gemm,
        "wukong_sgemm_nt" | "wukong_sgemm_nt_parallel" => GemmNt,
        "wukong_sgemm_nt_epi" | "wukong_sgemm_nt_epi_parallel" => GemmNtEpi,
        "wukong_i8gemm_nt" | "wukong_i8gemm_nt_parallel" => I8GemmNt,
        "wukong_norm_f32" | "wukong_norm_f32_parallel" => Norm,
        "wukong_norm_affine_f32" | "wukong_norm_affine_f32_parallel" => NormAffine,
        "wukong_vmath_f32" | "wukong_vmath_f32_parallel" | "wukong_vmath_bf16"
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
    pub reason: String,
    /// The recognized cooperative ops in entry order (program order within the block sweep).
    pub coop_ops: Vec<CoopOp>,
}

impl MegaPlan {
    fn ineligible(reason: impl Into<String>) -> Self {
        MegaPlan { eligible: false, reason: reason.into(), coop_ops: Vec::new() }
    }
}

/// The operand values an [`Op`] *reads* (used by the taint fixpoint). Address/value operands of a
/// `Store` are included even though `Store` has no result, so a callers iterating "reads" see them;
/// the taint pass only consults this for value-producing ops.
fn op_operands(op: &Op) -> Vec<ValueId> {
    match op {
        Op::ConstInt(..)
        | Op::ConstFloat(..)
        | Op::Alloca(_)
        | Op::FuncAddr(_)
        | Op::GlobalAddr(_)
        | Op::Iota(_) => Vec::new(),
        Op::Bin(_, a, b) | Op::Cmp(_, a, b) => vec![*a, *b],
        Op::Neg(a) | Op::Not(a) | Op::Sqrt(a) | Op::Splat(a) | Op::Round(_, a)
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
        Op::VecKernelCall { ptrs, scalars, n, .. } => vec![*ptrs, *scalars, *n],
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
            Terminator::CondBr { then_blk, then_args, else_blk, else_args, .. } => {
                incoming.entry(then_blk.0).or_default().push(then_args.clone());
                incoming.entry(else_blk.0).or_default().push(else_args.clone());
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
                    if edges.iter().any(|args| args.get(i).is_some_and(|a| tainted.contains(&a.0))) {
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

/// Analyze `entry` for cooperative-megakernel eligibility and inventory its recognized ops.
///
/// Eligible iff (in this first increment):
///  - the entry's only calls are recognized cooperative ops or deterministic side effects
///    (`print`/`assert`) — no other user functions, no unrecognized runtime symbols, no
///    `wukong_parallel_for` yet (its outlined body is a separate function — a later increment);
///  - control flow is data-independent (the safety gate above);
///  - cooperative-op arguments are themselves data-independent (so every thread computes the same
///    pointers/sizes — otherwise threads would disagree on what to cooperate on);
///  - at least one recognized cooperative op is present (else the single-thread path is already fine).
pub fn analyze(program: &Program, entry: Symbol, interner: &Interner) -> MegaPlan {
    let Some(func) = program.function(entry) else {
        return MegaPlan::ineligible("no entry function");
    };
    if !func.params.is_empty() {
        return MegaPlan::ineligible("entry takes parameters");
    }

    let tainted = mem_tainted(func);
    if has_data_dependent_control_flow(func, &tainted) {
        return MegaPlan::ineligible("data-dependent control flow (a branch reads memory)");
    }

    let mut coop_ops: Vec<CoopOp> = Vec::new();
    for (bi, b) in func.blocks.iter().enumerate() {
        for (ii, inst) in b.insts.iter().enumerate() {
            if let Op::Call { func: callee, args } = &inst.op {
                let name = interner.resolve(*callee).to_string();
                // A call to another user-defined function is opaque to this single-function analysis.
                if program.function(*callee).is_some() {
                    return MegaPlan::ineligible(format!("entry calls user function `{name}`"));
                }
                match classify_call(&name) {
                    CallClass::Coop(kind) => {
                        if kind == CoopKind::ParallelFor {
                            return MegaPlan::ineligible(
                                "generic @parallel (wukong_parallel_for) not yet cooperative",
                            );
                        }
                        // Cooperative-op args must be data-independent so all threads agree.
                        if args.iter().any(|a| tainted.contains(&a.0)) {
                            return MegaPlan::ineligible(format!(
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
                        return MegaPlan::ineligible(format!("entry calls unrecognized `{name}`"));
                    }
                }
            }
        }
    }

    if coop_ops.is_empty() {
        return MegaPlan::ineligible("no recognized cooperative op (single-thread path suffices)");
    }

    MegaPlan { eligible: true, reason: String::new(), coop_ops }
}

#[cfg(all(test, feature = "gpu"))]
mod tests {
    use super::*;
    use wukong_span::SourceMap;

    /// Build a program from `.wk` source at `opt` (lex -> parse -> sema -> mir_build -> opt).
    fn build(src: &str, opt: u8) -> Option<(Program, Interner)> {
        let mut sm = SourceMap::new();
        let id = sm.add("fusion_gate.wk".to_string(), src.to_string());
        let (tokens, ld) = wukong_lexer::tokenize(sm.source(id), id);
        if ld.iter().any(|d| d.is_error()) {
            return None;
        }
        let mut interner = Interner::new();
        let (module, pd) = wukong_parser::parse_module_tokens(&tokens, sm.source(id), &mut interner);
        if pd.iter().any(|d| d.is_error()) {
            return None;
        }
        let (sema, sd) = wukong_sema::check(&module, &interner);
        if sd.iter().any(|d| d.is_error()) {
            return None;
        }
        let (mut program, md) = wukong_mir_build::lower_program(&module, &sema, &mut interner);
        if md.iter().any(|d| d.is_error()) {
            return None;
        }
        wukong_opt::optimize(&mut program, opt);
        Some((program, interner))
    }

    fn plan(src: &str, opt: u8) -> (MegaPlan, Program, Interner) {
        let (program, mut interner) = build(src, opt).expect("frontend ok");
        let entry = interner.intern("main");
        let p = analyze(&program, entry, &interner);
        (p, program, interner)
    }

    #[test]
    fn classify_symbols() {
        assert_eq!(classify_call("wukong_sreduce_f32_parallel"), CallClass::Coop(CoopKind::Reduce));
        assert_eq!(classify_call("wukong_sgemm_nt_epi"), CallClass::Coop(CoopKind::GemmNtEpi));
        assert_eq!(classify_call("wukong_vmath_bf16"), CallClass::Coop(CoopKind::Vmath));
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
            assert!(matches!(a, CallClass::Coop(_)), "{serial} must be a cooperative op");
            assert_eq!(a, b, "`{parallel}` must classify exactly like `{serial}`");
        }
    }

    /// The **`@parallel` activation shape** is megakernel-eligible. In a *multi-statement* `@parallel`
    /// function the region runs with `parallel_fn = true`, so at -O2 mir_build interns
    /// `wukong_vmath_f32_parallel` and inlines it into `main` (verified with `wukongc --emit=mir -O2`;
    /// the same twin `tests/run/vmath_parallel.wk` gates on the CPU). Before the classification fix
    /// that name reached `CallClass::Other`, so `analyze` refused the whole program with "entry calls
    /// unrecognized `wukong_vmath_f32_parallel`" and the megakernel silently declined.
    #[test]
    fn parallel_activation_is_eligible() {
        const N: usize = 1024;
        let src = format!(
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
        );
        let (p, _prog, _i) = plan(&src, 2);
        assert!(p.eligible, "expected eligible, got: {}", p.reason);
        assert!(
            p.coop_ops.iter().any(|c| c.name == "wukong_vmath_f32_parallel"),
            "expected the @parallel vmath twin, got {:?}",
            p.coop_ops.iter().map(|c| (c.kind, c.name.clone())).collect::<Vec<_>>()
        );
    }

    /// A recognized-reduction program over statically-sized buffers: data-independent control flow,
    /// recognized cooperative ops, no other user calls -> eligible. (At -O2 the @parallel reductions
    /// become `wukong_sreduce_f32_parallel` and `main` inlines them.)
    #[test]
    fn parallel_reduce_is_eligible() {
        let src = r#"module t
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
        let (p, _prog, _i) = plan(src, 2);
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
        assert!(p.coop_ops.iter().any(|c| matches!(c.kind, CoopKind::Gemm | CoopKind::GemmNt)));
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
        assert!(p.reason.contains("no recognized cooperative op"), "reason: {}", p.reason);
    }
}
