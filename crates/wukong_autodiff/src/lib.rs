//! `wukong_autodiff` — reverse-mode automatic differentiation as a MIR->MIR transform.
//!
//! Wukong's forward path turns a `.wk` model into tuned kernels; this crate is the **backward
//! path**. Given a forward MIR [`Function`] that computes a scalar loss, [`grad`] emits a *new*
//! MIR function that runs the same forward pass and *also* accumulates the gradient of that loss
//! with respect to designated input buffers — the vector-Jacobian product (VJP), walked in reverse
//! topological order over the forward instructions.
//!
//! # The model
//!
//! The transform consumes **straight-line, SSA-form** MIR: a single basic block. This is exactly the
//! shape the optimizer produces after `mem2reg` + simplify-cfg, with every recognized loop nest
//! lowered to one tuned-kernel call. Two adjoint models meet in it: the scalar core (one adjoint SSA
//! value per forward value) and the buffer tape (`tape*.rs`: one gradient buffer per forward buffer,
//! with the kernels' own backward kernels).
//!
//! Contract for the forward function `f`:
//! - exactly one basic block, terminated by `ret <loss>` (the scalar being differentiated);
//! - buffers (`Ptr` parameters and local `f32` arrays) are read with `load` through a `gep` chain;
//!   a float `store` into a buffer whose gradient is live is differentiated as a memory write (the
//!   slot's adjoint moves to the stored value and the slot's own adjoint becomes zero), and a store
//!   into a buffer nobody differentiates (the loss sink) is skipped;
//! - `wrt` lists the parameter indices to differentiate. For each, the emitted gradient function
//!   appends one `Ptr` parameter that receives `d loss / d input` (accumulated, so the caller must
//!   pass a zeroed buffer).
//!
//! Calls to user functions are differentiated by [`inline_calls`] (straight-line callees are
//! spliced in before differentiation, at every `-O` level); second-order derivatives of scalar-path
//! losses come from [`grad_dot`] / [`hvp`].
//!
//! # Correctness gate
//!
//! Every rule is gated by **finite differences** (see the test module): perturb each input element
//! by +-eps, evaluate the loss, and assert the central difference `(L(x+eps) - L(x-eps)) / 2eps`
//! matches the analytic gradient the transform emits, over the full gradient buffer. For the scalar
//! rules the reference forward pass runs in `f64` (via `wukong_interp::run_kernel_f64`) so eps-noise
//! stays far below the tolerance. The buffer-tape rules can only run in `f32` — the runtime kernels
//! they call are f32-only — so there the loose f32 finite difference is the sanity check and a tight
//! **f64 closed form** is the real gate wherever one exists.
//! A wrong gradient is a miscompile — it must be fixed before any throughput counts.

use std::collections::{HashMap, HashSet};
use wukong_mir::{
    BasicBlock, BinOp, CastKind, Function, Inst, MirType, Op, Program, Terminator, ValueId,
};
use wukong_span::{Interner, Symbol};

mod coverage;
mod inline;
pub mod optim;
mod tape;
mod tape_linalg;
mod tape_nn;
use tape::{Kernel, Syms};

/// Which function the reverse pass builds.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Mode {
    /// `{name}_grad`: void, writes each `wrt` gradient into an appended buffer.
    Grad,
    /// `{name}_gdot`: returns the directional derivative `Σ_j <v_j, ∂L/∂p_j>`, one appended
    /// direction buffer `v_j` per `wrt` parameter — a single-block scalar function, so it is itself
    /// differentiable: its gradient is the Hessian-vector product.
    GradDot,
}

/// Differentiate `func` with respect to the buffer parameters named by `wrt` (indices into
/// `func.params`), returning a new function `{func.name}_grad`.
///
/// The returned function has the **same parameters** as `func` followed by **one `Ptr` parameter
/// per `wrt` entry**, in `wrt` order; each receives the accumulated gradient of the loss w.r.t.
/// that input buffer. It runs the forward pass (so any forward output store still happens) and then
/// the reverse pass.
///
/// Errors if `func` violates the straight-line / `ret <loss>` contract, if a `wrt` index is not a
/// pointer parameter, or if the reverse pass meets a differentiable value whose defining op has no
/// VJP rule yet (so a missing rule is a loud failure, never a silently-zero gradient).
pub fn grad(func: &Function, wrt: &[usize], interner: &mut Interner) -> Result<Function, String> {
    build(func, wrt, interner, Mode::Grad)
}

/// The **directional derivative** of a scalar-path loss: `{func.name}_gdot(params..., v_1..v_k)
/// -> Σ_j <v_j, ∂L/∂p_{wrt_j}>`, with one appended `Ptr` direction buffer per `wrt` entry.
///
/// This is the second-order entry point. The function `grad` emits is void and multi-block, so it
/// can never be fed back in; the directional derivative is a single-block function returning a
/// float, so [`grad`] of it is the **Hessian-vector product** `H·v` (forward-free reverse-over-reverse,
/// the same construction `jax.hessian`'s `hvp` uses). Supported subset, stated exactly: losses
/// whose gradient flows through the scalar core only — arithmetic, `sqrt`, `fma`, `select`, casts,
/// scalar trig, and loads of `wrt` parameters. A recognized tensor kernel, or a read of a local
/// buffer, declines by name.
pub fn grad_dot(
    func: &Function,
    wrt: &[usize],
    interner: &mut Interner,
) -> Result<Function, String> {
    build(func, wrt, interner, Mode::GradDot)
}

/// The **Hessian-vector product**: returns `(gdot, hvp)` where `gdot = grad_dot(func, wrt)` and
/// `hvp = grad(gdot, wrt)`. `hvp` (named `{func.name}_gdot_grad`) takes `func`'s parameters, the
/// direction buffers `v_1..v_k`, then one zeroed output buffer per `wrt` entry, which receives
/// `H·v` restricted to that parameter. Same supported subset as [`grad_dot`].
pub fn hvp(
    func: &Function,
    wrt: &[usize],
    interner: &mut Interner,
) -> Result<(Function, Function), String> {
    let gdot = grad_dot(func, wrt, interner)?;
    let hv = grad(&gdot, wrt, interner)?;
    Ok((gdot, hv))
}

/// `func` with every call to a straight-line (single-block) user function of `program` spliced in,
/// recursively — the form [`grad`] can differentiate across helper functions at every `-O` level
/// (`-O2`'s inliner already does this for small callees, but `-O0`/`-O1` and the inliner's size
/// budget left a loss calling a helper undifferentiable). A callee that is multi-block, recursive or
/// carries vectorized recipes stays a call; if its result is on the gradient path, [`grad`] names
/// it. Returns a clone when nothing is inlined.
pub fn inline_calls(program: &Program, func: &Function) -> Function {
    inline::flatten_calls(program, func).unwrap_or_else(|| func.clone())
}

fn build(
    func: &Function,
    wrt: &[usize],
    interner: &mut Interner,
    mode: Mode,
) -> Result<Function, String> {
    let fname = interner.resolve(func.name).to_string();
    // Second-order (grad-of-grad) through the emitted backward is a capability boundary, and one a
    // reader would otherwise have to infer: the function this transform EMITS returns `Void` and is
    // multi-block, so it satisfies neither half of the contract below. Say that, and name the
    // supported route.
    if fname.ends_with("_grad") && func.ret == MirType::Void {
        return Err(format!(
            "autodiff: `{fname}` is already a generated gradient function; second-order gradients \
             (the gradient of a gradient) are not supported through it — the emitted backward \
             returns void and is multi-block, so it is not itself re-differentiable. For a \
             scalar-path loss, differentiate its directional derivative instead \
             (`wukong_autodiff::grad_dot` / `hvp`, the Hessian-vector product)"
        ));
    }
    if func.blocks.len() != 1 {
        return Err(format!(
            "autodiff: `{}` has {} blocks; only straight-line (single-block) functions are \
             supported in this iteration (run -O1 to mem2reg + simplify-cfg first)",
            interner.resolve(func.name),
            func.blocks.len()
        ));
    }
    for &wi in wrt {
        let p = *func
            .params
            .get(wi)
            .ok_or_else(|| format!("autodiff: wrt index {wi} out of range"))?;
        if *func.value_type(p) != MirType::Ptr {
            return Err(format!(
                "autodiff: wrt parameter {wi} of `{}` is {}, not a buffer pointer",
                interner.resolve(func.name),
                func.value_type(p).display()
            ));
        }
    }
    let ret_ty = match mode {
        Mode::Grad => MirType::Void,
        Mode::GradDot => match func.blocks[0].term {
            Terminator::Ret(Some(v)) => func.value_type(v).clone(),
            _ => return Err("autodiff: forward function must end in `ret <loss>`".into()),
        },
    };
    // Pre-intern the kernel symbols (forward recognition is by Symbol equality — interning dedups,
    // so these match the symbols the forward calls were built with).
    let syms = Syms::new(interner);
    let suffix = match mode {
        Mode::Grad => "grad",
        Mode::GradDot => "gdot",
    };
    let gname = interner.intern(&format!("{fname}_{suffix}"));

    let mut vjp = Vjp::new(func, gname, wrt, syms, interner, mode, ret_ty);
    vjp.scan_kernel_outputs();
    vjp.replay();
    vjp.seed()?;
    vjp.reverse()?;
    Ok(vjp.finish())
}

/// The worker that builds one gradient function. Its three core maps are:
/// - `fwd_to_new`: a forward (old) value -> its replayed value in the new function;
/// - `adj`: a forward value -> the new value holding its *currently accumulated* adjoint;
/// - `grad_buf`: a forward `wrt` parameter -> the new gradient-output parameter it routes to (the
///   direction buffer in [`Mode::GradDot`]).
///
/// The buffer tape adds `buf_adj` / `buf_count` / `contributed` (see `tape.rs`), and `def_op` indexes
/// the forward block so pointer provenance and constant operands can be traced.
struct Vjp<'a> {
    fwd: &'a Function,
    b: wukong_mir::Builder,
    mode: Mode,
    /// The emitted function's return type (`Void`, or the loss type in [`Mode::GradDot`]).
    ret_ty: MirType,
    /// [`Mode::GradDot`]'s running `Σ v·∂L/∂p`.
    gdot_acc: Option<ValueId>,
    /// Defining op of each forward result value (for tracing `gep`/`load` pointer provenance).
    def_op: HashMap<ValueId, &'a Op>,
    fwd_to_new: HashMap<ValueId, ValueId>,
    /// Scalar adjoints: a forward SSA value -> its accumulated-adjoint value in the new function.
    adj: HashMap<ValueId, ValueId>,
    /// Gradient-output parameter for each `wrt` input buffer (the appended, caller-zeroed params).
    grad_buf: HashMap<ValueId, ValueId>,
    /// Interned kernel symbols, for recognizing/emitting tensor-kernel calls (see `tape.rs`).
    syms: Syms,
    /// Buffer adjoints: a forward buffer pointer -> the new buffer holding its accumulated gradient.
    /// The tensor-tape twin of `adj`; set by a consumer's VJP, read by the producer's.
    buf_adj: HashMap<ValueId, ValueId>,
    /// Element count of each intermediate (local `f32` array alloca) forward buffer, for sizing its
    /// gradient buffer.
    buf_count: HashMap<ValueId, u32>,
    /// Buffers that have already received a gradient contribution: the next one accumulates
    /// (`Vjp::begin`/`end` through a scratch, or a GEMM's `beta = 1`) instead of overwriting.
    contributed: HashSet<ValueId>,
    /// The last recognized kernel call that WRITES each buffer (see `tape::scan_kernel_outputs`),
    /// with its forward operands — so a non-differentiated parameter a kernel writes can still get a
    /// gradient sized from the writer's extent.
    kernel_out: HashMap<ValueId, (Kernel, &'a [ValueId])>,
    /// Buffers more than one recognized kernel writes (the write-after-write rule).
    multi_written: HashSet<ValueId>,
    /// Buffers a non-constant float `store` writes: a kernel that overwrites one must zero the
    /// region it wrote, or `diff_store` would hand the stored value an adjoint the kernel destroyed.
    store_written: HashSet<ValueId>,
    /// Heap scratch (`wukong_rt_alloc`) to release before the gradient returns.
    scratch: Vec<ValueId>,
    /// Read-only interner, so a decline can *name* the construct that blocked the gradient (a bare
    /// "unrecognized call" leaves the user with no way to find the offending source line).
    it: &'a Interner,
}

impl<'a> Vjp<'a> {
    fn new(
        fwd: &'a Function,
        gname: Symbol,
        wrt: &[usize],
        syms: Syms,
        it: &'a Interner,
        mode: Mode,
        ret_ty: MirType,
    ) -> Vjp<'a> {
        let mut b = wukong_mir::Builder::new(gname, ret_ty.clone());
        let mut fwd_to_new = HashMap::new();
        // Forward parameters, in order.
        for &p in &fwd.params {
            let np = b.add_param(fwd.value_type(p).clone());
            fwd_to_new.insert(p, np);
        }
        // One gradient-output (or direction) buffer per `wrt` input, appended after the forward
        // params.
        let mut grad_buf = HashMap::new();
        for &wi in wrt {
            let p = fwd.params[wi];
            let gb = b.add_param(MirType::Ptr);
            grad_buf.insert(p, gb);
        }
        let block = &fwd.blocks[0];
        let mut def_op = HashMap::new();
        let mut buf_count = HashMap::new();
        for inst in &block.insts {
            if let Some(r) = inst.result {
                def_op.insert(r, &inst.op);
                // The element count of each local f32 array (its gradient is an f32 buffer of the
                // same size). Other aggregates (byte blobs, pointer tables) carry no gradient.
                if let Op::Alloca {
                    ty: MirType::Array(elem, count),
                    space: wukong_mir::AddrSpace::Local,
                } = &inst.op
                {
                    if **elem == MirType::F32 {
                        buf_count.insert(r, *count);
                    }
                }
            }
        }
        Vjp {
            fwd,
            b,
            mode,
            ret_ty,
            gdot_acc: None,
            def_op,
            fwd_to_new,
            adj: HashMap::new(),
            grad_buf,
            syms,
            buf_adj: HashMap::new(),
            buf_count,
            contributed: HashSet::new(),
            kernel_out: HashMap::new(),
            multi_written: HashSet::new(),
            store_written: HashSet::new(),
            scratch: Vec::new(),
            it,
        }
    }

    fn entry_block(&self) -> &'a BasicBlock {
        &self.fwd.blocks[0]
    }

    /// Emit the forward instructions verbatim (operands remapped to the new value space), recording
    /// `fwd_to_new` so the reverse pass can reference any forward value.
    fn replay(&mut self) {
        // The source block belongs to the borrowed forward function, independently of the new
        // builder. Remap directly from it without a temporary clone of every op and operand list.
        for inst in &self.entry_block().insts {
            let new_op = self.remap_op(&inst.op);
            match inst.result {
                Some(old_r) => {
                    let ty = self.fwd.value_type(old_r).clone();
                    let new_r = self.b.build(ty, new_op);
                    self.fwd_to_new.insert(old_r, new_r);
                }
                None => self.b.build_void(new_op),
            }
        }
    }

    /// Seed the loss adjoint to 1.0 from the `ret <loss>` terminator.
    fn seed(&mut self) -> Result<(), String> {
        let loss = match self.entry_block().term {
            Terminator::Ret(Some(v)) => v,
            _ => {
                return Err("autodiff: forward function must end in `ret <loss>`".into());
            }
        };
        let ty = self.fwd.value_type(loss).clone();
        if !ty.is_float() {
            return Err(format!(
                "autodiff: loss value has type {}, expected a float scalar",
                ty.display()
            ));
        }
        let one = self.b.build(ty.clone(), Op::ConstFloat(1.0, ty));
        self.adj.insert(loss, one);
        Ok(())
    }

    /// Walk the forward instructions in reverse, applying each VJP rule, then release the heap
    /// scratch and return (the directional derivative, in [`Mode::GradDot`]).
    fn reverse(&mut self) -> Result<(), String> {
        for inst in self.entry_block().insts.iter().rev() {
            self.diff_inst(inst)?;
        }
        self.free_scratch();
        match self.mode {
            Mode::Grad => self.b.ret(None),
            Mode::GradDot => {
                let acc = match self.gdot_acc {
                    Some(a) => a,
                    None => {
                        let t = self.ret_ty.clone();
                        self.fconst(0.0, &t)
                    }
                };
                self.b.ret(Some(acc));
            }
        }
        Ok(())
    }

    fn finish(self) -> Function {
        self.b.finish()
    }

    // --- the per-instruction VJP dispatch -------------------------------------------------

    fn diff_inst(&mut self, inst: &Inst) -> Result<(), String> {
        // A recognized tensor-kernel call differentiates through the buffer-adjoint model (tape.rs).
        // Its gradient flows through buffers (memory), not the scalar `adj` map.
        if let Op::Call { func, args } = &inst.op {
            if self.is_kernel(*func) {
                if self.mode == Mode::GradDot {
                    return Err(format!(
                        "autodiff: second-order differentiation (grad_dot / hvp) supports \
                         scalar-path losses only; `{}` is a recognized tensor kernel",
                        self.it.resolve(*func)
                    ));
                }
                return self.diff_kernel_call(*func, args, inst.result);
            }
            // The ML I/O and randomness entries carry an explicit decision (see `tape.rs`).
            match tape::ml_call_gradient(self.it.resolve(*func)) {
                Some(tape::MlCallGradient::Constant) => return Ok(()),
                Some(tape::MlCallGradient::Refuse) => {
                    return Err(format!(
                        "autodiff: no VJP rule for buffer-writing call `{}`",
                        self.it.resolve(*func)
                    ))
                }
                None => {}
            }
            if inst.result.is_none() {
                let name = self.it.resolve(*func);
                // Printing, asserting, freeing and writing a file read values but write no
                // differentiated memory: they are replayed in the forward and contribute nothing.
                if coverage::is_effect_only(name) {
                    return Ok(());
                }
                // Any other void call could write a buffer that feeds the gradient; we cannot prove
                // its contribution is zero, so refuse rather than silently emit a wrong (zero)
                // gradient — naming the callee (the user's only handle on which source construct
                // blocked the gradient) and why, when the coverage table knows.
                return Err(match coverage::not_differentiable_reason(name) {
                    Some(why) => {
                        format!("autodiff: no VJP rule for buffer-writing call `{name}`: {why}")
                    }
                    None => format!("autodiff: no VJP rule for buffer-writing call `{name}`"),
                });
            }
        }
        // A synthesized vector kernel (the autovectorizer's output) writes through buffer pointers
        // with a void result. There is no VJP rule for it; refuse loudly.
        if matches!(inst.op, Op::VecKernelCall { .. }) {
            return Err("autodiff: no VJP rule for a synthesized vector kernel \
                        (veckernel); the vectorizer's output is not differentiable"
                .to_string());
        }
        if let Op::Store { ptr, value } = &inst.op {
            return self.diff_store(*ptr, *value);
        }
        let old_r = match inst.result {
            Some(r) => r,
            None => return Ok(()),
        };
        // No downstream value depends on `old_r` -> its adjoint is zero -> nothing to propagate.
        let g = match self.adj.get(&old_r).copied() {
            Some(g) => g,
            None => return Ok(()),
        };
        let ty = self.fwd.value_type(old_r).clone();

        match &inst.op {
            // Scalar trig is a value-returning operation, not a buffer-tape kernel. Its
            // runtime branch must not force a formerly straight-line loss into CFG/scratch
            // memory the current AD model cannot differentiate. Differentiate the function
            // itself, including large finite arguments, using its paired trig operation.
            Op::Call { func, args } if self.it.resolve(*func) == "wukong_rt_trig_f32" => {
                if args.len() != 2
                    || ty != MirType::F32
                    || self.fwd.value_type(args[0]) != &MirType::F32
                {
                    return Err("autodiff: invalid scalar trig ABI".into());
                }
                let Some(Op::ConstInt(op, MirType::I64)) = self.def_op.get(&args[1]).copied()
                else {
                    return Err("autodiff: scalar trig operation must be a constant i64".into());
                };
                let op = *op as i64;
                // Public runtime op codes: sin=15, cos=16; other codes are identity.
                if !matches!(op, 15 | 16) {
                    self.accum(args[0], g);
                } else {
                    let paired = self.b.build(
                        MirType::I64,
                        Op::ConstInt(if op == 15 { 16 } else { 15 }, MirType::I64),
                    );
                    let x = self.remap(args[0]);
                    let mut derivative = self.b.build(
                        MirType::F32,
                        Op::Call {
                            func: *func,
                            args: vec![x, paired],
                        },
                    );
                    if op == 16 {
                        derivative = self.neg(derivative, &ty);
                    }
                    let contribution = self.fmul(g, derivative, &ty);
                    self.accum(args[0], contribution);
                }
            }
            // The f64 transcendental library (`wukong_rt_math_f64(x, op)`): value-returning and
            // pure, with a closed-form derivative in `x` and the forward result, so like the f32
            // trig call it stays a single-block scalar AD candidate rather than a buffer kernel.
            Op::Call { func, args } if *func == self.syms.math_f64 => {
                if args.len() != 2
                    || ty != MirType::F64
                    || self.fwd.value_type(args[0]) != &MirType::F64
                {
                    return Err("autodiff: invalid f64 math ABI".into());
                }
                let Some(Op::ConstInt(op, MirType::I64)) = self.def_op.get(&args[1]).copied()
                else {
                    return Err("autodiff: f64 math operation must be a constant i64".into());
                };
                let op = *op as i64;
                let x = self.remap(args[0]);
                let r = self.remap(old_r);
                let derivative = self.math_f64_derivative(op, x, r);
                let contribution = self.fmul(g, derivative, &ty);
                self.accum(args[0], contribution);
            }
            // The two-argument twin (`pow`/`atan2`/`hypot`): both partials are closed forms.
            Op::Call { func, args } if *func == self.syms.math2_f64 => {
                if args.len() != 3
                    || ty != MirType::F64
                    || self.fwd.value_type(args[0]) != &MirType::F64
                    || self.fwd.value_type(args[1]) != &MirType::F64
                {
                    return Err("autodiff: invalid f64 two-argument math ABI".into());
                }
                let Some(Op::ConstInt(op, MirType::I64)) = self.def_op.get(&args[2]).copied()
                else {
                    return Err("autodiff: f64 math operation must be a constant i64".into());
                };
                let op = *op as i64;
                let a = self.remap(args[0]);
                let b = self.remap(args[1]);
                let r = self.remap(old_r);
                let (da, db) = self.math2_f64_derivatives(op, a, b, r);
                let ca = self.fmul(g, da, &ty);
                self.accum(args[0], ca);
                let cb = self.fmul(g, db, &ty);
                self.accum(args[1], cb);
            }
            // d(a + b) = (g, g)
            Op::Bin(BinOp::FAdd, a, b) => {
                self.accum(*a, g);
                self.accum(*b, g);
            }
            // d(a - b) = (g, -g)
            Op::Bin(BinOp::FSub, a, b) => {
                self.accum(*a, g);
                let ng = self.neg(g, &ty);
                self.accum(*b, ng);
            }
            // d(a * b) = (g*b, g*a)
            Op::Bin(BinOp::FMul, a, b) => {
                let (an, bn) = (self.remap(*a), self.remap(*b));
                let ca = self.fmul(g, bn, &ty);
                self.accum(*a, ca);
                let cb = self.fmul(g, an, &ty);
                self.accum(*b, cb);
            }
            // d(a / b) = (g/b, -g*a/b^2)
            Op::Bin(BinOp::FDiv, a, b) => {
                let (an, bn) = (self.remap(*a), self.remap(*b));
                let t = self.fdiv(g, bn, &ty); // g/b
                self.accum(*a, t);
                let ab = self.fdiv(an, bn, &ty); // a/b
                let t_ab = self.fmul(t, ab, &ty); // (g/b)*(a/b) = g*a/b^2
                let neg = self.neg(t_ab, &ty);
                self.accum(*b, neg);
            }
            // d(-a) = -g
            Op::Neg(a) => {
                let ng = self.neg(g, &ty);
                self.accum(*a, ng);
            }
            // d(a*b + c) = (g*b, g*a, g)
            Op::Fma(a, b, c) => {
                let (an, bn) = (self.remap(*a), self.remap(*b));
                let ca = self.fmul(g, bn, &ty);
                self.accum(*a, ca);
                let cb = self.fmul(g, an, &ty);
                self.accum(*b, cb);
                self.accum(*c, g);
            }
            // d(sqrt(a)) = g * 0.5 / sqrt(a); reuse the forward result to avoid recomputing the root.
            Op::Sqrt(a) => {
                let r_fwd = self.remap(old_r);
                let half = self.fconst(0.5, &ty);
                let inv = self.fdiv(half, r_fwd, &ty); // 0.5 / sqrt(a)
                let c = self.fmul(g, inv, &ty);
                self.accum(*a, c);
            }
            // select(c, a, b): da = c ? g : 0, db = c ? 0 : g (used by max/min-as-select, e.g. relu)
            Op::Select(c, a, b) => {
                let cn = self.remap(*c);
                let zero = self.fconst(0.0, &ty);
                let ta = self.select(cn, g, zero, &ty);
                self.accum(*a, ta);
                let tb = self.select(cn, zero, g, &ty);
                self.accum(*b, tb);
            }
            // A float load reads a buffer element; route its adjoint to that buffer's gradient.
            Op::Load(p, lty) if lty.is_float() => {
                self.diff_load(*p, g, lty.clone())?;
            }
            // A precision cast is the identity on values, so its VJP is the identity on adjoints —
            // routed back through the INVERSE cast so the contribution carries the operand's type
            // (an f32 adjoint added into an f64 adjoint would be a type-mismatched `fadd`, an ICE
            // in the verifier). Routing the adjoint back through the inverse cast QUANTIZES it to
            // the narrow type (bf16 keeps 8 mantissa bits), which is the arithmetic the forward
            // actually performed, not an approximation this rule introduces.
            Op::Cast(kind, a, _) if matches!(kind, CastKind::FpTrunc | CastKind::FpExt) => {
                let sty = self.fwd.value_type(*a).clone();
                let back = if *kind == CastKind::FpTrunc {
                    CastKind::FpExt
                } else {
                    CastKind::FpTrunc
                };
                let up = self.b.build(sty.clone(), Op::Cast(back, g, sty));
                self.accum(*a, up);
            }
            // An int -> float cast reads an INTEGER operand, which carries no gradient: the adjoint
            // terminates here, exactly like a constant — CONDITIONALLY. A float -> int -> float round
            // trip (`(x as i64) as f64`) would otherwise end the chain at the `sitofp` and never hand
            // the `fptosi` an adjoint at all, and dL/dx would come back zero at exit 0. Truncation
            // has a derivative of zero almost everywhere and is undefined on the grid, so that
            // composition stays a loud refusal.
            Op::Cast(CastKind::SiToFp | CastKind::UiToFp, a, _) => {
                if self.int_derives_from_float(*a) {
                    return Err(format!(
                        "autodiff: no VJP rule for a float -> int -> float round trip (v{} \
                         feeds `{}` through an integer produced by `fptosi`/`fptoui`); \
                         truncation has a derivative of zero almost everywhere and is \
                         undefined on the grid, so terminating the adjoint here would \
                         silently answer zero",
                        old_r.0,
                        op_name(&inst.op)
                    ));
                }
            }
            // A constant is a leaf with no inputs: its adjoint terminates here (constants carry no
            // gradient). This is what lets an accumulator seeded `acc = 0.0` then folded with `fma`
            // differentiate cleanly — the seed const simply absorbs the residual adjoint.
            Op::ConstFloat(..) | Op::ConstInt(..) => {}
            // Any other value-returning call on the gradient path: a user function `inline_calls`
            // could not splice (multi-block, recursive, vectorized), or a runtime routine with no
            // VJP. Name it — "no VJP rule for `call`" leaves the user nothing to grep for.
            Op::Call { func, .. } => {
                let name = self.it.resolve(*func);
                let why = coverage::not_differentiable_reason(name).unwrap_or(
                    "a user function is differentiated only when it is a single straight-line \
                     block `inline_calls` can splice into the caller",
                );
                return Err(format!(
                    "autodiff: no VJP rule for call to `{name}` (value v{} has a non-zero \
                     adjoint): {why}",
                    old_r.0
                ));
            }
            other => {
                return Err(format!(
                    "autodiff: no VJP rule for `{}` (value v{} has a non-zero adjoint)",
                    op_name(other),
                    old_r.0
                ));
            }
        }
        Ok(())
    }

    /// Does the integer value `v` derive, through SSA arithmetic, from a **float -> int** cast?
    ///
    /// A depth-first walk over the forward block's defining ops, following the operands of the
    /// integer forms lowering can put between an `fptosi`/`fptoui` and a later `sitofp` (`Bin`,
    /// `Cmp`, `Neg`, `Not`, `Select`, `Fma`, a further `Cast`, `Splat`, `ExtractLane`). Anything
    /// else — a constant, a parameter, a load, a call result — is an integer LEAF and carries no
    /// gradient.
    ///
    /// The one hole this leaves is a value laundered through MEMORY (`store fptosi(x)` then a
    /// separate `load`), which the straight-line model has no def-use edge for; that composition
    /// still terminates at zero, and it is stated here rather than silently assumed away.
    fn int_derives_from_float(&self, v: ValueId) -> bool {
        let mut stack = vec![v];
        let mut seen = HashSet::new();
        while let Some(cur) = stack.pop() {
            if !seen.insert(cur) {
                continue;
            }
            let Some(op) = self.def_op.get(&cur) else {
                continue; // a parameter or block parameter: a leaf
            };
            match op {
                Op::Cast(CastKind::FpToSi | CastKind::FpToUi, ..) => return true,
                Op::Cast(_, a, _)
                | Op::Neg(a)
                | Op::Not(a)
                | Op::IntUnary(_, a)
                | Op::Splat(a)
                | Op::ExtractLane(a, _) => stack.push(*a),
                Op::Bin(_, a, b) | Op::Cmp(_, a, b) => {
                    stack.push(*a);
                    stack.push(*b);
                }
                Op::Select(c, a, b) | Op::Fma(a, b, c) => {
                    stack.push(*a);
                    stack.push(*b);
                    stack.push(*c);
                }
                _ => {}
            }
        }
        false
    }

    /// Route the adjoint `g` of a value loaded from a buffer into that buffer's gradient,
    /// accumulating (read-add-write) so repeated loads of the same element sum correctly.
    fn diff_load(&mut self, ptr: ValueId, g: ValueId, ty: MirType) -> Result<(), String> {
        if self.mode == Mode::GradDot {
            return self.gdot_load(ptr, g, ty);
        }
        let (base, idx) = self.resolve_access(ptr, &ty)?;
        // `grad_target` SEEDS `buf_adj`, which every producer rule in `tape.rs` gates on — so a
        // scalar read of a kernel's output reaches that kernel's inputs. A parameter outside `wrt`
        // that no kernel writes (the target labels of an MSE loss) has no gradient: drop it.
        let gbuf = match self.grad_target(base)? {
            Some(gb) => gb,
            None => return Ok(()),
        };
        let gptr = match idx {
            Some(iv) => self.b.build(
                MirType::Ptr,
                Op::Gep {
                    ptr: gbuf,
                    index: iv,
                    elem: ty.clone(),
                },
            ),
            None => gbuf,
        };
        let old = self.b.build(ty.clone(), Op::Load(gptr, ty.clone()));
        let sum = self.fadd(old, g, &ty);
        self.b.build_void(Op::Store {
            ptr: gptr,
            value: sum,
        });
        // This buffer has now received a gradient contribution, so a later kernel rule accumulates
        // onto it instead of overwriting it (`Vjp::begin`, `beta_for`).
        self.contributed.insert(base);
        Ok(())
    }

    /// [`Mode::GradDot`]'s load rule: instead of accumulating `g` into a gradient buffer, add
    /// `v[idx]·g` into the directional derivative — `Σ_j <v_j, ∂L/∂p_j>` is linear in the gradient,
    /// so each scalar contribution maps to one term.
    fn gdot_load(&mut self, ptr: ValueId, g: ValueId, ty: MirType) -> Result<(), String> {
        let (base, idx) = self.resolve_access(ptr, &ty)?;
        if !self.fwd.params.contains(&base) {
            return Err(format!(
                "autodiff: second-order differentiation (grad_dot / hvp) supports losses that \
                 read parameters only; v{} reads a local buffer",
                ptr.0
            ));
        }
        let Some(&vb) = self.grad_buf.get(&base) else {
            return Ok(()); // not a differentiated parameter
        };
        if ty != self.ret_ty {
            return Err(format!(
                "autodiff: second-order differentiation needs every differentiated load to have \
                 the loss type {}; a {} load does not",
                self.ret_ty.display(),
                ty.display()
            ));
        }
        let vptr = match idx {
            Some(iv) => self.b.build(
                MirType::Ptr,
                Op::Gep {
                    ptr: vb,
                    index: iv,
                    elem: ty.clone(),
                },
            ),
            None => vb,
        };
        let vv = self.b.build(ty.clone(), Op::Load(vptr, ty.clone()));
        let term = self.fmul(vv, g, &ty);
        let prev = self.gdot_acc;
        let acc = match prev {
            Some(a) => self.fadd(a, term, &ty),
            None => term,
        };
        self.gdot_acc = Some(acc);
        Ok(())
    }

    /// A float store into a buffer whose gradient is live — the memory adjoint rule: the stored
    /// slot's accumulated adjoint flows to the stored VALUE, and the slot's own adjoint becomes zero
    /// (the store overwrote whatever was there, so nothing before it can influence the loss through
    /// that slot). This is what makes `h[0] = x[0]*x[0]; loss = Σ h` differentiable, and what makes
    /// a constant store after a kernel wrote the buffer (`h = relu(x); h[0] = 0.0`) zero that
    /// element's gradient instead of passing the kernel a stale one.
    ///
    /// A store into a buffer nobody differentiates (the loss sink, a zero-init before anything reads
    /// the buffer's gradient) is skipped.
    fn diff_store(&mut self, ptr: ValueId, value: ValueId) -> Result<(), String> {
        if self.buf_adj.is_empty() {
            return Ok(());
        }
        let base = self.store_base(ptr);
        let Some(gbuf) = self.buf_adj.get(&base).copied() else {
            return Ok(());
        };
        let vty = self.fwd.value_type(value).clone();
        let routed = if vty.is_float() {
            self.resolve_access(ptr, &vty)
                .ok()
                .filter(|(b, _)| *b == base)
        } else {
            None
        };
        let Some((_, idx)) = routed else {
            return Err(format!(
                "autodiff: store into buffer v{} has no VJP rule, but that buffer's gradient is \
                 live (it is read downstream) — only a float store through a gep chain of the \
                 stored element type can be routed",
                base.0
            ));
        };
        let gptr = match idx {
            Some(iv) => self.b.build(
                MirType::Ptr,
                Op::Gep {
                    ptr: gbuf,
                    index: iv,
                    elem: vty.clone(),
                },
            ),
            None => gbuf,
        };
        if !self.is_const(value) {
            let gv = self.b.build(vty.clone(), Op::Load(gptr, vty.clone()));
            self.accum(value, gv);
        }
        let z = self.fconst(0.0, &vty);
        self.b.build_void(Op::Store {
            ptr: gptr,
            value: z,
        });
        Ok(())
    }

    /// The base buffer a store pointer addresses: peel a whole chain of `gep`s back to the
    /// underlying alloca / parameter, whatever the index. (Unlike `tape::canon`, which peels only a
    /// whole-buffer `gep ..., 0` because a non-zero offset there is a genuine sub-slice operand,
    /// this must see through an element index: `h[3] = v` still writes into `h`.)
    fn store_base(&self, mut ptr: ValueId) -> ValueId {
        while let Some(Op::Gep { ptr: base, .. }) = self.def_op.get(&ptr) {
            ptr = *base;
        }
        ptr
    }

    /// Whether forward value `v` is a constant (a constant store carries no adjoint to its value).
    fn is_const(&self, v: ValueId) -> bool {
        matches!(
            self.def_op.get(&v),
            Some(Op::ConstFloat(..)) | Some(Op::ConstInt(..))
        )
    }

    /// Trace a load/store pointer to `(base buffer, optional element index in the NEW value space)`.
    ///
    /// The base is a function parameter or a local `f32` array, reached through a chain of `gep`s
    /// whose element type is the accessed type `ty` — so the indices are in the same unit and their
    /// sum addresses the gradient buffer exactly as the chain addressed the forward one (a helper
    /// spliced by [`inline_calls`] that indexes a sub-slice is a two-level chain). A `gep` of a
    /// different element type (a struct byte offset, a row of a nested array) has a different
    /// stride and is refused: routing its index into an `f32` gradient would misplace it silently.
    fn resolve_access(
        &mut self,
        ptr: ValueId,
        ty: &MirType,
    ) -> Result<(ValueId, Option<ValueId>), String> {
        let mut idxs: Vec<ValueId> = Vec::new();
        let mut cur = ptr;
        loop {
            if self.fwd.params.contains(&cur) {
                break;
            }
            if self.buf_count.contains_key(&cur) {
                if *ty != MirType::F32 {
                    return Err(format!(
                        "autodiff: cannot route gradient for a {} access to the local f32 \
                         array v{}",
                        ty.display(),
                        cur.0
                    ));
                }
                break;
            }
            match self.def_op.get(&cur) {
                Some(Op::Gep {
                    ptr: base,
                    index,
                    elem,
                }) if elem == ty => {
                    idxs.push(*index);
                    cur = *base;
                }
                _ => {
                    return Err(format!(
                        "autodiff: cannot route gradient for pointer v{} (not a parameter, a \
                         local f32 array, or a gep chain over one with element type {})",
                        ptr.0,
                        ty.display()
                    ))
                }
            }
        }
        let idx = match idxs.as_slice() {
            [] => None,
            [only] => Some(self.remap(*only)),
            many => {
                let mut acc: Option<ValueId> = None;
                for &i in many {
                    if *self.fwd.value_type(i) != MirType::I64 {
                        return Err(format!(
                            "autodiff: cannot route gradient for pointer v{}: a nested gep index \
                             has type {}, not i64",
                            ptr.0,
                            self.fwd.value_type(i).display()
                        ));
                    }
                    let ni = self.remap(i);
                    acc = Some(match acc {
                        None => ni,
                        Some(a) => self.b.build(MirType::I64, Op::Bin(BinOp::Add, a, ni)),
                    });
                }
                acc
            }
        };
        Ok((cur, idx))
    }

    // --- small emit helpers ---------------------------------------------------------------

    fn remap(&self, v: ValueId) -> ValueId {
        self.fwd_to_new[&v]
    }

    /// Add contribution `contrib` to the adjoint of forward value `old_v` (initialize if first).
    fn accum(&mut self, old_v: ValueId, contrib: ValueId) {
        let ty = self.fwd.value_type(old_v).clone();
        match self.adj.get(&old_v).copied() {
            Some(prev) => {
                let s = self.fadd(prev, contrib, &ty);
                self.adj.insert(old_v, s);
            }
            None => {
                self.adj.insert(old_v, contrib);
            }
        }
    }

    fn fadd(&mut self, l: ValueId, r: ValueId, ty: &MirType) -> ValueId {
        self.b.build(ty.clone(), Op::Bin(BinOp::FAdd, l, r))
    }
    fn fmul(&mut self, l: ValueId, r: ValueId, ty: &MirType) -> ValueId {
        self.b.build(ty.clone(), Op::Bin(BinOp::FMul, l, r))
    }
    fn fdiv(&mut self, l: ValueId, r: ValueId, ty: &MirType) -> ValueId {
        self.b.build(ty.clone(), Op::Bin(BinOp::FDiv, l, r))
    }
    fn neg(&mut self, v: ValueId, ty: &MirType) -> ValueId {
        self.b.build(ty.clone(), Op::Neg(v))
    }
    fn select(&mut self, c: ValueId, a: ValueId, b: ValueId, ty: &MirType) -> ValueId {
        self.b.build(ty.clone(), Op::Select(c, a, b))
    }
    fn fconst(&mut self, x: f64, ty: &MirType) -> ValueId {
        self.b.build(ty.clone(), Op::ConstFloat(x, ty.clone()))
    }

    /// One call of the f64 transcendental library in the new function: `wukong_rt_math_f64(x, op)`.
    fn math_f64_call(&mut self, op: i64, x: ValueId) -> ValueId {
        let code = self
            .b
            .build(MirType::I64, Op::ConstInt(op as i128, MirType::I64));
        self.b.build(
            MirType::F64,
            Op::Call {
                func: self.syms.math_f64,
                args: vec![x, code],
            },
        )
    }

    /// `wukong_rt_math2_f64(a, b, op)` in the new function.
    fn math2_f64_call(&mut self, op: i64, a: ValueId, b: ValueId) -> ValueId {
        let code = self
            .b
            .build(MirType::I64, Op::ConstInt(op as i128, MirType::I64));
        self.b.build(
            MirType::F64,
            Op::Call {
                func: self.syms.math2_f64,
                args: vec![a, b, code],
            },
        )
    }

    /// `f'(x)` for the f64 library code `op` (the runtime's public `VM_*` codes), given the replayed
    /// argument `x` and the replayed forward result `r = f(x)`, which the closed forms reuse where
    /// they can (`exp' = exp`, `tanh' = 1 − tanh²`, …). A code the runtime treats as the identity
    /// has derivative 1, matching.
    fn math_f64_derivative(&mut self, op: i64, x: ValueId, r: ValueId) -> ValueId {
        let ty = MirType::F64;
        let one = self.fconst(1.0, &ty);
        match op {
            // exp' = exp
            0 => r,
            // log' = 1/x
            1 => self.fdiv(one, x, &ty),
            // tanh' = 1 − tanh²
            2 => {
                let r2 = self.fmul(r, r, &ty);
                self.b.build(ty.clone(), Op::Bin(BinOp::FSub, one, r2))
            }
            // sin' = cos
            15 => self.math_f64_call(16, x),
            // cos' = −sin
            16 => {
                let s = self.math_f64_call(15, x);
                self.neg(s, &ty)
            }
            // erf' = (2/√π)·exp(−x²)
            17 => {
                let x2 = self.fmul(x, x, &ty);
                let nx2 = self.neg(x2, &ty);
                let e = self.math_f64_call(0, nx2);
                let c = self.fconst(std::f64::consts::FRAC_2_SQRT_PI, &ty);
                self.fmul(c, e, &ty)
            }
            // exp2' = exp2·ln2
            18 => {
                let c = self.fconst(std::f64::consts::LN_2, &ty);
                self.fmul(r, c, &ty)
            }
            // log2' = 1/(x·ln2)
            19 => {
                let c = self.fconst(std::f64::consts::LN_2, &ty);
                let d = self.fmul(x, c, &ty);
                self.fdiv(one, d, &ty)
            }
            // sinh' = cosh
            20 => self.math_f64_call(21, x),
            // cosh' = sinh
            21 => self.math_f64_call(20, x),
            // asinh' = 1/√(x²+1)
            22 => {
                let x2 = self.fmul(x, x, &ty);
                let s = self.fadd(x2, one, &ty);
                let root = self.b.build(ty.clone(), Op::Sqrt(s));
                self.fdiv(one, root, &ty)
            }
            // acosh' = 1/√(x²−1)
            23 => {
                let x2 = self.fmul(x, x, &ty);
                let s = self.b.build(ty.clone(), Op::Bin(BinOp::FSub, x2, one));
                let root = self.b.build(ty.clone(), Op::Sqrt(s));
                self.fdiv(one, root, &ty)
            }
            // atanh' = 1/(1−x²)
            24 => {
                let x2 = self.fmul(x, x, &ty);
                let d = self.b.build(ty.clone(), Op::Bin(BinOp::FSub, one, x2));
                self.fdiv(one, d, &ty)
            }
            // atan' = 1/(1+x²)
            25 => {
                let x2 = self.fmul(x, x, &ty);
                let d = self.fadd(one, x2, &ty);
                self.fdiv(one, d, &ty)
            }
            // expm1' = exp — computed directly, not as `expm1 + 1`: for x ≪ 0 the forward result is
            // −1 + e^x and adding 1 back cancels to the ≈1e-16 absolute error of that sum (at
            // x = −20.5 the relative error of `r + 1` is 3e-9; `exp(x)` stays ≤ 1 ULP).
            26 => self.math_f64_call(0, x),
            // log1p' = 1/(1+x)
            27 => {
                let d = self.fadd(one, x, &ty);
                self.fdiv(one, d, &ty)
            }
            // exp10' = exp10·ln10
            28 => {
                let c = self.fconst(std::f64::consts::LN_10, &ty);
                self.fmul(r, c, &ty)
            }
            // log10' = 1/(x·ln10)
            29 => {
                let c = self.fconst(std::f64::consts::LN_10, &ty);
                let d = self.fmul(x, c, &ty);
                self.fdiv(one, d, &ty)
            }
            // tan' = 1 + tan²
            32 => {
                let r2 = self.fmul(r, r, &ty);
                self.fadd(one, r2, &ty)
            }
            // asin' = 1/√(1−x²)
            33 => {
                let x2 = self.fmul(x, x, &ty);
                let s = self.b.build(ty.clone(), Op::Bin(BinOp::FSub, one, x2));
                let root = self.b.build(ty.clone(), Op::Sqrt(s));
                self.fdiv(one, root, &ty)
            }
            // acos' = −1/√(1−x²)
            34 => {
                let x2 = self.fmul(x, x, &ty);
                let s = self.b.build(ty.clone(), Op::Bin(BinOp::FSub, one, x2));
                let root = self.b.build(ty.clone(), Op::Sqrt(s));
                let d = self.fdiv(one, root, &ty);
                self.neg(d, &ty)
            }
            // cbrt' = 1/(3·cbrt²)
            35 => {
                let r2 = self.fmul(r, r, &ty);
                let three = self.fconst(3.0, &ty);
                let d = self.fmul(three, r2, &ty);
                self.fdiv(one, d, &ty)
            }
            // The runtime returns `x` for any other code.
            _ => one,
        }
    }

    /// `(∂f/∂a, ∂f/∂b)` for the two-argument f64 library code `op` (the runtime's `VM2_*` codes),
    /// given the replayed arguments and the replayed forward result `r = f(a, b)`.
    fn math2_f64_derivatives(
        &mut self,
        op: i64,
        a: ValueId,
        b: ValueId,
        r: ValueId,
    ) -> (ValueId, ValueId) {
        let ty = MirType::F64;
        match op {
            // pow(a, b): ∂/∂a = b·a^(b−1), ∂/∂b = ln(a)·a^b
            0 => {
                let one = self.fconst(1.0, &ty);
                let bm1 = self.b.build(ty.clone(), Op::Bin(BinOp::FSub, b, one));
                let p = self.math2_f64_call(0, a, bm1);
                let da = self.fmul(b, p, &ty);
                let la = self.math_f64_call(1, a);
                let db = self.fmul(la, r, &ty);
                (da, db)
            }
            // atan2(a, b): ∂/∂a = b/(a²+b²), ∂/∂b = −a/(a²+b²)
            1 => {
                let a2 = self.fmul(a, a, &ty);
                let b2 = self.fmul(b, b, &ty);
                let den = self.fadd(a2, b2, &ty);
                let da = self.fdiv(b, den, &ty);
                let q = self.fdiv(a, den, &ty);
                let db = self.neg(q, &ty);
                (da, db)
            }
            // hypot(a, b): ∂/∂a = a/r, ∂/∂b = b/r
            2 => {
                let da = self.fdiv(a, r, &ty);
                let db = self.fdiv(b, r, &ty);
                (da, db)
            }
            // The runtime returns `a` for any other code.
            _ => (self.fconst(1.0, &ty), self.fconst(0.0, &ty)),
        }
    }

    /// Clone a forward op with every operand value remapped into the new value space.
    fn remap_op(&self, op: &Op) -> Op {
        map_operands(op, |v| self.remap(v))
    }
}

/// Clone `op` with every operand `ValueId` passed through `f` (the `kernel` index of a
/// `VecKernelCall` is a function-local table index, not a value, and is copied unchanged).
pub(crate) fn map_operands(op: &Op, mut f: impl FnMut(ValueId) -> ValueId) -> Op {
    match op {
        Op::ConstInt(..)
        | Op::ConstFloat(..)
        | Op::Alloca { .. }
        | Op::Barrier
        | Op::FuncAddr(..)
        | Op::GlobalAddr(..) => op.clone(),
        Op::Bin(b, l, r) => Op::Bin(*b, f(*l), f(*r)),
        Op::Cmp(c, l, r) => Op::Cmp(*c, f(*l), f(*r)),
        Op::Neg(v) => Op::Neg(f(*v)),
        Op::Not(v) => Op::Not(f(*v)),
        Op::Cast(k, v, t) => Op::Cast(*k, f(*v), t.clone()),
        Op::Select(c, a, b) => Op::Select(f(*c), f(*a), f(*b)),
        Op::Load(p, t) => Op::Load(f(*p), t.clone()),
        Op::Store { ptr, value } => Op::Store {
            ptr: f(*ptr),
            value: f(*value),
        },
        Op::Gep { ptr, index, elem } => Op::Gep {
            ptr: f(*ptr),
            index: f(*index),
            elem: elem.clone(),
        },
        Op::Call { func, args } => Op::Call {
            func: *func,
            args: args.iter().map(|a| f(*a)).collect(),
        },
        Op::Splat(v) => Op::Splat(f(*v)),
        Op::ExtractLane(v, k) => Op::ExtractLane(f(*v), *k),
        Op::Iota(ty) => Op::Iota(ty.clone()),
        Op::Fma(a, b, c) => Op::Fma(f(*a), f(*b), f(*c)),
        Op::Sqrt(v) => Op::Sqrt(f(*v)),
        Op::Round(m, v) => Op::Round(*m, f(*v)),
        Op::IntUnary(o, v) => Op::IntUnary(*o, f(*v)),
        Op::VecKernelCall {
            kernel,
            ptrs,
            scalars,
            n,
        } => Op::VecKernelCall {
            kernel: *kernel,
            ptrs: f(*ptrs),
            scalars: f(*scalars),
            n: f(*n),
        },
    }
}

/// A short op name for diagnostics (the `Debug` of `Op` is verbose and includes operands).
fn op_name(op: &Op) -> &'static str {
    match op {
        Op::ConstInt(..) => "const.int",
        Op::ConstFloat(..) => "const.float",
        Op::Bin(b, ..) => b.name(),
        Op::Cmp(..) => "cmp",
        Op::Neg(..) => "neg",
        Op::Not(..) => "not",
        Op::Cast(k, ..) => k.name(),
        Op::Select(..) => "select",
        Op::Alloca { .. } => "alloca",
        Op::Barrier => "barrier",
        Op::Load(..) => "load",
        Op::Store { .. } => "store",
        Op::Gep { .. } => "gep",
        Op::Call { .. } => "call",
        Op::FuncAddr(..) => "func_addr",
        Op::GlobalAddr(..) => "global_addr",
        Op::Splat(..) => "splat",
        Op::ExtractLane(..) => "extractlane",
        Op::Iota(..) => "iota",
        Op::Fma(..) => "fma",
        Op::Sqrt(..) => "sqrt",
        Op::Round(..) => "round",
        Op::IntUnary(o, _) => o.name(),
        Op::VecKernelCall { .. } => "veckernel",
    }
}

#[cfg(test)]
mod tests;
