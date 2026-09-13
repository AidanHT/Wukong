//! `wukong_autodiff` — reverse-mode automatic differentiation as a MIR->MIR transform.
//!
//! Wukong's forward path turns a `.wk` model into tuned kernels; this crate is the **backward
//! path**. Given a forward MIR [`Function`] that computes a scalar loss, [`grad`] emits a *new*
//! MIR function that runs the same forward pass and *also* accumulates the gradient of that loss
//! with respect to designated input buffers — the vector-Jacobian product (VJP), walked in reverse
//! topological order over the forward instructions.
//!
//! # The model (this iteration)
//!
//! The transform consumes **straight-line, SSA-form** MIR: a single basic block whose only memory
//! traffic is reads/writes of buffer (`Ptr`) parameters. This is exactly the shape the optimizer
//! produces after `mem2reg` (scalar locals become SSA values; the only `load`/`store`/`gep` left
//! address tensor buffers passed as pointer parameters). Differentiating that form needs no memory
//! model for temporaries — every intermediate is an SSA value, available to the reverse pass by
//! construction.
//!
//! Contract for the forward function `f`:
//! - exactly one basic block, terminated by `ret <loss>` (the scalar being differentiated);
//! - every parameter is a `Ptr` (a buffer); inputs are read with `load`/`gep`, the loss is *also*
//!   stored to an output buffer so a caller can observe it (the transform skips a store whose
//!   destination buffer has no live gradient — the loss sink, and constant array zero-init — and
//!   *refuses* a store into a buffer a downstream recognized kernel reads, which has no VJP rule);
//! - `wrt` lists the parameter indices to differentiate. For each, the emitted gradient function
//!   appends one `Ptr` parameter that receives `d loss / d input` (accumulated, so the caller must
//!   pass a zeroed buffer).
//!
//! # Correctness gate
//!
//! Every rule is gated by **finite differences** (see the test module): perturb each input element
//! by +-eps, evaluate the loss, and assert the central difference `(L(x+eps) - L(x-eps)) / 2eps`
//! matches the analytic gradient the transform emits, over the full gradient buffer. For the scalar
//! rules the reference forward pass runs in `f64` (via `wukong_interp::run_kernel_f64`) so eps-noise
//! stays far below the tolerance. The buffer-tape rules (`tape.rs`) can only run in `f32` — the
//! runtime kernels they call are f32-only — so there the loose f32 finite difference is the sanity
//! check and a tight **f64 closed form** is the real gate wherever one exists (the smooth activations
//! whose kernel evaluates an f32 polynomial have none, and are FD-gated only).
//! A wrong gradient is a miscompile — it must be fixed before any throughput counts.

use std::collections::{HashMap, HashSet};
use wukong_mir::{BasicBlock, BinOp, CastKind, Function, Inst, MirType, Op, Terminator, ValueId};
use wukong_span::{Interner, Symbol};

pub mod optim;
mod tape;
use tape::Syms;

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
    let fname = interner.resolve(func.name).to_string();
    // Second-order (grad-of-grad) is a clean capability boundary, not a defect — but it is one a
    // reader has to infer today. The function this transform EMITS returns `Void` and is
    // multi-block (`tape.rs` synthesizes counted loops for the norm and activation backwards), so
    // it satisfies neither half of the contract below. Say that, instead of letting the generic
    // "has N blocks" message stand in for it; a boundary you cannot discover from the tool reads
    // as a bug.
    if fname.ends_with("_grad") && func.ret == MirType::Void {
        return Err(format!(
            "autodiff: `{fname}` is already a generated gradient function; second-order gradients \
             (the gradient of a gradient) are not supported — the emitted backward returns void \
             and is multi-block, so it is not itself re-differentiable"
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

    let base = fname;
    // Pre-intern the kernel symbols (forward recognition is by Symbol equality — interning dedups,
    // so these match the symbols the forward calls were built with).
    let syms = Syms::new(interner);
    let gname = interner.intern(&format!("{base}_grad"));

    let mut vjp = Vjp::new(func, gname, wrt, syms, interner);
    vjp.scan_kernel_outputs();
    vjp.replay();
    vjp.seed()?;
    vjp.reverse()?;
    Ok(vjp.finish())
}

/// The worker that builds one gradient function. Its three core maps are:
/// - `fwd_to_new`: a forward (old) value -> its replayed value in the new function;
/// - `adj`: a forward value -> the new value holding its *currently accumulated* adjoint;
/// - `grad_buf`: a forward `wrt` parameter -> the new gradient-output parameter it routes to.
///
/// The buffer tape adds `buf_adj` / `buf_count` / `contributed` (see `tape.rs`), and `def_op` indexes
/// the forward block so pointer provenance and constant operands can be traced.
struct Vjp<'a> {
    fwd: &'a Function,
    b: wukong_mir::Builder,
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
    /// The tensor-tape twin of `adj`; set by a consumer's VJP, read by the producer's VJP.
    buf_adj: HashMap<ValueId, ValueId>,
    /// Element count of each intermediate (alloca'd) forward buffer, for sizing its gradient buffer.
    buf_count: HashMap<ValueId, u32>,
    /// Buffers that have already received a gradient contribution — a second one would need
    /// accumulation (handled per-op for matmul via `beta`; a loud error elsewhere until supported).
    contributed: HashSet<ValueId>,
    /// Buffers a recognized kernel WRITES in the forward pass (see `tape::scan_kernel_outputs`).
    /// `diff_load` needs it to tell a read-only input, whose adjoint may be dropped, from a kernel
    /// output, whose dropped adjoint would silently zero that kernel's inputs.
    kernel_written: HashSet<ValueId>,
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
    ) -> Vjp<'a> {
        // The gradient function returns nothing — it writes gradients into its appended buffers.
        let mut b = wukong_mir::Builder::new(gname, MirType::Void);
        let mut fwd_to_new = HashMap::new();
        // Forward parameters, in order.
        for &p in &fwd.params {
            let np = b.add_param(fwd.value_type(p).clone());
            fwd_to_new.insert(p, np);
        }
        // One gradient-output buffer per `wrt` input, appended after the forward params.
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
                // Record the element count of each intermediate buffer (an array alloca), so its
                // gradient buffer can be sized to match.
                if let Op::Alloca {
                    ty: MirType::Array(_, count),
                    space: wukong_mir::AddrSpace::Local,
                } = &inst.op
                {
                    buf_count.insert(r, *count);
                }
            }
        }
        Vjp {
            fwd,
            b,
            def_op,
            fwd_to_new,
            adj: HashMap::new(),
            grad_buf,
            syms,
            buf_adj: HashMap::new(),
            buf_count,
            contributed: HashSet::new(),
            kernel_written: HashSet::new(),
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

    /// Walk the forward instructions in reverse, applying each VJP rule.
    fn reverse(&mut self) -> Result<(), String> {
        for inst in self.entry_block().insts.iter().rev() {
            self.diff_inst(inst)?;
        }
        self.b.ret(None);
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
                return self.diff_kernel_call(*func, args, inst.result);
            }
            // An unrecognized call that writes buffers (void result) could feed a downstream
            // gradient; we cannot prove its contribution is zero, so refuse rather than silently
            // emit a wrong (zero) gradient. (Value-returning unknown calls fall through to the
            // scalar path, which errors only if the value actually has a non-zero adjoint.)
            // Name the callee: it is the only handle the user has on which source construct
            // blocked the gradient (e.g. `wukong_sgemm_nt_epi` = a fused FFN epilogue).
            if inst.result.is_none() {
                return Err(format!(
                    "autodiff: no VJP rule for buffer-writing call `{}`",
                    self.it.resolve(*func)
                ));
            }
        }
        // A synthesized vector kernel (the autovectorizer's output) writes through buffer pointers
        // with a void result, so it would fall through to the `result.is_none() -> Ok(())` skip
        // below and contribute nothing — a silently-zero gradient for every buffer it writes.
        // There is no VJP rule for it; refuse loudly.
        if matches!(inst.op, Op::VecKernelCall { .. }) {
            return Err("autodiff: no VJP rule for a synthesized vector kernel \
                        (veckernel); the vectorizer's output is not differentiable"
                .to_string());
        }
        // Stores are normally the loss sink (or an output write) and propagate no adjoint in the
        // SSA-temporaries model. But a store whose destination buffer already has a *live* gradient
        // — i.e. the reverse walk has already seen a recognized kernel READ that buffer — is on the
        // gradient path, and skipping it silently drops the contribution (an all-zero gradient, the
        // one thing this crate promises never to emit). Read-after-write through intermediate memory
        // has no VJP rule, so refuse. Constant stores stay skippable: `[0.0; N]` array zero-init
        // carries no gradient, and the loss sink writes into a buffer no kernel reads.
        if let Op::Store { ptr, value } = &inst.op {
            // Without a live buffer adjoint, no destination can trigger this refusal. Constant
            // initialization is also always skippable, regardless of pointer-chain depth.
            if self.buf_adj.is_empty() {
                return Ok(());
            }
            let is_const = matches!(
                self.def_op.get(value),
                Some(Op::ConstFloat(..)) | Some(Op::ConstInt(..))
            );
            if is_const {
                return Ok(());
            }
            let base = self.store_base(*ptr);
            if self.buf_adj.contains_key(&base) {
                return Err(format!(
                    "autodiff: store into buffer v{} has no VJP rule, but that buffer's gradient \
                     is live (a recognized kernel reads it downstream) — read-after-write through \
                     intermediate memory is not supported",
                    base.0
                ));
            }
            return Ok(());
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
            // A float load reads an input element; route its adjoint to the matching gradient buffer.
            Op::Load(p, lty) if lty.is_float() => {
                self.diff_load(*p, g, lty.clone())?;
            }
            // A precision cast is the identity on values, so its VJP is the identity on adjoints —
            // routed back through the INVERSE cast so the contribution carries the operand's type
            // (an f32 adjoint added into an f64 adjoint would be a type-mismatched `fadd`, an ICE
            // in the verifier). This is what makes a mixed-precision loss — the whole point of the
            // bf16/f16 GEMM family — differentiable: `(x as bf16)` no longer stops the chain.
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
            // terminates here, exactly like a constant. (This is what makes `x / (n as f32)`
            // differentiable when `n` does not const-fold.) The float -> int direction keeps the
            // loud refusal below: its derivative is zero almost everywhere and undefined on the
            // grid, and silently answering zero is the one thing this crate promises never to do.
            Op::Cast(CastKind::SiToFp | CastKind::UiToFp, ..) => {}
            // A constant is a leaf with no inputs: its adjoint terminates here (constants carry no
            // gradient). This is what lets an accumulator seeded `acc = 0.0` then folded with `fma`
            // differentiate cleanly — the seed const simply absorbs the residual adjoint.
            Op::ConstFloat(..) | Op::ConstInt(..) => {}
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

    /// Route the adjoint `g` of a value loaded from a buffer into that buffer's gradient output,
    /// accumulating (read-add-write) so repeated loads of the same element sum correctly.
    fn diff_load(&mut self, ptr: ValueId, g: ValueId, ty: MirType) -> Result<(), String> {
        let (param, idx) = self.resolve_buffer_ptr(ptr)?;
        // Route through `grad_target`, NOT `grad_buf` directly. Both return the same buffer for a
        // `wrt` parameter, but only `grad_target` SEEDS `buf_adj` for it — and every producer rule
        // in `tape.rs` gates on `buf_adj[out]`, returning `Ok(())` and emitting nothing when it is
        // absent. So `k(x -> h); s = h[0]; loss = f(s)` with `h` a `wrt` buffer parameter used to
        // write into h's gradient and then find no output adjoint for the kernel that produced h:
        // dL/dx came back all zeros, exit 0, no diagnostic — the one thing this crate promises
        // never to do (docs/roadmap.md documents the case; nothing guarded it). With the seed in
        // place the producer finds its `dout` and either chains correctly or hits one of the
        // existing loud refusals (`single()`, the store guard).
        let gbuf = match self.grad_target(param)? {
            Some(gb) => gb,
            None => {
                // Not a `wrt` input, so its gradient is dropped — correct for a read-only input
                // (the target labels of an MSE loss). But when a recognized kernel WRITES this
                // buffer, dropping the adjoint breaks the chain to that kernel's own inputs and
                // silently zeroes them, and a parameter has no known element count to size a
                // scratch gradient from. Refuse, naming the fix.
                if self.kernel_written.contains(&param) {
                    return Err(format!(
                        "autodiff: scalar load of buffer v{p} reads the output of a recognized \
                         kernel, but v{p} is not in the differentiated (`wrt`) set, so its adjoint \
                         has nowhere to go and that kernel's inputs would silently receive a zero \
                         gradient — add v{p} to the wrt list",
                        p = param.0
                    ));
                }
                return Ok(());
            }
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
        // This buffer has now received a gradient contribution. The scalar path above ACCUMULATES
        // (read-add-write) while the kernel path (`fill_buf`/`velem_scale`/`velem_affine`) OVERWRITES,
        // so recording it is what stops a later kernel rule from clobbering this element: `single()`
        // turns the mix into a loud error, and `beta_for` switches a matmul adjoint to accumulate
        // (correct, since the ABI has the caller pass a zeroed gradient buffer). Repeated scalar
        // loads of the same buffer are unaffected — this ignores the "already present" result, and
        // they accumulate correctly by construction.
        self.contributed.insert(param);
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

    /// Trace a load pointer to `(input parameter, optional element index in the new value space)`.
    /// Handles a bare parameter (element 0) and a one-level `gep` of a parameter (the flat-index
    /// form a `buf[i]` lowers to). Anything deeper is rejected so a gradient is never misrouted.
    fn resolve_buffer_ptr(&self, ptr: ValueId) -> Result<(ValueId, Option<ValueId>), String> {
        if self.fwd.params.contains(&ptr) {
            return Ok((ptr, None));
        }
        match self.def_op.get(&ptr) {
            Some(Op::Gep {
                ptr: base, index, ..
            }) if self.fwd.params.contains(base) => Ok((*base, Some(self.remap(*index)))),
            _ => Err(format!(
                "autodiff: cannot route gradient for load pointer v{} (not a parameter or a \
                 one-level gep of a parameter)",
                ptr.0
            )),
        }
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

    /// Clone a forward op with every operand value remapped into the new value space.
    fn remap_op(&self, op: &Op) -> Op {
        match op {
            Op::ConstInt(..)
            | Op::ConstFloat(..)
            | Op::Alloca { .. }
            | Op::Barrier
            | Op::FuncAddr(..)
            | Op::GlobalAddr(..) => op.clone(),
            Op::Bin(b, l, r) => Op::Bin(*b, self.remap(*l), self.remap(*r)),
            Op::Cmp(c, l, r) => Op::Cmp(*c, self.remap(*l), self.remap(*r)),
            Op::Neg(v) => Op::Neg(self.remap(*v)),
            Op::Not(v) => Op::Not(self.remap(*v)),
            Op::Cast(k, v, t) => Op::Cast(*k, self.remap(*v), t.clone()),
            Op::Select(c, a, b) => Op::Select(self.remap(*c), self.remap(*a), self.remap(*b)),
            Op::Load(p, t) => Op::Load(self.remap(*p), t.clone()),
            Op::Store { ptr, value } => Op::Store {
                ptr: self.remap(*ptr),
                value: self.remap(*value),
            },
            Op::Gep { ptr, index, elem } => Op::Gep {
                ptr: self.remap(*ptr),
                index: self.remap(*index),
                elem: elem.clone(),
            },
            Op::Call { func, args } => Op::Call {
                func: *func,
                args: args.iter().map(|a| self.remap(*a)).collect(),
            },
            Op::Splat(v) => Op::Splat(self.remap(*v)),
            Op::ExtractLane(v, k) => Op::ExtractLane(self.remap(*v), *k),
            Op::Iota(ty) => Op::Iota(ty.clone()),
            Op::Fma(a, b, c) => Op::Fma(self.remap(*a), self.remap(*b), self.remap(*c)),
            Op::Sqrt(v) => Op::Sqrt(self.remap(*v)),
            Op::Round(m, v) => Op::Round(*m, self.remap(*v)),
            Op::VecKernelCall {
                kernel,
                ptrs,
                scalars,
                n,
            } => Op::VecKernelCall {
                kernel: *kernel,
                ptrs: self.remap(*ptrs),
                scalars: self.remap(*scalars),
                n: self.remap(*n),
            },
        }
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
        Op::VecKernelCall { .. } => "veckernel",
    }
}

#[cfg(test)]
mod tests;
