//! The MIR verifier — checks well-formedness invariants. A failure here is an *internal compiler
//! error* (a bug in a pass), not a user error, so violations are returned as plain strings.
//!
//! It runs after MIR construction, between optimization passes (under `--verify-each`), and
//! before backend entry. Checks: every value used is defined; block-parameter arities and types
//! line up across edges; per-operation operand/result types are consistent; and terminators are
//! type-correct (including the function return type).

use std::collections::{HashMap, HashSet};

use crate::{BinOp, Function, MirType, Op, Program, Terminator, ValueId};
use wukong_span::Symbol;

/// Declared signatures (parameter types, return type) of the functions defined in a program, so a
/// call can be checked against the function it names. Only callees with a MIR body are in here:
/// `print` and the `wukong_*` runtime kernels are external symbols whose signatures live outside
/// the IR, and a call to one is left unchecked.
type FnSigs = HashMap<Symbol, (Vec<MirType>, MirType)>;

/// Verify every function in a program. Returns a list of human-readable problems (empty = ok).
/// Unlike [`verify_function`] this also cross-checks each call against its callee's signature,
/// which needs the whole program in hand.
pub fn verify_program(p: &Program) -> Vec<String> {
    let mut sigs: FnSigs = HashMap::new();
    for f in &p.funcs {
        // A parameter outside the value arena is reported separately; skip registering such a
        // function rather than indexing past the end here.
        let params: Option<Vec<MirType>> = f
            .params
            .iter()
            .map(|v| f.value_types.get(v.0 as usize).cloned())
            .collect();
        if let Some(params) = params {
            // `Program::function` resolves a name to the FIRST function carrying it; mirror that.
            sigs.entry(f.name).or_insert((params, f.ret.clone()));
        }
    }
    let mut errors = Vec::new();
    for f in &p.funcs {
        errors.extend(verify(f, Some(&sigs)));
    }
    errors
}

/// Verify a single function. Calls are checked for operand definition only — cross-checking a call
/// against its callee needs [`verify_program`].
pub fn verify_function(f: &Function) -> Vec<String> {
    verify(f, None)
}

fn verify(f: &Function, sigs: Option<&FnSigs>) -> Vec<String> {
    let mut v = Verifier {
        f,
        sigs,
        errors: Vec::new(),
        defined: HashSet::new(),
    };
    v.run();
    v.errors
}

struct Verifier<'a> {
    f: &'a Function,
    /// Callee signatures, when the verifier was entered with a whole program.
    sigs: Option<&'a FnSigs>,
    errors: Vec<String>,
    defined: HashSet<u32>,
}

impl Verifier<'_> {
    fn err(&mut self, msg: impl Into<String>) {
        self.errors.push(msg.into());
    }

    fn run(&mut self) {
        // Collect all defined values (block params + instruction results).
        for b in &self.f.blocks {
            for p in &b.params {
                self.defined.insert(p.0);
            }
            for inst in &b.insts {
                if let Some(r) = inst.result {
                    self.defined.insert(r.0);
                }
            }
        }

        let nblocks = self.f.blocks.len() as u32;
        for (bi, b) in self.f.blocks.iter().enumerate() {
            if b.id.0 != bi as u32 {
                self.err(format!("block {} has inconsistent id {}", bi, b.id.0));
            }
            for inst in &b.insts {
                self.check_op(&inst.op, inst.result);
            }
            self.check_term(&b.term, nblocks);
        }
    }

    fn ty(&self, v: ValueId) -> Option<&MirType> {
        self.f.value_types.get(v.0 as usize)
    }

    fn use_val(&mut self, v: ValueId) -> bool {
        if !self.defined.contains(&v.0) {
            self.err(format!("use of undefined value {v:?}"));
            false
        } else {
            true
        }
    }

    fn expect_ty(&mut self, v: ValueId, want: &MirType, ctx: &str) {
        if let Some(t) = self.ty(v) {
            if t != want {
                self.err(format!(
                    "{ctx}: {v:?} has type {} but expected {}",
                    t.display(),
                    want.display()
                ));
            }
        }
    }

    fn result_ty(&self, result: Option<ValueId>) -> Option<MirType> {
        result.and_then(|r| self.ty(r).cloned())
    }

    fn check_op(&mut self, op: &Op, result: Option<ValueId>) {
        // `Store` never produces a result; `Call` may be void (e.g. the `print` intrinsic) or
        // value-producing. Every other op must produce exactly one result.
        let must_produce = !matches!(op, Op::Store { .. } | Op::Call { .. } | Op::VecKernelCall { .. });
        if must_produce && result.is_none() {
            self.err(format!("operation {op:?} must produce a result value"));
        }
        if matches!(op, Op::Store { .. }) && result.is_some() {
            self.err("Store must not produce a result value".to_string());
        }

        match op {
            Op::ConstInt(_, ty) => {
                if !ty.is_int() {
                    self.err(format!("const int has non-integer type {}", ty.display()));
                }
                self.check_result_is(result, ty);
            }
            Op::ConstFloat(_, ty) => {
                if !ty.is_float() {
                    self.err(format!("const float has non-float type {}", ty.display()));
                }
                self.check_result_is(result, ty);
            }
            Op::Bin(b, l, r) => {
                if self.use_val(*l) & self.use_val(*r) {
                    if let Some(res) = self.result_ty(result) {
                        self.expect_ty(*l, &res, b.name());
                        self.expect_ty(*r, &res, b.name());
                        // Classify by lane type so vector arithmetic (`<4 x f32> fadd`) is checked
                        // against its `f32` lanes, not the aggregate `Vec` type.
                        let lane = res.lane_type();
                        let float_op = b.is_float();
                        if float_op && !lane.is_float() {
                            self.err(format!(
                                "float op {} on non-float type {}",
                                b.name(),
                                res.display()
                            ));
                        }
                        if !float_op
                            && !lane.is_int()
                            && !matches!(b, BinOp::Xor | BinOp::And | BinOp::Or)
                        {
                            self.err(format!(
                                "int op {} on non-int type {}",
                                b.name(),
                                res.display()
                            ));
                        }
                    }
                }
            }
            Op::Cmp(c, l, r) => {
                if self.use_val(*l) & self.use_val(*r) {
                    if let (Some(lt), Some(rt)) = (self.ty(*l).cloned(), self.ty(*r).cloned()) {
                        if lt != rt {
                            self.err(format!(
                                "cmp operands differ: {} vs {}",
                                lt.display(),
                                rt.display()
                            ));
                        }
                        // Classify by lane type so a `<4 x f32>` compare reads as a float compare.
                        if c.is_float() != lt.lane_type().is_float() {
                            self.err(format!(
                                "cmp predicate {} mismatches operand type {}",
                                c.name(),
                                lt.display()
                            ));
                        }
                        // A vector compare yields a same-width lane mask; a scalar compare yields i1.
                        if let MirType::Vec(_, n) = lt {
                            match self.result_ty(result) {
                                Some(MirType::Vec(_, m)) if m == n => {}
                                Some(other) => self.err(format!(
                                    "vector cmp result must be an {n}-lane mask, got {}",
                                    other.display()
                                )),
                                None => {}
                            }
                            return;
                        }
                    }
                }
                self.check_result_is(result, &MirType::I1);
            }
            Op::Neg(v) | Op::Not(v) => {
                if self.use_val(*v) {
                    if let Some(res) = self.result_ty(result) {
                        self.expect_ty(*v, &res, "neg/not");
                    }
                }
            }
            Op::Cast(_, v, to) => {
                self.use_val(*v);
                self.check_result_is(result, to);
            }
            Op::Select(c, a, b) => {
                if self.use_val(*a) & self.use_val(*b) {
                    if let Some(res) = self.result_ty(result) {
                        self.expect_ty(*a, &res, "select");
                        self.expect_ty(*b, &res, "select");
                        // A vector select takes a same-width lane mask; a scalar select takes i1.
                        if let MirType::Vec(_, n) = &res {
                            if self.use_val(*c) {
                                match self.ty(*c) {
                                    Some(MirType::Vec(_, m)) if m == n => {}
                                    Some(other) => {
                                        let other = other.display();
                                        self.err(format!(
                                            "vector select mask must be an {n}-lane vector, got {other}"
                                        ));
                                    }
                                    None => {}
                                }
                            }
                            return;
                        }
                    }
                }
                if self.use_val(*c) {
                    self.expect_ty(*c, &MirType::I1, "select condition");
                }
            }
            Op::Alloca(_) => self.check_result_is(result, &MirType::Ptr),
            Op::Load(p, ty) => {
                if self.use_val(*p) {
                    self.expect_ty(*p, &MirType::Ptr, "load pointer");
                }
                self.check_result_is(result, ty);
            }
            Op::Store { ptr, value } => {
                if self.use_val(*ptr) {
                    self.expect_ty(*ptr, &MirType::Ptr, "store pointer");
                }
                self.use_val(*value);
            }
            Op::Gep { ptr, index, .. } => {
                if self.use_val(*ptr) {
                    self.expect_ty(*ptr, &MirType::Ptr, "gep base");
                }
                if self.use_val(*index) {
                    if let Some(t) = self.ty(*index) {
                        if !t.is_int() {
                            self.err(format!("gep index has non-int type {}", t.display()));
                        }
                    }
                }
                self.check_result_is(result, &MirType::Ptr);
            }
            Op::Call { func, args } => {
                for a in args {
                    self.use_val(*a);
                }
                // Both backends consume a call's arity and types as fact, and both truncate
                // silently on mismatch: the interpreter zips the callee's params with the args, so
                // surplus parameters keep reading as 0, while Cranelift builds the call against the
                // callee's signature and rejects or panics inside its own IR. An ABI desync is
                // therefore an interp-vs-native divergence with a silent zero on the oracle side.
                let sig = self.sigs.and_then(|s| s.get(func)).cloned();
                if let Some((params, ret)) = sig {
                    if params.len() != args.len() {
                        self.err(format!(
                            "call to {func:?} passes {} args but the function has {} parameters",
                            args.len(),
                            params.len()
                        ));
                    } else {
                        for (i, (a, pt)) in args.iter().zip(&params).enumerate() {
                            self.expect_ty(*a, pt, &format!("call argument {i} to {func:?}"));
                        }
                    }
                    // Discarding a returned value is legal (`result` absent), but taking one from a
                    // `void` callee is not, and a taken result must have the callee's return type.
                    if ret == MirType::Void {
                        if result.is_some() {
                            self.err(format!(
                                "call to {func:?} takes a result but the function returns void"
                            ));
                        }
                    } else {
                        self.check_result_is(result, &ret);
                    }
                }
            }
            Op::VecKernelCall {
                kernel,
                ptrs,
                scalars,
                n,
            } => {
                // `kernel` is a bare index into the *owning* function's `vec_kernels`; nothing else
                // in the instruction can recover it, and both backends index their kernel table
                // with it unchecked (Cranelift `kernel_refs[k]`, the interpreter `vec_kernels[k]`).
                // A pass that re-parents the call into another function leaves it dangling, so
                // range-check it here — this is the one operand the type rules below cannot see.
                let kind = self
                    .f
                    .vec_kernels
                    .get(*kernel as usize)
                    .map(|k| k.reduce.is_some());
                match kind {
                    None => {
                        let have = self.f.vec_kernels.len();
                        self.err(format!(
                            "veckernel kernel index {kernel} is out of range \
                             (the function has {have} kernels)"
                        ));
                    }
                    // A reduction kernel returns its horizontal fold as `f32`; an elementwise one
                    // writes through its output streams and yields nothing. The two have different
                    // backend signatures, so the call's result must agree with the recipe.
                    Some(true) if result.is_none() => self.err(format!(
                        "veckernel #{kernel} is a reduction but the call takes no result"
                    )),
                    Some(true) => self.check_result_is(result, &MirType::F32),
                    Some(false) if result.is_some() => self.err(format!(
                        "veckernel #{kernel} is elementwise but the call takes a result"
                    )),
                    Some(false) => {}
                }
                if self.use_val(*ptrs) {
                    self.expect_ty(*ptrs, &MirType::Ptr, "veckernel ptrs");
                }
                if self.use_val(*scalars) {
                    self.expect_ty(*scalars, &MirType::Ptr, "veckernel scalars");
                }
                if self.use_val(*n) {
                    if let Some(t) = self.ty(*n) {
                        if !t.is_int() {
                            self.err(format!("veckernel n has non-int type {}", t.display()));
                        }
                    }
                }
            }
            Op::FuncAddr(_) => {
                self.check_result_is(result, &MirType::Ptr);
            }
            Op::GlobalAddr(_) => {
                self.check_result_is(result, &MirType::Ptr);
            }
            Op::Fma(a, b, c) => {
                let ok = self.use_val(*a) & self.use_val(*b) & self.use_val(*c);
                if let Some(res) = self.result_ty(result) {
                    if !res.lane_type().is_float() {
                        self.err(format!("fma on non-float type {}", res.display()));
                    }
                    if ok {
                        self.expect_ty(*a, &res, "fma");
                        self.expect_ty(*b, &res, "fma");
                        self.expect_ty(*c, &res, "fma");
                    }
                }
            }
            Op::Sqrt(a) => {
                let ok = self.use_val(*a);
                if let Some(res) = self.result_ty(result) {
                    if !res.lane_type().is_float() {
                        self.err(format!("sqrt on non-float type {}", res.display()));
                    }
                    if ok {
                        self.expect_ty(*a, &res, "sqrt");
                    }
                }
            }
            Op::Round(_, a) => {
                let ok = self.use_val(*a);
                if let Some(res) = self.result_ty(result) {
                    if !res.lane_type().is_float() {
                        self.err(format!("round on non-float type {}", res.display()));
                    }
                    if ok {
                        self.expect_ty(*a, &res, "round");
                    }
                }
            }
            Op::Splat(v) => {
                if self.use_val(*v) {
                    if let (Some(vt), Some(res)) = (self.ty(*v).cloned(), self.result_ty(result)) {
                        match &res {
                            MirType::Vec(lane, _) if **lane == vt => {}
                            MirType::Vec(lane, _) => self.err(format!(
                                "splat: operand {} does not match result lane type {}",
                                vt.display(),
                                lane.display()
                            )),
                            _ => self.err(format!(
                                "splat result must be a vector, got {}",
                                res.display()
                            )),
                        }
                    }
                }
            }
        }
    }

    fn check_result_is(&mut self, result: Option<ValueId>, want: &MirType) {
        if let Some(r) = result {
            if let Some(t) = self.ty(r) {
                if t != want {
                    self.err(format!(
                        "result {r:?} has type {} but operation yields {}",
                        t.display(),
                        want.display()
                    ));
                }
            }
        }
    }

    fn check_term(&mut self, t: &Terminator, nblocks: u32) {
        match t {
            Terminator::Ret(Some(v)) => {
                if self.use_val(*v) {
                    self.expect_ty(*v, &self.f.ret.clone(), "return value");
                }
            }
            Terminator::Ret(None) => {
                if self.f.ret != MirType::Void {
                    self.err(format!(
                        "`ret` with no value but function returns {}",
                        self.f.ret.display()
                    ));
                }
            }
            Terminator::Br { target, args } => {
                self.check_edge(*target, args, nblocks);
            }
            Terminator::CondBr {
                cond,
                then_blk,
                then_args,
                else_blk,
                else_args,
            } => {
                if self.use_val(*cond) {
                    self.expect_ty(*cond, &MirType::I1, "cond_br condition");
                }
                self.check_edge(*then_blk, then_args, nblocks);
                self.check_edge(*else_blk, else_args, nblocks);
            }
            Terminator::Unreachable => {}
        }
    }

    fn check_edge(&mut self, target: crate::BlockId, args: &[ValueId], nblocks: u32) {
        if target.0 >= nblocks {
            self.err(format!("branch to nonexistent block bb{}", target.0));
            return;
        }
        let tparams = self.f.blocks[target.0 as usize].params.clone();
        if tparams.len() != args.len() {
            self.err(format!(
                "branch to bb{} passes {} args but the block has {} parameters",
                target.0,
                args.len(),
                tparams.len()
            ));
            return;
        }
        for (a, p) in args.iter().zip(&tparams) {
            if self.use_val(*a) {
                if let (Some(at), Some(pt)) = (self.ty(*a).cloned(), self.ty(*p).cloned()) {
                    if at != pt {
                        self.err(format!(
                            "branch arg {a:?} ({}) does not match bb{} param {p:?} ({})",
                            at.display(),
                            target.0,
                            pt.display()
                        ));
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{BinOp, Builder, MirType, Op};
    use wukong_span::Interner;

    #[test]
    fn valid_function_verifies() {
        let mut i = Interner::new();
        let mut b = Builder::new(i.intern("f"), MirType::I32);
        let x = b.add_param(MirType::I32);
        let one = b.build(MirType::I32, Op::ConstInt(1, MirType::I32));
        let y = b.build(MirType::I32, Op::Bin(BinOp::Add, x, one));
        b.ret(Some(y));
        assert!(verify_function(&b.finish()).is_empty());
    }

    #[test]
    fn detects_type_mismatch() {
        // add an i32 and an i64 -> operand type mismatch against the i32 result.
        let mut i = Interner::new();
        let mut b = Builder::new(i.intern("f"), MirType::I32);
        let x = b.add_param(MirType::I32);
        let big = b.build(MirType::I64, Op::ConstInt(1, MirType::I64));
        let y = b.build(MirType::I32, Op::Bin(BinOp::Add, x, big));
        b.ret(Some(y));
        assert!(!verify_function(&b.finish()).is_empty());
    }

    #[test]
    fn void_call_verifies() {
        // A result-less call (e.g. the `print` intrinsic) is valid and must not be flagged as
        // "must produce a result" — regression for the void-call verifier rule.
        use crate::Op;
        let mut i = Interner::new();
        let mut b = Builder::new(i.intern("main"), MirType::Void);
        let arg = b.build(MirType::I32, Op::ConstInt(42, MirType::I32));
        b.build_void(Op::Call {
            func: i.intern("print"),
            args: vec![arg],
        });
        b.ret(None);
        assert!(verify_function(&b.finish()).is_empty());
    }

    #[test]
    fn detects_block_param_arity() {
        let mut i = Interner::new();
        let mut b = Builder::new(i.intern("f"), MirType::Void);
        let target = b.new_block();
        let _p = b.block_param(target, MirType::I32);
        // branch with no args to a block expecting one parameter.
        b.br(target, vec![]);
        b.switch_to(target);
        b.ret(None);
        let errs = verify_function(&b.finish());
        assert!(errs.iter().any(|e| e.contains("parameters")), "{errs:?}");
    }

    /// `kernel` is a bare index into the *owning* function's `vec_kernels`; nothing else in the
    /// instruction can recover it, and Cranelift indexes `kernel_refs` with it unchecked. A pass
    /// that re-parents the call into a function that owns no kernels leaves it dangling.
    #[test]
    fn detects_veckernel_index_out_of_range() {
        let mut i = Interner::new();
        let mut b = Builder::new(i.intern("f"), MirType::Void);
        let ptrs = b.alloca(MirType::Ptr);
        let scalars = b.alloca(MirType::F32);
        let n = b.build(MirType::I64, Op::ConstInt(8, MirType::I64));
        b.build_void(Op::VecKernelCall {
            kernel: 0,
            ptrs,
            scalars,
            n,
        });
        b.ret(None);
        // The function registered no kernels, so `#0` refers to nothing.
        let errs = verify_function(&b.finish());
        assert!(
            errs.iter().any(|e| e.contains("out of range")),
            "{errs:?}"
        );
    }

    // ---- cross-function call checks (`verify_program`) ----

    fn two_ptr_void_callee(i: &mut Interner) -> Function {
        let mut b = Builder::new(i.intern("callee"), MirType::Void);
        let _a = b.add_param(MirType::Ptr);
        let _c = b.add_param(MirType::Ptr);
        b.ret(None);
        b.finish()
    }

    fn program_of(funcs: Vec<Function>) -> Program {
        Program {
            funcs,
            statics: Vec::new(),
            level: crate::MirLevel::Low,
        }
    }

    /// The exact shape a monomorphization-key collision produces: one argument passed to a
    /// two-parameter function, and a result taken from a `void` callee. Per-function verification
    /// accepted it; the interpreter then reported `expected a pointer` and the native backend
    /// aborted with `MIR value used before definition`.
    #[test]
    fn program_detects_call_arity_and_void_result_mismatch() {
        let mut i = Interner::new();
        let callee = two_ptr_void_callee(&mut i);
        let mut b = Builder::new(i.intern("main"), MirType::I32);
        let slot = b.alloca(MirType::I32);
        let bad = b.build(
            MirType::I32,
            Op::Call {
                func: i.intern("callee"),
                args: vec![slot],
            },
        );
        b.ret(Some(bad));
        let errs = verify_program(&program_of(vec![callee, b.finish()]));
        assert!(
            errs.iter().any(|e| e.contains("passes 1 args")),
            "arity: {errs:?}"
        );
        assert!(
            errs.iter().any(|e| e.contains("returns void")),
            "void result: {errs:?}"
        );
    }

    #[test]
    fn program_detects_call_argument_type_mismatch() {
        let mut i = Interner::new();
        let callee = two_ptr_void_callee(&mut i);
        let mut b = Builder::new(i.intern("main"), MirType::I32);
        let slot = b.alloca(MirType::I32);
        let wrong = b.build(MirType::I64, Op::ConstInt(0, MirType::I64));
        b.build_void(Op::Call {
            func: i.intern("callee"),
            args: vec![slot, wrong],
        });
        let z = b.build(MirType::I32, Op::ConstInt(0, MirType::I32));
        b.ret(Some(z));
        let errs = verify_program(&program_of(vec![callee, b.finish()]));
        assert!(
            errs.iter().any(|e| e.contains("call argument 1")),
            "{errs:?}"
        );
    }

    /// A call to a symbol with no MIR body (the `print` intrinsic, a `wukong_*` runtime kernel) is
    /// external and must be skipped, a well-formed internal call must stay clean, and discarding a
    /// non-void callee's result is legal.
    #[test]
    fn program_accepts_external_well_formed_and_discarded_calls() {
        let mut i = Interner::new();
        let callee = two_ptr_void_callee(&mut i);
        let mut vb = Builder::new(i.intern("answer"), MirType::I32);
        let a = vb.build(MirType::I32, Op::ConstInt(42, MirType::I32));
        vb.ret(Some(a));
        let answer = vb.finish();

        let mut b = Builder::new(i.intern("main"), MirType::I32);
        let p1 = b.alloca(MirType::I32);
        let p2 = b.alloca(MirType::I32);
        b.build_void(Op::Call {
            func: i.intern("callee"),
            args: vec![p1, p2],
        });
        b.build_void(Op::Call {
            func: i.intern("answer"),
            args: vec![],
        });
        let z = b.build(MirType::I32, Op::ConstInt(0, MirType::I32));
        b.build_void(Op::Call {
            func: i.intern("print"),
            args: vec![z],
        });
        b.ret(Some(z));
        let p = program_of(vec![callee, answer, b.finish()]);
        assert!(verify_program(&p).is_empty(), "{:?}", verify_program(&p));
    }
}
