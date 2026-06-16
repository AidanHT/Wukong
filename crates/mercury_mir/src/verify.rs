//! The MIR verifier — checks well-formedness invariants. A failure here is an *internal compiler
//! error* (a bug in a pass), not a user error, so violations are returned as plain strings.
//!
//! It runs after MIR construction, between optimization passes (under `--verify-each`), and
//! before backend entry. Checks: every value used is defined; block-parameter arities and types
//! line up across edges; per-operation operand/result types are consistent; and terminators are
//! type-correct (including the function return type).

use std::collections::HashSet;

use crate::{BinOp, Function, MirType, Op, Program, Terminator, ValueId};

/// Verify every function in a program. Returns a list of human-readable problems (empty = ok).
pub fn verify_program(p: &Program) -> Vec<String> {
    let mut errors = Vec::new();
    for f in &p.funcs {
        errors.extend(verify_function(f));
    }
    errors
}

/// Verify a single function.
pub fn verify_function(f: &Function) -> Vec<String> {
    let mut v = Verifier {
        f,
        errors: Vec::new(),
        defined: HashSet::new(),
    };
    v.run();
    v.errors
}

struct Verifier<'a> {
    f: &'a Function,
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
        let must_produce = !matches!(op, Op::Store { .. } | Op::Call { .. });
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
                        let float_op = b.is_float();
                        if float_op && !res.is_float() {
                            self.err(format!(
                                "float op {} on non-float type {}",
                                b.name(),
                                res.display()
                            ));
                        }
                        if !float_op
                            && !res.is_int()
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
                        if c.is_float() != lt.is_float() {
                            self.err(format!(
                                "cmp predicate {} mismatches operand type {}",
                                c.name(),
                                lt.display()
                            ));
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
                if self.use_val(*c) {
                    self.expect_ty(*c, &MirType::I1, "select condition");
                }
                if self.use_val(*a) & self.use_val(*b) {
                    if let Some(res) = self.result_ty(result) {
                        self.expect_ty(*a, &res, "select");
                        self.expect_ty(*b, &res, "select");
                    }
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
            Op::Call { args, .. } => {
                for a in args {
                    self.use_val(*a);
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
    use mercury_span::Interner;

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
}
