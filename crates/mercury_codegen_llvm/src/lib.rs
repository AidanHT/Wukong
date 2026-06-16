//! `mercury_codegen_llvm` — the LLVM backend.
//!
//! We emit **textual LLVM IR** rather than driving libLLVM in-process. This was a deliberate
//! call: on this Windows + MinGW host, linking `llvm-sys`/inkwell is fragile (ABI matching, the
//! missing dev libraries in the stock installer), whereas textual IR needs only the LLVM command
//! line tools on PATH and no build-time dependency at all. The same MIR lowering would feed an
//! inkwell builder; this implementation sits behind the shared [`Backend`] seam so it can be
//! swapped later without touching the driver.
//!
//! Mercury MIR uses block-parameter SSA (à la Cranelift/MLIR); LLVM uses phi nodes. The emitter
//! bridges the two: each non-entry block parameter becomes a `phi` whose incoming values are the
//! arguments each predecessor passes on its edge to that block. At `-O0` the front-end emits no
//! block parameters, so no phis are generated; after `mem2reg` they appear and are lowered here.

use std::fmt::Write as _;

use mercury_backend::{Artifact, Backend};
use mercury_mir::{
    BasicBlock, BinOp, CastKind, CmpOp, Function, MirType, Op, Program, Terminator, ValueId,
};
use mercury_span::{Interner, Symbol};
use std::collections::HashMap;

/// For each block, the incoming edges as `(predecessor block id, arguments passed)`. Used to build
/// phi nodes for block parameters.
type EdgeArgs = HashMap<u32, Vec<(u32, Vec<ValueId>)>>;

/// Collect, per target block, the arguments each predecessor passes along its edge.
fn edge_args(f: &Function) -> EdgeArgs {
    let mut map: EdgeArgs = HashMap::new();
    for b in &f.blocks {
        match &b.term {
            Terminator::Br { target, args } => {
                map.entry(target.0)
                    .or_default()
                    .push((b.id.0, args.clone()));
            }
            Terminator::CondBr {
                then_blk,
                then_args,
                else_blk,
                else_args,
                ..
            } => {
                map.entry(then_blk.0)
                    .or_default()
                    .push((b.id.0, then_args.clone()));
                // A conditional branch with both arms to the same block is one LLVM predecessor;
                // don't record a second, conflicting phi entry for it.
                if else_blk.0 != then_blk.0 {
                    map.entry(else_blk.0)
                        .or_default()
                        .push((b.id.0, else_args.clone()));
                }
            }
            Terminator::Ret(_) | Terminator::Unreachable => {}
        }
    }
    map
}

/// The LLVM backend. `compile` returns the emitted IR text in [`Artifact::Emitted`].
pub struct LlvmBackend;

impl Backend for LlvmBackend {
    fn name(&self) -> &'static str {
        "llvm"
    }

    fn compile(
        &self,
        program: &Program,
        _entry: Symbol,
        interner: &Interner,
    ) -> Result<Artifact, String> {
        Ok(Artifact::Emitted {
            llvm_ir: Some(emit_llvm_ir(program, interner)),
            object: None,
        })
    }
}

/// Lower a whole program to a textual LLVM IR module.
pub fn emit_llvm_ir(program: &Program, interner: &Interner) -> String {
    let mut out = String::new();
    out.push_str("; Mercury -> LLVM IR\n\n");
    for f in &program.funcs {
        emit_function(&mut out, f, interner);
        out.push('\n');
    }
    out
}

fn emit_function(out: &mut String, f: &Function, interner: &Interner) {
    // Inline constants: map const-producing values to literal operand strings.
    let mut consts: HashMap<u32, String> = HashMap::new();
    for b in &f.blocks {
        for inst in &b.insts {
            if let Some(r) = inst.result {
                match &inst.op {
                    Op::ConstInt(v, _) => {
                        consts.insert(r.0, v.to_string());
                    }
                    Op::ConstFloat(v, _) => {
                        consts.insert(r.0, fmt_float(*v));
                    }
                    _ => {}
                }
            }
        }
    }

    let params: Vec<String> = f
        .params
        .iter()
        .map(|p| format!("{} %v{}", llvm_ty(f.value_type(*p)), p.0))
        .collect();
    let _ = writeln!(
        out,
        "define {} @{}({}) {{",
        llvm_ty(&f.ret),
        interner.resolve(f.name),
        params.join(", ")
    );

    let edges = edge_args(f);
    let e = Emitter {
        f,
        interner,
        consts: &consts,
        edges: &edges,
    };
    for b in &f.blocks {
        e.emit_block(out, b);
    }
    out.push_str("}\n");
}

struct Emitter<'a> {
    f: &'a Function,
    interner: &'a Interner,
    consts: &'a HashMap<u32, String>,
    edges: &'a EdgeArgs,
}

impl Emitter<'_> {
    fn operand(&self, v: ValueId) -> String {
        match self.consts.get(&v.0) {
            Some(s) => s.clone(),
            None => format!("%v{}", v.0),
        }
    }

    fn ty(&self, v: ValueId) -> String {
        llvm_ty(self.f.value_type(v))
    }

    fn emit_block(&self, out: &mut String, b: &BasicBlock) {
        let _ = writeln!(out, "bb{}:", b.id.0);
        // Non-entry block parameters become phi nodes (entry parameters are the function's
        // arguments, declared in the signature). Phis must lead the block.
        if b.id != self.f.entry {
            self.emit_phis(out, b);
        }
        for inst in &b.insts {
            // Constants are inlined as operands; they emit no instruction.
            if matches!(inst.op, Op::ConstInt(..) | Op::ConstFloat(..)) {
                continue;
            }
            self.emit_inst(out, inst);
        }
        self.emit_term(out, &b.term);
    }

    fn emit_phis(&self, out: &mut String, b: &BasicBlock) {
        if b.params.is_empty() {
            return;
        }
        let preds = self.edges.get(&b.id.0);
        for (k, param) in b.params.iter().enumerate() {
            let ty = self.ty(*param);
            let mut entries: Vec<String> = Vec::new();
            if let Some(preds) = preds {
                for (pred, args) in preds {
                    if let Some(arg) = args.get(k) {
                        entries.push(format!("[ {}, %bb{} ]", self.operand(*arg), pred));
                    }
                }
            }
            let _ = writeln!(out, "  %v{} = phi {ty} {}", param.0, entries.join(", "));
        }
    }

    fn emit_inst(&self, out: &mut String, inst: &mercury_mir::Inst) {
        let res = inst
            .result
            .map(|r| format!("%v{} = ", r.0))
            .unwrap_or_default();
        let line = match &inst.op {
            Op::ConstInt(..) | Op::ConstFloat(..) => return,
            Op::Bin(op, l, r) => {
                let ty = self.ty(inst.result.unwrap());
                format!(
                    "{res}{} {ty} {}, {}",
                    bin_name(*op),
                    self.operand(*l),
                    self.operand(*r)
                )
            }
            Op::Cmp(op, l, r) => {
                let opty = self.ty(*l);
                let (instr, pred) = cmp_instr(*op);
                format!(
                    "{res}{instr} {pred} {opty} {}, {}",
                    self.operand(*l),
                    self.operand(*r)
                )
            }
            Op::Neg(v) => {
                let ty = self.ty(inst.result.unwrap());
                if self.f.value_type(inst.result.unwrap()).is_float() {
                    format!("{res}fneg {ty} {}", self.operand(*v))
                } else {
                    format!("{res}sub {ty} 0, {}", self.operand(*v))
                }
            }
            Op::Not(v) => {
                let ty = self.ty(inst.result.unwrap());
                format!("{res}xor {ty} {}, -1", self.operand(*v))
            }
            Op::Cast(kind, v, to) => {
                let from = self.ty(*v);
                format!(
                    "{res}{} {from} {} to {}",
                    cast_name(*kind),
                    self.operand(*v),
                    llvm_ty(to)
                )
            }
            Op::Select(c, a, b) => {
                let ty = self.ty(inst.result.unwrap());
                format!(
                    "{res}select i1 {}, {ty} {}, {ty} {}",
                    self.operand(*c),
                    self.operand(*a),
                    self.operand(*b)
                )
            }
            Op::Alloca(ty) => format!("{res}alloca {}", llvm_ty(ty)),
            Op::Load(p, ty) => format!("{res}load {}, ptr {}", llvm_ty(ty), self.operand(*p)),
            Op::Store { ptr, value } => {
                let ty = self.ty(*value);
                format!(
                    "store {ty} {}, ptr {}",
                    self.operand(*value),
                    self.operand(*ptr)
                )
            }
            Op::Gep { ptr, index, elem } => {
                let idxty = self.ty(*index);
                format!(
                    "{res}getelementptr {}, ptr {}, {idxty} {}",
                    llvm_ty(elem),
                    self.operand(*ptr),
                    self.operand(*index)
                )
            }
            Op::Call { func, args } => {
                // A void call (e.g. the `print` intrinsic) has no result value to type.
                let ret_ty = match inst.result {
                    Some(r) => self.ty(r),
                    None => "void".to_string(),
                };
                let argstr: Vec<String> = args
                    .iter()
                    .map(|a| format!("{} {}", self.ty(*a), self.operand(*a)))
                    .collect();
                format!(
                    "{res}call {ret_ty} @{}({})",
                    self.interner.resolve(*func),
                    argstr.join(", ")
                )
            }
            Op::FuncAddr(func) => {
                // The address of a function as a `ptr` (an identity GEP keeps it valid IR).
                format!(
                    "{res}getelementptr i8, ptr @{}, i64 0",
                    self.interner.resolve(*func)
                )
            }
            Op::Splat(v) => {
                // Broadcast a scalar into every lane: insert at lane 0, then shuffle with an
                // all-zero mask. (The native vectorizer is Cranelift-only, so this path is for
                // completeness/round-tripping rather than the hot path.)
                let r = inst.result.unwrap();
                let vty = self.ty(r);
                let lane = self.ty(*v);
                let width = match self.f.value_type(r) {
                    MirType::Vec(_, n) => *n,
                    _ => 1,
                };
                format!(
                    "%v{0}.s = insertelement {vty} poison, {lane} {1}, i32 0\n  \
                     {res}shufflevector {vty} %v{0}.s, {vty} poison, <{width} x i32> zeroinitializer",
                    r.0,
                    self.operand(*v),
                )
            }
        };
        let _ = writeln!(out, "  {line}");
    }

    fn emit_term(&self, out: &mut String, t: &Terminator) {
        let line = match t {
            Terminator::Ret(None) => "ret void".to_string(),
            Terminator::Ret(Some(v)) => format!("ret {} {}", self.ty(*v), self.operand(*v)),
            Terminator::Br { target, .. } => format!("br label %bb{}", target.0),
            Terminator::CondBr {
                cond,
                then_blk,
                else_blk,
                ..
            } => format!(
                "br i1 {}, label %bb{}, label %bb{}",
                self.operand(*cond),
                then_blk.0,
                else_blk.0
            ),
            Terminator::Unreachable => "unreachable".to_string(),
        };
        let _ = writeln!(out, "  {line}");
    }
}

fn llvm_ty(t: &MirType) -> String {
    match t {
        MirType::I1 => "i1".into(),
        MirType::I8 => "i8".into(),
        MirType::I16 => "i16".into(),
        MirType::I32 => "i32".into(),
        MirType::I64 => "i64".into(),
        MirType::F16 => "half".into(),
        MirType::BF16 => "bfloat".into(),
        MirType::F32 => "float".into(),
        MirType::F64 => "double".into(),
        MirType::Ptr => "ptr".into(),
        MirType::Vec(e, n) => format!("<{} x {}>", n, llvm_ty(e)),
        MirType::Array(e, n) => format!("[{} x {}]", n, llvm_ty(e)),
        MirType::Void => "void".into(),
    }
}

fn bin_name(op: BinOp) -> &'static str {
    use BinOp::*;
    match op {
        Add => "add",
        Sub => "sub",
        Mul => "mul",
        SDiv => "sdiv",
        UDiv => "udiv",
        SRem => "srem",
        URem => "urem",
        FAdd => "fadd",
        FSub => "fsub",
        FMul => "fmul",
        FDiv => "fdiv",
        And => "and",
        Or => "or",
        Xor => "xor",
        Shl => "shl",
        LShr => "lshr",
        AShr => "ashr",
    }
}

fn cmp_instr(op: CmpOp) -> (&'static str, &'static str) {
    use CmpOp::*;
    match op {
        Eq => ("icmp", "eq"),
        Ne => ("icmp", "ne"),
        Slt => ("icmp", "slt"),
        Sle => ("icmp", "sle"),
        Sgt => ("icmp", "sgt"),
        Sge => ("icmp", "sge"),
        Ult => ("icmp", "ult"),
        Ule => ("icmp", "ule"),
        Ugt => ("icmp", "ugt"),
        Uge => ("icmp", "uge"),
        Foeq => ("fcmp", "oeq"),
        Fone => ("fcmp", "one"),
        Folt => ("fcmp", "olt"),
        Fole => ("fcmp", "ole"),
        Fogt => ("fcmp", "ogt"),
        Foge => ("fcmp", "oge"),
    }
}

fn cast_name(kind: CastKind) -> &'static str {
    use CastKind::*;
    match kind {
        SExt => "sext",
        ZExt => "zext",
        Trunc => "trunc",
        FpToSi => "fptosi",
        FpToUi => "fptoui",
        SiToFp => "sitofp",
        UiToFp => "uitofp",
        FpExt => "fpext",
        FpTrunc => "fptrunc",
        Bitcast => "bitcast",
        IntToPtr => "inttoptr",
        PtrToInt => "ptrtoint",
    }
}

fn fmt_float(v: f64) -> String {
    if v == v.trunc() && v.is_finite() {
        format!("{v:.1}")
    } else {
        format!("{v:?}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mercury_span::SourceId;

    fn ir(src: &str) -> String {
        ir_opt(src, 0)
    }

    fn ir_opt(src: &str, opt: u8) -> String {
        let mut interner = Interner::new();
        let (module, _) = mercury_parser::parse_module(src, SourceId(0), &mut interner);
        let (sema, sd) = mercury_sema::check(&module, &interner);
        assert!(sd.iter().all(|d| !d.is_error()), "sema: {sd:?}");
        let (mut program, _) = mercury_mir_build::lower_program(&module, &sema, &mut interner);
        mercury_opt::optimize(&mut program, opt);
        emit_llvm_ir(&program, &interner)
    }

    #[test]
    fn emits_function_signature_and_arith() {
        let out = ir("fn add(a: i32, b: i32) -> i32 { return a + b; }");
        assert!(out.contains("define i32 @add(i32 %v0, i32 %v1)"), "{out}");
        assert!(out.contains("add i32"), "{out}");
        assert!(out.contains("ret i32"), "{out}");
    }

    #[test]
    fn emits_control_flow_and_calls() {
        let out = ir("fn fib(n: i32) -> i32 { if n < 2 { return n; } \
                      return fib(n-1) + fib(n-2); } fn main() -> i32 { return fib(10); }");
        assert!(out.contains("icmp slt i32"), "{out}");
        assert!(out.contains("br i1"), "{out}");
        assert!(out.contains("call i32 @fib"), "{out}");
        assert!(out.contains("alloca i32"), "{out}");
    }

    #[test]
    fn emits_void_call_for_print_intrinsic() {
        // A void call (no result) must emit `call void @print(...)`, not panic on a missing
        // result type (regression for indexing value_types with a dummy id).
        let out = ir("fn main() -> i32 { print(42); return 0; }");
        assert!(out.contains("call void @print(i32 42)"), "{out}");
    }

    #[test]
    fn emits_phi_nodes_for_optimized_loops() {
        // After mem2reg the loop carries its counter/accumulator as block parameters; the emitter
        // must lower those to phi nodes with one entry per predecessor edge.
        let out = ir_opt(
            "fn main() -> i32 { let mut s: i32 = 0; let mut i: i32 = 0; \
             while i < 10 { s = s + i; i = i + 1; } return s; }",
            2,
        );
        assert!(out.contains("phi i32"), "expected phi nodes:\n{out}");
        // Every phi should name its predecessor blocks.
        assert!(out.contains("phi i32 [") && out.contains(", %bb"), "{out}");
        // No leftover alloca: mem2reg promoted the scalars.
        assert!(
            !out.contains("alloca"),
            "scalars should be promoted:\n{out}"
        );
    }
}
