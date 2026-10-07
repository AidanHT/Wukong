//! `wukong_codegen_llvm` — the LLVM **IR emitter**. An inspection path, not the native fast path
//! (that is `wukong_codegen_cranelift`, which also needs no LLVM): nothing here assembles, links, or
//! runs anything, so it is never the differential oracle's peer.
//!
//! We emit **textual LLVM IR** rather than driving libLLVM in-process. This was a deliberate
//! call: on this Windows + MinGW host, linking `llvm-sys`/inkwell is fragile (ABI matching, the
//! missing dev libraries in the stock installer), whereas textual IR needs no build-time dependency
//! at all — turning it into a binary would need `clang`/`llc` on PATH, an out-of-process step this
//! crate never performs. The same MIR lowering would feed an inkwell builder.
//!
//! [`LlvmBackend`] declares the [`Backend`] seam, but it is not how the IR is reached: nothing in the
//! workspace calls [`Backend::compile`], and `--emit=llvm-ir` has `wukong_driver` call
//! [`try_emit_llvm_ir`] directly (no feature gate).
//!
//! Wukong MIR uses block-parameter SSA (à la Cranelift/MLIR); LLVM uses phi nodes. The emitter
//! bridges the two: each non-entry block parameter becomes a `phi` whose incoming values are the
//! arguments each predecessor passes on its edge to that block. At `-O0` the front-end emits no
//! block parameters, so no phis are generated; after `mem2reg` they appear and are lowered here.

use std::fmt::Write as _;

use std::collections::{BTreeMap, HashMap, HashSet};
use wukong_backend::{Artifact, Backend};
use wukong_mir::{
    BasicBlock, BinOp, CastKind, CmpOp, Function, MirType, Op, Program, Terminator, ValueId,
};
use wukong_span::{Interner, Symbol};

/// For each block, the incoming edges as `(predecessor block id, arguments passed)`. Used to build
/// phi nodes for block parameters.
type EdgeArgs = HashMap<u32, Vec<(String, Vec<ValueId>)>>;

/// Collect, per target block, the arguments each predecessor passes along its edge.
fn edge_args(f: &Function) -> EdgeArgs {
    let mut map: EdgeArgs = HashMap::new();
    for b in &f.blocks {
        match &b.term {
            Terminator::Br { target, args } => {
                map.entry(target.0)
                    .or_default()
                    .push((format!("bb{}", b.id.0), args.clone()));
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
                    .push((format!("bb{}", b.id.0), then_args.clone()));
                // Split a duplicate successor edge, so distinct edge arguments have distinct
                // predecessor labels in LLVM's phi representation.
                let pred = if else_blk == then_blk {
                    format!("bb{}.else", b.id.0)
                } else {
                    format!("bb{}", b.id.0)
                };
                map.entry(else_blk.0)
                    .or_default()
                    .push((pred, else_args.clone()));
            }
            // Every switch edge leaves through its own forwarding block (see `emit_term`), so each
            // edge is a distinct phi predecessor even when two cases share a target.
            Terminator::Switch {
                cases,
                default,
                default_args,
                ..
            } => {
                for (i, c) in cases.iter().enumerate() {
                    map.entry(c.target.0)
                        .or_default()
                        .push((format!("bb{}.s{i}", b.id.0), c.args.clone()));
                }
                map.entry(default.0)
                    .or_default()
                    .push((format!("bb{}.sd", b.id.0), default_args.clone()));
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
            llvm_ir: Some(try_emit_llvm_ir(program, interner)?),
            object: None,
        })
    }
}

/// Lower a whole program to a textual LLVM IR module.
/// Panics for unsupported MIR. Diagnostic-producing callers should use [`try_emit_llvm_ir`].
pub fn emit_llvm_ir(program: &Program, interner: &Interner) -> String {
    try_emit_llvm_ir(program, interner).expect("unsupported MIR in textual LLVM emission")
}

/// Emit an inspection module, declining constructs whose ABI cannot be inferred faithfully.
/// External declarations describe MIR call sites; their implementations must still be linked.
pub fn try_emit_llvm_ir(program: &Program, interner: &Interner) -> Result<String, String> {
    let overloads = print_overloads(program, interner);
    let declarations = declarations(program, interner, &overloads)?;
    let mut out = String::new();
    out.push_str("; Wukong -> LLVM IR\n\n");
    for (name, (ret, args)) in &declarations {
        if program
            .funcs
            .iter()
            .any(|f| interner.resolve(f.name) == name)
        {
            continue;
        }
        let _ = writeln!(out, "declare {ret} @{name}({})", args.join(", "));
    }
    out.push('\n');
    // Read-only string literals live as private constants; `Op::GlobalAddr(@name)` GEPs off them.
    for s in &program.statics {
        let mut bytes = String::new();
        for &b in &s.bytes {
            if b == b'\\' || b == b'"' || !(0x20..0x7f).contains(&b) {
                bytes.push_str(&format!("\\{b:02X}"));
            } else {
                bytes.push(b as char);
            }
        }
        out.push_str(&format!(
            "@{} = private unnamed_addr constant [{} x i8] c\"{}\"\n",
            interner.resolve(s.name),
            s.bytes.len(),
            bytes
        ));
    }
    if !program.statics.is_empty() {
        out.push('\n');
    }
    for f in &program.funcs {
        emit_function(&mut out, f, interner, &overloads, &declarations);
        out.push('\n');
    }
    Ok(out)
}

type Signature = (String, Vec<String>);

fn print_overloads(program: &Program, interner: &Interner) -> HashSet<Symbol> {
    let mut seen: HashMap<Symbol, Vec<MirType>> = HashMap::new();
    let mut overloaded = HashSet::new();
    for f in &program.funcs {
        for b in &f.blocks {
            for inst in &b.insts {
                if let Op::Call { func, args } = &inst.op {
                    if matches!(interner.resolve(*func), "print" | "println")
                        && !program.funcs.iter().any(|f| f.name == *func)
                    {
                        let types = args
                            .iter()
                            .map(|a| f.value_type(*a).clone())
                            .collect::<Vec<_>>();
                        if seen.get(func).is_some_and(|old| old != &types) {
                            overloaded.insert(*func);
                        }
                        seen.insert(*func, types);
                    }
                }
            }
        }
    }
    overloaded
}

fn call_name(
    f: &Function,
    func: Symbol,
    args: &[ValueId],
    interner: &Interner,
    overloads: &HashSet<Symbol>,
) -> String {
    let name = interner.resolve(func);
    if overloads.contains(&func) {
        let suffix = args
            .iter()
            .map(|a| llvm_ty(f.value_type(*a)))
            .collect::<Vec<_>>()
            .join(".");
        format!("__wukong_{name}.{suffix}")
    } else {
        name.to_owned()
    }
}

fn declarations(
    program: &Program,
    interner: &Interner,
    overloads: &HashSet<Symbol>,
) -> Result<BTreeMap<String, Signature>, String> {
    let mut signatures: BTreeMap<String, Signature> = BTreeMap::new();
    // Optional runtime pointers are sometimes represented as integer zero in MIR. Infer the
    // pointer ABI only when every non-pointer use at that position is a literal zero.
    let mut nullable: HashMap<(String, usize), bool> = HashMap::new();
    for f in &program.funcs {
        let zeros: HashSet<ValueId> = f
            .blocks
            .iter()
            .flat_map(|b| &b.insts)
            .filter(|ins| matches!(ins.op, Op::ConstInt(0, _)))
            .filter_map(|ins| ins.result)
            .collect();
        for b in &f.blocks {
            for inst in &b.insts {
                if let Op::Call { func, args } = &inst.op {
                    let name = interner.resolve(*func);
                    if name.starts_with("wukong_") {
                        for (k, a) in args.iter().enumerate() {
                            let valid = f.value_type(*a) == &MirType::Ptr
                                || (f.value_type(*a).is_int() && zeros.contains(a));
                            *nullable.entry((name.to_owned(), k)).or_insert(true) &= valid;
                        }
                    }
                }
            }
        }
    }
    for f in &program.funcs {
        signatures.insert(
            interner.resolve(f.name).to_owned(),
            (
                llvm_ty(&f.ret),
                f.params.iter().map(|v| llvm_ty(f.value_type(*v))).collect(),
            ),
        );
    }
    let defined = signatures.clone();
    for f in &program.funcs {
        for b in &f.blocks {
            for inst in &b.insts {
                if let Op::Cast(_, value, to) = &inst.op {
                    let from = f.value_type(*value);
                    if from != to
                        && to.is_vector()
                        && matches!(to.lane_type(), MirType::F16 | MirType::BF16)
                        && matches!(
                            from.lane_type(),
                            MirType::F16 | MirType::BF16 | MirType::F64
                        )
                    {
                        return Err(
                            "LLVM emission: vector low-precision casts require an f32 lane adapter"
                                .into(),
                        );
                    }
                }
                let call = match &inst.op {
                    Op::Call { func, args } => Some((
                        call_name(f, *func, args, interner, overloads),
                        (
                            inst.result
                                .map(|r| llvm_ty(f.value_type(r)))
                                .unwrap_or_else(|| {
                                    defined
                                        .get(interner.resolve(*func))
                                        .map(|s| s.0.clone())
                                        .unwrap_or_else(|| "void".into())
                                }),
                            args.iter().map(|v| llvm_ty(f.value_type(*v))).collect(),
                        ),
                    )),
                    Op::Sqrt(v) | Op::Round(_, v) => {
                        let ty = f.value_type(*v);
                        let name = math_intrinsic(&inst.op, ty)?;
                        Some((name, (llvm_ty(ty), vec![llvm_ty(ty)])))
                    }
                    Op::Fma(a, _, _) => {
                        let ty = f.value_type(*a);
                        if !matches!(ty.lane_type(), MirType::F32 | MirType::F64) {
                            return Err(
                                "LLVM emission: low-precision FMA requires promoted f32 SSA values"
                                    .into(),
                            );
                        }
                        Some((
                            math_intrinsic(&inst.op, ty)?,
                            (llvm_ty(ty), vec![llvm_ty(ty); 3]),
                        ))
                    }
                    Op::Cast(kind @ (CastKind::FpToSi | CastKind::FpToUi), v, to) => {
                        let from = f.value_type(*v);
                        let target = saturating_target(to);
                        Some((
                            saturating_cast_name(*kind, from, &target),
                            (llvm_ty(&target), vec![llvm_ty(from)]),
                        ))
                    }
                    Op::VecKernelCall { kernel, n, .. } => Some((
                        format!("__wukong_veckernel_{}_{}", interner.resolve(f.name), kernel),
                        (
                            inst.result
                                .map(|r| llvm_ty(f.value_type(r)))
                                .unwrap_or_else(|| "void".into()),
                            vec!["ptr".into(), "ptr".into(), llvm_ty(f.value_type(*n))],
                        ),
                    )),
                    _ => None,
                };
                if let Some((name, sig)) = call {
                    if let Some(previous) = signatures.get_mut(&name) {
                        if !defined.contains_key(&name)
                            && previous.0 == sig.0
                            && previous.1.len() == sig.1.len()
                        {
                            let mut merged = previous.1.clone();
                            for (k, (old, new)) in merged.iter_mut().zip(&sig.1).enumerate() {
                                if old != new
                                    && (old == "ptr" || new == "ptr")
                                    && nullable.get(&(name.clone(), k)) == Some(&true)
                                {
                                    *old = "ptr".into();
                                }
                            }
                            if merged
                                .iter()
                                .zip(&sig.1)
                                .enumerate()
                                .all(|(k, (old, new))| {
                                    old == new
                                        || (old == "ptr"
                                            && nullable.get(&(name.clone(), k)) == Some(&true))
                                })
                            {
                                previous.1 = merged;
                                continue;
                            }
                        }
                        if previous != &sig {
                            return Err(format!("LLVM emission: incompatible call signatures for `{name}`: {previous:?} and {sig:?}; an explicit ABI adapter is required"));
                        }
                    } else {
                        signatures.insert(name, sig);
                    }
                }
            }
        }
    }
    for f in &program.funcs {
        for b in &f.blocks {
            for inst in &b.insts {
                if let Op::FuncAddr(sym) = inst.op {
                    if !signatures.contains_key(interner.resolve(sym)) {
                        return Err(format!(
                            "LLVM emission: cannot infer signature of external function `{}`",
                            interner.resolve(sym)
                        ));
                    }
                }
            }
        }
    }
    Ok(signatures)
}

fn math_intrinsic(op: &Op, ty: &MirType) -> Result<String, String> {
    let suffix = match ty.lane_type() {
        MirType::F16 => "f16",
        MirType::BF16 => "bf16",
        MirType::F32 => "f32",
        MirType::F64 => "f64",
        _ => return Err(format!("LLVM emission: floating intrinsic on {ty:?}")),
    };
    let suffix = match ty {
        MirType::Vec(_, n) => format!("v{n}{suffix}"),
        _ => suffix.to_owned(),
    };
    let name = match op {
        Op::Fma(..) => "fma",
        Op::Sqrt(_) => "sqrt",
        Op::Round(mode, _) => match mode {
            wukong_mir::RoundMode::Nearest => "roundeven",
            wukong_mir::RoundMode::Floor => "floor",
            wukong_mir::RoundMode::Ceil => "ceil",
            wukong_mir::RoundMode::Trunc => "trunc",
        },
        _ => unreachable!(),
    };
    Ok(format!("llvm.{name}.{suffix}"))
}

fn intrinsic_type_suffix(ty: &MirType) -> String {
    match ty {
        MirType::Vec(lane, n) => format!("v{n}{}", intrinsic_type_suffix(lane)),
        MirType::F16 => "f16".into(),
        MirType::BF16 => "bf16".into(),
        MirType::F32 => "f32".into(),
        MirType::F64 => "f64".into(),
        _ => llvm_ty(ty),
    }
}

fn saturating_target(to: &MirType) -> MirType {
    // The interpreter's boolean fallback saturates at i32 width, then keeps the low bit.
    match to {
        MirType::I1 => MirType::I32,
        MirType::Vec(lane, n) if **lane == MirType::I1 => MirType::Vec(Box::new(MirType::I32), *n),
        _ => to.clone(),
    }
}

fn saturating_cast_name(kind: CastKind, from: &MirType, to: &MirType) -> String {
    format!(
        "llvm.{}.sat.{}.{}",
        cast_name(kind),
        intrinsic_type_suffix(to),
        intrinsic_type_suffix(from)
    )
}

fn emit_function(
    out: &mut String,
    f: &Function,
    interner: &Interner,
    overloads: &HashSet<Symbol>,
    declarations: &BTreeMap<String, Signature>,
) {
    // Inline constants: map const-producing values to literal operand strings.
    let mut consts: HashMap<u32, String> = HashMap::new();
    for b in &f.blocks {
        for inst in &b.insts {
            if let Some(r) = inst.result {
                match &inst.op {
                    Op::ConstInt(v, _) => {
                        consts.insert(r.0, v.to_string());
                    }
                    Op::ConstFloat(v, ty) => {
                        consts.insert(r.0, fmt_float(*v, ty));
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
        overloads,
        declarations,
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
    overloads: &'a HashSet<Symbol>,
    declarations: &'a BTreeMap<String, Signature>,
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
        self.emit_term(out, b);
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
                        entries.push(format!("[ {}, %{} ]", self.operand(*arg), pred));
                    }
                }
            }
            let _ = writeln!(out, "  %v{} = phi {ty} {}", param.0, entries.join(", "));
        }
    }

    fn emit_inst(&self, out: &mut String, inst: &wukong_mir::Inst) {
        let res = inst
            .result
            .map(|r| format!("%v{} = ", r.0))
            .unwrap_or_default();
        let line = match &inst.op {
            Op::ConstInt(..) | Op::ConstFloat(..) | Op::Barrier => return,
            Op::Bin(op, l, r) => {
                let ty = self.ty(inst.result.unwrap());
                let value = inst.result.unwrap();
                let mir_ty = self.f.value_type(value);
                if matches!(op, BinOp::Shl | BinOp::LShr | BinOp::AShr) {
                    let width = match mir_ty.lane_type() {
                        MirType::I1 => 1,
                        MirType::I8 => 8,
                        MirType::I16 => 16,
                        MirType::I32 => 32,
                        MirType::I64 => 64,
                        _ => unreachable!("integer shift type"),
                    };
                    let mask = int_literal(mir_ty, width - 1);
                    let _ = writeln!(
                        out,
                        "  %v{}.shift = and {ty} {}, {mask}",
                        value.0,
                        self.operand(*r)
                    );
                    let _ = writeln!(
                        out,
                        "  {res}{} {ty} {}, %v{}.shift",
                        bin_name(*op),
                        self.operand(*l),
                        value.0
                    );
                    return;
                }
                if matches!(op, BinOp::SDiv | BinOp::UDiv | BinOp::SRem | BinOp::URem) {
                    self.emit_integer_div(out, value, *op, *l, *r);
                    return;
                }
                format!(
                    "{res}{} {ty} {}, {}",
                    bin_name(*op),
                    self.operand(*l),
                    self.operand(*r)
                )
            }
            Op::Alloca { ty, .. } => format!("{res}alloca {}", llvm_ty(ty)),
            Op::Cmp(op, l, r) => {
                let opty = self.ty(*l);
                let (instr, pred) = cmp_instr(*op);
                let result = inst.result.unwrap();
                if let MirType::Vec(lane, n) = self.f.value_type(result) {
                    if **lane != MirType::I1 {
                        let ty = self.ty(result);
                        return {
                            let _ = writeln!(
                                out,
                                "  %v{}.mask = {instr} {pred} {opty} {}, {}",
                                result.0,
                                self.operand(*l),
                                self.operand(*r)
                            );
                            let _ =
                                writeln!(out, "  {res}sext <{n} x i1> %v{}.mask to {ty}", result.0);
                        };
                    }
                }
                format!(
                    "{res}{instr} {pred} {opty} {}, {}",
                    self.operand(*l),
                    self.operand(*r)
                )
            }
            Op::Neg(v) => {
                let ty = self.ty(inst.result.unwrap());
                if self
                    .f
                    .value_type(inst.result.unwrap())
                    .lane_type()
                    .is_float()
                {
                    format!("{res}fneg {ty} {}", self.operand(*v))
                } else {
                    format!("{res}sub {ty} zeroinitializer, {}", self.operand(*v))
                }
            }
            Op::Not(v) => {
                let ty = self.ty(inst.result.unwrap());
                let ones = match self.f.value_type(inst.result.unwrap()) {
                    MirType::Vec(lane, n) => format!(
                        "<{}>",
                        vec![format!("{} -1", llvm_ty(lane)); *n as usize].join(", ")
                    ),
                    _ => "-1".into(),
                };
                format!("{res}xor {ty} {}, {ones}", self.operand(*v))
            }
            Op::Cast(kind, v, to) => {
                let from = self.ty(*v);
                if matches!(kind, CastKind::FpToSi | CastKind::FpToUi) {
                    let target = saturating_target(to);
                    let name = saturating_cast_name(*kind, self.f.value_type(*v), &target);
                    let target_ty = llvm_ty(&target);
                    if &target == to {
                        let _ = writeln!(
                            out,
                            "  {res}call {target_ty} @{name}({from} {})",
                            self.operand(*v)
                        );
                    } else {
                        let r = inst.result.unwrap().0;
                        let _ = writeln!(
                            out,
                            "  %v{r}.sat = call {target_ty} @{name}({from} {})",
                            self.operand(*v)
                        );
                        let _ =
                            writeln!(out, "  {res}trunc {target_ty} %v{r}.sat to {}", llvm_ty(to));
                    }
                    return;
                }
                if self.f.value_type(*v) == to {
                    // A typed identity SSA binding, including half/bfloat and opaque pointers.
                    let value = self.operand(*v);
                    let _ = writeln!(out, "  {res}select i1 true, {from} {value}, {from} {value}");
                    return;
                }
                if matches!(to, MirType::F16 | MirType::BF16)
                    && matches!(
                        self.f.value_type(*v),
                        MirType::F16 | MirType::BF16 | MirType::F64
                    )
                {
                    // MIR lowp observations round through f32; half and bfloat also cannot be
                    // converted with LLVM's same-width fptrunc/fpext.
                    let r = inst.result.unwrap().0;
                    let via = if self.f.value_type(*v) == &MirType::F64 {
                        "fptrunc"
                    } else {
                        "fpext"
                    };
                    let _ = writeln!(
                        out,
                        "  %v{r}.wide = {via} {from} {} to float",
                        self.operand(*v)
                    );
                    let _ = writeln!(out, "  {res}fptrunc float %v{r}.wide to {}", llvm_ty(to));
                    return;
                }
                format!(
                    "{res}{} {from} {} to {}",
                    cast_name(*kind),
                    self.operand(*v),
                    llvm_ty(to)
                )
            }
            Op::Select(c, a, b) => {
                let ty = self.ty(inst.result.unwrap());
                if let MirType::Vec(lane, n) = self.f.value_type(*c) {
                    if **lane != MirType::I1 {
                        let r = inst.result.unwrap().0;
                        let compare = if lane.is_float() {
                            "fcmp une"
                        } else {
                            "icmp ne"
                        };
                        let _ = writeln!(
                            out,
                            "  %v{r}.cond = {compare} {} {}, zeroinitializer",
                            self.ty(*c),
                            self.operand(*c)
                        );
                        let _ = writeln!(
                            out,
                            "  {res}select <{n} x i1> %v{r}.cond, {ty} {}, {ty} {}",
                            self.operand(*a),
                            self.operand(*b)
                        );
                        return;
                    }
                }
                format!(
                    "{res}select {} {}, {ty} {}, {ty} {}",
                    self.ty(*c),
                    self.operand(*c),
                    self.operand(*a),
                    self.operand(*b)
                )
            }
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
                    None => self
                        .declarations
                        .get(self.interner.resolve(*func))
                        .map(|s| s.0.clone())
                        .unwrap_or_else(|| "void".to_string()),
                };
                let name = call_name(self.f, *func, args, self.interner, self.overloads);
                let signature = self.declarations.get(&name);
                let argstr: Vec<String> = args
                    .iter()
                    .enumerate()
                    .map(|(k, a)| {
                        if signature.is_some_and(|s| s.1[k] == "ptr")
                            && self.f.value_type(*a).is_int()
                        {
                            "ptr null".into()
                        } else {
                            format!("{} {}", self.ty(*a), self.operand(*a))
                        }
                    })
                    .collect();
                format!("{res}call {ret_ty} @{}({})", name, argstr.join(", "))
            }
            Op::FuncAddr(func) => {
                // The address of a function as a `ptr` (an identity GEP keeps it valid IR).
                format!(
                    "{res}getelementptr i8, ptr @{}, i64 0",
                    self.interner.resolve(*func)
                )
            }
            Op::GlobalAddr(data) => {
                // The read-only address of a `.rodata` string blob as a `ptr` (declared as a private
                // constant in the module preamble by `emit_llvm_ir`; an identity GEP keeps it valid).
                format!(
                    "{res}getelementptr i8, ptr @{}, i64 0",
                    self.interner.resolve(*data)
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
            Op::ExtractLane(v, k) => {
                let vty = self.ty(*v);
                format!("{res}extractelement {vty} {}, i32 {k}", self.operand(*v))
            }
            // The lane-index ramp is a constant, and LLVM writes a non-uniform vector constant
            // inline — so it is spelled as an `add` of that literal to a zero vector, which is the
            // shortest way to give it a name in a text path that emits one instruction per MIR op.
            Op::Iota(ty) => {
                let vty = self.ty(inst.result.expect("iota has a result"));
                let (lane, n) = match ty {
                    MirType::Vec(l, n) => (llvm_ty(l), *n),
                    other => (llvm_ty(other), 1),
                };
                let items: Vec<String> = (0..n).map(|k| format!("{lane} {k}")).collect();
                format!("{res}add {vty} zeroinitializer, <{}>", items.join(", "))
            }
            Op::Fma(a, b, c) => {
                let r = inst.result.unwrap();
                let ty = self.ty(r);
                let name =
                    math_intrinsic(&inst.op, self.f.value_type(r)).expect("validated FMA type");
                format!(
                    "{res}call {ty} @{name}({ty} {}, {ty} {}, {ty} {})",
                    self.operand(*a),
                    self.operand(*b),
                    self.operand(*c),
                )
            }
            Op::Sqrt(a) | Op::Round(_, a) => {
                let r = inst.result.unwrap();
                let ty = self.ty(r);
                let name = math_intrinsic(&inst.op, self.f.value_type(r))
                    .expect("intrinsic types checked before emission");
                format!("{res}call {ty} @{name}({ty} {})", self.operand(*a))
            }
            Op::VecKernelCall {
                kernel,
                ptrs,
                scalars,
                n,
            } => {
                // The synthesized 256-bit AVX2 kernel, emitted as an external void call (like the
                // recognized `wukong_*` runtime kernels above). This text-IR path is not the
                // differential oracle and is not assembled here — the AVX2 body lives only in the
                // Cranelift backend — so a well-formed, round-trippable call is the faithful form.
                format!(
                    "{res}call {} @__wukong_veckernel_{}_{kernel}(ptr {}, ptr {}, {} {})",
                    inst.result
                        .map(|r| self.ty(r))
                        .unwrap_or_else(|| "void".into()),
                    self.interner.resolve(self.f.name),
                    self.operand(*ptrs),
                    self.operand(*scalars),
                    self.ty(*n),
                    self.operand(*n),
                )
            }
        };
        let _ = writeln!(out, "  {line}");
    }

    fn emit_integer_div(
        &self,
        out: &mut String,
        result: ValueId,
        op: BinOp,
        l: ValueId,
        r: ValueId,
    ) {
        let id = result.0;
        let mir_ty = self.f.value_type(result);
        let ty = llvm_ty(mir_ty);
        let bool_ty = match mir_ty {
            MirType::Vec(_, n) => format!("<{n} x i1>"),
            _ => "i1".into(),
        };
        let one = int_literal(mir_ty, 1);
        let neg1 = int_literal(mir_ty, -1);
        let lhs = self.operand(l);
        let rhs = self.operand(r);
        let _ = writeln!(out, "  %v{id}.zero = icmp eq {ty} {rhs}, zeroinitializer");
        let signed = matches!(op, BinOp::SDiv | BinOp::SRem);
        let danger = if signed {
            let _ = writeln!(out, "  %v{id}.neg1 = icmp eq {ty} {rhs}, {neg1}");
            let _ = writeln!(
                out,
                "  %v{id}.danger = or {bool_ty} %v{id}.zero, %v{id}.neg1"
            );
            format!("%v{id}.danger")
        } else {
            format!("%v{id}.zero")
        };
        let _ = writeln!(
            out,
            "  %v{id}.safe = select {bool_ty} {danger}, {ty} {one}, {ty} {rhs}"
        );
        let _ = writeln!(
            out,
            "  %v{id}.quot = {} {ty} {lhs}, %v{id}.safe",
            bin_name(op)
        );
        let quotient = if op == BinOp::SDiv {
            let _ = writeln!(out, "  %v{id}.neg = sub {ty} zeroinitializer, {lhs}");
            let _ = writeln!(out, "  %v{id}.wrapped = select {bool_ty} %v{id}.neg1, {ty} %v{id}.neg, {ty} %v{id}.quot");
            format!("%v{id}.wrapped")
        } else {
            format!("%v{id}.quot")
        };
        let _ = writeln!(
            out,
            "  %v{id} = select {bool_ty} %v{id}.zero, {ty} zeroinitializer, {ty} {quotient}"
        );
    }

    fn emit_term(&self, out: &mut String, block: &BasicBlock) {
        let line = match &block.term {
            Terminator::Ret(None) => "ret void".to_string(),
            Terminator::Ret(Some(v)) => format!("ret {} {}", self.ty(*v), self.operand(*v)),
            Terminator::Br { target, .. } => format!("br label %bb{}", target.0),
            Terminator::CondBr {
                cond,
                then_blk,
                else_blk,
                ..
            } => {
                let else_label = if then_blk == else_blk {
                    format!("bb{}.else", block.id.0)
                } else {
                    format!("bb{}", else_blk.0)
                };
                let mut line = format!(
                    "br i1 {}, label %bb{}, label %{else_label}",
                    self.operand(*cond),
                    then_blk.0
                );
                if then_blk == else_blk {
                    let _ = write!(line, "\n{else_label}:\n  br label %bb{}", else_blk.0);
                }
                line
            }
            Terminator::Switch {
                scrutinee,
                cases,
                default,
                ..
            } => {
                let id = block.id.0;
                let ty = self.ty(*scrutinee);
                let mut line = format!(
                    "switch {ty} {}, label %bb{id}.sd [",
                    self.operand(*scrutinee)
                );
                for (i, c) in cases.iter().enumerate() {
                    let _ = write!(line, " {ty} {}, label %bb{id}.s{i}", c.value);
                }
                line.push_str(" ]");
                for (i, c) in cases.iter().enumerate() {
                    let _ = write!(line, "\nbb{id}.s{i}:\n  br label %bb{}", c.target.0);
                }
                let _ = write!(line, "\nbb{id}.sd:\n  br label %bb{}", default.0);
                line
            }
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
        FRem => "frem",
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
        Fone => ("fcmp", "une"),
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

// LLVM's hexadecimal float spelling uses the bits of an exactly widened double for f32.
// This handles tiny values, signed zero, infinities and NaNs without decimal representability
// restrictions. Low-precision literals share the runtime's rounding contract.
fn int_literal(ty: &MirType, value: i64) -> String {
    match ty {
        MirType::Vec(lane, n) => format!(
            "<{}>",
            vec![format!("{} {value}", llvm_ty(lane)); *n as usize].join(", ")
        ),
        _ => value.to_string(),
    }
}

fn fmt_float(v: f64, ty: &MirType) -> String {
    match ty {
        MirType::F16 => format!("0xH{:04X}", wukong_runtime::f32_to_f16_bits(v as f32)),
        MirType::BF16 => format!("0xR{:04X}", wukong_runtime::f32_to_bf16_bits(v as f32)),
        MirType::F32 => format!("0x{:016X}", (v as f32 as f64).to_bits()),
        MirType::F64 => format!("0x{:016X}", v.to_bits()),
        _ => unreachable!("floating constant has non-floating type"),
    }
}

#[cfg(test)]
mod closeout_tests;

#[cfg(test)]
mod semantic_tests;

#[cfg(test)]
mod tests {
    use super::*;
    use wukong_span::SourceId;

    fn ir(src: &str) -> String {
        ir_opt(src, 0)
    }

    fn ir_opt(src: &str, opt: u8) -> String {
        let mut interner = Interner::new();
        let (module, _) = wukong_parser::parse_module(src, SourceId(0), &mut interner);
        let (sema, sd) = wukong_sema::check(&module, &interner);
        assert!(sd.iter().all(|d| !d.is_error()), "sema: {sd:?}");
        let (mut program, _) = wukong_mir_build::lower_program(&module, &sema, &mut interner);
        wukong_opt::optimize(&mut program, opt);
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
