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
//!
//! ## The language the module implements is the oracle's
//!
//! The interpreter is the specification, and the module must mean what it means:
//!
//! * **`f16`/`bf16` values live in `float` registers.** The interpreter keeps a sub-`f32` value as
//!   its `f32` rounding and snaps it to the half/bfloat grid only at the observation boundaries — a
//!   constant, an `fpext`/`fptrunc`, a load and a store (`wukong_interp`'s "round at
//!   store/cast/load" model, which Cranelift mirrors with `f32` registers). Emitting `fadd half`
//!   would round every intermediate to the narrow grid and compute a different program, so a half
//!   value is a `float` SSA value here and `half`/`bfloat` appear only as memory types and as the
//!   target of the explicit grid rounding (`fptrunc` + `fpext`).
//! * **`!=` on floats is unordered** (`fcmp une`), integer division/remainder by zero is 0 and
//!   `INT_MIN / -1` wraps, and shift counts are masked to the width — each guarded explicitly, since
//!   LLVM makes all three poison.
//! * **Booleans occupy one byte in memory** with only the low bit significant, as in both executing
//!   backends; an `i1` (or `<n x i1>`) load/store goes through `i8` storage, never LLVM's packed
//!   vector-of-`i1` layout.
//!
//! ## The module links against the same runtime as `--emit=exe`
//!
//! `print`/`assert` are lowered exactly as the Cranelift object lowers them — to the
//! `wukong_rt_print_{i64,u64,f64,str}` / `wukong_rt_assert` ABI with the same argument widening —
//! and every `wukong_*` runtime kernel is called with its real C ABI (every integer argument
//! widened to `i64`, a literal-zero optional pointer passed as `ptr null`). The general vectorizer's
//! synthesized kernels, which the Cranelift backend installs as raw AVX2 bytes, are *defined* in the
//! module from their backend-agnostic recipe (see [`vec_kernels`]). So the only external symbols a
//! module references are the ones the AOT link already provides.

use std::cell::Cell;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fmt::Write as _;

use wukong_backend::{Artifact, Backend};
use wukong_mir::{
    BasicBlock, BinOp, CastKind, CmpOp, Function, IntUnOp, MirType, Op, Program, Terminator,
    ValueId,
};
use wukong_span::{Interner, Symbol};

mod vec_kernels;

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

/// The target triple the module is tagged with: the compiling host's, which is also what the
/// Cranelift object and the AOT runtime link are built for. No `target datalayout` is written —
/// `llc`, `opt` and `clang` all install the target's own layout for a module that names a triple,
/// and a hand-maintained layout string drifts with every LLVM release.
fn host_triple() -> Option<&'static str> {
    if cfg!(all(
        target_arch = "x86_64",
        target_os = "windows",
        target_env = "msvc"
    )) {
        Some("x86_64-pc-windows-msvc")
    } else if cfg!(all(
        target_arch = "x86_64",
        target_os = "windows",
        target_env = "gnu"
    )) {
        Some("x86_64-pc-windows-gnu")
    } else if cfg!(all(
        target_arch = "x86_64",
        target_os = "linux",
        target_env = "gnu"
    )) {
        Some("x86_64-unknown-linux-gnu")
    } else if cfg!(all(
        target_arch = "aarch64",
        target_os = "linux",
        target_env = "gnu"
    )) {
        Some("aarch64-unknown-linux-gnu")
    } else if cfg!(all(target_arch = "x86_64", target_os = "macos")) {
        Some("x86_64-apple-darwin")
    } else if cfg!(all(target_arch = "aarch64", target_os = "macos")) {
        Some("aarch64-apple-darwin")
    } else {
        None
    }
}

/// Emit a module, declining only constructs whose meaning cannot be expressed faithfully. External
/// declarations name exactly the runtime the AOT link provides (see the crate docs).
pub fn try_emit_llvm_ir(program: &Program, interner: &Interner) -> Result<String, String> {
    let defined: HashSet<Symbol> = program.funcs.iter().map(|f| f.name).collect();
    let declarations = declarations(program, interner, &defined)?;
    let mut out = String::new();
    out.push_str("; Wukong -> LLVM IR\n");
    if let Some(triple) = host_triple() {
        let _ = writeln!(out, "target triple = \"{triple}\"");
    }
    out.push('\n');
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
        emit_function(&mut out, f, interner, &declarations, &defined);
        out.push('\n');
    }
    Ok(out)
}

type Signature = (String, Vec<String>);

// The runtime ABI the Cranelift object calls for the print/assert intrinsics (see
// `wukong_codegen_cranelift::lower_call`). Linking a module against the `--emit=exe` runtime needs
// the module to use exactly these symbols and argument widths.
const RT_PRINT_I64: &str = "wukong_rt_print_i64";
const RT_PRINT_U64: &str = "wukong_rt_print_u64";
const RT_PRINT_F64: &str = "wukong_rt_print_f64";
const RT_PRINT_STR: &str = "wukong_rt_print_str";
const RT_ASSERT: &str = "wukong_rt_assert";

/// A built-in the MIR calls by its source name and the runtime implements under another.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Intrinsic {
    PrintInt,
    PrintUint,
    PrintFloat,
    PrintStr,
    Assert,
}

impl Intrinsic {
    /// The same classification the Cranelift backend applies (`classify_intrinsic`).
    fn classify(name: &str, first_is_float: bool) -> Option<Intrinsic> {
        match name {
            "print" | "println" => Some(if first_is_float {
                Intrinsic::PrintFloat
            } else {
                Intrinsic::PrintInt
            }),
            "print_str" | "println_str" => Some(Intrinsic::PrintStr),
            "print_u" | "println_u" => Some(Intrinsic::PrintUint),
            "assert" => Some(Intrinsic::Assert),
            _ => None,
        }
    }

    fn symbol(self) -> &'static str {
        match self {
            Intrinsic::PrintInt => RT_PRINT_I64,
            Intrinsic::PrintUint => RT_PRINT_U64,
            Intrinsic::PrintFloat => RT_PRINT_F64,
            Intrinsic::PrintStr => RT_PRINT_STR,
            Intrinsic::Assert => RT_ASSERT,
        }
    }

    fn param(self) -> &'static str {
        match self {
            Intrinsic::PrintInt | Intrinsic::PrintUint | Intrinsic::Assert => "i64",
            Intrinsic::PrintFloat => "double",
            Intrinsic::PrintStr => "ptr",
        }
    }
}

/// How a `Call` reaches its target.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Callee {
    /// A function defined in this program: called with its own MIR signature.
    Defined,
    /// A print/assert built-in, routed to the runtime's `wukong_rt_*` ABI.
    Intrinsic(Intrinsic),
    /// A `wukong_*` runtime kernel: every integer argument widened to `i64`, as Cranelift does.
    Runtime,
    /// Any other external symbol: declared from the call's MIR types.
    Foreign,
}

fn callee(
    f: &Function,
    func: Symbol,
    args: &[ValueId],
    interner: &Interner,
    defined: &HashSet<Symbol>,
) -> Callee {
    if defined.contains(&func) {
        return Callee::Defined;
    }
    let name = interner.resolve(func);
    let first_is_float = args.first().is_some_and(|a| f.value_type(*a).is_float());
    if let Some(intrinsic) = Intrinsic::classify(name, first_is_float) {
        Callee::Intrinsic(intrinsic)
    } else if name.starts_with("wukong_") {
        Callee::Runtime
    } else {
        Callee::Foreign
    }
}

/// The register (SSA) type of a MIR value type: `f16`/`bf16` lanes compute in `f32` (see the crate
/// docs), everything else is itself.
fn reg_mir(t: &MirType) -> MirType {
    match t {
        MirType::F16 | MirType::BF16 => MirType::F32,
        MirType::Vec(lane, n) if is_half(lane) => MirType::Vec(Box::new(MirType::F32), *n),
        other => other.clone(),
    }
}

fn reg_ty(t: &MirType) -> String {
    llvm_ty(&reg_mir(t))
}

/// Whether `t`'s lanes are a 16-bit float (stored narrow, computed in `f32`).
fn is_half(t: &MirType) -> bool {
    matches!(t.lane_type(), MirType::F16 | MirType::BF16)
}

/// The in-memory type of `t`: booleans take a whole byte (as in the interpreter and Cranelift —
/// LLVM would bit-pack a `<n x i1>`), everything else is its own LLVM type.
fn mem_ty(t: &MirType) -> String {
    match t {
        MirType::I1 => "i8".into(),
        MirType::Vec(lane, n) | MirType::Array(lane, n) if **lane == MirType::I1 => {
            if matches!(t, MirType::Vec(..)) {
                format!("<{n} x i8>")
            } else {
                format!("[{n} x i8]")
            }
        }
        MirType::Array(e, n) => format!("[{n} x {}]", mem_ty(e)),
        other => llvm_ty(other),
    }
}

/// The type of a runtime-kernel argument slot: integers are widened to the runtime's `i64`.
fn runtime_param_ty(t: &MirType) -> String {
    if t.is_int() {
        "i64".into()
    } else {
        reg_ty(t)
    }
}

/// For each `(runtime symbol, argument position)`, whether every non-pointer argument passed there
/// is a literal integer zero — the MIR spelling of an absent optional pointer (a null bias). Such a
/// slot is declared `ptr` and the zero is passed as `ptr null`.
fn nullable_pointer_slots(
    program: &Program,
    interner: &Interner,
    defined: &HashSet<Symbol>,
) -> HashMap<(String, usize), bool> {
    let mut nullable: HashMap<(String, usize), bool> = HashMap::new();
    for f in &program.funcs {
        let zeros: HashSet<ValueId> = f
            .blocks
            .iter()
            .flat_map(|b| &b.insts)
            .filter(|ins| matches!(ins.op, Op::ConstInt(0, _)))
            .filter_map(|ins| ins.result)
            .collect();
        for inst in f.blocks.iter().flat_map(|b| &b.insts) {
            if let Op::Call { func, args } = &inst.op {
                if callee(f, *func, args, interner, defined) != Callee::Runtime {
                    continue;
                }
                let name = interner.resolve(*func);
                for (k, a) in args.iter().enumerate() {
                    let valid = f.value_type(*a) == &MirType::Ptr
                        || (f.value_type(*a).is_int() && zeros.contains(a));
                    *nullable.entry((name.to_owned(), k)).or_insert(true) &= valid;
                }
            }
        }
    }
    nullable
}

fn declarations(
    program: &Program,
    interner: &Interner,
    defined: &HashSet<Symbol>,
) -> Result<BTreeMap<String, Signature>, String> {
    let mut signatures: BTreeMap<String, Signature> = BTreeMap::new();
    let nullable = nullable_pointer_slots(program, interner, defined);
    for f in &program.funcs {
        signatures.insert(
            interner.resolve(f.name).to_owned(),
            (
                reg_ty(&f.ret),
                f.params.iter().map(|v| reg_ty(f.value_type(*v))).collect(),
            ),
        );
    }
    let defined_signatures = signatures.clone();
    let mut required: Vec<(String, Signature)> = Vec::new();
    for f in &program.funcs {
        for kernel in &f.vec_kernels {
            vec_kernels::check(kernel)?;
            let (fma, sqrt) = vec_kernels::intrinsics(kernel);
            if fma {
                required.push((
                    "llvm.fma.f32".into(),
                    ("float".into(), vec!["float".into(); 3]),
                ));
            }
            if sqrt {
                required.push((
                    "llvm.sqrt.f32".into(),
                    ("float".into(), vec!["float".into()]),
                ));
            }
        }
        for inst in f.blocks.iter().flat_map(|b| &b.insts) {
            let call: Option<(String, Signature)> = match &inst.op {
                Op::Call { func, args } => {
                    let name = interner.resolve(*func).to_owned();
                    let ret = inst
                        .result
                        .map(|r| reg_ty(f.value_type(r)))
                        .unwrap_or_else(|| "void".into());
                    match callee(f, *func, args, interner, defined) {
                        Callee::Defined => None,
                        Callee::Intrinsic(intrinsic) => {
                            if intrinsic != Intrinsic::Assert
                                && args.first().is_some_and(|a| f.value_type(*a).is_vector())
                            {
                                return Err(format!(
                                    "LLVM emission: `{name}` of a vector value has no runtime ABI"
                                ));
                            }
                            Some((
                                intrinsic.symbol().to_owned(),
                                ("void".into(), vec![intrinsic.param().into()]),
                            ))
                        }
                        Callee::Runtime => Some((
                            name,
                            (
                                ret,
                                args.iter()
                                    .map(|a| runtime_param_ty(f.value_type(*a)))
                                    .collect(),
                            ),
                        )),
                        Callee::Foreign => Some((
                            name,
                            (ret, args.iter().map(|a| reg_ty(f.value_type(*a))).collect()),
                        )),
                    }
                }
                Op::Sqrt(v) | Op::Round(_, v) => {
                    let ty = reg_mir(f.value_type(*v));
                    let name = math_intrinsic(&inst.op, &ty)?;
                    Some((name, (llvm_ty(&ty), vec![llvm_ty(&ty)])))
                }
                Op::Fma(a, _, _) => {
                    let ty = reg_mir(f.value_type(*a));
                    Some((
                        math_intrinsic(&inst.op, &ty)?,
                        (llvm_ty(&ty), vec![llvm_ty(&ty); 3]),
                    ))
                }
                Op::IntUnary(op, v) => {
                    let ty = llvm_ty(f.value_type(*v));
                    let mut params = vec![ty.clone()];
                    if matches!(op, IntUnOp::Clz | IntUnOp::Ctz) {
                        params.push("i1".into());
                    }
                    Some((int_unary_intrinsic(*op, &ty), (ty, params)))
                }
                Op::Cast(kind @ (CastKind::FpToSi | CastKind::FpToUi), v, to) => {
                    let from = reg_mir(f.value_type(*v));
                    let target = saturating_target(to);
                    Some((
                        saturating_cast_name(*kind, &from, &target),
                        (llvm_ty(&target), vec![llvm_ty(&from)]),
                    ))
                }
                Op::Cast(CastKind::Bitcast, v, to) if is_half(f.value_type(*v)) || is_half(to) => {
                    return Err(
                        "LLVM emission: a bitcast to or from f16/bf16 has no f32-register meaning"
                            .into(),
                    );
                }
                Op::VecKernelCall { kernel, .. } => {
                    if f.vec_kernels.get(*kernel as usize).is_none() {
                        return Err(format!(
                            "LLVM emission: vector kernel #{kernel} of `{}` is not registered",
                            interner.resolve(f.name)
                        ));
                    }
                    None
                }
                _ => None,
            };
            if let Some(entry) = call {
                required.push(entry);
            }
        }
    }
    for (name, sig) in required {
        if let Some(previous) = signatures.get_mut(&name) {
            if !defined_signatures.contains_key(&name)
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
                            || (old == "ptr" && nullable.get(&(name.clone(), k)) == Some(&true))
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
    for f in &program.funcs {
        for inst in f.blocks.iter().flat_map(|b| &b.insts) {
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
    // An `extern "C"` declaration is authoritative over the call-site inference above: it carries
    // the C signedness a narrow integer needs (`signext`/`zeroext`), and it declares even a C
    // function no call site reaches.
    for e in &program.externs {
        signatures.insert(
            interner.resolve(e.name).to_owned(),
            (
                extern_abi(&e.ret, true),
                e.params.iter().map(|p| extern_abi(p, false)).collect(),
            ),
        );
    }
    Ok(signatures)
}

/// One `extern "C"` parameter or result as an LLVM type, with the `signext`/`zeroext` attribute a
/// sub-32-bit integer needs under the C ABI (written after a parameter type, before a result type).
fn extern_abi(p: &wukong_mir::ExternParam, is_ret: bool) -> String {
    let ty = llvm_ty(&p.ty);
    if !matches!(p.ty, MirType::I1 | MirType::I8 | MirType::I16) {
        return ty;
    }
    let attr = if p.signed { "signext" } else { "zeroext" };
    if is_ret {
        format!("{attr} {ty}")
    } else {
        format!("{ty} {attr}")
    }
}

fn math_intrinsic(op: &Op, ty: &MirType) -> Result<String, String> {
    let suffix = match ty.lane_type() {
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

/// The LLVM intrinsic an [`Op::IntUnary`] on the integer type `ty` (`i8`..`i64`) calls.
/// `ctlz`/`cttz` take a second `i1` operand, passed `false` so a zero input is defined (the full
/// width), which is the MIR semantics.
fn int_unary_intrinsic(op: IntUnOp, ty: &str) -> String {
    let name = match op {
        IntUnOp::Popcnt => "ctpop",
        IntUnOp::Clz => "ctlz",
        IntUnOp::Ctz => "cttz",
        IntUnOp::Bswap => "bswap",
    };
    format!("llvm.{name}.{ty}")
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
    declarations: &BTreeMap<String, Signature>,
    defined: &HashSet<Symbol>,
) {
    // Inline constants: map const-producing values to literal operand strings.
    let mut consts: HashMap<u32, String> = HashMap::new();
    for b in &f.blocks {
        for inst in &b.insts {
            if let Some(r) = inst.result {
                match &inst.op {
                    Op::ConstInt(v, ty) => {
                        consts.insert(r.0, int_const(*v, ty));
                    }
                    Op::ConstFloat(v, ty) => {
                        consts.insert(r.0, fmt_float(*v, ty));
                    }
                    _ => {}
                }
            }
        }
    }

    // A `@noalias` parameter is LLVM's `noalias` argument attribute verbatim: for the duration of
    // the call, memory reached through it is not reached through any pointer not based on it. The
    // attribute is only legal on a pointer, which every promised (buffer) parameter is.
    let params: Vec<String> = f
        .params
        .iter()
        .enumerate()
        .map(|(i, p)| {
            let attr = if f.param_attr(i).noalias && *f.value_type(*p) == MirType::Ptr {
                " noalias"
            } else {
                ""
            };
            format!("{}{} %v{}", reg_ty(f.value_type(*p)), attr, p.0)
        })
        .collect();
    let _ = writeln!(
        out,
        "define {} @{}({}) {{",
        reg_ty(&f.ret),
        interner.resolve(f.name),
        params.join(", ")
    );

    let edges = edge_args(f);
    let e = Emitter {
        f,
        interner,
        consts: &consts,
        edges: &edges,
        declarations,
        defined,
        tmp: Cell::new(0),
    };
    for b in &f.blocks {
        e.emit_block(out, b);
    }
    out.push_str("}\n");

    let name = interner.resolve(f.name);
    for (k, kernel) in f.vec_kernels.iter().enumerate() {
        out.push('\n');
        vec_kernels::emit_definition(out, &vec_kernels::kernel_symbol(name, k as u32), kernel);
    }
}

struct Emitter<'a> {
    f: &'a Function,
    interner: &'a Interner,
    consts: &'a HashMap<u32, String>,
    edges: &'a EdgeArgs,
    declarations: &'a BTreeMap<String, Signature>,
    defined: &'a HashSet<Symbol>,
    /// Counter for the `%tN` temporaries an instruction needs besides its own result.
    tmp: Cell<u32>,
}

impl Emitter<'_> {
    fn operand(&self, v: ValueId) -> String {
        match self.consts.get(&v.0) {
            Some(s) => s.clone(),
            None => format!("%v{}", v.0),
        }
    }

    /// The register type of value `v`.
    fn ty(&self, v: ValueId) -> String {
        reg_ty(self.f.value_type(v))
    }

    fn mir(&self, v: ValueId) -> &MirType {
        self.f.value_type(v)
    }

    fn fresh(&self) -> String {
        let n = self.tmp.get();
        self.tmp.set(n + 1);
        format!("%t{n}")
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

    /// A typed identity binding of `value` to `dst` (LLVM has no plain copy instruction).
    fn bind(&self, out: &mut String, dst: &str, ty: &str, value: &str) {
        let _ = writeln!(out, "  {dst} = select i1 true, {ty} {value}, {ty} {value}");
    }

    /// Widen an integer-like value to `i64` (`i1` always zero-extends; pointers become their
    /// address), returning the operand spelling.
    fn to_i64(&self, out: &mut String, v: ValueId, signed: bool) -> String {
        let ty = self.mir(v).clone();
        let operand = self.operand(v);
        match ty {
            MirType::I64 => operand,
            MirType::Ptr => {
                let t = self.fresh();
                let _ = writeln!(out, "  {t} = ptrtoint ptr {operand} to i64");
                t
            }
            MirType::I1 | MirType::I8 | MirType::I16 | MirType::I32 => {
                let t = self.fresh();
                let ext = if signed && ty != MirType::I1 {
                    "sext"
                } else {
                    "zext"
                };
                let _ = writeln!(out, "  {t} = {ext} {} {operand} to i64", llvm_ty(&ty));
                t
            }
            _ => operand,
        }
    }

    /// A float register value as `double`, returning the operand spelling.
    fn to_f64(&self, out: &mut String, v: ValueId) -> String {
        let operand = self.operand(v);
        if reg_mir(self.mir(v)) == MirType::F64 {
            operand
        } else {
            let t = self.fresh();
            let _ = writeln!(out, "  {t} = fpext float {operand} to double");
            t
        }
    }

    /// Round the `f32`-register value `value` (of register type `reg`) onto the grid of `grid`'s
    /// half/bfloat lane and widen it back, returning the rounded register value.
    fn round_to_grid(
        &self,
        out: &mut String,
        value: &str,
        reg: &MirType,
        grid: &MirType,
    ) -> String {
        let lane = grid.lane_type().clone();
        let narrow = match reg {
            MirType::Vec(_, n) => MirType::Vec(Box::new(lane), *n),
            _ => lane,
        };
        let (reg, narrow) = (llvm_ty(reg), llvm_ty(&narrow));
        let t = self.fresh();
        let w = self.fresh();
        let _ = writeln!(out, "  {t} = fptrunc {reg} {value} to {narrow}");
        let _ = writeln!(out, "  {w} = fpext {narrow} {t} to {reg}");
        w
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
            Op::Alloca { ty, .. } => format!("{res}alloca {}", mem_ty(ty)),
            Op::Cmp(op, l, r) => {
                let opty = self.ty(*l);
                let (instr, pred) = cmp_instr(*op);
                let result = inst.result.unwrap();
                if let MirType::Vec(lane, n) = self.f.value_type(result) {
                    if **lane != MirType::I1 {
                        let ty = self.ty(result);
                        let _ = writeln!(
                            out,
                            "  %v{}.mask = {instr} {pred} {opty} {}, {}",
                            result.0,
                            self.operand(*l),
                            self.operand(*r)
                        );
                        let _ = writeln!(out, "  {res}sext <{n} x i1> %v{}.mask to {ty}", result.0);
                        return;
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
                self.emit_cast(out, inst.result.unwrap(), *kind, *v, to);
                return;
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
            Op::Load(p, ty) => {
                let r = inst.result.expect("a load has a result").0;
                let ptr = self.operand(*p);
                if is_half(ty) || ty.lane_type() == &MirType::I1 {
                    // Narrow storage: load the stored bits, then widen to the register form (a
                    // half/bfloat widens exactly; a boolean byte keeps only its low bit, as the
                    // interpreter's load does).
                    let mem = mem_ty(ty);
                    let widen = if is_half(ty) { "fpext" } else { "trunc" };
                    let _ = writeln!(out, "  %v{r}.mem = load {mem}, ptr {ptr}");
                    format!("{res}{widen} {mem} %v{r}.mem to {}", reg_ty(ty))
                } else {
                    format!("{res}load {}, ptr {ptr}", llvm_ty(ty))
                }
            }
            Op::Store { ptr, value } => {
                let vt = self.mir(*value);
                let reg = self.ty(*value);
                let v = self.operand(*value);
                let p = self.operand(*ptr);
                if is_half(vt) || vt.lane_type() == &MirType::I1 {
                    // Round to the stored grid (or widen a boolean to its byte) and store narrow.
                    let mem = mem_ty(vt);
                    let narrow = if is_half(vt) { "fptrunc" } else { "zext" };
                    let t = self.fresh();
                    let _ = writeln!(out, "  {t} = {narrow} {reg} {v} to {mem}");
                    format!("store {mem} {t}, ptr {p}")
                } else {
                    format!("store {reg} {v}, ptr {p}")
                }
            }
            Op::Gep { ptr, index, elem } => {
                let idxty = self.ty(*index);
                format!(
                    "{res}getelementptr {}, ptr {}, {idxty} {}",
                    mem_ty(elem),
                    self.operand(*ptr),
                    self.operand(*index)
                )
            }
            Op::Call { func, args } => {
                match callee(self.f, *func, args, self.interner, self.defined) {
                    Callee::Intrinsic(intrinsic) => {
                        self.emit_intrinsic(out, intrinsic, args);
                        return;
                    }
                    Callee::Runtime => {
                        self.emit_runtime_call(out, &res, inst.result, *func, args);
                        return;
                    }
                    Callee::Defined | Callee::Foreign => {}
                }
                let name = self.interner.resolve(*func);
                // A discarded result of a defined function is still produced by the callee.
                let ret_ty = match inst.result {
                    Some(r) => self.ty(r),
                    None => self
                        .declarations
                        .get(name)
                        .map(|s| s.0.clone())
                        .unwrap_or_else(|| "void".to_string()),
                };
                let argstr: Vec<String> = args
                    .iter()
                    .map(|a| format!("{} {}", self.ty(*a), self.operand(*a)))
                    .collect();
                format!("{res}call {ret_ty} @{name}({})", argstr.join(", "))
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
                // all-zero mask.
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
                let name = math_intrinsic(&inst.op, &reg_mir(self.f.value_type(r)))
                    .expect("validated FMA type");
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
                let name = math_intrinsic(&inst.op, &reg_mir(self.f.value_type(r)))
                    .expect("intrinsic types checked before emission");
                format!("{res}call {ty} @{name}({ty} {})", self.operand(*a))
            }
            Op::IntUnary(op, a) => {
                let ty = self.ty(*a);
                let name = int_unary_intrinsic(*op, &ty);
                if matches!(op, IntUnOp::Clz | IntUnOp::Ctz) {
                    format!(
                        "{res}call {ty} @{name}({ty} {}, i1 false)",
                        self.operand(*a)
                    )
                } else {
                    format!("{res}call {ty} @{name}({ty} {})", self.operand(*a))
                }
            }
            Op::VecKernelCall {
                kernel,
                ptrs,
                scalars,
                n,
            } => {
                // The synthesized vector kernel, defined in this module from its recipe (see
                // `vec_kernels`). The element count is zero-extended, as the Cranelift call does.
                let count = self.to_i64(out, *n, false);
                format!(
                    "{res}call {} @{}(ptr {}, ptr {}, i64 {count})",
                    inst.result
                        .map(|r| self.ty(r))
                        .unwrap_or_else(|| "void".into()),
                    vec_kernels::kernel_symbol(self.interner.resolve(self.f.name), *kernel),
                    self.operand(*ptrs),
                    self.operand(*scalars),
                )
            }
        };
        let _ = writeln!(out, "  {line}");
    }

    /// Lower a cast with the interpreter's semantics (`wukong_interp::apply_cast`): a widening from a
    /// half/bfloat source first snaps the `f32` register onto the source grid, a narrowing to a
    /// half/bfloat target rounds through `f32` onto the target grid, and int→float conversion
    /// rounds once to the target's register precision.
    fn emit_cast(
        &self,
        out: &mut String,
        result: ValueId,
        kind: CastKind,
        v: ValueId,
        to: &MirType,
    ) {
        let dst = format!("%v{}", result.0);
        let from = self.mir(v).clone();
        let src = self.operand(v);
        let from_reg = reg_mir(&from);
        let to_reg = reg_mir(to);
        match kind {
            CastKind::FpToSi | CastKind::FpToUi => {
                let target = saturating_target(to);
                let name = saturating_cast_name(kind, &from_reg, &target);
                let target_ty = llvm_ty(&target);
                let from_ty = llvm_ty(&from_reg);
                if &target == to {
                    let _ = writeln!(out, "  {dst} = call {target_ty} @{name}({from_ty} {src})");
                } else {
                    let _ = writeln!(
                        out,
                        "  {dst}.sat = call {target_ty} @{name}({from_ty} {src})"
                    );
                    let _ = writeln!(
                        out,
                        "  {dst} = trunc {target_ty} {dst}.sat to {}",
                        llvm_ty(to)
                    );
                }
            }
            CastKind::FpExt if is_half(&from) => {
                let rounded = self.round_to_grid(out, &src, &from_reg, &from);
                self.convert_float(out, &dst, &rounded, &from_reg, &to_reg);
            }
            CastKind::FpTrunc if is_half(to) => {
                // `round_xx(v as f32)`: a double first rounds to f32, then onto the target grid.
                let single = shape_f32(&from_reg);
                let narrowed = if from_reg.lane_type() == &MirType::F64 {
                    let t = self.fresh();
                    let _ = writeln!(
                        out,
                        "  {t} = fptrunc {} {src} to {}",
                        llvm_ty(&from_reg),
                        llvm_ty(&single)
                    );
                    t
                } else {
                    src
                };
                let rounded = self.round_to_grid(out, &narrowed, &single, to);
                self.bind(out, &dst, &llvm_ty(&single), &rounded);
            }
            _ if from_reg == to_reg => self.bind(out, &dst, &llvm_ty(&from_reg), &src),
            _ => {
                let _ = writeln!(
                    out,
                    "  {dst} = {} {} {src} to {}",
                    cast_name(kind),
                    llvm_ty(&from_reg),
                    llvm_ty(&to_reg)
                );
            }
        }
    }

    /// Move a float register value between `float` and `double` shapes (or bind it unchanged).
    fn convert_float(
        &self,
        out: &mut String,
        dst: &str,
        value: &str,
        from_reg: &MirType,
        to_reg: &MirType,
    ) {
        if from_reg == to_reg {
            self.bind(out, dst, &llvm_ty(from_reg), value);
        } else {
            let op = if to_reg.lane_type() == &MirType::F64 {
                "fpext"
            } else {
                "fptrunc"
            };
            let _ = writeln!(
                out,
                "  {dst} = {op} {} {value} to {}",
                llvm_ty(from_reg),
                llvm_ty(to_reg)
            );
        }
    }

    /// `print`/`assert`, lowered to the runtime ABI exactly as the Cranelift object lowers them.
    fn emit_intrinsic(&self, out: &mut String, intrinsic: Intrinsic, args: &[ValueId]) {
        let arg = match intrinsic {
            Intrinsic::Assert => {
                // The runtime aborts on 0. A float is truthy when `!= 0.0` (unordered, so NaN
                // passes, as in the interpreter); a vector is never truthy.
                let ok = match args.first() {
                    None => "0".to_string(),
                    Some(&a) if self.mir(a).is_vector() => "0".to_string(),
                    Some(&a) => {
                        let ty = self.mir(a).clone();
                        let v = self.operand(a);
                        let rty = self.ty(a);
                        let c = self.fresh();
                        if ty.is_float() {
                            let _ = writeln!(out, "  {c} = fcmp une {rty} {v}, 0.0");
                        } else if ty == MirType::Ptr {
                            let _ = writeln!(out, "  {c} = icmp ne ptr {v}, null");
                        } else {
                            let _ = writeln!(out, "  {c} = icmp ne {rty} {v}, 0");
                        }
                        let w = self.fresh();
                        let _ = writeln!(out, "  {w} = zext i1 {c} to i64");
                        w
                    }
                };
                format!("i64 {ok}")
            }
            // An argument-less print lowers to nothing, as in the Cranelift backend.
            _ => {
                let Some(&a) = args.first() else { return };
                match intrinsic {
                    Intrinsic::PrintFloat => format!("double {}", self.to_f64(out, a)),
                    Intrinsic::PrintStr => format!("ptr {}", self.operand(a)),
                    _ => format!("i64 {}", self.to_i64(out, a, true)),
                }
            }
        };
        let _ = writeln!(out, "  call void @{}({arg})", intrinsic.symbol());
    }

    /// A `wukong_*` runtime kernel call with the C ABI the runtime exports: integer arguments
    /// sign-extended to `i64` (booleans zero-extended), a literal-zero optional pointer as `ptr null`.
    fn emit_runtime_call(
        &self,
        out: &mut String,
        res: &str,
        result: Option<ValueId>,
        func: Symbol,
        args: &[ValueId],
    ) {
        let name = self.interner.resolve(func);
        let slots = self.declarations.get(name).map(|s| &s.1);
        let mut argstr: Vec<String> = Vec::with_capacity(args.len());
        for (k, a) in args.iter().enumerate() {
            let slot = slots.and_then(|s| s.get(k)).map(String::as_str);
            let ty = self.mir(*a);
            argstr.push(if slot == Some("ptr") && ty.is_int() {
                "ptr null".to_string()
            } else if ty.is_int() {
                format!("i64 {}", self.to_i64(out, *a, true))
            } else {
                format!("{} {}", self.ty(*a), self.operand(*a))
            });
        }
        let ret = result.map(|r| self.ty(r)).unwrap_or_else(|| "void".into());
        let _ = writeln!(out, "  {res}call {ret} @{name}({})", argstr.join(", "));
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

/// `float` with the vector shape of `t` (`<n x float>` for an `n`-lane vector).
fn shape_f32(t: &MirType) -> MirType {
    match t {
        MirType::Vec(_, n) => MirType::Vec(Box::new(MirType::F32), *n),
        _ => MirType::F32,
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

/// The LLVM predicate for each MIR comparison. `Fone` is the oracle's `!=` — **unordered**
/// (`une`, true when either side is NaN) — not LLVM's ordered `one`.
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

fn int_literal(ty: &MirType, value: i64) -> String {
    match ty {
        MirType::Vec(lane, n) => format!(
            "<{}>",
            vec![format!("{} {value}", llvm_ty(lane)); *n as usize].join(", ")
        ),
        _ => value.to_string(),
    }
}

/// An integer constant spelled in its type's signed range (the interpreter masks every constant to
/// its width; `i1` keeps only its low bit). A pointer constant is `null` or an `inttoptr`.
fn int_const(v: i128, ty: &MirType) -> String {
    let bits: u32 = match ty {
        MirType::I1 => return (v & 1).to_string(),
        MirType::I8 => 8,
        MirType::I16 => 16,
        MirType::I32 => 32,
        MirType::I64 => 64,
        MirType::Ptr if v == 0 => return "null".into(),
        MirType::Ptr => return format!("inttoptr (i64 {} to ptr)", v as i64),
        _ => return v.to_string(),
    };
    let shift = 128 - bits;
    ((v << shift) >> shift).to_string()
}

/// A floating constant in LLVM's hexadecimal spelling (the bits of the exactly widened double), so
/// tiny values, signed zero, infinities and NaNs need no decimal-representability argument. A
/// half/bfloat constant is its grid-rounded value held in a `float` register — the interpreter's
/// `ConstFloat` rounding, through the same runtime conversion.
fn fmt_float(v: f64, ty: &MirType) -> String {
    let x = match ty {
        MirType::F16 => wukong_runtime::round_f16(v as f32) as f64,
        MirType::BF16 => wukong_runtime::round_bf16(v as f32) as f64,
        MirType::F32 => v as f32 as f64,
        MirType::F64 => v,
        _ => unreachable!("floating constant has non-floating type"),
    };
    format!("0x{:016X}", x.to_bits())
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
        // A void call (no result) must not panic on a missing result type (regression for indexing
        // value_types with a dummy id), and `print` reaches the runtime's i64 print ABI — the
        // symbol the `--emit=exe` runtime defines — never an undefined `@print`.
        let out = ir("fn main() -> i32 { print(42); return 0; }");
        assert!(
            out.contains("declare void @wukong_rt_print_i64(i64)"),
            "{out}"
        );
        assert!(out.contains("call void @wukong_rt_print_i64(i64 "), "{out}");
        assert!(!out.contains("@print("), "{out}");
    }

    fn corpus_fixture(name: &str) -> String {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("..")
            .join("tests")
            .join("run")
            .join(name);
        std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
    }

    /// P15.4: the emitted module must implement the oracle's scalar contracts on the corpus
    /// programs that pin them, at -O0 and -O2. `div_edge.wk` pins division/remainder by zero == 0
    /// and `INT_MIN / -1` wrapping, `shift_width.wk` pins the masked shift count, and
    /// `vectorized_edge_values.wk` returns 7777 from `if v != v` (unordered `!=`). The IR cannot be
    /// executed here, so the guard instructions themselves are the assertion; an unguarded
    /// `sdiv`/`shl` or an ordered `fcmp one` would be poison or the wrong answer under LLVM. At -O2
    /// the optimizer folds the constant operands away (with the same oracle semantics), so the
    /// guards are asserted on the -O0 form and the module-level properties at both levels.
    #[test]
    fn emitted_ir_matches_oracle_scalar_contracts() {
        for opt in [0, 2] {
            let div = ir_opt(&corpus_fixture("div_edge.wk"), opt);
            let shift = ir_opt(&corpus_fixture("shift_width.wk"), opt);
            let edge = ir_opt(&corpus_fixture("vectorized_edge_values.wk"), opt);
            if opt == 0 {
                assert!(div.contains(".zero = icmp eq"), "{div}");
                assert!(div.contains(".safe = select"), "{div}");
                assert!(div.contains(".wrapped = select"), "{div}");
                assert!(shift.contains(".shift = and"), "{shift}");
                assert!(edge.contains("fcmp une"), "{edge}");
            }
            for text in [&div, &shift, &edge] {
                assert!(!text.contains("fcmp one"), "-O{opt}:\n{text}");
                // Every external symbol is the AOT runtime's or an LLVM intrinsic: no module
                // references a symbol the `--emit=exe` link cannot provide.
                for line in text.lines().filter(|l| l.starts_with("declare ")) {
                    let name = line.split('@').nth(1).unwrap_or("");
                    assert!(
                        name.starts_with("wukong_") || name.starts_with("llvm."),
                        "-O{opt}: undefined non-runtime symbol: {line}"
                    );
                }
                assert!(!text.contains("declare void @__wukong_veckernel"), "{text}");
            }
        }
    }

    #[test]
    fn emits_phi_nodes_for_optimized_loops() {
        // After mem2reg the loop carries its counter/accumulator as block parameters; the emitter
        // must lower those to phi nodes with one entry per predecessor edge. The trip count is a
        // parameter: since 975bc417 (complete unrolling of known trip counts <= 32) a constant
        // `while i < 10` is fully unrolled and folds to `ret i32 45`, leaving no loop to carry phis.
        let out = ir_opt(
            "fn sum(n: i32) -> i32 { let mut s: i32 = 0; let mut i: i32 = 0; \
             while i < n { s = s + i; i = i + 1; } return s; }",
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

    /// A `@noalias` parameter is LLVM's `noalias` argument attribute, on that parameter only.
    #[test]
    fn a_promised_parameter_is_emitted_noalias() {
        let out = ir(
            "fn k(@noalias mut o: [f32; 4], x: [f32; 4]) { o[0] = x[0]; } \
             fn main() -> i32 { return 0; }",
        );
        assert!(
            out.contains("define void @k(ptr noalias %v0, ptr %v1)"),
            "{out}"
        );
        assert_eq!(out.matches("noalias").count(), 1, "{out}");
    }
}
