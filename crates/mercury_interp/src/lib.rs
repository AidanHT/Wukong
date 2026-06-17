//! `mercury_interp` — a from-scratch MIR interpreter.
//!
//! Zero external dependencies by design: this is the always-available execution path and the
//! oracle that differential tests compare LLVM output against. It walks the CFG block by block,
//! keeping a per-call register file and a shared flat memory for `alloca`/`load`/`store`/`gep`.

use mercury_backend::{Artifact, Backend};
use mercury_mir::{
    BasicBlock, BinOp, CastKind, CmpOp, Function, MirType, Op, Program, Terminator, ValueId,
};
use mercury_span::{Interner, Symbol};

/// A runtime value. Integers are stored width-agnostically in an `i128` and masked per result
/// type; pointers are indices into the interpreter's flat memory.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Value {
    Int(i128),
    Float(f64),
    Ptr(usize),
    /// A SIMD vector value: an index into the interpreter's `vecs` side arena (keeps `Value` cheap
    /// and `Copy`). Produced by `Splat`/vector `Load`/vector `Bin`; consumed by vector `Store`.
    VecRef(u32),
    Unit,
}

impl Value {
    fn as_int(self) -> i128 {
        match self {
            Value::Int(i) => i,
            Value::Ptr(p) => p as i128,
            Value::Float(f) => f as i128,
            Value::VecRef(_) | Value::Unit => 0,
        }
    }

    fn as_float(self) -> f64 {
        match self {
            Value::Float(f) => f,
            Value::Int(i) => i as f64,
            _ => 0.0,
        }
    }

    fn truthy(self) -> bool {
        self.as_int() != 0
    }
}

/// Pointers at or above this value encode a function index (`FUNC_TAG + idx`) rather than a memory
/// address — how `Op::FuncAddr` is represented for `parallel_for` to call back. Real `alloca`
/// memory never grows near it.
const FUNC_TAG: usize = 1 << 60;

/// The interpreter backend.
pub struct Interpreter;

impl Backend for Interpreter {
    fn name(&self) -> &'static str {
        "interpreter"
    }

    fn compile(
        &self,
        program: &Program,
        entry: Symbol,
        interner: &Interner,
    ) -> Result<Artifact, String> {
        let (exit_code, stdout) = run_with_output(program, entry, interner)?;
        Ok(Artifact::Executed { exit_code, stdout })
    }
}

/// Run `entry` (typically `main`) and return its integer result as a process exit code.
pub fn run(program: &Program, entry: Symbol, interner: &Interner) -> Result<i64, String> {
    Ok(run_with_output(program, entry, interner)?.0)
}

/// Run `entry` and return both its exit code and anything it printed.
pub fn run_with_output(
    program: &Program,
    entry: Symbol,
    interner: &Interner,
) -> Result<(i64, Vec<u8>), String> {
    let func = program
        .function(entry)
        .ok_or_else(|| format!("no entry function `{}`", interner.resolve(entry)))?;
    let mut interp = Interp {
        program,
        interner,
        memory: Vec::new(),
        stdout: Vec::new(),
        frames: Vec::new(),
        scratch: Vec::with_capacity(8),
        vecs: Vec::new(),
    };
    let result = interp.run_function(func, Vec::new())?;
    Ok((result.as_int() as i64, interp.stdout))
}

struct Interp<'a> {
    program: &'a Program,
    interner: &'a Interner,
    memory: Vec<Value>,
    stdout: Vec<u8>,
    /// Recycled register files, one per active call depth. Pooling them keeps recursion and
    /// call-heavy code from allocating a fresh vector on every call.
    frames: Vec<Vec<Value>>,
    /// Scratch buffer reused when passing block-parameter arguments across an edge.
    scratch: Vec<Value>,
    /// Backing store for SIMD vector values; `Value::VecRef(i)` indexes this. Vectors only live in
    /// registers (never in `memory`, which stays scalar), so this grows but is never aliased.
    vecs: Vec<Vec<Value>>,
}

impl<'a> Interp<'a> {
    fn run_function(&mut self, func: &Function, args: Vec<Value>) -> Result<Value, String> {
        // Take a recycled register file (or a fresh one) and size it for this function. Values
        // start as `Unit`, the interpreter's "undefined"; well-formed MIR writes before it reads.
        let mut regs = self.frames.pop().unwrap_or_default();
        regs.clear();
        regs.resize(func.value_types.len(), Value::Unit);
        for (p, a) in func.params.iter().zip(args) {
            regs[p.0 as usize] = a;
        }
        let result = self.exec(func, &mut regs);
        self.frames.push(regs);
        result
    }

    fn exec(&mut self, func: &Function, regs: &mut [Value]) -> Result<Value, String> {
        let mut cur = func.entry;
        let mut steps = 0u64;
        loop {
            steps += 1;
            if steps > 100_000_000 {
                return Err("interpreter step limit exceeded (likely an infinite loop)".into());
            }
            let block = func.block(cur);
            for inst in &block.insts {
                let rty = inst.result.map(|r| func.value_type(r));
                let v = self.eval(&inst.op, func, regs, rty)?;
                if let Some(r) = inst.result {
                    // Normalize integer results to their declared width: this gives correct
                    // two's-complement wrapping and keeps booleans (`i1`) as 0/1 (so e.g. `!true`
                    // is 0, not a sign-extended -2). Float results narrower than `f64` are rounded
                    // to their declared precision so the interpreter matches the native backend
                    // bit-for-bit (`f32` arithmetic must round at `f32`, not `f64`).
                    let rty = func.value_type(r);
                    let v = match v {
                        Value::Int(i) if rty.is_int() => Value::Int(mask(i, rty)),
                        Value::Float(f) if is_narrow_float(rty) => Value::Float(f as f32 as f64),
                        other => other,
                    };
                    regs[r.0 as usize] = v;
                }
            }
            match &block.term {
                Terminator::Ret(None) => return Ok(Value::Unit),
                Terminator::Ret(Some(v)) => return Ok(reg(regs, *v)),
                Terminator::Br { target, args } => {
                    pass_args(regs, &mut self.scratch, func.block(*target), args);
                    cur = *target;
                }
                Terminator::CondBr {
                    cond,
                    then_blk,
                    then_args,
                    else_blk,
                    else_args,
                } => {
                    let (tgt, bargs) = if reg(regs, *cond).truthy() {
                        (*then_blk, then_args)
                    } else {
                        (*else_blk, else_args)
                    };
                    pass_args(regs, &mut self.scratch, func.block(tgt), bargs);
                    cur = tgt;
                }
                Terminator::Unreachable => return Err("execution reached `unreachable`".into()),
            }
        }
    }

    fn eval(
        &mut self,
        op: &Op,
        func: &Function,
        regs: &[Value],
        rty: Option<&MirType>,
    ) -> Result<Value, String> {
        Ok(match op {
            Op::ConstInt(v, ty) => Value::Int(mask(*v, ty)),
            Op::ConstFloat(v, _) => Value::Float(*v),
            // Vector arithmetic is lane-wise, each lane rounded to the lane type (so `<n x f32>`
            // ops round at f32, matching the native backend). Scalar bins go the fast path.
            Op::Bin(b, l, r) => {
                if let Some(MirType::Vec(lane, n)) = rty {
                    let av = self.vec_lanes(reg(regs, *l));
                    let bv = self.vec_lanes(reg(regs, *r));
                    let lane = (**lane).clone();
                    let lanes: Vec<Value> = (0..*n as usize)
                        .map(|i| apply_bin(*b, av[i], bv[i], Some(&lane)))
                        .collect();
                    self.push_vec(lanes)
                } else {
                    apply_bin(*b, reg(regs, *l), reg(regs, *r), rty)
                }
            }
            Op::Cmp(c, l, r) => {
                if let Some(MirType::Vec(_, n)) = rty {
                    // Lane-wise compare → a mask vector of 1/0 lanes.
                    let av = self.vec_lanes(reg(regs, *l));
                    let bv = self.vec_lanes(reg(regs, *r));
                    let lanes: Vec<Value> = (0..*n as usize)
                        .map(|i| Value::Int(apply_cmp(*c, av[i], bv[i]) as i128))
                        .collect();
                    self.push_vec(lanes)
                } else {
                    Value::Int(apply_cmp(*c, reg(regs, *l), reg(regs, *r)) as i128)
                }
            }
            Op::Neg(v) => match reg(regs, *v) {
                Value::Float(f) => Value::Float(-f),
                other => Value::Int(-other.as_int()),
            },
            Op::Not(v) => match reg(regs, *v) {
                Value::Int(i) => Value::Int(!i),
                other => Value::Int(!other.as_int()),
            },
            Op::Cast(kind, v, to) => apply_cast(*kind, reg(regs, *v), func.value_type(*v), to),
            Op::Select(c, a, b) => {
                if let Some(MirType::Vec(_, n)) = rty {
                    // Lane-wise blend by a mask vector.
                    let m = self.vec_lanes(reg(regs, *c));
                    let av = self.vec_lanes(reg(regs, *a));
                    let bv = self.vec_lanes(reg(regs, *b));
                    let lanes: Vec<Value> = (0..*n as usize)
                        .map(|i| if m[i].truthy() { av[i] } else { bv[i] })
                        .collect();
                    self.push_vec(lanes)
                } else if reg(regs, *c).truthy() {
                    reg(regs, *a)
                } else {
                    reg(regs, *b)
                }
            }
            Op::Alloca(ty) => {
                let idx = self.memory.len();
                // An array alloca reserves `count` contiguous element slots; its result points at
                // the first. `gep` then computes `base + index` into this run.
                if let MirType::Array(elem, count) = ty {
                    let d = default_value(elem);
                    for _ in 0..*count {
                        self.memory.push(d);
                    }
                } else {
                    self.memory.push(default_value(ty));
                }
                Value::Ptr(idx)
            }
            // A vector load gathers `n` contiguous scalar slots (memory stays scalar); a scalar
            // load reads one. Both `gep` to the element address first.
            Op::Load(p, ty) => {
                let idx = ptr(reg(regs, *p))?;
                if let MirType::Vec(_, n) = ty {
                    let mut lanes = Vec::with_capacity(*n as usize);
                    for i in 0..*n as usize {
                        lanes.push(
                            *self
                                .memory
                                .get(idx + i)
                                .ok_or("vector load out of bounds")?,
                        );
                    }
                    self.push_vec(lanes)
                } else {
                    *self.memory.get(idx).ok_or("load out of bounds")?
                }
            }
            Op::Store { ptr: p, value } => {
                let idx = ptr(reg(regs, *p))?;
                match reg(regs, *value) {
                    // A vector store scatters its lanes across contiguous scalar slots.
                    Value::VecRef(vi) => {
                        let lanes = self.vecs[vi as usize].clone();
                        for (i, lane) in lanes.iter().enumerate() {
                            *self
                                .memory
                                .get_mut(idx + i)
                                .ok_or("vector store out of bounds")? = *lane;
                        }
                    }
                    val => *self.memory.get_mut(idx).ok_or("store out of bounds")? = val,
                }
                Value::Unit
            }
            Op::Gep { ptr: p, index, .. } => {
                let base = ptr(reg(regs, *p))?;
                let off = reg(regs, *index).as_int();
                Value::Ptr((base as i128 + off) as usize)
            }
            Op::Call { func, args } => {
                let argv: Vec<Value> = args.iter().map(|a| reg(regs, *a)).collect();
                let prog = self.program;
                if let Some(callee) = prog.function(*func) {
                    self.run_function(callee, argv)?
                } else {
                    let name = self.interner.resolve(*func).to_string();
                    self.intrinsic(&name, &argv)?
                }
            }
            // A function address: a pointer the interpreter tags with the function's index so the
            // `parallel_for` intrinsic can call it back. (The native backend uses a real address.)
            Op::FuncAddr(sym) => {
                let idx = self
                    .program
                    .funcs
                    .iter()
                    .position(|f| f.name == *sym)
                    .ok_or("func_addr of unknown function")?;
                Value::Ptr(FUNC_TAG + idx)
            }
            // Broadcast a scalar to every lane.
            Op::Splat(v) => {
                let n = match rty {
                    Some(MirType::Vec(_, n)) => *n as usize,
                    _ => 1,
                };
                let s = reg(regs, *v);
                self.push_vec(vec![s; n])
            }
            // Fused multiply-add `a*b + c`, single-rounded via `mul_add` so it stays bit-identical
            // to the native `fma`. Lane-wise for vectors, each lane rounded to its lane type.
            Op::Fma(a, b, c) => {
                if let Some(MirType::Vec(lane, n)) = rty {
                    let av = self.vec_lanes(reg(regs, *a));
                    let bv = self.vec_lanes(reg(regs, *b));
                    let cv = self.vec_lanes(reg(regs, *c));
                    let lane = (**lane).clone();
                    let lanes: Vec<Value> = (0..*n as usize)
                        .map(|i| apply_fma(av[i], bv[i], cv[i], Some(&lane)))
                        .collect();
                    self.push_vec(lanes)
                } else {
                    apply_fma(reg(regs, *a), reg(regs, *b), reg(regs, *c), rty)
                }
            }
        })
    }

    /// Intern a freshly-computed vector value into the side arena, returning its handle.
    fn push_vec(&mut self, lanes: Vec<Value>) -> Value {
        let i = self.vecs.len() as u32;
        self.vecs.push(lanes);
        Value::VecRef(i)
    }

    /// The lanes behind a `VecRef` (cloned out so callers can borrow `self` mutably afterwards).
    fn vec_lanes(&self, v: Value) -> Vec<Value> {
        match v {
            Value::VecRef(i) => self.vecs[i as usize].clone(),
            // A non-vector reaching a vector op is a lowering bug; treat as a single lane.
            other => vec![other],
        }
    }

    fn intrinsic(&mut self, name: &str, args: &[Value]) -> Result<Value, String> {
        match name {
            "print" | "println" => {
                let text = match args.first().copied().unwrap_or(Value::Unit) {
                    Value::Int(i) => format!("{i}\n"),
                    Value::Float(f) => format!("{f}\n"),
                    Value::Ptr(p) => format!("{p}\n"),
                    Value::VecRef(_) => "<vector>\n".to_string(),
                    Value::Unit => "\n".to_string(),
                };
                self.stdout.extend_from_slice(text.as_bytes());
                Ok(Value::Unit)
            }
            "assert" => {
                let ok = match args.first().copied().unwrap_or(Value::Unit) {
                    Value::Int(i) => i != 0,
                    Value::Float(f) => f != 0.0,
                    Value::Ptr(p) => p != 0,
                    Value::VecRef(_) | Value::Unit => false,
                };
                if ok {
                    Ok(Value::Unit)
                } else {
                    Err("assertion failed".to_string())
                }
            }
            // `mercury_parallel_for(n, body, env)` — run `body(start, end, env)` over the index
            // range. The interpreter executes the whole range sequentially in one call; since
            // parallel-for bodies have no cross-iteration dependencies this is exactly the result
            // the multi-threaded native runtime produces (so the two stay differential-equal).
            "mercury_parallel_for" => {
                let n = args.first().copied().unwrap_or(Value::Unit).as_int();
                let body = match args.get(1).copied() {
                    Some(Value::Ptr(p)) if p >= FUNC_TAG => p - FUNC_TAG,
                    _ => return Err("parallel_for body is not a function address".into()),
                };
                let env = args.get(2).copied().unwrap_or(Value::Ptr(0));
                let prog = self.program;
                let func = prog
                    .funcs
                    .get(body)
                    .ok_or("parallel_for body index out of range")?;
                self.run_function(func, vec![Value::Int(0), Value::Int(n), env])?;
                Ok(Value::Unit)
            }
            // `mercury_sgemm(a, b, c, m, k, n, beta)` (and the parallel / transposed `nn.Linear`
            // variants) — the GEMM microkernels the compiler lowers a matmul nest to. The interpreter
            // marshals its abstract (tagged-`Value`) memory into real f32 buffers and calls the
            // *identical* runtime kernel the native backend calls, then marshals the result back, so
            // the differential oracle stays bit-for-bit exact. The serial kernel is used for both
            // serial and parallel names (numerically identical).
            "mercury_sgemm"
            | "mercury_sgemm_parallel"
            | "mercury_sgemm_nt"
            | "mercury_sgemm_nt_parallel" => {
                let a = ptr(args[0])?;
                let b = ptr(args[1])?;
                let c = ptr(args[2])?;
                let m = args[3].as_int() as usize;
                let k = args[4].as_int() as usize;
                let n = args[5].as_int() as usize;
                let beta = args[6].as_int() as i64;
                let read = |mem: &[Value], base: usize, len: usize| -> Result<Vec<f32>, String> {
                    let mut v = Vec::with_capacity(len);
                    for t in 0..len {
                        v.push(
                            mem.get(base + t)
                                .ok_or("sgemm operand out of bounds")?
                                .as_float() as f32,
                        );
                    }
                    Ok(v)
                };
                let abuf = read(&self.memory, a, m * k)?;
                let bbuf = read(&self.memory, b, k * n)?;
                let mut cbuf = read(&self.memory, c, m * n)?;
                let transposed = name.contains("_nt");
                // SAFETY: buffers are exactly m*k, k*n (= n*k), m*n long — the kernel's contract.
                unsafe {
                    if transposed {
                        mercury_runtime::mercury_sgemm_nt(
                            abuf.as_ptr(),
                            bbuf.as_ptr(),
                            cbuf.as_mut_ptr(),
                            m as i64,
                            k as i64,
                            n as i64,
                            beta,
                        );
                    } else {
                        mercury_runtime::mercury_sgemm(
                            abuf.as_ptr(),
                            bbuf.as_ptr(),
                            cbuf.as_mut_ptr(),
                            m as i64,
                            k as i64,
                            n as i64,
                            beta,
                        );
                    }
                }
                for (t, &val) in cbuf.iter().enumerate() {
                    *self
                        .memory
                        .get_mut(c + t)
                        .ok_or("sgemm output out of bounds")? = Value::Float(val as f64);
                }
                Ok(Value::Unit)
            }
            other => Err(format!("call to unknown function or intrinsic `{other}`")),
        }
    }
}

fn reg(regs: &[Value], v: ValueId) -> Value {
    regs[v.0 as usize]
}

/// Pass `args` to `target`'s block parameters, using `scratch` (cleared and reused) so no
/// allocation happens per edge. All argument values are snapshotted before any parameter is
/// written, in case an argument names a value the target also defines.
fn pass_args(regs: &mut [Value], scratch: &mut Vec<Value>, target: &BasicBlock, args: &[ValueId]) {
    if args.is_empty() {
        return;
    }
    scratch.clear();
    for a in args {
        scratch.push(reg(regs, *a));
    }
    for (p, val) in target.params.iter().zip(scratch.iter()) {
        regs[p.0 as usize] = *val;
    }
}

fn ptr(v: Value) -> Result<usize, String> {
    match v {
        Value::Ptr(p) => Ok(p),
        _ => Err("expected a pointer".into()),
    }
}

fn default_value(ty: &MirType) -> Value {
    if ty.is_float() {
        Value::Float(0.0)
    } else if matches!(ty, MirType::Ptr) {
        Value::Ptr(0)
    } else {
        Value::Int(0)
    }
}

fn int_bits(ty: &MirType) -> u32 {
    match ty {
        MirType::I1 => 1,
        MirType::I8 => 8,
        MirType::I16 => 16,
        MirType::I32 => 32,
        MirType::I64 => 64,
        _ => 64,
    }
}

/// Reinterpret the low `bits` of a (sign-extended) integer value as unsigned. The interpreter keeps
/// integers sign-extended in an `i128`; unsigned ops (`UDiv`/`URem`/`UiToFp`/`ZExt`) must read only
/// the value's own `bits`-wide window as unsigned, else a high-bit-set value (e.g. a `u32` ≥ 2^31)
/// diverges from the native backend, which operates on the true machine width.
fn uval(v: i128, bits: u32) -> u128 {
    if bits >= 128 {
        v as u128
    } else {
        (v as u128) & ((1u128 << bits) - 1)
    }
}

/// Truncate an integer value to a result type's bit width (signed wrap).
fn mask(v: i128, ty: &MirType) -> i128 {
    // `i1` is a boolean: keep the low bit unsigned (true == 1, not a sign-extended -1).
    if matches!(ty, MirType::I1) {
        return v & 1;
    }
    let bits = int_bits(ty);
    if bits >= 128 {
        return v;
    }
    let shift = 128 - bits;
    (v << shift) >> shift // sign-extend from `bits`
}

/// `f16`/`bf16`/`f32` results are rounded to `f32` precision (the interpreter promotes sub-`f32`
/// storage types to `f32`); `f64` keeps full precision.
fn is_narrow_float(ty: &MirType) -> bool {
    matches!(ty, MirType::F16 | MirType::BF16 | MirType::F32)
}

/// `a * b + c` with a *single* rounding, matching the native `fma` and the front-end's float
/// contraction of `x + y*z`. Narrow (`f32`) operands round in `f32`, like `apply_bin`.
fn apply_fma(a: Value, b: Value, c: Value, rty: Option<&MirType>) -> Value {
    if rty.is_some_and(is_narrow_float) {
        let r = (a.as_float() as f32).mul_add(b.as_float() as f32, c.as_float() as f32);
        Value::Float(r as f64)
    } else {
        Value::Float(a.as_float().mul_add(b.as_float(), c.as_float()))
    }
}

fn apply_bin(op: BinOp, a: Value, b: Value, rty: Option<&MirType>) -> Value {
    use BinOp::*;
    if op.is_float() {
        // Compute `f32`-typed operations in actual `f32` (single rounding), matching the native
        // backend exactly; a single op on two exact `f32` values would otherwise double-round.
        if rty.is_some_and(is_narrow_float) {
            let (x, y) = (a.as_float() as f32, b.as_float() as f32);
            let r = match op {
                FAdd => x + y,
                FSub => x - y,
                FMul => x * y,
                FDiv => x / y,
                FRem => x % y,
                _ => unreachable!(),
            };
            return Value::Float(r as f64);
        }
        let (x, y) = (a.as_float(), b.as_float());
        return Value::Float(match op {
            FAdd => x + y,
            FSub => x - y,
            FMul => x * y,
            FDiv => x / y,
            FRem => x % y,
            _ => unreachable!(),
        });
    }
    let (x, y) = (a.as_int(), b.as_int());
    Value::Int(match op {
        Add => x.wrapping_add(y),
        Sub => x.wrapping_sub(y),
        Mul => x.wrapping_mul(y),
        SDiv => {
            if y == 0 {
                0
            } else {
                x.wrapping_div(y)
            }
        }
        UDiv => {
            let w = rty.map(int_bits).unwrap_or(64);
            let (xu, yu) = (uval(x, w), uval(y, w));
            if yu == 0 {
                0
            } else {
                (xu / yu) as i128
            }
        }
        SRem => {
            if y == 0 {
                0
            } else {
                x.wrapping_rem(y)
            }
        }
        URem => {
            let w = rty.map(int_bits).unwrap_or(64);
            let (xu, yu) = (uval(x, w), uval(y, w));
            if yu == 0 {
                0
            } else {
                (xu % yu) as i128
            }
        }
        And => x & y,
        Or => x | y,
        Xor => x ^ y,
        Shl => x.wrapping_shl(y as u32),
        LShr => ((x as u128) >> (y as u32)) as i128,
        AShr => x >> (y as u32),
        FAdd | FSub | FMul | FDiv | FRem => unreachable!(),
    })
}

fn apply_cmp(op: CmpOp, a: Value, b: Value) -> bool {
    use CmpOp::*;
    if op.is_float() {
        let (x, y) = (a.as_float(), b.as_float());
        return match op {
            Foeq => x == y,
            Fone => x != y,
            Folt => x < y,
            Fole => x <= y,
            Fogt => x > y,
            Foge => x >= y,
            _ => unreachable!(),
        };
    }
    let (x, y) = (a.as_int(), b.as_int());
    let (ux, uy) = (x as u128, y as u128);
    match op {
        Eq => x == y,
        Ne => x != y,
        Slt => x < y,
        Sle => x <= y,
        Sgt => x > y,
        Sge => x >= y,
        Ult => ux < uy,
        Ule => ux <= uy,
        Ugt => ux > uy,
        Uge => ux >= uy,
        _ => unreachable!(),
    }
}

fn apply_cast(kind: CastKind, v: Value, from: &MirType, to: &MirType) -> Value {
    use CastKind::*;
    match kind {
        SExt | Trunc => Value::Int(mask(v.as_int(), to)),
        // Zero-extend the source's own `from`-width bits. Because ints are stored sign-extended, a
        // high-bit-set unsigned source (e.g. `u32` ≥ 2^31) would otherwise widen as negative.
        ZExt => Value::Int(mask(uval(v.as_int(), int_bits(from)) as i128, to)),
        SiToFp => Value::Float(v.as_int() as f64),
        // Unsigned→float: read the source as unsigned in its own width first (matches native
        // `fcvt_from_uint`); `as_int() as f64` would be negative for a high-bit-set value.
        UiToFp => Value::Float(uval(v.as_int(), int_bits(from)) as f64),
        // Saturating fp→int, matching the native backend's `fcvt_to_{sint,uint}_sat` (NaN→0, clamp to
        // the target range, negatives→0 for unsigned). Rust's `as` has exactly these semantics; the
        // old bit-mask of an `i128` cast diverged from native for out-of-range / negative-to-unsigned
        // values (a latent differential-oracle break — `u`-typed targets make `FpToUi` reachable).
        FpToSi => {
            let f = v.as_float();
            let i = match to {
                MirType::I8 => f as i8 as i128,
                MirType::I16 => f as i16 as i128,
                MirType::I64 => f as i64 as i128,
                _ => f as i32 as i128, // I32 (and the I1 fallback)
            };
            Value::Int(mask(i, to))
        }
        FpToUi => {
            let f = v.as_float();
            let i = match to {
                MirType::I8 => f as u8 as i128,
                MirType::I16 => f as u16 as i128,
                MirType::I64 => f as u64 as i128,
                _ => f as u32 as i128, // I32 (and the I1 fallback)
            };
            Value::Int(mask(i, to))
        }
        FpExt | FpTrunc => Value::Float(v.as_float()),
        Bitcast => v,
        IntToPtr => Value::Ptr(v.as_int() as usize),
        PtrToInt => Value::Int(v.as_int()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mercury_span::SourceId;

    fn run_main(src: &str) -> i64 {
        let mut interner = Interner::new();
        let (module, pd) = mercury_parser::parse_module(src, SourceId(0), &mut interner);
        assert!(pd.is_empty(), "parse: {pd:?}");
        let (sema, sd) = mercury_sema::check(&module, &interner);
        assert!(sd.iter().all(|d| !d.is_error()), "sema: {sd:?}");
        let (program, _ld) = mercury_mir_build::lower_program(&module, &sema, &mut interner);
        let main = interner.intern("main");
        run(&program, main, &interner).unwrap()
    }

    #[test]
    fn runs_loop_sum() {
        let src = "fn main() -> i32 { let mut s: i32 = 0; let mut i: i32 = 0; \
                   while i < 10 { s += i; i += 1; } return s; }";
        assert_eq!(run_main(src), 45);
    }

    #[test]
    fn runs_recursive_fib() {
        let src = "fn fib(n: i32) -> i32 { if n < 2 { return n; } \
                   return fib(n - 1) + fib(n - 2); } \
                   fn main() -> i32 { return fib(10); }";
        assert_eq!(run_main(src), 55);
    }

    #[test]
    fn print_intrinsic_captures_stdout() {
        let mut interner = Interner::new();
        let src = "fn main() -> i32 { print(42); print(7 * 6); return 0; }";
        let (module, _) = mercury_parser::parse_module(src, SourceId(0), &mut interner);
        let (sema, _) = mercury_sema::check(&module, &interner);
        let (program, _) = mercury_mir_build::lower_program(&module, &sema, &mut interner);
        let main = interner.intern("main");
        let (code, out) = run_with_output(&program, main, &interner).unwrap();
        assert_eq!(code, 0);
        assert_eq!(String::from_utf8(out).unwrap(), "42\n42\n");
    }

    #[test]
    fn runs_for_range_with_step() {
        let src = "fn main() -> i32 { let mut s: i32 = 0; \
                   for i in 0..10 step 2 { s += i; } return s; }";
        // 0 + 2 + 4 + 6 + 8 = 20
        assert_eq!(run_main(src), 20);
    }

    #[test]
    fn runs_array_index_and_reduction() {
        let src = "fn main() -> i32 { let mut xs: [i32; 4] = [0,0,0,0]; \
                   let mut i: i32 = 0; while i < 4 { xs[i] = i * i; i = i + 1; } \
                   let mut s: i32 = 0; let mut j: i32 = 0; \
                   while j < 4 { s = s + xs[j]; j = j + 1; } return s; }";
        // 0 + 1 + 4 + 9 = 14
        assert_eq!(run_main(src), 14);
    }

    #[test]
    fn runs_large_array_repeat_fill() {
        // A large `[v; n]` initializer lowers to a fill loop rather than n unrolled stores; every
        // slot must still hold the value.
        let src = "fn main() -> i32 { let a: [i32; 100] = [7; 100]; \
                   return a[0] + a[50] + a[99]; }";
        assert_eq!(run_main(src), 21);
    }

    #[test]
    fn runs_array_parameter_by_reference() {
        let src = "fn fill(a: [i32; 3]) { a[0] = 7; a[1] = 8; a[2] = 9; } \
                   fn main() -> i32 { let mut a: [i32;3] = [0,0,0]; fill(a); \
                   return a[0] + a[1] + a[2]; }";
        assert_eq!(run_main(src), 24);
    }

    #[test]
    fn runs_casts_and_signed_modulo() {
        // -7 % 3 == -1, i64->i32 cast, f32->i32 truncation.
        assert_eq!(run_main("fn main() -> i32 { return -7 % 3; }"), -1);
        assert_eq!(
            run_main("fn main() -> i32 { let a: i64 = 300; return a as i32; }"),
            300
        );
        assert_eq!(
            run_main("fn main() -> i32 { let f: f32 = 9.9; return f as i32; }"),
            9
        );
    }

    #[test]
    fn failed_assert_is_an_error() {
        let mut interner = Interner::new();
        let src = "fn main() -> i32 { assert(1 > 2); return 0; }";
        let (module, _) = mercury_parser::parse_module(src, SourceId(0), &mut interner);
        let (sema, _) = mercury_sema::check(&module, &interner);
        let (program, _) = mercury_mir_build::lower_program(&module, &sema, &mut interner);
        let main = interner.intern("main");
        assert!(run(&program, main, &interner).is_err());
    }
}
