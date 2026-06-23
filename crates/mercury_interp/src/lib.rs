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

/// An accelerator that can run recognized runtime kernels in place of the CPU `mercury_runtime`
/// microkernels — the seam the **GPU backend** plugs into. The interpreter stays zero-dependency by
/// knowing only this trait; the driver supplies a GPU implementation behind its `gpu` feature, and
/// the offload runs over the *same* tree-walked MIR (so control flow, buffer layout, and every
/// non-kernel op are identical — only the recognized kernel calls move to the device).
///
/// Each method mirrors one `mercury_runtime` kernel over already-marshalled f32 buffers. Returning
/// `None` means "not handled here — fall back to the CPU kernel", so partial / precision-restricted
/// support is fine. With no accelerator installed (the default, and every existing caller), the
/// differential oracle path is **byte-for-byte unchanged**; the CPU↔GPU gate is therefore a
/// *tolerance* differential (interp-CPU vs interp-with-this-accelerator over identical inputs).
pub trait Accelerator {
    /// `C = A·Bᵀ` (the `_nt` form) with `beta == 0` (overwrite). Return `None` for the `beta != 0`
    /// or non-transposed forms the GPU GEMM wrapper does not cover.
    fn sgemm_nt(
        &mut self,
        _a: &[f32],
        _b: &[f32],
        _c: &mut [f32],
        _m: usize,
        _k: usize,
        _n: usize,
    ) -> Option<Result<(), String>> {
        None
    }
    /// `out[i] = f_op(x[i])` — the 256-bit activation kernel (`op` is the shared runtime op code).
    fn vmath(&mut self, _op: i64, _x: &[f32], _out: &mut [f32]) -> Option<Result<(), String>> {
        None
    }
    /// Fused row norm (softmax / LayerNorm / RMSNorm): `out` = norm over each `cols`-wide row of `x`.
    fn norm(
        &mut self,
        _op: i64,
        _x: &[f32],
        _out: &mut [f32],
        _rows: usize,
        _cols: usize,
        _eps: f32,
    ) -> Option<Result<(), String>> {
        None
    }
    /// Deterministic reduction returning a scalar (sum / dot / max / …); `y` is the second operand
    /// (used by dot/ssd, ignored otherwise).
    fn sreduce(&mut self, _op: i64, _x: &[f32], _y: &[f32]) -> Option<Result<f32, String>> {
        None
    }
    /// **Fused-epilogue Linear** `C = act(A·Bᵀ + bias)` with `beta == 0` (the `mercury_sgemm_nt_epi`
    /// shape a `act(matmul(x,w)[+bias])` Mercury expression lowers to). `bias` is `None` for the
    /// bias-free SwiGLU form; `act` is the runtime `ACT_*` code (1=relu, 2=gelu, 3=silu). Return `None`
    /// for any case the device kernel doesn't cover (non-zero beta, a bias it can't fuse, an unsupported
    /// activation, or an unaligned shape) so it falls back to the CPU fused kernel.
    fn sgemm_nt_epi(
        &mut self,
        _a: &[f32],
        _b: &[f32],
        _c: &mut [f32],
        _m: usize,
        _k: usize,
        _n: usize,
        _beta: i64,
        _bias: Option<&[f32]>,
        _act: i64,
    ) -> Option<Result<(), String>> {
        None
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
        accel: None,
    };
    let result = interp.run_function(func, Vec::new())?;
    Ok((result.as_int() as i64, interp.stdout))
}

/// Like [`run_with_output`] but offloads recognized kernel calls to `accel` (the GPU backend). This
/// is the `--backend=gpu` execution path: the whole program is still tree-walked here, but every
/// recognized GEMM / norm / activation / reduction call runs on the device instead of the CPU
/// microkernel. Falls back to the CPU kernel for any call `accel` declines (returns `None`).
pub fn run_with_output_accel(
    program: &Program,
    entry: Symbol,
    interner: &Interner,
    accel: &mut dyn Accelerator,
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
        accel: Some(accel),
    };
    let result = interp.run_function(func, Vec::new())?;
    Ok((result.as_int() as i64, interp.stdout))
}

/// Run an f32-buffer kernel (`entry`) over caller-provided buffers — the interpreter's typed
/// kernel-entry ABI, the counterpart to the native backend's `jit_module().func_ptr()` + raw call.
///
/// `bufs` are the kernel's buffer parameters in declaration order: each slice is copied into the
/// interpreter's flat memory, the kernel is invoked with a pointer to each, and the final memory
/// contents are copied back into the slices (so an output or in-place buffer reflects the result).
/// Every parameter of `entry` must be a pointer — i.e. an `[f32; N]` array parameter, which lowers
/// to [`MirType::Ptr`]; pass exactly one slice per parameter.
///
/// This is what lets a differential fuzzer run the *same* kernel on the interpreter and the native
/// backend over identical random buffers and compare the **full output buffer**, closing the gap
/// left by the stdout/exit-code-only oracle (which could only see whatever a `main` chose to print).
pub fn run_kernel_f32(
    program: &Program,
    entry: Symbol,
    bufs: &mut [&mut [f32]],
    interner: &Interner,
) -> Result<(), String> {
    run_kernel_f32_inner(program, entry, bufs, interner, None)
}

/// Like [`run_kernel_f32`] but offloads recognized kernel calls to `accel` — the full-buffer
/// **tolerance** gate for the GPU backend: run the same kernel on the CPU oracle (no accelerator)
/// and on the GPU (this) over identical random buffers, then compare within `c·√K·ε`.
pub fn run_kernel_f32_accel(
    program: &Program,
    entry: Symbol,
    bufs: &mut [&mut [f32]],
    interner: &Interner,
    accel: &mut dyn Accelerator,
) -> Result<(), String> {
    run_kernel_f32_inner(program, entry, bufs, interner, Some(accel))
}

fn run_kernel_f32_inner(
    program: &Program,
    entry: Symbol,
    bufs: &mut [&mut [f32]],
    interner: &Interner,
    accel: Option<&mut dyn Accelerator>,
) -> Result<(), String> {
    let func = program
        .function(entry)
        .ok_or_else(|| format!("no entry function `{}`", interner.resolve(entry)))?;
    if func.params.len() != bufs.len() {
        return Err(format!(
            "kernel `{}` takes {} parameter(s) but {} buffer(s) were provided",
            interner.resolve(entry),
            func.params.len(),
            bufs.len()
        ));
    }
    let mut interp = Interp {
        program,
        interner,
        memory: Vec::new(),
        stdout: Vec::new(),
        frames: Vec::new(),
        scratch: Vec::with_capacity(8),
        vecs: Vec::new(),
        accel,
    };
    // Lay each buffer out contiguously in flat memory and remember its base slot. Any `alloca`
    // the kernel performs internally grows memory *past* these regions, so it never clobbers them.
    let mut bases = Vec::with_capacity(bufs.len());
    for buf in bufs.iter() {
        bases.push(interp.memory.len());
        interp
            .memory
            .extend(buf.iter().map(|&v| Value::Float(v as f64)));
    }
    let args: Vec<Value> = bases.iter().map(|&b| Value::Ptr(b)).collect();
    interp.run_function(func, args)?;
    // Copy the final contents back out (captures both outputs and in-place mutation).
    for (buf, &base) in bufs.iter_mut().zip(bases.iter()) {
        for (i, slot) in buf.iter_mut().enumerate() {
            *slot = match interp.memory[base + i] {
                Value::Float(f) => f as f32,
                Value::Int(n) => n as f32,
                _ => 0.0,
            };
        }
    }
    Ok(())
}

/// Run an f64-buffer kernel over caller-provided buffers — the double-precision twin of
/// [`run_kernel_f32`]. The autodiff finite-difference gradient gate evaluates its reference forward
/// pass through this: an f32 central difference's eps-noise (`~f32_eps/eps`) would swamp the
/// gradient, whereas f64 keeps it orders of magnitude below the tolerance. Identical to
/// `run_kernel_f32` except buffers marshal as `Value::Float(v)` directly (no `as f32` rounding), so
/// a kernel built with `F64` ops computes in full double precision.
pub fn run_kernel_f64(
    program: &Program,
    entry: Symbol,
    bufs: &mut [&mut [f64]],
    interner: &Interner,
) -> Result<(), String> {
    let func = program
        .function(entry)
        .ok_or_else(|| format!("no entry function `{}`", interner.resolve(entry)))?;
    if func.params.len() != bufs.len() {
        return Err(format!(
            "kernel `{}` takes {} parameter(s) but {} buffer(s) were provided",
            interner.resolve(entry),
            func.params.len(),
            bufs.len()
        ));
    }
    let mut interp = Interp {
        program,
        interner,
        memory: Vec::new(),
        stdout: Vec::new(),
        frames: Vec::new(),
        scratch: Vec::with_capacity(8),
        vecs: Vec::new(),
        accel: None,
    };
    let mut bases = Vec::with_capacity(bufs.len());
    for buf in bufs.iter() {
        bases.push(interp.memory.len());
        interp.memory.extend(buf.iter().map(|&v| Value::Float(v)));
    }
    let args: Vec<Value> = bases.iter().map(|&b| Value::Ptr(b)).collect();
    interp.run_function(func, args)?;
    for (buf, &base) in bufs.iter_mut().zip(bases.iter()) {
        for (i, slot) in buf.iter_mut().enumerate() {
            *slot = match interp.memory[base + i] {
                Value::Float(f) => f,
                Value::Int(n) => n as f64,
                _ => 0.0,
            };
        }
    }
    Ok(())
}

/// Run an int8 quantized GEMM kernel `fn k(a: [u8; _], b: [i8; _], c: [i32; _])` over caller
/// buffers — the integer twin of [`run_kernel_f32`], for the full-buffer int8 differential fuzzer.
///
/// `a`/`b`/`c` are laid out contiguously in flat `Value::Int` memory (`a` zero-extended from `u8`,
/// `b` sign-extended from `i8` — matching the kernel's widening), the kernel runs with a pointer to
/// each, and the `i32` results are copied back into `c`. The kernel must take exactly three
/// (pointer) parameters in `a, b, c` order.
pub fn run_kernel_i8(
    program: &Program,
    entry: Symbol,
    a: &[u8],
    b: &[i8],
    c: &mut [i32],
    interner: &Interner,
) -> Result<(), String> {
    let func = program
        .function(entry)
        .ok_or_else(|| format!("no entry function `{}`", interner.resolve(entry)))?;
    if func.params.len() != 3 {
        return Err(format!(
            "int8 kernel `{}` takes {} parameter(s), expected 3 (a, b, c)",
            interner.resolve(entry),
            func.params.len()
        ));
    }
    let mut interp = Interp {
        program,
        interner,
        memory: Vec::new(),
        stdout: Vec::new(),
        frames: Vec::new(),
        scratch: Vec::with_capacity(8),
        vecs: Vec::new(),
        accel: None,
    };
    // u8 zero-extends, i8 sign-extends — the `as i128` casts do exactly that, matching the kernel.
    let a_base = interp.memory.len();
    interp
        .memory
        .extend(a.iter().map(|&v| Value::Int(v as i128)));
    let b_base = interp.memory.len();
    interp
        .memory
        .extend(b.iter().map(|&v| Value::Int(v as i128)));
    let c_base = interp.memory.len();
    interp
        .memory
        .extend(c.iter().map(|&v| Value::Int(v as i128)));
    let args = vec![Value::Ptr(a_base), Value::Ptr(b_base), Value::Ptr(c_base)];
    interp.run_function(func, args)?;
    for (i, slot) in c.iter_mut().enumerate() {
        *slot = match interp.memory[c_base + i] {
            Value::Int(n) => n as i32,
            Value::Float(f) => f as i32,
            _ => 0,
        };
    }
    Ok(())
}

struct Interp<'a, 'k> {
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
    /// Optional accelerator the recognized kernel calls offload to (the GPU backend). `None` for the
    /// reference oracle and every plain run, so their numerics are untouched.
    accel: Option<&'k mut (dyn Accelerator + 'k)>,
}

impl<'a, 'k> Interp<'a, 'k> {
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
            Op::Cast(kind, v, to) => {
                if let Some(MirType::Vec(to_lane, n)) = rty {
                    // Lane-wise cast (e.g. the vectorized exp's f32->i32 and i32->f32 bitcast).
                    let xs = self.vec_lanes(reg(regs, *v));
                    let from_lane = match func.value_type(*v) {
                        MirType::Vec(l, _) => (**l).clone(),
                        other => other.clone(),
                    };
                    let to_lane = (**to_lane).clone();
                    let lanes: Vec<Value> = (0..*n as usize)
                        .map(|i| apply_cast(*kind, xs[i], &from_lane, &to_lane))
                        .collect();
                    self.push_vec(lanes)
                } else {
                    apply_cast(*kind, reg(regs, *v), func.value_type(*v), to)
                }
            }
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
                    let v = *self.memory.get(idx).ok_or("load out of bounds")?;
                    // `[bf16; N]`/`[f16; N]` storage holds reduced precision: round on read (the native
                    // backend stores the rounded 16 bits, so a load there observes the same value).
                    match (ty, v) {
                        (MirType::BF16, Value::Float(f)) => {
                            Value::Float(mercury_runtime::round_bf16(f as f32) as f64)
                        }
                        (MirType::F16, Value::Float(f)) => {
                            Value::Float(mercury_runtime::round_f16(f as f32) as f64)
                        }
                        _ => v,
                    }
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
            // Square root, lane-wise for vectors; each lane rounded to its lane type so an `f32`
            // sqrt stays bit-identical to the native `fsqrt`.
            Op::Sqrt(v) => {
                if let Some(MirType::Vec(lane, n)) = rty {
                    let xv = self.vec_lanes(reg(regs, *v));
                    let lane = (**lane).clone();
                    let lanes: Vec<Value> = (0..*n as usize)
                        .map(|i| apply_sqrt(xv[i], Some(&lane)))
                        .collect();
                    self.push_vec(lanes)
                } else {
                    apply_sqrt(reg(regs, *v), rty)
                }
            }
            Op::Round(mode, v) => {
                if let Some(MirType::Vec(lane, n)) = rty {
                    let xv = self.vec_lanes(reg(regs, *v));
                    let lane = (**lane).clone();
                    let lanes: Vec<Value> = (0..*n as usize)
                        .map(|i| apply_round(*mode, xv[i], Some(&lane)))
                        .collect();
                    self.push_vec(lanes)
                } else {
                    apply_round(*mode, reg(regs, *v), rty)
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
                // Offload to the accelerator (GPU) for the `C = A·Bᵀ` overwrite form it covers; any
                // other shape (beta != 0, non-transposed) or a declined call falls back to the CPU
                // kernel. With no accelerator (the oracle) this is always the CPU path.
                let offloaded = if transposed && beta == 0 {
                    self.accel
                        .as_mut()
                        .and_then(|acc| acc.sgemm_nt(&abuf, &bbuf, &mut cbuf, m, k, n))
                } else {
                    None
                };
                match offloaded {
                    Some(Ok(())) => {}
                    Some(Err(e)) => return Err(e),
                    // SAFETY: buffers are exactly m*k, k*n (= n*k), m*n long — the kernel's contract.
                    None => unsafe {
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
                    },
                }
                for (t, &val) in cbuf.iter().enumerate() {
                    *self
                        .memory
                        .get_mut(c + t)
                        .ok_or("sgemm output out of bounds")? = Value::Float(val as f64);
                }
                Ok(Value::Unit)
            }
            // `mercury_i8gemm_nt[_parallel](a, b, c, m, k, n)` — the int8 quantized `nn.Linear` kernel
            // (`u8` activations × `i8` weights → `i32`, `C = A·Bᵀ`). The interpreter recovers each
            // operand byte (`as u8`/`as i8` takes the low 8 bits — bit-identical to the native buffer
            // regardless of how the abstract value was sign-extended) and writes the `i32` result,
            // calling the *serial* runtime kernel (bit-identical to the parallel one — rows are
            // independent), so the differential oracle stays exact. Integer math, so no rounding at all.
            "mercury_i8gemm_nt" | "mercury_i8gemm_nt_parallel" => {
                let a = ptr(args[0])?;
                let b = ptr(args[1])?;
                let c = ptr(args[2])?;
                let m = args[3].as_int() as usize;
                let k = args[4].as_int() as usize;
                let n = args[5].as_int() as usize;
                let mut abuf: Vec<u8> = Vec::with_capacity(m * k);
                for t in 0..m * k {
                    abuf.push(
                        self.memory
                            .get(a + t)
                            .ok_or("i8gemm a out of bounds")?
                            .as_int() as u8,
                    );
                }
                let mut bbuf: Vec<i8> = Vec::with_capacity(n * k);
                for t in 0..n * k {
                    bbuf.push(
                        self.memory
                            .get(b + t)
                            .ok_or("i8gemm b out of bounds")?
                            .as_int() as i8,
                    );
                }
                let mut cbuf = vec![0i32; m * n];
                // SAFETY: buffers are exactly m*k, n*k, m*n long — the kernel's contract.
                unsafe {
                    mercury_runtime::mercury_i8gemm_nt(
                        abuf.as_ptr(),
                        bbuf.as_ptr(),
                        cbuf.as_mut_ptr(),
                        m as i64,
                        k as i64,
                        n as i64,
                    );
                }
                for (t, &val) in cbuf.iter().enumerate() {
                    *self
                        .memory
                        .get_mut(c + t)
                        .ok_or("i8gemm output out of bounds")? = Value::Int(val as i128);
                }
                Ok(Value::Unit)
            }
            // `mercury_sgemm_nt_epi[_parallel](a, b, c, m, k, n, beta, bias, act)` — the fused-epilogue
            // Linear (`C = act(A·Bᵀ + bias)`). Like the plain GEMM, the interpreter marshals operands
            // into real f32 buffers and calls the *serial* runtime kernel as the oracle. The native
            // `@parallel` variant is bit-identical to it (each C tile owned by one task, same per-(i,j)
            // accumulation order), so calling the serial form here keeps the two backends bit-for-bit
            // exact without spawning threads in the interpreter.
            "mercury_sgemm_nt_epi" | "mercury_sgemm_nt_epi_parallel" => {
                let a = ptr(args[0])?;
                let b = ptr(args[1])?;
                let c = ptr(args[2])?;
                let m = args[3].as_int() as usize;
                let k = args[4].as_int() as usize;
                let n = args[5].as_int() as usize;
                let beta = args[6].as_int() as i64;
                // An absent bias arrives as a `Value::Int(0)` (the null built as an integer 0) vs a
                // real array's `Value::Ptr` — distinguished by variant, like the affine-norm params.
                let bias_idx = match args[7] {
                    Value::Ptr(p) => Some(p),
                    _ => None,
                };
                let act = args[8].as_int() as i64;
                let read = |mem: &[Value], base: usize, len: usize| -> Result<Vec<f32>, String> {
                    let mut v = Vec::with_capacity(len);
                    for t in 0..len {
                        v.push(
                            mem.get(base + t)
                                .ok_or("sgemm_epi operand out of bounds")?
                                .as_float() as f32,
                        );
                    }
                    Ok(v)
                };
                let abuf = read(&self.memory, a, m * k)?;
                let bbuf = read(&self.memory, b, n * k)?;
                let mut cbuf = read(&self.memory, c, m * n)?;
                let biasbuf = match bias_idx {
                    Some(base) => read(&self.memory, base, n)?,
                    None => Vec::new(),
                };
                let bias_ptr = if bias_idx.is_some() {
                    biasbuf.as_ptr()
                } else {
                    std::ptr::null()
                };
                // Offload the fused epilogue to the accelerator (GPU) when it covers this case; a decline
                // (`None`) or no accelerator falls back to the identical CPU kernel — same contract as the
                // plain GEMM above. With no accelerator (the oracle) this is always the CPU path.
                let bias_opt = bias_idx.is_some().then_some(biasbuf.as_slice());
                let offloaded = self
                    .accel
                    .as_mut()
                    .and_then(|acc| acc.sgemm_nt_epi(&abuf, &bbuf, &mut cbuf, m, k, n, beta, bias_opt, act));
                match offloaded {
                    Some(Ok(())) => {}
                    Some(Err(e)) => return Err(e),
                    // SAFETY: buffers are exactly m*k, n*k, m*n long; bias is null or n long — kernel contract.
                    None => unsafe {
                        mercury_runtime::mercury_sgemm_nt_epi(
                            abuf.as_ptr(),
                            bbuf.as_ptr(),
                            cbuf.as_mut_ptr(),
                            m as i64,
                            k as i64,
                            n as i64,
                            beta,
                            bias_ptr,
                            act,
                        );
                    },
                }
                for (t, &val) in cbuf.iter().enumerate() {
                    *self
                        .memory
                        .get_mut(c + t)
                        .ok_or("sgemm_epi output out of bounds")? = Value::Float(val as f64);
                }
                Ok(Value::Unit)
            }
            // `mercury_vmath_f32(x, out, n, op)` — the 256-bit AVX2 elementwise transcendental kernel
            // (exp/log/tanh/sigmoid) the compiler lowers an `out[i] = f(x[i])` loop to. Marshal `n`
            // f32 from x, call the *identical* runtime kernel the native backend calls, write the
            // result back — so the differential oracle stays exact despite the kernel's wider lanes.
            // Reading all of x before writing out makes the in-place (x == out) case correct.
            "mercury_vmath_f32" => {
                let x = ptr(args[0])?;
                let out = ptr(args[1])?;
                let n = args[2].as_int() as usize;
                let op = args[3].as_int() as i64;
                let mut xbuf = Vec::with_capacity(n);
                for t in 0..n {
                    xbuf.push(
                        self.memory
                            .get(x + t)
                            .ok_or("vmath operand out of bounds")?
                            .as_float() as f32,
                    );
                }
                let mut obuf = vec![0.0f32; n];
                // Offload the activation to the accelerator (GPU) when it claims this op; otherwise
                // (or with no accelerator — the oracle) fall back to the identical CPU kernel.
                let offloaded = self
                    .accel
                    .as_mut()
                    .and_then(|acc| acc.vmath(op, &xbuf, &mut obuf));
                match offloaded {
                    Some(Ok(())) => {}
                    Some(Err(e)) => return Err(e),
                    // SAFETY: xbuf/obuf are exactly n f32 long — the kernel's contract.
                    None => unsafe {
                        mercury_runtime::mercury_vmath_f32(
                            xbuf.as_ptr(),
                            obuf.as_mut_ptr(),
                            n as i64,
                            op,
                        );
                    },
                }
                for (t, &val) in obuf.iter().enumerate() {
                    *self
                        .memory
                        .get_mut(out + t)
                        .ok_or("vmath output out of bounds")? = Value::Float(val as f64);
                }
                Ok(Value::Unit)
            }
            // `mercury_vmath2_f32(x, y, out, n, op)` — the two-input 256-bit kernel (pow/atan2/hypot)
            // an `out[i] = f(x[i], y[i])` loop lowers to. Marshal `n` f32 from x AND y, call the
            // *identical* runtime kernel the native backend calls, write the result back — so the
            // differential oracle stays exact. Read both inputs before writing (in-place safe).
            "mercury_vmath2_f32" => {
                let x = ptr(args[0])?;
                let y = ptr(args[1])?;
                let out = ptr(args[2])?;
                let n = args[3].as_int() as usize;
                let op = args[4].as_int() as i64;
                let mut xbuf = Vec::with_capacity(n);
                let mut ybuf = Vec::with_capacity(n);
                for t in 0..n {
                    xbuf.push(
                        self.memory
                            .get(x + t)
                            .ok_or("vmath2 operand out of bounds")?
                            .as_float() as f32,
                    );
                    ybuf.push(
                        self.memory
                            .get(y + t)
                            .ok_or("vmath2 operand out of bounds")?
                            .as_float() as f32,
                    );
                }
                let mut obuf = vec![0.0f32; n];
                // SAFETY: xbuf/ybuf/obuf are exactly n f32 long — the kernel's contract.
                unsafe {
                    mercury_runtime::mercury_vmath2_f32(
                        xbuf.as_ptr(),
                        ybuf.as_ptr(),
                        obuf.as_mut_ptr(),
                        n as i64,
                        op,
                    );
                }
                for (t, &val) in obuf.iter().enumerate() {
                    *self
                        .memory
                        .get_mut(out + t)
                        .ok_or("vmath2 output out of bounds")? = Value::Float(val as f64);
                }
                Ok(Value::Unit)
            }
            // `mercury_vmath_bf16(x, out, n, op)` — the bf16-input twin of `mercury_vmath_f32` an
            // `out[i] = f((x[i] as f32))` loop over a `[bf16]` array lowers to. Reconstruct the exact
            // bf16 input bits (as the bf16 reductions/axpby do — the stored value is already bf16-
            // rounded, so `f32_to_bf16_bits` is exact), call the *identical* runtime kernel the native
            // backend calls, write the f32 result back — so the differential oracle stays exact.
            "mercury_vmath_bf16" => {
                let x = ptr(args[0])?;
                let out = ptr(args[1])?;
                let n = args[2].as_int() as usize;
                let op = args[3].as_int() as i64;
                let mut xbuf = Vec::with_capacity(n);
                for t in 0..n {
                    xbuf.push(mercury_runtime::f32_to_bf16_bits(
                        self.memory
                            .get(x + t)
                            .ok_or("vmath bf16 operand out of bounds")?
                            .as_float() as f32,
                    ));
                }
                let mut obuf = vec![0.0f32; n];
                // SAFETY: xbuf is n u16, obuf is n f32 — the kernel's contract.
                unsafe {
                    mercury_runtime::mercury_vmath_bf16(
                        xbuf.as_ptr(),
                        obuf.as_mut_ptr(),
                        n as i64,
                        op,
                    );
                }
                for (t, &val) in obuf.iter().enumerate() {
                    *self
                        .memory
                        .get_mut(out + t)
                        .ok_or("vmath bf16 output out of bounds")? = Value::Float(val as f64);
                }
                Ok(Value::Unit)
            }
            // `mercury_vmath_f16` — the F16C twin of `mercury_vmath_bf16`: reconstruct exact f16 bits
            // via `f32_to_f16_bits` and call the identical kernel, so interp == native.
            "mercury_vmath_f16" => {
                let x = ptr(args[0])?;
                let out = ptr(args[1])?;
                let n = args[2].as_int() as usize;
                let op = args[3].as_int() as i64;
                let mut xbuf = Vec::with_capacity(n);
                for t in 0..n {
                    xbuf.push(mercury_runtime::f32_to_f16_bits(
                        self.memory
                            .get(x + t)
                            .ok_or("vmath f16 operand out of bounds")?
                            .as_float() as f32,
                    ));
                }
                let mut obuf = vec![0.0f32; n];
                // SAFETY: xbuf is n u16, obuf is n f32 — the kernel's contract.
                unsafe {
                    mercury_runtime::mercury_vmath_f16(
                        xbuf.as_ptr(),
                        obuf.as_mut_ptr(),
                        n as i64,
                        op,
                    );
                }
                for (t, &val) in obuf.iter().enumerate() {
                    *self
                        .memory
                        .get_mut(out + t)
                        .ok_or("vmath f16 output out of bounds")? = Value::Float(val as f64);
                }
                Ok(Value::Unit)
            }
            // `mercury_velem_f32(x, y, out, n, a, b, c, op)` — the streaming affine+activation kernel
            // (saxpy / scale / residual-add / bias / ReLU) a recognized `out[i] = act(a·x[i] + b·y[i]
            // + c)` map lowers to. Marshal `n` f32 from x and y, call the *identical* runtime kernel
            // the native backend calls, write the result back — so the differential oracle stays exact
            // despite the wider lanes / non-temporal stores (which write the same bits). Reading all of
            // x/y before writing out makes the in-place (x == out) case correct. When `y` is unused the
            // recognizer passes the x pointer for it (never dereferenced by the kernel), so marshalling
            // y unconditionally is harmless.
            "mercury_velem_f32" => {
                let x = ptr(args[0])?;
                let y = ptr(args[1])?;
                let out = ptr(args[2])?;
                let n = args[3].as_int() as usize;
                let a = args[4].as_float() as f32;
                let b = args[5].as_float() as f32;
                let c = args[6].as_float() as f32;
                let op = args[7].as_int() as i64;
                let mut xbuf = Vec::with_capacity(n);
                let mut ybuf = Vec::with_capacity(n);
                for t in 0..n {
                    xbuf.push(
                        self.memory
                            .get(x + t)
                            .ok_or("velem operand out of bounds")?
                            .as_float() as f32,
                    );
                    ybuf.push(
                        self.memory
                            .get(y + t)
                            .ok_or("velem operand out of bounds")?
                            .as_float() as f32,
                    );
                }
                let mut obuf = vec![0.0f32; n];
                // SAFETY: xbuf/ybuf/obuf are exactly n f32 long — the kernel's contract.
                unsafe {
                    mercury_runtime::mercury_velem_f32(
                        xbuf.as_ptr(),
                        ybuf.as_ptr(),
                        obuf.as_mut_ptr(),
                        n as i64,
                        a,
                        b,
                        c,
                        op,
                    );
                }
                for (t, &val) in obuf.iter().enumerate() {
                    *self
                        .memory
                        .get_mut(out + t)
                        .ok_or("velem output out of bounds")? = Value::Float(val as f64);
                }
                Ok(Value::Unit)
            }
            // `mercury_vhorner_f32(x, out, n, coeffs, ncoeff)` — the streaming Horner-polynomial kernel
            // a recognized `r = c0; r = r*x + c1; …; out[i] = r` loop lowers to. Marshal `n` f32 from x
            // and `ncoeff` f32 from coeffs, call the identical runtime kernel, write out — so the
            // differential oracle stays exact despite the wider lanes / non-temporal stores.
            "mercury_vhorner_f32" => {
                let x = ptr(args[0])?;
                let out = ptr(args[1])?;
                let n = args[2].as_int() as usize;
                let coeffs = ptr(args[3])?;
                let ncoeff = args[4].as_int() as usize;
                let mut xbuf = Vec::with_capacity(n);
                for t in 0..n {
                    xbuf.push(
                        self.memory
                            .get(x + t)
                            .ok_or("vhorner operand out of bounds")?
                            .as_float() as f32,
                    );
                }
                let mut cbuf = Vec::with_capacity(ncoeff);
                for t in 0..ncoeff {
                    cbuf.push(
                        self.memory
                            .get(coeffs + t)
                            .ok_or("vhorner coeff out of bounds")?
                            .as_float() as f32,
                    );
                }
                let mut obuf = vec![0.0f32; n];
                // SAFETY: xbuf/obuf are n f32, cbuf is ncoeff f32 — the kernel's contract.
                unsafe {
                    mercury_runtime::mercury_vhorner_f32(
                        xbuf.as_ptr(),
                        obuf.as_mut_ptr(),
                        n as i64,
                        cbuf.as_ptr(),
                        ncoeff as i64,
                    );
                }
                for (t, &val) in obuf.iter().enumerate() {
                    *self
                        .memory
                        .get_mut(out + t)
                        .ok_or("vhorner output out of bounds")? = Value::Float(val as f64);
                }
                Ok(Value::Unit)
            }
            // `mercury_sreduce_f32[_parallel](x, y, n, op) -> f32` — the deterministic reduction kernel
            // a `@parallel` reduction loop lowers to (dot / ssd / sum). Marshal `n` f32 out of x and y
            // and call the *serial* runtime kernel, which is bit-identical to the parallel one the
            // native backend runs — so the differential oracle stays exact. The unary sum passes
            // y == x (harmless redundant reads); the kernel ignores y for it.
            "mercury_sreduce_f32" | "mercury_sreduce_f32_parallel" => {
                let x = ptr(args[0])?;
                let y = ptr(args[1])?;
                let n = args[2].as_int() as usize;
                let op = args[3].as_int() as i64;
                let mut xbuf = Vec::with_capacity(n);
                let mut ybuf = Vec::with_capacity(n);
                for t in 0..n {
                    xbuf.push(
                        self.memory
                            .get(x + t)
                            .ok_or("reduce operand out of bounds")?
                            .as_float() as f32,
                    );
                    ybuf.push(
                        self.memory
                            .get(y + t)
                            .ok_or("reduce operand out of bounds")?
                            .as_float() as f32,
                    );
                }
                // Offload the reduction to the accelerator (GPU) when it covers this op; otherwise
                // (or with no accelerator) call the serial CPU kernel.
                let offloaded = self
                    .accel
                    .as_mut()
                    .and_then(|acc| acc.sreduce(op, &xbuf, &ybuf));
                let r = match offloaded {
                    Some(Ok(v)) => v,
                    Some(Err(e)) => return Err(e),
                    // SAFETY: xbuf/ybuf are exactly n f32 long — the kernel's contract.
                    None => unsafe {
                        mercury_runtime::mercury_sreduce_f32(
                            xbuf.as_ptr(),
                            ybuf.as_ptr(),
                            n as i64,
                            op,
                        )
                    },
                };
                Ok(Value::Float(r as f64))
            }
            // `mercury_dot_bf16(x, y, n) -> f32` / `mercury_sum_bf16(x, n) -> f32` — the bf16
            // mixed-precision reduction a `s += (x[k] as f32) [* (y[k] as f32)]` loop over `[bf16; _]`
            // arrays lowers to (bf16 storage, f32 accumulate). The interpreter stores bf16 as the
            // bf16-rounded f32 value; reconstruct the exact 16 stored bits via `f32_to_bf16_bits`
            // (idempotent on an already-bf16-rounded value) so the buffer is identical to the native
            // backend's 2-byte storage, then call the identical kernel — the differential gate stays
            // exact despite the kernel's reassociated 8-lane accumulation.
            "mercury_dot_bf16" | "mercury_sum_bf16" => {
                let is_dot = name == "mercury_dot_bf16";
                let x = ptr(args[0])?;
                let (y, n) = if is_dot {
                    (ptr(args[1])?, args[2].as_int() as usize)
                } else {
                    (x, args[1].as_int() as usize)
                };
                let bits = |idx: usize, t: usize| -> Result<u16, String> {
                    Ok(mercury_runtime::f32_to_bf16_bits(
                        self.memory
                            .get(idx + t)
                            .ok_or("bf16 reduce operand out of bounds")?
                            .as_float() as f32,
                    ))
                };
                let mut xbuf = Vec::with_capacity(n);
                let mut ybuf = Vec::with_capacity(n);
                for t in 0..n {
                    xbuf.push(bits(x, t)?);
                    if is_dot {
                        ybuf.push(bits(y, t)?);
                    }
                }
                // SAFETY: the buffers are exactly n u16 long — the kernels' contract.
                let r = unsafe {
                    if is_dot {
                        mercury_runtime::mercury_dot_bf16(xbuf.as_ptr(), ybuf.as_ptr(), n as i64)
                    } else {
                        mercury_runtime::mercury_sum_bf16(xbuf.as_ptr(), n as i64)
                    }
                };
                Ok(Value::Float(r as f64))
            }
            // `mercury_reduce_bf16(x, n, op) -> f32` — the bf16 max-family reduction (max/min/absmax)
            // a `m = fmax/fmin(m, (x[k] as f32))` loop over `[bf16]` lowers to. Reconstruct the exact
            // bf16 bits (as the dot/sum path does), call the identical kernel — the widen is lossless
            // and max/min round nothing, so interp == native exactly.
            "mercury_reduce_bf16" => {
                let x = ptr(args[0])?;
                let n = args[1].as_int() as usize;
                let op = args[2].as_int() as i64;
                let mut xbuf = Vec::with_capacity(n);
                for t in 0..n {
                    xbuf.push(mercury_runtime::f32_to_bf16_bits(
                        self.memory
                            .get(x + t)
                            .ok_or("bf16 reduce operand out of bounds")?
                            .as_float() as f32,
                    ));
                }
                // SAFETY: xbuf is exactly n u16 long — the kernel's contract.
                let r =
                    unsafe { mercury_runtime::mercury_reduce_bf16(xbuf.as_ptr(), n as i64, op) };
                Ok(Value::Float(r as f64))
            }
            // The IEEE-f16 twins: `mercury_dot_f16` / `mercury_sum_f16` / `mercury_reduce_f16` — same
            // marshaling as the bf16 reductions, but reconstruct the exact f16 bits via
            // `f32_to_f16_bits` (the stored value is already f16-rounded, so this is exact) and call
            // the F16C kernels. interp == native bit-for-bit (the widen is lossless, identical kernel).
            "mercury_dot_f16" | "mercury_sum_f16" => {
                let is_dot = name == "mercury_dot_f16";
                let x = ptr(args[0])?;
                let (y, n) = if is_dot {
                    (ptr(args[1])?, args[2].as_int() as usize)
                } else {
                    (x, args[1].as_int() as usize)
                };
                let bits = |idx: usize, t: usize| -> Result<u16, String> {
                    Ok(mercury_runtime::f32_to_f16_bits(
                        self.memory
                            .get(idx + t)
                            .ok_or("f16 reduce operand out of bounds")?
                            .as_float() as f32,
                    ))
                };
                let mut xbuf = Vec::with_capacity(n);
                let mut ybuf = Vec::with_capacity(n);
                for t in 0..n {
                    xbuf.push(bits(x, t)?);
                    if is_dot {
                        ybuf.push(bits(y, t)?);
                    }
                }
                // SAFETY: the buffers are exactly n u16 long — the kernels' contract.
                let r = unsafe {
                    if is_dot {
                        mercury_runtime::mercury_dot_f16(xbuf.as_ptr(), ybuf.as_ptr(), n as i64)
                    } else {
                        mercury_runtime::mercury_sum_f16(xbuf.as_ptr(), n as i64)
                    }
                };
                Ok(Value::Float(r as f64))
            }
            "mercury_reduce_f16" => {
                let x = ptr(args[0])?;
                let n = args[1].as_int() as usize;
                let op = args[2].as_int() as i64;
                let mut xbuf = Vec::with_capacity(n);
                for t in 0..n {
                    xbuf.push(mercury_runtime::f32_to_f16_bits(
                        self.memory
                            .get(x + t)
                            .ok_or("f16 reduce operand out of bounds")?
                            .as_float() as f32,
                    ));
                }
                // SAFETY: xbuf is exactly n u16 long — the kernel's contract.
                let r = unsafe { mercury_runtime::mercury_reduce_f16(xbuf.as_ptr(), n as i64, op) };
                Ok(Value::Float(r as f64))
            }
            // `mercury_axpby_bf16(x, y, out, n, a, b)` — the bf16→f32 streaming axpby a recognized
            // `out[k] = a*(x[k] as f32) + b*(y[k] as f32)` map over `[bf16; _]` inputs (f32 output)
            // lowers to. Reconstruct the bf16 input bits exactly (as the reductions do), call the
            // identical kernel, write the f32 result back — so the differential gate stays exact.
            "mercury_axpby_bf16" | "mercury_axpby_f16" => {
                let is_f16 = name == "mercury_axpby_f16";
                let x = ptr(args[0])?;
                let y = ptr(args[1])?;
                let out = ptr(args[2])?;
                let n = args[3].as_int() as usize;
                let a = args[4].as_float() as f32;
                let b = args[5].as_float() as f32;
                let bits = |idx: usize, t: usize| -> Result<u16, String> {
                    let f = self
                        .memory
                        .get(idx + t)
                        .ok_or("lowp axpby operand out of bounds")?
                        .as_float() as f32;
                    Ok(if is_f16 {
                        mercury_runtime::f32_to_f16_bits(f)
                    } else {
                        mercury_runtime::f32_to_bf16_bits(f)
                    })
                };
                let mut xbuf = Vec::with_capacity(n);
                let mut ybuf = Vec::with_capacity(n);
                for t in 0..n {
                    xbuf.push(bits(x, t)?);
                    ybuf.push(bits(y, t)?);
                }
                let mut obuf = vec![0.0f32; n];
                // SAFETY: xbuf/ybuf are n u16, obuf is n f32 — the kernel's contract.
                unsafe {
                    if is_f16 {
                        mercury_runtime::mercury_axpby_f16(
                            xbuf.as_ptr(),
                            ybuf.as_ptr(),
                            obuf.as_mut_ptr(),
                            n as i64,
                            a,
                            b,
                        );
                    } else {
                        mercury_runtime::mercury_axpby_bf16(
                            xbuf.as_ptr(),
                            ybuf.as_ptr(),
                            obuf.as_mut_ptr(),
                            n as i64,
                            a,
                            b,
                        );
                    }
                }
                for (t, &val) in obuf.iter().enumerate() {
                    *self
                        .memory
                        .get_mut(out + t)
                        .ok_or("lowp axpby output out of bounds")? = Value::Float(val as f64);
                }
                Ok(Value::Unit)
            }
            // `mercury_norm_f32[_parallel](x, out, rows, cols, eps_bits, op)` — the fused row-wise
            // softmax / LayerNorm / RMSNorm kernel a recognized multi-pass norm lowers to. Marshal the
            // `rows*cols` f32 out of x, call the *serial* runtime kernel (bit-identical to the parallel
            // one the native backend runs, since rows are independent), write the result to out.
            // Reading all of x before writing out makes the in-place (x == out) case correct.
            "mercury_norm_f32" | "mercury_norm_f32_parallel" => {
                let x = ptr(args[0])?;
                let out = ptr(args[1])?;
                let rows = args[2].as_int() as usize;
                let cols = args[3].as_int() as usize;
                let eps_bits = args[4].as_int() as i64;
                let op = args[5].as_int() as i64;
                let n = rows * cols;
                let mut xbuf = Vec::with_capacity(n);
                for t in 0..n {
                    xbuf.push(
                        self.memory
                            .get(x + t)
                            .ok_or("norm operand out of bounds")?
                            .as_float() as f32,
                    );
                }
                let mut obuf = vec![0.0f32; n];
                // Offload the fused norm to the accelerator (GPU) when present; else the CPU kernel.
                // The runtime ABI carries eps as raw f32 bits; the GPU wrapper takes an f32, so widen.
                let offloaded = self.accel.as_mut().and_then(|acc| {
                    acc.norm(
                        op,
                        &xbuf,
                        &mut obuf,
                        rows,
                        cols,
                        f32::from_bits(eps_bits as u32),
                    )
                });
                match offloaded {
                    Some(Ok(())) => {}
                    Some(Err(e)) => return Err(e),
                    // SAFETY: xbuf/obuf are exactly rows*cols f32 long — the kernel's contract.
                    None => unsafe {
                        mercury_runtime::mercury_norm_f32(
                            xbuf.as_ptr(),
                            obuf.as_mut_ptr(),
                            rows as i64,
                            cols as i64,
                            eps_bits,
                            op,
                        );
                    },
                }
                for (t, &val) in obuf.iter().enumerate() {
                    *self
                        .memory
                        .get_mut(out + t)
                        .ok_or("norm output out of bounds")? = Value::Float(val as f64);
                }
                Ok(Value::Unit)
            }
            // `mercury_norm_affine_f32[_parallel](x, out, gamma, beta, rows, cols, eps_bits, op)` — the
            // affine (per-column scale gamma + optional shift beta) fused LayerNorm/RMSNorm. Same
            // marshalling as the plain norm, plus the gamma/beta arrays (length cols). An absent param
            // lowers to a `Ptr`-typed `ConstInt(0)` → a `Value::Int(0)` here (distinct from a real array's
            // `Value::Ptr`), so match the variant: pass a null pointer (kernel uses scale 1 / shift 0).
            // Both names marshal through the *serial* runtime kernel (bit-identical — rows independent).
            "mercury_norm_affine_f32" | "mercury_norm_affine_f32_parallel" => {
                let x = ptr(args[0])?;
                let out = ptr(args[1])?;
                let gamma_idx = match args[2] {
                    Value::Ptr(p) => Some(p),
                    _ => None,
                };
                let beta_idx = match args[3] {
                    Value::Ptr(p) => Some(p),
                    _ => None,
                };
                let rows = args[4].as_int() as usize;
                let cols = args[5].as_int() as usize;
                let eps_bits = args[6].as_int() as i64;
                let op = args[7].as_int() as i64;
                let n = rows * cols;
                let mut xbuf = Vec::with_capacity(n);
                for t in 0..n {
                    xbuf.push(
                        self.memory
                            .get(x + t)
                            .ok_or("norm operand out of bounds")?
                            .as_float() as f32,
                    );
                }
                let mut gbuf = Vec::new();
                if let Some(g) = gamma_idx {
                    for t in 0..cols {
                        gbuf.push(
                            self.memory
                                .get(g + t)
                                .ok_or("norm gamma out of bounds")?
                                .as_float() as f32,
                        );
                    }
                }
                let mut bbuf = Vec::new();
                if let Some(bb) = beta_idx {
                    for t in 0..cols {
                        bbuf.push(
                            self.memory
                                .get(bb + t)
                                .ok_or("norm beta out of bounds")?
                                .as_float() as f32,
                        );
                    }
                }
                let gptr = if gamma_idx.is_some() {
                    gbuf.as_ptr()
                } else {
                    std::ptr::null()
                };
                let bptr = if beta_idx.is_some() {
                    bbuf.as_ptr()
                } else {
                    std::ptr::null()
                };
                let mut obuf = vec![0.0f32; n];
                // SAFETY: xbuf is rows*cols; gamma/beta (when present) are cols long — kernel contract.
                unsafe {
                    mercury_runtime::mercury_norm_affine_f32(
                        xbuf.as_ptr(),
                        obuf.as_mut_ptr(),
                        gptr,
                        bptr,
                        rows as i64,
                        cols as i64,
                        eps_bits,
                        op,
                    );
                }
                for (t, &val) in obuf.iter().enumerate() {
                    *self
                        .memory
                        .get_mut(out + t)
                        .ok_or("norm output out of bounds")? = Value::Float(val as f64);
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

/// Square root with the result rounded to its (`f32` or full `f64`) type, matching the native
/// `sqrt`. Both backends use the hardware square root, so they agree bit-for-bit.
fn apply_sqrt(v: Value, rty: Option<&MirType>) -> Value {
    if rty.is_some_and(is_narrow_float) {
        Value::Float((v.as_float() as f32).sqrt() as f64)
    } else {
        Value::Float(v.as_float().sqrt())
    }
}

/// Round to an integral, mirroring the native `roundss`/`roundps` (and `Op::Round`'s contract):
/// `Nearest` is round-to-nearest-ties-to-**even** (`round_ties_even`, the IEEE form `nearest` emits —
/// *not* `round`, which is ties-away). Narrow-float results round in `f32` first, like every other op.
fn apply_round(mode: mercury_mir::RoundMode, v: Value, rty: Option<&MirType>) -> Value {
    use mercury_mir::RoundMode::*;
    let f = |x: f64| -> f64 {
        match mode {
            Nearest => x.round_ties_even(),
            Floor => x.floor(),
            Ceil => x.ceil(),
            Trunc => x.trunc(),
        }
    };
    if rty.is_some_and(is_narrow_float) {
        let x = v.as_float() as f32;
        let r = match mode {
            Nearest => x.round_ties_even(),
            Floor => x.floor(),
            Ceil => x.ceil(),
            Trunc => x.trunc(),
        };
        Value::Float(r as f64)
    } else {
        Value::Float(f(v.as_float()))
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
        // Shift counts are masked to the result width (as x86/Cranelift do): `1i32 << 32 == 1`, not
        // 0. `LShr` is logical, so it shifts the value's own unsigned width window, not the
        // sign-extended i128 (else a high-bit-set operand shifts in 1s and diverges from native).
        Shl => {
            let bits = rty.map(int_bits).unwrap_or(64);
            x.wrapping_shl((y as u32) & (bits - 1))
        }
        LShr => {
            let bits = rty.map(int_bits).unwrap_or(64);
            (uval(x, bits) >> ((y as u32) & (bits - 1))) as i128
        }
        AShr => {
            let bits = rty.map(int_bits).unwrap_or(64);
            x >> ((y as u32) & (bits - 1))
        }
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
        // Widening is exact. Narrowing to bf16/f16 rounds to that grid (the native backend rounds on
        // store / cast identically); narrowing to f32 is left to the per-op f32 rounding in `exec`.
        FpExt => Value::Float(v.as_float()),
        FpTrunc => match to {
            MirType::BF16 => Value::Float(mercury_runtime::round_bf16(v.as_float() as f32) as f64),
            MirType::F16 => Value::Float(mercury_runtime::round_f16(v.as_float() as f32) as f64),
            _ => Value::Float(v.as_float()),
        },
        // Reinterpret the raw bits between an int and a same-width float (matches the native
        // `bitcast`); the `exp` polynomial reconstructs `2^n` this way. `cast_kind` never produces
        // an int↔float bitcast from source, so this path is exercised only by hand-built MIR.
        // Same-kind scalar bitcasts and vectors pass through (vector casts are handled lane-wise
        // in `eval`).
        Bitcast => match (from, to) {
            (MirType::I32, MirType::F32) => Value::Float(f32::from_bits(v.as_int() as u32) as f64),
            (MirType::I64, MirType::F64) => Value::Float(f64::from_bits(v.as_int() as u64)),
            (MirType::F32, MirType::I32) => Value::Int((v.as_float() as f32).to_bits() as i128),
            (MirType::F64, MirType::I64) => Value::Int(v.as_float().to_bits() as i128),
            _ => v,
        },
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

    /// The typed kernel-entry ABI: lower an f32-buffer kernel and run it over caller-provided
    /// buffers (no `main` wrapper), then read the full output back. This is the interpreter side of
    /// the full-buffer differential gate.
    #[test]
    fn run_kernel_f32_saxpy_and_dot() {
        let mut interner = Interner::new();
        let src = "module m\n\
            fn saxpy(x: [f32; 8], y: [f32; 8], out: [f32; 8]) { \
                for i in 0..8 { out[i] = 2.0 * x[i] + y[i]; } }\n\
            fn dot(x: [f32; 8], y: [f32; 8], out: [f32; 8]) { \
                let mut s: f32 = 0.0; for i in 0..8 { s = s + x[i] * y[i]; } out[0] = s; }\n";
        let (module, pd) = mercury_parser::parse_module(src, SourceId(0), &mut interner);
        assert!(pd.iter().all(|d| !d.is_error()), "parse: {pd:?}");
        let (sema, sd) = mercury_sema::check(&module, &interner);
        assert!(sd.iter().all(|d| !d.is_error()), "sema: {sd:?}");
        let (program, ld) = mercury_mir_build::lower_program(&module, &sema, &mut interner);
        assert!(ld.iter().all(|d| !d.is_error()), "lower: {ld:?}");

        let x: Vec<f32> = (0..8).map(|i| i as f32).collect();
        let y: Vec<f32> = (0..8).map(|i| (i as f32) * 0.5 - 1.0).collect();

        // saxpy: out = 2*x + y, full-buffer.
        let mut out = vec![0.0f32; 8];
        let (mut xb, mut yb) = (x.clone(), y.clone());
        run_kernel_f32(
            &program,
            interner.intern("saxpy"),
            &mut [&mut xb, &mut yb, &mut out],
            &interner,
        )
        .unwrap();
        for i in 0..8 {
            assert_eq!(out[i], 2.0 * x[i] + y[i], "saxpy[{i}]");
        }

        // dot: out[0] = sum(x*y).
        let mut out = vec![0.0f32; 8];
        let (mut xb, mut yb) = (x.clone(), y.clone());
        run_kernel_f32(
            &program,
            interner.intern("dot"),
            &mut [&mut xb, &mut yb, &mut out],
            &interner,
        )
        .unwrap();
        let expect: f32 = (0..8).map(|i| x[i] * y[i]).sum();
        assert_eq!(out[0], expect, "dot");
    }

    #[test]
    fn run_kernel_i8_linear() {
        // C[2,2] = A[2,4](u8) · B[2,4](i8)ᵀ — the int8 kernel-entry ABI, checked against a hand
        // i64 reference. A=[1..8], B=[-6..1]; the recognizer dispatches this to mercury_i8gemm_nt.
        let mut interner = Interner::new();
        let src = "module m\n\
            fn lin(a: [u8; 8], b: [i8; 8], c: [i32; 4]) { \
                for i in 0..2 { for j in 0..2 { let mut s: i32 = 0; \
                for k in 0..4 { s = s + (a[i*4+k] as i32) * (b[j*4+k] as i32); } \
                c[i*2+j] = s; } } }\n";
        let (module, pd) = mercury_parser::parse_module(src, SourceId(0), &mut interner);
        assert!(pd.iter().all(|d| !d.is_error()), "parse: {pd:?}");
        let (sema, sd) = mercury_sema::check(&module, &interner);
        assert!(sd.iter().all(|d| !d.is_error()), "sema: {sd:?}");
        let (program, ld) = mercury_mir_build::lower_program(&module, &sema, &mut interner);
        assert!(ld.iter().all(|d| !d.is_error()), "lower: {ld:?}");

        let a: [u8; 8] = [1, 2, 3, 4, 5, 6, 7, 8];
        let b: [i8; 8] = [-6, -5, -4, -3, -2, -1, 0, 1];
        let mut c = [0i32; 4];
        run_kernel_i8(&program, interner.intern("lin"), &a, &b, &mut c, &interner).unwrap();

        // Independent i64 reference: c[i,j] = sum_k a[i*4+k] * b[j*4+k].
        let mut want = [0i32; 4];
        for i in 0..2 {
            for j in 0..2 {
                let mut s = 0i64;
                for k in 0..4 {
                    s += a[i * 4 + k] as i64 * b[j * 4 + k] as i64;
                }
                want[i * 2 + j] = s as i32;
            }
        }
        assert_eq!(c, want, "i8 linear");
        assert_eq!(c, [-40, 0, -112, -8], "i8 linear expected values");
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
