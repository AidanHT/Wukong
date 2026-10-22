//! `wukong_interp` — a from-scratch MIR interpreter.
//!
//! No crates.io dependencies of its own by design: this is the always-available execution path and
//! the oracle differential tests compare the **native** (Cranelift) backend against — the textual
//! LLVM emitter is never assembled or run, so it is not what the gate judges. It walks the CFG block
//! by block, keeping a per-call register file and a shared flat memory for
//! `alloca`/`load`/`store`/`gep`. Scalar MIR arrays use one `Value` slot per element. Source
//! aggregates lowered to byte buffers use one slot per layout byte, including padding; fields
//! occupy their starting byte's slot, not packed host bytes (see `slot_count`). It
//! does depend on the workspace's `wukong_runtime`, so a recognized kernel call runs the *identical*
//! microkernel the native backend links.

use std::cell::Cell;
use std::collections::{BTreeMap, HashMap};
use wukong_backend::{Artifact, Backend};
use wukong_mir::{
    BasicBlock, BinOp, BlockId, CastKind, CmpOp, Function, MirType, Op, Program, Terminator,
    ValueId, VecKernel, VecLane, VecOp, VecValue,
};
use wukong_span::{Interner, Symbol};

/// The view a kernel helper validates an operand against ([`marshal::Arena`]): the interpreter's
/// `memory` with its live allocations, each borrowed as a field so the interpreter's other fields
/// (the accelerator, the scratch pool) stay usable while the view is alive.
macro_rules! arena {
    ($interp:expr) => {
        $crate::marshal::Arena {
            slots: &$interp.memory,
            live: &$interp.slots.live,
        }
    };
}

/// [`arena!`] for a helper that writes ([`marshal::ArenaMut`]).
macro_rules! arena_mut {
    ($interp:expr) => {
        $crate::marshal::ArenaMut {
            slots: &mut $interp.memory,
            live: &$interp.slots.live,
        }
    };
}

mod marshal;
use marshal::{Arena, LiveBlocks, Overlap};
// Every intrinsic name the interpreter implements, as a dense id (`Callee::Intrinsic`).
mod intrinsics;
use intrinsics::Intrinsic;

/// A runtime value. Integers are stored width-agnostically in an `i128` and masked per result
/// type; pointers are **slot** indices into flat memory, not host byte addresses. Scalar arrays
/// use one slot per element; lowered aggregate byte buffers use one slot per layout byte.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Value {
    Int(i128),
    Float(f64),
    Ptr(usize),
    /// A SIMD vector value: an index into the interpreter's `vecs` side arena (keeps `Value` cheap
    /// and `Copy`). Produced by `Splat`, a vector `Load`, and every lane-wise arm of `eval`
    /// (`Bin`/`Cmp`/`Neg`/`Not`/`Cast`/`Select`/`Fma`/`Sqrt`/`Round`); consumed by a vector `Store`
    /// and by `vec_lanes` on the way into those arms.
    VecRef(usize),
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

    /// Kernel buffers contain numeric scalar slots. Keep ordinary scalar evaluation permissive,
    /// but never hide type-confused or undefined buffer memory by manufacturing a zero.
    fn kernel_float(self, context: &str) -> Result<f64, String> {
        match self {
            Value::Float(f) => Ok(f),
            Value::Int(i) => Ok(i as f64),
            Value::Ptr(_) => Err(format!(
                "{context}: floating buffer contains a pointer slot"
            )),
            Value::VecRef(_) => Err(format!("{context}: floating buffer contains a vector slot")),
            Value::Unit => Err(format!(
                "{context}: floating buffer contains an uninitialized slot"
            )),
        }
    }

    /// The integer-buffer twin of [`Self::kernel_float`]: an integer slot is an `Int`, and anything
    /// else in a buffer a kernel reads as integers is type confusion, reported rather than coerced.
    fn kernel_int(self, context: &str) -> Result<i128, String> {
        match self {
            Value::Int(i) => Ok(i),
            Value::Float(_) => Err(format!("{context}: integer buffer contains a float slot")),
            Value::Ptr(_) => Err(format!("{context}: integer buffer contains a pointer slot")),
            Value::VecRef(_) => Err(format!("{context}: integer buffer contains a vector slot")),
            Value::Unit => Err(format!(
                "{context}: integer buffer contains an uninitialized slot"
            )),
        }
    }

    fn truthy(self) -> bool {
        self.as_int() != 0
    }
}

/// How one [`VecLane`]'s values travel through slot memory: the same representation the scalar
/// `Load`/`Store` arms use for that MIR type (a float slot holds the value as an `f64`, an integer
/// slot the sign-extended value in an `i128`), so a recipe and the scalar tail beside it read and
/// write identical slots.
trait KernelLaneValue: VecValue {
    fn read(v: Value, context: &str) -> Result<Self, String>;
    fn write(self) -> Value;
}

impl KernelLaneValue for f32 {
    fn read(v: Value, context: &str) -> Result<Self, String> {
        Ok(v.kernel_float(context)? as f32)
    }
    fn write(self) -> Value {
        Value::Float(self as f64)
    }
}

impl KernelLaneValue for f64 {
    fn read(v: Value, context: &str) -> Result<Self, String> {
        v.kernel_float(context)
    }
    fn write(self) -> Value {
        Value::Float(self)
    }
}

impl KernelLaneValue for i32 {
    fn read(v: Value, context: &str) -> Result<Self, String> {
        // Slots hold the value sign-extended; the low 32 bits are the lane.
        Ok(v.kernel_int(context)? as i32)
    }
    fn write(self) -> Value {
        Value::Int(self as i128)
    }
}

impl KernelLaneValue for i64 {
    fn read(v: Value, context: &str) -> Result<Self, String> {
        Ok(v.kernel_int(context)? as i64)
    }
    fn write(self) -> Value {
        Value::Int(self as i128)
    }
}

/// Allocate an exact scalar-lane result without invoking the allocator's abort-on-OOM path.
fn collect_lanes(values: impl ExactSizeIterator<Item = Value>) -> Result<Vec<Value>, String> {
    let mut lanes = Vec::new();
    lanes
        .try_reserve_exact(values.len())
        .map_err(|_| "SIMD lane allocation failed")?;
    for value in values {
        if matches!(value, Value::VecRef(_)) {
            return Err("SIMD lanes must be scalar; nested vectors are unsupported".into());
        }
        lanes.push(value);
    }
    Ok(lanes)
}

/// Validate a whole kernel operand — operand `k` of the call, `rows * cols` slots from `base` —
/// before allocating scratch: the extent product is checked, the range must lie in the arena, and
/// then inside ONE live allocation (`marshal::LiveBlocks`; an aggregate is one allocation).
/// Callers validate every used operand before marshalling.
fn kernel_values<'a>(
    arena: Arena<'a>,
    k: usize,
    base: usize,
    rows: usize,
    cols: usize,
    context: &str,
) -> Result<&'a [Value], String> {
    let len = rows
        .checked_mul(cols)
        .ok_or_else(|| format!("{context}: operand extent overflow"))?;
    let end = base
        .checked_add(len)
        .ok_or_else(|| format!("{context}: operand out of bounds"))?;
    let values = arena
        .slots
        .get(base..end)
        .ok_or_else(|| format!("{context}: operand out of bounds"))?;
    arena.live.check(context, k, base, len)?;
    Ok(values)
}

/// Reserve fallibly before any push/extend, including byte reconstruction and output-only buffers.
fn kernel_scratch<T>(len: usize, context: &str) -> Result<Vec<T>, String> {
    let mut buffer = Vec::new();
    buffer
        .try_reserve_exact(len)
        .map_err(|_| format!("{context}: operand allocation failed"))?;
    Ok(buffer)
}

fn kernel_buffer<T>(
    values: &[Value],
    context: &str,
    mut convert: impl FnMut(Value) -> Result<T, String>,
) -> Result<Vec<T>, String> {
    let mut buffer = kernel_scratch(values.len(), context)?;
    for &value in values {
        buffer.push(convert(value)?);
    }
    Ok(buffer)
}

/// Reconstruct read paths without infallible growth, retaining low-byte and lossy-UTF-8 semantics.
fn file_read_path(memory: &[Value], base: usize) -> Result<String, String> {
    let values = memory.get(base..).unwrap_or(&[]);
    let len = values
        .iter()
        .position(|v| v.as_int() as u8 == 0)
        .unwrap_or(values.len());
    let bytes = kernel_buffer(
        &values[..len],
        "file-io read path",
        |v| Ok(v.as_int() as u8),
    )?;
    let len = bytes.utf8_chunks().try_fold(0usize, |len, chunk| {
        len.checked_add(chunk.valid().len())
            .and_then(|len| len.checked_add(if chunk.invalid().is_empty() { 0 } else { 3 }))
            .ok_or("file-io read: path extent overflow")
    })?;
    let mut path = String::new();
    path.try_reserve_exact(len)
        .map_err(|_| "file-io read: path allocation failed")?;
    for chunk in bytes.utf8_chunks() {
        path.push_str(chunk.valid());
        if !chunk.invalid().is_empty() {
            path.push('\u{fffd}');
        }
    }
    Ok(path)
}

/// Decode only after the complete actual destination is validated. A read error leaves the arena
/// untouched; a short file is already clamped by the caller's metadata check. For malformed
/// destinations, the range diagnostic now takes precedence over a simultaneous read failure.
fn read_file_elements(
    memory: &mut [Value],
    reader: &mut impl std::io::Read,
    data_base: usize,
    n: usize,
    elem: usize,
    name: &str,
) -> Result<Value, String> {
    if n > 0 {
        let end = data_base
            .checked_add(n)
            .ok_or("file-io read: data out of bounds")?;
        memory
            .get(data_base..end)
            .ok_or("file-io read: data out of bounds")?;
        let byte_len = n
            .checked_mul(elem)
            .ok_or("file-io read: byte extent overflow")?;
        let mut bytes = kernel_scratch(byte_len, "file-io read")?;
        bytes.resize(byte_len, 0u8);
        if reader.read_exact(&mut bytes).is_err() {
            return Ok(Value::Int(-2));
        }
        for k in 0..n {
            let off = k * elem;
            let slot = data_base + k;
            if slot >= memory.len() {
                return Err("file-io read: data out of bounds".into());
            }
            memory[slot] = match name {
                "wukong_rt_read_f32" => {
                    let x = f32::from_le_bytes([
                        bytes[off],
                        bytes[off + 1],
                        bytes[off + 2],
                        bytes[off + 3],
                    ]);
                    Value::Float(x as f64)
                }
                "wukong_rt_read_i32" => {
                    let x = i32::from_le_bytes([
                        bytes[off],
                        bytes[off + 1],
                        bytes[off + 2],
                        bytes[off + 3],
                    ]);
                    Value::Int(x as i128)
                }
                "wukong_rt_read_i64" => {
                    let x = i64::from_le_bytes([
                        bytes[off],
                        bytes[off + 1],
                        bytes[off + 2],
                        bytes[off + 3],
                        bytes[off + 4],
                        bytes[off + 5],
                        bytes[off + 6],
                        bytes[off + 7],
                    ]);
                    Value::Int(x as i128)
                }
                "wukong_rt_read_f64" => {
                    let x = f64::from_le_bytes([
                        bytes[off],
                        bytes[off + 1],
                        bytes[off + 2],
                        bytes[off + 3],
                        bytes[off + 4],
                        bytes[off + 5],
                        bytes[off + 6],
                        bytes[off + 7],
                    ]);
                    Value::Float(x)
                }
                "wukong_rt_read_i8" => {
                    // i8 is signed: reinterpret the byte's bit pattern then sign-extend.
                    let x = bytes[off] as i8;
                    Value::Int(x as i128)
                }
                // wukong_rt_read_u8 — sign-extended like every integer slot: `exec` masks each
                // integer result to its width sign-extended and a `Load` returns the slot verbatim,
                // so the `u8` byte 0xC8 must be `Int(-56)`, exactly what a `200u8` constant or a
                // `u8` store leaves. Zero-extended, `b == 200` and `b > 150` (unsigned) compared a
                // different representation of the same byte: 0 here, 1 natively.
                _ => Value::Int(bytes[off] as i8 as i128),
            };
        }
    }
    Ok(Value::Int(n as i128))
}

// The interpreter twins of the ML builtins' runtime entries, beside the frozen decoder above.
mod ml_io;
// The interpreter twins of the decode builtins' entries (KV-cache attention, half / quantized GEMV).
mod decode;

// The interpreter twins of the text runtime entries (`printf`, `sprintf`, structural `print`) and
// of the process interface (`arg`, `env`, `read_stdin_u8`, `exit`).
mod text_io;

/// Install the program's argument vector (`arg(0)` is the program) for an in-process run — the
/// interpreter's, and the JIT's, which reads the same `wukong_runtime` state.
pub use wukong_runtime::process::set_program_args;

/// Pointers at or above this value encode a function index (`FUNC_TAG + idx`) rather than a memory
/// address — how `Op::FuncAddr` is represented for `parallel_for` to call back. Real `alloca`
/// memory never grows near it.
const FUNC_TAG: usize = 1 << 60;

/// What one callee name reaches.
#[derive(Clone, Copy)]
enum Callee {
    /// The program's function at this index.
    Func(usize),
    /// The program's `extern "C"` declaration at this index.
    Extern(usize),
    /// A runtime intrinsic the interpreter implements (`intrinsics.rs`).
    Intrinsic(Intrinsic),
    /// None of the above: calling it is the "unknown function or intrinsic" error.
    Unknown,
}

/// Every callee name of a program resolved once, as a dense array indexed by `Symbol` id. A call
/// used to hash its callee into a SipHash map and, on every miss — every intrinsic call, including
/// the scalar transcendentals — scan the extern list linearly before falling through to the
/// intrinsic name match; now it is one bounds-checked load (Wasm resolves `call` to an index at
/// validation time for the same reason). Resolution order is unchanged: a function wins over an
/// extern of the same name, the first of duplicated names wins, and only a name that is neither
/// is looked up in the intrinsic table — once, the first time the run calls it, after which the
/// call dispatches on the cached [`Intrinsic`] id rather than on its name.
///
/// It also caches, per function, which registers hold SIMD vectors (see [`Interp::enter_frame`]).
struct CalleeTable {
    /// By `Symbol` id; `None` until the run first calls a name that is not a function or extern.
    callees: Vec<Option<Callee>>,
    /// By function index: the registers of a vector type, computed the first time it is entered.
    vector_regs: Vec<Option<Box<[u32]>>>,
}

impl CalleeTable {
    fn new(program: &Program) -> Self {
        let names = program.funcs.iter().map(|f| f.name);
        let len = names
            .chain(program.externs.iter().map(|e| e.name))
            .map(|s| s.0 as usize + 1)
            .max()
            .unwrap_or(0);
        let mut table = vec![None; len];
        // Lowest priority first, each list in reverse, so the earliest entry is written last.
        for (index, ext) in program.externs.iter().enumerate().rev() {
            table[ext.name.0 as usize] = Some(Callee::Extern(index));
        }
        for (index, func) in program.funcs.iter().enumerate().rev() {
            table[func.name.0 as usize] = Some(Callee::Func(index));
        }
        CalleeTable {
            callees: table,
            vector_regs: vec![None; program.funcs.len()],
        }
    }

    /// What `symbol` reaches, resolving it through the intrinsic table on its first call (`name`
    /// is consulted only then).
    fn resolve<'n>(&mut self, symbol: Symbol, name: impl FnOnce() -> &'n str) -> Callee {
        let index = symbol.0 as usize;
        if let Some(Some(callee)) = self.callees.get(index) {
            return *callee;
        }
        let callee = Intrinsic::from_name(name()).map_or(Callee::Unknown, Callee::Intrinsic);
        if index >= self.callees.len() {
            self.callees.resize(index + 1, None);
        }
        self.callees[index] = Some(callee);
        callee
    }

    /// The vector-typed registers of `func`, the program's function at `index`.
    fn vector_registers(&mut self, index: usize, func: &Function) -> &[u32] {
        self.vector_regs[index].get_or_insert_with(|| {
            (0u32..)
                .zip(&func.value_types)
                .filter(|(_, ty)| matches!(ty, MirType::Vec(..)))
                .map(|(r, _)| r)
                .collect()
        })
    }
}

fn symbol_indices(symbols: impl ExactSizeIterator<Item = Symbol>) -> HashMap<Symbol, usize> {
    let mut indices = HashMap::with_capacity(symbols.len());
    for (index, symbol) in symbols.enumerate() {
        indices.entry(symbol).or_insert(index);
    }
    indices
}

/// The interpreter backend.
///
/// The `Backend` impl below is a *declared* seam, not the live dispatch path: nothing in the
/// workspace calls `Backend::compile`. `wukong_driver` matches its own `BackendKind::Interp` and
/// calls [`run_with_output`] directly, and the fuzzers/autodiff gate go straight to the
/// `run_kernel_*` entry points. Keep the impl three lines and put behavior in the free functions.
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

/// An accelerator that can run recognized runtime kernels in place of the CPU `wukong_runtime`
/// microkernels — the seam the **GPU backend** plugs into. The interpreter stays zero-dependency by
/// knowing only this trait; the driver supplies a GPU implementation behind its `gpu` feature, and
/// the offload runs over the *same* tree-walked MIR (so control flow, buffer layout, and every
/// non-kernel op are identical — only the recognized kernel calls move to the device).
///
/// Each method mirrors one `wukong_runtime` kernel over already-marshalled f32 buffers. Returning
/// `None` means "not handled here — fall back to the CPU kernel", so partial / precision-restricted
/// support is fine. With no accelerator installed — the default, and every entry point except
/// [`run_with_output_accel`] / [`run_kernel_f32_accel`] (which only `--backend=gpu` and the driver's
/// GPU tolerance gates reach) — the differential oracle path is **byte-for-byte unchanged**; the
/// CPU↔GPU gate is therefore a *tolerance* differential (interp-CPU vs interp-with-this-accelerator
/// over identical inputs).
pub trait Accelerator {
    /// `C = A·Bᵀ` (the `_nt` form) with `beta == 0` (overwrite). Only that form is ever offered —
    /// the GEMM arm pre-filters `beta != 0`, the plain `A·B`, and `_tn` (`C = Aᵀ·B`) to the CPU
    /// kernel — so return `None` only for shapes the device kernel itself cannot take.
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
    /// Fused row norm: `out` = norm over each `cols`-wide row of `x`. `op` is the runtime `NORM_*`
    /// code, and the interpreter forwards **all** of them (softmax, LayerNorm, RMSNorm, log-softmax,
    /// L2) — decline the ones the device has no kernel for or it aborts instead of falling back.
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
    /// **Fused-epilogue Linear** `C = act(A·Bᵀ + bias)` with `beta == 0` (the `wukong_sgemm_nt_epi`
    /// shape a `act(matmul(x,w)[+bias])` Wukong expression lowers to). `bias` is `None` for the
    /// bias-free SwiGLU form; `act` is the runtime `ACT_*` code (0=identity — the bias-only affine
    /// Linear, 1=relu, 2=gelu, 3=silu). Return `None` for any case the device kernel doesn't cover
    /// (non-zero beta, a bias it can't fuse, an unsupported activation, or an unaligned shape) so it
    /// falls back to the CPU fused kernel.
    // One parameter per `wukong_sgemm_nt_epi` C-ABI operand, deliberately: the trait mirrors the
    // kernel ABI, and a params struct here would ripple through every Accelerator impl
    // (`wukong_driver/src/gpu_accel.rs`) without making the seam clearer.
    #[allow(clippy::too_many_arguments)]
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

/// The slot index the interpreter reserves as the **null pointer**.
///
/// Every public entry point starts its flat memory with one never-allocated sentinel at slot 0, so
/// no `alloca`, heap allocation, static blob or kernel-entry buffer can ever be placed there and
/// `Value::Ptr(0)` means exactly what `inttoptr 0` means natively: null. Before this, slot 0 was the
/// first live allocation of a run, which is why a pointer condition could never be lowered as
/// `PtrToInt(p) != 0` (reverted at 0900352: `let p = &mut v; if p` read the first local as null in
/// the interpreter and non-null natively), why mem2reg could not materialize a null for a
/// read-before-write pointer slot, and why the interpreter could not share native's null at all.
/// A load or store through it is reported as a null dereference instead of reading the sentinel.
pub const NULL_SLOT: usize = 0;

/// The value parked in [`NULL_SLOT`]. A vector handle no arena can hold (`usize::MAX`), so it is
/// never produced by a legal store — vector stores scatter lanes, and nested vector handles are
/// rejected — and a read that slips past the explicit null checks fails as an invalid handle or a
/// type-confused kernel operand instead of yielding a plausible number. It also marks the
/// reservation: the null checks below fire only when slot 0 holds it, so the unit tests that build
/// an `Interp` by hand with a buffer at slot 0 keep their original addressing.
const NULL_SENTINEL: Value = Value::VecRef(usize::MAX);

/// Fresh run memory with [`NULL_SLOT`] reserved.
fn null_reserved_memory() -> Vec<Value> {
    vec![NULL_SENTINEL]
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
    let (result, stdout) = run_capturing(program, entry, interner);
    result.map(|code| (code, stdout))
}

/// How a run's stdout leaves the interpreter — see [`run_streaming`]. Thread-local, held at one
/// run's mode by a [`StdoutModeGuard`] for the duration of [`run_with_stdout`] and `Capture` at
/// every other time, so a kernel entry or a hand-built `Interp` on the same thread still captures.
#[derive(Clone, Copy, PartialEq, Eq)]
enum StdoutMode {
    /// Collect everything for the caller (`run_capturing`, the kernel entries, the tests).
    Capture,
    /// Write through to the process's stdout at every newline (an interactive terminal).
    LineBuffered,
    /// Write through to the process's stdout in blocks of [`STREAM_BLOCK`] bytes (a pipe or file).
    BlockBuffered,
}

thread_local! {
    static STDOUT_MODE: Cell<StdoutMode> = const { Cell::new(StdoutMode::Capture) };
}

/// Sets [`STDOUT_MODE`] for one run and restores the previous mode when the run ends, including
/// by unwinding. A run executes on the caller's thread (there is no per-run worker whose
/// thread-local dies with it), so the mode must not outlive the run that set it.
struct StdoutModeGuard(StdoutMode);

impl StdoutModeGuard {
    fn set(mode: StdoutMode) -> Self {
        Self(STDOUT_MODE.with(|m| m.replace(mode)))
    }
}

impl Drop for StdoutModeGuard {
    fn drop(&mut self) {
        STDOUT_MODE.with(|m| m.set(self.0));
    }
}

/// The block a block-buffered streaming run writes stdout in.
const STREAM_BLOCK: usize = 1 << 16;

/// Run `entry` and return its exit code or its run error **together with everything it printed
/// before it stopped** — also when it stopped on a runtime failure (a failed `assert`, a slice
/// bounds failure). A linked executable has always kept that output (its prints reach stdout as
/// they happen); `wukongc --run` prints it before the error so all three execution paths show the
/// same thing, and the differential gate compares it.
pub fn run_capturing(
    program: &Program,
    entry: Symbol,
    interner: &Interner,
) -> (Result<i64, String>, Vec<u8>) {
    run_with_stdout(program, entry, interner, StdoutMode::Capture)
}

/// Run `entry` with its stdout **streamed** to the process's stdout as it is produced, rather than
/// collected until it ends — what `wukongc --run` does, so a long run shows its progress and a run
/// that fails has already shown everything it printed before the failure. `line_buffered` writes at
/// every newline (pass it for an interactive terminal); otherwise output goes out in 64 KiB blocks,
/// as C's stdio and CPython buffer a pipe. Whatever is still buffered is flushed before this returns,
/// on success and on a runtime failure alike. The bytes are exactly the ones [`run_capturing`]
/// would have returned.
pub fn run_streaming(
    program: &Program,
    entry: Symbol,
    interner: &Interner,
    line_buffered: bool,
) -> Result<i64, String> {
    let mode = if line_buffered {
        StdoutMode::LineBuffered
    } else {
        StdoutMode::BlockBuffered
    };
    run_with_stdout(program, entry, interner, mode).0
}

fn run_with_stdout(
    program: &Program,
    entry: Symbol,
    interner: &Interner,
    mode: StdoutMode,
) -> (Result<i64, String>, Vec<u8>) {
    // Wukong calls do not recurse on the host stack (`Interp::run_function` keeps an explicit
    // frame stack), so the run executes right here on the caller's thread: no worker thread, no
    // stack reservation, and the call-depth ceiling is a language-level number (`max_call_depth`)
    // rather than a measurement of this host's frame size.
    let _mode = StdoutModeGuard::set(mode);
    let Some(func) = program.function(entry) else {
        return (
            Err(format!("no entry function `{}`", interner.resolve(entry))),
            Vec::new(),
        );
    };
    let mut interp = Interp {
        program,
        interner,
        memory: null_reserved_memory(),
        stdout: Vec::new(),
        scratch_arena: ScratchSlots::default(),
        slots: SlotReuse::tracked(),
        frames: Vec::new(),
        call_args: Vec::new(),
        scratch: Vec::with_capacity(8),
        vecs: Vec::new(),
        #[cfg(test)]
        vector_peak: 0,
        data_addrs: HashMap::new(),
        function_indices: None,
        static_indices: None,
        accel: None,
        depth: 0,
    };
    let result = match interp.run_function(func, &[]) {
        Ok(v) => Ok(v.as_int() as i64),
        // `exit(code)` stops the run with the sentinel error; the code is the run's result.
        Err(e) => wukong_runtime::process::exit_code_of(&e).ok_or(e),
    };
    interp.flush_stdout();
    (result, interp.stdout)
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
        memory: null_reserved_memory(),
        stdout: Vec::new(),
        scratch_arena: ScratchSlots::default(),
        slots: SlotReuse::tracked(),
        frames: Vec::new(),
        call_args: Vec::new(),
        scratch: Vec::with_capacity(8),
        vecs: Vec::new(),
        #[cfg(test)]
        vector_peak: 0,
        data_addrs: HashMap::new(),
        function_indices: None,
        static_indices: None,
        accel: Some(accel),
        depth: 0,
    };
    let code = match interp.run_function(func, &[]) {
        Ok(v) => v.as_int() as i64,
        Err(e) => wukong_runtime::process::exit_code_of(&e).ok_or(e)?,
    };
    Ok((code, interp.stdout))
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
    run_kernel_f32_inner(program, entry, bufs, interner, None, &mut Vec::new())
}

/// [`run_kernel_f32`], also returning the bytes the kernel printed — what a linked executable
/// running the same kernel writes to its stdout — so the caller can forward them. `wukongc --train`
/// does: a `--train --emit=exe` artifact's kernels print straight to stdout, and the in-process run
/// must print the same bytes in the same order. Like [`run_capturing`], the bytes come back **also
/// when the kernel stopped on a runtime failure** (a failed `assert`, a bounds failure): everything
/// it printed before the failing statement, which the executable has already written.
pub fn run_kernel_f32_with_output(
    program: &Program,
    entry: Symbol,
    bufs: &mut [&mut [f32]],
    interner: &Interner,
) -> (Result<(), String>, Vec<u8>) {
    let mut printed = Vec::new();
    let result = run_kernel_f32_inner(program, entry, bufs, interner, None, &mut printed);
    (result, printed)
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
    run_kernel_f32_inner(program, entry, bufs, interner, Some(accel), &mut Vec::new())
}

// Host-buffer marshalling is a bit-preserving transfer, not floating-point arithmetic.
// Numeric casts quiet signaling NaNs on some targets even when the kernel never writes
// the input. Preserve their sign/payload explicitly at this boundary only.
fn widen_buffer_f32(value: f32) -> f64 {
    if value.is_nan() {
        let bits = value.to_bits();
        f64::from_bits(
            (u64::from(bits & 0x8000_0000) << 32)
                | 0x7ff0_0000_0000_0000
                | (u64::from(bits & 0x007f_ffff) << 29),
        )
    } else {
        value as f64
    }
}

fn narrow_buffer_f32(value: f64) -> f32 {
    if value.is_nan() {
        let bits = value.to_bits();
        let payload = ((bits & 0x000f_ffff_ffff_ffff) >> 29) as u32;
        // A NaN produced in f64 may carry only low payload bits; truncation must
        // not turn that NaN into infinity.
        let payload = if payload == 0 { 0x0040_0000 } else { payload };
        f32::from_bits(((bits >> 32) as u32 & 0x8000_0000) | 0x7f80_0000 | payload)
    } else {
        value as f32
    }
}

fn run_kernel_f32_inner(
    program: &Program,
    entry: Symbol,
    bufs: &mut [&mut [f32]],
    interner: &Interner,
    accel: Option<&mut dyn Accelerator>,
    printed: &mut Vec<u8>,
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
        memory: null_reserved_memory(),
        stdout: Vec::new(),
        scratch_arena: ScratchSlots::default(),
        slots: SlotReuse::tracked(),
        frames: Vec::new(),
        call_args: Vec::new(),
        scratch: Vec::with_capacity(8),
        vecs: Vec::new(),
        #[cfg(test)]
        vector_peak: 0,
        data_addrs: HashMap::new(),
        function_indices: None,
        static_indices: None,
        accel,
        depth: 0,
    };
    // Lay each buffer out contiguously in flat memory and remember its base slot. Any `alloca`
    // the kernel performs internally grows memory *past* these regions, so it never clobbers
    // them. Each buffer is its own allocation: a kernel operand may not run from one into the next.
    let mut bases = Vec::with_capacity(bufs.len());
    for buf in bufs.iter() {
        bases.push(interp.memory.len());
        interp.slots.live.insert(interp.memory.len(), buf.len());
        interp
            .memory
            .extend(buf.iter().map(|&v| Value::Float(widen_buffer_f32(v))));
    }
    let args: Vec<Value> = bases.iter().map(|&b| Value::Ptr(b)).collect();
    let ran = interp.run_function(func, &args);
    // What the kernel printed, also when it stopped on a failure (see `run_kernel_f32_with_output`).
    *printed = std::mem::take(&mut interp.stdout);
    ran?;
    // Copy the final contents back out (captures both outputs and in-place mutation).
    for (buf, &base) in bufs.iter_mut().zip(bases.iter()) {
        for (i, slot) in buf.iter_mut().enumerate() {
            *slot = match interp.memory[base + i] {
                Value::Float(f) => narrow_buffer_f32(f),
                Value::Int(n) => n as f32,
                value => value.kernel_float("kernel f32 copy-back")? as f32,
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
        memory: null_reserved_memory(),
        stdout: Vec::new(),
        scratch_arena: ScratchSlots::default(),
        slots: SlotReuse::tracked(),
        frames: Vec::new(),
        call_args: Vec::new(),
        scratch: Vec::with_capacity(8),
        vecs: Vec::new(),
        #[cfg(test)]
        vector_peak: 0,
        data_addrs: HashMap::new(),
        function_indices: None,
        static_indices: None,
        accel: None,
        depth: 0,
    };
    let mut bases = Vec::with_capacity(bufs.len());
    for buf in bufs.iter() {
        bases.push(interp.memory.len());
        interp.slots.live.insert(interp.memory.len(), buf.len());
        interp.memory.extend(buf.iter().map(|&v| Value::Float(v)));
    }
    let args: Vec<Value> = bases.iter().map(|&b| Value::Ptr(b)).collect();
    interp.run_function(func, &args)?;
    for (buf, &base) in bufs.iter_mut().zip(bases.iter()) {
        for (i, slot) in buf.iter_mut().enumerate() {
            *slot = interp.memory[base + i].kernel_float("kernel f64 copy-back")?;
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
        memory: null_reserved_memory(),
        stdout: Vec::new(),
        scratch_arena: ScratchSlots::default(),
        slots: SlotReuse::tracked(),
        frames: Vec::new(),
        call_args: Vec::new(),
        scratch: Vec::with_capacity(8),
        vecs: Vec::new(),
        #[cfg(test)]
        vector_peak: 0,
        data_addrs: HashMap::new(),
        function_indices: None,
        static_indices: None,
        accel: None,
        depth: 0,
    };
    // u8 zero-extends, i8 sign-extends — the `as i128` casts do exactly that, matching the
    // kernel.
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
    // Three allocations, one per buffer.
    for (base, len) in [(a_base, a.len()), (b_base, b.len()), (c_base, c.len())] {
        interp.slots.live.insert(base, len);
    }
    let args = [Value::Ptr(a_base), Value::Ptr(b_base), Value::Ptr(c_base)];
    interp.run_function(func, &args)?;
    for (i, slot) in c.iter_mut().enumerate() {
        *slot = match interp.memory[c_base + i] {
            Value::Int(n) => n as i32,
            Value::Float(f) => f as i32,
            _ => 0,
        };
    }
    Ok(())
}

/// The interpreter's model of the scratch allocator (`scratch_<T>` / `scratch_mark` /
/// `scratch_release`; the contract and the native arena are in `wukong_runtime::scratch`). The
/// mark is the number of live scratch allocations — the same count native reports — and a
/// released region becomes reusable by the next allocation made at its depth.
#[derive(Default)]
struct ScratchSlots {
    /// One `(base slot, capacity in slots, length in slots)` per live scratch allocation, oldest
    /// first. The length is what [`SlotReuse::live`] holds as the allocation.
    live: Vec<(usize, usize, usize)>,
    /// `free[d]`: the region most recently released from depth `d`, if not yet reused.
    free: Vec<Option<(usize, usize)>>,
}

/// How the run's slot memory ([`Interp::memory`]) is reused — the oracle's counterpart of native
/// stack frames and of `free`. Without it every call's allocas and every freed heap block stayed
/// in `memory` for the rest of the run, so a decode loop that calls a layer function per token, or
/// allocates and frees per step, grew the oracle's memory without bound.
///
/// * **Frames.** An `Op::Alloca` pushes its slots at the end of `memory`. When a frame returns its
///   allocas are dead — an aggregate returns through a caller-allocated sret buffer and an array
///   parameter is the caller's storage, so only a dangling pointer (undefined natively) could still
///   reach them — and `memory` is cut back to the frame's entry length, but never below
///   `persist_floor`: the end of the last growth that was *not* an alloca (a heap or scratch
///   allocation, a materialized string, a kernel-entry buffer), which must outlive the frame that
///   made it. Such growth is detected rather than enumerated: between two alloca/release points
///   `memory` only grows, so a length that differs from `stack_end` there is persistent growth. An
///   alloca below such growth is never released, since no release goes below the floor —
///   conservative, never unsound. Re-pushed alloca slots get
///   their typed defaults again, exactly as fresh ones do.
/// * **`free`.** A `wukong_rt_alloc` block is recorded with its capacity; `free` moves it to a
///   size-bucketed free list, and the next allocation that fits a freed block (the smallest
///   capacity at least the request and at most twice it) takes it back, re-zeroed with the
///   requested element type's zero — what native's calloc'd bytes decode to. A pointer that is not
///   a live block (a double free, a stack or sub-slice, null) is ignored, as before; a use after
///   free reads whatever the block holds now, undefined natively and outside the differential
///   contract.
#[derive(Default)]
struct SlotReuse {
    /// `memory.len()` as the last alloca or frame release left it.
    stack_end: usize,
    /// Slots below this survive every frame release.
    persist_floor: usize,
    /// Live `wukong_rt_alloc` blocks: base slot -> capacity in slots.
    heap_live: HashMap<usize, usize>,
    /// Freed blocks: capacity -> base slots, most recently freed last.
    heap_free: BTreeMap<usize, Vec<usize>>,
    /// Every live allocation — alloca, heap, scratch, static, kernel-entry buffer — that a kernel
    /// operand must stay inside (`marshal::LiveBlocks`). Kept in step with the reuse above: a frame
    /// release drops the released allocas, `free` the heap block, `scratch_release` the region.
    live: LiveBlocks,
}

impl SlotReuse {
    /// The reuse state of a run whose kernel operands are checked against their allocations —
    /// every public entry point's. (`SlotReuse::default()`, untracked, is the hand-built unit-test
    /// interpreter's, whose memory is assigned directly and has no allocations to record.)
    fn tracked() -> Self {
        SlotReuse {
            live: LiveBlocks::tracking(),
            ..SlotReuse::default()
        }
    }

    /// Record any growth of `memory` (now `len` slots) since the last alloca or release as
    /// persistent.
    fn note_growth(&mut self, len: usize) {
        if len != self.stack_end {
            self.persist_floor = self.persist_floor.max(len);
            self.stack_end = len;
        }
    }

    /// The length `memory` (now `len` slots) is cut back to when a frame entered at `mark` returns.
    fn release_to(&mut self, mark: usize, len: usize) -> usize {
        self.note_growth(len);
        let keep = mark.max(self.persist_floor).min(len);
        self.stack_end = keep;
        self.live.release_from(keep);
        keep
    }

    /// Take a freed block that holds `count` slots without wasting more than half of itself.
    fn take_free(&mut self, count: usize) -> Option<(usize, usize)> {
        let (&cap, bases) = self.heap_free.range_mut(count..).next()?;
        if cap > count.saturating_mul(2) {
            return None;
        }
        let base = bases.pop()?;
        if bases.is_empty() {
            self.heap_free.remove(&cap);
        }
        Some((base, cap))
    }

    /// `free(base)`: a live block becomes reusable (and no longer a live allocation); anything
    /// else is ignored.
    fn free(&mut self, base: usize) {
        if let Some(cap) = self.heap_live.remove(&base) {
            self.heap_free.entry(cap).or_default().push(base);
            self.live.remove(base);
        }
    }
}

struct Interp<'a, 'k> {
    program: &'a Program,
    interner: &'a Interner,
    memory: Vec<Value>,
    stdout: Vec<u8>,
    /// The scratch allocations of this run (see [`ScratchSlots`]).
    scratch_arena: ScratchSlots,
    /// Frame and `free` reuse of `memory` (see [`SlotReuse`]).
    slots: SlotReuse,
    /// Recycled register files, one per active call depth. Pooling them keeps recursion and
    /// call-heavy code from allocating a fresh vector on every call.
    frames: Vec<Vec<Value>>,
    /// Argument snapshots available for reuse. Each active call owns its buffer until the
    /// callee returns, so nested calls cannot overwrite an outer call's arguments.
    call_args: Vec<Vec<Value>>,
    /// Scratch buffer reused when passing block-parameter arguments across an edge.
    scratch: Vec<Value>,
    /// Immutable scalar-lane vectors, addressed by handles that may alias across registers.
    /// Each active call owns a suffix; block boundaries compact that suffix and remap its registers.
    /// Caller handles stay below the callee watermark. Nested vector lanes are rejected so handles
    /// cannot escape through scalar memory or through another vector's lanes.
    vecs: Vec<Vec<Value>>,
    #[cfg(test)]
    vector_peak: usize,
    /// Optional accelerator the recognized kernel calls offload to (the GPU backend). `None` for the
    /// reference oracle and every plain run, so their numerics are untouched.
    accel: Option<&'k mut (dyn Accelerator + 'k)>,
    /// Materialized addresses of `Program::statics` string blobs, keyed by the blob's symbol. An
    /// `Op::GlobalAddr(sym)` materializes the bytes into persistent `memory` **once** and caches the
    /// base here, so every use of a (deduped) literal yields the *same* address — matching the
    /// native `.rodata` (a returned/threaded `*u8` stays valid, and `*u8` pointer equality agrees
    /// across backends). Persistent memory is never freed, so the address outlives its defining frame.
    data_addrs: HashMap<Symbol, usize>,
    /// Build name indices only when calls/function addresses or static data need them. The
    /// program is immutably borrowed for this run, so indices cannot become stale. Retain the
    /// first occurrence of duplicate names, matching the previous linear searches. Callees resolve
    /// through the dense [`CalleeTable`], one array index per call.
    function_indices: Option<CalleeTable>,
    static_indices: Option<HashMap<Symbol, usize>>,
    /// Active Wukong call depth: the frames [`Interp::run_function`] holds, bounded by
    /// [`max_call_depth`]. The 100M-step guard counts the blocks of a *single* frame and can never
    /// fire on recursion; this is the counter that does.
    depth: usize,
}

/// Default Wukong call depth at which the interpreter gives up (the
/// `WUKONG_INTERP_MAX_DEPTH` environment variable overrides it; see [`max_call_depth`]).
///
/// A Wukong call does not recurse on the host stack: [`Interp::run_function`] keeps an explicit
/// frame stack (CPython 3.11's "Python calls do not consume C stack", Lua's and wasm3's heap
/// frames), so a frame costs its heap register file and a few words — never host stack — and the
/// limit is a language-level number that is the same in a debug and a release `wukongc`, on every
/// host. It replaced two constants (40_000 debug / 300_000 release) that were bisected against a
/// 512 MiB host-stack reservation and silently went stale whenever a host frame changed size.
///
/// The value is the old release ceiling, so no depth that ran before is refused now; in a debug
/// build it raises the ceiling 7.5x. It clears the depth the differential gate exercises
/// (`wukong_codegen_cranelift`'s `differential_deep_recursion` runs `sum(30000)`), and an unbounded
/// recursion reaches it after a few tens of MiB of frames rather than an uncatchable host overflow.
const MAX_CALL_DEPTH: usize = 300_000;

/// The call-depth ceiling for one run: `WUKONG_INTERP_MAX_DEPTH` when it holds a positive integer,
/// [`MAX_CALL_DEPTH`] otherwise. Read once per [`Interp::run_function`] (once per run), never per
/// call. Like `sys.setrecursionlimit` or Wasmtime's `max_wasm_stack`, it is configuration, not a
/// property of the host.
fn max_call_depth() -> usize {
    std::env::var("WUKONG_INTERP_MAX_DEPTH")
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .filter(|&d| d > 0)
        .unwrap_or(MAX_CALL_DEPTH)
}

/// One Wukong call on [`Interp::run_function`]'s explicit frame stack.
struct Frame<'a> {
    func: &'a Function,
    /// `func`'s index among the program's functions (`None` only for a test harness's function).
    index: Option<usize>,
    /// The SSA register file, recycled through [`Interp::frames`].
    regs: Vec<Value>,
    /// The block executing and the index of its next instruction. While the frame is suspended
    /// under a callee, `ip` indexes the `Op::Call` waiting for the callee's result.
    block: BlockId,
    ip: usize,
    /// First [`Interp::vecs`] handle this frame owns (its suffix of the SIMD arena).
    vec_base: usize,
    /// `memory.len()` at entry: the frame's allocas live at or above it (see [`SlotReuse`]).
    mem_mark: usize,
    /// Blocks this frame has entered, for the per-frame 100M-step guard.
    steps: u64,
    /// The frame runs a `wukong_parallel_for` body, whose call yields `Unit` whatever it returns.
    unit_result: bool,
}

/// Why [`Interp::exec_frame`] stopped running a frame.
enum Flow {
    /// It reached a call of a Wukong function — the program's function at this index (the
    /// arguments, and whether the call yields `Unit`).
    Call(usize, Vec<Value>, bool),
    /// It returned this value.
    Ret(Value),
}

/// What one `Op::Call` resolved to.
enum CallOutcome {
    /// A Wukong function (or a `wukong_parallel_for` body), by index, to push a frame for.
    Enter(usize, Vec<Value>, bool),
    /// An extern or intrinsic that has already run, and its result.
    Done(Value),
}

/// Normalize a result about to be written to its register. Integer results are masked to their
/// declared width: this gives correct two's-complement wrapping and keeps booleans (`i1`) as 0/1
/// (so e.g. `!true` is 0, not a sign-extended -2). Float results narrower than `f64` are rounded to
/// `f32` (`is_narrow_float`) so the interpreter matches the native backend bit-for-bit (`f32`
/// arithmetic must round at `f32`, not `f64`). A `bf16`/`f16` result is rounded to `f32` here too,
/// *not* to its own grid: sub-`f32` types live as their rounded `f32` and take the grid rounding at
/// `ConstFloat`, at every cast, and on load.
///
/// **`Load` is exempt from the integer mask**: memory is one *typed value* per scalar slot, so a
/// load returns the slot's value verbatim. A same-typed load (every load a well-typed program's own
/// reads produce) holds an already-masked value, so the exemption is a no-op there. It matters for
/// the compiler's byte-wise aggregate copies (`emit_copy_bytes` — whole-enum assignment): those move
/// each *slot* through an `I8`-typed load/store pair, and masking the load would truncate a wider
/// payload slot (an `i64` payload of 5000 read through the copy became -120: interp ≠ native).
/// Native moves real bytes and reassembles the full value, so the verbatim slot move is exactly
/// what matches it.
fn canonical_result(v: Value, rty: &MirType, op: &Op) -> Value {
    match v {
        Value::Int(i) if rty.is_int() && !matches!(op, Op::Load(..)) => Value::Int(mask(i, rty)),
        Value::Float(f) if is_narrow_float(rty) => Value::Float(f as f32 as f64),
        other => other,
    }
}

/// Marshal a recognized-kernel **extent** argument (`rows`, `cols`, `m`, `k`, `n`, `t`, `h`, `s`,
/// `d`, `half`, `ncoeff`, ...) into a slot count, bailing out of the arm with the kernel's
/// documented no-op when the extent is non-positive.
///
/// A non-positive extent is reachable from ordinary source: these shapes are loop bounds, and
/// `for i in 0..n` with `n <= 0` runs zero iterations — a well-defined no-op in the language — so
/// the *recognized* form of the same nest must be a no-op too. Every `wukong_runtime` kernel
/// implements exactly that, opening with `if rows <= 0 || cols <= 0 { return; }` (`norm.rs:769`,
/// `gemm.rs:638`, `attention.rs:44`, `rowarg.rs:154`, `velem.rs:192`, ...), so `--backend=native`
/// completes and leaves the output buffer untouched.
///
/// Reading the argument with a bare `as usize` instead wrapped a negative extent to ~`usize::MAX`,
/// and the `Vec::with_capacity(rows * cols)` marshalling that follows aborted the whole process
/// with a raw Rust `capacity overflow` panic — an uncatchable abort out of the backend that is
/// supposed to be the semantic oracle, on input the native backend runs cleanly.
///
/// Bailing on a **zero** extent as well is deliberate, and is likewise what the kernels do: e.g.
/// `wukong_colargmax_i32` with `rows == 0` writes nothing, whereas marshalling it here would have
/// written `cols` zero indices into the output.
///
/// That rule matches the KERNEL, and it matches the SOURCE only because of what the compiler sends
/// here. A recognized REDUCTION whose reduction extent is zero is not a no-op in the source: the
/// nest still writes its seed or finalizer (`out[j] = 0.0` for a column sum over zero rows, the
/// zeroed `C` of a GEMM over `K == 0`). `wukong_mir_build`'s `emit_nonempty_reduction` therefore
/// dispatches the kernel only for a positive reduction extent and lowers the source loop
/// otherwise, for every reduction family (GEMM and its fused/lowp/int8 forms, GEMV, GEVM, the
/// column, row and arg reductions, the losses, log-sum-exp). A zero extent that reaches an arm here
/// is therefore one whose source loop really does nothing at zero trips.
///
/// Reductions that *return* a value (`wukong_sreduce_f32`, `wukong_argreduce_f32`, the bf16/f16
/// dot/sum/reduce family) deliberately do **not** use this: they clamp the count with `.max(0)` and
/// let the kernel supply its own empty-input identity (`reduce.rs:234` returns 0 / -inf / -1 for
/// `n <= 0`), which is exactly the value native computes for a negative `n`.
macro_rules! dim {
    ($v:expr) => {
        match $v.as_int() {
            e if e <= 0 => return Ok(Value::Unit),
            e => e as usize,
        }
    };
}

impl<'a, 'k> Interp<'a, 'k> {
    /// Is `slot` the reserved null pointer of this run? See [`NULL_SLOT`] / [`NULL_SENTINEL`].
    fn is_null(&self, slot: usize) -> bool {
        slot == NULL_SLOT && self.memory.first() == Some(&NULL_SENTINEL)
    }

    /// This run's callee table, built on first use (see [`CalleeTable`]).
    fn callee_table(&mut self) -> &mut CalleeTable {
        let program = self.program;
        self.function_indices
            .get_or_insert_with(|| CalleeTable::new(program))
    }

    /// What a call of `symbol` reaches, resolved once per run (see [`CalleeTable`]).
    fn callee(&mut self, symbol: Symbol) -> Callee {
        let interner = self.interner;
        self.callee_table()
            .resolve(symbol, || interner.resolve(symbol))
    }

    fn function_index(&mut self, symbol: Symbol) -> Option<usize> {
        match self.callee(symbol) {
            Callee::Func(index) => Some(index),
            Callee::Extern(_) | Callee::Intrinsic(_) | Callee::Unknown => None,
        }
    }

    /// Run `func` on `args` to completion and return its result.
    ///
    /// **Stackless.** A Wukong call does not recurse on the host stack: the frames live on an
    /// explicit stack (`suspended` plus the running `frame`), [`Interp::exec_frame`] runs the top
    /// frame until it calls or returns, and a call pushes a frame and keeps going in this loop. The
    /// host stack depth of a run is therefore the same for `sum(10)` and `sum(299999)`, which is what
    /// lets every entry point run on its caller's thread and makes the depth ceiling
    /// ([`max_call_depth`]) a language-level number instead of a host-stack measurement.
    ///
    /// On an error every frame this invocation entered is unwound exactly as a returning call
    /// would leave it (SIMD arena suffix, register pool, depth), so a reused `Interp` is clean.
    fn run_function(&mut self, func: &'a Function, args: &[Value]) -> Result<Value, String> {
        let limit = max_call_depth();
        let program: &'a Program = self.program;
        // Once per invocation: the entry function's index (calls carry their callee's).
        let index = program.funcs.iter().position(|f| std::ptr::eq(f, func));
        let mut suspended: Vec<Frame<'a>> = Vec::new();
        let mut frame = self.enter_frame(index, func, args, limit, false)?;
        loop {
            let flow = match self.exec_frame(&mut frame) {
                Ok(flow) => flow,
                Err(e) => return self.unwind(frame, &mut suspended, e),
            };
            match flow {
                Flow::Call(index, mut argv, unit_result) => {
                    let callee = &program.funcs[index];
                    let entered = self.enter_frame(Some(index), callee, &argv, limit, unit_result);
                    argv.clear();
                    self.call_args.push(argv);
                    match entered {
                        Ok(callee_frame) => {
                            suspended.push(std::mem::replace(&mut frame, callee_frame));
                        }
                        Err(e) => return self.unwind(frame, &mut suspended, e),
                    }
                }
                Flow::Ret(value) => {
                    let Some(caller) = suspended.pop() else {
                        return self.leave_frame(frame, Ok(value));
                    };
                    let done = std::mem::replace(&mut frame, caller);
                    let unit_result = done.unit_result;
                    match self.leave_frame(done, Ok(value)) {
                        Ok(v) => self.resume(&mut frame, if unit_result { Value::Unit } else { v }),
                        Err(e) => return self.unwind(frame, &mut suspended, e),
                    }
                }
            }
        }
    }

    /// Push a frame for a call of `func` (the program's function at `index`, when it is one):
    /// check the depth ceiling, take a recycled register file (or a fresh one) and size it for
    /// this function.
    ///
    /// **A recycled register file is not refilled.** Every call used to `clear` it and `resize` it
    /// back with `Unit` — a write of every SSA register (32 bytes each) per call, however few the
    /// callee touches. That fill is unobservable on verified MIR: the verifier requires every use
    /// to be dominated by its definition (`wukong_mir::verify`, "not dominated by its definition"),
    /// and the driver verifies before every backend entry, so no instruction reads a register this
    /// frame has not written. The one reader that scans the WHOLE file is
    /// [`Interp::compact_vectors`], which keeps every SIMD handle it finds alive — a stale handle
    /// left by an earlier call would be kept, or rejected as out of range. So a pooled file never
    /// holds one: handles only ever live in vector-typed registers (every op that yields one has a
    /// vector result type), and [`Interp::leave_frame`] resets exactly those, from a per-function
    /// list ([`CalleeTable::vector_registers`]) — none at all for a scalar function. Here the file
    /// only grows (filling the new tail with `Unit`) or shrinks to this function's size. A
    /// function that is not one of the program's (a test harness's) gets the old full fill.
    fn enter_frame(
        &mut self,
        index: Option<usize>,
        func: &'a Function,
        args: &[Value],
        limit: usize,
        unit_result: bool,
    ) -> Result<Frame<'a>, String> {
        if self.depth >= limit {
            return Err(
                "interpreter call-depth limit exceeded (likely unbounded recursion)".into(),
            );
        }
        self.depth += 1;
        let mut regs = self.frames.pop().unwrap_or_default();
        let size = func.value_types.len();
        if index.is_none() {
            regs.clear();
        }
        if regs.len() > size {
            regs.truncate(size);
        } else {
            regs.resize(size, Value::Unit);
        }
        for (p, &a) in func.params.iter().zip(args) {
            regs[p.0 as usize] = a;
        }
        self.slots.note_growth(self.memory.len());
        Ok(Frame {
            func,
            index,
            regs,
            block: func.entry,
            ip: 0,
            vec_base: self.vecs.len(),
            mem_mark: self.memory.len(),
            steps: 0,
            unit_result,
        })
    }

    /// Pop `frame`, which finished with `result`, and release what it owned.
    fn leave_frame(
        &mut self,
        frame: Frame<'a>,
        result: Result<Value, String>,
    ) -> Result<Value, String> {
        let Frame {
            func,
            index,
            mut regs,
            vec_base,
            mem_mark,
            ..
        } = frame;
        // A callee may return either a borrowed caller handle or one of its own vectors. Move
        // the latter out before reclaiming the suffix, then promote exactly one owned result.
        let mut returned_lanes = None;
        let result = match result {
            Ok(Value::VecRef(index)) if index >= vec_base => match self.vecs.get_mut(index) {
                Some(lanes) => {
                    returned_lanes = Some(std::mem::take(lanes));
                    Ok(Value::Unit)
                }
                None => Err("invalid SIMD vector handle on return".into()),
            },
            other => other,
        };
        self.vecs.truncate(vec_base);
        self.scratch.clear();
        // Pool the register file without its SIMD handles, and otherwise as it is (see
        // `enter_frame`): only the vector-typed registers can hold one.
        match index {
            Some(index) => {
                for &r in self.callee_table().vector_registers(index, func) {
                    regs[r as usize] = Value::Unit;
                }
            }
            None => regs.clear(),
        }
        self.frames.push(regs);
        self.depth -= 1;
        // The frame's allocas are dead: cut `memory` back (see `SlotReuse`).
        let keep = self.slots.release_to(mem_mark, self.memory.len());
        self.memory.truncate(keep);
        match returned_lanes {
            Some(lanes) => self.push_vec(lanes),
            None => result,
        }
    }

    /// Unwind `frame` and every frame in `suspended`, innermost first, after `error`.
    fn unwind(
        &mut self,
        frame: Frame<'a>,
        suspended: &mut Vec<Frame<'a>>,
        error: String,
    ) -> Result<Value, String> {
        let mut result = self.leave_frame(frame, Err(error));
        while let Some(outer) = suspended.pop() {
            result = self.leave_frame(outer, result);
        }
        result
    }

    /// Continue `frame` after the callee its pending `Op::Call` entered returned `value`.
    fn resume(&mut self, frame: &mut Frame<'a>, value: Value) {
        let func = frame.func;
        let inst = &func.block(frame.block).insts[frame.ip];
        if let Some(r) = inst.result {
            frame.regs[r.0 as usize] = canonical_result(value, func.value_type(r), &inst.op);
        }
        frame.ip += 1;
    }

    /// Run `frame` from its current block and instruction until it calls a Wukong function
    /// ([`Flow::Call`], with `frame.ip` left on the call) or returns ([`Flow::Ret`]).
    fn exec_frame(&mut self, frame: &mut Frame<'a>) -> Result<Flow, String> {
        let func = frame.func;
        let mut remap = Vec::new();
        loop {
            let block = func.block(frame.block);
            if frame.ip == 0 {
                // Block entry is the safe point: no eval/call/edge argument snapshot is pending.
                // Keeping every register (including stale SSA slots) is conservative and bounded
                // by function size.
                self.compact_vectors(frame.vec_base, &mut frame.regs, &mut remap)?;
                self.scratch.clear();
                frame.steps += 1;
                if frame.steps > 100_000_000 {
                    return Err("interpreter step limit exceeded (likely an infinite loop)".into());
                }
            }
            while let Some(inst) = block.insts.get(frame.ip) {
                let v = match &inst.op {
                    Op::Call { func: callee, args } => {
                        match self.dispatch_call(func, *callee, args, &frame.regs)? {
                            CallOutcome::Enter(target, argv, unit_result) => {
                                return Ok(Flow::Call(target, argv, unit_result));
                            }
                            CallOutcome::Done(v) => v,
                        }
                    }
                    op => {
                        let rty = inst.result.map(|r| func.value_type(r));
                        self.eval(op, func, &frame.regs, rty)?
                    }
                };
                if let Some(r) = inst.result {
                    frame.regs[r.0 as usize] = canonical_result(v, func.value_type(r), &inst.op);
                }
                frame.ip += 1;
            }
            frame.ip = 0;
            let regs = &mut frame.regs;
            match &block.term {
                Terminator::Ret(None) => return Ok(Flow::Ret(Value::Unit)),
                Terminator::Ret(Some(v)) => return Ok(Flow::Ret(reg(regs, *v))),
                Terminator::Br { target, args } => {
                    pass_args(regs, &mut self.scratch, func.block(*target), args);
                    frame.block = *target;
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
                    frame.block = tgt;
                }
                // Exactly a `cmp.eq` chain in case order: the scrutinee's integer value (held
                // sign-extended, which is the canonical form every case value is stored in) against
                // each case, falling back to the default.
                Terminator::Switch {
                    scrutinee,
                    cases,
                    default,
                    default_args,
                } => {
                    let x = reg(regs, *scrutinee).as_int();
                    let (tgt, bargs) = match cases.iter().find(|c| c.value == x) {
                        Some(c) => (c.target, &c.args),
                        None => (*default, default_args),
                    };
                    pass_args(regs, &mut self.scratch, func.block(tgt), bargs);
                    frame.block = tgt;
                }
                Terminator::Unreachable => return Err("execution reached `unreachable`".into()),
            }
        }
    }

    /// Resolve one `Op::Call`. A Wukong callee — or the body of a `wukong_parallel_for`, which the
    /// interpreter runs as one sequential call over the whole range — comes back as
    /// [`CallOutcome::Enter`], so the frame loop pushes a frame instead of recursing on the host
    /// stack; an extern or an intrinsic runs here.
    fn dispatch_call(
        &mut self,
        func: &Function,
        callee: Symbol,
        args: &[ValueId],
        regs: &[Value],
    ) -> Result<CallOutcome, String> {
        let mut argv = self.call_args.pop().unwrap_or_default();
        argv.extend(args.iter().map(|a| reg(regs, *a)));
        let program: &'a Program = self.program;
        let result = match self.callee(callee) {
            Callee::Func(index) => {
                return Ok(CallOutcome::Enter(index, argv, false));
            }
            Callee::Extern(index) => call_extern(&program.externs[index], &argv, self.interner),
            Callee::Unknown => Err(format!(
                "call to unknown function or intrinsic `{}`",
                self.interner.resolve(callee)
            )),
            Callee::Intrinsic(id) => {
                if id == Intrinsic::ParallelFor {
                    match self.parallel_for_body(&argv) {
                        Ok((body, body_args)) => {
                            argv.clear();
                            argv.extend(body_args);
                            return Ok(CallOutcome::Enter(body, argv, true));
                        }
                        Err(e) => Err(e),
                    }
                } else {
                    // A float `print` picks its printer by the argument's MIR type, as native does
                    // (`classify_intrinsic`): an `f32` (or an `f16`/`bf16`, which lives in an `f32`)
                    // prints at its own precision. Every other print routes by name below.
                    let narrow_float = args.first().is_some_and(|a| {
                        matches!(
                            func.value_type(*a),
                            MirType::F32
                                | MirType::F16
                                | MirType::BF16
                                | MirType::F8E4M3
                                | MirType::F8E5M2
                        )
                    });
                    if narrow_float && matches!(id, Intrinsic::Print | Intrinsic::Println) {
                        self.print_f32(&argv)
                    } else {
                        self.call_intrinsic(id, &argv)
                    }
                }
            }
        };
        argv.clear();
        self.call_args.push(argv);
        result.map(CallOutcome::Done)
    }

    /// The body of a `wukong_parallel_for(n, body, env)` call and the arguments it runs on: the
    /// interpreter runs `body(0, n, env)`, the whole index range sequentially in one call. Since
    /// parallel-for bodies have no cross-iteration dependencies this is exactly the result the
    /// multi-threaded native runtime produces (so the two stay differential-equal).
    fn parallel_for_body(&self, args: &[Value]) -> Result<(usize, [Value; 3]), String> {
        let n = args.first().copied().unwrap_or(Value::Unit).as_int();
        let body = match args.get(1).copied() {
            Some(Value::Ptr(p)) if p >= FUNC_TAG => p - FUNC_TAG,
            _ => return Err("parallel_for body is not a function address".into()),
        };
        let env = args.get(2).copied().unwrap_or(Value::Ptr(0));
        if body >= self.program.funcs.len() {
            return Err("parallel_for body index out of range".into());
        }
        Ok((body, [Value::Int(0), Value::Int(n), env]))
    }

    /// Marshal one `Op::VecKernelCall` of a recipe whose lanes are `T` through slot memory: the
    /// interpreter half of the differential pair whose other half is `codegen_cranelift::avx2`.
    /// `stream_bases`/`scalars` are read from the two pointer tables, `count` is the vector part's
    /// element count. A reduction returns its horizontal fold (steps 1-5 of the canonical tree in
    /// `wukong_mir::vec_kernel`); an elementwise recipe writes its stores and returns `Unit`.
    ///
    /// Every extent is checked before anything is reserved or read: the two tables (every stream
    /// pointer, and the scalars up to the last one the recipe reads), and each stream the recipe
    /// loads or stores over the elements it touches (none, on a zero-trip call, so a base that
    /// call never dereferences is never checked), must lie in the arena and inside ONE live
    /// allocation (a diagnostic names a stream by its index as the operand). The table sizes and
    /// the whole-lane-group length are checked arithmetic and fallible reservations.
    fn run_vec_kernel<T: KernelLaneValue>(
        &mut self,
        kern: &VecKernel,
        ptrs_base: usize,
        scalars_base: usize,
        count: usize,
    ) -> Result<Value, String> {
        let streams = kern.streams as usize;
        let scalars = kern.scalars as usize;
        // Output-only streams need not contain initialized numeric values. Only recipe
        // loads read memory; scalar tails beyond count remain untouched and unchecked.
        let mut read_streams = kernel_scratch(streams, "veckernel streams")?;
        read_streams.resize(streams, false);
        let mut stored_streams = read_streams.clone();
        let mut read_scalars = kernel_scratch(scalars, "veckernel scalars")?;
        read_scalars.resize(scalars, false);
        for op in &kern.ops {
            if let VecOp::Splat { scalar } = op {
                *read_scalars
                    .get_mut(*scalar as usize)
                    .ok_or("veckernel scalar index out of bounds")? = true;
            }
            // `Store`/`StoreF16` and `Load`/`LoadF16`: an f16 stream is one slot per element like
            // any other; the recipe's evaluator rounds what it reads and writes.
            if let Some(stream) = op.stored_stream() {
                *stored_streams
                    .get_mut(stream as usize)
                    .ok_or("veckernel store stream out of bounds")? = true;
            }
            if let Some(stream) = op.loaded_stream() {
                if stream as usize >= streams {
                    return Err("veckernel load stream out of bounds".into());
                }
            }
        }
        // Resolve the stream base slots and the invariant scalars once, each table validated
        // whole first: every stream pointer is read, and the scalars up to the last one used.
        let ptr_slots = kernel_values(arena!(self), 0, ptrs_base, streams, 1, "veckernel ptrs")?;
        let mut stream_bases: Vec<usize> = kernel_scratch(streams, "veckernel ptrs")?;
        for &slot in ptr_slots {
            stream_bases.push(ptr(slot)?);
        }
        // Each stream's alias class: the first stream with the same base. Two streams may share a
        // base, because the optimizer's runtime alias guard admits two slices that are one buffer
        // when every access sits at one element (`wukong_opt::vectorize`'s `guard_allows_equal`).
        // They then name one element per lane, and the evaluator forwards a store to a later load
        // of any stream of its class (`VecKernel::eval_lane_aliased`), as the machine code does by
        // re-reading memory.
        let mut alias: Vec<u32> = kernel_scratch(streams, "veckernel streams")?;
        for &b in &stream_bases {
            let first = stream_bases.iter().position(|&c| c == b).unwrap_or(0);
            alias.push(first as u32);
        }
        for (index, op) in kern.ops.iter().enumerate() {
            if let Some(stream) = op.loaded_stream() {
                // The evaluator forwards a preceding store to this stream's class; that load never
                // consumes the original slot, which may be undefined.
                let class = alias[stream as usize];
                if !kern.ops[..index].iter().any(|previous| {
                    previous
                        .stored_stream()
                        .is_some_and(|written| alias[written as usize] == class)
                }) {
                    read_streams[stream as usize] = true;
                }
            }
        }
        let used_scalars = read_scalars
            .iter()
            .rposition(|&read| read)
            .map_or(0, |k| k + 1);
        let scalar_slots = kernel_values(
            arena!(self),
            0,
            scalars_base,
            used_scalars,
            1,
            "veckernel scalars",
        )?;
        let mut scalar_vals: Vec<T> = kernel_scratch(scalars, "veckernel scalars")?;
        for (k, &read) in read_scalars.iter().enumerate() {
            scalar_vals.push(if read {
                T::read(scalar_slots[k], "veckernel scalar")?
            } else {
                T::unused()
            });
        }
        if kern.reduce.is_some() {
            // Reduction kernel: fold every lane's addend and return the horizontal result in the
            // lane type (the caller combines the initial accumulator and folds the scalar tail).
            // Reads only — no stores. Same reassociation `eval_reduction` pins for both backends,
            // so this equals the native kernel's return bit-for-bit.
            let lanes = kern.lane.lanes() as usize;
            // Whole lane groups only: a partial group beyond `vlen` is the caller's scalar tail.
            let vlen = count
                .checked_div(lanes)
                .and_then(|groups| groups.checked_mul(lanes))
                .ok_or("veckernel reduction extent overflow")?;
            // Validate and convert every element the evaluator will read up front, so its
            // callbacks cannot fail; a partial lane group beyond `vlen` is never read.
            let mut columns: Vec<Vec<T>> = kernel_scratch(streams, "veckernel reduction")?;
            for (s, &b) in stream_bases.iter().enumerate() {
                // A zero-trip fold (`vlen == 0`) reads no stream, so it constrains no base: the
                // caller still offsets every base to the range start, which may lie past the end
                // of the arena (see the elementwise branch).
                if read_streams[s] && vlen > 0 {
                    let end = b
                        .checked_add(vlen)
                        .ok_or("veckernel reduction load out of bounds")?;
                    let values = self
                        .memory
                        .get(b..end)
                        .ok_or("veckernel reduction load out of bounds")?;
                    self.slots.live.check("veckernel stream", s, b, vlen)?;
                    let mut column = kernel_scratch(vlen, "veckernel reduction")?;
                    for value in values {
                        column.push(T::read(*value, "veckernel reduction")?);
                    }
                    columns.push(column);
                } else {
                    columns.push(Vec::new());
                }
            }
            let r = kern.eval_reduction(
                vlen,
                |s, e| columns[s as usize][e],
                |k| scalar_vals[k as usize],
            );
            Ok(r.write())
        } else {
            // Elementwise: one `eval_lane_aliased` per element — snapshot this lane's stream
            // inputs, run the recipe, write its stores back. Load-before-store within a lane keeps
            // in-place streams exact; distinct lanes never alias (the vectorizer's no-alias
            // precondition), so no hazard. Streams that share a BASE name one element per lane,
            // so the evaluator serves a load from the lane's latest earlier store to any stream of
            // its alias class (its own included) and reads the snapshot only when there is none:
            // memory, as the machine code reads it, decides every load. Every stream the recipe
            // loads or stores is checked over all `count` elements first, so a stream that would
            // run past its allocation is refused before the first store.
            //
            // A zero-trip call touches nothing and so checks nothing: `marshal_vec_kernel` passes
            // `n = 0` for a short or empty range but still offsets each base to the range START,
            // which can lie past the end of the arena (`for i in 100000..n` over an empty slice).
            // The native kernel does nothing there; refusing a base it never dereferences would
            // be an interp/native divergence on a valid program.
            if count == 0 {
                return Ok(Value::Unit);
            }
            for (s, &b) in stream_bases.iter().enumerate() {
                if read_streams[s] || stored_streams[s] {
                    let end = b
                        .checked_add(count)
                        .ok_or("veckernel stream out of bounds")?;
                    if end > self.memory.len() {
                        return Err("veckernel stream out of bounds".into());
                    }
                    self.slots.live.check("veckernel stream", s, b, count)?;
                }
            }
            let mut stores: Vec<(u32, T)> = Vec::new();
            let mut loads: Vec<T> = kernel_scratch(streams, "veckernel loads")?;
            let mut scratch = wukong_mir::vec_kernel::VecKernelScratch::<T>::default();
            for i in 0..count {
                loads.clear();
                for (s, &b) in stream_bases.iter().enumerate() {
                    loads.push(if read_streams[s] {
                        T::read(
                            *self
                                .memory
                                .get(b + i)
                                .ok_or("veckernel load out of bounds")?,
                            "veckernel load",
                        )?
                    } else {
                        T::unused()
                    });
                }
                stores.clear();
                kern.eval_lane_aliased(
                    &mut scratch,
                    &alias,
                    |s| loads[s as usize],
                    |k| scalar_vals[k as usize],
                    |s, v| stores.push((s, v)),
                );
                for &(s, v) in &stores {
                    let slot = stream_bases[s as usize] + i;
                    *self
                        .memory
                        .get_mut(slot)
                        .ok_or("veckernel store out of bounds")? = v.write();
                }
            }
            Ok(Value::Unit)
        }
    }

    fn eval(
        &mut self,
        op: &Op,
        func: &Function,
        regs: &[Value],
        rty: Option<&MirType>,
    ) -> Result<Value, String> {
        if let Some(ty @ MirType::Vec(..)) = rty {
            if matches!(
                op,
                Op::Bin(..)
                    | Op::Cmp(..)
                    | Op::Neg(..)
                    | Op::Not(..)
                    | Op::Cast(..)
                    | Op::Select(..)
                    | Op::Splat(..)
                    | Op::Iota(..)
                    | Op::Widen { .. }
                    | Op::Narrow { .. }
                    | Op::Shuffle { .. }
                    | Op::InsertLane(..)
                    | Op::Fma(..)
                    | Op::Sqrt(..)
                    | Op::Round(..)
            ) {
                return self.eval_vector(op, func, regs, ty);
            }
        }
        Ok(match op {
            Op::ConstInt(v, ty) => Value::Int(mask(*v, ty)),
            Op::ConstFloat(v, ty) => Value::Float(match ty {
                // A bf16/f16 const IS its grid-rounded value (see the native ConstFloat lowering) —
                // round at materialization so optimizer value-forwarding can't drop the rounding and
                // make -O2 disagree with -O0. Same `half`-crate path used at every bf16/f16 store/cast.
                MirType::BF16 => wukong_runtime::round_bf16(*v as f32) as f64,
                MirType::F16 => wukong_runtime::round_f16(*v as f32) as f64,
                // An f8 const is the literal rounded ONCE onto the grid (from its f64 value).
                MirType::F8E4M3 | MirType::F8E5M2 => round_f8(*v, ty),
                _ => *v,
            }),
            // Vector arithmetic is lane-wise, each lane rounded to the lane type (so `<n x f32>`
            // ops round at f32, matching the native backend). Scalar bins go the fast path.
            Op::Bin(b, l, r) => apply_bin(*b, reg(regs, *l), reg(regs, *r), rty),
            Op::Cmp(c, l, r) => Value::Int(apply_cmp(*c, reg(regs, *l), reg(regs, *r)) as i128),
            Op::Neg(v) => match reg(regs, *v) {
                Value::Float(f) => Value::Float(-f),
                other => Value::Int(other.as_int().wrapping_neg()),
            },
            Op::Not(v) => match reg(regs, *v) {
                Value::Int(i) => Value::Int(!i),
                other => Value::Int(!other.as_int()),
            },
            Op::Cast(kind, v, to) => apply_cast(*kind, reg(regs, *v), func.value_type(*v), to),
            Op::Select(c, a, b) => {
                if reg(regs, *c).truthy() {
                    reg(regs, *a)
                } else {
                    reg(regs, *b)
                }
            }
            Op::Alloca { ty, .. } => {
                self.slots.note_growth(self.memory.len());
                let idx = self.memory.len();
                // Reserve every leaf slot of `ty` (recursing into nested arrays so an array of
                // structs/tuples is fully sized, not one slot per element); the result points at the
                // first. `gep` then computes `base + index * slot_count(elem)` into this run. The
                // frame that executed it releases the slots when it returns (`SlotReuse`). The whole
                // alloca is ONE allocation, however its aggregate nests: a kernel operand may span
                // its fields or elements, never run past its end.
                push_defaults(ty, &mut self.memory);
                self.slots.stack_end = self.memory.len();
                self.slots.live.insert(idx, self.memory.len() - idx);
                Value::Ptr(idx)
            }
            // A vector load gathers `n` contiguous scalar slots (memory stays scalar); a scalar
            // load reads one. Both `gep` to the element address first.
            Op::Load(p, ty) => {
                let idx = ptr(reg(regs, *p))?;
                if self.is_null(idx) {
                    return Err("null pointer dereference (load)".into());
                }
                if let MirType::Vec(lane, n) = ty {
                    let mut lanes = Vec::new();
                    lanes
                        .try_reserve_exact(*n as usize)
                        .map_err(|_| "SIMD lane allocation failed")?;
                    for i in 0..*n as usize {
                        let value = *self
                            .memory
                            .get(idx + i)
                            .ok_or("vector load out of bounds")?;
                        // I1 occupies one byte natively but represents only its low bit.
                        // Normalize before Select/branches observe a noncanonical stored byte.
                        // A bf16/f16 lane rounds to its storage grid on read, exactly as the
                        // scalar load below does (the native backend stores the 16 bits).
                        lanes.push(match (&**lane, value) {
                            (MirType::I1, _) => mask_lane(value, lane),
                            (MirType::BF16, Value::Float(f)) => {
                                Value::Float(wukong_runtime::round_bf16(f as f32) as f64)
                            }
                            (MirType::F16, Value::Float(f)) => {
                                Value::Float(wukong_runtime::round_f16(f as f32) as f64)
                            }
                            _ => value,
                        });
                    }
                    self.push_vec(lanes)?
                } else {
                    let v = *self.memory.get(idx).ok_or("load out of bounds")?;
                    // `[bf16; N]`/`[f16; N]` storage holds reduced precision: round on read (the native
                    // backend stores the rounded 16 bits, so a load there observes the same value).
                    match (ty, v) {
                        // Keep the general integer-load exemption used by bytewise aggregate
                        // copies, but match native's unconditional I1 result normalization.
                        (MirType::I1, Value::Int(i)) => Value::Int(i & 1),
                        (MirType::BF16, Value::Float(f)) => {
                            Value::Float(wukong_runtime::round_bf16(f as f32) as f64)
                        }
                        (MirType::F16, Value::Float(f)) => {
                            Value::Float(wukong_runtime::round_f16(f as f32) as f64)
                        }
                        // `[f8e4m3; N]` storage holds a code: read the value it stands for.
                        (MirType::F8E4M3 | MirType::F8E5M2, Value::Float(f)) => {
                            Value::Float(round_f8(f, ty))
                        }
                        _ => v,
                    }
                }
            }
            Op::Store { ptr: p, value } => {
                let idx = ptr(reg(regs, *p))?;
                if self.is_null(idx) {
                    return Err("null pointer dereference (store)".into());
                }
                match reg(regs, *value) {
                    // A vector store scatters its lanes across contiguous scalar slots.
                    Value::VecRef(vi) => {
                        let lanes = self
                            .vecs
                            .get(vi)
                            .ok_or("invalid SIMD vector handle in store")?;
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
            // An integer read-modify-write, wrapping at the value's width. The oracle runs one
            // thread, so "atomic" is free here; what it pins down is the arithmetic every native
            // interleaving must reproduce (integer addition mod 2^n commutes, so they all do).
            Op::AtomicAdd { ptr: p, value } => {
                let idx = ptr(reg(regs, *p))?;
                if self.is_null(idx) {
                    return Err("null pointer dereference (atomic add)".into());
                }
                let add = reg(regs, *value).as_int();
                let slot = self.memory.get_mut(idx).ok_or("atomic add out of bounds")?;
                *slot = Value::Int(mask(
                    slot.as_int().wrapping_add(add),
                    func.value_type(*value),
                ));
                Value::Unit
            }
            Op::Gep {
                ptr: p,
                index,
                elem,
            } => {
                let base = ptr(reg(regs, *p))?;
                let off = reg(regs, *index).as_int();
                // Scale the index by the element's slot footprint so an aggregate-element array
                // (`[Struct; N]`) strides one whole element per index, matching native's
                // `index * size_of(elem)`. `slot_count` is 1 for scalar/byte elements, so scalar
                // arrays and struct/tuple byte-offset field GEPs (`elem = I8`) are unchanged.
                Value::Ptr((base as i128 + off * slot_count(elem) as i128) as usize)
            }
            Op::Barrier => Value::Unit,
            // Calls never reach `eval`: `exec_frame` dispatches every `Op::Call` (`dispatch_call`)
            // so a Wukong callee gets a frame on the explicit stack rather than a host recursion.
            Op::Call { .. } => {
                return Err("calls are dispatched by the frame loop, not `eval`".into())
            }
            // A synthesized 256-bit AVX2 vector kernel (the general vectorizer's output). The native
            // backend executes hand-assembled AVX2; the oracle marshals the same `VecKernel` recipe
            // element-by-element through slot memory so the two agree bit-for-bit. `ptrs` points at
            // `streams` slots each holding a stream's (already `start`-offset) base pointer, `scalars`
            // at `scalars` invariant values of the recipe's lane type, and `n` is the
            // multiple-of-lanes element count (the caller runs the scalar tail). The recipe is a
            // function-local list borrowed from `func` (a shared reference), independent of the
            // `&mut self` the marshalling below needs.
            Op::VecKernelCall {
                kernel,
                ptrs,
                scalars,
                n,
            } => {
                let kern = func
                    .vec_kernels
                    .get(*kernel as usize)
                    .ok_or("veckernel: unknown kernel index")?;
                let ptrs_base = ptr(reg(regs, *ptrs))?;
                let scalars_base = ptr(reg(regs, *scalars))?;
                // A count past the address space is no element count (not a truncated one).
                let count = usize::try_from(reg(regs, *n).as_int().max(0))
                    .map_err(|_| "veckernel element count overflow")?;
                match kern.lane {
                    VecLane::F32 => {
                        self.run_vec_kernel::<f32>(kern, ptrs_base, scalars_base, count)?
                    }
                    VecLane::F64 => {
                        self.run_vec_kernel::<f64>(kern, ptrs_base, scalars_base, count)?
                    }
                    VecLane::I32 => {
                        self.run_vec_kernel::<i32>(kern, ptrs_base, scalars_base, count)?
                    }
                    VecLane::I64 => {
                        self.run_vec_kernel::<i64>(kern, ptrs_base, scalars_base, count)?
                    }
                }
            }
            // A function address: a pointer the interpreter tags with the function's index so the
            // `parallel_for` intrinsic can call it back. (The native backend uses a real address.)
            Op::FuncAddr(sym) => {
                let idx = self
                    .function_index(*sym)
                    .ok_or("func_addr of unknown function")?;
                Value::Ptr(FUNC_TAG + idx)
            }
            // The read-only address of a static-data blob (a string literal): materialize its bytes
            // into persistent memory **once**, caching the base by symbol so every use of a deduped
            // literal yields the same address (matching native `.rodata`; `*u8` pointer equality then
            // agrees across backends). Persistent memory is never freed, so a returned/threaded `*u8`
            // stays valid after its defining frame is gone — closing the native stack-dangle divergence.
            Op::GlobalAddr(sym) => {
                if let Some(&base) = self.data_addrs.get(sym) {
                    Value::Ptr(base)
                } else {
                    let index = *self
                        .static_indices
                        .get_or_insert_with(|| {
                            symbol_indices(self.program.statics.iter().map(|s| s.name))
                        })
                        .get(sym)
                        .ok_or("global_addr of unknown static data")?;
                    let bytes = &self.program.statics[index].bytes;
                    let base = self.memory.len();
                    // Each byte in the representation every other byte slot holds — the
                    // sign-extended `i8` an `I8` store or a runtime byte write leaves — because a
                    // load is verbatim (`canonical_result`). Zero-extended here, the bytes of a
                    // string literal compared unequal to the same bytes anywhere else on this
                    // backend only: `s[0] == t[0]` for `s: str = "\u{e9}"` and `t` viewing
                    // `[0xC3, 0xA9]` was false here and true natively.
                    self.memory
                        .extend(bytes.iter().map(|&b| Value::Int(i128::from(b as i8))));
                    self.slots.live.insert(base, bytes.len());
                    self.data_addrs.insert(*sym, base);
                    Value::Ptr(base)
                }
            }
            // Broadcast a scalar to every lane.
            Op::Splat(_) => return Err("SIMD operation requires a vector result".into()),
            // One lane out of a vector, by constant index — the inverse of `Splat`, and the only
            // way back to scalar form. The lane count comes from the *operand's* type, not the
            // result's (the result is the lane type). Both the index and the vector-ness of the
            // operand are checked by the MIR verifier, so a bad index here is a lowering bug, not
            // a program error; `vec_lanes` still reports it rather than panicking.
            Op::ExtractLane(v, k) => {
                let n = match func.value_type(*v) {
                    MirType::Vec(_, n) => *n as usize,
                    other => {
                        return Err(format!(
                            "extractlane operand is {}, not a vector",
                            other.display()
                        ))
                    }
                };
                let lanes = self.vec_lanes(reg(regs, *v), n)?;
                *lanes
                    .get(*k as usize)
                    .ok_or("extractlane index out of range")?
            }
            // The lane-index ramp `<0, 1, .., n-1>`. The lane values are stored untruncated (an
            // `i128` per lane, exactly as `Op::ConstInt` does) — every consumer of a narrow integer
            // lane masks to its own width, and `n` is at most 16, so no lane can exceed any lane
            // type this op accepts.
            Op::Iota(_) => return Err("SIMD operation requires a vector result".into()),
            // Like `Splat`/`Iota`, the lane-moving ops always produce a vector (`eval_vector`).
            Op::Widen { .. } | Op::Narrow { .. } | Op::Shuffle { .. } | Op::InsertLane(..) => {
                return Err("SIMD operation requires a vector result".into())
            }
            // Fused multiply-add `a*b + c`, single-rounded via `mul_add` so it stays bit-identical
            // to the native `fma`. Lane-wise for vectors, each lane rounded to its lane type.
            Op::Fma(a, b, c) => apply_fma(reg(regs, *a), reg(regs, *b), reg(regs, *c), rty),
            // Square root, lane-wise for vectors; each lane rounded to its lane type so an `f32`
            // sqrt stays bit-identical to the native `fsqrt`.
            Op::Sqrt(v) => apply_sqrt(reg(regs, *v), rty),
            Op::Round(mode, v) => apply_round(*mode, reg(regs, *v), rty),
            // Bit counting / byte reversal on the operand's own-width bit pattern (the verifier
            // admits scalar i8..i128 only). `IntUnOp::eval_wide` is the definition the hardware
            // instruction computes; `exec` re-canonicalizes the result to the type's width.
            Op::IntUnary(op, v) => {
                let bits = rty.map(int_bits).unwrap_or(64);
                let x = uval(reg(regs, *v).as_int(), bits);
                Value::Int(op.eval_wide(x, bits) as i128)
            }
        })
    }

    /// Keep SIMD lane temporaries off scalar call frames: the interpreter's recursion ceiling
    /// also protects programs that never evaluate a vector operation.
    fn eval_vector(
        &mut self,
        op: &Op,
        func: &Function,
        regs: &[Value],
        ty: &MirType,
    ) -> Result<Value, String> {
        let MirType::Vec(lane, n) = ty else {
            return Err("SIMD operation requires a vector result".into());
        };
        Ok(match op {
            Op::Bin(b, l, r) => {
                let av = self.vec_lanes(reg(regs, *l), *n as usize)?;
                let lane = (**lane).clone();
                let lanes = if func.value_type(*r).is_vector() {
                    let bv = self.vec_lanes(reg(regs, *r), *n as usize)?;
                    collect_lanes(
                        (0..*n as usize)
                            .map(|i| mask_lane(apply_bin(*b, av[i], bv[i], Some(&lane)), &lane)),
                    )?
                } else {
                    // A vector shifted by one scalar amount (`Op::Bin`'s one operand-shape
                    // exception, which the verifier admits for shifts only): every lane is the
                    // scalar shift of that lane by the same amount, masked to the lane width.
                    let amount = reg(regs, *r);
                    collect_lanes(
                        (0..*n as usize)
                            .map(|i| mask_lane(apply_bin(*b, av[i], amount, Some(&lane)), &lane)),
                    )?
                };
                self.push_vec(lanes)?
            }
            // Result lane `k` is source lane `first + k` under the scalar cast of the same kind, so
            // a widened lane can never differ from what the scalar loop computes for it.
            Op::Widen { v, half, kind } => {
                let (from_lane, src_n) = match func.value_type(*v) {
                    MirType::Vec(l, m) => ((**l).clone(), *m),
                    other => {
                        return Err(format!(
                            "widen operand is {}, not a vector",
                            other.display()
                        ))
                    }
                };
                let first = half.first_lane(src_n) as usize;
                if first + *n as usize > src_n as usize {
                    return Err(
                        "widen reads past its operand's lanes (MIR invariant violation)".into(),
                    );
                }
                let xs = self.vec_lanes(reg(regs, *v), src_n as usize)?;
                let to_lane = (**lane).clone();
                let lanes = collect_lanes((0..*n as usize).map(|i| {
                    mask_lane(
                        apply_cast(*kind, xs[first + i], &from_lane, &to_lane),
                        &to_lane,
                    )
                }))?;
                self.push_vec(lanes)?
            }
            // The lanes of `lo`, then the lanes of `hi`, each reduced on its own.
            Op::Narrow { lo, hi, kind } => {
                let half = (*n / 2) as usize;
                let lo_lanes = self.vec_lanes(reg(regs, *lo), half)?;
                let hi_lanes = self.vec_lanes(reg(regs, *hi), half)?;
                let to_lane = (**lane).clone();
                let lanes = collect_lanes((0..*n as usize).map(|i| {
                    let x = if i < half {
                        lo_lanes[i]
                    } else {
                        hi_lanes[i - half]
                    };
                    narrow_lane(*kind, x, &to_lane)
                }))?;
                self.push_vec(lanes)?
            }
            // Lane `k` is lane `mask[k]` of `a ++ b`: a pure copy, whatever the lane type.
            Op::Shuffle { a, b, mask } => {
                let av = self.vec_lanes(reg(regs, *a), *n as usize)?;
                let bv = self.vec_lanes(reg(regs, *b), *n as usize)?;
                let width = *n as usize;
                if mask.len() != width || mask.iter().any(|&m| m as usize >= 2 * width) {
                    return Err(
                        "shuffle mask does not fit its operands (MIR invariant violation)".into(),
                    );
                }
                let lanes = collect_lanes(mask.iter().map(|&m| {
                    let m = m as usize;
                    if m < width {
                        av[m]
                    } else {
                        bv[m - width]
                    }
                }))?;
                self.push_vec(lanes)?
            }
            Op::InsertLane(v, s, k) => {
                let xs = self.vec_lanes(reg(regs, *v), *n as usize)?;
                if *k >= *n {
                    return Err("insertlane index out of range".into());
                }
                let value = reg(regs, *s);
                let lanes = collect_lanes((0..*n as usize).map(|i| {
                    if i == *k as usize {
                        value
                    } else {
                        xs[i]
                    }
                }))?;
                self.push_vec(lanes)?
            }
            Op::Cmp(c, l, r) => {
                // Lane-wise compare → a mask vector of 1/0 lanes.
                let av = self.vec_lanes(reg(regs, *l), *n as usize)?;
                let bv = self.vec_lanes(reg(regs, *r), *n as usize)?;
                let lanes = collect_lanes(
                    (0..*n as usize).map(|i| Value::Int(apply_cmp(*c, av[i], bv[i]) as i128)),
                )?;
                self.push_vec(lanes)?
            }
            Op::Neg(v) => {
                // Lane-wise negate (the autovectorized `-x[k]`). Without this arm a `VecRef`
                // operand falls into the scalar path below, where `as_int()` yields 0, so every
                // lane is silently zeroed — a miscompile the native backend does not share.
                let xs = self.vec_lanes(reg(regs, *v), *n as usize)?;
                let is_float = lane.is_float();
                let lanes = collect_lanes((0..*n as usize).map(|i| {
                    if is_float {
                        Value::Float(-xs[i].as_float())
                    } else {
                        // Wrap at the lane width: negating `i32::MIN` is `i32::MIN`.
                        mask_lane(Value::Int(xs[i].as_int().wrapping_neg()), lane)
                    }
                }))?;
                self.push_vec(lanes)?
            }
            Op::Not(v) => {
                // Lane-wise bitwise complement, truncated to the lane width. Latent today (the
                // vectorizer does not yet emit a vector `Not`), but mirror `Neg` so it can
                // never silently zero lanes or carry `i128` bits the backend does not have.
                let xs = self.vec_lanes(reg(regs, *v), *n as usize)?;
                let lanes = collect_lanes(
                    (0..*n as usize).map(|i| mask_lane(Value::Int(!xs[i].as_int()), lane)),
                )?;
                self.push_vec(lanes)?
            }
            Op::Cast(kind, v, _) => {
                // Lane-wise cast (e.g. the vectorized exp's f32->i32 and i32->f32 bitcast).
                let xs = self.vec_lanes(reg(regs, *v), *n as usize)?;
                let from_lane = match func.value_type(*v) {
                    MirType::Vec(l, _) => (**l).clone(),
                    other => other.clone(),
                };
                let to_lane = (**lane).clone();
                let lanes = collect_lanes(
                    (0..*n as usize).map(|i| apply_cast(*kind, xs[i], &from_lane, &to_lane)),
                )?;
                self.push_vec(lanes)?
            }
            Op::Select(c, a, b) => {
                // Lane-wise blend by a mask vector.
                let m = self.vec_lanes(reg(regs, *c), *n as usize)?;
                let av = self.vec_lanes(reg(regs, *a), *n as usize)?;
                let bv = self.vec_lanes(reg(regs, *b), *n as usize)?;
                let lanes =
                    collect_lanes((0..*n as usize).map(
                        |i| {
                            if m[i].truthy() {
                                av[i]
                            } else {
                                bv[i]
                            }
                        },
                    ))?;
                self.push_vec(lanes)?
            }
            Op::Fma(a, b, c) => {
                let av = self.vec_lanes(reg(regs, *a), *n as usize)?;
                let bv = self.vec_lanes(reg(regs, *b), *n as usize)?;
                let cv = self.vec_lanes(reg(regs, *c), *n as usize)?;
                let lane = (**lane).clone();
                let lanes = collect_lanes(
                    (0..*n as usize).map(|i| apply_fma(av[i], bv[i], cv[i], Some(&lane))),
                )?;
                self.push_vec(lanes)?
            }
            Op::Sqrt(v) => {
                let xv = self.vec_lanes(reg(regs, *v), *n as usize)?;
                let lane = (**lane).clone();
                let lanes =
                    collect_lanes((0..*n as usize).map(|i| apply_sqrt(xv[i], Some(&lane))))?;
                self.push_vec(lanes)?
            }
            Op::Round(mode, v) => {
                let xv = self.vec_lanes(reg(regs, *v), *n as usize)?;
                let lane = (**lane).clone();
                let lanes = collect_lanes(
                    (0..*n as usize).map(|i| apply_round(*mode, xv[i], Some(&lane))),
                )?;
                self.push_vec(lanes)?
            }
            Op::Splat(v) => self.push_vec(collect_lanes(std::iter::repeat_n(
                reg(regs, *v),
                *n as usize,
            ))?)?,
            Op::Iota(_) => self.push_vec(collect_lanes(
                (0..*n as usize).map(|k| Value::Int(k as i128)),
            )?)?,
            _ => return Err("unsupported SIMD operation".into()),
        })
    }

    /// Intern a freshly-computed vector value into the side arena, returning its handle.
    fn push_vec(&mut self, lanes: Vec<Value>) -> Result<Value, String> {
        if lanes.iter().any(|v| matches!(v, Value::VecRef(_))) {
            return Err("SIMD lanes must be scalar; nested vectors are unsupported".into());
        }
        self.vecs
            .try_reserve(1)
            .map_err(|_| "SIMD arena allocation failed")?;
        let index = self.vecs.len();
        self.vecs.push(lanes);
        #[cfg(test)]
        {
            self.vector_peak = self.vector_peak.max(self.vecs.len());
        }
        Ok(Value::VecRef(index))
    }

    /// Compact only this call's suffix. Aliases share one retained vector and one remapped handle.
    /// Suspended caller registers and argument buffers only contain handles below `base`.
    fn compact_vectors(
        &mut self,
        base: usize,
        regs: &mut [Value],
        remap: &mut Vec<usize>,
    ) -> Result<(), String> {
        let count = self
            .vecs
            .len()
            .checked_sub(base)
            .ok_or("invalid SIMD frame watermark")?;
        if count == 0 {
            return Ok(());
        }
        remap.clear();
        remap
            .try_reserve(count)
            .map_err(|_| "SIMD compaction allocation failed")?;
        remap.resize(count, usize::MAX);
        for value in regs.iter() {
            if let Value::VecRef(index) = *value {
                if index >= base {
                    *remap
                        .get_mut(index - base)
                        .ok_or("invalid SIMD vector handle in register")? = 0;
                }
            }
        }
        let mut retained = 0;
        for (offset, destination) in remap.iter_mut().enumerate() {
            if *destination != usize::MAX {
                *destination = base + retained;
                self.vecs.swap(base + offset, *destination);
                retained += 1;
            }
        }
        self.vecs.truncate(base + retained);
        for value in regs {
            if let Value::VecRef(index) = value {
                if *index >= base {
                    *index = remap[*index - base];
                }
            }
        }
        Ok(())
    }

    /// Borrow the `n` immutable lanes behind a `VecRef`. Callers compute their result before
    /// appending it to the arena, so operands need no temporary copies.
    ///
    /// An operand that is not an `n`-lane vector is a lowering bug, and every caller indexes
    /// `0..n` — this used to hand back `vec![other]` for a scalar, so the next index panicked with
    /// a raw `index out of bounds` instead of producing a diagnostic. The MIR verifier shields the
    /// `Bin`/`Neg`/`Not`/`Select` arms (it forces operand type == vector result type) but **not**
    /// `Op::Cast`, whose arm only checks the result: `%r: <4 x i32> = cast sext %s` with a scalar
    /// `%s` verifies clean and then reached that panic. Report it instead of broadcasting, which
    /// would hide the lowering bug; the durable fix is an operand check in `wukong_mir::verify`.
    fn vec_lanes(&self, v: Value, n: usize) -> Result<&[Value], String> {
        match v {
            Value::VecRef(i) if self.vecs.get(i).is_some_and(|lanes| lanes.len() == n) => {
                Ok(&self.vecs[i])
            }
            _ => Err(format!(
                "vector op operand is not a {n}-lane vector (MIR invariant violation)"
            )),
        }
    }

    /// Append printed bytes to this run's stdout. A streaming run ([`run_streaming`]) writes them
    /// through to the process's stdout at a newline (line-buffered) or once a block has filled.
    fn emit_stdout(&mut self, bytes: &[u8]) {
        self.stdout.extend_from_slice(bytes);
        let write = match STDOUT_MODE.with(Cell::get) {
            StdoutMode::Capture => false,
            StdoutMode::LineBuffered => bytes.contains(&b'\n'),
            StdoutMode::BlockBuffered => self.stdout.len() >= STREAM_BLOCK,
        };
        if write {
            self.flush_stdout();
        }
    }

    /// In a streaming run, write out everything buffered so far (a capturing run keeps it).
    fn flush_stdout(&mut self) {
        use std::io::Write as _;
        if STDOUT_MODE.with(Cell::get) == StdoutMode::Capture {
            return;
        }
        let out = std::io::stdout();
        let mut out = out.lock();
        let _ = out.write_all(&self.stdout);
        let _ = out.flush();
        self.stdout.clear();
    }

    /// `print`/`println` of an `f32` (or an `f16`/`bf16` held in one): the value's shortest
    /// round-trip digits at `f32` precision. The slot holds the value's exact `f64` image, so the
    /// narrowing is exact.
    fn print_f32(&mut self, args: &[Value]) -> Result<Value, String> {
        let x = args.first().map_or(0.0, |v| v.as_float()) as f32;
        let text = wukong_runtime::fmt::f32_line(x);
        self.emit_stdout(text.as_bytes());
        Ok(Value::Unit)
    }

    /// Call the runtime intrinsic `name` by its callee name — the by-name entry the unit tests
    /// drive arms through. A run resolves each callee once ([`CalleeTable`]) and dispatches on the
    /// id; a name outside the intrinsic table fails exactly as an unresolved call does.
    #[cfg(test)]
    fn intrinsic(&mut self, name: &str, args: &[Value]) -> Result<Value, String> {
        match Intrinsic::from_name(name) {
            Some(id) => self.call_intrinsic(id, args),
            None => Err(format!("call to unknown function or intrinsic `{name}`")),
        }
    }

    /// Run the intrinsic `id` on `args`. Each arm keeps `name` (the id's callee name) for its
    /// diagnostics; the text, ML and decode builtins are dispatched by their own modules.
    fn call_intrinsic(&mut self, id: Intrinsic, args: &[Value]) -> Result<Value, String> {
        use Intrinsic as I;
        let name = id.name();
        match id {
            I::Print | I::Println => {
                let text = match args.first().copied().unwrap_or(Value::Unit) {
                    Value::Int(i) => format!("{i}\n"),
                    Value::Float(f) => format!("{f}\n"),
                    Value::Ptr(p) => format!("{p}\n"),
                    Value::VecRef(_) => "<vector>\n".to_string(),
                    Value::Unit => "\n".to_string(),
                };
                self.emit_stdout(text.as_bytes());
                Ok(Value::Unit)
            }
            // The unsigned twin of `print`/`println` (mir_build routes an unsigned-integer argument
            // here after zero-extending it to 64 bits): render the low 64 bits as `u64`, so a
            // high-bit-set value prints its magnitude. Native's `rt_print_u64` formats the identical
            // bits the same way, so the differential gate holds.
            I::PrintU | I::PrintlnU => {
                let text = match args.first().copied().unwrap_or(Value::Unit) {
                    Value::Int(i) => format!("{}\n", i as u64),
                    other => format!("{}\n", other.as_int() as u64),
                };
                self.emit_stdout(text.as_bytes());
                Ok(Value::Unit)
            }
            // A `bool` / `char` argument to `print`/`println` (mir_build zero-extends it to 64 bits
            // and routes it here): `true`/`false`, or the character's glyph. Native's
            // `rt_print_bool`/`rt_print_char` render through the same `wukong_runtime::fmt`.
            I::PrintBool | I::PrintlnBool => {
                let v = args.first().map_or(0, |v| v.as_int() as i64);
                let text = wukong_runtime::fmt::bool_line(v);
                self.emit_stdout(text.as_bytes());
                Ok(Value::Unit)
            }
            I::PrintChar | I::PrintlnChar => {
                let v = args.first().map_or(0, |v| v.as_int() as i64);
                let text = wukong_runtime::fmt::char_line(v);
                self.emit_stdout(text.as_bytes());
                Ok(Value::Unit)
            }
            // A `*u8` string argument to `print`/`println` (lowered to these symbols by mir_build):
            // walk `memory` from the base pointer, collecting bytes (each stored as a `Value::Int`
            // low byte) until the NUL terminator, then render as UTF-8. Native renders the identical
            // bytes via `rt_print_str`, so the differential gate holds.
            I::PrintStr | I::PrintlnStr => {
                // Native's `rt_print_str` returns without writing anything (not even the newline)
                // for a null string. Slot 0 is the shared null now (`NULL_SLOT`), so mirror it.
                if let Some(Value::Ptr(p)) = args.first().copied() {
                    if self.is_null(p) {
                        return Ok(Value::Unit);
                    }
                }
                let mut bytes = Vec::new();
                if let Some(Value::Ptr(base)) = args.first().copied() {
                    let mut i = base;
                    while i < self.memory.len() {
                        let b = self.memory[i].as_int() as u8;
                        if b == 0 {
                            break;
                        }
                        bytes.push(b);
                        i += 1;
                    }
                }
                let mut text = String::from_utf8_lossy(&bytes).into_owned();
                text.push('\n');
                self.emit_stdout(text.as_bytes());
                Ok(Value::Unit)
            }
            I::Assert => {
                let ok = match args.first().copied().unwrap_or(Value::Unit) {
                    Value::Int(i) => i != 0,
                    Value::Float(f) => f != 0.0,
                    Value::Ptr(p) => p != 0,
                    Value::VecRef(_) | Value::Unit => false,
                };
                if ok {
                    Ok(Value::Unit)
                } else {
                    // `assert(cond, site[, message])`: the lowering appends the call's `.rodata`
                    // `file:line:col` (when it had a source map) and the message literal (when the
                    // program wrote one); a bare `assert(cond)` carries neither. The runtime's one
                    // formatter words it, as for the JIT and a linked executable.
                    let text = |i: usize| -> Result<String, String> {
                        match args.get(i).copied() {
                            Some(Value::Ptr(base)) if !self.is_null(base) => {
                                file_read_path(&self.memory, base)
                            }
                            _ => Ok(String::new()),
                        }
                    };
                    let (site, message) = (text(1)?, text(2)?);
                    Err(wukong_runtime::assert_failure_message(&site, &message))
                }
            }
            // `wukong_rt_bounds_fail(index, len, site)` — the failure edge of a checked slice access
            // `s[i]` (`index <u len` was false). `site` is the `.rodata` `file:line:col` string the
            // lowering attached. The message comes from the one formatter the JIT and the linked
            // executable also use, so all three paths report the same text and stop at this point.
            I::RtBoundsFail => {
                let index = args.first().map_or(0, |v| v.as_int()) as i64;
                let len = args.get(1).map_or(0, |v| v.as_int()) as i64;
                let site = match args.get(2).copied() {
                    Some(Value::Ptr(base)) if !self.is_null(base) => {
                        file_read_path(&self.memory, base)?
                    }
                    _ => String::new(),
                };
                Err(wukong_runtime::bounds_failure_message(index, len, &site))
            }
            // `wukong_rt_check_fail(kind, a, b, site)` — every other failed run-time check (a
            // range index, `str_from_utf8`), reported through the runtime's one formatter exactly
            // as a bounds failure is above.
            I::RtCheckFail => {
                let int = |i: usize| args.get(i).map_or(0, |v| v.as_int()) as i64;
                let site = match args.get(3).copied() {
                    Some(Value::Ptr(base)) if !self.is_null(base) => {
                        file_read_path(&self.memory, base)?
                    }
                    _ => String::new(),
                };
                Err(wukong_runtime::check_failure_message(
                    int(0),
                    int(1),
                    int(2),
                    &site,
                ))
            }
            // `wukong_rt_alloc(count, elem_size, elem_is_float) -> ptr` — the heap allocation
            // backing the `alloc_<T>(n)` builtins. A heap object in the slot-indexed memory model
            // is exactly an alloca-shaped run: `count` contiguous typed-zero slots (one per scalar
            // element; `elem_size` is native's byte stride and irrelevant here), whose
            // function-independent lifetime falls out of `memory` never being reclaimed. The zero
            // is `Float(0.0)` for float elements / `Int(0)` otherwise — the same typed value
            // native's calloc'd zero bits decode to, so a read-before-write observes an identical
            // zero on both backends (the determinism contract).
            //
            // **Exhaustion.** The native `wukong_rt_alloc` returns null when the allocator cannot
            // satisfy the request (crates/wukong_runtime/src/lib.rs). Slot 0 is now a reserved
            // null (`NULL_SLOT`), so the interpreter *could* return the same null — but a null
            // slice whose length field still says `count` would only move the failure to the first
            // element access (a null dereference here, a segfault natively). It reports a
            // diagnostic at the allocation instead —
            // `try_reserve` rather than `resize`, so the request never reaches Rust's allocation-
            // error handler, which `abort()`s the process with a raw `memory allocation of N bytes
            // failed` message no caller can catch. One element costs one 32-byte `Value` slot here
            // against 4 bytes natively, so the interpreter reaches exhaustion ~8x earlier; that
            // difference is why this path is reachable at all.
            I::RtAlloc => {
                let count = args.first().map(|v| v.as_int()).unwrap_or(0).max(0) as usize;
                let is_float = args.get(2).map(|v| v.as_int()).unwrap_or(0) != 0;
                let zero = if is_float {
                    Value::Float(0.0)
                } else {
                    Value::Int(0)
                };
                // A freed block that fits comes back re-zeroed with this request's typed zero
                // (`SlotReuse`), so an alloc/free-per-step loop runs in bounded memory.
                if count > 0 {
                    if let Some((base, cap)) = self.slots.take_free(count) {
                        if let Some(block) = self.memory.get_mut(base..base + count) {
                            block.fill(zero);
                            self.slots.heap_live.insert(base, cap);
                            // The allocation is the `count` slots asked for, not the capacity.
                            self.slots.live.insert(base, count);
                            return Ok(Value::Ptr(base));
                        }
                    }
                }
                let idx = self.memory.len();
                // An explicit ceiling, because `try_reserve` alone is NOT a portable guard. It
                // checks the RESERVE; the `resize` below is what COMMITS the pages. Windows does
                // not overcommit, so an absurd request fails in `try_reserve` and the diagnostic
                // fires — but under Linux overcommit the same reservation SUCCEEDS and the resize
                // then faults pages in until the OOM killer takes the process. Observed
                // 2026-08-08 on a Modal 8-core/16 GiB Linux container: this very arm, reached
                // from `heap_exhaustion_is_a_diagnostic_not_an_allocator_abort` (a 100e9-element
                // = 3.2 TB request), stalled >60 s and then SIGKILLed the container with exit 137
                // — an uncatchable abort, which is the exact outcome this arm exists to prevent.
                // 2^31 slots is 64 GiB at 32 B/`Value`, far above any real program (GPT-2 124M is
                // ~124M slots) and far below anything a host can satisfy.
                const MAX_SLOTS: usize = 1 << 31;
                if count > MAX_SLOTS {
                    return Err(format!(
                        "interpreter heap exhausted allocating {count} element(s)"
                    ));
                }
                self.memory.try_reserve(count).map_err(|_| {
                    format!("interpreter heap exhausted allocating {count} element(s)")
                })?;
                self.memory.resize(idx + count, zero);
                if count > 0 {
                    self.slots.heap_live.insert(idx, count);
                    self.slots.live.insert(idx, count);
                }
                Ok(Value::Ptr(idx))
            }
            // `wukong_rt_free(data)` — a live `wukong_rt_alloc` block joins the size-bucketed free
            // list the next fitting allocation reuses (`SlotReuse`); native releases the real
            // allocation. Double-free or freeing a non-alloc slice is ignored, so it never crashes
            // the interpreter; a use-after-free reads whatever the block holds now while being
            // undefined on native — such programs are outside the differential contract.
            I::RtFree => {
                if let Some(Value::Ptr(base)) = args.first() {
                    self.slots.free(*base);
                }
                Ok(Value::Unit)
            }
            // `wukong_rt_scratch_alloc(count, elem_size, elem_is_float) -> ptr` — `scratch_<T>(n)`
            // (the contract is in `wukong_runtime::scratch`). `count` typed-zero slots exactly like
            // `wukong_rt_alloc`, counted as one live scratch allocation — a zero-length one too, as
            // natively, so `scratch_mark()` agrees. The region released last from this depth is
            // handed out again when it is large enough, re-zeroed as native re-zeroes its reused
            // bytes, so a balanced `mark; scratch_…; release` loop reuses one region instead of
            // growing the run's memory every iteration.
            I::RtScratchAlloc => {
                let count = args.first().map(|v| v.as_int()).unwrap_or(0).max(0) as usize;
                let is_float = args.get(2).map(|v| v.as_int()).unwrap_or(0) != 0;
                let zero = if is_float {
                    Value::Float(0.0)
                } else {
                    Value::Int(0)
                };
                let depth = self.scratch_arena.live.len();
                let reuse = self
                    .scratch_arena
                    .free
                    .get_mut(depth)
                    .and_then(Option::take)
                    .filter(|&(_, cap)| cap >= count);
                let (base, cap) = match reuse {
                    Some(region) => region,
                    None => {
                        // The same ceiling and the same `try_reserve` guard as `wukong_rt_alloc`.
                        const MAX_SLOTS: usize = 1 << 31;
                        if count > MAX_SLOTS {
                            return Err(format!(
                                "interpreter heap exhausted allocating {count} element(s)"
                            ));
                        }
                        let idx = self.memory.len();
                        self.memory.try_reserve(count).map_err(|_| {
                            format!("interpreter heap exhausted allocating {count} element(s)")
                        })?;
                        self.memory.resize(idx + count, zero);
                        (idx, count)
                    }
                };
                self.memory[base..base + count].fill(zero);
                self.scratch_arena.live.push((base, cap, count));
                self.slots.live.insert(base, count);
                Ok(Value::Ptr(base))
            }
            // `wukong_rt_scratch_mark() -> i64` — `scratch_mark()`: the live scratch-allocation
            // count, the backend-independent value native's arena reports.
            I::RtScratchMark => Ok(Value::Int(self.scratch_arena.live.len() as i128)),
            // `wukong_rt_scratch_release(mark)` — `scratch_release(m)`: pop the allocations made
            // after `mark`, each becoming the reusable region of its depth. A negative mark
            // releases them all; a mark at or past the count releases nothing.
            I::RtScratchRelease => {
                let mark = args.first().map(|v| v.as_int()).unwrap_or(0).max(0);
                let mark = usize::try_from(mark).unwrap_or(usize::MAX);
                let arena = &mut self.scratch_arena;
                while arena.live.len() > mark {
                    let depth = arena.live.len() - 1;
                    let region = arena.live.pop();
                    if arena.free.len() <= depth {
                        arena.free.resize(depth + 1, None);
                    }
                    arena.free[depth] = region.map(|(base, cap, _)| (base, cap));
                    // The region is no longer an allocation (an empty one never was).
                    if let Some((base, _, 1..)) = region {
                        self.slots.live.remove(base);
                    }
                }
                Ok(Value::Unit)
            }
            // `wukong_now_ns() -> i64` — the `now_ns()` timing intrinsic. Call the *same*
            // `wukong_runtime` clock the native backend links (a monotonic process-lifetime epoch),
            // so both backends read one clock. The value is inherently non-deterministic, but a
            // conforming program observes only *differences* between two reads (never the raw stamp),
            // so its printed output stays interp==native — the value never reaches a differential
            // fixture's stdout.
            I::NowNs => Ok(Value::Int(wukong_runtime::wukong_now_ns() as i128)),
            // `wukong_rt_read_{f32,i32,i64,u8,f64,i8}(path: *u8, data: *T, len: i64) -> i64` — the read
            // half of the file-I/O intrinsic family. The on-disk format is headerless: raw contiguous
            // little-endian elements (f32=4, i32=4, i64=8, u8=1, f64=8, i8=1 bytes; no magic, no length prefix).
            // Reconstruct the NUL-terminated `path` by walking `memory` from the base pointer (the
            // same loop as `print_str`), then open it read-only: on open failure return -1 and leave
            // `data` untouched. Otherwise `avail = file_len / sizeof(T)` (floor), `n = min(len, avail)`;
            // read `n * sizeof(T)` bytes, decode each element little-endian and store into `data[0..n]`
            // as `Value::Float(x as f64)` (f32) / `Value::Int(x as i128)` (the integer widths) — the
            // identical typed value native's calloc'd load observes — leaving `data[n..]` unchanged. A
            // read error mid-way returns -2; success returns `n` (>= 0). The LE decode, the min(len,avail)
            // truncation, and the -1/-2/n codes are byte-identical to `wukong_runtime`, so the
            // interp==native differential gate holds.
            I::RtReadF32
            | I::RtReadI32
            | I::RtReadI64
            | I::RtReadU8
            | I::RtReadF64
            | I::RtReadI8 => {
                // `path` is a NUL-terminated `*u8`: walk `memory` collecting low bytes until NUL.
                let path_base = match args.first().copied() {
                    Some(Value::Ptr(base)) => base,
                    _ => return Err("file-io read: path is not a pointer".into()),
                };
                let path = file_read_path(&self.memory, path_base)?;
                let data_base = match args.get(1).copied() {
                    Some(Value::Ptr(base)) => base,
                    _ => return Err("file-io read: data is not a pointer".into()),
                };
                let len = args.get(2).map(|v| v.as_int()).unwrap_or(0);
                let elem: usize = match name {
                    "wukong_rt_read_i64" | "wukong_rt_read_f64" => 8,
                    "wukong_rt_read_u8" | "wukong_rt_read_i8" => 1,
                    _ => 4, // f32, i32
                };
                let mut file = match std::fs::File::open(&path) {
                    Ok(f) => f,
                    Err(_) => return Ok(Value::Int(-1)),
                };
                let file_len = match file.metadata() {
                    Ok(m) => m.len() as i128,
                    Err(_) => return Ok(Value::Int(-2)),
                };
                let avail = file_len / elem as i128;
                // n = min(len, avail), clamped to >= 0 (guards a non-positive len — 0 => read nothing).
                let n = len.min(avail);
                let n = usize::try_from(n.max(0))
                    .map_err(|_| "file-io read: element extent overflow")?;
                if data_base
                    .checked_add(n)
                    .is_some_and(|end| end <= self.memory.len())
                {
                    // In the arena (`read_file_elements` reports it otherwise): inside ONE live
                    // allocation too, before a byte is read.
                    self.slots.live.check(name, 1, data_base, n)?;
                }
                read_file_elements(&mut self.memory, &mut file, data_base, n, elem, name)
            }
            // `wukong_rt_write_{f32,i32,i64,u8,f64,i8}(path: *u8, data: *T, len: i64) -> i64` — the write
            // half of the file-I/O family. Reconstruct the NUL-terminated `path` (as `print_str`),
            // then create/truncate it: on create failure return -1. Otherwise read back `len` elements
            // from `data[0..len]`, narrow each to `T` (`as f32`/`as i32`/`as i64`/`as u8` — the same
            // truncation native's typed store performs) and encode little-endian into one contiguous
            // headerless buffer, then write it. A write error returns -2; success returns `len`. The
            // narrowing, the byte layout, and the -1/-2/len codes are byte-identical to
            // `wukong_runtime`, so the interp==native differential gate holds.
            I::RtWriteF32
            | I::RtWriteI32
            | I::RtWriteI64
            | I::RtWriteU8
            | I::RtWriteF64
            | I::RtWriteI8 => {
                use std::io::Write;
                let path_base = match args.first().copied() {
                    Some(Value::Ptr(base)) => base,
                    _ => return Err("file-io write: path is not a pointer".into()),
                };
                let mut path_bytes = Vec::new();
                let mut i = path_base;
                while i < self.memory.len() {
                    let b = self.memory[i].as_int() as u8;
                    if b == 0 {
                        break;
                    }
                    path_bytes.push(b);
                    i += 1;
                }
                let path = String::from_utf8_lossy(&path_bytes).into_owned();
                let data_base = match args.get(1).copied() {
                    Some(Value::Ptr(base)) => base,
                    _ => return Err("file-io write: data is not a pointer".into()),
                };
                let len = args.get(2).map(|v| v.as_int()).unwrap_or(0);
                // Guard a non-positive len: nothing to serialize (the file is still created/truncated
                // below, so a subsequent read observes an empty file — matching native).
                let count = if len < 0 { 0 } else { len as usize };
                // The elements serialized must be one allocation's, checked before the file is
                // created (a range past the arena is still reported element by element below).
                if data_base
                    .checked_add(count)
                    .is_some_and(|end| end <= self.memory.len())
                {
                    self.slots.live.check(name, 1, data_base, count)?;
                }
                let mut file = match std::fs::File::create(&path) {
                    Ok(f) => f,
                    Err(_) => return Ok(Value::Int(-1)),
                };
                // Serialize all elements little-endian, then write once — the resulting file bytes are
                // identical to native's per-element `write_all`.
                let mut out: Vec<u8> = Vec::new();
                for k in 0..count {
                    let slot = data_base + k;
                    let v = match self.memory.get(slot) {
                        Some(v) => *v,
                        None => return Err("file-io write: data out of bounds".into()),
                    };
                    match name {
                        "wukong_rt_write_f32" => {
                            out.extend_from_slice(&(v.as_float() as f32).to_le_bytes())
                        }
                        "wukong_rt_write_i32" => {
                            out.extend_from_slice(&(v.as_int() as i32).to_le_bytes())
                        }
                        "wukong_rt_write_i64" => {
                            out.extend_from_slice(&(v.as_int() as i64).to_le_bytes())
                        }
                        "wukong_rt_write_f64" => out.extend_from_slice(&v.as_float().to_le_bytes()),
                        "wukong_rt_write_i8" => {
                            out.extend_from_slice(&(v.as_int() as i8).to_le_bytes())
                        }
                        // wukong_rt_write_u8
                        _ => out.push(v.as_int() as u8),
                    }
                }
                if file.write_all(&out).is_err() {
                    return Ok(Value::Int(-2));
                }
                Ok(Value::Int(count as i128))
            }
            // (`wukong_parallel_for` never reaches here: `dispatch_call` runs its body as a frame on
            // the explicit stack — see `Interp::parallel_for_body`.)
            // `wukong_sgemm(a, b, c, m, k, n, beta)` (and the parallel / transposed `nn.Linear`
            // variants) — the GEMM microkernels the compiler lowers a matmul nest to. The interpreter
            // marshals its abstract (tagged-`Value`) memory into real f32 buffers and calls the
            // *identical* runtime kernel the native backend calls, then marshals the result back. The
            // serial kernel is used for both serial and parallel names (numerically identical).
            //
            // THE MARSHALLING CONTRACT, stated exactly, because the comment that used to sit here
            // said "so the differential oracle stays bit-for-bit exact" and that is only true on
            // DISJOINT operands. Copying each operand into a private `Vec` and writing the result
            // back is copy-in/copy-out: it de-aliases the call. On overlapping operands the
            // interpreter therefore computes a program that neither the native backend nor the
            // source nest has, and — because the oracle's answer is the more plausible one — a
            // differential failure on such an input would name NATIVE as the deviant side.
            //
            // What keeps that unreachable is the lowering PLUS a check here, and both halves are
            // stated because the comment this one replaces listed three groups the lowering did not
            // gate. `wukong_mir_build` now routes every multi-buffer recognizer through one
            // classifier (`alias::plan_kernel_alias`) and, for a pair it cannot separate at compile
            // time, a run-time overlap test (`kernel_guard.rs`) over every constrained operand's
            // extent that dispatches the kernel only when the operands are disjoint — or EXACTLY in
            // place where the runtime entry documents in-place operation (velem, vmath, the scans,
            // RoPE, the norms, the `dx`-over-`dy`/`dx`-over-`x` backward forms) — and lowers the
            // ORIGINAL source loop otherwise. The test is built with `Gep`, so it runs in this
            // interpreter's slot address space and reaches the same verdict the native backend does.
            //
            // So for every guarded family copy-in/copy-out is unobservable, and each arm makes that
            // a checked statement with `marshal::check_output`: operands that overlap in a way the
            // kernel's contract forbids are an ERROR, not a de-aliased answer. Exact in-place
            // operation passes where permitted, and is equal to out of place for a row-local or
            // elementwise kernel, which is what makes the copy sound there. The check can only fire
            // on a call that bypassed the guard — a hand-written `sdpa(..)`, an alias-analysis hole
            // (`alias.rs`'s `# Known limits`), or a direct intrinsic call — and there the honest
            // oracle answer is "outside the kernel's contract".
            //
            // Two families stay off the copy entirely: the embedding gather and the scatter-add run
            // against live memory in source order (their arms below), so they mean the source nest
            // under any aliasing. And the purely elementwise maps (vmath, vmath2, velem, vhorner)
            // answer a SHIFTED overlap the same way, one element at a time on live memory
            // (`marshal::elementwise_live`): the compiler versions their dispatch like every other
            // kernel, but an elementwise map has one meaning under ANY overlap — the source loop's —
            // so the oracle computes it rather than refusing, and a call that reached the kernel
            // with such operands (a direct call, an alias-analysis hole) is judged against it.
            //
            // Operand scratch is recycled per thread (`marshal::f32_scratch`/`recycle_f32`): a
            // kernel inside a loop no longer allocates its operands on every iteration.
            I::Sgemm
            | I::SgemmParallel
            | I::SgemmNt
            | I::SgemmNtParallel
            | I::SgemmTn
            | I::SgemmTnParallel => {
                let a = ptr(args[0])?;
                let b = ptr(args[1])?;
                let c = ptr(args[2])?;
                let m = dim!(args[3]);
                let k = dim!(args[4]);
                let n = dim!(args[5]);
                let beta = args[6].as_int() as i64;
                let matrix = |op: usize, base: usize, rows: usize, cols: usize| {
                    let len = rows
                        .checked_mul(cols)
                        .ok_or("sgemm operand extent overflow")?;
                    let end = base.checked_add(len).ok_or("sgemm operand out of bounds")?;
                    let values = self
                        .memory
                        .get(base..end)
                        .ok_or("sgemm operand out of bounds")?;
                    self.slots.live.check(name, op, base, len)?;
                    Ok::<_, String>(values)
                };
                // Validate every product, arena range and allocation before any scratch
                // allocation: an operand that fits the arena but runs into the next allocation is
                // refused, not multiplied.
                let av = matrix(0, a, m, k)?;
                let bv = matrix(1, b, k, n)?;
                let cv = matrix(2, c, m, n)?;
                // gemm.rs: `C` is written in tiled order while `A`/`B` are read — no overlap.
                marshal::check_output(
                    arena!(self),
                    name,
                    (2, c, cv.len()),
                    &[
                        ((0, a, av.len()), Overlap::Forbidden),
                        ((1, b, bv.len()), Overlap::Forbidden),
                    ],
                )?;
                // Pooled scratch (`marshal::f32_scratch`): a GEMM inside a loop no longer allocates
                // three operand buffers per iteration.
                let read = |values: &[Value], input: bool| -> Result<Vec<f32>, String> {
                    let mut buffer = marshal::f32_scratch(values.len(), name)?;
                    for value in values {
                        buffer.push(if input {
                            value.kernel_float(name)? as f32
                        } else {
                            0.0
                        });
                    }
                    Ok(buffer)
                };
                let abuf = read(av, true)?;
                let bbuf = read(bv, true)?;
                let mut cbuf = read(cv, beta != 0)?;
                let transposed = name.contains("_nt");
                // `_tn` is `C = Aᵀ·B` (A stored `[k,m]`, B `[k,n]`); element counts m*k / k*n / m*n
                // match the plain/`_nt` reads above, so only the kernel differs. The kernel handles
                // the transpose internally; the GPU offload below is `A·Bᵀ`-only (transposed) so it
                // correctly declines `_tn` and this falls through to the CPU kernel.
                let tn = name.contains("_tn");
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
                        if tn {
                            // C = Aᵀ·B. Serial kernel for both serial/parallel names — `_tn`'s serial
                            // and parallel forms are numerically identical (rows independent), so the
                            // differential oracle (interp-serial vs native-parallel) stays bit-exact.
                            wukong_runtime::wukong_sgemm_tn(
                                abuf.as_ptr(),
                                bbuf.as_ptr(),
                                cbuf.as_mut_ptr(),
                                m as i64,
                                k as i64,
                                n as i64,
                                beta,
                            );
                        } else if transposed {
                            wukong_runtime::wukong_sgemm_nt(
                                abuf.as_ptr(),
                                bbuf.as_ptr(),
                                cbuf.as_mut_ptr(),
                                m as i64,
                                k as i64,
                                n as i64,
                                beta,
                            );
                        } else {
                            wukong_runtime::wukong_sgemm(
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
                marshal::write_f32(arena_mut!(self), 2, c, &cbuf, name)?;
                marshal::recycle_f32(abuf);
                marshal::recycle_f32(bbuf);
                marshal::recycle_f32(cbuf);
                Ok(Value::Unit)
            }
            // `wukong_sgemv[_parallel](a, x, y, m, n)` — matrix-times-vector `y[i] = Σ_j a[i,j]·x[j]`.
            // The interpreter marshals the abstract memory into real f32 buffers and calls the *serial*
            // runtime kernel as the oracle (rows independent → the parallel form is bit-identical), so
            // the differential gate stays exact despite the kernel's 8-wide reassociated row dot.
            I::Sgemv | I::SgemvParallel => {
                let a = ptr(args[0])?;
                let x = ptr(args[1])?;
                let y = ptr(args[2])?;
                let m = dim!(args[3]);
                let n = dim!(args[4]);
                let av = kernel_values(arena!(self), 0, a, m, n, name)?;
                let xv = kernel_values(arena!(self), 1, x, 1, n, name)?;
                let yv = kernel_values(arena!(self), 2, y, 1, m, name)?;
                // gemv.rs: `y` must not overlap `a` or `x`.
                marshal::check_output(
                    arena!(self),
                    name,
                    (2, y, m),
                    &[
                        ((0, a, av.len()), Overlap::Forbidden),
                        ((1, x, n), Overlap::Forbidden),
                    ],
                )?;
                let abuf = kernel_buffer(av, name, |v| Ok(v.kernel_float(name)? as f32))?;
                let xbuf = kernel_buffer(xv, name, |v| Ok(v.kernel_float(name)? as f32))?;
                let mut ybuf = kernel_buffer(yv, name, |_| Ok(0.0f32))?;
                // SAFETY: buffers are exactly m*n, n, m long — the kernel's contract.
                unsafe {
                    wukong_runtime::wukong_sgemv(
                        abuf.as_ptr(),
                        xbuf.as_ptr(),
                        ybuf.as_mut_ptr(),
                        m as i64,
                        n as i64,
                    );
                }
                for (t, &val) in ybuf.iter().enumerate() {
                    *self
                        .memory
                        .get_mut(y + t)
                        .ok_or("sgemv output out of bounds")? = Value::Float(val as f64);
                }
                Ok(Value::Unit)
            }
            // `wukong_sgemv_alpha[_parallel](a, x, y, m, n, alpha)` — the α-scaled GEMV
            // `y[i] = alpha · Σ_j a[i,j]·x[j]` (the decode-attention score `(K·q)·(1/√d)`). Marshals
            // exactly like the plain GEMV and calls the *serial* runtime kernel (bit-identical to the
            // parallel one — rows independent), so the differential gate stays exact.
            I::SgemvAlpha | I::SgemvAlphaParallel => {
                let a = ptr(args[0])?;
                let x = ptr(args[1])?;
                let y = ptr(args[2])?;
                let m = dim!(args[3]);
                let n = dim!(args[4]);
                let alpha = args[5].as_float() as f32;
                let av = kernel_values(arena!(self), 0, a, m, n, name)?;
                let xv = kernel_values(arena!(self), 1, x, 1, n, name)?;
                let yv = kernel_values(arena!(self), 2, y, 1, m, name)?;
                // gemv.rs: `y` must not overlap `a` or `x`.
                marshal::check_output(
                    arena!(self),
                    name,
                    (2, y, m),
                    &[
                        ((0, a, av.len()), Overlap::Forbidden),
                        ((1, x, n), Overlap::Forbidden),
                    ],
                )?;
                let abuf = kernel_buffer(av, name, |v| Ok(v.kernel_float(name)? as f32))?;
                let xbuf = kernel_buffer(xv, name, |v| Ok(v.kernel_float(name)? as f32))?;
                let mut ybuf = kernel_buffer(yv, name, |_| Ok(0.0f32))?;
                // SAFETY: buffers are exactly m*n, n, m long — the kernel's contract.
                unsafe {
                    wukong_runtime::wukong_sgemv_alpha(
                        abuf.as_ptr(),
                        xbuf.as_ptr(),
                        ybuf.as_mut_ptr(),
                        m as i64,
                        n as i64,
                        alpha,
                    );
                }
                for (t, &val) in ybuf.iter().enumerate() {
                    *self
                        .memory
                        .get_mut(y + t)
                        .ok_or("sgemv_alpha output out of bounds")? = Value::Float(val as f64);
                }
                Ok(Value::Unit)
            }
            // `wukong_sgevm_f32[_parallel](w, a, out, rows, cols, alpha)` — the vector·matrix product
            // `out[j] = alpha · Σ_i w[i]·a[i,j]` (the KV-decode attention read-out `out = scoresᵀ·V`).
            // Marshals w/a and calls the *serial* runtime kernel (bit-identical to the parallel one —
            // disjoint output-column stripes, each folding ascending i), so the gate stays exact.
            I::SgevmF32 | I::SgevmF32Parallel => {
                let w = ptr(args[0])?;
                let a = ptr(args[1])?;
                let out = ptr(args[2])?;
                let rows = dim!(args[3]);
                let cols = dim!(args[4]);
                let alpha = args[5].as_float() as f32;
                let av = kernel_values(arena!(self), 1, a, rows, cols, name)?;
                let xv = kernel_values(arena!(self), 0, w, 1, rows, name)?;
                let yv = kernel_values(arena!(self), 2, out, 1, cols, name)?;
                // gevm.rs: `out` must not overlap `w` or `a`.
                marshal::check_output(
                    arena!(self),
                    name,
                    (2, out, cols),
                    &[
                        ((0, w, rows), Overlap::Forbidden),
                        ((1, a, av.len()), Overlap::Forbidden),
                    ],
                )?;
                let abuf = kernel_buffer(av, name, |v| Ok(v.kernel_float(name)? as f32))?;
                let wbuf = kernel_buffer(xv, name, |v| Ok(v.kernel_float(name)? as f32))?;
                let mut obuf = kernel_buffer(yv, name, |_| Ok(0.0f32))?;
                // SAFETY: buffers are exactly rows, rows*cols, cols long — the kernel's contract.
                unsafe {
                    wukong_runtime::wukong_sgevm_f32(
                        wbuf.as_ptr(),
                        abuf.as_ptr(),
                        obuf.as_mut_ptr(),
                        rows as i64,
                        cols as i64,
                        alpha,
                    );
                }
                for (t, &val) in obuf.iter().enumerate() {
                    *self
                        .memory
                        .get_mut(out + t)
                        .ok_or("sgevm output out of bounds")? = Value::Float(val as f64);
                }
                Ok(Value::Unit)
            }
            // `wukong_i8gemm_nt[_parallel](a, b, c, m, k, n)` — the int8 quantized `nn.Linear` kernel
            // (`u8` activations × `i8` weights → `i32`, `C = A·Bᵀ`). The interpreter recovers each
            // operand byte (`as u8`/`as i8` takes the low 8 bits — bit-identical to the native buffer
            // regardless of how the abstract value was sign-extended) and writes the `i32` result,
            // calling the *serial* runtime kernel (bit-identical to the parallel one — rows are
            // independent), so the differential oracle stays exact. Integer math, so no rounding at all.
            I::I8gemmNt | I::I8gemmNtParallel => {
                let a = ptr(args[0])?;
                let b = ptr(args[1])?;
                let c = ptr(args[2])?;
                let m = dim!(args[3]);
                let k = dim!(args[4]);
                let n = dim!(args[5]);
                let av = kernel_values(arena!(self), 0, a, m, k, name)?;
                let bv = kernel_values(arena!(self), 1, b, n, k, name)?;
                let cv = kernel_values(arena!(self), 2, c, m, n, name)?;
                // The `i32` accumulator is written while the int8 operands are read — no overlap.
                marshal::check_output(
                    arena!(self),
                    name,
                    (2, c, cv.len()),
                    &[
                        ((0, a, av.len()), Overlap::Forbidden),
                        ((1, b, bv.len()), Overlap::Forbidden),
                    ],
                )?;
                let abuf = kernel_buffer(av, name, |v| Ok(v.as_int() as u8))?;
                let bbuf = kernel_buffer(bv, name, |v| Ok(v.as_int() as i8))?;
                let mut cbuf = kernel_buffer(cv, name, |_| Ok(0i32))?;
                // SAFETY: buffers are exactly m*k, n*k, m*n long — the kernel's contract.
                unsafe {
                    wukong_runtime::wukong_i8gemm_nt(
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
            // `wukong_i8gemm_nt_deq[_parallel](a, b, out, m, k, n, scale_a, scale_b, bias, act)` — the
            // *fused-dequant* int8 `nn.Linear`: `out_f32 = act((A_u8·B_i8ᵀ as f32)·scale_a·scale_b[j]
            // (+ bias[j]))`. Marshals the integer operands exactly like `wukong_i8gemm_nt` (each
            // `Value::Int`'s low byte → `u8`/`i8`), plus a per-column `scale_b` f32 array (length `n`),
            // an `f32` `scale_a` scalar (read like velem's affine scalars), and an *optional* `bias` f32
            // array (length `n`): an absent bias lowers to a `Ptr`-typed `ConstInt(0)` → a `Value::Int(0)`
            // here (distinct from a real array's `Value::Ptr`), so match the variant and pass a null
            // pointer (same convention as the affine-norm gamma/beta and the epilogue GEMM bias). The
            // result is f32 (rounding under dequant/activation), written back to the `out` buffer. Both
            // names marshal through the *serial* runtime kernel (bit-identical — rows independent).
            I::I8gemmNtDeq | I::I8gemmNtDeqParallel => {
                let a = ptr(args[0])?;
                let b = ptr(args[1])?;
                let out = ptr(args[2])?;
                let m = dim!(args[3]);
                let k = dim!(args[4]);
                let n = dim!(args[5]);
                let scale_a = args[6].as_float() as f32;
                let scale_b_idx = ptr(args[7])?;
                let bias_idx = match args[8] {
                    Value::Ptr(p) => Some(p),
                    _ => None,
                };
                let act = args[9].as_int() as i64;
                let av = kernel_values(arena!(self), 0, a, m, k, name)?;
                let bv = kernel_values(arena!(self), 1, b, n, k, name)?;
                let cv = kernel_values(arena!(self), 2, out, m, n, name)?;
                let sbv = kernel_values(arena!(self), 7, scale_b_idx, 1, n, name)?;
                let biasv = bias_idx
                    .map(|base| kernel_values(arena!(self), 8, base, 1, n, name))
                    .transpose()?;
                // The f32 output is written while the int8 operands, the per-channel scale and the
                // bias are read — it must overlap none of them.
                marshal::check_output(
                    arena!(self),
                    name,
                    (2, out, cv.len()),
                    &[
                        ((0, a, av.len()), Overlap::Forbidden),
                        ((1, b, bv.len()), Overlap::Forbidden),
                        ((7, scale_b_idx, n), Overlap::Forbidden),
                        (
                            (8, bias_idx.unwrap_or(0), biasv.map_or(0, <[Value]>::len)),
                            Overlap::Forbidden,
                        ),
                    ],
                )?;
                let abuf = kernel_buffer(av, name, |v| Ok(v.as_int() as u8))?;
                let bbuf = kernel_buffer(bv, name, |v| Ok(v.as_int() as i8))?;
                let sbbuf = kernel_buffer(sbv, name, |v| Ok(v.kernel_float(name)? as f32))?;
                let biasbuf = kernel_buffer(biasv.unwrap_or(&[]), name, |v| {
                    Ok(v.kernel_float(name)? as f32)
                })?;
                let bias_ptr = if bias_idx.is_some() {
                    biasbuf.as_ptr()
                } else {
                    std::ptr::null()
                };
                let mut obuf = kernel_buffer(cv, name, |_| Ok(0.0f32))?;
                // SAFETY: abuf/bbuf/obuf are exactly m*k, n*k, m*n long; scale_b (and bias when present)
                // are n long — the kernel's contract.
                unsafe {
                    wukong_runtime::wukong_i8gemm_nt_deq(
                        abuf.as_ptr(),
                        bbuf.as_ptr(),
                        obuf.as_mut_ptr(),
                        m as i64,
                        k as i64,
                        n as i64,
                        scale_a,
                        sbbuf.as_ptr(),
                        bias_ptr,
                        act,
                    );
                }
                for (t, &val) in obuf.iter().enumerate() {
                    *self
                        .memory
                        .get_mut(out + t)
                        .ok_or("i8gemm_deq output out of bounds")? = Value::Float(val as f64);
                }
                Ok(Value::Unit)
            }
            // `wukong_dequant_f32[_parallel](q, out, n, scale, op)` — int-input dequant
            // `out[j] = act((q[j] as f32)·scale)` over an `[i8]`/`[u8]`/`[i32]` array (the width is in
            // `op`). The interpreter reconstructs the input array's **real bytes** (its memory is 1 slot
            // per scalar leaf; each slot's low bits are the stored integer) into a width-matched buffer,
            // so the kernel — cast to `*const i8`/`u8`/`i32` internally — sees the identical layout the
            // native backend passes, then marshals the f32 result back. Integer→f32 is exact, so this is
            // bit-for-bit; the *serial* runtime kernel backs both names (elementwise → parallel is
            // identical). `wukong_runtime::DQ_*` name the width codes.
            I::DequantF32 | I::DequantF32Parallel => {
                let q = ptr(args[0])?;
                let out = ptr(args[1])?;
                let n = dim!(args[2]);
                let scale = args[3].as_float() as f32;
                let op = args[4].as_int() as i64;
                let width = op & (0xff << 8);
                let byte_width = if width == wukong_runtime::DQ_I32 {
                    4
                } else {
                    1
                };
                let byte_len = n
                    .checked_mul(byte_width)
                    .ok_or("dequant byte extent overflow")?;
                let qv = kernel_values(arena!(self), 0, q, 1, n, name)?;
                let ov = kernel_values(arena!(self), 1, out, 1, n, name)?;
                // An integer input and an f32 output are never the same array; any overlap is a
                // type pun the source loop does not perform.
                marshal::check_output(
                    arena!(self),
                    name,
                    (1, out, n),
                    &[((0, q, n), Overlap::Forbidden)],
                )?;
                // DQ_I32 scalar/tail loads dereference *const i32 in the runtime. A Vec<u8>
                // does not guarantee that alignment, so keep reconstructed words in typed storage.
                let qwords = if width == wukong_runtime::DQ_I32 {
                    kernel_buffer(qv, name, |v| Ok(v.as_int() as i32))?
                } else {
                    Vec::new()
                };
                let mut qbytes = kernel_scratch(
                    if width == wukong_runtime::DQ_I32 {
                        0
                    } else {
                        byte_len
                    },
                    name,
                )?;
                if width != wukong_runtime::DQ_I32 {
                    for value in qv {
                        qbytes.push(value.as_int() as u8);
                    }
                }
                let qptr = if width == wukong_runtime::DQ_I32 {
                    qwords.as_ptr().cast::<u8>()
                } else {
                    qbytes.as_ptr()
                };
                let mut obuf = kernel_buffer(ov, name, |_| Ok(0.0f32))?;
                // SAFETY: qptr references n width-matched, aligned elements; obuf is n long.
                unsafe {
                    wukong_runtime::wukong_dequant_f32(
                        qptr,
                        obuf.as_mut_ptr(),
                        n as i64,
                        scale,
                        op,
                    );
                }
                for (t, &val) in obuf.iter().enumerate() {
                    *self
                        .memory
                        .get_mut(out + t)
                        .ok_or("dequant output out of bounds")? = Value::Float(val as f64);
                }
                Ok(Value::Unit)
            }
            // `wukong_dequant_perchan_f32[_parallel](q, out, rows, cols, scale, op)` — per-channel
            // dequant `out[i*cols+j] = act((q[i*cols+j] as f32)·scale[j])` with a per-output-column
            // `scale[j]` (length `cols`). Marshals the integer input exactly like `wukong_dequant_f32`
            // (width-matched real bytes) plus the `cols`-long f32 scale vector; writes the f32 result
            // back. Rows are independent so the serial kernel backs both names.
            I::DequantPerchanF32 | I::DequantPerchanF32Parallel => {
                let q = ptr(args[0])?;
                let out = ptr(args[1])?;
                let rows = dim!(args[2]);
                let cols = dim!(args[3]);
                let scale_idx = ptr(args[4])?;
                let op = args[5].as_int() as i64;
                let width = op & (0xff << 8);
                let n = rows
                    .checked_mul(cols)
                    .ok_or("dequant_perchan operand extent overflow")?;
                let byte_width = if width == wukong_runtime::DQ_I32 {
                    4
                } else {
                    1
                };
                let byte_len = n
                    .checked_mul(byte_width)
                    .ok_or("dequant byte extent overflow")?;
                let qv = kernel_values(arena!(self), 0, q, 1, n, name)?;
                let ov = kernel_values(arena!(self), 1, out, 1, n, name)?;
                let sv = kernel_values(arena!(self), 4, scale_idx, 1, cols, name)?;
                // The per-channel scale is re-read for every row, so the output must overlap neither
                // it nor the integer input.
                marshal::check_output(
                    arena!(self),
                    name,
                    (1, out, n),
                    &[
                        ((0, q, n), Overlap::Forbidden),
                        ((4, scale_idx, cols), Overlap::Forbidden),
                    ],
                )?;
                // DQ_I32 scalar/tail loads dereference *const i32 in the runtime. A Vec<u8>
                // does not guarantee that alignment, so keep reconstructed words in typed storage.
                let qwords = if width == wukong_runtime::DQ_I32 {
                    kernel_buffer(qv, name, |v| Ok(v.as_int() as i32))?
                } else {
                    Vec::new()
                };
                let mut qbytes = kernel_scratch(
                    if width == wukong_runtime::DQ_I32 {
                        0
                    } else {
                        byte_len
                    },
                    name,
                )?;
                if width != wukong_runtime::DQ_I32 {
                    for value in qv {
                        qbytes.push(value.as_int() as u8);
                    }
                }
                let qptr = if width == wukong_runtime::DQ_I32 {
                    qwords.as_ptr().cast::<u8>()
                } else {
                    qbytes.as_ptr()
                };
                let sbuf = kernel_buffer(sv, name, |v| Ok(v.kernel_float(name)? as f32))?;
                let mut obuf = kernel_buffer(ov, name, |_| Ok(0.0f32))?;
                // SAFETY: qptr references n width-matched, aligned elements; obuf is n and sbuf cols long.
                unsafe {
                    wukong_runtime::wukong_dequant_perchan_f32(
                        qptr,
                        obuf.as_mut_ptr(),
                        rows as i64,
                        cols as i64,
                        sbuf.as_ptr(),
                        op,
                    );
                }
                for (t, &val) in obuf.iter().enumerate() {
                    *self
                        .memory
                        .get_mut(out + t)
                        .ok_or("dequant_perchan output out of bounds")? = Value::Float(val as f64);
                }
                Ok(Value::Unit)
            }
            // `wukong_sgemm_nt_epi[_parallel](a, b, c, m, k, n, beta, bias, act)` — the fused-epilogue
            // Linear (`C = act(A·Bᵀ + bias)`). Like the plain GEMM, the interpreter marshals operands
            // into real f32 buffers and calls the *serial* runtime kernel as the oracle. The native
            // `@parallel` variant is bit-identical to it (each C tile owned by one task, same per-(i,j)
            // accumulation order), so calling the serial form here keeps the two backends bit-for-bit
            // exact without spawning threads in the interpreter.
            I::SgemmNtEpi | I::SgemmNtEpiParallel => {
                let a = ptr(args[0])?;
                let b = ptr(args[1])?;
                let c = ptr(args[2])?;
                let m = dim!(args[3]);
                let k = dim!(args[4]);
                let n = dim!(args[5]);
                let beta = args[6].as_int() as i64;
                // An absent bias arrives as a `Value::Int(0)` (the null built as an integer 0) vs a
                // real array's `Value::Ptr` — distinguished by variant, like the affine-norm params.
                let bias_idx = match args[7] {
                    Value::Ptr(p) => Some(p),
                    _ => None,
                };
                let act = args[8].as_int() as i64;
                let matrix = |op: usize, base: usize, rows: usize, cols: usize| {
                    let len = rows
                        .checked_mul(cols)
                        .ok_or("sgemm_epi operand extent overflow")?;
                    let end = base
                        .checked_add(len)
                        .ok_or("sgemm_epi operand out of bounds")?;
                    let values = self
                        .memory
                        .get(base..end)
                        .ok_or("sgemm_epi operand out of bounds")?;
                    self.slots.live.check(name, op, base, len)?;
                    Ok::<_, String>(values)
                };
                // Check every arena range and allocation before allocating scratch, including the
                // unread beta-zero output.
                let av = matrix(0, a, m, k)?;
                let bv = matrix(1, b, n, k)?;
                let cv = matrix(2, c, m, n)?;
                let biasv = bias_idx.map(|base| matrix(7, base, 1, n)).transpose()?;
                // gemm.rs: `C` is written while `A`, `B` and the epilogue bias are read. (A
                // `beta = 1` residual reads `C` itself, which is in place by construction.)
                marshal::check_output(
                    arena!(self),
                    name,
                    (2, c, cv.len()),
                    &[
                        ((0, a, av.len()), Overlap::Forbidden),
                        ((1, b, bv.len()), Overlap::Forbidden),
                        (
                            (7, bias_idx.unwrap_or(0), biasv.map_or(0, <[Value]>::len)),
                            Overlap::Forbidden,
                        ),
                    ],
                )?;
                // Pooled scratch (`marshal::f32_scratch`): a GEMM inside a loop no longer allocates
                // three operand buffers per iteration.
                let read = |values: &[Value], input: bool| -> Result<Vec<f32>, String> {
                    let mut buffer = marshal::f32_scratch(values.len(), name)?;
                    for value in values {
                        buffer.push(if input {
                            value.kernel_float(name)? as f32
                        } else {
                            0.0
                        });
                    }
                    Ok(buffer)
                };
                let abuf = read(av, true)?;
                let bbuf = read(bv, true)?;
                let mut cbuf = read(cv, beta != 0)?;
                let biasbuf = match biasv {
                    Some(values) => read(values, true)?,
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
                let offloaded = self.accel.as_mut().and_then(|acc| {
                    acc.sgemm_nt_epi(&abuf, &bbuf, &mut cbuf, m, k, n, beta, bias_opt, act)
                });
                match offloaded {
                    Some(Ok(())) => {}
                    Some(Err(e)) => return Err(e),
                    // SAFETY: buffers are exactly m*k, n*k, m*n long; bias is null or n long — kernel contract.
                    None => unsafe {
                        wukong_runtime::wukong_sgemm_nt_epi(
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
                marshal::write_f32(arena_mut!(self), 2, c, &cbuf, name)?;
                marshal::recycle_f32(abuf);
                marshal::recycle_f32(bbuf);
                marshal::recycle_f32(cbuf);
                marshal::recycle_f32(biasbuf);
                Ok(Value::Unit)
            }
            // `wukong_sgemm_nt_alpha[_parallel](a, b, c, m, k, n, beta, alpha)` — the α-scaled Linear
            // (`C = alpha·(A·Bᵀ)`, the attention score matmul `QKᵀ/√d`). Like the plain GEMM, the
            // interpreter marshals operands into real f32 buffers and calls the *serial* runtime kernel
            // as the oracle (the parallel form is bit-identical — each C tile owned by one task, same
            // per-(i,j) accumulation order). `alpha` rides an f32 slot, read as a scalar value.
            I::SgemmNtAlpha | I::SgemmNtAlphaParallel => {
                let a = ptr(args[0])?;
                let b = ptr(args[1])?;
                let c = ptr(args[2])?;
                let m = dim!(args[3]);
                let k = dim!(args[4]);
                let n = dim!(args[5]);
                let beta = args[6].as_int() as i64;
                let alpha = args[7].as_float() as f32;
                let matrix = |op: usize, base: usize, rows: usize, cols: usize| {
                    let len = rows
                        .checked_mul(cols)
                        .ok_or("sgemm_alpha operand extent overflow")?;
                    let end = base
                        .checked_add(len)
                        .ok_or("sgemm_alpha operand out of bounds")?;
                    let values = self
                        .memory
                        .get(base..end)
                        .ok_or("sgemm_alpha operand out of bounds")?;
                    self.slots.live.check(name, op, base, len)?;
                    Ok::<_, String>(values)
                };
                // Check every arena range and allocation before allocating scratch, including the
                // unread beta-zero output.
                let av = matrix(0, a, m, k)?;
                let bv = matrix(1, b, n, k)?;
                let cv = matrix(2, c, m, n)?;
                // gemm.rs: `C` is written in tiled order while `A`/`B` are read — no overlap.
                marshal::check_output(
                    arena!(self),
                    name,
                    (2, c, cv.len()),
                    &[
                        ((0, a, av.len()), Overlap::Forbidden),
                        ((1, b, bv.len()), Overlap::Forbidden),
                    ],
                )?;
                // Pooled scratch (`marshal::f32_scratch`): a GEMM inside a loop no longer allocates
                // three operand buffers per iteration.
                let read = |values: &[Value], input: bool| -> Result<Vec<f32>, String> {
                    let mut buffer = marshal::f32_scratch(values.len(), name)?;
                    for value in values {
                        buffer.push(if input {
                            value.kernel_float(name)? as f32
                        } else {
                            0.0
                        });
                    }
                    Ok(buffer)
                };
                let abuf = read(av, true)?;
                let bbuf = read(bv, true)?;
                let mut cbuf = read(cv, beta != 0)?;
                // SAFETY: buffers are exactly m*k, n*k, m*n long — the kernel's contract.
                unsafe {
                    wukong_runtime::wukong_sgemm_nt_alpha(
                        abuf.as_ptr(),
                        bbuf.as_ptr(),
                        cbuf.as_mut_ptr(),
                        m as i64,
                        k as i64,
                        n as i64,
                        beta,
                        alpha,
                    );
                }
                marshal::write_f32(arena_mut!(self), 2, c, &cbuf, name)?;
                marshal::recycle_f32(abuf);
                marshal::recycle_f32(bbuf);
                marshal::recycle_f32(cbuf);
                Ok(Value::Unit)
            }
            // `wukong_vmath_f32(x, out, n, op)` — the 256-bit AVX2 elementwise transcendental kernel
            // (exp/log/tanh/sigmoid) the compiler lowers an `out[i] = f(x[i])` loop to. Marshal `n`
            // f32 from x, call the *identical* runtime kernel the native backend calls, write the
            // result back — so the differential oracle stays exact despite the kernel's wider lanes.
            // Reading all of x before writing out makes the in-place (x == out) case correct. The
            // `_parallel` twin is the same per-element map chunked across cores, so it marshals here
            // identically (bit-for-bit) — the differential gate sees one kernel.
            //
            // A SHIFTED overlap (`out = x + 1` through two pointer views) is the one input where
            // the copy and the source loop disagree: the loop feeds each store into a later read.
            // The compiler's run-time guard keeps a recognized call off such operands (the source
            // loop runs instead), so this can only be reached by a call that bypassed it; the
            // oracle still answers it the way the loop does — element by element on live memory
            // (`marshal::elementwise_live`) — so a native kernel that disagrees is the side the
            // differential gate names.
            I::VmathF32 | I::VmathF32Parallel => {
                let x = ptr(args[0])?;
                let out = ptr(args[1])?;
                let n = dim!(args[2]);
                let op = args[3].as_int() as i64;
                if marshal::shifted_overlap((out, n), &[(x, n)]) {
                    marshal::elementwise_live(
                        arena_mut!(self),
                        &[(0, x)],
                        (1, out),
                        n,
                        name,
                        |lane| {
                            let mut o = 0.0f32;
                            // SAFETY: one input and one output element — the kernel's contract at n = 1.
                            unsafe { wukong_runtime::wukong_vmath_f32(&lane[0], &mut o, 1, op) };
                            o
                        },
                    )?;
                    return Ok(Value::Unit);
                }
                // vmath.rs: "`x` and `out` may alias" — EXACTLY in place, which an elementwise map
                // computes identically out of place.
                marshal::check_output(
                    arena!(self),
                    name,
                    (1, out, n),
                    &[((0, x, n), Overlap::ExactOnly)],
                )?;
                let xbuf = marshal::read_f32(arena!(self), 0, x, n, name)?;
                let mut obuf = marshal::output_f32(arena!(self), 1, out, n, name)?;
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
                        wukong_runtime::wukong_vmath_f32(
                            xbuf.as_ptr(),
                            obuf.as_mut_ptr(),
                            n as i64,
                            op,
                        );
                    },
                }
                marshal::write_f32(arena_mut!(self), 1, out, &obuf, name)?;
                marshal::recycle_f32(xbuf);
                marshal::recycle_f32(obuf);
                Ok(Value::Unit)
            }
            // `wukong_transpose_f32[_parallel](src, dst, rows, cols)` — the cache-blocked transpose a
            // recognized `dst[j,i] = src[i,j]` nest lowers to. Marshal the `rows*cols` f32 out of src,
            // call the *serial* runtime kernel (a permutation, so serial == parallel bit-for-bit), write
            // the `cols*rows` result to dst.
            //
            // Reading all of src first does NOT make overlap robust — it makes overlap compute a
            // fourth thing. transpose.rs declares `src`/`dst` non-overlapping; an aliased 4x4
            // `tr(x, x)` printed `1 5 9 13 2 6 10 14 …` here (a true transpose, because of this
            // copy), `1 2 3 4 2 6 7 8 …` natively, and the scalar nest mirrors the upper triangle
            // down — three answers, and at >= 16x16 the blocked kernel adds a fourth. The recognizer
            // now versions an alias-uncertain transpose on a run-time overlap test whose overlapping
            // arm is the nest (`kernel_guard.rs` in wukong_mir_build), so this arm is reached only for
            // disjoint buffers, the copy is pure marshalling, and an overlapping call is refused.
            I::TransposeF32 | I::TransposeF32Parallel => {
                let src = ptr(args[0])?;
                let dst = ptr(args[1])?;
                let rows = dim!(args[2]);
                let cols = dim!(args[3]);
                let input = kernel_values(arena!(self), 0, src, rows, cols, name)?;
                let output = kernel_values(arena!(self), 1, dst, rows, cols, name)?;
                // transpose.rs: `src`/`dst` non-overlapping.
                marshal::check_output(
                    arena!(self),
                    name,
                    (1, dst, output.len()),
                    &[((0, src, input.len()), Overlap::Forbidden)],
                )?;
                let sbuf = kernel_buffer(input, name, |v| Ok(v.kernel_float(name)? as f32))?;
                let mut dbuf = kernel_buffer(output, name, |_| Ok(0.0f32))?;
                // SAFETY: sbuf is rows*cols, dbuf is cols*rows = rows*cols f32 — the kernel's contract.
                unsafe {
                    wukong_runtime::wukong_transpose_f32(
                        sbuf.as_ptr(),
                        dbuf.as_mut_ptr(),
                        rows as i64,
                        cols as i64,
                    );
                }
                for (t, &val) in dbuf.iter().enumerate() {
                    *self
                        .memory
                        .get_mut(dst + t)
                        .ok_or("transpose output out of bounds")? = Value::Float(val as f64);
                }
                Ok(Value::Unit)
            }
            // `wukong_{max,avg}pool2d_f32[_parallel](x, out, channels, h, w, kh, kw, sh, sw)` — 2D
            // max/avg pooling over a [channels, h, w] row-major input (no padding) a recognized pooling
            // nest lowers to. Marshal `channels*h*w` f32 from x, call the *serial* runtime kernel
            // (bit-identical to the parallel one — channels independent, max idempotent, avg sum order
            // fixed), size the output through the shared runtime shape helper. Invalid windows
            // and nonpositive dimensions write nothing. An `out` overlapping `x` is outside
            // pool2d.rs's contract and is refused rather than answered from a copy of `x`.
            I::Maxpool2dF32
            | I::Maxpool2dF32Parallel
            | I::Avgpool2dF32
            | I::Avgpool2dF32Parallel => {
                let x = ptr(args[0])?;
                let out = ptr(args[1])?;
                // Empty/invalid dimensions are no-ops, including zero channels. Check
                // before narrowing: interpreter integers are wider than the runtime ABI.
                if args[2..9].iter().any(|v| v.as_int() <= 0) {
                    return Ok(Value::Unit);
                }
                let mut dims = [0i64; 7];
                for (dim, arg) in dims.iter_mut().zip(&args[2..9]) {
                    *dim = i64::try_from(arg.as_int()).map_err(|_| "pool2d extent exceeds i64")?;
                }
                let [channels, h, w, kh, kw, sh, sw] = dims;
                let Some((oh, ow)) = wukong_runtime::pool2d_output_dims(h, w, kh, kw, sh, sw)
                else {
                    return Ok(Value::Unit);
                };
                let c = usize::try_from(channels).map_err(|_| "pool2d channel extent overflow")?;
                let n_in = c
                    .checked_mul(h as usize)
                    .and_then(|n| n.checked_mul(w as usize))
                    .ok_or("pool2d operand extent overflow")?;
                let out_len = c
                    .checked_mul(oh)
                    .and_then(|n| n.checked_mul(ow))
                    .ok_or("pool2d output extent overflow")?;
                let x_end = x.checked_add(n_in).ok_or("pool2d operand out of bounds")?;
                let input = self
                    .memory
                    .get(x..x_end)
                    .ok_or("pool2d operand out of bounds")?;
                let out_end = out
                    .checked_add(out_len)
                    .ok_or("pool2d output out of bounds")?;
                self.memory
                    .get(out..out_end)
                    .ok_or("pool2d output out of bounds")?;
                // pool2d.rs: `x`/`out` non-overlapping; each inside its own allocation.
                marshal::check_output(
                    arena!(self),
                    name,
                    (1, out, out_len),
                    &[((0, x, n_in), Overlap::Forbidden)],
                )?;
                let mut xbuf = Vec::new();
                xbuf.try_reserve_exact(n_in)
                    .map_err(|_| "pool2d scratch allocation failed")?;
                for value in input {
                    xbuf.push(value.kernel_float(name)? as f32);
                }
                let mut obuf = Vec::new();
                obuf.try_reserve_exact(out_len)
                    .map_err(|_| "pool2d scratch allocation failed")?;
                obuf.resize(out_len, 0.0f32);
                let kernel: unsafe extern "C" fn(
                    *const f32,
                    *mut f32,
                    i64,
                    i64,
                    i64,
                    i64,
                    i64,
                    i64,
                    i64,
                ) = if name.starts_with("wukong_avgpool2d") {
                    wukong_runtime::wukong_avgpool2d_f32
                } else {
                    wukong_runtime::wukong_maxpool2d_f32
                };
                // SAFETY: xbuf is channels*h*w, obuf is channels*oh*ow f32 — the kernel's contract.
                unsafe {
                    kernel(
                        xbuf.as_ptr(),
                        obuf.as_mut_ptr(),
                        channels,
                        h,
                        w,
                        kh,
                        kw,
                        sh,
                        sw,
                    );
                }
                for (t, &val) in obuf.iter().enumerate() {
                    *self
                        .memory
                        .get_mut(out + t)
                        .ok_or("pool2d output out of bounds")? = Value::Float(val as f64);
                }
                Ok(Value::Unit)
            }
            // `wukong_transpose_u16[_parallel](src, dst, rows, cols)` — the 16-bit (bf16/f16) transpose.
            // A transpose is a permutation, so the interpreter moves the `Value`s directly (precision-
            // agnostic): the bf16/f16 elements are stored as their rounded `Value::Float`, which the
            // native u16 kernel's moved bits decode back to the same value — so interp == native without
            // any bf16-vs-f16 bit conversion. An overlapping `src`/`dst` is refused, as for the f32 entry.
            I::TransposeU16 | I::TransposeU16Parallel => {
                let src = ptr(args[0])?;
                let dst = ptr(args[1])?;
                let rows = dim!(args[2]);
                let cols = dim!(args[3]);
                let input = kernel_values(arena!(self), 0, src, rows, cols, name)?;
                kernel_values(arena!(self), 1, dst, rows, cols, name)?;
                // transpose.rs: `src`/`dst` non-overlapping (the same contract as the f32 entry).
                marshal::check_output(
                    arena!(self),
                    name,
                    (1, dst, input.len()),
                    &[((0, src, input.len()), Overlap::Forbidden)],
                )?;
                let buf = kernel_buffer(input, name, |value| {
                    value.kernel_float(name)?;
                    Ok(value)
                })?;
                for i in 0..rows {
                    for j in 0..cols {
                        *self
                            .memory
                            .get_mut(dst + j * rows + i)
                            .ok_or("transpose output out of bounds")? = buf[i * cols + j];
                    }
                }
                Ok(Value::Unit)
            }
            // `wukong_col{sum,max,min}_f32[_parallel](x, out, rows, cols)` — the SIMD column reductions
            // (`out[j] = Σ/max/min_i x[i*cols+j]`) a recognized column-reduce nest lowers to. Marshal the
            // `rows*cols` f32 from x, call the *serial* runtime kernel (bit-identical to the parallel
            // one — each column folded in the same i-order, disjoint stripes), write the `cols`-long
            // result to out.
            I::ColsumF32
            | I::ColsumF32Parallel
            | I::ColmaxF32
            | I::ColmaxF32Parallel
            | I::ColminF32
            | I::ColminF32Parallel
            | I::ColmaxabsF32
            | I::ColmaxabsF32Parallel
            | I::ColmeanF32
            | I::ColmeanF32Parallel
            | I::ColsumsqF32
            | I::ColsumsqF32Parallel
            | I::Coll2F32
            | I::Coll2F32Parallel
            | I::ColrmsF32
            | I::ColrmsF32Parallel => {
                let x = ptr(args[0])?;
                let out = ptr(args[1])?;
                let rows = dim!(args[2]);
                let cols = dim!(args[3]);
                let n = marshal::span(rows, cols, name)?;
                // colreduce.rs: `x`/`out` non-overlapping.
                marshal::check_output(
                    arena!(self),
                    name,
                    (1, out, cols),
                    &[((0, x, n), Overlap::Forbidden)],
                )?;
                let xbuf = marshal::read_f32(arena!(self), 0, x, n, name)?;
                let mut obuf = marshal::output_f32(arena!(self), 1, out, cols, name)?;
                // The serial kernel matching the name (bit-identical to its `_parallel` twin).
                let kernel: unsafe extern "C" fn(*const f32, *mut f32, i64, i64) =
                    if name.starts_with("wukong_colmaxabs") {
                        wukong_runtime::wukong_colmaxabs_f32
                    } else if name.starts_with("wukong_colmax") {
                        wukong_runtime::wukong_colmax_f32
                    } else if name.starts_with("wukong_colmin") {
                        wukong_runtime::wukong_colmin_f32
                    } else if name.starts_with("wukong_colmean") {
                        wukong_runtime::wukong_colmean_f32
                    } else if name.starts_with("wukong_colsumsq") {
                        wukong_runtime::wukong_colsumsq_f32
                    } else if name.starts_with("wukong_coll2") {
                        wukong_runtime::wukong_coll2_f32
                    } else if name.starts_with("wukong_colrms") {
                        wukong_runtime::wukong_colrms_f32
                    } else {
                        wukong_runtime::wukong_colsum_f32
                    };
                // SAFETY: xbuf is rows*cols, obuf is cols f32 — the kernel's contract.
                unsafe {
                    kernel(xbuf.as_ptr(), obuf.as_mut_ptr(), rows as i64, cols as i64);
                }
                marshal::write_f32(arena_mut!(self), 1, out, &obuf, name)?;
                marshal::recycle_f32(xbuf);
                marshal::recycle_f32(obuf);
                Ok(Value::Unit)
            }
            // Scalar math ABI: shared full-range trig, including exceptional SIMD lanes.
            I::RtTrigF32 => {
                if args.len() != 2 {
                    return Err("wukong_rt_trig_f32 expects two arguments".into());
                }
                Ok(Value::Float(wukong_runtime::wukong_rt_trig_f32(
                    args[0].as_float() as f32,
                    args[1].as_int() as i64,
                ) as f64))
            }
            // The double-precision transcendental library an `f64` `exp`/`log`/`sin`/… lowers to:
            // the identical pure-Rust function the JIT binds and the AOT link resolves, called on
            // the f64 the slot holds (no narrowing), so the oracle is the native result bit-for-bit.
            I::RtMathF64 => {
                if args.len() != 2 {
                    return Err("wukong_rt_math_f64 expects two arguments".into());
                }
                Ok(Value::Float(wukong_runtime::wukong_rt_math_f64(
                    args[0].as_float(),
                    args[1].as_int() as i64,
                )))
            }
            I::RtMath2F64 => {
                if args.len() != 3 {
                    return Err("wukong_rt_math2_f64 expects three arguments".into());
                }
                Ok(Value::Float(wukong_runtime::wukong_rt_math2_f64(
                    args[0].as_float(),
                    args[1].as_float(),
                    args[2].as_int() as i64,
                )))
            }
            // `wukong_vmath2_f32(x, y, out, n, op)` — the two-input 256-bit kernel (pow/atan2/hypot)
            // an `out[i] = f(x[i], y[i])` loop lowers to. Marshal `n` f32 from x AND y, call the
            // *identical* runtime kernel the native backend calls, write the result back — so the
            // differential oracle stays exact. Read both inputs before writing (in-place safe).
            I::Vmath2F32 => {
                let x = ptr(args[0])?;
                let y = ptr(args[1])?;
                let out = ptr(args[2])?;
                let n = dim!(args[3]);
                let op = args[4].as_int() as i64;
                // A shifted overlap runs the source loop's order on live memory (see the
                // `wukong_vmath_f32` arm); exact in place equals out of place and takes the copy.
                if marshal::shifted_overlap((out, n), &[(x, n), (y, n)]) {
                    marshal::elementwise_live(
                        arena_mut!(self),
                        &[(0, x), (1, y)],
                        (2, out),
                        n,
                        name,
                        |lane| {
                            let mut o = 0.0f32;
                            // SAFETY: one element per operand — the kernel's contract at n = 1.
                            unsafe {
                                wukong_runtime::wukong_vmath2_f32(&lane[0], &lane[1], &mut o, 1, op)
                            };
                            o
                        },
                    )?;
                    return Ok(Value::Unit);
                }
                // Elementwise: either input may be EXACTLY the output.
                marshal::check_output(
                    arena!(self),
                    name,
                    (2, out, n),
                    &[
                        ((0, x, n), Overlap::ExactOnly),
                        ((1, y, n), Overlap::ExactOnly),
                    ],
                )?;
                let xbuf = marshal::read_f32(arena!(self), 0, x, n, name)?;
                let ybuf = marshal::read_f32(arena!(self), 1, y, n, name)?;
                let mut obuf = marshal::output_f32(arena!(self), 2, out, n, name)?;
                // SAFETY: xbuf/ybuf/obuf are exactly n f32 long — the kernel's contract.
                unsafe {
                    wukong_runtime::wukong_vmath2_f32(
                        xbuf.as_ptr(),
                        ybuf.as_ptr(),
                        obuf.as_mut_ptr(),
                        n as i64,
                        op,
                    );
                }
                marshal::write_f32(arena_mut!(self), 2, out, &obuf, name)?;
                marshal::recycle_f32(xbuf);
                marshal::recycle_f32(ybuf);
                marshal::recycle_f32(obuf);
                Ok(Value::Unit)
            }
            // `wukong_softmax_bwd_f32[_parallel](y, dy, dx, rows, cols)` — the fused softmax-backward
            // (`dx = y·(dy − Σ y·dy)`) a recognized batched nest lowers to. Marshal `rows*cols` f32 from
            // y AND dy, call the *serial* runtime kernel (bit-identical to the parallel one — rows are
            // independent), write `dx`. Read both inputs first, which makes the `dx == dy` aliasing
            // softmax_bwd.rs explicitly ALLOWS safe here too. It does NOT make the aliasing that file
            // FORBIDS safe: "`dx` must not overlap `y` (its row is read during the apply)". Under
            // `dx == y` this arm computes an out-of-place result and native computes something else,
            // so the recognizer declines that case at lowering (`provably_disjoint(nest.y, nest.dx)`)
            // and only the permitted aliasing reaches here.
            I::SoftmaxBwdF32 | I::SoftmaxBwdF32Parallel => {
                let y = ptr(args[0])?;
                let dy = ptr(args[1])?;
                let dx = ptr(args[2])?;
                let rows = dim!(args[3]);
                let cols = dim!(args[4]);
                let n = marshal::span(rows, cols, name)?;
                // softmax_bwd.rs: `dx` may be EXACTLY `dy`, and must not overlap `y` at all.
                marshal::check_output(
                    arena!(self),
                    name,
                    (2, dx, n),
                    &[
                        ((0, y, n), Overlap::Forbidden),
                        ((1, dy, n), Overlap::ExactOnly),
                    ],
                )?;
                let ybuf = marshal::read_f32(arena!(self), 0, y, n, name)?;
                let dybuf = marshal::read_f32(arena!(self), 1, dy, n, name)?;
                let mut dxbuf = marshal::output_f32(arena!(self), 2, dx, n, name)?;
                // SAFETY: ybuf/dybuf/dxbuf are exactly rows*cols f32 — the kernel's contract.
                unsafe {
                    wukong_runtime::wukong_softmax_bwd_f32(
                        ybuf.as_ptr(),
                        dybuf.as_ptr(),
                        dxbuf.as_mut_ptr(),
                        rows as i64,
                        cols as i64,
                    );
                }
                marshal::write_f32(arena_mut!(self), 2, dx, &dxbuf, name)?;
                marshal::recycle_f32(ybuf);
                marshal::recycle_f32(dybuf);
                marshal::recycle_f32(dxbuf);
                Ok(Value::Unit)
            }
            // `wukong_lrscan_f32[_parallel](a, b, out, rows, cols)` — the first-order linear-recurrence
            // / selective scan (SSM/Mamba/EMA) `out[r,t] = a[r,t]·h_{t-1} + b[r,t]`, `h_{-1}=0` per row.
            // Marshal the `rows*cols` f32 gate `a` and input `b`, call the *serial* kernel (bit-identical
            // to the parallel one — rows independent, no cross-row combine), write the `rows*cols` result.
            I::LrscanF32 | I::LrscanF32Parallel => {
                let a = ptr(args[0])?;
                let b = ptr(args[1])?;
                let out = ptr(args[2])?;
                let rows = dim!(args[3]);
                let cols = dim!(args[4]);
                let n = marshal::span(rows, cols, name)?;
                // lrscan.rs: `out` may be EXACTLY `a` or `b` (in place per row), nothing else.
                marshal::check_output(
                    arena!(self),
                    name,
                    (2, out, n),
                    &[
                        ((0, a, n), Overlap::ExactOnly),
                        ((1, b, n), Overlap::ExactOnly),
                    ],
                )?;
                let abuf = marshal::read_f32(arena!(self), 0, a, n, name)?;
                let bbuf = marshal::read_f32(arena!(self), 1, b, n, name)?;
                let mut outbuf = marshal::output_f32(arena!(self), 2, out, n, name)?;
                // SAFETY: abuf/bbuf/outbuf are exactly rows*cols f32 — the kernel's contract.
                unsafe {
                    wukong_runtime::wukong_lrscan_f32(
                        abuf.as_ptr(),
                        bbuf.as_ptr(),
                        outbuf.as_mut_ptr(),
                        rows as i64,
                        cols as i64,
                    );
                }
                marshal::write_f32(arena_mut!(self), 2, out, &outbuf, name)?;
                marshal::recycle_f32(abuf);
                marshal::recycle_f32(bbuf);
                marshal::recycle_f32(outbuf);
                Ok(Value::Unit)
            }
            // `wukong_rmsnorm_bwd_f32[_parallel](x, dy, gamma, dx, rows, cols, eps_bits)` — the fused
            // RMSNorm input-gradient (`dx = r·(g − x·r²·Σg·x/C)`, `g = dy·γ`) a recognized batched nest
            // lowers to. Marshal `rows*cols` f32 from x AND dy plus the `cols`-long gamma, call the
            // *serial* runtime kernel (bit-identical to the parallel one — rows are independent), write
            // `dx`. The EXACT in-place `dx == dy` the RMSNorm kernel allows is equal to out of place
            // (each row's `dy` is read before that row's `dx` is written). The aliasing it FORBIDS is
            // `dx` over `x` (rmsnorm_bwd.rs: "each row's `x` is read during the apply after the
            // reductions"; layernorm_bwd.rs forbids `dx` over `dy` as well) — the standard way to
            // halve a backward pass's memory. The recognizer versions it to the source loop, and an
            // overlapping call that reaches this arm anyway is refused rather than copied.
            // The recognizer always binds a real gamma, but a null (`Value::Int(0)`) is still handled by
            // variant, mirroring the affine-norm convention.
            // LayerNorm backward shares the identical 7-arg ABI and marshalling; the kernel fn is
            // selected by name below. (Both serial/parallel names route through the serial kernel —
            // bit-identical, rows independent.)
            I::RmsnormBwdF32
            | I::RmsnormBwdF32Parallel
            | I::LayernormBwdF32
            | I::LayernormBwdF32Parallel => {
                let x = ptr(args[0])?;
                let dy = ptr(args[1])?;
                let gamma_idx = match args[2] {
                    Value::Ptr(p) => Some(p),
                    _ => None,
                };
                let dx = ptr(args[3])?;
                let rows = dim!(args[4]);
                let cols = dim!(args[5]);
                let eps_bits = args[6].as_int() as i64;
                let n = marshal::span(rows, cols, name)?;
                let layernorm = name.starts_with("wukong_layernorm_bwd");
                // rmsnorm_bwd.rs: `dx` may be EXACTLY `dy`; layernorm_bwd.rs: `dx` must not overlap
                // `dy` either. Neither permits `dx` over `x` or over `gamma`.
                let dy_rule = if layernorm {
                    Overlap::Forbidden
                } else {
                    Overlap::ExactOnly
                };
                marshal::check_output(
                    arena!(self),
                    name,
                    (3, dx, n),
                    &[
                        ((0, x, n), Overlap::Forbidden),
                        ((1, dy, n), dy_rule),
                        (
                            (
                                2,
                                gamma_idx.unwrap_or(0),
                                if gamma_idx.is_some() { cols } else { 0 },
                            ),
                            Overlap::Forbidden,
                        ),
                    ],
                )?;
                let xbuf = marshal::read_f32(arena!(self), 0, x, n, name)?;
                let dybuf = marshal::read_f32(arena!(self), 1, dy, n, name)?;
                let gbuf = match gamma_idx {
                    Some(g) => marshal::read_f32(arena!(self), 2, g, cols, name)?,
                    None => Vec::new(),
                };
                let gptr = if gamma_idx.is_some() {
                    gbuf.as_ptr()
                } else {
                    std::ptr::null()
                };
                let mut dxbuf = marshal::output_f32(arena!(self), 3, dx, n, name)?;
                // SAFETY: xbuf/dybuf/dxbuf are exactly rows*cols f32; gamma (when present) is cols —
                // the kernel's contract. The kernel fn is chosen by name (rmsnorm vs layernorm).
                let kernel = if layernorm {
                    wukong_runtime::wukong_layernorm_bwd_f32
                } else {
                    wukong_runtime::wukong_rmsnorm_bwd_f32
                };
                unsafe {
                    kernel(
                        xbuf.as_ptr(),
                        dybuf.as_ptr(),
                        gptr,
                        dxbuf.as_mut_ptr(),
                        rows as i64,
                        cols as i64,
                        eps_bits,
                    );
                }
                marshal::write_f32(arena_mut!(self), 3, dx, &dxbuf, name)?;
                marshal::recycle_f32(xbuf);
                marshal::recycle_f32(dybuf);
                marshal::recycle_f32(gbuf);
                marshal::recycle_f32(dxbuf);
                Ok(Value::Unit)
            }
            // `wukong_xent_fwd_f32[_parallel](x, target, loss, rows, cols)` — the fused softmax
            // cross-entropy forward loss (`loss[r] = m + log(Σexp(x[r,·]−m)) − x[r, target[r]]`) a
            // recognized batched nest lowers to. Marshal `rows*cols` f32 from x, the `rows` i32 labels
            // from target (each `Value::Int`'s value), call the *serial* kernel (bit-identical to the
            // parallel one — rows independent), write the `rows`-long loss vector.
            I::XentFwdF32 | I::XentFwdF32Parallel => {
                let x = ptr(args[0])?;
                let target = ptr(args[1])?;
                let loss = ptr(args[2])?;
                let rows = dim!(args[3]);
                let cols = dim!(args[4]);
                let n = marshal::span(rows, cols, name)?;
                // The per-row loss must not overlap the logits or the `i32` targets.
                marshal::check_output(
                    arena!(self),
                    name,
                    (2, loss, rows),
                    &[
                        ((0, x, n), Overlap::Forbidden),
                        ((1, target, rows), Overlap::Forbidden),
                    ],
                )?;
                let xbuf = marshal::read_f32(arena!(self), 0, x, n, name)?;
                let tbuf = marshal::read_i32(arena!(self), 1, target, rows, name)?;
                let mut lbuf = marshal::output_f32(arena!(self), 2, loss, rows, name)?;
                // SAFETY: xbuf is rows*cols f32; tbuf/lbuf are rows long — the kernel's contract.
                unsafe {
                    wukong_runtime::wukong_xent_fwd_f32(
                        xbuf.as_ptr(),
                        tbuf.as_ptr(),
                        lbuf.as_mut_ptr(),
                        rows as i64,
                        cols as i64,
                    );
                }
                marshal::write_f32(arena_mut!(self), 2, loss, &lbuf, name)?;
                marshal::recycle_f32(xbuf);
                marshal::recycle_f32(lbuf);
                Ok(Value::Unit)
            }
            // `wukong_xent_bwd_f32[_parallel](x, target, dx, rows, cols)` — the softmax cross-entropy
            // backward (`dx = softmax(x) − onehot(target)`). Marshal `rows*cols` f32 from x and `rows`
            // i32 labels, call the *serial* kernel (bit-identical to parallel), write the `rows*cols` dx.
            I::XentBwdF32 | I::XentBwdF32Parallel => {
                let x = ptr(args[0])?;
                let target = ptr(args[1])?;
                let dx = ptr(args[2])?;
                let rows = dim!(args[3]);
                let cols = dim!(args[4]);
                let n = marshal::span(rows, cols, name)?;
                // xent_bwd.rs: `dx` may be EXACTLY `x`; the `i32` targets must not overlap it.
                marshal::check_output(
                    arena!(self),
                    name,
                    (2, dx, n),
                    &[
                        ((0, x, n), Overlap::ExactOnly),
                        ((1, target, rows), Overlap::Forbidden),
                    ],
                )?;
                let xbuf = marshal::read_f32(arena!(self), 0, x, n, name)?;
                let tbuf = marshal::read_i32(arena!(self), 1, target, rows, name)?;
                let mut dxbuf = marshal::output_f32(arena!(self), 2, dx, n, name)?;
                // SAFETY: xbuf/dxbuf are rows*cols f32; tbuf is rows — the kernel's contract.
                unsafe {
                    wukong_runtime::wukong_xent_bwd_f32(
                        xbuf.as_ptr(),
                        tbuf.as_ptr(),
                        dxbuf.as_mut_ptr(),
                        rows as i64,
                        cols as i64,
                    );
                }
                marshal::write_f32(arena_mut!(self), 2, dx, &dxbuf, name)?;
                marshal::recycle_f32(xbuf);
                marshal::recycle_f32(dxbuf);
                Ok(Value::Unit)
            }
            // `wukong_rope_f32[_parallel](x, inv_freq, out, rows, half)` — the inline-sincos rotary
            // position embedding a recognized RoPE nest lowers to. Marshal `rows*2*half` f32 from x and
            // `half` f32 from inv_freq, call the *serial* kernel (bit-identical to the parallel one —
            // rows independent), write `out` (reading x first, so in-place x==out is correct).
            I::RopeF32 | I::RopeF32Parallel => {
                let x = ptr(args[0])?;
                let inv_freq = ptr(args[1])?;
                let out = ptr(args[2])?;
                let rows = dim!(args[3]);
                let half = dim!(args[4]);
                let n = marshal::span(rows, marshal::span(2, half, name)?, name)?;
                // rope.rs: `out` may be EXACTLY `x`; the frequency table must not overlap it.
                marshal::check_output(
                    arena!(self),
                    name,
                    (2, out, n),
                    &[
                        ((0, x, n), Overlap::ExactOnly),
                        ((1, inv_freq, half), Overlap::Forbidden),
                    ],
                )?;
                let xbuf = marshal::read_f32(arena!(self), 0, x, n, name)?;
                let fbuf = marshal::read_f32(arena!(self), 1, inv_freq, half, name)?;
                let mut obuf = marshal::output_f32(arena!(self), 2, out, n, name)?;
                // SAFETY: xbuf/obuf are rows*2*half f32; fbuf is half — the kernel's contract.
                unsafe {
                    wukong_runtime::wukong_rope_f32(
                        xbuf.as_ptr(),
                        fbuf.as_ptr(),
                        obuf.as_mut_ptr(),
                        rows as i64,
                        half as i64,
                    );
                }
                marshal::write_f32(arena_mut!(self), 2, out, &obuf, name)?;
                marshal::recycle_f32(xbuf);
                marshal::recycle_f32(fbuf);
                marshal::recycle_f32(obuf);
                Ok(Value::Unit)
            }
            // `wukong_rope_bwd_f32[_parallel](g, inv_freq, dx, rows, half)` — the RoPE backward (the
            // transpose/inverse rotation). Identical marshalling to the forward, only the kernel differs.
            I::RopeBwdF32 | I::RopeBwdF32Parallel => {
                let g = ptr(args[0])?;
                let inv_freq = ptr(args[1])?;
                let dx = ptr(args[2])?;
                let rows = dim!(args[3]);
                let half = dim!(args[4]);
                let n = marshal::span(rows, marshal::span(2, half, name)?, name)?;
                // rope_bwd.rs: `dx` may be EXACTLY `g`; the frequency table must not overlap it.
                marshal::check_output(
                    arena!(self),
                    name,
                    (2, dx, n),
                    &[
                        ((0, g, n), Overlap::ExactOnly),
                        ((1, inv_freq, half), Overlap::Forbidden),
                    ],
                )?;
                let gbuf = marshal::read_f32(arena!(self), 0, g, n, name)?;
                let fbuf = marshal::read_f32(arena!(self), 1, inv_freq, half, name)?;
                let mut obuf = marshal::output_f32(arena!(self), 2, dx, n, name)?;
                // SAFETY: gbuf/obuf are rows*2*half f32; fbuf is half — the kernel's contract.
                unsafe {
                    wukong_runtime::wukong_rope_bwd_f32(
                        gbuf.as_ptr(),
                        fbuf.as_ptr(),
                        obuf.as_mut_ptr(),
                        rows as i64,
                        half as i64,
                    );
                }
                marshal::write_f32(arena_mut!(self), 2, dx, &obuf, name)?;
                marshal::recycle_f32(gbuf);
                marshal::recycle_f32(fbuf);
                marshal::recycle_f32(obuf);
                Ok(Value::Unit)
            }
            // `wukong_logsumexp_f32[_parallel](x, out, rows, cols)` — the batched log-partition a
            // recognized `out[r] = m + log(Σexp(x[r,·]−m))` nest lowers to. Marshal `rows*cols` f32 from
            // x, call the *serial* kernel (bit-identical to the parallel one — rows independent), write
            // the `rows`-long out vector.
            // Log-sum-exp and row-entropy share the (x, out, rows, cols) shape — `rows*cols` f32 in,
            // a `rows`-long scalar out; the kernel fn is chosen by name.
            I::LogsumexpF32 | I::LogsumexpF32Parallel | I::EntropyF32 | I::EntropyF32Parallel => {
                let x = ptr(args[0])?;
                let out = ptr(args[1])?;
                let rows = dim!(args[2]);
                let cols = dim!(args[3]);
                let n = marshal::span(rows, cols, name)?;
                // The `rows`-long result must not overlap the input it is folded from.
                marshal::check_output(
                    arena!(self),
                    name,
                    (1, out, rows),
                    &[((0, x, n), Overlap::Forbidden)],
                )?;
                let xbuf = marshal::read_f32(arena!(self), 0, x, n, name)?;
                let mut obuf = marshal::output_f32(arena!(self), 1, out, rows, name)?;
                let kernel = if name.starts_with("wukong_entropy") {
                    wukong_runtime::wukong_entropy_f32
                } else {
                    wukong_runtime::wukong_logsumexp_f32
                };
                // SAFETY: xbuf is rows*cols f32; obuf is rows — the kernel's contract.
                unsafe {
                    kernel(xbuf.as_ptr(), obuf.as_mut_ptr(), rows as i64, cols as i64);
                }
                marshal::write_f32(arena_mut!(self), 1, out, &obuf, name)?;
                marshal::recycle_f32(xbuf);
                marshal::recycle_f32(obuf);
                Ok(Value::Unit)
            }
            // `wukong_{row,col}arg{max,min}_i32[_parallel](x, out, rows, cols)` — the per-row /
            // per-column argmax/argmin a recognized classification-head / axis-0 top-1 nest lowers to.
            // Marshal `rows*cols` f32 from x, call the *serial* kernel (bit-identical to the parallel one
            // — rows/columns independent), write the index vector back as `Value::Int` (one of the
            // two output-buffer families that write integers rather than floats — `wukong_i8gemm_nt`
            // and its `i32` accumulator is the other). row-arg writes `rows` indices,
            // col-arg writes `cols`; the kernel and the output length are selected by name.
            I::RowargmaxI32
            | I::RowargmaxI32Parallel
            | I::RowargminI32
            | I::RowargminI32Parallel
            | I::ColargmaxI32
            | I::ColargmaxI32Parallel
            | I::ColargminI32
            | I::ColargminI32Parallel => {
                let x = ptr(args[0])?;
                let out = ptr(args[1])?;
                let rows = dim!(args[2]);
                let cols = dim!(args[3]);
                let n = marshal::span(rows, cols, name)?;
                let is_col = name.contains("colarg");
                let is_min = name.contains("argmin");
                // row-arg → one index per row; col-arg → one index per column.
                let out_len = if is_col { cols } else { rows };
                // rowarg.rs / colarg.rs: the `i32` index output must not overlap the f32 input.
                marshal::check_output(
                    arena!(self),
                    name,
                    (1, out, out_len),
                    &[((0, x, n), Overlap::Forbidden)],
                )?;
                let xbuf = marshal::read_f32(arena!(self), 0, x, n, name)?;
                let mut obuf = marshal::output::<i32>(arena!(self), 1, out, out_len, name)?;
                let kernel = match (is_col, is_min) {
                    (false, false) => wukong_runtime::wukong_rowargmax_i32,
                    (false, true) => wukong_runtime::wukong_rowargmin_i32,
                    (true, false) => wukong_runtime::wukong_colargmax_i32,
                    (true, true) => wukong_runtime::wukong_colargmin_i32,
                };
                // SAFETY: xbuf is rows*cols f32; obuf is out_len i32 — the kernel's contract.
                unsafe {
                    kernel(xbuf.as_ptr(), obuf.as_mut_ptr(), rows as i64, cols as i64);
                }
                marshal::write_back(arena_mut!(self), 1, out, &obuf, name, |v| {
                    Value::Int(v as i128)
                })?;
                marshal::recycle_f32(xbuf);
                Ok(Value::Unit)
            }
            // `wukong_cumsum_f32[_parallel]` / `wukong_cum{max,min}_f32[_parallel](x, out, rows, cols)`
            // — the per-row inclusive prefix scan a recognized cumsum / cummax / cummin nest lowers to.
            // Marshal `rows*cols` f32 from x, call the *serial* kernel (bit-identical to the parallel one
            // — rows independent), write the `rows*cols` out (reading all of x first, so in-place
            // `x==out` is safe). cumsum's in-lane scan reassociates (both backends run the identical
            // kernel, so interp == native holds); cummax/cummin select a value, so they are bit-exact.
            // The kernel fn is chosen by name.
            I::CumsumF32
            | I::CumsumF32Parallel
            | I::CummaxF32
            | I::CummaxF32Parallel
            | I::CumminF32
            | I::CumminF32Parallel
            | I::CumprodF32
            | I::CumprodF32Parallel => {
                let x = ptr(args[0])?;
                let out = ptr(args[1])?;
                let rows = dim!(args[2]);
                let cols = dim!(args[3]);
                let n = marshal::span(rows, cols, name)?;
                // cumsum.rs / cumminmax.rs / cumprod.rs: `out` may be EXACTLY `x`; a shifted
                // overlap is not in place (the row scan loads 8 lanes before storing them).
                marshal::check_output(
                    arena!(self),
                    name,
                    (1, out, n),
                    &[((0, x, n), Overlap::ExactOnly)],
                )?;
                let xbuf = marshal::read_f32(arena!(self), 0, x, n, name)?;
                let mut obuf = marshal::output_f32(arena!(self), 1, out, n, name)?;
                let kernel = if name.starts_with("wukong_cummax") {
                    wukong_runtime::wukong_cummax_f32
                } else if name.starts_with("wukong_cummin") {
                    wukong_runtime::wukong_cummin_f32
                } else if name.starts_with("wukong_cumprod") {
                    wukong_runtime::wukong_cumprod_f32
                } else {
                    wukong_runtime::wukong_cumsum_f32
                };
                // SAFETY: xbuf and obuf are both rows*cols f32 — the kernel's contract.
                unsafe {
                    kernel(xbuf.as_ptr(), obuf.as_mut_ptr(), rows as i64, cols as i64);
                }
                marshal::write_f32(arena_mut!(self), 1, out, &obuf, name)?;
                marshal::recycle_f32(xbuf);
                marshal::recycle_f32(obuf);
                Ok(Value::Unit)
            }
            // KL divergence and soft-label cross-entropy share the (a, b, out, rows, cols) shape — two
            // `rows*cols` f32 inputs, a `rows`-long scalar out; the kernel fn is chosen by name.
            I::KldivF32 | I::KldivF32Parallel | I::KdLossF32 | I::KdLossF32Parallel => {
                let a = ptr(args[0])?;
                let b = ptr(args[1])?;
                let out = ptr(args[2])?;
                let rows = dim!(args[3]);
                let cols = dim!(args[4]);
                let n = marshal::span(rows, cols, name)?;
                // The per-row loss must not overlap either distribution it is folded from.
                marshal::check_output(
                    arena!(self),
                    name,
                    (2, out, rows),
                    &[
                        ((0, a, n), Overlap::Forbidden),
                        ((1, b, n), Overlap::Forbidden),
                    ],
                )?;
                let abuf = marshal::read_f32(arena!(self), 0, a, n, name)?;
                let bbuf = marshal::read_f32(arena!(self), 1, b, n, name)?;
                let mut obuf = marshal::output_f32(arena!(self), 2, out, rows, name)?;
                let kernel = if name.starts_with("wukong_kd_loss") {
                    wukong_runtime::wukong_kd_loss_f32
                } else {
                    wukong_runtime::wukong_kldiv_f32
                };
                // SAFETY: abuf/bbuf are rows*cols f32; obuf is rows — the kernel's contract.
                unsafe {
                    kernel(
                        abuf.as_ptr(),
                        bbuf.as_ptr(),
                        obuf.as_mut_ptr(),
                        rows as i64,
                        cols as i64,
                    );
                }
                marshal::write_f32(arena_mut!(self), 2, out, &obuf, name)?;
                marshal::recycle_f32(abuf);
                marshal::recycle_f32(bbuf);
                marshal::recycle_f32(obuf);
                Ok(Value::Unit)
            }
            // `wukong_vmath_bf16(x, out, n, op)` — the bf16-input twin of `wukong_vmath_f32` an
            // `out[i] = f((x[i] as f32))` loop over a `[bf16]` array lowers to. Reconstruct the exact
            // bf16 input bits (as the bf16 reductions/axpby do — the stored value is already bf16-
            // rounded, so `f32_to_bf16_bits` is exact), call the *identical* runtime kernel the native
            // backend calls, write the f32 result back — so the differential oracle stays exact.
            I::VmathBf16 => {
                let x = ptr(args[0])?;
                let out = ptr(args[1])?;
                let n = dim!(args[2]);
                let op = args[3].as_int() as i64;
                // A `[bf16]` input and an f32 output are never one array; any overlap is a pun.
                marshal::check_output(
                    arena!(self),
                    name,
                    (1, out, n),
                    &[((0, x, n), Overlap::Forbidden)],
                )?;
                let input = kernel_values(arena!(self), 0, x, n, 1, name)?;
                let xbuf = kernel_buffer(input, name, |v| {
                    Ok(wukong_runtime::f32_to_bf16_bits(
                        v.kernel_float(name)? as f32
                    ))
                })?;
                let mut obuf = marshal::output_f32(arena!(self), 1, out, n, name)?;
                // SAFETY: xbuf is n u16, obuf is n f32 — the kernel's contract.
                unsafe {
                    wukong_runtime::wukong_vmath_bf16(
                        xbuf.as_ptr(),
                        obuf.as_mut_ptr(),
                        n as i64,
                        op,
                    );
                }
                marshal::write_f32(arena_mut!(self), 1, out, &obuf, name)?;
                marshal::recycle_f32(obuf);
                Ok(Value::Unit)
            }
            // `wukong_vmath_f16` — the F16C twin of `wukong_vmath_bf16`: reconstruct exact f16 bits
            // via `f32_to_f16_bits` and call the identical kernel, so interp == native.
            I::VmathF16 => {
                let x = ptr(args[0])?;
                let out = ptr(args[1])?;
                let n = dim!(args[2]);
                let op = args[3].as_int() as i64;
                // An `[f16]` input and an f32 output are never one array; any overlap is a pun.
                marshal::check_output(
                    arena!(self),
                    name,
                    (1, out, n),
                    &[((0, x, n), Overlap::Forbidden)],
                )?;
                let input = kernel_values(arena!(self), 0, x, n, 1, name)?;
                let xbuf = kernel_buffer(input, name, |v| {
                    Ok(wukong_runtime::f32_to_f16_bits(v.kernel_float(name)? as f32))
                })?;
                let mut obuf = marshal::output_f32(arena!(self), 1, out, n, name)?;
                // SAFETY: xbuf is n u16, obuf is n f32 — the kernel's contract.
                unsafe {
                    wukong_runtime::wukong_vmath_f16(
                        xbuf.as_ptr(),
                        obuf.as_mut_ptr(),
                        n as i64,
                        op,
                    );
                }
                marshal::write_f32(arena_mut!(self), 1, out, &obuf, name)?;
                marshal::recycle_f32(obuf);
                Ok(Value::Unit)
            }
            // `wukong_vmath_{bf16,f16}_out(x, out, n, op)` — the **half-output** activation twin: read a
            // `[bf16]`/`[f16]` input, call the *identical* runtime kernel (which computes the activation
            // in f32 and narrows the result back to the half width through the shared shim), then widen
            // the stored half bits back to the interpreter's f32 slot. Same reconstruct-exact-bits +
            // call-the-kernel discipline as the reduction/axpby-out arms, so interp == native bit-exact.
            I::VmathBf16Out | I::VmathF16Out => {
                let is_f16 = name == "wukong_vmath_f16_out";
                let x = ptr(args[0])?;
                let out = ptr(args[1])?;
                let n = dim!(args[2]);
                let op = args[3].as_int() as i64;
                // Half in, half out, elementwise: EXACTLY in place is the same map.
                marshal::check_output(
                    arena!(self),
                    name,
                    (1, out, n),
                    &[((0, x, n), Overlap::ExactOnly)],
                )?;
                let input = kernel_values(arena!(self), 0, x, n, 1, name)?;
                let xbuf = kernel_buffer(input, name, |v| {
                    let f = v.kernel_float(name)? as f32;
                    Ok(if is_f16 {
                        wukong_runtime::f32_to_f16_bits(f)
                    } else {
                        wukong_runtime::f32_to_bf16_bits(f)
                    })
                })?;
                let mut obuf = marshal::output::<u16>(arena!(self), 1, out, n, name)?;
                // SAFETY: xbuf and obuf are each n u16 — the kernel's contract.
                unsafe {
                    if is_f16 {
                        wukong_runtime::wukong_vmath_f16_out(
                            xbuf.as_ptr(),
                            obuf.as_mut_ptr(),
                            n as i64,
                            op,
                        );
                    } else {
                        wukong_runtime::wukong_vmath_bf16_out(
                            xbuf.as_ptr(),
                            obuf.as_mut_ptr(),
                            n as i64,
                            op,
                        );
                    }
                }
                marshal::write_back(arena_mut!(self), 1, out, &obuf, name, |obits| {
                    let widened = if is_f16 {
                        wukong_runtime::f16_bits_to_f32(obits)
                    } else {
                        wukong_runtime::bf16_bits_to_f32(obits)
                    };
                    Value::Float(widened as f64)
                })?;
                Ok(Value::Unit)
            }
            // `wukong_velem_f32[_parallel](x, y, out, n, a, b, c, op)` — the streaming affine+
            // activation kernel (saxpy / scale / residual-add / bias / ReLU) a recognized `out[i] =
            // act(a·x[i] + b·y[i] + c)` map lowers to. Marshal `n` f32 from x and y, call the *serial*
            // runtime kernel (bit-identical to the parallel one the native backend runs under
            // `@parallel`, since the map is elementwise), write the result back — so the differential
            // oracle stays exact despite the wider lanes / non-temporal stores (which write the same
            // bits). Reading all of x/y before writing out makes the in-place (x == out) case correct.
            // An unused y contributes no data: do not type-check or dereference its memory.
            // The scratch buffer still supplies valid storage to every runtime implementation.
            I::VelemF32 | I::VelemF32Parallel => {
                let x = ptr(args[0])?;
                let y = ptr(args[1])?;
                let out = ptr(args[2])?;
                let n = dim!(args[3]);
                let a = args[4].as_float() as f32;
                let b = args[5].as_float() as f32;
                let c = args[6].as_float() as f32;
                let op = args[7].as_int() as i64;
                // 512/1024 are the runtime Hadamard/division flags.
                let reads_y = op & (wukong_runtime::VE_USE_Y | 512 | 1024) != 0;
                // A shifted overlap runs the source loop's order on live memory (see the
                // `wukong_vmath_f32` arm). An unread `y` is no operand and constrains nothing.
                let y_len = if reads_y { n } else { 0 };
                if marshal::shifted_overlap((out, n), &[(x, n), (y, y_len)]) {
                    let both = [(0, x), (1, y)];
                    let inputs = if reads_y { &both[..] } else { &both[..1] };
                    marshal::elementwise_live(
                        arena_mut!(self),
                        inputs,
                        (2, out),
                        n,
                        name,
                        |lane| {
                            let yv = if reads_y { lane[1] } else { 0.0 };
                            let mut o = 0.0f32;
                            // SAFETY: one element per operand — the kernel's contract at n = 1 (`y` is
                            // a valid one-element read whether or not `op` reads it).
                            unsafe {
                                wukong_runtime::wukong_velem_f32(
                                    &lane[0], &yv, &mut o, 1, a, b, c, op,
                                )
                            };
                            o
                        },
                    )?;
                    return Ok(Value::Unit);
                }
                // velem.rs: "`x`/`y`/`out` may alias" — EXACTLY, the elementwise map's in-place form.
                marshal::check_output(
                    arena!(self),
                    name,
                    (2, out, n),
                    &[
                        ((0, x, n), Overlap::ExactOnly),
                        ((1, y, y_len), Overlap::ExactOnly),
                    ],
                )?;
                let xbuf = marshal::read_f32(arena!(self), 0, x, n, name)?;
                // An unused y contributes no data: do not type-check or dereference its memory. The
                // zeroed scratch still supplies valid storage to every runtime implementation.
                let ybuf = if reads_y {
                    marshal::read_f32(arena!(self), 1, y, n, name)?
                } else {
                    marshal::output_f32(arena!(self), 0, x, n, name)?
                };
                let mut obuf = marshal::output_f32(arena!(self), 2, out, n, name)?;
                // SAFETY: xbuf/ybuf/obuf are exactly n f32 long — the kernel's contract.
                unsafe {
                    wukong_runtime::wukong_velem_f32(
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
                marshal::write_f32(arena_mut!(self), 2, out, &obuf, name)?;
                marshal::recycle_f32(xbuf);
                marshal::recycle_f32(ybuf);
                marshal::recycle_f32(obuf);
                Ok(Value::Unit)
            }
            // `wukong_bias_bcast_f32[_parallel](x, b, out, rows, cols, op)` — the broadcast-bias add a
            // recognized `for i { for j { out[i*C+j] = act(x[i*C+j] + b[j]) } }` nest lowers to (a
            // `cols`-long bias added across every row, optional fused activation). Marshal `rows*cols`
            // f32 from x and `cols` from b, call the *serial* runtime kernel (serial == parallel
            // bit-for-bit, rows independent), write the result back — so the differential oracle stays
            // exact for both the serial and `_parallel` names. Read all of x/b before writing out so an
            // in-place (x == out) case is correct.
            I::BiasBcastF32 | I::BiasBcastF32Parallel => {
                let x = ptr(args[0])?;
                let b = ptr(args[1])?;
                let out = ptr(args[2])?;
                let rows = dim!(args[3]);
                let cols = dim!(args[4]);
                let op = args[5].as_int() as i64;
                let n = marshal::span(rows, cols, name)?;
                // bias.rs: `x`/`out` identical or disjoint; the bias row must not overlap `out`.
                marshal::check_output(
                    arena!(self),
                    name,
                    (2, out, n),
                    &[
                        ((0, x, n), Overlap::ExactOnly),
                        ((1, b, cols), Overlap::Forbidden),
                    ],
                )?;
                let xbuf = marshal::read_f32(arena!(self), 0, x, n, name)?;
                let bbuf = marshal::read_f32(arena!(self), 1, b, cols, name)?;
                let mut obuf = marshal::output_f32(arena!(self), 2, out, n, name)?;
                // SAFETY: xbuf/obuf are n f32, bbuf is cols f32 — the kernel's contract.
                unsafe {
                    wukong_runtime::wukong_bias_bcast_f32(
                        xbuf.as_ptr(),
                        bbuf.as_ptr(),
                        obuf.as_mut_ptr(),
                        rows as i64,
                        cols as i64,
                        op,
                    );
                }
                marshal::write_f32(arena_mut!(self), 2, out, &obuf, name)?;
                marshal::recycle_f32(xbuf);
                marshal::recycle_f32(bbuf);
                marshal::recycle_f32(obuf);
                Ok(Value::Unit)
            }
            // `wukong_vhorner_f32(x, out, n, coeffs, ncoeff)` — the streaming Horner-polynomial kernel
            // a recognized `r = c0; r = r*x + c1; …; out[i] = r` loop lowers to. Marshal `n` f32 from x
            // and `ncoeff` f32 from coeffs, call the identical runtime kernel, write out — so the
            // differential oracle stays exact despite the wider lanes / non-temporal stores.
            I::VhornerF32 => {
                let x = ptr(args[0])?;
                let out = ptr(args[1])?;
                let n = dim!(args[2]);
                let coeffs = ptr(args[3])?;
                let ncoeff = dim!(args[4]);
                // The coefficient table is read for every element, so an output over it changes what
                // later elements compute: refused (the compiler builds the table itself, so only a
                // hand-written call can get here).
                marshal::check_output(
                    arena!(self),
                    name,
                    (1, out, n),
                    &[((3, coeffs, ncoeff), Overlap::Forbidden)],
                )?;
                let cbuf = marshal::read_f32(arena!(self), 3, coeffs, ncoeff, name)?;
                // Elementwise in `x`: a shifted overlap runs the source loop's order on live memory
                // (see the `wukong_vmath_f32` arm); exact in place equals out of place.
                if marshal::shifted_overlap((out, n), &[(x, n)]) {
                    marshal::elementwise_live(
                        arena_mut!(self),
                        &[(0, x)],
                        (1, out),
                        n,
                        name,
                        |lane| {
                            let mut o = 0.0f32;
                            // SAFETY: one element in and out; cbuf is ncoeff f32 — the kernel's contract.
                            unsafe {
                                wukong_runtime::wukong_vhorner_f32(
                                    &lane[0],
                                    &mut o,
                                    1,
                                    cbuf.as_ptr(),
                                    ncoeff as i64,
                                )
                            };
                            o
                        },
                    )?;
                    marshal::recycle_f32(cbuf);
                    return Ok(Value::Unit);
                }
                let xbuf = marshal::read_f32(arena!(self), 0, x, n, name)?;
                let mut obuf = marshal::output_f32(arena!(self), 1, out, n, name)?;
                // SAFETY: xbuf/obuf are n f32, cbuf is ncoeff f32 — the kernel's contract.
                unsafe {
                    wukong_runtime::wukong_vhorner_f32(
                        xbuf.as_ptr(),
                        obuf.as_mut_ptr(),
                        n as i64,
                        cbuf.as_ptr(),
                        ncoeff as i64,
                    );
                }
                marshal::write_f32(arena_mut!(self), 1, out, &obuf, name)?;
                marshal::recycle_f32(xbuf);
                marshal::recycle_f32(cbuf);
                marshal::recycle_f32(obuf);
                Ok(Value::Unit)
            }
            // `wukong_sreduce_f32[_parallel](x, y, n, op) -> f32` — the deterministic reduction kernel
            // a `@parallel` reduction loop lowers to (dot / ssd / sum). Marshal `n` f32 out of x and y
            // and call the *serial* runtime kernel, which is bit-identical to the parallel one the
            // native backend runs — so the differential oracle stays exact. The unary sum passes
            // y == x (harmless redundant reads); the kernel ignores y for it.
            I::SreduceF32 | I::SreduceF32Parallel => {
                let x = ptr(args[0])?;
                let y = ptr(args[1])?;
                let n = args[2].as_int().max(0) as usize;
                let op = args[3].as_int() as i64;
                // Reduction opcode 10 is the runtime binary absolute-difference fold.
                let reads_y = matches!(op, wukong_runtime::RED_DOT | wukong_runtime::RED_SSD | 10);
                // Both operands are only read, so no aliasing contract applies; the ranges are still
                // validated before any scratch is reserved (a positive-but-absurd `n` is a
                // diagnostic, not an allocator abort).
                let xbuf = marshal::read_f32(arena!(self), 0, x, n, name)?;
                let ybuf = if reads_y {
                    marshal::read_f32(arena!(self), 1, y, n, name)?
                } else {
                    marshal::output_f32(arena!(self), 0, x, n, name)?
                };
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
                        wukong_runtime::wukong_sreduce_f32(
                            xbuf.as_ptr(),
                            ybuf.as_ptr(),
                            n as i64,
                            op,
                        )
                    },
                };
                marshal::recycle_f32(xbuf);
                marshal::recycle_f32(ybuf);
                Ok(Value::Float(r as f64))
            }
            // `wukong_argreduce_f32(x, n, op) -> i64` — the deterministic argmax/argmin a recognized
            // `for k { if x[k] CMP bv { bv=x[k]; bi=k } }` loop reconciles against. Read `n` f32 from x
            // and call the *serial* kernel (bit-identical to the parallel one — fixed RCHUNK, ascending
            // index-order combine, lowest-index tie-break), returning the index. CPU-only (no
            // accelerator seam), so the differential oracle is unaffected.
            I::ArgreduceF32 | I::ArgreduceF32Parallel => {
                let x = ptr(args[0])?;
                let n = args[1].as_int().max(0) as usize;
                let op = args[2].as_int() as i64;
                let xbuf = marshal::read_f32(arena!(self), 0, x, n, name)?;
                // SAFETY: xbuf is exactly n f32 long — the kernel's contract.
                let r =
                    unsafe { wukong_runtime::wukong_argreduce_f32(xbuf.as_ptr(), n as i64, op) };
                marshal::recycle_f32(xbuf);
                Ok(Value::Int(r as i128))
            }
            // `wukong_dot_bf16(x, y, n) -> f32` / `wukong_sum_bf16(x, n) -> f32` — the bf16
            // mixed-precision reduction a `s += (x[k] as f32) [* (y[k] as f32)]` loop over `[bf16; _]`
            // arrays lowers to (bf16 storage, f32 accumulate). The interpreter stores bf16 as the
            // bf16-rounded f32 value; reconstruct the exact 16 stored bits via `f32_to_bf16_bits`
            // (idempotent on an already-bf16-rounded value) so the buffer is identical to the native
            // backend's 2-byte storage, then call the identical kernel — the differential gate stays
            // exact despite the kernel's reassociated 8-lane accumulation. The `_parallel` twins (a
            // `@parallel` low-precision reduction) marshal identically but call the **parallel** kernel:
            // unlike the f32 reductions (whose serial form is itself chunked, so the interp calls it),
            // the bf16/f16 serial kernels reduce the *whole* array flat, which reassociates vs the
            // chunked parallel fold — so we call the deterministic parallel kernel native runs, exactly.
            I::DotBf16 | I::SumBf16 | I::DotBf16Parallel | I::SumBf16Parallel => {
                let is_dot = name == "wukong_dot_bf16" || name == "wukong_dot_bf16_parallel";
                let is_par = name.ends_with("_parallel");
                let x = ptr(args[0])?;
                let (y, n) = if is_dot {
                    (ptr(args[1])?, args[2].as_int().max(0) as usize)
                } else {
                    (x, args[1].as_int().max(0) as usize)
                };
                let xbuf = marshal::read_half_bits(arena!(self), 0, x, n, name, false)?;
                let ybuf = if is_dot {
                    marshal::read_half_bits(arena!(self), 1, y, n, name, false)?
                } else {
                    Vec::new()
                };
                // SAFETY: the buffers are exactly n u16 long — the kernels' contract.
                let r = unsafe {
                    match (is_dot, is_par) {
                        (true, false) => {
                            wukong_runtime::wukong_dot_bf16(xbuf.as_ptr(), ybuf.as_ptr(), n as i64)
                        }
                        (true, true) => wukong_runtime::wukong_dot_bf16_parallel(
                            xbuf.as_ptr(),
                            ybuf.as_ptr(),
                            n as i64,
                        ),
                        (false, false) => wukong_runtime::wukong_sum_bf16(xbuf.as_ptr(), n as i64),
                        (false, true) => {
                            wukong_runtime::wukong_sum_bf16_parallel(xbuf.as_ptr(), n as i64)
                        }
                    }
                };
                Ok(Value::Float(r as f64))
            }
            // `wukong_reduce_bf16(x, n, op) -> f32` — the bf16 max-family reduction (max/min/absmax)
            // a `m = fmax/fmin(m, (x[k] as f32))` loop over `[bf16]` lowers to. Reconstruct the exact
            // bf16 bits (as the dot/sum path does), call the identical kernel — the widen is lossless
            // and max/min round nothing, so interp == native exactly.
            I::ReduceBf16 | I::ReduceBf16Parallel => {
                let is_par = name.ends_with("_parallel");
                let x = ptr(args[0])?;
                let n = args[1].as_int().max(0) as usize;
                let op = args[2].as_int() as i64;
                let xbuf = marshal::read_half_bits(arena!(self), 0, x, n, name, false)?;
                // SAFETY: xbuf is exactly n u16 long — the kernel's contract.
                let r = unsafe {
                    if is_par {
                        wukong_runtime::wukong_reduce_bf16_parallel(xbuf.as_ptr(), n as i64, op)
                    } else {
                        wukong_runtime::wukong_reduce_bf16(xbuf.as_ptr(), n as i64, op)
                    }
                };
                Ok(Value::Float(r as f64))
            }
            // The IEEE-f16 twins: `wukong_dot_f16` / `wukong_sum_f16` / `wukong_reduce_f16` — same
            // marshaling as the bf16 reductions, but reconstruct the exact f16 bits via
            // `f32_to_f16_bits` (the stored value is already f16-rounded, so this is exact) and call
            // the F16C kernels. interp == native bit-for-bit (the widen is lossless, identical kernel).
            I::DotF16 | I::SumF16 | I::DotF16Parallel | I::SumF16Parallel => {
                let is_dot = name == "wukong_dot_f16" || name == "wukong_dot_f16_parallel";
                let is_par = name.ends_with("_parallel");
                let x = ptr(args[0])?;
                let (y, n) = if is_dot {
                    (ptr(args[1])?, args[2].as_int().max(0) as usize)
                } else {
                    (x, args[1].as_int().max(0) as usize)
                };
                let xbuf = marshal::read_half_bits(arena!(self), 0, x, n, name, true)?;
                let ybuf = if is_dot {
                    marshal::read_half_bits(arena!(self), 1, y, n, name, true)?
                } else {
                    Vec::new()
                };
                // SAFETY: the buffers are exactly n u16 long — the kernels' contract. The `_parallel`
                // names call the deterministic parallel kernel (see the bf16 dot/sum arm above).
                let r = unsafe {
                    match (is_dot, is_par) {
                        (true, false) => {
                            wukong_runtime::wukong_dot_f16(xbuf.as_ptr(), ybuf.as_ptr(), n as i64)
                        }
                        (true, true) => wukong_runtime::wukong_dot_f16_parallel(
                            xbuf.as_ptr(),
                            ybuf.as_ptr(),
                            n as i64,
                        ),
                        (false, false) => wukong_runtime::wukong_sum_f16(xbuf.as_ptr(), n as i64),
                        (false, true) => {
                            wukong_runtime::wukong_sum_f16_parallel(xbuf.as_ptr(), n as i64)
                        }
                    }
                };
                Ok(Value::Float(r as f64))
            }
            I::ReduceF16 | I::ReduceF16Parallel => {
                let is_par = name.ends_with("_parallel");
                let x = ptr(args[0])?;
                let n = args[1].as_int().max(0) as usize;
                let op = args[2].as_int() as i64;
                let xbuf = marshal::read_half_bits(arena!(self), 0, x, n, name, true)?;
                // SAFETY: xbuf is exactly n u16 long — the kernel's contract.
                let r = unsafe {
                    if is_par {
                        wukong_runtime::wukong_reduce_f16_parallel(xbuf.as_ptr(), n as i64, op)
                    } else {
                        wukong_runtime::wukong_reduce_f16(xbuf.as_ptr(), n as i64, op)
                    }
                };
                Ok(Value::Float(r as f64))
            }
            // `wukong_axpby_bf16(x, y, out, n, a, b)` — the bf16→f32 streaming axpby a recognized
            // `out[k] = a*(x[k] as f32) + b*(y[k] as f32)` map over `[bf16; _]` inputs (f32 output)
            // lowers to. Reconstruct the bf16 input bits exactly (as the reductions do), call the
            // identical kernel, write the f32 result back — so the differential gate stays exact.
            I::AxpbyBf16 | I::AxpbyF16 => {
                let is_f16 = name == "wukong_axpby_f16";
                let x = ptr(args[0])?;
                let y = ptr(args[1])?;
                let out = ptr(args[2])?;
                let n = dim!(args[3]);
                let a = args[4].as_float() as f32;
                let b = args[5].as_float() as f32;
                // Half inputs and an f32 output: an overlap between them is a mixed-width pun the
                // slot memory cannot even represent faithfully (one slot per element against 2 and 4
                // bytes natively), so it is refused rather than answered.
                marshal::check_output(
                    arena!(self),
                    name,
                    (2, out, n),
                    &[
                        ((0, x, n), Overlap::Forbidden),
                        ((1, y, n), Overlap::Forbidden),
                    ],
                )?;
                let xbuf = marshal::read_half_bits(arena!(self), 0, x, n, name, is_f16)?;
                let ybuf = marshal::read_half_bits(arena!(self), 1, y, n, name, is_f16)?;
                let mut obuf = marshal::output_f32(arena!(self), 2, out, n, name)?;
                // SAFETY: xbuf/ybuf are n u16, obuf is n f32 — the kernel's contract.
                unsafe {
                    if is_f16 {
                        wukong_runtime::wukong_axpby_f16(
                            xbuf.as_ptr(),
                            ybuf.as_ptr(),
                            obuf.as_mut_ptr(),
                            n as i64,
                            a,
                            b,
                        );
                    } else {
                        wukong_runtime::wukong_axpby_bf16(
                            xbuf.as_ptr(),
                            ybuf.as_ptr(),
                            obuf.as_mut_ptr(),
                            n as i64,
                            a,
                            b,
                        );
                    }
                }
                marshal::write_f32(arena_mut!(self), 2, out, &obuf, name)?;
                marshal::recycle_f32(obuf);
                Ok(Value::Unit)
            }
            // `wukong_axpby_{bf16,f16}_out(x, y, out, n, a, b)` — the **all-half** streaming axpby (half
            // in AND a half output): the axpby sum is narrowed back to bf16/f16 on store. Reconstruct the
            // half input bits exactly, call the identical kernel into a real `u16` output buffer, then
            // widen each stored half back to the `f32` an `[bf16]`/`[f16]` element observes (`Value::Float`
            // of the widened bits) — bit-identical to the native backend's write-then-load. The
            // narrowing round lives in the shared shim (== the recognizer's `as bf16` store), so interp
            // == native and `-O0` == `-O2` (value-rounded).
            I::AxpbyBf16Out | I::AxpbyF16Out => {
                let is_f16 = name == "wukong_axpby_f16_out";
                let x = ptr(args[0])?;
                let y = ptr(args[1])?;
                let out = ptr(args[2])?;
                let n = dim!(args[3]);
                let a = args[4].as_float() as f32;
                let b = args[5].as_float() as f32;
                // Half in, half out, elementwise: EXACTLY in place is the same map. The runtime also
                // runs a SHIFTED same-width overlap with live ascending reads; copied operands cannot
                // model that, so it is refused here rather than answered with a snapshot value.
                marshal::check_output(
                    arena!(self),
                    name,
                    (2, out, n),
                    &[
                        ((0, x, n), Overlap::ExactOnly),
                        ((1, y, n), Overlap::ExactOnly),
                    ],
                )?;
                let xbuf = marshal::read_half_bits(arena!(self), 0, x, n, name, is_f16)?;
                let ybuf = marshal::read_half_bits(arena!(self), 1, y, n, name, is_f16)?;
                let mut obuf = marshal::output::<u16>(arena!(self), 2, out, n, name)?;
                // SAFETY: xbuf/ybuf/obuf are each n u16 — the kernel's contract.
                unsafe {
                    if is_f16 {
                        wukong_runtime::wukong_axpby_f16_out(
                            xbuf.as_ptr(),
                            ybuf.as_ptr(),
                            obuf.as_mut_ptr(),
                            n as i64,
                            a,
                            b,
                        );
                    } else {
                        wukong_runtime::wukong_axpby_bf16_out(
                            xbuf.as_ptr(),
                            ybuf.as_ptr(),
                            obuf.as_mut_ptr(),
                            n as i64,
                            a,
                            b,
                        );
                    }
                }
                marshal::write_back(arena_mut!(self), 2, out, &obuf, name, |obits| {
                    let widened = if is_f16 {
                        wukong_runtime::f16_bits_to_f32(obits)
                    } else {
                        wukong_runtime::bf16_bits_to_f32(obits)
                    };
                    Value::Float(widened as f64)
                })?;
                Ok(Value::Unit)
            }
            // `wukong_sgemm_{bf16,f16}_{nt,tn}[_parallel](a, b, c, m, k, n, beta)` — the low-precision
            // GEMM (bf16/f16 inputs stored as `u16` bits, f32 accumulate, f32 output) a matmul nest over
            // `[bf16; _]`/`[f16; _]` operands lowers to: NT = `C = A·Bᵀ` (nn.Linear forward), TN =
            // `C = Aᵀ·B` (the `dW = dYᵀ·X` weight gradient). The flat operand element counts are the
            // same either way (A is m·k, B is n·k regardless of layout — only the kernel's stride
            // interpretation differs), so the marshalling is identical; only the runtime fn called
            // differs. Reconstruct the exact stored 16 bits of each input via `f32_to_{bf16,f16}_bits`
            // (idempotent on an already-rounded value, so the buffer is bit-identical to the native
            // backend's 2-byte storage — same as the bf16/f16 reductions/axpby). A nonzero beta is
            // the runtime's accumulate flag, so marshal the incoming C as well; zero beta keeps
            // a fresh output buffer. This must hold even though current recognizers emit beta=0.
            // Call the *serial* runtime kernel for BOTH the
            // serial and `_parallel` names: the runtime pins serial == parallel == interpreter
            // bit-for-bit, and the interpreter is the oracle, so the differential gate stays exact
            // despite the kernel's wider/reassociated accumulation.
            I::SgemmBf16Nt
            | I::SgemmBf16NtParallel
            | I::SgemmF16Nt
            | I::SgemmF16NtParallel
            | I::SgemmBf16Tn
            | I::SgemmBf16TnParallel
            | I::SgemmF16Tn
            | I::SgemmF16TnParallel => {
                let is_f16 = name.contains("_f16");
                let is_tn = name.contains("_tn");
                let a = ptr(args[0])?;
                let b = ptr(args[1])?;
                let c = ptr(args[2])?;
                let m = dim!(args[3]);
                let k = dim!(args[4]);
                let n = dim!(args[5]);
                let beta = args[6].as_int() as i64;
                let av = kernel_values(arena!(self), 0, a, m, k, name)?;
                let bv = kernel_values(arena!(self), 1, b, n, k, name)?;
                let cv = kernel_values(arena!(self), 2, c, m, n, name)?;
                // The f32 `C` is written in tiled order while the half `A`/`B` are read.
                marshal::check_output(
                    arena!(self),
                    name,
                    (2, c, cv.len()),
                    &[
                        ((0, a, av.len()), Overlap::Forbidden),
                        ((1, b, bv.len()), Overlap::Forbidden),
                    ],
                )?;
                // Reconstruct the exact half bits after validating every operand range.
                let bits = |value: Value| -> Result<u16, String> {
                    let f = value.kernel_float(name)? as f32;
                    Ok(if is_f16 {
                        wukong_runtime::f32_to_f16_bits(f)
                    } else {
                        wukong_runtime::f32_to_bf16_bits(f)
                    })
                };
                let abuf = kernel_buffer(av, name, bits)?;
                let bbuf = kernel_buffer(bv, name, bits)?;
                // Beta zero requires valid output storage but does not read its prior values.
                let mut cbuf = kernel_buffer(cv, name, |v| {
                    Ok(if beta != 0 {
                        v.kernel_float(name)? as f32
                    } else {
                        0.0
                    })
                })?;
                let (ai, ki, ni) = (m as i64, k as i64, n as i64);
                let (ap, bp, cp) = (abuf.as_ptr(), bbuf.as_ptr(), cbuf.as_mut_ptr());
                // SAFETY: abuf is m*k, bbuf is n*k u16; cbuf is m*n f32 — the kernels' contract.
                // (TN reads A as [k,m] and B as [k,n], but those have the same flat element counts.)
                unsafe {
                    match (is_f16, is_tn) {
                        (false, false) => {
                            wukong_runtime::wukong_sgemm_bf16_nt(ap, bp, cp, ai, ki, ni, beta)
                        }
                        (true, false) => {
                            wukong_runtime::wukong_sgemm_f16_nt(ap, bp, cp, ai, ki, ni, beta)
                        }
                        (false, true) => {
                            wukong_runtime::wukong_sgemm_bf16_tn(ap, bp, cp, ai, ki, ni, beta)
                        }
                        (true, true) => {
                            wukong_runtime::wukong_sgemm_f16_tn(ap, bp, cp, ai, ki, ni, beta)
                        }
                    }
                }
                for (t, &val) in cbuf.iter().enumerate() {
                    *self
                        .memory
                        .get_mut(c + t)
                        .ok_or("lowp sgemm output out of bounds")? = Value::Float(val as f64);
                }
                Ok(Value::Unit)
            }
            // `wukong_sgemm_{bf16,f16}_nt_epi[_parallel](a, b, c, m, k, n, beta, bias, act)` — the
            // fused-epilogue mixed-precision Linear (`C = act(A·Bᵀ + bias)`, half inputs / f32 output).
            // Combines the bf16/f16 GEMM's u16-input marshalling (read each `[bf16]`/`[f16]` element back
            // to its exact 16 stored bits via `f32_to_{bf16,f16}_bits`) with the `nt_epi` epilogue's
            // bias/act marshalling (an absent bias arrives as `Value::Int(0)` → a null pointer). Calls
            // the *serial* runtime kernel for both the serial and `_parallel` names (serial == parallel
            // == interpreter bit-for-bit — rows independent, identical per-(i,j) order), so the
            // differential gate stays exact despite the kernel's wider/reassociated accumulation.
            I::SgemmBf16NtEpi
            | I::SgemmBf16NtEpiParallel
            | I::SgemmF16NtEpi
            | I::SgemmF16NtEpiParallel => {
                let is_f16 = name.contains("_f16");
                let a = ptr(args[0])?;
                let b = ptr(args[1])?;
                let c = ptr(args[2])?;
                let m = dim!(args[3]);
                let k = dim!(args[4]);
                let n = dim!(args[5]);
                let beta = args[6].as_int() as i64;
                // An absent bias arrives as a `Value::Int(0)` (the null built as an integer 0) vs a real
                // array's `Value::Ptr` — distinguished by variant, like the f32 `nt_epi` epilogue.
                let bias_idx = match args[7] {
                    Value::Ptr(p) => Some(p),
                    _ => None,
                };
                let act = args[8].as_int() as i64;
                let av = kernel_values(arena!(self), 0, a, m, k, name)?;
                let bv = kernel_values(arena!(self), 1, b, n, k, name)?;
                let cv = kernel_values(arena!(self), 2, c, m, n, name)?;
                let biasv = bias_idx
                    .map(|base| kernel_values(arena!(self), 7, base, 1, n, name))
                    .transpose()?;
                // The f32 `C` is written while the half operands and the epilogue bias are read.
                marshal::check_output(
                    arena!(self),
                    name,
                    (2, c, cv.len()),
                    &[
                        ((0, a, av.len()), Overlap::Forbidden),
                        ((1, b, bv.len()), Overlap::Forbidden),
                        (
                            (7, bias_idx.unwrap_or(0), biasv.map_or(0, <[Value]>::len)),
                            Overlap::Forbidden,
                        ),
                    ],
                )?;
                // Reconstruct the exact half bits after validating every operand range.
                let bits = |value: Value| -> Result<u16, String> {
                    let f = value.kernel_float(name)? as f32;
                    Ok(if is_f16 {
                        wukong_runtime::f32_to_f16_bits(f)
                    } else {
                        wukong_runtime::f32_to_bf16_bits(f)
                    })
                };
                let abuf = kernel_buffer(av, name, bits)?;
                let bbuf = kernel_buffer(bv, name, bits)?;
                // Beta zero requires valid output storage but does not read its prior values.
                let mut cbuf = kernel_buffer(cv, name, |v| {
                    Ok(if beta != 0 {
                        v.kernel_float(name)? as f32
                    } else {
                        0.0
                    })
                })?;
                let biasbuf = kernel_buffer(biasv.unwrap_or(&[]), name, |v| {
                    Ok(v.kernel_float(name)? as f32)
                })?;
                let bias_ptr = if bias_idx.is_some() {
                    biasbuf.as_ptr()
                } else {
                    std::ptr::null()
                };
                // SAFETY: abuf is m*k, bbuf is n*k u16; cbuf is m*n f32; bias is null or n long — the
                // kernels' contract. The bf16/f16 kernels differ only in which runtime fn widens the bits.
                unsafe {
                    if is_f16 {
                        wukong_runtime::wukong_sgemm_f16_nt_epi(
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
                    } else {
                        wukong_runtime::wukong_sgemm_bf16_nt_epi(
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
                    }
                }
                for (t, &val) in cbuf.iter().enumerate() {
                    *self
                        .memory
                        .get_mut(c + t)
                        .ok_or("lowp sgemm_epi output out of bounds")? = Value::Float(val as f64);
                }
                Ok(Value::Unit)
            }
            // `wukong_norm_f32[_parallel](x, out, rows, cols, eps_bits, op)` — the fused row-wise
            // softmax / LayerNorm / RMSNorm kernel a recognized multi-pass norm lowers to. Marshal the
            // `rows*cols` f32 out of x, call the *serial* runtime kernel (bit-identical to the parallel
            // one the native backend runs, since rows are independent), write the result to out.
            // Reading all of x before writing out makes the in-place (x == out) case correct.
            I::NormF32 | I::NormF32Parallel => {
                let x = ptr(args[0])?;
                let out = ptr(args[1])?;
                let rows = dim!(args[2]);
                let cols = dim!(args[3]);
                let eps_bits = args[4].as_int() as i64;
                let op = args[5].as_int() as i64;
                let n = marshal::span(rows, cols, name)?;
                // norm.rs: `x`/`out` may alias for in-place — EXACTLY; each row is read whole before
                // it is written, so in place equals out of place. A shifted overlap would let one
                // row's writes reach the next row's reads.
                marshal::check_output(
                    arena!(self),
                    name,
                    (1, out, n),
                    &[((0, x, n), Overlap::ExactOnly)],
                )?;
                let xbuf = marshal::read_f32(arena!(self), 0, x, n, name)?;
                let mut obuf = marshal::output_f32(arena!(self), 1, out, n, name)?;
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
                        wukong_runtime::wukong_norm_f32(
                            xbuf.as_ptr(),
                            obuf.as_mut_ptr(),
                            rows as i64,
                            cols as i64,
                            eps_bits,
                            op,
                        );
                    },
                }
                marshal::write_f32(arena_mut!(self), 1, out, &obuf, name)?;
                marshal::recycle_f32(xbuf);
                marshal::recycle_f32(obuf);
                Ok(Value::Unit)
            }
            // `wukong_norm_affine_f32[_parallel](x, out, gamma, beta, rows, cols, eps_bits, op)` — the
            // affine (per-column scale gamma + optional shift beta) fused LayerNorm/RMSNorm. Same
            // marshalling as the plain norm, plus the gamma/beta arrays (length cols). An absent param
            // lowers to a `Ptr`-typed `ConstInt(0)` → a `Value::Int(0)` here (distinct from a real array's
            // `Value::Ptr`), so match the variant: pass a null pointer (kernel uses scale 1 / shift 0).
            // Both names marshal through the *serial* runtime kernel (bit-identical — rows independent).
            I::NormAffineF32 | I::NormAffineF32Parallel => {
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
                let rows = dim!(args[4]);
                let cols = dim!(args[5]);
                let eps_bits = args[6].as_int() as i64;
                let op = args[7].as_int() as i64;
                let n = marshal::span(rows, cols, name)?;
                // `x`/`out` may be EXACTLY in place, as for the plain norm. `gamma`/`beta` are re-read
                // on every row's apply, so an `out` overlapping either would change what a later row
                // reads: the kernel and the source loop see that, a copy here would not.
                let affine = |k: usize, idx: Option<usize>| {
                    (k, idx.unwrap_or(0), if idx.is_some() { cols } else { 0 })
                };
                marshal::check_output(
                    arena!(self),
                    name,
                    (1, out, n),
                    &[
                        ((0, x, n), Overlap::ExactOnly),
                        (affine(2, gamma_idx), Overlap::Forbidden),
                        (affine(3, beta_idx), Overlap::Forbidden),
                    ],
                )?;
                let xbuf = marshal::read_f32(arena!(self), 0, x, n, name)?;
                let gbuf = match gamma_idx {
                    Some(g) => marshal::read_f32(arena!(self), 2, g, cols, name)?,
                    None => Vec::new(),
                };
                let bbuf = match beta_idx {
                    Some(bb) => marshal::read_f32(arena!(self), 3, bb, cols, name)?,
                    None => Vec::new(),
                };
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
                let mut obuf = marshal::output_f32(arena!(self), 1, out, n, name)?;
                // SAFETY: xbuf is rows*cols; gamma/beta (when present) are cols long — kernel contract.
                unsafe {
                    wukong_runtime::wukong_norm_affine_f32(
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
                marshal::write_f32(arena_mut!(self), 1, out, &obuf, name)?;
                marshal::recycle_f32(xbuf);
                marshal::recycle_f32(gbuf);
                marshal::recycle_f32(bbuf);
                marshal::recycle_f32(obuf);
                Ok(Value::Unit)
            }
            // `wukong_embedding_f32[_parallel](out, weight, ids, t, h, v)` — embedding lookup (the first
            // layer of every LLM): `out[r,:] = weight[ids[r],:]`. NOTE the output pointer is the FIRST
            // arg.
            //
            // The gather runs **row at a time straight against `self.memory`**, in ascending `t`, which
            // is exactly the source nest `for t { for d { out[t*h+d] = weight[ids[t]*h+d] } }`. It must
            // not marshal the weight table into a scratch buffer first: that snapshots `weight` before
            // the first row is written, so when `out` and `weight` are the *same* buffer (an in-place
            // gather — a legal Wukong program, and the case a shared token/position table produces) the
            // interpreter computed a different program than the source and than the serial native
            // kernel, which reads live memory. It also removes the old `v_eff = max(ids) + 1` extent
            // heuristic the snapshot needed to size that buffer.
            //
            // Element-wise this is the identical gather (pure data movement — every output element is a
            // verbatim copy, so there is no reassociation to keep in sync with the runtime kernel), with
            // the runtime's own out-of-range rule preserved: an id outside `[0, v)` zeroes its row.
            I::EmbeddingF32 | I::EmbeddingF32Parallel => {
                let out = ptr(args[0])?;
                let weight = ptr(args[1])?;
                let ids = ptr(args[2])?;
                let t = dim!(args[3]);
                let h = dim!(args[4]);
                let v = args[5].as_int().max(0) as usize;
                // Prove IDs/output addresses before writes, but never snapshot either input:
                // an earlier output can change a later ID or weight in the live gather contract.
                // Each selected row is bounds-checked before its first lane; earlier rows remain
                // written on a later bounds error. Type errors retain earlier lanes too.
                kernel_values(arena!(self), 2, ids, 1, t, name)?;
                kernel_values(arena!(self), 0, out, t, h, name)?;
                for r in 0..t {
                    let id = self
                        .memory
                        .get(ids + r)
                        .ok_or("embedding ids out of bounds")?
                        .as_int();
                    let row = if id >= 0 && (id as usize) < v {
                        let offset = (id as usize)
                            .checked_mul(h)
                            .ok_or("embedding weight extent overflow")?;
                        let base = weight
                            .checked_add(offset)
                            .ok_or("embedding weight out of bounds")?;
                        let end = base
                            .checked_add(h)
                            .ok_or("embedding weight out of bounds")?;
                        if end > self.memory.len() {
                            return Err("embedding weight: operand out of bounds".into());
                        }
                        // The row must lie inside the allocation holding `weight` ITSELF. `v` is
                        // no bound here — a runtime-sized table is passed a sentinel height — so
                        // a row checked only against "some allocation" could sit wholly inside the
                        // next one and gather the neighbour's values.
                        self.slots.live.check_within(name, 1, weight, base, h)?;
                        Some(base)
                    } else {
                        None
                    };
                    for d in 0..h {
                        let val = if let Some(base) = row {
                            self.memory[base + d]
                        } else {
                            Value::Float(0.0)
                        };
                        // Re-marshal through f32 so an `[f32]` table that happens to hold an
                        // integer-typed slot lands as the f32 the native gather would copy.
                        *self
                            .memory
                            .get_mut(out + r * h + d)
                            .ok_or("embedding output out of bounds")? =
                            Value::Float(val.kernel_float(name)? as f32 as f64);
                    }
                }
                Ok(Value::Unit)
            }
            // `wukong_scatter_add_f32[_parallel](grad_w, grad_out, ids, t, h, v)` — the embedding-gradient
            // backward `grad_w[ids[t], :] += grad_out[t, :]` (the gather's dual).
            //
            // Like the gather above, this runs **straight against `self.memory`** rather than
            // marshalling its operands into scratch Vecs, and for the same reason. The marshalled
            // form snapshotted `grad_out` before the first accumulation, so when `grad_w` and
            // `grad_out` are the same buffer — which `tests/run/embedding_alias_parallel.wk` calls
            // as `scatter(ids, acc, acc)`, and which the old run-time guard deliberately sent to the
            // serial kernel (today the guard sends it to the source loop instead, with all three
            // operands range-tested) — the oracle folded a stale copy and answered **49120** where the
            // source nest and the native run both answer **49121**. That was a live interp-vs-native
            // divergence shipping in the corpus, invisible only because the fixture computed the
            // value and declined to print it.
            //
            // The order below is exactly `scatter.rs::scatter_rows` with `r0 = 0, r1 = v`: scan all
            // `t` tokens in ascending `ti`, skip an id outside `[0, v)`, and accumulate the row
            // element-wise. That is also exactly the source nest, so all three implementations now
            // agree on every input, aliased or not. Element-wise f32 accumulation has no
            // reassociation to keep in sync with the AVX2 `add_row` (eight independent lane adds),
            // and the `_parallel` kernel splits *output rows* while keeping each row's ascending-`ti`
            // fold, so one implementation serves both symbols.
            //
            // TWO PROPERTIES OF THE MARSHALLED FORM ARE KEPT DELIBERATELY, because dropping either
            // would be a silent behaviour change rather than the fix this arm is:
            //  * it validated the whole `grad_out[0..t*h]` and `grad_w[0..v*h]` extents BEFORE
            //    computing anything, so an out-of-arena extent was a diagnostic and not a quietly
            //    shorter loop. Reading live memory only touches the slots an in-range id reaches, so
            //    the two range ends are probed up front instead (memory is one flat slot arena, so
            //    the last slot standing in for the range is exact).
            //  * it truncated the id with `as i32` before handing it to the kernel, which is what
            //    NATIVE does — `ids` is an `[i32]` buffer and the kernel's parameter is
            //    `*const i32`, so a value outside i32 is reinterpreted, not skipped. A well-typed
            //    program cannot produce one; matching native costs one cast.
            I::ScatterAddF32 | I::ScatterAddF32Parallel => {
                let grad_w = ptr(args[0])?;
                let grad_out = ptr(args[1])?;
                let ids = ptr(args[2])?;
                let t = dim!(args[3]);
                let h = dim!(args[4]);
                let v = dim!(args[5]);
                let end = self.memory.len();
                // Checked throughout: `t`/`h`/`v` are program-supplied dims, so `t*h` is an
                // attacker-shaped product and a wrap would turn the guard into a pass.
                let in_arena = |base: usize, n: Option<usize>| {
                    n.and_then(|n| base.checked_add(n))
                        .is_some_and(|last| last <= end)
                };
                if !in_arena(ids, Some(t)) {
                    return Err("scatter_add ids out of bounds".into());
                }
                if !in_arena(grad_out, t.checked_mul(h)) {
                    return Err("scatter_add grad_out out of bounds".into());
                }
                if !in_arena(grad_w, v.checked_mul(h)) {
                    return Err("scatter_add grad_w out of bounds".into());
                }
                // ...and each inside ONE live allocation: a table shorter than `v` rows that runs
                // into the next allocation is refused rather than accumulated into. (The products
                // are known not to overflow here.)
                for (k, base, len) in [
                    (2, ids, Some(t)),
                    (1, grad_out, t.checked_mul(h)),
                    (0, grad_w, v.checked_mul(h)),
                ] {
                    self.slots
                        .live
                        .check(name, k, base, len.unwrap_or(usize::MAX))?;
                }
                for ti in 0..t {
                    let id = self
                        .memory
                        .get(ids + ti)
                        .ok_or("scatter_add ids out of bounds")?
                        .as_int() as i32;
                    if id < 0 || (id as usize) >= v {
                        continue;
                    }
                    let row = id as usize * h;
                    for j in 0..h {
                        let g = self
                            .memory
                            .get(grad_out + ti * h + j)
                            .ok_or("scatter_add grad_out out of bounds")?
                            .kernel_float(name)? as f32;
                        let slot = self
                            .memory
                            .get_mut(grad_w + row + j)
                            .ok_or("scatter_add grad_w out of bounds")?;
                        let acc = slot.kernel_float(name)? as f32;
                        *slot = Value::Float((acc + g) as f64);
                    }
                }
                Ok(Value::Unit)
            }
            // `wukong_attention_f32(q, k, v, out, s, d, scale, causal)` — fused scaled-dot-product
            // attention (the flash-attention kernel). Like the GEMM path, the interpreter marshals
            // q/k/v out of its abstract memory into real f32 buffers, calls the *identical* runtime
            // kernel the native backend calls, then writes the output back — so the online-softmax
            // reassociation stays bit-for-bit exact across the two backends.
            I::AttentionF32 => {
                let q = ptr(args[0])?;
                let k = ptr(args[1])?;
                let v = ptr(args[2])?;
                let out = ptr(args[3])?;
                let s = dim!(args[4]);
                let d = dim!(args[5]);
                let scale = args[6].as_float() as f32;
                let causal = args[7].as_int() as i64;
                let len = marshal::span(s, d, name)?;
                // attention.rs: the output must not overlap `q`, `k` or `v` (each output row is
                // written while later rows still read all three). `sdpa(...)` is a user-level call
                // with no source loop to fall back to, so an overlapping one is reported here
                // rather than answered with a copied, de-aliased value.
                marshal::check_output(
                    arena!(self),
                    name,
                    (3, out, len),
                    &[
                        ((0, q, len), Overlap::Forbidden),
                        ((1, k, len), Overlap::Forbidden),
                        ((2, v, len), Overlap::Forbidden),
                    ],
                )?;
                let qbuf = marshal::read_f32(arena!(self), 0, q, len, name)?;
                let kbuf = marshal::read_f32(arena!(self), 1, k, len, name)?;
                let vbuf = marshal::read_f32(arena!(self), 2, v, len, name)?;
                let mut obuf = marshal::output_f32(arena!(self), 3, out, len, name)?;
                // SAFETY: buffers are exactly s*d long — the kernel's operand contract.
                unsafe {
                    wukong_runtime::wukong_attention_f32(
                        qbuf.as_ptr(),
                        kbuf.as_ptr(),
                        vbuf.as_ptr(),
                        obuf.as_mut_ptr(),
                        s as i64,
                        d as i64,
                        scale,
                        causal,
                    );
                }
                marshal::write_f32(arena_mut!(self), 3, out, &obuf, name)?;
                marshal::recycle_f32(qbuf);
                marshal::recycle_f32(kbuf);
                marshal::recycle_f32(vbuf);
                marshal::recycle_f32(obuf);
                Ok(Value::Unit)
            }
            // The text, ML and decode builtins' runtime entries live in `text_io.rs`, `ml_io.rs`
            // and `decode.rs`; anything none of them claims is an unknown callee.
            other => self
                .text_intrinsic(other, args)
                .or_else(|| self.ml_intrinsic(other, args))
                .or_else(|| self.decode_intrinsic(other, args))
                .unwrap_or_else(|| Err(format!("call to unknown function or intrinsic `{name}`"))),
        }
    }
}

/// The result class the interpreter reads an `extern "C"` function's return value as, or why it
/// cannot call the function at all. The interpreter's memory is slot-indexed, not machine memory,
/// so only scalar signatures cross (see `wukong_runtime::cffi`): integers, `bool`, `f32` and `f64`
/// parameters — at most `MAX_SCALAR_ARGS` of them — and a `void`, integer or float result.
fn extern_ret_class(ext: &wukong_mir::ExternFn) -> Result<wukong_runtime::cffi::CRet, String> {
    use wukong_runtime::cffi::{CRet, MAX_SCALAR_ARGS};
    if ext.params.len() > MAX_SCALAR_ARGS {
        return Err(format!(
            "it takes {} parameters and the interpreter marshals at most {MAX_SCALAR_ARGS}",
            ext.params.len()
        ));
    }
    for p in &ext.params {
        if p.ty == MirType::I128 || !(p.ty.is_int() || matches!(p.ty, MirType::F32 | MirType::F64))
        {
            return Err(format!(
                "a `{}` parameter cannot leave the interpreter's slot memory",
                p.ty.display()
            ));
        }
    }
    Ok(match &ext.ret.ty {
        MirType::Void => CRet::Void,
        MirType::F32 => CRet::F32,
        MirType::F64 => CRet::F64,
        MirType::I64 => CRet::Int64,
        MirType::I1 | MirType::I8 | MirType::I16 | MirType::I32 => CRet::Int32,
        other => {
            return Err(format!(
                "a `{}` result cannot enter the interpreter's slot memory",
                other.display()
            ))
        }
    })
}

/// Every `extern "C"` function of `program` the interpreter cannot call, one message each: a
/// non-scalar signature, or a symbol no C library loaded into this process defines. The driver
/// checks this before an interpreted run so the refusal is a compile-time diagnostic, not a
/// failure halfway through the program's output.
pub fn unsupported_externs(program: &Program, interner: &Interner) -> Vec<String> {
    let mut out = Vec::new();
    for ext in &program.externs {
        let name = interner.resolve(ext.name);
        if let Err(why) = extern_ret_class(ext) {
            out.push(format!(
                "extern \"C\" function `{name}` cannot run on the interpreter: {why} (use \
                 --backend=native)"
            ));
        } else if wukong_runtime::cffi::resolve_c_symbol(name).is_none() {
            out.push(format!(
                "unresolved external symbol `{name}`: no C library loaded into this process \
                 defines it"
            ));
        }
    }
    out
}

/// Call an `extern "C"` function: the host symbol `wukong_runtime::cffi` resolves — the very
/// address the native JIT binds — through the scalar marshaller, so interp and native run the same
/// machine code. Integer arguments are extended to 64 bits per their declared signedness (the
/// interpreter keeps every integer sign-extended at its width); an integer result is re-normalized
/// to the declared width.
fn call_extern(
    ext: &wukong_mir::ExternFn,
    args: &[Value],
    interner: &Interner,
) -> Result<Value, String> {
    use wukong_runtime::cffi::{call_scalar, resolve_c_symbol, CArg, CValue};
    let name = interner.resolve(ext.name);
    let ret = extern_ret_class(ext).map_err(|why| {
        format!("extern \"C\" function `{name}` cannot run on the interpreter: {why}")
    })?;
    let addr = resolve_c_symbol(name).ok_or_else(|| {
        format!(
            "unresolved external symbol `{name}`: no C library loaded into this process defines it"
        )
    })?;
    let mut cargs = Vec::with_capacity(ext.params.len());
    for (p, v) in ext.params.iter().zip(args) {
        cargs.push(match &p.ty {
            MirType::F32 => CArg::F32(v.as_float() as f32),
            MirType::F64 => CArg::F64(v.as_float()),
            ty => {
                let raw = v.as_int();
                let wide = if p.signed {
                    mask(raw, ty)
                } else {
                    uval(raw, int_bits(ty)) as i128
                };
                CArg::Int(wide as i64)
            }
        });
    }
    // SAFETY: `addr` is the C symbol the program declared with exactly this signature (sema typed
    // every call against it and the MIR verifier checked the arity and types), each argument is
    // marshalled in its declared register class, and `ret` is the declared result's class. A
    // declaration that lies about the C function is undefined behavior, as in C itself.
    let out = unsafe { call_scalar(addr, &cargs, ret) }
        .ok_or_else(|| format!("extern \"C\" function `{name}`: unsupported signature shape"))?;
    Ok(match out {
        CValue::Void => Value::Unit,
        CValue::Int(i) => Value::Int(mask(i128::from(i), &ext.ret.ty)),
        CValue::F32(x) => Value::Float(f64::from(x)),
        CValue::F64(x) => Value::Float(x),
    })
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

/// The number of flat-memory slots a value of `ty` occupies — the interpreter's analogue of
/// `size_of`. Every scalar (any width), pointer, or vector reference is one slot; an `Array(elem, n)`
/// is `n` element-runs laid out contiguously, so it occupies `n * slot_count(elem)` slots (recursing
/// for an array of aggregates). This is the stride a `Gep` over such an element must use so that
/// element `i` lands at `base + i * slot_count(elem)` — matching native, which scales the GEP index
/// by `size_of(elem)` bytes. For a scalar/byte (`I8`) element this is `1`, so a scalar array and a
/// struct/tuple byte-buffer field GEP are unchanged; only an *aggregate-element* array (`[Struct; N]`,
/// `[(..); N]`) is affected — previously mis-strided by a single slot.
fn slot_count(ty: &MirType) -> usize {
    match ty {
        MirType::Array(elem, count) => *count as usize * slot_count(elem),
        _ => 1,
    }
}

/// Reserve the flat-memory slots for an `alloca` of `ty`, pushing a typed zero per leaf slot. An
/// array recurses element-by-element (so a nested `Array(Array(I8, 8), 2)` reserves all 16 leaf
/// slots, not 2), keeping the per-leaf default type (`Float(0.0)` for an f32 array, `Int(0)` for a
/// byte buffer) the way the old single-level loop did for a scalar array.
fn push_defaults(ty: &MirType, out: &mut Vec<Value>) {
    if let MirType::Array(elem, count) = ty {
        for _ in 0..*count {
            push_defaults(elem, out);
        }
    } else {
        out.push(default_value(ty));
    }
}

fn int_bits(ty: &MirType) -> u32 {
    match ty {
        MirType::I1 => 1,
        MirType::I8 => 8,
        MirType::I16 => 16,
        MirType::I32 => 32,
        MirType::I64 => 64,
        MirType::I128 => 128,
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

/// Pick the correctly single-rounded int→float result for the cast target. A narrow float
/// (`f32`/`bf16`/`f16`) uses `narrow` — the integer rounded straight to `f32` — matching native's
/// single `fcvt_from_{sint,uint}(F32)`; an `f64` target uses `wide` (the `f64` rounding). Both are
/// computed by the caller with one Rust `as` conversion (IEEE round-to-nearest-even). Going through
/// `f64` and re-rounding to `f32` double-rounds and disagrees with native above 2^53.
fn int_to_float(wide: f64, narrow: f32, to: &MirType) -> f64 {
    if matches!(
        to,
        MirType::F32 | MirType::BF16 | MirType::F16 | MirType::F8E4M3 | MirType::F8E5M2
    ) {
        narrow as f64
    } else {
        wide
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

/// Truncate one *lane* of a SIMD result to its lane width — the vector twin of the normalization
/// `exec` applies to scalar results.
///
/// `exec` only masks a `Value::Int` whose declared result type `is_int()`, and a vector result is a
/// `Value::VecRef` of a `MirType::Vec` (not an int type), so neither guard reaches a lane. The lane
/// closures below compute in `i128` (`apply_bin` wraps with `i128::wrapping_*`, `Op::Neg`/`Op::Not`
/// negate/complement an `i128`), so without this an `<4 x i32>` add of 2000000000 + 2000000000
/// yields 4000000000 where the native backend wraps to -294967296: the oracle contradicting the
/// backend it certifies. Float lanes are already rounded to the lane type by the callers and pass
/// through unchanged.
fn mask_lane(v: Value, lane: &MirType) -> Value {
    match v {
        Value::Int(i) if lane.is_int() => Value::Int(mask(i, lane)),
        other => other,
    }
}

/// One lane of an `Op::Narrow` into the lane type `to`, from a source lane held (like every
/// interpreter integer) as its sign-extended value. The saturating kinds clamp that signed value —
/// `packss` to the signed range, `packus` to `0 ..= 2^w - 1` — and the result is then held in the
/// narrow type's own canonical form, so `SatUnsigned` 255 into `i8` is the byte `0xFF`, i.e. `-1`.
fn narrow_lane(kind: wukong_mir::NarrowKind, x: Value, to: &MirType) -> Value {
    use wukong_mir::NarrowKind;
    let bits = int_bits(to);
    match kind {
        NarrowKind::Trunc => Value::Int(mask(x.as_int(), to)),
        NarrowKind::SatSigned => {
            let (min, max) = (-(1i128 << (bits - 1)), (1i128 << (bits - 1)) - 1);
            Value::Int(x.as_int().clamp(min, max))
        }
        NarrowKind::SatUnsigned => {
            let max = (1i128 << bits) - 1;
            Value::Int(mask(x.as_int().clamp(0, max), to))
        }
        NarrowKind::FpTrunc => Value::Float(x.as_float() as f32 as f64),
    }
}

/// `f16`/`bf16`/`f32` results are rounded to `f32` precision (the interpreter promotes sub-`f32`
/// storage types to `f32`); `f64` keeps full precision.
fn is_narrow_float(ty: &MirType) -> bool {
    matches!(
        ty,
        MirType::F16 | MirType::BF16 | MirType::F8E4M3 | MirType::F8E5M2 | MirType::F32
    )
}

/// `v` rounded onto the grid of the 8-bit float `ty` (nearest, ties to even, ONE rounding from
/// the `f64` it is held as), by the bit-level codec every backend runs
/// (`wukong_runtime::fp8`): E4M3FN saturates at ±448, E5M2 overflows to ±∞, a NaN becomes the
/// format's canonical NaN. Any other `ty` returns `v` unchanged.
fn round_f8(v: f64, ty: &MirType) -> f64 {
    use wukong_runtime::fp8::{decode_f32, encode_f64, Fp8Format};
    let fmt = match ty {
        MirType::F8E4M3 => Fp8Format::E4M3,
        MirType::F8E5M2 => Fp8Format::E5M2,
        _ => return v,
    };
    f64::from(decode_f32(encode_f64(v, fmt), fmt))
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
fn apply_round(mode: wukong_mir::RoundMode, v: Value, rty: Option<&MirType>) -> Value {
    use wukong_mir::RoundMode::*;
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
            // Division by zero yields 0 (matched to Cranelift), same as the sibling arms.
            xu.checked_div(yu).unwrap_or(0) as i128
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
        // Int→float must round in ONE step to the target's precision. A narrow (`f32`/`bf16`/`f16`)
        // target rounds the integer directly to `f32` (`as f32`), matching native's single
        // `fcvt_from_sint(F32)`; routing through `f64` first (`as f64`, then `exec`'s `as f32`)
        // double-rounds and disagrees with native for magnitudes above 2^53. An `f64` target rounds
        // to `f64` (a single rounding, and `exec` does not re-round `f64`).
        SiToFp => Value::Float(int_to_float(v.as_int() as f64, v.as_int() as f32, to)),
        // Unsigned→float: read the source as unsigned in its own width first (matches native
        // `fcvt_from_uint`); `as_int()` would be negative for a high-bit-set value.
        UiToFp => {
            let u = uval(v.as_int(), int_bits(from));
            Value::Float(int_to_float(u as f64, u as f32, to))
        }
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
                MirType::I128 => f as i128,
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
                // The u128 bit pattern, held (like every 128-bit value) as its i128 reinterpretation.
                MirType::I128 => f as u128 as i128,
                _ => f as u32 as i128, // I32 (and the I1 fallback)
            };
            Value::Int(mask(i, to))
        }
        // Widening to f32/f64 is value-preserving for an f32 source, but a bf16/f16 source must first
        // round to its grid: the per-store rounding alone is fragile (an optimizer can promote/forward
        // a bf16 op result past its store, leaving an unrounded f32), so the observation boundary
        // rounds too — the "round at store/cast/load" model — keeping -O0 == -O2. Narrowing to bf16/f16
        // rounds to that grid (the native backend rounds on store/cast identically); narrowing to f32
        // is left to the per-op f32 rounding in `exec`.
        FpExt => match from {
            MirType::BF16 => Value::Float(wukong_runtime::round_bf16(v.as_float() as f32) as f64),
            MirType::F16 => Value::Float(wukong_runtime::round_f16(v.as_float() as f32) as f64),
            MirType::F8E4M3 | MirType::F8E5M2 => Value::Float(round_f8(v.as_float(), from)),
            _ => Value::Float(v.as_float()),
        },
        // An f8 target rounds the source's exact value once — an `f64` straight to the grid, not
        // through `f32` (which would round twice).
        FpTrunc => match to {
            MirType::BF16 => Value::Float(wukong_runtime::round_bf16(v.as_float() as f32) as f64),
            MirType::F16 => Value::Float(wukong_runtime::round_f16(v.as_float() as f32) as f64),
            MirType::F8E4M3 | MirType::F8E5M2 => Value::Float(round_f8(v.as_float(), to)),
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
    use wukong_span::SourceId;

    fn run_main(src: &str) -> i64 {
        let mut interner = Interner::new();
        let (module, pd) = wukong_parser::parse_module(src, SourceId(0), &mut interner);
        assert!(pd.is_empty(), "parse: {pd:?}");
        let (sema, sd) = wukong_sema::check(&module, &interner);
        assert!(sd.iter().all(|d| !d.is_error()), "sema: {sd:?}");
        let (program, _ld) = wukong_mir_build::lower_program(&module, &sema, &mut interner);
        let main = interner.intern("main");
        run(&program, main, &interner).unwrap()
    }

    /// Like [`run_main`] but returns what the program printed (for results wider than the `i32`
    /// exit code).
    fn run_main_stdout(src: &str) -> String {
        let mut interner = Interner::new();
        let (module, pd) = wukong_parser::parse_module(src, SourceId(0), &mut interner);
        assert!(pd.is_empty(), "parse: {pd:?}");
        let (sema, sd) = wukong_sema::check(&module, &interner);
        assert!(sd.iter().all(|d| !d.is_error()), "sema: {sd:?}");
        let (program, _ld) = wukong_mir_build::lower_program(&module, &sema, &mut interner);
        let main = interner.intern("main");
        let (_, out) = run_with_output(&program, main, &interner).unwrap();
        String::from_utf8(out).unwrap()
    }

    /// An integer SIMD lane must wrap at its **lane width**, exactly as the scalar path does — the
    /// interpreter is the oracle the native backend is certified against, and Cranelift wraps at
    /// 32/64 bits. The reference values here are hand-computed two's-complement truths, not native
    /// readings: 2000000000+2000000000 = 4000000000 ≡ -294967296 (mod 2^32), 100000*100000 =
    /// 10^10 ≡ 1410065408 (mod 2^32), -(-2^31) ≡ -2^31, and 9e18+9e18 = 1.8e19 ≡ -446744073709551616
    /// (mod 2^64). `mir_build`'s AST vectorizer turns each `for k in 0..64` body into `<4 x i32>` /
    /// `<2 x i64>` ops, so these exercise the vector arms of `eval`, not the scalar ones.
    #[test]
    fn integer_vector_lanes_wrap_at_lane_width() {
        let fill = "let mut a: [i32; 64] = [0; 64]; let mut b: [i32; 64] = [0; 64]; \
                    let mut i: i32 = 0; \
                    while i < 64 { a[i] = 2000000000; b[i] = 2000000000; i = i + 1; }";
        assert_eq!(
            run_main(&format!(
                "fn main() -> i32 {{ {fill} for k in 0..64 {{ a[k] = a[k] + b[k]; }} return a[0]; }}"
            )),
            -294967296,
            "i32 vector add must wrap"
        );

        let fill_mul = "let mut a: [i32; 64] = [0; 64]; let mut b: [i32; 64] = [0; 64]; \
                        let mut i: i32 = 0; \
                        while i < 64 { a[i] = 100000; b[i] = 100000; i = i + 1; }";
        assert_eq!(
            run_main(&format!(
                "fn main() -> i32 {{ {fill_mul} for k in 0..64 {{ a[k] = a[k] * b[k]; }} return a[0]; }}"
            )),
            1410065408,
            "i32 vector multiply must wrap"
        );

        let fill_neg = "let mut a: [i32; 64] = [0; 64]; let mut i: i32 = 0; \
                        while i < 64 { a[i] = -2147483647 - 1; i = i + 1; }";
        assert_eq!(
            run_main(&format!(
                "fn main() -> i32 {{ {fill_neg} for k in 0..64 {{ a[k] = -a[k]; }} return a[0]; }}"
            )),
            -2147483648,
            "negating i32::MIN in a lane must wrap back to i32::MIN"
        );

        let fill64 = "let mut a: [i64; 64] = [0; 64]; let mut b: [i64; 64] = [0; 64]; \
                      let mut i: i64 = 0; \
                      while i < 64 { a[i] = 9000000000000000000; \
                      b[i] = 9000000000000000000; i = i + 1; }";
        assert_eq!(
            run_main_stdout(&format!(
                "fn main() -> i32 {{ {fill64} for k in 0..64 {{ a[k] = a[k] + b[k]; }} \
                 print(a[0]); return 0; }}"
            )),
            "-446744073709551616\n",
            "i64 vector add must wrap"
        );
    }

    /// Unbounded recursion must produce a diagnostic, not a stack overflow. A stack overflow is an
    /// uncatchable process abort: it takes the whole differential-oracle run (or test binary) down
    /// and prints an internal Rust message in place of a catalogued error. `exec`'s 100M-step guard
    /// cannot cover this — it is a local of one `exec` invocation, so a recursion that runs a few
    /// blocks per frame never increments a shared counter.
    #[test]
    fn unbounded_recursion_is_a_diagnostic_not_a_process_abort() {
        let mut interner = Interner::new();
        let src = "fn f(n: i32) -> i32 { return f(n + 1); } fn main() -> i32 { return f(0); }";
        let (module, _) = wukong_parser::parse_module(src, SourceId(0), &mut interner);
        let (sema, _) = wukong_sema::check(&module, &interner);
        let (program, _) = wukong_mir_build::lower_program(&module, &sema, &mut interner);
        let main = interner.intern("main");
        let err = run(&program, main, &interner).unwrap_err();
        assert!(err.contains("call-depth limit exceeded"), "got: {err}");
    }

    /// The typed kernel-entry ABI must give the tree-walker the same recursion headroom `run` gets
    /// (`deep_recursion_does_not_overflow_oracle` below). When calls recursed on the host stack, a
    /// kernel entry on the caller's default stack aborted the whole test binary at this depth with
    /// STATUS_STACK_OVERFLOW; calls are now stackless, so it runs on the caller's thread.
    #[test]
    fn kernel_entry_deep_recursion_does_not_overflow_oracle() {
        let mut interner = Interner::new();
        let src = "module m\n\
            fn rec(v: f32, n: i32) -> f32 { if n == 0 { return v; } return rec(v, n - 1) + 1.0; }\n\
            fn k(x: [f32; 1], mut out: [f32; 1]) { out[0] = rec(x[0], 20000); }\n";
        let (module, pd) = wukong_parser::parse_module(src, SourceId(0), &mut interner);
        assert!(pd.iter().all(|d| !d.is_error()), "parse: {pd:?}");
        let (sema, sd) = wukong_sema::check(&module, &interner);
        assert!(sd.iter().all(|d| !d.is_error()), "sema: {sd:?}");
        let (program, ld) = wukong_mir_build::lower_program(&module, &sema, &mut interner);
        assert!(ld.iter().all(|d| !d.is_error()), "lower: {ld:?}");
        let mut x = vec![0.0f32; 1];
        let mut out = vec![0.0f32; 1];
        run_kernel_f32(
            &program,
            interner.intern("k"),
            &mut [&mut x, &mut out],
            &interner,
        )
        .unwrap();
        assert_eq!(out[0], 20000.0);
    }

    /// Scalar-to-vector casts must be rejected by the verifier. Direct interpreter callers
    /// can still supply unverified MIR, so retain the runtime error guard as well: this used
    /// to index a one-element lane list and panic instead of returning a diagnostic.
    #[test]
    fn vector_op_with_a_scalar_operand_is_an_error_not_a_panic() {
        use wukong_mir::{
            BasicBlock, BlockId, CastKind, Function, Inst, MirLevel, Op, Terminator, ValueId,
        };
        let mut interner = Interner::new();
        let name = interner.intern("main");
        // %0: i32 = const 7 ; %1: <4 x i32> = cast sext %0 ; ret %0
        let f = Function {
            name,
            params: Vec::new(),
            ret: MirType::I32,
            blocks: vec![BasicBlock {
                id: BlockId(0),
                params: Vec::new(),
                insts: vec![
                    Inst {
                        result: Some(ValueId(0)),
                        op: Op::ConstInt(7, MirType::I32),
                    },
                    Inst {
                        result: Some(ValueId(1)),
                        op: Op::Cast(
                            CastKind::SExt,
                            ValueId(0),
                            MirType::Vec(Box::new(MirType::I32), 4),
                        ),
                    },
                ],
                term: Terminator::Ret(Some(ValueId(0))),
            }],
            value_types: vec![MirType::I32, MirType::Vec(Box::new(MirType::I32), 4)],
            entry: BlockId(0),
            vec_kernels: Vec::new(),
            param_attrs: Vec::new(),
            locs: Default::default(),
        };
        let mut program = Program::new();
        program.funcs.push(f);
        program.level = MirLevel::Low;
        assert!(
            !wukong_mir::verify::verify_program(&program).is_empty(),
            "the verifier must reject a scalar-to-vector extension"
        );
        let err = run(&program, name, &interner).unwrap_err();
        assert!(err.contains("not a 4-lane vector"), "got: {err}");
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
    fn deep_recursion_does_not_overflow_oracle() {
        // 1+2+...+1000 = 500500.
        let src = "fn sum(n: i32) -> i32 { if n == 0 { return 0; } return n + sum(n - 1); } \
                   fn main() -> i32 { return sum(1000); }";
        assert_eq!(run_main(src), 500500);
    }

    /// Calls are stackless, so a recursion far deeper than any host stack holds runs right on the
    /// test thread (2 MiB), with the same ceiling in a debug and a release build. The host-recursive
    /// tree-walker refused this depth in a debug build (its bisected ceiling was 40_000) and needed a
    /// 512 MiB worker thread for the depths it did accept. 1+2+...+123456 = 7620753696, and
    /// 7620753696 mod 1000 = 696.
    #[test]
    fn stackless_recursion_depth_is_profile_independent() {
        let src = "fn sum(n: i64) -> i64 { if n == 0 { return 0; } return n + sum(n - 1); } \
                   fn main() -> i32 { return (sum(123456) % 1000) as i32; }";
        assert_eq!(run_main(src), 696);
    }

    /// A recognized kernel called with a **non-positive extent** is a no-op: the shapes come from
    /// loop bounds, and `for i in 0..n` with `n < 0` runs zero iterations, so the recognized form
    /// must leave the output buffer alone — which is what every `wukong_runtime` kernel does
    /// (`if rows <= 0 || cols <= 0 { return; }`) and therefore what `--backend=native` does. The
    /// interpreter used to read the extent as a bare `as usize`, wrapping to ~`usize::MAX`, and the
    /// `Vec::with_capacity(rows * cols)` that followed aborted the process with a raw Rust
    /// `capacity overflow` panic. Each case below dispatches to a different kernel family.
    #[test]
    fn non_positive_kernel_extent_is_a_no_op() {
        // Fused norm (`wukong_norm_f32`): negative row count.
        assert_eq!(
            run_main(
                "fn norm_rows(mut x: [f32; 12], r: i64) { \
                   for rr in 0..r { let mut ss: f32 = 0.0; \
                     for i in 0..4 { ss = ss + x[rr*4+i] * x[rr*4+i]; } \
                     let inv = 1.0 / sqrt(ss / 4.0 + 0.00001); \
                     for i in 0..4 { x[rr*4+i] = x[rr*4+i] * inv; } } } \
                 fn main() -> i32 { let mut x: [f32; 12] = [1.0; 12]; let n: i64 = 0 - 1; \
                   norm_rows(x, n); return (x[0] * 10.0) as i32; }"
            ),
            10,
            "negative rows must leave x untouched"
        );

        // GEMM (`wukong_sgemm_nt`): negative M.
        assert_eq!(
            run_main(
                "fn mm(a: [f32; 16], b: [f32; 16], mut c: [f32; 16], m: i64) { \
                   for i in 0..m { for j in 0..4 { let mut s: f32 = 0.0; \
                     for k in 0..4 { s = s + a[i*4+k] * b[j*4+k]; } c[i*4+j] = s; } } } \
                 fn main() -> i32 { let mut a: [f32;16] = [1.0;16]; let mut b: [f32;16] = [1.0;16]; \
                   let mut c: [f32;16] = [2.0;16]; let m: i64 = 0 - 1; mm(a, b, c, m); \
                   return (c[0] * 10.0) as i32; }"
            ),
            20,
            "negative M must leave C untouched"
        );

        // Fused attention (`wukong_attention_f32`): non-positive sequence length.
        //
        // The source-level `sdpa(..)` is a shape-checked kernel builtin (65541ffd): a runtime
        // extent outside `0..=i32::MAX` traps through `assert` BEFORE the kernel is reached, so a
        // negative `s` can no longer arrive at `wukong_attention_f32` from source. An empty `s`
        // still passes the builtin's checks and reaches the kernel, which must be a no-op.
        let sdpa_with_s = |s: &str| {
            format!(
                "fn main() -> i32 {{ let mut q: [f32; 16] = [0.0; 16]; let mut k: [f32; 16] = [0.0; 16]; \
                   let mut v: [f32; 16] = [0.0; 16]; let mut o: [f32; 16] = [1.0; 16]; \
                   for i in 0..16 {{ q[i] = (i as f32) * 0.1; k[i] = (i as f32) * 0.2; \
                     v[i] = (i as f32) * 0.5; }} \
                   sdpa(q, k, v, o, {s}, 4, 0.5, 0); return (o[0] * 100.0) as i32; }}"
            )
        };
        assert_eq!(
            run_main(&sdpa_with_s("0")),
            100,
            "an empty S must leave the attention output untouched"
        );
        {
            let src = sdpa_with_s("-1");
            let mut interner = Interner::new();
            let (module, pd) = wukong_parser::parse_module(&src, SourceId(0), &mut interner);
            assert!(pd.is_empty(), "parse: {pd:?}");
            let (sema, sd) = wukong_sema::check(&module, &interner);
            assert!(sd.iter().all(|d| !d.is_error()), "sema: {sd:?}");
            let (program, _ld) = wukong_mir_build::lower_program(&module, &sema, &mut interner);
            let main = interner.intern("main");
            assert_eq!(
                run(&program, main, &interner).unwrap_err(),
                "assertion failed",
                "the sdpa builtin must trap a negative runtime S before the kernel"
            );
        }

        // The kernel-level contract the builtin check now shields: a negative `s` handed straight
        // to `wukong_attention_f32` (as hand-built MIR or a recognizer could) is a no-op, exactly
        // as the runtime kernel the native backend calls treats it — never a wrapped `usize`.
        let program = Program::default();
        let interner = Interner::new();
        let mut interp = Interp {
            program: &program,
            interner: &interner,
            memory: Vec::new(),
            stdout: Vec::new(),
            scratch_arena: ScratchSlots::default(),
            slots: SlotReuse::default(),
            frames: Vec::new(),
            call_args: Vec::new(),
            scratch: Vec::new(),
            vecs: Vec::new(),
            #[cfg(test)]
            vector_peak: 0,
            data_addrs: HashMap::new(),
            function_indices: None,
            static_indices: None,
            accel: None,
            depth: 0,
        };
        let initial: Vec<Value> = (0..64).map(|i| Value::Float(f64::from(i) * 0.25)).collect();
        for s in [0, -1, i128::from(i64::MIN)] {
            interp.memory.clone_from(&initial);
            interp
                .intrinsic(
                    "wukong_attention_f32",
                    &[
                        Value::Ptr(0),
                        Value::Ptr(16),
                        Value::Ptr(32),
                        Value::Ptr(48),
                        Value::Int(s),
                        Value::Int(4),
                        Value::Float(0.5),
                        Value::Int(0),
                    ],
                )
                .unwrap();
            assert_eq!(
                interp.memory, initial,
                "S = {s} must leave the attention output untouched"
            );
        }
    }

    #[test]
    fn pool2d_shared_shape_marshalling_and_checked_extents() {
        let program = Program::default();
        let interner = Interner::new();
        let mut interp = Interp {
            program: &program,
            interner: &interner,
            memory: Vec::new(),
            stdout: Vec::new(),
            scratch_arena: ScratchSlots::default(),
            slots: SlotReuse::default(),
            frames: Vec::new(),
            call_args: Vec::new(),
            scratch: Vec::new(),
            vecs: Vec::new(),
            #[cfg(test)]
            vector_peak: 0,
            data_addrs: HashMap::new(),
            function_indices: None,
            static_indices: None,
            accel: None,
            depth: 0,
        };
        for name in [
            "wukong_maxpool2d_f32",
            "wukong_maxpool2d_f32_parallel",
            "wukong_avgpool2d_f32",
            "wukong_avgpool2d_f32_parallel",
        ] {
            let initial: Vec<Value> = [77.0]
                .into_iter()
                .chain((1..=12).map(f64::from))
                .chain([19.0; 5])
                .map(Value::Float)
                .collect();
            let args = |x, out, dims: [i128; 7]| {
                let mut args = vec![Value::Ptr(x), Value::Ptr(out)];
                args.extend(dims.map(Value::Int));
                args
            };
            interp.memory.clone_from(&initial);
            interp
                .intrinsic(name, &args(1, 13, [1, 3, 4, 2, 2, 1, 2]))
                .unwrap();
            let expected = if name.contains("avg") {
                [3.5, 5.5, 7.5, 9.5]
            } else {
                [6.0, 8.0, 10.0, 12.0]
            };
            assert_eq!(
                &interp.memory[13..17],
                &expected.map(Value::Float),
                "{name}"
            );
            assert_eq!(&interp.memory[..13], &initial[..13]);
            assert_eq!(interp.memory[17], initial[17]);
            for index in 0..7 {
                for invalid in [0, -1, i128::MIN] {
                    interp.memory.clone_from(&initial);
                    let mut dims = [1, 3, 4, 2, 2, 1, 2];
                    dims[index] = invalid;
                    interp
                        .intrinsic(name, &args(usize::MAX, usize::MAX, dims))
                        .unwrap();
                    assert_eq!(interp.memory, initial);
                }
                let mut dims = [1, 3, 4, 2, 2, 1, 2];
                dims[index] = i128::from(i64::MAX) + 1;
                assert_eq!(
                    interp.intrinsic(name, &args(1, 13, dims)).unwrap_err(),
                    "pool2d extent exceeds i64"
                );
                assert_eq!(interp.memory, initial);
            }
            for dims in [
                [1, 3, 4, 4, 2, 1, 2],
                [1, 3, 4, 2, 5, 1, 2],
                [0, i128::MAX, 4, 2, 2, 1, 2],
            ] {
                interp
                    .intrinsic(name, &args(usize::MAX, usize::MAX, dims))
                    .unwrap();
                assert_eq!(interp.memory, initial);
            }
            for (x, out, dims, error) in [
                (
                    1,
                    13,
                    [i128::from(i64::MAX), 3, 4, 2, 2, 1, 2],
                    "pool2d operand extent overflow",
                ),
                (
                    usize::MAX,
                    13,
                    [1, 3, 4, 2, 2, 1, 2],
                    "pool2d operand out of bounds",
                ),
                (
                    1,
                    usize::MAX,
                    [1, 3, 4, 2, 2, 1, 2],
                    "pool2d output out of bounds",
                ),
                (1, 15, [1, 3, 4, 2, 2, 1, 2], "pool2d output out of bounds"),
            ] {
                assert_eq!(
                    interp.intrinsic(name, &args(x, out, dims)).unwrap_err(),
                    error,
                    "{name}"
                );
                assert_eq!(interp.memory, initial);
            }
        }
    }

    #[test]
    fn sgemm_checks_extents_before_allocating_scratch() {
        let program = Program::default();
        let interner = Interner::new();
        let mut interp = Interp {
            program: &program,
            interner: &interner,
            memory: Vec::new(),
            stdout: Vec::new(),
            scratch_arena: ScratchSlots::default(),
            slots: SlotReuse::default(),
            frames: Vec::new(),
            call_args: Vec::new(),
            scratch: Vec::new(),
            vecs: Vec::new(),
            #[cfg(test)]
            vector_peak: 0,
            data_addrs: HashMap::new(),
            function_indices: None,
            static_indices: None,
            accel: None,
            depth: 0,
        };
        for layout in ["", "_nt", "_tn"] {
            for suffix in ["", "_parallel"] {
                let name = format!("wukong_sgemm{layout}{suffix}");
                let a = if layout == "_tn" {
                    [1.0, 3.0, 2.0, 4.0]
                } else {
                    [1.0, 2.0, 3.0, 4.0]
                };
                let b = if layout == "_nt" {
                    [5.0, 7.0, 6.0, 8.0]
                } else {
                    [5.0, 6.0, 7.0, 8.0]
                };
                let initial: Vec<Value> = [99.0]
                    .into_iter()
                    .chain(a)
                    .chain(b)
                    .chain([1.0, 2.0, 3.0, 4.0, 77.0])
                    .map(Value::Float)
                    .collect();
                for (m, k, n, ap, bp, cp, error) in [
                    (i64::MAX, 4, 1, 1, 5, 9, "sgemm operand extent overflow"),
                    (1, i64::MAX, 1, 1, 5, 9, "sgemm operand out of bounds"),
                    (2, 2, 2, usize::MAX, 5, 9, "sgemm operand out of bounds"),
                    (2, 2, 2, 1, 13, 9, "sgemm operand out of bounds"),
                    (2, 2, 2, 1, 5, 13, "sgemm operand out of bounds"),
                ] {
                    interp.memory.clone_from(&initial);
                    let args = [
                        Value::Ptr(ap),
                        Value::Ptr(bp),
                        Value::Ptr(cp),
                        Value::Int(m as i128),
                        Value::Int(k as i128),
                        Value::Int(n as i128),
                        Value::Int(1),
                    ];
                    // These requests are rejected against a fourteen-slot arena before reserving
                    // scratch. No test relies on an allocator trying an enormous request.
                    assert_eq!(interp.intrinsic(&name, &args).unwrap_err(), error, "{name}");
                    assert_eq!(
                        interp.memory, initial,
                        "a failed range check must not write memory"
                    );
                }
                for beta in [0, 1] {
                    interp.memory.clone_from(&initial);
                    let args = [
                        Value::Ptr(1),
                        Value::Ptr(5),
                        Value::Ptr(9),
                        Value::Int(2),
                        Value::Int(2),
                        Value::Int(2),
                        Value::Int(beta),
                    ];
                    interp.intrinsic(&name, &args).unwrap();
                    let expected = if beta == 0 {
                        [19.0, 22.0, 43.0, 50.0]
                    } else {
                        [20.0, 24.0, 46.0, 54.0]
                    };
                    assert_eq!(
                        &interp.memory[9..13],
                        &expected.map(Value::Float),
                        "{name}, beta={beta}"
                    );
                    assert_eq!(&interp.memory[..9], &initial[..9]);
                    assert_eq!(interp.memory[13], initial[13]);
                }
            }
        }
    }

    #[test]
    fn empty_inner_reductions_match_the_source_with_recognition_on_and_off() {
        let source =
            include_str!("../../wukong_mir_build/tests/fixtures/zero_inner_recognition.wk");
        let expected = source
            .lines()
            .filter_map(|line| line.strip_prefix("// EXPECT-OUT: "))
            .map(|line| format!("{line}\n"))
            .collect::<String>();
        for recognize in [false, true] {
            let mut interner = Interner::new();
            let (module, diagnostics) =
                wukong_parser::parse_module(source, SourceId(0), &mut interner);
            assert!(diagnostics.iter().all(|d| !d.is_error()), "{diagnostics:?}");
            let (sema, diagnostics) = wukong_sema::check(&module, &interner);
            assert!(diagnostics.iter().all(|d| !d.is_error()), "{diagnostics:?}");
            let (program, diagnostics) = wukong_mir_build::lower_program_with(
                &module,
                &sema,
                &mut interner,
                wukong_mir_build::LowerOptions {
                    recognize,
                    ..Default::default()
                },
            );
            assert!(diagnostics.iter().all(|d| !d.is_error()), "{diagnostics:?}");
            let errors = wukong_mir::verify::verify_program(&program);
            assert!(errors.is_empty(), "{errors:?}");
            let entry = interner.intern("main");
            let (code, output) = run_with_output(&program, entry, &interner).unwrap();
            assert_eq!(code, 0);
            assert_eq!(
                String::from_utf8(output).unwrap(),
                expected,
                "recognize={recognize}"
            );
        }
    }

    #[test]
    fn i1_loads_use_the_low_bit_for_scalar_and_vector_masks() {
        use wukong_mir::Builder;

        for storage in [MirType::I8, MirType::I1] {
            for vector in [false, true] {
                for (lane, expected) in [22, 11, 22, 11, 11].into_iter().enumerate() {
                    let mut interner = Interner::new();
                    let name = interner.intern("main");
                    let mut b = Builder::new(name, MirType::I32);
                    let buffer = b.alloca(MirType::Array(Box::new(storage.clone()), 5));
                    let mut pointers = Vec::new();
                    for (i, byte) in [0, 1, 2, 3, 255].into_iter().enumerate() {
                        let index = b.build(MirType::I64, Op::ConstInt(i as i128, MirType::I64));
                        let ptr = b.build(
                            MirType::Ptr,
                            Op::Gep {
                                ptr: buffer,
                                index,
                                elem: storage.clone(),
                            },
                        );
                        // The I1 control uses canonical typed stores; I8 stores retain raw bytes.
                        let value = if storage == MirType::I1 {
                            byte & 1
                        } else {
                            byte
                        };
                        let value = b.build(storage.clone(), Op::ConstInt(value, storage.clone()));
                        b.build_void(Op::Store { ptr, value });
                        pointers.push(ptr);
                    }
                    let yes = b.build(MirType::I32, Op::ConstInt(11, MirType::I32));
                    let no = b.build(MirType::I32, Op::ConstInt(22, MirType::I32));
                    let result = if vector {
                        let mask_ty = MirType::Vec(Box::new(MirType::I1), 5);
                        let values_ty = MirType::Vec(Box::new(MirType::I32), 5);
                        let mask = b.build(mask_ty.clone(), Op::Load(buffer, mask_ty));
                        let yes = b.build(values_ty.clone(), Op::Splat(yes));
                        let no = b.build(values_ty.clone(), Op::Splat(no));
                        let selected = b.build(values_ty, Op::Select(mask, yes, no));
                        b.build(MirType::I32, Op::ExtractLane(selected, lane as u32))
                    } else {
                        let mask = b.build(MirType::I1, Op::Load(pointers[lane], MirType::I1));
                        b.build(MirType::I32, Op::Select(mask, yes, no))
                    };
                    b.ret(Some(result));
                    let program = Program {
                        funcs: vec![b.finish()],
                        ..Program::default()
                    };
                    assert!(wukong_mir::verify::verify_program(&program).is_empty());
                    assert_eq!(
                        run(&program, name, &interner).unwrap(),
                        expected,
                        "storage={storage:?}, vector={vector}, lane={lane}"
                    );
                }
            }
        }
    }

    #[test]
    fn lowp_gemm_preserves_existing_c_for_nonzero_beta() {
        // A = [[1,2,3],[-1,0,2]], B = [[4,2],[-2,3],[1,-1]].
        // The exact integer product is [[3,5],[-2,-4]], independently of
        // half format, NT/TN storage, or the kernel used by the interpreter.
        let program = Program::default();
        let interner = Interner::new();
        for half in ["bf16", "f16"] {
            for layout in ["nt", "tn", "nt_epi"] {
                for suffix in ["", "_parallel"] {
                    for beta in [0, 1, -1, 2] {
                        let name = format!("wukong_sgemm_{half}_{layout}{suffix}");
                        let a = if layout == "tn" {
                            [1.0, -1.0, 2.0, 0.0, 3.0, 2.0]
                        } else {
                            [1.0, 2.0, 3.0, -1.0, 0.0, 2.0]
                        };
                        let b = if layout == "tn" {
                            [4.0, 2.0, -2.0, 3.0, 1.0, -1.0]
                        } else {
                            [4.0, -2.0, 1.0, 2.0, 3.0, -1.0]
                        };
                        for act in if layout == "nt_epi" { 0..2 } else { 0..1 } {
                            let c = [10.0, -20.0, 30.0, -40.0];
                            let mut interp = Interp {
                                program: &program,
                                interner: &interner,
                                memory: a
                                    .into_iter()
                                    .chain(b)
                                    .chain(c)
                                    .chain([-4.0, 6.0])
                                    .map(Value::Float)
                                    .collect(),
                                stdout: Vec::new(),
                                scratch_arena: ScratchSlots::default(),
                                slots: SlotReuse::default(),
                                frames: Vec::new(),
                                call_args: Vec::new(),
                                scratch: Vec::new(),
                                vecs: Vec::new(),
                                #[cfg(test)]
                                vector_peak: 0,
                                data_addrs: HashMap::new(),
                                function_indices: None,
                                static_indices: None,
                                accel: None,
                                depth: 0,
                            };
                            let mut args = vec![
                                Value::Ptr(0),
                                Value::Ptr(6),
                                Value::Ptr(12),
                                Value::Int(2),
                                Value::Int(3),
                                Value::Int(2),
                                Value::Int(beta),
                            ];
                            if layout == "nt_epi" {
                                args.extend([Value::Ptr(16), Value::Int(act)]);
                            }
                            interp.intrinsic(&name, &args).unwrap();
                            let product = [3.0f32, 5.0, -2.0, -4.0];
                            for (i, dot) in product.into_iter().enumerate() {
                                // beta is a flag, not a multiplier; every nonzero value adds C.
                                let mut expected = dot + if beta == 0 { 0.0 } else { c[i] as f32 };
                                if layout == "nt_epi" {
                                    expected += [-4.0, 6.0][i % 2];
                                    if act == 1 {
                                        expected = expected.max(0.0);
                                    }
                                }
                                assert_eq!(
                                    (interp.memory[12 + i].as_float() as f32).to_bits(),
                                    expected.to_bits(),
                                    "{name}, beta={beta}, act={act}, lane={i}"
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    /// Every `wukong_*` symbol the lowered program calls. Kernel arms are gate-blind — a fixture
    /// that stopped dispatching would still print the right answer, and the test below would pass
    /// while proving nothing — so an aliasing test has to assert the dispatch as well as the value.
    fn kernel_calls_of(src: &str) -> Vec<String> {
        let mut interner = Interner::new();
        let (module, pd) = wukong_parser::parse_module(src, SourceId(0), &mut interner);
        assert!(pd.is_empty(), "parse: {pd:?}");
        let (sema, sd) = wukong_sema::check(&module, &interner);
        assert!(sd.iter().all(|d| !d.is_error()), "sema: {sd:?}");
        let (program, _ld) = wukong_mir_build::lower_program(&module, &sema, &mut interner);
        let mut out = Vec::new();
        for f in &program.funcs {
            for b in &f.blocks {
                for i in &b.insts {
                    if let wukong_mir::Op::Call { func, .. } = &i.op {
                        let n = interner.resolve(*func);
                        if n.starts_with("wukong_") && n != "wukong_parallel_for" {
                            out.push(n.to_string());
                        }
                    }
                }
            }
        }
        out
    }

    /// The scatter-add half of the aliasing story, and the one quantified interp-vs-native
    /// divergence that was shipping in the corpus.
    ///
    /// `wukong_scatter_add_f32` accumulates `grad_w[ids[t], :] += grad_out[t, :]` scanning `t`
    /// ascending. The interpreter used to marshal `grad_out` into a scratch `Vec` before calling
    /// the kernel, so when the two buffers are the SAME array it folded a pre-call SNAPSHOT: over
    /// `tests/run/embedding_alias_parallel.wk`'s own inputs it answered 49120 where the nest and
    /// the native run both answer 49121. That fixture states the divergence in its header and then
    /// declines to print the value, so `native_matches_interpreter` stayed green over it.
    ///
    /// 49121 is the hand reference: the nest `grad_w[ids[t]*4+d] += grad_out[t*4+d]` run over
    /// `ids[t] = (3t+1) % 8` and `acc[i] = 0.5i - 3.0` in f32, in ascending `t`, reading live
    /// memory — identical to `scatter.rs::scatter_rows` with `r0 = 0, r1 = v`. It is NOT a reading
    /// taken from another backend. The snapshot answer 49120 is asserted against explicitly so a
    /// regression to marshalling fails loudly rather than drifting.
    #[test]
    fn scatter_add_reads_live_memory_when_grad_w_aliases_grad_out() {
        let src = "fn scatter(ids: [i32;80], grad_out: [f32;320], mut grad_w: [f32;320]) { \
                     for t in 0..80 { for d in 0..4 { \
                       grad_w[ids[t]*4+d] += grad_out[t*4+d]; } } } \
                   fn main() -> i32 { let mut ids: [i32; 80] = [0; 80]; \
                     let mut acc: [f32; 320] = [0.0; 320]; \
                     for i in 0..80 { ids[i] = ((i * 3 + 1) % 8) as i32; } \
                     for i in 0..320 { acc[i] = (i as f32) * 0.5 - 3.0; } \
                     scatter(ids, acc, acc); \
                     let mut s: f32 = 0.0; for i in 0..320 { s = s + acc[i]; } \
                     return s as i32; }";
        let calls = kernel_calls_of(src);
        assert!(
            calls
                .iter()
                .any(|c| c.starts_with("wukong_scatter_add_f32")),
            "the scatter recognizer must fire, or this test is vacuous: {calls:?}"
        );
        let got = run_main(src);
        assert_ne!(
            got, 49120,
            "the oracle folded a pre-call snapshot of grad_out"
        );
        assert_eq!(got, 49121);

        // The ordinary disjoint scatter must be unchanged: every token's row lands once into a
        // zeroed accumulator, so the total is the sum of all 320 gradient elements = 24560.
        let disjoint = "fn scatter(ids: [i32;80], grad_out: [f32;320], mut grad_w: [f32;320]) { \
                          for t in 0..80 { for d in 0..4 { \
                            grad_w[ids[t]*4+d] += grad_out[t*4+d]; } } } \
                        fn main() -> i32 { let mut ids: [i32; 80] = [0; 80]; \
                          let mut g: [f32; 320] = [0.0; 320]; \
                          let mut gw: [f32; 320] = [0.0; 320]; \
                          for i in 0..80 { ids[i] = ((i * 3 + 1) % 8) as i32; } \
                          for i in 0..320 { g[i] = (i as f32) * 0.5 - 3.0; } \
                          scatter(ids, g, gw); \
                          let mut s: f32 = 0.0; for i in 0..320 { s = s + gw[i]; } \
                          return s as i32; }";
        assert_eq!(run_main(disjoint), 24560);
    }

    /// An in-place embedding gather (`out` and `weight` are the *same* buffer) must reproduce the
    /// sequential source nest, which reads `weight[ids[t]]` **after** the earlier rows have already
    /// overwritten it. The interpreter used to marshal the whole weight table into a scratch buffer
    /// first and gather from that snapshot, which is a different program. 766 is the hand reference
    /// (the nest run over the same inputs), not a reading taken from another backend.
    #[test]
    fn embedding_gather_reads_live_memory_when_out_aliases_weight() {
        let src = "fn embed(ids: [i32;80], weight: [f32;320], mut out: [f32;320]) { \
                     for t in 0..80 { for d in 0..4 { out[t*4+d] = weight[ids[t]*4+d]; } } } \
                   fn main() -> i32 { let mut ids: [i32; 80] = [0; 80]; \
                     let mut buf: [f32; 320] = [0.0; 320]; \
                     for i in 0..80 { ids[i] = 79 - i; } \
                     for i in 0..320 { buf[i] = (i as f32); } \
                     embed(ids, buf, buf); \
                     let mut s: f32 = 0.0; for i in 0..320 { s = s + buf[i]; } \
                     return (s / 100.0) as i32; }";
        assert_eq!(run_main(src), 766);

        // The ordinary disjoint gather must be unchanged: reversing 80 rows of a table whose row `r`
        // holds `4r..4r+3` sums to the same 0..319 total, and row 0 of the output is the last table
        // row (316+317+318+319 = 1270).
        let disjoint = "fn embed(ids: [i32;80], weight: [f32;320], mut out: [f32;320]) { \
                          for t in 0..80 { for d in 0..4 { out[t*4+d] = weight[ids[t]*4+d]; } } } \
                        fn main() -> i32 { let mut ids: [i32; 80] = [0; 80]; \
                          let mut w: [f32; 320] = [0.0; 320]; let mut o: [f32; 320] = [0.0; 320]; \
                          for i in 0..80 { ids[i] = 79 - i; } \
                          for i in 0..320 { w[i] = (i as f32); } \
                          embed(ids, w, o); \
                          let mut s: f32 = 0.0; for i in 0..320 { s = s + o[i]; } \
                          return (s as i32) * 10000 + ((o[0] + o[1] + o[2] + o[3]) as i32); }";
        assert_eq!(run_main(disjoint), 51040 * 10000 + 1270);
    }

    /// An `alloc_<T>(n)` the interpreter's arena cannot satisfy must report a diagnostic. It used
    /// to reach Rust's allocation-error handler through `Vec::resize` and `abort()` the process
    /// with `memory allocation of N bytes failed` — an internal runtime message, and an abort no
    /// caller (fuzzer, differential harness, test binary) can catch.
    #[test]
    fn heap_exhaustion_is_a_diagnostic_not_an_allocator_abort() {
        let mut interner = Interner::new();
        let src = "fn main() -> i32 { let mut d: []f32 = alloc_f32(100000000000); \
                   d[0] = 1.0; free(d); return 0; }";
        let (module, _) = wukong_parser::parse_module(src, SourceId(0), &mut interner);
        let (sema, _) = wukong_sema::check(&module, &interner);
        let (program, _) = wukong_mir_build::lower_program(&module, &sema, &mut interner);
        let main = interner.intern("main");
        let err = run(&program, main, &interner).unwrap_err();
        assert!(err.contains("interpreter heap exhausted"), "got: {err}");
    }

    /// An `extern "C"` function runs on the oracle by calling the same host C symbol the native JIT
    /// binds, through every scalar register class (f64, int, f32, i64). Every expected value is
    /// exact by hand: ldexp(3, 4) = 3 * 2^4 = 48; fmod(17.5, 5) = 2.5; fmodf(7.5, 2) = 1.5;
    /// llabs(-9e9) = 9e9; toupper('a' = 97) = 'A' = 65.
    #[test]
    fn extern_c_functions_call_the_host_c_library() {
        let out = run_main_stdout(
            "extern \"C\" {\n\
                 fn ldexp(x: f64, e: i32) -> f64;\n\
                 fn fmod(x: f64, y: f64) -> f64;\n\
                 fn fmodf(x: f32, y: f32) -> f32;\n\
                 fn llabs(x: i64) -> i64;\n\
                 fn toupper(c: i32) -> i32;\n\
             }\n\
             fn main() -> i32 {\n\
                 print(ldexp(3.0, 4) as i64);\n\
                 print((fmod(17.5, 5.0) * 10.0) as i64);\n\
                 print((fmodf(7.5, 2.0) * 10.0) as i32);\n\
                 print(llabs(-9000000000));\n\
                 print(toupper(97));\n\
                 return 0;\n\
             }\n",
        );
        assert_eq!(out, "48\n25\n15\n9000000000\n65\n");
    }

    /// What the oracle cannot call is named before anything runs: a pointer parameter (its memory
    /// is slots, not bytes) and a symbol no loaded C library defines.
    #[test]
    fn externs_the_interpreter_cannot_call_are_reported_up_front() {
        let src = "extern \"C\" {\n\
                       fn strlen(s: *u8) -> i64;\n\
                       fn zz_no_such_c_symbol_anywhere(x: i32) -> i32;\n\
                       fn toupper(c: i32) -> i32;\n\
                   }\n\
                   fn main() -> i32 { return 0; }\n";
        let mut interner = Interner::new();
        let (module, pd) = wukong_parser::parse_module(src, SourceId(0), &mut interner);
        assert!(pd.is_empty(), "parse: {pd:?}");
        let (sema, sd) = wukong_sema::check(&module, &interner);
        assert!(sd.iter().all(|d| !d.is_error()), "sema: {sd:?}");
        let (program, ld) = wukong_mir_build::lower_program(&module, &sema, &mut interner);
        assert!(ld.iter().all(|d| !d.is_error()), "lower: {ld:?}");
        assert_eq!(program.externs.len(), 3);
        let refused = unsupported_externs(&program, &interner);
        assert_eq!(refused.len(), 2, "{refused:?}");
        assert!(
            refused[0].contains("strlen") && refused[0].contains("ptr"),
            "{refused:?}"
        );
        assert!(
            refused[1].contains("zz_no_such_c_symbol_anywhere"),
            "{refused:?}"
        );
    }

    #[test]
    fn print_intrinsic_captures_stdout() {
        let mut interner = Interner::new();
        let src = "fn main() -> i32 { print(42); print(7 * 6); return 0; }";
        let (module, _) = wukong_parser::parse_module(src, SourceId(0), &mut interner);
        let (sema, _) = wukong_sema::check(&module, &interner);
        let (program, _) = wukong_mir_build::lower_program(&module, &sema, &mut interner);
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
            fn saxpy(x: [f32; 8], y: [f32; 8], mut out: [f32; 8]) { \
                for i in 0..8 { out[i] = 2.0 * x[i] + y[i]; } }\n\
            fn dot(x: [f32; 8], y: [f32; 8], mut out: [f32; 8]) { \
                let mut s: f32 = 0.0; for i in 0..8 { s = s + x[i] * y[i]; } out[0] = s; }\n";
        let (module, pd) = wukong_parser::parse_module(src, SourceId(0), &mut interner);
        assert!(pd.iter().all(|d| !d.is_error()), "parse: {pd:?}");
        let (sema, sd) = wukong_sema::check(&module, &interner);
        assert!(sd.iter().all(|d| !d.is_error()), "sema: {sd:?}");
        let (program, ld) = wukong_mir_build::lower_program(&module, &sema, &mut interner);
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
        // i64 reference. A=[1..8], B=[-6..1]; the recognizer dispatches this to wukong_i8gemm_nt.
        let mut interner = Interner::new();
        let src = "module m\n\
            fn lin(a: [u8; 8], b: [i8; 8], mut c: [i32; 4]) { \
                for i in 0..2 { for j in 0..2 { let mut s: i32 = 0; \
                for k in 0..4 { s = s + (a[i*4+k] as i32) * (b[j*4+k] as i32); } \
                c[i*2+j] = s; } } }\n";
        let (module, pd) = wukong_parser::parse_module(src, SourceId(0), &mut interner);
        assert!(pd.iter().all(|d| !d.is_error()), "parse: {pd:?}");
        let (sema, sd) = wukong_sema::check(&module, &interner);
        assert!(sd.iter().all(|d| !d.is_error()), "sema: {sd:?}");
        let (program, ld) = wukong_mir_build::lower_program(&module, &sema, &mut interner);
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
        let src = "fn fill(mut a: [i32; 3]) { a[0] = 7; a[1] = 8; a[2] = 9; } \
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
        let (module, _) = wukong_parser::parse_module(src, SourceId(0), &mut interner);
        let (sema, _) = wukong_sema::check(&module, &interner);
        let (program, _) = wukong_mir_build::lower_program(&module, &sema, &mut interner);
        let main = interner.intern("main");
        assert!(run(&program, main, &interner).is_err());
    }

    /// `NULL_SLOT` is reserved in every run: the first `alloca` of a run is not address 0, so a
    /// pointer null test agrees with native (`inttoptr 0` is null there, and nothing else is). The
    /// first case is the exact reproducer the `PtrToInt(p) != 0` lowering was reverted on at
    /// 0900352 — `p` addresses the first local, which used to *be* slot 0 and read as null.
    #[test]
    fn address_zero_is_never_a_live_allocation() {
        let lower = |src: &str| {
            let mut interner = Interner::new();
            let (module, pd) = wukong_parser::parse_module(src, SourceId(0), &mut interner);
            assert!(pd.is_empty(), "parse: {pd:?}");
            let (sema, sd) = wukong_sema::check(&module, &interner);
            assert!(sd.iter().all(|d| !d.is_error()), "sema: {sd:?}");
            let (program, ld) = wukong_mir_build::lower_program(&module, &sema, &mut interner);
            assert!(ld.iter().all(|d| !d.is_error()), "lower: {ld:?}");
            assert!(
                wukong_mir::verify::verify_program(&program).is_empty(),
                "a pointer condition must lower to verifier-clean MIR: {src}"
            );
            (program, interner)
        };
        let first_local = "fn main() -> i32 { let mut v: i32 = 7; let p: *mut i32 = &mut v; \
                           if p { return 1; } return 0; }";
        let (program, mut interner) = lower(first_local);
        let main = interner.intern("main");
        assert_eq!(run(&program, main, &interner), Ok(1));

        let null = "fn main() -> i32 { let q: *mut i32 = 0 as *mut i32; \
                    if q { return 1; } return 2; }";
        let (program, mut interner) = lower(null);
        let main = interner.intern("main");
        assert_eq!(run(&program, main, &interner), Ok(2));

        // Dereferencing the null is a diagnostic, not a read of the reserved sentinel slot.
        let deref = "fn main() -> i32 { let q: *mut i32 = 0 as *mut i32; return *q; }";
        let (program, mut interner) = lower(deref);
        let main = interner.intern("main");
        let err = run(&program, main, &interner).unwrap_err();
        assert!(err.contains("null pointer dereference"), "{err}");

        // A kernel-entry buffer is laid out after the reserved slot too.
        let kernel = "fn k(mut a: [f32; 2]) { a[0] = a[0] + 1.0; a[1] = a[1] * 2.0; }";
        let (program, mut interner) = lower(kernel);
        let k = interner.intern("k");
        let mut buf = [3.0f32, 5.0];
        run_kernel_f32(&program, k, &mut [&mut buf], &interner).unwrap();
        assert_eq!(buf, [4.0, 10.0]);
    }

    /// `atomic.add` wraps at the value's width, like the `add` it stands for. The expected values
    /// are hand-derived two's-complement truths: u8 200 + 100 = 300 = 44 (mod 2^8); i16 -1 + 2 = 1;
    /// i32 MAX + 1 = MIN; i64 MAX + 1 = MIN. Each is read back through an ordinary load.
    #[test]
    fn atomic_add_wraps_at_the_value_width() {
        use wukong_mir::Builder;
        let cases: [(MirType, i128, i128, i128); 4] = [
            (MirType::I8, 200, 100, 44),
            (MirType::I16, -1, 2, 1),
            (MirType::I32, i128::from(i32::MAX), 1, i128::from(i32::MIN)),
            (MirType::I64, i128::from(i64::MAX), 1, i128::from(i64::MIN)),
        ];
        for (ty, init, add, want) in cases {
            let mut interner = Interner::new();
            let name = interner.intern("main");
            let mut b = Builder::new(name, MirType::I64);
            let slot = b.alloca(ty.clone());
            let a = b.build(ty.clone(), Op::ConstInt(init, ty.clone()));
            b.build_void(Op::Store {
                ptr: slot,
                value: a,
            });
            let v = b.build(ty.clone(), Op::ConstInt(add, ty.clone()));
            b.atomic_add(slot, v);
            let got = b.build(ty.clone(), Op::Load(slot, ty.clone()));
            let wide = if ty == MirType::I64 {
                got
            } else {
                b.build(MirType::I64, Op::Cast(CastKind::SExt, got, MirType::I64))
            };
            b.ret(Some(wide));
            let f = b.finish();
            assert!(wukong_mir::verify::verify_function(&f).is_empty());
            let mut program = Program::new();
            program.funcs.push(f);
            assert_eq!(
                run(&program, name, &interner),
                Ok(i64::try_from(want).unwrap()),
                "{ty:?}: {init} + {add}"
            );
        }

        // A null pointer is a diagnostic, not an add into the reserved slot.
        let mut interner = Interner::new();
        let name = interner.intern("main");
        let mut b = Builder::new(name, MirType::I64);
        let zero = b.build(MirType::I64, Op::ConstInt(0, MirType::I64));
        let null = b.build(
            MirType::Ptr,
            Op::Cast(CastKind::IntToPtr, zero, MirType::Ptr),
        );
        let one = b.build(MirType::I64, Op::ConstInt(1, MirType::I64));
        b.atomic_add(null, one);
        b.ret(Some(one));
        let mut program = Program::new();
        program.funcs.push(b.finish());
        let err = run(&program, name, &interner).unwrap_err();
        assert!(
            err.contains("null pointer dereference (atomic add)"),
            "{err}"
        );
    }

    /// N89: a failed `assert` reports the site the lowering attached (and its message literal),
    /// and so does a value `match` no arm matched, naming the `match`, instead of reaching a bare
    /// `unreachable`. Without a source locator a bare `assert(cond)` keeps its old message.
    #[test]
    fn a_failed_assert_and_an_unmatched_match_name_their_sites() {
        let run_src = |src: &str, located: bool| -> Result<i64, String> {
            let mut sm = wukong_span::SourceMap::new();
            let id = sm.add("t.wk".to_string(), src.to_string());
            let mut interner = Interner::new();
            let (module, pd) = wukong_parser::parse_module(src, id, &mut interner);
            assert!(pd.is_empty(), "parse: {pd:?}");
            let (sema, sd) = wukong_sema::check(&module, &interner);
            assert!(sd.iter().all(|d| !d.is_error()), "sema: {sd:?}");
            let locate = |span: wukong_span::Span| {
                let loc = sm.span_location(span);
                format!("{}:{}:{}", sm.name(span.source), loc.line, loc.col)
            };
            let opts = wukong_mir_build::LowerOptions::default();
            let (program, ld) = if located {
                wukong_mir_build::lower_program_located(
                    &module,
                    &sema,
                    &mut interner,
                    opts,
                    &locate,
                )
            } else {
                wukong_mir_build::lower_program_with(&module, &sema, &mut interner, opts)
            };
            assert!(ld.iter().all(|d| !d.is_error()), "lower: {ld:?}");
            let main = interner.intern("main");
            run(&program, main, &interner)
        };
        let bare = "fn main() -> i32 {\n    assert(1 > 2);\n    return 0;\n}\n";
        assert_eq!(
            run_src(bare, true),
            Err("assertion failed at t.wk:2:5".to_string())
        );
        assert_eq!(run_src(bare, false), Err("assertion failed".to_string()));
        assert_eq!(
            run_src(
                "fn main() -> i32 {\n    assert(1 > 2, \"one is not two\");\n    return 0;\n}\n",
                true
            ),
            Err("assertion failed: one is not two at t.wk:2:5".to_string())
        );
        // `    return ` is 11 characters, so the `match` opens column 12 of line 4.
        assert_eq!(
            run_src(
                "enum E { A = 1, B = 2 }\nfn main() -> i32 {\n    let e: E;\n    \
                 return match e { E::A => 1, E::B => 2 };\n}\n",
                true
            ),
            Err("assertion failed: no `match` arm matched its value at t.wk:4:12".to_string())
        );
    }
}

#[cfg(test)]
mod vector_arena_tests;

#[cfg(test)]
mod kernel_marshalling_tests;

#[cfg(test)]
mod kernel_extent_marshalling_tests;

#[cfg(test)]
mod transfer_bounds_tests;

#[cfg(test)]
mod str_utf8_tests;

#[cfg(test)]
mod scratch_tests;

#[cfg(test)]
mod kernel_alias_contract_tests;

#[cfg(test)]
mod parallel_twin_tests;

#[cfg(test)]
mod scalar_contract_tests;

#[cfg(test)]
mod runtime_surface_tests;

#[cfg(test)]
mod slot_reuse_tests;
