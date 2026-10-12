//! `wukong_codegen_cranelift` — native code generation via Cranelift (no LLVM toolchain required).
//!
//! Lowers fully-lowered (Low) MIR to Cranelift IR and either JIT-compiles and runs it in
//! process (`jit_run`, the fast execution path and a differential oracle alongside the interpreter)
//! or emits a native object file (`emit_object`). Linking an object into an executable is **not**
//! done here — that is `wukong_driver::emit_native`.
//!
//! The MIR maps onto Cranelift almost one-to-one: it is already block-parameter SSA (exactly
//! Cranelift's model), integers are signless with signedness on the op, and memory is explicit
//! `alloca`/`load`/`store`/`gep`. It is mostly scalar but not exclusively: `mir_build`'s vectorizer
//! leaves `MirType::Vec` values (CLIF-legal only up to 128 bits) and `Op::VecKernelCall`s whose
//! 256-bit bodies this crate assembles as raw machine code itself (see the `avx2` module).
//!
//! The semantic gaps we bridge to stay bit-identical to the interpreter oracle: divide-by-zero yields
//! 0 (no trap), float→int casts saturate, `i1` results are normalised to their low bit, float `%`
//! calls an `rt_fmod_*` shim (Cranelift has no `frem`), and f16/bf16 are rounded at the value —
//! inline for bf16, through the runtime's `half`-crate shims for f16.

use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicBool, AtomicI64, AtomicU8, Ordering};
use std::sync::Mutex;

use cranelift_codegen::ir::condcodes::{FloatCC, IntCC};
use cranelift_codegen::ir::{
    types, AbiParam, AliasRegion, AtomicRmwOp, Block, BlockArg, FuncRef, GlobalValue, InstBuilder,
    MemFlags, Signature, SourceLoc, StackSlot, StackSlotData, StackSlotKind, Type, Value,
};
use cranelift_codegen::settings::{self, Configurable};
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};
use cranelift_module::{DataDescription, DataId, FuncId, Linkage, Module};

use wukong_mir::{BinOp, CastKind, CmpOp, Function, MirType, Op, Program, Terminator, ValueId};
use wukong_span::{Interner, Span, Symbol};

mod backend;
mod bool_vectors;
pub use backend::CraneliftBackend;

// `--target-cpu`: the ISA level native code is generated for (host detection by default).
mod target;
pub use target::TargetCpu;

// P4: raw-AVX2 emitter for the general vectorizer's 256-bit path (Cranelift caps CLIF vectors at
// 128-bit). Owned by this session; additive.
mod avx2;

// The runtime entries of the ML builtins, declared, JIT-bound and called from one table.
mod ml_rt;

// The runtime entries of the text builtins (the per-type scalar printers), the same way.
mod text_rt;
// The `i128` operations x64 has no instruction for: wide constants, division, float conversion.
mod int128;

// Where each function's machine code went: the DWARF of a `-g` object and the JIT's perf map
// (`WUKONG_PERF_MAP=1`).
mod debuginfo;
pub use debuginfo::{DebugInfo, DW_LANG_WUKONG};

// The AOT entry contract (a synthesized C `main` on a big-stack worker) and the object's symbol
// policy (internal, mangled user functions in an executable).
mod entry;
pub use entry::ObjectKind;

// Stack-slot coloring: allocas with disjoint lifetimes share one frame slot.
mod stack_color;
pub use stack_color::{frame_bytes, FrameBytes};

// --- The minimal runtime the generated code calls back into ---------------------------------
//
// `print`/`println`/`assert` lower to calls to these symbols. For the JIT they are real Rust
// functions registered with the module; output is captured into a process-global buffer so the
// native path produces exactly the bytes the interpreter would, which is what the differential
// gate compares. A global run lock serialises native runs sharing that buffer.

static OUTPUT: Mutex<Vec<u8>> = Mutex::new(Vec::new());
static ASSERT_FAILED: AtomicBool = AtomicBool::new(false);
/// The diagnostic of the first runtime failure of the current guarded run (`assertion failed`, or a
/// slice bounds failure's `index out of bounds: …`). Set alongside [`ASSERT_FAILED`], first one wins,
/// and reported by the guarded run APIs as the run's error.
static FAILURE_MESSAGE: Mutex<Option<String>> = Mutex::new(None);
static RUN_LOCK: Mutex<()> = Mutex::new(());

thread_local! {
    /// The failure state of the `@parallel` region span this thread is running, if it is running
    /// one: `None` outside every span, `Some(None)` inside a span that has not failed, and
    /// `Some(Some(message))` once it has. A span's failure stays off the shared run state until
    /// [`rt_parallel_for`] picks the region's failure by iteration order — so which span's thread
    /// happened to fail first can neither change the reported diagnostic nor stop another span
    /// early. Saved and restored around every span, so a span nested inside another (a work-stealing
    /// worker running a second region while it waits) keeps its own state.
    static SPAN_FAILURE: RefCell<Option<Option<String>>> = const { RefCell::new(None) };
}

/// Record `message` as the run's failure unless an earlier one is already recorded, and raise the
/// failure flag the generated code polls — the shared run flag, or, inside a parallel-region span,
/// that span's own (see [`SPAN_FAILURE`]).
fn record_failure(message: impl FnOnce() -> String) {
    let mut message = Some(message);
    let in_span = SPAN_FAILURE.with(|span| match span.borrow_mut().as_mut() {
        Some(slot) => {
            if slot.is_none() {
                *slot = message.take().map(|m| m());
            }
            true
        }
        None => false,
    });
    if in_span {
        return;
    }
    if let Some(message) = message {
        if let Ok(mut m) = FAILURE_MESSAGE.lock() {
            if m.is_none() {
                *m = Some(message());
            }
        }
    }
    ASSERT_FAILED.store(true, Ordering::SeqCst);
}

/// The exit code of a guarded run that `exit(code)` ended — read with the failure it is recorded
/// as ([`wukong_runtime::process::EXIT_SENTINEL`]).
static EXIT_CODE: AtomicI64 = AtomicI64::new(0);

/// `exit(code)` in a guarded run: remember the code and stop the run the way a failed `assert`
/// does — every generated caller returns at once — so the host reports `code` as the run's result.
pub(crate) fn record_exit(code: i64) {
    EXIT_CODE.store(code, Ordering::SeqCst);
    record_failure(|| wukong_runtime::process::EXIT_SENTINEL.to_owned());
}

/// The outcome of a guarded run whose entry returned `exit_code`: its failure, the code `exit`
/// ended it with, or `exit_code`.
fn run_outcome(exit_code: i64) -> Result<i64, String> {
    match take_failure() {
        Some(f) if f == wukong_runtime::process::EXIT_SENTINEL => {
            Ok(EXIT_CODE.load(Ordering::SeqCst))
        }
        Some(failure) => Err(failure),
        None => Ok(exit_code),
    }
}

/// Clear the failure state at the start of a guarded run.
fn reset_failure() {
    ASSERT_FAILED.store(false, Ordering::SeqCst);
    if let Ok(mut m) = FAILURE_MESSAGE.lock() {
        *m = None;
    }
}

/// The failure a guarded run ended with, if any: the recorded diagnostic, or `assertion failed`.
fn take_failure() -> Option<String> {
    if !ASSERT_FAILED.load(Ordering::SeqCst) {
        return None;
    }
    let recorded = FAILURE_MESSAGE.lock().ok().and_then(|mut m| m.take());
    Some(recorded.unwrap_or_else(|| "assertion failed".to_owned()))
}

/// How a guarded run's stdout leaves [`OUTPUT`]: [`STREAM_CAPTURE`] keeps it for the caller
/// ([`JitProgram::run_capturing`]); the two streaming modes ([`JitProgram::run_streaming`]) write it
/// through to the process's stdout at every newline, or once [`STREAM_BLOCK`] bytes are buffered.
static STREAM_MODE: AtomicU8 = AtomicU8::new(STREAM_CAPTURE);
const STREAM_CAPTURE: u8 = 0;
const STREAM_LINES: u8 = 1;
const STREAM_BLOCKS: u8 = 2;
/// The block a block-buffered streaming run writes stdout in (the interpreter's, too).
const STREAM_BLOCK: usize = 1 << 16;

/// Write everything buffered in `o` to the process's stdout and clear it.
fn drain_output(o: &mut Vec<u8>) {
    use std::io::Write as _;
    let out = std::io::stdout();
    let mut out = out.lock();
    let _ = out.write_all(o);
    let _ = out.flush();
    o.clear();
}

/// Append `bytes` to the run's stdout — the one sink every JIT print entry writes through. A
/// streaming run writes it on to the process's stdout per [`STREAM_MODE`].
pub(crate) fn emit_stdout(bytes: &[u8]) {
    let mode = STREAM_MODE.load(Ordering::SeqCst);
    if let Ok(mut o) = OUTPUT.lock() {
        o.extend_from_slice(bytes);
        let write = match mode {
            STREAM_LINES => bytes.contains(&b'\n'),
            STREAM_BLOCKS => o.len() >= STREAM_BLOCK,
            _ => false,
        };
        if write {
            drain_output(&mut o);
        }
    }
}

/// Turns streaming on for one guarded run; dropping it (also while a panic unwinds) turns it off
/// and writes out what is still buffered.
struct StreamGuard;

impl StreamGuard {
    fn on(line_buffered: bool) -> StreamGuard {
        let mode = if line_buffered {
            STREAM_LINES
        } else {
            STREAM_BLOCKS
        };
        STREAM_MODE.store(mode, Ordering::SeqCst);
        StreamGuard
    }
}

impl Drop for StreamGuard {
    fn drop(&mut self) {
        STREAM_MODE.store(STREAM_CAPTURE, Ordering::SeqCst);
        drain_output(&mut OUTPUT.lock().unwrap_or_else(|e| e.into_inner()));
    }
}

extern "C" fn rt_print_i64(x: i64) {
    emit_stdout(format!("{x}\n").as_bytes());
}

// `print`/`println` of an *unsigned* integer lowers to this (mir_build zero-extends the value to 64
// bits first). Renders the bits as `u64` — the identical text the interpreter's `print_u` produces.
extern "C" fn rt_print_u64(x: u64) {
    emit_stdout(format!("{x}\n").as_bytes());
}

extern "C" fn rt_print_f64(x: f64) {
    emit_stdout(format!("{x}\n").as_bytes());
}

// `print`/`println` of a `*u8` string lowers to this. Reads the NUL-terminated byte buffer the
// string literal materialized on the stack and appends its bytes plus a newline — the identical
// bytes the interpreter's `print_str` produces (the buffer is always valid UTF-8 by construction).
extern "C" fn rt_print_str(ptr: *const u8) {
    if ptr.is_null() {
        return;
    }
    // SAFETY: `ptr` is a Wukong string buffer, always NUL-terminated by the `ExprKind::Str` lowering.
    let mut line = unsafe {
        let mut len = 0usize;
        while *ptr.add(len) != 0 {
            len += 1;
        }
        std::slice::from_raw_parts(ptr, len).to_vec()
    };
    line.push(b'\n');
    emit_stdout(&line);
}

// `wukong_rt_assert(cond, site, message)`: the lowering passes the `assert` call's `.rodata`
// `file:line:col` and its message literal, each null when the MIR call carried none. A failure is
// recorded like a bounds failure; the generated code then returns through its callers.
extern "C" fn rt_assert(cond: i64, site: *const u8, message: *const u8) {
    if cond == 0 {
        // SAFETY: each is null or a NUL-terminated `.rodata` string the compiler emitted.
        let (site, message) = unsafe {
            (
                wukong_runtime::site_str(site),
                wukong_runtime::site_str(message),
            )
        };
        record_failure(|| wukong_runtime::assert_failure_message(&site, &message));
    }
}

// Every other failed run-time check — a range index `s[a..b]`, `str_from_utf8` of bytes that are
// not UTF-8 — reported and returned through exactly as a failed bounds check is below.
extern "C" fn rt_check_fail(kind: i64, a: i64, b: i64, site: *const u8) {
    // SAFETY: `site` is a NUL-terminated `.rodata` string emitted by the compiler.
    let site = unsafe { wukong_runtime::site_str(site) };
    record_failure(|| wukong_runtime::check_failure_message(kind, a, b, &site));
}

// The raw-function-pointer binding of `wukong_rt_check_fail`: fatal, as `rt_bounds_fail_abort`.
extern "C" fn rt_check_fail_abort(kind: i64, a: i64, b: i64, site: *const u8) {
    // SAFETY: as in `rt_check_fail`.
    let site = unsafe { wukong_runtime::site_str(site) };
    eprintln!(
        "error: {}",
        wukong_runtime::check_failure_message(kind, a, b, &site)
    );
    std::process::abort();
}

// A slice index failed its bounds check — the failure edge `mir_build` emits for every checked
// `s[i]` (see `wukong_mir_build/src/bounds.rs`). Record the shared diagnostic and raise the failure
// flag; the generated code then returns through its callers exactly as for a failed `assert`, so
// the run stops at this access on the JIT just as it does in the interpreter and the linked exe.
extern "C" fn rt_bounds_fail(index: i64, len: i64, site: *const u8) {
    // SAFETY: `site` is a NUL-terminated `.rodata` string emitted by the compiler.
    let site = unsafe { wukong_runtime::site_str(site) };
    record_failure(|| wukong_runtime::bounds_failure_message(index, len, &site));
}

// The raw-function-pointer (`jit_module`) binding of the same symbol. A raw caller has no run
// handle a failure could be reported through, and early-returning a dummy value would be a silent
// wrong answer, so an out-of-bounds slice access is fatal there — Rust's `panic = "abort"`.
extern "C" fn rt_bounds_fail_abort(index: i64, len: i64, site: *const u8) {
    // SAFETY: as in `rt_bounds_fail`.
    let site = unsafe { wukong_runtime::site_str(site) };
    eprintln!(
        "error: {}",
        wukong_runtime::bounds_failure_message(index, len, &site)
    );
    std::process::abort();
}

// The raw-function-pointer (`jit_module`) binding of `wukong_rt_assert`. Like a bounds failure
// there, a failed assertion is fatal (Rust's `panic = "abort"`): a raw caller has no run handle a
// failure could be reported through, and returning a dummy value would be a silent wrong answer.
// The assertion lowering only calls this on its failure edge, so `cond` is always 0 in practice.
extern "C" fn rt_assert_abort(cond: i64, site: *const u8, message: *const u8) {
    if cond == 0 {
        // SAFETY: as in `rt_assert`.
        let (site, message) = unsafe {
            (
                wukong_runtime::site_str(site),
                wukong_runtime::site_str(message),
            )
        };
        eprintln!(
            "error: {}",
            wukong_runtime::assert_failure_message(&site, &message)
        );
        std::process::abort();
    }
}

// The raw-function-pointer (`jit_module`) binding of `wukong_rt_exit`: a raw caller has no run an
// `exit(code)` could end and report, so it is fatal there, as a failed assertion is.
extern "C" fn rt_exit_abort(code: i64) {
    eprintln!("error: exit({code}) called from code run through a raw function pointer");
    std::process::abort();
}

// The raw-mode binding of `wukong_rt_assert_failed`. A raw module aborts on any failure (an
// `assert`, a bounds check), so no failure can ever be pending in it — and reading the shared flag
// would let an unrelated guarded run on another thread stop raw code.
extern "C" fn rt_assert_never_failed() -> i64 {
    0
}

// The poll the generated code makes after every call that can fail: has the run failed? Inside a
// parallel-region span only that span's own failure counts (see `SPAN_FAILURE`).
extern "C" fn rt_assert_failed() -> i64 {
    let span = SPAN_FAILURE.with(|span| span.borrow().as_ref().map(Option::is_some));
    i64::from(span.unwrap_or_else(|| ASSERT_FAILED.load(Ordering::SeqCst)))
}

/// One `@parallel` region in flight under [`rt_parallel_for`]: the outlined body and its captured
/// environment, and the failure of the lowest-starting span that has failed so far.
struct ParallelRegion {
    body: extern "C" fn(i64, i64, *const u8),
    env: *const u8,
    first: Mutex<Option<(i64, String)>>,
}

// The guarded JIT's binding of `wukong_parallel_for` — the cancellation ABI for failure-capable
// region bodies. The runtime splits `[0, n)` into contiguous spans and runs them concurrently, so
// a body that fails (an `assert`, a slice bounds check) can fail on several spans at once, in any
// order. Each span runs under `rt_parallel_span`, which returns early once the span fails and
// keeps its diagnostic off the shared run state; the region then reports the failure of the
// LOWEST-starting failed span. That is exactly the failure the interpreter reports, which runs
// the range in order: by the data-parallel contract every iteration behaves as it does serially,
// so no span before the first failing iteration fails, and the span holding it stops there. The
// chosen failure is recorded on the calling thread — the run's state, or the enclosing span's when
// regions nest — and the caller, which is failure-capable itself, polls and returns at once.
// Spans that start after an already-recorded failure are skipped: serially they never run.
//
// A linked executable binds `wukong_parallel_for` to the runtime directly: there a failure exits
// the process at its source, so nothing propagates and no span outlives it.
extern "C" fn rt_parallel_for(n: i64, body: extern "C" fn(i64, i64, *const u8), env: *const u8) {
    let region = ParallelRegion {
        body,
        env,
        first: Mutex::new(None),
    };
    // SAFETY: `wukong_parallel_for` returns only after every span has completed, so `region`
    // outlives every `rt_parallel_span` call that reads it, and the spans share it only through
    // `&` (its mutable part is behind the mutex). `body`/`env` are forwarded unchanged.
    unsafe {
        wukong_runtime::wukong_parallel_for(
            n,
            rt_parallel_span,
            &region as *const ParallelRegion as *const u8,
        );
    }
    let first = region.first.into_inner().unwrap_or_else(|e| e.into_inner());
    if let Some((_, message)) = first {
        record_failure(|| message);
    }
}

// One span `[start, end)` of a region run by `rt_parallel_for` (see there).
extern "C" fn rt_parallel_span(start: i64, end: i64, region: *const u8) {
    // SAFETY: `region` is the `ParallelRegion` `rt_parallel_for` passed, alive for this call.
    let region = unsafe { &*(region as *const ParallelRegion) };
    let earlier_failed = region
        .first
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .as_ref()
        .is_some_and(|(failed_at, _)| *failed_at < start);
    if earlier_failed {
        return;
    }
    let outer = SPAN_FAILURE.with(|span| span.replace(Some(None)));
    (region.body)(start, end, region.env);
    let failure = SPAN_FAILURE.with(|span| span.replace(outer)).flatten();
    if let Some(message) = failure {
        let mut first = region.first.lock().unwrap_or_else(|e| e.into_inner());
        if first
            .as_ref()
            .is_none_or(|(failed_at, _)| start < *failed_at)
        {
            *first = Some((start, message));
        }
    }
}

// Float remainder. Cranelift has no `frem`; the interpreter uses Rust `%` (true fmod), so the native
// backend calls these to stay bit-identical instead of the lossy `a - trunc(a/b)*b` identity (which
// drifts once `a/b` exceeds the float's integer-precision range, e.g. `1e18 % 3`).
extern "C" fn rt_fmod_f64(a: f64, b: f64) -> f64 {
    a % b
}

extern "C" fn rt_fmod_f32(a: f32, b: f32) -> f32 {
    a % b
}

/// Names of the runtime symbols, shared by the JIT (which binds each to its in-process
/// implementation — one of the `rt_*` shims above, or the matching `wukong_runtime` kernel) and the
/// object emitter (which leaves them as undefined imports resolved at link time).
const RT_PRINT_I64: &str = "wukong_rt_print_i64";
const RT_PRINT_U64: &str = "wukong_rt_print_u64";
const RT_PRINT_F64: &str = "wukong_rt_print_f64";
const RT_PRINT_STR: &str = "wukong_rt_print_str";
const RT_ASSERT: &str = "wukong_rt_assert";
const RT_ASSERT_FAILED: &str = "wukong_rt_assert_failed";
// The failure edge of a checked slice access: (index: i64, len: i64, site: ptr) -> void.
const RT_BOUNDS_FAIL: &str = "wukong_rt_bounds_fail";
// Any other failed run-time check (a range index, `str_from_utf8`): (kind, a, b: i64, site: ptr).
const RT_CHECK_FAIL: &str = "wukong_rt_check_fail";
// The heap builtins: zeroed allocation (count, elem_size, elem_is_float) -> ptr, and its release.
const RT_ALLOC: &str = "wukong_rt_alloc";
const RT_FREE: &str = "wukong_rt_free";
// The scratch allocator: alloc shares RT_ALLOC's ABI; mark () -> i64; release (i64) -> void.
const RT_SCRATCH_ALLOC: &str = "wukong_rt_scratch_alloc";
const RT_SCRATCH_MARK: &str = "wukong_rt_scratch_mark";
const RT_SCRATCH_RELEASE: &str = "wukong_rt_scratch_release";
// The file-I/O intrinsics: read_/write_ × {f32,f64,i32,i64,i8,u8}, each (path, data, len) -> i64.
// All twelve share one ABI (two pointers + an i64 count) so cranelift declares them from one signature.
const RT_READ_F32: &str = "wukong_rt_read_f32";
const RT_READ_F64: &str = "wukong_rt_read_f64";
const RT_READ_I32: &str = "wukong_rt_read_i32";
const RT_READ_I64: &str = "wukong_rt_read_i64";
const RT_READ_I8: &str = "wukong_rt_read_i8";
const RT_READ_U8: &str = "wukong_rt_read_u8";
const RT_WRITE_F32: &str = "wukong_rt_write_f32";
const RT_WRITE_F64: &str = "wukong_rt_write_f64";
const RT_WRITE_I32: &str = "wukong_rt_write_i32";
const RT_WRITE_I64: &str = "wukong_rt_write_i64";
const RT_WRITE_I8: &str = "wukong_rt_write_i8";
const RT_WRITE_U8: &str = "wukong_rt_write_u8";
// The `now_ns()` timing intrinsic: `wukong_now_ns() -> i64`, a monotonic process-lifetime
// nanosecond clock (no parameters) for in-process benchmarking.
const RT_NOW_NS: &str = "wukong_now_ns";
const RT_PARALLEL_FOR: &str = "wukong_parallel_for";
const RT_SGEMM: &str = "wukong_sgemm";
const RT_SGEMM_PARALLEL: &str = "wukong_sgemm_parallel";
const RT_SGEMM_NT: &str = "wukong_sgemm_nt";
const RT_SGEMM_NT_PARALLEL: &str = "wukong_sgemm_nt_parallel";
const RT_SGEMM_TN: &str = "wukong_sgemm_tn";
const RT_SGEMM_TN_PARALLEL: &str = "wukong_sgemm_tn_parallel";
const RT_SGEMM_BF16_NT: &str = "wukong_sgemm_bf16_nt";
const RT_SGEMM_BF16_NT_PARALLEL: &str = "wukong_sgemm_bf16_nt_parallel";
const RT_SGEMM_F16_NT: &str = "wukong_sgemm_f16_nt";
const RT_SGEMM_F16_NT_PARALLEL: &str = "wukong_sgemm_f16_nt_parallel";
const RT_SGEMM_BF16_TN: &str = "wukong_sgemm_bf16_tn";
const RT_SGEMM_BF16_TN_PARALLEL: &str = "wukong_sgemm_bf16_tn_parallel";
const RT_SGEMM_F16_TN: &str = "wukong_sgemm_f16_tn";
const RT_SGEMM_F16_TN_PARALLEL: &str = "wukong_sgemm_f16_tn_parallel";
const RT_SGEMM_NT_EPI: &str = "wukong_sgemm_nt_epi";
const RT_SGEMM_NT_EPI_PAR: &str = "wukong_sgemm_nt_epi_parallel";
const RT_SGEMV: &str = "wukong_sgemv";
const RT_SGEMV_PAR: &str = "wukong_sgemv_parallel";
const RT_SGEMV_ALPHA: &str = "wukong_sgemv_alpha";
const RT_SGEMV_ALPHA_PAR: &str = "wukong_sgemv_alpha_parallel";
const RT_SGEVM: &str = "wukong_sgevm_f32";
const RT_SGEVM_PAR: &str = "wukong_sgevm_f32_parallel";
const RT_SGEMM_NT_ALPHA: &str = "wukong_sgemm_nt_alpha";
const RT_SGEMM_NT_ALPHA_PAR: &str = "wukong_sgemm_nt_alpha_parallel";
const RT_SGEMM_BF16_NT_EPI: &str = "wukong_sgemm_bf16_nt_epi";
const RT_SGEMM_BF16_NT_EPI_PAR: &str = "wukong_sgemm_bf16_nt_epi_parallel";
const RT_SGEMM_F16_NT_EPI: &str = "wukong_sgemm_f16_nt_epi";
const RT_SGEMM_F16_NT_EPI_PAR: &str = "wukong_sgemm_f16_nt_epi_parallel";
const RT_VMATH: &str = "wukong_vmath_f32";
const RT_TRIG_F32: &str = "wukong_rt_trig_f32";
const RT_MATH_F64: &str = "wukong_rt_math_f64";
const RT_MATH2_F64: &str = "wukong_rt_math2_f64";
const RT_VMATH_PARALLEL: &str = "wukong_vmath_f32_parallel";
const RT_VMATH2: &str = "wukong_vmath2_f32";
const RT_SOFTMAX_BWD: &str = "wukong_softmax_bwd_f32";
const RT_SOFTMAX_BWD_PAR: &str = "wukong_softmax_bwd_f32_parallel";
const RT_LRSCAN: &str = "wukong_lrscan_f32";
const RT_LRSCAN_PAR: &str = "wukong_lrscan_f32_parallel";
const RT_RMSNORM_BWD: &str = "wukong_rmsnorm_bwd_f32";
const RT_RMSNORM_BWD_PAR: &str = "wukong_rmsnorm_bwd_f32_parallel";
const RT_LAYERNORM_BWD: &str = "wukong_layernorm_bwd_f32";
const RT_LAYERNORM_BWD_PAR: &str = "wukong_layernorm_bwd_f32_parallel";
const RT_XENT: &str = "wukong_xent_fwd_f32";
const RT_XENT_PAR: &str = "wukong_xent_fwd_f32_parallel";
const RT_XENT_BWD: &str = "wukong_xent_bwd_f32";
const RT_XENT_BWD_PAR: &str = "wukong_xent_bwd_f32_parallel";
const RT_ROPE: &str = "wukong_rope_f32";
const RT_ROPE_PAR: &str = "wukong_rope_f32_parallel";
const RT_ROPE_BWD: &str = "wukong_rope_bwd_f32";
const RT_ROPE_BWD_PAR: &str = "wukong_rope_bwd_f32_parallel";
const RT_LOGSUMEXP: &str = "wukong_logsumexp_f32";
const RT_LOGSUMEXP_PAR: &str = "wukong_logsumexp_f32_parallel";
const RT_KLDIV: &str = "wukong_kldiv_f32";
const RT_KLDIV_PAR: &str = "wukong_kldiv_f32_parallel";
const RT_ENTROPY: &str = "wukong_entropy_f32";
const RT_ENTROPY_PAR: &str = "wukong_entropy_f32_parallel";
const RT_KD_LOSS: &str = "wukong_kd_loss_f32";
const RT_KD_LOSS_PAR: &str = "wukong_kd_loss_f32_parallel";
const RT_ROWARGMAX: &str = "wukong_rowargmax_i32";
const RT_ROWARGMAX_PAR: &str = "wukong_rowargmax_i32_parallel";
const RT_ROWARGMIN: &str = "wukong_rowargmin_i32";
const RT_ROWARGMIN_PAR: &str = "wukong_rowargmin_i32_parallel";
const RT_COLARGMAX: &str = "wukong_colargmax_i32";
const RT_COLARGMAX_PAR: &str = "wukong_colargmax_i32_parallel";
const RT_COLARGMIN: &str = "wukong_colargmin_i32";
const RT_COLARGMIN_PAR: &str = "wukong_colargmin_i32_parallel";
const RT_CUMSUM: &str = "wukong_cumsum_f32";
const RT_CUMSUM_PAR: &str = "wukong_cumsum_f32_parallel";
const RT_CUMPROD: &str = "wukong_cumprod_f32";
const RT_CUMPROD_PAR: &str = "wukong_cumprod_f32_parallel";
const RT_CUMMAX: &str = "wukong_cummax_f32";
const RT_CUMMAX_PAR: &str = "wukong_cummax_f32_parallel";
const RT_CUMMIN: &str = "wukong_cummin_f32";
const RT_CUMMIN_PAR: &str = "wukong_cummin_f32_parallel";
const RT_MAXPOOL2D: &str = "wukong_maxpool2d_f32";
const RT_MAXPOOL2D_PAR: &str = "wukong_maxpool2d_f32_parallel";
const RT_AVGPOOL2D: &str = "wukong_avgpool2d_f32";
const RT_AVGPOOL2D_PAR: &str = "wukong_avgpool2d_f32_parallel";
const RT_VMATH_BF16: &str = "wukong_vmath_bf16";
const RT_VMATH_F16: &str = "wukong_vmath_f16";
const RT_VMATH_BF16_OUT: &str = "wukong_vmath_bf16_out";
const RT_VMATH_F16_OUT: &str = "wukong_vmath_f16_out";
const RT_TRANSPOSE: &str = "wukong_transpose_f32";
const RT_TRANSPOSE_PAR: &str = "wukong_transpose_f32_parallel";
const RT_TRANSPOSE_U16: &str = "wukong_transpose_u16";
const RT_TRANSPOSE_U16_PAR: &str = "wukong_transpose_u16_parallel";
const RT_COLSUM: &str = "wukong_colsum_f32";
const RT_COLSUM_PAR: &str = "wukong_colsum_f32_parallel";
const RT_COLMAX: &str = "wukong_colmax_f32";
const RT_COLMAX_PAR: &str = "wukong_colmax_f32_parallel";
const RT_COLMIN: &str = "wukong_colmin_f32";
const RT_COLMIN_PAR: &str = "wukong_colmin_f32_parallel";
const RT_COLMAXABS: &str = "wukong_colmaxabs_f32";
const RT_COLMAXABS_PAR: &str = "wukong_colmaxabs_f32_parallel";
const RT_COLMEAN: &str = "wukong_colmean_f32";
const RT_COLMEAN_PAR: &str = "wukong_colmean_f32_parallel";
const RT_COLSUMSQ: &str = "wukong_colsumsq_f32";
const RT_COLSUMSQ_PAR: &str = "wukong_colsumsq_f32_parallel";
const RT_COLL2: &str = "wukong_coll2_f32";
const RT_COLL2_PAR: &str = "wukong_coll2_f32_parallel";
const RT_COLRMS: &str = "wukong_colrms_f32";
const RT_COLRMS_PAR: &str = "wukong_colrms_f32_parallel";
const RT_VELEM: &str = "wukong_velem_f32";
const RT_VELEM_PARALLEL: &str = "wukong_velem_f32_parallel";
const RT_VHORNER: &str = "wukong_vhorner_f32";
const RT_BIAS_BCAST: &str = "wukong_bias_bcast_f32";
const RT_BIAS_BCAST_PAR: &str = "wukong_bias_bcast_f32_parallel";
const RT_SREDUCE: &str = "wukong_sreduce_f32";
const RT_SREDUCE_PARALLEL: &str = "wukong_sreduce_f32_parallel";
const RT_ARGREDUCE: &str = "wukong_argreduce_f32";
const RT_ARGREDUCE_PARALLEL: &str = "wukong_argreduce_f32_parallel";
const RT_NORM: &str = "wukong_norm_f32";
const RT_NORM_PARALLEL: &str = "wukong_norm_f32_parallel";
const RT_NORM_AFFINE: &str = "wukong_norm_affine_f32";
const RT_NORM_AFFINE_PARALLEL: &str = "wukong_norm_affine_f32_parallel";
const RT_I8GEMM_NT: &str = "wukong_i8gemm_nt";
const RT_I8GEMM_NT_PARALLEL: &str = "wukong_i8gemm_nt_parallel";
const RT_I8GEMM_NT_DEQ: &str = "wukong_i8gemm_nt_deq";
const RT_I8GEMM_NT_DEQ_PARALLEL: &str = "wukong_i8gemm_nt_deq_parallel";
const RT_DEQUANT: &str = "wukong_dequant_f32";
const RT_DEQUANT_PAR: &str = "wukong_dequant_f32_parallel";
const RT_DEQUANT_PERCHAN: &str = "wukong_dequant_perchan_f32";
const RT_DEQUANT_PERCHAN_PAR: &str = "wukong_dequant_perchan_f32_parallel";
const RT_EMBEDDING: &str = "wukong_embedding_f32";
const RT_EMBEDDING_PAR: &str = "wukong_embedding_f32_parallel";
const RT_ATTENTION: &str = "wukong_attention_f32";
const RT_SCATTER_ADD: &str = "wukong_scatter_add_f32";
const RT_SCATTER_ADD_PAR: &str = "wukong_scatter_add_f32_parallel";
const RT_DOT_BF16: &str = "wukong_dot_bf16";
const RT_SUM_BF16: &str = "wukong_sum_bf16";
const RT_REDUCE_BF16: &str = "wukong_reduce_bf16";
const RT_DOT_F16: &str = "wukong_dot_f16";
const RT_SUM_F16: &str = "wukong_sum_f16";
const RT_REDUCE_F16: &str = "wukong_reduce_f16";
// The `@parallel` multicore twins (same ABI; deterministic fixed-RCHUNK chunk fold).
const RT_DOT_BF16_PAR: &str = "wukong_dot_bf16_parallel";
const RT_SUM_BF16_PAR: &str = "wukong_sum_bf16_parallel";
const RT_REDUCE_BF16_PAR: &str = "wukong_reduce_bf16_parallel";
const RT_DOT_F16_PAR: &str = "wukong_dot_f16_parallel";
const RT_SUM_F16_PAR: &str = "wukong_sum_f16_parallel";
const RT_REDUCE_F16_PAR: &str = "wukong_reduce_f16_parallel";
// f16 has no cheap inline round (unlike bf16's `<<16`), so f16 load/store/cast call these shims —
// the *same* `half`-crate conversion the interpreter uses, keeping native == interp bit-for-bit.
const RT_F32_TO_F16: &str = "wukong_f32_to_f16_bits";
const RT_F16_TO_F32: &str = "wukong_f16_bits_to_f32";
const RT_AXPBY_BF16: &str = "wukong_axpby_bf16";
const RT_AXPBY_F16: &str = "wukong_axpby_f16";
const RT_AXPBY_BF16_OUT: &str = "wukong_axpby_bf16_out";
const RT_AXPBY_F16_OUT: &str = "wukong_axpby_f16_out";
const RT_FMOD_F64: &str = "wukong_rt_fmod_f64";
const RT_FMOD_F32: &str = "wukong_rt_fmod_f32";

/// Flags for every load/store this backend emits: `notrap` (the address is in-bounds by
/// construction — bounds checks happen upstream) but deliberately **not** `aligned`.
///
/// Cranelift's `trusted()` is `notrap | aligned`, and it documents `aligned` as a promise the
/// caller makes: "the instruction is permitted to trap or return a wrong result if the effective
/// address is misaligned" (`ir/memflags.rs`). This backend cannot make that promise. The 128-bit
/// CLIF vectorizer emits `load.f32x4`/`store.f32x4` at addresses it *knows* are not 16-byte
/// aligned — `for i in 1..4095 { c[i] = ... }` over `[f32; 4096]` locals lowers to a vector load at
/// `alloca_base + 4`. On this AVX2 host nothing faults because the x64 ISLE rules pick VEX forms,
/// which tolerate misalignment; on an x86-64 host without AVX, `xmm_mem_to_xmm_mem_aligned`
/// (`isa/x64/lower/isle.rs`) sees the `aligned` bit, embeds the misaligned operand straight into a
/// legacy SSE instruction instead of loading it through `movups`, and the program takes a #GP on
/// the first vectorized iteration. Dropping the flag is strictly weaker: it can only make Cranelift
/// materialize an unaligned load it would otherwise have folded, never introduce a fault.
fn mem_flags() -> MemFlags {
    MemFlags::new().with_notrap()
}

/// The alias region every access to a **private stack slot** carries — a slot created by an
/// `Op::Alloca` of this function whose address provably never escapes it
/// ([`wukong_opt::AliasInfo::is_private_stack`]).
///
/// Cranelift 0.124 partitions memory into four static categories — `None` ("other"), `Heap`,
/// `Table`, `Vmctx` — and its `alias_analysis.rs` tracks a *separate* last-store per category, so a
/// store tagged `None` does not invalidate a load tagged `Vmctx`. That is the only channel this
/// version offers for "these two accesses are disjoint"; the `readonly` bit is a strictly stronger
/// claim (nothing writes the location for the whole function) and was measured to change nothing on
/// the shapes Wukong emits, so it is deliberately not set. The category names are wasmtime's; the
/// analysis only ever compares them for equality within one function, so the choice of `Vmctx` is
/// arbitrary.
///
/// SOUNDNESS: Cranelift requires that one memory location is never accessed under two different
/// categories within a function. A slot that never escapes can, by the definition of escaping, only
/// be reached through values whose provenance is that same `alloca` — its address never becomes a
/// store's value, a call argument, a branch argument or an integer — so every access to it is tagged
/// here, and no untracked pointer can name it. Every *other* access keeps `None`. A frame slot that
/// [`stack_color`] shares between allocas with disjoint lifetimes keeps the rule: at `-O1`+ its
/// members are all private or all not, and at `-O0` a private member of a mixed slot is demoted to
/// `None` ([`FnTranslator::demoted_slot`]).
const PRIVATE_STACK_REGION: AliasRegion = AliasRegion::Vmctx;

/// The alias region of every access to **`@noalias` parameter storage this function writes** and
/// reaches only through pointers the MIR analysis tracks
/// ([`wukong_opt::AliasInfo::promised_storage`] answering `Written`): the parameter's elements, and
/// for a slice its fat pointer too.
///
/// SOUNDNESS, by the same argument as [`PRIVATE_STACK_REGION`]: nothing derived from the parameter
/// escapes, so no untracked pointer of this function was derived from it, and the `@noalias`
/// promise excludes every pointer *not* derived from it — so every access that can touch this
/// storage is one of the accesses tagged here. A store to it therefore no longer clobbers a load
/// tagged `None` (an unpromised input) or [`PROMISED_READONLY_REGION`]. Calls stay full fences in
/// Cranelift's analysis, so nothing a callee does is affected. Two written promised parameters
/// share this one category, which only costs precision between them.
const PROMISED_WRITTEN_REGION: AliasRegion = AliasRegion::Heap;

/// The alias region of every access to **`@noalias` parameter storage this function never
/// writes** (`promised_storage` answering `ReadOnly`). No store at all carries this category, so a
/// load tagged with it is clobbered by nothing but a call: a re-read of a promised input is
/// forwarded across every store in the function.
const PROMISED_READONLY_REGION: AliasRegion = AliasRegion::Table;

/// The phrase every oversized-layout failure carries ([`FnTranslator::size_of_or_err`]); the anchor
/// [`classify_error`] keys [`NativeErrorKind::LayoutTooLarge`] on.
const LAYOUT_TOO_LARGE: &str = "is too large to lay out";

/// Message prefixes of the failures this backend raises *by design* for a well-typed program it
/// does not lower (an external call with no ABI, an assertion inside a foreign callback, an entry
/// point with parameters, a target CPU it cannot serve).
const UNSUPPORTED_PREFIXES: &[&str] = &[
    "unsupported ",
    "native entry point",
    "native buffer dispatcher",
    "no entry function",
    "raw JIT function pointers",
];

/// What a native-backend failure means to the user. Every Cranelift entry point reports failure as
/// a message (a `String`, or a [`NativeFailure`] that also says where); this is how the driver turns
/// one into a catalogued diagnostic (a stable code,
/// `--error-format=json`, `--explain`) instead of printing it bare. The phrases it keys on are
/// produced in this crate, next to this function, so the two cannot drift apart unnoticed (pinned
/// by `error_tests`).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum NativeErrorKind {
    /// An array, vector or aggregate whose byte size exceeds the backend's 32-bit layout (`C0002`).
    LayoutTooLarge,
    /// A construct the native backend declines by design (`C0001`).
    Unsupported,
    /// A failure inside the backend itself — an internal compiler error, not the user's program.
    Internal,
}

/// A native-backend failure together with the source construct it was raised on — what the
/// driver-facing entry points ([`jit_compile_for_with`], [`emit_object_kind_for_with`] and
/// [`emit_object_kind_timed_for_with`]) return, so `C0002`, `C0001` and a backend ICE can carry a
/// primary span. `site` is the span of the MIR value being lowered when the failure was recorded
/// ([`wukong_mir::Function::value_loc`]), else the failing function's declaration, else `None` (a
/// hand-built program, or a failure no single function owns). Every other entry point still
/// reports the message alone, through `From<NativeFailure> for String`.
#[derive(Clone, Debug, PartialEq)]
pub struct NativeFailure {
    pub message: String,
    pub site: Option<Span>,
}

impl std::fmt::Display for NativeFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for NativeFailure {}

impl From<String> for NativeFailure {
    fn from(message: String) -> NativeFailure {
        NativeFailure {
            message,
            site: None,
        }
    }
}

impl From<NativeFailure> for String {
    fn from(failure: NativeFailure) -> String {
        failure.message
    }
}

/// Classify a failure returned by [`jit_compile`], [`jit_run`] or [`emit_object`].
pub fn classify_error(message: &str) -> NativeErrorKind {
    if message.contains(LAYOUT_TOO_LARGE) {
        NativeErrorKind::LayoutTooLarge
    } else if UNSUPPORTED_PREFIXES.iter().any(|p| message.starts_with(p)) {
        NativeErrorKind::Unsupported
    } else {
        NativeErrorKind::Internal
    }
}

/// The runtime function an intrinsic call lowers to.
#[derive(Clone, Copy)]
enum Intrinsic {
    PrintInt,
    PrintUint,
    /// An `f64` value: `wukong_rt_print_f64`.
    PrintFloat,
    /// An `f32` value, or an `f16`/`bf16` one held in an `f32` register: `wukong_rt_print_f32`,
    /// which prints the shortest round-trip digits at `f32` precision (`wukong_runtime::fmt`).
    PrintF32,
    PrintStr,
    PrintBool,
    PrintChar,
    Assert,
}

/// `arg` is the MIR type of the call's first argument, when it has one. A `print` of a float picks
/// its printer by that type, so an `f32` prints at its own precision instead of as the `f64` it
/// widens to — the interpreter makes the same choice from the same type.
fn classify_intrinsic(name: &str, arg: Option<&MirType>) -> Option<Intrinsic> {
    match name {
        "print" | "println" => Some(match arg {
            Some(MirType::F32 | MirType::F16 | MirType::BF16) => Intrinsic::PrintF32,
            Some(t) if t.is_float() => Intrinsic::PrintFloat,
            _ => Intrinsic::PrintInt,
        }),
        // mir_build routes a `*u8` (string) argument to these symbols (see `print_str` in GemmSyms).
        "print_str" | "println_str" => Some(Intrinsic::PrintStr),
        // mir_build routes an unsigned-integer argument here (zero-extended to 64 bits).
        "print_u" | "println_u" => Some(Intrinsic::PrintUint),
        // mir_build routes a `bool` / `char` argument here (zero-extended to 64 bits).
        "print_bool" | "println_bool" => Some(Intrinsic::PrintBool),
        "print_char" | "println_char" => Some(Intrinsic::PrintChar),
        "assert" => Some(Intrinsic::Assert),
        _ => None,
    }
}

// --- Type and size mapping ------------------------------------------------------------------

/// Is `t` the general vectorizer's `<4 x bf16>`: four bf16 lanes held as an `f32x4`, stored as four
/// 2-byte elements? Its loads, stores and grid-rounding casts are lowered on their own.
fn is_bf16_x4(t: &MirType) -> bool {
    matches!(t, MirType::Vec(lane, 4) if **lane == MirType::BF16)
}

/// Map a MIR type to its Cranelift type, or `None` for `Void`. `i1` becomes `i8`; `f16`/`bf16` are
/// computed as `f32` (matching the interpreter's promotion of sub-`f32` floats).
fn cl_type(t: &MirType, ptr_ty: Type) -> Option<Type> {
    Some(match t {
        MirType::I1 | MirType::I8 => types::I8,
        MirType::I16 => types::I16,
        MirType::I32 => types::I32,
        MirType::I64 => types::I64,
        MirType::I128 => types::I128,
        MirType::F16 | MirType::BF16 | MirType::F32 => types::F32,
        MirType::F64 => types::F64,
        MirType::Ptr => ptr_ty,
        MirType::Vec(elem, n) => {
            let lane = cl_type(elem, ptr_ty)?;
            lane.by(*n)?
        }
        MirType::Array(..) => ptr_ty,
        MirType::Void => return None,
    })
}

/// Size in bytes of a MIR type (for `gep` scaling and stack-slot sizing). Returns `None` when the
/// layout does not fit in a `u32` byte count — an oversized array/vector extent (e.g.
/// `[i32; 999999999999]`) whose `elem_size * count` overflows. Callers turn that into a clean
/// "type too large to lay out" compiler error via [`FnTranslator::size_of_or_err`] instead of
/// letting the `u32` multiply overflow-panic.
fn size_of(t: &MirType) -> Option<u32> {
    Some(match t {
        MirType::I1 | MirType::I8 => 1,
        MirType::I16 | MirType::F16 | MirType::BF16 => 2,
        MirType::I32 | MirType::F32 => 4,
        MirType::I64 | MirType::F64 | MirType::Ptr => 8,
        MirType::I128 => 16,
        MirType::Vec(e, n) | MirType::Array(e, n) => size_of(e)?.checked_mul(*n)?,
        MirType::Void => 0,
    })
}

fn int_cc(op: CmpOp) -> IntCC {
    use CmpOp::*;
    match op {
        Eq => IntCC::Equal,
        Ne => IntCC::NotEqual,
        Slt => IntCC::SignedLessThan,
        Sle => IntCC::SignedLessThanOrEqual,
        Sgt => IntCC::SignedGreaterThan,
        Sge => IntCC::SignedGreaterThanOrEqual,
        Ult => IntCC::UnsignedLessThan,
        Ule => IntCC::UnsignedLessThanOrEqual,
        Ugt => IntCC::UnsignedGreaterThan,
        Uge => IntCC::UnsignedGreaterThanOrEqual,
        _ => unreachable!("float predicate in int_cc"),
    }
}

fn float_cc(op: CmpOp) -> FloatCC {
    use CmpOp::*;
    match op {
        Foeq => FloatCC::Equal,
        // The interpreter implements `!=` with Rust's `f64::ne` (unordered-or-not-equal), so match
        // that rather than the ordered `one`.
        Fone => FloatCC::NotEqual,
        Folt => FloatCC::LessThan,
        Fole => FloatCC::LessThanOrEqual,
        Fogt => FloatCC::GreaterThan,
        Foge => FloatCC::GreaterThanOrEqual,
        _ => unreachable!("int predicate in float_cc"),
    }
}

/// The slot size from which a stack slot's base is 32-byte aligned: one YMM register. Below it the
/// cap stays 16 bytes (the stack pointer's own alignment, so it costs nothing).
const WIDE_SLOT_BYTES: u32 = 32;

/// The alignment the x64 ABI keeps the stack pointer at, and so the most a slot's frame offset
/// alone can guarantee (see [`over_aligned_slot_bytes`]).
const FRAME_ALIGN: u32 = 16;

fn align_shift(bytes: u32) -> u8 {
    // Align stack slots to their (rounded-up power-of-two) size, capped at 16 bytes — or at 32 for
    // a slot of 32 bytes or more, so a 256-bit access to a local array never splits a cache line
    // at its base. This is the slot BASE only — an interior address such as `base + 4` is not
    // aligned, which is why [`mem_flags`] must never set Cranelift's `aligned` bit.
    let cap = if bytes >= WIDE_SLOT_BYTES {
        WIDE_SLOT_BYTES
    } else {
        16
    };
    let mut a = 1u32;
    let mut shift = 0u8;
    while a < bytes && a < cap {
        a <<= 1;
        shift += 1;
    }
    shift
}

/// The bytes a stack slot of `bytes` bytes reserves when its base must be `want`-aligned with
/// `want > FRAME_ALIGN`: a wide slot (`>= WIDE_SLOT_BYTES`, `want = 32`) or an `@align(N)` local.
///
/// Cranelift 0.124 lays slots out at their requested alignment *relative to the stack pointer*,
/// which the x64 ABI aligns to 16 bytes only, and it never realigns a frame (`machinst/abi.rs`
/// rounds slot offsets; the prologue never `and`s `rsp`). So a 32-byte request alone gives a
/// 32-byte-aligned address only when the caller's stack happened to be — every other call depth
/// it is 16 mod 32. The slot therefore reserves `want - 16` bytes of slack, and the lowering rounds
/// its address up to the next `want` boundary at run time (`(base + want - 16) & -want`, which
/// stays inside the slot because the base is always 16-aligned). Saturating: an absurd size fails
/// Cranelift's own frame limit instead of wrapping.
fn over_aligned_slot_bytes(bytes: u32, want: u32) -> u32 {
    bytes.saturating_add(want - FRAME_ALIGN)
}

// --- Per-function lowering ------------------------------------------------------------------

/// Allocation capacity reused by serial lowering and by each parallel iterator state. All
/// function-local handles are cleared before the next function's DFG is populated.
#[derive(Default)]
struct TranslationScratch {
    vmap: Vec<Option<Value>>,
    blocks: Vec<Block>,
    func_refs: HashMap<Symbol, FuncRef>,
    data_refs: HashMap<Symbol, GlobalValue>,
    rt_refs: HashMap<&'static str, FuncRef>,
    kernel_refs: Vec<FuncRef>,
    call_args: Vec<Value>,
    branch_args: Vec<BlockArg>,
}

/// Lowers one MIR `Function` into a Cranelift function body already attached to `builder`.
struct FnTranslator<'a> {
    builder: FunctionBuilder<'a>,
    func: &'a Function,
    interner: &'a Interner,
    ptr_ty: Type,
    /// MIR `ValueId` -> Cranelift value. Block params and instruction results fill this in.
    vmap: &'a mut Vec<Option<Value>>,
    /// MIR block index -> Cranelift block.
    blocks: &'a [Block],
    /// Pre-declared FuncRefs for callees and runtime imports in this function.
    func_refs: &'a HashMap<Symbol, FuncRef>,
    decls: &'a Decls<'a>,
    rt_ids: &'a HashMap<&'static str, FuncId>,
    rt_refs: &'a mut HashMap<&'static str, FuncRef>,
    /// Temporary arguments are copied into the DFG by the instruction builder. Reuse their
    /// capacity instead of allocating a new vector at every user call and branch.
    call_args: &'a mut Vec<Value>,
    branch_args: &'a mut Vec<BlockArg>,
    /// Pre-declared GlobalValues for the read-only static-data blobs (`Op::GlobalAddr`), one per
    /// unique string literal, imported into this function's DFG.
    data_refs: &'a HashMap<Symbol, GlobalValue>,
    /// FuncRefs for this function's synthesized AVX2 vector kernels, indexed by `Op::VecKernelCall`'s
    /// `kernel` field (the position in the owning `Function::vec_kernels`).
    kernel_refs: &'a [FuncRef],
    /// MIR provenance facts for `func`, used only to pick the [`MemFlags`] alias region of each
    /// load/store (see [`PRIVATE_STACK_REGION`]). Purely an optimization hint: dropping it would
    /// emit the same instructions with weaker flags.
    alias: wukong_opt::AliasInfo,
    /// First unrecoverable lowering failure seen while lowering this function, if any: a "type too
    /// large to lay out" overflow ([`FnTranslator::size_of_or_err`]), or a runtime kernel that
    /// reached [`FnTranslator::lower_call`] with an arity no dispatch arm accepts. Recorded
    /// (instead of panicking, or silently dropping the instruction) so `populate_module` can abort
    /// with a clean error before the half-built function is finalized/defined.
    layout_err: Option<String>,
    /// The located MIR value whose code is being emitted, or `None` when that code has no known
    /// location (see [`FnTranslator::translate`] for which instructions carry which). Its `ValueId`
    /// is also the Cranelift `SourceLoc` of every instruction emitted since, so a machine-code range
    /// maps back to a span through [`Function::value_loc`] (what a DWARF line table consumes); code
    /// with no location carries the default `SourceLoc`. Locations never change code:
    /// `srcloc_tests` pins the object bytes with and without them.
    cur_loc: Option<ValueId>,
    /// Where `layout_err` was raised: the span of `cur_loc` at that moment, else the function's
    /// declaration. Becomes [`NativeFailure::site`].
    err_site: Option<Span>,
    /// Which allocas share a frame slot ([`stack_color`]); empty when coloring is off.
    slot_plan: stack_color::SlotPlan,
    /// The Cranelift slot of each shared color, created when its first member is lowered.
    color_slots: Vec<Option<StackSlot>>,
}

/// The value a terminator's code is attributed to: a branch's condition, a switch's scrutinee, a
/// return's value.
fn term_operand(term: &Terminator) -> Option<ValueId> {
    match term {
        Terminator::CondBr { cond, .. } => Some(*cond),
        Terminator::Switch { scrutinee, .. } => Some(*scrutinee),
        Terminator::Ret(v) => *v,
        Terminator::Br { .. } | Terminator::Unreachable => None,
    }
}

/// Does the result-less instruction `op` read `v`? Lists the ops that produce no value; any other
/// op answers `false`, which drops a location rather than borrowing one.
fn void_op_reads(op: &Op, v: ValueId) -> bool {
    match op {
        Op::Store { ptr, value } | Op::AtomicAdd { ptr, value } => *ptr == v || *value == v,
        Op::Call { args, .. } => args.contains(&v),
        Op::VecKernelCall {
            ptrs, scalars, n, ..
        } => [*ptrs, *scalars, *n].contains(&v),
        _ => false,
    }
}

/// Does `term` read `v` — as its condition, scrutinee or return value, or as an edge argument?
fn term_reads(term: &Terminator, v: ValueId) -> bool {
    let mut reads = term_operand(term) == Some(v);
    term.for_each_edge(|_, args| reads |= args.contains(&v));
    reads
}

impl<'a> FnTranslator<'a> {
    /// Import a runtime signature only when an emitted instruction needs it. The cache is
    /// keyed by the registry's static names; its iteration order never enters generated code.
    fn runtime_ref(&mut self, name: &str) -> FuncRef {
        if let Some(&fref) = self.rt_refs.get(name) {
            return fref;
        }
        let (&name, &id) = self
            .rt_ids
            .get_key_value(name)
            .expect("unknown runtime symbol");
        let fref = self.decls.declare_func_in_func(id, self.builder.func);
        self.rt_refs.insert(name, fref);
        fref
    }

    fn val(&self, v: ValueId) -> Value {
        self.vmap[v.0 as usize].expect("MIR value used before definition (non-SSA input?)")
    }

    fn set(&mut self, v: ValueId, cv: Value) {
        self.vmap[v.0 as usize] = Some(cv);
    }

    /// The Cranelift stack slot the alloca `res` lives in: its color's shared slot when
    /// [`stack_color`] put it in one (created by the first member lowered, at the color's size and
    /// alignment, which cover `spec`), else a slot of its own shaped `spec`.
    fn alloca_stack_slot(
        &mut self,
        res: Option<ValueId>,
        spec: stack_color::SlotSpec,
    ) -> StackSlot {
        let shared = res.and_then(|v| self.slot_plan.color_of(v));
        if let Some((c, _)) = shared {
            if let Some(slot) = self.color_slots.get(c as usize).copied().flatten() {
                return slot;
            }
        }
        let shape = shared.map_or(spec, |(_, color)| color);
        let slot = self.builder.create_sized_stack_slot(StackSlotData::new(
            StackSlotKind::ExplicitSlot,
            shape.bytes,
            shape.align_shift,
        ));
        if let Some((c, _)) = shared {
            let c = c as usize;
            if self.color_slots.len() <= c {
                self.color_slots.resize(c + 1, None);
            }
            self.color_slots[c] = Some(slot);
        }
        slot
    }

    /// Byte size of `ty` for stack-slot sizing / `gep` scaling. On an unrepresentable (overflowing)
    /// layout it records a clean compiler error and returns a harmless 1-byte placeholder so the
    /// half-built Cranelift IR stays well-formed; `populate_module` checks `layout_err` right after
    /// lowering and aborts the compile before this function is defined. This is why an oversized
    /// type (e.g. `[i32; 999999999999]`) yields a diagnostic + nonzero exit, not a Rust panic.
    fn size_of_or_err(&mut self, ty: &MirType) -> u32 {
        match size_of(ty) {
            Some(b) => b,
            None => {
                if self.layout_err.is_none() {
                    self.layout_err = Some(format!(
                        "type `{}` in function `{}` {LAYOUT_TOO_LARGE}: its byte size exceeds \
                         2^32 (4 GiB), the native backend's addressing limit",
                        ty.display(),
                        self.interner.resolve(self.func.name)
                    ));
                }
                1
            }
        }
    }

    fn ty_of(&self, v: ValueId) -> &MirType {
        self.func.value_type(v)
    }

    /// [`mem_flags`] plus the alias region for an access through `ptr`. See
    /// [`PRIVATE_STACK_REGION`] and [`PROMISED_WRITTEN_REGION`] for the two classes that get one —
    /// a never-escaping stack slot, and never-escaping `@noalias` parameter storage — and why that
    /// is sound. Every other access keeps `None`.
    fn mem_flags_for(&self, ptr: ValueId) -> MemFlags {
        let mut fl = mem_flags();
        if self.alias.is_private_stack(ptr) && !self.demoted_slot(ptr) {
            fl.set_alias_region(Some(PRIVATE_STACK_REGION));
        } else if let Some(s) = self.alias.promised_storage(ptr) {
            fl.set_alias_region(Some(match s {
                wukong_opt::alias::PromisedStorage::Written => PROMISED_WRITTEN_REGION,
                wukong_opt::alias::PromisedStorage::ReadOnly => PROMISED_READONLY_REGION,
            }));
        }
        fl
    }

    /// Does `ptr` point into a private slot that [`stack_color`] shares with a non-private one, so
    /// that it must give up [`PRIVATE_STACK_REGION`] (only at `-O0`; see `SlotPlan::demotes`)?
    fn demoted_slot(&self, ptr: ValueId) -> bool {
        match self.alias.prov(ptr) {
            wukong_opt::Prov::Alloca(a) => self.slot_plan.demotes(a),
            _ => false,
        }
    }

    /// [`Self::mem_flags_for`] for a load of `ty` through `ptr`. A load from the fat pointer of a
    /// `@noalias` slice that nothing in this function writes or publishes
    /// ([`wukong_opt::AliasInfo::is_invariant_slice_header`]) is additionally `readonly` +
    /// `can_move`: the header holds one value for the whole call and is a live 16-byte object the
    /// caller keeps, so the load can neither fault nor observe a change wherever it runs. Cranelift
    /// then treats it as a pure node — GVN merges the per-subscript re-reads of the data pointer
    /// and its LICM lifts them out of the loop.
    fn load_flags_for(&self, ptr: ValueId, ty: &MirType) -> MemFlags {
        let fl = self.mem_flags_for(ptr);
        match size_of(ty) {
            Some(bytes) if self.alias.is_invariant_slice_header(ptr, bytes) => {
                fl.with_readonly().with_can_move()
            }
            _ => fl,
        }
    }

    fn dfg_ty(&self, v: Value) -> Type {
        self.builder.func.dfg.value_type(v)
    }

    /// Reverse-postorder of blocks so every definition is emitted before its uses.
    fn rpo(func: &Function) -> Vec<u32> {
        let mut visited = vec![false; func.blocks.len()];
        let mut post = Vec::with_capacity(func.blocks.len());
        let mut stack = vec![(func.entry.0, 0usize)];
        visited[func.entry.0 as usize] = true;
        while let Some(&mut (b, ref mut i)) = stack.last_mut() {
            // Edge order (then before else, switch cases before the default); a target reached
            // twice is skipped by `visited`, so the order matches a deduplicated successor list.
            let succ = func.blocks[b as usize].term.edge_target(*i).map(|t| t.0);
            if let Some(s) = succ {
                *i += 1;
                if !visited[s as usize] {
                    visited[s as usize] = true;
                    stack.push((s, 0));
                }
            } else {
                post.push(b);
                stack.pop();
            }
        }
        post.reverse();
        post
    }

    /// Attribute the instructions emitted from here on to the located value `v`.
    fn mark(&mut self, v: ValueId) {
        self.cur_loc = Some(v);
        self.builder.set_srcloc(SourceLoc::new(v.0));
    }

    /// Attribute the instructions emitted from here on to nothing (the default `SourceLoc`).
    fn unmark(&mut self) {
        if self.cur_loc.take().is_some() {
            self.builder.set_srcloc(SourceLoc::default());
        }
    }

    /// Keep the current mark only when `reads` says the code about to be emitted reads the marked
    /// value; otherwise drop it.
    fn keep_mark_if(&mut self, reads: impl FnOnce(ValueId) -> bool) {
        if !self.cur_loc.is_some_and(reads) {
            self.unmark();
        }
    }

    /// `v` when it has a recorded source location.
    fn located(&self, v: Option<ValueId>) -> Option<ValueId> {
        v.filter(|&v| self.func.value_loc(v).is_some())
    }

    /// The span a failure raised now is reported at: the value being lowered, else the
    /// function's declaration.
    fn current_site(&self) -> Option<Span> {
        self.cur_loc
            .and_then(|v| self.func.value_loc(v))
            .or(self.func.decl_span())
    }

    /// Lower the whole function, attributing each Cranelift instruction to a MIR value
    /// ([`FnTranslator::mark`]) — or to nothing. A location is dropped rather than borrowed, since a
    /// wrong one would point a failure (and a debugger) at code that did not produce it:
    /// - a block starts with no location, never the last one of the block RPO lowered before it;
    /// - an instruction with a result carries that result's location, or none when it is unknown;
    /// - an instruction without a result (a store, a void call) has no table entry of its own: it
    ///   keeps the mark only while it reads the marked value (`store p, v` or `f(.., v)` right after
    ///   `v`), and otherwise carries none;
    /// - a terminator carries its condition's, scrutinee's or return value's location, else keeps the
    ///   mark only while it reads the marked value (an edge argument).
    fn translate(&mut self) {
        // Give each Cranelift block its params (the entry block's are the function params).
        for (i, mb) in self.func.blocks.iter().enumerate() {
            let cb = self.blocks[i];
            if mb.id == self.func.entry {
                self.builder.append_block_params_for_function_params(cb);
                for (p, &cv) in mb.params.iter().zip(self.builder.block_params(cb)) {
                    self.vmap[p.0 as usize] = Some(cv);
                }
            } else {
                for p in &mb.params {
                    let t = cl_type(self.ty_of(*p), self.ptr_ty).unwrap_or(self.ptr_ty);
                    let cv = self.builder.append_block_param(cb, t);
                    self.set(*p, cv);
                }
            }
        }

        let order = Self::rpo(self.func);
        // Copy the shared reference so lowering can mutate the translator without cloning MIR.
        let func = self.func;
        for &bi in &order {
            let cb = self.blocks[bi as usize];
            self.builder.switch_to_block(cb);
            let mb = &func.blocks[bi as usize];
            self.unmark();
            for inst in &mb.insts {
                match (inst.result, self.located(inst.result)) {
                    (_, Some(v)) => self.mark(v),
                    (Some(_), None) => self.unmark(),
                    (None, None) => self.keep_mark_if(|m| void_op_reads(&inst.op, m)),
                }
                self.lower_inst(inst);
                if self.layout_err.is_some() {
                    self.err_site = self.current_site();
                    // Failed value-producing calls do not populate vmap. Stop before any use
                    // (including the terminator) can obscure the diagnostic with a panic.
                    return;
                }
            }
            match self.located(term_operand(&mb.term)) {
                Some(v) => self.mark(v),
                None => self.keep_mark_if(|m| term_reads(&mb.term, m)),
            }
            self.lower_term(&mb.term);
        }
        self.builder.seal_all_blocks();
    }

    fn lower_inst(&mut self, inst: &wukong_mir::Inst) {
        let res = inst.result;
        let cv = match &inst.op {
            Op::ConstInt(v, ty) => {
                if matches!(ty, MirType::I1) {
                    self.builder.ins().iconst(types::I8, (*v as i64) & 1)
                } else if matches!(ty, MirType::I128) {
                    self.i128_const(*v)
                } else {
                    let t = cl_type(ty, self.ptr_ty).unwrap_or(types::I64);
                    self.builder.ins().iconst(t, *v as i64)
                }
            }
            Op::ConstFloat(v, ty) => match ty {
                MirType::F64 => self.builder.ins().f64const(*v),
                // A bf16/f16 const IS its grid-rounded value; materialize it rounded so it stays
                // correct however the optimizer forwards or folds it. The per-store rounding alone is
                // fragile — mem2reg and CSE load-forwarding bypass the store, dropping the rounding and
                // making -O2 disagree with -O0 (`let b: bf16 = 0.1` then forwarded the f32 0.1, not the
                // bf16-grid 0.10009765625). Rounding here is the same `half`-crate path the interp uses.
                MirType::BF16 => self
                    .builder
                    .ins()
                    .f32const(wukong_runtime::round_bf16(*v as f32)),
                MirType::F16 => self
                    .builder
                    .ins()
                    .f32const(wukong_runtime::round_f16(*v as f32)),
                _ => self.builder.ins().f32const(*v as f32),
            },
            Op::Bin(op, l, r) => {
                let rt = self.func.value_type(inst.result.unwrap());
                self.lower_bin(*op, *l, *r, rt)
            }
            Op::Cmp(op, l, r) => {
                if op.is_float() {
                    // The front-end's float literals are loosely typed, so operands may differ in
                    // width; compare in the wider type (the interpreter compares in f64).
                    let (a0, b0) = (self.val(*l), self.val(*r));
                    let common = if self.dfg_ty(a0) == types::F64 || self.dfg_ty(b0) == types::F64 {
                        types::F64
                    } else {
                        types::F32
                    };
                    let a = self.coerce_float(a0, common);
                    let b = self.coerce_float(b0, common);
                    self.builder.ins().fcmp(float_cc(*op), a, b)
                } else {
                    let (a, b) = (self.val(*l), self.val(*r));
                    self.builder.ins().icmp(int_cc(*op), a, b)
                }
            }
            Op::Neg(v) => {
                let x = self.val(*v);
                // A lane-float vector (`Vec(f32, N)`) is not itself `is_float()`, so dispatch on the
                // lane type for vectors. Otherwise an autovectorized `-x[k]` over `[f32; N]` emits
                // `ineg` on a float vector — invalid CLIF that the verifier rejects.
                let is_float = match self.ty_of(*v) {
                    MirType::Vec(lane, _) => lane.is_float(),
                    t => t.is_float(),
                };
                if is_float {
                    self.builder.ins().fneg(x)
                } else {
                    self.builder.ins().ineg(x)
                }
            }
            Op::Not(v) => {
                let x = self.val(*v);
                self.builder.ins().bnot(x)
            }
            Op::Cast(kind, v, to) => self.lower_cast(*kind, *v, to),
            Op::Select(c, a, b) => {
                let cc = self.val(*c);
                let av0 = self.val(*a);
                let bv0 = self.val(*b);
                let avty = self.dfg_ty(av0);
                if avty.is_vector() {
                    // Vector blend: the mask is a per-lane all-ones/all-zeros vector (from a vector
                    // compare). Reinterpret it to the value vector type and select bitwise.
                    let cty = self.dfg_ty(cc);
                    let mask = if cty == avty {
                        cc
                    } else {
                        self.builder.ins().bitcast(avty, MemFlags::new(), cc)
                    };
                    self.builder.ins().bitselect(mask, av0, bv0)
                } else {
                    // Cranelift requires both arms to share a type; unify mismatched float widths.
                    let (av, bv) = if self.dfg_ty(av0).is_float() || self.dfg_ty(bv0).is_float() {
                        let common =
                            if self.dfg_ty(av0) == types::F64 || self.dfg_ty(bv0) == types::F64 {
                                types::F64
                            } else {
                                types::F32
                            };
                        (
                            self.coerce_float(av0, common),
                            self.coerce_float(bv0, common),
                        )
                    } else {
                        (av0, bv0)
                    };
                    if self.dfg_ty(av) == types::I128 {
                        // Cranelift's mid-end rewrites `select(icmp(x, y), x, y)` into
                        // `smin`/`umax`/..., which x64 cannot lower at 128 bits; selecting each
                        // half on its own is the same value and gives the rewrite no `x`/`y`.
                        self.i128_select(cc, av, bv)
                    } else {
                        self.builder.ins().select(cc, av, bv)
                    }
                }
            }
            Op::Alloca { ty, align, .. } => {
                let bytes = self.size_of_or_err(ty).max(1);
                // `align` (from `@align(N)`) only ever raises the slot's alignment; `0` is natural.
                let (spec, want) = stack_color::alloca_slot(bytes, *align);
                let slot = self.alloca_stack_slot(res, spec);
                let base = self.builder.ins().stack_addr(self.ptr_ty, slot, 0);
                if want > FRAME_ALIGN {
                    // Past the stack pointer's own alignment the base needs a run-time round-up
                    // inside the slot's slack: see `over_aligned_slot_bytes`.
                    let slack = want - FRAME_ALIGN;
                    let bumped = self.builder.ins().iadd_imm(base, i64::from(slack));
                    self.builder.ins().band_imm(bumped, -i64::from(want))
                } else {
                    base
                }
            }
            Op::Barrier => {
                self.builder.ins().fence();
                return;
            }
            // `atomic_add(place, v)`: one sequentially consistent `lock xadd`-class RMW whose old
            // value is discarded. Plain `mem_flags()`, never `mem_flags_for`: the escape scan's
            // fallback marks the atomic's pointer escaped, so its location is neither a private
            // stack slot nor live `@noalias` storage, and every other access to it carries the
            // same `None` alias region this does. Never `aligned` either (see `mem_flags`).
            Op::AtomicAdd { ptr, value } => {
                let addr = self.val(*ptr);
                let v = self.val(*value);
                let ty = self.dfg_ty(v);
                self.builder
                    .ins()
                    .atomic_rmw(ty, mem_flags(), AtomicRmwOp::Add, addr, v);
                return;
            }
            Op::Load(p, ty) => {
                let addr = self.val(*p);
                let flags = self.load_flags_for(*p, ty);
                if is_bf16_x4(ty) {
                    // `<4 x bf16>` (the general vectorizer's bf16 lanes): four 2-byte elements,
                    // each widened exactly as the scalar load above widens one — zero-extended
                    // into the top half of an f32 lane. Register type is `f32x4` (see `cl_type`).
                    let ext = self.builder.ins().uload16x4(flags, addr, 0);
                    let shifted = self.builder.ins().ishl_imm(ext, 16);
                    self.vec_bitcast(types::F32X4, shifted)
                } else if matches!(ty, MirType::BF16) {
                    // bf16 storage is 2 bytes; load the 16 bits and widen to f32 exactly (the bits
                    // become the top half of the f32). Register type is f32 (see `cl_type`).
                    let half = self.builder.ins().load(types::I16, flags, addr, 0);
                    let ext = self.builder.ins().uextend(types::I32, half);
                    let shifted = self.builder.ins().ishl_imm(ext, 16);
                    self.builder
                        .ins()
                        .bitcast(types::F32, MemFlags::new(), shifted)
                } else if matches!(ty, MirType::F16) {
                    // f16 storage is 2 bytes; widen via the runtime shim (no cheap inline bit-extend
                    // like bf16). Register type is f32.
                    let half = self.builder.ins().load(types::I16, flags, addr, 0);
                    let ext = self.builder.ins().uextend(types::I32, half);
                    let fref = self.runtime_ref(RT_F16_TO_F32);
                    let call = self.builder.ins().call(fref, &[ext]);
                    self.builder.inst_results(call)[0]
                } else {
                    let t = cl_type(ty, self.ptr_ty).unwrap_or(self.ptr_ty);
                    self.builder.ins().load(t, flags, addr, 0)
                }
            }
            Op::Store { ptr, value } => {
                let addr = self.val(*ptr);
                let flags = self.mem_flags_for(*ptr);
                if is_bf16_x4(self.ty_of(*value)) {
                    // Each lane rounded to bf16 exactly as the scalar store below rounds one, the
                    // four top halves packed into 8 bytes. After the shift every lane is in
                    // `0..=0xffff`, so the saturating pack (`packusdw`) is a plain truncation.
                    let v = self.val(*value);
                    let rounded = self.round_to_bf16_x4(v);
                    let bits = self.vec_bitcast(types::I32X4, rounded);
                    let hi = self.builder.ins().ushr_imm(bits, 16);
                    let packed = self.builder.ins().unarrow(hi, hi);
                    let wide = self.vec_bitcast(types::I64X2, packed);
                    let low = self.builder.ins().extractlane(wide, 0);
                    self.builder.ins().store(flags, low, addr, 0);
                } else if matches!(self.ty_of(*value), MirType::BF16) {
                    // Round the f32 register to bf16 and store the top 16 bits (2 bytes).
                    let v = self.val(*value);
                    let rounded = self.round_to_bf16(v);
                    let bits = self
                        .builder
                        .ins()
                        .bitcast(types::I32, MemFlags::new(), rounded);
                    let hi = self.builder.ins().ushr_imm(bits, 16);
                    let half = self.builder.ins().ireduce(types::I16, hi);
                    self.builder.ins().store(flags, half, addr, 0);
                } else if matches!(self.ty_of(*value), MirType::F16) {
                    // Round the f32 register to f16 via the runtime shim, store the 16 bits (2 bytes).
                    let v = self.val(*value);
                    let fref = self.runtime_ref(RT_F32_TO_F16);
                    let call = self.builder.ins().call(fref, &[v]);
                    let bits = self.builder.inst_results(call)[0]; // i32, low 16 = f16 bits
                    let half = self.builder.ins().ireduce(types::I16, bits);
                    self.builder.ins().store(flags, half, addr, 0);
                } else {
                    let v = self.val(*value);
                    self.builder.ins().store(flags, v, addr, 0);
                }
                return;
            }
            Op::Gep { ptr, index, elem } => {
                let base = self.val(*ptr);
                let idx = self.ptr_int(*index);
                let scale = self.size_of_or_err(elem) as i64;
                let scaled = self.builder.ins().imul_imm(idx, scale);
                self.builder.ins().iadd(base, scaled)
            }
            Op::Call { func, args } => match self.lower_call(*func, args) {
                Some(v) => v,
                None => return,
            },
            // A synthesized 256-bit AVX2 vector kernel. The kernel is installed as raw machine code
            // (see `avx2::assemble_kernel` + `define_function_bytes`) under the module's default call
            // conv (Win64 here: ptrs->RCX, scalars->RDX, n->R8), so a plain Cranelift `call` reaches
            // it. It writes through the stream pointers; an elementwise kernel returns nothing, a
            // reduction kernel returns its horizontal fold in its lane type — XMM0 for an f32/f64
            // lane, RAX/EAX for an i32/i64 one (see the `res` match below).
            Op::VecKernelCall {
                kernel,
                ptrs,
                scalars,
                n,
            } => {
                let fref = self.kernel_refs[*kernel as usize];
                let ptrs_v = self.val(*ptrs);
                let scalars_v = self.val(*scalars);
                // The kernel's `n` parameter is i64; the count is non-negative, so zero-extend.
                let n_raw = self.val(*n);
                let n_ty = self.dfg_ty(n_raw);
                let n_v = if n_ty == types::I64 {
                    n_raw
                } else {
                    self.resize_int(n_raw, n_ty, types::I64, false)
                };
                let call = self.builder.ins().call(fref, &[ptrs_v, scalars_v, n_v]);
                // A reduction kernel returns its horizontal fold (lane-typed, matching the declared
                // signature); an elementwise kernel is void.
                match res {
                    Some(_) => self.builder.inst_results(call)[0],
                    None => return,
                }
            }
            Op::FuncAddr(sym) => {
                let fref = self.func_refs[sym];
                self.builder.ins().func_addr(self.ptr_ty, fref)
            }
            // The read-only address of a `.rodata` string blob: materialize the pre-declared data
            // symbol's address. In the JIT this resolves to the blob's in-process address; in the
            // object it emits a relocation into the `.rodata` section — so a returned `*u8` is a real
            // static address, not a reclaimed stack slot.
            Op::GlobalAddr(sym) => {
                let gv = self.data_refs[sym];
                self.builder.ins().global_value(self.ptr_ty, gv)
            }
            Op::Splat(v) => {
                let rty = self.ty_of(inst.result.unwrap());
                let vec_ty = cl_type(rty, self.ptr_ty).unwrap_or(self.ptr_ty);
                let x = self.val(*v);
                self.builder.ins().splat(vec_ty, x)
            }
            // One lane of a vector, by constant index: one shuffle, and free for lane 0. This is
            // how the loop vectorizer keeps a float reduction's accumulate serial without a stack
            // round-trip. The MIR verifier has already checked the index is in range.
            Op::ExtractLane(v, k) => {
                let x = self.val(*v);
                self.builder.ins().extractlane(x, *k as u8)
            }
            // The lane-index ramp, as a constant-pool vector. Cranelift has no `iota`, and it does
            // not need one: the value is a compile-time constant, so it becomes one `vconst` (a
            // rip-relative 16-byte load the loop's preheader hoists) rather than any per-iteration
            // work. `ConstantData` is a flat little-endian byte image of the whole register, which
            // is exactly how the x64 backend reads it back, so lane `k` occupies bytes
            // `[k*w, (k+1)*w)` with its own little-endian encoding — and every lane index here is
            // below 16, so only the lowest byte of a lane is ever non-zero.
            Op::Iota(ty) => {
                // The fallback mirrors `ConstInt`'s: `lower_inst` has no error channel, and a type
                // `cl_type` cannot render is malformed MIR the verifier rejects before any backend
                // runs (`wukong_driver::verify_or_ice`).
                let vec_ty = cl_type(ty, self.ptr_ty).unwrap_or(types::I32X4);
                let lanes = vec_ty.lane_count() as usize;
                let width = vec_ty.lane_bits() as usize / 8;
                let mut bytes = vec![0u8; lanes * width];
                for (k, chunk) in bytes.chunks_mut(width).enumerate() {
                    chunk[..8.min(width)]
                        .copy_from_slice(&(k as u64).to_le_bytes()[..8.min(width)]);
                }
                let handle = self
                    .builder
                    .func
                    .dfg
                    .constants
                    .insert(cranelift_codegen::ir::ConstantData::from(bytes.as_slice()));
                self.builder.ins().vconst(vec_ty, handle)
            }
            Op::Widen { v, half, kind } => self.lower_widen(*v, *half, *kind),
            Op::Narrow { lo, hi, kind } => self.lower_narrow(*lo, *hi, *kind),
            Op::Shuffle { a, b, mask } => self.lower_shuffle(*a, *b, mask),
            // `insertlane` exists for every 128-bit lane type the verifier admits.
            Op::InsertLane(v, s, k) => {
                let x = self.val(*v);
                let y = self.val(*s);
                self.builder.ins().insertlane(x, y, *k as u8)
            }
            // Fused multiply-add: Cranelift `fma(a, b, c)` is `a*b + c` with one rounding, lowering
            // to a hardware `vfmadd` (scalar or 128-bit vector) on FMA3 hosts.
            Op::Fma(a, b, c) => {
                let av = self.val(*a);
                let bv = self.val(*b);
                let cvv = self.val(*c);
                self.builder.ins().fma(av, bv, cvv)
            }
            // Hardware square root (scalar or 128-bit vector); the interpreter mirrors it with
            // `f32`/`f64::sqrt`, so the two agree bit-for-bit.
            Op::Sqrt(v) => {
                let x = self.val(*v);
                self.builder.ins().sqrt(x)
            }
            // Hardware round-to-integral (scalar or 128-bit vector); the interpreter mirrors each mode
            // with the matching `f32`/`f64` method (`Nearest` ⇒ `round_ties_even`), so the two agree.
            Op::Round(mode, v) => {
                let x = self.val(*v);
                match mode {
                    wukong_mir::RoundMode::Nearest => self.builder.ins().nearest(x),
                    wukong_mir::RoundMode::Floor => self.builder.ins().floor(x),
                    wukong_mir::RoundMode::Ceil => self.builder.ins().ceil(x),
                    wukong_mir::RoundMode::Trunc => self.builder.ins().trunc(x),
                }
            }
            // One scalar integer instruction each (x64: `popcnt`/`lzcnt`/`tzcnt`/`bswap`, or the
            // backend's fallback sequence where the CPU lacks the extension). The result has the
            // operand's type; the verifier admits only i8..i64 scalars, and no i8 `bswap`.
            Op::IntUnary(op, v) => {
                let x = self.val(*v);
                match op {
                    wukong_mir::IntUnOp::Popcnt => self.builder.ins().popcnt(x),
                    wukong_mir::IntUnOp::Clz => self.builder.ins().clz(x),
                    wukong_mir::IntUnOp::Ctz => self.builder.ins().ctz(x),
                    wukong_mir::IntUnOp::Bswap => self.builder.ins().bswap(x),
                }
            }
        };
        if let Some(r) = res {
            // Normalise the result to its declared type so `vmap[r]` always has the MIR type's
            // Cranelift type: `i1` -> low bit, and narrow floats rounded to width — both matching
            // the interpreter's per-result normalisation.
            let rty = self.func.value_type(r);
            let cv = if matches!(rty, MirType::I1) {
                self.builder.ins().band_imm(cv, 1)
            } else if rty.is_float() {
                let want = cl_type(rty, self.ptr_ty).unwrap_or(types::F64);
                self.coerce_float(cv, want)
            } else {
                cv
            };
            self.set(r, cv);
        }
    }

    fn lower_bin(&mut self, op: BinOp, l: ValueId, r: ValueId, res_ty: &MirType) -> Value {
        use BinOp::*;
        // Float ops compute in their result type with operands coerced to it (the front-end's
        // float literals are loosely typed); this mirrors the interpreter, keeping the two exact.
        if op.is_float() {
            let target = cl_type(res_ty, self.ptr_ty).unwrap_or(types::F64);
            let a0 = self.val(l);
            let b0 = self.val(r);
            let a = self.coerce_float(a0, target);
            let b = self.coerce_float(b0, target);
            return match op {
                FAdd => self.builder.ins().fadd(a, b),
                FSub => self.builder.ins().fsub(a, b),
                FMul => self.builder.ins().fmul(a, b),
                FDiv => self.builder.ins().fdiv(a, b),
                // True fmod via a runtime call (Rust `%`), bit-identical to the interpreter. The old
                // `a - trunc(a/b)*b` identity drifts once `a/b` exceeds the mantissa's integer range
                // (e.g. `1e18 % 3` gave 0 instead of 1). y == 0 yields NaN in both, as before.
                FRem => {
                    let name = if target == types::F32 {
                        RT_FMOD_F32
                    } else {
                        RT_FMOD_F64
                    };
                    let fref = self.runtime_ref(name);
                    let call = self.builder.ins().call(fref, &[a, b]);
                    self.builder.inst_results(call)[0]
                }
                _ => unreachable!(),
            };
        }
        let (a, b) = (self.val(l), self.val(r));
        if *res_ty == MirType::I1 && matches!(op, Shl | LShr | AShr) {
            // MIR masks shift counts by the declared width minus one. Every I1 shift
            // therefore shifts by zero, although its native register is an I8.
            return a;
        }
        match op {
            Add => self.builder.ins().iadd(a, b),
            Sub => self.builder.ins().isub(a, b),
            Mul => self.builder.ins().imul(a, b),
            FAdd | FSub | FMul | FDiv | FRem => unreachable!("handled above"),
            And => self.builder.ins().band(a, b),
            Or => self.builder.ins().bor(a, b),
            Xor => self.builder.ins().bxor(a, b),
            Shl => self.builder.ins().ishl(a, b),
            LShr => self.builder.ins().ushr(a, b),
            AShr => self.builder.ins().sshr(a, b),
            // x64 has no 128-bit divide; the runtime entry carries the zero/overflow guards below
            // itself (`int128.rs`).
            SDiv | UDiv | SRem | URem if self.dfg_ty(a) == types::I128 => {
                self.i128_divrem(op, a, b)
            }
            // Guard the divisor so a zero yields 0 instead of trapping, matching the interpreter.
            // Signed div/rem additionally trap the hardware on `INT_MIN / -1` (2's-complement
            // overflow), so fold a -1 divisor into the guard and patch the result to the
            // interpreter's wrapping value (`INT_MIN / -1 == INT_MIN`, `x % -1 == 0`). Unsigned
            // division never overflows, so -1 there is just an ordinary large divisor.
            SDiv | UDiv | SRem | URem => {
                let ty = self.dfg_ty(a);
                let zero = self.builder.ins().iconst(ty, 0);
                let one = self.builder.ins().iconst(ty, 1);
                let is_zero = self.builder.ins().icmp(IntCC::Equal, b, zero);
                let is_neg1 = if matches!(op, SDiv | SRem) {
                    let neg1 = self.builder.ins().iconst(ty, -1);
                    Some(self.builder.ins().icmp(IntCC::Equal, b, neg1))
                } else {
                    None
                };
                let danger = match is_neg1 {
                    Some(n) => self.builder.ins().bor(is_zero, n),
                    None => is_zero,
                };
                let safe = self.builder.ins().select(danger, one, b);
                let q = match op {
                    SDiv => self.builder.ins().sdiv(a, safe),
                    UDiv => self.builder.ins().udiv(a, safe),
                    SRem => self.builder.ins().srem(a, safe),
                    URem => self.builder.ins().urem(a, safe),
                    _ => unreachable!(),
                };
                // `INT_MIN / -1`: the divisor was swapped to 1, so sdiv produced `a` — negate it to
                // get `-a` (== INT_MIN for INT_MIN). `x % -1 == 0` already falls out of `srem(a, 1)`.
                let q = match (op, is_neg1) {
                    (SDiv, Some(n)) => {
                        let nega = self.builder.ins().ineg(a);
                        self.builder.ins().select(n, nega, q)
                    }
                    _ => q,
                };
                self.builder.ins().select(is_zero, zero, q)
            }
        }
    }

    fn lower_cast(&mut self, kind: CastKind, v: ValueId, to: &MirType) -> Value {
        let x = self.val(v);
        let from_ty = self.dfg_ty(x);
        let to_ty = cl_type(to, self.ptr_ty).unwrap_or(self.ptr_ty);
        use CastKind::*;
        if matches!(to, MirType::I1) && matches!(kind, FpToSi | FpToUi) {
            // MIR's boolean fp-cast contract saturates at i32/u32 width, then keeps
            // the low bit. I1 uses an I8 register, but clamping at I8/U8 would change
            // parity above 127/255 (128.0 and 256.0 must both become false).
            let wide = self.fcvt_to_int_sat(x, types::I32, kind == FpToSi);
            // lower_inst applies the ordinary I1 low-bit normalization afterwards.
            return self.builder.ins().ireduce(to_ty, wide);
        }
        match kind {
            SExt => self.resize_int(x, from_ty, to_ty, true),
            ZExt | Trunc => self.resize_int(x, from_ty, to_ty, false),
            // x64 converts no 128-bit integer to or from a float (`int128.rs`).
            SiToFp | UiToFp if from_ty == types::I128 => {
                self.i128_to_float(x, to_ty, kind == SiToFp)
            }
            FpToSi | FpToUi if to_ty == types::I128 => self.float_to_i128(x, kind == FpToSi),
            FpToSi => self.fcvt_to_int_sat(x, to_ty, true),
            FpToUi => self.fcvt_to_int_sat(x, to_ty, false),
            SiToFp => self.builder.ins().fcvt_from_sint(to_ty, x),
            UiToFp => self.builder.ins().fcvt_from_uint(to_ty, x),
            // The vectorizer's `<4 x bf16>` casts: a lane-wise `FpExt` to `f32` and `FpTrunc` to
            // bf16 both round each lane to the bf16 grid, exactly as the scalar forms below do.
            FpExt | FpTrunc if is_bf16_x4(self.ty_of(v)) || is_bf16_x4(to) => {
                debug_assert!(
                    matches!(
                        (kind, self.ty_of(v).lane_type(), to.lane_type()),
                        (FpExt, MirType::BF16, MirType::F32)
                            | (FpTrunc, MirType::F32 | MirType::BF16, MirType::BF16)
                    ),
                    "a bf16 vector cast the vectorizer never builds"
                );
                self.round_to_bf16_x4(x)
            }
            FpExt => {
                // A bf16/f16 source must round to its grid before widening. bf16/f16 share f32's
                // register, so the per-store rounding is fragile — an optimizer can forward/promote a
                // bf16 op result past its store, leaving an unrounded f32, and this widen would then be
                // a no-op passing the wrong value (so -O2 disagreed with -O0). Round at the observation
                // boundary too (the interpreter's `apply_cast` FpExt arm does the same).
                let xr = match self.ty_of(v) {
                    MirType::BF16 => self.round_to_bf16(x),
                    MirType::F16 => self.round_to_f16(x),
                    _ => x,
                };
                let xr_ty = self.dfg_ty(xr);
                if to_ty == xr_ty {
                    xr
                } else {
                    self.builder.ins().fpromote(to_ty, xr)
                }
            }
            FpTrunc => {
                if matches!(to, MirType::BF16) {
                    // bf16's register type is f32; demote an f64 source first, then round to bf16.
                    let f = if from_ty == types::F64 {
                        self.builder.ins().fdemote(types::F32, x)
                    } else {
                        x
                    };
                    self.round_to_bf16(f)
                } else if matches!(to, MirType::F16) {
                    // f16's register type is f32; demote an f64 source first, then round to f16.
                    let f = if from_ty == types::F64 {
                        self.builder.ins().fdemote(types::F32, x)
                    } else {
                        x
                    };
                    self.round_to_f16(f)
                } else if to_ty == from_ty {
                    x
                } else {
                    self.builder.ins().fdemote(to_ty, x)
                }
            }
            Bitcast => {
                if to_ty == from_ty {
                    x
                } else {
                    self.builder.ins().bitcast(to_ty, MemFlags::new(), x)
                }
            }
            IntToPtr | PtrToInt => self.resize_int(x, from_ty, to_ty, false),
        }
    }

    /// Saturating float→int conversion that also supports the narrow result types `i8`/`i16`, which
    /// Cranelift's `fcvt_to_{sint,uint}_sat` cannot target directly on x64 (the emitter hits
    /// `unreachable!`). For a narrow target, convert to `i32` saturating (NaN→0, out-of-range
    /// clamped to the i32 range), then clamp to the *narrow* type's range and `ireduce`. This
    /// reproduces Rust `as` / the interpreter's saturating cast bit-for-bit — including out-of-range
    /// and NaN inputs (`300.0 as u8 == 255`, `-1.0 as u8 == 0`, `NaN as i8 == 0`) — keeping the
    /// differential gate exact. A `>= 32`-bit target uses the direct instruction unchanged.
    fn fcvt_to_int_sat(&mut self, x: Value, to_ty: types::Type, signed: bool) -> Value {
        if to_ty.bits() >= 32 {
            return if signed {
                self.builder.ins().fcvt_to_sint_sat(to_ty, x)
            } else {
                self.builder.ins().fcvt_to_uint_sat(to_ty, x)
            };
        }
        if signed {
            let wide = self.builder.ins().fcvt_to_sint_sat(types::I32, x);
            let (lo, hi) = if to_ty.bits() == 8 {
                (-128i64, 127i64)
            } else {
                (-32768i64, 32767i64)
            };
            let hic = self.builder.ins().iconst(types::I32, hi);
            let loc = self.builder.ins().iconst(types::I32, lo);
            let capped = self.builder.ins().smin(wide, hic);
            let clamped = self.builder.ins().smax(capped, loc);
            self.builder.ins().ireduce(to_ty, clamped)
        } else {
            let wide = self.builder.ins().fcvt_to_uint_sat(types::I32, x);
            let hi = if to_ty.bits() == 8 { 255i64 } else { 65535i64 };
            let hic = self.builder.ins().iconst(types::I32, hi);
            // The lower bound is already 0 from the unsigned saturation; only the upper bound needs
            // clamping (unsigned min, so a saturated `u32::MAX` reads as larger than the narrow max).
            let clamped = self.builder.ins().umin(wide, hic);
            self.builder.ins().ireduce(to_ty, clamped)
        }
    }

    /// Reinterpret a 128-bit vector as `ty` (the same bits). Cranelift requires a byte order on a
    /// `bitcast` that changes the lane count; every vector here lives in a little-endian register.
    fn vec_bitcast(&mut self, ty: Type, x: Value) -> Value {
        if self.dfg_ty(x) == ty {
            return x;
        }
        let flags = MemFlags::new().with_endianness(cranelift_codegen::ir::Endianness::Little);
        self.builder.ins().bitcast(ty, flags, x)
    }

    /// A byte shuffle of two 128-bit vectors: result byte `i` is byte `bytes[i]` of `a ++ b`.
    fn byte_shuffle(&mut self, a: Value, b: Value, bytes: &[u8], ty: Type) -> Value {
        let a8 = self.vec_bitcast(types::I8X16, a);
        let b8 = self.vec_bitcast(types::I8X16, b);
        let imm = self
            .builder
            .func
            .dfg
            .immediates
            .push(cranelift_codegen::ir::ConstantData::from(bytes));
        let r = self.builder.ins().shuffle(a8, b8, imm);
        self.vec_bitcast(ty, r)
    }

    /// `Op::Widen`: one half of the source lanes at twice the width. `f32 -> f64` has only a
    /// low-half instruction (`cvtps2pd`), so the high half is first moved down with a `pshufd`-shaped
    /// byte shuffle — one Cranelift lowers without SSSE3.
    fn lower_widen(&mut self, v: ValueId, half: wukong_mir::Half, kind: CastKind) -> Value {
        use wukong_mir::Half;
        let x = self.val(v);
        match (kind, half) {
            (CastKind::SExt, Half::Lo) => self.builder.ins().swiden_low(x),
            (CastKind::SExt, Half::Hi) => self.builder.ins().swiden_high(x),
            (CastKind::ZExt, Half::Lo) => self.builder.ins().uwiden_low(x),
            (CastKind::ZExt, Half::Hi) => self.builder.ins().uwiden_high(x),
            // `FpExt`, `f32x4 -> f64x2`: the only other kind the verifier admits.
            (_, Half::Lo) => self.builder.ins().fvpromote_low(x),
            (_, Half::Hi) => {
                let bytes: Vec<u8> = (8..16).chain(8..16).collect();
                let hi = self.byte_shuffle(x, x, &bytes, types::F32X4);
                self.builder.ins().fvpromote_low(hi)
            }
        }
    }

    /// `Op::Narrow`. The saturating kinds are `snarrow`/`unarrow` (`packss`/`packus`). Truncation
    /// first masks every lane to its low half-width bits — which puts it in range, so the unsigned
    /// saturating pack then keeps exactly those bits — except at 64-bit lanes, where no pack exists
    /// and the low dwords are picked out with a `shufps`-shaped shuffle. `f64 -> f32` demotes each
    /// operand (`cvtpd2ps`, high half zeroed) and joins the two low halves (`punpcklqdq`).
    fn lower_narrow(&mut self, lo: ValueId, hi: ValueId, kind: wukong_mir::NarrowKind) -> Value {
        use wukong_mir::NarrowKind;
        let (l, h) = (self.val(lo), self.val(hi));
        let wide = self.dfg_ty(l);
        match kind {
            NarrowKind::SatSigned => self.builder.ins().snarrow(l, h),
            NarrowKind::SatUnsigned => self.builder.ins().unarrow(l, h),
            NarrowKind::Trunc if wide == types::I64X2 => {
                let bytes: [u8; 16] = [0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27];
                self.byte_shuffle(l, h, &bytes, types::I32X4)
            }
            NarrowKind::Trunc => {
                let lane = wide.lane_type();
                let low_bits: i64 = if lane == types::I16 { 0xFF } else { 0xFFFF };
                let m = self.builder.ins().iconst(lane, low_bits);
                let mask = self.builder.ins().splat(wide, m);
                let lm = self.builder.ins().band(l, mask);
                let hm = self.builder.ins().band(h, mask);
                self.builder.ins().unarrow(lm, hm)
            }
            NarrowKind::FpTrunc => {
                let ld = self.builder.ins().fvdemote(l);
                let hd = self.builder.ins().fvdemote(h);
                let bytes: Vec<u8> = (0..8).chain(16..24).collect();
                self.byte_shuffle(ld, hd, &bytes, types::F32X4)
            }
        }
    }

    /// `Op::Shuffle`: lane `k` is lane `mask[k]` of `a ++ b`, as a byte shuffle over the lane width.
    /// Without SSSE3 Cranelift lowers a general byte shuffle to a `pshufb` *libcall* that neither
    /// the JIT nor a plain link provides, so a baseline (`--target-cpu=x86-64`) build moves the
    /// lanes one at a time instead — slower, and the same bits.
    fn lower_shuffle(&mut self, a: ValueId, b: ValueId, mask: &[u8]) -> Value {
        let (av, bv) = (self.val(a), self.val(b));
        let vty = self.dfg_ty(av);
        let n = vty.lane_count();
        if !self.decls.ssse3 {
            let mut acc = av;
            for (k, &m) in mask.iter().enumerate() {
                let (src, j) = if u32::from(m) < n {
                    (av, m)
                } else {
                    (bv, m - n as u8)
                };
                let x = self.builder.ins().extractlane(src, j);
                acc = self.builder.ins().insertlane(acc, x, k as u8);
            }
            return acc;
        }
        let w = (vty.lane_bits() / 8) as u8;
        let mut bytes = Vec::with_capacity(16);
        for &m in mask {
            for j in 0..w {
                bytes.push(m * w + j);
            }
        }
        self.byte_shuffle(av, bv, &bytes, vty)
    }

    /// Round an `f32` to bf16 precision and back to `f32`, emitting the *identical* integer
    /// arithmetic as `wukong_runtime::round_bf16` so the native backend and the interpreter agree
    /// bit-for-bit. Returns the rounded `f32` (its low 16 mantissa bits are zero).
    fn round_to_bf16(&mut self, x: Value) -> Value {
        let bits = self.builder.ins().bitcast(types::I32, MemFlags::new(), x);
        // non-NaN, round to nearest even: (bits + 0x7fff + ((bits>>16)&1)) & 0xffff0000
        let shr = self.builder.ins().ushr_imm(bits, 16);
        let lsb = self.builder.ins().band_imm(shr, 1);
        let bias = self.builder.ins().iadd_imm(lsb, 0x7fff);
        let summed = self.builder.ins().iadd(bits, bias);
        let nonnan = self
            .builder
            .ins()
            .band_imm(summed, 0xffff_0000u32 as i32 as i64);
        // NaN stays NaN (quiet): (bits & 0xffff0000) | 0x00400000
        let masked = self
            .builder
            .ins()
            .band_imm(bits, 0xffff_0000u32 as i32 as i64);
        let nanres = self.builder.ins().bor_imm(masked, 0x0040_0000);
        let absb = self.builder.ins().band_imm(bits, 0x7fff_ffff);
        let isnan = self
            .builder
            .ins()
            .icmp_imm(IntCC::UnsignedGreaterThan, absb, 0x7f80_0000);
        let resbits = self.builder.ins().select(isnan, nanres, nonnan);
        self.builder
            .ins()
            .bitcast(types::F32, MemFlags::new(), resbits)
    }

    /// [`Self::round_to_bf16`] on each lane of an `f32x4`: the same integer round-to-nearest-even on
    /// the bits, and the same quiet-NaN rule (keep the top half, set the quiet bit), with a lane
    /// mask from an unordered compare where the scalar form tests `|bits| > 0x7f800000`.
    fn round_to_bf16_x4(&mut self, x: Value) -> Value {
        let splat = |me: &mut Self, k: u32| {
            let c = me.builder.ins().iconst(types::I32, i64::from(k as i32));
            me.builder.ins().splat(types::I32X4, c)
        };
        let bits = self.vec_bitcast(types::I32X4, x);
        let shr = self.builder.ins().ushr_imm(bits, 16);
        let one = splat(self, 1);
        let lsb = self.builder.ins().band(shr, one);
        let half = splat(self, 0x7fff);
        let bias = self.builder.ins().iadd(lsb, half);
        let summed = self.builder.ins().iadd(bits, bias);
        let top = splat(self, 0xffff_0000);
        let nonnan = self.builder.ins().band(summed, top);
        let masked = self.builder.ins().band(bits, top);
        let quiet = splat(self, 0x0040_0000);
        let nanres = self.builder.ins().bor(masked, quiet);
        let isnan = self.builder.ins().fcmp(FloatCC::Unordered, x, x);
        let isnan = self.vec_bitcast(types::I32X4, isnan);
        let resbits = self.builder.ins().bitselect(isnan, nanres, nonnan);
        self.vec_bitcast(types::F32X4, resbits)
    }

    /// Round an `f32` to f16 precision and back to `f32`, by calling the runtime shims
    /// (`wukong_f32_to_f16_bits` then `wukong_f16_bits_to_f32`). f16's exponent/mantissa layout has
    /// no cheap inline round like bf16, so a call to the *identical* `half`-crate conversion the
    /// interpreter uses keeps native == interp bit-for-bit. Returns the f16-rounded `f32`.
    fn round_to_f16(&mut self, x: Value) -> Value {
        let pack = self.runtime_ref(RT_F32_TO_F16);
        let c1 = self.builder.ins().call(pack, &[x]);
        let bits = self.builder.inst_results(c1)[0]; // i32, low 16 = f16 bits
        let unpack = self.runtime_ref(RT_F16_TO_F32);
        let c2 = self.builder.ins().call(unpack, &[bits]);
        self.builder.inst_results(c2)[0]
    }

    /// Sign- or zero-extend, truncate, or pass through an integer to a target width.
    fn resize_int(&mut self, x: Value, from: Type, to: Type, signed: bool) -> Value {
        match to.bits().cmp(&from.bits()) {
            std::cmp::Ordering::Equal => x,
            std::cmp::Ordering::Greater => {
                if signed {
                    self.builder.ins().sextend(to, x)
                } else {
                    self.builder.ins().uextend(to, x)
                }
            }
            std::cmp::Ordering::Less => self.builder.ins().ireduce(to, x),
        }
    }

    /// Promote/demote a float value to a target float width (no-op if already that width).
    fn coerce_float(&mut self, x: Value, to: Type) -> Value {
        let from = self.dfg_ty(x);
        if from == to || !from.is_float() || !to.is_float() {
            return x;
        }
        if to.bits() > from.bits() {
            self.builder.ins().fpromote(to, x)
        } else {
            self.builder.ins().fdemote(to, x)
        }
    }

    /// Coerce an index value to the pointer-width integer used for address arithmetic.
    fn ptr_int(&mut self, index: ValueId) -> Value {
        let x = self.val(index);
        let from = self.dfg_ty(x);
        self.resize_int(x, from, self.ptr_ty, true)
    }

    fn lower_call(&mut self, func: Symbol, args: &[ValueId]) -> Option<Value> {
        if let Some(&fref) = self.func_refs.get(&func) {
            self.call_args.clear();
            self.call_args.reserve(args.len());
            for &arg in args {
                let value = self.val(arg);
                self.call_args.push(value);
            }
            let call = self.builder.ins().call(fref, self.call_args);
            let result = self.builder.inst_results(call).first().copied();
            if self.decls.asserting.contains(&func) {
                let status = self.runtime_ref(RT_ASSERT_FAILED);
                let check = self.builder.ins().call(status, &[]);
                let failed = self.builder.inst_results(check)[0];
                self.return_on_assert_failure(failed, None);
            }
            return result;
        }
        let name = self.interner.resolve(func);
        // The parallel-for runtime entry: wukong_parallel_for(n, body_ptr, env_ptr).
        if name == RT_PARALLEL_FOR && args.len() == 3 {
            let n = self.coerce_to_i64(args[0]);
            let body = self.val(args[1]);
            let env = self.val(args[2]);
            let fref = self.runtime_ref(RT_PARALLEL_FOR);
            self.builder.ins().call(fref, &[n, body, env]);
            // A failure-capable region body hands its failure back through the JIT's
            // `rt_parallel_for`, so a failure-capable caller polls here exactly as it does after
            // an asserting user call. (Polling where the body cannot fail reads a clear flag.)
            if self.decls.asserting.contains(&self.func.name) {
                let status = self.runtime_ref(RT_ASSERT_FAILED);
                let check = self.builder.ins().call(status, &[]);
                let failed = self.builder.inst_results(check)[0];
                self.return_on_assert_failure(failed, None);
            }
            return None;
        }
        // The GEMM microkernel: wukong_sgemm(a, b, c, m, k, n, beta) and its parallel / transposed
        // (nn.Linear, C = A·Bᵀ) variants — all share the (ptr,ptr,ptr,i64,i64,i64,i64) signature.
        if matches!(
            name,
            RT_SGEMM
                | RT_SGEMM_PARALLEL
                | RT_SGEMM_NT
                | RT_SGEMM_NT_PARALLEL
                | RT_SGEMM_TN
                | RT_SGEMM_TN_PARALLEL
                | RT_SGEMM_BF16_NT
                | RT_SGEMM_BF16_NT_PARALLEL
                | RT_SGEMM_F16_NT
                | RT_SGEMM_F16_NT_PARALLEL
                | RT_SGEMM_BF16_TN
                | RT_SGEMM_BF16_TN_PARALLEL
                | RT_SGEMM_F16_TN
                | RT_SGEMM_F16_TN_PARALLEL
        ) && args.len() == 7
        {
            let a = self.val(args[0]);
            let b = self.val(args[1]);
            let c = self.val(args[2]);
            let m = self.coerce_to_i64(args[3]);
            let k = self.coerce_to_i64(args[4]);
            let n = self.coerce_to_i64(args[5]);
            let beta = self.coerce_to_i64(args[6]);
            let fref = self.runtime_ref(name);
            self.builder.ins().call(fref, &[a, b, c, m, k, n, beta]);
            return None;
        }
        // The fused-epilogue Linear: wukong_sgemm_nt_epi[_parallel](a, b, c, m, k, n, beta, bias, act)
        // — three pointers, four i64 (m,k,n,beta), a bias pointer, and an i64 activation code. The
        // `@parallel` variant and the bf16/f16 half-input twins (a/b are 2-byte-element pointers; the
        // kernel widens losslessly) all share the identical 9-arg signature, so just route by `name`.
        if matches!(
            name,
            RT_SGEMM_NT_EPI
                | RT_SGEMM_NT_EPI_PAR
                | RT_SGEMM_BF16_NT_EPI
                | RT_SGEMM_BF16_NT_EPI_PAR
                | RT_SGEMM_F16_NT_EPI
                | RT_SGEMM_F16_NT_EPI_PAR
        ) && args.len() == 9
        {
            let a = self.val(args[0]);
            let b = self.val(args[1]);
            let c = self.val(args[2]);
            let m = self.coerce_to_i64(args[3]);
            let k = self.coerce_to_i64(args[4]);
            let n = self.coerce_to_i64(args[5]);
            let beta = self.coerce_to_i64(args[6]);
            let bias = self.val(args[7]);
            let act = self.coerce_to_i64(args[8]);
            let fref = self.runtime_ref(name);
            self.builder
                .ins()
                .call(fref, &[a, b, c, m, k, n, beta, bias, act]);
            return None;
        }
        // GEMV: wukong_sgemv[_parallel](a, x, y, m, n) — three pointers and two i64. Void.
        if matches!(name, RT_SGEMV | RT_SGEMV_PAR) && args.len() == 5 {
            let a = self.val(args[0]);
            let x = self.val(args[1]);
            let y = self.val(args[2]);
            let m = self.coerce_to_i64(args[3]);
            let n = self.coerce_to_i64(args[4]);
            let fref = self.runtime_ref(name);
            self.builder.ins().call(fref, &[a, x, y, m, n]);
            return None;
        }
        // α-scaled GEMV: wukong_sgemv_alpha[_parallel](a, x, y, m, n, alpha), and the vector·matrix
        // wukong_sgevm_f32[_parallel](w, a, out, rows, cols, alpha) — the same three pointers, two
        // i64, and one trailing f32 scale by value (like the α-scaled GEMM's alpha). Route by name. Void.
        if matches!(
            name,
            RT_SGEMV_ALPHA | RT_SGEMV_ALPHA_PAR | RT_SGEVM | RT_SGEVM_PAR
        ) && args.len() == 6
        {
            let a = self.val(args[0]);
            let x = self.val(args[1]);
            let y = self.val(args[2]);
            let m = self.coerce_to_i64(args[3]);
            let n = self.coerce_to_i64(args[4]);
            let alpha = self.val(args[5]);
            let fref = self.runtime_ref(name);
            self.builder.ins().call(fref, &[a, x, y, m, n, alpha]);
            return None;
        }
        // α-scaled Linear: wukong_sgemm_nt_alpha[_parallel](a, b, c, m, k, n, beta, alpha) — three
        // pointers, four i64, and one f32 scalar `alpha` (passed by value, like the i8 dequant scale).
        if matches!(name, RT_SGEMM_NT_ALPHA | RT_SGEMM_NT_ALPHA_PAR) && args.len() == 8 {
            let a = self.val(args[0]);
            let b = self.val(args[1]);
            let c = self.val(args[2]);
            let m = self.coerce_to_i64(args[3]);
            let k = self.coerce_to_i64(args[4]);
            let n = self.coerce_to_i64(args[5]);
            let beta = self.coerce_to_i64(args[6]);
            let alpha = self.val(args[7]);
            let fref = self.runtime_ref(name);
            self.builder
                .ins()
                .call(fref, &[a, b, c, m, k, n, beta, alpha]);
            return None;
        }
        if name == RT_TRIG_F32 && args.len() == 2 {
            let x = self.val(args[0]);
            let op = self.coerce_to_i64(args[1]);
            let fref = self.runtime_ref(RT_TRIG_F32);
            let call = self.builder.ins().call(fref, &[x, op]);
            return Some(self.builder.inst_results(call)[0]);
        }
        // The f64 transcendental library: wukong_rt_math_f64(x: f64, op: i64) -> f64 and its
        // two-argument twin — value-returning, the same pure-Rust function the interpreter calls.
        if name == RT_MATH_F64 && args.len() == 2 {
            let x = self.val(args[0]);
            let op = self.coerce_to_i64(args[1]);
            let fref = self.runtime_ref(RT_MATH_F64);
            let call = self.builder.ins().call(fref, &[x, op]);
            return Some(self.builder.inst_results(call)[0]);
        }
        if name == RT_MATH2_F64 && args.len() == 3 {
            let a = self.val(args[0]);
            let b = self.val(args[1]);
            let op = self.coerce_to_i64(args[2]);
            let fref = self.runtime_ref(RT_MATH2_F64);
            let call = self.builder.ins().call(fref, &[a, b, op]);
            return Some(self.builder.inst_results(call)[0]);
        }
        // The vectorized elementwise transcendental: wukong_vmath_f32(x, out, n, op) — two pointers
        // and two i64 (element count, op code). The 256-bit AVX2 kernel an `out[i]=f(x[i])` loop
        // lowers to.
        // Same 4-arg shape for the bf16/f16-input twins (x is a 2-byte-element pointer; the kernel
        // widens losslessly). Identical signature, so just route by name.
        if (name == RT_VMATH
            || name == RT_VMATH_PARALLEL
            || name == RT_VMATH_BF16
            || name == RT_VMATH_F16
            || name == RT_VMATH_BF16_OUT
            || name == RT_VMATH_F16_OUT)
            && args.len() == 4
        {
            let x = self.val(args[0]);
            let out = self.val(args[1]);
            let n = self.coerce_to_i64(args[2]);
            let op = self.coerce_to_i64(args[3]);
            let fref = self.runtime_ref(name);
            self.builder.ins().call(fref, &[x, out, n, op]);
            return None;
        }
        // Batched log-sum-exp: wukong_logsumexp_f32[_parallel](x, out, rows, cols) — two pointers +
        // two i64, the same vmath shape; route by name. Void. Row-entropy and the per-row arg-reductions
        // (`wukong_rowarg{max,min}_i32`, whose `out` is an i32 buffer — a pointer rides the slot either
        // way) share the exact `(ptr, ptr, i64, i64)` shape.
        if matches!(
            name,
            RT_LOGSUMEXP
                | RT_LOGSUMEXP_PAR
                | RT_ENTROPY
                | RT_ENTROPY_PAR
                | RT_ROWARGMAX
                | RT_ROWARGMAX_PAR
                | RT_ROWARGMIN
                | RT_ROWARGMIN_PAR
                | RT_COLARGMAX
                | RT_COLARGMAX_PAR
                | RT_COLARGMIN
                | RT_COLARGMIN_PAR
                | RT_CUMSUM
                | RT_CUMSUM_PAR
                | RT_CUMMAX
                | RT_CUMMAX_PAR
                | RT_CUMMIN
                | RT_CUMMIN_PAR
                | RT_CUMPROD
                | RT_CUMPROD_PAR
        ) && args.len() == 4
        {
            let x = self.val(args[0]);
            let out = self.val(args[1]);
            let rows = self.coerce_to_i64(args[2]);
            let cols = self.coerce_to_i64(args[3]);
            let fref = self.runtime_ref(name);
            self.builder.ins().call(fref, &[x, out, rows, cols]);
            return None;
        }
        // KL divergence / soft-label cross-entropy: (p|x, q, out, rows, cols) — three pointers + two
        // i64, the same vmath2 shape; route by name. Void.
        if matches!(name, RT_KLDIV | RT_KLDIV_PAR | RT_KD_LOSS | RT_KD_LOSS_PAR) && args.len() == 5
        {
            let a = self.val(args[0]);
            let b = self.val(args[1]);
            let out = self.val(args[2]);
            let rows = self.coerce_to_i64(args[3]);
            let cols = self.coerce_to_i64(args[4]);
            let fref = self.runtime_ref(name);
            self.builder.ins().call(fref, &[a, b, out, rows, cols]);
            return None;
        }
        // The cache-blocked transpose: wukong_transpose_f32[_parallel](src, dst, rows, cols) — two
        // pointers and two i64. Same (ptr, ptr, i64, i64) signature as vmath; route by name.
        if matches!(
            name,
            RT_TRANSPOSE | RT_TRANSPOSE_PAR | RT_TRANSPOSE_U16 | RT_TRANSPOSE_U16_PAR
        ) && args.len() == 4
        {
            let src = self.val(args[0]);
            let dst = self.val(args[1]);
            let rows = self.coerce_to_i64(args[2]);
            let cols = self.coerce_to_i64(args[3]);
            let fref = self.runtime_ref(name);
            self.builder.ins().call(fref, &[src, dst, rows, cols]);
            return None;
        }
        // 2D pooling: wukong_{max,avg}pool2d_f32[_parallel](x, out, channels, h, w, kh, kw, sh, sw) —
        // two pointers and seven i64. Route by name to the declared import.
        if matches!(
            name,
            RT_MAXPOOL2D | RT_MAXPOOL2D_PAR | RT_AVGPOOL2D | RT_AVGPOOL2D_PAR
        ) && args.len() == 9
        {
            let x = self.val(args[0]);
            let out = self.val(args[1]);
            let mut call_args = vec![x, out];
            for a in &args[2..] {
                call_args.push(self.coerce_to_i64(*a));
            }
            let fref = self.runtime_ref(name);
            self.builder.ins().call(fref, &call_args);
            return None;
        }
        // The SIMD column reductions: wukong_col{sum,max,min,maxabs,mean,sumsq,l2,rms}_f32[_parallel]
        // (x, out, rows, cols) — two pointers and two i64. Same (ptr, ptr, i64, i64) signature; by name.
        if matches!(
            name,
            RT_COLSUM
                | RT_COLSUM_PAR
                | RT_COLMAX
                | RT_COLMAX_PAR
                | RT_COLMIN
                | RT_COLMIN_PAR
                | RT_COLMAXABS
                | RT_COLMAXABS_PAR
                | RT_COLMEAN
                | RT_COLMEAN_PAR
                | RT_COLSUMSQ
                | RT_COLSUMSQ_PAR
                | RT_COLL2
                | RT_COLL2_PAR
                | RT_COLRMS
                | RT_COLRMS_PAR
        ) && args.len() == 4
        {
            let x = self.val(args[0]);
            let out = self.val(args[1]);
            let rows = self.coerce_to_i64(args[2]);
            let cols = self.coerce_to_i64(args[3]);
            let fref = self.runtime_ref(name);
            self.builder.ins().call(fref, &[x, out, rows, cols]);
            return None;
        }
        // The two-input transcendental: wukong_vmath2_f32(x, y, out, n, op) — three pointers, two i64
        // (element count, op code). The 256-bit AVX2 kernel an `out[i]=pow/atan2/hypot(x[i],y[i])`
        // loop lowers to.
        if name == RT_VMATH2 && args.len() == 5 {
            let x = self.val(args[0]);
            let y = self.val(args[1]);
            let out = self.val(args[2]);
            let n = self.coerce_to_i64(args[3]);
            let op = self.coerce_to_i64(args[4]);
            let fref = self.runtime_ref(RT_VMATH2);
            self.builder.ins().call(fref, &[x, y, out, n, op]);
            return None;
        }
        // Fused softmax-backward: wukong_softmax_bwd_f32[_parallel](y, dy, dx, rows, cols) — three
        // pointers, two i64. Same (ptr, ptr, ptr, i64, i64) shape as vmath2; route by name. Void.
        if matches!(name, RT_SOFTMAX_BWD | RT_SOFTMAX_BWD_PAR) && args.len() == 5 {
            let y = self.val(args[0]);
            let dy = self.val(args[1]);
            let dx = self.val(args[2]);
            let rows = self.coerce_to_i64(args[3]);
            let cols = self.coerce_to_i64(args[4]);
            let fref = self.runtime_ref(name);
            self.builder.ins().call(fref, &[y, dy, dx, rows, cols]);
            return None;
        }

        // Linear-recurrence scan: wukong_lrscan_f32[_parallel](a, b, out, rows, cols) — three pointers,
        // two i64 (same (ptr,ptr,ptr,i64,i64) shape as softmax_bwd); route by name. Void.
        if matches!(name, RT_LRSCAN | RT_LRSCAN_PAR) && args.len() == 5 {
            let a = self.val(args[0]);
            let b = self.val(args[1]);
            let out = self.val(args[2]);
            let rows = self.coerce_to_i64(args[3]);
            let cols = self.coerce_to_i64(args[4]);
            let fref = self.runtime_ref(name);
            self.builder.ins().call(fref, &[a, b, out, rows, cols]);
            return None;
        }
        // Fused cross-entropy loss: wukong_xent_fwd_f32[_parallel](x, target, loss, rows, cols) — three
        // pointers (x f32, target i32, loss f32 — a ptr is a ptr) + two i64, the same vmath2 shape. Void.
        if matches!(name, RT_XENT | RT_XENT_PAR) && args.len() == 5 {
            let x = self.val(args[0]);
            let target = self.val(args[1]);
            let loss = self.val(args[2]);
            let rows = self.coerce_to_i64(args[3]);
            let cols = self.coerce_to_i64(args[4]);
            let fref = self.runtime_ref(name);
            self.builder
                .ins()
                .call(fref, &[x, target, loss, rows, cols]);
            return None;
        }
        // Cross-entropy backward: wukong_xent_bwd_f32[_parallel](x, target, dx, rows, cols) — same
        // vmath2 shape (3 ptr + 2 i64). Void.
        if matches!(name, RT_XENT_BWD | RT_XENT_BWD_PAR) && args.len() == 5 {
            let x = self.val(args[0]);
            let target = self.val(args[1]);
            let dx = self.val(args[2]);
            let rows = self.coerce_to_i64(args[3]);
            let cols = self.coerce_to_i64(args[4]);
            let fref = self.runtime_ref(name);
            self.builder.ins().call(fref, &[x, target, dx, rows, cols]);
            return None;
        }
        // RoPE: wukong_rope_f32[_parallel](x, inv_freq, out, rows, half) — three f32 pointers + two
        // i64, the same vmath2 shape. Void.
        if matches!(name, RT_ROPE | RT_ROPE_PAR | RT_ROPE_BWD | RT_ROPE_BWD_PAR) && args.len() == 5
        {
            let x = self.val(args[0]);
            let inv_freq = self.val(args[1]);
            let out = self.val(args[2]);
            let rows = self.coerce_to_i64(args[3]);
            let half = self.coerce_to_i64(args[4]);
            let fref = self.runtime_ref(name);
            self.builder
                .ins()
                .call(fref, &[x, inv_freq, out, rows, half]);
            return None;
        }
        // Fused RMSNorm-backward: wukong_rmsnorm_bwd_f32[_parallel](x, dy, gamma, dx, rows, cols,
        // eps_bits) — four pointers, three i64. The two per-row reductions fold 8-wide. Void.
        if matches!(
            name,
            RT_RMSNORM_BWD | RT_RMSNORM_BWD_PAR | RT_LAYERNORM_BWD | RT_LAYERNORM_BWD_PAR
        ) && args.len() == 7
        {
            let x = self.val(args[0]);
            let dy = self.val(args[1]);
            let gamma = self.val(args[2]);
            let dx = self.val(args[3]);
            let rows = self.coerce_to_i64(args[4]);
            let cols = self.coerce_to_i64(args[5]);
            let eps = self.coerce_to_i64(args[6]);
            let fref = self.runtime_ref(name);
            self.builder
                .ins()
                .call(fref, &[x, dy, gamma, dx, rows, cols, eps]);
            return None;
        }
        // The streaming affine+activation kernel: wukong_velem_f32[_parallel](x, y, out, n, a, b, c,
        // op) — three pointers, one i64 count, three f32 coefficients, one i64 op (256-bit AVX2 +
        // non-temporal stores, the saxpy/scale/add/bias/ReLU map a recognized streaming loop lowers
        // to). The `_parallel` twin has the identical signature and is bit-equal (elementwise). Void.
        if matches!(name, RT_VELEM | RT_VELEM_PARALLEL) && args.len() == 8 {
            let x = self.val(args[0]);
            let y = self.val(args[1]);
            let out = self.val(args[2]);
            let n = self.coerce_to_i64(args[3]);
            let a = self.val(args[4]);
            let b = self.val(args[5]);
            let c = self.val(args[6]);
            let op = self.coerce_to_i64(args[7]);
            let fref = self.runtime_ref(name);
            self.builder.ins().call(fref, &[x, y, out, n, a, b, c, op]);
            return None;
        }
        // The broadcast-bias kernel: wukong_bias_bcast_f32[_parallel](x, b, out, rows, cols, op) —
        // three pointers (x, b, out) and three i64 (rows, cols, activation op). Void, like velem.
        if matches!(name, RT_BIAS_BCAST | RT_BIAS_BCAST_PAR) && args.len() == 6 {
            let x = self.val(args[0]);
            let b = self.val(args[1]);
            let out = self.val(args[2]);
            let rows = self.coerce_to_i64(args[3]);
            let cols = self.coerce_to_i64(args[4]);
            let op = self.coerce_to_i64(args[5]);
            let fref = self.runtime_ref(name);
            self.builder.ins().call(fref, &[x, b, out, rows, cols, op]);
            return None;
        }
        // The streaming Horner-polynomial kernel: wukong_vhorner_f32(x, out, n, coeffs, ncoeff) —
        // three pointers (x, out, coeffs) and two i64 (element count, coefficient count). Void.
        if name == RT_VHORNER && args.len() == 5 {
            let x = self.val(args[0]);
            let out = self.val(args[1]);
            let n = self.coerce_to_i64(args[2]);
            let coeffs = self.val(args[3]);
            let ncoeff = self.coerce_to_i64(args[4]);
            let fref = self.runtime_ref(RT_VHORNER);
            self.builder.ins().call(fref, &[x, out, n, coeffs, ncoeff]);
            return None;
        }
        // The heap builtins. wukong_rt_alloc(count, elem_size, elem_is_float) -> ptr returns the
        // zeroed data pointer (a value-returning runtime call, bound like sreduce below);
        // wukong_rt_free(ptr) is void.
        if name == RT_ALLOC && args.len() == 3 {
            let count = self.coerce_to_i64(args[0]);
            let esize = self.coerce_to_i64(args[1]);
            let isf = self.coerce_to_i64(args[2]);
            let fref = self.runtime_ref(RT_ALLOC);
            let call = self.builder.ins().call(fref, &[count, esize, isf]);
            return self.builder.inst_results(call).first().copied();
        }
        if name == RT_FREE && args.len() == 1 {
            let data = self.val(args[0]);
            let fref = self.runtime_ref(RT_FREE);
            self.builder.ins().call(fref, &[data]);
            return None;
        }
        // The scratch builtins (`wukong_runtime::scratch`): alloc with `wukong_rt_alloc`'s ABI,
        // mark returning the live-allocation count, release taking a mark.
        if name == RT_SCRATCH_ALLOC && args.len() == 3 {
            let count = self.coerce_to_i64(args[0]);
            let esize = self.coerce_to_i64(args[1]);
            let isf = self.coerce_to_i64(args[2]);
            let fref = self.runtime_ref(RT_SCRATCH_ALLOC);
            let call = self.builder.ins().call(fref, &[count, esize, isf]);
            return self.builder.inst_results(call).first().copied();
        }
        if name == RT_SCRATCH_MARK && args.is_empty() {
            let fref = self.runtime_ref(RT_SCRATCH_MARK);
            let call = self.builder.ins().call(fref, &[]);
            return self.builder.inst_results(call).first().copied();
        }
        if name == RT_SCRATCH_RELEASE && args.len() == 1 {
            let mark = self.coerce_to_i64(args[0]);
            let fref = self.runtime_ref(RT_SCRATCH_RELEASE);
            self.builder.ins().call(fref, &[mark]);
            return None;
        }
        // The `now_ns()` timing intrinsic: wukong_now_ns() -> i64, a monotonic process-lifetime
        // nanosecond clock. No arguments; returns the i64 stamp, bound like the value-returning
        // runtime calls below.
        if name == RT_NOW_NS && args.is_empty() {
            let fref = self.runtime_ref(RT_NOW_NS);
            let call = self.builder.ins().call(fref, &[]);
            return self.builder.inst_results(call).first().copied();
        }
        // The failure edge of a checked slice access: wukong_rt_bounds_fail(index, len, site).
        // It records the diagnostic; this function then returns at once through the same path a
        // failed `assert` takes (its callers poll the shared flag), so no statement after the bad
        // access runs. The MIR block's `unreachable` lands in the dead resume block.
        if name == RT_BOUNDS_FAIL && args.len() == 3 {
            let index = self.coerce_to_i64(args[0]);
            let len = self.coerce_to_i64(args[1]);
            let site = self.val(args[2]);
            let fref = self.runtime_ref(RT_BOUNDS_FAIL);
            self.builder.ins().call(fref, &[index, len, site]);
            let failed = self.builder.ins().iconst(types::I8, 1);
            self.return_on_assert_failure(failed, None);
            return None;
        }
        // Every other failed run-time check: wukong_rt_check_fail(kind, a, b, site), the same
        // record-and-return as a bounds failure.
        if name == RT_CHECK_FAIL && args.len() == 4 {
            let kind = self.coerce_to_i64(args[0]);
            let a = self.coerce_to_i64(args[1]);
            let b = self.coerce_to_i64(args[2]);
            let site = self.val(args[3]);
            let fref = self.runtime_ref(RT_CHECK_FAIL);
            self.builder.ins().call(fref, &[kind, a, b, site]);
            let failed = self.builder.ins().iconst(types::I8, 1);
            self.return_on_assert_failure(failed, None);
            return None;
        }
        // The file-I/O intrinsics: wukong_rt_{read,write}_{f32,f64,i32,i64,i8,u8}(path, data, len) -> i64.
        // All twelve share one ABI — the NUL-terminated path pointer, the slice's data pointer, and
        // an i64 element count — so one branch keyed on `name` emits the call. Returns the i64
        // status (element count / -1 open-fail / -2 io-error), bound like the sreduce kernel below.
        if matches!(
            name,
            RT_READ_F32
                | RT_READ_F64
                | RT_READ_I32
                | RT_READ_I64
                | RT_READ_I8
                | RT_READ_U8
                | RT_WRITE_F32
                | RT_WRITE_F64
                | RT_WRITE_I32
                | RT_WRITE_I64
                | RT_WRITE_I8
                | RT_WRITE_U8
        ) && args.len() == 3
        {
            let path = self.val(args[0]);
            let data = self.val(args[1]);
            let len = self.coerce_to_i64(args[2]);
            let fref = self.runtime_ref(name);
            let call = self.builder.ins().call(fref, &[path, data, len]);
            return self.builder.inst_results(call).first().copied();
        }
        // The deterministic reduction kernel: wukong_sreduce_f32[_parallel](x, y, n, op) -> f32 —
        // two pointers, two i64, and an f32 scalar result (the dot/ssd/sum a `@parallel` reduction
        // loop lowers to). Unlike the void kernels above, this returns the accumulated value.
        if matches!(name, RT_SREDUCE | RT_SREDUCE_PARALLEL) && args.len() == 4 {
            let x = self.val(args[0]);
            let y = self.val(args[1]);
            let n = self.coerce_to_i64(args[2]);
            let op = self.coerce_to_i64(args[3]);
            let fref = self.runtime_ref(name);
            let call = self.builder.ins().call(fref, &[x, y, n, op]);
            return self.builder.inst_results(call).first().copied();
        }
        // The deterministic arg-reduction: wukong_argreduce_f32(x, n, op) -> i64 (the argmax/argmin a
        // recognized `if x[k] CMP bv {...}` loop reconciles against). One pointer, two i64, an i64
        // index result — bind the result like the sreduce kernel above.
        if (name == RT_ARGREDUCE || name == RT_ARGREDUCE_PARALLEL) && args.len() == 3 {
            let x = self.val(args[0]);
            let n = self.coerce_to_i64(args[1]);
            let op = self.coerce_to_i64(args[2]);
            let fref = self.runtime_ref(name);
            let call = self.builder.ins().call(fref, &[x, n, op]);
            return self.builder.inst_results(call).first().copied();
        }
        // The bf16 mixed-precision reductions: wukong_dot_bf16(x, y, n) -> f32 (3 args) and
        // wukong_sum_bf16(x, n) -> f32 (2 args). bf16 storage, f32 accumulate; both return the
        // accumulated f32, so bind the call result like the sreduce kernel above.
        if matches!(
            name,
            RT_DOT_BF16 | RT_DOT_F16 | RT_DOT_BF16_PAR | RT_DOT_F16_PAR
        ) && args.len() == 3
        {
            let x = self.val(args[0]);
            let y = self.val(args[1]);
            let n = self.coerce_to_i64(args[2]);
            let fref = self.runtime_ref(name);
            let call = self.builder.ins().call(fref, &[x, y, n]);
            return self.builder.inst_results(call).first().copied();
        }
        if matches!(
            name,
            RT_SUM_BF16 | RT_SUM_F16 | RT_SUM_BF16_PAR | RT_SUM_F16_PAR
        ) && args.len() == 2
        {
            let x = self.val(args[0]);
            let n = self.coerce_to_i64(args[1]);
            let fref = self.runtime_ref(name);
            let call = self.builder.ins().call(fref, &[x, n]);
            return self.builder.inst_results(call).first().copied();
        }
        // wukong_reduce_{bf16,f16}[_parallel](x, n, op) -> f32 — the max-family reduction
        // (max/min/absmax). Same f32 return as the sum/dot kernels above; the op selects the fold.
        if matches!(
            name,
            RT_REDUCE_BF16 | RT_REDUCE_F16 | RT_REDUCE_BF16_PAR | RT_REDUCE_F16_PAR
        ) && args.len() == 3
        {
            let x = self.val(args[0]);
            let n = self.coerce_to_i64(args[1]);
            let op = self.coerce_to_i64(args[2]);
            let fref = self.runtime_ref(name);
            let call = self.builder.ins().call(fref, &[x, n, op]);
            return self.builder.inst_results(call).first().copied();
        }
        // The bf16/f16 mixed-precision streaming axpby: wukong_axpby_{bf16,f16}(x, y, out, n, a, b) —
        // two half-precision input pointers, one f32 output pointer, an i64 count, and two f32
        // coefficients (half in, f32 out, f32 math — the saxpy/axpby a recognized loop lowers to). Void.
        if matches!(
            name,
            RT_AXPBY_BF16 | RT_AXPBY_F16 | RT_AXPBY_BF16_OUT | RT_AXPBY_F16_OUT
        ) && args.len() == 6
        {
            let x = self.val(args[0]);
            let y = self.val(args[1]);
            let out = self.val(args[2]);
            let n = self.coerce_to_i64(args[3]);
            let a = self.val(args[4]);
            let b = self.val(args[5]);
            let fref = self.runtime_ref(name);
            self.builder.ins().call(fref, &[x, y, out, n, a, b]);
            return None;
        }
        // The fused normalization kernel: wukong_norm_f32[_parallel](x, out, rows, cols, eps_bits,
        // op) — two pointers and four i64 (the softmax/LayerNorm/RMSNorm a recognized multi-pass norm
        // lowers to). Void, like the GEMM/vmath kernels.
        if matches!(name, RT_NORM | RT_NORM_PARALLEL) && args.len() == 6 {
            let x = self.val(args[0]);
            let out = self.val(args[1]);
            let rows = self.coerce_to_i64(args[2]);
            let cols = self.coerce_to_i64(args[3]);
            let eps = self.coerce_to_i64(args[4]);
            let op = self.coerce_to_i64(args[5]);
            let fref = self.runtime_ref(name);
            self.builder
                .ins()
                .call(fref, &[x, out, rows, cols, eps, op]);
            return None;
        }
        // The affine fused norm: wukong_norm_affine_f32[_parallel](x, out, gamma, beta, rows, cols,
        // eps_bits, op) — four pointers (gamma/beta may be null) and four i64 (LayerNorm/RMSNorm with a
        // learned per-column scale/shift). Void, like the plain norm; the _parallel one maps rows across
        // cores (bit-equal to serial — rows independent).
        if matches!(name, RT_NORM_AFFINE | RT_NORM_AFFINE_PARALLEL) && args.len() == 8 {
            let x = self.val(args[0]);
            let out = self.val(args[1]);
            let gamma = self.val(args[2]);
            let beta = self.val(args[3]);
            let rows = self.coerce_to_i64(args[4]);
            let cols = self.coerce_to_i64(args[5]);
            let eps = self.coerce_to_i64(args[6]);
            let op = self.coerce_to_i64(args[7]);
            let fref = self.runtime_ref(name);
            self.builder
                .ins()
                .call(fref, &[x, out, gamma, beta, rows, cols, eps, op]);
            return None;
        }
        // The int8 quantized nn.Linear: wukong_i8gemm_nt[_parallel](a, b, c, m, k, n) — three
        // pointers and three i64 (no beta; the kernel always overwrites C). Void.
        if matches!(name, RT_I8GEMM_NT | RT_I8GEMM_NT_PARALLEL) && args.len() == 6 {
            let a = self.val(args[0]);
            let b = self.val(args[1]);
            let c = self.val(args[2]);
            let m = self.coerce_to_i64(args[3]);
            let k = self.coerce_to_i64(args[4]);
            let n = self.coerce_to_i64(args[5]);
            let fref = self.runtime_ref(name);
            self.builder.ins().call(fref, &[a, b, c, m, k, n]);
            return None;
        }
        // Embedding lookup: wukong_embedding_f32[_parallel](out, weight, ids, t, h, v) — three pointers
        // (out/weight f32*, ids i32*) and three i64 dims (the table height `v` rides in as a large
        // sentinel from the recognizer, so the kernel's out-of-range clamp never fires). Void.
        if matches!(name, RT_EMBEDDING | RT_EMBEDDING_PAR) && args.len() == 6 {
            let out = self.val(args[0]);
            let weight = self.val(args[1]);
            let ids = self.val(args[2]);
            let t = self.coerce_to_i64(args[3]);
            let h = self.coerce_to_i64(args[4]);
            let v = self.coerce_to_i64(args[5]);
            let fref = self.runtime_ref(name);
            self.builder.ins().call(fref, &[out, weight, ids, t, h, v]);
            return None;
        }
        // Fused scaled-dot-product attention: wukong_attention_f32(q, k, v, out, s, d, scale, causal)
        // — four f32* pointers, two i64 dims (seq-len `s`, head-dim `d`), an f32 `scale`, and an i64
        // `causal` flag. Online-softmax kernel (no S×S scores materialized). Void.
        if name == RT_ATTENTION && args.len() == 8 {
            let q = self.val(args[0]);
            let k = self.val(args[1]);
            let v = self.val(args[2]);
            let out = self.val(args[3]);
            let s = self.coerce_to_i64(args[4]);
            let d = self.coerce_to_i64(args[5]);
            let scale = self.coerce_to_f32(args[6]);
            let causal = self.coerce_to_i64(args[7]);
            let fref = self.runtime_ref(name);
            self.builder
                .ins()
                .call(fref, &[q, k, v, out, s, d, scale, causal]);
            return None;
        }

        // Scatter-add / embedding-gradient backward: wukong_scatter_add_f32[_parallel](grad_w, grad_out,
        // ids, t, h, v) — same 3-ptr + 3-i64 ABI as the embedding gather (grad_w/grad_out f32*, ids i32*),
        // but `v` is the real table height (the parallel kernel partitions its output rows). Void.
        if matches!(name, RT_SCATTER_ADD | RT_SCATTER_ADD_PAR) && args.len() == 6 {
            let grad_w = self.val(args[0]);
            let grad_out = self.val(args[1]);
            let ids = self.val(args[2]);
            let t = self.coerce_to_i64(args[3]);
            let h = self.coerce_to_i64(args[4]);
            let v = self.coerce_to_i64(args[5]);
            let fref = self.runtime_ref(name);
            self.builder
                .ins()
                .call(fref, &[grad_w, grad_out, ids, t, h, v]);
            return None;
        }
        // The int8 quantized nn.Linear with fused dequant epilogue:
        // wukong_i8gemm_nt_deq[_parallel](a, b, out, m, k, n, scale_a, scale_b, bias, act) — three
        // pointers, three i64, one f32 scalar (scale_a), two pointers (scale_b, bias; bias may be
        // null), one i64 act code. Void.
        if matches!(name, RT_I8GEMM_NT_DEQ | RT_I8GEMM_NT_DEQ_PARALLEL) && args.len() == 10 {
            let a = self.val(args[0]);
            let b = self.val(args[1]);
            let out = self.val(args[2]);
            let m = self.coerce_to_i64(args[3]);
            let k = self.coerce_to_i64(args[4]);
            let n = self.coerce_to_i64(args[5]);
            let scale_a = self.val(args[6]);
            let scale_b = self.val(args[7]);
            let bias = self.val(args[8]);
            let act = self.coerce_to_i64(args[9]);
            let fref = self.runtime_ref(name);
            self.builder
                .ins()
                .call(fref, &[a, b, out, m, k, n, scale_a, scale_b, bias, act]);
            return None;
        }
        // Int-input dequant: wukong_dequant_f32[_parallel](q, out, n, scale, op) — two pointers, one
        // i64 count, one f32 scale, one i64 op (input width + activation). 256-bit widen+scale. Void.
        if matches!(name, RT_DEQUANT | RT_DEQUANT_PAR) && args.len() == 5 {
            let q = self.val(args[0]);
            let out = self.val(args[1]);
            let n = self.coerce_to_i64(args[2]);
            let scale = self.val(args[3]);
            let op = self.coerce_to_i64(args[4]);
            let fref = self.runtime_ref(name);
            self.builder.ins().call(fref, &[q, out, n, scale, op]);
            return None;
        }
        // Per-channel dequant: wukong_dequant_perchan_f32[_parallel](q, out, rows, cols, scale, op) —
        // two data pointers, two i64 dims, one f32* per-column scale, one i64 op. Void.
        if matches!(name, RT_DEQUANT_PERCHAN | RT_DEQUANT_PERCHAN_PAR) && args.len() == 6 {
            let q = self.val(args[0]);
            let out = self.val(args[1]);
            let rows = self.coerce_to_i64(args[2]);
            let cols = self.coerce_to_i64(args[3]);
            let scale = self.val(args[4]);
            let op = self.coerce_to_i64(args[5]);
            let fref = self.runtime_ref(name);
            self.builder
                .ins()
                .call(fref, &[q, out, rows, cols, scale, op]);
            return None;
        }
        // The ML builtins' entries (the `ml_rt` table).
        if let Some(result) = self.lower_ml_rt_call(name, args) {
            return result;
        }
        // `exit(code)`: record the code, then return through every caller at once, exactly as after
        // a failed `assert` (a linked executable's `wukong_rt_exit` ends the process instead).
        if name == text_rt::RT_EXIT && args.len() == 1 {
            let code = self.coerce_to_i64(args[0]);
            let fref = self.runtime_ref(text_rt::RT_EXIT);
            self.builder.ins().call(fref, &[code]);
            let status = self.runtime_ref(RT_ASSERT_FAILED);
            let check = self.builder.ins().call(status, &[]);
            let stopped = self.builder.inst_results(check)[0];
            self.return_on_assert_failure(stopped, None);
            return None;
        }
        // The text builtins' entries (the `text_rt` table).
        if let Some(result) = self.lower_text_rt_call(name, args) {
            return result;
        }
        // Everything above dispatches a runtime kernel on (name, arity). `rt_ids` holds exactly
        // the runtime symbols this function is responsible for, so reaching here with one of them
        // means a recognizer emitted an arity no arm accepts. Unknown `wukong_` names must also
        // fail: the interpreter dispatches some kernel families by prefix. Without this guard `?` below returns
        // `None`, `Op::Call` lowers to *nothing*, and the call is silently deleted: a void kernel
        // (`wukong_sgemm`, `wukong_norm_f32`) leaves the destination buffer holding whatever it
        // held before — a silent wrong answer that the interpreter, which dispatches on the symbol
        // name alone with no arity check, does not reproduce — and a value-returning kernel never
        // enters `vmap`, so the first use panics with `MIR value used before definition`. Record it
        // as a lowering error instead; `build_function_clif` aborts before the function is defined.
        if self.rt_ids.contains_key(name) || name.starts_with("wukong_") {
            if self.layout_err.is_none() {
                self.layout_err = Some(format!(
                    "internal compiler error: runtime kernel `{name}` reached codegen with {} \
                     argument(s), which matches no declared ABI",
                    args.len()
                ));
            }
            return None;
        }
        let arg_ty = args.first().map(|a| self.ty_of(*a).clone());
        let Some(intr) = classify_intrinsic(name, arg_ty.as_ref()) else {
            if self.layout_err.is_none() {
                self.layout_err = Some(format!(
                    "unsupported external call `{name}`: native codegen has no declared ABI or symbol binding"
                ));
            }
            return None;
        };
        match intr {
            Intrinsic::PrintInt => {
                if let Some(&a) = args.first() {
                    let v = self.coerce_to_i64(a);
                    let fref = self.runtime_ref(RT_PRINT_I64);
                    self.builder.ins().call(fref, &[v]);
                }
            }
            Intrinsic::PrintUint => {
                if let Some(&a) = args.first() {
                    // mir_build already zero-extended the value to 64 bits; pass the bits through
                    // (Cranelift has no unsigned types — `rt_print_u64` formats the same i64 bits
                    // as `u64`).
                    let v = self.coerce_to_i64(a);
                    let fref = self.runtime_ref(RT_PRINT_U64);
                    self.builder.ins().call(fref, &[v]);
                }
            }
            Intrinsic::PrintFloat => {
                if let Some(&a) = args.first() {
                    let v = self.coerce_to_f64(a);
                    let fref = self.runtime_ref(RT_PRINT_F64);
                    self.builder.ins().call(fref, &[v]);
                }
            }
            // The per-type scalar printers live in the `text_rt` table; each takes the one value.
            Intrinsic::PrintF32 | Intrinsic::PrintBool | Intrinsic::PrintChar => {
                let entry = match intr {
                    Intrinsic::PrintF32 => text_rt::RT_PRINT_F32,
                    Intrinsic::PrintBool => text_rt::RT_PRINT_BOOL,
                    _ => text_rt::RT_PRINT_CHAR,
                };
                if let Some(a) = args.first() {
                    self.lower_text_rt_call(entry, std::slice::from_ref(a));
                }
            }
            Intrinsic::PrintStr => {
                if let Some(&a) = args.first() {
                    // The argument is already a pointer (the string buffer's base); pass it through.
                    let v = self.val(a);
                    let fref = self.runtime_ref(RT_PRINT_STR);
                    self.builder.ins().call(fref, &[v]);
                }
            }
            Intrinsic::Assert => {
                // `assert(cond[, site[, message]])`: the operands after the condition are the
                // `.rodata` strings the failure reports (`wukong_mir_build/src/bounds.rs`).
                let site = args.get(1).map(|&a| self.val(a));
                let message = args.get(2).map(|&a| self.val(a));
                let failed = if let Some(&a) = args.first() {
                    let value = self.val(a);
                    let ty = self.dfg_ty(value);
                    if ty == types::F32 {
                        let zero = self.builder.ins().f32const(0.0);
                        self.builder.ins().fcmp(FloatCC::Equal, value, zero)
                    } else if ty == types::F64 {
                        let zero = self.builder.ins().f64const(0.0);
                        self.builder.ins().fcmp(FloatCC::Equal, value, zero)
                    } else if ty.is_vector() {
                        self.builder.ins().iconst(types::I8, 1)
                    } else if ty == types::I128 {
                        // `icmp_imm` legalizes through an `iconst` of the operand's type, which
                        // has no 128-bit form.
                        let zero = self.i128_const(0);
                        self.builder.ins().icmp(IntCC::Equal, value, zero)
                    } else {
                        self.builder.ins().icmp_imm(IntCC::Equal, value, 0)
                    }
                } else {
                    self.builder.ins().iconst(types::I8, 1)
                };
                self.return_on_assert_failure(failed, Some([site, message]));
            }
        }
        None
    }

    /// Return normally through generated callers; never unwind Rust through an extern-C frame.
    /// A caller checks the shared failure flag before it can observe the dummy return value.
    /// `record` is `Some([site, message])` for a failed `assert`, which the stop block reports
    /// through `wukong_rt_assert(0, site, message)` (a missing operand passes null); `None` when
    /// the failure was already recorded by the call that raised it.
    fn return_on_assert_failure(&mut self, failed: Value, record: Option<[Option<Value>; 2]>) {
        let stop = self.builder.create_block();
        let resume = self.builder.create_block();
        self.builder.ins().brif(failed, stop, &[], resume, &[]);
        self.builder.switch_to_block(stop);
        if let Some(operands) = record {
            let zero = self.builder.ins().iconst(types::I64, 0);
            let mut call_args = vec![zero];
            for operand in operands {
                let v = match operand {
                    Some(v) => v,
                    None => self.builder.ins().iconst(self.ptr_ty, 0),
                };
                call_args.push(v);
            }
            let assert = self.runtime_ref(RT_ASSERT);
            self.builder.ins().call(assert, &call_args);
        }
        match cl_type(&self.func.ret, self.ptr_ty) {
            None => {
                self.builder.ins().return_(&[]);
            }
            Some(ty) => {
                let lane = ty.lane_type();
                let zero = if lane == types::F32 {
                    self.builder.ins().f32const(0.0)
                } else if lane == types::F64 {
                    self.builder.ins().f64const(0.0)
                } else if lane == types::I128 {
                    // `iconst` has no 128-bit form (`int128.rs`).
                    self.i128_const(0)
                } else {
                    self.builder.ins().iconst(lane, 0)
                };
                let zero = if ty.is_vector() {
                    self.builder.ins().splat(ty, zero)
                } else {
                    zero
                };
                self.builder.ins().return_(&[zero]);
            }
        }
        self.builder.switch_to_block(resume);
    }

    fn coerce_to_i64(&mut self, v: ValueId) -> Value {
        let x = self.val(v);
        let from = self.dfg_ty(x);
        if from.is_int() {
            self.resize_int(x, from, types::I64, true)
        } else {
            self.builder.ins().fcvt_to_sint_sat(types::I64, x)
        }
    }

    fn coerce_to_f64(&mut self, v: ValueId) -> Value {
        let x = self.val(v);
        let from = self.dfg_ty(x);
        if from == types::F64 {
            x
        } else {
            self.builder.ins().fpromote(types::F64, x)
        }
    }

    fn coerce_to_f32(&mut self, v: ValueId) -> Value {
        let x = self.val(v);
        let from = self.dfg_ty(x);
        if from == types::F32 {
            x
        } else if from == types::F64 {
            self.builder.ins().fdemote(types::F32, x)
        } else {
            // Integer scale (unusual, but keep the call well-typed): signed int → f32.
            self.builder.ins().fcvt_from_sint(types::F32, x)
        }
    }

    fn lower_term(&mut self, term: &Terminator) {
        match term {
            Terminator::Ret(None) => {
                self.builder.ins().return_(&[]);
            }
            Terminator::Ret(Some(v)) => {
                let x = self.val(*v);
                self.builder.ins().return_(&[x]);
            }
            Terminator::Br { target, args } => {
                self.branch_args.clear();
                self.branch_args.reserve(args.len());
                for &arg in args {
                    let value = self.val(arg);
                    self.branch_args.push(BlockArg::Value(value));
                }
                let cb = self.blocks[target.0 as usize];
                self.builder.ins().jump(cb, &*self.branch_args);
            }
            Terminator::CondBr {
                cond,
                then_blk,
                then_args,
                else_blk,
                else_args,
            } => {
                let c = self.val(*cond);
                self.branch_args.clear();
                self.branch_args.reserve(then_args.len() + else_args.len());
                for &arg in then_args.iter().chain(else_args) {
                    let value = self.val(arg);
                    self.branch_args.push(BlockArg::Value(value));
                }
                let (ta, ea) = self.branch_args.split_at(then_args.len());
                let tb = self.blocks[then_blk.0 as usize];
                let eb = self.blocks[else_blk.0 as usize];
                self.builder.ins().brif(c, tb, ta, eb, ea);
            }
            Terminator::Switch {
                scrutinee,
                cases,
                default,
                default_args,
            } => {
                // `cranelift_frontend::Switch` picks a jump table, bit tests or a search tree. It
                // keys entries by the scrutinee's *unsigned* bit pattern, so each canonical
                // (sign-extended) case value is masked to the type's width; distinct canonical
                // values stay distinct. Its targets take no arguments, so an edge that passes
                // some goes through a trampoline block that forwards them.
                let x = self.val(*scrutinee);
                let bits = self.dfg_ty(x).bits();
                let mask = if bits >= 128 {
                    u128::MAX
                } else {
                    (1u128 << bits) - 1
                };
                let mut forwards: Vec<(Block, u32, &[ValueId])> = Vec::new();
                let mut switch = cranelift_frontend::Switch::new();
                for c in cases {
                    let dest = if c.args.is_empty() {
                        self.blocks[c.target.0 as usize]
                    } else {
                        let tb = self.builder.create_block();
                        forwards.push((tb, c.target.0, c.args.as_slice()));
                        tb
                    };
                    switch.set_entry((c.value as u128) & mask, dest);
                }
                let otherwise = if default_args.is_empty() {
                    self.blocks[default.0 as usize]
                } else {
                    let tb = self.builder.create_block();
                    forwards.push((tb, default.0, default_args.as_slice()));
                    tb
                };
                switch.emit(&mut self.builder, x, otherwise);
                for (tb, target, args) in forwards {
                    self.builder.switch_to_block(tb);
                    self.branch_args.clear();
                    self.branch_args.reserve(args.len());
                    for &arg in args {
                        let value = self.val(arg);
                        self.branch_args.push(BlockArg::Value(value));
                    }
                    let cb = self.blocks[target as usize];
                    self.builder.ins().jump(cb, &*self.branch_args);
                }
            }
            Terminator::Unreachable => {
                let tc = cranelift_codegen::ir::TrapCode::user(1).unwrap();
                self.builder.ins().trap(tc);
            }
        }
    }
}

// --- Module-level driving (shared by JIT and object emission) -------------------------------

struct RtFuncs {
    print_i64: FuncId,
    print_u64: FuncId,
    print_f64: FuncId,
    print_str: FuncId,
    assert: FuncId,
    assert_failed: FuncId,
    bounds_fail: FuncId,
    check_fail: FuncId,
    rt_alloc: FuncId,
    rt_free: FuncId,
    rt_scratch_alloc: FuncId,
    rt_scratch_mark: FuncId,
    rt_scratch_release: FuncId,
    rt_read_f32: FuncId,
    rt_read_f64: FuncId,
    rt_read_i32: FuncId,
    rt_read_i64: FuncId,
    rt_read_i8: FuncId,
    rt_read_u8: FuncId,
    rt_write_f32: FuncId,
    rt_write_f64: FuncId,
    rt_write_i32: FuncId,
    rt_write_i64: FuncId,
    rt_write_i8: FuncId,
    rt_write_u8: FuncId,
    now_ns: FuncId,
    parallel_for: FuncId,
    sgemm: FuncId,
    sgemm_parallel: FuncId,
    sgemm_nt: FuncId,
    sgemm_nt_parallel: FuncId,
    sgemm_tn: FuncId,
    sgemm_tn_parallel: FuncId,
    sgemm_bf16_nt: FuncId,
    sgemm_bf16_nt_parallel: FuncId,
    sgemm_f16_nt: FuncId,
    sgemm_f16_nt_parallel: FuncId,
    sgemm_bf16_tn: FuncId,
    sgemm_bf16_tn_parallel: FuncId,
    sgemm_f16_tn: FuncId,
    sgemm_f16_tn_parallel: FuncId,
    sgemm_nt_epi: FuncId,
    sgemm_nt_epi_par: FuncId,
    sgemv: FuncId,
    sgemv_par: FuncId,
    sgemv_alpha: FuncId,
    sgemv_alpha_par: FuncId,
    gevm: FuncId,
    gevm_par: FuncId,
    sgemm_nt_alpha: FuncId,
    sgemm_nt_alpha_par: FuncId,
    sgemm_bf16_nt_epi: FuncId,
    sgemm_bf16_nt_epi_par: FuncId,
    sgemm_f16_nt_epi: FuncId,
    sgemm_f16_nt_epi_par: FuncId,
    vmath: FuncId,
    trig_f32: FuncId,
    math_f64: FuncId,
    math2_f64: FuncId,
    vmath_par: FuncId,
    vmath2: FuncId,
    softmax_bwd: FuncId,
    softmax_bwd_par: FuncId,
    lrscan: FuncId,
    lrscan_par: FuncId,
    rmsnorm_bwd: FuncId,
    rmsnorm_bwd_par: FuncId,
    layernorm_bwd: FuncId,
    layernorm_bwd_par: FuncId,
    xent: FuncId,
    xent_par: FuncId,
    xent_bwd: FuncId,
    xent_bwd_par: FuncId,
    rope: FuncId,
    rope_par: FuncId,
    rope_bwd: FuncId,
    rope_bwd_par: FuncId,
    logsumexp: FuncId,
    logsumexp_par: FuncId,
    kldiv: FuncId,
    kldiv_par: FuncId,
    entropy: FuncId,
    entropy_par: FuncId,
    rowargmax: FuncId,
    rowargmax_par: FuncId,
    rowargmin: FuncId,
    rowargmin_par: FuncId,
    colargmax: FuncId,
    colargmax_par: FuncId,
    colargmin: FuncId,
    colargmin_par: FuncId,
    cumsum: FuncId,
    cumsum_par: FuncId,
    cumprod: FuncId,
    cumprod_par: FuncId,
    cummax: FuncId,
    cummax_par: FuncId,
    cummin: FuncId,
    cummin_par: FuncId,
    maxpool2d: FuncId,
    maxpool2d_par: FuncId,
    avgpool2d: FuncId,
    avgpool2d_par: FuncId,
    kd_loss: FuncId,
    kd_loss_par: FuncId,
    vmath_bf16: FuncId,
    vmath_f16: FuncId,
    vmath_bf16_out: FuncId,
    vmath_f16_out: FuncId,
    transpose: FuncId,
    transpose_par: FuncId,
    transpose_u16: FuncId,
    transpose_u16_par: FuncId,
    colsum: FuncId,
    colsum_par: FuncId,
    colmax: FuncId,
    colmax_par: FuncId,
    colmin: FuncId,
    colmin_par: FuncId,
    colmaxabs: FuncId,
    colmaxabs_par: FuncId,
    colmean: FuncId,
    colmean_par: FuncId,
    colsumsq: FuncId,
    colsumsq_par: FuncId,
    coll2: FuncId,
    coll2_par: FuncId,
    colrms: FuncId,
    colrms_par: FuncId,
    velem: FuncId,
    velem_par: FuncId,
    vhorner: FuncId,
    bias_bcast: FuncId,
    bias_bcast_par: FuncId,
    sred: FuncId,
    sred_par: FuncId,
    argreduce: FuncId,
    argreduce_par: FuncId,
    norm: FuncId,
    norm_par: FuncId,
    norm_affine: FuncId,
    norm_affine_par: FuncId,
    i8nt: FuncId,
    i8nt_par: FuncId,
    i8nt_deq: FuncId,
    i8nt_deq_par: FuncId,
    /// Int-input dequant `wukong_dequant_f32[_parallel](q, out, n, scale, op)` — 2 ptr, i64, f32, i64.
    dequant: FuncId,
    dequant_par: FuncId,
    /// Per-channel dequant `wukong_dequant_perchan_f32[_parallel](q, out, rows, cols, scale, op)` —
    /// ptr, ptr, i64, i64, ptr, i64.
    dequant_perchan: FuncId,
    dequant_perchan_par: FuncId,
    /// Embedding lookup `wukong_embedding_f32[_parallel](out, weight, ids, t, h, v)` — token-id row
    /// gather (the first layer of every LLM). Reuses the 3-ptr + 3-i64 void `sig_i8gemm` signature.
    embedding: FuncId,
    embedding_par: FuncId,
    /// Fused scaled-dot-product attention `wukong_attention_f32(q, k, v, out, s, d, scale, causal)` —
    /// four ptrs, two i64 dims, an f32 scale, an i64 causal flag (its own 8-arg signature).
    attention: FuncId,
    scatter_add: FuncId,
    scatter_add_par: FuncId,
    dot_bf16: FuncId,
    sum_bf16: FuncId,
    reduce_bf16: FuncId,
    axpby_f16: FuncId,
    dot_f16: FuncId,
    sum_f16: FuncId,
    reduce_f16: FuncId,
    dot_bf16_par: FuncId,
    sum_bf16_par: FuncId,
    reduce_bf16_par: FuncId,
    dot_f16_par: FuncId,
    sum_f16_par: FuncId,
    reduce_f16_par: FuncId,
    f32_to_f16: FuncId,
    f16_to_f32: FuncId,
    axpby_bf16: FuncId,
    axpby_bf16_out: FuncId,
    axpby_f16_out: FuncId,
    fmod_f64: FuncId,
    fmod_f32: FuncId,
}

fn signature_of(
    f: &Function,
    ptr_ty: Type,
    call_conv: cranelift_codegen::isa::CallConv,
) -> Signature {
    let mut sig = Signature::new(call_conv);
    for p in &f.params {
        if let Some(t) = cl_type(f.value_type(*p), ptr_ty) {
            sig.params.push(AbiParam::new(t));
        }
    }
    if let Some(t) = cl_type(&f.ret, ptr_ty) {
        sig.returns.push(AbiParam::new(t));
    }
    sig
}

/// The C-ABI signature of an `extern "C"` import. A narrow integer is marked for sign or zero
/// extension per its declared signedness — the C ABI leaves extending a `char`/`short`/`bool`
/// argument to the caller — and `bool` (`i1`, an `i8` in CLIF) is unsigned.
fn extern_signature(
    e: &wukong_mir::ExternFn,
    ptr_ty: Type,
    call_conv: cranelift_codegen::isa::CallConv,
) -> Signature {
    let abi = |p: &wukong_mir::ExternParam| -> Option<AbiParam> {
        let t = cl_type(&p.ty, ptr_ty)?;
        let param = AbiParam::new(t);
        Some(if p.ty.is_int() && t.bits() < 64 {
            if p.signed {
                param.sext()
            } else {
                param.uext()
            }
        } else {
            param
        })
    };
    let mut sig = Signature::new(call_conv);
    sig.params.extend(e.params.iter().filter_map(&abi));
    if let Some(r) = abi(&e.ret) {
        sig.returns.push(r);
    }
    sig
}

/// Bind every `extern "C"` import of `program` to the address `wukong_runtime::cffi` resolves —
/// the very address the interpreter calls, which is what keeps the two backends bit-exact on a C
/// function. An unresolvable symbol is a clean error here instead of cranelift-jit's panic when it
/// finalizes the relocation.
fn bind_extern_symbols(
    builder: &mut cranelift_jit::JITBuilder,
    program: &Program,
    interner: &Interner,
) -> Result<(), String> {
    for e in &program.externs {
        let name = interner.resolve(e.name);
        let addr = wukong_runtime::cffi::resolve_c_symbol(name).ok_or_else(|| {
            format!(
                "unresolved external symbol `{name}`: no C library loaded into this process \
                 defines it (an `extern \"C\"` function must come from the C runtime, libm, or \
                 another library the process has loaded)"
            )
        })?;
        builder.symbol(name, addr);
    }
    Ok(())
}

/// Should the Cranelift IR verifier run? It is a **check-only** pass — it inspects the CLIF for
/// well-formedness and never modifies the function or the emitted machine code, so toggling it is
/// byte-identical. Cranelift defaults it *on* ("makes compilation slower but catches many bugs"),
/// which is pure compile-time cost once our lowering is trusted. We keep it on under
/// `debug_assertions` (so `cargo test` — a debug build — still validates every lowering's IR) and off
/// in release (the latency path), with an env override for release debugging.
fn default_verify() -> bool {
    match std::env::var("WUKONG_CL_VERIFY").ok().as_deref() {
        Some("0") => false,
        Some(_) => true,
        None => cfg!(debug_assertions),
    }
}

/// Should the backend compile the user functions concurrently — for `emit_object` and for both JIT
/// entry points ([`jit_compile`], [`jit_module`])? Each function is an independent compilation
/// unit, so parallel isel + serial in-order definition is byte-identical to serial codegen (proven
/// for objects by `serial_parallel_object_bytes_identical`; the JIT twin in `backend_opt_tests`
/// checks the two JIT paths agree). Default on; the `WUKONG_PAR_CODEGEN=0` kill-switch forces serial.
fn default_parallel() -> bool {
    !matches!(
        std::env::var("WUKONG_PAR_CODEGEN").ok().as_deref(),
        Some("0")
    )
}

/// How hard Cranelift works on the code it is handed — the backend half of the source `-O` level.
///
/// The mid-end (e-graph rewrites, GVN, alias analysis) is worth its compile time only on code the
/// optimizer has already shaped. At `-O0` `wukong_opt` runs no passes, so Cranelift receives the
/// largest, least-SSA input (every local an alloca with loads and stores) — and `-O0` is the CLI
/// default, the configuration where compile latency matters most. Mapping it to Cranelift's
/// `opt_level=none` is what `rustc_codegen_cranelift` does for debug builds. No Cranelift opt level
/// reassociates, contracts or otherwise changes floating-point results, so the interpreter/native
/// differential stays bit-exact at every level; only code speed differs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BackendOpt {
    /// The source optimization level (`-O0` .. `-O3`). `0` selects Cranelift `opt_level=none`;
    /// anything higher selects `speed`.
    pub level: u8,
}

impl BackendOpt {
    /// The backend configuration for source optimization level `level`.
    pub fn for_level(level: u8) -> Self {
        BackendOpt { level }
    }

    /// The Cranelift `opt_level` flag value this configuration selects.
    fn cranelift_opt_level(self) -> &'static str {
        if self.level == 0 {
            "none"
        } else {
            "speed"
        }
    }
}

impl Default for BackendOpt {
    /// The optimizing configuration (`speed`): what every entry point without an explicit
    /// [`BackendOpt`] — the benches, the harnesses, the library API — has always compiled with.
    fn default() -> Self {
        BackendOpt { level: 2 }
    }
}

fn make_isa(pic: bool) -> Result<std::sync::Arc<dyn cranelift_codegen::isa::TargetIsa>, String> {
    make_isa_for(pic, default_verify(), TargetCpu::Native)
}

/// The ISA for `target` with the optimizing backend configuration ([`BackendOpt::default`]).
fn make_isa_for(
    pic: bool,
    verify: bool,
    target: TargetCpu,
) -> Result<std::sync::Arc<dyn cranelift_codegen::isa::TargetIsa>, String> {
    make_isa_full(pic, verify, target, BackendOpt::default())
}

/// The host ISA ([`TargetCpu::Native`]) with an explicit backend configuration.
///
/// Test-only since the `--target-cpu` merge: every production caller (the JIT and the object
/// emitter) now picks its target explicitly through [`make_isa_full`]; this host shorthand remains
/// for `backend_opt_tests`, which pins the `-O` level → Cranelift `opt_level` mapping.
#[cfg(test)]
fn make_isa_with(
    pic: bool,
    verify: bool,
    backend: BackendOpt,
) -> Result<std::sync::Arc<dyn cranelift_codegen::isa::TargetIsa>, String> {
    make_isa_full(pic, verify, TargetCpu::Native, backend)
}

/// Both axes at once: the instruction-set level code is generated for (`--target-cpu`) and how
/// hard Cranelift works on it (the source `-O` level). They are independent — a target level
/// fixes the feature flags, the backend level only the mid-end — so every pair is valid.
fn make_isa_full(
    pic: bool,
    verify: bool,
    target: TargetCpu,
    backend: BackendOpt,
) -> Result<std::sync::Arc<dyn cranelift_codegen::isa::TargetIsa>, String> {
    let mut flag_builder = settings::builder();
    flag_builder
        .set("opt_level", backend.cranelift_opt_level())
        .map_err(|e| e.to_string())?;
    // The IR verifier is a diagnostic-only pass (see `default_verify`); disabling it in release
    // trims compile time without changing a single emitted byte.
    flag_builder
        .set("enable_verifier", if verify { "true" } else { "false" })
        .map_err(|e| e.to_string())?;
    // The JIT requires non-PIC; the object emitter wants PIC. Configure per caller.
    flag_builder
        .set("is_pic", if pic { "true" } else { "false" })
        .map_err(|e| e.to_string())?;
    // Emit stack-probe code in any prologue whose frame exceeds a page, so a function with a large
    // local array (e.g. an im2col scratch buffer) touches each guard page instead of jumping past it
    // and faulting. The *inline* strategy emits the probe loop directly — no external `__chkstk`
    // symbol to bind, which the JIT cannot provide. Probing only commits stack pages, so it changes
    // no computed value and keeps the interpreter/native differential exact.
    flag_builder
        .set("enable_probestack", "true")
        .map_err(|e| e.to_string())?;
    flag_builder
        .set("probestack_strategy", "inline")
        .map_err(|e| e.to_string())?;
    // An `i128` parameter or result of a Wukong function travels as two consecutive 64-bit
    // registers (or stack words), low half first — LLVM's convention, which Cranelift's x64 ABI
    // implements only under this flag and otherwise refuses with a panic. Every caller and callee of
    // such a signature is code this backend generates, so the convention only has to agree with
    // itself; the runtime entries take `i128` operands as explicit halves and never see it. Beyond
    // `i128` (and the `f16`/`f128` this backend never passes) the flag changes nothing but the
    // third integer return register of a multi-value SysV return, which no Wukong signature has.
    flag_builder
        .set("enable_llvm_abi_extensions", "true")
        .map_err(|e| e.to_string())?;
    // `WUKONG_MAX_ISA` narrows `native` (read once per process by `wukong_runtime::isa::cap`), so
    // every JIT and object entry point that targets the host inherits the cap from this one place.
    let isa_builder = target::isa_builder(target, wukong_runtime::isa::cap())?;
    isa_builder
        .finish(settings::Flags::new(flag_builder))
        .map_err(|e| e.to_string())
}

/// Is `func` (not a user function) one of the runtime failure intrinsics — a builtin `assert`, the
/// failure edge of a checked slice access, the failure edge of any other run-time check (a range
/// index, `str_from_utf8`), or `exit`? Each one stops the run. A failure entry missing here is a
/// silent JIT divergence: the callee records the failure and returns, and its caller, never
/// polled, runs on and reads whatever the callee did not write.
fn is_failure_intrinsic(func: Symbol, interner: &Interner) -> bool {
    matches!(
        interner.resolve(func),
        "assert" | RT_BOUNDS_FAIL | RT_CHECK_FAIL | text_rt::RT_EXIT
    )
}

/// The functions that can stop the run and so must be polled after every call to them: one that
/// calls a runtime failure intrinsic ([`is_failure_intrinsic`]: a builtin `assert`, a slice bounds
/// failure, any other run-time check's failure, `exit`), calls such a function, or runs a parallel
/// region whose body is one. A region body's failure comes back
/// through the JIT's `rt_parallel_for` (the cancellation ABI, see there) to the caller of
/// `wukong_parallel_for`; in a linked executable it exits the process where it happens.
///
/// One scan builds the reverse call graph and the seeds, then a worklist walks it, linear in
/// instructions plus call edges. A direct call to a defined function is an edge; so is a parallel
/// region whose body is a `func_addr g` (`g` failing makes the region's caller fail), while a body
/// of any other provenance is assumed to fail ([`callback_can_fail`]) and seeds its caller. The
/// sweep-to-fixpoint this replaces re-scanned every instruction per sweep and found one more
/// call-chain level per sweep when callers precede their callees in source order (O(chain length x
/// instructions)); both compute the same least fixpoint.
fn asserting_functions(program: &Program, interner: &Interner) -> HashSet<Symbol> {
    let defined: HashSet<Symbol> = program.funcs.iter().map(|f| f.name).collect();
    let mut callers: HashMap<Symbol, Vec<Symbol>> = HashMap::new();
    let mut asserting = HashSet::new();
    let mut work = Vec::new();
    for f in &program.funcs {
        for inst in f.blocks.iter().flat_map(|b| &b.insts) {
            let Op::Call { func, args } = &inst.op else {
                continue;
            };
            if defined.contains(func) {
                callers.entry(*func).or_default().push(f.name);
                continue;
            }
            let seed = if is_failure_intrinsic(*func, interner) {
                true
            } else if interner.resolve(*func) == RT_PARALLEL_FOR {
                match args.get(1).map(|&body| callback_target(f, body)) {
                    // A `func_addr g` body: `f` fails exactly when `g` does — an edge.
                    Some(Some(g)) => {
                        callers.entry(g).or_default().push(f.name);
                        false
                    }
                    // A body of any other provenance is assumed to fail.
                    Some(None) => callback_can_fail(f, args[1], &asserting),
                    None => false,
                }
            } else {
                false
            };
            if seed && asserting.insert(f.name) {
                work.push(f.name);
            }
        }
    }
    while let Some(callee) = work.pop() {
        for &caller in callers.get(&callee).map_or(&[][..], |v| &v[..]) {
            if asserting.insert(caller) {
                work.push(caller);
            }
        }
    }
    asserting
}

/// The function a parallel region's `body` operand names when `f` defines it as `func_addr g`;
/// `None` for a body of any other provenance.
fn callback_target(f: &Function, body: ValueId) -> Option<Symbol> {
    let def = f
        .blocks
        .iter()
        .flat_map(|b| &b.insts)
        .find(|inst| inst.result == Some(body));
    match def.map(|inst| &inst.op) {
        Some(Op::FuncAddr(g)) => Some(*g),
        _ => None,
    }
}

/// Can the callback whose address `f` passes as `body` fail? A direct `func_addr g` can exactly
/// when `g` is failure-capable; a body of any other provenance is assumed to.
fn callback_can_fail(f: &Function, body: ValueId, asserting: &HashSet<Symbol>) -> bool {
    let def = f
        .blocks
        .iter()
        .flat_map(|b| &b.insts)
        .find(|inst| inst.result == Some(body));
    match def.map(|inst| &inst.op) {
        Some(Op::FuncAddr(g)) => asserting.contains(g),
        _ => true,
    }
}

/// Declare every user function and the runtime imports into `module`, then define each body.
fn populate_module<M: Module>(
    module: &mut M,
    program: &Program,
    interner: &Interner,
    parallel: bool,
    mode: entry::SymbolMode,
    mut log: Option<&mut debuginfo::CodeLog>,
) -> Result<HashMap<Symbol, FuncId>, NativeFailure> {
    // Compact boolean-vector casts are legalized per function inside `build_function_clif`.
    let asserting = asserting_functions(program, interner);
    // Stack-slot coloring needs to know which calls may retain a pointer (`stack_color`).
    let captures = stack_color::enabled().then(|| wukong_opt::CaptureSummary::analyze(program));
    let ptr_ty = module.target_config().pointer_type();
    let call_conv = module.target_config().default_call_conv;

    let mut sig_i = Signature::new(call_conv);
    sig_i.params.push(AbiParam::new(types::I64));
    let mut sig_f = Signature::new(call_conv);
    sig_f.params.push(AbiParam::new(types::F64));
    // wukong_parallel_for(n: i64, body: ptr, env: ptr)
    let mut sig_par = Signature::new(call_conv);
    sig_par.params.push(AbiParam::new(types::I64));
    sig_par.params.push(AbiParam::new(ptr_ty));
    sig_par.params.push(AbiParam::new(ptr_ty));
    // wukong_sgemm(a: ptr, b: ptr, c: ptr, m: i64, k: i64, n: i64, beta: i64)
    let mut sig_gemm = Signature::new(call_conv);
    for _ in 0..3 {
        sig_gemm.params.push(AbiParam::new(ptr_ty));
    }
    for _ in 0..4 {
        sig_gemm.params.push(AbiParam::new(types::I64));
    }
    // wukong_sgemm_nt_epi(a, b, c: ptr, m, k, n, beta: i64, bias: ptr, act: i64) — fused Linear.
    let mut sig_gemm_epi = Signature::new(call_conv);
    for _ in 0..3 {
        sig_gemm_epi.params.push(AbiParam::new(ptr_ty));
    }
    for _ in 0..4 {
        sig_gemm_epi.params.push(AbiParam::new(types::I64));
    }
    sig_gemm_epi.params.push(AbiParam::new(ptr_ty)); // bias
    sig_gemm_epi.params.push(AbiParam::new(types::I64)); // act
                                                         // wukong_sgemv[_parallel](a: ptr, x: ptr, y: ptr, m: i64, n: i64) — matrix-times-vector (void).
    let mut sig_gemv = Signature::new(call_conv);
    for _ in 0..3 {
        sig_gemv.params.push(AbiParam::new(ptr_ty));
    }
    for _ in 0..2 {
        sig_gemv.params.push(AbiParam::new(types::I64));
    }
    // wukong_sgemv_alpha[_parallel](a, x, y: ptr, m, n: i64, alpha: f32) — the α-scaled GEMV, and
    // the same 3-ptr + 2-i64 + trailing-f32 shape for wukong_sgevm_f32[_parallel](w, a, out, rows,
    // cols, alpha), the vector·matrix product. Void.
    let mut sig_gemv_alpha = Signature::new(call_conv);
    for _ in 0..3 {
        sig_gemv_alpha.params.push(AbiParam::new(ptr_ty));
    }
    for _ in 0..2 {
        sig_gemv_alpha.params.push(AbiParam::new(types::I64));
    }
    sig_gemv_alpha.params.push(AbiParam::new(types::F32)); // alpha
                                                           // wukong_sgemm_nt_alpha[_parallel](a, b, c: ptr, m, k, n, beta: i64, alpha: f32) — α-scaled Linear.
    let mut sig_gemm_alpha = Signature::new(call_conv);
    for _ in 0..3 {
        sig_gemm_alpha.params.push(AbiParam::new(ptr_ty));
    }
    for _ in 0..4 {
        sig_gemm_alpha.params.push(AbiParam::new(types::I64));
    }
    sig_gemm_alpha.params.push(AbiParam::new(types::F32)); // alpha
                                                           // wukong_vmath_f32(x: ptr, out: ptr, n: i64, op: i64) — vectorized elementwise transcendental.
    let mut sig_vmath = Signature::new(call_conv);
    sig_vmath.params.push(AbiParam::new(ptr_ty));
    sig_vmath.params.push(AbiParam::new(ptr_ty));
    sig_vmath.params.push(AbiParam::new(types::I64));
    sig_vmath.params.push(AbiParam::new(types::I64));
    let mut sig_trig = Signature::new(call_conv);
    sig_trig.params.push(AbiParam::new(types::F32));
    sig_trig.params.push(AbiParam::new(types::I64));
    sig_trig.returns.push(AbiParam::new(types::F32));
    // wukong_rt_math_f64(x: f64, op: i64) -> f64 and wukong_rt_math2_f64(a, b: f64, op: i64) -> f64:
    // the double-precision transcendental library every f64 transcendental lowers to.
    let mut sig_math_f64 = Signature::new(call_conv);
    sig_math_f64.params.push(AbiParam::new(types::F64));
    sig_math_f64.params.push(AbiParam::new(types::I64));
    sig_math_f64.returns.push(AbiParam::new(types::F64));
    let mut sig_math2_f64 = Signature::new(call_conv);
    sig_math2_f64.params.push(AbiParam::new(types::F64));
    sig_math2_f64.params.push(AbiParam::new(types::F64));
    sig_math2_f64.params.push(AbiParam::new(types::I64));
    sig_math2_f64.returns.push(AbiParam::new(types::F64));
    // wukong_vmath2_f32(x, y, out: ptr, n: i64, op: i64) — two-input transcendental (pow/atan2/hypot).
    let mut sig_vmath2 = Signature::new(call_conv);
    sig_vmath2.params.push(AbiParam::new(ptr_ty));
    sig_vmath2.params.push(AbiParam::new(ptr_ty));
    sig_vmath2.params.push(AbiParam::new(ptr_ty));
    sig_vmath2.params.push(AbiParam::new(types::I64));
    sig_vmath2.params.push(AbiParam::new(types::I64));
    // wukong_velem_f32(x, y, out: ptr, n: i64, a, b, c: f32, op: i64) — streaming affine+activation.
    let mut sig_velem = Signature::new(call_conv);
    sig_velem.params.push(AbiParam::new(ptr_ty));
    sig_velem.params.push(AbiParam::new(ptr_ty));
    sig_velem.params.push(AbiParam::new(ptr_ty));
    sig_velem.params.push(AbiParam::new(types::I64));
    sig_velem.params.push(AbiParam::new(types::F32));
    sig_velem.params.push(AbiParam::new(types::F32));
    sig_velem.params.push(AbiParam::new(types::F32));
    sig_velem.params.push(AbiParam::new(types::I64));
    // wukong_vhorner_f32(x, out: ptr, n: i64, coeffs: ptr, ncoeff: i64) — streaming Horner poly.
    let mut sig_vhorner = Signature::new(call_conv);
    sig_vhorner.params.push(AbiParam::new(ptr_ty));
    sig_vhorner.params.push(AbiParam::new(ptr_ty));
    sig_vhorner.params.push(AbiParam::new(types::I64));
    sig_vhorner.params.push(AbiParam::new(ptr_ty));
    sig_vhorner.params.push(AbiParam::new(types::I64));
    // wukong_bias_bcast_f32[_parallel](x, b, out: ptr, rows, cols, op: i64) — broadcast-bias add
    // (3 ptr + 3 i64, void).
    let mut sig_bias_bcast = Signature::new(call_conv);
    for _ in 0..3 {
        sig_bias_bcast.params.push(AbiParam::new(ptr_ty));
    }
    for _ in 0..3 {
        sig_bias_bcast.params.push(AbiParam::new(types::I64));
    }
    // wukong_sreduce_f32[_parallel](x, y: ptr, n, op: i64) -> f32 — deterministic reduction kernel.
    let mut sig_sreduce = Signature::new(call_conv);
    sig_sreduce.params.push(AbiParam::new(ptr_ty));
    sig_sreduce.params.push(AbiParam::new(ptr_ty));
    sig_sreduce.params.push(AbiParam::new(types::I64));
    sig_sreduce.params.push(AbiParam::new(types::I64));
    sig_sreduce.returns.push(AbiParam::new(types::F32));
    // wukong_argreduce_f32(x: ptr, n, op: i64) -> i64 — the deterministic argmax/argmin index.
    let mut sig_argreduce = Signature::new(call_conv);
    sig_argreduce.params.push(AbiParam::new(ptr_ty));
    sig_argreduce.params.push(AbiParam::new(types::I64));
    sig_argreduce.params.push(AbiParam::new(types::I64));
    sig_argreduce.returns.push(AbiParam::new(types::I64));
    // wukong_norm_f32[_parallel](x, out: ptr, rows, cols, eps_bits, op: i64) — fused row-wise norm.
    let mut sig_norm = Signature::new(call_conv);
    sig_norm.params.push(AbiParam::new(ptr_ty));
    sig_norm.params.push(AbiParam::new(ptr_ty));
    for _ in 0..4 {
        sig_norm.params.push(AbiParam::new(types::I64));
    }
    // wukong_norm_affine_f32(x, out, gamma, beta: ptr, rows, cols, eps_bits, op: i64) — affine norm.
    let mut sig_norm_affine = Signature::new(call_conv);
    for _ in 0..4 {
        sig_norm_affine.params.push(AbiParam::new(ptr_ty));
    }
    for _ in 0..4 {
        sig_norm_affine.params.push(AbiParam::new(types::I64));
    }
    // wukong_{max,avg}pool2d_f32[_parallel](x, out: ptr, channels, h, w, kh, kw, sh, sw: i64) — 2D
    // pooling (2 ptr + 7 i64, void).
    let mut sig_pool2d = Signature::new(call_conv);
    sig_pool2d.params.push(AbiParam::new(ptr_ty));
    sig_pool2d.params.push(AbiParam::new(ptr_ty));
    for _ in 0..7 {
        sig_pool2d.params.push(AbiParam::new(types::I64));
    }
    // wukong_rmsnorm_bwd_f32[_parallel](x, dy, gamma, dx: ptr, rows, cols, eps_bits: i64) — fused
    // batched RMSNorm input-gradient (4 ptr + 3 i64, void).
    let mut sig_rmsnorm_bwd = Signature::new(call_conv);
    for _ in 0..4 {
        sig_rmsnorm_bwd.params.push(AbiParam::new(ptr_ty));
    }
    for _ in 0..3 {
        sig_rmsnorm_bwd.params.push(AbiParam::new(types::I64));
    }
    // wukong_i8gemm_nt[_parallel](a, b, c: ptr, m, k, n: i64) — int8 quantized nn.Linear (void).
    let mut sig_i8gemm = Signature::new(call_conv);
    for _ in 0..3 {
        sig_i8gemm.params.push(AbiParam::new(ptr_ty));
    }
    for _ in 0..3 {
        sig_i8gemm.params.push(AbiParam::new(types::I64));
    }
    // wukong_attention_f32(q, k, v, out: ptr, s, d: i64, scale: f32, causal: i64) — fused SDPA (void).
    let mut sig_attn = Signature::new(call_conv);
    for _ in 0..4 {
        sig_attn.params.push(AbiParam::new(ptr_ty));
    }
    sig_attn.params.push(AbiParam::new(types::I64));
    sig_attn.params.push(AbiParam::new(types::I64));
    sig_attn.params.push(AbiParam::new(types::F32));
    sig_attn.params.push(AbiParam::new(types::I64));
    // wukong_i8gemm_nt_deq[_parallel](a, b, out: ptr, m, k, n: i64, scale_a: f32, scale_b, bias: ptr,
    // act: i64) — int8 quantized nn.Linear with fused dequant epilogue (void).
    let mut sig_i8gemm_deq = Signature::new(call_conv);
    for _ in 0..3 {
        sig_i8gemm_deq.params.push(AbiParam::new(ptr_ty));
    }
    for _ in 0..3 {
        sig_i8gemm_deq.params.push(AbiParam::new(types::I64));
    }
    sig_i8gemm_deq.params.push(AbiParam::new(types::F32));
    sig_i8gemm_deq.params.push(AbiParam::new(ptr_ty));
    sig_i8gemm_deq.params.push(AbiParam::new(ptr_ty));
    sig_i8gemm_deq.params.push(AbiParam::new(types::I64));
    // wukong_dequant_f32[_parallel](q, out: ptr, n: i64, scale: f32, op: i64) — int-input dequant (void).
    let mut sig_dequant = Signature::new(call_conv);
    sig_dequant.params.push(AbiParam::new(ptr_ty));
    sig_dequant.params.push(AbiParam::new(ptr_ty));
    sig_dequant.params.push(AbiParam::new(types::I64));
    sig_dequant.params.push(AbiParam::new(types::F32));
    sig_dequant.params.push(AbiParam::new(types::I64));
    // wukong_dequant_perchan_f32[_parallel](q, out: ptr, rows, cols: i64, scale: ptr, op: i64) — per-
    // channel dequant (void).
    let mut sig_dequant_perchan = Signature::new(call_conv);
    sig_dequant_perchan.params.push(AbiParam::new(ptr_ty));
    sig_dequant_perchan.params.push(AbiParam::new(ptr_ty));
    sig_dequant_perchan.params.push(AbiParam::new(types::I64));
    sig_dequant_perchan.params.push(AbiParam::new(types::I64));
    sig_dequant_perchan.params.push(AbiParam::new(ptr_ty));
    sig_dequant_perchan.params.push(AbiParam::new(types::I64));
    // wukong_dot_bf16(x, y: ptr, n: i64) -> f32 — bf16 mixed-precision dot (f32 accumulate).
    let mut sig_dot_bf16 = Signature::new(call_conv);
    sig_dot_bf16.params.push(AbiParam::new(ptr_ty));
    sig_dot_bf16.params.push(AbiParam::new(ptr_ty));
    sig_dot_bf16.params.push(AbiParam::new(types::I64));
    sig_dot_bf16.returns.push(AbiParam::new(types::F32));
    // wukong_sum_bf16(x: ptr, n: i64) -> f32 — bf16 mixed-precision sum (f32 accumulate).
    let mut sig_sum_bf16 = Signature::new(call_conv);
    sig_sum_bf16.params.push(AbiParam::new(ptr_ty));
    sig_sum_bf16.params.push(AbiParam::new(types::I64));
    sig_sum_bf16.returns.push(AbiParam::new(types::F32));
    // wukong_reduce_bf16(x: ptr, n: i64, op: i64) -> f32 — bf16 max/min/absmax reduction.
    let mut sig_reduce_bf16 = Signature::new(call_conv);
    sig_reduce_bf16.params.push(AbiParam::new(ptr_ty));
    sig_reduce_bf16.params.push(AbiParam::new(types::I64));
    sig_reduce_bf16.params.push(AbiParam::new(types::I64));
    sig_reduce_bf16.returns.push(AbiParam::new(types::F32));
    // wukong_f32_to_f16_bits(f32) -> i32 (low 16 = f16 bits); wukong_f16_bits_to_f32(i32) -> f32.
    let mut sig_f32_to_f16 = Signature::new(call_conv);
    sig_f32_to_f16.params.push(AbiParam::new(types::F32));
    sig_f32_to_f16.returns.push(AbiParam::new(types::I32));
    let mut sig_f16_to_f32 = Signature::new(call_conv);
    sig_f16_to_f32.params.push(AbiParam::new(types::I32));
    sig_f16_to_f32.returns.push(AbiParam::new(types::F32));
    // wukong_axpby_bf16(x, y: ptr<bf16>, out: ptr<f32>, n: i64, a, b: f32) — bf16→f32 axpby. Void.
    let mut sig_axpby_bf16 = Signature::new(call_conv);
    sig_axpby_bf16.params.push(AbiParam::new(ptr_ty));
    sig_axpby_bf16.params.push(AbiParam::new(ptr_ty));
    sig_axpby_bf16.params.push(AbiParam::new(ptr_ty));
    sig_axpby_bf16.params.push(AbiParam::new(types::I64));
    sig_axpby_bf16.params.push(AbiParam::new(types::F32));
    sig_axpby_bf16.params.push(AbiParam::new(types::F32));
    // wukong_rt_fmod_f64(a, b) -> f64 and the f32 variant — true fmod backing float `%`.
    let mut sig_fmod_f64 = Signature::new(call_conv);
    sig_fmod_f64.params.push(AbiParam::new(types::F64));
    sig_fmod_f64.params.push(AbiParam::new(types::F64));
    sig_fmod_f64.returns.push(AbiParam::new(types::F64));
    let mut sig_fmod_f32 = Signature::new(call_conv);
    sig_fmod_f32.params.push(AbiParam::new(types::F32));
    sig_fmod_f32.params.push(AbiParam::new(types::F32));
    sig_fmod_f32.returns.push(AbiParam::new(types::F32));
    // wukong_rt_print_str(ptr) — render a NUL-terminated string buffer. Void.
    let mut sig_print_str = Signature::new(call_conv);
    sig_print_str.params.push(AbiParam::new(ptr_ty));
    // wukong_rt_alloc(count, elem_size, elem_is_float: i64) -> ptr — zeroed heap allocation
    // backing the `alloc_<T>(n)` builtins; wukong_rt_free(ptr) releases it. Void free.
    let mut sig_rt_alloc = Signature::new(call_conv);
    for _ in 0..3 {
        sig_rt_alloc.params.push(AbiParam::new(types::I64));
    }
    sig_rt_alloc.returns.push(AbiParam::new(ptr_ty));
    let mut sig_rt_free = Signature::new(call_conv);
    sig_rt_free.params.push(AbiParam::new(ptr_ty));
    // wukong_rt_{read,write}_{f32,i32,i64,u8}(path: ptr, data: ptr, len: i64) -> i64 — the file-I/O
    // intrinsics. Identical ABI across all eight dtypes/directions, so one signature declares them.
    let mut sig_rt_fileio = Signature::new(call_conv);
    sig_rt_fileio.params.push(AbiParam::new(ptr_ty));
    sig_rt_fileio.params.push(AbiParam::new(ptr_ty));
    sig_rt_fileio.params.push(AbiParam::new(types::I64));
    sig_rt_fileio.returns.push(AbiParam::new(types::I64));
    // wukong_now_ns() -> i64 — the monotonic process-lifetime nanosecond clock backing `now_ns()`.
    // No parameters; returns the elapsed nanoseconds as an i64.
    let mut sig_now_ns = Signature::new(call_conv);
    sig_now_ns.returns.push(AbiParam::new(types::I64));
    // wukong_rt_bounds_fail(index: i64, len: i64, site: ptr)
    let mut sig_bounds_fail = Signature::new(call_conv);
    sig_bounds_fail.params.push(AbiParam::new(types::I64));
    sig_bounds_fail.params.push(AbiParam::new(types::I64));
    sig_bounds_fail.params.push(AbiParam::new(ptr_ty));
    // wukong_rt_check_fail(kind: i64, a: i64, b: i64, site: ptr)
    let mut sig_check_fail = Signature::new(call_conv);
    for _ in 0..3 {
        sig_check_fail.params.push(AbiParam::new(types::I64));
    }
    sig_check_fail.params.push(AbiParam::new(ptr_ty));
    // wukong_rt_assert(cond: i64, site: ptr, message: ptr) -- either pointer may be null
    let mut sig_assert = Signature::new(call_conv);
    sig_assert.params.push(AbiParam::new(types::I64));
    sig_assert.params.push(AbiParam::new(ptr_ty));
    sig_assert.params.push(AbiParam::new(ptr_ty));
    let rt = RtFuncs {
        print_i64: module
            .declare_function(RT_PRINT_I64, Linkage::Import, &sig_i)
            .map_err(|e| e.to_string())?,
        print_u64: module
            .declare_function(RT_PRINT_U64, Linkage::Import, &sig_i)
            .map_err(|e| e.to_string())?,
        print_f64: module
            .declare_function(RT_PRINT_F64, Linkage::Import, &sig_f)
            .map_err(|e| e.to_string())?,
        print_str: module
            .declare_function(RT_PRINT_STR, Linkage::Import, &sig_print_str)
            .map_err(|e| e.to_string())?,
        assert: module
            .declare_function(RT_ASSERT, Linkage::Import, &sig_assert)
            .map_err(|e| e.to_string())?,
        assert_failed: module
            .declare_function(RT_ASSERT_FAILED, Linkage::Import, &sig_now_ns)
            .map_err(|e| e.to_string())?,
        bounds_fail: module
            .declare_function(RT_BOUNDS_FAIL, Linkage::Import, &sig_bounds_fail)
            .map_err(|e| e.to_string())?,
        check_fail: module
            .declare_function(RT_CHECK_FAIL, Linkage::Import, &sig_check_fail)
            .map_err(|e| e.to_string())?,
        rt_alloc: module
            .declare_function(RT_ALLOC, Linkage::Import, &sig_rt_alloc)
            .map_err(|e| e.to_string())?,
        rt_free: module
            .declare_function(RT_FREE, Linkage::Import, &sig_rt_free)
            .map_err(|e| e.to_string())?,
        rt_scratch_alloc: module
            .declare_function(RT_SCRATCH_ALLOC, Linkage::Import, &sig_rt_alloc)
            .map_err(|e| e.to_string())?,
        // () -> i64, the `now_ns` shape.
        rt_scratch_mark: module
            .declare_function(RT_SCRATCH_MARK, Linkage::Import, &sig_now_ns)
            .map_err(|e| e.to_string())?,
        // (i64) -> void, the `print_i64` shape.
        rt_scratch_release: module
            .declare_function(RT_SCRATCH_RELEASE, Linkage::Import, &sig_i)
            .map_err(|e| e.to_string())?,
        rt_read_f32: module
            .declare_function(RT_READ_F32, Linkage::Import, &sig_rt_fileio)
            .map_err(|e| e.to_string())?,
        rt_read_f64: module
            .declare_function(RT_READ_F64, Linkage::Import, &sig_rt_fileio)
            .map_err(|e| e.to_string())?,
        rt_read_i32: module
            .declare_function(RT_READ_I32, Linkage::Import, &sig_rt_fileio)
            .map_err(|e| e.to_string())?,
        rt_read_i64: module
            .declare_function(RT_READ_I64, Linkage::Import, &sig_rt_fileio)
            .map_err(|e| e.to_string())?,
        rt_read_i8: module
            .declare_function(RT_READ_I8, Linkage::Import, &sig_rt_fileio)
            .map_err(|e| e.to_string())?,
        rt_read_u8: module
            .declare_function(RT_READ_U8, Linkage::Import, &sig_rt_fileio)
            .map_err(|e| e.to_string())?,
        rt_write_f32: module
            .declare_function(RT_WRITE_F32, Linkage::Import, &sig_rt_fileio)
            .map_err(|e| e.to_string())?,
        rt_write_f64: module
            .declare_function(RT_WRITE_F64, Linkage::Import, &sig_rt_fileio)
            .map_err(|e| e.to_string())?,
        rt_write_i32: module
            .declare_function(RT_WRITE_I32, Linkage::Import, &sig_rt_fileio)
            .map_err(|e| e.to_string())?,
        rt_write_i64: module
            .declare_function(RT_WRITE_I64, Linkage::Import, &sig_rt_fileio)
            .map_err(|e| e.to_string())?,
        rt_write_i8: module
            .declare_function(RT_WRITE_I8, Linkage::Import, &sig_rt_fileio)
            .map_err(|e| e.to_string())?,
        rt_write_u8: module
            .declare_function(RT_WRITE_U8, Linkage::Import, &sig_rt_fileio)
            .map_err(|e| e.to_string())?,
        now_ns: module
            .declare_function(RT_NOW_NS, Linkage::Import, &sig_now_ns)
            .map_err(|e| e.to_string())?,
        parallel_for: module
            .declare_function(RT_PARALLEL_FOR, Linkage::Import, &sig_par)
            .map_err(|e| e.to_string())?,
        sgemm: module
            .declare_function(RT_SGEMM, Linkage::Import, &sig_gemm)
            .map_err(|e| e.to_string())?,
        sgemm_parallel: module
            .declare_function(RT_SGEMM_PARALLEL, Linkage::Import, &sig_gemm)
            .map_err(|e| e.to_string())?,
        sgemm_nt: module
            .declare_function(RT_SGEMM_NT, Linkage::Import, &sig_gemm)
            .map_err(|e| e.to_string())?,
        sgemm_nt_parallel: module
            .declare_function(RT_SGEMM_NT_PARALLEL, Linkage::Import, &sig_gemm)
            .map_err(|e| e.to_string())?,
        sgemm_tn: module
            .declare_function(RT_SGEMM_TN, Linkage::Import, &sig_gemm)
            .map_err(|e| e.to_string())?,
        sgemm_tn_parallel: module
            .declare_function(RT_SGEMM_TN_PARALLEL, Linkage::Import, &sig_gemm)
            .map_err(|e| e.to_string())?,
        sgemm_bf16_nt: module
            .declare_function(RT_SGEMM_BF16_NT, Linkage::Import, &sig_gemm)
            .map_err(|e| e.to_string())?,
        sgemm_bf16_nt_parallel: module
            .declare_function(RT_SGEMM_BF16_NT_PARALLEL, Linkage::Import, &sig_gemm)
            .map_err(|e| e.to_string())?,
        sgemm_f16_nt: module
            .declare_function(RT_SGEMM_F16_NT, Linkage::Import, &sig_gemm)
            .map_err(|e| e.to_string())?,
        sgemm_f16_nt_parallel: module
            .declare_function(RT_SGEMM_F16_NT_PARALLEL, Linkage::Import, &sig_gemm)
            .map_err(|e| e.to_string())?,
        sgemm_bf16_tn: module
            .declare_function(RT_SGEMM_BF16_TN, Linkage::Import, &sig_gemm)
            .map_err(|e| e.to_string())?,
        sgemm_bf16_tn_parallel: module
            .declare_function(RT_SGEMM_BF16_TN_PARALLEL, Linkage::Import, &sig_gemm)
            .map_err(|e| e.to_string())?,
        sgemm_f16_tn: module
            .declare_function(RT_SGEMM_F16_TN, Linkage::Import, &sig_gemm)
            .map_err(|e| e.to_string())?,
        sgemm_f16_tn_parallel: module
            .declare_function(RT_SGEMM_F16_TN_PARALLEL, Linkage::Import, &sig_gemm)
            .map_err(|e| e.to_string())?,
        sgemm_nt_epi: module
            .declare_function(RT_SGEMM_NT_EPI, Linkage::Import, &sig_gemm_epi)
            .map_err(|e| e.to_string())?,
        sgemm_nt_epi_par: module
            .declare_function(RT_SGEMM_NT_EPI_PAR, Linkage::Import, &sig_gemm_epi)
            .map_err(|e| e.to_string())?,
        sgemv: module
            .declare_function(RT_SGEMV, Linkage::Import, &sig_gemv)
            .map_err(|e| e.to_string())?,
        sgemv_par: module
            .declare_function(RT_SGEMV_PAR, Linkage::Import, &sig_gemv)
            .map_err(|e| e.to_string())?,
        sgemv_alpha: module
            .declare_function(RT_SGEMV_ALPHA, Linkage::Import, &sig_gemv_alpha)
            .map_err(|e| e.to_string())?,
        sgemv_alpha_par: module
            .declare_function(RT_SGEMV_ALPHA_PAR, Linkage::Import, &sig_gemv_alpha)
            .map_err(|e| e.to_string())?,
        gevm: module
            .declare_function(RT_SGEVM, Linkage::Import, &sig_gemv_alpha)
            .map_err(|e| e.to_string())?,
        gevm_par: module
            .declare_function(RT_SGEVM_PAR, Linkage::Import, &sig_gemv_alpha)
            .map_err(|e| e.to_string())?,
        sgemm_nt_alpha: module
            .declare_function(RT_SGEMM_NT_ALPHA, Linkage::Import, &sig_gemm_alpha)
            .map_err(|e| e.to_string())?,
        sgemm_nt_alpha_par: module
            .declare_function(RT_SGEMM_NT_ALPHA_PAR, Linkage::Import, &sig_gemm_alpha)
            .map_err(|e| e.to_string())?,
        sgemm_bf16_nt_epi: module
            .declare_function(RT_SGEMM_BF16_NT_EPI, Linkage::Import, &sig_gemm_epi)
            .map_err(|e| e.to_string())?,
        sgemm_bf16_nt_epi_par: module
            .declare_function(RT_SGEMM_BF16_NT_EPI_PAR, Linkage::Import, &sig_gemm_epi)
            .map_err(|e| e.to_string())?,
        sgemm_f16_nt_epi: module
            .declare_function(RT_SGEMM_F16_NT_EPI, Linkage::Import, &sig_gemm_epi)
            .map_err(|e| e.to_string())?,
        sgemm_f16_nt_epi_par: module
            .declare_function(RT_SGEMM_F16_NT_EPI_PAR, Linkage::Import, &sig_gemm_epi)
            .map_err(|e| e.to_string())?,
        vmath: module
            .declare_function(RT_VMATH, Linkage::Import, &sig_vmath)
            .map_err(|e| e.to_string())?,
        trig_f32: module
            .declare_function(RT_TRIG_F32, Linkage::Import, &sig_trig)
            .map_err(|e| e.to_string())?,
        math_f64: module
            .declare_function(RT_MATH_F64, Linkage::Import, &sig_math_f64)
            .map_err(|e| e.to_string())?,
        math2_f64: module
            .declare_function(RT_MATH2_F64, Linkage::Import, &sig_math2_f64)
            .map_err(|e| e.to_string())?,
        vmath_par: module
            .declare_function(RT_VMATH_PARALLEL, Linkage::Import, &sig_vmath)
            .map_err(|e| e.to_string())?,
        vmath2: module
            .declare_function(RT_VMATH2, Linkage::Import, &sig_vmath2)
            .map_err(|e| e.to_string())?,
        softmax_bwd: module
            .declare_function(RT_SOFTMAX_BWD, Linkage::Import, &sig_vmath2)
            .map_err(|e| e.to_string())?,
        softmax_bwd_par: module
            .declare_function(RT_SOFTMAX_BWD_PAR, Linkage::Import, &sig_vmath2)
            .map_err(|e| e.to_string())?,
        lrscan: module
            .declare_function(RT_LRSCAN, Linkage::Import, &sig_vmath2)
            .map_err(|e| e.to_string())?,
        lrscan_par: module
            .declare_function(RT_LRSCAN_PAR, Linkage::Import, &sig_vmath2)
            .map_err(|e| e.to_string())?,
        rmsnorm_bwd: module
            .declare_function(RT_RMSNORM_BWD, Linkage::Import, &sig_rmsnorm_bwd)
            .map_err(|e| e.to_string())?,
        rmsnorm_bwd_par: module
            .declare_function(RT_RMSNORM_BWD_PAR, Linkage::Import, &sig_rmsnorm_bwd)
            .map_err(|e| e.to_string())?,
        layernorm_bwd: module
            .declare_function(RT_LAYERNORM_BWD, Linkage::Import, &sig_rmsnorm_bwd)
            .map_err(|e| e.to_string())?,
        layernorm_bwd_par: module
            .declare_function(RT_LAYERNORM_BWD_PAR, Linkage::Import, &sig_rmsnorm_bwd)
            .map_err(|e| e.to_string())?,
        xent: module
            .declare_function(RT_XENT, Linkage::Import, &sig_vmath2)
            .map_err(|e| e.to_string())?,
        xent_par: module
            .declare_function(RT_XENT_PAR, Linkage::Import, &sig_vmath2)
            .map_err(|e| e.to_string())?,
        xent_bwd: module
            .declare_function(RT_XENT_BWD, Linkage::Import, &sig_vmath2)
            .map_err(|e| e.to_string())?,
        xent_bwd_par: module
            .declare_function(RT_XENT_BWD_PAR, Linkage::Import, &sig_vmath2)
            .map_err(|e| e.to_string())?,
        rope: module
            .declare_function(RT_ROPE, Linkage::Import, &sig_vmath2)
            .map_err(|e| e.to_string())?,
        rope_par: module
            .declare_function(RT_ROPE_PAR, Linkage::Import, &sig_vmath2)
            .map_err(|e| e.to_string())?,
        rope_bwd: module
            .declare_function(RT_ROPE_BWD, Linkage::Import, &sig_vmath2)
            .map_err(|e| e.to_string())?,
        rope_bwd_par: module
            .declare_function(RT_ROPE_BWD_PAR, Linkage::Import, &sig_vmath2)
            .map_err(|e| e.to_string())?,
        logsumexp: module
            .declare_function(RT_LOGSUMEXP, Linkage::Import, &sig_vmath)
            .map_err(|e| e.to_string())?,
        logsumexp_par: module
            .declare_function(RT_LOGSUMEXP_PAR, Linkage::Import, &sig_vmath)
            .map_err(|e| e.to_string())?,
        kldiv: module
            .declare_function(RT_KLDIV, Linkage::Import, &sig_vmath2)
            .map_err(|e| e.to_string())?,
        kldiv_par: module
            .declare_function(RT_KLDIV_PAR, Linkage::Import, &sig_vmath2)
            .map_err(|e| e.to_string())?,
        entropy: module
            .declare_function(RT_ENTROPY, Linkage::Import, &sig_vmath)
            .map_err(|e| e.to_string())?,
        entropy_par: module
            .declare_function(RT_ENTROPY_PAR, Linkage::Import, &sig_vmath)
            .map_err(|e| e.to_string())?,
        kd_loss: module
            .declare_function(RT_KD_LOSS, Linkage::Import, &sig_vmath2)
            .map_err(|e| e.to_string())?,
        kd_loss_par: module
            .declare_function(RT_KD_LOSS_PAR, Linkage::Import, &sig_vmath2)
            .map_err(|e| e.to_string())?,
        rowargmax: module
            .declare_function(RT_ROWARGMAX, Linkage::Import, &sig_vmath)
            .map_err(|e| e.to_string())?,
        rowargmax_par: module
            .declare_function(RT_ROWARGMAX_PAR, Linkage::Import, &sig_vmath)
            .map_err(|e| e.to_string())?,
        rowargmin: module
            .declare_function(RT_ROWARGMIN, Linkage::Import, &sig_vmath)
            .map_err(|e| e.to_string())?,
        rowargmin_par: module
            .declare_function(RT_ROWARGMIN_PAR, Linkage::Import, &sig_vmath)
            .map_err(|e| e.to_string())?,
        colargmax: module
            .declare_function(RT_COLARGMAX, Linkage::Import, &sig_vmath)
            .map_err(|e| e.to_string())?,
        colargmax_par: module
            .declare_function(RT_COLARGMAX_PAR, Linkage::Import, &sig_vmath)
            .map_err(|e| e.to_string())?,
        colargmin: module
            .declare_function(RT_COLARGMIN, Linkage::Import, &sig_vmath)
            .map_err(|e| e.to_string())?,
        colargmin_par: module
            .declare_function(RT_COLARGMIN_PAR, Linkage::Import, &sig_vmath)
            .map_err(|e| e.to_string())?,
        cumsum: module
            .declare_function(RT_CUMSUM, Linkage::Import, &sig_vmath)
            .map_err(|e| e.to_string())?,
        cumsum_par: module
            .declare_function(RT_CUMSUM_PAR, Linkage::Import, &sig_vmath)
            .map_err(|e| e.to_string())?,
        cumprod: module
            .declare_function(RT_CUMPROD, Linkage::Import, &sig_vmath)
            .map_err(|e| e.to_string())?,
        cumprod_par: module
            .declare_function(RT_CUMPROD_PAR, Linkage::Import, &sig_vmath)
            .map_err(|e| e.to_string())?,
        cummax: module
            .declare_function(RT_CUMMAX, Linkage::Import, &sig_vmath)
            .map_err(|e| e.to_string())?,
        cummax_par: module
            .declare_function(RT_CUMMAX_PAR, Linkage::Import, &sig_vmath)
            .map_err(|e| e.to_string())?,
        cummin: module
            .declare_function(RT_CUMMIN, Linkage::Import, &sig_vmath)
            .map_err(|e| e.to_string())?,
        cummin_par: module
            .declare_function(RT_CUMMIN_PAR, Linkage::Import, &sig_vmath)
            .map_err(|e| e.to_string())?,
        maxpool2d: module
            .declare_function(RT_MAXPOOL2D, Linkage::Import, &sig_pool2d)
            .map_err(|e| e.to_string())?,
        maxpool2d_par: module
            .declare_function(RT_MAXPOOL2D_PAR, Linkage::Import, &sig_pool2d)
            .map_err(|e| e.to_string())?,
        avgpool2d: module
            .declare_function(RT_AVGPOOL2D, Linkage::Import, &sig_pool2d)
            .map_err(|e| e.to_string())?,
        avgpool2d_par: module
            .declare_function(RT_AVGPOOL2D_PAR, Linkage::Import, &sig_pool2d)
            .map_err(|e| e.to_string())?,
        // bf16/f16-input twins: identical (ptr, ptr, i64, i64) signature.
        vmath_bf16: module
            .declare_function(RT_VMATH_BF16, Linkage::Import, &sig_vmath)
            .map_err(|e| e.to_string())?,
        vmath_f16: module
            .declare_function(RT_VMATH_F16, Linkage::Import, &sig_vmath)
            .map_err(|e| e.to_string())?,
        // Half-output activation twins (bf16/f16 in AND out): same (ptr, ptr, i64, i64) signature.
        vmath_bf16_out: module
            .declare_function(RT_VMATH_BF16_OUT, Linkage::Import, &sig_vmath)
            .map_err(|e| e.to_string())?,
        vmath_f16_out: module
            .declare_function(RT_VMATH_F16_OUT, Linkage::Import, &sig_vmath)
            .map_err(|e| e.to_string())?,
        transpose: module
            .declare_function(RT_TRANSPOSE, Linkage::Import, &sig_vmath)
            .map_err(|e| e.to_string())?,
        transpose_par: module
            .declare_function(RT_TRANSPOSE_PAR, Linkage::Import, &sig_vmath)
            .map_err(|e| e.to_string())?,
        transpose_u16: module
            .declare_function(RT_TRANSPOSE_U16, Linkage::Import, &sig_vmath)
            .map_err(|e| e.to_string())?,
        transpose_u16_par: module
            .declare_function(RT_TRANSPOSE_U16_PAR, Linkage::Import, &sig_vmath)
            .map_err(|e| e.to_string())?,
        colsum: module
            .declare_function(RT_COLSUM, Linkage::Import, &sig_vmath)
            .map_err(|e| e.to_string())?,
        colsum_par: module
            .declare_function(RT_COLSUM_PAR, Linkage::Import, &sig_vmath)
            .map_err(|e| e.to_string())?,
        colmax: module
            .declare_function(RT_COLMAX, Linkage::Import, &sig_vmath)
            .map_err(|e| e.to_string())?,
        colmax_par: module
            .declare_function(RT_COLMAX_PAR, Linkage::Import, &sig_vmath)
            .map_err(|e| e.to_string())?,
        colmin: module
            .declare_function(RT_COLMIN, Linkage::Import, &sig_vmath)
            .map_err(|e| e.to_string())?,
        colmin_par: module
            .declare_function(RT_COLMIN_PAR, Linkage::Import, &sig_vmath)
            .map_err(|e| e.to_string())?,
        colmaxabs: module
            .declare_function(RT_COLMAXABS, Linkage::Import, &sig_vmath)
            .map_err(|e| e.to_string())?,
        colmaxabs_par: module
            .declare_function(RT_COLMAXABS_PAR, Linkage::Import, &sig_vmath)
            .map_err(|e| e.to_string())?,
        colmean: module
            .declare_function(RT_COLMEAN, Linkage::Import, &sig_vmath)
            .map_err(|e| e.to_string())?,
        colmean_par: module
            .declare_function(RT_COLMEAN_PAR, Linkage::Import, &sig_vmath)
            .map_err(|e| e.to_string())?,
        colsumsq: module
            .declare_function(RT_COLSUMSQ, Linkage::Import, &sig_vmath)
            .map_err(|e| e.to_string())?,
        colsumsq_par: module
            .declare_function(RT_COLSUMSQ_PAR, Linkage::Import, &sig_vmath)
            .map_err(|e| e.to_string())?,
        coll2: module
            .declare_function(RT_COLL2, Linkage::Import, &sig_vmath)
            .map_err(|e| e.to_string())?,
        coll2_par: module
            .declare_function(RT_COLL2_PAR, Linkage::Import, &sig_vmath)
            .map_err(|e| e.to_string())?,
        colrms: module
            .declare_function(RT_COLRMS, Linkage::Import, &sig_vmath)
            .map_err(|e| e.to_string())?,
        colrms_par: module
            .declare_function(RT_COLRMS_PAR, Linkage::Import, &sig_vmath)
            .map_err(|e| e.to_string())?,
        velem: module
            .declare_function(RT_VELEM, Linkage::Import, &sig_velem)
            .map_err(|e| e.to_string())?,
        velem_par: module
            .declare_function(RT_VELEM_PARALLEL, Linkage::Import, &sig_velem)
            .map_err(|e| e.to_string())?,
        vhorner: module
            .declare_function(RT_VHORNER, Linkage::Import, &sig_vhorner)
            .map_err(|e| e.to_string())?,
        bias_bcast: module
            .declare_function(RT_BIAS_BCAST, Linkage::Import, &sig_bias_bcast)
            .map_err(|e| e.to_string())?,
        bias_bcast_par: module
            .declare_function(RT_BIAS_BCAST_PAR, Linkage::Import, &sig_bias_bcast)
            .map_err(|e| e.to_string())?,
        sred: module
            .declare_function(RT_SREDUCE, Linkage::Import, &sig_sreduce)
            .map_err(|e| e.to_string())?,
        sred_par: module
            .declare_function(RT_SREDUCE_PARALLEL, Linkage::Import, &sig_sreduce)
            .map_err(|e| e.to_string())?,
        argreduce: module
            .declare_function(RT_ARGREDUCE, Linkage::Import, &sig_argreduce)
            .map_err(|e| e.to_string())?,
        argreduce_par: module
            .declare_function(RT_ARGREDUCE_PARALLEL, Linkage::Import, &sig_argreduce)
            .map_err(|e| e.to_string())?,
        norm: module
            .declare_function(RT_NORM, Linkage::Import, &sig_norm)
            .map_err(|e| e.to_string())?,
        norm_par: module
            .declare_function(RT_NORM_PARALLEL, Linkage::Import, &sig_norm)
            .map_err(|e| e.to_string())?,
        norm_affine: module
            .declare_function(RT_NORM_AFFINE, Linkage::Import, &sig_norm_affine)
            .map_err(|e| e.to_string())?,
        norm_affine_par: module
            .declare_function(RT_NORM_AFFINE_PARALLEL, Linkage::Import, &sig_norm_affine)
            .map_err(|e| e.to_string())?,
        i8nt: module
            .declare_function(RT_I8GEMM_NT, Linkage::Import, &sig_i8gemm)
            .map_err(|e| e.to_string())?,
        i8nt_par: module
            .declare_function(RT_I8GEMM_NT_PARALLEL, Linkage::Import, &sig_i8gemm)
            .map_err(|e| e.to_string())?,
        i8nt_deq: module
            .declare_function(RT_I8GEMM_NT_DEQ, Linkage::Import, &sig_i8gemm_deq)
            .map_err(|e| e.to_string())?,
        i8nt_deq_par: module
            .declare_function(RT_I8GEMM_NT_DEQ_PARALLEL, Linkage::Import, &sig_i8gemm_deq)
            .map_err(|e| e.to_string())?,
        dequant: module
            .declare_function(RT_DEQUANT, Linkage::Import, &sig_dequant)
            .map_err(|e| e.to_string())?,
        dequant_par: module
            .declare_function(RT_DEQUANT_PAR, Linkage::Import, &sig_dequant)
            .map_err(|e| e.to_string())?,
        dequant_perchan: module
            .declare_function(RT_DEQUANT_PERCHAN, Linkage::Import, &sig_dequant_perchan)
            .map_err(|e| e.to_string())?,
        dequant_perchan_par: module
            .declare_function(
                RT_DEQUANT_PERCHAN_PAR,
                Linkage::Import,
                &sig_dequant_perchan,
            )
            .map_err(|e| e.to_string())?,
        // Embedding lookup reuses the 3-ptr + 3-i64 void signature (the pointer element type is
        // irrelevant to the ABI — out/weight are f32*, ids is i32*).
        embedding: module
            .declare_function(RT_EMBEDDING, Linkage::Import, &sig_i8gemm)
            .map_err(|e| e.to_string())?,
        embedding_par: module
            .declare_function(RT_EMBEDDING_PAR, Linkage::Import, &sig_i8gemm)
            .map_err(|e| e.to_string())?,
        attention: module
            .declare_function(RT_ATTENTION, Linkage::Import, &sig_attn)
            .map_err(|e| e.to_string())?,
        scatter_add: module
            .declare_function(RT_SCATTER_ADD, Linkage::Import, &sig_i8gemm)
            .map_err(|e| e.to_string())?,
        scatter_add_par: module
            .declare_function(RT_SCATTER_ADD_PAR, Linkage::Import, &sig_i8gemm)
            .map_err(|e| e.to_string())?,
        axpby_bf16: module
            .declare_function(RT_AXPBY_BF16, Linkage::Import, &sig_axpby_bf16)
            .map_err(|e| e.to_string())?,
        axpby_bf16_out: module
            .declare_function(RT_AXPBY_BF16_OUT, Linkage::Import, &sig_axpby_bf16)
            .map_err(|e| e.to_string())?,
        axpby_f16_out: module
            .declare_function(RT_AXPBY_F16_OUT, Linkage::Import, &sig_axpby_bf16)
            .map_err(|e| e.to_string())?,
        dot_bf16: module
            .declare_function(RT_DOT_BF16, Linkage::Import, &sig_dot_bf16)
            .map_err(|e| e.to_string())?,
        sum_bf16: module
            .declare_function(RT_SUM_BF16, Linkage::Import, &sig_sum_bf16)
            .map_err(|e| e.to_string())?,
        reduce_bf16: module
            .declare_function(RT_REDUCE_BF16, Linkage::Import, &sig_reduce_bf16)
            .map_err(|e| e.to_string())?,
        // f16 streaming/reductions reuse the bf16 signatures (same shapes; only the widen differs).
        axpby_f16: module
            .declare_function(RT_AXPBY_F16, Linkage::Import, &sig_axpby_bf16)
            .map_err(|e| e.to_string())?,
        dot_f16: module
            .declare_function(RT_DOT_F16, Linkage::Import, &sig_dot_bf16)
            .map_err(|e| e.to_string())?,
        sum_f16: module
            .declare_function(RT_SUM_F16, Linkage::Import, &sig_sum_bf16)
            .map_err(|e| e.to_string())?,
        reduce_f16: module
            .declare_function(RT_REDUCE_F16, Linkage::Import, &sig_reduce_bf16)
            .map_err(|e| e.to_string())?,
        // The `@parallel` twins share the serial signatures (identical ABI; only the impl chunks).
        dot_bf16_par: module
            .declare_function(RT_DOT_BF16_PAR, Linkage::Import, &sig_dot_bf16)
            .map_err(|e| e.to_string())?,
        sum_bf16_par: module
            .declare_function(RT_SUM_BF16_PAR, Linkage::Import, &sig_sum_bf16)
            .map_err(|e| e.to_string())?,
        reduce_bf16_par: module
            .declare_function(RT_REDUCE_BF16_PAR, Linkage::Import, &sig_reduce_bf16)
            .map_err(|e| e.to_string())?,
        dot_f16_par: module
            .declare_function(RT_DOT_F16_PAR, Linkage::Import, &sig_dot_bf16)
            .map_err(|e| e.to_string())?,
        sum_f16_par: module
            .declare_function(RT_SUM_F16_PAR, Linkage::Import, &sig_sum_bf16)
            .map_err(|e| e.to_string())?,
        reduce_f16_par: module
            .declare_function(RT_REDUCE_F16_PAR, Linkage::Import, &sig_reduce_bf16)
            .map_err(|e| e.to_string())?,
        f32_to_f16: module
            .declare_function(RT_F32_TO_F16, Linkage::Import, &sig_f32_to_f16)
            .map_err(|e| e.to_string())?,
        f16_to_f32: module
            .declare_function(RT_F16_TO_F32, Linkage::Import, &sig_f16_to_f32)
            .map_err(|e| e.to_string())?,
        fmod_f64: module
            .declare_function(RT_FMOD_F64, Linkage::Import, &sig_fmod_f64)
            .map_err(|e| e.to_string())?,
        fmod_f32: module
            .declare_function(RT_FMOD_F32, Linkage::Import, &sig_fmod_f32)
            .map_err(|e| e.to_string())?,
    };

    // Module-wide symbol membership is also the arity-error guard. Per-function
    // signature imports are created lazily by FnTranslator::runtime_ref.
    let mut rt_ids: HashMap<&'static str, FuncId> = HashMap::from([
        (RT_PRINT_I64, rt.print_i64),
        (RT_PRINT_U64, rt.print_u64),
        (RT_PRINT_STR, rt.print_str),
        (RT_PRINT_F64, rt.print_f64),
        (RT_ASSERT, rt.assert),
        (RT_ASSERT_FAILED, rt.assert_failed),
        (RT_BOUNDS_FAIL, rt.bounds_fail),
        (RT_CHECK_FAIL, rt.check_fail),
        (RT_ALLOC, rt.rt_alloc),
        (RT_FREE, rt.rt_free),
        (RT_SCRATCH_ALLOC, rt.rt_scratch_alloc),
        (RT_SCRATCH_MARK, rt.rt_scratch_mark),
        (RT_SCRATCH_RELEASE, rt.rt_scratch_release),
        (RT_READ_F32, rt.rt_read_f32),
        (RT_READ_F64, rt.rt_read_f64),
        (RT_READ_I32, rt.rt_read_i32),
        (RT_READ_I64, rt.rt_read_i64),
        (RT_READ_I8, rt.rt_read_i8),
        (RT_READ_U8, rt.rt_read_u8),
        (RT_WRITE_F32, rt.rt_write_f32),
        (RT_WRITE_F64, rt.rt_write_f64),
        (RT_WRITE_I32, rt.rt_write_i32),
        (RT_WRITE_I64, rt.rt_write_i64),
        (RT_WRITE_I8, rt.rt_write_i8),
        (RT_WRITE_U8, rt.rt_write_u8),
        (RT_NOW_NS, rt.now_ns),
        (RT_PARALLEL_FOR, rt.parallel_for),
        (RT_SGEMM, rt.sgemm),
        (RT_SGEMM_PARALLEL, rt.sgemm_parallel),
        (RT_SGEMM_NT, rt.sgemm_nt),
        (RT_SGEMM_NT_PARALLEL, rt.sgemm_nt_parallel),
        (RT_SGEMM_TN, rt.sgemm_tn),
        (RT_SGEMM_TN_PARALLEL, rt.sgemm_tn_parallel),
        (RT_SGEMM_BF16_NT, rt.sgemm_bf16_nt),
        (RT_SGEMM_BF16_NT_PARALLEL, rt.sgemm_bf16_nt_parallel),
        (RT_SGEMM_F16_NT, rt.sgemm_f16_nt),
        (RT_SGEMM_F16_NT_PARALLEL, rt.sgemm_f16_nt_parallel),
        (RT_SGEMM_BF16_TN, rt.sgemm_bf16_tn),
        (RT_SGEMM_BF16_TN_PARALLEL, rt.sgemm_bf16_tn_parallel),
        (RT_SGEMM_F16_TN, rt.sgemm_f16_tn),
        (RT_SGEMM_F16_TN_PARALLEL, rt.sgemm_f16_tn_parallel),
        (RT_SGEMM_NT_EPI, rt.sgemm_nt_epi),
        (RT_SGEMM_NT_EPI_PAR, rt.sgemm_nt_epi_par),
        (RT_SGEMV, rt.sgemv),
        (RT_SGEMV_PAR, rt.sgemv_par),
        (RT_SGEMV_ALPHA, rt.sgemv_alpha),
        (RT_SGEMV_ALPHA_PAR, rt.sgemv_alpha_par),
        (RT_SGEVM, rt.gevm),
        (RT_SGEVM_PAR, rt.gevm_par),
        (RT_SGEMM_NT_ALPHA, rt.sgemm_nt_alpha),
        (RT_SGEMM_NT_ALPHA_PAR, rt.sgemm_nt_alpha_par),
        (RT_SGEMM_BF16_NT_EPI, rt.sgemm_bf16_nt_epi),
        (RT_SGEMM_BF16_NT_EPI_PAR, rt.sgemm_bf16_nt_epi_par),
        (RT_SGEMM_F16_NT_EPI, rt.sgemm_f16_nt_epi),
        (RT_SGEMM_F16_NT_EPI_PAR, rt.sgemm_f16_nt_epi_par),
        (RT_VMATH, rt.vmath),
        (RT_TRIG_F32, rt.trig_f32),
        (RT_MATH_F64, rt.math_f64),
        (RT_MATH2_F64, rt.math2_f64),
        (RT_VMATH_PARALLEL, rt.vmath_par),
        (RT_VMATH2, rt.vmath2),
        (RT_SOFTMAX_BWD, rt.softmax_bwd),
        (RT_SOFTMAX_BWD_PAR, rt.softmax_bwd_par),
        (RT_LRSCAN, rt.lrscan),
        (RT_LRSCAN_PAR, rt.lrscan_par),
        (RT_RMSNORM_BWD, rt.rmsnorm_bwd),
        (RT_RMSNORM_BWD_PAR, rt.rmsnorm_bwd_par),
        (RT_LAYERNORM_BWD, rt.layernorm_bwd),
        (RT_LAYERNORM_BWD_PAR, rt.layernorm_bwd_par),
        (RT_XENT, rt.xent),
        (RT_XENT_PAR, rt.xent_par),
        (RT_XENT_BWD, rt.xent_bwd),
        (RT_XENT_BWD_PAR, rt.xent_bwd_par),
        (RT_ROPE, rt.rope),
        (RT_ROPE_PAR, rt.rope_par),
        (RT_ROPE_BWD, rt.rope_bwd),
        (RT_ROPE_BWD_PAR, rt.rope_bwd_par),
        (RT_LOGSUMEXP, rt.logsumexp),
        (RT_LOGSUMEXP_PAR, rt.logsumexp_par),
        (RT_KLDIV, rt.kldiv),
        (RT_KLDIV_PAR, rt.kldiv_par),
        (RT_ENTROPY, rt.entropy),
        (RT_ENTROPY_PAR, rt.entropy_par),
        (RT_KD_LOSS, rt.kd_loss),
        (RT_KD_LOSS_PAR, rt.kd_loss_par),
        (RT_ROWARGMAX, rt.rowargmax),
        (RT_ROWARGMAX_PAR, rt.rowargmax_par),
        (RT_ROWARGMIN, rt.rowargmin),
        (RT_ROWARGMIN_PAR, rt.rowargmin_par),
        (RT_COLARGMAX, rt.colargmax),
        (RT_COLARGMAX_PAR, rt.colargmax_par),
        (RT_COLARGMIN, rt.colargmin),
        (RT_COLARGMIN_PAR, rt.colargmin_par),
        (RT_CUMSUM, rt.cumsum),
        (RT_CUMSUM_PAR, rt.cumsum_par),
        (RT_CUMPROD, rt.cumprod),
        (RT_CUMPROD_PAR, rt.cumprod_par),
        (RT_CUMMAX, rt.cummax),
        (RT_CUMMAX_PAR, rt.cummax_par),
        (RT_CUMMIN, rt.cummin),
        (RT_CUMMIN_PAR, rt.cummin_par),
        (RT_MAXPOOL2D, rt.maxpool2d),
        (RT_MAXPOOL2D_PAR, rt.maxpool2d_par),
        (RT_AVGPOOL2D, rt.avgpool2d),
        (RT_AVGPOOL2D_PAR, rt.avgpool2d_par),
        (RT_VMATH_BF16, rt.vmath_bf16),
        (RT_VMATH_F16, rt.vmath_f16),
        (RT_VMATH_BF16_OUT, rt.vmath_bf16_out),
        (RT_VMATH_F16_OUT, rt.vmath_f16_out),
        (RT_TRANSPOSE, rt.transpose),
        (RT_TRANSPOSE_PAR, rt.transpose_par),
        (RT_TRANSPOSE_U16, rt.transpose_u16),
        (RT_TRANSPOSE_U16_PAR, rt.transpose_u16_par),
        (RT_COLSUM, rt.colsum),
        (RT_COLSUM_PAR, rt.colsum_par),
        (RT_COLMAX, rt.colmax),
        (RT_COLMAX_PAR, rt.colmax_par),
        (RT_COLMIN, rt.colmin),
        (RT_COLMIN_PAR, rt.colmin_par),
        (RT_COLMAXABS, rt.colmaxabs),
        (RT_COLMAXABS_PAR, rt.colmaxabs_par),
        (RT_COLMEAN, rt.colmean),
        (RT_COLMEAN_PAR, rt.colmean_par),
        (RT_COLSUMSQ, rt.colsumsq),
        (RT_COLSUMSQ_PAR, rt.colsumsq_par),
        (RT_COLL2, rt.coll2),
        (RT_COLL2_PAR, rt.coll2_par),
        (RT_COLRMS, rt.colrms),
        (RT_COLRMS_PAR, rt.colrms_par),
        (RT_VELEM, rt.velem),
        (RT_VELEM_PARALLEL, rt.velem_par),
        (RT_VHORNER, rt.vhorner),
        (RT_BIAS_BCAST, rt.bias_bcast),
        (RT_BIAS_BCAST_PAR, rt.bias_bcast_par),
        (RT_SREDUCE, rt.sred),
        (RT_SREDUCE_PARALLEL, rt.sred_par),
        (RT_ARGREDUCE, rt.argreduce),
        (RT_ARGREDUCE_PARALLEL, rt.argreduce_par),
        (RT_NORM, rt.norm),
        (RT_NORM_PARALLEL, rt.norm_par),
        (RT_NORM_AFFINE, rt.norm_affine),
        (RT_NORM_AFFINE_PARALLEL, rt.norm_affine_par),
        (RT_I8GEMM_NT, rt.i8nt),
        (RT_I8GEMM_NT_PARALLEL, rt.i8nt_par),
        (RT_I8GEMM_NT_DEQ, rt.i8nt_deq),
        (RT_I8GEMM_NT_DEQ_PARALLEL, rt.i8nt_deq_par),
        (RT_DEQUANT, rt.dequant),
        (RT_DEQUANT_PAR, rt.dequant_par),
        (RT_DEQUANT_PERCHAN, rt.dequant_perchan),
        (RT_DEQUANT_PERCHAN_PAR, rt.dequant_perchan_par),
        (RT_EMBEDDING, rt.embedding),
        (RT_EMBEDDING_PAR, rt.embedding_par),
        (RT_ATTENTION, rt.attention),
        (RT_SCATTER_ADD, rt.scatter_add),
        (RT_SCATTER_ADD_PAR, rt.scatter_add_par),
        (RT_AXPBY_BF16, rt.axpby_bf16),
        (RT_AXPBY_BF16_OUT, rt.axpby_bf16_out),
        (RT_AXPBY_F16_OUT, rt.axpby_f16_out),
        (RT_DOT_BF16, rt.dot_bf16),
        (RT_SUM_BF16, rt.sum_bf16),
        (RT_REDUCE_BF16, rt.reduce_bf16),
        (RT_AXPBY_F16, rt.axpby_f16),
        (RT_DOT_F16, rt.dot_f16),
        (RT_SUM_F16, rt.sum_f16),
        (RT_REDUCE_F16, rt.reduce_f16),
        (RT_DOT_BF16_PAR, rt.dot_bf16_par),
        (RT_SUM_BF16_PAR, rt.sum_bf16_par),
        (RT_REDUCE_BF16_PAR, rt.reduce_bf16_par),
        (RT_DOT_F16_PAR, rt.dot_f16_par),
        (RT_SUM_F16_PAR, rt.sum_f16_par),
        (RT_REDUCE_F16_PAR, rt.reduce_f16_par),
        (RT_F32_TO_F16, rt.f32_to_f16),
        (RT_F16_TO_F32, rt.f16_to_f32),
        (RT_FMOD_F64, rt.fmod_f64),
        (RT_FMOD_F32, rt.fmod_f32),
    ]);
    ml_rt::declare(module, ptr_ty, call_conv, &mut rt_ids)?;
    text_rt::declare(module, ptr_ty, call_conv, &mut rt_ids)?;

    // Declare all user functions first so calls resolve regardless of definition order. An object
    // names and links them per `mode` (see `entry`): internal and mangled in an executable, so a
    // `.wk` function called `write` or `fmod` can neither collide with nor interpose on libc.
    let entry_fn = entry::entry_function(program, interner, mode, ptr_ty, call_conv)?;
    let entry_name = entry_fn.map(|f| f.name);
    let mut ids: HashMap<Symbol, FuncId> = HashMap::new();
    for f in &program.funcs {
        let sig = signature_of(f, ptr_ty, call_conv);
        let (name, linkage) =
            entry::user_symbol(interner.resolve(f.name), mode, Some(f.name) == entry_name);
        let id = module
            .declare_function(&name, linkage, &sig)
            .map_err(|e| e.to_string())?;
        ids.insert(f.name, id);
    }
    // `extern "C"` functions: imported under their C names with the C-ABI signature. They join
    // `ids` only while the bodies are lowered — so a call to one takes the ordinary user-call path
    // in `lower_call` — and leave it before `ids` is returned, since an import has no finalized
    // body to hand out. The JIT binds each one up front (`bind_extern_symbols`); an object leaves it
    // to the linker.
    for e in &program.externs {
        let sig = extern_signature(e, ptr_ty, call_conv);
        let id = module
            .declare_function(interner.resolve(e.name), Linkage::Import, &sig)
            .map_err(|err| err.to_string())?;
        ids.insert(e.name, id);
    }

    // Declare + define each read-only static-data blob (string literals). `writable = false` lands
    // them in the read-only data section (`.rodata`/`.rdata`); `Linkage::Local` keeps them internal.
    // The bytes include the trailing NUL, so the address is directly usable by `print_str`. This is
    // what gives a returned `*u8` a real static address instead of a reclaimed stack frame.
    let mut data_ids: HashMap<Symbol, DataId> = HashMap::new();
    for s in &program.statics {
        let name = interner.resolve(s.name);
        let id = module
            .declare_data(name, Linkage::Local, false, false)
            .map_err(|e| e.to_string())?;
        let mut desc = DataDescription::new();
        desc.define(s.bytes.clone().into_boxed_slice());
        module.define_data(id, &desc).map_err(|e| e.to_string())?;
        data_ids.insert(s.name, id);
    }

    // Declare and install the synthesized 256-bit AVX2 vector kernels as raw machine code. They are
    // function-local (a loop the vectorizer widened registers its recipe on that function), so we
    // walk every function's `vec_kernels` and give each a module-unique FuncId keyed by (function
    // index, kernel index). Each is a self-contained `fn(ptrs: *const *mut u8, scalars: *const f32,
    // n: u64)` under the module's default call conv (Win64 on this target), so ordinary Cranelift
    // `call`s from the owning function reach it. `define_function_bytes` copies the bytes into the
    // module's code region; the empty reloc slice reflects that the kernels reference no externals.
    let mut kernel_sig = Signature::new(call_conv);
    kernel_sig.params.push(AbiParam::new(ptr_ty)); // ptrs
    kernel_sig.params.push(AbiParam::new(ptr_ty)); // scalars
    kernel_sig.params.push(AbiParam::new(types::I64)); // n
    let mut kernel_ids: Vec<Vec<FuncId>> = Vec::with_capacity(program.funcs.len());
    for (fi, f) in program.funcs.iter().enumerate() {
        let mut per_fn: Vec<FuncId> = Vec::with_capacity(f.vec_kernels.len());
        for (ki, k) in f.vec_kernels.iter().enumerate() {
            let bytes = crate::avx2::assemble_kernel(k)
                .map_err(|e| format!("avx2 assemble vec kernel {fi}.{ki}: {e}"))?;
            // A reduction kernel returns its horizontal fold in its lane type (XMM0 for a float
            // lane, RAX/EAX for an integer one); an elementwise kernel is void.
            let mut sig = kernel_sig.clone();
            if k.reduce.is_some() {
                sig.returns
                    .push(AbiParam::new(crate::avx2::lane_clif_type(k.lane)));
            }
            let id = module
                .declare_function(
                    &format!("__wukong_veckernel_{fi}_{ki}"),
                    Linkage::Local,
                    &sig,
                )
                .map_err(|e| e.to_string())?;
            module
                .define_function_bytes(id, 16, &bytes, &[])
                .map_err(|e| format!("cranelift define vec kernel {fi}.{ki}: {e:?}"))?;
            if let Some(log) = log.as_deref_mut() {
                let extent = debuginfo::Extent {
                    size: bytes.len() as u32,
                    srclocs: Vec::new(),
                    cfi: None,
                };
                log.record(id, debuginfo::CodeOrigin::VecKernel(fi, ki), extent);
            }
            per_fn.push(id);
        }
        kernel_ids.push(per_fn);
    }

    if parallel && program.funcs.len() > 1 {
        define_functions_parallel(
            module,
            program,
            interner,
            ptr_ty,
            call_conv,
            &rt_ids,
            &ids,
            &data_ids,
            &kernel_ids,
            &asserting,
            captures.as_ref(),
            log.as_deref_mut(),
        )?;
    } else {
        define_functions_serial(
            module,
            program,
            interner,
            ptr_ty,
            call_conv,
            &rt_ids,
            &ids,
            &data_ids,
            &kernel_ids,
            &asserting,
            captures.as_ref(),
            log.as_deref_mut(),
        )?;
    }
    if let Some(main) = entry_name {
        entry::define_entry(module, ids[&main], ptr_ty, call_conv, log)?;
    }
    for e in &program.externs {
        ids.remove(&e.name);
    }
    Ok(ids)
}

/// Lower one MIR function to a complete Cranelift IR function (the instruction-selection input),
/// building into `clif`. The module-declarations dependency is a read-only [`Decls`] view, so
/// [`define_functions_parallel`] can build and compile functions concurrently. User symbols are
/// imported in MIR order and runtime signatures on first use; both paths use the same ordering.
/// `Decls`' shims replicate `Module::declare_{func,data}_in_func` verbatim.
#[allow(clippy::too_many_arguments)]
fn build_function_clif(
    module: &Decls,
    f: &Function,
    fi: usize,
    interner: &Interner,
    ptr_ty: Type,
    call_conv: cranelift_codegen::isa::CallConv,
    rt_ids: &HashMap<&'static str, FuncId>,
    ids: &HashMap<Symbol, FuncId>,
    data_ids: &HashMap<Symbol, DataId>,
    kernel_ids: &[Vec<FuncId>],
    fbctx: &mut FunctionBuilderContext,
    clif: &mut cranelift_codegen::ir::Function,
    scratch: &mut TranslationScratch,
) -> Result<(), NativeFailure> {
    // A function with a float-to-`i1`-vector cast is scalarized here, on its own codegen worker;
    // every other function is borrowed as is (see `bool_vectors::legalize`). A decline names no
    // instruction, so it is reported at the function's declaration.
    let legalized = bool_vectors::legalize(f, interner).map_err(|message| at_decl(f, message))?;
    let f: &Function = &legalized;
    clif.signature = signature_of(f, ptr_ty, call_conv);
    let TranslationScratch {
        vmap,
        blocks,
        func_refs,
        data_refs,
        rt_refs,
        kernel_refs,
        call_args,
        branch_args,
    } = scratch;
    vmap.clear();
    vmap.resize(f.value_types.len(), None);
    blocks.clear();
    func_refs.clear();
    data_refs.clear();
    rt_refs.clear();
    kernel_refs.clear();
    call_args.clear();
    branch_args.clear();
    {
        {
            let mut builder = FunctionBuilder::new(clif, fbctx);

            // Import only referenced symbols, in deterministic MIR instruction order. Importing
            // every function and static into every DFG grows quadratically with the module size,
            // even when each function only calls one other function. Keep declarations module-wide
            // so forward references, recursion, and address-taken parallel helpers still resolve.
            for inst in f.blocks.iter().flat_map(|block| &block.insts) {
                match &inst.op {
                    Op::Call { func, .. } | Op::FuncAddr(func) => {
                        if let Some(&id) = ids.get(func) {
                            func_refs
                                .entry(*func)
                                .or_insert_with(|| module.declare_func_in_func(id, builder.func));
                        }
                    }
                    Op::GlobalAddr(sym) => {
                        data_refs.entry(*sym).or_insert_with(|| {
                            module.declare_data_in_func(data_ids[sym], builder.func)
                        });
                    }
                    _ => {}
                }
            }
            // This function's own vector-kernel FuncRefs, indexed by `Op::VecKernelCall`'s `kernel`.
            kernel_refs.extend(
                kernel_ids[fi]
                    .iter()
                    .map(|&kid| module.declare_func_in_func(kid, builder.func)),
            );
            blocks.extend(f.blocks.iter().map(|_| builder.create_block()));
            let alias = wukong_opt::AliasInfo::analyze(f);
            let slot_plan = module
                .coloring
                .map(|cfg| stack_color::SlotPlan::new(f, cfg, &alias))
                .unwrap_or_default();
            let mut t = FnTranslator {
                builder,
                func: f,
                interner,
                ptr_ty,
                vmap,
                blocks,
                func_refs,
                decls: module,
                rt_ids,
                rt_refs,
                call_args,
                branch_args,
                data_refs,
                kernel_refs,
                alias,
                layout_err: None,
                cur_loc: None,
                err_site: None,
                slot_plan,
                color_slots: Vec::new(),
            };
            t.translate();
            // `layout_err` carries every unrecoverable lowering failure — a type whose byte layout
            // overflows `u32`, and a runtime kernel that reached `lower_call` with an arity no
            // dispatch arm accepts — recorded during lowering rather than panicking. Abort now,
            // before this half-built function is finalized/defined (the `?` at each
            // `populate_module` call site surfaces it as `error: <msg>`, exit 1), at the source
            // construct whose value was being lowered (`err_site`; a failure in a terminator is
            // located here instead).
            if let Some(message) = t.layout_err.take() {
                let site = t.err_site.or_else(|| t.current_site());
                return Err(NativeFailure { message, site });
            }
            t.builder.finalize();
        }
    }
    Ok(())
}

/// A failure that belongs to `f` as a whole rather than to one of its instructions (a declined
/// legalization, a function Cranelift rejects): reported at `f`'s declaration.
fn at_decl(f: &Function, message: String) -> NativeFailure {
    NativeFailure {
        message,
        site: f.decl_span(),
    }
}

/// A read-only view of a [`Module`]'s declarations, exposing exactly the two ref-import shims the
/// per-function lowering needs — replicated **verbatim** from `cranelift_module::Module`'s default
/// methods (which `ObjectModule`/`JITModule` do not override) so the built CLIF is byte-identical.
/// Because it borrows the declarations `&`, not the module `&mut`, it is `Sync` and can be shared
/// across the rayon workers in [`define_functions_parallel`].
struct Decls<'a> {
    d: &'a cranelift_module::ModuleDeclarations,
    asserting: &'a HashSet<Symbol>,
    /// What stack-slot coloring needs when it is on ([`stack_color`]).
    coloring: Option<stack_color::ColorConfig<'a>>,
    /// Whether the target ISA has SSSE3 (`pshufb`), which decides how `Op::Shuffle` is lowered.
    ssse3: bool,
}

/// The stack-coloring configuration for code generated by `isa`: on when the program's capture
/// summary was built, mixing alias regions only where Cranelift runs no alias analysis
/// (`opt_level=none`, the source `-O0`; see [`stack_color`]).
fn color_config<'a>(
    captures: Option<&'a wukong_opt::CaptureSummary>,
    isa: &dyn cranelift_codegen::isa::TargetIsa,
) -> Option<stack_color::ColorConfig<'a>> {
    captures.map(|calls| stack_color::ColorConfig {
        calls,
        mix_regions: isa.flags().opt_level() == settings::OptLevel::None,
    })
}

/// Does `isa` have SSSE3? Read off the ISA's own flags, so `--target-cpu` levels and host detection
/// answer the same question the instruction selector asks.
fn isa_has_ssse3(isa: &dyn cranelift_codegen::isa::TargetIsa) -> bool {
    cranelift_codegen::isa::TargetIsa::isa_flags(isa)
        .iter()
        .any(|v| v.name == "has_ssse3" && v.as_bool() == Some(true))
}

impl Decls<'_> {
    /// Mirror of `Module::declare_func_in_func`.
    fn declare_func_in_func(
        &self,
        func_id: FuncId,
        func: &mut cranelift_codegen::ir::Function,
    ) -> FuncRef {
        let decl = self.d.get_function_decl(func_id);
        let signature = func.import_signature(decl.signature.clone());
        let user_name_ref =
            func.declare_imported_user_function(cranelift_codegen::ir::UserExternalName {
                namespace: 0,
                index: func_id.as_u32(),
            });
        let colocated = decl.linkage.is_final();
        func.import_function(cranelift_codegen::ir::ExtFuncData {
            name: cranelift_codegen::ir::ExternalName::user(user_name_ref),
            signature,
            colocated,
        })
    }

    /// Mirror of `Module::declare_data_in_func`.
    fn declare_data_in_func(
        &self,
        data: DataId,
        func: &mut cranelift_codegen::ir::Function,
    ) -> GlobalValue {
        let decl = self.d.get_data_decl(data);
        let colocated = decl.linkage.is_final();
        let user_name_ref =
            func.declare_imported_user_function(cranelift_codegen::ir::UserExternalName {
                namespace: 1,
                index: data.as_u32(),
            });
        func.create_global_value(cranelift_codegen::ir::GlobalValueData::Symbol {
            name: cranelift_codegen::ir::ExternalName::user(user_name_ref),
            offset: cranelift_codegen::ir::immediates::Imm64::new(0),
            colocated,
            tls: decl.tls,
        })
    }
}

/// Serial backend: lower, compile and define one function at a time in source order. Limit the
/// declarations borrow to lowering so the module can then be mutated, and reuse the Cranelift
/// context instead of retaining a module's worth of CLIF before defining its first function.
#[allow(clippy::too_many_arguments)]
fn define_functions_serial<M: Module>(
    module: &mut M,
    program: &Program,
    interner: &Interner,
    ptr_ty: Type,
    call_conv: cranelift_codegen::isa::CallConv,
    rt_ids: &HashMap<&'static str, FuncId>,
    ids: &HashMap<Symbol, FuncId>,
    data_ids: &HashMap<Symbol, DataId>,
    kernel_ids: &[Vec<FuncId>],
    asserting: &HashSet<Symbol>,
    captures: Option<&wukong_opt::CaptureSummary>,
    mut log: Option<&mut debuginfo::CodeLog>,
) -> Result<(), NativeFailure> {
    let mut ctx = module.make_context();
    let mut fbctx = FunctionBuilderContext::new();
    let mut scratch = TranslationScratch::default();
    let ssse3 = isa_has_ssse3(module.isa());
    for (fi, f) in program.funcs.iter().enumerate() {
        {
            let decls = Decls {
                d: module.declarations(),
                asserting,
                coloring: color_config(captures, module.isa()),
                ssse3,
            };
            build_function_clif(
                &decls,
                f,
                fi,
                interner,
                ptr_ty,
                call_conv,
                rt_ids,
                ids,
                data_ids,
                kernel_ids,
                &mut fbctx,
                &mut ctx.func,
                &mut scratch,
            )?;
        }
        module
            .define_function(ids[&f.name], &mut ctx)
            .map_err(|e| at_decl(f, format!("cranelift define fn: {e:?}")))?;
        #[cfg(test)]
        srcloc_tests::capture(f.name, &ctx);
        if let Some(log) = log.as_deref_mut() {
            let code = ctx
                .compiled_code()
                .expect("a defined function has machine code");
            let origin = debuginfo::CodeOrigin::User(fi);
            log.record_compiled(ids[&f.name], origin, code, module.isa());
        }
        module.clear_context(&mut ctx);
    }
    Ok(())
}

/// Parallel backend: build + compile each function concurrently (independent compilation units,
/// shared read-only ISA + [`Decls`]), then define them into the object **in source order**. The
/// object bytes are identical to [`define_functions_serial`] because (a) each function's machine
/// code is a deterministic function of its CLIF + the ISA (default `ControlPlane`, no chaos), and
/// (b) `define_function_bytes` records them in the same order with the same declared `FuncId`s.
/// `WUKONG_PAR_CODEGEN=0` disables this path. Proven equal by `serial_parallel_object_bytes_identical`.
#[allow(clippy::too_many_arguments)]
fn define_functions_parallel<M: Module>(
    module: &mut M,
    program: &Program,
    interner: &Interner,
    ptr_ty: Type,
    call_conv: cranelift_codegen::isa::CallConv,
    rt_ids: &HashMap<&'static str, FuncId>,
    ids: &HashMap<Symbol, FuncId>,
    data_ids: &HashMap<Symbol, DataId>,
    kernel_ids: &[Vec<FuncId>],
    asserting: &HashSet<Symbol>,
    captures: Option<&wukong_opt::CaptureSummary>,
    mut log: Option<&mut debuginfo::CodeLog>,
) -> Result<(), NativeFailure> {
    use cranelift_module::ModuleReloc;
    use rayon::prelude::*;

    // What each worker reports for the code log (`None`: nothing is being logged); `Some(true)`
    // also takes the srcloc ranges and the call-frame information.
    let want_log = log.as_deref().map(|l| l.srclocs);
    // Each worker owns its FunctionBuilderContext + Cranelift Context; `Decls` and the ISA are shared
    // immutable borrows of the module (both `Sync`). Produces owned bytes + relocs so nothing borrows
    // the module past the parallel region.
    // One worker's output: the function, its alignment, bytes and relocations, and what the code
    // log records of it.
    type Compiled = (
        FuncId,
        u64,
        Vec<u8>,
        Vec<ModuleReloc>,
        Option<debuginfo::Extent>,
    );
    let compiled: Vec<Compiled> = {
        let decls = Decls {
            d: module.declarations(),
            asserting,
            coloring: color_config(captures, module.isa()),
            ssse3: isa_has_ssse3(module.isa()),
        };
        let isa = module.isa();
        program
            .funcs
            .par_iter()
            .enumerate()
            .map_init(
                || {
                    (
                        FunctionBuilderContext::new(),
                        cranelift_codegen::Context::new(),
                        TranslationScratch::default(),
                    )
                },
                |(fbctx, ctx, scratch), (fi, f)| -> Result<_, NativeFailure> {
                    let fid = ids[&f.name];
                    ctx.clear();
                    if let Err(error) = build_function_clif(
                        &decls,
                        f,
                        fi,
                        interner,
                        ptr_ty,
                        call_conv,
                        rt_ids,
                        ids,
                        data_ids,
                        kernel_ids,
                        fbctx,
                        &mut ctx.func,
                        scratch,
                    ) {
                        // Lowering errors return before FunctionBuilder::finalize, which normally
                        // clears its context. Positional error collection can visit another
                        // function with this same iterator state, so discard that incomplete SSA
                        // state before reuse. The next invocation clears ctx and scratch as usual.
                        *fbctx = FunctionBuilderContext::new();
                        return Err(error);
                    }
                    let mut cp = cranelift_codegen::control::ControlPlane::default();
                    ctx.compile(isa, &mut cp)
                        .map_err(|e| at_decl(f, format!("cranelift compile fn: {:?}", e.inner)))?;
                    let cc = ctx.compiled_code().unwrap();
                    let alignment = cc.buffer.alignment as u64;
                    let bytes = cc.buffer.data().to_vec();
                    let relocs: Vec<ModuleReloc> = cc
                        .buffer
                        .relocs()
                        .iter()
                        .map(|r| ModuleReloc::from_mach_reloc(r, &ctx.func, fid))
                        .collect();
                    let extent =
                        want_log.map(|full| debuginfo::compiled_extent(cc, full.then_some(isa)));
                    Ok((fid, alignment, bytes, relocs, extent))
                },
            )
            // Collect the per-function `Result`s **positionally**, then take the first `Err` in
            // source order. Collecting straight into `Result<Vec<_>, _>` reports whichever worker
            // failed first in wall-clock time, so a program with two bad functions produced a
            // different diagnostic run to run — `two_huge.wk` (an oversized array in `a()` and
            // another in `b()`) alternated between the `[... x i32]` and `[... x i64]` messages
            // across identical invocations, while the serial path always reported `a()`. Indexing
            // the errors makes the parallel path report the same function the serial path does, so
            // the compiler is a pure function of its input (U3).
            .collect::<Vec<Result<_, NativeFailure>>>()
            .into_iter()
            .collect::<Result<Vec<_>, NativeFailure>>()?
    };

    for (fi, ((fid, alignment, bytes, relocs, extent), f)) in
        compiled.into_iter().zip(&program.funcs).enumerate()
    {
        module
            .define_function_bytes(fid, alignment, &bytes, &relocs)
            .map_err(|e| at_decl(f, format!("cranelift define fn bytes: {e:?}")))?;
        if let (Some(log), Some(extent)) = (log.as_deref_mut(), extent) {
            log.record(fid, debuginfo::CodeOrigin::User(fi), extent);
        }
    }
    Ok(())
}

/// A JIT-compiled program: holds the executable memory and a pointer to the entry point. Compile
/// once with [`jit_compile`], then [`JitProgram::run`] (capturing output, for correctness) or
/// [`JitProgram::call`] (raw, for timing) as many times as needed.
pub struct JitProgram {
    module: Option<cranelift_jit::JITModule>,
    code: *const u8,
    ret: MirType,
}

impl JitProgram {
    /// Invoke the entry point, transmuting to the right ABI for its return type. Caller must hold
    /// the run lock (so the shared capture buffer isn't raced).
    unsafe fn invoke(&self) -> i64 {
        invoke_code(self.code as usize, &self.ret)
    }

    /// Run once, returning the exit code and captured stdout (the native counterpart to the
    /// interpreter's `run_with_output`).
    pub fn run(&self) -> Result<(i64, Vec<u8>), String> {
        let (result, out) = self.run_capturing();
        result.map(|code| (code, out))
    }

    /// Run once, returning the exit code or the run's failure **together with the stdout captured
    /// before the run stopped** — the native counterpart to the interpreter's `run_capturing`. A
    /// failed `assert` or slice bounds check returns at once through every generated caller, so the
    /// output is exactly what the program printed before the failing statement.
    pub fn run_capturing(&self) -> (Result<i64, String>, Vec<u8>) {
        let _guard = RUN_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        OUTPUT.lock().unwrap_or_else(|e| e.into_inner()).clear();
        reset_failure();
        // Run the JIT'd entry on a worker thread with a large stack. The native code recurses on
        // the host call stack (one machine frame per Wukong call), so a deeply recursive program
        // would overflow the default ~8 MiB main-thread stack and *abort* the process — while the
        // interpreter oracle, whose calls never touch the host stack, completes. Without the
        // headroom the two backends diverge on deep recursion (a stack
        // overflow vs a correct result), breaking the differential gate. Only the code pointer
        // (passed as a `usize`, since a raw pointer isn't `Send`) and the return type cross the
        // boundary; the `OUTPUT`/`ASSERT_FAILED` statics the JIT'd `rt_*` calls touch are global.
        let code = self.code as usize;
        let ret = self.ret.clone();
        let exit_code = run_on_big_stack(move || unsafe { invoke_code(code, &ret) });
        let out = OUTPUT.lock().unwrap_or_else(|e| e.into_inner()).clone();
        (run_outcome(exit_code), out)
    }

    /// Run once with stdout **streamed** to the process's stdout as it is produced — what
    /// `wukongc --run --backend=native` does, the native twin of `wukong_interp::run_streaming`:
    /// `line_buffered` writes at every newline (an interactive terminal), otherwise in 64 KiB blocks.
    /// Everything still buffered is written out before this returns, also when the run stopped on a
    /// runtime failure, which is returned as the error. Same stack and failure contract as
    /// [`JitProgram::run_capturing`]; the bytes are exactly the ones it would have captured.
    pub fn run_streaming(&self, line_buffered: bool) -> Result<i64, String> {
        let _guard = RUN_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        OUTPUT.lock().unwrap_or_else(|e| e.into_inner()).clear();
        reset_failure();
        let stream = StreamGuard::on(line_buffered);
        let code = self.code as usize;
        let ret = self.ret.clone();
        let exit_code = run_on_big_stack(move || unsafe { invoke_code(code, &ret) });
        drop(stream);
        run_outcome(exit_code)
    }

    /// Run once for timing: clears (but does not clone) the capture buffer so repeated prints don't
    /// grow memory, and returns just the exit code (1 on assertion failure).
    ///
    /// **Stack contract — differs from [`JitProgram::run`].** This runs the JIT'd entry on the
    /// *caller's* stack, deliberately: spawning a thread per iteration would land inside the
    /// measured region and swamp a short kernel. The caller therefore owns the recursion headroom.
    /// On the default ~8 MiB main-thread stack a deeply recursive program (the class
    /// `run_on_big_stack` exists for) overflows and aborts the whole process, where the same source
    /// under `wukongc --run --backend=native` completes. A timing harness must wrap its entire
    /// warm-up + measurement loop in one `std::thread::Builder::new().stack_size(..)` worker so the
    /// spawn is paid once, outside the timed region.
    pub fn call(&self) -> i64 {
        let _guard = RUN_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        OUTPUT.lock().unwrap_or_else(|e| e.into_inner()).clear();
        reset_failure();
        // This entry runs on the caller's thread, which keeps its scratch arena between calls: start
        // every call at scratch mark 0, as a fresh run (and the interpreter) does.
        wukong_runtime::scratch_reset();
        let result = unsafe { self.invoke() };
        // A failure exits 1, as the process would; `exit(code)` exits `code`.
        run_outcome(result).unwrap_or(1)
    }
}

impl Drop for JitProgram {
    fn drop(&mut self) {
        if let Some(m) = self.module.take() {
            // SAFETY: no JIT'd code from this module runs after the handle is dropped.
            unsafe { m.free_memory() };
        }
    }
}

/// Invoke a JIT entry's code pointer (passed as a `usize` so it can cross a thread boundary into the
/// big-stack worker), transmuting to the ABI implied by its return type. Mirrors `JitProgram::invoke`.
///
/// # Safety
/// `code` must be a finalized function pointer of the ABI implied by `ret`, valid for the call.
unsafe fn invoke_code(code: usize, ret: &MirType) -> i64 {
    let code = code as *const u8;
    match ret {
        MirType::Void => {
            let f: extern "C" fn() = std::mem::transmute(code);
            f();
            0
        }
        MirType::F64 => {
            let f: extern "C" fn() -> f64 = std::mem::transmute(code);
            f() as i64
        }
        // The narrow floats must NOT share the `f64` arm. `signature_of` returns whatever
        // `cl_type` says, and `cl_type` computes `f16`/`bf16` in `f32` registers, so all three
        // return a *single-precision* XMM0: `movss` leaves the value in XMM0[31:0] and zeroes
        // [127:32]. Reading those 64 bits as an `f64` reinterprets a live float as a denormal —
        // `fn main() -> f32 { return 42.9; }` (XMM0[31:0] = 0x422b_999a) came out as 5.4e-315,
        // which `as i64` saturates to 0, while the interpreter oracle exits 42.
        MirType::F32 | MirType::F16 | MirType::BF16 => {
            let f: extern "C" fn() -> f32 = std::mem::transmute(code);
            f() as i64
        }
        MirType::I64 => {
            let f: extern "C" fn() -> i64 = std::mem::transmute(code);
            f()
        }
        _ => {
            let f: extern "C" fn() -> i32 = std::mem::transmute(code);
            f() as i64
        }
    }
}

/// Run `f` on a worker thread with a large (512 MiB) stack, re-raising any panic on the caller so
/// behavior is otherwise identical to a direct call. This gives JIT'd Wukong recursion headroom
/// comparable to the interpreter's stackless call frames — without it the native backend overflows
/// the host's ~8 MiB default stack and aborts the process where the interpreter completes, a
/// divergence the differential gate would otherwise miss (it only re-runs the interpreter across opt
/// levels). `thread::scope` lets the worker borrow non-`'static` captures.
fn run_on_big_stack<T: Send>(f: impl FnOnce() -> T + Send) -> T {
    const STACK: usize = 512 * 1024 * 1024;
    std::thread::scope(|s| {
        let handle = std::thread::Builder::new()
            .name("wukong-native".into())
            .stack_size(STACK)
            .spawn_scoped(s, f)
            .expect("spawn native worker thread");
        match handle.join() {
            Ok(v) => v,
            Err(payload) => std::panic::resume_unwind(payload),
        }
    })
}

/// JIT-compile `program`, returning a callable handle to `entry`. `entry` must name a function that
/// takes **no parameters** — a parameterized entry is an `Err`, not a wrong call. Functions compile
/// concurrently, exactly as for `emit_object` (see [`default_parallel`]), with the optimizing
/// backend configuration ([`BackendOpt::default`]).
pub fn jit_compile(
    program: &Program,
    entry: Symbol,
    interner: &Interner,
) -> Result<JitProgram, String> {
    jit_compile_with(program, entry, interner, BackendOpt::default())
}

/// As [`jit_compile`], generating code for `target` instead of the full host feature set — so the
/// code an `--emit=obj --target-cpu=<level>` object would contain can be *run* on this machine. The
/// host must implement every feature of the level (a JIT cannot run code its CPU faults on).
pub fn jit_compile_for(
    program: &Program,
    entry: Symbol,
    interner: &Interner,
    target: TargetCpu,
) -> Result<JitProgram, String> {
    jit_compile_for_with(program, entry, interner, target, BackendOpt::default())
        .map_err(String::from)
}

/// As [`jit_compile`], with an explicit backend configuration (the driver passes the source `-O`
/// level, so `-O0` compiles with Cranelift `opt_level=none`).
pub fn jit_compile_with(
    program: &Program,
    entry: Symbol,
    interner: &Interner,
    backend: BackendOpt,
) -> Result<JitProgram, String> {
    jit_compile_for_with(program, entry, interner, TargetCpu::Native, backend).map_err(String::from)
}

/// Both [`jit_compile_for`]'s instruction-set level and [`jit_compile_with`]'s backend
/// configuration — what the driver uses for `--run --backend=native`, which honours both
/// `--target-cpu` and `-O`. A failure carries the source construct it was raised on
/// ([`NativeFailure::site`]).
pub fn jit_compile_for_with(
    program: &Program,
    entry: Symbol,
    interner: &Interner,
    target: TargetCpu,
    backend: BackendOpt,
) -> Result<JitProgram, NativeFailure> {
    let missing = target::missing_on_host(target);
    if !missing.is_empty() {
        return Err(format!(
            "unsupported target CPU `{}` for the JIT: this host lacks {}; use --emit=obj/exe to \
             build for it",
            target.name(),
            missing.join(", ")
        )
        .into());
    }
    jit_compile_entry(
        program,
        entry,
        interner,
        false,
        target,
        backend,
        default_parallel(),
    )
}

// The parameterized handle remains private inside JitBufferKernel: exposing it as JitProgram
// would let the safe no-argument run API invoke a function with the wrong ABI.
fn jit_compile_entry(
    program: &Program,
    entry: Symbol,
    interner: &Interner,
    buffer_dispatch: bool,
    target: TargetCpu,
    backend: BackendOpt,
    parallel: bool,
) -> Result<JitProgram, NativeFailure> {
    jit_compile_entry_mapped(
        program,
        entry,
        interner,
        buffer_dispatch,
        target,
        backend,
        parallel,
        debuginfo::perf_map::from_env().as_deref(),
    )
}

/// [`jit_compile_entry`], appending a line per compiled function to the perf map `perf_map` (see
/// [`debuginfo::perf_map`]) when it is `Some`.
#[allow(clippy::too_many_arguments)]
fn jit_compile_entry_mapped(
    program: &Program,
    entry: Symbol,
    interner: &Interner,
    buffer_dispatch: bool,
    target: TargetCpu,
    backend: BackendOpt,
    parallel: bool,
    perf_map: Option<&std::path::Path>,
) -> Result<JitProgram, NativeFailure> {
    use cranelift_jit::{JITBuilder, JITModule};

    target::check_vec_kernels(program, interner, target)?;
    let isa = make_isa_full(false, default_verify(), target, backend)?;
    let mut builder = JITBuilder::with_isa(isa, cranelift_module::default_libcall_names());
    builder.symbol(RT_PRINT_I64, rt_print_i64 as *const u8);
    builder.symbol(RT_PRINT_U64, rt_print_u64 as *const u8);
    builder.symbol(RT_PRINT_F64, rt_print_f64 as *const u8);
    builder.symbol(RT_PRINT_STR, rt_print_str as *const u8);
    builder.symbol(RT_ASSERT, rt_assert as *const u8);
    builder.symbol(RT_ASSERT_FAILED, rt_assert_failed as *const u8);
    builder.symbol(RT_BOUNDS_FAIL, rt_bounds_fail as *const u8);
    builder.symbol(RT_CHECK_FAIL, rt_check_fail as *const u8);
    builder.symbol(RT_ALLOC, wukong_runtime::wukong_rt_alloc as *const u8);
    builder.symbol(RT_FREE, wukong_runtime::wukong_rt_free as *const u8);
    builder.symbol(
        RT_SCRATCH_ALLOC,
        wukong_runtime::wukong_rt_scratch_alloc as *const u8,
    );
    builder.symbol(
        RT_SCRATCH_MARK,
        wukong_runtime::wukong_rt_scratch_mark as *const u8,
    );
    builder.symbol(
        RT_SCRATCH_RELEASE,
        wukong_runtime::wukong_rt_scratch_release as *const u8,
    );
    builder.symbol(RT_READ_F32, wukong_runtime::wukong_rt_read_f32 as *const u8);
    builder.symbol(RT_READ_F64, wukong_runtime::wukong_rt_read_f64 as *const u8);
    builder.symbol(RT_READ_I32, wukong_runtime::wukong_rt_read_i32 as *const u8);
    builder.symbol(RT_READ_I64, wukong_runtime::wukong_rt_read_i64 as *const u8);
    builder.symbol(RT_READ_I8, wukong_runtime::wukong_rt_read_i8 as *const u8);
    builder.symbol(RT_READ_U8, wukong_runtime::wukong_rt_read_u8 as *const u8);
    builder.symbol(
        RT_WRITE_F32,
        wukong_runtime::wukong_rt_write_f32 as *const u8,
    );
    builder.symbol(
        RT_WRITE_F64,
        wukong_runtime::wukong_rt_write_f64 as *const u8,
    );
    builder.symbol(
        RT_WRITE_I32,
        wukong_runtime::wukong_rt_write_i32 as *const u8,
    );
    builder.symbol(
        RT_WRITE_I64,
        wukong_runtime::wukong_rt_write_i64 as *const u8,
    );
    builder.symbol(RT_WRITE_I8, wukong_runtime::wukong_rt_write_i8 as *const u8);
    builder.symbol(RT_WRITE_U8, wukong_runtime::wukong_rt_write_u8 as *const u8);
    // The ML builtins' entries (the `ml_rt` table), and the text builtins' (the `text_rt` table).
    ml_rt::bind_jit(&mut builder);
    text_rt::bind_jit(&mut builder);
    builder.symbol(RT_NOW_NS, wukong_runtime::wukong_now_ns as *const u8);
    // The guarded run's parallel-for: the runtime's, wrapped in the cancellation ABI that brings a
    // failing region body's diagnostic back to its caller in iteration order (`rt_parallel_for`).
    builder.symbol(RT_PARALLEL_FOR, rt_parallel_for as *const u8);
    builder.symbol(RT_SGEMM, wukong_runtime::wukong_sgemm as *const u8);
    builder.symbol(
        RT_SGEMM_PARALLEL,
        wukong_runtime::wukong_sgemm_parallel as *const u8,
    );
    builder.symbol(RT_SGEMM_NT, wukong_runtime::wukong_sgemm_nt as *const u8);
    builder.symbol(
        RT_SGEMM_NT_PARALLEL,
        wukong_runtime::wukong_sgemm_nt_parallel as *const u8,
    );
    builder.symbol(RT_SGEMM_TN, wukong_runtime::wukong_sgemm_tn as *const u8);
    builder.symbol(
        RT_SGEMM_TN_PARALLEL,
        wukong_runtime::wukong_sgemm_tn_parallel as *const u8,
    );
    builder.symbol(
        RT_SGEMM_BF16_NT,
        wukong_runtime::wukong_sgemm_bf16_nt as *const u8,
    );
    builder.symbol(
        RT_SGEMM_BF16_NT_PARALLEL,
        wukong_runtime::wukong_sgemm_bf16_nt_parallel as *const u8,
    );
    builder.symbol(
        RT_SGEMM_F16_NT,
        wukong_runtime::wukong_sgemm_f16_nt as *const u8,
    );
    builder.symbol(
        RT_SGEMM_F16_NT_PARALLEL,
        wukong_runtime::wukong_sgemm_f16_nt_parallel as *const u8,
    );
    builder.symbol(
        RT_SGEMM_BF16_TN,
        wukong_runtime::wukong_sgemm_bf16_tn as *const u8,
    );
    builder.symbol(
        RT_SGEMM_BF16_TN_PARALLEL,
        wukong_runtime::wukong_sgemm_bf16_tn_parallel as *const u8,
    );
    builder.symbol(
        RT_SGEMM_F16_TN,
        wukong_runtime::wukong_sgemm_f16_tn as *const u8,
    );
    builder.symbol(
        RT_SGEMM_F16_TN_PARALLEL,
        wukong_runtime::wukong_sgemm_f16_tn_parallel as *const u8,
    );
    builder.symbol(
        RT_SGEMM_NT_EPI,
        wukong_runtime::wukong_sgemm_nt_epi as *const u8,
    );
    builder.symbol(
        RT_SGEMM_NT_EPI_PAR,
        wukong_runtime::wukong_sgemm_nt_epi_parallel as *const u8,
    );
    builder.symbol(RT_SGEMV, wukong_runtime::wukong_sgemv as *const u8);
    builder.symbol(
        RT_SGEMV_PAR,
        wukong_runtime::wukong_sgemv_parallel as *const u8,
    );
    builder.symbol(
        RT_SGEMV_ALPHA,
        wukong_runtime::wukong_sgemv_alpha as *const u8,
    );
    builder.symbol(
        RT_SGEMV_ALPHA_PAR,
        wukong_runtime::wukong_sgemv_alpha_parallel as *const u8,
    );
    builder.symbol(RT_SGEVM, wukong_runtime::wukong_sgevm_f32 as *const u8);
    builder.symbol(
        RT_SGEVM_PAR,
        wukong_runtime::wukong_sgevm_f32_parallel as *const u8,
    );
    builder.symbol(
        RT_SGEMM_NT_ALPHA,
        wukong_runtime::wukong_sgemm_nt_alpha as *const u8,
    );
    builder.symbol(
        RT_SGEMM_NT_ALPHA_PAR,
        wukong_runtime::wukong_sgemm_nt_alpha_parallel as *const u8,
    );
    builder.symbol(
        RT_SGEMM_BF16_NT_EPI,
        wukong_runtime::wukong_sgemm_bf16_nt_epi as *const u8,
    );
    builder.symbol(
        RT_SGEMM_BF16_NT_EPI_PAR,
        wukong_runtime::wukong_sgemm_bf16_nt_epi_parallel as *const u8,
    );
    builder.symbol(
        RT_SGEMM_F16_NT_EPI,
        wukong_runtime::wukong_sgemm_f16_nt_epi as *const u8,
    );
    builder.symbol(
        RT_SGEMM_F16_NT_EPI_PAR,
        wukong_runtime::wukong_sgemm_f16_nt_epi_parallel as *const u8,
    );
    builder.symbol(RT_VMATH, wukong_runtime::wukong_vmath_f32 as *const u8);
    builder.symbol(RT_TRIG_F32, wukong_runtime::wukong_rt_trig_f32 as *const u8);
    builder.symbol(RT_MATH_F64, wukong_runtime::wukong_rt_math_f64 as *const u8);
    builder.symbol(
        RT_MATH2_F64,
        wukong_runtime::wukong_rt_math2_f64 as *const u8,
    );
    builder.symbol(
        RT_VMATH_PARALLEL,
        wukong_runtime::wukong_vmath_f32_parallel as *const u8,
    );
    builder.symbol(RT_VMATH2, wukong_runtime::wukong_vmath2_f32 as *const u8);
    builder.symbol(
        RT_SOFTMAX_BWD,
        wukong_runtime::wukong_softmax_bwd_f32 as *const u8,
    );
    builder.symbol(
        RT_SOFTMAX_BWD_PAR,
        wukong_runtime::wukong_softmax_bwd_f32_parallel as *const u8,
    );
    builder.symbol(RT_LRSCAN, wukong_runtime::wukong_lrscan_f32 as *const u8);
    builder.symbol(
        RT_LRSCAN_PAR,
        wukong_runtime::wukong_lrscan_f32_parallel as *const u8,
    );
    builder.symbol(
        RT_RMSNORM_BWD,
        wukong_runtime::wukong_rmsnorm_bwd_f32 as *const u8,
    );
    builder.symbol(
        RT_RMSNORM_BWD_PAR,
        wukong_runtime::wukong_rmsnorm_bwd_f32_parallel as *const u8,
    );
    builder.symbol(
        RT_LAYERNORM_BWD,
        wukong_runtime::wukong_layernorm_bwd_f32 as *const u8,
    );
    builder.symbol(
        RT_LAYERNORM_BWD_PAR,
        wukong_runtime::wukong_layernorm_bwd_f32_parallel as *const u8,
    );
    builder.symbol(RT_XENT, wukong_runtime::wukong_xent_fwd_f32 as *const u8);
    builder.symbol(
        RT_XENT_PAR,
        wukong_runtime::wukong_xent_fwd_f32_parallel as *const u8,
    );
    builder.symbol(
        RT_XENT_BWD,
        wukong_runtime::wukong_xent_bwd_f32 as *const u8,
    );
    builder.symbol(
        RT_XENT_BWD_PAR,
        wukong_runtime::wukong_xent_bwd_f32_parallel as *const u8,
    );
    builder.symbol(RT_ROPE, wukong_runtime::wukong_rope_f32 as *const u8);
    builder.symbol(
        RT_ROPE_PAR,
        wukong_runtime::wukong_rope_f32_parallel as *const u8,
    );
    builder.symbol(
        RT_ROPE_BWD,
        wukong_runtime::wukong_rope_bwd_f32 as *const u8,
    );
    builder.symbol(
        RT_ROPE_BWD_PAR,
        wukong_runtime::wukong_rope_bwd_f32_parallel as *const u8,
    );
    builder.symbol(
        RT_LOGSUMEXP,
        wukong_runtime::wukong_logsumexp_f32 as *const u8,
    );
    builder.symbol(
        RT_LOGSUMEXP_PAR,
        wukong_runtime::wukong_logsumexp_f32_parallel as *const u8,
    );
    builder.symbol(RT_KLDIV, wukong_runtime::wukong_kldiv_f32 as *const u8);
    builder.symbol(
        RT_KLDIV_PAR,
        wukong_runtime::wukong_kldiv_f32_parallel as *const u8,
    );
    builder.symbol(RT_ENTROPY, wukong_runtime::wukong_entropy_f32 as *const u8);
    builder.symbol(
        RT_ENTROPY_PAR,
        wukong_runtime::wukong_entropy_f32_parallel as *const u8,
    );
    builder.symbol(RT_KD_LOSS, wukong_runtime::wukong_kd_loss_f32 as *const u8);
    builder.symbol(
        RT_KD_LOSS_PAR,
        wukong_runtime::wukong_kd_loss_f32_parallel as *const u8,
    );
    builder.symbol(
        RT_ROWARGMAX,
        wukong_runtime::wukong_rowargmax_i32 as *const u8,
    );
    builder.symbol(
        RT_ROWARGMAX_PAR,
        wukong_runtime::wukong_rowargmax_i32_parallel as *const u8,
    );
    builder.symbol(
        RT_ROWARGMIN,
        wukong_runtime::wukong_rowargmin_i32 as *const u8,
    );
    builder.symbol(
        RT_ROWARGMIN_PAR,
        wukong_runtime::wukong_rowargmin_i32_parallel as *const u8,
    );
    builder.symbol(
        RT_COLARGMAX,
        wukong_runtime::wukong_colargmax_i32 as *const u8,
    );
    builder.symbol(
        RT_COLARGMAX_PAR,
        wukong_runtime::wukong_colargmax_i32_parallel as *const u8,
    );
    builder.symbol(
        RT_COLARGMIN,
        wukong_runtime::wukong_colargmin_i32 as *const u8,
    );
    builder.symbol(
        RT_COLARGMIN_PAR,
        wukong_runtime::wukong_colargmin_i32_parallel as *const u8,
    );
    builder.symbol(RT_CUMSUM, wukong_runtime::wukong_cumsum_f32 as *const u8);
    builder.symbol(
        RT_CUMSUM_PAR,
        wukong_runtime::wukong_cumsum_f32_parallel as *const u8,
    );
    builder.symbol(RT_CUMPROD, wukong_runtime::wukong_cumprod_f32 as *const u8);
    builder.symbol(
        RT_CUMPROD_PAR,
        wukong_runtime::wukong_cumprod_f32_parallel as *const u8,
    );
    builder.symbol(RT_CUMMAX, wukong_runtime::wukong_cummax_f32 as *const u8);
    builder.symbol(
        RT_CUMMAX_PAR,
        wukong_runtime::wukong_cummax_f32_parallel as *const u8,
    );
    builder.symbol(RT_CUMMIN, wukong_runtime::wukong_cummin_f32 as *const u8);
    builder.symbol(
        RT_CUMMIN_PAR,
        wukong_runtime::wukong_cummin_f32_parallel as *const u8,
    );
    builder.symbol(
        RT_MAXPOOL2D,
        wukong_runtime::wukong_maxpool2d_f32 as *const u8,
    );
    builder.symbol(
        RT_MAXPOOL2D_PAR,
        wukong_runtime::wukong_maxpool2d_f32_parallel as *const u8,
    );
    builder.symbol(
        RT_AVGPOOL2D,
        wukong_runtime::wukong_avgpool2d_f32 as *const u8,
    );
    builder.symbol(
        RT_AVGPOOL2D_PAR,
        wukong_runtime::wukong_avgpool2d_f32_parallel as *const u8,
    );
    builder.symbol(
        RT_VMATH_BF16,
        wukong_runtime::wukong_vmath_bf16 as *const u8,
    );
    builder.symbol(RT_VMATH_F16, wukong_runtime::wukong_vmath_f16 as *const u8);
    builder.symbol(
        RT_VMATH_BF16_OUT,
        wukong_runtime::wukong_vmath_bf16_out as *const u8,
    );
    builder.symbol(
        RT_VMATH_F16_OUT,
        wukong_runtime::wukong_vmath_f16_out as *const u8,
    );
    builder.symbol(
        RT_TRANSPOSE,
        wukong_runtime::wukong_transpose_f32 as *const u8,
    );
    builder.symbol(
        RT_TRANSPOSE_PAR,
        wukong_runtime::wukong_transpose_f32_parallel as *const u8,
    );
    builder.symbol(
        RT_TRANSPOSE_U16,
        wukong_runtime::wukong_transpose_u16 as *const u8,
    );
    builder.symbol(
        RT_TRANSPOSE_U16_PAR,
        wukong_runtime::wukong_transpose_u16_parallel as *const u8,
    );
    builder.symbol(RT_COLSUM, wukong_runtime::wukong_colsum_f32 as *const u8);
    builder.symbol(
        RT_COLSUM_PAR,
        wukong_runtime::wukong_colsum_f32_parallel as *const u8,
    );
    builder.symbol(RT_COLMAX, wukong_runtime::wukong_colmax_f32 as *const u8);
    builder.symbol(
        RT_COLMAX_PAR,
        wukong_runtime::wukong_colmax_f32_parallel as *const u8,
    );
    builder.symbol(RT_COLMIN, wukong_runtime::wukong_colmin_f32 as *const u8);
    builder.symbol(
        RT_COLMIN_PAR,
        wukong_runtime::wukong_colmin_f32_parallel as *const u8,
    );
    builder.symbol(
        RT_COLMAXABS,
        wukong_runtime::wukong_colmaxabs_f32 as *const u8,
    );
    builder.symbol(
        RT_COLMAXABS_PAR,
        wukong_runtime::wukong_colmaxabs_f32_parallel as *const u8,
    );
    builder.symbol(RT_COLMEAN, wukong_runtime::wukong_colmean_f32 as *const u8);
    builder.symbol(
        RT_COLMEAN_PAR,
        wukong_runtime::wukong_colmean_f32_parallel as *const u8,
    );
    builder.symbol(
        RT_COLSUMSQ,
        wukong_runtime::wukong_colsumsq_f32 as *const u8,
    );
    builder.symbol(
        RT_COLSUMSQ_PAR,
        wukong_runtime::wukong_colsumsq_f32_parallel as *const u8,
    );
    builder.symbol(RT_COLL2, wukong_runtime::wukong_coll2_f32 as *const u8);
    builder.symbol(
        RT_COLL2_PAR,
        wukong_runtime::wukong_coll2_f32_parallel as *const u8,
    );
    builder.symbol(RT_COLRMS, wukong_runtime::wukong_colrms_f32 as *const u8);
    builder.symbol(
        RT_COLRMS_PAR,
        wukong_runtime::wukong_colrms_f32_parallel as *const u8,
    );
    builder.symbol(RT_VELEM, wukong_runtime::wukong_velem_f32 as *const u8);
    builder.symbol(
        RT_VELEM_PARALLEL,
        wukong_runtime::wukong_velem_f32_parallel as *const u8,
    );
    builder.symbol(RT_VHORNER, wukong_runtime::wukong_vhorner_f32 as *const u8);
    builder.symbol(
        RT_BIAS_BCAST,
        wukong_runtime::wukong_bias_bcast_f32 as *const u8,
    );
    builder.symbol(
        RT_BIAS_BCAST_PAR,
        wukong_runtime::wukong_bias_bcast_f32_parallel as *const u8,
    );
    builder.symbol(RT_SREDUCE, wukong_runtime::wukong_sreduce_f32 as *const u8);
    builder.symbol(
        RT_SREDUCE_PARALLEL,
        wukong_runtime::wukong_sreduce_f32_parallel as *const u8,
    );
    builder.symbol(
        RT_ARGREDUCE,
        wukong_runtime::wukong_argreduce_f32 as *const u8,
    );
    builder.symbol(
        RT_ARGREDUCE_PARALLEL,
        wukong_runtime::wukong_argreduce_f32_parallel as *const u8,
    );
    builder.symbol(RT_NORM, wukong_runtime::wukong_norm_f32 as *const u8);
    builder.symbol(
        RT_NORM_PARALLEL,
        wukong_runtime::wukong_norm_f32_parallel as *const u8,
    );
    builder.symbol(
        RT_NORM_AFFINE,
        wukong_runtime::wukong_norm_affine_f32 as *const u8,
    );
    builder.symbol(
        RT_NORM_AFFINE_PARALLEL,
        wukong_runtime::wukong_norm_affine_f32_parallel as *const u8,
    );
    builder.symbol(RT_I8GEMM_NT, wukong_runtime::wukong_i8gemm_nt as *const u8);
    builder.symbol(
        RT_I8GEMM_NT_PARALLEL,
        wukong_runtime::wukong_i8gemm_nt_parallel as *const u8,
    );
    builder.symbol(
        RT_I8GEMM_NT_DEQ,
        wukong_runtime::wukong_i8gemm_nt_deq as *const u8,
    );
    builder.symbol(
        RT_I8GEMM_NT_DEQ_PARALLEL,
        wukong_runtime::wukong_i8gemm_nt_deq_parallel as *const u8,
    );
    builder.symbol(RT_DEQUANT, wukong_runtime::wukong_dequant_f32 as *const u8);
    builder.symbol(
        RT_DEQUANT_PAR,
        wukong_runtime::wukong_dequant_f32_parallel as *const u8,
    );
    builder.symbol(
        RT_DEQUANT_PERCHAN,
        wukong_runtime::wukong_dequant_perchan_f32 as *const u8,
    );
    builder.symbol(
        RT_DEQUANT_PERCHAN_PAR,
        wukong_runtime::wukong_dequant_perchan_f32_parallel as *const u8,
    );
    builder.symbol(
        RT_EMBEDDING,
        wukong_runtime::wukong_embedding_f32 as *const u8,
    );
    builder.symbol(
        RT_EMBEDDING_PAR,
        wukong_runtime::wukong_embedding_f32_parallel as *const u8,
    );
    builder.symbol(
        RT_ATTENTION,
        wukong_runtime::wukong_attention_f32 as *const u8,
    );
    builder.symbol(
        RT_SCATTER_ADD,
        wukong_runtime::wukong_scatter_add_f32 as *const u8,
    );
    builder.symbol(
        RT_SCATTER_ADD_PAR,
        wukong_runtime::wukong_scatter_add_f32_parallel as *const u8,
    );
    builder.symbol(RT_DOT_BF16, wukong_runtime::wukong_dot_bf16 as *const u8);
    builder.symbol(RT_SUM_BF16, wukong_runtime::wukong_sum_bf16 as *const u8);
    builder.symbol(
        RT_REDUCE_BF16,
        wukong_runtime::wukong_reduce_bf16 as *const u8,
    );
    builder.symbol(RT_AXPBY_F16, wukong_runtime::wukong_axpby_f16 as *const u8);
    builder.symbol(RT_DOT_F16, wukong_runtime::wukong_dot_f16 as *const u8);
    builder.symbol(RT_SUM_F16, wukong_runtime::wukong_sum_f16 as *const u8);
    builder.symbol(
        RT_REDUCE_F16,
        wukong_runtime::wukong_reduce_f16 as *const u8,
    );
    builder.symbol(
        RT_DOT_BF16_PAR,
        wukong_runtime::wukong_dot_bf16_parallel as *const u8,
    );
    builder.symbol(
        RT_SUM_BF16_PAR,
        wukong_runtime::wukong_sum_bf16_parallel as *const u8,
    );
    builder.symbol(
        RT_REDUCE_BF16_PAR,
        wukong_runtime::wukong_reduce_bf16_parallel as *const u8,
    );
    builder.symbol(
        RT_DOT_F16_PAR,
        wukong_runtime::wukong_dot_f16_parallel as *const u8,
    );
    builder.symbol(
        RT_SUM_F16_PAR,
        wukong_runtime::wukong_sum_f16_parallel as *const u8,
    );
    builder.symbol(
        RT_REDUCE_F16_PAR,
        wukong_runtime::wukong_reduce_f16_parallel as *const u8,
    );
    builder.symbol(
        RT_F32_TO_F16,
        wukong_runtime::wukong_f32_to_f16_bits as *const u8,
    );
    builder.symbol(
        RT_F16_TO_F32,
        wukong_runtime::wukong_f16_bits_to_f32 as *const u8,
    );
    builder.symbol(
        RT_AXPBY_BF16,
        wukong_runtime::wukong_axpby_bf16 as *const u8,
    );
    builder.symbol(
        RT_AXPBY_BF16_OUT,
        wukong_runtime::wukong_axpby_bf16_out as *const u8,
    );
    builder.symbol(
        RT_AXPBY_F16_OUT,
        wukong_runtime::wukong_axpby_f16_out as *const u8,
    );
    builder.symbol(RT_FMOD_F64, rt_fmod_f64 as *const u8);
    builder.symbol(RT_FMOD_F32, rt_fmod_f32 as *const u8);
    bind_extern_symbols(&mut builder, program, interner)?;
    let mut module = JITModule::new(builder);

    let mut log = perf_map.map(|_| debuginfo::CodeLog::new(false));
    let ids = populate_module(
        &mut module,
        program,
        interner,
        parallel,
        entry::SymbolMode::Jit,
        log.as_mut(),
    )?;
    module.finalize_definitions().map_err(|e| e.to_string())?;
    if let (Some(path), Some(log)) = (perf_map, &log) {
        write_perf_map(&module, log, program, interner, path);
    }

    let entry_fn = program
        .function(entry)
        .ok_or_else(|| format!("no entry function `{}`", interner.resolve(entry)))?;
    let valid_entry = if buffer_dispatch {
        entry_fn.ret == MirType::Void
            && entry_fn.params.len() == 2
            && entry_fn.value_type(entry_fn.params[0]) == &MirType::Ptr
            && entry_fn.value_type(entry_fn.params[1]) == &MirType::I64
    } else {
        entry_fn.params.is_empty()
    };
    if !valid_entry {
        unsafe { module.free_memory() };
        return Err(NativeFailure {
            message: if buffer_dispatch {
                "native buffer dispatcher must have signature (Ptr, I64) -> Void"
            } else {
                "native entry point must take no parameters"
            }
            .into(),
            site: entry_fn.decl_span(),
        });
    }
    let entry_id = ids[&entry];
    let code = module.get_finalized_function(entry_id);

    Ok(JitProgram {
        module: Some(module),
        code,
        ret: entry_fn.ret.clone(),
    })
}

/// Append one perf-map line per function `log` recorded, at its finalized address in `module`. The
/// map is a profiling aid: failing to write it warns and never fails the compile.
fn write_perf_map(
    module: &cranelift_jit::JITModule,
    log: &debuginfo::CodeLog,
    program: &Program,
    interner: &Interner,
    path: &std::path::Path,
) {
    let entries: Vec<(usize, u32, String)> = log
        .defined
        .iter()
        .map(|code| {
            (
                module.get_finalized_function(code.id) as usize,
                code.size,
                debuginfo::display_name(code, program, interner),
            )
        })
        .collect();
    if let Err(e) = debuginfo::perf_map::append(path, &entries) {
        eprintln!(
            "warning: could not write the perf map `{}` ({}=1): {e}",
            path.display(),
            debuginfo::perf_map::ENV
        );
    }
}

/// JIT-compile and execute `entry`, returning its exit code and captured stdout.
pub fn jit_run(
    program: &Program,
    entry: Symbol,
    interner: &Interner,
) -> Result<(i64, Vec<u8>), String> {
    jit_compile(program, entry, interner)?.run()
}

/// JIT-compile and execute `entry`, returning its exit code or failure together with the stdout
/// printed before it stopped (see [`JitProgram::run_capturing`]). A compile error has no output.
pub fn jit_run_capturing(
    program: &Program,
    entry: Symbol,
    interner: &Interner,
) -> (Result<i64, String>, Vec<u8>) {
    match jit_compile(program, entry, interner) {
        Ok(native) => native.run_capturing(),
        Err(e) => (Err(e), Vec::new()),
    }
}

/// As [`jit_run`], with an explicit backend configuration (see [`jit_compile_with`]).
pub fn jit_run_with(
    program: &Program,
    entry: Symbol,
    interner: &Interner,
    backend: BackendOpt,
) -> Result<(i64, Vec<u8>), String> {
    jit_compile_with(program, entry, interner, backend)?.run()
}

/// A JIT-compiled module exposing raw pointers to any of its functions, for callers that know a
/// function's ABI and want to invoke it directly (e.g. timing a kernel `(*const f32, …)`).
pub struct JitModuleHandle {
    module: Option<cranelift_jit::JITModule>,
    ids: HashMap<Symbol, FuncId>,
}

impl JitModuleHandle {
    /// The finalized machine-code pointer for `sym`, if the module defines it. The caller must
    /// transmute it to the correct ABI; the pointer is valid until this handle is dropped.
    pub fn func_ptr(&self, sym: Symbol) -> Option<*const u8> {
        let id = *self.ids.get(&sym)?;
        Some(self.module.as_ref().unwrap().get_finalized_function(id))
    }
}

impl Drop for JitModuleHandle {
    fn drop(&mut self) {
        if let Some(m) = self.module.take() {
            unsafe { m.free_memory() };
        }
    }
}

/// JIT-compile every function in `program`, returning a handle for fetching their code pointers.
///
/// Runtime failures have a raw-mode meaning here. A raw caller has no run handle a failure could be
/// reported through, and returning a dummy value would be a silently wrong answer, so a failed
/// `assert` or an out-of-bounds slice access **aborts the process** with the shared diagnostic —
/// Rust's `panic = "abort"`, and what a linked executable does (exit 1). A run that must survive a
/// failure and report it uses the guarded [`jit_compile`] + [`JitProgram::run`] API instead.
pub fn jit_module(program: &Program, interner: &Interner) -> Result<JitModuleHandle, String> {
    use cranelift_jit::{JITBuilder, JITModule};

    let isa = make_isa(false)?;
    let mut builder = JITBuilder::with_isa(isa, cranelift_module::default_libcall_names());
    builder.symbol(RT_PRINT_I64, rt_print_i64 as *const u8);
    builder.symbol(RT_PRINT_U64, rt_print_u64 as *const u8);
    builder.symbol(RT_PRINT_F64, rt_print_f64 as *const u8);
    builder.symbol(RT_PRINT_STR, rt_print_str as *const u8);
    builder.symbol(RT_ASSERT, rt_assert_abort as *const u8);
    // Raw mode: every failure aborts where it happens, so none is ever pending — never read the
    // flag a concurrent guarded run may have raised.
    builder.symbol(RT_ASSERT_FAILED, rt_assert_never_failed as *const u8);
    builder.symbol(RT_BOUNDS_FAIL, rt_bounds_fail_abort as *const u8);
    builder.symbol(RT_CHECK_FAIL, rt_check_fail_abort as *const u8);
    builder.symbol(RT_ALLOC, wukong_runtime::wukong_rt_alloc as *const u8);
    builder.symbol(RT_FREE, wukong_runtime::wukong_rt_free as *const u8);
    builder.symbol(
        RT_SCRATCH_ALLOC,
        wukong_runtime::wukong_rt_scratch_alloc as *const u8,
    );
    builder.symbol(
        RT_SCRATCH_MARK,
        wukong_runtime::wukong_rt_scratch_mark as *const u8,
    );
    builder.symbol(
        RT_SCRATCH_RELEASE,
        wukong_runtime::wukong_rt_scratch_release as *const u8,
    );
    builder.symbol(RT_READ_F32, wukong_runtime::wukong_rt_read_f32 as *const u8);
    builder.symbol(RT_READ_F64, wukong_runtime::wukong_rt_read_f64 as *const u8);
    builder.symbol(RT_READ_I32, wukong_runtime::wukong_rt_read_i32 as *const u8);
    builder.symbol(RT_READ_I64, wukong_runtime::wukong_rt_read_i64 as *const u8);
    builder.symbol(RT_READ_I8, wukong_runtime::wukong_rt_read_i8 as *const u8);
    builder.symbol(RT_READ_U8, wukong_runtime::wukong_rt_read_u8 as *const u8);
    builder.symbol(
        RT_WRITE_F32,
        wukong_runtime::wukong_rt_write_f32 as *const u8,
    );
    builder.symbol(
        RT_WRITE_F64,
        wukong_runtime::wukong_rt_write_f64 as *const u8,
    );
    builder.symbol(
        RT_WRITE_I32,
        wukong_runtime::wukong_rt_write_i32 as *const u8,
    );
    builder.symbol(
        RT_WRITE_I64,
        wukong_runtime::wukong_rt_write_i64 as *const u8,
    );
    builder.symbol(RT_WRITE_I8, wukong_runtime::wukong_rt_write_i8 as *const u8);
    builder.symbol(RT_WRITE_U8, wukong_runtime::wukong_rt_write_u8 as *const u8);
    // The ML builtins' entries (the `ml_rt` table), and the text builtins' (the `text_rt` table).
    ml_rt::bind_jit(&mut builder);
    text_rt::bind_jit(&mut builder);
    // Raw mode: a raw caller has no run an `exit` could end, so it is fatal, as a failed assert is.
    builder.symbol(text_rt::RT_EXIT, rt_exit_abort as *const u8);
    builder.symbol(RT_NOW_NS, wukong_runtime::wukong_now_ns as *const u8);
    builder.symbol(
        RT_PARALLEL_FOR,
        wukong_runtime::wukong_parallel_for as *const u8,
    );
    builder.symbol(RT_SGEMM, wukong_runtime::wukong_sgemm as *const u8);
    builder.symbol(
        RT_SGEMM_PARALLEL,
        wukong_runtime::wukong_sgemm_parallel as *const u8,
    );
    builder.symbol(RT_SGEMM_NT, wukong_runtime::wukong_sgemm_nt as *const u8);
    builder.symbol(
        RT_SGEMM_NT_PARALLEL,
        wukong_runtime::wukong_sgemm_nt_parallel as *const u8,
    );
    builder.symbol(RT_SGEMM_TN, wukong_runtime::wukong_sgemm_tn as *const u8);
    builder.symbol(
        RT_SGEMM_TN_PARALLEL,
        wukong_runtime::wukong_sgemm_tn_parallel as *const u8,
    );
    builder.symbol(
        RT_SGEMM_BF16_NT,
        wukong_runtime::wukong_sgemm_bf16_nt as *const u8,
    );
    builder.symbol(
        RT_SGEMM_BF16_NT_PARALLEL,
        wukong_runtime::wukong_sgemm_bf16_nt_parallel as *const u8,
    );
    builder.symbol(
        RT_SGEMM_F16_NT,
        wukong_runtime::wukong_sgemm_f16_nt as *const u8,
    );
    builder.symbol(
        RT_SGEMM_F16_NT_PARALLEL,
        wukong_runtime::wukong_sgemm_f16_nt_parallel as *const u8,
    );
    builder.symbol(
        RT_SGEMM_BF16_TN,
        wukong_runtime::wukong_sgemm_bf16_tn as *const u8,
    );
    builder.symbol(
        RT_SGEMM_BF16_TN_PARALLEL,
        wukong_runtime::wukong_sgemm_bf16_tn_parallel as *const u8,
    );
    builder.symbol(
        RT_SGEMM_F16_TN,
        wukong_runtime::wukong_sgemm_f16_tn as *const u8,
    );
    builder.symbol(
        RT_SGEMM_F16_TN_PARALLEL,
        wukong_runtime::wukong_sgemm_f16_tn_parallel as *const u8,
    );
    builder.symbol(
        RT_SGEMM_NT_EPI,
        wukong_runtime::wukong_sgemm_nt_epi as *const u8,
    );
    builder.symbol(
        RT_SGEMM_NT_EPI_PAR,
        wukong_runtime::wukong_sgemm_nt_epi_parallel as *const u8,
    );
    builder.symbol(RT_SGEMV, wukong_runtime::wukong_sgemv as *const u8);
    builder.symbol(
        RT_SGEMV_PAR,
        wukong_runtime::wukong_sgemv_parallel as *const u8,
    );
    builder.symbol(
        RT_SGEMV_ALPHA,
        wukong_runtime::wukong_sgemv_alpha as *const u8,
    );
    builder.symbol(
        RT_SGEMV_ALPHA_PAR,
        wukong_runtime::wukong_sgemv_alpha_parallel as *const u8,
    );
    builder.symbol(RT_SGEVM, wukong_runtime::wukong_sgevm_f32 as *const u8);
    builder.symbol(
        RT_SGEVM_PAR,
        wukong_runtime::wukong_sgevm_f32_parallel as *const u8,
    );
    builder.symbol(
        RT_SGEMM_NT_ALPHA,
        wukong_runtime::wukong_sgemm_nt_alpha as *const u8,
    );
    builder.symbol(
        RT_SGEMM_NT_ALPHA_PAR,
        wukong_runtime::wukong_sgemm_nt_alpha_parallel as *const u8,
    );
    builder.symbol(
        RT_SGEMM_BF16_NT_EPI,
        wukong_runtime::wukong_sgemm_bf16_nt_epi as *const u8,
    );
    builder.symbol(
        RT_SGEMM_BF16_NT_EPI_PAR,
        wukong_runtime::wukong_sgemm_bf16_nt_epi_parallel as *const u8,
    );
    builder.symbol(
        RT_SGEMM_F16_NT_EPI,
        wukong_runtime::wukong_sgemm_f16_nt_epi as *const u8,
    );
    builder.symbol(
        RT_SGEMM_F16_NT_EPI_PAR,
        wukong_runtime::wukong_sgemm_f16_nt_epi_parallel as *const u8,
    );
    builder.symbol(RT_VMATH, wukong_runtime::wukong_vmath_f32 as *const u8);
    builder.symbol(RT_TRIG_F32, wukong_runtime::wukong_rt_trig_f32 as *const u8);
    builder.symbol(RT_MATH_F64, wukong_runtime::wukong_rt_math_f64 as *const u8);
    builder.symbol(
        RT_MATH2_F64,
        wukong_runtime::wukong_rt_math2_f64 as *const u8,
    );
    builder.symbol(
        RT_VMATH_PARALLEL,
        wukong_runtime::wukong_vmath_f32_parallel as *const u8,
    );
    builder.symbol(RT_VMATH2, wukong_runtime::wukong_vmath2_f32 as *const u8);
    builder.symbol(
        RT_SOFTMAX_BWD,
        wukong_runtime::wukong_softmax_bwd_f32 as *const u8,
    );
    builder.symbol(
        RT_SOFTMAX_BWD_PAR,
        wukong_runtime::wukong_softmax_bwd_f32_parallel as *const u8,
    );
    builder.symbol(RT_LRSCAN, wukong_runtime::wukong_lrscan_f32 as *const u8);
    builder.symbol(
        RT_LRSCAN_PAR,
        wukong_runtime::wukong_lrscan_f32_parallel as *const u8,
    );
    builder.symbol(
        RT_RMSNORM_BWD,
        wukong_runtime::wukong_rmsnorm_bwd_f32 as *const u8,
    );
    builder.symbol(
        RT_RMSNORM_BWD_PAR,
        wukong_runtime::wukong_rmsnorm_bwd_f32_parallel as *const u8,
    );
    builder.symbol(
        RT_LAYERNORM_BWD,
        wukong_runtime::wukong_layernorm_bwd_f32 as *const u8,
    );
    builder.symbol(
        RT_LAYERNORM_BWD_PAR,
        wukong_runtime::wukong_layernorm_bwd_f32_parallel as *const u8,
    );
    builder.symbol(RT_XENT, wukong_runtime::wukong_xent_fwd_f32 as *const u8);
    builder.symbol(
        RT_XENT_PAR,
        wukong_runtime::wukong_xent_fwd_f32_parallel as *const u8,
    );
    builder.symbol(
        RT_XENT_BWD,
        wukong_runtime::wukong_xent_bwd_f32 as *const u8,
    );
    builder.symbol(
        RT_XENT_BWD_PAR,
        wukong_runtime::wukong_xent_bwd_f32_parallel as *const u8,
    );
    builder.symbol(RT_ROPE, wukong_runtime::wukong_rope_f32 as *const u8);
    builder.symbol(
        RT_ROPE_PAR,
        wukong_runtime::wukong_rope_f32_parallel as *const u8,
    );
    builder.symbol(
        RT_ROPE_BWD,
        wukong_runtime::wukong_rope_bwd_f32 as *const u8,
    );
    builder.symbol(
        RT_ROPE_BWD_PAR,
        wukong_runtime::wukong_rope_bwd_f32_parallel as *const u8,
    );
    builder.symbol(
        RT_LOGSUMEXP,
        wukong_runtime::wukong_logsumexp_f32 as *const u8,
    );
    builder.symbol(
        RT_LOGSUMEXP_PAR,
        wukong_runtime::wukong_logsumexp_f32_parallel as *const u8,
    );
    builder.symbol(RT_KLDIV, wukong_runtime::wukong_kldiv_f32 as *const u8);
    builder.symbol(
        RT_KLDIV_PAR,
        wukong_runtime::wukong_kldiv_f32_parallel as *const u8,
    );
    builder.symbol(RT_ENTROPY, wukong_runtime::wukong_entropy_f32 as *const u8);
    builder.symbol(
        RT_ENTROPY_PAR,
        wukong_runtime::wukong_entropy_f32_parallel as *const u8,
    );
    builder.symbol(RT_KD_LOSS, wukong_runtime::wukong_kd_loss_f32 as *const u8);
    builder.symbol(
        RT_KD_LOSS_PAR,
        wukong_runtime::wukong_kd_loss_f32_parallel as *const u8,
    );
    builder.symbol(
        RT_ROWARGMAX,
        wukong_runtime::wukong_rowargmax_i32 as *const u8,
    );
    builder.symbol(
        RT_ROWARGMAX_PAR,
        wukong_runtime::wukong_rowargmax_i32_parallel as *const u8,
    );
    builder.symbol(
        RT_ROWARGMIN,
        wukong_runtime::wukong_rowargmin_i32 as *const u8,
    );
    builder.symbol(
        RT_ROWARGMIN_PAR,
        wukong_runtime::wukong_rowargmin_i32_parallel as *const u8,
    );
    builder.symbol(
        RT_COLARGMAX,
        wukong_runtime::wukong_colargmax_i32 as *const u8,
    );
    builder.symbol(
        RT_COLARGMAX_PAR,
        wukong_runtime::wukong_colargmax_i32_parallel as *const u8,
    );
    builder.symbol(
        RT_COLARGMIN,
        wukong_runtime::wukong_colargmin_i32 as *const u8,
    );
    builder.symbol(
        RT_COLARGMIN_PAR,
        wukong_runtime::wukong_colargmin_i32_parallel as *const u8,
    );
    builder.symbol(RT_CUMSUM, wukong_runtime::wukong_cumsum_f32 as *const u8);
    builder.symbol(
        RT_CUMSUM_PAR,
        wukong_runtime::wukong_cumsum_f32_parallel as *const u8,
    );
    builder.symbol(RT_CUMPROD, wukong_runtime::wukong_cumprod_f32 as *const u8);
    builder.symbol(
        RT_CUMPROD_PAR,
        wukong_runtime::wukong_cumprod_f32_parallel as *const u8,
    );
    builder.symbol(RT_CUMMAX, wukong_runtime::wukong_cummax_f32 as *const u8);
    builder.symbol(
        RT_CUMMAX_PAR,
        wukong_runtime::wukong_cummax_f32_parallel as *const u8,
    );
    builder.symbol(RT_CUMMIN, wukong_runtime::wukong_cummin_f32 as *const u8);
    builder.symbol(
        RT_CUMMIN_PAR,
        wukong_runtime::wukong_cummin_f32_parallel as *const u8,
    );
    builder.symbol(
        RT_MAXPOOL2D,
        wukong_runtime::wukong_maxpool2d_f32 as *const u8,
    );
    builder.symbol(
        RT_MAXPOOL2D_PAR,
        wukong_runtime::wukong_maxpool2d_f32_parallel as *const u8,
    );
    builder.symbol(
        RT_AVGPOOL2D,
        wukong_runtime::wukong_avgpool2d_f32 as *const u8,
    );
    builder.symbol(
        RT_AVGPOOL2D_PAR,
        wukong_runtime::wukong_avgpool2d_f32_parallel as *const u8,
    );
    builder.symbol(
        RT_VMATH_BF16,
        wukong_runtime::wukong_vmath_bf16 as *const u8,
    );
    builder.symbol(RT_VMATH_F16, wukong_runtime::wukong_vmath_f16 as *const u8);
    builder.symbol(
        RT_VMATH_BF16_OUT,
        wukong_runtime::wukong_vmath_bf16_out as *const u8,
    );
    builder.symbol(
        RT_VMATH_F16_OUT,
        wukong_runtime::wukong_vmath_f16_out as *const u8,
    );
    builder.symbol(
        RT_TRANSPOSE,
        wukong_runtime::wukong_transpose_f32 as *const u8,
    );
    builder.symbol(
        RT_TRANSPOSE_PAR,
        wukong_runtime::wukong_transpose_f32_parallel as *const u8,
    );
    builder.symbol(
        RT_TRANSPOSE_U16,
        wukong_runtime::wukong_transpose_u16 as *const u8,
    );
    builder.symbol(
        RT_TRANSPOSE_U16_PAR,
        wukong_runtime::wukong_transpose_u16_parallel as *const u8,
    );
    builder.symbol(RT_COLSUM, wukong_runtime::wukong_colsum_f32 as *const u8);
    builder.symbol(
        RT_COLSUM_PAR,
        wukong_runtime::wukong_colsum_f32_parallel as *const u8,
    );
    builder.symbol(RT_COLMAX, wukong_runtime::wukong_colmax_f32 as *const u8);
    builder.symbol(
        RT_COLMAX_PAR,
        wukong_runtime::wukong_colmax_f32_parallel as *const u8,
    );
    builder.symbol(RT_COLMIN, wukong_runtime::wukong_colmin_f32 as *const u8);
    builder.symbol(
        RT_COLMIN_PAR,
        wukong_runtime::wukong_colmin_f32_parallel as *const u8,
    );
    builder.symbol(
        RT_COLMAXABS,
        wukong_runtime::wukong_colmaxabs_f32 as *const u8,
    );
    builder.symbol(
        RT_COLMAXABS_PAR,
        wukong_runtime::wukong_colmaxabs_f32_parallel as *const u8,
    );
    builder.symbol(RT_COLMEAN, wukong_runtime::wukong_colmean_f32 as *const u8);
    builder.symbol(
        RT_COLMEAN_PAR,
        wukong_runtime::wukong_colmean_f32_parallel as *const u8,
    );
    builder.symbol(
        RT_COLSUMSQ,
        wukong_runtime::wukong_colsumsq_f32 as *const u8,
    );
    builder.symbol(
        RT_COLSUMSQ_PAR,
        wukong_runtime::wukong_colsumsq_f32_parallel as *const u8,
    );
    builder.symbol(RT_COLL2, wukong_runtime::wukong_coll2_f32 as *const u8);
    builder.symbol(
        RT_COLL2_PAR,
        wukong_runtime::wukong_coll2_f32_parallel as *const u8,
    );
    builder.symbol(RT_COLRMS, wukong_runtime::wukong_colrms_f32 as *const u8);
    builder.symbol(
        RT_COLRMS_PAR,
        wukong_runtime::wukong_colrms_f32_parallel as *const u8,
    );
    builder.symbol(RT_VELEM, wukong_runtime::wukong_velem_f32 as *const u8);
    builder.symbol(
        RT_VELEM_PARALLEL,
        wukong_runtime::wukong_velem_f32_parallel as *const u8,
    );
    builder.symbol(RT_VHORNER, wukong_runtime::wukong_vhorner_f32 as *const u8);
    builder.symbol(
        RT_BIAS_BCAST,
        wukong_runtime::wukong_bias_bcast_f32 as *const u8,
    );
    builder.symbol(
        RT_BIAS_BCAST_PAR,
        wukong_runtime::wukong_bias_bcast_f32_parallel as *const u8,
    );
    builder.symbol(RT_SREDUCE, wukong_runtime::wukong_sreduce_f32 as *const u8);
    builder.symbol(
        RT_SREDUCE_PARALLEL,
        wukong_runtime::wukong_sreduce_f32_parallel as *const u8,
    );
    builder.symbol(
        RT_ARGREDUCE,
        wukong_runtime::wukong_argreduce_f32 as *const u8,
    );
    builder.symbol(
        RT_ARGREDUCE_PARALLEL,
        wukong_runtime::wukong_argreduce_f32_parallel as *const u8,
    );
    builder.symbol(RT_NORM, wukong_runtime::wukong_norm_f32 as *const u8);
    builder.symbol(
        RT_NORM_PARALLEL,
        wukong_runtime::wukong_norm_f32_parallel as *const u8,
    );
    builder.symbol(
        RT_NORM_AFFINE,
        wukong_runtime::wukong_norm_affine_f32 as *const u8,
    );
    builder.symbol(
        RT_NORM_AFFINE_PARALLEL,
        wukong_runtime::wukong_norm_affine_f32_parallel as *const u8,
    );
    builder.symbol(RT_I8GEMM_NT, wukong_runtime::wukong_i8gemm_nt as *const u8);
    builder.symbol(
        RT_I8GEMM_NT_PARALLEL,
        wukong_runtime::wukong_i8gemm_nt_parallel as *const u8,
    );
    builder.symbol(
        RT_I8GEMM_NT_DEQ,
        wukong_runtime::wukong_i8gemm_nt_deq as *const u8,
    );
    builder.symbol(
        RT_I8GEMM_NT_DEQ_PARALLEL,
        wukong_runtime::wukong_i8gemm_nt_deq_parallel as *const u8,
    );
    builder.symbol(RT_DEQUANT, wukong_runtime::wukong_dequant_f32 as *const u8);
    builder.symbol(
        RT_DEQUANT_PAR,
        wukong_runtime::wukong_dequant_f32_parallel as *const u8,
    );
    builder.symbol(
        RT_DEQUANT_PERCHAN,
        wukong_runtime::wukong_dequant_perchan_f32 as *const u8,
    );
    builder.symbol(
        RT_DEQUANT_PERCHAN_PAR,
        wukong_runtime::wukong_dequant_perchan_f32_parallel as *const u8,
    );
    builder.symbol(
        RT_EMBEDDING,
        wukong_runtime::wukong_embedding_f32 as *const u8,
    );
    builder.symbol(
        RT_EMBEDDING_PAR,
        wukong_runtime::wukong_embedding_f32_parallel as *const u8,
    );
    builder.symbol(
        RT_ATTENTION,
        wukong_runtime::wukong_attention_f32 as *const u8,
    );
    builder.symbol(
        RT_SCATTER_ADD,
        wukong_runtime::wukong_scatter_add_f32 as *const u8,
    );
    builder.symbol(
        RT_SCATTER_ADD_PAR,
        wukong_runtime::wukong_scatter_add_f32_parallel as *const u8,
    );
    builder.symbol(RT_DOT_BF16, wukong_runtime::wukong_dot_bf16 as *const u8);
    builder.symbol(RT_SUM_BF16, wukong_runtime::wukong_sum_bf16 as *const u8);
    builder.symbol(
        RT_REDUCE_BF16,
        wukong_runtime::wukong_reduce_bf16 as *const u8,
    );
    builder.symbol(RT_AXPBY_F16, wukong_runtime::wukong_axpby_f16 as *const u8);
    builder.symbol(RT_DOT_F16, wukong_runtime::wukong_dot_f16 as *const u8);
    builder.symbol(RT_SUM_F16, wukong_runtime::wukong_sum_f16 as *const u8);
    builder.symbol(
        RT_REDUCE_F16,
        wukong_runtime::wukong_reduce_f16 as *const u8,
    );
    builder.symbol(
        RT_DOT_BF16_PAR,
        wukong_runtime::wukong_dot_bf16_parallel as *const u8,
    );
    builder.symbol(
        RT_SUM_BF16_PAR,
        wukong_runtime::wukong_sum_bf16_parallel as *const u8,
    );
    builder.symbol(
        RT_REDUCE_BF16_PAR,
        wukong_runtime::wukong_reduce_bf16_parallel as *const u8,
    );
    builder.symbol(
        RT_DOT_F16_PAR,
        wukong_runtime::wukong_dot_f16_parallel as *const u8,
    );
    builder.symbol(
        RT_SUM_F16_PAR,
        wukong_runtime::wukong_sum_f16_parallel as *const u8,
    );
    builder.symbol(
        RT_REDUCE_F16_PAR,
        wukong_runtime::wukong_reduce_f16_parallel as *const u8,
    );
    builder.symbol(
        RT_F32_TO_F16,
        wukong_runtime::wukong_f32_to_f16_bits as *const u8,
    );
    builder.symbol(
        RT_F16_TO_F32,
        wukong_runtime::wukong_f16_bits_to_f32 as *const u8,
    );
    builder.symbol(
        RT_AXPBY_BF16,
        wukong_runtime::wukong_axpby_bf16 as *const u8,
    );
    builder.symbol(
        RT_AXPBY_BF16_OUT,
        wukong_runtime::wukong_axpby_bf16_out as *const u8,
    );
    builder.symbol(
        RT_AXPBY_F16_OUT,
        wukong_runtime::wukong_axpby_f16_out as *const u8,
    );
    builder.symbol(RT_FMOD_F64, rt_fmod_f64 as *const u8);
    builder.symbol(RT_FMOD_F32, rt_fmod_f32 as *const u8);
    bind_extern_symbols(&mut builder, program, interner)?;
    let mut module = JITModule::new(builder);
    let perf_map = debuginfo::perf_map::from_env();
    let mut log = perf_map.as_ref().map(|_| debuginfo::CodeLog::new(false));
    let ids = populate_module(
        &mut module,
        program,
        interner,
        default_parallel(),
        entry::SymbolMode::Jit,
        log.as_mut(),
    )?;
    module.finalize_definitions().map_err(|e| e.to_string())?;
    if let (Some(path), Some(log)) = (&perf_map, &log) {
        write_perf_map(&module, log, program, interner, path);
    }
    Ok(JitModuleHandle {
        module: Some(module),
        ids,
    })
}

/// Levers that trim `emit_object` compile time **without changing a single emitted byte**. Every
/// field toggles a check-only or scheduling concern, never the machine code: the differential suite
/// and the serial-vs-parallel byte-identity test (`tests.rs`) pin that invariant.
#[derive(Clone, Copy)]
struct EmitOptions {
    /// Run the Cranelift IR verifier (diagnostic-only; see [`default_verify`]).
    verify: bool,
    /// Compile the user functions concurrently, then define them into the object in source order.
    /// Byte-identical to serial because each function is an independent compilation unit and the
    /// object is assembled in the same deterministic order either way (see [`default_parallel`]).
    parallel: bool,
}

impl EmitOptions {
    /// The shipping configuration: verifier per [`default_verify`], parallelism per
    /// [`default_parallel`].
    fn production() -> Self {
        EmitOptions {
            verify: default_verify(),
            parallel: default_parallel(),
        }
    }
}

/// Wall-time split of `emit_object`, exposed by [`emit_object_timed`] so `compile-profile` can report
/// Cranelift codegen (isel/regalloc/machine-code emit) separately from object-container
/// serialization — the two halves the `codegen+obj` stage previously lumped together.
pub struct ObjTimings {
    /// ISA/module setup + `populate_module`: declaration, per-function instruction selection,
    /// register allocation, and machine-code emission.
    pub codegen: std::time::Duration,
    /// `ObjectModule::finish` + `ObjectProduct::emit`: writing the machine code, relocations, and
    /// symbol table into the COFF/ELF container.
    pub object_write: std::time::Duration,
}

/// Compile `program` to a native object file (bytes) for the host target. The runtime symbols (the
/// `wukong_rt_*` shims and every recognized `wukong_*` kernel) are left as undefined imports; this
/// crate never links. Resolving them belongs to `wukong_driver::emit_native`, which prefers a
/// **rustc-driven** link (rustc pulls in the `wukong_runtime` rlib, so kernel symbols and the
/// read-only string relocations both resolve) and only falls back to `$CC`/`cc` plus a generated C
/// runtime shim when rustc or that rlib is unavailable.
pub fn emit_object(program: &Program, interner: &Interner) -> Result<Vec<u8>, String> {
    emit_object_ex(program, interner, EmitOptions::production()).map(|(bytes, _)| bytes)
}

/// As [`emit_object`], for the instruction-set level `target` (`--target-cpu`) instead of the host's.
/// A program carrying raw AVX2+FMA vector kernels is refused for a level below x86-64-v3 rather
/// than embedding bytes that level cannot execute.
pub fn emit_object_for(
    program: &Program,
    interner: &Interner,
    target: TargetCpu,
) -> Result<Vec<u8>, String> {
    emit_object_for_with(program, interner, target, BackendOpt::default())
}

/// As [`emit_object`], with an explicit backend configuration (the driver passes the source `-O`
/// level, so an `-O0` object compiles with Cranelift `opt_level=none`).
pub fn emit_object_with(
    program: &Program,
    interner: &Interner,
    backend: BackendOpt,
) -> Result<Vec<u8>, String> {
    emit_object_for_with(program, interner, TargetCpu::Native, backend)
}

/// Both [`emit_object_for`]'s instruction-set level and [`emit_object_with`]'s backend
/// configuration — what `--emit=obj|exe` uses, honouring `--target-cpu` and `-O` together.
pub fn emit_object_for_with(
    program: &Program,
    interner: &Interner,
    target: TargetCpu,
    backend: BackendOpt,
) -> Result<Vec<u8>, String> {
    emit_object_kind_for_with(program, interner, target, backend, ObjectKind::Library)
        .map_err(String::from)
}

/// As [`emit_object_for_with`], for an object of `kind`. `--emit=exe` passes
/// [`ObjectKind::Executable`]: every user function becomes an internal, mangled symbol, and the
/// program's `main` runs behind a synthesized C `main` (see the `entry` module). Every object with
/// a zero-parameter scalar `main` gets that C `main`, so it imports `wukong_rt_run_main`. A failure
/// carries the source construct it was raised on ([`NativeFailure::site`]).
pub fn emit_object_kind_for_with(
    program: &Program,
    interner: &Interner,
    target: TargetCpu,
    backend: BackendOpt,
    kind: ObjectKind,
) -> Result<Vec<u8>, NativeFailure> {
    emit_object_full(
        program,
        interner,
        EmitOptions::production(),
        target,
        backend,
        kind,
    )
    .map(|(bytes, _)| bytes)
}

/// As [`emit_object`], but also returns the codegen-vs-object-write [`ObjTimings`] split. The bytes
/// are identical to `emit_object`; only measurement is added.
pub fn emit_object_timed(
    program: &Program,
    interner: &Interner,
) -> Result<(Vec<u8>, ObjTimings), String> {
    emit_object_ex(program, interner, EmitOptions::production())
}

/// As [`emit_object_timed`], with an explicit backend configuration — so the timed object is the
/// one the requested `-O` level produces.
pub fn emit_object_timed_with(
    program: &Program,
    interner: &Interner,
    backend: BackendOpt,
) -> Result<(Vec<u8>, ObjTimings), String> {
    emit_object_timed_for_with(program, interner, TargetCpu::Native, backend)
}

/// As [`emit_object_timed_with`], for the instruction-set level `target` — what `wukongc
/// --time-passes` uses, so the timed object is exactly the one `--target-cpu` and `-O` produce.
pub fn emit_object_timed_for_with(
    program: &Program,
    interner: &Interner,
    target: TargetCpu,
    backend: BackendOpt,
) -> Result<(Vec<u8>, ObjTimings), String> {
    emit_object_kind_timed_for_with(program, interner, target, backend, ObjectKind::Library)
        .map_err(String::from)
}

/// As [`emit_object_timed_for_with`], for an object of `kind` (see [`emit_object_kind_for_with`]).
pub fn emit_object_kind_timed_for_with(
    program: &Program,
    interner: &Interner,
    target: TargetCpu,
    backend: BackendOpt,
    kind: ObjectKind,
) -> Result<(Vec<u8>, ObjTimings), NativeFailure> {
    emit_object_full(
        program,
        interner,
        EmitOptions::production(),
        target,
        backend,
        kind,
    )
}

/// As [`emit_object_kind_timed_for_with`], and with `debug` (`-g`) the object also carries DWARF:
/// a line table mapping every function's machine code to the source lines it was lowered from and
/// a compile unit with one subprogram per function (see the `debuginfo` module). The machine code
/// is the same with or without it; with `debug` `None` the object is byte-identical to
/// [`emit_object_kind_timed_for_with`]'s. ELF and COFF targets only.
pub fn emit_object_debug_for_with(
    program: &Program,
    interner: &Interner,
    target: TargetCpu,
    backend: BackendOpt,
    kind: ObjectKind,
    debug: Option<&DebugInfo<'_>>,
) -> Result<(Vec<u8>, ObjTimings), NativeFailure> {
    emit_object_debug_full(
        program,
        interner,
        EmitOptions::production(),
        target,
        backend,
        kind,
        debug,
    )
}

fn emit_object_ex(
    program: &Program,
    interner: &Interner,
    opts: EmitOptions,
) -> Result<(Vec<u8>, ObjTimings), String> {
    emit_object_full(
        program,
        interner,
        opts,
        TargetCpu::Native,
        BackendOpt::default(),
        ObjectKind::Library,
    )
    .map_err(String::from)
}

fn emit_object_full(
    program: &Program,
    interner: &Interner,
    opts: EmitOptions,
    target: TargetCpu,
    backend: BackendOpt,
    kind: ObjectKind,
) -> Result<(Vec<u8>, ObjTimings), NativeFailure> {
    emit_object_debug_full(program, interner, opts, target, backend, kind, None)
}

/// [`emit_object_full`], with DWARF when `debug` is `Some` (`-g`). Without it the object is
/// byte-identical to [`emit_object_full`]'s: nothing is recorded and nothing is added.
fn emit_object_debug_full(
    program: &Program,
    interner: &Interner,
    opts: EmitOptions,
    target: TargetCpu,
    backend: BackendOpt,
    kind: ObjectKind,
    debug: Option<&DebugInfo<'_>>,
) -> Result<(Vec<u8>, ObjTimings), NativeFailure> {
    target::check_vec_kernels(program, interner, target)?;
    let t_cg = std::time::Instant::now();
    let isa = make_isa_full(true, opts.verify, target, backend)?;
    emit_object_for_isa(isa, program, interner, opts.parallel, kind, debug, t_cg)
}

/// The object for `program` on `isa` (whose triple picks the container: COFF on Windows, ELF on
/// Linux), plus its DWARF when `debug` is `Some`. `t_cg` is when codegen timing started. Split out
/// so the DWARF tests can build an ELF object on a Windows host, and the reverse.
fn emit_object_for_isa(
    isa: std::sync::Arc<dyn cranelift_codegen::isa::TargetIsa>,
    program: &Program,
    interner: &Interner,
    parallel: bool,
    kind: ObjectKind,
    debug: Option<&DebugInfo<'_>>,
    t_cg: std::time::Instant,
) -> Result<(Vec<u8>, ObjTimings), NativeFailure> {
    use cranelift_object::{ObjectBuilder, ObjectModule};
    use std::time::Instant;

    let builder = ObjectBuilder::new(
        isa.clone(),
        "wukong",
        cranelift_module::default_libcall_names(),
    )
    .map_err(|e| e.to_string())?;
    let mut module = ObjectModule::new(builder);
    let mut log = debug.map(|_| debuginfo::CodeLog::new(true));
    populate_module(
        &mut module,
        program,
        interner,
        parallel,
        entry::SymbolMode::Object(kind),
        log.as_mut(),
    )?;
    let codegen = t_cg.elapsed();

    let t_ow = Instant::now();
    let mut product = module.finish();
    if let (Some(debug), Some(log)) = (debug, &log) {
        debuginfo::attach_dwarf(&mut product, &*isa, program, interner, log, debug)?;
    }
    let bytes = product.emit().map_err(|e| e.to_string())?;
    let object_write = t_ow.elapsed();

    Ok((
        bytes,
        ObjTimings {
            codegen,
            object_write,
        },
    ))
}

#[cfg(test)]
mod tests;

#[cfg(test)]
mod lane_op_tests;

#[cfg(test)]
mod assertion_tests;

#[cfg(test)]
mod error_tests;

#[cfg(test)]
mod srcloc_tests;

#[cfg(test)]
mod dwarf_tests;

#[cfg(test)]
mod target_tests;

#[cfg(test)]
mod fuzz;

#[cfg(test)]
mod fuzz_grammar;

#[cfg(test)]
mod backend_opt_tests;

mod guarded_kernel;
pub use guarded_kernel::{jit_compile_buffer_kernel, JitBufferKernel};

/// P11.7: a stack slot of 32 bytes or more has a 32-byte-aligned base at every call depth, so a
/// 256-bit access at its base never splits a cache line. The address is observable only in MIR
/// (the language has no pointer-to-integer cast), so the probes are built by hand.
#[cfg(test)]
mod stack_slot_alignment_tests {
    use super::*;
    use wukong_mir::Builder;

    /// `probe(d) -> i64`: allocate a wide slot of type `wide` and a 16-byte one, recurse `d` deep,
    /// and sum one packed word per frame — bits 32.. the wide base's misalignment (`addr & 31`),
    /// bits 16.. whether the small slot sat at 16 mod 32 (it is never realigned, so it reports the
    /// stack pointer's parity), bits 0.. one per call. Two callers with different frame sizes run it
    /// too, so the frames land at both parities whatever this function's own frame size is.
    fn probe_program(wide: MirType) -> (Program, Interner, Symbol) {
        let mut i = Interner::new();
        let probe = i.intern("probe");
        let print = i.intern("print");
        let main = i.intern("main");
        let mut program = Program::new();

        let mut b = Builder::new(probe, MirType::I64);
        let d = b.add_param(MirType::I64);
        let wide_slot = b.alloca(wide);
        let small_slot = b.alloca(MirType::Array(Box::new(MirType::I64), 2));
        let c = |b: &mut Builder, v: i128| b.build(MirType::I64, Op::ConstInt(v, MirType::I64));
        let zero = c(&mut b, 0);
        b.build_void(Op::Store {
            ptr: small_slot,
            value: zero,
        });
        let wide_addr = b.build(
            MirType::I64,
            Op::Cast(CastKind::PtrToInt, wide_slot, MirType::I64),
        );
        let small_addr = b.build(
            MirType::I64,
            Op::Cast(CastKind::PtrToInt, small_slot, MirType::I64),
        );
        let m31 = c(&mut b, 31);
        let m16 = c(&mut b, 16);
        let mis = b.build(MirType::I64, Op::Bin(BinOp::And, wide_addr, m31));
        let par = b.build(MirType::I64, Op::Bin(BinOp::And, small_addr, m16));
        let s32 = c(&mut b, 32);
        let s12 = c(&mut b, 12);
        let mis_field = b.build(MirType::I64, Op::Bin(BinOp::Shl, mis, s32));
        let par_field = b.build(MirType::I64, Op::Bin(BinOp::Shl, par, s12));
        let one = c(&mut b, 1);
        let word = b.build(MirType::I64, Op::Bin(BinOp::Or, mis_field, par_field));
        let word = b.build(MirType::I64, Op::Bin(BinOp::Or, word, one));
        let more = b.build(MirType::I1, Op::Cmp(CmpOp::Sgt, d, zero));
        let deeper = b.new_block();
        let leaf = b.new_block();
        b.cond_br(more, deeper, vec![], leaf, vec![]);
        b.switch_to(deeper);
        let d1 = b.build(MirType::I64, Op::Bin(BinOp::Sub, d, one));
        let rest = b.build(
            MirType::I64,
            Op::Call {
                func: probe,
                args: vec![d1],
            },
        );
        let sum = b.build(MirType::I64, Op::Bin(BinOp::Add, rest, word));
        b.ret(Some(sum));
        b.switch_to(leaf);
        b.ret(Some(word));
        program.funcs.push(b.finish());

        // Two pass-through callers whose own frames differ by 16 bytes (an 8- and a 24-byte slot).
        let mut pads = Vec::new();
        for (name, pad) in [("pad_a", 1u32), ("pad_b", 3)] {
            let sym = i.intern(name);
            let mut b = Builder::new(sym, MirType::I64);
            let d = b.add_param(MirType::I64);
            let slot = b.alloca(MirType::Array(Box::new(MirType::I64), pad));
            b.build_void(Op::Store {
                ptr: slot,
                value: d,
            });
            let r = b.build(
                MirType::I64,
                Op::Call {
                    func: probe,
                    args: vec![d],
                },
            );
            b.ret(Some(r));
            program.funcs.push(b.finish());
            pads.push(sym);
        }

        let mut b = Builder::new(main, MirType::I32);
        let depth = b.build(MirType::I64, Op::ConstInt(6, MirType::I64));
        let mut total = b.build(
            MirType::I64,
            Op::Call {
                func: probe,
                args: vec![depth],
            },
        );
        for pad in pads {
            let r = b.build(
                MirType::I64,
                Op::Call {
                    func: pad,
                    args: vec![depth],
                },
            );
            total = b.build(MirType::I64, Op::Bin(BinOp::Add, total, r));
        }
        b.build_void(Op::Call {
            func: print,
            args: vec![total],
        });
        let ok = b.build(MirType::I32, Op::ConstInt(0, MirType::I32));
        b.ret(Some(ok));
        program.funcs.push(b.finish());
        assert!(wukong_mir::verify::verify_program(&program).is_empty());
        (program, i, main)
    }

    /// Run the probe natively and return (summed wide misalignment, frames at 16 mod 32, calls).
    fn probe(wide: MirType) -> (i64, i64, i64) {
        let (program, interner, main) = probe_program(wide);
        let (code, out) = jit_run(&program, main, &interner).expect("jit");
        assert_eq!(code, 0);
        let total: i64 = String::from_utf8(out).unwrap().trim().parse().unwrap();
        (total >> 32, (total >> 16) & 0xffff, total & 0xffff)
    }

    fn assert_aligned_at_both_parities(wide: MirType) {
        let (misaligned, odd_frames, calls) = probe(wide.clone());
        // 7 frames from `main`'s direct call plus 7 under each of the two pads.
        assert_eq!(calls, 21, "{wide:?}");
        // Not vacuous: the frames really sat at both 32-byte parities, so a merely 16-aligned base
        // would have been misaligned in some of them.
        assert!(
            odd_frames > 0 && odd_frames < calls,
            "{wide:?}: every frame had one parity ({odd_frames}/{calls}), so the probe proves nothing"
        );
        assert_eq!(
            misaligned, 0,
            "{wide:?}: a wide slot base was not 32-byte aligned"
        );
    }

    #[test]
    fn stack_slot_alignment_f32x8_local_is_32_byte_aligned() {
        assert_aligned_at_both_parities(MirType::Array(Box::new(MirType::F32), 8));
    }

    #[test]
    fn stack_slot_alignment_f64x16_local_is_32_byte_aligned() {
        assert_aligned_at_both_parities(MirType::Array(Box::new(MirType::F64), 16));
    }

    /// The cap is per size class, and no access is ever *promised* aligned: interior addresses
    /// (`base + 4`) are not, and Cranelift may fault on a broken `aligned` promise.
    #[test]
    fn stack_slot_alignment_caps_and_never_promises_aligned_access() {
        assert_eq!(align_shift(1), 0);
        assert_eq!(align_shift(8), 3);
        assert_eq!(align_shift(16), 4);
        assert_eq!(align_shift(24), 4, "below 32 bytes the cap stays 16");
        assert_eq!(align_shift(32), 5);
        assert_eq!(align_shift(128), 5, "the cap is 32 bytes");
        assert_eq!(over_aligned_slot_bytes(32, WIDE_SLOT_BYTES), 48);
        assert_eq!(over_aligned_slot_bytes(24, 64), 72);
        assert_eq!(over_aligned_slot_bytes(u32::MAX, WIDE_SLOT_BYTES), u32::MAX);
        assert!(!mem_flags().aligned());
    }
}
