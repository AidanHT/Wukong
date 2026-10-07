//! Kernel-operand marshalling for the recognized-kernel arms of `Interp::intrinsic`: checked
//! extents, the aliasing contract each runtime kernel states, and typed scratch.
//!
//! # Why the oracle checks aliasing at all
//!
//! The interpreter cannot hand a runtime kernel its own memory — slots are tagged `Value`s, the
//! kernel wants contiguous `f32` — so every buffer arm copies its inputs into scratch, runs the
//! kernel there and writes the result back. That is copy-in/copy-out, and on OVERLAPPING operands it
//! computes a program neither the native backend nor the source loop has: the native kernel reads
//! values it has itself just overwritten, the source nest reads them in its own order, and the
//! oracle reads neither. Worse, the oracle's answer is the plausible one, so a differential failure
//! on such an input would name the native backend as the deviant side.
//!
//! The compiler now keeps such inputs away from every guarded family: `wukong_mir_build`'s
//! `plan_kernel_alias` and its run-time guard (`kernel_guard.rs`) dispatch a kernel only when its
//! operands are disjoint, or exactly in place where the runtime documents in-place operation, and
//! lower the source loop otherwise. So for those families copy-in/copy-out is unobservable — and
//! [`check_overlap`] makes that a checked statement rather than an assumption: a call that reaches
//! an arm with operands the kernel's contract forbids is reported as an error instead of being
//! answered with a value no backend computes. Exact in-place operation, where permitted, is equal
//! to out-of-place on a row-local or elementwise kernel, which is what makes copying sound there.
//!
//! # Why extents are checked before anything is allocated
//!
//! A kernel extent is a program value. Multiplying two of them unchecked overflowed (a debug-build
//! panic), and reserving scratch for a product that is positive but absurd reached the allocator's
//! abort path before any bounds check could reject it — an uncatchable process abort out of the
//! backend that is supposed to be the semantic oracle. Every helper here validates the whole operand
//! range against the arena FIRST and reserves fallibly SECOND, so a bad extent is a diagnostic.
//! (The bound is still the arena, not the allocation: provenance is not tracked per allocation.)
//!
//! # Why the f32 scratch is recycled
//!
//! `--run` defaults to this interpreter, and a recognized kernel inside a loop (a per-head GEMM, a
//! per-row norm) used to allocate and free a fresh `Vec` for every operand on every iteration. The
//! tagged-slot memory model makes the element conversion itself unavoidable, but the allocation is
//! not: [`read_f32`] and [`output_f32`] draw from a small per-thread pool and the arms hand their
//! buffers back with [`recycle_f32`]. A buffer that is not handed back (an error path) is simply
//! dropped, so the pool can only ever save work, never change a value — every buffer is cleared and
//! fully rewritten before a kernel sees it.

use std::cell::RefCell;

use crate::{kernel_buffer, kernel_scratch, kernel_values, Value};

/// How many recycled f32 buffers one thread keeps: enough for the widest arm (GEMM's three operands
/// plus a bias), small enough that the pool never matters to a program's footprint.
const F32_POOL_DEPTH: usize = 8;

/// The largest buffer the pool retains, in elements (16 MiB of `f32`). A larger operand is a
/// one-off; keeping its capacity alive for the rest of the run would trade the allocation saved for
/// a resident footprint the program never asked for.
const F32_POOL_MAX_RETAINED: usize = 1 << 22;

thread_local! {
    /// Recycled f32 kernel scratch for this thread's interpreter (see the module doc).
    static F32_POOL: RefCell<Vec<Vec<f32>>> = const { RefCell::new(Vec::new()) };
}

/// An empty f32 buffer with room for `len` elements — a recycled one when the pool has it, reserved
/// fallibly either way (an absurd `len` is a diagnostic, not an allocator abort).
pub(crate) fn f32_scratch(len: usize, name: &str) -> Result<Vec<f32>, String> {
    let mut buffer = F32_POOL
        .with(|pool| pool.borrow_mut().pop())
        .unwrap_or_default();
    buffer.clear();
    buffer
        .try_reserve_exact(len)
        .map_err(|_| format!("{name}: operand allocation failed"))?;
    Ok(buffer)
}

/// Hand a kernel buffer back to this thread's pool once the arm is done with it.
pub(crate) fn recycle_f32(buffer: Vec<f32>) {
    if buffer.capacity() == 0 || buffer.capacity() > F32_POOL_MAX_RETAINED {
        return;
    }
    F32_POOL.with(|pool| {
        let mut pool = pool.borrow_mut();
        if pool.len() < F32_POOL_DEPTH {
            pool.push(buffer);
        }
    });
}

/// `rows * cols` as a slot count, or a diagnostic instead of an overflowed product.
pub(crate) fn span(rows: usize, cols: usize, name: &str) -> Result<usize, String> {
    rows.checked_mul(cols)
        .ok_or_else(|| format!("{name}: operand extent overflow"))
}

/// Read `len` floating operand slots from `base` as `f32`, validating the whole range before any
/// scratch is reserved. A pointer, vector or undefined slot is an error, not a manufactured zero.
pub(crate) fn read_f32(
    memory: &[Value],
    base: usize,
    len: usize,
    name: &str,
) -> Result<Vec<f32>, String> {
    let values = kernel_values(memory, base, len, 1, name)?;
    let mut buffer = f32_scratch(values.len(), name)?;
    for &value in values {
        buffer.push(value.kernel_float(name)? as f32);
    }
    Ok(buffer)
}

/// [`output`] for an `f32` destination, drawing its storage from the recycled pool.
pub(crate) fn output_f32(
    memory: &[Value],
    base: usize,
    len: usize,
    name: &str,
) -> Result<Vec<f32>, String> {
    kernel_values(memory, base, len, 1, name)?;
    let mut buffer = f32_scratch(len, name)?;
    // Within the capacity `f32_scratch` reserved fallibly, so this cannot reallocate.
    buffer.resize(len, 0.0);
    Ok(buffer)
}

/// Read `len` integer operand slots from `base` as `i32` (index buffers: cross-entropy targets).
pub(crate) fn read_i32(
    memory: &[Value],
    base: usize,
    len: usize,
    name: &str,
) -> Result<Vec<i32>, String> {
    let values = kernel_values(memory, base, len, 1, name)?;
    kernel_buffer(values, name, |v| Ok(v.as_int() as i32))
}

/// Read `len` half-precision operand slots as their exact stored 16 bits — `f16` when `f16` is set,
/// `bf16` otherwise. The slot holds the value already rounded to the half grid, so the conversion
/// is exact and the buffer is bit-identical to the native backend's 2-byte storage.
pub(crate) fn read_half_bits(
    memory: &[Value],
    base: usize,
    len: usize,
    name: &str,
    f16: bool,
) -> Result<Vec<u16>, String> {
    let values = kernel_values(memory, base, len, 1, name)?;
    kernel_buffer(values, name, |v| {
        let f = v.kernel_float(name)? as f32;
        Ok(if f16 {
            wukong_runtime::f32_to_f16_bits(f)
        } else {
            wukong_runtime::f32_to_bf16_bits(f)
        })
    })
}

/// A zeroed output buffer for `len` slots at `base`, after validating the DESTINATION range — an
/// output the kernel overwrites need not be read, but it must exist before scratch is reserved.
pub(crate) fn output<T: Copy + Default>(
    memory: &[Value],
    base: usize,
    len: usize,
    name: &str,
) -> Result<Vec<T>, String> {
    kernel_values(memory, base, len, 1, name)?;
    let mut buffer = kernel_scratch(len, name)?;
    // Within the capacity `kernel_scratch` reserved fallibly, so this cannot reallocate.
    buffer.resize(len, T::default());
    Ok(buffer)
}

/// Write a kernel result back to `base..base + data.len()` with ONE range check, converting each
/// element with `to_value`.
pub(crate) fn write_back<T: Copy>(
    memory: &mut [Value],
    base: usize,
    data: &[T],
    name: &str,
    to_value: impl Fn(T) -> Value,
) -> Result<(), String> {
    let end = base
        .checked_add(data.len())
        .ok_or_else(|| format!("{name}: output out of bounds"))?;
    let slots = memory
        .get_mut(base..end)
        .ok_or_else(|| format!("{name}: output out of bounds"))?;
    for (slot, &value) in slots.iter_mut().zip(data) {
        *slot = to_value(value);
    }
    Ok(())
}

/// [`write_back`] for an `f32` result.
pub(crate) fn write_f32(
    memory: &mut [Value],
    base: usize,
    data: &[f32],
    name: &str,
) -> Result<(), String> {
    write_back(memory, base, data, name, |v| Value::Float(v as f64))
}

/// What a kernel's contract permits for one operand pair.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Overlap {
    /// The two ranges must not share a slot.
    Forbidden,
    /// The two ranges may be IDENTICAL (exact in-place operation, documented by the runtime entry)
    /// and must otherwise be disjoint. A shifted overlap is not in place: the kernel's vector body
    /// reads lanes before storing them, the source loop does not.
    ExactOnly,
}

/// Refuse a kernel call whose operands overlap in a way its contract forbids.
///
/// `ops` are `(base, len)` slot ranges; `pairs` name the constrained pairs by index. Empty ranges
/// constrain nothing. The compiler versions every guarded dispatch on the same test (in the same
/// slot address space, through `Gep`), so for those families this can only fire on a call that
/// bypassed the guard — a hole in the alias analysis, or a hand-written kernel call — and there the
/// honest oracle answer is "this call is outside the kernel's contract", not a copy-in/copy-out
/// value.
pub(crate) fn check_overlap(
    name: &str,
    ops: &[(usize, usize)],
    pairs: &[(usize, usize, Overlap)],
) -> Result<(), String> {
    for &(i, j, rule) in pairs {
        let (Some(&(a, a_len)), Some(&(b, b_len))) = (ops.get(i), ops.get(j)) else {
            return Err(format!(
                "{name}: operand index out of range in overlap check"
            ));
        };
        if a_len == 0 || b_len == 0 {
            continue;
        }
        if a.saturating_add(a_len) <= b || b.saturating_add(b_len) <= a {
            continue;
        }
        if rule == Overlap::ExactOnly && a == b && a_len == b_len {
            continue;
        }
        return Err(format!(
            "{name}: kernel operands {i} and {j} overlap (slots {a}..{} and {b}..{}), which the \
             kernel's contract forbids; a copied answer here would be a value no backend computes",
            a.saturating_add(a_len),
            b.saturating_add(b_len),
        ));
    }
    Ok(())
}

/// One output range checked against each read range, each pair under its own rule — the shape of
/// every recognized kernel (one destination, several sources). Ranges are `(base, len)` in slots.
pub(crate) fn check_output(
    name: &str,
    out: (usize, usize),
    reads: &[((usize, usize), Overlap)],
) -> Result<(), String> {
    let mut ops = vec![out];
    let mut pairs = Vec::new();
    for (i, &(range, rule)) in reads.iter().enumerate() {
        ops.push(range);
        pairs.push((0, i + 1, rule));
    }
    check_overlap(name, &ops, &pairs)
}

/// Whether the output range shares a slot with any read range WITHOUT being exactly that range —
/// the overlaps an elementwise kernel's vector body and its source loop disagree on. Empty ranges
/// overlap nothing; exact in-place operation is not "shifted".
pub(crate) fn shifted_overlap(out: (usize, usize), reads: &[(usize, usize)]) -> bool {
    let (a, a_len) = out;
    reads.iter().any(|&(b, b_len)| {
        a_len != 0
            && b_len != 0
            && a.saturating_add(a_len) > b
            && b.saturating_add(b_len) > a
            && (a, a_len) != (b, b_len)
    })
}

/// Run an ELEMENTWISE kernel exactly the way its source loop runs: one element at a time, in
/// ascending order, each input read from LIVE memory and each result stored before the next
/// element is read. `apply` computes one output element from that element's inputs (one value per
/// base in `inputs`, in order), by calling the runtime kernel at `n = 1` — the same per-element map
/// its vector body computes, which the runtime pins bit-exactly to the scalar twin.
///
/// This is the oracle's answer for a SHIFTED overlap between an elementwise kernel's output and an
/// input (`out = x + 1` through two pointer views): the source loop then feeds each store into a
/// later read, which neither a copy-in/copy-out run nor the kernel's 8-lane body reproduces. For a
/// disjoint or exactly-in-place call it computes the same bits as the whole-buffer call, which is
/// why the arms keep the faster whole-buffer path there.
///
/// Every range, and every input slot's type, is validated before the first store, so a bad operand
/// is a diagnostic with memory untouched. (A store only ever writes a float, so an input slot that
/// is a float before the loop stays one.)
pub(crate) fn elementwise_live(
    memory: &mut [Value],
    inputs: &[usize],
    out: usize,
    n: usize,
    name: &str,
    mut apply: impl FnMut(&[f32]) -> f32,
) -> Result<(), String> {
    kernel_values(memory, out, n, 1, name)?;
    for &base in inputs {
        for value in kernel_values(memory, base, n, 1, name)? {
            value.kernel_float(name)?;
        }
    }
    let mut lane = Vec::new();
    lane.try_reserve_exact(inputs.len())
        .map_err(|_| format!("{name}: operand allocation failed"))?;
    for t in 0..n {
        lane.clear();
        for &base in inputs {
            lane.push(memory[base + t].kernel_float(name)? as f32);
        }
        memory[out + t] = Value::Float(f64::from(apply(lane.as_slice())));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn overlap_rules_match_the_compiler_guard() {
        use Overlap::{ExactOnly, Forbidden};
        // Disjoint, adjacent ranges are fine under either rule.
        assert!(check_overlap("k", &[(0, 4), (4, 4)], &[(0, 1, Forbidden)]).is_ok());
        assert!(check_overlap("k", &[(4, 4), (0, 4)], &[(0, 1, ExactOnly)]).is_ok());
        // One shared slot is an overlap.
        assert!(check_overlap("k", &[(0, 5), (4, 4)], &[(0, 1, Forbidden)]).is_err());
        assert!(check_overlap("k", &[(0, 5), (4, 4)], &[(0, 1, ExactOnly)]).is_err());
        // Exactly in place: forbidden for a cross-element kernel, permitted for an in-place one.
        assert!(check_overlap("k", &[(8, 4), (8, 4)], &[(0, 1, Forbidden)]).is_err());
        assert!(check_overlap("k", &[(8, 4), (8, 4)], &[(0, 1, ExactOnly)]).is_ok());
        // A SHIFTED overlap is never in place, nor is a same-base range of another length.
        assert!(check_overlap("k", &[(8, 4), (9, 4)], &[(0, 1, ExactOnly)]).is_err());
        assert!(check_overlap("k", &[(8, 4), (8, 2)], &[(0, 1, ExactOnly)]).is_err());
        // Empty ranges constrain nothing; unlisted pairs are not checked.
        assert!(check_overlap("k", &[(8, 0), (8, 4)], &[(0, 1, Forbidden)]).is_ok());
        assert!(check_overlap("k", &[(0, 4), (0, 4), (8, 4)], &[(0, 2, Forbidden)]).is_ok());
        // A bad operand index is an error, never a silent pass.
        assert!(check_overlap("k", &[(0, 4)], &[(0, 1, Forbidden)]).is_err());
    }

    #[test]
    fn one_output_is_checked_against_each_read_under_its_own_rule() {
        use Overlap::{ExactOnly, Forbidden};
        // `out` exactly over `x` (permitted) and disjoint from the table.
        assert!(check_output("k", (0, 4), &[((0, 4), ExactOnly), ((8, 2), Forbidden)]).is_ok());
        // The same layout with the in-place pair under a `Forbidden` rule.
        let error = check_output("k", (0, 4), &[((0, 4), Forbidden)]).unwrap_err();
        assert!(error.contains("k:") && error.contains("overlap"), "{error}");
        // The table overlapping the output is refused even when the in-place pair is fine.
        assert!(check_output("k", (0, 4), &[((0, 4), ExactOnly), ((3, 2), Forbidden)]).is_err());
        // Two READS may overlap each other freely: only output pairs are constrained.
        assert!(check_output("k", (8, 4), &[((0, 4), Forbidden), ((2, 4), Forbidden)]).is_ok());
    }

    #[test]
    fn extents_are_checked_before_scratch_is_reserved() {
        let memory = vec![Value::Float(1.5); 8];
        assert!(span(usize::MAX, 2, "k").is_err());
        assert_eq!(span(3, 4, "k"), Ok(12));
        // In range: the values come back as f32.
        assert_eq!(read_f32(&memory, 2, 3, "k"), Ok(vec![1.5f32; 3]));
        // Out of range by one slot, and an absurd length: both diagnostics, neither an abort.
        assert!(read_f32(&memory, 6, 3, "k").is_err());
        assert!(read_f32(&memory, 0, usize::MAX, "k").is_err());
        assert!(output::<f32>(&memory, 0, usize::MAX / 2, "k").is_err());
        assert_eq!(output::<i32>(&memory, 4, 4, "k"), Ok(vec![0i32; 4]));
        // An undefined slot is an error, not a zero.
        let mut holes = memory.clone();
        holes[3] = Value::Unit;
        assert!(read_f32(&holes, 0, 8, "k").is_err());
    }

    #[test]
    fn shifted_overlap_excludes_disjoint_exact_and_empty_ranges() {
        // Disjoint and adjacent: no overlap.
        assert!(!shifted_overlap((4, 4), &[(0, 4)]));
        assert!(!shifted_overlap((0, 4), &[(4, 4)]));
        // Exactly in place is not shifted.
        assert!(!shifted_overlap((2, 4), &[(2, 4)]));
        // Empty ranges overlap nothing.
        assert!(!shifted_overlap((2, 0), &[(2, 4)]));
        assert!(!shifted_overlap((2, 4), &[(3, 0)]));
        // Shifted either way, and a same-base range of another length, are shifted.
        assert!(shifted_overlap((1, 4), &[(0, 4)]));
        assert!(shifted_overlap((0, 4), &[(1, 4)]));
        assert!(shifted_overlap((0, 4), &[(0, 2)]));
        // Any one of several reads suffices.
        assert!(shifted_overlap((8, 4), &[(0, 4), (9, 4)]));
    }

    /// The source loop's semantics on a FORWARD shift: `out = x + 1` makes every store feed the
    /// next element's read, so `out[t] = 2*x[t] + 1` over `[1, 2, 3, 4, 5]` chains to
    /// `3, 7, 15, 31` — where a copied input would give `3, 5, 7, 9`.
    #[test]
    fn elementwise_live_runs_the_source_loop_order() {
        let mut memory = [1.0, 2.0, 3.0, 4.0, 5.0].map(Value::Float).to_vec();
        elementwise_live(&mut memory, &[0], 1, 4, "k", |lane| 2.0 * lane[0] + 1.0).unwrap();
        let expected = [1.0, 3.0, 7.0, 15.0, 31.0].map(Value::Float);
        assert_eq!(memory, expected);
        // A BACKWARD shift reads ahead of every store, so it equals the copied answer:
        // `out = x - 1`, `out[t] = x[t] + 10` over `[1, 2, 3, 4, 5]` -> `12, 13, 14, 15, 5`.
        let mut memory = [1.0, 2.0, 3.0, 4.0, 5.0].map(Value::Float).to_vec();
        elementwise_live(&mut memory, &[1], 0, 4, "k", |lane| lane[0] + 10.0).unwrap();
        let expected = [12.0, 13.0, 14.0, 15.0, 5.0].map(Value::Float);
        assert_eq!(memory, expected);
        // Two inputs, read element by element: `out[t] = x[t] * y[t]` with `out == y + 1`.
        // x = [2, 2, 2], y = [1, 1, 1, 1] at slots 3..7: y[1] = 2*1, y[2] = 2*2, y[3] = 2*4.
        let mut memory = [2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0]
            .map(Value::Float)
            .to_vec();
        elementwise_live(&mut memory, &[0, 3], 4, 3, "k", |lane| lane[0] * lane[1]).unwrap();
        let expected = [2.0, 2.0, 2.0, 1.0, 2.0, 4.0, 8.0].map(Value::Float);
        assert_eq!(memory, expected);
    }

    #[test]
    fn elementwise_live_validates_before_the_first_store() {
        // An undefined input slot late in the range: nothing may have been written.
        let mut memory = vec![Value::Float(1.0); 6];
        memory[4] = Value::Unit;
        let before = memory.clone();
        assert!(elementwise_live(&mut memory, &[0], 1, 5, "k", |lane| lane[0]).is_err());
        assert_eq!(memory, before);
        // An output range past the arena: refused, memory untouched.
        assert!(elementwise_live(&mut memory, &[0], 3, 4, "k", |lane| lane[0]).is_err());
        assert_eq!(memory, before);
    }

    /// The pool can only save an allocation, never change a value: a recycled buffer comes back
    /// empty with at least the requested room, and an output is zeroed however dirty the buffer
    /// it was drawn from.
    #[test]
    fn recycled_scratch_is_cleared_before_reuse() {
        recycle_f32(vec![9.0f32; 64]);
        let buffer = f32_scratch(16, "k").unwrap();
        assert!(buffer.is_empty());
        assert!(buffer.capacity() >= 16);
        recycle_f32(buffer);
        recycle_f32(vec![7.0f32; 8]);
        let memory = vec![Value::Float(0.5); 8];
        assert_eq!(output_f32(&memory, 0, 8, "k").unwrap(), vec![0.0f32; 8]);
        assert_eq!(read_f32(&memory, 2, 3, "k").unwrap(), vec![0.5f32; 3]);
        // An absurd request is still a diagnostic, with or without a pooled buffer.
        assert!(f32_scratch(usize::MAX, "k").is_err());
    }

    #[test]
    fn write_back_is_one_range_check() {
        let mut memory = vec![Value::Float(0.0); 4];
        assert!(write_f32(&mut memory, 1, &[2.0, 3.0], "k").is_ok());
        assert_eq!(memory[1], Value::Float(2.0));
        assert_eq!(memory[2], Value::Float(3.0));
        // A write that would run past the arena changes nothing.
        assert!(write_f32(&mut memory, 3, &[9.0, 9.0], "k").is_err());
        assert_eq!(memory[3], Value::Float(0.0));
        assert!(write_back(&mut memory, 0, &[7i32], "k", |v| Value::Int(v as i128)).is_ok());
        assert_eq!(memory[0], Value::Int(7));
    }
}
