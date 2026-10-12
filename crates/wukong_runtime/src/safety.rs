//! The language's runtime safety failures, shared by every execution path.
//!
//! A slice access `s[i]` is bounds-checked against the length in the slice's own fat pointer: the
//! lowering emits `i <u len` and, on the failure edge, a call to `wukong_rt_bounds_fail(i, len,
//! site)` where `site` is a NUL-terminated `file:line:col` string in `.rodata` (empty when the
//! compiler had no source map). The three execution paths must say the same thing and stop at the
//! same point, so the message is formatted here, once:
//!
//! * the interpreter returns it as its run error (the driver prints `error: {message}`, exit 1);
//! * the Cranelift JIT records it and returns through its callers like a failed `assert`, and
//!   `JitProgram::run` reports it as the run error;
//! * a linked executable resolves `wukong_rt_bounds_fail` to the C-ABI entry below, which prints
//!   the same `error: {message}` line and exits 1 (the `cc` fallback runtime mirrors it in C).

use std::io::Write;

/// The diagnostic for an out-of-bounds slice access. `site` is the `file:line:col` the compiler
/// recorded, or empty.
pub fn bounds_failure_message(index: i64, len: i64, site: &str) -> String {
    if site.is_empty() {
        format!("index out of bounds: the len is {len} but the index is {index}")
    } else {
        format!("index out of bounds: the len is {len} but the index is {index} at {site}")
    }
}

/// [`check_failure_message`] kind: a range index `s[lo..hi]` whose start is past its end
/// (`a = lo`, `b = hi`).
pub const CHECK_RANGE_ORDER: i64 = 1;
/// [`check_failure_message`] kind: a range index whose end is past the slice's length
/// (`a = hi`, `b = len`).
pub const CHECK_RANGE_END: i64 = 2;
/// [`check_failure_message`] kind: `str_from_utf8` of bytes that are not UTF-8 (`a` = the packed
/// fault of `crate::utf8::pack_fault`, `b` = the byte length).
pub const CHECK_UTF8: i64 = 3;

/// The diagnostic for a failed run-time check other than an element access (which keeps
/// [`bounds_failure_message`]): `wukong_rt_check_fail(kind, a, b, site)`. The wording of the range
/// checks is Rust's, so `s[5..3]` no longer reports its start as an "index" against a "len" that is
/// really its end.
pub fn check_failure_message(kind: i64, a: i64, b: i64, site: &str) -> String {
    let what = match kind {
        CHECK_RANGE_ORDER => format!("slice index starts at {a} but ends at {b}"),
        CHECK_RANGE_END => format!("range end index {a} out of range for slice of length {b}"),
        CHECK_UTF8 => match crate::utf8::unpack_fault(a) {
            Some(crate::utf8::Utf8Fault {
                valid_up_to,
                error_len: Some(n),
            }) => format!(
                "str_from_utf8: invalid utf-8 sequence of {n} bytes from index {valid_up_to} \
                 (the len is {b})"
            ),
            Some(f) => format!(
                "str_from_utf8: incomplete utf-8 byte sequence from index {} (the len is {b})",
                f.valid_up_to
            ),
            None => format!("str_from_utf8: invalid utf-8 (the len is {b})"),
        },
        _ => format!("run-time check {kind} failed ({a}, {b})"),
    };
    if site.is_empty() {
        what
    } else {
        format!("{what} at {site}")
    }
}

/// Read a NUL-terminated site string. Lossy, because the compiler only ever writes UTF-8 here.
///
/// # Safety
/// `site` must be null or point at a NUL-terminated byte string valid for the duration of the call.
pub unsafe fn site_str(site: *const u8) -> String {
    if site.is_null() {
        return String::new();
    }
    // The caller guarantees a NUL terminator inside the same allocation; `CStr` does the walk.
    std::ffi::CStr::from_ptr(site as *const std::os::raw::c_char)
        .to_string_lossy()
        .into_owned()
}

/// The ahead-of-time (`--emit=exe`) entry for a failed slice bounds check: print the shared
/// diagnostic and exit with the same status `wukongc --run` reports for a runtime failure (1).
/// Output already written by the program stays written, exactly as it does for a failed `assert`.
///
/// The JIT never binds this symbol: it registers its own recording entry so an in-process run can
/// report the failure without exiting the host process.
#[allow(clippy::not_unsafe_ptr_arg_deref)]
#[no_mangle]
pub extern "C" fn wukong_rt_bounds_fail(index: i64, len: i64, site: *const u8) {
    // SAFETY-EXEMPT: `site` is a compiler-emitted NUL-terminated `.rodata` string -- no extent.
    // SAFETY: `site` is a `.rodata` string the compiler emitted with a NUL terminator.
    let site = unsafe { site_str(site) };
    let _ = std::io::stdout().flush();
    eprintln!("error: {}", bounds_failure_message(index, len, &site));
    std::process::exit(1);
}

/// The ahead-of-time entry for every other failed run-time check (a range index, `str_from_utf8`):
/// [`check_failure_message`], printed and exited on exactly as [`wukong_rt_bounds_fail`] does. The
/// JIT binds its own recording entry instead.
#[allow(clippy::not_unsafe_ptr_arg_deref)]
#[no_mangle]
pub extern "C" fn wukong_rt_check_fail(kind: i64, a: i64, b: i64, site: *const u8) {
    // SAFETY-EXEMPT: `site` is a compiler-emitted NUL-terminated `.rodata` string -- no extent.
    // SAFETY: `site` is a `.rodata` string the compiler emitted with a NUL terminator.
    let site = unsafe { site_str(site) };
    let _ = std::io::stdout().flush();
    eprintln!("error: {}", check_failure_message(kind, a, b, &site));
    std::process::exit(1);
}

// ---------------------------------------------------------------------------------------------
// P14.5: debug-only precondition checks for the C-ABI kernel entries.
//
// Every exported pointer-taking `extern "C" fn wukong_*` calls these at its top, against the extents
// and aliasing rules its `# Safety` section documents, or carries a `// SAFETY-EXEMPT: <reason>`
// comment saying why there is nothing to check (a NUL-terminated string has no extent, an opaque
// closure environment has no shape). `entry_law::every_pointer_entry_checks_its_contract` below
// enforces that textually over every runtime module.
//
// The checks compile OUT of the release perf builds (each body is `if cfg!(debug_assertions)`) and
// are a no-op for a zero or negative extent, which every entry treats as a legal no-op. An entry
// taking a SHAPE (`rows, cols`; GEMM's `m, k, n`) wraps its checks in
// `if cfg!(debug_assertions) && <every dimension> > 0`, its own no-op condition negated, because a
// derived extent is not enough: two negative dimensions multiply to a positive span (`-7 * -1`, and
// `i64::MIN * -1` saturates to `i64::MAX`), and a one-dimension extent (`b: cols`) is positive while
// the OTHER dimension is empty. So a call the entry treats as a no-op never asserts, whatever
// pointers it carries (`tests::no_op_calls_never_assert`). They cannot check a *memory bound*: a
// C-ABI entry carries no capacity, only the extent, so the facts expressible from
// `(pointer, extent)` are that a nonempty buffer pointer is non-null and that the extent does not
// overflow the `p.add(n)` the kernel performs, plus -- for entries whose `# Safety` contract
// documents operands as disjoint, or as "exactly in place, otherwise disjoint" -- that two ranges
// the caller promised not to overlap (or to overlap only exactly) in fact do not. A firing assert
// on a legitimate call is a contract bug: fix the contract or the caller, never weaken the assert.
// A failure aborts the process (a panic cannot unwind out of an `extern "C"` entry), which in a
// debug build is the point: the alternative is undefined behavior.
// ---------------------------------------------------------------------------------------------

/// The one failure path of the checks below: a panic reported at the ENTRY's call site (every
/// helper is `#[track_caller]`), so the message names the kernel whose contract was broken.
#[cold]
#[inline(never)]
#[track_caller]
fn contract_violation(what: &str) {
    panic!("P14.5 kernel contract violated: {what}");
}

/// Assert a single buffer's extent is castable and, when nonempty, its pointer is non-null and the
/// `n`-element span does not overrun `isize::MAX` (the precondition of `p.add(n)`).
#[inline(always)]
#[track_caller]
pub(crate) fn debug_extent<T>(p: *const T, n: i64) {
    if cfg!(debug_assertions)
        && n > 0
        && (p.is_null() || (n as usize) > isize::MAX as usize / core::mem::size_of::<T>().max(1))
    {
        contract_violation(
            "a nonempty extent's pointer is null, or the extent overflows isize::MAX",
        );
    }
}

/// [`debug_extent`] for an OPTIONAL operand: null means "absent" (a norm with no `gamma`, a GEMM
/// epilogue with no bias) and is legal at any extent; a non-null pointer must satisfy the extent.
#[inline(always)]
#[track_caller]
pub(crate) fn debug_extent_opt<T>(p: *const T, n: i64) {
    if cfg!(debug_assertions) && !p.is_null() {
        debug_extent(p, n);
    }
}

/// Two-dimensional form of [`debug_extent`]: the span is `rows*cols` elements. A `rows*cols` that
/// overflows `usize` is itself a contract violation and trips the assert.
#[inline(always)]
#[track_caller]
pub(crate) fn debug_extent_2d<T>(p: *const T, rows: i64, cols: i64) {
    if !cfg!(debug_assertions) || rows <= 0 || cols <= 0 {
        return;
    }
    match (rows as usize).checked_mul(cols as usize) {
        Some(n) if n <= i64::MAX as usize => debug_extent(p, n as i64),
        _ => contract_violation("rows*cols overflows the addressable range"),
    }
}

/// A byte buffer of `n` elements `elem` bytes wide -- the dtype-coded entries (`wukong_rt_read_dt`,
/// the format sink's slice operand), whose pointer is `*u8` but whose extent counts elements.
#[inline(always)]
#[track_caller]
pub(crate) fn debug_extent_bytes(p: *const u8, n: i64, elem: usize) {
    if !cfg!(debug_assertions) || n <= 0 {
        return;
    }
    match (n as usize).checked_mul(elem.max(1)) {
        Some(b) if b <= i64::MAX as usize => debug_extent(p, b as i64),
        _ => contract_violation("element count x width overflows the addressable range"),
    }
}

/// Byte range `[p, p + n*size_of::<T>())`, saturating.
#[inline(always)]
fn byte_range<T>(p: *const T, n: i64) -> (usize, usize) {
    let a0 = p as usize;
    let a1 = a0.saturating_add((n as usize).saturating_mul(core::mem::size_of::<T>()));
    (a0, a1)
}

/// Assert that two ranges a caller's contract documents as DISJOINT do not overlap. Only valid for
/// entries that forbid overlap between the two given operands; never call it on a pair an entry
/// permits to alias in place (that is [`debug_inplace_or_disjoint`]).
#[inline(always)]
#[track_caller]
pub(crate) fn debug_disjoint<T, U>(a: *const T, an: i64, b: *const U, bn: i64) {
    if !cfg!(debug_assertions) || an <= 0 || bn <= 0 {
        return;
    }
    let (a0, a1) = byte_range(a, an);
    let (b0, b1) = byte_range(b, bn);
    if !(a1 <= b0 || b1 <= a0) {
        contract_violation("operands documented disjoint have overlapping byte ranges");
    }
}

/// The "exactly in place, otherwise disjoint" aliasing class: the two ranges either start at the
/// same address (`out == x`, the in-place map every such entry supports) or do not overlap at all.
/// A SHIFTED overlap -- `out` a few elements into `x` -- is not the elementwise source loop, which is
/// why the class forbids it.
#[inline(always)]
#[track_caller]
pub(crate) fn debug_inplace_or_disjoint<T, U>(a: *const T, an: i64, b: *const U, bn: i64) {
    if !cfg!(debug_assertions) || an <= 0 || bn <= 0 {
        return;
    }
    let (a0, a1) = byte_range(a, an);
    let (b0, b1) = byte_range(b, bn);
    if !(a0 == b0 || a1 <= b0 || b1 <= a0) {
        contract_violation("operands documented exactly-in-place-or-disjoint overlap shifted");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn debug_precondition_helpers_accept_legal_and_reject_overlap() {
        // Legal: zero/negative extents are no-ops; a well-formed buffer passes.
        let buf = [0.0f32; 8];
        debug_extent(buf.as_ptr(), 0);
        debug_extent(buf.as_ptr(), -3);
        debug_extent(buf.as_ptr(), 8);
        debug_extent_2d(buf.as_ptr(), 2, 4);
        debug_extent_2d(buf.as_ptr(), 0, 4);
        // Disjoint ranges pass; the degenerate-extent short-circuit passes.
        let other = [0.0f32; 8];
        debug_disjoint(buf.as_ptr(), 8, other.as_ptr(), 8);
        debug_disjoint(buf.as_ptr(), 0, buf.as_ptr(), 8);
        // Exactly in place, or disjoint, passes the in-place class; null is "absent" for an
        // optional operand at any extent.
        debug_inplace_or_disjoint(buf.as_ptr(), 8, buf.as_ptr(), 8);
        debug_inplace_or_disjoint(buf.as_ptr(), 8, other.as_ptr(), 8);
        debug_inplace_or_disjoint(buf.as_ptr(), 4, buf[4..].as_ptr(), 4);
        debug_extent_opt(std::ptr::null::<f32>(), 8);
        debug_extent_opt(buf.as_ptr(), 8);
        debug_extent_bytes(buf.as_ptr().cast::<u8>(), 8, 4);
        debug_extent_bytes(std::ptr::null(), 0, 4);
        // Every violation is rejected (only observable in a debug build; release compiles the
        // checks out). Each runs through `catch_unwind` here because these helpers are ordinary
        // functions; inside an `extern "C"` entry the same panic aborts.
        if cfg!(debug_assertions) {
            let trips = |what: &str, f: &dyn Fn()| {
                let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(f));
                assert!(r.is_err(), "{what} must trip the assert");
            };
            trips("overlapping documented-disjoint ranges", &|| {
                debug_disjoint(buf.as_ptr(), 8, buf.as_ptr(), 8)
            });
            trips("a shifted in-place overlap", &|| {
                debug_inplace_or_disjoint(buf.as_ptr(), 8, buf[1..].as_ptr(), 7)
            });
            trips("a null nonempty buffer", &|| {
                debug_extent(std::ptr::null::<f32>(), 1)
            });
            trips("a null nonempty 2-D buffer", &|| {
                debug_extent_2d(std::ptr::null::<f32>(), 2, 3)
            });
            trips("an extent past isize::MAX bytes", &|| {
                debug_extent(buf.as_ptr(), i64::MAX)
            });
            trips("a rows*cols overflow", &|| {
                debug_extent_2d(buf.as_ptr(), i64::MAX, 4)
            });
            trips("a null nonempty byte buffer", &|| {
                debug_extent_bytes(std::ptr::null(), 2, 8)
            });
        }
    }

    /// The checks' no-op guarantee, end to end: a call the ENTRY treats as a no-op -- some
    /// dimension non-positive, so it returns before touching memory -- never asserts, whatever
    /// pointers it carries. Every shape-taking entry is driven through every non-positive mix of
    /// its dimensions with two operand layouts: all null, and all shifted views of one buffer. A
    /// check that runs on such a call trips on one of them (a positive extent on null, or an
    /// overlapping documented-disjoint / shifted in-place pair). The traps this pins: the product
    /// of two negative dimensions is positive (`-7 * -1`, and `i64::MIN * -1` saturates to
    /// `i64::MAX`), a one-dimension extent (`b: cols`) is positive when only the OTHER dimension
    /// is empty, and the GEMM family is a no-op for `k <= 0` too. A violation aborts the test
    /// binary (a panic cannot unwind out of an `extern "C"` entry). Release builds compile the
    /// checks out, so there the sweep only proves the entries' early returns.
    #[test]
    #[allow(clippy::too_many_lines)]
    fn no_op_calls_never_assert() {
        // 4 is the one positive value: a tuple is a no-op exactly when some entry dimension is not.
        const V: [i64; 5] = [4, 0, -1, -7, i64::MIN];
        let mut buf = [0.0f32; 16];
        let base = buf.as_mut_ptr();
        let mut calls = 0usize;
        for shifted in [false, true] {
            // Operand `i`: null, or a view `i` elements into the one shared buffer.
            let p = |i: usize| {
                if shifted {
                    base.wrapping_add(i)
                } else {
                    std::ptr::null_mut()
                }
            };
            let (p0, p1, p2, p3, p4) = (p(0), p(1), p(2), p(3), p(4));
            let (h0, h1) = (p0.cast::<u16>(), p1.cast::<u16>());
            let i32p = |q: *mut f32| q.cast::<i32>();
            for a in V {
                for b in V {
                    if a > 0 && b > 0 {
                        continue;
                    }
                    calls += 1;
                    // SAFETY: `a` or `b` is non-positive, a no-op dimension for every entry here,
                    // so each returns before dereferencing an operand.
                    unsafe {
                        crate::wukong_attention_f32(p0, p1, p2, p3, a, b, 1.0, 0);
                        crate::wukong_bias_bcast_f32(p0, p1, p2, a, b, 0);
                        crate::wukong_bias_bcast_f32_parallel(p0, p1, p2, a, b, 0);
                        crate::wukong_colargmax_i32(p0, i32p(p1), a, b);
                        crate::wukong_colargmax_i32_parallel(p0, i32p(p1), a, b);
                        crate::wukong_colargmin_i32(p0, i32p(p1), a, b);
                        crate::wukong_colargmin_i32_parallel(p0, i32p(p1), a, b);
                        crate::wukong_colsum_f32(p0, p1, a, b);
                        crate::wukong_colsum_f32_parallel(p0, p1, a, b);
                        crate::wukong_colmax_f32(p0, p1, a, b);
                        crate::wukong_colmax_f32_parallel(p0, p1, a, b);
                        crate::wukong_colmin_f32(p0, p1, a, b);
                        crate::wukong_colmin_f32_parallel(p0, p1, a, b);
                        crate::wukong_colmaxabs_f32(p0, p1, a, b);
                        crate::wukong_colmaxabs_f32_parallel(p0, p1, a, b);
                        crate::wukong_colmean_f32(p0, p1, a, b);
                        crate::wukong_colmean_f32_parallel(p0, p1, a, b);
                        crate::wukong_coll2_f32(p0, p1, a, b);
                        crate::wukong_coll2_f32_parallel(p0, p1, a, b);
                        crate::wukong_colrms_f32(p0, p1, a, b);
                        crate::wukong_colrms_f32_parallel(p0, p1, a, b);
                        crate::wukong_colsumsq_f32(p0, p1, a, b);
                        crate::wukong_colsumsq_f32_parallel(p0, p1, a, b);
                        crate::wukong_cummax_f32(p0, p1, a, b);
                        crate::wukong_cummax_f32_parallel(p0, p1, a, b);
                        crate::wukong_cummin_f32(p0, p1, a, b);
                        crate::wukong_cummin_f32_parallel(p0, p1, a, b);
                        crate::wukong_cumprod_f32(p0, p1, a, b);
                        crate::wukong_cumprod_f32_parallel(p0, p1, a, b);
                        crate::wukong_cumsum_f32(p0, p1, a, b);
                        crate::wukong_cumsum_f32_parallel(p0, p1, a, b);
                        crate::wukong_dequant_perchan_f32(p0.cast::<u8>(), p1, a, b, p2, 0);
                        crate::wukong_dequant_perchan_f32_parallel(
                            p0.cast::<u8>(),
                            p1,
                            a,
                            b,
                            p2,
                            0,
                        );
                        crate::wukong_entropy_f32(p0, p1, a, b);
                        crate::wukong_entropy_f32_parallel(p0, p1, a, b);
                        crate::wukong_sgemv(p0, p1, p2, a, b);
                        crate::wukong_sgemv_parallel(p0, p1, p2, a, b);
                        crate::wukong_sgemv_alpha(p0, p1, p2, a, b, 1.0);
                        crate::wukong_sgemv_alpha_parallel(p0, p1, p2, a, b, 1.0);
                        crate::wukong_sgevm_f32(p0, p1, p2, a, b, 1.0);
                        crate::wukong_sgevm_f32_parallel(p0, p1, p2, a, b, 1.0);
                        crate::wukong_kd_loss_f32(p0, p1, p2, a, b);
                        crate::wukong_kd_loss_f32_parallel(p0, p1, p2, a, b);
                        crate::wukong_kldiv_f32(p0, p1, p2, a, b);
                        crate::wukong_kldiv_f32_parallel(p0, p1, p2, a, b);
                        crate::wukong_layernorm_bwd_f32(p0, p1, p2, p3, a, b, 0);
                        crate::wukong_layernorm_bwd_f32_parallel(p0, p1, p2, p3, a, b, 0);
                        crate::wukong_logsoftmax_f32(p0, p1, a, b);
                        crate::wukong_logsoftmax_f32_parallel(p0, p1, a, b);
                        crate::wukong_logsumexp_f32(p0, p1, a, b);
                        crate::wukong_logsumexp_f32_parallel(p0, p1, a, b);
                        crate::wukong_lrscan_f32(p0, p1, p2, a, b);
                        crate::wukong_lrscan_f32_parallel(p0, p1, p2, a, b);
                        crate::wukong_norm_f32(p0, p1, a, b, 0, 0);
                        crate::wukong_norm_f32_parallel(p0, p1, a, b, 0, 0);
                        crate::wukong_norm_affine_f32(p0, p1, p2, p3, a, b, 0, 0);
                        crate::wukong_norm_affine_f32_parallel(p0, p1, p2, p3, a, b, 0, 0);
                        crate::wukong_rmsnorm_bwd_f32(p0, p1, p2, p3, a, b, 0);
                        crate::wukong_rmsnorm_bwd_f32_parallel(p0, p1, p2, p3, a, b, 0);
                        crate::wukong_rope_f32(p0, p1, p2, a, b);
                        crate::wukong_rope_f32_parallel(p0, p1, p2, a, b);
                        crate::wukong_rope_bwd_f32(p0, p1, p2, a, b);
                        crate::wukong_rope_bwd_f32_parallel(p0, p1, p2, a, b);
                        crate::wukong_rowargmax_i32(p0, i32p(p1), a, b);
                        crate::wukong_rowargmax_i32_parallel(p0, i32p(p1), a, b);
                        crate::wukong_rowargmin_i32(p0, i32p(p1), a, b);
                        crate::wukong_rowargmin_i32_parallel(p0, i32p(p1), a, b);
                        crate::wukong_softmax_bwd_f32(p0, p1, p2, a, b);
                        crate::wukong_softmax_bwd_f32_parallel(p0, p1, p2, a, b);
                        crate::wukong_transpose_f32(p0, p1, a, b);
                        crate::wukong_transpose_f32_parallel(p0, p1, a, b);
                        crate::wukong_transpose_u16(h0, h1, a, b);
                        crate::wukong_transpose_u16_parallel(h0, h1, a, b);
                        crate::wukong_xent_fwd_f32(p0, i32p(p1), p2, a, b);
                        crate::wukong_xent_fwd_f32_parallel(p0, i32p(p1), p2, a, b);
                        crate::wukong_xent_bwd_f32(p0, i32p(p1), p2, a, b);
                        crate::wukong_xent_bwd_f32_parallel(p0, i32p(p1), p2, a, b);
                        // `(n, ncoeff)`: either non-positive is the no-op.
                        crate::wukong_vhorner_f32(p0, p1, a, p2, b);
                        // `(t, h)` with any `v`; and scatter's `(t, h, v)` below.
                        for v in V {
                            crate::wukong_embedding_f32(p0, p1, i32p(p2), a, b, v);
                            crate::wukong_embedding_f32_parallel(p0, p1, i32p(p2), a, b, v);
                        }
                    }
                    // `(m, k, n)`: the GEMM family returns when ANY of the three is non-positive,
                    // `k` included.
                    for c in V {
                        // SAFETY: each `(m, k, n)` holds a non-positive `a` or `b`, a no-op
                        // dimension for every entry here, so none dereferences an operand.
                        unsafe {
                            for (m, k, n) in [(a, b, c), (a, c, b), (c, a, b)] {
                                crate::wukong_sgemm(p0, p1, p2, m, k, n, 0);
                                crate::wukong_sgemm_parallel(p0, p1, p2, m, k, n, 0);
                                crate::wukong_sgemm_nt(p0, p1, p2, m, k, n, 0);
                                crate::wukong_sgemm_nt_parallel(p0, p1, p2, m, k, n, 0);
                                crate::wukong_sgemm_nt_epi(p0, p1, p2, m, k, n, 0, p3, 0);
                                crate::wukong_sgemm_nt_epi_parallel(p0, p1, p2, m, k, n, 0, p3, 0);
                                crate::wukong_sgemm_nt_alpha(p0, p1, p2, m, k, n, 0, 1.0);
                                crate::wukong_sgemm_nt_alpha_parallel(p0, p1, p2, m, k, n, 0, 1.0);
                                crate::wukong_sgemm_tn(p0, p1, p2, m, k, n, 0);
                                crate::wukong_sgemm_tn_parallel(p0, p1, p2, m, k, n, 0);
                                crate::wukong_sgemm_bf16_nt(h0, h1, p2, m, k, n, 0);
                                crate::wukong_sgemm_bf16_nt_parallel(h0, h1, p2, m, k, n, 0);
                                crate::wukong_sgemm_f16_nt(h0, h1, p2, m, k, n, 0);
                                crate::wukong_sgemm_f16_nt_parallel(h0, h1, p2, m, k, n, 0);
                                crate::wukong_sgemm_bf16_nt_epi(h0, h1, p2, m, k, n, 0, p3, 0);
                                crate::wukong_sgemm_bf16_nt_epi_parallel(
                                    h0, h1, p2, m, k, n, 0, p3, 0,
                                );
                                crate::wukong_sgemm_f16_nt_epi(h0, h1, p2, m, k, n, 0, p3, 0);
                                crate::wukong_sgemm_f16_nt_epi_parallel(
                                    h0, h1, p2, m, k, n, 0, p3, 0,
                                );
                                crate::wukong_sgemm_bf16_tn(h0, h1, p2, m, k, n, 0);
                                crate::wukong_sgemm_bf16_tn_parallel(h0, h1, p2, m, k, n, 0);
                                crate::wukong_sgemm_f16_tn(h0, h1, p2, m, k, n, 0);
                                crate::wukong_sgemm_f16_tn_parallel(h0, h1, p2, m, k, n, 0);
                                let (u8p, i8p) = (p0.cast::<u8>(), p1.cast::<i8>());
                                crate::wukong_i8gemm_nt(u8p, i8p, i32p(p2), m, k, n);
                                crate::wukong_i8gemm_nt_parallel(u8p, i8p, i32p(p2), m, k, n);
                                crate::wukong_i8gemm_nt_deq(u8p, i8p, p2, m, k, n, 1.0, p3, p4, 0);
                                crate::wukong_i8gemm_nt_deq_parallel(
                                    u8p, i8p, p2, m, k, n, 1.0, p3, p4, 0,
                                );
                                crate::wukong_scatter_add_f32(p0, p1, i32p(p2), m, k, n);
                                crate::wukong_scatter_add_f32_parallel(p0, p1, i32p(p2), m, k, n);
                            }
                        }
                    }
                }
            }
            // Pooling: a no-op for `channels <= 0` or for a window that yields no output.
            for ch in V {
                for (h, w, kh, kw, sh, sw) in [
                    (4, 4, 2, 2, 2, 2),
                    (0, 4, 2, 2, 2, 2),
                    (-7, -1, 2, 2, 2, 2),
                    (4, 4, 5, 2, 2, 2),
                    (4, 4, 2, 2, 0, -1),
                    (i64::MIN, -1, -1, -1, 1, 1),
                ] {
                    if ch > 0 && crate::pool2d::pool2d_output_dims(h, w, kh, kw, sh, sw).is_some() {
                        continue;
                    }
                    // SAFETY: no channels or no output window -- each entry returns untouched.
                    unsafe {
                        crate::wukong_maxpool2d_f32(p0, p1, ch, h, w, kh, kw, sh, sw);
                        crate::wukong_maxpool2d_f32_parallel(p0, p1, ch, h, w, kh, kw, sh, sw);
                        crate::wukong_avgpool2d_f32(p0, p1, ch, h, w, kh, kw, sh, sw);
                        crate::wukong_avgpool2d_f32_parallel(p0, p1, ch, h, w, kh, kw, sh, sw);
                    }
                }
            }
            // One-extent entries: every helper is already a no-op for `n <= 0`.
            for n in V.into_iter().filter(|&n| n <= 0) {
                // SAFETY: `n <= 0` is a no-op for every entry here.
                unsafe {
                    crate::wukong_vmath_f32(p0, p1, n, 0);
                    crate::wukong_vmath_f32_parallel(p0, p1, n, 0);
                    crate::wukong_vmath2_f32(p0, p1, p2, n, 0);
                    crate::wukong_velem_f32(p0, p1, p2, n, 1.0, 1.0, 0.0, -1);
                    crate::wukong_velem_f32_parallel(p0, p1, p2, n, 1.0, 1.0, 0.0, -1);
                    crate::wukong_dequant_f32(p0.cast::<u8>(), p1, n, 1.0, 0);
                    crate::wukong_dequant_f32_parallel(p0.cast::<u8>(), p1, n, 1.0, 0);
                    crate::wukong_rand_fill_f32(p0, n, 0, 0);
                }
            }
        }
        assert!(calls >= 2 * 24, "swept only {calls} two-dimension tuples");
    }

    #[test]
    fn bounds_message_names_len_index_and_site() {
        assert_eq!(
            bounds_failure_message(4, 4, ""),
            "index out of bounds: the len is 4 but the index is 4"
        );
        assert_eq!(
            bounds_failure_message(-1, 3, "main.wk:7:12"),
            "index out of bounds: the len is 3 but the index is -1 at main.wk:7:12"
        );
        assert_eq!(
            check_failure_message(CHECK_RANGE_ORDER, 5, 3, "m.wk:2:9"),
            "slice index starts at 5 but ends at 3 at m.wk:2:9"
        );
        assert_eq!(
            check_failure_message(CHECK_RANGE_END, 9, 6, ""),
            "range end index 9 out of range for slice of length 6"
        );
        let fault = |valid_up_to, error_len| {
            crate::utf8::pack_fault(Some(crate::utf8::Utf8Fault {
                valid_up_to,
                error_len,
            }))
        };
        assert_eq!(
            check_failure_message(CHECK_UTF8, fault(3, Some(1)), 7, ""),
            "str_from_utf8: invalid utf-8 sequence of 1 bytes from index 3 (the len is 7)"
        );
        assert_eq!(
            check_failure_message(CHECK_UTF8, fault(5, None), 6, "a.wk:3:1"),
            "str_from_utf8: incomplete utf-8 byte sequence from index 5 (the len is 6) at a.wk:3:1"
        );
        let site = b"a.wk:1:2\0";
        // SAFETY: `site` points at a NUL-terminated byte string valid for the call.
        assert_eq!(unsafe { site_str(site.as_ptr()) }, "a.wk:1:2");
        // SAFETY: a null pointer is an explicitly handled input to `site_str`.
        assert_eq!(unsafe { site_str(std::ptr::null()) }, "");
    }
}

/// P14.5's textual law: every exported pointer-taking C entry checks its contract in debug builds.
///
/// Scans each runtime module's source (`include_str!`, so the scan is of exactly what was compiled)
/// for `#[no_mangle]` items named `wukong_*` whose parameter list takes a raw pointer, and requires
/// the body to call one of the `crate::safety::debug_*` helpers above or to carry a
/// `// SAFETY-EXEMPT: <reason>` comment. `LAW_MODULES` must list every `src/*.rs` file: a new module
/// fails `the_law_scans_every_runtime_module` until it is added, so an entry cannot slip in unseen.
#[cfg(test)]
mod entry_law {
    /// Every runtime module, by file name. Keep in sync with `src/` (enforced below).
    const LAW_MODULES: &[(&str, &str)] = &[
        ("attention.rs", include_str!("attention.rs")),
        ("attention_kv.rs", include_str!("attention_kv.rs")),
        ("bias.rs", include_str!("bias.rs")),
        ("cffi.rs", include_str!("cffi.rs")),
        ("colarg.rs", include_str!("colarg.rs")),
        ("colreduce.rs", include_str!("colreduce.rs")),
        ("cumminmax.rs", include_str!("cumminmax.rs")),
        ("cumprod.rs", include_str!("cumprod.rs")),
        ("cumsum.rs", include_str!("cumsum.rs")),
        ("dequant.rs", include_str!("dequant.rs")),
        ("embedding.rs", include_str!("embedding.rs")),
        ("entropy.rs", include_str!("entropy.rs")),
        ("fileio.rs", include_str!("fileio.rs")),
        ("fmt.rs", include_str!("fmt.rs")),
        ("fp8.rs", include_str!("fp8.rs")),
        ("gemm.rs", include_str!("gemm.rs")),
        ("gemv.rs", include_str!("gemv.rs")),
        ("gemv_half.rs", include_str!("gemv_half.rs")),
        ("gevm.rs", include_str!("gevm.rs")),
        ("gguf.rs", include_str!("gguf.rs")),
        ("i8gemm.rs", include_str!("i8gemm.rs")),
        ("int128.rs", include_str!("int128.rs")),
        ("isa.rs", include_str!("isa.rs")),
        ("kd_loss.rs", include_str!("kd_loss.rs")),
        ("kldiv.rs", include_str!("kldiv.rs")),
        ("layernorm_bwd.rs", include_str!("layernorm_bwd.rs")),
        ("lib.rs", include_str!("lib.rs")),
        ("logsoftmax.rs", include_str!("logsoftmax.rs")),
        ("lowp.rs", include_str!("lowp.rs")),
        ("lrscan.rs", include_str!("lrscan.rs")),
        ("math64.rs", include_str!("math64.rs")),
        ("mmap.rs", include_str!("mmap.rs")),
        ("norm.rs", include_str!("norm.rs")),
        ("npy.rs", include_str!("npy.rs")),
        ("nt_policy.rs", include_str!("nt_policy.rs")),
        ("par.rs", include_str!("par.rs")),
        ("pool2d.rs", include_str!("pool2d.rs")),
        ("process.rs", include_str!("process.rs")),
        ("quant.rs", include_str!("quant.rs")),
        ("random.rs", include_str!("random.rs")),
        ("reduce.rs", include_str!("reduce.rs")),
        ("rmsnorm_bwd.rs", include_str!("rmsnorm_bwd.rs")),
        ("rope.rs", include_str!("rope.rs")),
        ("rope_bwd.rs", include_str!("rope_bwd.rs")),
        ("rowarg.rs", include_str!("rowarg.rs")),
        ("safetensors.rs", include_str!("safetensors.rs")),
        ("safety.rs", include_str!("safety.rs")),
        ("scatter.rs", include_str!("scatter.rs")),
        ("scratch.rs", include_str!("scratch.rs")),
        ("softmax_bwd.rs", include_str!("softmax_bwd.rs")),
        ("special.rs", include_str!("special.rs")),
        ("topk.rs", include_str!("topk.rs")),
        ("topology.rs", include_str!("topology.rs")),
        ("train.rs", include_str!("train.rs")),
        ("transpose.rs", include_str!("transpose.rs")),
        ("utf8.rs", include_str!("utf8.rs")),
        ("velem.rs", include_str!("velem.rs")),
        ("vmath.rs", include_str!("vmath.rs")),
        ("xent.rs", include_str!("xent.rs")),
        ("xent_bwd.rs", include_str!("xent_bwd.rs")),
    ];

    /// The aliasing rule an entry's doc block states in its `**Aliasing...` paragraph.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum Alias {
        /// No `**Aliasing` paragraph: the entry documents no aliasing rule.
        Unstated,
        /// `**Aliasing: none constrained**`: every operand is only read.
        Unconstrained,
        /// "all operands disjoint" or a named disjoint pair: [`super::debug_disjoint`]'s class.
        Disjoint,
        /// "exactly in place, otherwise disjoint" (or a "may be EXACTLY `x`" restatement):
        /// [`super::debug_inplace_or_disjoint`]'s class.
        InPlace,
    }

    /// One scanned `#[no_mangle] ... fn wukong_*` entry.
    struct Entry {
        name: String,
        takes_pointer: bool,
        checked: bool,
        /// `Some(reason)` when the body carries `// SAFETY-EXEMPT: reason`.
        exempt: Option<String>,
        /// The aliasing class the doc block above `#[no_mangle]` documents.
        alias: Alias,
        /// The body calls `safety::debug_disjoint` / `safety::debug_inplace_or_disjoint`.
        checks_disjoint: bool,
        checks_inplace: bool,
    }

    /// Entries whose kernel DETECTS a documented-forbidden overlap and defines the result itself,
    /// each pinned by an overlap test (`mixed_width_overlap_matches_live_byte_reference`, the
    /// `*_narrow_overlap_preserves_ascending_reads_and_writes` pair): asserting the documented class
    /// would abort a call the kernel defines, so these eight check extents only. The list is exact --
    /// a name here that does check its class, or that documents no rule, fails the law.
    const ALIASING_DEFINED_IN_KERNEL: &[&str] = &[
        "wukong_axpby_bf16",
        "wukong_axpby_f16",
        "wukong_axpby_bf16_out",
        "wukong_axpby_f16_out",
        "wukong_vmath_bf16",
        "wukong_vmath_f16",
        "wukong_vmath_bf16_out",
        "wukong_vmath_f16_out",
    ];

    /// Classify the `**Aliasing` paragraph of a doc block (the `///` lines above `#[no_mangle]`):
    /// from the line that opens it to the next blank doc line.
    fn alias_class(doc: &[&str]) -> Alias {
        let Some(start) = doc.iter().position(|l| l.contains("**Aliasing")) else {
            return Alias::Unstated;
        };
        let para: Vec<&str> = doc[start..]
            .iter()
            .take_while(|l| !l.is_empty())
            .copied()
            .collect();
        let para = para.join(" ");
        if para.contains("none constrained") {
            Alias::Unconstrained
        } else if para.contains("exactly in place") || para.contains("EXACTLY") {
            Alias::InPlace
        } else {
            Alias::Disjoint
        }
    }

    /// Scan one module's text. An entry is a `#[no_mangle]` attribute line followed by an item
    /// whose `fn` is named `wukong_*`. Entries are top-level items, so rustfmt closes their body
    /// with a `}` alone at column 0 -- the scan cuts the body there (and fails loudly on an entry
    /// that is not top-level, rather than guess). `#[cfg(test)] mod tests` is not scanned.
    fn scan(module: &str, src: &str) -> Vec<Entry> {
        let src = src.replace("\r\n", "\n");
        let src = match src.find("#[cfg(test)]\nmod tests") {
            Some(i) => &src[..i],
            None => &src[..],
        };
        let lines: Vec<&str> = src.lines().collect();
        let mut out = Vec::new();
        for (i, line) in lines.iter().enumerate() {
            if line.trim() != "#[no_mangle]" {
                continue;
            }
            // The doc block: the `///` lines (and attributes) directly above `#[no_mangle]`.
            let mut doc: Vec<&str> = lines[..i]
                .iter()
                .rev()
                .map(|l| l.trim())
                .take_while(|l| l.starts_with("///") || l.starts_with("#["))
                .filter_map(|l| l.strip_prefix("///"))
                .map(str::trim)
                .collect();
            doc.reverse();
            let alias = alias_class(&doc);
            // The item's `fn` line: the first later non-comment, non-attribute line naming a fn.
            let Some(fi) = (i + 1..lines.len()).find(|&j| {
                let t = lines[j].trim_start();
                t.contains("fn ") && !t.starts_with("//") && !t.starts_with("#[")
            }) else {
                panic!("{module}:{}: `#[no_mangle]` with no fn after it", i + 1);
            };
            let after_fn = &lines[fi][lines[fi].find("fn ").unwrap() + 3..];
            let name: String = after_fn
                .chars()
                .take_while(|c| c.is_ascii_alphanumeric() || *c == '_' || *c == '$')
                .collect();
            if !name.starts_with("wukong_") {
                // A macro-generated entry (`fn $read`) forwards to a scanned `wukong_*` entry.
                continue;
            }
            assert!(
                !lines[fi].starts_with(char::is_whitespace),
                "{module}:{}: `{name}` is not a top-level item; the law cannot scan it",
                fi + 1
            );
            // Signature: from the `fn` line to the line that opens the body.
            let Some(bi) = (fi..lines.len()).find(|&j| lines[j].trim_end().ends_with('{')) else {
                panic!("{module}:{}: `{name}` has no body", fi + 1);
            };
            // The parameter list only -- a returned pointer (`-> *mut u8`) is not an operand.
            let sig = lines[fi..=bi].join("\n");
            let open = sig.find('(').unwrap();
            let mut depth = 0usize;
            let close = sig[open..]
                .char_indices()
                .find_map(|(k, c)| {
                    match c {
                        '(' => depth += 1,
                        ')' => {
                            depth -= 1;
                            if depth == 0 {
                                return Some(open + k);
                            }
                        }
                        _ => {}
                    }
                    None
                })
                .unwrap_or(sig.len());
            let params = &sig[open..close];
            let takes_pointer = params.contains("*const") || params.contains("*mut");
            let Some(ei) = (bi + 1..lines.len()).find(|&j| lines[j] == "}") else {
                panic!(
                    "{module}:{}: `{name}`'s body never closes at column 0",
                    fi + 1
                );
            };
            let body = lines[bi + 1..ei].join("\n");
            let checked = body.contains("safety::debug_");
            let exempt = body.lines().find_map(|l| {
                l.trim()
                    .strip_prefix("// SAFETY-EXEMPT:")
                    .map(|r| r.trim().to_string())
            });
            out.push(Entry {
                name,
                takes_pointer,
                checked,
                exempt,
                alias,
                checks_disjoint: body.contains("safety::debug_disjoint("),
                checks_inplace: body.contains("safety::debug_inplace_or_disjoint("),
            });
        }
        out
    }

    #[test]
    fn every_pointer_entry_checks_its_contract() {
        let mut violations = Vec::new();
        let (mut entries, mut pointer_entries, mut checked) = (0usize, 0usize, 0usize);
        for (module, src) in LAW_MODULES {
            for e in scan(module, src) {
                entries += 1;
                if !e.takes_pointer {
                    continue;
                }
                pointer_entries += 1;
                match (&e.exempt, e.checked) {
                    (Some(r), _) if r.len() < 10 => violations.push(format!(
                        "{module}: `{}` has a SAFETY-EXEMPT with no real reason",
                        e.name
                    )),
                    (Some(_), true) => violations.push(format!(
                        "{module}: `{}` both checks and claims SAFETY-EXEMPT; keep one",
                        e.name
                    )),
                    (Some(_), false) => {}
                    (None, true) => checked += 1,
                    (None, false) => violations.push(format!(
                        "{module}: `{}` takes a raw pointer but calls no `safety::debug_*` check \
                         and has no `// SAFETY-EXEMPT: <reason>`",
                        e.name
                    )),
                }
            }
        }
        assert!(
            violations.is_empty(),
            "P14.5 entry law violated:\n{}",
            violations.join("\n")
        );
        // Floors, so a broken scanner cannot pass vacuously: the crate exports ~180 entries, nearly
        // all of them pointer-taking kernels.
        assert!(entries >= 150, "scanned only {entries} entries");
        assert!(
            pointer_entries >= 140 && checked >= 140,
            "scanned {pointer_entries} pointer entries, {checked} checked"
        );
    }

    /// The aliasing half of the law: an entry whose doc states an aliasing rule asserts that rule,
    /// not just its extents. "Calls some `safety::debug_*`" cannot see an entry that documents
    /// "exactly in place, otherwise disjoint" and checks only extents -- `wukong_vmath_f32` and
    /// `wukong_vmath2_f32` did exactly that while their `_parallel` twin checked the class.
    #[test]
    fn every_documented_aliasing_rule_is_checked() {
        let mut violations = Vec::new();
        let mut seen_allowed = Vec::new();
        let (mut disjoint, mut inplace) = (0usize, 0usize);
        for (module, src) in LAW_MODULES {
            for e in scan(module, src) {
                let allowed = ALIASING_DEFINED_IN_KERNEL.contains(&e.name.as_str());
                let checks = match e.alias {
                    Alias::Unstated | Alias::Unconstrained => {
                        if allowed {
                            violations.push(format!(
                                "{module}: `{}` is in ALIASING_DEFINED_IN_KERNEL but documents no \
                                 aliasing rule",
                                e.name
                            ));
                        }
                        continue;
                    }
                    Alias::Disjoint => {
                        disjoint += 1;
                        e.checks_disjoint
                    }
                    Alias::InPlace => {
                        inplace += 1;
                        e.checks_inplace
                    }
                };
                match (checks, allowed) {
                    (true, false) => {}
                    (false, true) => seen_allowed.push(e.name.clone()),
                    (true, true) => violations.push(format!(
                        "{module}: `{}` checks its aliasing class; drop it from \
                         ALIASING_DEFINED_IN_KERNEL",
                        e.name
                    )),
                    (false, false) => {
                        let (class, helper) = if e.alias == Alias::InPlace {
                            ("exactly-in-place", "safety::debug_inplace_or_disjoint")
                        } else {
                            ("all-disjoint", "safety::debug_disjoint")
                        };
                        violations.push(format!(
                            "{module}: `{}` documents the {class} aliasing rule but never \
                             calls `{helper}`",
                            e.name
                        ))
                    }
                }
            }
        }
        seen_allowed.sort();
        let mut want: Vec<String> = ALIASING_DEFINED_IN_KERNEL
            .iter()
            .map(|n| n.to_string())
            .collect();
        want.sort();
        if seen_allowed != want {
            violations.push(format!(
                "ALIASING_DEFINED_IN_KERNEL lists {want:?}, but the scan matched {seen_allowed:?}"
            ));
        }
        assert!(
            violations.is_empty(),
            "P14.5 aliasing law violated:\n{}",
            violations.join("\n")
        );
        // Floors, so a scanner that stops reading doc blocks cannot pass vacuously.
        assert!(
            disjoint >= 80 && inplace >= 25,
            "classified only {disjoint} disjoint / {inplace} in-place entries"
        );
    }

    #[test]
    fn the_law_scans_every_runtime_module() {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
        let mut on_disk: Vec<String> = std::fs::read_dir(&dir)
            .expect("read src/")
            .filter_map(|e| e.ok())
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .filter(|n| n.ends_with(".rs"))
            .collect();
        on_disk.sort();
        let mut listed: Vec<String> = LAW_MODULES.iter().map(|(n, _)| n.to_string()).collect();
        listed.sort();
        assert_eq!(
            listed, on_disk,
            "LAW_MODULES must list every src/*.rs runtime module (add the new one with include_str!)"
        );
    }

    #[test]
    fn the_scanner_flags_an_unchecked_pointer_entry() {
        let src = [
            "#[no_mangle]",
            "pub unsafe extern \"C\" fn wukong_demo(x: *const f32, n: i64) {",
            "    let _ = (x, n);",
            "}",
            "",
            "#[no_mangle]",
            "pub unsafe extern \"C\" fn wukong_ok(",
            "    x: *const f32,",
            "    n: i64,",
            ") {",
            "    crate::safety::debug_extent(x, n);",
            "}",
            "",
            "#[no_mangle]",
            "pub extern \"C\" fn wukong_scalar(x: f32) -> f32 {",
            "    x",
            "}",
            "",
            "#[no_mangle]",
            "pub extern \"C\" fn wukong_alloc(n: i64) -> *mut u8 {",
            "    std::ptr::null_mut()",
            "}",
            "",
            "#[no_mangle]",
            "pub unsafe extern \"C\" fn wukong_ex(s: *const u8) {",
            "    // SAFETY-EXEMPT: a NUL-terminated string has no extent.",
            "    let _ = s;",
            "}",
            "",
            "/// # Safety",
            "/// **Aliasing: exactly in place, otherwise disjoint** -- `out` may be EXACTLY `x`.",
            "#[no_mangle]",
            "pub unsafe extern \"C\" fn wukong_inplace_unchecked(x: *const f32, out: *mut f32) {",
            "    crate::safety::debug_extent(x, 1);",
            "}",
            "",
            "/// **Aliasing:** that of [`wukong_inplace_unchecked`] -- `out` may be EXACTLY `x`.",
            "#[no_mangle]",
            "#[allow(clippy::too_many_arguments)]",
            "pub unsafe extern \"C\" fn wukong_inplace_ok(x: *const f32, out: *mut f32) {",
            "    crate::safety::debug_inplace_or_disjoint(out, 1, x, 1);",
            "}",
            "",
            "/// **Aliasing: all operands disjoint** -- `out` does not overlap `x`.",
            "///",
            "/// A later paragraph that says EXACTLY is not part of the rule.",
            "#[no_mangle]",
            "pub unsafe extern \"C\" fn wukong_disjoint_ok(x: *const f32, out: *mut f32) {",
            "    crate::safety::debug_disjoint(out, 1, x, 1);",
            "}",
            "",
            "/// **Aliasing: none constrained** -- every operand is only read.",
            "#[no_mangle]",
            "pub unsafe extern \"C\" fn wukong_read_only(x: *const f32, y: *const f32) {",
            "    crate::safety::debug_extent(x, 1);",
            "}",
        ]
        .join("\n");
        let got: Vec<(String, bool, bool, bool)> = scan("demo.rs", &src)
            .iter()
            .map(|e| {
                (
                    e.name.clone(),
                    e.takes_pointer,
                    e.checked,
                    e.exempt.is_some(),
                )
            })
            .collect();
        let want: Vec<(String, bool, bool, bool)> = [
            ("wukong_demo", true, false, false),
            ("wukong_ok", true, true, false),
            ("wukong_scalar", false, false, false),
            // A returned pointer is not an operand.
            ("wukong_alloc", false, false, false),
            ("wukong_ex", true, false, true),
            ("wukong_inplace_unchecked", true, true, false),
            ("wukong_inplace_ok", true, true, false),
            ("wukong_disjoint_ok", true, true, false),
            ("wukong_read_only", true, true, false),
        ]
        .into_iter()
        .map(|(n, a, b, c)| (n.to_string(), a, b, c))
        .collect();
        assert_eq!(got, want);
        // The aliasing class comes from the `**Aliasing` paragraph only, and the in-place entry that
        // checks only extents is told apart from the one that checks its class.
        let classes: Vec<(String, Alias, bool, bool)> = scan("demo.rs", &src)
            .into_iter()
            .map(|e| (e.name, e.alias, e.checks_disjoint, e.checks_inplace))
            .filter(|(n, ..)| !["wukong_ok", "wukong_scalar", "wukong_alloc"].contains(&n.as_str()))
            .collect();
        let want: Vec<(String, Alias, bool, bool)> = [
            ("wukong_demo", Alias::Unstated, false, false),
            ("wukong_ex", Alias::Unstated, false, false),
            ("wukong_inplace_unchecked", Alias::InPlace, false, false),
            ("wukong_inplace_ok", Alias::InPlace, false, true),
            ("wukong_disjoint_ok", Alias::Disjoint, true, false),
            ("wukong_read_only", Alias::Unconstrained, false, false),
        ]
        .into_iter()
        .map(|(n, a, d, i)| (n.to_string(), a, d, i))
        .collect();
        assert_eq!(classes, want);
    }
}
