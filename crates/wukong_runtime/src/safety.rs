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
    // SAFETY: `site` is a `.rodata` string the compiler emitted with a NUL terminator.
    let site = unsafe { site_str(site) };
    let _ = std::io::stdout().flush();
    eprintln!("error: {}", bounds_failure_message(index, len, &site));
    std::process::exit(1);
}

// ---------------------------------------------------------------------------------------------
// P14.5: debug-only precondition checks for the C-ABI kernel entries.
//
// These compile OUT of the release perf builds (`debug_assert!`) and are a no-op for a zero or
// negative extent, which every entry treats as a legal no-op -- so they never fail on an input a
// valid caller is allowed to pass. They cannot check a *memory bound*: a C-ABI entry carries no
// capacity, only the extent, so the only facts expressible from `(pointer, extent)` are that a
// nonempty buffer pointer is non-null and that the extent does not overflow the `p.add(n)` the
// kernel performs, plus -- for entries whose `# Safety` contract documents operands as disjoint --
// that two ranges the caller promised not to overlap in fact do not. An entry that permits exact
// in-place aliasing (`out == x`) must NOT pass that pair to `debug_disjoint`.
// ---------------------------------------------------------------------------------------------

/// Assert a single buffer's extent is castable and, when nonempty, its pointer is non-null and the
/// `n`-element span does not overrun `isize::MAX` (the precondition of `p.add(n)`).
#[inline(always)]
pub(crate) fn debug_extent<T>(p: *const T, n: i64) {
    debug_assert!(
        n <= 0
            || (!p.is_null()
                && (n as usize) <= isize::MAX as usize / core::mem::size_of::<T>().max(1)),
        "kernel extent is null/negative-cast or overflows isize::MAX for the element type"
    );
}

/// Two-dimensional form of [`debug_extent`]: the span is `rows*cols` elements. A `rows*cols` that
/// overflows `usize` is itself a contract violation and trips the assert.
#[inline(always)]
pub(crate) fn debug_extent_2d<T>(p: *const T, rows: i64, cols: i64) {
    if rows <= 0 || cols <= 0 {
        return;
    }
    match (rows as usize).checked_mul(cols as usize) {
        Some(n) if n <= i64::MAX as usize => debug_extent(p, n as i64),
        _ => debug_assert!(false, "kernel rows*cols overflows the addressable range"),
    }
}

/// Assert that two ranges a caller's contract documents as DISJOINT do not overlap. Only valid for
/// entries that forbid overlap between the two given operands; never call it on a pair an entry
/// permits to alias in place.
#[inline(always)]
pub(crate) fn debug_disjoint<T, U>(a: *const T, an: i64, b: *const U, bn: i64) {
    if an <= 0 || bn <= 0 {
        return;
    }
    let a0 = a as usize;
    let b0 = b as usize;
    let a1 = a0.saturating_add((an as usize).saturating_mul(core::mem::size_of::<T>()));
    let b1 = b0.saturating_add((bn as usize).saturating_mul(core::mem::size_of::<U>()));
    debug_assert!(
        a1 <= b0 || b1 <= a0,
        "kernel operands documented disjoint but their byte ranges overlap"
    );
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
        // Overlap is rejected (only observable in a debug build; release compiles it out).
        if cfg!(debug_assertions) {
            let overlap =
                std::panic::catch_unwind(|| debug_disjoint(buf.as_ptr(), 8, buf.as_ptr(), 8));
            assert!(
                overlap.is_err(),
                "overlapping documented-disjoint ranges must trip the assert"
            );
        }
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
        let site = b"a.wk:1:2\0";
        assert_eq!(unsafe { site_str(site.as_ptr()) }, "a.wk:1:2");
        assert_eq!(unsafe { site_str(std::ptr::null()) }, "");
    }
}
