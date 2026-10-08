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

#[cfg(test)]
mod tests {
    use super::*;

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
