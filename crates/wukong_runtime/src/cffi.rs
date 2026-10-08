//! Calling C from Wukong programs: the one symbol resolver every backend shares, and the
//! interpreter's scalar call marshalling.
//!
//! An `extern "C"` function a `.wk` program declares is an ordinary C symbol. The native backend
//! imports it — the JIT binds it to the address [`resolve_c_symbol`] returns, and an `--emit=exe`
//! object leaves it to the linker. The interpreter calls the SAME address through
//! [`call_scalar`], so a `--run` on either backend runs the identical machine code and the
//! interpreter-vs-native differential stays bit-exact even for a libm function whose last bit
//! differs between C runtimes.
//!
//! Resolution order: on Windows the Universal CRT (`ucrtbase.dll`, present on every supported
//! Windows and loaded on demand so all of C99's libm is reachable), then `msvcrt.dll` (the CRT a
//! `windows-gnu` binary links), then the executable itself; on Unix `dlsym(RTLD_DEFAULT, ..)` — the
//! process's C library, libm and anything else already loaded.
//!
//! The interpreter's slot-indexed memory is not machine memory, so only *scalar* signatures cross:
//! up to [`MAX_SCALAR_ARGS`] parameters, each an integer (passed as a 64-bit value the caller has
//! already sign- or zero-extended from its C type — correct for both the System V and the Win64
//! convention, where the callee reads the low bits it declared), an `f32` or an `f64`; and a `void`,
//! integer, `f32` or `f64` result. A pointer parameter or result is native-only.

/// The most parameters [`call_scalar`] marshals.
pub const MAX_SCALAR_ARGS: usize = 4;

/// One scalar argument, classified by the register class the C ABI passes it in.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum CArg {
    /// Any C integer type (and `bool`), already extended to 64 bits per its signedness.
    Int(i64),
    F32(f32),
    F64(f64),
}

/// The class of a C function's result.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CRet {
    Void,
    /// A C integer no wider than 32 bits (`int`, `char`, `short`, `bool`): read from the low 32
    /// bits of the return register; the caller truncates it to the declared width.
    Int32,
    /// A 64-bit C integer (`long long`, `int64_t`, `size_t` on 64-bit targets).
    Int64,
    F32,
    F64,
}

/// A C function's result.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum CValue {
    Void,
    Int(i64),
    F32(f32),
    F64(f64),
}

/// The address of the C function `name` in the running process, or `None` when no loaded C
/// library defines it. See the module docs for the search order.
pub fn resolve_c_symbol(name: &str) -> Option<*const u8> {
    let c_name = std::ffi::CString::new(name).ok()?;
    imp::lookup(&c_name)
}

#[cfg(windows)]
mod imp {
    use std::ffi::{c_char, c_void, CStr};

    #[link(name = "kernel32")]
    extern "system" {
        fn GetModuleHandleA(name: *const c_char) -> *mut c_void;
        fn LoadLibraryA(name: *const c_char) -> *mut c_void;
        fn GetProcAddress(module: *mut c_void, name: *const c_char) -> *mut c_void;
    }

    pub(super) fn lookup(name: &CStr) -> Option<*const u8> {
        for lib in [c"ucrtbase.dll", c"msvcrt.dll"] {
            // SAFETY: both arguments are valid NUL-terminated strings; a null handle is skipped.
            let module = unsafe {
                let loaded = GetModuleHandleA(lib.as_ptr());
                if loaded.is_null() {
                    LoadLibraryA(lib.as_ptr())
                } else {
                    loaded
                }
            };
            if module.is_null() {
                continue;
            }
            // SAFETY: `module` is a live module handle and `name` is NUL-terminated.
            let addr = unsafe { GetProcAddress(module, name.as_ptr()) };
            if !addr.is_null() {
                return Some(addr as *const u8);
            }
        }
        // SAFETY: a null name asks for the executable's own handle, which is always valid.
        let exe = unsafe { GetModuleHandleA(std::ptr::null()) };
        // SAFETY: as above.
        let addr = unsafe { GetProcAddress(exe, name.as_ptr()) };
        (!addr.is_null()).then_some(addr as *const u8)
    }
}

#[cfg(unix)]
mod imp {
    use std::ffi::{c_char, c_void, CStr};

    extern "C" {
        fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
    }

    /// `RTLD_DEFAULT`: search every object loaded into the process.
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    fn rtld_default() -> *mut c_void {
        -2isize as *mut c_void
    }
    #[cfg(not(any(target_os = "macos", target_os = "ios")))]
    fn rtld_default() -> *mut c_void {
        std::ptr::null_mut()
    }

    pub(super) fn lookup(name: &CStr) -> Option<*const u8> {
        // SAFETY: `RTLD_DEFAULT` is a valid pseudo-handle and `name` is NUL-terminated.
        let addr = unsafe { dlsym(rtld_default(), name.as_ptr()) };
        (!addr.is_null()).then_some(addr as *const u8)
    }
}

#[cfg(not(any(windows, unix)))]
mod imp {
    use std::ffi::CStr;

    pub(super) fn lookup(_name: &CStr) -> Option<*const u8> {
        None
    }
}

/// Call the C function at `f` with `args`, reading a result of class `ret`. `None` when the
/// signature shape is outside what this marshaller supports (more than [`MAX_SCALAR_ARGS`]
/// parameters) — the caller reports that as a diagnostic instead of guessing.
///
/// # Safety
///
/// `f` must be the address of a C function (`extern "C"` ABI) whose parameter list is exactly the
/// classes of `args`, in order, and whose result is of class `ret`. Calling anything else is
/// undefined behavior, exactly as a mismatched `extern` declaration is in C.
pub unsafe fn call_scalar(f: *const u8, args: &[CArg], ret: CRet) -> Option<CValue> {
    macro_rules! call {
        ($($t:ty),* ; $($a:expr),*) => {
            match ret {
                CRet::Void => {
                    let g = std::mem::transmute::<*const u8, extern "C" fn($($t),*)>(f);
                    g($($a),*);
                    CValue::Void
                }
                CRet::Int32 => {
                    let g = std::mem::transmute::<*const u8, extern "C" fn($($t),*) -> i32>(f);
                    CValue::Int(i64::from(g($($a),*)))
                }
                CRet::Int64 => {
                    let g = std::mem::transmute::<*const u8, extern "C" fn($($t),*) -> i64>(f);
                    CValue::Int(g($($a),*))
                }
                CRet::F32 => {
                    let g = std::mem::transmute::<*const u8, extern "C" fn($($t),*) -> f32>(f);
                    CValue::F32(g($($a),*))
                }
                CRet::F64 => {
                    let g = std::mem::transmute::<*const u8, extern "C" fn($($t),*) -> f64>(f);
                    CValue::F64(g($($a),*))
                }
            }
        };
    }
    use CArg::{Int as I, F32 as S, F64 as D};
    Some(match args {
        [] => call!(; ),
        [I(a)] => call!(i64; *a),
        [S(a)] => call!(f32; *a),
        [D(a)] => call!(f64; *a),
        [I(a), I(b)] => call!(i64, i64; *a, *b),
        [I(a), S(b)] => call!(i64, f32; *a, *b),
        [I(a), D(b)] => call!(i64, f64; *a, *b),
        [S(a), I(b)] => call!(f32, i64; *a, *b),
        [S(a), S(b)] => call!(f32, f32; *a, *b),
        [S(a), D(b)] => call!(f32, f64; *a, *b),
        [D(a), I(b)] => call!(f64, i64; *a, *b),
        [D(a), S(b)] => call!(f64, f32; *a, *b),
        [D(a), D(b)] => call!(f64, f64; *a, *b),
        [I(a), I(b), I(c)] => call!(i64, i64, i64; *a, *b, *c),
        [I(a), I(b), S(c)] => call!(i64, i64, f32; *a, *b, *c),
        [I(a), I(b), D(c)] => call!(i64, i64, f64; *a, *b, *c),
        [I(a), S(b), I(c)] => call!(i64, f32, i64; *a, *b, *c),
        [I(a), S(b), S(c)] => call!(i64, f32, f32; *a, *b, *c),
        [I(a), S(b), D(c)] => call!(i64, f32, f64; *a, *b, *c),
        [I(a), D(b), I(c)] => call!(i64, f64, i64; *a, *b, *c),
        [I(a), D(b), S(c)] => call!(i64, f64, f32; *a, *b, *c),
        [I(a), D(b), D(c)] => call!(i64, f64, f64; *a, *b, *c),
        [S(a), I(b), I(c)] => call!(f32, i64, i64; *a, *b, *c),
        [S(a), I(b), S(c)] => call!(f32, i64, f32; *a, *b, *c),
        [S(a), I(b), D(c)] => call!(f32, i64, f64; *a, *b, *c),
        [S(a), S(b), I(c)] => call!(f32, f32, i64; *a, *b, *c),
        [S(a), S(b), S(c)] => call!(f32, f32, f32; *a, *b, *c),
        [S(a), S(b), D(c)] => call!(f32, f32, f64; *a, *b, *c),
        [S(a), D(b), I(c)] => call!(f32, f64, i64; *a, *b, *c),
        [S(a), D(b), S(c)] => call!(f32, f64, f32; *a, *b, *c),
        [S(a), D(b), D(c)] => call!(f32, f64, f64; *a, *b, *c),
        [D(a), I(b), I(c)] => call!(f64, i64, i64; *a, *b, *c),
        [D(a), I(b), S(c)] => call!(f64, i64, f32; *a, *b, *c),
        [D(a), I(b), D(c)] => call!(f64, i64, f64; *a, *b, *c),
        [D(a), S(b), I(c)] => call!(f64, f32, i64; *a, *b, *c),
        [D(a), S(b), S(c)] => call!(f64, f32, f32; *a, *b, *c),
        [D(a), S(b), D(c)] => call!(f64, f32, f64; *a, *b, *c),
        [D(a), D(b), I(c)] => call!(f64, f64, i64; *a, *b, *c),
        [D(a), D(b), S(c)] => call!(f64, f64, f32; *a, *b, *c),
        [D(a), D(b), D(c)] => call!(f64, f64, f64; *a, *b, *c),
        [I(a), I(b), I(c), I(d)] => call!(i64, i64, i64, i64; *a, *b, *c, *d),
        [I(a), I(b), I(c), S(d)] => call!(i64, i64, i64, f32; *a, *b, *c, *d),
        [I(a), I(b), I(c), D(d)] => call!(i64, i64, i64, f64; *a, *b, *c, *d),
        [I(a), I(b), S(c), I(d)] => call!(i64, i64, f32, i64; *a, *b, *c, *d),
        [I(a), I(b), S(c), S(d)] => call!(i64, i64, f32, f32; *a, *b, *c, *d),
        [I(a), I(b), S(c), D(d)] => call!(i64, i64, f32, f64; *a, *b, *c, *d),
        [I(a), I(b), D(c), I(d)] => call!(i64, i64, f64, i64; *a, *b, *c, *d),
        [I(a), I(b), D(c), S(d)] => call!(i64, i64, f64, f32; *a, *b, *c, *d),
        [I(a), I(b), D(c), D(d)] => call!(i64, i64, f64, f64; *a, *b, *c, *d),
        [I(a), S(b), I(c), I(d)] => call!(i64, f32, i64, i64; *a, *b, *c, *d),
        [I(a), S(b), I(c), S(d)] => call!(i64, f32, i64, f32; *a, *b, *c, *d),
        [I(a), S(b), I(c), D(d)] => call!(i64, f32, i64, f64; *a, *b, *c, *d),
        [I(a), S(b), S(c), I(d)] => call!(i64, f32, f32, i64; *a, *b, *c, *d),
        [I(a), S(b), S(c), S(d)] => call!(i64, f32, f32, f32; *a, *b, *c, *d),
        [I(a), S(b), S(c), D(d)] => call!(i64, f32, f32, f64; *a, *b, *c, *d),
        [I(a), S(b), D(c), I(d)] => call!(i64, f32, f64, i64; *a, *b, *c, *d),
        [I(a), S(b), D(c), S(d)] => call!(i64, f32, f64, f32; *a, *b, *c, *d),
        [I(a), S(b), D(c), D(d)] => call!(i64, f32, f64, f64; *a, *b, *c, *d),
        [I(a), D(b), I(c), I(d)] => call!(i64, f64, i64, i64; *a, *b, *c, *d),
        [I(a), D(b), I(c), S(d)] => call!(i64, f64, i64, f32; *a, *b, *c, *d),
        [I(a), D(b), I(c), D(d)] => call!(i64, f64, i64, f64; *a, *b, *c, *d),
        [I(a), D(b), S(c), I(d)] => call!(i64, f64, f32, i64; *a, *b, *c, *d),
        [I(a), D(b), S(c), S(d)] => call!(i64, f64, f32, f32; *a, *b, *c, *d),
        [I(a), D(b), S(c), D(d)] => call!(i64, f64, f32, f64; *a, *b, *c, *d),
        [I(a), D(b), D(c), I(d)] => call!(i64, f64, f64, i64; *a, *b, *c, *d),
        [I(a), D(b), D(c), S(d)] => call!(i64, f64, f64, f32; *a, *b, *c, *d),
        [I(a), D(b), D(c), D(d)] => call!(i64, f64, f64, f64; *a, *b, *c, *d),
        [S(a), I(b), I(c), I(d)] => call!(f32, i64, i64, i64; *a, *b, *c, *d),
        [S(a), I(b), I(c), S(d)] => call!(f32, i64, i64, f32; *a, *b, *c, *d),
        [S(a), I(b), I(c), D(d)] => call!(f32, i64, i64, f64; *a, *b, *c, *d),
        [S(a), I(b), S(c), I(d)] => call!(f32, i64, f32, i64; *a, *b, *c, *d),
        [S(a), I(b), S(c), S(d)] => call!(f32, i64, f32, f32; *a, *b, *c, *d),
        [S(a), I(b), S(c), D(d)] => call!(f32, i64, f32, f64; *a, *b, *c, *d),
        [S(a), I(b), D(c), I(d)] => call!(f32, i64, f64, i64; *a, *b, *c, *d),
        [S(a), I(b), D(c), S(d)] => call!(f32, i64, f64, f32; *a, *b, *c, *d),
        [S(a), I(b), D(c), D(d)] => call!(f32, i64, f64, f64; *a, *b, *c, *d),
        [S(a), S(b), I(c), I(d)] => call!(f32, f32, i64, i64; *a, *b, *c, *d),
        [S(a), S(b), I(c), S(d)] => call!(f32, f32, i64, f32; *a, *b, *c, *d),
        [S(a), S(b), I(c), D(d)] => call!(f32, f32, i64, f64; *a, *b, *c, *d),
        [S(a), S(b), S(c), I(d)] => call!(f32, f32, f32, i64; *a, *b, *c, *d),
        [S(a), S(b), S(c), S(d)] => call!(f32, f32, f32, f32; *a, *b, *c, *d),
        [S(a), S(b), S(c), D(d)] => call!(f32, f32, f32, f64; *a, *b, *c, *d),
        [S(a), S(b), D(c), I(d)] => call!(f32, f32, f64, i64; *a, *b, *c, *d),
        [S(a), S(b), D(c), S(d)] => call!(f32, f32, f64, f32; *a, *b, *c, *d),
        [S(a), S(b), D(c), D(d)] => call!(f32, f32, f64, f64; *a, *b, *c, *d),
        [S(a), D(b), I(c), I(d)] => call!(f32, f64, i64, i64; *a, *b, *c, *d),
        [S(a), D(b), I(c), S(d)] => call!(f32, f64, i64, f32; *a, *b, *c, *d),
        [S(a), D(b), I(c), D(d)] => call!(f32, f64, i64, f64; *a, *b, *c, *d),
        [S(a), D(b), S(c), I(d)] => call!(f32, f64, f32, i64; *a, *b, *c, *d),
        [S(a), D(b), S(c), S(d)] => call!(f32, f64, f32, f32; *a, *b, *c, *d),
        [S(a), D(b), S(c), D(d)] => call!(f32, f64, f32, f64; *a, *b, *c, *d),
        [S(a), D(b), D(c), I(d)] => call!(f32, f64, f64, i64; *a, *b, *c, *d),
        [S(a), D(b), D(c), S(d)] => call!(f32, f64, f64, f32; *a, *b, *c, *d),
        [S(a), D(b), D(c), D(d)] => call!(f32, f64, f64, f64; *a, *b, *c, *d),
        [D(a), I(b), I(c), I(d)] => call!(f64, i64, i64, i64; *a, *b, *c, *d),
        [D(a), I(b), I(c), S(d)] => call!(f64, i64, i64, f32; *a, *b, *c, *d),
        [D(a), I(b), I(c), D(d)] => call!(f64, i64, i64, f64; *a, *b, *c, *d),
        [D(a), I(b), S(c), I(d)] => call!(f64, i64, f32, i64; *a, *b, *c, *d),
        [D(a), I(b), S(c), S(d)] => call!(f64, i64, f32, f32; *a, *b, *c, *d),
        [D(a), I(b), S(c), D(d)] => call!(f64, i64, f32, f64; *a, *b, *c, *d),
        [D(a), I(b), D(c), I(d)] => call!(f64, i64, f64, i64; *a, *b, *c, *d),
        [D(a), I(b), D(c), S(d)] => call!(f64, i64, f64, f32; *a, *b, *c, *d),
        [D(a), I(b), D(c), D(d)] => call!(f64, i64, f64, f64; *a, *b, *c, *d),
        [D(a), S(b), I(c), I(d)] => call!(f64, f32, i64, i64; *a, *b, *c, *d),
        [D(a), S(b), I(c), S(d)] => call!(f64, f32, i64, f32; *a, *b, *c, *d),
        [D(a), S(b), I(c), D(d)] => call!(f64, f32, i64, f64; *a, *b, *c, *d),
        [D(a), S(b), S(c), I(d)] => call!(f64, f32, f32, i64; *a, *b, *c, *d),
        [D(a), S(b), S(c), S(d)] => call!(f64, f32, f32, f32; *a, *b, *c, *d),
        [D(a), S(b), S(c), D(d)] => call!(f64, f32, f32, f64; *a, *b, *c, *d),
        [D(a), S(b), D(c), I(d)] => call!(f64, f32, f64, i64; *a, *b, *c, *d),
        [D(a), S(b), D(c), S(d)] => call!(f64, f32, f64, f32; *a, *b, *c, *d),
        [D(a), S(b), D(c), D(d)] => call!(f64, f32, f64, f64; *a, *b, *c, *d),
        [D(a), D(b), I(c), I(d)] => call!(f64, f64, i64, i64; *a, *b, *c, *d),
        [D(a), D(b), I(c), S(d)] => call!(f64, f64, i64, f32; *a, *b, *c, *d),
        [D(a), D(b), I(c), D(d)] => call!(f64, f64, i64, f64; *a, *b, *c, *d),
        [D(a), D(b), S(c), I(d)] => call!(f64, f64, f32, i64; *a, *b, *c, *d),
        [D(a), D(b), S(c), S(d)] => call!(f64, f64, f32, f32; *a, *b, *c, *d),
        [D(a), D(b), S(c), D(d)] => call!(f64, f64, f32, f64; *a, *b, *c, *d),
        [D(a), D(b), D(c), I(d)] => call!(f64, f64, f64, i64; *a, *b, *c, *d),
        [D(a), D(b), D(c), S(d)] => call!(f64, f64, f64, f32; *a, *b, *c, *d),
        [D(a), D(b), D(c), D(d)] => call!(f64, f64, f64, f64; *a, *b, *c, *d),
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The resolver finds a libm function in every supported environment, and the marshaller calls
    /// it through each register class with a hand-checkable, exactly representable result.
    #[test]
    fn resolves_and_calls_libm_through_every_register_class() {
        let ldexp = resolve_c_symbol("ldexp").expect("ldexp is in every C runtime");
        // SAFETY: C's `double ldexp(double, int)`; the int is passed extended to 64 bits.
        let r = unsafe { call_scalar(ldexp, &[CArg::F64(3.0), CArg::Int(4)], CRet::F64) };
        assert_eq!(r, Some(CValue::F64(48.0)));

        let fmod = resolve_c_symbol("fmod").expect("fmod is in every C runtime");
        // SAFETY: C's `double fmod(double, double)`.
        let r = unsafe { call_scalar(fmod, &[CArg::F64(17.5), CArg::F64(5.0)], CRet::F64) };
        assert_eq!(r, Some(CValue::F64(2.5)));

        let toupper = resolve_c_symbol("toupper").expect("toupper is in every C runtime");
        // SAFETY: C's `int toupper(int)`.
        let r = unsafe { call_scalar(toupper, &[CArg::Int(97)], CRet::Int32) };
        assert_eq!(r, Some(CValue::Int(65)));

        let llabs = resolve_c_symbol("llabs").expect("llabs is in every C99 runtime");
        // SAFETY: C's `long long llabs(long long)`.
        let r = unsafe { call_scalar(llabs, &[CArg::Int(-9_000_000_000)], CRet::Int64) };
        assert_eq!(r, Some(CValue::Int(9_000_000_000)));

        let fmodf = resolve_c_symbol("fmodf").expect("fmodf is in every C99 runtime");
        // SAFETY: C's `float fmodf(float, float)`.
        let r = unsafe { call_scalar(fmodf, &[CArg::F32(7.5), CArg::F32(2.0)], CRet::F32) };
        assert_eq!(r, Some(CValue::F32(1.5)));
    }

    #[test]
    fn unknown_symbols_and_oversized_signatures_are_refused() {
        assert!(resolve_c_symbol("wukong_no_such_c_symbol_anywhere").is_none());
        assert!(resolve_c_symbol("bad\0name").is_none());
        let ldexp = resolve_c_symbol("ldexp").expect("ldexp");
        let five = [CArg::Int(0); MAX_SCALAR_ARGS + 1];
        // SAFETY: rejected before any call is made.
        assert_eq!(unsafe { call_scalar(ldexp, &five, CRet::Void) }, None);
    }
}
