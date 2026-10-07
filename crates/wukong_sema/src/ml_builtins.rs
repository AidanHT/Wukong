//! Typing for the ML I/O builtins: the typed file-I/O family over every element type (whole-file
//! and byte-offset reads, writes, `file_size`) and the safetensors / `.npy` container readers.
//!
//! Like the heap and timing builtins, these are typed nominally here — the call's result is a
//! concrete type, so `let n = read_bf16_at(p, off, w)` composes in arithmetic — and every misuse is
//! rejected with the standard codes before `wukong_mir_build` lowers the call structurally:
//! a wrong arity is `E0503`, a wrong operand kind or element type `E0401`, and a builtin that writes
//! a non-`mut` aggregate parameter `E0304` (exactly what `p[i] = …` through that parameter is).
//! [`ml_builtin`] is the one name table; `mir_build` matches the same enum, so the two cannot
//! disagree about which names are builtins or what element type each one moves.
//!
//! **Buffers.** A buffer operand is a `[]T` slice, a fixed `[T; N]` array, or a row-major
//! (`Contiguous`) `Tensor[T, …]` view whose every dimension is known at the call — a constant or a
//! bound dimension variable, never `?`. The element type must be exactly the builtin's `T`: a
//! `[]i32` handed to `read_f32` would read the wrong stride and decode the wrong type. A container
//! read into a *tensor* also checks the file's shape against the tensor's dims at load time.
//!
//! **Integers.** A byte offset may be any integer type (the value's bits, as `x as u64` would give
//! them); an unsuffixed literal adapts to `i64`.
//!
//! A user-defined function of the same name shadows every builtin here (the resolved-callee path in
//! `type_call` runs first).

use wukong_ast::{Expr, ExprKind};
use wukong_span::Span;
use wukong_types::{Dim, Layout, Scalar, Ty};

use crate::Sema;

/// One ML builtin. Each typed file-I/O variant carries its element type.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum MlBuiltin {
    /// `read_<T>(path: *u8, buf) -> i64` — the whole file from byte 0.
    Read(Scalar),
    /// `write_<T>(path: *u8, buf) -> i64`.
    Write(Scalar),
    /// `read_<T>_at(path: *u8, byte_off: int, buf) -> i64` — from a byte offset.
    ReadAt(Scalar),
    /// `file_size(path: *u8) -> i64`.
    FileSize,
    /// `st_read_<T>(path: *u8, name: *u8, buf) -> i64` — one tensor of a safetensors file.
    StRead(Scalar),
    /// `npy_read_<T>(path: *u8, buf) -> i64` — the array of a NumPy `.npy` file.
    NpyRead(Scalar),
}

/// The element type a file-I/O name suffix denotes (`f32`, `bf16`, `u64`, …), or `None`. The twelve
/// fixed-width numeric scalars — every type an `alloc_<T>` buffer can hold — and not `bool`, `char`
/// or the pointer-width `usize`/`isize`, whose on-disk width would depend on the target.
pub fn io_dtype(suffix: &str) -> Option<Scalar> {
    Some(match suffix {
        "f32" => Scalar::F32,
        "f64" => Scalar::F64,
        "f16" => Scalar::F16,
        "bf16" => Scalar::Bf16,
        "i8" => Scalar::I8,
        "i16" => Scalar::I16,
        "i32" => Scalar::I32,
        "i64" => Scalar::I64,
        "u8" => Scalar::U8,
        "u16" => Scalar::U16,
        "u32" => Scalar::U32,
        "u64" => Scalar::U64,
        _ => return None,
    })
}

/// The runtime dtype code of an I/O element type — the compiler's copy of
/// `wukong_runtime::fileio::DT_*` (this crate does not depend on the runtime; a test in
/// `wukong_interp`, which sees both, pins every row). `-1` for a type [`io_dtype`] never yields.
pub fn io_dtype_code(s: Scalar) -> i64 {
    match s {
        Scalar::F32 => 0,
        Scalar::F64 => 1,
        Scalar::F16 => 2,
        Scalar::Bf16 => 3,
        Scalar::I8 => 4,
        Scalar::I16 => 5,
        Scalar::I32 => 6,
        Scalar::I64 => 7,
        Scalar::U8 => 8,
        Scalar::U16 => 9,
        Scalar::U32 => 10,
        Scalar::U64 => 11,
        _ => -1,
    }
}

/// The builtin `name` denotes, or `None`.
pub fn ml_builtin(name: &str) -> Option<MlBuiltin> {
    if let Some(rest) = name.strip_prefix("read_") {
        if let Some(elem) = rest.strip_suffix("_at") {
            return io_dtype(elem).map(MlBuiltin::ReadAt);
        }
        return io_dtype(rest).map(MlBuiltin::Read);
    }
    if let Some(rest) = name.strip_prefix("write_") {
        return io_dtype(rest).map(MlBuiltin::Write);
    }
    if let Some(rest) = name.strip_prefix("st_read_") {
        return io_dtype(rest).map(MlBuiltin::StRead);
    }
    if let Some(rest) = name.strip_prefix("npy_read_") {
        return io_dtype(rest).map(MlBuiltin::NpyRead);
    }
    Some(match name {
        "file_size" => MlBuiltin::FileSize,
        _ => return None,
    })
}

/// The spelling of a buffer of `elem` in a diagnostic.
fn buffer_desc(elem: Scalar) -> String {
    let t = elem.name();
    format!("`[]{t}`, `[{t}; N]` or contiguous `Tensor[{t}, …]`")
}

/// One argument position of a builtin's signature.
#[derive(Clone, Copy)]
enum Arg {
    /// A NUL-terminated `*u8` string (a path or a tensor name).
    Str(&'static str),
    /// An integer (`i64` ABI).
    Int(&'static str),
    /// A buffer of this element type; `true` when the builtin writes it.
    Buf(Scalar, bool),
}

/// The ordinal word for argument `i` in a diagnostic.
fn ordinal(i: usize) -> &'static str {
    ["first", "second", "third", "fourth"]
        .get(i)
        .copied()
        .unwrap_or("next")
}

impl Sema<'_> {
    /// Type a call to a builtin of this family, or return `None` when `name` is not one (the caller
    /// keeps looking). Diagnoses arity and operand errors itself; the returned type is the
    /// builtin's result type either way, so one bad call does not cascade.
    pub(crate) fn type_ml_builtin(
        &mut self,
        name: &str,
        args: &[Expr],
        arg_tys: &mut [Ty],
        span: Span,
    ) -> Option<Ty> {
        use Arg::{Buf, Int, Str};
        let b = ml_builtin(name)?;
        let i64_ty = Ty::Scalar(Scalar::I64);
        let (sig, ret): (Vec<Arg>, Ty) = match b {
            MlBuiltin::Read(t) | MlBuiltin::NpyRead(t) => (vec![Str("path"), Buf(t, true)], i64_ty),
            MlBuiltin::Write(t) => (vec![Str("path"), Buf(t, false)], i64_ty),
            MlBuiltin::ReadAt(t) => (vec![Str("path"), Int("byte offset"), Buf(t, true)], i64_ty),
            MlBuiltin::FileSize => (vec![Str("path")], i64_ty),
            MlBuiltin::StRead(t) => (vec![Str("path"), Str("tensor name"), Buf(t, true)], i64_ty),
        };
        if args.len() != sig.len() {
            let want: Vec<String> = sig
                .iter()
                .map(|a| match a {
                    Str(what) => format!("a `*u8` {what}"),
                    Int(what) => format!("an integer {what}"),
                    Buf(t, _) => format!("a {} buffer", buffer_desc(*t)),
                })
                .collect();
            let plural = if sig.len() == 1 { "" } else { "s" };
            self.error(
                span,
                "E0503",
                format!(
                    "`{name}` takes exactly {} argument{plural} ({}), but {} were supplied",
                    sig.len(),
                    want.join(", "),
                    args.len()
                ),
            );
            return Some(ret);
        }
        for (i, a) in sig.iter().enumerate() {
            match *a {
                Str(what) => self.ml_str_arg(name, &args[i], &arg_tys[i], i, what),
                Int(what) => self.ml_scalar_arg(name, &args[i], &mut arg_tys[i], false, what),
                Buf(t, written) => self.ml_buffer_arg(name, &args[i], &arg_tys[i], t, i, written),
            }
        }
        Some(ret)
    }

    /// A NUL-terminated `*u8` string (a literal or a `*u8` binding); anything else is `E0401`.
    fn ml_str_arg(&mut self, name: &str, arg: &Expr, ty: &Ty, i: usize, what: &str) {
        match ty {
            Ty::Ptr { pointee, .. } if matches!(pointee.as_ref(), Ty::Scalar(Scalar::U8)) => {}
            Ty::Unknown | Ty::Error => {}
            other => self.error(
                arg.span,
                "E0401",
                format!(
                    "`{name}` expects a `*u8` {what} as its {} argument, found a value of type \
                     `{}`",
                    ordinal(i),
                    other.display(self.interner)
                ),
            ),
        }
    }

    /// An integer (`float == false`) or floating-point (`float == true`) scalar operand. An
    /// unsuffixed literal adapts to the ABI width (`i64` / `f32`), exactly as it would to a
    /// parameter of that type, so `read_f32_at(p, 4096, w)` needs no suffix.
    fn ml_scalar_arg(&mut self, name: &str, arg: &Expr, ty: &mut Ty, float: bool, what: &str) {
        let target = Ty::Scalar(if float { Scalar::F32 } else { Scalar::I64 });
        if self.literal_adapts(&target, arg) {
            self.retype_adapted_literal(arg, &target);
            self.range_check_int_literal(arg, &target);
            *ty = target;
        }
        let ok = match ty {
            Ty::Scalar(s) => {
                if float {
                    s.is_float()
                } else {
                    s.is_int()
                }
            }
            Ty::Unknown | Ty::Error => true,
            _ => false,
        };
        if !ok {
            let kind = if float { "a float" } else { "an integer" };
            self.error(
                arg.span,
                "E0401",
                format!(
                    "`{name}` expects {kind} {what}, found a value of type `{}`",
                    ty.display(self.interner)
                ),
            );
        }
    }

    /// A buffer operand of element type `elem` (see the module docs for the accepted forms). When
    /// the builtin writes the buffer, a non-`mut` aggregate parameter is `E0304` — the same rule as
    /// `p[i] = …`, because the write lands in the caller's storage.
    fn ml_buffer_arg(
        &mut self,
        name: &str,
        arg: &Expr,
        ty: &Ty,
        elem: Scalar,
        i: usize,
        written: bool,
    ) {
        let span = arg.span;
        let want = Ty::Scalar(elem);
        let ok = match ty {
            Ty::Slice(e) => **e == want,
            Ty::Array { elem: e, .. } => **e == want,
            Ty::Tensor {
                elem: e,
                shape,
                layout,
            } if *e == elem => {
                if *layout != Layout::Contiguous {
                    self.error(
                        span,
                        "E0401",
                        format!(
                            "`{name}` moves elements in row-major order, so its {} argument must \
                             be a contiguous tensor, not a `{}` one",
                            ordinal(i),
                            ty.display(self.interner)
                        ),
                    );
                    return;
                }
                if shape.0.iter().any(|d| matches!(d, Dim::Dynamic)) {
                    self.error(
                        span,
                        "E0401",
                        format!(
                            "`{name}` needs the element count of its {} argument, but `{}` has a \
                             `?` dimension; give the dimension a name or a size",
                            ordinal(i),
                            ty.display(self.interner)
                        ),
                    );
                    return;
                }
                true
            }
            Ty::Unknown | Ty::Error => true,
            _ => false,
        };
        if !ok {
            self.error(
                span,
                "E0401",
                format!(
                    "`{name}` expects a {} buffer as its {} argument, found a value of type `{}`",
                    buffer_desc(elem),
                    ordinal(i),
                    ty.display(self.interner)
                ),
            );
            return;
        }
        if written {
            if let ExprKind::Path(p) = &arg.kind {
                if p.is_single() {
                    let sym = p.first().sym;
                    if self.is_immutable_param(sym) && self.param_is_aggregate(sym) {
                        let nm = self.sym_str(sym).to_string();
                        self.error(
                            span,
                            "E0304",
                            format!(
                                "`{name}` writes its {} argument, but `{nm}` is a non-`mut` \
                                 parameter passed by reference — add `mut` to the parameter \
                                 (`fn …(mut {nm}: …)`) to allow the write",
                                ordinal(i)
                            ),
                        );
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use wukong_span::{Interner, SourceId};

    fn errors(src: &str) -> Vec<&'static str> {
        let mut interner = Interner::new();
        let (module, pd) = wukong_parser::parse_module(src, SourceId(0), &mut interner);
        assert!(pd.is_empty(), "parse errors: {pd:?}");
        let (_res, diags) = crate::check(&module, &interner);
        diags.into_iter().filter_map(|d| d.code).collect()
    }

    /// Each source must produce exactly the one code given.
    fn assert_rejected(cases: &[(&str, &str)]) {
        for &(src, code) in cases {
            assert_eq!(errors(src), [code], "{src}");
        }
    }

    #[test]
    fn io_name_table_and_dtype_codes() {
        assert_eq!(ml_builtin("read_bf16"), Some(MlBuiltin::Read(Scalar::Bf16)));
        assert_eq!(
            ml_builtin("read_bf16_at"),
            Some(MlBuiltin::ReadAt(Scalar::Bf16))
        );
        assert_eq!(ml_builtin("write_u64"), Some(MlBuiltin::Write(Scalar::U64)));
        assert_eq!(
            ml_builtin("st_read_i16"),
            Some(MlBuiltin::StRead(Scalar::I16))
        );
        assert_eq!(
            ml_builtin("npy_read_f16"),
            Some(MlBuiltin::NpyRead(Scalar::F16))
        );
        assert_eq!(ml_builtin("file_size"), Some(MlBuiltin::FileSize));
        for not_one in [
            "read_bool",
            "read_usize",
            "read_f32_at_x",
            "write_char",
            "st_read_",
            "npy_read",
        ] {
            assert_eq!(ml_builtin(not_one), None, "{not_one}");
        }
        // The dtype codes are a dense, duplicate-free table over the twelve I/O element types.
        let mut codes: Vec<i64> = [
            "f32", "f64", "f16", "bf16", "i8", "i16", "i32", "i64", "u8", "u16", "u32", "u64",
        ]
        .iter()
        .map(|s| io_dtype_code(io_dtype(s).unwrap()))
        .collect();
        codes.sort_unstable();
        assert_eq!(codes, (0..12).collect::<Vec<i64>>());
        assert_eq!(io_dtype_code(Scalar::Bool), -1);
    }

    #[test]
    fn io_accepts_every_buffer_form_and_types_its_result() {
        let ok = "fn f(mut t: Tensor[f32, 2, 3], mut a: [u16; 4], x: []f32) -> i64 { \
                  let s: []bf16 = alloc_bf16(6); \
                  let n: i64 = read_bf16(\"a.bin\", s) + read_u16_at(\"a.bin\", 4, a) \
                      + st_read_f32(\"a.st\", \"w\", t) + npy_read_u16(\"a.npy\", a) \
                      + file_size(\"a.bin\") + write_bf16(\"b.bin\", s) + write_f32(\"c.bin\", x); \
                  let off: u32 = 8; \
                  return n + read_bf16_at(\"a.bin\", off, s); }";
        assert!(errors(ok).is_empty(), "unexpected: {:?}", errors(ok));
        // The result is a typed `i64`, not a lenient `Unknown`.
        assert_eq!(
            errors("fn f() { let x: f32 = file_size(\"a.bin\"); }"),
            ["E0401"]
        );
        // A generic tensor's dims are bound variables, so its element count is known at the call.
        let generic = "fn g<R, C>(mut t: Tensor[u32, R, C]) -> i64 { return read_u32(\"a\", t); }";
        assert!(errors(generic).is_empty(), "{:?}", errors(generic));
        // Reading a non-`mut` parameter is fine: only a written operand needs `mut`.
        let read_only = "fn f(t: Tensor[f32, 4]) -> i64 { return write_f32(\"a\", t); }";
        assert!(errors(read_only).is_empty(), "{:?}", errors(read_only));
    }

    #[test]
    fn io_misuse_is_rejected_with_the_standard_codes() {
        assert_rejected(&[
            // A `?` dimension leaves the element count unknown.
            (
                "fn f(mut t: Tensor[f32, ?]) -> i64 { return read_f32(\"a\", t); }",
                "E0401",
            ),
            // The element type is exact: f16 is not bf16.
            (
                "fn f() -> i64 { return read_bf16(\"a\", alloc_f16(2)); }",
                "E0401",
            ),
            // Writing a non-`mut` by-reference parameter.
            (
                "fn f(t: Tensor[f32, 4]) -> i64 { return read_f32(\"a\", t); }",
                "E0304",
            ),
            (
                "fn f(w: [f32; 4]) -> i64 { return npy_read_f32(\"a\", w); }",
                "E0304",
            ),
            // A tensor name must be a string, an offset an integer.
            (
                "fn f(mut w: [f32; 4]) -> i64 { return st_read_f32(\"a\", 3, w); }",
                "E0401",
            ),
            (
                "fn f(mut w: [f32; 4]) -> i64 { return read_f32_at(\"a\", 1.5, w); }",
                "E0401",
            ),
            // Arity.
            ("fn f() -> i64 { return file_size(); }", "E0503"),
        ]);
    }
}
