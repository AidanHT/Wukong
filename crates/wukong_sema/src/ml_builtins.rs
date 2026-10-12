//! Typing for the ML I/O and randomness builtins: the typed file-I/O family over every element type
//! (whole-file and byte-offset reads, writes, `file_size`), the safetensors / `.npy` / GGUF
//! container readers and the GGUF tensor-table queries, the memory-mapped loads, the Philox
//! counter-based RNG, and the top-k / argsort / categorical-sampling kernels.
//!
//! Like the heap and timing builtins, these are typed nominally here — the call's result is a
//! concrete type, so `let n = read_bf16_at(p, off, w)` composes in arithmetic — and every misuse is
//! rejected with the standard codes before `wukong_mir_build` lowers the call structurally:
//! a wrong arity is `E0503`, a wrong operand kind or element type `E0437`, and a builtin that writes
//! a non-`mut` aggregate parameter `E0304` (exactly what `p[i] = …` through that parameter is).
//! [`ml_builtin`] is the one name table; `mir_build` matches the same enum, so the two cannot
//! disagree about which names are builtins or what element type each one moves.
//!
//! **Buffers.** A buffer operand is a `[]T` slice, a fixed `[T; N]` array, or a row-major
//! (`Contiguous`) `Tensor[T, …]` view whose every dimension is known at the call — a constant or a
//! bound dimension variable, never `?`. The element type must be exactly the builtin's `T`: a
//! `[]i32` handed to `read_f32` would read the wrong stride and decode the wrong type. A container
//! read into a *tensor* also checks the file's shape against the tensor's dims at load time. A
//! dtype-generic `Tensor[T, …]` is checked per instantiation (`dtype.rs`): an instance binding `T`
//! to the builtin's element reaches it like the concrete function, any other binding is `E0416`.
//!
//! **Integers.** Byte offsets, seeds and counters accept any integer type (the value's bits, as
//! `x as u64` would give them); an unsuffixed literal adapts to `i64`.
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
    /// `gguf_read_<T>(path: *u8, name: *u8, buf) -> i64` — one tensor of a GGUF file: verbatim,
    /// widened or dequantized into `f32`, or as raw quantized blocks into `u8`.
    GgufRead(Scalar),
    /// `gguf_find` / `gguf_rank` / `gguf_type(path: *u8, name: *u8) -> i64` and
    /// `gguf_dim(path: *u8, name: *u8, k: int) -> i64` — a GGUF tensor-table lookup.
    GgufQuery(GgufQuery),
    /// `map_<T>(path: *u8) -> []T`, `st_map_<T>(path: *u8, name: *u8) -> []T`,
    /// `npy_map_<T>(path: *u8) -> []T` and `gguf_map_<T>(path: *u8, name: *u8) -> []T` — a file's
    /// bytes (or one stored tensor's, in its own element type) as a slice, memory-mapped
    /// copy-on-write where the OS allows; released with `free`, empty on any failure.
    Map(MapSource, Scalar),
    /// `rand_u32(seed: int, ctr: int) -> u32` — Philox4x32-10, word 0.
    RandU32,
    /// `rand_f32(seed: int, ctr: int) -> f32` — uniform in `[0, 1)`.
    RandF32,
    /// `randn_f32(seed: int, ctr: int) -> f32` — standard normal (Box–Muller).
    RandnF32,
    /// `rand_fill_f32(buf, seed: int, ctr0: int)` — `buf[i] = rand_f32(seed, ctr0 + i)`.
    RandFill,
    /// `topk_f32(x, vals, idx) -> i64` — the largest `k` values and their indices.
    TopK,
    /// `argsort_f32(x, idx) -> i64` / `argsort_desc_f32(x, idx) -> i64` — a stable argsort.
    Argsort {
        /// `argsort_desc_f32`.
        descending: bool,
    },
    /// `sample_categorical(probs, u: float) -> i64` — an inverse-CDF draw.
    SampleCategorical,
}

/// What a GGUF tensor-table query asks for.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum GgufQuery {
    /// `gguf_find`: the tensor's index in the file's tensor table.
    Find,
    /// `gguf_rank`: its number of dimensions.
    Rank,
    /// `gguf_dim`: dimension `k`, outermost first.
    Dim,
    /// `gguf_type`: its ggml type id (`0` F32, `1` F16, `2` Q4_0, `8` Q8_0, `30` BF16, …).
    Type,
}

impl GgufQuery {
    /// The runtime's query code — the compiler's copy of `wukong_runtime::gguf::GGUF_Q_*` (a test
    /// in `wukong_interp`, which sees both, pins every row).
    pub fn code(self) -> i64 {
        match self {
            GgufQuery::Find => 0,
            GgufQuery::Rank => 1,
            GgufQuery::Dim => 2,
            GgufQuery::Type => 3,
        }
    }
}

/// What a memory-mapped load maps.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum MapSource {
    /// `map_<T>`: the whole file.
    File,
    /// `st_map_<T>`: one safetensors tensor.
    Safetensors,
    /// `npy_map_<T>`: the array of a `.npy` file.
    Npy,
    /// `gguf_map_<T>`: one GGUF tensor (its raw blocks, for a quantized one).
    Gguf,
}

impl MapSource {
    /// The runtime's container code — the compiler's copy of `wukong_runtime::mmap::MAP_*` (pinned
    /// by a test in `wukong_interp`).
    pub fn code(self) -> i64 {
        match self {
            MapSource::File => 0,
            MapSource::Safetensors => 1,
            MapSource::Npy => 2,
            MapSource::Gguf => 3,
        }
    }

    /// Whether the map names a tensor (`st_map_<T>`, `gguf_map_<T>`).
    pub fn named(self) -> bool {
        matches!(self, MapSource::Safetensors | MapSource::Gguf)
    }
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
    if let Some(rest) = name.strip_prefix("gguf_read_") {
        return io_dtype(rest).map(MlBuiltin::GgufRead);
    }
    for (prefix, src) in [
        ("map_", MapSource::File),
        ("st_map_", MapSource::Safetensors),
        ("npy_map_", MapSource::Npy),
        ("gguf_map_", MapSource::Gguf),
    ] {
        if let Some(rest) = name.strip_prefix(prefix) {
            return io_dtype(rest).map(|t| MlBuiltin::Map(src, t));
        }
    }
    Some(match name {
        "file_size" => MlBuiltin::FileSize,
        "gguf_find" => MlBuiltin::GgufQuery(GgufQuery::Find),
        "gguf_rank" => MlBuiltin::GgufQuery(GgufQuery::Rank),
        "gguf_dim" => MlBuiltin::GgufQuery(GgufQuery::Dim),
        "gguf_type" => MlBuiltin::GgufQuery(GgufQuery::Type),
        "rand_u32" => MlBuiltin::RandU32,
        "rand_f32" => MlBuiltin::RandF32,
        "randn_f32" => MlBuiltin::RandnF32,
        "rand_fill_f32" => MlBuiltin::RandFill,
        "topk_f32" => MlBuiltin::TopK,
        "argsort_f32" => MlBuiltin::Argsort { descending: false },
        "argsort_desc_f32" => MlBuiltin::Argsort { descending: true },
        "sample_categorical" => MlBuiltin::SampleCategorical,
        _ => return None,
    })
}

/// Whether the builtin `name` is a pure function of its scalar arguments — no memory traffic, no
/// state — so a `@parallel` region may evaluate it per iteration in any order (the RNG scalars).
pub fn ml_builtin_is_pure(name: &str) -> bool {
    matches!(
        ml_builtin(name),
        Some(MlBuiltin::RandU32 | MlBuiltin::RandF32 | MlBuiltin::RandnF32)
    )
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
    /// A floating-point scalar (`f32` ABI).
    Float(&'static str),
    /// A buffer of this element type; `true` when the builtin writes it.
    Buf(Scalar, bool),
}

/// The argument list and result type of `b` — the one table both the type checker
/// ([`Sema::type_ml_builtin`]) and the lowering's element backstop ([`MlBuiltin::buffer_elem`])
/// read.
fn signature(b: MlBuiltin) -> (Vec<Arg>, Ty) {
    use Arg::{Buf, Float, Int, Str};
    let i64_ty = Ty::Scalar(Scalar::I64);
    let f32_ty = Ty::Scalar(Scalar::F32);
    match b {
        MlBuiltin::Read(t) | MlBuiltin::NpyRead(t) => (vec![Str("path"), Buf(t, true)], i64_ty),
        MlBuiltin::Write(t) => (vec![Str("path"), Buf(t, false)], i64_ty),
        MlBuiltin::ReadAt(t) => (vec![Str("path"), Int("byte offset"), Buf(t, true)], i64_ty),
        MlBuiltin::FileSize => (vec![Str("path")], i64_ty),
        MlBuiltin::StRead(t) | MlBuiltin::GgufRead(t) => {
            (vec![Str("path"), Str("tensor name"), Buf(t, true)], i64_ty)
        }
        MlBuiltin::GgufQuery(GgufQuery::Dim) => (
            vec![Str("path"), Str("tensor name"), Int("dimension index")],
            i64_ty,
        ),
        MlBuiltin::GgufQuery(_) => (vec![Str("path"), Str("tensor name")], i64_ty),
        MlBuiltin::Map(src, t) => {
            let args = if src.named() {
                vec![Str("path"), Str("tensor name")]
            } else {
                vec![Str("path")]
            };
            (args, Ty::Slice(Box::new(Ty::Scalar(t))))
        }
        MlBuiltin::RandU32 => (vec![Int("seed"), Int("counter")], Ty::Scalar(Scalar::U32)),
        MlBuiltin::RandF32 | MlBuiltin::RandnF32 => (vec![Int("seed"), Int("counter")], f32_ty),
        MlBuiltin::RandFill => (
            vec![Buf(Scalar::F32, true), Int("seed"), Int("counter")],
            Ty::Unit,
        ),
        MlBuiltin::TopK => (
            vec![
                Buf(Scalar::F32, false),
                Buf(Scalar::F32, true),
                Buf(Scalar::I64, true),
            ],
            i64_ty,
        ),
        MlBuiltin::Argsort { .. } => (
            vec![Buf(Scalar::F32, false), Buf(Scalar::I64, true)],
            i64_ty,
        ),
        MlBuiltin::SampleCategorical => (
            vec![Buf(Scalar::F32, false), Float("uniform variate")],
            i64_ty,
        ),
    }
}

impl MlBuiltin {
    /// The element type argument `i` must hold when it is a buffer operand, else `None`.
    ///
    /// For the lowering's backstop (`wukong_mir_build::ml_builtins`): a dtype-generic
    /// `Tensor[T, ..]` operand is accepted in the generic body and its element is checked per
    /// instantiation (`E0416`), but `instantiate.rs` re-checks a bounded number of instances, while
    /// the monomorphizer lowers all of them. An instance past that budget must not hand a buffer of
    /// another element to the runtime, which reads the bytes as this type.
    pub fn buffer_elem(self, i: usize) -> Option<Scalar> {
        match signature(self).0.get(i)? {
            Arg::Buf(t, _) => Some(*t),
            _ => None,
        }
    }
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
        use Arg::{Buf, Float, Int, Str};
        let b = ml_builtin(name)?;
        let (sig, ret) = signature(b);
        if args.len() != sig.len() {
            let want: Vec<String> = sig
                .iter()
                .map(|a| match a {
                    Str(what) => format!("a `*u8` {what}"),
                    Int(what) => format!("an integer {what}"),
                    Float(what) => format!("a float {what}"),
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
                Float(what) => self.ml_scalar_arg(name, &args[i], &mut arg_tys[i], true, what),
                Buf(t, written) => self.ml_buffer_arg(name, &args[i], &arg_tys[i], t, i, written),
            }
        }
        Some(ret)
    }

    /// A NUL-terminated `*u8` string (a literal or a `*u8` binding); anything else is `E0437`.
    fn ml_str_arg(&mut self, name: &str, arg: &Expr, ty: &Ty, i: usize, what: &str) {
        match ty {
            Ty::Ptr { pointee, .. } if matches!(pointee.as_ref(), Ty::Scalar(Scalar::U8)) => {}
            Ty::Unknown | Ty::Error => {}
            other => self.error(
                arg.span,
                "E0437",
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
    pub(crate) fn ml_scalar_arg(
        &mut self,
        name: &str,
        arg: &Expr,
        ty: &mut Ty,
        float: bool,
        what: &str,
    ) {
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
                "E0437",
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
    pub(crate) fn ml_buffer_arg(
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
        // Only this family's own builtins defer a dtype-generic element: their lowering refuses an
        // instance the re-check never reached (`MlBuiltin::buffer_elem`). The decode builtins
        // (`decode_builtins.rs`) share this operand check but not that backstop, so for them a
        // `Tensor[T, ..]` stays the plain `E0437` of the generic body.
        let defers = ml_builtin(name).is_some();
        let ok = match ty {
            Ty::Slice(e) => **e == want,
            Ty::Array { elem: e, .. } => **e == want,
            // A dtype-generic tensor (`elem_var: Some`) is checked per instantiation (`dtype.rs`):
            // the generic body accepts it here, and each concrete instance is re-checked with `T`
            // substituted — `T` bound to this builtin's element reaches it, any other is `E0416`
            // (the arm below).
            Ty::Tensor {
                elem: e,
                shape,
                layout,
                elem_var,
            } if (elem_var.is_some() && defers) || (elem_var.is_none() && *e == elem) => {
                if *layout != Layout::Contiguous {
                    self.error(
                        span,
                        "E0437",
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
                        "E0437",
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
            // While a concrete instantiation is re-checked, a tensor of another element was a
            // deferred `Tensor[T, ..]` the substitution made concrete (`dtype.rs`).
            Ty::Tensor {
                elem: e,
                elem_var: None,
                ..
            } if defers && self.rechecking_instance() => {
                self.error(
                    span,
                    "E0416",
                    format!(
                        "`{name}` moves `{}` elements, but in this instantiation its {} argument \
                         holds `{}` (a dtype-generic `Tensor[T, ..]` reaches `{name}` only when \
                         `T = {}`)",
                        elem.name(),
                        ordinal(i),
                        e.name(),
                        elem.name()
                    ),
                );
                return;
            }
            Ty::Unknown | Ty::Error => true,
            _ => false,
        };
        if !ok {
            self.error(
                span,
                "E0437",
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
                "E0437",
            ),
            // The element type is exact: f16 is not bf16.
            (
                "fn f() -> i64 { return read_bf16(\"a\", alloc_f16(2)); }",
                "E0437",
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
                "E0437",
            ),
            (
                "fn f(mut w: [f32; 4]) -> i64 { return read_f32_at(\"a\", 1.5, w); }",
                "E0437",
            ),
            // Arity.
            ("fn f() -> i64 { return file_size(); }", "E0503"),
        ]);
    }

    #[test]
    fn gguf_builtins_are_named_and_typed() {
        assert_eq!(
            ml_builtin("gguf_read_f32"),
            Some(MlBuiltin::GgufRead(Scalar::F32))
        );
        assert_eq!(
            ml_builtin("gguf_read_u8"),
            Some(MlBuiltin::GgufRead(Scalar::U8))
        );
        assert_eq!(
            ml_builtin("gguf_dim"),
            Some(MlBuiltin::GgufQuery(GgufQuery::Dim))
        );
        for not_one in [
            "gguf_read",
            "gguf_read_bool",
            "gguf_read_f32_at",
            "gguf_shape",
        ] {
            assert_eq!(ml_builtin(not_one), None, "{not_one}");
        }
        let ok = "fn f(mut t: Tensor[f32, 2, 3], mut q: []u8, mut h: [bf16; 4]) -> i64 { \
                  let k: u32 = 1; \
                  return gguf_read_f32(\"m.gguf\", \"w\", t) + gguf_read_u8(\"m.gguf\", \"q\", q) \
                      + gguf_read_bf16(\"m.gguf\", \"h\", h) + gguf_find(\"m.gguf\", \"w\") \
                      + gguf_rank(\"m.gguf\", \"w\") + gguf_dim(\"m.gguf\", \"w\", k) \
                      + gguf_dim(\"m.gguf\", \"w\", 0) + gguf_type(\"m.gguf\", \"w\"); }";
        assert!(errors(ok).is_empty(), "unexpected: {:?}", errors(ok));
        assert_rejected(&[
            // The read writes its buffer.
            (
                "fn f(t: Tensor[f32, 4]) -> i64 { return gguf_read_f32(\"a\", \"w\", t); }",
                "E0304",
            ),
            // A tensor name is a string, a dimension index an integer, the element type exact.
            (
                "fn f(mut w: [f32; 4]) -> i64 { return gguf_read_f32(\"a\", 3, w); }",
                "E0437",
            ),
            (
                "fn f() -> i64 { return gguf_dim(\"a\", \"w\", 1.5); }",
                "E0437",
            ),
            (
                "fn f(mut w: []i32) -> i64 { return gguf_read_f32(\"a\", \"w\", w); }",
                "E0437",
            ),
            // Arity.
            ("fn f() -> i64 { return gguf_dim(\"a\", \"w\"); }", "E0503"),
            (
                "fn f() -> i64 { return gguf_rank(\"a\", \"w\", 0); }",
                "E0503",
            ),
            // The result is a typed `i64`.
            ("fn f() { let x: f32 = gguf_type(\"a\", \"w\"); }", "E0401"),
        ]);
    }

    #[test]
    fn map_builtins_are_named_and_typed() {
        assert_eq!(
            ml_builtin("map_f32"),
            Some(MlBuiltin::Map(MapSource::File, Scalar::F32))
        );
        assert_eq!(
            ml_builtin("st_map_bf16"),
            Some(MlBuiltin::Map(MapSource::Safetensors, Scalar::Bf16))
        );
        assert_eq!(
            ml_builtin("npy_map_i16"),
            Some(MlBuiltin::Map(MapSource::Npy, Scalar::I16))
        );
        assert_eq!(
            ml_builtin("gguf_map_u8"),
            Some(MlBuiltin::Map(MapSource::Gguf, Scalar::U8))
        );
        for not_one in ["map_", "map_bool", "map", "st_map", "map_f32_at"] {
            assert_eq!(ml_builtin(not_one), None, "{not_one}");
        }
        let ok = "fn f() -> i64 { \
                  let a: []f32 = map_f32(\"w.bin\"); \
                  let b: []bf16 = st_map_bf16(\"m.st\", \"w\"); \
                  let c: []i16 = npy_map_i16(\"a.npy\"); \
                  let d: []u8 = gguf_map_u8(\"m.gguf\", \"q\"); \
                  let n: i64 = a.len() + b.len() + c.len() + d.len(); \
                  free(a); free(b); free(c); free(d); \
                  return n; }";
        assert!(errors(ok).is_empty(), "unexpected: {:?}", errors(ok));
        assert_rejected(&[
            // The result is the stored element type's slice.
            (
                "fn f() { let a: []f64 = map_f32(\"w.bin\"); free(a); }",
                "E0401",
            ),
            // A tensor name is a string; the whole-file maps take none.
            (
                "fn f() { let a: []f32 = st_map_f32(\"m.st\", 1); free(a); }",
                "E0437",
            ),
            (
                "fn f() { let a: []f32 = map_f32(\"w.bin\", \"w\"); free(a); }",
                "E0503",
            ),
            (
                "fn f() { let a: []f32 = gguf_map_f32(\"m.gguf\"); free(a); }",
                "E0503",
            ),
        ]);
    }

    #[test]
    fn rng_builtins_are_named_pure_and_typed() {
        assert_eq!(ml_builtin("rand_u32"), Some(MlBuiltin::RandU32));
        assert_eq!(ml_builtin("rand_fill_f32"), Some(MlBuiltin::RandFill));
        assert_eq!(ml_builtin("randn_f64"), None);
        assert!(["rand_u32", "rand_f32", "randn_f32"]
            .iter()
            .all(|n| ml_builtin_is_pure(n)));
        assert!(!ml_builtin_is_pure("rand_fill_f32") && !ml_builtin_is_pure("read_f32"));
        let ok = "fn f(mut w: [f32; 8]) -> f32 { \
                  let seed: u64 = 7; \
                  let ctr: i32 = 3; \
                  let u: u32 = rand_u32(seed, ctr); \
                  rand_fill_f32(w, seed, -1); \
                  return rand_f32(1, 2) + randn_f32(-3, 4); }";
        assert!(errors(ok).is_empty(), "unexpected: {:?}", errors(ok));
        assert_rejected(&[
            // The results are typed, not lenient: a u32 is not an f32.
            ("fn f() { let x: f32 = rand_u32(1, 2); }", "E0401"),
            // The seed is an integer's bits; a float has none to offer.
            ("fn f() -> f32 { return rand_f32(0.5, 1); }", "E0437"),
            // The fill writes its buffer.
            ("fn f(w: [f32; 4]) { rand_fill_f32(w, 1, 2); }", "E0304"),
            ("fn f() -> f32 { return randn_f32(1); }", "E0503"),
        ]);
    }

    #[test]
    fn sampling_kernels_are_named_and_typed() {
        assert_eq!(
            ml_builtin("argsort_desc_f32"),
            Some(MlBuiltin::Argsort { descending: true })
        );
        assert_eq!(ml_builtin("topk"), None);
        let ok = "fn f(x: []f32, mut v: []f32, mut i: []i64, w: [f32; 8]) -> i64 { \
                  let u: f64 = 0.5; \
                  return topk_f32(x, v, i) + argsort_f32(x, i) + argsort_desc_f32(w, i) \
                      + sample_categorical(x, 0.5) + sample_categorical(w, u); }";
        assert!(errors(ok).is_empty(), "unexpected: {:?}", errors(ok));
        assert_rejected(&[
            // `vals` is written, so a non-`mut` slice parameter there is rejected; `x` is only read.
            (
                "fn f(x: []f32, v: []f32, mut i: []i64) -> i64 { return topk_f32(x, v, i); }",
                "E0304",
            ),
            // The variate is a float, the index buffer i64.
            (
                "fn f(p: []f32) -> i64 { return sample_categorical(p, 1); }",
                "E0437",
            ),
            (
                "fn f(x: []f32, mut i: []i32) -> i64 { return argsort_f32(x, i); }",
                "E0437",
            ),
            (
                "fn f(p: []f32) -> i64 { return sample_categorical(p); }",
                "E0503",
            ),
        ]);
    }
}
