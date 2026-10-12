//! Lowering of the printing builtins `wukong_sema::text` types.
//!
//! **`print` / `println` of a scalar** lower to a void call the backends route to a per-type
//! runtime printer. Most of the routing is visible in MIR alone: an `f64` argument and an `f32`
//! (or `f16` / `bf16`, which live in `f32` registers) one differ in MIR type, and every backend
//! picks `wukong_rt_print_f64` or `wukong_rt_print_f32` from it. What MIR cannot see is a `bool`
//! (an `i1`, the same register as a comparison result) or a `char` (an `i32`, the same register as
//! an integer): those are routed here, by their sema type, to the `print_bool` / `print_char`
//! symbols, zero-extended to `i64` — so `print(true)` prints `true` and `print('A')` prints `A`.
//!
//! **Everything else prints through the formatting runtime** (`wukong_runtime::fmt`): `print` of
//! an aggregate or a reference, `printf` / `eprintf` / `sprintf`, `print_raw` / `eprint` /
//! `eprintln` and `write_stdout` / `write_stderr`. Each piece of output is one call to a
//! `wukong_rt_fmt_*` entry taking the sink operands `(target, buf, cap, pos)` and returning the
//! position after it; a `sprintf` threads that position from call to call, a stream ignores it.
//!
//! * A **format string** is parsed here at compile time (`wukong_sema::text::parse_format`, the
//!   parser sema checked it with). Its literal runs are printed straight out of the string's own
//!   `.rodata` blob (`wukong_rt_fmt_str` with a byte offset and length), and each placeholder's
//!   spec is packed into one `i64` constant. Every argument is evaluated, in order, before
//!   anything prints — Rust's `format_args!` order.
//! * An **aggregate** prints **structurally**, like Rust's `{:?}`, expanded at compile time over
//!   its type: `P { x: 1, y: 2.5 }`, `(1, 'a')`, `[[1, 2], [3, 4]]`. A run of scalars — an array or
//!   slice of numbers, `bool`s or `char`s — is one `wukong_rt_fmt_slice` call over its storage, so
//!   a slice's runtime length needs no loop in MIR and a long one summarizes NumPy-style at run
//!   time; an array of aggregates is unrolled by the same summarizing rule. Generated punctuation
//!   and names (`", "`, `"P { "`) travel as up to eight bytes packed into an `i64` constant
//!   (`wukong_rt_fmt_lit`), so no data section is needed for them.
//!
//! * A **`str`** prints its text — `wukong_rt_fmt_str` over its data pointer and byte length, read
//!   from the fat pointer — wherever a `*u8` string would print. Sema publishes a `str` as the
//!   `[]u8` it is laid out as, so the type cannot say which slices are text: the arguments sema
//!   recorded in `SemaResult::str_print_args` can, and they are routed here by their `NodeId`. A
//!   `str` INSIDE a printed aggregate (a field, an element, behind a reference) is found by walking
//!   the argument's unerased type, which sema keeps for exactly this (`StrInfo::print_types` and
//!   `StrInfo::struct_fields`); it prints quoted, as every string element does.
//! * **`str_format(fmt, args…)`** formats twice over the same evaluated arguments: once into no
//!   buffer, which measures, then into a fresh heap allocation of that size plus a NUL.
//!
//! **The process interface** (`arg_count`, `arg`, `arg_i64`, `arg_f32`, `env`, `read_stdin_u8`,
//! `exit`) lowers to one call each of a `wukong_rt_*` entry of `wukong_runtime::process`, its
//! integer operands widened to `i64` and its float operand to `f32`.
//!
//! Every call is an opaque `Op::Call`, so no optimizer moves output across another side effect;
//! the interpreter and the native backends render through the same runtime functions, so the bytes
//! printed cannot depend on the backend. `--backend=gpu-native` declines the `wukong_rt_fmt_*`
//! calls (the usual UNSUPPORTED skip).

use wukong_ast::{Expr, ExprKind};
use wukong_mir::{BinOp, CmpOp, MirType, Op, ValueId};
use wukong_sema::text::{self as st, FmtPiece, FmtSpec, TextBuiltin};
use wukong_sema::DefKind;
use wukong_span::{Interner, Span, Symbol};
use wukong_types::{Scalar, Ty};

use crate::{FnLowerer, SLICE_LEN_OFF, SLICE_PTR_OFF};

/// The symbols this module lowers to, interned once at the top of `lower_program` (`FnLowerer`
/// holds only a shared `&Interner`) and carried on `GemmSyms`.
#[derive(Clone, Copy)]
pub(crate) struct TextSyms {
    print_bool: Symbol,
    println_bool: Symbol,
    print_char: Symbol,
    println_char: Symbol,
    fmt_int: Symbol,
    fmt_i128: Symbol,
    fmt_f64: Symbol,
    fmt_f32: Symbol,
    fmt_str: Symbol,
    fmt_lit: Symbol,
    fmt_slice: Symbol,
    arg_count: Symbol,
    arg: Symbol,
    arg_i64: Symbol,
    arg_f32: Symbol,
    env: Symbol,
    exit: Symbol,
    read_stdin: Symbol,
}

impl TextSyms {
    pub(crate) fn new(interner: &mut Interner) -> TextSyms {
        TextSyms {
            print_bool: interner.intern("print_bool"),
            println_bool: interner.intern("println_bool"),
            print_char: interner.intern("print_char"),
            println_char: interner.intern("println_char"),
            fmt_int: interner.intern("wukong_rt_fmt_int"),
            fmt_i128: interner.intern("wukong_rt_fmt_i128"),
            fmt_f64: interner.intern("wukong_rt_fmt_f64"),
            fmt_f32: interner.intern("wukong_rt_fmt_f32"),
            fmt_str: interner.intern("wukong_rt_fmt_str"),
            fmt_lit: interner.intern("wukong_rt_fmt_lit"),
            fmt_slice: interner.intern("wukong_rt_fmt_slice"),
            arg_count: interner.intern("wukong_rt_arg_count"),
            arg: interner.intern("wukong_rt_arg"),
            arg_i64: interner.intern("wukong_rt_arg_i64"),
            arg_f32: interner.intern("wukong_rt_arg_f32"),
            env: interner.intern("wukong_rt_env"),
            exit: interner.intern("wukong_rt_exit"),
            read_stdin: interner.intern("wukong_rt_read_stdin"),
        }
    }
}

/// Where the text of one print goes: the sink operands every `wukong_rt_fmt_*` call leads with.
struct Sink {
    /// `1` stdout, `2` stderr, `0` the buffer.
    target: ValueId,
    /// The buffer's data pointer (the integer `0`, MIR's null, for a stream).
    buf: ValueId,
    /// The buffer's byte capacity (`0` for a stream).
    cap: ValueId,
    /// The write position: threaded from call to call into a buffer, constant `0` for a stream.
    pos: ValueId,
    buffered: bool,
    /// The printing call, for a lowering diagnostic.
    span: Span,
}

/// The spec an aggregate applies to its elements: a plain `{}` renders them in `Debug` style
/// (`[1.0, 2.5]`, `'a'`, `"s"`), as Rust's `{:?}` of a slice does (the runtime's `element_spec`).
fn element_spec(spec: &FmtSpec) -> FmtSpec {
    if spec.style == st::STYLE_DISPLAY {
        FmtSpec {
            style: st::STYLE_DEBUG,
            ..*spec
        }
    } else {
        *spec
    }
}

/// A type's name as printed: the last segment of its program-unique symbol (an imported item is
/// renamed `module.Name`).
fn short_name(full: &str) -> &str {
    full.rsplit('.').next().unwrap_or(full)
}

impl FnLowerer<'_> {
    /// `print(x)` / `println(x)` of a `bool` or a `char`: its own printer, the value zero-extended
    /// to `i64`; and of a `str`: its text. `None` for any other argument, which keeps the generic
    /// print path.
    pub(crate) fn lower_print_scalar(&mut self, newline: bool, arg: &Expr) -> Option<ValueId> {
        if self.is_str_print_arg(arg) {
            let mut sink = self.text_stream(false, arg.span);
            let v = self.lower_expr(arg);
            self.text_str(&mut sink, v, &FmtSpec::default());
            if newline {
                self.text_lit(&mut sink, b"\n");
            }
            return Some(self.const_zero(MirType::I32));
        }
        let syms = self.gemm.text;
        // A 128-bit integer has no per-type printer: it prints through the formatting entry, then
        // the newline every integer `print` ends with (`printf("{}\n", x)`).
        // An 8-bit float prints as its `f32` value (exact: every f8 value is an `f32`), through
        // the same formatting entry, so no backend's `print` needs an f8 arm.
        if let ty @ Ty::Scalar(Scalar::I128 | Scalar::U128 | Scalar::F8E4M3 | Scalar::F8E5M2) =
            self.expr_ty(arg)
        {
            let mut sink = self.text_stream(false, arg.span);
            let v = self.lower_expr(arg);
            self.text_scalar(&mut sink, v, &ty, &FmtSpec::default());
            self.text_lit(&mut sink, b"\n");
            return Some(self.const_zero(MirType::I32));
        }
        let (plain, line) = match self.expr_ty(arg) {
            Ty::Scalar(Scalar::Bool) => (syms.print_bool, syms.println_bool),
            Ty::Scalar(Scalar::Char) => (syms.print_char, syms.println_char),
            _ => return None,
        };
        let from = self.expr_mir(arg);
        let v0 = self.lower_expr(arg);
        let v = self.coerce_to(v0, &from, &MirType::I64, false);
        let func = if newline { line } else { plain };
        self.builder.build_void(Op::Call {
            func,
            args: vec![v],
        });
        Some(self.const_zero(MirType::I32))
    }

    /// `print(x)` / `println(x)` of an aggregate or a reference: the structural rendering, then a
    /// newline (both spellings append one). `None` for a scalar or a string, which keep their
    /// per-type printers.
    pub(crate) fn lower_print_value(&mut self, arg: &Expr, span: Span) -> Option<ValueId> {
        let ty = self.print_ty(arg);
        if !matches!(ty, Ty::Ref { .. }) && !self.text_by_pointer(&ty) {
            return None;
        }
        let mut sink = self.text_stream(false, span);
        let v = self.lower_expr(arg);
        self.text_value(&mut sink, v, &ty, &FmtSpec::default());
        self.text_lit(&mut sink, b"\n");
        Some(self.const_zero(MirType::I32))
    }

    /// Lower a call to a text builtin, or return `None` for any other callee. A user function of
    /// the same name never reaches here: `lower_call` resolves it first.
    pub(crate) fn lower_text_builtin(
        &mut self,
        name: Symbol,
        args: &[Expr],
        span: Span,
    ) -> Option<ValueId> {
        let b = st::text_builtin(self.interner.resolve(name))?;
        match b {
            TextBuiltin::Printf { stderr } => {
                let (fmt, rest) = args.split_first()?;
                let mut sink = self.text_stream(stderr, span);
                self.text_format(&mut sink, fmt, rest)?;
                Some(self.const_zero(MirType::I32))
            }
            TextBuiltin::Sprintf => {
                let (buf, rest) = args.split_first()?;
                let (fmt, rest) = rest.split_first()?;
                let (data, cap) = self.text_u8_buffer(buf)?;
                let target = self.text_const(0);
                let pos = self.text_const(0);
                let mut sink = Sink {
                    target,
                    buf: data,
                    cap,
                    pos,
                    buffered: true,
                    span,
                };
                // An empty write first: every write leaves the buffer NUL-terminated, and this one
                // guarantees there is at least one, so even an empty format yields "".
                let none = self.text_const(0);
                let f = self.gemm.text.fmt_lit;
                self.text_call(&mut sink, f, &[none, none]);
                self.text_format(&mut sink, fmt, rest)?;
                Some(sink.pos)
            }
            TextBuiltin::StrFormat => {
                let (fmt, rest) = args.split_first()?;
                // Every argument once, in order; then the pieces twice over those values: into
                // no buffer (capacity 0 writes nothing and returns the full length), and into a
                // fresh allocation of that length plus the NUL every buffer write leaves.
                let vals = self.text_format_args(rest);
                let target = self.text_const(0);
                let zero = self.text_const(0);
                let mut measure = Sink {
                    target,
                    buf: zero,
                    cap: zero,
                    pos: zero,
                    buffered: true,
                    span,
                };
                self.text_format_pieces(&mut measure, fmt, &vals)?;
                let len = measure.pos;
                let data = self.heap_bytes(len);
                let one = self.text_const(1);
                let cap = self
                    .builder
                    .build(MirType::I64, Op::Bin(BinOp::Add, len, one));
                let mut sink = Sink {
                    target,
                    buf: data,
                    cap,
                    pos: zero,
                    buffered: true,
                    span,
                };
                self.text_format_pieces(&mut sink, fmt, &vals)?;
                Some(self.fat_ptr(data, len))
            }
            TextBuiltin::PrintValue { stderr, newline } => {
                let mut sink = self.text_stream(stderr, span);
                if let Some(arg) = args.first() {
                    let ty = self.print_ty(arg);
                    let v = self.lower_expr(arg);
                    if self.is_str_print_arg(arg) {
                        self.text_str(&mut sink, v, &FmtSpec::default());
                    } else {
                        self.text_value(&mut sink, v, &ty, &FmtSpec::default());
                    }
                }
                if newline {
                    self.text_lit(&mut sink, b"\n");
                }
                Some(self.const_zero(MirType::I32))
            }
            TextBuiltin::WriteBytes { stderr } => {
                let (buf, n) = (args.first()?, args.get(1)?);
                let (data, len) = self.text_u8_buffer(buf)?;
                let count = self.text_clamped_count(n, len);
                let mut sink = self.text_stream(stderr, span);
                let spec = self.text_const(0);
                let f = self.gemm.text.fmt_str;
                Some(self.text_call(&mut sink, f, &[data, count, spec]))
            }
            TextBuiltin::WriteText => {
                let (path, buf, n) = (args.first()?, args.get(1)?, args.get(2)?);
                let p = self.lower_expr(path);
                let (data, len) = self.text_u8_buffer(buf)?;
                let count = self.text_clamped_count(n, len);
                let func = self.gemm.rt_write_u8;
                Some(self.builder.build(
                    MirType::I64,
                    Op::Call {
                        func,
                        args: vec![p, data, count],
                    },
                ))
            }
            TextBuiltin::ArgCount
            | TextBuiltin::Arg
            | TextBuiltin::ArgI64
            | TextBuiltin::ArgF32
            | TextBuiltin::Env
            | TextBuiltin::Exit
            | TextBuiltin::ReadStdin => self.lower_process_builtin(b, args),
        }
    }

    /// The process interface: one `wukong_runtime::process` entry per call.
    fn lower_process_builtin(&mut self, b: TextBuiltin, args: &[Expr]) -> Option<ValueId> {
        let syms = self.gemm.text;
        let (func, ret, operands) = match b {
            TextBuiltin::ArgCount => (syms.arg_count, MirType::I64, Vec::new()),
            TextBuiltin::Arg => {
                let i = self.text_i64_operand(args.first()?);
                (syms.arg, MirType::Ptr, vec![i])
            }
            TextBuiltin::ArgI64 => {
                let i = self.text_i64_operand(args.first()?);
                let d = self.text_i64_operand(args.get(1)?);
                (syms.arg_i64, MirType::I64, vec![i, d])
            }
            TextBuiltin::ArgF32 => {
                let i = self.text_i64_operand(args.first()?);
                let e = args.get(1)?;
                let from = self.expr_mir(e);
                let signed = self.signed(e);
                let v = self.lower_expr(e);
                let d = self.coerce_to(v, &from, &MirType::F32, signed);
                (syms.arg_f32, MirType::F32, vec![i, d])
            }
            TextBuiltin::Env => {
                let name = self.lower_expr(args.first()?);
                (syms.env, MirType::Ptr, vec![name])
            }
            TextBuiltin::ReadStdin => {
                let (data, cap) = self.text_u8_buffer(args.first()?)?;
                (syms.read_stdin, MirType::I64, vec![data, cap])
            }
            TextBuiltin::Exit => {
                let code = self.text_i64_operand(args.first()?);
                self.builder.build_void(Op::Call {
                    func: syms.exit,
                    args: vec![code],
                });
                return Some(self.const_zero(MirType::I32));
            }
            _ => return None,
        };
        Some(self.builder.build(
            ret,
            Op::Call {
                func,
                args: operands,
            },
        ))
    }

    /// An integer operand widened to `i64` by its own signedness.
    fn text_i64_operand(&mut self, e: &Expr) -> ValueId {
        let from = self.expr_mir(e);
        let signed = self.signed(e);
        let v = self.lower_expr(e);
        self.coerce_to(v, &from, &MirType::I64, signed)
    }

    fn text_const(&mut self, v: i64) -> ValueId {
        self.builder
            .build(MirType::I64, Op::ConstInt(i128::from(v), MirType::I64))
    }

    /// A stream sink: stdout, or stderr.
    fn text_stream(&mut self, stderr: bool, span: Span) -> Sink {
        let target = self.text_const(if stderr { 2 } else { 1 });
        let zero = self.text_const(0);
        Sink {
            target,
            buf: zero,
            cap: zero,
            pos: zero,
            buffered: false,
            span,
        }
    }

    /// One `wukong_rt_fmt_*` call: the sink operands, then `rest`. Returns the position after its
    /// output, which a buffer sink threads into its next call.
    fn text_call(&mut self, sink: &mut Sink, func: Symbol, rest: &[ValueId]) -> ValueId {
        let mut args = Vec::with_capacity(4 + rest.len());
        args.extend([sink.target, sink.buf, sink.cap, sink.pos]);
        args.extend_from_slice(rest);
        let end = self.builder.build(MirType::I64, Op::Call { func, args });
        if sink.buffered {
            sink.pos = end;
        }
        end
    }

    /// Literal text the compiler generates (punctuation, a type or field name): eight bytes per
    /// `wukong_rt_fmt_lit` call, packed little-endian into an `i64` constant.
    fn text_lit(&mut self, sink: &mut Sink, bytes: &[u8]) {
        let f = self.gemm.text.fmt_lit;
        for chunk in bytes.chunks(8) {
            let mut word = [0u8; 8];
            word[..chunk.len()].copy_from_slice(chunk);
            let packed = self.text_const(i64::from_le_bytes(word));
            let n = self.text_const(chunk.len() as i64);
            self.text_call(sink, f, &[packed, n]);
        }
    }

    /// Does a value of type `t` live behind its base pointer (the by-pointer aggregate convention):
    /// an array, a tuple, a slice's fat pointer, a struct or a data-carrying enum?
    fn text_by_pointer(&self, t: &Ty) -> bool {
        match t {
            Ty::Array { .. } | Ty::Tuple(_) | Ty::Slice(_) | Ty::Str => true,
            Ty::Named(n) => {
                matches!(
                    self.sema.defs.lookup(*n).map(|d| &d.kind),
                    Some(DefKind::Struct(_))
                ) || self.enum_is_data_carrying(*n)
            }
            _ => false,
        }
    }

    /// The data pointer and `i64` element count of a `[]u8` / `[u8; N]` operand.
    pub(crate) fn text_u8_buffer(&mut self, e: &Expr) -> Option<(ValueId, ValueId)> {
        match self.expr_ty(e) {
            Ty::Slice(_) => {
                let s = self.lower_expr(e);
                Some(self.text_slice_parts(s))
            }
            Ty::Array { len, .. } => {
                let n = i64::try_from(len).ok()?;
                let base = self.lower_expr(e);
                let n = self.text_const(n);
                Some((base, n))
            }
            _ => None,
        }
    }

    /// A slice's data pointer and `i64` length, read from its fat pointer.
    pub(crate) fn text_slice_parts(&mut self, s: ValueId) -> (ValueId, ValueId) {
        let dp = self.field_ptr(s, SLICE_PTR_OFF);
        let data = self.builder.build(MirType::Ptr, Op::Load(dp, MirType::Ptr));
        let lp = self.field_ptr(s, SLICE_LEN_OFF);
        let len = self.builder.build(MirType::I64, Op::Load(lp, MirType::I64));
        (data, len)
    }

    /// The byte count operand `n` of a raw write, clamped to `[0, len]`: a negative count writes
    /// nothing and a count past the buffer writes the whole buffer, never beyond it.
    fn text_clamped_count(&mut self, n: &Expr, len: ValueId) -> ValueId {
        let v = self.text_i64_operand(n);
        let zero = self.text_const(0);
        let neg = self
            .builder
            .build(MirType::I1, Op::Cmp(CmpOp::Slt, v, zero));
        let v = self.builder.build(MirType::I64, Op::Select(neg, zero, v));
        let over = self.builder.build(MirType::I1, Op::Cmp(CmpOp::Sgt, v, len));
        self.builder.build(MirType::I64, Op::Select(over, len, v))
    }

    /// The pieces of a format string literal and their arguments.
    fn text_format(&mut self, sink: &mut Sink, fmt: &Expr, args: &[Expr]) -> Option<()> {
        let vals = self.text_format_args(args);
        self.text_format_pieces(sink, fmt, &vals)
    }

    /// Every format argument, evaluated in order before anything prints, with the type it prints
    /// at. A `str` argument is flagged here, while its node is at hand: its type alone reads as
    /// `[]u8`.
    fn text_format_args(&mut self, args: &[Expr]) -> Vec<(ValueId, Ty, bool)> {
        let mut vals: Vec<(ValueId, Ty, bool)> = Vec::with_capacity(args.len());
        for a in args {
            let t = self.print_ty(a);
            let is_str = self.is_str_print_arg(a);
            let v = self.lower_expr(a);
            vals.push((v, t, is_str));
        }
        vals
    }

    /// The pieces of the format string literal `fmt`, over the evaluated arguments `vals`.
    fn text_format_pieces(
        &mut self,
        sink: &mut Sink,
        fmt: &Expr,
        vals: &[(ValueId, Ty, bool)],
    ) -> Option<()> {
        let ExprKind::Str(sym) = &fmt.kind else {
            return None;
        };
        let bytes = crate::decode_string_literal(self.interner.resolve(*sym));
        let pieces = st::parse_format(&bytes).ok()?;
        let base = self.lower_expr(fmt);
        let f = self.gemm.text.fmt_str;
        let mut next = vals.iter();
        for piece in &pieces {
            match piece {
                FmtPiece::Lit { start, len } => {
                    let off = self.text_const(i64::try_from(*start).ok()?);
                    let p = self.builder.build(
                        MirType::Ptr,
                        Op::Gep {
                            ptr: base,
                            index: off,
                            elem: MirType::I8,
                        },
                    );
                    let n = self.text_const(i64::try_from(*len).ok()?);
                    let spec = self.text_const(0);
                    self.text_call(sink, f, &[p, n, spec]);
                }
                FmtPiece::Arg(spec) => {
                    let (v, t, is_str) = next.next()?;
                    if *is_str {
                        self.text_str(sink, *v, spec);
                    } else {
                        self.text_value(sink, *v, t, spec);
                    }
                }
            }
        }
        Some(())
    }

    /// Whether sema recorded the print or format argument `arg` as a `str`.
    fn is_str_print_arg(&self, arg: &Expr) -> bool {
        self.sema.str_print_args.contains(&arg.id)
    }

    /// The type a print or format argument prints at: its unerased type when it holds a `str`
    /// inside an aggregate or behind a reference (`StrInfo::print_types`), else its own. Only the
    /// printer reads it; a `str` and the `[]u8` it erases to have one layout.
    fn print_ty(&self, arg: &Expr) -> Ty {
        match self.sema.str_info.print_types.get(&arg.id) {
            Some(t) => t.clone(),
            None => self.expr_ty(arg),
        }
    }

    /// The text of the `str` whose fat pointer is `v`, per `spec` — the `*u8` string's printer,
    /// with the byte length from the fat pointer in place of the NUL scan.
    fn text_str(&mut self, sink: &mut Sink, v: ValueId, spec: &FmtSpec) {
        let (data, len) = self.text_slice_parts(v);
        let sp = self.text_const(spec.pack(st::KIND_SIGNED));
        let f = self.gemm.text.fmt_str;
        self.text_call(sink, f, &[data, len, sp]);
    }

    /// Print the value `v` of semantic type `ty` per `spec`: a reference prints its referent, an
    /// aggregate prints structurally, everything else is one scalar.
    fn text_value(&mut self, sink: &mut Sink, v: ValueId, ty: &Ty, spec: &FmtSpec) {
        match ty {
            Ty::Ref { pointee, .. } => {
                let inner = if self.text_by_pointer(pointee) {
                    v
                } else {
                    let mty = self.mir_ty_of(pointee);
                    self.builder.build(mty.clone(), Op::Load(v, mty))
                };
                self.text_value(sink, inner, pointee, spec);
            }
            Ty::Array { elem, len, .. } => self.text_array(sink, v, elem, *len, spec),
            // A `str` inside an aggregate (the unerased print type): its text.
            Ty::Str => self.text_str(sink, v, spec),
            Ty::Slice(elem) => {
                let (data, len) = self.text_slice_parts(v);
                match elem.as_ref() {
                    Ty::Scalar(s) => self.text_buffer(sink, data, len, *s, spec),
                    _ => self.unsupported(sink.span, "printing a slice of non-scalars"),
                }
            }
            Ty::Tuple(fields) => {
                // A `str` field is laid out as the `[]u8` it erases to.
                let erased: Vec<Ty> = fields.iter().map(Ty::erase_str).collect();
                let Some((offsets, _, _)) = self.aggregate_layout(&erased) else {
                    self.unsupported(sink.span, "printing an unsized tuple");
                    return;
                };
                let espec = element_spec(spec);
                self.text_lit(sink, b"(");
                for (i, (fty, off)) in fields.iter().zip(offsets).enumerate() {
                    if i > 0 {
                        self.text_lit(sink, b", ");
                    }
                    let p = self.field_ptr(v, off);
                    self.text_field(sink, p, fty, &espec);
                }
                if fields.len() == 1 {
                    self.text_lit(sink, b",");
                }
                self.text_lit(sink, b")");
            }
            Ty::Named(n) if self.text_by_pointer(ty) => self.text_struct(sink, v, *n, spec),
            _ => self.text_scalar(sink, v, ty, spec),
        }
    }

    /// One field or element at `p`: an aggregate is printed from its place, a scalar loaded first.
    fn text_field(&mut self, sink: &mut Sink, p: ValueId, fty: &Ty, spec: &FmtSpec) {
        if self.text_by_pointer(fty) {
            self.text_value(sink, p, fty, spec);
        } else {
            let mty = self.mir_ty_of(fty);
            let v = self.builder.build(mty.clone(), Op::Load(p, mty));
            self.text_value(sink, v, fty, spec);
        }
    }

    /// A struct: `Name { field: value, … }`, or just `Name` when it has no fields.
    fn text_struct(&mut self, sink: &mut Sink, v: ValueId, name: Symbol, spec: &FmtSpec) {
        let interner = self.interner;
        let names: Option<Vec<String>> = match self.sema.defs.lookup(name).map(|d| &d.kind) {
            Some(DefKind::Struct(fields)) => Some(
                fields
                    .iter()
                    .map(|(f, _)| interner.resolve(*f).to_string())
                    .collect(),
            ),
            _ => None,
        };
        // A data-carrying enum (sema rejects it, E0448) or an unsized struct.
        let (Some(names), Some(layout)) = (names, self.struct_field_tys(name)) else {
            self.unsupported(sink.span, "printing a value of this type");
            return;
        };
        // A struct with a `str` field: print each field at its unerased type (`strs.rs`).
        let layout: Vec<(u64, Ty)> = match self.sema.str_info.struct_fields.get(&name) {
            Some(tys) if tys.len() == layout.len() => layout
                .into_iter()
                .zip(tys)
                .map(|((off, _), t)| (off, t.clone()))
                .collect(),
            _ => layout,
        };
        let tname = short_name(self.interner.resolve(name)).to_string();
        self.text_lit(sink, tname.as_bytes());
        if names.is_empty() {
            return;
        }
        let espec = element_spec(spec);
        self.text_lit(sink, b" { ");
        for (i, (fname, (off, fty))) in names.iter().zip(layout).enumerate() {
            if i > 0 {
                self.text_lit(sink, b", ");
            }
            self.text_lit(sink, format!("{fname}: ").as_bytes());
            let p = self.field_ptr(v, off);
            self.text_field(sink, p, &fty, &espec);
        }
        self.text_lit(sink, b" }");
    }

    /// An array: one runtime call over the storage of a run of scalars, else unrolled — the
    /// elements summarized past `SUMMARIZE_THRESHOLD` exactly as the runtime summarizes a buffer.
    fn text_array(&mut self, sink: &mut Sink, base: ValueId, elem: &Ty, len: u64, spec: &FmtSpec) {
        if let Ty::Scalar(s) = elem {
            let n = self.text_const(i64::try_from(len).unwrap_or(i64::MAX));
            self.text_buffer(sink, base, n, *s, spec);
            return;
        }
        let espec = element_spec(spec);
        let emty = self.mir_ty_of(&elem.erase_str());
        let summarize = len > st::SUMMARIZE_THRESHOLD;
        let indices: Vec<u64> = if summarize {
            (0..st::EDGE_ITEMS)
                .chain(len - st::EDGE_ITEMS..len)
                .collect()
        } else {
            (0..len).collect()
        };
        self.text_lit(sink, b"[");
        for (k, &i) in indices.iter().enumerate() {
            if k > 0 {
                self.text_lit(sink, b", ");
            }
            if summarize && k as u64 == st::EDGE_ITEMS {
                self.text_lit(sink, b"..., ");
            }
            let index = self.text_const(i64::try_from(i).unwrap_or(i64::MAX));
            let p = self.builder.build(
                MirType::Ptr,
                Op::Gep {
                    ptr: base,
                    index,
                    elem: emty.clone(),
                },
            );
            self.text_field(sink, p, elem, &espec);
        }
        self.text_lit(sink, b"]");
    }

    /// `len` scalar elements of type `s` at `data`: one `wukong_rt_fmt_slice` call.
    fn text_buffer(
        &mut self,
        sink: &mut Sink,
        data: ValueId,
        len: ValueId,
        s: Scalar,
        spec: &FmtSpec,
    ) {
        let dt = self.text_const(st::print_dtype(s));
        let sp = self.text_const(spec.pack(st::KIND_SIGNED));
        let f = self.gemm.text.fmt_slice;
        self.text_call(sink, f, &[data, len, dt, sp]);
    }

    /// One scalar: an integer, a float, a `bool`, a `char`, a C-style enum or a `*u8` string.
    fn text_scalar(&mut self, sink: &mut Sink, v: ValueId, ty: &Ty, spec: &FmtSpec) {
        let syms = self.gemm.text;
        let from = self.mir_ty_of(ty);
        match ty {
            Ty::Scalar(Scalar::F64) => {
                let x = self.coerce_to(v, &from, &MirType::F64, true);
                let sp = self.text_const(spec.pack(st::KIND_SIGNED));
                self.text_call(sink, syms.fmt_f64, &[x, sp]);
            }
            Ty::Scalar(s) if s.is_float() => {
                let x = self.coerce_to(v, &from, &MirType::F32, true);
                let sp = self.text_const(spec.pack(st::KIND_SIGNED));
                self.text_call(sink, syms.fmt_f32, &[x, sp]);
            }
            // A 128-bit integer: its whole value, through the 128-bit entry (two `i64` halves on
            // the native call). A radix style prints a signed value's bits, as below.
            Ty::Scalar(s @ (Scalar::I128 | Scalar::U128)) => {
                let kind = if *s == Scalar::I128 && !spec.is_radix() {
                    st::KIND_SIGNED
                } else {
                    st::KIND_UNSIGNED
                };
                let sp = self.text_const(spec.pack(kind));
                self.text_call(sink, syms.fmt_i128, &[v, sp]);
            }
            Ty::Scalar(s) => {
                // A signed integer under a radix style prints its own width's bits, as Rust's
                // `{:x}` of `-1i8` prints `ff`: zero-extend it and print it unsigned.
                let (kind, signed) = match s {
                    Scalar::Bool => (st::KIND_BOOL, false),
                    Scalar::Char => (st::KIND_CHAR, false),
                    s if s.is_signed() && !spec.is_radix() => (st::KIND_SIGNED, true),
                    _ => (st::KIND_UNSIGNED, false),
                };
                let x = self.coerce_to(v, &from, &MirType::I64, signed);
                let sp = self.text_const(spec.pack(kind));
                self.text_call(sink, syms.fmt_int, &[x, sp]);
            }
            // A C-style enum is its signed `i32` discriminant.
            Ty::Named(n) if !self.enum_is_data_carrying(*n) => {
                let (kind, signed) = if spec.is_radix() {
                    (st::KIND_UNSIGNED, false)
                } else {
                    (st::KIND_SIGNED, true)
                };
                let x = self.coerce_to(v, &from, &MirType::I64, signed);
                let sp = self.text_const(spec.pack(kind));
                self.text_call(sink, syms.fmt_int, &[x, sp]);
            }
            // A `*u8` string, read to its NUL.
            Ty::Ptr { pointee, .. } if matches!(pointee.as_ref(), Ty::Scalar(Scalar::U8)) => {
                let n = self.text_const(-1);
                let sp = self.text_const(spec.pack(st::KIND_SIGNED));
                self.text_call(sink, syms.fmt_str, &[v, n, sp]);
            }
            // A value sema typed leniently (`Unknown`: an unmodeled builtin's result) prints by its
            // MIR type, as the plain `print` path routes it — never a C0001 for an argument that
            // `print` itself accepts.
            Ty::Unknown | Ty::Error => {
                let s = match self.builder.value_type(v) {
                    MirType::F64 => Scalar::F64,
                    MirType::F32 => Scalar::F32,
                    MirType::F16 => Scalar::F16,
                    MirType::BF16 => Scalar::Bf16,
                    MirType::F8E4M3 => Scalar::F8E4M3,
                    MirType::F8E5M2 => Scalar::F8E5M2,
                    MirType::I1 => Scalar::Bool,
                    MirType::I8 => Scalar::I8,
                    MirType::I16 => Scalar::I16,
                    MirType::I32 => Scalar::I32,
                    MirType::I64 => Scalar::I64,
                    MirType::I128 => Scalar::I128,
                    _ => {
                        self.unsupported(sink.span, "printing a value of this type");
                        return;
                    }
                };
                self.text_scalar(sink, v, &Ty::Scalar(s), spec);
            }
            // Sema rejects every other type (E0448); a generic instantiation can still reach here.
            _ => self.unsupported(sink.span, "printing a value of this type"),
        }
    }
}
