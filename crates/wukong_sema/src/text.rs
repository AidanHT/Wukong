//! Typing for the text builtins: what `print` / `println` accept, the formatted-output family
//! `printf` / `eprintf` / `sprintf`, the newline-free and standard-error printers `print_raw` /
//! `eprint` / `eprintln`, and the raw byte writers `write_stdout` / `write_stderr` / `write_text`.
//!
//! | builtin | prints to | result |
//! |---|---|---|
//! | `print(x)` / `println(x)` | stdout, then a newline (both) | — |
//! | `print_raw(x)` | stdout, no newline | — |
//! | `eprint(x)` / `eprintln(x)` | stderr, then a newline (both) | — |
//! | `printf(fmt, args…)` / `eprintf(fmt, args…)` | stdout / stderr, no newline | — |
//! | `sprintf(buf, fmt, args…)` | the `u8` buffer `buf`, NUL-terminated | `i64` length |
//! | `write_stdout(buf, n)` / `write_stderr(buf, n)` | the first `n` bytes of `buf` | `i64` count |
//! | `write_text(path, buf, n)` | the file `path`, created or truncated | `i64` count or `< 0` |
//!
//! And the process interface (`wukong_runtime::process`): `arg_count() -> i64`, `arg(i) -> *u8`
//! (the empty string when out of range; `arg(0)` is the program), `arg_i64(i, default) -> i64`,
//! `arg_f32(i, default) -> f32`, `env(name) -> *u8` (the empty string when unset),
//! `read_stdin_u8(buf) -> i64` (standard input until the buffer is full or input ends) and
//! `exit(code)` (end the program with `code`, as returning it from `main` would).
//!
//! **What prints (E0448).** A scalar (integer, float, `bool`, `char`), a `*u8` string, a C-style
//! enum (its discriminant), and — rendered structurally, like Rust's `{:?}` — an array, a slice of
//! scalars, a tuple or a struct whose every element prints; a reference prints what it refers to.
//! Anything else is `E0448`: a raw pointer other than `*u8` (an address, which the interpreter and
//! a native run cannot agree on), a reference or a slice of non-scalars inside an aggregate, a
//! data-carrying enum, a SIMD vector, a function value. A whole tensor and `()` are `E0437` under
//! `print`/`println`.
//!
//! **Format strings (E0449, E0450).** The format is a string literal, parsed at compile time:
//! `{}` / `{:spec}` placeholders take the arguments in order, `{{` / `}}` print a brace, and a spec
//! is Rust's `[[fill]align][+][#][0][width][.precision][type]` with `type` one of `?`, `x`, `X`,
//! `o`, `b`, `e`, `E` (width and precision at most [`MAX_FMT_WIDTH`]). A malformed format string is
//! `E0449`; a placeholder count that differs from the argument count, or a spec that does not apply
//! to its argument's type (`{:x}` of a float, `{:.2}` of an integer), is `E0450`.
//!
//! A user-defined function of the same name shadows every builtin here (the resolved-callee path
//! in `type_call` runs first).

use wukong_ast::{Expr, ExprKind};
use wukong_span::Span;
use wukong_types::{Scalar, Ty};

use crate::{DefKind, Sema};

// ---- String literals ---------------------------------------------------------------------------

/// Decode a string literal's raw source text (including the surrounding `"`) into its UTF-8 bytes,
/// resolving the escapes `\n` `\r` `\t` `\\` `\'` `\"` `\0`, `\xHH` and `\u{…}`; each decoded code
/// point is re-encoded as UTF-8 (an invalid one is kept as its low byte). The caller appends a NUL
/// terminator if it needs one. The one decoder: `wukong_mir_build` materializes a literal's bytes
/// with it, and the format-string checks below parse the same bytes the program will print.
pub fn decode_string_literal(text: &str) -> Vec<u8> {
    let inner = text
        .strip_prefix('"')
        .and_then(|t| t.strip_suffix('"'))
        .unwrap_or(text);
    let mut out = Vec::new();
    let mut buf = [0u8; 4];
    let mut chars = inner.chars();
    while let Some(c) = chars.next() {
        let cp = if c == '\\' {
            decode_escape(&mut chars)
        } else {
            c as u32
        };
        match char::from_u32(cp) {
            Some(ch) => out.extend_from_slice(ch.encode_utf8(&mut buf).as_bytes()),
            None => out.push(cp as u8),
        }
    }
    out
}

/// Decode the body of a backslash escape (the `\` already consumed) to a code point.
pub fn decode_escape(chars: &mut std::str::Chars) -> u32 {
    match chars.next() {
        Some('n') => '\n' as u32,
        Some('r') => '\r' as u32,
        Some('t') => '\t' as u32,
        Some('\\') => '\\' as u32,
        Some('\'') => '\'' as u32,
        Some('"') => '"' as u32,
        Some('0') => 0,
        // `\xHH` — up to two hex digits (the lexer consumed at most two).
        Some('x') => chars
            .by_ref()
            .take(2)
            .fold(0u32, |v, c| c.to_digit(16).map_or(v, |d| v * 16 + d)),
        // `\u{HHHH}` — the hex digits between the braces. Saturating, so an over-long escape
        // (`\u{100000000}`, ≥ 9 hex digits) clamps to an invalid code point instead of overflowing
        // the accumulator and panicking the compiler; a valid code point is ≤ 6 hex digits anyway.
        Some('u') => chars
            .by_ref()
            .skip_while(|&c| c != '{')
            .skip(1)
            .take_while(|&c| c != '}')
            .fold(0u32, |v, c| {
                c.to_digit(16)
                    .map_or(v, |d| v.saturating_mul(16).saturating_add(d))
            }),
        Some(c) => c as u32,
        None => 0,
    }
}

// ---- Format strings ----------------------------------------------------------------------------

/// The largest width or precision a format spec may name.
pub const MAX_FMT_WIDTH: u16 = 255;

/// `{}` — Rust's `Display`.
pub const STYLE_DISPLAY: u8 = 0;
/// `{:?}` — Rust's `Debug` (a string or `char` quoted, a float always with a point).
pub const STYLE_DEBUG: u8 = 1;
/// `{:x}`
pub const STYLE_LOWER_HEX: u8 = 2;
/// `{:X}`
pub const STYLE_UPPER_HEX: u8 = 3;
/// `{:o}`
pub const STYLE_OCTAL: u8 = 4;
/// `{:b}`
pub const STYLE_BINARY: u8 = 5;
/// `{:e}`
pub const STYLE_LOWER_EXP: u8 = 6;
/// `{:E}`
pub const STYLE_UPPER_EXP: u8 = 7;

/// No alignment: a number right-aligns, everything else left-aligns.
pub const ALIGN_NONE: u8 = 0;
/// `<`
pub const ALIGN_LEFT: u8 = 1;
/// `^`
pub const ALIGN_CENTER: u8 = 2;
/// `>`
pub const ALIGN_RIGHT: u8 = 3;

/// The register value is a signed integer.
pub const KIND_SIGNED: u8 = 0;
/// The register value is an unsigned integer's bits, zero-extended to 64.
pub const KIND_UNSIGNED: u8 = 1;
/// The register value is a `bool`.
pub const KIND_BOOL: u8 = 2;
/// The register value is a `char`'s code point.
pub const KIND_CHAR: u8 = 3;

/// The runtime dtype code of a printed `bool` buffer (beyond the twelve I/O dtypes of
/// [`crate::io_dtype_code`]); the runtime's `fmt::DT_BOOL`.
pub const PRINT_DT_BOOL: i64 = 12;
/// The runtime dtype code of a printed `char` buffer; the runtime's `fmt::DT_CHAR`.
pub const PRINT_DT_CHAR: i64 = 13;
/// An array or slice longer than this prints summarized, NumPy-style (the runtime's
/// `fmt::SUMMARIZE_THRESHOLD`; the compiler unrolls an array of aggregates by the same rule).
pub const SUMMARIZE_THRESHOLD: u64 = 1000;
/// How many elements a summarized array prints at each end (the runtime's `fmt::EDGE_ITEMS`).
pub const EDGE_ITEMS: u64 = 3;

/// The runtime dtype code of a printed buffer whose element is `s`.
pub fn print_dtype(s: Scalar) -> i64 {
    match s {
        Scalar::Bool => PRINT_DT_BOOL,
        Scalar::Char => PRINT_DT_CHAR,
        Scalar::Usize => crate::io_dtype_code(Scalar::U64),
        Scalar::Isize => crate::io_dtype_code(Scalar::I64),
        s => crate::io_dtype_code(s),
    }
}

/// One parsed `{:spec}` placeholder.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FmtSpec {
    pub width: u16,
    pub precision: Option<u16>,
    /// The ASCII fill byte; 0 is a space.
    pub fill: u8,
    /// One of the `ALIGN_*` codes.
    pub align: u8,
    pub plus: bool,
    pub alt: bool,
    pub zero: bool,
    /// One of the `STYLE_*` codes.
    pub style: u8,
}

impl FmtSpec {
    /// Pack this spec, with the integer register `kind`, into the `i64` the runtime's
    /// `wukong_runtime::fmt::Spec::unpack` reads: width in bits 0..16, precision + 1 in 16..32, the
    /// fill byte in 32..40, the alignment in 40..42, `+` / `#` / `0` at bits 42 / 43 / 44, the style
    /// in 48..52 and the kind in 52..56. (This crate does not depend on the runtime; a test in
    /// `wukong_interp`, which sees both, pins the two layouts.)
    pub fn pack(&self, kind: u8) -> i64 {
        let prec = self.precision.map_or(0, |p| i64::from(p) + 1);
        let mut bits = i64::from(self.width)
            | (prec << 16)
            | (i64::from(self.fill) << 32)
            | (i64::from(self.align & 0x3) << 40)
            | (i64::from(self.style & 0xF) << 48)
            | (i64::from(kind & 0xF) << 52);
        if self.plus {
            bits |= 1 << 42;
        }
        if self.alt {
            bits |= 1 << 43;
        }
        if self.zero {
            bits |= 1 << 44;
        }
        bits
    }

    /// `x`, `X`, `o` or `b`: the integer's bits in a radix.
    pub fn is_radix(&self) -> bool {
        matches!(
            self.style,
            STYLE_LOWER_HEX | STYLE_UPPER_HEX | STYLE_OCTAL | STYLE_BINARY
        )
    }

    /// `e` or `E`.
    pub fn is_exp(&self) -> bool {
        matches!(self.style, STYLE_LOWER_EXP | STYLE_UPPER_EXP)
    }
}

/// One piece of a parsed format string.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FmtPiece {
    /// Literal text: `len` bytes of the decoded format string starting at byte `start`.
    Lit { start: usize, len: usize },
    /// The next argument, formatted per the spec.
    Arg(FmtSpec),
}

/// Parse a decoded format string into literal runs and placeholders. `Err` names what is wrong.
pub fn parse_format(s: &[u8]) -> Result<Vec<FmtPiece>, String> {
    fn push_lit(pieces: &mut Vec<FmtPiece>, start: usize, end: usize) {
        if end > start {
            pieces.push(FmtPiece::Lit {
                start,
                len: end - start,
            });
        }
    }
    let mut pieces = Vec::new();
    let mut lit = 0usize;
    let mut i = 0usize;
    while i < s.len() {
        match s[i] {
            // An escaped brace: the literal run keeps the first one and skips the second.
            b'{' | b'}' if s.get(i + 1) == Some(&s[i]) => {
                push_lit(&mut pieces, lit, i + 1);
                i += 2;
                lit = i;
            }
            b'{' => {
                push_lit(&mut pieces, lit, i);
                let close = s[i + 1..]
                    .iter()
                    .position(|&c| c == b'}')
                    .ok_or("an unclosed `{` (write `{{` for a literal brace)")?;
                let inner = &s[i + 1..i + 1 + close];
                if inner.contains(&b'{') {
                    return Err(
                        "a `{` inside a placeholder (write `{{` for a literal brace)".into(),
                    );
                }
                pieces.push(FmtPiece::Arg(parse_spec(inner)?));
                i += close + 2;
                lit = i;
            }
            b'}' => return Err("an unmatched `}` (write `}}` for a literal brace)".into()),
            _ => i += 1,
        }
    }
    push_lit(&mut pieces, lit, s.len());
    Ok(pieces)
}

/// The leading decimal digits of `p` as a width or precision, and how many bytes they took.
fn fmt_number(p: &[u8]) -> Result<(u16, usize), String> {
    let n = p.iter().take_while(|c| c.is_ascii_digit()).count();
    if n == 0 {
        return Ok((0, 0));
    }
    let value = p[..n]
        .iter()
        .try_fold(0u32, |v, &d| {
            v.checked_mul(10)?.checked_add(u32::from(d - b'0'))
        })
        .filter(|&v| v <= u32::from(MAX_FMT_WIDTH))
        .ok_or_else(|| format!("a width or precision above {MAX_FMT_WIDTH}"))?;
    Ok((value as u16, n))
}

/// Parse the inside of one `{…}` placeholder.
fn parse_spec(inner: &[u8]) -> Result<FmtSpec, String> {
    let mut spec = FmtSpec::default();
    if inner.is_empty() {
        return Ok(spec);
    }
    let Some(mut p) = inner.strip_prefix(b":") else {
        return Err(format!(
            "the placeholder `{{{}}}` names an argument; write `{{}}` or `{{:spec}}` and pass the \
             arguments in order",
            String::from_utf8_lossy(inner)
        ));
    };
    let align_of = |c: u8| match c {
        b'<' => Some(ALIGN_LEFT),
        b'^' => Some(ALIGN_CENTER),
        b'>' => Some(ALIGN_RIGHT),
        _ => None,
    };
    if p.first().is_some_and(|c| !c.is_ascii()) {
        return Err("a non-ASCII fill character (the fill must be ASCII)".into());
    }
    if let Some(a) = p.get(1).and_then(|&c| align_of(c)) {
        if p[0].is_ascii_control() {
            return Err("a control character as the fill".into());
        }
        spec.fill = p[0];
        spec.align = a;
        p = &p[2..];
    } else if let Some(a) = p.first().and_then(|&c| align_of(c)) {
        spec.align = a;
        p = &p[1..];
    }
    if let Some(rest) = p.strip_prefix(b"+") {
        spec.plus = true;
        p = rest;
    }
    if p.first() == Some(&b'-') {
        return Err("the `-` flag (a sign is printed only when negative unless `+` asks)".into());
    }
    if let Some(rest) = p.strip_prefix(b"#") {
        spec.alt = true;
        p = rest;
    }
    if let Some(rest) = p.strip_prefix(b"0") {
        spec.zero = true;
        p = rest;
    }
    let (width, used) = fmt_number(p)?;
    spec.width = width;
    p = &p[used..];
    if let Some(rest) = p.strip_prefix(b".") {
        let (precision, used) = fmt_number(rest)?;
        if used == 0 {
            return Err("a `.` with no precision after it (write `{:.3}`)".into());
        }
        spec.precision = Some(precision);
        p = &rest[used..];
    }
    spec.style = match p {
        b"" => STYLE_DISPLAY,
        b"?" => STYLE_DEBUG,
        b"x" => STYLE_LOWER_HEX,
        b"X" => STYLE_UPPER_HEX,
        b"o" => STYLE_OCTAL,
        b"b" => STYLE_BINARY,
        b"e" => STYLE_LOWER_EXP,
        b"E" => STYLE_UPPER_EXP,
        other => {
            return Err(format!(
                "the format type `{}` (expected one of `?`, `x`, `X`, `o`, `b`, `e`, `E`)",
                String::from_utf8_lossy(other)
            ))
        }
    };
    Ok(spec)
}

// ---- The builtins ------------------------------------------------------------------------------

/// One text builtin (see the module docs).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TextBuiltin {
    /// `printf(fmt, args…)` (stdout) / `eprintf(fmt, args…)` (stderr).
    Printf { stderr: bool },
    /// `sprintf(buf, fmt, args…) -> i64`.
    Sprintf,
    /// `print_raw(x)` (stdout, no newline) / `eprint(x)` and `eprintln(x)` (stderr, newline).
    PrintValue { stderr: bool, newline: bool },
    /// `write_stdout(buf, n) -> i64` / `write_stderr(buf, n) -> i64`.
    WriteBytes { stderr: bool },
    /// `write_text(path, buf, n) -> i64`.
    WriteText,
    /// `arg_count() -> i64`.
    ArgCount,
    /// `arg(i) -> *u8`.
    Arg,
    /// `arg_i64(i, default) -> i64`.
    ArgI64,
    /// `arg_f32(i, default) -> f32`.
    ArgF32,
    /// `env(name) -> *u8`.
    Env,
    /// `exit(code)`.
    Exit,
    /// `read_stdin_u8(buf) -> i64`.
    ReadStdin,
}

/// The text builtin `name` denotes, or `None`. (`print` / `println` are the original intrinsics,
/// checked by [`Sema::check_print_args`].)
pub fn text_builtin(name: &str) -> Option<TextBuiltin> {
    Some(match name {
        "printf" => TextBuiltin::Printf { stderr: false },
        "eprintf" => TextBuiltin::Printf { stderr: true },
        "sprintf" => TextBuiltin::Sprintf,
        "print_raw" => TextBuiltin::PrintValue {
            stderr: false,
            newline: false,
        },
        "eprint" | "eprintln" => TextBuiltin::PrintValue {
            stderr: true,
            newline: true,
        },
        "write_stdout" => TextBuiltin::WriteBytes { stderr: false },
        "write_stderr" => TextBuiltin::WriteBytes { stderr: true },
        "write_text" => TextBuiltin::WriteText,
        "arg_count" => TextBuiltin::ArgCount,
        "arg" => TextBuiltin::Arg,
        "arg_i64" => TextBuiltin::ArgI64,
        "arg_f32" => TextBuiltin::ArgF32,
        "env" => TextBuiltin::Env,
        "exit" => TextBuiltin::Exit,
        "read_stdin_u8" => TextBuiltin::ReadStdin,
        _ => return None,
    })
}

/// How a printed value is classified for the spec check.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Printed {
    Int,
    Float,
    Bool,
    Char,
    Str,
    Aggregate,
    /// A generic parameter or an unknown type: checked per instance.
    Lenient,
}

/// Why `spec` does not apply to a value of class `class`, or `None` when it does.
fn spec_mismatch(spec: &FmtSpec, class: Printed) -> Option<&'static str> {
    use Printed::*;
    if class == Lenient {
        return None;
    }
    if spec.is_radix() && class != Int {
        return Some("the `x`, `X`, `o` and `b` styles format an integer");
    }
    if spec.is_exp() && !matches!(class, Int | Float) {
        return Some("the `e` and `E` styles format a number");
    }
    if spec.alt && !spec.is_radix() {
        return Some("`#` adds the `0x` / `0o` / `0b` prefix of the `x`, `X`, `o` and `b` styles");
    }
    if spec.precision.is_some() && matches!(class, Int | Bool | Char) {
        return Some(
            "a precision applies to a float (its digits after the point), a string (its maximum \
             length) or the elements of an aggregate",
        );
    }
    if (spec.plus || spec.zero) && matches!(class, Bool | Char | Str) {
        return Some("the `+` and `0` flags format a number");
    }
    None
}

/// The spellings of a spec, for a diagnostic: `{:x}`.
fn spec_text(spec: &FmtSpec) -> String {
    let mut s = String::from("{:");
    if spec.align != ALIGN_NONE {
        if spec.fill != 0 {
            s.push(char::from(spec.fill));
        }
        s.push(match spec.align {
            ALIGN_LEFT => '<',
            ALIGN_CENTER => '^',
            _ => '>',
        });
    }
    if spec.plus {
        s.push('+');
    }
    if spec.alt {
        s.push('#');
    }
    if spec.zero {
        s.push('0');
    }
    if spec.width != 0 {
        s.push_str(&spec.width.to_string());
    }
    if let Some(p) = spec.precision {
        s.push('.');
        s.push_str(&p.to_string());
    }
    s.push_str(match spec.style {
        STYLE_DEBUG => "?",
        STYLE_LOWER_HEX => "x",
        STYLE_UPPER_HEX => "X",
        STYLE_OCTAL => "o",
        STYLE_BINARY => "b",
        STYLE_LOWER_EXP => "e",
        STYLE_UPPER_EXP => "E",
        _ => "",
    });
    s.push('}');
    s
}

impl Sema<'_> {
    /// Why a value of type `t` cannot be printed, or `None` when it can (see the module docs).
    /// `top` is the printed value itself, where a reference prints what it refers to.
    fn unprintable(&self, t: &Ty, top: bool) -> Option<String> {
        match t {
            Ty::Scalar(_) | Ty::Unknown | Ty::Error => None,
            Ty::Ptr { pointee, .. } if matches!(pointee.as_ref(), Ty::Scalar(Scalar::U8)) => None,
            Ty::Ptr { .. } => Some(
                "a raw pointer is an address, not a value; dereference it (`*p`) to print what \
                 it points at"
                    .into(),
            ),
            Ty::Ref { pointee, .. } if top => self.unprintable(pointee, false),
            Ty::Ref { .. } => Some(
                "a reference inside an aggregate does not print; print the value it refers to"
                    .into(),
            ),
            Ty::Array { elem, .. } => self.unprintable(elem, false),
            Ty::Slice(elem) => match elem.as_ref() {
                Ty::Scalar(_) => None,
                _ => Some(
                    "only a slice of scalars (numbers, `bool`, `char`) prints; print the elements \
                     in a loop"
                        .into(),
                ),
            },
            Ty::Tuple(ts) => ts.iter().find_map(|t| self.unprintable(t, false)),
            Ty::Named(n) => match self.defs.lookup(*n).map(|d| &d.kind) {
                Some(DefKind::Struct(fields)) => {
                    fields.iter().find_map(|(_, t)| self.unprintable(t, false))
                }
                Some(DefKind::Enum(_)) if self.enum_is_data_carrying(*n) => Some(format!(
                    "a data-carrying enum (`{}`) prints by matching on it and printing the \
                     variant's fields",
                    self.sym_str(*n)
                )),
                // A C-style enum prints its discriminant; a generic parameter prints per instance.
                _ => None,
            },
            Ty::Tensor { .. } => {
                Some("a whole tensor prints element by element; index every axis in a loop".into())
            }
            Ty::Vector { .. } => {
                Some("a SIMD vector prints lane by lane; extract each lane to print it".into())
            }
            Ty::Fn { .. } => Some("a function value has nothing to print".into()),
            Ty::Unit => Some("`()` has no value to print".into()),
        }
    }

    /// The class of a printed value for the spec check.
    fn printed_class(&self, t: &Ty) -> Printed {
        match t {
            Ty::Scalar(s) if s.is_float() => Printed::Float,
            Ty::Scalar(Scalar::Bool) => Printed::Bool,
            Ty::Scalar(Scalar::Char) => Printed::Char,
            Ty::Scalar(_) => Printed::Int,
            Ty::Ptr { .. } => Printed::Str,
            Ty::Ref { pointee, .. } => self.printed_class(pointee),
            Ty::Array { .. } | Ty::Slice(_) | Ty::Tuple(_) => Printed::Aggregate,
            Ty::Named(n) => match self.defs.lookup(*n).map(|d| &d.kind) {
                Some(DefKind::Struct(_)) => Printed::Aggregate,
                Some(DefKind::Enum(_)) => Printed::Int,
                _ => Printed::Lenient,
            },
            _ => Printed::Lenient,
        }
    }

    /// Report `E0448` at `arg` when `ty` cannot be printed by `name`; `true` when it can.
    fn check_printable(&mut self, name: &str, arg: &Expr, ty: &Ty) -> bool {
        let Some(why) = self.unprintable(ty, true) else {
            return true;
        };
        self.error(
            arg.span,
            "E0448",
            format!(
                "`{name}` cannot print a value of type `{}`: {why}",
                ty.display(self.interner)
            ),
        );
        false
    }

    /// `print` / `println` and the other one-value printers: at most one argument, which must
    /// print. A whole tensor and `()` are `E0437` (an intrinsic argument of the wrong kind); anything else that
    /// cannot print is `E0448`. Returns whether the call is well-formed.
    pub(crate) fn check_print_args(
        &mut self,
        name: &str,
        args: &[Expr],
        arg_tys: &[Ty],
        span: Span,
    ) -> bool {
        if args.len() > 1 {
            self.error(
                span,
                "E0503",
                format!(
                    "`{name}` takes a single value to print, but {} were supplied",
                    args.len()
                ),
            );
            return false;
        }
        let (Some(arg), Some(t)) = (args.first(), arg_tys.first()) else {
            return true;
        };
        // A `()`/unit argument has no printable value — e.g. `print(x)` where `x = if c { 1 }` (an
        // else-less `if` is unit) or `x = match … { … }` with statement arms. mir_build lowered the
        // unit `let` to an `alloca void` and the read to a `load void`: native -O0 read
        // uninitialized stack (nondeterministic garbage), while the interpreter and -O2 (mem2reg)
        // produced 0 — a divergence that broke BOTH invariants at once. Rejected (E0437), as `()`
        // is already rejected as an arithmetic operand.
        if let t @ Ty::Tensor { .. } = t {
            let what = format!("`{name}`");
            let msg = self.whole_tensor_message(t, &what);
            self.error(span, "E0437", msg);
            return false;
        }
        if matches!(t, Ty::Unit) {
            self.error(
                span,
                "E0437",
                format!(
                    "`{name}` cannot print a `()` value; supply a printable scalar (an `if` with \
                     no `else`, or a `match` with statement arms, yields `()`)"
                ),
            );
            return false;
        }
        self.check_printable(name, arg, t)
    }

    /// Type a call to a text builtin, or return `None` when `name` is not one (the caller keeps
    /// looking). Diagnoses arity and operand errors itself; the returned type is the builtin's
    /// result type either way, so one bad call does not cascade.
    pub(crate) fn type_text_builtin(
        &mut self,
        name: &str,
        args: &[Expr],
        arg_tys: &mut [Ty],
        span: Span,
    ) -> Option<Ty> {
        let b = text_builtin(name)?;
        let i64_ty = Ty::Scalar(Scalar::I64);
        Some(match b {
            TextBuiltin::Printf { .. } => {
                if args.is_empty() {
                    self.error(
                        span,
                        "E0503",
                        format!("`{name}` takes a format string literal, then its arguments"),
                    );
                } else {
                    self.check_format(name, &args[0], &args[1..], &arg_tys[1..], span);
                }
                Ty::Unit
            }
            TextBuiltin::Sprintf => {
                if args.len() < 2 {
                    self.error(
                        span,
                        "E0503",
                        format!(
                            "`{name}` takes a `u8` buffer and a format string literal, then its \
                             arguments, but {} were supplied",
                            args.len()
                        ),
                    );
                } else {
                    self.check_u8_buffer(name, &args[0], &arg_tys[0], 0, true);
                    self.check_format(name, &args[1], &args[2..], &arg_tys[2..], span);
                }
                i64_ty
            }
            TextBuiltin::PrintValue { .. } => {
                self.check_print_args(name, args, arg_tys, span);
                Ty::Unit
            }
            TextBuiltin::WriteBytes { .. } | TextBuiltin::WriteText => {
                let with_path = b == TextBuiltin::WriteText;
                let want = if with_path { 3 } else { 2 };
                if args.len() != want {
                    let shape = if with_path {
                        "a `*u8` path, a `u8` buffer and an integer byte count"
                    } else {
                        "a `u8` buffer and an integer byte count"
                    };
                    self.error(
                        span,
                        "E0503",
                        format!(
                            "`{name}` takes exactly {want} arguments ({shape}), but {} were \
                             supplied",
                            args.len()
                        ),
                    );
                    return Some(i64_ty);
                }
                let first = usize::from(with_path);
                if with_path {
                    match &arg_tys[0] {
                        Ty::Ptr { pointee, .. }
                            if matches!(pointee.as_ref(), Ty::Scalar(Scalar::U8)) => {}
                        Ty::Unknown | Ty::Error => {}
                        other => self.error(
                            args[0].span,
                            "E0437",
                            format!(
                                "`{name}` expects a `*u8` path as its first argument, found a \
                                 value of type `{}`",
                                other.display(self.interner)
                            ),
                        ),
                    }
                }
                self.check_u8_buffer(name, &args[first], &arg_tys[first], first, false);
                self.check_int_arg(
                    name,
                    &args[first + 1],
                    &mut arg_tys[first + 1],
                    "byte count",
                );
                i64_ty
            }
            TextBuiltin::ArgCount
            | TextBuiltin::Arg
            | TextBuiltin::ArgI64
            | TextBuiltin::ArgF32
            | TextBuiltin::Env
            | TextBuiltin::Exit
            | TextBuiltin::ReadStdin => self.type_process_builtin(b, name, args, arg_tys, span),
        })
    }

    /// The process interface: `arg_count`, `arg`, `arg_i64`, `arg_f32`, `env`, `exit`,
    /// `read_stdin_u8` (see the module docs).
    fn type_process_builtin(
        &mut self,
        b: TextBuiltin,
        name: &str,
        args: &[Expr],
        arg_tys: &mut [Ty],
        span: Span,
    ) -> Ty {
        let str_ty = Ty::Ptr {
            mutable: false,
            pointee: Box::new(Ty::Scalar(Scalar::U8)),
        };
        let (shape, ret): (&str, Ty) = match b {
            TextBuiltin::ArgCount => ("no arguments", Ty::Scalar(Scalar::I64)),
            TextBuiltin::Arg => ("an integer index", str_ty.clone()),
            TextBuiltin::ArgI64 => (
                "an integer index and an integer default",
                Ty::Scalar(Scalar::I64),
            ),
            TextBuiltin::ArgF32 => (
                "an integer index and a float default",
                Ty::Scalar(Scalar::F32),
            ),
            TextBuiltin::Env => ("a `*u8` variable name", str_ty.clone()),
            TextBuiltin::Exit => ("an integer exit code", Ty::Unit),
            _ => ("a `[]u8` or `[u8; N]` buffer", Ty::Scalar(Scalar::I64)),
        };
        let want = match b {
            TextBuiltin::ArgCount => 0,
            TextBuiltin::ArgI64 | TextBuiltin::ArgF32 => 2,
            _ => 1,
        };
        if args.len() != want {
            self.error(
                span,
                "E0503",
                format!(
                    "`{name}` takes {shape}, but {} argument(s) were supplied",
                    args.len()
                ),
            );
            return ret;
        }
        match b {
            TextBuiltin::Arg | TextBuiltin::ArgI64 | TextBuiltin::ArgF32 => {
                self.check_int_arg(name, &args[0], &mut arg_tys[0], "argument index");
                if b == TextBuiltin::ArgI64 {
                    self.check_int_arg(name, &args[1], &mut arg_tys[1], "default");
                } else if b == TextBuiltin::ArgF32 {
                    self.check_float_arg(name, &args[1], &mut arg_tys[1], "default");
                }
            }
            TextBuiltin::Exit => self.check_int_arg(name, &args[0], &mut arg_tys[0], "exit code"),
            TextBuiltin::Env => match &arg_tys[0] {
                Ty::Ptr { pointee, .. } if matches!(pointee.as_ref(), Ty::Scalar(Scalar::U8)) => {}
                Ty::Unknown | Ty::Error => {}
                other => self.error(
                    args[0].span,
                    "E0437",
                    format!(
                        "`{name}` expects a `*u8` variable name, found a value of type `{}`",
                        other.display(self.interner)
                    ),
                ),
            },
            TextBuiltin::ReadStdin => {
                self.check_u8_buffer(name, &args[0], &arg_tys[0], 0, true);
            }
            _ => {}
        }
        ret
    }

    /// A float operand; an unsuffixed literal adapts to `f32`.
    fn check_float_arg(&mut self, name: &str, arg: &Expr, ty: &mut Ty, what: &str) {
        let target = Ty::Scalar(Scalar::F32);
        if self.literal_adapts(&target, arg) {
            self.retype_adapted_literal(arg, &target);
            *ty = target;
        }
        let ok = match ty {
            Ty::Scalar(s) => s.is_float(),
            Ty::Unknown | Ty::Error => true,
            _ => false,
        };
        if !ok {
            self.error(
                arg.span,
                "E0437",
                format!(
                    "`{name}` expects a float {what}, found a value of type `{}`",
                    ty.display(self.interner)
                ),
            );
        }
    }

    /// The format string and arguments of `printf` / `eprintf` / `sprintf`.
    fn check_format(&mut self, name: &str, fmt: &Expr, args: &[Expr], tys: &[Ty], span: Span) {
        let ExprKind::Str(sym) = &fmt.kind else {
            self.error(
                fmt.span,
                "E0449",
                format!(
                    "the format string of `{name}` must be a string literal: it is parsed when \
                     the program is compiled"
                ),
            );
            return;
        };
        let bytes = decode_string_literal(self.sym_str(*sym));
        let pieces = match parse_format(&bytes) {
            Ok(p) => p,
            Err(why) => {
                self.error(
                    fmt.span,
                    "E0449",
                    format!("invalid format string for `{name}`: {why}"),
                );
                return;
            }
        };
        let specs: Vec<FmtSpec> = pieces
            .iter()
            .filter_map(|p| match p {
                FmtPiece::Arg(s) => Some(*s),
                FmtPiece::Lit { .. } => None,
            })
            .collect();
        if specs.len() != args.len() {
            let at = args.get(specs.len()).map_or(span, |a| a.span);
            self.error(
                at,
                "E0450",
                format!(
                    "the format string of `{name}` has {} placeholder(s), but {} argument(s) \
                     were supplied",
                    specs.len(),
                    args.len()
                ),
            );
            return;
        }
        for ((spec, arg), ty) in specs.iter().zip(args).zip(tys) {
            if !self.check_printable(name, arg, ty) {
                continue;
            }
            if let Some(why) = spec_mismatch(spec, self.printed_class(ty)) {
                self.error(
                    arg.span,
                    "E0450",
                    format!(
                        "the placeholder `{}` of `{name}` does not apply to a value of type \
                         `{}`: {why}",
                        spec_text(spec),
                        ty.display(self.interner)
                    ),
                );
            }
        }
    }

    /// A `[]u8` slice or `[u8; N]` array operand (argument `i`). When the builtin writes it, a
    /// non-`mut` by-reference parameter is `E0304`, as `p[i] = …` through it would be.
    fn check_u8_buffer(&mut self, name: &str, arg: &Expr, ty: &Ty, i: usize, written: bool) {
        let u8_ty = Ty::Scalar(Scalar::U8);
        let ok = match ty {
            Ty::Slice(e) | Ty::Array { elem: e, .. } => **e == u8_ty,
            Ty::Unknown | Ty::Error => true,
            _ => false,
        };
        if !ok {
            self.error(
                arg.span,
                "E0437",
                format!(
                    "`{name}` expects a `[]u8` or `[u8; N]` buffer as argument {}, found a value \
                     of type `{}`",
                    i + 1,
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
                            arg.span,
                            "E0304",
                            format!(
                                "`{name}` writes its buffer, but `{nm}` is a non-`mut` parameter \
                                 passed by reference — add `mut` to the parameter \
                                 (`fn …(mut {nm}: …)`) to allow the write"
                            ),
                        );
                    }
                }
            }
        }
    }

    /// An integer operand; an unsuffixed literal adapts to `i64`.
    fn check_int_arg(&mut self, name: &str, arg: &Expr, ty: &mut Ty, what: &str) {
        let target = Ty::Scalar(Scalar::I64);
        if self.literal_adapts(&target, arg) {
            self.retype_adapted_literal(arg, &target);
            self.range_check_int_literal(arg, &target);
            *ty = target;
        }
        let ok = match ty {
            Ty::Scalar(s) => s.is_int(),
            Ty::Unknown | Ty::Error => true,
            _ => false,
        };
        if !ok {
            self.error(
                arg.span,
                "E0437",
                format!(
                    "`{name}` expects an integer {what}, found a value of type `{}`",
                    ty.display(self.interner)
                ),
            );
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

    fn lit(start: usize, len: usize) -> FmtPiece {
        FmtPiece::Lit { start, len }
    }

    #[test]
    fn format_strings_parse_into_literals_and_specs() {
        let d = FmtSpec::default();
        assert_eq!(parse_format(b""), Ok(vec![]));
        assert_eq!(
            parse_format(b"step {} loss {:.4}\n"),
            Ok(vec![
                lit(0, 5),
                FmtPiece::Arg(d),
                lit(7, 6),
                FmtPiece::Arg(FmtSpec {
                    precision: Some(4),
                    ..d
                }),
                lit(18, 1),
            ])
        );
        // `{{` / `}}` keep one brace in the literal run.
        // "{{x}}" decodes to "{x}": the run through the first `{`, then "x}" through the first `}`.
        assert_eq!(parse_format(b"{{x}}"), Ok(vec![lit(0, 1), lit(2, 2)]));
        assert_eq!(
            parse_format(b"{:*^9.2e}"),
            Ok(vec![FmtPiece::Arg(FmtSpec {
                width: 9,
                precision: Some(2),
                fill: b'*',
                align: ALIGN_CENTER,
                style: STYLE_LOWER_EXP,
                ..d
            })])
        );
        assert_eq!(
            parse_format(b"{:+#010x}"),
            Ok(vec![FmtPiece::Arg(FmtSpec {
                width: 10,
                plus: true,
                alt: true,
                zero: true,
                style: STYLE_LOWER_HEX,
                ..d
            })])
        );
        assert_eq!(
            parse_format(b"{:0>4}"),
            Ok(vec![FmtPiece::Arg(FmtSpec {
                width: 4,
                fill: b'0',
                align: ALIGN_RIGHT,
                ..d
            })])
        );
        for bad in [
            &b"{"[..],
            b"}",
            b"a { b",
            b"{0}",
            b"{x}",
            b"{:.}",
            b"{:256}",
            b"{:q}",
            b"{:-5}",
            b"{{{",
        ] {
            assert!(
                parse_format(bad).is_err(),
                "{:?}",
                String::from_utf8_lossy(bad)
            );
        }
    }

    #[test]
    fn spec_packing_matches_the_runtime_layout() {
        let s = FmtSpec {
            width: 7,
            precision: Some(3),
            fill: b'_',
            align: ALIGN_LEFT,
            plus: true,
            alt: false,
            zero: true,
            style: STYLE_UPPER_HEX,
        };
        let bits = s.pack(KIND_UNSIGNED);
        assert_eq!(bits & 0xFFFF, 7);
        assert_eq!((bits >> 16) & 0xFFFF, 4);
        assert_eq!((bits >> 32) & 0xFF, i64::from(b'_'));
        assert_eq!((bits >> 40) & 0x3, i64::from(ALIGN_LEFT));
        assert_eq!(bits & (1 << 42), 1 << 42);
        assert_eq!(bits & (1 << 43), 0);
        assert_eq!(bits & (1 << 44), 1 << 44);
        assert_eq!((bits >> 48) & 0xF, i64::from(STYLE_UPPER_HEX));
        assert_eq!((bits >> 52) & 0xF, i64::from(KIND_UNSIGNED));
        assert_eq!(FmtSpec::default().pack(KIND_SIGNED), 0);
    }

    #[test]
    fn print_accepts_scalars_strings_and_aggregates() {
        // A reference is a `&T` parameter, the one place `&T` is a type of its own: `&x` has the
        // raw-pointer type `*T` (see `unprintable_values_are_e0448`) and `let r: &T = &x` is
        // E0401, so the referents print through `show`.
        let ok = "struct P { x: i32, y: f32 }\n\
                  fn show(r: &P, v: &f64) { print(r); println(v); }\n\
                  fn main() -> i32 { \
                  let p = P { x: 1, y: 2.5 }; print(p); \
                  let a: [i32; 3] = [1, 2, 3]; println(a); \
                  let t = (1, 2.5); print(t); \
                  let s: []f32 = alloc_f32(2); print(s); \
                  let w: f64 = 0.1; show(&p, &w); \
                  print(\"hi\"); print('c'); print(true); \
                  print_raw(1); eprint(2.5); eprintln(\"x\"); \
                  return 0; }";
        assert!(errors(ok).is_empty(), "unexpected: {:?}", errors(ok));
    }

    #[test]
    fn unprintable_values_are_e0448() {
        let cases = [
            "fn main() -> i32 { let x: i32 = 1; let p: *i32 = 0 as *i32; print(p); return x; }",
            // `&x` is a raw pointer (`*T`), not a reference: unannotated, it does not print.
            "struct P { x: i32 } fn main() -> i32 { let p = P { x: 1 }; let r = &p; print(r); \
             return 0; }",
            "enum E { A(i32), B } fn main() -> i32 { let e = E::B; print(e); return 0; }",
            "fn f(s: []*u8) { print(s); } fn main() -> i32 { return 0; }",
            "fn main() -> i32 { let x: i32 = 1; let p: *i32 = 0 as *i32; printf(\"{}\", p); \
             return x; }",
        ];
        for src in cases {
            let got = errors(src);
            assert!(got.first() == Some(&"E0448"), "{src}: {got:?}");
        }
        // `()` and a whole tensor are a builtin argument of the wrong kind (E0437) under print.
        assert_eq!(
            errors("fn main() -> i32 { let x = if true { 1 }; print(x); return 0; }"),
            ["E0437"]
        );
    }

    #[test]
    fn format_calls_are_checked() {
        let ok = "fn main() -> i32 { let mut b: [u8; 16] = [0; 16]; \
                  printf(\"{} {:.2} {:>5} {:#x} {:?} {}\", 1, 2.5, \"s\", 255, 'c', true); \
                  eprintf(\"{{}}\"); \
                  let n: i64 = sprintf(b, \"x={:03}\", 7); \
                  let m: i64 = write_stdout(b, n) + write_stderr(b, 2) + write_text(\"f\", b, n); \
                  return (m + n) as i32; }";
        assert!(errors(ok).is_empty(), "unexpected: {:?}", errors(ok));
        let cases = [
            ("fn main() -> i32 { printf(\"{\", 1); return 0; }", "E0449"),
            (
                "fn main() -> i32 { let f: *u8 = \"{}\"; printf(f, 1); return 0; }",
                "E0449",
            ),
            (
                "fn main() -> i32 { printf(\"{0}\", 1); return 0; }",
                "E0449",
            ),
            (
                "fn main() -> i32 { printf(\"{} {}\", 1); return 0; }",
                "E0450",
            ),
            (
                "fn main() -> i32 { printf(\"{}\", 1, 2); return 0; }",
                "E0450",
            ),
            (
                "fn main() -> i32 { printf(\"{:x}\", 1.5); return 0; }",
                "E0450",
            ),
            (
                "fn main() -> i32 { printf(\"{:.2}\", 3); return 0; }",
                "E0450",
            ),
            (
                "fn main() -> i32 { printf(\"{:+}\", true); return 0; }",
                "E0450",
            ),
            (
                "fn main() -> i32 { printf(\"{:#}\", 3); return 0; }",
                "E0450",
            ),
            ("fn main() -> i32 { printf(); return 0; }", "E0503"),
            (
                "fn main() -> i32 { let b: [i32; 4] = [0; 4]; return sprintf(b, \"x\") as i32; }",
                "E0437",
            ),
            (
                "fn f(b: [u8; 4]) -> i64 { return sprintf(b, \"x\"); } \
                 fn main() -> i32 { return 0; }",
                "E0304",
            ),
            (
                "fn main() -> i32 { let b: [u8; 4] = [0; 4]; return write_stdout(b, 1.5) as i32; }",
                "E0437",
            ),
            (
                "fn main() -> i32 { let b: [u8; 4] = [0; 4]; return write_text(b, 1) as i32; }",
                "E0503",
            ),
        ];
        for (src, code) in cases {
            let got = errors(src);
            assert!(got.first() == Some(&code), "{src}: {got:?}");
        }
    }

    #[test]
    fn process_builtins_are_typed() {
        let ok = "fn main() -> i32 { let n: i64 = arg_count(); let p: *u8 = arg(1); \
                  let k: i64 = arg_i64(2, 10); let x: f32 = arg_f32(3, 0.5); \
                  let h: *u8 = env(\"HOME\"); let mut b: [u8; 8] = [0; 8]; \
                  let r: i64 = read_stdin_u8(b); if n > 9 { exit(2); } \
                  print(p); print(h); return (k + r) as i32 + x as i32; }";
        assert!(errors(ok).is_empty(), "unexpected: {:?}", errors(ok));
        let cases = [
            ("fn main() -> i32 { return arg_count(1) as i32; }", "E0503"),
            ("fn main() -> i32 { print(arg(1.5)); return 0; }", "E0437"),
            (
                "fn main() -> i32 { return arg_f32(1, true) as i32; }",
                "E0437",
            ),
            ("fn main() -> i32 { print(env(3)); return 0; }", "E0437"),
            ("fn main() -> i32 { exit(); return 0; }", "E0503"),
            (
                "fn f(b: [u8; 4]) -> i64 { return read_stdin_u8(b); } \
                 fn main() -> i32 { return 0; }",
                "E0304",
            ),
            // `exit` returns nothing to bind.
            (
                "fn main() -> i32 { let x: i32 = exit(1); return x; }",
                "E0401",
            ),
        ];
        for (src, code) in cases {
            let got = errors(src);
            assert!(got.first() == Some(&code), "{src}: {got:?}");
        }
    }

    #[test]
    fn the_one_decoder_handles_every_escape() {
        assert_eq!(decode_string_literal("\"a\\n\\t\\\\\""), b"a\n\t\\");
        assert_eq!(
            decode_string_literal("\"\\x41\\u{e9}\\0\""),
            "A\u{e9}\0".as_bytes()
        );
    }
}
