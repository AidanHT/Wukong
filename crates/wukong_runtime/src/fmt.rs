//! Text rendering shared by every execution path: the interpreter (the spec), the native JIT and a
//! linked executable all render a printed value through these functions, so the bytes a program
//! prints cannot depend on which backend ran it.
//!
//! **Scalars print at their own type.** A value is rendered the way Rust's `Display` renders the
//! same type:
//!
//! * an `f32` prints the *shortest* decimal digit string that parses back to that `f32` (Rust's
//!   `{}` for `f32`: `0.1f32` prints `0.1`, never the `0.10000000149011612` of the `f64` it widens
//!   to). `f16` and `bf16` values live in `f32` registers on every backend and print through the
//!   same rule, exactly as the `half` crate's `Display` does (it formats the widened `f32`);
//! * an `f64` prints its own shortest round-trip digits;
//! * a `bool` prints `true` / `false`;
//! * a `char` prints its glyph, UTF-8 encoded. A code point that is not a Unicode scalar value (a
//!   surrogate, or above U+10FFFF — reachable through an integer `as char`) prints U+FFFD, the
//!   replacement character, as `String::from_utf8_lossy` would.
//!
//! The C fallback runtime of `wukongc --emit=exe` (`wukong_driver`'s `WUKONG_RT_C`) carries C
//! twins of the scalar printers; its float printer derives the same shortest digits by
//! round-tripping `%e` candidates through `strtof` / `strtod`.
//!
//! **Formatting.** `printf` / `eprintf` / `sprintf` and the structural printing of aggregates
//! lower to the `wukong_rt_fmt_*` entries below, one call per piece of output. Every entry takes
//! the same leading **sink** operands — `(target, buf, cap, pos)` — and returns the position after
//! its output:
//!
//! * `target` [`TARGET_STDOUT`] / [`TARGET_STDERR`] writes to a stream (`buf`/`cap`/`pos` unused);
//! * `target` [`TARGET_BUFFER`] writes into the `cap`-byte buffer `buf` at byte `pos` with
//!   `snprintf` semantics: bytes past `cap - 1` are dropped, the buffer is always NUL-terminated,
//!   and the returned position counts every byte *formatted*, dropped or not — so the final
//!   position is the full length of the text, and `>= cap` reports truncation.
//!
//! How one value renders is fixed by a packed [`Spec`] — the parsed `{:…}` placeholder (fill,
//! alignment, sign, `#`, `0`, width, precision, style) plus the integer's kind (signed, unsigned,
//! `bool`, `char`). The compiler's copy of the packing is `wukong_sema::text::FmtSpec`; a test in
//! `wukong_interp`, which sees both crates, pins the two.
//!
//! The rendering follows Rust's `std::fmt` for the same spec: `{:>8.3}` of an `f32` prints what
//! `format!("{:>8.3}", x)` prints, `{:#x}` of a `u8` prints `0xff`, `{:?}` of a string prints it
//! quoted and escaped. An aggregate (array, slice, tuple, struct) renders like Rust's `{:?}`: its
//! elements in `Debug` style, `[1.0, 2.5]`. A buffer longer than [`SUMMARIZE_THRESHOLD`] elements
//! prints summarized, NumPy-style: the first and last [`EDGE_ITEMS`] around `...`.

use std::borrow::Cow;
use std::fmt::{Debug, Display, LowerExp, UpperExp};
use std::io::Write;

use crate::fileio::{
    DT_BF16, DT_F16, DT_F32, DT_F64, DT_I16, DT_I32, DT_I8, DT_U16, DT_U32, DT_U64, DT_U8,
};

/// The text a `bool` prints: `true` for any non-zero register value, `false` for zero.
pub fn bool_text(v: i64) -> &'static str {
    if v != 0 {
        "true"
    } else {
        "false"
    }
}

/// The character a `char` register value prints: the Unicode scalar value it holds, or U+FFFD
/// when the low 32 bits are not one (a surrogate, or a value above U+10FFFF).
pub fn char_glyph(code: i64) -> char {
    char::from_u32(code as u32).unwrap_or(char::REPLACEMENT_CHARACTER)
}

/// `print(x)` of an `f32` (or an `f16` / `bf16` held in an `f32` register): the shortest
/// round-trip digits of `x` at `f32` precision, then a newline.
pub fn f32_line(x: f32) -> String {
    format!("{x}\n")
}

/// `print(b)` of a `bool` register value, with its newline.
pub fn bool_line(v: i64) -> String {
    format!("{}\n", bool_text(v))
}

/// `print(c)` of a `char` register value, with its newline.
pub fn char_line(code: i64) -> String {
    format!("{}\n", char_glyph(code))
}

// ---- Sinks ---------------------------------------------------------------------------------

/// Sink `target`: the caller's byte buffer (`buf`, `cap`, `pos`).
pub const TARGET_BUFFER: i64 = 0;
/// Sink `target`: standard output.
pub const TARGET_STDOUT: i64 = 1;
/// Sink `target`: standard error.
pub const TARGET_STDERR: i64 = 2;

/// Where a write of `n` formatted bytes at position `pos` lands in a `cap`-byte buffer, with
/// `snprintf` semantics: `count` bytes are copied to `start`, the NUL terminator goes to `nul`, and
/// `end` is the position after the write, counting every formatted byte. One plan serves the
/// native entries (raw bytes) and the interpreter (byte slots), so the two cannot disagree on a
/// truncated write.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BufWrite {
    pub start: usize,
    pub count: usize,
    /// `None` for a buffer with no room at all (`cap <= 0`).
    pub nul: Option<usize>,
    pub end: i64,
}

/// The [`BufWrite`] plan for `n` bytes at `pos` into a `cap`-byte buffer. A negative `pos` is
/// treated as 0. The last byte of the buffer is reserved for the terminating NUL, which lands just
/// after the last byte written (at `cap - 1` once the text no longer fits).
pub fn buffer_write(cap: i64, pos: i64, n: usize) -> BufWrite {
    let pos = pos.max(0);
    let end = pos.saturating_add(i64::try_from(n).unwrap_or(i64::MAX));
    if cap <= 0 {
        return BufWrite {
            start: 0,
            count: 0,
            nul: None,
            end,
        };
    }
    let limit = cap - 1;
    let count = if pos < limit {
        (limit - pos).min(end - pos) as usize
    } else {
        0
    };
    BufWrite {
        start: pos.min(limit) as usize,
        count,
        nul: Some(end.min(limit) as usize),
        end,
    }
}

/// Write `bytes` into the `cap`-byte buffer `buf` at `pos` ([`buffer_write`]'s plan) and return the
/// position after them.
///
/// # Safety
/// When `cap > 0` and `buf` is non-null, `buf` must be valid for writes of `cap` bytes.
pub unsafe fn write_buffer(buf: *mut u8, cap: i64, pos: i64, bytes: &[u8]) -> i64 {
    let w = buffer_write(cap, pos, bytes.len());
    if !buf.is_null() {
        if w.count > 0 {
            // SAFETY: `start + count <= cap - 1` by construction, inside the caller's buffer.
            unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), buf.add(w.start), w.count) };
        }
        if let Some(n) = w.nul {
            // SAFETY: `n <= cap - 1`, inside the caller's buffer.
            unsafe { *buf.add(n) = 0 };
        }
    }
    w.end
}

/// A program's standard error, collected rather than written while it is `Some`: see
/// [`begin_stderr_capture`].
static STDERR_CAPTURE: std::sync::Mutex<Option<Vec<u8>>> = std::sync::Mutex::new(None);

/// From now until [`end_stderr_capture`], collect what programs write to standard error (through
/// [`write_stream`], which the interpreter and the JIT both write it with) instead of writing it.
/// The run's standard output is already captured by the run APIs; `wukongc --repl` captures this
/// stream too, because a session replays earlier inputs and shows only the current input's part.
pub fn begin_stderr_capture() {
    *STDERR_CAPTURE.lock().unwrap_or_else(|e| e.into_inner()) = Some(Vec::new());
}

/// Stop capturing standard error ([`begin_stderr_capture`]) and return what was collected.
pub fn end_stderr_capture() -> Vec<u8> {
    STDERR_CAPTURE
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .take()
        .unwrap_or_default()
}

/// Write `bytes` to the process's standard error ([`TARGET_STDERR`]) or standard output (any other
/// stream target) — the streams of a linked executable. A write error is ignored, as a failed
/// `print` is everywhere else. Standard error goes to the capture buffer instead while one is open
/// ([`begin_stderr_capture`]).
pub fn write_stream(target: i64, bytes: &[u8]) {
    if target == TARGET_STDERR {
        if let Some(captured) = STDERR_CAPTURE
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .as_mut()
        {
            captured.extend_from_slice(bytes);
            return;
        }
        let _ = std::io::stderr().lock().write_all(bytes);
    } else {
        flush_stdout_at_exit();
        let _ = std::io::stdout().lock().write_all(bytes);
    }
}

/// Arrange for Rust's stdout to be flushed when the process exits. A linked executable's entry
/// point is the generated object's C `main`, not Rust's, so nothing runs the standard library's
/// exit-time flush when it returns, and Rust's stdout is line-buffered: a final `printf("done")` or
/// `print_raw(x)` with no newline after it would be lost. The first stdout write registers a C
/// `atexit` handler that flushes. (`wukong_rt_exit` flushes explicitly before it ends the process.)
fn flush_stdout_at_exit() {
    extern "C" fn flush() {
        let _ = std::io::stdout().flush();
    }
    extern "C" {
        fn atexit(callback: extern "C" fn()) -> std::ffi::c_int;
    }
    static REGISTER: std::sync::Once = std::sync::Once::new();
    REGISTER.call_once(|| {
        // SAFETY: `atexit` (C89) takes a plain function pointer with no arguments; `flush` has
        // that ABI and cannot unwind. A failed registration only leaves the old behaviour.
        let _ = unsafe { atexit(flush) };
    });
}

/// Deliver formatted `bytes` to a sink — a stream of the running process, or the caller's buffer —
/// and return the position after them.
///
/// # Safety
/// As [`write_buffer`] when `target` is [`TARGET_BUFFER`].
pub unsafe fn emit(target: i64, buf: *mut u8, cap: i64, pos: i64, bytes: &[u8]) -> i64 {
    if target == TARGET_BUFFER {
        // SAFETY: forwarded from this function's contract.
        unsafe { write_buffer(buf, cap, pos, bytes) }
    } else {
        write_stream(target, bytes);
        pos.saturating_add(i64::try_from(bytes.len()).unwrap_or(i64::MAX))
    }
}

// ---- The format spec -----------------------------------------------------------------------

/// Packed-spec bit layout. `width` is bits 0..16.
const PREC_SHIFT: u32 = 16; // bits 16..32: precision + 1 (0 = none)
const FILL_SHIFT: u32 = 32; // bits 32..40: the ASCII fill byte (0 = space)
const ALIGN_SHIFT: u32 = 40; // bits 40..42
const PLUS_BIT: i64 = 1 << 42;
const ALT_BIT: i64 = 1 << 43;
const ZERO_BIT: i64 = 1 << 44;
const STYLE_SHIFT: u32 = 48; // bits 48..52
const KIND_SHIFT: u32 = 52; // bits 52..56

/// No alignment requested: numbers right-align, everything else left-aligns.
pub const ALIGN_NONE: u8 = 0;
/// `<`
pub const ALIGN_LEFT: u8 = 1;
/// `^`
pub const ALIGN_CENTER: u8 = 2;
/// `>`
pub const ALIGN_RIGHT: u8 = 3;

/// `{}`
pub const STYLE_DISPLAY: u8 = 0;
/// `{:?}`
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

/// An integer register holding a signed value.
pub const KIND_SIGNED: u8 = 0;
/// An integer register holding an unsigned value (its bits, zero-extended to 64).
pub const KIND_UNSIGNED: u8 = 1;
/// A `bool` (0 / 1).
pub const KIND_BOOL: u8 = 2;
/// A `char` (its code point).
pub const KIND_CHAR: u8 = 3;

/// One placeholder's formatting request, unpacked.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Spec {
    /// Minimum width in characters; 0 is none.
    pub width: u16,
    /// `.N`: digits after the point for a float, the maximum character count for a string.
    pub precision: Option<u16>,
    /// The ASCII fill byte; 0 is a space.
    pub fill: u8,
    /// One of the `ALIGN_*` codes.
    pub align: u8,
    /// `+`: always print a number's sign.
    pub plus: bool,
    /// `#`: the `0x` / `0o` / `0b` prefix.
    pub alt: bool,
    /// `0`: pad a number with zeros after its sign and prefix.
    pub zero: bool,
    /// One of the `STYLE_*` codes.
    pub style: u8,
    /// One of the `KIND_*` codes (read by the integer entry only).
    pub kind: u8,
}

impl Spec {
    /// Unpack the `i64` the compiler emitted.
    pub fn unpack(bits: i64) -> Spec {
        let prec = ((bits >> PREC_SHIFT) & 0xFFFF) as u16;
        Spec {
            width: (bits & 0xFFFF) as u16,
            precision: prec.checked_sub(1),
            fill: ((bits >> FILL_SHIFT) & 0xFF) as u8,
            align: ((bits >> ALIGN_SHIFT) & 0x3) as u8,
            plus: bits & PLUS_BIT != 0,
            alt: bits & ALT_BIT != 0,
            zero: bits & ZERO_BIT != 0,
            style: ((bits >> STYLE_SHIFT) & 0xF) as u8,
            kind: ((bits >> KIND_SHIFT) & 0xF) as u8,
        }
    }

    /// Pack into the `i64` the compiler emits — the inverse of [`Spec::unpack`] for every spec
    /// whose precision is below `u16::MAX`.
    pub fn pack(self) -> i64 {
        let prec = self.precision.map_or(0, |p| i64::from(p) + 1);
        let mut bits = i64::from(self.width)
            | (prec << PREC_SHIFT)
            | (i64::from(self.fill) << FILL_SHIFT)
            | (i64::from(self.align & 0x3) << ALIGN_SHIFT)
            | (i64::from(self.style & 0xF) << STYLE_SHIFT)
            | (i64::from(self.kind & 0xF) << KIND_SHIFT);
        if self.plus {
            bits |= PLUS_BIT;
        }
        if self.alt {
            bits |= ALT_BIT;
        }
        if self.zero {
            bits |= ZERO_BIT;
        }
        bits
    }
}

// ---- Rendering -------------------------------------------------------------------------------

/// Append `body` padded to the spec's width with its fill and alignment; `numeric` makes right
/// alignment the default, as for a number.
fn pad_into(out: &mut Vec<u8>, body: &str, spec: &Spec, numeric: bool) {
    let width = usize::from(spec.width);
    let len = body.chars().count();
    if len >= width {
        out.extend_from_slice(body.as_bytes());
        return;
    }
    let pad = width - len;
    let fill = if spec.fill == 0 { b' ' } else { spec.fill };
    let align = match spec.align {
        ALIGN_NONE if numeric => ALIGN_RIGHT,
        ALIGN_NONE => ALIGN_LEFT,
        a => a,
    };
    let (before, after) = match align {
        ALIGN_LEFT => (0, pad),
        ALIGN_CENTER => (pad / 2, pad - pad / 2),
        _ => (pad, 0),
    };
    out.extend(std::iter::repeat_n(fill, before));
    out.extend_from_slice(body.as_bytes());
    out.extend(std::iter::repeat_n(fill, after));
}

/// Append a number: its sign, its radix prefix and its digits, padded per the spec. The `0` flag
/// pads with zeros between the prefix and the digits and overrides fill and alignment, as Rust's
/// sign-aware zero padding does.
fn pad_number(out: &mut Vec<u8>, sign: &str, prefix: &str, digits: &str, spec: &Spec) {
    if spec.zero {
        let len = sign.len() + prefix.len() + digits.chars().count();
        out.extend_from_slice(sign.as_bytes());
        out.extend_from_slice(prefix.as_bytes());
        out.extend(std::iter::repeat_n(
            b'0',
            usize::from(spec.width).saturating_sub(len),
        ));
        out.extend_from_slice(digits.as_bytes());
    } else {
        pad_into(out, &format!("{sign}{prefix}{digits}"), spec, true);
    }
}

/// Render an integer register value per `spec` (its `kind` says how to read the bits).
pub fn format_int(out: &mut Vec<u8>, x: i64, spec: Spec) {
    match spec.kind {
        KIND_BOOL => {
            pad_into(out, bool_text(x), &spec, false);
            return;
        }
        KIND_CHAR => {
            let c = char_glyph(x);
            if spec.style == STYLE_DEBUG {
                pad_into(out, &format!("{c:?}"), &spec, false);
            } else {
                let mut b = [0u8; 4];
                pad_into(out, c.encode_utf8(&mut b), &spec, false);
            }
            return;
        }
        _ => {}
    }
    let (neg, mag) = if spec.kind == KIND_UNSIGNED {
        (false, x as u64)
    } else {
        (x < 0, x.unsigned_abs())
    };
    let digits = match spec.style {
        STYLE_LOWER_HEX => format!("{mag:x}"),
        STYLE_UPPER_HEX => format!("{mag:X}"),
        STYLE_OCTAL => format!("{mag:o}"),
        STYLE_BINARY => format!("{mag:b}"),
        STYLE_LOWER_EXP => format!("{mag:e}"),
        STYLE_UPPER_EXP => format!("{mag:E}"),
        _ => mag.to_string(),
    };
    let prefix = match (spec.alt, spec.style) {
        (true, STYLE_LOWER_HEX | STYLE_UPPER_HEX) => "0x",
        (true, STYLE_OCTAL) => "0o",
        (true, STYLE_BINARY) => "0b",
        _ => "",
    };
    let sign = if neg {
        "-"
    } else if spec.plus {
        "+"
    } else {
        ""
    };
    pad_number(out, sign, prefix, &digits, &spec);
}

/// The unpadded text of a float per the spec's style and precision — Rust's own `Display` /
/// `Debug` / `LowerExp` / `UpperExp` for the value's type.
fn float_text<T: Display + Debug + LowerExp + UpperExp>(x: T, spec: &Spec) -> String {
    match (spec.style, spec.precision.map(usize::from)) {
        (STYLE_LOWER_EXP, None) => format!("{x:e}"),
        (STYLE_LOWER_EXP, Some(p)) => format!("{x:.p$e}"),
        (STYLE_UPPER_EXP, None) => format!("{x:E}"),
        (STYLE_UPPER_EXP, Some(p)) => format!("{x:.p$E}"),
        (STYLE_DEBUG, None) => format!("{x:?}"),
        (STYLE_DEBUG, Some(p)) => format!("{x:.p$?}"),
        (_, None) => format!("{x}"),
        (_, Some(p)) => format!("{x:.p$}"),
    }
}

/// Pad a rendered float: its sign is split off so `+` and the `0` flag place it first. A NaN never
/// carries a sign, as in Rust.
fn pad_float(out: &mut Vec<u8>, text: &str, nan: bool, spec: &Spec) {
    let (sign, body) = match text.strip_prefix('-') {
        Some(rest) => ("-", rest),
        None if spec.plus && !nan => ("+", text),
        None => ("", text),
    };
    pad_number(out, sign, "", body, spec);
}

/// Render an `f64` per `spec`.
pub fn format_f64(out: &mut Vec<u8>, x: f64, spec: Spec) {
    pad_float(out, &float_text(x, &spec), x.is_nan(), &spec);
}

/// Render an `f32` (or an `f16` / `bf16` held in one) per `spec`, at `f32` precision.
pub fn format_f32(out: &mut Vec<u8>, x: f32, spec: Spec) {
    pad_float(out, &float_text(x, &spec), x.is_nan(), &spec);
}

/// Render a byte string per `spec`: verbatim under a plain `{}` (invalid UTF-8 included); with a
/// width, a precision (the maximum character count) or `{:?}` (quoted and escaped), as text.
pub fn format_str(out: &mut Vec<u8>, s: &[u8], spec: Spec) {
    if spec.style != STYLE_DEBUG && spec.width == 0 && spec.precision.is_none() {
        out.extend_from_slice(s);
        return;
    }
    let text = String::from_utf8_lossy(s);
    let text: Cow<str> = match spec.precision {
        Some(p) => Cow::Owned(text.chars().take(usize::from(p)).collect()),
        None => text,
    };
    if spec.style == STYLE_DEBUG {
        pad_into(out, &format!("{text:?}"), &spec, false);
    } else {
        pad_into(out, &text, &spec, false);
    }
}

// ---- Buffers ---------------------------------------------------------------------------------

/// Element dtype code of a `bool` buffer (one byte, 0 / 1) — beyond `fileio`'s numeric `DT_*`.
pub const DT_BOOL: i64 = 12;
/// Element dtype code of a `char` buffer (a 4-byte code point).
pub const DT_CHAR: i64 = 13;

/// Buffers longer than this print summarized (NumPy's `threshold`).
pub const SUMMARIZE_THRESHOLD: usize = 1000;
/// How many elements a summarized buffer prints at each end (NumPy's `edgeitems`).
pub const EDGE_ITEMS: usize = 3;

/// One element of a printed buffer.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Elem {
    Int(i64),
    Uint(u64),
    F32(f32),
    F64(f64),
    Bool(bool),
    Char(i64),
}

/// Byte width of a buffer element of dtype `dt`, or `None` for an unknown code.
pub fn elem_size(dt: i64) -> Option<usize> {
    match dt {
        DT_BOOL => Some(1),
        DT_CHAR => Some(4),
        _ => crate::fileio::dtype_size(dt),
    }
}

/// Is `style` one of the radix styles (`x`, `X`, `o`, `b`)? A signed element prints its bits there,
/// as Rust prints a negative `i8` in hex as its two's complement (`{:x}` of `-1i8` is `ff`).
fn radix_style(style: u8) -> bool {
    matches!(
        style,
        STYLE_LOWER_HEX | STYLE_UPPER_HEX | STYLE_OCTAL | STYLE_BINARY
    )
}

/// The element of dtype `dt` whose integer image is `bits` (any extension of the element's own
/// width — only the low bits are read) or whose float value is `float`.
pub fn elem_of(dt: i64, bits: i64, float: f64, style: u8) -> Elem {
    let width = match dt {
        DT_F32 => return Elem::F32(float as f32),
        // A half-precision element is read at its storage precision, as the interpreter's `Load`
        // rounds it and a native buffer holds only the 16 rounded bits (a no-op on a value that is
        // already rounded).
        DT_F16 => return Elem::F32(crate::round_f16(float as f32)),
        DT_BF16 => return Elem::F32(crate::round_bf16(float as f32)),
        DT_F64 => return Elem::F64(float),
        DT_BOOL => return Elem::Bool(bits & 0xFF != 0),
        DT_CHAR => return Elem::Char(bits & 0xFFFF_FFFF),
        DT_I8 | DT_U8 => 8,
        DT_I16 | DT_U16 => 16,
        DT_I32 | DT_U32 => 32,
        _ => 64,
    };
    let unsigned = matches!(dt, DT_U8 | DT_U16 | DT_U32 | DT_U64);
    let raw = if width == 64 {
        bits as u64
    } else {
        (bits as u64) & ((1u64 << width) - 1)
    };
    if unsigned || radix_style(style) {
        Elem::Uint(raw)
    } else {
        let shift = 64 - width;
        Elem::Int(((raw << shift) as i64) >> shift)
    }
}

/// Read element `i` of a native buffer of dtype `dt`.
///
/// # Safety
/// `data` must address at least `i + 1` readable elements of dtype `dt`.
pub unsafe fn read_elem(data: *const u8, i: usize, dt: i64, style: u8) -> Elem {
    let size = elem_size(dt).unwrap_or(1);
    // SAFETY: the caller guarantees element `i` is readable; every read below is unaligned-safe.
    unsafe {
        let p = data.add(i * size);
        match dt {
            DT_F32 => Elem::F32(f32::from_bits((p as *const u32).read_unaligned())),
            DT_F64 => Elem::F64(f64::from_bits((p as *const u64).read_unaligned())),
            DT_F16 => Elem::F32(crate::f16_bits_to_f32((p as *const u16).read_unaligned())),
            DT_BF16 => Elem::F32(crate::bf16_bits_to_f32((p as *const u16).read_unaligned())),
            _ => {
                let bits = match size {
                    1 => i64::from(*p),
                    2 => i64::from((p as *const u16).read_unaligned()),
                    4 => i64::from((p as *const u32).read_unaligned()),
                    _ => (p as *const u64).read_unaligned() as i64,
                };
                elem_of(dt, bits, 0.0, style)
            }
        }
    }
}

/// Render one buffer element per `spec`.
pub fn format_elem(out: &mut Vec<u8>, e: Elem, spec: Spec) {
    match e {
        Elem::Int(v) => format_int(
            out,
            v,
            Spec {
                kind: KIND_SIGNED,
                ..spec
            },
        ),
        Elem::Uint(v) => format_int(
            out,
            v as i64,
            Spec {
                kind: KIND_UNSIGNED,
                ..spec
            },
        ),
        Elem::F32(v) => format_f32(out, v, spec),
        Elem::F64(v) => format_f64(out, v, spec),
        Elem::Bool(b) => format_int(
            out,
            i64::from(b),
            Spec {
                kind: KIND_BOOL,
                ..spec
            },
        ),
        Elem::Char(c) => format_int(
            out,
            c,
            Spec {
                kind: KIND_CHAR,
                ..spec
            },
        ),
    }
}

/// The spec an aggregate applies to each element: a plain `{}` renders the elements in `Debug`
/// style (`[1.0, 2.5]`, `'a'`), as Rust's `{:?}` of a slice does; any other style applies as is.
pub fn element_spec(spec: Spec) -> Spec {
    if spec.style == STYLE_DISPLAY {
        Spec {
            style: STYLE_DEBUG,
            ..spec
        }
    } else {
        spec
    }
}

/// Render a buffer of `len` elements — `elem(i)` reads element `i` — as `[e0, e1, …]`, summarized
/// past [`SUMMARIZE_THRESHOLD`] elements.
pub fn format_seq(out: &mut Vec<u8>, len: usize, mut elem: impl FnMut(usize) -> Elem, spec: Spec) {
    let espec = element_spec(spec);
    out.push(b'[');
    let mut put = |out: &mut Vec<u8>, i: usize, first: bool| {
        if !first {
            out.extend_from_slice(b", ");
        }
        format_elem(out, elem(i), espec);
    };
    if len > SUMMARIZE_THRESHOLD {
        for i in 0..EDGE_ITEMS {
            put(out, i, i == 0);
        }
        out.extend_from_slice(b", ...");
        for i in len - EDGE_ITEMS..len {
            put(out, i, false);
        }
    } else {
        for i in 0..len {
            put(out, i, i == 0);
        }
    }
    out.push(b']');
}

// ---- The native entries ----------------------------------------------------------------------
//
// What a linked executable calls (the Cranelift object imports these; rustc pulls them out of this
// crate's rlib). The JIT binds its own capturing twins (`wukong_codegen_cranelift`'s `text_rt`) and
// the interpreter its slot-memory twins (`wukong_interp`'s `text_io`); all three render through the
// functions above.

/// The bytes of a string operand: `n` bytes at `s`, or up to the NUL when `n < 0`; nothing for null.
///
/// # Safety
/// `s` must be null, address `n` readable bytes (`n >= 0`), or a NUL-terminated string (`n < 0`).
pub unsafe fn str_operand<'a>(s: *const u8, n: i64) -> &'a [u8] {
    if s.is_null() {
        return &[];
    }
    // SAFETY: forwarded from this function's contract.
    unsafe {
        match usize::try_from(n) {
            Ok(n) => std::slice::from_raw_parts(s, n),
            Err(_) => std::ffi::CStr::from_ptr(s as *const std::ffi::c_char).to_bytes(),
        }
    }
}

/// `wukong_rt_fmt_int(target, buf, cap, pos, x, spec) -> pos'`: an integer, `bool` or `char`.
///
/// # Safety
/// The sink operands obey [`emit`]'s contract.
#[no_mangle]
pub unsafe extern "C" fn wukong_rt_fmt_int(
    target: i64,
    buf: *mut u8,
    cap: i64,
    pos: i64,
    x: i64,
    spec: i64,
) -> i64 {
    // P14.5: debug-build contract checks (the `# Safety` section above); a null buffer sink is the
    // counting mode; a stream target ignores `buf`.
    if target == TARGET_BUFFER {
        crate::safety::debug_extent_opt(buf, cap);
    }
    let mut s = Vec::new();
    format_int(&mut s, x, Spec::unpack(spec));
    // SAFETY: forwarded from this function's contract.
    unsafe { emit(target, buf, cap, pos, &s) }
}

/// `wukong_rt_fmt_f64(target, buf, cap, pos, x, spec) -> pos'`.
///
/// # Safety
/// The sink operands obey [`emit`]'s contract.
#[no_mangle]
pub unsafe extern "C" fn wukong_rt_fmt_f64(
    target: i64,
    buf: *mut u8,
    cap: i64,
    pos: i64,
    x: f64,
    spec: i64,
) -> i64 {
    // P14.5: debug-build contract checks (the `# Safety` section above); a null buffer sink is the
    // counting mode; a stream target ignores `buf`.
    if target == TARGET_BUFFER {
        crate::safety::debug_extent_opt(buf, cap);
    }
    let mut s = Vec::new();
    format_f64(&mut s, x, Spec::unpack(spec));
    // SAFETY: forwarded from this function's contract.
    unsafe { emit(target, buf, cap, pos, &s) }
}

/// `wukong_rt_fmt_f32(target, buf, cap, pos, x, spec) -> pos'`.
///
/// # Safety
/// The sink operands obey [`emit`]'s contract.
#[no_mangle]
pub unsafe extern "C" fn wukong_rt_fmt_f32(
    target: i64,
    buf: *mut u8,
    cap: i64,
    pos: i64,
    x: f32,
    spec: i64,
) -> i64 {
    // P14.5: debug-build contract checks (the `# Safety` section above); a null buffer sink is the
    // counting mode; a stream target ignores `buf`.
    if target == TARGET_BUFFER {
        crate::safety::debug_extent_opt(buf, cap);
    }
    let mut s = Vec::new();
    format_f32(&mut s, x, Spec::unpack(spec));
    // SAFETY: forwarded from this function's contract.
    unsafe { emit(target, buf, cap, pos, &s) }
}

/// `wukong_rt_fmt_str(target, buf, cap, pos, s, n, spec) -> pos'`: `n` bytes at `s`, or the
/// NUL-terminated string at `s` when `n < 0`.
///
/// # Safety
/// The sink operands obey [`emit`]'s contract; `s` and `n` obey [`str_operand`]'s.
#[no_mangle]
pub unsafe extern "C" fn wukong_rt_fmt_str(
    target: i64,
    buf: *mut u8,
    cap: i64,
    pos: i64,
    s: *const u8,
    n: i64,
    spec: i64,
) -> i64 {
    // P14.5: debug-build contract checks (the `# Safety` section above); a null buffer sink is the
    // counting mode; `n < 0` is a NUL-terminated `s` (no extent).
    if target == TARGET_BUFFER {
        crate::safety::debug_extent_opt(buf, cap);
    }
    crate::safety::debug_extent_opt(s, n);
    let mut out = Vec::new();
    // SAFETY: forwarded from this function's contract.
    format_str(&mut out, unsafe { str_operand(s, n) }, Spec::unpack(spec));
    // SAFETY: forwarded from this function's contract.
    unsafe { emit(target, buf, cap, pos, &out) }
}

/// The bytes of a literal operand: the low `n` (at most 8) bytes of `packed`, little-endian — how
/// the compiler carries a short piece of generated text (`", "`, `"Point { "`) without a data
/// section.
pub fn lit_operand(packed: i64, n: i64) -> ([u8; 8], usize) {
    (packed.to_le_bytes(), n.clamp(0, 8) as usize)
}

/// `wukong_rt_fmt_lit(target, buf, cap, pos, packed, n) -> pos'`: up to 8 literal bytes.
///
/// # Safety
/// The sink operands obey [`emit`]'s contract.
#[no_mangle]
pub unsafe extern "C" fn wukong_rt_fmt_lit(
    target: i64,
    buf: *mut u8,
    cap: i64,
    pos: i64,
    packed: i64,
    n: i64,
) -> i64 {
    // P14.5: debug-build contract checks (the `# Safety` section above); a null buffer sink is the
    // counting mode; a stream target ignores `buf`.
    if target == TARGET_BUFFER {
        crate::safety::debug_extent_opt(buf, cap);
    }
    let (bytes, n) = lit_operand(packed, n);
    // SAFETY: forwarded from this function's contract.
    unsafe { emit(target, buf, cap, pos, &bytes[..n]) }
}

/// `wukong_rt_fmt_slice(target, buf, cap, pos, data, len, dtype, spec) -> pos'`: a buffer of `len`
/// scalar elements of `dtype` (`fileio`'s `DT_*`, [`DT_BOOL`], [`DT_CHAR`]).
///
/// # Safety
/// The sink operands obey [`emit`]'s contract; `data` addresses `len` readable elements of `dtype`
/// (it may be null when `len <= 0`).
#[no_mangle]
pub unsafe extern "C" fn wukong_rt_fmt_slice(
    target: i64,
    buf: *mut u8,
    cap: i64,
    pos: i64,
    data: *const u8,
    len: i64,
    dtype: i64,
    spec: i64,
) -> i64 {
    // P14.5: debug-build contract checks (the `# Safety` section above); a null buffer sink is the
    // counting mode; a stream target ignores `buf`.
    if target == TARGET_BUFFER {
        crate::safety::debug_extent_opt(buf, cap);
    }
    if let Some(sz) = elem_size(dtype) {
        crate::safety::debug_extent_bytes(data, len, sz);
    }
    let spec = Spec::unpack(spec);
    let len = if data.is_null() || elem_size(dtype).is_none() {
        0
    } else {
        usize::try_from(len).unwrap_or(0)
    };
    let mut out = Vec::new();
    // SAFETY: `data` addresses `len` elements of `dtype` (the caller's contract).
    format_seq(
        &mut out,
        len,
        |i| unsafe { read_elem(data, i, dtype, spec.style) },
        spec,
    );
    // SAFETY: forwarded from this function's contract.
    unsafe { emit(target, buf, cap, pos, &out) }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// While a capture is open, standard error written through `write_stream` (and `emit`) is
    /// collected in order instead of written; closing it returns the bytes, and a second close
    /// returns nothing.
    #[test]
    fn stderr_capture_collects_what_a_program_writes() {
        begin_stderr_capture();
        write_stream(TARGET_STDERR, b"first ");
        // SAFETY: a stream target touches no buffer.
        let end = unsafe { emit(TARGET_STDERR, std::ptr::null_mut(), 0, 5, b"second") };
        assert_eq!(end, 11);
        assert_eq!(end_stderr_capture(), b"first second");
        assert!(end_stderr_capture().is_empty());
    }

    fn int(x: i64, spec: Spec) -> String {
        let mut out = Vec::new();
        format_int(&mut out, x, spec);
        String::from_utf8(out).unwrap()
    }

    fn f32s(x: f32, spec: Spec) -> String {
        let mut out = Vec::new();
        format_f32(&mut out, x, spec);
        String::from_utf8(out).unwrap()
    }

    fn f64s(x: f64, spec: Spec) -> String {
        let mut out = Vec::new();
        format_f64(&mut out, x, spec);
        String::from_utf8(out).unwrap()
    }

    fn strs(s: &[u8], spec: Spec) -> String {
        let mut out = Vec::new();
        format_str(&mut out, s, spec);
        String::from_utf8(out).unwrap()
    }

    /// Every expected string below is derived from the rule in the module docs, not recorded.
    #[test]
    fn scalars_print_at_their_own_type() {
        // f32 0.1 is 13421773 * 2^-27; "0.1" is the shortest decimal that rounds back to it.
        assert_eq!(f32_line(0.1), "0.1\n");
        // f32(0.1) + f32(0.2) is 0x3E99999A, which is also f32(0.3).
        assert_eq!(f32_line(0.1f32 + 0.2f32), "0.3\n");
        assert_eq!(f32_line(1.0 / 3.0), "0.33333334\n");
        assert_eq!(f32_line(16777216.0), "16777216\n");
        assert_eq!(f32_line(-0.0), "-0\n");
        assert_eq!(f32_line(f32::NAN), "NaN\n");
        assert_eq!(f32_line(f32::NEG_INFINITY), "-inf\n");
        // f16 0.1 is 0x2E66 = 0.0999755859375 exactly; its shortest f32 digits are 0.099975586.
        assert_eq!(f32_line(0.099_975_586), "0.099975586\n");
        assert_eq!(bool_line(1), "true\n");
        assert_eq!(bool_line(0), "false\n");
        assert_eq!(char_line('A' as i64), "A\n");
        assert_eq!(char_line(0x1F600), "\u{1F600}\n");
        // A surrogate and an out-of-range value are not Unicode scalar values.
        assert_eq!(char_line(0xD800), "\u{FFFD}\n");
        assert_eq!(char_line(0x11_0000), "\u{FFFD}\n");
    }

    #[test]
    fn spec_packing_round_trips() {
        let specs = [
            Spec::default(),
            Spec {
                width: 65535,
                precision: Some(255),
                fill: b'*',
                align: ALIGN_CENTER,
                plus: true,
                alt: true,
                zero: true,
                style: STYLE_UPPER_EXP,
                kind: KIND_CHAR,
            },
            Spec {
                precision: Some(0),
                style: STYLE_BINARY,
                kind: KIND_UNSIGNED,
                ..Spec::default()
            },
        ];
        for s in specs {
            assert_eq!(Spec::unpack(s.pack()), s, "{s:?}");
        }
        assert_eq!(Spec::default().pack(), 0);
    }

    /// Each line is what Rust's `format!` prints for the same spec and type.
    #[test]
    fn specs_render_like_rust_format() {
        let d = Spec::default();
        assert_eq!(int(-42, d), "-42");
        assert_eq!(int(42, Spec { plus: true, ..d }), "+42");
        assert_eq!(int(7, Spec { width: 4, ..d }), "   7");
        assert_eq!(
            int(
                7,
                Spec {
                    width: 4,
                    align: ALIGN_LEFT,
                    ..d
                }
            ),
            "7   "
        );
        assert_eq!(
            int(
                7,
                Spec {
                    width: 5,
                    align: ALIGN_CENTER,
                    fill: b'*',
                    ..d
                }
            ),
            "**7**"
        );
        assert_eq!(
            int(
                -7,
                Spec {
                    width: 5,
                    zero: true,
                    ..d
                }
            ),
            "-0007"
        );
        // `{:#06x}` of 255u8: the prefix is inside the width, the zeros after it.
        assert_eq!(
            int(
                255,
                Spec {
                    width: 6,
                    zero: true,
                    alt: true,
                    style: STYLE_LOWER_HEX,
                    kind: KIND_UNSIGNED,
                    ..d
                }
            ),
            "0x00ff"
        );
        // `{:X}` of -1i32: the compiler hands over the bits zero-extended from 32, as unsigned.
        assert_eq!(
            int(
                0xFFFF_FFFF,
                Spec {
                    style: STYLE_UPPER_HEX,
                    kind: KIND_UNSIGNED,
                    ..d
                }
            ),
            "FFFFFFFF"
        );
        assert_eq!(
            int(
                5,
                Spec {
                    alt: true,
                    style: STYLE_BINARY,
                    ..d
                }
            ),
            "0b101"
        );
        assert_eq!(
            int(
                -1,
                Spec {
                    kind: KIND_UNSIGNED,
                    ..d
                }
            ),
            "18446744073709551615"
        );
        assert_eq!(
            int(
                1,
                Spec {
                    kind: KIND_BOOL,
                    width: 6,
                    ..d
                }
            ),
            "true  "
        );
        assert_eq!(
            int(
                'x' as i64,
                Spec {
                    kind: KIND_CHAR,
                    style: STYLE_DEBUG,
                    ..d
                }
            ),
            "'x'"
        );
        // Floats: `{:.3}`, `{:8.2}`, `{:+}`, `{:e}`, `{:?}` and the sign-aware `0` flag. (Not
        // 3.14159: clippy's deny-by-default `approx_constant` rejects a truncated PI literal.) The
        // f32 1.23456 is exactly 1.2345600128173828125, so three decimals round up to 1.235.
        assert_eq!(
            f32s(
                1.23456,
                Spec {
                    precision: Some(3),
                    ..d
                }
            ),
            "1.235"
        );
        assert_eq!(
            f64s(
                -2.5,
                Spec {
                    width: 8,
                    precision: Some(2),
                    ..d
                }
            ),
            "   -2.50"
        );
        assert_eq!(f32s(1.5, Spec { plus: true, ..d }), "+1.5");
        assert_eq!(f32s(f32::NAN, Spec { plus: true, ..d }), "NaN");
        assert_eq!(
            f64s(
                1234.5,
                Spec {
                    style: STYLE_LOWER_EXP,
                    ..d
                }
            ),
            "1.2345e3"
        );
        assert_eq!(
            f32s(
                1.0,
                Spec {
                    style: STYLE_DEBUG,
                    ..d
                }
            ),
            "1.0"
        );
        assert_eq!(
            f64s(
                -1.5,
                Spec {
                    width: 7,
                    zero: true,
                    precision: Some(1),
                    ..d
                }
            ),
            "-0001.5"
        );
        // Strings: verbatim, padded, truncated, quoted.
        assert_eq!(strs(b"hi", d), "hi");
        assert_eq!(strs(b"hi", Spec { width: 4, ..d }), "hi  ");
        assert_eq!(
            strs(
                b"hello",
                Spec {
                    precision: Some(3),
                    ..d
                }
            ),
            "hel"
        );
        assert_eq!(
            strs(
                b"a\"b",
                Spec {
                    style: STYLE_DEBUG,
                    ..d
                }
            ),
            "\"a\\\"b\""
        );
    }

    #[test]
    fn buffer_writes_truncate_like_snprintf() {
        let mut buf = [0xAAu8; 6];
        let p = buf.as_mut_ptr();
        // SAFETY: `buf` is 6 writable bytes.
        let end = unsafe { write_buffer(p, 6, 0, b"abc") };
        assert_eq!((end, &buf[..4]), (3, &b"abc\0"[..]));
        // A write that only partly fits keeps the NUL in the last byte and still counts every byte.
        // SAFETY: as above.
        let end = unsafe { write_buffer(p, 6, end, b"defg") };
        assert_eq!((end, &buf[..]), (7, &b"abcde\0"[..]));
        // Past the end: nothing more is copied; the position keeps counting.
        // SAFETY: as above.
        let end = unsafe { write_buffer(p, 6, end, b"hi") };
        assert_eq!((end, &buf[..]), (9, &b"abcde\0"[..]));
        assert_eq!(
            buffer_write(0, 4, 2),
            BufWrite {
                start: 0,
                count: 0,
                nul: None,
                end: 6
            }
        );
    }

    #[test]
    fn buffers_render_like_a_debug_slice_and_summarize() {
        let d = Spec::default();
        let mut out = Vec::new();
        let xs = [1.0f32, 2.5, -0.0];
        format_seq(&mut out, xs.len(), |i| Elem::F32(xs[i]), d);
        assert_eq!(String::from_utf8(out).unwrap(), "[1.0, 2.5, -0.0]");
        let mut out = Vec::new();
        format_seq(&mut out, 0, |_| Elem::Int(0), d);
        assert_eq!(out, b"[]");
        let mut out = Vec::new();
        format_seq(&mut out, 1001, |i| Elem::Int(i as i64), d);
        assert_eq!(
            String::from_utf8(out).unwrap(),
            "[0, 1, 2, ..., 998, 999, 1000]"
        );
        // A signed element in a radix style prints its own width's bits.
        assert_eq!(elem_of(DT_I8, -1, 0.0, STYLE_LOWER_HEX), Elem::Uint(0xFF));
        assert_eq!(elem_of(DT_I16, 0xFFFF, 0.0, STYLE_DISPLAY), Elem::Int(-1));
        assert_eq!(elem_of(DT_U8, -56, 0.0, STYLE_DISPLAY), Elem::Uint(200));
        // Native buffers decode at their own width.
        let raw: [i16; 2] = [-3, 7];
        // SAFETY: `raw` holds two i16 elements.
        let e = unsafe { read_elem(raw.as_ptr() as *const u8, 0, DT_I16, STYLE_DISPLAY) };
        assert_eq!(e, Elem::Int(-3));
        let (bytes, n) = lit_operand(i64::from_le_bytes(*b"ab\0\0\0\0\0\0"), 2);
        assert_eq!(&bytes[..n], b"ab");
    }
}
