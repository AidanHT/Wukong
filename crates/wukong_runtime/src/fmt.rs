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

#[cfg(test)]
mod tests {
    use super::*;

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
}
