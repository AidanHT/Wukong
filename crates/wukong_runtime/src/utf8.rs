//! UTF-8 for the `str` operations: validation (`str_from_utf8`, `utf8_valid_up_to`), decoding
//! (`for c in s.chars()`) and counting (`s.char_count()`), shared by every execution path — the
//! interpreter calls the pure functions below on the bytes it reads out of its slots, and the JIT
//! and a linked executable call the `extern "C"` entries, which wrap the same functions.
//!
//! **Well-formedness** is the Unicode Standard's (Table 3-7, "Well-Formed UTF-8 Byte Sequences"),
//! which is exactly what `std::str::from_utf8` accepts: no overlong form (`C0`, `C1`, `E0 80..9F`,
//! `F0 80..8F`), no surrogate (`ED A0..BF`), nothing above U+10FFFF (`F4 90..`, `F5..FF`).
//!
//! **A fault** is reported as Rust's `Utf8Error` reports it: the length of the longest valid prefix
//! (`valid_up_to`), and the length of the invalid sequence after it (`error_len`, `None` when the
//! input ends in the middle of a character). The `extern "C"` entry packs the two into one `i64`
//! ([`pack_fault`]).
//!
//! **Decoding never fails.** A `str` built by `str_from_utf8` was valid when it was built, but the
//! language has no borrow checker: the buffer it views can be written afterwards, and `as_bytes()`
//! hands out the bytes of any `str`. So [`decode_at`] decodes ANY byte sequence, deterministically:
//! each ill-formed sequence becomes one U+FFFD REPLACEMENT CHARACTER covering its *maximal
//! subpart* (Unicode §3.9, "U+FFFD Substitution of Maximal Subparts") — the policy of
//! `String::from_utf8_lossy` and the WHATWG Encoding Standard, which the tests below use as the
//! oracle. On valid input it is plain UTF-8 decoding.

/// The packed answer for valid input: no fault.
pub const UTF8_VALID: i64 = -1;

/// The bit position of `error_len` in a packed fault ([`pack_fault`]).
const ERROR_LEN_SHIFT: u32 = 56;

/// The code point an ill-formed sequence decodes to: U+FFFD REPLACEMENT CHARACTER.
pub const REPLACEMENT: u32 = 0xFFFD;

/// The first ill-formed sequence of a byte string: Rust's `Utf8Error`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Utf8Fault {
    /// The length of the longest valid UTF-8 prefix.
    pub valid_up_to: usize,
    /// The length (1 to 3) of the invalid sequence that starts at `valid_up_to`, or `None` when
    /// the input ends in the middle of a character that was well-formed so far.
    pub error_len: Option<u8>,
}

/// The inclusive range the byte after a lead byte must fall in (Unicode Table 3-7), and how many
/// continuation bytes the lead announces. `None` for a byte that cannot start a sequence: a
/// continuation byte `80..=BF`, an overlong lead `C0`/`C1`, or `F5..=FF`.
fn lead(b: u8) -> Option<(u8, u8, usize)> {
    match b {
        0xC2..=0xDF => Some((0x80, 0xBF, 1)),
        0xE0 => Some((0xA0, 0xBF, 2)),
        0xE1..=0xEC | 0xEE..=0xEF => Some((0x80, 0xBF, 2)),
        0xED => Some((0x80, 0x9F, 2)),
        0xF0 => Some((0x90, 0xBF, 3)),
        0xF1..=0xF3 => Some((0x80, 0xBF, 3)),
        0xF4 => Some((0x80, 0x8F, 3)),
        _ => None,
    }
}

/// One step of decoding at `i < bytes.len()`.
enum Step {
    /// A well-formed character: its code point and byte width.
    Char(u32, usize),
    /// An ill-formed sequence: the length of its maximal subpart (at least 1), and whether it was
    /// cut short by the end of the input (`Utf8Error::error_len() == None`).
    Bad(usize, bool),
}

/// Decode the sequence that starts at `bytes[i]` (`i < bytes.len()`).
fn step(bytes: &[u8], i: usize) -> Step {
    let b0 = bytes[i];
    if b0 < 0x80 {
        return Step::Char(u32::from(b0), 1);
    }
    let Some((lo, hi, extra)) = lead(b0) else {
        return Step::Bad(1, false);
    };
    // The lead's payload bits: 5, 4 or 3 of them for a 2-, 3- or 4-byte sequence.
    let mut cp = u32::from(b0) & (0x7F >> (extra + 1));
    for k in 1..=extra {
        let Some(&b) = bytes.get(i + k) else {
            // Every byte so far was right, and the input ended: an incomplete character.
            return Step::Bad(k, true);
        };
        // The second byte has the lead's own range; every later one is a plain continuation.
        let (l, h) = if k == 1 { (lo, hi) } else { (0x80, 0xBF) };
        if !(l..=h).contains(&b) {
            return Step::Bad(k, false);
        }
        cp = (cp << 6) | (u32::from(b) & 0x3F);
    }
    Step::Char(cp, extra + 1)
}

/// The first ill-formed sequence of `bytes`, or `None` when it is all valid UTF-8 — what
/// `std::str::from_utf8(bytes).err()` reports.
pub fn validate(bytes: &[u8]) -> Option<Utf8Fault> {
    let mut i = 0;
    while i < bytes.len() {
        // The common case first: a run of ASCII needs no table.
        if bytes[i] < 0x80 {
            i += 1;
            continue;
        }
        match step(bytes, i) {
            Step::Char(_, w) => i += w,
            Step::Bad(w, at_end) => {
                return Some(Utf8Fault {
                    valid_up_to: i,
                    error_len: if at_end { None } else { Some(w as u8) },
                })
            }
        }
    }
    None
}

/// The code point at byte `i` of `bytes` and the number of bytes it spans, decoding ill-formed
/// input as U+FFFD per maximal subpart (see the module docs). `i` must be `< bytes.len()`; the
/// width is always at least 1, so a loop that advances by it terminates.
pub fn decode_at(bytes: &[u8], i: usize) -> (u32, usize) {
    match step(bytes, i) {
        Step::Char(cp, w) => (cp, w),
        Step::Bad(w, _) => (REPLACEMENT, w),
    }
}

/// How many characters a `for c in s.chars()` loop over `bytes` yields: one per well-formed
/// character and one per ill-formed sequence ([`decode_at`]).
pub fn char_count(bytes: &[u8]) -> usize {
    let (mut i, mut n) = (0, 0);
    while i < bytes.len() {
        i += if bytes[i] < 0x80 {
            1
        } else {
            decode_at(bytes, i).1
        };
        n += 1;
    }
    n
}

/// Pack a validation result into the `i64` the `extern "C"` entry returns: [`UTF8_VALID`] (`-1`)
/// for valid input, else `valid_up_to | error_len << 56`, with `error_len` 0 for an incomplete
/// character at the end.
pub fn pack_fault(fault: Option<Utf8Fault>) -> i64 {
    match fault {
        None => UTF8_VALID,
        Some(f) => {
            let len = i64::from(f.error_len.unwrap_or(0));
            (f.valid_up_to as i64 & ((1 << ERROR_LEN_SHIFT) - 1)) | (len << ERROR_LEN_SHIFT)
        }
    }
}

/// The inverse of [`pack_fault`].
pub fn unpack_fault(packed: i64) -> Option<Utf8Fault> {
    if packed < 0 {
        return None;
    }
    let len = (packed >> ERROR_LEN_SHIFT) as u8;
    Some(Utf8Fault {
        valid_up_to: (packed & ((1 << ERROR_LEN_SHIFT) - 1)) as usize,
        error_len: (len != 0).then_some(len),
    })
}

/// The longest valid prefix of `len` bytes whose packed validation result is `packed`: `len` when
/// they are all valid. What `utf8_valid_up_to(b)` returns.
pub fn valid_up_to(packed: i64, len: i64) -> i64 {
    match unpack_fault(packed) {
        None => len.max(0),
        Some(f) => f.valid_up_to as i64,
    }
}

/// Pack a decoded character for the `extern "C"` entry: `code point | width << 32`.
pub fn pack_char(cp: u32, width: usize) -> i64 {
    i64::from(cp) | ((width as i64) << 32)
}

/// The bytes `data[..len]`, or an empty slice for a null pointer or a non-positive length.
///
/// # Safety
/// When `len > 0` and `data` is non-null, `data` must be valid for reads of `len` bytes.
unsafe fn bytes<'a>(data: *const u8, len: i64) -> &'a [u8] {
    match usize::try_from(len) {
        Ok(n) if n > 0 && !data.is_null() => std::slice::from_raw_parts(data, n),
        _ => &[],
    }
}

/// `wukong_rt_utf8_check(data, len) -> i64`: validate `data[..len]` ([`validate`]), packed by
/// [`pack_fault`] — `-1` when it is valid UTF-8.
///
/// # Safety
/// `data` must be valid for reads of `len` bytes when `len > 0`.
#[no_mangle]
pub unsafe extern "C" fn wukong_rt_utf8_check(data: *const u8, len: i64) -> i64 {
    crate::safety::debug_extent(data, len);
    // SAFETY: forwarded from this function's contract.
    pack_fault(validate(unsafe { bytes(data, len) }))
}

/// `wukong_rt_utf8_decode(data, len, i) -> i64`: the character at byte `i` of `data[..len]`
/// ([`decode_at`]), packed by [`pack_char`]. An `i` outside `0..len` decodes as U+FFFD of width 1,
/// so a caller's loop still advances (the lowering never asks for one).
///
/// # Safety
/// `data` must be valid for reads of `len` bytes when `len > 0`.
#[no_mangle]
pub unsafe extern "C" fn wukong_rt_utf8_decode(data: *const u8, len: i64, i: i64) -> i64 {
    crate::safety::debug_extent(data, len);
    // SAFETY: forwarded from this function's contract.
    let b = unsafe { bytes(data, len) };
    match usize::try_from(i) {
        Ok(i) if i < b.len() => {
            let (cp, w) = decode_at(b, i);
            pack_char(cp, w)
        }
        _ => pack_char(REPLACEMENT, 1),
    }
}

/// `wukong_rt_utf8_count(data, len) -> i64`: [`char_count`] of `data[..len]`.
///
/// # Safety
/// `data` must be valid for reads of `len` bytes when `len > 0`.
#[no_mangle]
pub unsafe extern "C" fn wukong_rt_utf8_count(data: *const u8, len: i64) -> i64 {
    crate::safety::debug_extent(data, len);
    // SAFETY: forwarded from this function's contract.
    char_count(unsafe { bytes(data, len) }) as i64
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every byte a well-formedness boundary sits on (Table 3-7), plus a few interior values.
    const EDGES: &[u8] = &[
        0x00, 0x41, 0x7F, 0x80, 0x8F, 0x90, 0x9F, 0xA0, 0xBF, 0xC0, 0xC1, 0xC2, 0xDF, 0xE0, 0xE1,
        0xEC, 0xED, 0xEE, 0xEF, 0xF0, 0xF1, 0xF3, 0xF4, 0xF5, 0xFF,
    ];

    /// `bytes` judged against the standard library: validation against `std::str::from_utf8`,
    /// decoding and counting against `String::from_utf8_lossy`, which substitutes maximal
    /// subparts exactly as Unicode §3.9 recommends.
    #[track_caller]
    fn agrees_with_std(bytes: &[u8]) {
        let want = std::str::from_utf8(bytes).err().map(|e| Utf8Fault {
            valid_up_to: e.valid_up_to(),
            error_len: e.error_len().map(|n| n as u8),
        });
        assert_eq!(validate(bytes), want, "validate {bytes:02X?}");
        assert_eq!(unpack_fault(pack_fault(want)), want, "{bytes:02X?}");
        let lossy: Vec<u32> = String::from_utf8_lossy(bytes)
            .chars()
            .map(u32::from)
            .collect();
        let mut got = Vec::new();
        let mut i = 0;
        while i < bytes.len() {
            let (cp, w) = decode_at(bytes, i);
            assert!(
                w >= 1 && i + w <= bytes.len(),
                "width {w} at {i} of {bytes:02X?}"
            );
            got.push(cp);
            i += w;
        }
        assert_eq!(got, lossy, "decode {bytes:02X?}");
        assert_eq!(char_count(bytes), lossy.len(), "count {bytes:02X?}");
    }

    /// Every byte string of length 0, 1 and 2 — all 65,793 of them.
    #[test]
    fn exhaustive_up_to_two_bytes() {
        agrees_with_std(&[]);
        for a in 0..=255u8 {
            agrees_with_std(&[a]);
            for b in 0..=255u8 {
                agrees_with_std(&[a, b]);
            }
        }
    }

    /// Every three-byte string — 16.7 million — through validation and decoding, without an
    /// allocation per case: the decoding oracle is `<[u8]>::utf8_chunks`, the iterator
    /// `String::from_utf8_lossy` is built on (each invalid chunk is one U+FFFD).
    #[test]
    fn exhaustive_three_bytes() {
        for a in 0..=255u8 {
            for b in 0..=255u8 {
                for c in 0..=255u8 {
                    let s = [a, b, c];
                    let want = std::str::from_utf8(&s).err().map(|e| Utf8Fault {
                        valid_up_to: e.valid_up_to(),
                        error_len: e.error_len().map(|n| n as u8),
                    });
                    if validate(&s) != want {
                        panic!("validate {s:02X?}: {:?} vs {want:?}", validate(&s));
                    }
                    let mut i = 0;
                    let mut n = 0;
                    for chunk in s.utf8_chunks() {
                        for ch in chunk.valid().chars() {
                            let (cp, w) = decode_at(&s, i);
                            assert_eq!((cp, w), (u32::from(ch), ch.len_utf8()), "{s:02X?} @{i}");
                            i += w;
                            n += 1;
                        }
                        if !chunk.invalid().is_empty() {
                            let (cp, w) = decode_at(&s, i);
                            assert_eq!((cp, w), (REPLACEMENT, chunk.invalid().len()), "{s:02X?}");
                            i += w;
                            n += 1;
                        }
                    }
                    assert_eq!((i, n), (3, char_count(&s)), "{s:02X?}");
                }
            }
        }
    }

    /// Lengths 3 and 4 over the boundary alphabet, through every check (decode and count too).
    #[test]
    fn edge_alphabet_three_and_four_bytes() {
        for &a in EDGES {
            for &b in EDGES {
                for &c in EDGES {
                    agrees_with_std(&[a, b, c]);
                    for &d in EDGES {
                        agrees_with_std(&[a, b, c, d]);
                    }
                }
            }
        }
    }

    /// Every Unicode scalar value encodes and decodes back, at every width; a surrogate never
    /// decodes as itself.
    #[test]
    fn every_scalar_value_round_trips() {
        let mut buf = [0u8; 4];
        for cp in (0..=0x10FFFFu32).filter_map(char::from_u32) {
            let s = cp.encode_utf8(&mut buf).as_bytes();
            assert_eq!(validate(s), None, "{cp:?}");
            assert_eq!(decode_at(s, 0), (u32::from(cp), s.len()), "{cp:?}");
            assert_eq!(char_count(s), 1);
        }
    }

    #[test]
    fn the_c_entries_wrap_the_pure_functions() {
        let s = "a\u{e9}\u{20ac}\u{1F600}".as_bytes();
        // SAFETY: `s` is valid for `s.len()` reads; a null pointer is legal with length 0.
        unsafe {
            assert_eq!(wukong_rt_utf8_check(s.as_ptr(), s.len() as i64), UTF8_VALID);
            assert_eq!(wukong_rt_utf8_count(s.as_ptr(), s.len() as i64), 4);
            assert_eq!(
                wukong_rt_utf8_decode(s.as_ptr(), s.len() as i64, 3),
                pack_char(0x20AC, 3)
            );
            assert_eq!(
                wukong_rt_utf8_decode(s.as_ptr(), s.len() as i64, 99),
                pack_char(REPLACEMENT, 1)
            );
            assert_eq!(wukong_rt_utf8_check(std::ptr::null(), 0), UTF8_VALID);
            assert_eq!(wukong_rt_utf8_count(std::ptr::null(), 0), 0);
            let bad = [b'o', b'k', 0xE2, 0x82];
            let packed = wukong_rt_utf8_check(bad.as_ptr(), 4);
            assert_eq!(
                unpack_fault(packed),
                Some(Utf8Fault {
                    valid_up_to: 2,
                    error_len: None
                })
            );
            assert_eq!(valid_up_to(packed, 4), 2);
            assert_eq!(valid_up_to(UTF8_VALID, 4), 4);
            let bad = [0xC3, 0x28];
            assert_eq!(
                unpack_fault(wukong_rt_utf8_check(bad.as_ptr(), 2)),
                Some(Utf8Fault {
                    valid_up_to: 0,
                    error_len: Some(1)
                })
            );
        }
    }
}
