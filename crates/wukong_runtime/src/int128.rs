//! The 128-bit integer operations the native backend has no instruction for.
//!
//! Cranelift's x64 backend holds an `i128` in a register pair and lowers addition, multiplication,
//! the bitwise operations, shifts and compares itself, but it has no 128-bit divide and no
//! conversion between a 128-bit integer and a float. Those lower to calls of the entries below,
//! which compute exactly what the interpreter computes on its own `i128` values — the language's
//! definition:
//!
//! * division and remainder **wrap** and **never trap**: a zero divisor yields `0` (as for every
//!   other integer width), and `i128::MIN / -1` is `i128::MIN` with remainder `0`;
//! * an integer converts to a float **rounded once**, to nearest, ties to even, straight to the
//!   target precision (an `f32` result is never rounded through `f64` first);
//! * a float converts to an integer **saturating**: NaN is `0`, and a value outside the target's
//!   range clamps to its nearest end (Rust's `as`).
//!
//! An `i128` operand crosses the call as its two 64-bit halves, low then high, and a 128-bit result
//! comes back through a pointer to two `i64`s (low first) — never the platform's own `i128` calling
//! convention, which compilers have disagreed on.

use crate::fmt::i128_from_halves;

/// `wukong_rt_i128_divrem`'s operation codes.
pub const I128_SDIV: i64 = 0;
pub const I128_UDIV: i64 = 1;
pub const I128_SREM: i64 = 2;
pub const I128_UREM: i64 = 3;

/// Divide or take the remainder of the 128-bit patterns `a` and `b` (`op` is one of the
/// `I128_*` codes), with the language's integer semantics: wrapping, and `0` for a zero divisor.
/// The unsigned codes read both patterns as `u128`.
pub fn i128_divrem(op: i64, a: i128, b: i128) -> i128 {
    match op {
        _ if b == 0 => 0,
        I128_SDIV => a.wrapping_div(b),
        I128_UDIV => ((a as u128) / (b as u128)) as i128,
        I128_SREM => a.wrapping_rem(b),
        _ => ((a as u128) % (b as u128)) as i128,
    }
}

/// The 128-bit pattern `x` as a float of type `f64`, read as signed when `signed`, rounded once.
pub fn i128_to_f64(x: i128, signed: bool) -> f64 {
    if signed {
        x as f64
    } else {
        x as u128 as f64
    }
}

/// [`i128_to_f64`] rounded once to `f32`.
pub fn i128_to_f32(x: i128, signed: bool) -> f32 {
    if signed {
        x as f32
    } else {
        x as u128 as f32
    }
}

/// The float `x` converted to a 128-bit integer (its bit pattern), saturating: to `i128` when
/// `signed`, else to `u128`. An `f32` operand converts exactly through this entry, because every
/// `f32` is an `f64` and the conversion truncates toward zero.
pub fn f64_to_i128(x: f64, signed: bool) -> i128 {
    if signed {
        x as i128
    } else {
        x as u128 as i128
    }
}

/// Store `r` as two `i64` halves, low first.
///
/// # Safety
/// `out` must be valid for writes of two `i64`.
unsafe fn store_halves(out: *mut i64, r: i128) {
    // SAFETY: forwarded from this function's contract.
    unsafe {
        out.write_unaligned(r as i64);
        out.add(1).write_unaligned((r >> 64) as i64);
    }
}

/// `wukong_rt_i128_divrem(a_lo, a_hi, b_lo, b_hi, op, out)`: [`i128_divrem`] of the two operands
/// given as halves, the result stored at `out` (low half first).
///
/// # Safety
/// `out` must be valid for writes of two `i64`.
#[no_mangle]
pub unsafe extern "C" fn wukong_rt_i128_divrem(
    a_lo: i64,
    a_hi: i64,
    b_lo: i64,
    b_hi: i64,
    op: i64,
    out: *mut i64,
) {
    // P14.5: debug-build contract check (the `# Safety` section above).
    crate::safety::debug_extent(out, 2);
    let r = i128_divrem(
        op,
        i128_from_halves(a_lo, a_hi),
        i128_from_halves(b_lo, b_hi),
    );
    // SAFETY: forwarded from this function's contract.
    unsafe { store_halves(out, r) }
}

/// `wukong_rt_i128_to_f64(lo, hi, signed) -> f64`: [`i128_to_f64`].
#[no_mangle]
pub extern "C" fn wukong_rt_i128_to_f64(lo: i64, hi: i64, signed: i64) -> f64 {
    i128_to_f64(i128_from_halves(lo, hi), signed != 0)
}

/// `wukong_rt_i128_to_f32(lo, hi, signed) -> f32`: [`i128_to_f32`].
#[no_mangle]
pub extern "C" fn wukong_rt_i128_to_f32(lo: i64, hi: i64, signed: i64) -> f32 {
    i128_to_f32(i128_from_halves(lo, hi), signed != 0)
}

/// `wukong_rt_f64_to_i128(x, signed, out)`: [`f64_to_i128`], the result stored at `out` (low
/// half first).
///
/// # Safety
/// `out` must be valid for writes of two `i64`.
#[no_mangle]
pub unsafe extern "C" fn wukong_rt_f64_to_i128(x: f64, signed: i64, out: *mut i64) {
    // P14.5: debug-build contract check (the `# Safety` section above).
    crate::safety::debug_extent(out, 2);
    // SAFETY: forwarded from this function's contract.
    unsafe { store_halves(out, f64_to_i128(x, signed != 0)) }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EDGES: [i128; 12] = [
        0,
        1,
        -1,
        2,
        -2,
        7,
        -7,
        i128::MAX,
        i128::MIN,
        i128::MIN + 1,
        1 << 64,
        -(1 << 64) + 3,
    ];

    fn halves(x: i128) -> (i64, i64) {
        (x as i64, (x >> 64) as i64)
    }

    #[test]
    fn halves_round_trip() {
        for x in EDGES {
            let (lo, hi) = halves(x);
            assert_eq!(i128_from_halves(lo, hi), x);
        }
    }

    /// Every operation against a direct Rust evaluation of the definition, through the C entry.
    #[test]
    fn divrem_matches_the_definition_through_the_entry() {
        for a in EDGES {
            for b in EDGES {
                let want = [
                    if b == 0 { 0 } else { a.wrapping_div(b) },
                    if b == 0 {
                        0
                    } else {
                        ((a as u128) / (b as u128)) as i128
                    },
                    if b == 0 { 0 } else { a.wrapping_rem(b) },
                    if b == 0 {
                        0
                    } else {
                        ((a as u128) % (b as u128)) as i128
                    },
                ];
                for (op, want) in [I128_SDIV, I128_UDIV, I128_SREM, I128_UREM]
                    .into_iter()
                    .zip(want)
                {
                    let (al, ah) = halves(a);
                    let (bl, bh) = halves(b);
                    let mut out = [0i64; 2];
                    // SAFETY: `out` holds two `i64`.
                    unsafe { wukong_rt_i128_divrem(al, ah, bl, bh, op, out.as_mut_ptr()) };
                    assert_eq!(i128_from_halves(out[0], out[1]), want, "op {op}: {a} {b}");
                }
            }
        }
        // The one signed overflow wraps instead of trapping.
        assert_eq!(i128_divrem(I128_SDIV, i128::MIN, -1), i128::MIN);
        assert_eq!(i128_divrem(I128_SREM, i128::MIN, -1), 0);
    }

    #[test]
    fn conversions_round_once_and_saturate() {
        // 2^64 + 1 is not an f64; it rounds to 2^64. u128::MAX rounds up to 2^128 in f64 and f32.
        let (lo, hi) = halves((1i128 << 64) + 1);
        assert_eq!(wukong_rt_i128_to_f64(lo, hi, 1), 18446744073709551616.0);
        let (lo, hi) = halves(-1);
        assert_eq!(
            wukong_rt_i128_to_f64(lo, hi, 0),
            340282366920938463463374607431768211456.0
        );
        assert_eq!(wukong_rt_i128_to_f64(lo, hi, 1), -1.0);
        // ... which is past f32::MAX, so the f32 conversion overflows to infinity.
        assert_eq!(wukong_rt_i128_to_f32(lo, hi, 0), f32::INFINITY);
        // f32 rounds directly: 2^24 + 1 is a tie between 2^24 and 2^24 + 2, so it goes to even.
        let (lo, hi) = halves((1 << 24) + 1);
        assert_eq!(wukong_rt_i128_to_f32(lo, hi, 1), 16777216.0);
        let mut out = [0i64; 2];
        for (x, signed, want) in [
            (f64::NAN, 1, 0),
            (-1.5, 1, -1),
            (-1.5, 0, 0),
            (1e40, 1, i128::MAX),
            (-1e40, 1, i128::MIN),
            (1e40, 0, -1),
            (2f64.powi(100), 1, 1 << 100),
            (-2f64.powi(127), 1, i128::MIN),
        ] {
            // SAFETY: `out` holds two `i64`.
            unsafe { wukong_rt_f64_to_i128(x, signed, out.as_mut_ptr()) };
            assert_eq!(
                i128_from_halves(out[0], out[1]),
                want,
                "{x} signed={signed}"
            );
        }
    }
}
