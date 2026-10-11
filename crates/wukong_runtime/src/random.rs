//! The counter-based random number generator behind `rand_u32`, `rand_f32`, `randn_f32` and
//! `rand_fill_f32`: **Philox4x32-10** (Salmon et al., "Parallel random numbers: as easy as 1, 2, 3",
//! SC'11 — the generator of Random123, cuRAND, PyTorch's CUDA RNG and Triton's `tl.rand`).
//!
//! A counter-based generator is a pure function `(key, counter) → random bits`, so there is no
//! state to share, seed or advance: element `i` of a fill is `f(seed, ctr0 + i)` however the fill is
//! split across threads, backends or calls. That is what makes the stream reproducible under
//! `@parallel` and bit-identical between the interpreter and native code.
//!
//! **The mapping.** `rand_u32(seed, ctr)` is word 0 of the Philox4x32-10 block with key
//! `(seed mod 2³², seed div 2³²)` and counter `(ctr mod 2³², ctr div 2³², 0, 0)` — Random123's
//! `philox4x32_R(10, ctr, key)` taken verbatim, checked below against its published known-answer
//! vectors. `rand_f32` is the top 24 bits of that word scaled by 2⁻²⁴: an exact multiple of 2⁻²⁴ in
//! `[0, 1)`, every value equally likely. `randn_f32` (lowered by `wukong_mir_build` only, since it
//! needs the compiler's own `log`/`cos`) uses words 0 and 1 of the same block in a Box–Muller step.
//!
//! The scalar builtins are lowered by the compiler to plain integer MIR (a 32×32→64 multiply,
//! shifts, xors), so they need no runtime call; this module is their reference — the oracle the
//! interpreter's differential test checks the lowering against — and the body of the explicit
//! `rand_fill_f32(buf, seed, ctr0)` fill.

/// Philox4x32 round multipliers.
pub const PHILOX_M0: u32 = 0xD251_1F53;
/// Philox4x32 round multipliers.
pub const PHILOX_M1: u32 = 0xCD9E_8D57;
/// Philox4x32 key schedule (Weyl) increments: the golden ratio and √3 − 1, as 32-bit fractions.
pub const PHILOX_W0: u32 = 0x9E37_79B9;
/// Philox4x32 key schedule (Weyl) increments.
pub const PHILOX_W1: u32 = 0xBB67_AE85;

/// `2⁻²⁴`, the spacing of `rand_f32`'s outputs.
const INV_2_24: f32 = 1.0 / 16_777_216.0;

/// One Philox4x32 round: two 32×32→64 multiplies, the high halves xored with the other words and
/// the round key.
#[inline]
fn round(c: [u32; 4], k: [u32; 2]) -> [u32; 4] {
    let p0 = u64::from(PHILOX_M0) * u64::from(c[0]);
    let p1 = u64::from(PHILOX_M1) * u64::from(c[2]);
    [
        ((p1 >> 32) as u32) ^ c[1] ^ k[0],
        p1 as u32,
        ((p0 >> 32) as u32) ^ c[3] ^ k[1],
        p0 as u32,
    ]
}

/// The Philox4x32-10 block function: ten rounds, the key bumped by the Weyl increments before every
/// round after the first.
pub fn philox4x32_10(ctr: [u32; 4], key: [u32; 2]) -> [u32; 4] {
    let mut c = ctr;
    let mut k = key;
    for r in 0..10 {
        if r > 0 {
            k = [k[0].wrapping_add(PHILOX_W0), k[1].wrapping_add(PHILOX_W1)];
        }
        c = round(c, k);
    }
    c
}

/// The full 4-word block for `(seed, ctr)` under the builtins' key/counter mapping.
pub fn philox_block(seed: u64, ctr: u64) -> [u32; 4] {
    philox4x32_10(
        [ctr as u32, (ctr >> 32) as u32, 0, 0],
        [seed as u32, (seed >> 32) as u32],
    )
}

/// `rand_u32(seed, ctr)`: word 0 of the block.
pub fn rand_u32(seed: u64, ctr: u64) -> u32 {
    philox_block(seed, ctr)[0]
}

/// `rand_f32(seed, ctr)`: `(rand_u32 >> 8) · 2⁻²⁴`, exact, in `[0, 1)`.
pub fn rand_f32(seed: u64, ctr: u64) -> f32 {
    (rand_u32(seed, ctr) >> 8) as f32 * INV_2_24
}

/// `wukong_rand_fill_f32(out, n, seed, ctr0)` — `out[i] = rand_f32(seed, ctr0 + i)` for
/// `i in 0..n` (the counter wraps modulo 2⁶⁴, as the `i64` sum `ctr0 + i` does in the program).
/// The body of the `rand_fill_f32(buf, seed, ctr0)` builtin; the interpreter calls the same
/// [`rand_f32`] slot by slot, and a fixture checks every lane against the scalar builtin.
///
/// # Safety
/// `out` must address `n` writable `f32`s (it is not touched when `n <= 0`).
#[no_mangle]
pub unsafe extern "C" fn wukong_rand_fill_f32(out: *mut f32, n: i64, seed: i64, ctr0: i64) {
    // P14.5: debug-build contract checks (the `# Safety` section above).
    crate::safety::debug_extent(out, n);
    if n <= 0 {
        return;
    }
    // SAFETY: forwarded from this function's contract.
    let out = unsafe { std::slice::from_raw_parts_mut(out, n as usize) };
    for (i, x) in out.iter_mut().enumerate() {
        *x = rand_f32(seed as u64, (ctr0 as u64).wrapping_add(i as u64));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Random123's `kat_vectors` entries for `philox4x32 10`: (counter, key, expected output).
    #[test]
    fn philox_matches_the_random123_known_answers() {
        let kat: [([u32; 4], [u32; 2], [u32; 4]); 3] = [
            (
                [0, 0, 0, 0],
                [0, 0],
                [0x6627_e8d5, 0xe169_c58d, 0xbc57_ac4c, 0x9b00_dbd8],
            ),
            (
                [u32::MAX; 4],
                [u32::MAX; 2],
                [0x408f_276d, 0x41c8_3b0e, 0xa20b_c7c6, 0x6d54_51fd],
            ),
            (
                [0x243f_6a88, 0x85a3_08d3, 0x1319_8a2e, 0x0370_7344],
                [0xa409_3822, 0x299f_31d0],
                [0xd16c_fe09, 0x94fd_cceb, 0x5001_e420, 0x2412_6ea1],
            ),
        ];
        for (ctr, key, want) in kat {
            assert_eq!(philox4x32_10(ctr, key), want, "ctr {ctr:x?} key {key:x?}");
        }
    }

    /// The builtin mapping: seed/counter split into low and high words. Expected words are from an
    /// independent Python transcription of the same block function, which reproduces all three
    /// known-answer vectors above.
    #[test]
    fn builtin_mapping_and_uniform_scaling() {
        assert_eq!(rand_u32(0, 0), 0x6627_e8d5);
        assert_eq!(rand_u32(0, 1), 0xf8e4_cca4);
        assert_eq!(rand_u32(1, 0), 0xe3e8_0670);
        assert_eq!(rand_u32(42, 7), 0x5bbd_83b1);
        assert_eq!(
            rand_u32(0xDEAD_BEEF_CAFE_F00D, 0x1234_5678_9ABC_DEF0),
            0xdecb_b6f2
        );
        assert_eq!(rand_u32(u64::MAX, 3), 0x8e8f_d014);
        // rand_f32 is the top 24 bits over 2^24, exactly.
        assert_eq!(rand_f32(0, 0), 6_694_888.0 / 16_777_216.0);
        assert_eq!(rand_f32(42, 7), 6_012_291.0 / 16_777_216.0);
        for c in 0..1000u64 {
            let x = rand_f32(9, c);
            assert!((0.0..1.0).contains(&x), "{x}");
            assert_eq!(x * 16_777_216.0, (x * 16_777_216.0).trunc());
        }
    }

    #[test]
    fn fill_is_the_scalar_stream_and_wraps_the_counter() {
        let mut out = [0f32; 5];
        // SAFETY: `out` holds 5 f32 and the call fills exactly 5 (the negative arg is a stream offset, not a length).
        unsafe { wukong_rand_fill_f32(out.as_mut_ptr(), 5, 42, -2) };
        for (i, &x) in out.iter().enumerate() {
            let ctr = (-2i64 as u64).wrapping_add(i as u64);
            assert_eq!(x.to_bits(), rand_f32(42, ctr).to_bits(), "lane {i}");
        }
        // Lane 2 is counter 0 after the wrap.
        assert_eq!(out[2], rand_f32(42, 0));
        let mut untouched = [7f32; 2];
        // SAFETY: n == 0 is a no-op; `untouched` holds 2 f32.
        unsafe { wukong_rand_fill_f32(untouched.as_mut_ptr(), 0, 1, 1) };
        // SAFETY: negative n is a no-op; `untouched` holds 2 f32.
        unsafe { wukong_rand_fill_f32(untouched.as_mut_ptr(), -3, 1, 1) };
        assert_eq!(untouched, [7.0; 2]);
    }
}
