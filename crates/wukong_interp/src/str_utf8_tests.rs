//! The interpreter's UTF-8 entries (`wukong_rt_utf8_check` / `_decode` / `_count`, which the `str`
//! operations lower to) against the standard library, over byte strings held in slot memory the
//! way a program holds them.
//!
//! The pure functions are pinned exhaustively in `wukong_runtime::utf8`; what can go wrong HERE is
//! the marshalling: a byte slot is an `i8`-typed value, which the interpreter holds sign-extended
//! (an `I8` store, a runtime byte write, and — since the fix that came with these entries — a
//! string literal's static bytes), so a byte at or above 0x80 must be read back as its unsigned
//! value. Each case is run with the bytes in both representations, so a reader that used the slot
//! value directly would fail here, not in some program comparing a literal against a buffer.
use super::*;
use wukong_runtime::utf8;
use Value::{Int as I, Ptr as P};

fn interpreter<'a>(program: &'a Program, interner: &'a Interner) -> Interp<'a, 'a> {
    Interp {
        program,
        interner,
        memory: Vec::new(),
        stdout: Vec::new(),
        scratch_arena: ScratchSlots::default(),
        slots: SlotReuse::default(),
        frames: Vec::new(),
        call_args: Vec::new(),
        scratch: Vec::new(),
        vecs: Vec::new(),
        vector_peak: 0,
        data_addrs: HashMap::new(),
        function_indices: None,
        static_indices: None,
        accel: None,
        depth: 0,
    }
}

/// Lay `bytes` out from slot 1 (slot 0 is the reserved null, as in a run), sign-extended like an `I8` store or
/// zero-extended like the static bytes used to be, and return their base.
fn place(interp: &mut Interp, bytes: &[u8], sign_extended: bool) -> usize {
    interp.memory = null_reserved_memory();
    interp.memory.extend(bytes.iter().map(|&b| {
        if sign_extended {
            I(i128::from(b as i8))
        } else {
            I(i128::from(b))
        }
    }));
    interp.slots = SlotReuse::tracked();
    interp.slots.live.insert(1, bytes.len().max(1));
    1
}

/// One byte string through all three entries, in both slot representations.
#[track_caller]
fn agrees(interp: &mut Interp, bytes: &[u8]) {
    let want_fault = std::str::from_utf8(bytes).err().map(|e| utf8::Utf8Fault {
        valid_up_to: e.valid_up_to(),
        error_len: e.error_len().map(|n| n as u8),
    });
    let lossy: Vec<u32> = String::from_utf8_lossy(bytes)
        .chars()
        .map(u32::from)
        .collect();
    let n = bytes.len() as i128;
    for sign_extended in [true, false] {
        let base = place(interp, bytes, sign_extended);
        let check = interp.call_intrinsic(Intrinsic::RtUtf8Check, &[P(base), I(n)]);
        assert_eq!(
            check,
            Ok(I(i128::from(utf8::pack_fault(want_fault)))),
            "check {bytes:02X?} (sign-extended: {sign_extended})"
        );
        let count = interp.call_intrinsic(Intrinsic::RtUtf8Count, &[P(base), I(n)]);
        assert_eq!(count, Ok(I(lossy.len() as i128)), "count {bytes:02X?}");
        let mut i = 0i128;
        let mut got = Vec::new();
        while i < n {
            let Ok(I(packed)) =
                interp.call_intrinsic(Intrinsic::RtUtf8Decode, &[P(base), I(n), I(i)])
            else {
                panic!("decode {bytes:02X?} at {i}");
            };
            got.push((packed & 0xFFFF_FFFF) as u32);
            let width = packed >> 32;
            assert!(width >= 1, "width {width} at {i} of {bytes:02X?}");
            i += width;
        }
        assert_eq!(i, n, "{bytes:02X?}");
        assert_eq!(
            got, lossy,
            "decode {bytes:02X?} (sign-extended: {sign_extended})"
        );
    }
}

#[test]
fn every_byte_string_up_to_two_bytes_matches_std() {
    let program = Program::new();
    let interner = Interner::new();
    let mut interp = interpreter(&program, &interner);
    agrees(&mut interp, &[]);
    for a in 0..=255u8 {
        agrees(&mut interp, &[a]);
        for b in 0..=255u8 {
            agrees(&mut interp, &[a, b]);
        }
    }
}

#[test]
fn boundary_bytes_at_three_and_four_bytes_match_std() {
    const EDGES: &[u8] = &[
        0x00, 0x7F, 0x80, 0x8F, 0x90, 0x9F, 0xA0, 0xBF, 0xC1, 0xC2, 0xDF, 0xE0, 0xED, 0xEF, 0xF0,
        0xF4, 0xF5, 0xFF,
    ];
    let program = Program::new();
    let interner = Interner::new();
    let mut interp = interpreter(&program, &interner);
    for &a in EDGES {
        for &b in EDGES {
            for &c in EDGES {
                agrees(&mut interp, &[a, b, c]);
                for &d in EDGES {
                    agrees(&mut interp, &[a, b, c, d]);
                }
            }
        }
    }
}

#[test]
fn decoding_at_every_index_of_a_longer_string_matches_the_whole_string() {
    // The interpreter's decode reads only the four bytes from `i` (a step never looks further), so
    // it must agree with `utf8::decode_at` over the WHOLE string at every index, including
    // continuation bytes and the incomplete sequences at the end, for strings longer than the
    // window. The fragments are well-formed 1- to 4-byte characters, truncated and ill-formed
    // sequences, a surrogate, and lone continuation and lead bytes; every ordered triple of them
    // is a string of 3 to 12 bytes.
    const FRAGMENTS: &[&[u8]] = &[
        b"a",
        &[0xC3, 0xA9],
        &[0xE2, 0x82, 0xAC],
        &[0xF0, 0x9F, 0x98, 0x80],
        &[0xF0, 0x9F, 0x98],
        &[0xE2, 0x82],
        &[0xC3],
        &[0xED, 0xA0, 0x80],
        &[0x80],
        &[0xFF],
    ];
    let program = Program::new();
    let interner = Interner::new();
    let mut interp = interpreter(&program, &interner);
    for a in FRAGMENTS {
        for b in FRAGMENTS {
            for c in FRAGMENTS {
                let bytes: Vec<u8> = [*a, *b, *c].concat();
                let n = bytes.len() as i128;
                for sign_extended in [true, false] {
                    let base = place(&mut interp, &bytes, sign_extended);
                    for i in 0..bytes.len() {
                        let (cp, w) = utf8::decode_at(&bytes, i);
                        let got = interp.call_intrinsic(
                            Intrinsic::RtUtf8Decode,
                            &[P(base), I(n), I(i as i128)],
                        );
                        assert_eq!(
                            got,
                            Ok(I(i128::from(utf8::pack_char(cp, w)))),
                            "decode {bytes:02X?} at {i} (sign-extended: {sign_extended})"
                        );
                    }
                    // Past either end: U+FFFD of width 1, as natively.
                    for i in [-1, n, n + 7] {
                        let got =
                            interp.call_intrinsic(Intrinsic::RtUtf8Decode, &[P(base), I(n), I(i)]);
                        let want = utf8::pack_char(utf8::REPLACEMENT, 1);
                        assert_eq!(got, Ok(I(i128::from(want))), "{bytes:02X?} at {i}");
                    }
                }
            }
        }
    }
}

#[test]
fn a_null_or_empty_operand_is_valid_and_empty() {
    let program = Program::new();
    let interner = Interner::new();
    let mut interp = interpreter(&program, &interner);
    // Slot 0 is the reserved null, as in a real run.
    interp.memory = null_reserved_memory();
    interp.memory.push(I(0x61));
    for (ptr, n) in [(P(0), I(0)), (P(0), I(5)), (P(1), I(0)), (P(1), I(-3))] {
        assert_eq!(
            interp.call_intrinsic(Intrinsic::RtUtf8Check, &[ptr, n]),
            Ok(I(i128::from(utf8::UTF8_VALID)))
        );
        assert_eq!(
            interp.call_intrinsic(Intrinsic::RtUtf8Count, &[ptr, n]),
            Ok(I(0))
        );
    }
}

/// The failure edge of a range index or `str_from_utf8` reports through the runtime's one
/// formatter, the message the JIT and a linked executable print too.
#[test]
fn check_fail_reports_the_shared_message() {
    let program = Program::new();
    let interner = Interner::new();
    let mut interp = interpreter(&program, &interner);
    interp.memory = vec![I(0)];
    let fault = utf8::pack_fault(Some(utf8::Utf8Fault {
        valid_up_to: 2,
        error_len: Some(1),
    }));
    for (kind, a, b, want) in [
        (
            wukong_runtime::CHECK_RANGE_ORDER,
            5,
            3,
            "slice index starts at 5 but ends at 3",
        ),
        (
            wukong_runtime::CHECK_RANGE_END,
            9,
            6,
            "range end index 9 out of range for slice of length 6",
        ),
        (
            wukong_runtime::CHECK_UTF8,
            fault,
            4,
            "str_from_utf8: invalid utf-8 sequence of 1 bytes from index 2 (the len is 4)",
        ),
    ] {
        let got = interp.call_intrinsic(
            Intrinsic::RtCheckFail,
            &[
                I(i128::from(kind)),
                I(i128::from(a)),
                I(i128::from(b)),
                P(0),
            ],
        );
        assert_eq!(got, Err(want.to_string()));
    }
}
