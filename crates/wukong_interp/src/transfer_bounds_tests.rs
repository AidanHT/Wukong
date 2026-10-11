//! Tiny interpreter arenas and owned small files only; malformed buffers never reach native code.
use super::*;
use std::io::{Cursor, Read, Write};
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use Value::{Float as F, Int as I, Ptr as P};

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

fn exact_values(actual: &[Value], expected: &[Value]) {
    assert_eq!(actual.len(), expected.len());
    for (a, b) in actual.iter().zip(expected) {
        if let (F(a), F(b)) = (a, b) {
            assert_eq!(a.to_bits(), b.to_bits());
        } else {
            assert_eq!(a, b);
        }
    }
}

#[test]
fn transpose_checks_extents_and_ranges_before_marshalling() {
    let program = Program::new();
    let interner = Interner::new();
    let mut interp = interpreter(&program, &interner);
    for name in [
        "wukong_transpose_f32",
        "wukong_transpose_f32_parallel",
        "wukong_transpose_u16",
        "wukong_transpose_u16_parallel",
    ] {
        for args in [
            [P(0), P(6), I(i64::MAX.into()), I(4)],
            [P(0), P(6), I(1), I(i64::MAX.into())],
            [P(7), P(6), I(2), I(3)],
            [P(0), P(7), I(2), I(3)],
            [P(usize::MAX), P(6), I(2), I(3)],
            [P(0), P(usize::MAX), I(2), I(3)],
        ] {
            interp.memory = vec![Value::Unit; 12];
            let before = interp.memory.clone();
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                interp.intrinsic(name, &args)
            }));
            let error = result
                .expect("transpose must diagnose, not panic")
                .unwrap_err();
            assert!(
                error.contains("extent overflow") || error.contains("out of bounds"),
                "{name}: {error}"
            );
            exact_values(&interp.memory, &before);
        }
        for dim in [2, 3] {
            for empty in [0, -1] {
                let mut args = [P(usize::MAX), P(usize::MAX), I(2), I(3)];
                args[dim] = I(empty);
                interp.memory = vec![Value::Unit; 12];
                assert_eq!(interp.intrinsic(name, &args).unwrap(), Value::Unit);
                assert_eq!(interp.memory, vec![Value::Unit; 12]);
            }
        }
        for bad in [P(0), Value::VecRef(0), Value::Unit] {
            interp.memory = vec![F(1.0); 12];
            interp.memory[5] = bad;
            let before = interp.memory.clone();
            assert!(interp
                .intrinsic(name, &[P(0), P(6), I(2), I(3)])
                .unwrap_err()
                .contains("floating buffer"));
            exact_values(&interp.memory, &before);
        }
    }
}

#[test]
fn transpose_preserves_exact_conversion_permutation_and_refuses_overlap() {
    let program = Program::new();
    let interner = Interner::new();
    let mut interp = interpreter(&program, &interner);
    for name in [
        "wukong_transpose_f32",
        "wukong_transpose_f32_parallel",
        "wukong_transpose_u16",
        "wukong_transpose_u16_parallel",
    ] {
        let input = [I(16_777_217), F(-0.0), F(1.5), I(-2), F(2.0), F(3.0)];
        interp.memory = vec![Value::Unit; 12];
        interp.memory[..6].copy_from_slice(&input);
        interp.intrinsic(name, &[P(0), P(6), I(2), I(3)]).unwrap();
        let mut expected = [input[0], input[3], input[1], input[4], input[2], input[5]];
        if name.contains("f32") {
            for v in &mut expected {
                *v = F(v.as_float() as f32 as f64);
            }
        }
        exact_values(&interp.memory[6..12], &expected);
        // In place (`dst == src`) and shifted (`dst == src + 1`) are outside transpose.rs's
        // contract, and the source nest, the blocked kernel and a snapshot give three different
        // answers there. The compiler versions the dispatch so neither reaches this arm; a call
        // that does is refused with memory untouched rather than answered with the snapshot.
        for out in [0, 1] {
            interp.memory = vec![Value::Unit; 12];
            interp.memory[..6].copy_from_slice(&input);
            let before = interp.memory.clone();
            let error = interp
                .intrinsic(name, &[P(0), P(out), I(2), I(3)])
                .unwrap_err();
            assert!(error.contains("overlap"), "{name} dst {out}: {error}");
            exact_values(&interp.memory, &before);
        }
    }
}

struct Fixture(PathBuf);
impl Fixture {
    fn new(bytes: &[u8]) -> Self {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let path = std::env::temp_dir().join(format!(
            "wukong-transfer-bounds-{}-{}.bin",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)
            .unwrap();
        file.write_all(bytes).unwrap();
        Self(path)
    }
    fn arena(&self) -> Vec<Value> {
        let mut values: Vec<_> = self
            .0
            .to_str()
            .unwrap()
            .bytes()
            .map(|b| I(b.into()))
            .collect();
        values.push(I(0));
        values
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

#[test]
fn file_reads_validate_actual_destination_and_preserve_decoding_and_clamps() {
    let program = Program::new();
    let interner = Interner::new();
    let mut interp = interpreter(&program, &interner);
    let cases = [
        (
            "wukong_rt_read_f32",
            [-0.0f32, 1.5]
                .into_iter()
                .flat_map(f32::to_le_bytes)
                .collect::<Vec<_>>(),
            [F(-0.0), F(1.5)],
            4,
        ),
        (
            "wukong_rt_read_f64",
            [-0.0f64, 1.0000000000000002]
                .into_iter()
                .flat_map(f64::to_le_bytes)
                .collect(),
            [F(-0.0), F(1.0000000000000002)],
            8,
        ),
        (
            "wukong_rt_read_i32",
            [i32::MIN, 17]
                .into_iter()
                .flat_map(i32::to_le_bytes)
                .collect(),
            [I(i32::MIN.into()), I(17)],
            4,
        ),
        (
            "wukong_rt_read_i64",
            [i64::MIN, i64::MAX]
                .into_iter()
                .flat_map(i64::to_le_bytes)
                .collect(),
            [I(i64::MIN.into()), I(i64::MAX.into())],
            8,
        ),
        // A `u8` slot holds the byte sign-extended, the canonical integer slot value.
        ("wukong_rt_read_u8", vec![0, 255], [I(0), I(-1)], 1),
        ("wukong_rt_read_i8", vec![128, 255], [I(-128), I(-1)], 1),
    ];
    for (name, mut bytes, expected, width) in cases {
        if width > 1 {
            bytes.push(0xab);
        } // Metadata floor ignores an incomplete trailing element.
        let fixture = Fixture::new(&bytes);
        for requested in [1, 2, i128::from(i64::MAX)] {
            interp.memory = fixture.arena();
            let base = interp.memory.len();
            interp.memory.extend([I(99); 3]);
            let n = if requested == 1 { 1 } else { 2 };
            assert_eq!(
                interp
                    .intrinsic(name, &[P(0), P(base), I(requested)])
                    .unwrap(),
                I(n as i128)
            );
            exact_values(&interp.memory[base..base + n], &expected[..n]);
            assert_eq!(&interp.memory[base + n..], &vec![I(99); 3 - n]);
        }
        for empty in [0, -1] {
            interp.memory = fixture.arena();
            let before = interp.memory.clone();
            assert_eq!(
                interp
                    .intrinsic(name, &[P(0), P(usize::MAX), I(empty)])
                    .unwrap(),
                I(0)
            );
            exact_values(&interp.memory, &before);
        }
        for overflow in [false, true] {
            interp.memory = fixture.arena();
            let base = interp.memory.len();
            interp.memory.push(I(99));
            let before = interp.memory.clone();
            let output = if overflow { usize::MAX } else { base };
            let error = interp
                .intrinsic(name, &[P(0), P(output), I(i64::MAX.into())])
                .unwrap_err();
            assert!(error.contains("data out of bounds"), "{name}: {error}");
            exact_values(&interp.memory, &before);
        }
        // Delete only this test's fixture: failed open wins even with an invalid destination.
        std::fs::remove_file(&fixture.0).unwrap();
        interp.memory = fixture.arena();
        let before = interp.memory.clone();
        assert_eq!(
            interp
                .intrinsic(name, &[P(0), P(usize::MAX), I(2)])
                .unwrap(),
            I(-1)
        );
        exact_values(&interp.memory, &before);
    }
    let empty = Fixture::new(&[]);
    interp.memory = empty.arena();
    assert_eq!(
        interp
            .intrinsic(
                "wukong_rt_read_f64",
                &[P(0), P(usize::MAX), I(i64::MAX.into())]
            )
            .unwrap(),
        I(0)
    );
}

#[test]
fn file_read_errors_do_not_modify_memory_and_bounds_precede_reads() {
    struct Fail;
    impl Read for Fail {
        fn read(&mut self, _: &mut [u8]) -> std::io::Result<usize> {
            Err(std::io::Error::other("controlled test error"))
        }
    }
    struct NeverRead;
    impl Read for NeverRead {
        fn read(&mut self, _: &mut [u8]) -> std::io::Result<usize> {
            panic!("invalid/empty destination must be handled before reading")
        }
    }
    let mut memory = [I(77); 2];
    assert_eq!(
        read_file_elements(&mut memory, &mut Fail, 0, 2, 4, "wukong_rt_read_i32").unwrap(),
        I(-2)
    );
    assert_eq!(memory, [I(77); 2]);
    assert_eq!(
        read_file_elements(
            &mut memory,
            &mut Cursor::new([1, 2, 3, 4, 5]),
            0,
            2,
            4,
            "wukong_rt_read_i32"
        )
        .unwrap(),
        I(-2)
    );
    assert_eq!(memory, [I(77); 2]);
    for base in [1, usize::MAX] {
        assert!(read_file_elements(
            &mut memory,
            &mut NeverRead,
            base,
            2,
            4,
            "wukong_rt_read_i32"
        )
        .unwrap_err()
        .contains("data out of bounds"));
        assert_eq!(memory, [I(77); 2]);
    }
    assert!(read_file_elements(
        &mut memory,
        &mut NeverRead,
        0,
        2,
        usize::MAX,
        "wukong_rt_read_i32"
    )
    .unwrap_err()
    .contains("byte extent overflow"));
    assert_eq!(
        read_file_elements(
            &mut memory,
            &mut NeverRead,
            usize::MAX,
            0,
            4,
            "wukong_rt_read_i32"
        )
        .unwrap(),
        I(0)
    );
}

#[test]
fn file_read_path_preserves_lossy_utf8_and_low_byte_rules() {
    for (bytes, want) in [
        (&b"abc"[..], "abc"),
        (&[0x66, 0x80, 0x6f][..], "f\u{fffd}o"),
        (&[0xe2, 0x82][..], "\u{fffd}"),
        (&[0xf0, 0x28, 0x8c, 0x28][..], "\u{fffd}(\u{fffd}("),
        (&[0xc0, 0xaf][..], "\u{fffd}\u{fffd}"),
        (&[0xf0, 0x9f, 0x98, 0x80][..], "\u{1f600}"),
    ] {
        let values: Vec<_> = bytes.iter().map(|&b| I(i128::from(b) + 256)).collect();
        assert_eq!(file_read_path(&values, 0).unwrap(), want);
    }
    assert_eq!(file_read_path(&[I(65), I(256), I(66)], 0).unwrap(), "A");
    assert_eq!(file_read_path(&[I(65)], usize::MAX).unwrap(), "");
}

#[test]
fn embedding_checks_extents_and_selected_rows_without_snapshotting() {
    let program = Program::new();
    let interner = Interner::new();
    let mut interp = interpreter(&program, &interner);
    for name in ["wukong_embedding_f32", "wukong_embedding_f32_parallel"] {
        for args in [
            [P(0), P(4), P(8), I(i64::MAX.into()), I(4), I(2)],
            [P(0), P(4), P(8), I(1), I(i64::MAX.into()), I(2)],
            [P(11), P(4), P(8), I(1), I(2), I(2)],
            [P(0), P(4), P(11), I(2), I(2), I(2)],
            [P(usize::MAX), P(4), P(8), I(1), I(2), I(2)],
            [P(0), P(4), P(usize::MAX), I(1), I(2), I(2)],
            [P(0), P(usize::MAX), P(8), I(1), I(2), I(2)],
        ] {
            interp.memory = vec![I(0); 12];
            let before = interp.memory.clone();
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                interp.intrinsic(name, &args)
            }));
            let error = result
                .expect("embedding must diagnose rather than panic")
                .unwrap_err();
            assert!(
                error.contains("out of bounds") || error.contains("extent overflow"),
                "{name}: {error}"
            );
            exact_values(&interp.memory, &before);
        }
        interp.memory = vec![I(0); 12];
        interp.memory[8] = I(i128::from(i64::MAX) - 1);
        let before = interp.memory.clone();
        assert!(interp
            .intrinsic(name, &[P(0), P(4), P(8), I(1), I(4), I(i64::MAX.into())])
            .unwrap_err()
            .contains("extent overflow"));
        exact_values(&interp.memory, &before);
        // Invalid IDs must not validate an unused weight pointer or an entire declared table.
        for id in [-1, 2] {
            interp.memory = vec![I(id), Value::Unit, Value::Unit];
            interp
                .intrinsic(name, &[P(1), P(usize::MAX), P(0), I(1), I(2), I(2)])
                .unwrap();
            exact_values(&interp.memory[1..], &[F(0.0), F(0.0)]);
        }
        interp.memory = vec![I(0), I(16_777_217), Value::Unit];
        interp
            .intrinsic(name, &[P(2), P(1), P(0), I(1), I(1), I(i64::MAX.into())])
            .unwrap();
        exact_values(&interp.memory[2..], &[F(16_777_216.0)]);
        for dim in [3, 4] {
            for empty in [0, -1] {
                let mut args = [
                    P(usize::MAX),
                    P(usize::MAX),
                    P(usize::MAX),
                    I(1),
                    I(1),
                    I(2),
                ];
                args[dim] = I(empty);
                interp.memory = vec![Value::Unit; 2];
                assert_eq!(interp.intrinsic(name, &args).unwrap(), Value::Unit);
                assert_eq!(interp.memory, vec![Value::Unit; 2]);
            }
        }
        // Earlier output changes a later ID; the second ID must be read live.
        interp.memory = vec![I(1), I(0), F(3.0), F(4.0)];
        interp
            .intrinsic(name, &[P(1), P(2), P(0), I(2), I(1), I(2)])
            .unwrap();
        exact_values(&interp.memory, &[I(1), F(4.0), F(0.0), F(4.0)]);
        // Earlier row changes later weights, and shifted overlap changes later lanes of one row.
        interp.memory = vec![F(1.0), F(2.0), F(3.0), F(4.0), I(1), I(0)];
        interp
            .intrinsic(name, &[P(0), P(0), P(4), I(2), I(2), I(2)])
            .unwrap();
        exact_values(&interp.memory[..4], &[F(3.0), F(4.0), F(3.0), F(4.0)]);
        interp.memory = vec![F(1.0), F(2.0), F(3.0), Value::Unit, I(0)];
        interp
            .intrinsic(name, &[P(1), P(0), P(4), I(1), I(3), I(1)])
            .unwrap();
        exact_values(&interp.memory[1..4], &[F(1.0); 3]);
        // A later invalid selected row retains the already-written earlier row.
        interp.memory = vec![Value::Unit, Value::Unit, I(0), I(1), F(7.0)];
        assert!(interp
            .intrinsic(name, &[P(0), P(4), P(2), I(2), I(1), I(2)])
            .unwrap_err()
            .contains("out of bounds"));
        exact_values(&interp.memory[..2], &[F(7.0), Value::Unit]);
        // A later lane's type error preserves the written prefix and untouched suffix.
        interp.memory = vec![Value::Unit, I(91), I(92), I(0), F(7.0), Value::Unit, F(9.0)];
        assert!(interp
            .intrinsic(name, &[P(0), P(4), P(3), I(1), I(3), I(1)])
            .unwrap_err()
            .contains("floating buffer"));
        exact_values(
            &interp.memory,
            &[F(7.0), I(91), I(92), I(0), F(7.0), Value::Unit, F(9.0)],
        );
        for bad in [P(0), Value::VecRef(0), Value::Unit] {
            interp.memory = vec![Value::Unit, I(0), bad];
            let before = interp.memory.clone();
            assert!(interp
                .intrinsic(name, &[P(0), P(2), P(1), I(1), I(1), I(1)])
                .unwrap_err()
                .contains("floating buffer"));
            exact_values(&interp.memory, &before);
        }
    }
}

/// A tracked interpreter (every public entry point's) over `memory`, each `(base, len)` of
/// `blocks` registered as its own allocation.
fn tracked<'a>(
    program: &'a Program,
    interner: &'a Interner,
    memory: Vec<Value>,
    blocks: &[(usize, usize)],
) -> Interp<'a, 'a> {
    let mut interp = interpreter(program, interner);
    interp.memory = memory;
    interp.slots = SlotReuse::tracked();
    for &(base, len) in blocks {
        interp.slots.live.insert(base, len);
    }
    interp
}

/// Transpose: exact-fit allocations run; a source or destination that fits the arena but runs
/// one slot into the adjacent allocation is refused by argument index with memory untouched; an
/// absurd extent is still the overflow diagnostic.
#[test]
fn transpose_operands_must_stay_inside_their_allocations() {
    let program = Program::new();
    let interner = Interner::new();
    let input = [F(1.0), F(2.0), F(3.0), F(4.0), F(5.0), F(6.0)];
    let mut memory = vec![Value::Unit; 12];
    memory[..6].copy_from_slice(&input);
    for name in [
        "wukong_transpose_f32",
        "wukong_transpose_f32_parallel",
        "wukong_transpose_u16",
        "wukong_transpose_u16_parallel",
    ] {
        let args = [P(0), P(6), I(2), I(3)];
        let mut interp = tracked(&program, &interner, memory.clone(), &[(0, 6), (6, 6)]);
        interp.intrinsic(name, &args).unwrap();
        let expected = [input[0], input[3], input[1], input[4], input[2], input[5]];
        exact_values(&interp.memory[6..12], &expected);
        for (operand, blocks) in [
            (0, [(0, 5), (5, 1), (6, 6)]),
            (1, [(0, 6), (6, 5), (11, 1)]),
        ] {
            let mut interp = tracked(&program, &interner, memory.clone(), &blocks);
            let error = interp.intrinsic(name, &args).unwrap_err();
            assert_eq!(
                error,
                format!("{name}: operand {operand} extends past its allocation")
            );
            exact_values(&interp.memory, &memory);
        }
        let mut interp = tracked(&program, &interner, memory.clone(), &[(0, 6), (6, 6)]);
        let error = interp
            .intrinsic(name, &[P(0), P(6), I(i64::MAX.into()), I(4)])
            .unwrap_err();
        assert!(error.contains("extent overflow"), "{name}: {error}");
        exact_values(&interp.memory, &memory);
    }
}

/// Embedding: the selected weight row must lie inside the weight table's allocation — a row that
/// runs into the next allocation is refused (earlier rows stay written, the live-gather rule);
/// the ids and the output are checked whole, by argument index.
#[test]
fn embedding_rows_must_stay_inside_the_weight_allocation() {
    let program = Program::new();
    let interner = Interner::new();
    for name in ["wukong_embedding_f32", "wukong_embedding_f32_parallel"] {
        // out [0, 4) = 2 rows of h = 2; weight [4, 8) = 2 rows; ids [8, 10) = [1, 0].
        let mut memory = vec![Value::Unit; 10];
        memory[4..8].copy_from_slice(&[F(1.0), F(2.0), F(3.0), F(4.0)]);
        memory[8..10].copy_from_slice(&[I(1), I(0)]);
        let args = [P(0), P(4), P(8), I(2), I(2), I(2)];
        let exact = [(0, 4), (4, 4), (8, 2)];
        let mut interp = tracked(&program, &interner, memory.clone(), &exact);
        interp.intrinsic(name, &args).unwrap();
        exact_values(&interp.memory[..4], &[F(3.0), F(4.0), F(1.0), F(2.0)]);
        // The weight table one slot short: row 1 (the first selected) crosses into its neighbour.
        let mut interp = tracked(
            &program,
            &interner,
            memory.clone(),
            &[(0, 4), (4, 3), (7, 1), (8, 2)],
        );
        assert_eq!(
            interp.intrinsic(name, &args).unwrap_err(),
            "embedding weight: operand 1 extends past its allocation"
        );
        exact_values(&interp.memory, &memory);
        // The output, or the ids, one slot short: refused before the first row is written.
        for (operand, blocks) in [
            (0, [(0, 3), (3, 1), (4, 4), (8, 2)]),
            (2, [(0, 4), (4, 4), (8, 1), (9, 1)]),
        ] {
            let mut interp = tracked(&program, &interner, memory.clone(), &blocks);
            assert_eq!(
                interp.intrinsic(name, &args).unwrap_err(),
                format!("{name}: operand {operand} extends past its allocation")
            );
            exact_values(&interp.memory, &memory);
        }
    }
}

/// A file read lands only inside the destination's allocation: a request that fits the arena but
/// not the destination is refused before a byte is decoded; an exact fit reads.
#[test]
fn file_reads_stay_inside_the_destination_allocation() {
    let program = Program::new();
    let interner = Interner::new();
    let fixture = Fixture::new(
        &[1.5f32, -2.0, 4.0]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect::<Vec<_>>(),
    );
    let path = fixture.arena();
    let base = path.len();
    let mut memory = path.clone();
    memory.extend([I(99); 4]);
    // The path string, then a 2-slot destination followed by a 2-slot neighbour.
    let blocks = [(0, base), (base, 2), (base + 2, 2)];
    let mut interp = tracked(&program, &interner, memory.clone(), &blocks);
    assert_eq!(
        interp
            .intrinsic("wukong_rt_read_f32", &[P(0), P(base), I(2)])
            .unwrap(),
        I(2)
    );
    exact_values(
        &interp.memory[base..base + 4],
        &[F(1.5), F(-2.0), I(99), I(99)],
    );
    let mut interp = tracked(&program, &interner, memory.clone(), &blocks);
    assert_eq!(
        interp
            .intrinsic("wukong_rt_read_f32", &[P(0), P(base), I(3)])
            .unwrap_err(),
        "wukong_rt_read_f32: operand 1 extends past its allocation"
    );
    exact_values(&interp.memory, &memory);
}
