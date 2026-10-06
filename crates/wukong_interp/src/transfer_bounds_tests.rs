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
fn transpose_preserves_exact_conversion_permutation_and_snapshots() {
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
        for out in [0, 1, 6] {
            interp.memory = vec![Value::Unit; 12];
            interp.memory[..6].copy_from_slice(&input);
            interp.intrinsic(name, &[P(0), P(out), I(2), I(3)]).unwrap();
            let mut expected = [input[0], input[3], input[1], input[4], input[2], input[5]];
            if name.contains("f32") {
                for v in &mut expected {
                    *v = F(v.as_float() as f32 as f64);
                }
            }
            exact_values(&interp.memory[out..out + 6], &expected);
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
        ("wukong_rt_read_u8", vec![0, 255], [I(0), I(255)], 1),
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
