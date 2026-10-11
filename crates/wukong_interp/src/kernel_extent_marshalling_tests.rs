//! Direct interpreter calls only: malformed buffers must never reach native runtime calls.
use super::*;
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

struct Case {
    name: String,
    args: Vec<Value>,
    dims: Vec<usize>,
    // Argument indices and complete lengths of every used buffer, including output.
    ranges: Vec<(usize, usize)>,
    floats: Vec<usize>,
    expected: Vec<Value>,
}

fn memory() -> Vec<Value> {
    let mut memory = vec![Value::Unit; 24];
    memory[..4].copy_from_slice(&[F(2.0), I(3), F(4.0), I(5)]);
    memory[4..8].copy_from_slice(&[I(6), F(7.0), I(8), F(9.0)]);
    memory[16..18].copy_from_slice(&[F(2.0), I(4)]);
    memory[20..22].copy_from_slice(&[I(1), F(-2.0)]);
    memory
}

fn cases() -> Vec<Case> {
    let mut cases = Vec::new();
    for parallel in [false, true] {
        let suffix = if parallel { "_parallel" } else { "" };
        for family in ["wukong_sgemv", "wukong_sgemv_alpha", "wukong_sgevm_f32"] {
            let mut args = vec![P(0), P(4), P(12), I(2), I(2)];
            if family != "wukong_sgemv" {
                args.push(F(0.5));
            }
            let gevm = family == "wukong_sgevm_f32";
            cases.push(Case {
                name: format!("{family}{suffix}"),
                args,
                dims: vec![3, 4],
                ranges: vec![
                    (0, if gevm { 2 } else { 4 }),
                    (1, if gevm { 4 } else { 2 }),
                    (2, 2),
                ],
                floats: vec![0, 4],
                expected: if gevm {
                    vec![F(18.0), F(20.5)]
                } else if family.ends_with("alpha") {
                    vec![F(16.5), F(29.5)]
                } else {
                    vec![F(33.0), F(59.0)]
                },
            });
        }
        for deq in [false, true] {
            let family = if deq {
                "wukong_i8gemm_nt_deq"
            } else {
                "wukong_i8gemm_nt"
            };
            let mut args = vec![P(0), P(4), P(12), I(2), I(2), I(2)];
            let mut ranges = vec![(0, 4), (1, 4), (2, 4)];
            if deq {
                args.extend([F(0.5), P(16), P(20), I(0)]);
                ranges.extend([(7, 2), (8, 2)]);
            }
            cases.push(Case {
                name: format!("{family}{suffix}"),
                args,
                dims: vec![3, 4, 5],
                ranges,
                floats: if deq { vec![16, 20] } else { vec![] },
                expected: if deq {
                    vec![F(34.0), F(84.0), F(60.0), F(152.0)]
                } else {
                    vec![I(33), I(43), I(59), I(77)]
                },
            });
        }
        for perchan in [false, true] {
            for width in [
                wukong_runtime::DQ_I8,
                wukong_runtime::DQ_U8,
                wukong_runtime::DQ_I32,
            ] {
                let family = if perchan {
                    "wukong_dequant_perchan_f32"
                } else {
                    "wukong_dequant_f32"
                };
                let args = if perchan {
                    vec![P(0), P(12), I(2), I(2), P(16), I(width.into())]
                } else {
                    vec![P(0), P(12), I(4), F(0.5), I(width.into())]
                };
                cases.push(Case {
                    name: format!("{family}{suffix}"),
                    args,
                    dims: if perchan { vec![2, 3] } else { vec![2] },
                    ranges: if perchan {
                        vec![(0, 4), (1, 4), (4, 2)]
                    } else {
                        vec![(0, 4), (1, 4)]
                    },
                    floats: if perchan { vec![16] } else { vec![] },
                    expected: if perchan {
                        vec![F(4.0), F(12.0), F(8.0), F(20.0)]
                    } else {
                        vec![F(1.0), F(1.5), F(2.0), F(2.5)]
                    },
                });
            }
        }
        for precision in ["bf16", "f16"] {
            for layout in ["nt", "tn", "nt_epi"] {
                let epi = layout == "nt_epi";
                let mut args = vec![P(0), P(4), P(12), I(2), I(2), I(2), I(0)];
                let mut ranges = vec![(0, 4), (1, 4), (2, 4)];
                let mut floats = vec![0, 4];
                if epi {
                    args.extend([P(20), I(0)]);
                    ranges.push((7, 2));
                    floats.push(20);
                }
                cases.push(Case {
                    name: format!("wukong_sgemm_{precision}_{layout}{suffix}"),
                    args,
                    dims: vec![3, 4, 5],
                    ranges,
                    floats,
                    expected: if epi {
                        vec![F(34.0), F(41.0), F(60.0), F(75.0)]
                    } else if layout == "tn" {
                        vec![F(44.0), F(50.0), F(58.0), F(66.0)]
                    } else {
                        vec![F(33.0), F(43.0), F(59.0), F(77.0)]
                    },
                });
            }
        }
    }
    cases
}

#[test]
fn extents_reject_huge_dimensions_without_panic_or_writes() {
    let program = Program::new();
    let interner = Interner::new();
    let mut interp = interpreter(&program, &interner);
    for case in cases() {
        for &dim in &case.dims {
            for huge in [i128::from(i64::MAX), i128::from(i64::MAX) / 2 + 1] {
                let mut args = case.args.clone();
                for &other in &case.dims {
                    args[other] = I(4);
                }
                args[dim] = I(huge);
                interp.memory = memory();
                let before = interp.memory.clone();
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    interp.intrinsic(&case.name, &args)
                }));
                let error = result
                    .expect("extent must diagnose rather than panic")
                    .unwrap_err();
                assert!(
                    error.contains("extent overflow") || error.contains("out of bounds"),
                    "{} dim {dim}: {error}",
                    case.name
                );
                assert_eq!(interp.memory, before, "{} dim {dim}", case.name);
            }
        }
    }
}

#[test]
fn extents_validate_every_range_before_scratch_marshalling() {
    let program = Program::new();
    let interner = Interner::new();
    let mut interp = interpreter(&program, &interner);
    for case in cases() {
        for &(arg, len) in &case.ranges {
            for base in [24 - len + 1, usize::MAX] {
                let mut args = case.args.clone();
                args[arg] = P(base);
                // Invalid float values in earlier operands must not hide a later invalid range.
                interp.memory = vec![Value::Unit; 24];
                let before = interp.memory.clone();
                let error = interp.intrinsic(&case.name, &args).unwrap_err();
                assert!(
                    error.contains("out of bounds"),
                    "{} arg {arg}: {error}",
                    case.name
                );
                assert_eq!(interp.memory, before, "{} arg {arg}", case.name);
            }
        }
    }
}

#[test]
fn extents_preserve_empty_noops_exact_values_and_strict_float_inputs() {
    let program = Program::new();
    let interner = Interner::new();
    let mut interp = interpreter(&program, &interner);
    for case in cases() {
        interp.memory = memory();
        interp.intrinsic(&case.name, &case.args).unwrap();
        assert_eq!(
            &interp.memory[12..12 + case.expected.len()],
            &case.expected,
            "{} {:?}",
            case.name,
            case.args
        );
        for &dim in &case.dims {
            for empty in [0, -1] {
                let mut args = case.args.clone();
                args[dim] = I(empty);
                for &(arg, _) in &case.ranges {
                    args[arg] = P(usize::MAX);
                }
                interp.memory = vec![Value::Unit; 24];
                let before = interp.memory.clone();
                assert_eq!(
                    interp.intrinsic(&case.name, &args).unwrap(),
                    Value::Unit,
                    "{}",
                    case.name
                );
                assert_eq!(interp.memory, before, "{} empty", case.name);
            }
        }
        for &slot in &case.floats {
            for bad in [P(0), Value::VecRef(0), Value::Unit] {
                interp.memory = memory();
                interp.memory[slot] = bad;
                let before = interp.memory.clone();
                let error = interp.intrinsic(&case.name, &case.args).unwrap_err();
                assert!(
                    error.contains("floating buffer contains"),
                    "{}: {error}",
                    case.name
                );
                assert_eq!(interp.memory, before, "{} bad float", case.name);
            }
        }
    }
}

#[test]
fn extents_preserve_lowp_beta_bias_rounding_and_refuse_aliased_output() {
    let program = Program::new();
    let interner = Interner::new();
    let mut interp = interpreter(&program, &interner);
    for case in cases()
        .into_iter()
        .filter(|c| c.name.contains("sgemm_bf16") || c.name.contains("sgemm_f16"))
    {
        let mut args = case.args.clone();
        args[6] = I(1);
        interp.memory = memory();
        interp.memory[12..16].fill(F(2.0));
        interp.intrinsic(&case.name, &args).unwrap();
        let expected: Vec<_> = case
            .expected
            .iter()
            .map(|v| F(v.as_float() + 2.0))
            .collect();
        assert_eq!(&interp.memory[12..16], &expected, "{} beta", case.name);
        for bad in [P(0), Value::VecRef(0), Value::Unit] {
            interp.memory = memory();
            interp.memory[12..16].fill(bad);
            let before = interp.memory.clone();
            assert!(interp
                .intrinsic(&case.name, &args)
                .unwrap_err()
                .contains("floating buffer contains"));
            assert_eq!(interp.memory, before);
            args[6] = I(0);
            interp.intrinsic(&case.name, &args).unwrap();
            assert_eq!(&interp.memory[12..16], &case.expected);
            args[6] = I(1);
        }
        // Single-element exact expected conversion: bf16 rounds this halfway value to 1;
        // f16 represents it exactly. No runtime-produced oracle or tolerance. `C` is disjoint from
        // every input: an output overlapping `A` or the bias is outside the kernel's contract and is
        // refused below, so this no longer doubles as a snapshot test (the bias may still share
        // `A`'s slots, because both are only read).
        args[3..6].fill(I(1));
        args[6] = I(0);
        args[2] = P(12);
        let epi = case.name.contains("_epi");
        for bias in [I(0), P(0)] {
            if epi {
                args[7] = bias;
            }
            interp.memory = memory();
            interp.memory[0] = F(1.00390625);
            interp.memory[4] = I(2);
            interp.intrinsic(&case.name, &args).unwrap();
            let product = if case.name.contains("bf16") {
                2.0
            } else {
                2.0078125
            };
            let expected = product
                + if epi && matches!(bias, P(_)) {
                    1.00390625
                } else {
                    0.0
                };
            assert_eq!(
                interp.memory[12],
                F(expected),
                "{} bias {bias:?}",
                case.name
            );
        }
        // The aliased spelling the snapshot used to answer — `C` over `A` (and, for the epilogue,
        // over the bias) — is refused with memory untouched: neither the native kernel nor the
        // source loop computes the copied value.
        args[2] = P(0);
        interp.memory = memory();
        interp.memory[0] = F(1.00390625);
        interp.memory[4] = I(2);
        let before = interp.memory.clone();
        let error = interp.intrinsic(&case.name, &args).unwrap_err();
        assert!(error.contains("overlap"), "{}: {error}", case.name);
        assert_eq!(interp.memory, before, "{} aliased output", case.name);
    }
}

#[test]
fn extents_dequant_integer_widths_aligned_i32_tail_and_optional_bias() {
    let program = Program::new();
    let interner = Interner::new();
    let mut interp = interpreter(&program, &interner);
    for parallel in [false, true] {
        let suffix = if parallel { "_parallel" } else { "" };
        for perchan in [false, true] {
            let name = format!(
                "wukong_dequant{}_f32{suffix}",
                if perchan { "_perchan" } else { "" }
            );
            for (width, expected) in [
                (wukong_runtime::DQ_I8, -1.0),
                (wukong_runtime::DQ_U8, 255.0),
                (wukong_runtime::DQ_I32, 511.0),
            ] {
                // Nine columns cover an AVX vector plus a scalar tail; two rows test stride.
                interp.memory = vec![I(511); 54];
                interp.memory[18..36].fill(Value::Unit);
                interp.memory[36..45].fill(F(1.0));
                let args = if perchan {
                    vec![P(0), P(18), I(2), I(9), P(36), I(width.into())]
                } else {
                    vec![P(0), P(18), I(18), F(1.0), I(width.into())]
                };
                interp.intrinsic(&name, &args).unwrap();
                assert_eq!(
                    &interp.memory[18..36],
                    &[F(expected); 18],
                    "{name} width {width}"
                );
            }
        }
        let name = format!("wukong_i8gemm_nt_deq{suffix}");
        for bias in [I(0), P(0)] {
            interp.memory = memory();
            // The output is disjoint from `A`; the bias may share `A`'s slots (both are reads).
            let mut args = [
                P(0),
                P(4),
                P(12),
                I(1),
                I(2),
                I(2),
                F(0.5),
                P(16),
                bias,
                I(0),
            ];
            interp.intrinsic(&name, &args).unwrap();
            // An output over `A` is outside the kernel's contract: refused, memory untouched.
            let after = interp.memory.clone();
            args[2] = P(0);
            let error = interp.intrinsic(&name, &args).unwrap_err();
            assert!(error.contains("overlap"), "{name}: {error}");
            assert_eq!(interp.memory, after, "{name}");
            assert_eq!(
                &interp.memory[12..14],
                if matches!(bias, P(_)) {
                    &[F(35.0), F(89.0)]
                } else {
                    &[F(33.0), F(86.0)]
                }
            );
        }
    }
}

/// The same cases on a TRACKED interpreter (every public entry point's): each used operand range
/// registered as its own allocation.
fn tracked_case(interp: &mut Interp<'_, '_>, case: &Case, shrink: Option<usize>) {
    interp.memory = memory();
    interp.slots = SlotReuse::tracked();
    for (i, &(arg, len)) in case.ranges.iter().enumerate() {
        let P(base) = case.args[arg] else {
            panic!("{}: argument {arg} is not a pointer", case.name)
        };
        if shrink == Some(i) {
            // One slot short, with the last slot a separate (adjacent) allocation: the operand
            // still fits the arena, but no longer its allocation.
            interp.slots.live.insert(base, len - 1);
            interp.slots.live.insert(base + len - 1, 1);
        } else {
            interp.slots.live.insert(base, len);
        }
    }
}

/// Exact fit: every operand exactly its allocation still runs, with the untracked answers.
#[test]
fn extents_exact_fit_allocations_still_run() {
    let program = Program::new();
    let interner = Interner::new();
    let mut interp = interpreter(&program, &interner);
    for case in cases() {
        tracked_case(&mut interp, &case, None);
        interp.intrinsic(&case.name, &case.args).unwrap();
        assert_eq!(
            &interp.memory[12..12 + case.expected.len()],
            &case.expected,
            "{}",
            case.name
        );
    }
}

/// An operand that fits the ARENA but runs one slot into the adjacent allocation is refused —
/// named by its argument index, before any scratch is reserved and with memory untouched — for
/// every operand of every case, inputs and outputs alike.
#[test]
fn extents_reject_an_operand_overrunning_into_the_adjacent_allocation() {
    let program = Program::new();
    let interner = Interner::new();
    let mut interp = interpreter(&program, &interner);
    for case in cases() {
        for (i, &(arg, _)) in case.ranges.iter().enumerate() {
            tracked_case(&mut interp, &case, Some(i));
            let before = interp.memory.clone();
            let error = interp.intrinsic(&case.name, &case.args).unwrap_err();
            assert_eq!(
                error,
                format!("{}: operand {arg} extends past its allocation", case.name),
                "{} operand {arg}",
                case.name
            );
            assert_eq!(interp.memory, before, "{} operand {arg}", case.name);
        }
    }
}

/// Absurd extents on a tracked interpreter are still the overflow / arena diagnostics (checked
/// before the allocation is looked up), never a panic or an allocator abort.
#[test]
fn extents_absurd_products_stay_diagnostics_on_a_tracked_interpreter() {
    let program = Program::new();
    let interner = Interner::new();
    let mut interp = interpreter(&program, &interner);
    for case in cases() {
        for &dim in &case.dims {
            for huge in [i128::from(i64::MAX), i128::from(i64::MAX) / 2 + 1] {
                let mut args = case.args.clone();
                args[dim] = I(huge);
                tracked_case(&mut interp, &case, None);
                let before = interp.memory.clone();
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    interp.intrinsic(&case.name, &args)
                }));
                let error = result
                    .expect("extent must diagnose rather than panic")
                    .unwrap_err();
                assert!(
                    error.contains("extent overflow")
                        || error.contains("out of bounds")
                        || error.contains("extends past its allocation"),
                    "{} dim {dim}: {error}",
                    case.name
                );
                assert_eq!(interp.memory, before, "{} dim {dim}", case.name);
            }
        }
    }
}

/// End to end through a public entry point (which always tracks): `main` allocates `x: [f32; 4]`,
/// a neighbouring `[f32; 4]`, and `out: [f32; 8]`, then calls `wukong_vmath_f32(x, out, n, exp)`.
/// At `n = 4` every operand is its allocation and the run succeeds; at `n = 8` the read of `x`
/// still fits the arena (it runs into the neighbour) but not `x`, and the run stops with the
/// diagnostic instead of exponentiating the neighbour's slots.
#[test]
fn a_run_refuses_a_kernel_operand_that_runs_into_the_next_alloca() {
    for (n, expected) in [
        (4, Ok((0, Vec::new()))),
        (
            8,
            Err("wukong_vmath_f32: operand 0 extends past its allocation".to_string()),
        ),
    ] {
        let mut interner = Interner::new();
        let main = interner.intern("main");
        let vmath = interner.intern("wukong_vmath_f32");
        let mut b = wukong_mir::Builder::new(main, MirType::I64);
        let x = b.alloca(MirType::Array(Box::new(MirType::F32), 4));
        b.alloca(MirType::Array(Box::new(MirType::F32), 4));
        let out = b.alloca(MirType::Array(Box::new(MirType::F32), 8));
        let len = b.build(MirType::I64, Op::ConstInt(n, MirType::I64));
        let exp = b.build(
            MirType::I64,
            Op::ConstInt(wukong_runtime::VM_EXP.into(), MirType::I64),
        );
        b.build_void(Op::Call {
            func: vmath,
            args: vec![x, out, len, exp],
        });
        let zero = b.build(MirType::I64, Op::ConstInt(0, MirType::I64));
        b.ret(Some(zero));
        let mut program = Program::new();
        program.funcs.push(b.finish());
        assert_eq!(
            run_with_output(&program, main, &interner),
            expected,
            "n = {n}"
        );
    }
}
