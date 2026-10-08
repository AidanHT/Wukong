//! Interpreter-only malformed-memory checks: never pass type-confused buffers to native code.
use super::*;
use wukong_mir::{Builder, VecBin, VecKernel, VecOp, VecRedOp, VecReduce};
use Value::{Float as F, Int as I, Ptr as P};

fn interpreter<'a>(program: &'a Program, interner: &'a Interner) -> Interp<'a, 'a> {
    Interp {
        program,
        interner,
        memory: Vec::new(),
        stdout: Vec::new(),
        scratch_arena: ScratchSlots::default(),
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
    name: &'static str,
    args: Vec<Value>,
    reads: Vec<usize>,
    zero_dim: usize,
}

// One or more representatives of every floating-buffer marshalling arm, including optional
// affine/bias inputs, accumulator outputs, low-precision reconstruction and live gather/scatter.
// Input bases are separated by four slots; output is 24; integer IDs/targets live at 28.
fn cases() -> Vec<Case> {
    let mut cases = Vec::new();
    macro_rules! add {
        ($names:expr, $args:expr, $reads:expr, $zero:expr) => {
            for name in $names {
                cases.push(Case {
                    name,
                    args: $args.to_vec(),
                    reads: $reads.to_vec(),
                    zero_dim: $zero,
                });
            }
        };
    }
    add!(
        ["wukong_sgemm", "wukong_sgemm_nt", "wukong_sgemm_tn"],
        [P(0), P(4), P(24), I(1), I(1), I(1), I(1)],
        [0, 4, 24],
        3
    );
    add!(["wukong_sgemv"], [P(0), P(4), P(24), I(1), I(1)], [0, 4], 3);
    add!(
        ["wukong_sgemv_alpha", "wukong_sgevm_f32"],
        [P(0), P(4), P(24), I(1), I(1), F(1.0)],
        [0, 4],
        3
    );
    add!(
        ["wukong_i8gemm_nt_deq"],
        [
            P(0),
            P(4),
            P(24),
            I(1),
            I(1),
            I(1),
            F(1.0),
            P(8),
            P(12),
            I(0)
        ],
        [8, 12],
        3
    );
    add!(
        ["wukong_dequant_perchan_f32"],
        [P(0), P(24), I(1), I(1), P(8), I(0)],
        [8],
        2
    );
    add!(
        ["wukong_sgemm_nt_epi"],
        [P(0), P(4), P(24), I(1), I(1), I(1), I(1), P(8), I(0)],
        [0, 4, 24, 8],
        3
    );
    add!(
        ["wukong_sgemm_nt_alpha"],
        [P(0), P(4), P(24), I(1), I(1), I(1), I(1), F(1.0)],
        [0, 4, 24],
        3
    );
    add!(
        [
            "wukong_vmath_f32",
            "wukong_vmath_bf16",
            "wukong_vmath_f16",
            "wukong_vmath_bf16_out",
            "wukong_vmath_f16_out"
        ],
        [P(0), P(24), I(1), I(0)],
        [0],
        2
    );
    add!(
        [
            "wukong_transpose_f32",
            "wukong_transpose_u16",
            "wukong_colsum_f32",
            "wukong_colmax_f32",
            "wukong_colmin_f32",
            "wukong_colmaxabs_f32",
            "wukong_colmean_f32",
            "wukong_colsumsq_f32",
            "wukong_coll2_f32",
            "wukong_colrms_f32",
            "wukong_logsumexp_f32",
            "wukong_entropy_f32",
            "wukong_rowargmax_i32",
            "wukong_rowargmin_i32",
            "wukong_colargmax_i32",
            "wukong_colargmin_i32",
            "wukong_cumsum_f32",
            "wukong_cummax_f32",
            "wukong_cummin_f32",
            "wukong_cumprod_f32"
        ],
        [P(0), P(24), I(1), I(1)],
        [0],
        2
    );
    add!(
        ["wukong_maxpool2d_f32", "wukong_avgpool2d_f32"],
        [P(0), P(24), I(1), I(1), I(1), I(1), I(1), I(1), I(1)],
        [0],
        2
    );
    add!(
        ["wukong_vmath2_f32"],
        [P(0), P(4), P(24), I(1), I(0)],
        [0, 4],
        3
    );
    add!(
        [
            "wukong_softmax_bwd_f32",
            "wukong_lrscan_f32",
            "wukong_kldiv_f32",
            "wukong_kd_loss_f32"
        ],
        [P(0), P(4), P(24), I(1), I(1)],
        [0, 4],
        3
    );
    add!(
        ["wukong_rmsnorm_bwd_f32", "wukong_layernorm_bwd_f32"],
        [P(0), P(4), P(8), P(24), I(1), I(1), I(0)],
        [0, 4, 8],
        4
    );
    add!(
        ["wukong_xent_fwd_f32", "wukong_xent_bwd_f32"],
        [P(0), P(28), P(24), I(1), I(1)],
        [0],
        3
    );
    add!(
        ["wukong_rope_f32", "wukong_rope_bwd_f32"],
        [P(0), P(4), P(24), I(1), I(1)],
        [0, 1, 4],
        3
    );
    add!(
        ["wukong_velem_f32"],
        [
            P(0),
            P(4),
            P(24),
            I(1),
            F(1.0),
            F(1.0),
            F(0.0),
            I(i128::from(wukong_runtime::VE_USE_Y))
        ],
        [0, 4],
        3
    );
    add!(
        ["wukong_bias_bcast_f32"],
        [P(0), P(4), P(24), I(1), I(1), I(0)],
        [0, 4],
        3
    );
    add!(
        ["wukong_vhorner_f32"],
        [P(0), P(24), I(1), P(4), I(1)],
        [0, 4],
        2
    );
    add!(
        ["wukong_sreduce_f32"],
        [P(0), P(4), I(1), I(i128::from(wukong_runtime::RED_DOT))],
        [0, 4],
        2
    );
    add!(
        [
            "wukong_argreduce_f32",
            "wukong_reduce_bf16",
            "wukong_reduce_f16"
        ],
        [P(0), I(1), I(0)],
        [0],
        1
    );
    add!(
        ["wukong_dot_bf16", "wukong_dot_f16"],
        [P(0), P(4), I(1)],
        [0, 4],
        2
    );
    add!(["wukong_sum_bf16", "wukong_sum_f16"], [P(0), I(1)], [0], 1);
    add!(
        [
            "wukong_axpby_bf16",
            "wukong_axpby_f16",
            "wukong_axpby_bf16_out",
            "wukong_axpby_f16_out"
        ],
        [P(0), P(4), P(24), I(1), F(1.0), F(1.0)],
        [0, 4],
        3
    );
    add!(
        [
            "wukong_sgemm_bf16_nt",
            "wukong_sgemm_f16_nt",
            "wukong_sgemm_bf16_tn",
            "wukong_sgemm_f16_tn"
        ],
        [P(0), P(4), P(24), I(1), I(1), I(1), I(1)],
        [0, 4, 24],
        3
    );
    add!(
        ["wukong_sgemm_bf16_nt_epi", "wukong_sgemm_f16_nt_epi"],
        [P(0), P(4), P(24), I(1), I(1), I(1), I(1), P(8), I(0)],
        [0, 4, 24, 8],
        3
    );
    add!(
        ["wukong_norm_f32"],
        [P(0), P(24), I(1), I(1), I(0), I(0)],
        [0],
        2
    );
    add!(
        ["wukong_norm_affine_f32"],
        [P(0), P(24), P(4), P(8), I(1), I(1), I(0), I(0)],
        [0, 4, 8],
        4
    );
    add!(
        ["wukong_embedding_f32"],
        [P(24), P(0), P(28), I(1), I(1), I(1)],
        [0],
        3
    );
    add!(
        ["wukong_scatter_add_f32"],
        [P(24), P(0), P(28), I(1), I(1), I(1)],
        [0, 24],
        3
    );
    add!(
        ["wukong_attention_f32"],
        [P(0), P(4), P(8), P(24), I(1), I(1), F(1.0), I(0)],
        [0, 4, 8],
        4
    );
    cases
}

#[test]
fn floating_kernel_operands_reject_pointer_vector_and_undefined_slots() {
    let program = Program::new();
    let interner = Interner::new();
    let mut interp = interpreter(&program, &interner);
    for case in cases() {
        for &slot in &case.reads {
            for (bad, kind) in [
                (P(0), "pointer"),
                (Value::VecRef(0), "vector"),
                (Value::Unit, "uninitialized"),
            ] {
                interp.memory = vec![F(2.0); 32];
                interp.memory[28] = I(0);
                interp.memory[slot] = bad;
                let before = interp.memory.clone();
                let error = interp.intrinsic(case.name, &case.args).unwrap_err();
                assert!(
                    error.contains(case.name) && error.contains(kind),
                    "{} slot {slot}: {error}",
                    case.name
                );
                assert_eq!(interp.memory, before, "{} slot {slot}", case.name);
            }
        }
    }
}

#[test]
fn valid_numeric_slots_and_empty_extents_remain_accepted() {
    let program = Program::new();
    let interner = Interner::new();
    let mut interp = interpreter(&program, &interner);
    for case in cases() {
        for numeric in [F(2.0), I(2)] {
            interp.memory = vec![F(2.0); 32];
            interp.memory[28] = I(0);
            for &slot in &case.reads {
                interp.memory[slot] = numeric;
            }
            interp
                .intrinsic(case.name, &case.args)
                .unwrap_or_else(|e| panic!("{}: {e}", case.name));
        }
        if case.name.starts_with("wukong_sgemm") && case.args.len() >= 7 {
            interp.memory = vec![F(2.0); 32];
            interp.memory[24] = Value::Unit;
            let mut args = case.args.clone();
            args[6] = I(0);
            interp.intrinsic(case.name, &args).unwrap();
            assert!(matches!(interp.memory[24], F(_)));
        }
        for empty in [0, -1] {
            interp.memory = vec![Value::Unit; 32];
            let mut args = case.args.clone();
            args[case.zero_dim] = I(empty);
            interp
                .intrinsic(case.name, &args)
                .unwrap_or_else(|e| panic!("{} empty: {e}", case.name));
            assert_eq!(interp.memory, vec![Value::Unit; 32], "{} empty", case.name);
        }
    }
}

#[test]
fn valid_low_precision_and_live_alias_values_are_unchanged() {
    let program = Program::new();
    let interner = Interner::new();
    let mut interp = interpreter(&program, &interner);
    for name in [
        "wukong_axpby_bf16",
        "wukong_axpby_f16",
        "wukong_axpby_bf16_out",
        "wukong_axpby_f16_out",
    ] {
        interp.memory = vec![F(1.5), F(-2.0), Value::Unit];
        interp
            .intrinsic(name, &[P(0), P(1), P(2), I(1), F(2.0), F(0.5)])
            .unwrap();
        assert_eq!(interp.memory, [F(1.5), F(-2.0), F(2.0)]);
    }
    // First gather overwrites the source needed by the next row: snapshots would yield [2,1].
    interp.memory = vec![F(1.0), F(2.0), I(1), I(0)];
    interp
        .intrinsic(
            "wukong_embedding_f32",
            &[P(0), P(0), P(2), I(2), I(1), I(2)],
        )
        .unwrap();
    assert_eq!(&interp.memory[..2], &[F(2.0), F(2.0)]);
    // The first scatter update changes the live gradient consumed by the second token.
    interp.memory = vec![F(1.0), F(2.0), I(1), I(0)];
    interp
        .intrinsic(
            "wukong_scatter_add_f32",
            &[P(0), P(0), P(2), I(2), I(1), I(2)],
        )
        .unwrap();
    assert_eq!(&interp.memory[..2], &[F(4.0), F(3.0)]);
}

fn vector_program(interner: &mut Interner, reduction: bool) -> Program {
    let name = interner.intern("vector_kernel");
    let mut b = Builder::new(
        name,
        if reduction {
            MirType::F32
        } else {
            MirType::Void
        },
    );
    let ptrs = b.add_param(MirType::Ptr);
    let scalars = b.add_param(MirType::Ptr);
    let n = b.add_param(MirType::I64);
    let op = Op::VecKernelCall {
        kernel: 0,
        ptrs,
        scalars,
        n,
    };
    if reduction {
        let result = b.build(MirType::F32, op);
        b.ret(Some(result));
    } else {
        b.build_void(op);
        b.ret(None);
    }
    let mut func = b.finish();
    let mut ops = vec![
        VecOp::Load { stream: 0 },
        VecOp::Splat { scalar: 0 },
        VecOp::Bin {
            op: VecBin::Mul,
            a: 0,
            b: 1,
        },
    ];
    if !reduction {
        ops.push(VecOp::Store { stream: 1, val: 2 });
        // This load forwards the store, so it must not validate the old output slot.
        ops.push(VecOp::Load { stream: 1 });
        ops.push(VecOp::Store { stream: 1, val: 4 });
    }
    func.vec_kernels.push(VecKernel {
        name,
        lane: wukong_mir::VecLane::F32,
        streams: 2,
        scalars: 2,
        unroll: 1,
        reduce: reduction.then_some(VecReduce {
            op: VecRedOp::Add,
            value: 2,
            fma: None,
        }),
        ops,
    });
    let mut program = Program::new();
    program.funcs.push(func);
    program
}

#[test]
fn vector_recipe_checks_reads_but_leaves_tail_and_output_only_slots_alone() {
    let mut interner = Interner::new();
    for reduction in [false, true] {
        let program = vector_program(&mut interner, reduction);
        assert!(wukong_mir::verify::verify_program(&program).is_empty());
        let mut interp = interpreter(&program, &interner);
        let mut memory = vec![Value::Unit; 25];
        memory[0] = P(4);
        memory[1] = P(16);
        memory[2] = F(3.0);
        memory[4..12].fill(F(2.0));
        interp.memory = memory.clone();
        let result = interp
            .run_function(&program.funcs[0], &[P(0), P(2), I(8)])
            .unwrap();
        if reduction {
            assert_eq!(result, F(48.0));
        } else {
            assert_eq!(&interp.memory[16..24], &[F(6.0); 8]);
        }
        assert_eq!(interp.memory[12], Value::Unit);
        assert_eq!(interp.memory[24], Value::Unit);
        if reduction {
            interp.memory = memory.clone();
            assert_eq!(
                interp
                    .run_function(&program.funcs[0], &[P(0), P(2), I(9)])
                    .unwrap(),
                F(48.0)
            );
        }
        for slot in [2, 4, 11] {
            for bad in [P(0), Value::Unit, Value::VecRef(0)] {
                interp.memory = memory.clone();
                interp.memory[slot] = bad;
                let error = interp
                    .run_function(&program.funcs[0], &[P(0), P(2), I(8)])
                    .unwrap_err();
                assert!(
                    error.contains("veckernel") && error.contains("floating buffer"),
                    "{error}"
                );
            }
        }
        interp.memory = memory;
        interp.memory[4..12].fill(Value::Unit);
        interp
            .run_function(&program.funcs[0], &[P(0), P(2), I(0)])
            .unwrap();
        assert_eq!(&interp.memory[16..25], &[Value::Unit; 9]);
    }
}

#[test]
fn host_float_buffers_reject_pointer_copyback_and_preserve_untouched_nan_bits() {
    let mut interner = Interner::new();
    for write_pointer in [false, true] {
        let name = interner.intern("host_buffer");
        let mut b = Builder::new(name, MirType::Void);
        let buf = b.add_param(MirType::Ptr);
        if write_pointer {
            b.build_void(Op::Store {
                ptr: buf,
                value: buf,
            });
        }
        b.ret(None);
        let mut program = Program::new();
        program.funcs.push(b.finish());
        let bits32 = [0x7f80_0001, 0xff80_1234, 0x7fc0_1234, 0x8000_0000];
        let mut floats32 = bits32.map(f32::from_bits);
        let result = run_kernel_f32(&program, name, &mut [&mut floats32], &interner);
        if write_pointer {
            assert!(result.unwrap_err().contains("pointer slot"));
        } else {
            result.unwrap();
            assert_eq!(floats32.map(f32::to_bits), bits32);
        }
        let bits64 = [
            0x7ff0_0000_0000_0001,
            0xfff0_0000_0000_1234,
            0x8000_0000_0000_0000,
        ];
        let mut floats64 = bits64.map(f64::from_bits);
        let result = run_kernel_f64(&program, name, &mut [&mut floats64], &interner);
        if write_pointer {
            assert!(result.unwrap_err().contains("pointer slot"));
        } else {
            result.unwrap();
            assert_eq!(floats64.map(f64::to_bits), bits64);
        }
    }
}

#[test]
fn unary_modes_do_not_validate_unused_second_inputs() {
    let program = Program::new();
    let interner = Interner::new();
    let mut interp = interpreter(&program, &interner);
    for unused in [Value::Unit, P(0), Value::VecRef(0)] {
        interp.memory = vec![F(2.0), unused, Value::Unit];
        interp
            .intrinsic(
                "wukong_velem_f32",
                &[P(0), P(1), P(2), I(1), F(3.0), F(0.0), F(0.0), I(0)],
            )
            .unwrap();
        assert_eq!(interp.memory[2], F(6.0));
        let sum = interp
            .intrinsic(
                "wukong_sreduce_f32",
                &[P(0), P(1), I(1), I(i128::from(wukong_runtime::RED_SUM))],
            )
            .unwrap();
        assert_eq!(sum, F(2.0));
    }
    // These ABI modes don't need y storage at all; only x and the output are dereferenced.
    interp.memory = vec![F(2.0), Value::Unit];
    interp
        .intrinsic(
            "wukong_velem_f32",
            &[
                P(0),
                P(usize::MAX),
                P(1),
                I(1),
                F(3.0),
                F(0.0),
                F(0.0),
                I(0),
            ],
        )
        .unwrap();
    assert_eq!(interp.memory[1], F(6.0));
    assert_eq!(
        interp
            .intrinsic(
                "wukong_sreduce_f32",
                &[
                    P(0),
                    P(usize::MAX),
                    I(1),
                    I(i128::from(wukong_runtime::RED_SUM))
                ]
            )
            .unwrap(),
        F(2.0)
    );
}

fn fused_gemm_cases() -> impl Iterator<Item = (&'static str, Vec<Value>)> {
    [
        "wukong_sgemm_nt_epi",
        "wukong_sgemm_nt_epi_parallel",
        "wukong_sgemm_nt_alpha",
        "wukong_sgemm_nt_alpha_parallel",
    ]
    .into_iter()
    .map(|name| {
        let mut args = vec![P(0), P(2), P(6), I(1), I(2), I(2), I(0)];
        if name.contains("_epi") {
            args.extend([P(8), I(0)]);
        } else {
            args.push(F(1.0));
        }
        (name, args)
    })
}

#[test]
fn fused_gemm_rejects_huge_extents_before_allocation() {
    let program = Program::new();
    let interner = Interner::new();
    let mut interp = interpreter(&program, &interner);
    for (name, args) in fused_gemm_cases() {
        for (m, k, n, diagnostic) in [
            (i64::MAX, 4, 1, "extent overflow"),
            (1, i64::MAX, 1, "out of bounds"),
            (1, 1, i64::MAX, "out of bounds"),
        ] {
            interp.memory = vec![F(2.0); 10];
            let before = interp.memory.clone();
            let mut args = args.clone();
            args[3..6].copy_from_slice(&[I(m.into()), I(k.into()), I(n.into())]);
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                interp.intrinsic(name, &args)
            }));
            let error = result
                .expect("fused GEMM must diagnose, not panic")
                .unwrap_err();
            assert!(error.contains(diagnostic), "{name}: {error}");
            assert_eq!(interp.memory, before, "{name}");
        }
    }
}

#[test]
fn fused_gemm_validates_all_ranges_before_marshalling() {
    let program = Program::new();
    let interner = Interner::new();
    let mut interp = interpreter(&program, &interner);
    for (name, args) in fused_gemm_cases() {
        let pointers: &[usize] = if name.contains("_epi") {
            &[0, 1, 2, 7]
        } else {
            &[0, 1, 2]
        };
        for &arg in pointers {
            for base in [9, usize::MAX] {
                interp.memory = vec![Value::Unit; 10];
                let before = interp.memory.clone();
                let mut args = args.clone();
                args[arg] = P(base);
                let error = interp.intrinsic(name, &args).unwrap_err();
                assert!(error.contains("out of bounds"), "{name} arg {arg}: {error}");
                assert_eq!(interp.memory, before, "{name} arg {arg}");
            }
        }
    }
}

#[test]
fn fused_gemm_preserves_numeric_types_bias_and_unread_output() {
    let program = Program::new();
    let interner = Interner::new();
    let mut interp = interpreter(&program, &interner);
    for (name, args) in fused_gemm_cases() {
        let epi = name.contains("_epi");
        for beta in [0, 1] {
            let mut args = args.clone();
            args[6] = I(beta);
            interp.memory = vec![
                F(2.0),
                I(3),
                F(4.0),
                I(5),
                F(6.0),
                I(7),
                F(1.0),
                F(2.0),
                F(2.0),
                I(3),
            ];
            let baseline = interp.memory.clone();
            let reads: &[usize] = if epi { &[0, 2, 6, 8] } else { &[0, 2, 6] };
            for &slot in reads {
                for bad in [P(0), Value::VecRef(0), Value::Unit] {
                    interp.memory = baseline.clone();
                    interp.memory[slot] = bad;
                    let before = interp.memory.clone();
                    let result = interp.intrinsic(name, &args);
                    if slot == 6 && beta == 0 {
                        result.unwrap();
                        let expected = if epi {
                            [F(25.0), F(36.0)]
                        } else {
                            [F(23.0), F(33.0)]
                        };
                        assert_eq!(&interp.memory[6..8], &expected, "{name}");
                    } else {
                        let error = result.unwrap_err();
                        assert!(
                            error.contains(name) && error.contains("floating buffer contains"),
                            "{error}"
                        );
                        assert_eq!(interp.memory, before, "{name}");
                    }
                }
            }
            interp.memory = baseline;
            interp.intrinsic(name, &args).unwrap();
            let expected = if epi {
                [25.0 + beta as f64, 36.0 + 2.0 * beta as f64]
            } else {
                [23.0 + beta as f64, 33.0 + 2.0 * beta as f64]
            };
            assert_eq!(&interp.memory[6..8], &expected.map(F), "{name}");
        }
        // Slot zero is a real bias pointer; null is an integer. All inputs are snapshotted
        // before output writes, including overlapping A, C, and optional bias.
        for bias in [I(0), P(0)] {
            let mut args = args.clone();
            args[2] = P(0);
            if epi {
                args[7] = bias;
            } else {
                args[7] = F(0.5);
            }
            interp.memory = vec![
                F(2.0),
                F(3.0),
                F(4.0),
                F(5.0),
                F(6.0),
                F(7.0),
                Value::Unit,
                Value::Unit,
                F(2.0),
                F(3.0),
            ];
            interp.intrinsic(name, &args).unwrap();
            let expected = if !epi {
                [F(11.5), F(16.5)]
            } else if matches!(bias, P(_)) {
                [F(25.0), F(36.0)]
            } else {
                [F(23.0), F(33.0)]
            };
            assert_eq!(&interp.memory[..2], &expected, "{name}");
        }
    }
}
