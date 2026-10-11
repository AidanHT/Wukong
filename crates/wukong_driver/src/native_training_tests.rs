use super::*;
use wukong_mir::{BinOp, Builder, MirType, Op, Program};
use wukong_types::{Dim, Layout, Scalar, Shape, Ty};

fn square_loss(it: &mut Interner) -> Program {
    let mut b = Builder::new(it.intern("loss"), MirType::F32);
    let weights = b.add_param(MirType::Ptr);
    let data = b.add_param(MirType::Ptr);
    let output = b.add_param(MirType::Ptr);
    let w = b.build(MirType::F32, Op::Load(weights, MirType::F32));
    let x = b.build(MirType::F32, Op::Load(data, MirType::F32));
    let delta = b.build(MirType::F32, Op::Bin(BinOp::FSub, w, x));
    let loss = b.build(MirType::F32, Op::Bin(BinOp::FMul, delta, delta));
    b.build_void(Op::Store {
        ptr: output,
        value: loss,
    });
    b.ret(Some(loss));
    let mut program = Program::new();
    program.funcs.push(b.finish());
    program
}

fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|x| x.to_bits()).collect()
}

#[test]
fn native_matches_exact_reference_training_and_preserves_fixed_inputs() {
    let mut it = Interner::new();
    let program = square_loss(&mut it);
    for (optimizer, decay) in [
        (TrainOpt::Sgd, 0.0),
        (TrainOpt::AdamW, 0.0),
        (TrainOpt::AdamW, 0.125),
    ] {
        for level in 0..=3 {
            for verified in [false, true] {
                let mut opts = Options {
                    opt_level: level,
                    verify_each: verified,
                    ..Options::default()
                };
                opts.grad = GradOptions {
                    train: true,
                    wrt: vec![0],
                    train_steps: 4,
                    train_lr: 0.125,
                    train_seed: 7,
                    train_opt: optimizer,
                    train_weight_decay: decay,
                    ..GradOptions::default()
                };
                let mut seed = opts.grad.train_seed;
                let mut w = train_rand(&mut seed, 1)[0];
                let fixed = train_rand(&mut seed, 1)[0];
                let mut expected = vec![(w - fixed) * (w - fixed)];
                let (mut m, mut v) = (0.0f32, 0.0f32);
                for t in 1..=4 {
                    let delta = w - fixed;
                    let gradient = delta + delta;
                    if optimizer == TrainOpt::Sgd {
                        w -= 0.125 * gradient;
                    } else {
                        m = 0.9f32.mul_add(m, (1.0 - 0.9f32) * gradient);
                        v = 0.999f32.mul_add(v, (1.0 - 0.999f32) * (gradient * gradient));
                        let mhat = m / (1.0 - 0.9f64.powi(t)) as f32;
                        let vhat = v / (1.0 - 0.999f64.powi(t)) as f32;
                        w -= 0.125 * (mhat / (vhat.sqrt() + 1e-8f32) + decay * w);
                    }
                    expected.push((w - fixed) * (w - fixed));
                }
                for backend in [BackendKind::Interp, BackendKind::Native] {
                    opts.backend = backend;
                    let (trajectory, buffers) =
                        train_loop_with_buffers(&program, &mut it, &opts, &[1, 1, 1]).unwrap();
                    assert_eq!(
                        bits(&trajectory),
                        bits(&expected),
                        "{backend:?} O{level} verified={verified} {optimizer:?} decay={decay}"
                    );
                    assert_eq!(buffers[0][0].to_bits(), w.to_bits());
                    assert_eq!(
                        buffers[1][0].to_bits(),
                        fixed.to_bits(),
                        "fixed input changed"
                    );
                }
            }
        }
    }
}

#[test]
fn training_rejects_non_buffer_mir_and_unrepresentable_extents() {
    let mut it = Interner::new();
    let mut program = square_loss(&mut it);
    let mut opts = Options {
        backend: BackendKind::Native,
        ..Options::default()
    };
    opts.grad.wrt = vec![0];
    let error = train_loop(&program, &mut it, &opts, &[usize::MAX, 1, 1]).unwrap_err();
    assert!(error.contains("byte extent"), "{error}");
    let scalar = program.funcs[0].params[1];
    program.funcs[0].value_types[scalar.0 as usize] = MirType::F32;
    let error = train_loop(&program, &mut it, &opts, &[1, 1, 1]).unwrap_err();
    assert!(error.contains("every loss parameter"), "{error}");
    for backend in [BackendKind::Gpu, BackendKind::GpuLower] {
        opts.backend = backend;
        let error = train_loop(&program, &mut it, &opts, &[1, 1, 1]).unwrap_err();
        assert!(
            error.contains("supports only --backend=interp or --backend=native"),
            "{error}"
        );
    }
}

fn array(elem: Ty, len: u64) -> Ty {
    Ty::Array {
        elem: Box::new(elem),
        len,
        shared: false,
    }
}

#[test]
fn semantic_f32_buffer_extents_are_checked_before_allocation() {
    let f32ty = Ty::Scalar(Scalar::F32);
    assert_eq!(ty_elem_count(&array(array(f32ty.clone(), 2), 3)), Some(6));
    assert_eq!(ty_elem_count(&array(f32ty.clone(), 0)), Some(0));
    for layout in [Layout::Contiguous, Layout::ColMajor] {
        let tensor = Ty::Tensor {
            elem: Scalar::F32,
            shape: Shape(vec![Dim::Const(2), Dim::Const(3)]),
            layout,
            elem_var: None,
        };
        assert_eq!(ty_elem_count(&tensor), Some(6));
    }
    let limit = (isize::MAX as usize / 4) as u64;
    assert_eq!(
        ty_elem_count(&array(f32ty.clone(), limit)),
        Some(limit as usize)
    );
    for ty in [
        f32ty.clone(),
        array(Ty::Scalar(Scalar::F64), 1),
        array(Ty::Scalar(Scalar::I64), 1),
        array(Ty::Scalar(Scalar::F16), 1),
        array(f32ty.clone(), limit + 1),
        array(array(f32ty.clone(), u64::MAX), 2),
        Ty::Slice(Box::new(f32ty.clone())),
        Ty::Ptr {
            mutable: true,
            pointee: Box::new(f32ty.clone()),
        },
        Ty::Tensor {
            elem: Scalar::F32,
            shape: Shape(vec![Dim::Dynamic]),
            layout: Layout::Contiguous,
            elem_var: None,
        },
        Ty::Tensor {
            elem: Scalar::F64,
            shape: Shape(vec![Dim::Const(1)]),
            layout: Layout::Contiguous,
            elem_var: None,
        },
        Ty::Tensor {
            elem: Scalar::F32,
            shape: Shape(vec![Dim::Const(u64::MAX), Dim::Const(2)]),
            layout: Layout::Contiguous,
            elem_var: None,
        },
        Ty::Tensor {
            elem: Scalar::F32,
            shape: Shape(vec![Dim::Const(2)]),
            layout: Layout::Strided,
            elem_var: None,
        },
    ] {
        assert_eq!(ty_elem_count(&ty), None, "{ty:?}");
    }
}

#[test]
fn native_dispatcher_is_generated_and_checks_buffer_lengths() {
    let mut it = Interner::new();
    let mut program = square_loss(&mut it);
    let entry = program.funcs[0].name;
    let interp = training_native::TrainingExecutor::new(
        &mut program,
        &mut it,
        BackendKind::Interp,
        vec![(entry, vec![1, 1, 1])],
    )
    .unwrap();
    assert!(!interp.is_native());
    program.statics.push(wukong_mir::StaticData {
        name: it.intern("__wukong_training_dispatch_0"),
        bytes: vec![0],
    });
    let mut conflict = Builder::new(it.intern("__wukong_training_dispatch_1"), MirType::Void);
    conflict.ret(None);
    program.funcs.push(conflict.finish());
    let executor = training_native::TrainingExecutor::new(
        &mut program,
        &mut it,
        BackendKind::Native,
        vec![(entry, vec![1, 1, 1])],
    )
    .unwrap();
    assert!(executor.is_native());
    assert_eq!(program.funcs.len(), 3);
    let dispatcher = &program.funcs[2];
    assert_eq!(it.resolve(dispatcher.name), "__wukong_training_dispatch_2");
    assert_eq!(dispatcher.params.len(), 2);
    assert!(dispatcher
        .blocks
        .iter()
        .flat_map(|b| &b.insts)
        .any(|i| matches!(i.op, Op::Call { func, .. } if func == entry)));
    let mut buffers = vec![vec![3.0], vec![1.0], vec![0.0]];
    executor.run(&program, entry, &mut buffers, &it).unwrap();
    assert_eq!(buffers, vec![vec![3.0], vec![1.0], vec![4.0]]);
    buffers[0].clear();
    assert!(executor
        .run(&program, entry, &mut buffers, &it)
        .unwrap_err()
        .contains("buffer lengths"));
}

#[test]
fn native_generated_backward_loop_preserves_all_lanes() {
    let mut it = Interner::new();
    let reduce = it.intern("wukong_sreduce_f32");
    let mut b = Builder::new(it.intern("loss"), MirType::F32);
    let input = b.add_param(MirType::Ptr);
    let output = b.add_param(MirType::Ptr);
    let count = b.build(MirType::I64, Op::ConstInt(9, MirType::I64));
    let sumabs = b.build(MirType::I64, Op::ConstInt(9, MirType::I64));
    let loss = b.build(
        MirType::F32,
        Op::Call {
            func: reduce,
            args: vec![input, input, count, sumabs],
        },
    );
    b.build_void(Op::Store {
        ptr: output,
        value: loss,
    });
    b.ret(Some(loss));
    let mut program = Program::new();
    program.funcs.push(b.finish());
    for opt_level in 0..=3 {
        for verify_each in [false, true] {
            let mut opts = Options {
                opt_level,
                verify_each,
                ..Options::default()
            };
            opts.grad.wrt = vec![0];
            let (fwd, grad) = build_grad(&program, &mut it, &opts).unwrap();
            assert!(grad.blocks.len() > 1);
            let entry = grad.name;
            let mut generated = program.clone();
            generated.funcs = vec![fwd, grad];
            let executor = training_native::TrainingExecutor::new(
                &mut generated,
                &mut it,
                BackendKind::Native,
                vec![(entry, vec![9, 1, 9])],
            )
            .unwrap();
            let input = vec![-4.0, -2.0, -1.0, -0.0, 0.0, 0.5, 1.0, 2.0, 4.0];
            let mut buffers = vec![input.clone(), vec![0.0], vec![0.0; 9]];
            executor.run(&generated, entry, &mut buffers, &it).unwrap();
            assert_eq!(bits(&buffers[0]), bits(&input));
            assert_eq!(bits(&buffers[1]), bits(&[14.5]));
            assert_eq!(
                bits(&buffers[2]),
                bits(&[-1.0, -1.0, -1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0])
            );
        }
    }
}

#[test]
fn native_adamw_dispatches_multiple_sizes_after_vec_replacements() {
    let mut it = Interner::new();
    let mut b = Builder::new(it.intern("loss"), MirType::F32);
    let w = b.add_param(MirType::Ptr);
    let u = b.add_param(MirType::Ptr);
    let fixed = b.add_param(MirType::Ptr);
    let out = b.add_param(MirType::Ptr);
    let one = b.build(MirType::I64, Op::ConstInt(1, MirType::I64));
    let u1 = b.build(
        MirType::Ptr,
        Op::Gep {
            ptr: u,
            index: one,
            elem: MirType::F32,
        },
    );
    let values: Vec<_> = [w, u, u1, fixed]
        .iter()
        .map(|&ptr| b.build(MirType::F32, Op::Load(ptr, MirType::F32)))
        .collect();
    let mut sum = values[0];
    for &v in &values[1..] {
        sum = b.build(MirType::F32, Op::Bin(BinOp::FAdd, sum, v));
    }
    b.build_void(Op::Store {
        ptr: out,
        value: sum,
    });
    b.ret(Some(sum));
    let mut program = Program::new();
    program.funcs.push(b.finish());
    let mut opts = Options {
        opt_level: 2,
        verify_each: true,
        ..Options::default()
    };
    opts.grad = GradOptions {
        train: true,
        wrt: vec![0, 1],
        train_steps: 3,
        train_lr: 0.125,
        train_opt: TrainOpt::AdamW,
        train_weight_decay: 0.125,
        train_seed: 7,
        ..GradOptions::default()
    };
    let mut seed = opts.grad.train_seed;
    let mut w = train_rand(&mut seed, 1);
    let mut u = train_rand(&mut seed, 2);
    let fixed = train_rand(&mut seed, 1);
    let mut expected = vec![((w[0] + u[0]) + u[1]) + fixed[0]];
    let (mut m, mut v) = (0.0f32, 0.0f32);
    for t in 1..=3 {
        m = 0.9f32.mul_add(m, 1.0 - 0.9f32);
        v = 0.999f32.mul_add(v, 1.0 - 0.999f32);
        let mhat = m / (1.0 - 0.9f64.powi(t)) as f32;
        let vhat = v / (1.0 - 0.999f64.powi(t)) as f32;
        for value in w.iter_mut().chain(&mut u) {
            *value -= 0.125 * (mhat / (vhat.sqrt() + 1e-8f32) + 0.125 * *value);
        }
        expected.push(((w[0] + u[0]) + u[1]) + fixed[0]);
    }
    for backend in [BackendKind::Interp, BackendKind::Native] {
        opts.backend = backend;
        let (trajectory, buffers) =
            train_loop_with_buffers(&program, &mut it, &opts, &[1, 2, 1, 1]).unwrap();
        assert_eq!(bits(&trajectory), bits(&expected));
        assert_eq!(bits(&buffers[0]), bits(&w));
        assert_eq!(bits(&buffers[1]), bits(&u));
        assert_eq!(bits(&buffers[2]), bits(&fixed));
    }
}

/// The AdamW hyperparameter layout has two copies — the kernel builder's (`wukong_autodiff`) and
/// the loop's (`wukong_runtime::train`) — and they must agree index for index. (A `const` assertion
/// beside `run_train` makes a drift a build error; this names each field when one fails.)
#[test]
fn native_training_hp_layout_matches_the_adamw_kernel() {
    use wukong_autodiff::optim::hp as kernel;
    use wukong_runtime::train::hp as runtime;
    for (field, k, r) in [
        ("LR", kernel::LR, runtime::LR),
        ("BETA1", kernel::BETA1, runtime::BETA1),
        ("BETA2", kernel::BETA2, runtime::BETA2),
        ("EPS", kernel::EPS, runtime::EPS),
        ("WD", kernel::WD, runtime::WD),
        ("BC1", kernel::BC1, runtime::BC1),
        ("BC2", kernel::BC2, runtime::BC2),
        ("LEN", kernel::LEN, runtime::LEN),
    ] {
        assert_eq!(k, r, "hp::{field}");
    }
}
