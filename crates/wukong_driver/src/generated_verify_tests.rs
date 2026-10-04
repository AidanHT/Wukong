//! Generated MIR must pass the driver gates even without --verify-each.

use super::*;
use wukong_mir::{BinOp, Builder, MirType, Op, Program};
use wukong_span::Symbol;

fn loss_program(interner: &mut Interner, bad_store: bool, call: Option<Symbol>) -> Program {
    let mut b = Builder::new(interner.intern("loss"), MirType::F32);
    let input = b.add_param(MirType::Ptr);
    let output = b.add_param(MirType::Ptr);
    let x = b.build(MirType::F32, Op::Load(input, MirType::F32));
    let y = b.build(MirType::F32, Op::Bin(BinOp::FMul, x, x));
    b.build_void(Op::Store {
        ptr: if bad_store { y } else { output },
        value: y,
    });
    if let Some(func) = call {
        // An unused scalar call is replayed by autodiff without requiring a VJP rule.
        b.build(MirType::F32, Op::Call { func, args: vec![] });
    }
    b.ret(Some(y));
    let mut program = Program::new();
    program.funcs.push(b.finish());
    program
}

#[test]
fn default_grad_and_training_reject_replayed_invalid_store() {
    let mut interner = Interner::new();
    let program = loss_program(&mut interner, true, None);
    let raw = wukong_autodiff::grad(&program.funcs[0], &[0], &mut interner).unwrap();
    assert!(!wukong_mir::verify::verify_function(&raw).is_empty());
    for verify_each in [false, true] {
        let mut opts = Options {
            verify_each,
            ..Options::default()
        };
        opts.grad.wrt = vec![0];
        let error = build_grad(&program, &mut interner, &opts).unwrap_err();
        assert!(error.contains("after `autodiff`"), "{error}");
        assert!(error.contains("store"), "{error}");
        assert_eq!(
            emit_grad(&program, &mut interner, &opts),
            exit::COMPILE_ERROR
        );
        opts.grad.train = true;
        opts.grad.train_steps = 0;
        for train_opt in [TrainOpt::Sgd, TrainOpt::AdamW] {
            opts.grad.train_opt = train_opt;
            assert_eq!(
                train_loop(&program, &mut interner, &opts, &[1, 1]).unwrap_err(),
                error
            );
        }
    }
}

#[test]
fn default_grad_verifies_the_complete_callee_set() {
    let mut interner = Interner::new();
    let mut program = loss_program(&mut interner, false, None);
    let mut callee = Builder::new(interner.intern("callee"), MirType::Void);
    callee.add_param(MirType::Ptr);
    callee.ret(None);
    let callee = callee.finish();
    let mut caller = Builder::new(interner.intern("caller"), MirType::Void);
    caller.build_void(Op::Call {
        func: callee.name,
        args: vec![],
    });
    caller.ret(None);
    program.funcs.extend([callee, caller.finish()]);
    assert!(program
        .funcs
        .iter()
        .all(|f| wukong_mir::verify::verify_function(f).is_empty()));
    let error = build_grad(&program, &mut interner, &Options::default()).unwrap_err();
    assert!(error.contains("after `autodiff`"), "{error}");
    assert!(error.contains("call"), "{error}");
}

#[test]
fn default_grad_checks_calls_introduced_only_by_the_backward() {
    let mut interner = Interner::new();
    let reduce = interner.intern("wukong_sreduce_f32");
    let velem = interner.intern("wukong_velem_f32");
    let mut b = Builder::new(interner.intern("loss"), MirType::F32);
    let input = b.add_param(MirType::Ptr);
    let output = b.add_param(MirType::Ptr);
    let count = b.build(MirType::I64, Op::ConstInt(1, MirType::I64));
    let sum = b.build(MirType::I64, Op::ConstInt(2, MirType::I64));
    let loss = b.build(
        MirType::F32,
        Op::Call {
            func: reduce,
            args: vec![input, input, count, sum],
        },
    );
    b.build_void(Op::Store {
        ptr: output,
        value: loss,
    });
    b.ret(Some(loss));
    // Only the backward calls velem, to fill the gradient with ones. This unused definition
    // is valid MIR, but its zero-parameter signature conflicts with that generated call.
    let mut callee = Builder::new(velem, MirType::Void);
    callee.ret(None);
    let mut program = Program::new();
    program.funcs = vec![b.finish(), callee.finish()];
    assert!(wukong_mir::verify::verify_program(&program).is_empty());
    let raw = wukong_autodiff::grad(&program.funcs[0], &[0], &mut interner).unwrap();
    assert!(raw.blocks.iter().flat_map(|block| &block.insts).any(
        |inst| matches!(&inst.op, Op::Call { func, args } if *func == velem && args.len() == 8)
    ));
    assert!(wukong_mir::verify::verify_function(&raw).is_empty());
    let mut generated = program.clone();
    generated.funcs.push(raw);
    assert!(!wukong_mir::verify::verify_program(&generated).is_empty());
    for verify_each in [false, true] {
        let opts = Options {
            verify_each,
            grad: GradOptions {
                wrt: vec![0],
                ..GradOptions::default()
            },
            ..Options::default()
        };
        let error = build_grad(&program, &mut interner, &opts).unwrap_err();
        assert!(error.contains("after `autodiff`"), "{error}");
        assert!(error.contains("call"), "{error}");
    }
}

#[test]
fn default_training_checks_newly_defined_adamw_signature_before_execution() {
    let mut interner = Interner::new();
    let adamw = interner.intern("adamw_step_1");
    let program = loss_program(&mut interner, false, Some(adamw));
    for verify_each in [false, true] {
        let opts = Options {
            verify_each,
            grad: GradOptions {
                train: true,
                train_steps: 0,
                train_opt: TrainOpt::AdamW,
                ..GradOptions::default()
            },
            ..Options::default()
        };
        // The external call has no MIR signature yet. Only training synthesis introduces the
        // five-pointer, void-returning AdamW definition and exposes the incompatible call.
        build_grad(&program, &mut interner, &opts).unwrap();
        let error = train_loop(&program, &mut interner, &opts, &[1, 1]).unwrap_err();
        assert!(error.contains("after `training kernels`"), "{error}");
        assert!(error.contains("call"), "{error}");
    }
}

#[test]
fn default_verification_preserves_gradient_and_training_values() {
    let mut interner = Interner::new();
    let program = loss_program(&mut interner, false, None);
    for train_opt in [TrainOpt::Sgd, TrainOpt::AdamW] {
        let mut opts = Options {
            grad: GradOptions {
                train: true,
                train_steps: 1,
                train_lr: 0.25,
                train_opt,
                ..GradOptions::default()
            },
            ..Options::default()
        };
        let (forward, backward) = build_grad(&program, &mut interner, &opts).unwrap();
        let entry = backward.name;
        let mut generated = Program::new();
        generated.funcs = vec![forward, backward];
        let mut input = [2.0];
        let mut loss = [0.0];
        let mut gradient = [0.0];
        wukong_interp::run_kernel_f32(
            &generated,
            entry,
            &mut [&mut input, &mut loss, &mut gradient],
            &interner,
        )
        .unwrap();
        assert_eq!(loss, [4.0]);
        assert_eq!(gradient, [4.0]);
        let ordinary = train_loop(&program, &mut interner, &opts, &[1, 1]).unwrap();
        opts.verify_each = true;
        let checked = train_loop(&program, &mut interner, &opts, &[1, 1]).unwrap();
        assert_eq!(ordinary, checked);
        assert_eq!(ordinary.len(), 2);
        if train_opt == TrainOpt::Sgd {
            let mut seed = opts.grad.train_seed;
            let x = train_rand(&mut seed, 1)[0];
            assert_eq!(ordinary, vec![x * x, (x * 0.5) * (x * 0.5)]);
        }
    }
}
