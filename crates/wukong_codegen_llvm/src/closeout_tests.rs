use super::*;
use wukong_mir::{Builder, RoundMode};

fn module(f: Function, interner: &Interner) -> String {
    let mut p = Program::new();
    p.funcs.push(f);
    let text = try_emit_llvm_ir(&p, interner).expect("representable MIR");
    // The separate LLVM verifier command consumes these exact regression modules. No external
    // executable is required to run the portable Rust assertions.
    if let Some(dir) = std::env::var_os("WUKONG_LLVM_TEST_OUTPUT") {
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            std::path::Path::new(&dir).join(format!("{}.ll", interner.resolve(p.funcs[0].name))),
            &text,
        )
        .unwrap();
    }
    text
}

#[test]
fn exact_float_literals_all_widths() {
    assert_eq!(fmt_float(0.1, &MirType::F32), "0x3FB99999A0000000");
    assert_eq!(fmt_float(-0.0, &MirType::F64), "0x8000000000000000");
    assert_eq!(
        fmt_float(f64::INFINITY, &MirType::F32),
        "0x7FF0000000000000"
    );
    assert_eq!(
        fmt_float(f64::NEG_INFINITY, &MirType::F32),
        "0xFFF0000000000000"
    );
    assert_eq!(fmt_float(1.0, &MirType::F16), "0xH3C00");
    assert_eq!(fmt_float(1.0, &MirType::BF16), "0xR3F80");
    let mut i = Interner::new();
    let mut b = Builder::new(i.intern("float_literals"), MirType::Void);
    for ty in [MirType::F16, MirType::BF16, MirType::F32, MirType::F64] {
        let ptr = b.alloca(ty.clone());
        for v in [
            0.1,
            1e-5,
            1e-40,
            -0.0,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NAN,
        ] {
            let value = b.build(ty.clone(), Op::ConstFloat(v, ty.clone()));
            b.build_void(Op::Store { ptr, value });
        }
    }
    b.ret(None);
    let text = module(b.finish(), &i);
    assert!(
        !text.contains("NaN") && !text.contains(" inf") && !text.contains(" -inf"),
        "{text}"
    );
}

#[test]
fn vector_neg_not_and_identity_casts() {
    let mut i = Interner::new();
    let mut b = Builder::new(i.intern("vector_ops"), MirType::Void);
    let vi = MirType::Vec(Box::new(MirType::I32), 4);
    let vf = MirType::Vec(Box::new(MirType::F32), 4);
    let a = b.add_param(vi.clone());
    let x = b.add_param(vf.clone());
    b.build(vi.clone(), Op::Neg(a));
    b.build(vi, Op::Not(a));
    b.build(vf, Op::Neg(x));
    for ty in [MirType::F16, MirType::BF16, MirType::Ptr, MirType::I32] {
        let p = b.add_param(ty.clone());
        b.build(ty.clone(), Op::Cast(CastKind::FpTrunc, p, ty));
    }
    b.ret(None);
    let text = module(b.finish(), &i);
    assert!(text.contains("sub <4 x i32> zeroinitializer"), "{text}");
    assert!(text.contains("<i32 -1, i32 -1, i32 -1, i32 -1>"), "{text}");
    assert!(text.contains("fneg <4 x float>"), "{text}");
    assert!(!text.contains("fptrunc"), "{text}");
    assert_eq!(text.matches("select i1 true").count(), 4);
}

#[test]
fn math_calls_have_declarations_and_result_bindings() {
    let mut i = Interner::new();
    let mut b = Builder::new(i.intern("math_calls"), MirType::F32);
    let p = b.add_param(MirType::F32);
    let sqrt = b.build(MirType::F32, Op::Sqrt(p));
    let round = b.build(MirType::F32, Op::Round(RoundMode::Nearest, sqrt));
    b.ret(Some(round));
    let text = module(b.finish(), &i);
    assert!(
        text.contains("declare float @llvm.sqrt.f32(float)"),
        "{text}"
    );
    assert!(
        text.contains("declare float @llvm.roundeven.f32(float)"),
        "{text}"
    );
    assert!(text.contains("%v1 = call float @llvm.sqrt.f32"), "{text}");
    assert!(
        text.contains("%v2 = call float @llvm.roundeven.f32(float %v1)"),
        "{text}"
    );
}

#[test]
fn external_declarations_and_conflicting_abis() {
    let mut i = Interner::new();
    let print = i.intern("foreign_sink");
    let mut b = Builder::new(i.intern("external_call"), MirType::Void);
    let a = b.add_param(MirType::I32);
    b.build_void(Op::Call {
        func: print,
        args: vec![a],
    });
    b.ret(None);
    let text = module(b.finish(), &i);
    assert!(text.contains("declare void @foreign_sink(i32)"), "{text}");
    let mut p = Program::new();
    let mut b = Builder::new(i.intern("conflicting_calls"), MirType::Void);
    let a = b.add_param(MirType::I32);
    let x = b.add_param(MirType::F32);
    b.build_void(Op::Call {
        func: print,
        args: vec![a],
    });
    b.build_void(Op::Call {
        func: print,
        args: vec![x],
    });
    b.ret(None);
    p.funcs.push(b.finish());
    let err = try_emit_llvm_ir(&p, &i).unwrap_err();
    assert!(
        err.contains("incompatible call signatures") && err.contains("foreign_sink"),
        "{err}"
    );
    assert!(LlvmBackend.compile(&p, p.funcs[0].name, &i).is_err());
}

#[test]
fn duplicate_successor_edges_keep_distinct_phi_values() {
    let mut i = Interner::new();
    let mut b = Builder::new(i.intern("duplicate_edges"), MirType::I32);
    let cond = b.add_param(MirType::I1);
    let a = b.add_param(MirType::I32);
    let z = b.add_param(MirType::I32);
    let target = b.new_block();
    let phi = b.block_param(target, MirType::I32);
    b.set_term(Terminator::CondBr {
        cond,
        then_blk: target,
        then_args: vec![a],
        else_blk: target,
        else_args: vec![z],
    });
    b.switch_to(target);
    b.ret(Some(phi));
    let text = module(b.finish(), &i);
    assert!(
        text.contains("phi i32 [ %v1, %bb0 ], [ %v2, %bb0.else ]"),
        "{text}"
    );
    assert!(text.contains("bb0.else:\n  br label %bb1"), "{text}");
}

#[test]
fn integer_guards_and_unordered_not_equal() {
    let mut i = Interner::new();
    let mut b = Builder::new(i.intern("integer_guards"), MirType::Void);
    for ty in [
        MirType::I8,
        MirType::I16,
        MirType::I32,
        MirType::I64,
        MirType::Vec(Box::new(MirType::I32), 4),
    ] {
        let a = b.add_param(ty.clone());
        let d = b.add_param(ty.clone());
        for op in [
            BinOp::SDiv,
            BinOp::UDiv,
            BinOp::SRem,
            BinOp::URem,
            BinOp::Shl,
            BinOp::LShr,
            BinOp::AShr,
        ] {
            b.build(ty.clone(), Op::Bin(op, a, d));
        }
    }
    let x = b.add_param(MirType::F32);
    b.build(MirType::I1, Op::Cmp(CmpOp::Fone, x, x));
    b.ret(None);
    let text = module(b.finish(), &i);
    assert!(text.contains("fcmp une float"), "{text}");
    assert_eq!(text.matches(".safe = select").count(), 20);
    assert_eq!(text.matches(".shift = and").count(), 15);
    assert_eq!(text.matches(".wrapped = select").count(), 5);
}

#[test]
fn scalar_guard_execution_modules() {
    let mut i = Interner::new();
    for (name, op) in [
        ("sdiv", BinOp::SDiv),
        ("udiv", BinOp::UDiv),
        ("srem", BinOp::SRem),
        ("urem", BinOp::URem),
        ("shl", BinOp::Shl),
        ("lshr", BinOp::LShr),
        ("ashr", BinOp::AShr),
    ] {
        let mut b = Builder::new(i.intern(name), MirType::I32);
        let a = b.add_param(MirType::I32);
        let d = b.add_param(MirType::I32);
        let result = b.build(MirType::I32, Op::Bin(op, a, d));
        b.ret(Some(result));
        let text = module(b.finish(), &i);
        assert!(text.contains("ret i32 %v2"), "{text}");
    }
    let mut b = Builder::new(i.intern("unordered_ne"), MirType::I1);
    let a = b.add_param(MirType::F64);
    let d = b.add_param(MirType::F64);
    let result = b.build(MirType::I1, Op::Cmp(CmpOp::Fone, a, d));
    b.ret(Some(result));
    let text = module(b.finish(), &i);
    assert!(text.contains("fcmp une double"), "{text}");
}

#[test]
fn overloaded_prints_have_typed_external_names() {
    let mut i = Interner::new();
    let print = i.intern("print");
    let mut b = Builder::new(i.intern("overloaded_prints"), MirType::Void);
    for ty in [
        MirType::I8,
        MirType::I32,
        MirType::F32,
        MirType::F64,
        MirType::F16,
        MirType::BF16,
    ] {
        let a = b.add_param(ty);
        b.build_void(Op::Call {
            func: print,
            args: vec![a],
        });
    }
    b.ret(None);
    let text = module(b.finish(), &i);
    for ty in ["i8", "i32", "float", "double", "half", "bfloat"] {
        assert!(
            text.contains(&format!("declare void @__wukong_print.{ty}({ty})")),
            "{text}"
        );
        assert!(
            text.contains(&format!("call void @__wukong_print.{ty}({ty}")),
            "{text}"
        );
    }
}

#[test]
fn runtime_nullable_pointer_calls_share_one_abi() {
    let mut i = Interner::new();
    let kernel = i.intern("wukong_optional_bias");
    for null_first in [false, true] {
        let mut b = Builder::new(
            i.intern(if null_first {
                "nullable_first"
            } else {
                "pointer_first"
            }),
            MirType::Void,
        );
        let ptr = b.add_param(MirType::Ptr);
        let null = b.build(MirType::I64, Op::ConstInt(0, MirType::I64));
        for a in if null_first { [null, ptr] } else { [ptr, null] } {
            b.build_void(Op::Call {
                func: kernel,
                args: vec![a],
            });
        }
        b.ret(None);
        let text = module(b.finish(), &i);
        assert!(
            text.contains("declare void @wukong_optional_bias(ptr)"),
            "{text}"
        );
        assert!(
            text.contains("call void @wukong_optional_bias(ptr null)"),
            "{text}"
        );
    }
    let mut b = Builder::new(i.intern("nonzero_pointer"), MirType::Void);
    let ptr = b.add_param(MirType::Ptr);
    let integer = b.build(MirType::I64, Op::ConstInt(7, MirType::I64));
    for a in [ptr, integer] {
        b.build_void(Op::Call {
            func: kernel,
            args: vec![a],
        });
    }
    b.ret(None);
    let mut p = Program::new();
    p.funcs.push(b.finish());
    assert!(try_emit_llvm_ir(&p, &i)
        .unwrap_err()
        .contains("incompatible call signatures"));
}

#[test]
fn discarded_defined_return_and_vector_kernel_reduction() {
    let mut i = Interner::new();
    let name = i.intern("discarded_return");
    let mut b = Builder::new(name, MirType::I32);
    let a = b.add_param(MirType::I32);
    b.build_void(Op::Call {
        func: name,
        args: vec![a],
    });
    b.ret(Some(a));
    let text = module(b.finish(), &i);
    assert!(
        text.contains("call i32 @discarded_return(i32 %v0)"),
        "{text}"
    );
    assert!(!text.contains("declare i32 @discarded_return"), "{text}");
    let mut b = Builder::new(i.intern("kernel_reduction"), MirType::F32);
    let ptrs = b.add_param(MirType::Ptr);
    let scalars = b.add_param(MirType::Ptr);
    let n = b.add_param(MirType::I64);
    let sum = b.build(
        MirType::F32,
        Op::VecKernelCall {
            kernel: 0,
            ptrs,
            scalars,
            n,
        },
    );
    b.ret(Some(sum));
    let text = module(b.finish(), &i);
    assert!(
        text.contains("declare float @__wukong_veckernel_kernel_reduction_0(ptr, ptr, i64)"),
        "{text}"
    );
    assert!(
        text.contains("%v3 = call float @__wukong_veckernel_kernel_reduction_0"),
        "{text}"
    );
}

#[test]
fn vector_integer_masks_and_lowp_casts() {
    let mut i = Interner::new();
    let mut b = Builder::new(i.intern("vector_masks_lowp_casts"), MirType::Void);
    let vf = MirType::Vec(Box::new(MirType::F32), 4);
    let vi = MirType::Vec(Box::new(MirType::I32), 4);
    let x = b.add_param(vf.clone());
    let y = b.add_param(vf.clone());
    let mask = b.build(vi, Op::Cmp(CmpOp::Folt, x, y));
    b.build(vf.clone(), Op::Select(mask, x, y));
    b.build(vf, Op::Select(x, x, y));
    for (from, to) in [
        (MirType::F16, MirType::BF16),
        (MirType::BF16, MirType::F16),
        (MirType::F64, MirType::F16),
        (MirType::F64, MirType::BF16),
    ] {
        let a = b.add_param(from);
        b.build(to.clone(), Op::Cast(CastKind::FpTrunc, a, to));
    }
    b.ret(None);
    let text = module(b.finish(), &i);
    assert!(
        text.contains("sext <4 x i1> %v2.mask to <4 x i32>"),
        "{text}"
    );
    assert!(text.contains("select <4 x i1> %v3.cond"), "{text}");
    assert_eq!(text.matches(".wide = fpext").count(), 2);
    assert_eq!(text.matches(".wide = fptrunc double").count(), 2);
    assert!(
        text.contains("fcmp une <4 x float> %v0, zeroinitializer"),
        "{text}"
    );

    let mut b = Builder::new(i.intern("unsupported_vector_lowp"), MirType::Void);
    let from = MirType::Vec(Box::new(MirType::F64), 4);
    let to = MirType::Vec(Box::new(MirType::F16), 4);
    let a = b.add_param(from);
    b.build(to.clone(), Op::Cast(CastKind::FpTrunc, a, to));
    b.ret(None);
    let mut p = Program::new();
    p.funcs.push(b.finish());
    assert!(try_emit_llvm_ir(&p, &i)
        .unwrap_err()
        .contains("vector low-precision casts"));
}
