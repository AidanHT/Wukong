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
    // A half/bfloat constant is its grid-rounded value in a `float` register (the interpreter's
    // `ConstFloat` rounding): 0.1 rounds to 0.0999755859375 on the f16 grid and to 0.10009765625
    // on the bf16 grid (hand-derived: f32 0x3DCCCCCD rounds to f16 0x2E66 and to bf16 0x3DCD).
    assert_eq!(fmt_float(1.0, &MirType::F16), "0x3FF0000000000000");
    assert_eq!(fmt_float(0.1, &MirType::F16), "0x3FB9980000000000");
    assert_eq!(fmt_float(0.1, &MirType::BF16), "0x3FB9A00000000000");
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
    // Pointer and integer same-type casts are identities. A same-type f16/bf16 `fptrunc` is not:
    // the interpreter rounds the f32 register onto the grid, so it is a narrowing round trip.
    // Values: a=%v0, x=%v1, the three ops %v2..%v4, then each param/cast pair from %v5.
    assert!(text.contains("fptrunc float %v5 to half"), "{text}");
    assert!(text.contains("fptrunc float %v7 to bfloat"), "{text}");
    assert!(
        !text.contains("fptrunc half") && !text.contains("fptrunc bfloat"),
        "{text}"
    );
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
fn prints_and_asserts_use_the_aot_runtime_abi() {
    let mut i = Interner::new();
    let print = i.intern("print");
    let mut b = Builder::new(i.intern("runtime_prints"), MirType::Void);
    for ty in [
        MirType::I8,
        MirType::I32,
        MirType::F32,
        MirType::F64,
        MirType::F16,
        MirType::BF16,
        MirType::I1,
        MirType::I64,
        MirType::Ptr,
    ] {
        let a = b.add_param(ty);
        b.build_void(Op::Call {
            func: print,
            args: vec![a],
        });
    }
    let unsigned = b.add_param(MirType::I64);
    b.build_void(Op::Call {
        func: i.intern("println_u"),
        args: vec![unsigned],
    });
    let text_ptr = b.add_param(MirType::Ptr);
    b.build_void(Op::Call {
        func: i.intern("print_str"),
        args: vec![text_ptr],
    });
    let flag = b.add_param(MirType::F32);
    b.build_void(Op::Call {
        func: i.intern("assert"),
        args: vec![flag],
    });
    b.ret(None);
    let text = module(b.finish(), &i);
    for decl in [
        "declare void @wukong_rt_print_i64(i64)",
        "declare void @wukong_rt_print_f64(double)",
        "declare void @wukong_rt_print_u64(i64)",
        "declare void @wukong_rt_print_str(ptr)",
        "declare void @wukong_rt_assert(i64)",
    ] {
        assert!(text.contains(decl), "{decl}:\n{text}");
    }
    // Exactly Cranelift's widening: ints sign-extend, booleans zero-extend, every float (f16/bf16
    // live in f32 registers) widens exactly to double, pointers print their address.
    for line in [
        "sext i8 %v0 to i64",
        "sext i32 %v1 to i64",
        "fpext float %v2 to double",
        "call void @wukong_rt_print_f64(double %v3)",
        "fpext float %v4 to double",
        "fpext float %v5 to double",
        "zext i1 %v6 to i64",
        "call void @wukong_rt_print_i64(i64 %v7)",
        "ptrtoint ptr %v8 to i64",
        "call void @wukong_rt_print_u64(i64 %v9)",
        "call void @wukong_rt_print_str(ptr %v10)",
        "fcmp une float %v11, 0.0",
    ] {
        assert!(text.contains(line), "{line}:\n{text}");
    }
    assert!(
        !text.contains("@print(") && !text.contains("__wukong_print"),
        "{text}"
    );
}

#[test]
fn runtime_kernel_integers_widen_to_the_c_abi() {
    let mut i = Interner::new();
    let kernel = i.intern("wukong_probe_kernel");
    let mut b = Builder::new(i.intern("kernel_abi"), MirType::Void);
    let p = b.add_param(MirType::Ptr);
    let n = b.add_param(MirType::I32);
    let alpha = b.add_param(MirType::F32);
    let flag = b.add_param(MirType::I1);
    b.build_void(Op::Call {
        func: kernel,
        args: vec![p, n, alpha, flag],
    });
    b.ret(None);
    let text = module(b.finish(), &i);
    assert!(
        text.contains("declare void @wukong_probe_kernel(ptr, i64, float, i64)"),
        "{text}"
    );
    assert!(text.contains("%t0 = sext i32 %v1 to i64"), "{text}");
    assert!(text.contains("%t1 = zext i1 %v3 to i64"), "{text}");
    assert!(
        text.contains("call void @wukong_probe_kernel(ptr %v0, i64 %t0, float %v2, i64 %t1)"),
        "{text}"
    );
}

#[test]
fn half_values_compute_in_f32_registers_and_round_at_memory() {
    let mut i = Interner::new();
    let mut b = Builder::new(i.intern("half_registers"), MirType::BF16);
    let x = b.add_param(MirType::BF16);
    let y = b.add_param(MirType::BF16);
    let sum = b.build(MirType::BF16, Op::Bin(BinOp::FAdd, x, y));
    let slot = b.alloca(MirType::BF16);
    b.build_void(Op::Store {
        ptr: slot,
        value: sum,
    });
    let back = b.build(MirType::BF16, Op::Load(slot, MirType::BF16));
    let widened = b.build(MirType::F32, Op::Cast(CastKind::FpExt, back, MirType::F32));
    let narrowed = b.build(
        MirType::BF16,
        Op::Cast(CastKind::FpTrunc, widened, MirType::BF16),
    );
    b.ret(Some(narrowed));
    let text = module(b.finish(), &i);
    assert!(
        text.contains("define float @half_registers(float %v0, float %v1)"),
        "{text}"
    );
    // The add is f32 arithmetic on f32 registers: no intermediate rounding to the bf16 grid.
    assert!(text.contains("%v2 = fadd float %v0, %v1"), "{text}");
    assert!(!text.contains("fadd bfloat"), "{text}");
    assert!(text.contains("alloca bfloat"), "{text}");
    assert!(text.contains("%t0 = fptrunc float %v2 to bfloat"), "{text}");
    assert!(text.contains("store bfloat %t0"), "{text}");
    assert!(text.contains("%v4.mem = load bfloat"), "{text}");
    assert!(
        text.contains("%v4 = fpext bfloat %v4.mem to float"),
        "{text}"
    );
    // Widening a bf16 value snaps the register onto the grid first; narrowing rounds again.
    assert!(text.contains("fptrunc float %v4 to bfloat"), "{text}");
    assert!(text.contains("fptrunc float %v5 to bfloat"), "{text}");
    assert!(text.contains("ret float %v6"), "{text}");
}

#[test]
fn booleans_are_bytes_in_memory() {
    let mut i = Interner::new();
    let mut b = Builder::new(i.intern("bool_memory"), MirType::I1);
    let flag = b.add_param(MirType::I1);
    let mask = b.add_param(MirType::Vec(Box::new(MirType::I1), 4));
    let slot = b.alloca(MirType::I1);
    b.build_void(Op::Store {
        ptr: slot,
        value: flag,
    });
    let vslot = b.alloca(MirType::Vec(Box::new(MirType::I1), 4));
    b.build_void(Op::Store {
        ptr: vslot,
        value: mask,
    });
    let back = b.build(MirType::I1, Op::Load(slot, MirType::I1));
    b.ret(Some(back));
    let text = module(b.finish(), &i);
    assert!(text.contains("alloca i8"), "{text}");
    assert!(text.contains("alloca <4 x i8>"), "{text}");
    assert!(text.contains("zext i1 %v0 to i8"), "{text}");
    assert!(text.contains("zext <4 x i1> %v1 to <4 x i8>"), "{text}");
    assert!(text.contains("trunc i8 %v"), "{text}");
    assert!(
        !text.contains("store i1") && !text.contains("load i1"),
        "{text}"
    );
}

#[test]
fn module_names_the_host_triple() {
    let mut i = Interner::new();
    let mut b = Builder::new(i.intern("triple"), MirType::Void);
    b.ret(None);
    let text = module(b.finish(), &i);
    if let Some(triple) = host_triple() {
        assert!(
            text.contains(&format!("target triple = \"{triple}\"")),
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
    let n = b.add_param(MirType::I32);
    // dot(x, y): two unit-stride loads, folded with a fused `acc = fma(x, y, acc)`.
    let kernel = b.add_vec_kernel(wukong_mir::VecKernel {
        name: i.intern("dot"),
        lane: wukong_mir::VecLane::F32,
        streams: 2,
        scalars: 0,
        ops: vec![
            wukong_mir::VecOp::Load { stream: 0 },
            wukong_mir::VecOp::Load { stream: 1 },
        ],
        unroll: 2,
        reduce: Some(wukong_mir::VecReduce {
            op: wukong_mir::VecRedOp::Add,
            value: 0,
            fma: Some((0, 1)),
        }),
    });
    let sum = b.build(
        MirType::F32,
        Op::VecKernelCall {
            kernel,
            ptrs,
            scalars,
            n,
        },
    );
    b.ret(Some(sum));
    let f = b.finish();
    let copies = f.vec_kernels[0].reduction_unroll();
    let text = module(f, &i);
    // The kernel is DEFINED in the module (no external symbol to link) ...
    assert!(
        text.contains(
            "define internal float @__wukong_veckernel_kernel_reduction_0(ptr %ptrs, ptr %scalars, i64 %n)"
        ),
        "{text}"
    );
    assert!(
        !text.contains("declare float @__wukong_veckernel"),
        "{text}"
    );
    assert!(
        text.contains("declare float @llvm.fma.f32(float, float, float)"),
        "{text}"
    );
    // ... with the evaluator's accumulator layout (U copies of 8 lanes) and fused fold ...
    assert!(
        text.contains(&format!("%acc = alloca [{} x float]", copies * 8)),
        "{text}"
    );
    assert!(
        text.contains("%m.acc = call float @llvm.fma.f32(float %m0, float %m1, float %j.old)"),
        "{text}"
    );
    assert!(
        text.contains("%c.acc = call float @llvm.fma.f32(float %c0, float %c1, float %l.old)"),
        "{text}"
    );
    // ... and the in-order horizontal fold ends at lane 7.
    assert!(text.contains("ret float %h7"), "{text}");
    // The call zero-extends the element count to the kernel's i64, as Cranelift does.
    assert!(text.contains("%t0 = zext i32 %v2 to i64"), "{text}");
    assert!(
        text.contains(
            "%v3 = call float @__wukong_veckernel_kernel_reduction_0(ptr %v0, ptr %v1, i64 %t0)"
        ),
        "{text}"
    );

    let mut b = Builder::new(i.intern("kernel_map"), MirType::Void);
    let ptrs = b.add_param(MirType::Ptr);
    let scalars = b.add_param(MirType::Ptr);
    let n = b.add_param(MirType::I64);
    // out[i] = x[i] != x[i] ? x[i] : -(x[i] * s)
    let kernel = b.add_vec_kernel(wukong_mir::VecKernel {
        name: i.intern("map"),
        lane: wukong_mir::VecLane::F32,
        streams: 2,
        scalars: 1,
        ops: vec![
            wukong_mir::VecOp::Load { stream: 0 },
            wukong_mir::VecOp::Splat { scalar: 0 },
            wukong_mir::VecOp::Bin {
                op: wukong_mir::VecBin::Mul,
                a: 0,
                b: 1,
            },
            wukong_mir::VecOp::Neg { a: 2 },
            wukong_mir::VecOp::Cmp {
                pred: wukong_mir::VecCmp::Ne,
                a: 0,
                b: 0,
            },
            wukong_mir::VecOp::Select {
                mask: 4,
                a: 0,
                b: 3,
            },
            wukong_mir::VecOp::Store { stream: 1, val: 5 },
        ],
        unroll: 1,
        reduce: None,
    });
    b.build_void(Op::VecKernelCall {
        kernel,
        ptrs,
        scalars,
        n,
    });
    b.ret(None);
    let text = module(b.finish(), &i);
    for line in [
        "define internal void @__wukong_veckernel_kernel_map_0(ptr %ptrs, ptr %scalars, i64 %n)",
        "%scalar0 = load float, ptr %scalar0.p",
        "%x2 = fmul float %x0, %scalar0",
        "%x3 = fneg float %x2",
        "%x4.c = fcmp une float %x0, %x0",
        "%x5 = select i1 %x5.c, float %x0, float %x3",
        "store float %x5, ptr %x6.a",
        "call void @__wukong_veckernel_kernel_map_0(ptr %v0, ptr %v1, i64 %v2)",
    ] {
        assert!(text.contains(line), "{line}:\n{text}");
    }

    let mut b = Builder::new(i.intern("kernel_missing"), MirType::Void);
    let ptrs = b.add_param(MirType::Ptr);
    let n = b.add_param(MirType::I64);
    b.build_void(Op::VecKernelCall {
        kernel: 3,
        ptrs,
        scalars: ptrs,
        n,
    });
    b.ret(None);
    let mut p = Program::new();
    p.funcs.push(b.finish());
    assert!(try_emit_llvm_ir(&p, &i)
        .unwrap_err()
        .contains("vector kernel #3"));
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
    // f16 <-> bf16 rounds the f32 register straight onto the target grid; a double source first
    // rounds to f32 (`round_xx(v as f32)`, the interpreter's double rounding).
    assert!(text.contains("fptrunc float %v5 to bfloat"), "{text}");
    assert!(text.contains("fptrunc float %v7 to half"), "{text}");
    assert_eq!(text.matches("fptrunc double").count(), 2, "{text}");
    assert!(
        text.contains("fcmp une <4 x float> %v0, zeroinitializer"),
        "{text}"
    );

    // The vector form takes the same f32 lane adapter instead of declining.
    let mut b = Builder::new(i.intern("vector_lowp"), MirType::Void);
    let from = MirType::Vec(Box::new(MirType::F64), 4);
    let to = MirType::Vec(Box::new(MirType::F16), 4);
    let a = b.add_param(from);
    b.build(to.clone(), Op::Cast(CastKind::FpTrunc, a, to));
    b.ret(None);
    let text = module(b.finish(), &i);
    for line in [
        "%t0 = fptrunc <4 x double> %v0 to <4 x float>",
        "%t1 = fptrunc <4 x float> %t0 to <4 x half>",
        "%t2 = fpext <4 x half> %t1 to <4 x float>",
        "%v1 = select i1 true, <4 x float> %t2, <4 x float> %t2",
    ] {
        assert!(text.contains(line), "{line}:\n{text}");
    }
}
