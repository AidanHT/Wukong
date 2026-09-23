use wukong_mir::{Builder, CastKind, CmpOp, MirType, Op, ValueId};
use wukong_span::{Interner, SourceId, Symbol};

fn print_bits(b: &mut Builder, print: Symbol, value: ValueId) {
    let bits = b.build(
        MirType::I32,
        Op::Cast(CastKind::Bitcast, value, MirType::I32),
    );
    let nan = b.build(MirType::I1, Op::Cmp(CmpOp::Fone, value, value));
    let canonical = b.build(MirType::I32, Op::ConstInt(0x7fc0_0000, MirType::I32));
    let bits = b.build(MirType::I32, Op::Select(nan, canonical, bits));
    b.build_void(Op::Call {
        func: print,
        args: vec![bits],
    });
}

#[test]
fn half_abs_promotes_before_bit_manipulation() {
    for (name, ty) in [("f16", MirType::F16), ("bf16", MirType::BF16)] {
        let source = format!("fn magnitude(x:{name})->{name} {{ return abs(x); }}");
        let mut interner = Interner::new();
        let (ast, diagnostics) = wukong_parser::parse_module(&source, SourceId(0), &mut interner);
        assert!(diagnostics.iter().all(|d| !d.is_error()), "{diagnostics:?}");
        let (sema, diagnostics) = wukong_sema::check(&ast, &interner);
        assert!(diagnostics.iter().all(|d| !d.is_error()), "{diagnostics:?}");
        let (mut program, diagnostics) =
            wukong_mir_build::lower_program(&ast, &sema, &mut interner);
        assert!(diagnostics.iter().all(|d| !d.is_error()), "{diagnostics:?}");
        let main = interner.intern("main");
        let magnitude = interner.intern("magnitude");
        let print = interner.intern("print");
        let mut b = Builder::new(main, MirType::I32);
        let cases = [
            0.0f32,
            -0.0,
            1.0,
            -1.0,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            -f32::NAN,
        ];
        let mut expected = String::new();
        for x in cases {
            let value = b.build(MirType::F32, Op::ConstFloat(x as f64, MirType::F32));
            let value = b.build(ty.clone(), Op::Cast(CastKind::FpTrunc, value, ty.clone()));
            let result = b.build(
                ty.clone(),
                Op::Call {
                    func: magnitude,
                    args: vec![value],
                },
            );
            let result = b.build(
                MirType::F32,
                Op::Cast(CastKind::FpExt, result, MirType::F32),
            );
            print_bits(&mut b, print, result);
            expected.push_str(&format!(
                "{}\n",
                if x.is_nan() {
                    0x7fc0_0000
                } else {
                    x.abs().to_bits()
                } as i32
            ));
        }
        let zero = b.build(MirType::I32, Op::ConstInt(0, MirType::I32));
        b.ret(Some(zero));
        program.funcs.push(b.finish());
        for level in [0, 2] {
            let mut optimized = program.clone();
            wukong_opt::optimize(&mut optimized, level);
            let errors = wukong_mir::verify::verify_program(&optimized);
            assert!(errors.is_empty(), "{errors:?}");
            assert_eq!(
                wukong_interp::run_with_output(&optimized, main, &interner).unwrap(),
                (0, expected.as_bytes().to_vec()),
                "{name} interp O{level}"
            );
            assert_eq!(
                wukong_codegen_cranelift::jit_run(&optimized, main, &interner).unwrap(),
                (0, expected.as_bytes().to_vec()),
                "{name} native O{level}"
            );
        }
    }
}

/// Check actual source lowering, both scalar formulas and SIMD bodies plus tails, against
/// runtime kernels. Separate exact endpoint assertions prevent shared mistakes from passing.
/// NaNs are compared by class here: the interpreter stores f32 values widened to f64.
#[test]
fn inlined_math_specials_match_runtime_and_explicit_endpoints() {
    use wukong_runtime::{
        VM_ASINH, VM_CBRT, VM_EXP, VM_EXPM1, VM_LOG, VM_LOG1P, VM_LOGSIGMOID, VM_SIGMOID,
        VM_SOFTPLUS,
    };
    let cases = [
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
        -f32::NAN,
        0.0,
        -0.0,
        -1.0,
        -100.0,
        1.0,
        f32::from_bits(1),
        f32::from_bits(2),
        f32::from_bits(0x007f_ffff),
        f32::MIN_POSITIVE,
        f32::from_bits(0x0080_0001),
        0.5,
        2.0,
        88.0,
        89.0,
        -87.0,
        -89.0,
        f32::MAX,
        -f32::MAX,
        f32::from_bits(0x7fa1_2345),
        f32::from_bits(0xffa1_2345),
        f32::from_bits(0x0040_0000),
        f32::from_bits(0x0020_0000),
        -f32::MIN_POSITIVE,
        -f32::from_bits(1),
        1e-6,
        -1e-6,
        f32::from_bits(0x3f7f_ffff),
        f32::from_bits(0x3f80_0001),
        f32::INFINITY,
    ];
    let n = cases.len();
    for (name, op) in [
        ("exp", VM_EXP),
        ("log", VM_LOG),
        ("expm1", VM_EXPM1),
        ("log1p", VM_LOG1P),
        ("sigmoid", VM_SIGMOID),
        ("softplus", VM_SOFTPLUS),
        ("logsigmoid", VM_LOGSIGMOID),
        ("asinh", VM_ASINH),
        ("cbrt", VM_CBRT),
    ] {
        let mut expected = vec![0.0; n];
        unsafe {
            wukong_runtime::wukong_vmath_f32(cases.as_ptr(), expected.as_mut_ptr(), n as i64, op);
        }
        let positive_infinity_result = match op {
            VM_SIGMOID => 1.0,
            VM_LOGSIGMOID => -0.0,
            _ => f32::INFINITY,
        };
        assert_eq!(
            expected[0].to_bits(),
            positive_infinity_result.to_bits(),
            "{name}(+inf)"
        );
        assert!(expected[2].is_nan() && expected[3].is_nan(), "{name}(NaN)");
        match op {
            VM_EXP => {
                assert_eq!(expected[1].to_bits(), 0, "exp(-inf)");
                assert_eq!(expected[4], 1.0);
                assert_eq!(expected[5], 1.0);
            }
            VM_LOG => {
                assert!(expected[1].is_nan() && expected[6].is_nan(), "log domain");
                assert_eq!(expected[4], f32::NEG_INFINITY);
                assert_eq!(expected[5], f32::NEG_INFINITY);
                assert_eq!(expected[8].to_bits(), 0, "log(1)");
                assert!(
                    expected[9].is_finite() && expected[9] < -100.0,
                    "log min subnormal"
                );
            }
            VM_EXPM1 => {
                assert_eq!(expected[1], -1.0);
                assert_eq!(expected[4].to_bits(), 0);
                assert_eq!(expected[5].to_bits(), 0x8000_0000);
                assert_eq!(expected[7], -1.0);
            }
            VM_LOG1P => {
                assert!(expected[1].is_nan() && expected[7].is_nan());
                assert_eq!(expected[4].to_bits(), 0);
                assert_eq!(expected[5].to_bits(), 0x8000_0000);
                assert_eq!(expected[6], f32::NEG_INFINITY);
            }
            VM_ASINH | VM_CBRT => {
                assert_eq!(expected[1], f32::NEG_INFINITY);
                assert_eq!(expected[4].to_bits(), 0);
                assert_eq!(expected[5].to_bits(), 0x8000_0000);
            }
            VM_SIGMOID | VM_SOFTPLUS => assert_eq!(expected[1].to_bits(), 0),
            VM_LOGSIGMOID => assert_eq!(expected[1], f32::NEG_INFINITY),
            _ => unreachable!(),
        }
        let source = format!(
            "@generic fn scalar(x:f32)->f32 {{ return {name}(x); }}
             @generic fn vector(x:[f32;{n}], mut out:[f32;{n}]) {{
                 for i in 0..{n} {{ out[i] = {name}(x[i]); }}
             }}
             fn main()->i32 {{ let x:[f32;{n}]=[1.0;{n}]; let mut out:[f32;{n}]=[0.0;{n}];
                 vector(x,out); return scalar(1.0) as i32; }}"
        );
        let mut interner = Interner::new();
        let (ast, diagnostics) = wukong_parser::parse_module(&source, SourceId(0), &mut interner);
        assert!(diagnostics.iter().all(|d| !d.is_error()), "{diagnostics:?}");
        let (sema, diagnostics) = wukong_sema::check(&ast, &interner);
        assert!(diagnostics.iter().all(|d| !d.is_error()), "{diagnostics:?}");
        let (mut program, diagnostics) =
            wukong_mir_build::lower_program(&ast, &sema, &mut interner);
        assert!(diagnostics.iter().all(|d| !d.is_error()), "{diagnostics:?}");
        let main = interner.intern("main");
        let scalar = interner.intern("scalar");
        let vector = interner.intern("vector");
        let print = interner.intern("print");
        assert!(
            program
                .function(vector)
                .unwrap()
                .value_types
                .iter()
                .any(|ty| matches!(ty, MirType::Vec(_, _))),
            "{name} must exercise SIMD MIR"
        );
        program.funcs.retain(|f| f.name != main);
        let mut b = Builder::new(main, MirType::I32);
        let input = b.alloca(MirType::Array(Box::new(MirType::F32), n as u32));
        let output = b.alloca(MirType::Array(Box::new(MirType::F32), n as u32));
        for (i, &value) in cases.iter().enumerate() {
            let value = b.build(MirType::F32, Op::ConstFloat(value as f64, MirType::F32));
            let index = b.build(MirType::I64, Op::ConstInt(i as i128, MirType::I64));
            let ptr = b.build(
                MirType::Ptr,
                Op::Gep {
                    ptr: input,
                    index,
                    elem: MirType::F32,
                },
            );
            b.build_void(Op::Store { ptr, value });
            let result = b.build(
                MirType::F32,
                Op::Call {
                    func: scalar,
                    args: vec![value],
                },
            );
            print_bits(&mut b, print, result);
        }
        b.build_void(Op::Call {
            func: vector,
            args: vec![input, output],
        });
        for i in 0..n {
            let index = b.build(MirType::I64, Op::ConstInt(i as i128, MirType::I64));
            let ptr = b.build(
                MirType::Ptr,
                Op::Gep {
                    ptr: output,
                    index,
                    elem: MirType::F32,
                },
            );
            let value = b.build(MirType::F32, Op::Load(ptr, MirType::F32));
            print_bits(&mut b, print, value);
        }
        let zero = b.build(MirType::I32, Op::ConstInt(0, MirType::I32));
        b.ret(Some(zero));
        program.funcs.push(b.finish());
        let line = |x: &f32| {
            format!(
                "{}\n",
                if x.is_nan() {
                    0x7fc0_0000u32
                } else {
                    x.to_bits()
                } as i32
            )
        };
        let expected: String = expected.iter().chain(&expected).map(line).collect();
        // The source scalar entry also lets us check raw NaN payloads/signs without the
        // interpreter's f32-to-f64 storage conversion, including signaling input payloads.
        let handle = wukong_codegen_cranelift::jit_module(&program, &interner).unwrap();
        let scalar_call: unsafe extern "C" fn(f32) -> f32 =
            unsafe { std::mem::transmute(handle.func_ptr(scalar).unwrap()) };
        for input in [0x7fc1_2345, 0xffc1_2345, 0x7fa1_2345, 0xffa1_2345] {
            let x = f32::from_bits(input);
            let mut kernel = 0.0f32;
            unsafe {
                wukong_runtime::wukong_vmath_f32(&x, &mut kernel, 1, op);
            }
            assert_eq!(
                unsafe { scalar_call(x) }.to_bits(),
                kernel.to_bits(),
                "{name} NaN {input:#x}"
            );
        }
        for level in [0, 2] {
            let mut optimized = program.clone();
            wukong_opt::optimize(&mut optimized, level);
            let errors = wukong_mir::verify::verify_program(&optimized);
            assert!(errors.is_empty(), "{name} O{level}: {errors:?}");
            let interpreted = wukong_interp::run_with_output(&optimized, main, &interner).unwrap();
            let native = wukong_codegen_cranelift::jit_run(&optimized, main, &interner).unwrap();
            assert_eq!(
                interpreted,
                (0, expected.as_bytes().to_vec()),
                "{name} interpreter O{level}"
            );
            assert_eq!(
                native,
                (0, expected.as_bytes().to_vec()),
                "{name} native O{level}"
            );
        }
    }
}
