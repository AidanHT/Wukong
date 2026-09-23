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

fn element(b: &mut Builder, ptr: ValueId, i: usize) -> ValueId {
    let index = b.build(MirType::I64, Op::ConstInt(i as i128, MirType::I64));
    b.build(
        MirType::Ptr,
        Op::Gep {
            ptr,
            index,
            elem: MirType::F32,
        },
    )
}

/// Explicit endpoint oracles catch shared implementation mistakes; finite ordinary cases
/// additionally compare the dispatched runtime with scalar and SIMD source lowering.
#[test]
fn binary_math_endpoints_across_lowering_and_backends() {
    use wukong_runtime::{VM2_ATAN2, VM2_HYPOT, VM2_POW};
    let inf = f32::INFINITY;
    let nan = f32::NAN;
    for (name, op) in [("atan2", VM2_ATAN2), ("hypot", VM2_HYPOT), ("pow", VM2_POW)] {
        let mut cases: Vec<(f32, f32, Option<f32>)> = Vec::new();
        match op {
            VM2_ATAN2 => {
                for a in [0.0f32, -0.0, 1.0, -1.0, inf, -inf, nan, -nan] {
                    for b in [0.0f32, -0.0, 1.0, -1.0, inf, -inf, nan, -nan] {
                        cases.push((a, b, Some((a as f64).atan2(b as f64) as f32)));
                    }
                }
                cases.extend([
                    (-f32::from_bits(1), f32::MAX, Some(-0.0)),
                    (f32::from_bits(1), f32::MAX, Some(0.0)),
                ]);
            }
            VM2_HYPOT => {
                for a in [0.0f32, -0.0, inf, -inf, nan, -nan] {
                    for b in [0.0f32, -0.0, inf, -inf, nan, -nan] {
                        let expected = if a.is_infinite() || b.is_infinite() {
                            inf
                        } else if a.is_nan() || b.is_nan() {
                            nan
                        } else {
                            0.0
                        };
                        cases.push((a, b, Some(expected)));
                    }
                }
                cases.extend([
                    (3.0, 4.0, Some(5.0)),
                    (f32::MAX, 0.0, Some(f32::MAX)),
                    (f32::from_bits(1), 0.0, Some(f32::from_bits(1))),
                ]);
            }
            VM2_POW => {
                for a in [0.0f32, -0.0, 1.0, -1.0, inf, -inf, nan, -nan] {
                    cases.push((a, 0.0, Some(1.0)));
                    cases.push((a, -0.0, Some(1.0)));
                    cases.push((1.0, a, Some(1.0)));
                }
                cases.extend([
                    (nan, 2.0, Some(nan)),
                    (2.0, nan, Some(nan)),
                    (inf, 2.0, Some(inf)),
                    (inf, -2.0, Some(0.0)),
                    (0.0, 2.0, Some(0.0)),
                    (0.0, -2.0, Some(inf)),
                ]);
            }
            _ => unreachable!(),
        }
        cases.extend([(0.75, 1.25, None), (2.0, 3.0, None)]);
        // Keep a scalar tail, placing an exceptional pair there as well.
        while cases.len() % 8 != 0 {
            cases.push((1.0, 1.0, None));
        }
        cases.push((nan, 0.0, Some(if op == VM2_POW { 1.0 } else { nan })));
        let n = cases.len();
        let xs: Vec<_> = cases.iter().map(|c| c.0).collect();
        let ys: Vec<_> = cases.iter().map(|c| c.1).collect();
        let mut expected = vec![0.0; n];
        unsafe {
            wukong_runtime::wukong_vmath2_f32(
                xs.as_ptr(),
                ys.as_ptr(),
                expected.as_mut_ptr(),
                n as i64,
                op,
            );
        }
        for ((a, b, oracle), actual) in cases.iter().zip(&expected) {
            if let Some(oracle) = oracle {
                assert!(
                    if oracle.is_nan() {
                        actual.is_nan()
                    } else {
                        actual.to_bits() == oracle.to_bits()
                    },
                    "{name}({a:?}, {b:?}): {actual:?} expected {oracle:?}"
                );
            }
        }
        let source = format!(
            "@generic fn scalar(x:f32,y:f32)->f32 {{ return {name}(x,y); }}
             @generic fn vector(x:[f32;{n}],y:[f32;{n}],mut out:[f32;{n}]) {{
                 for i in 0..{n} {{ out[i]={name}(x[i],y[i]); }} }}
             fn main()->i32 {{ let x:[f32;{n}]=[1.0;{n}]; let y:[f32;{n}]=[2.0;{n}];
                 let mut out:[f32;{n}]=[0.0;{n}]; vector(x,y,out); return 0; }}"
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
            "{name} must exercise SIMD"
        );
        program.funcs.retain(|f| f.name != main);
        let mut b = Builder::new(main, MirType::I32);
        let xptr = b.alloca(MirType::Array(Box::new(MirType::F32), n as u32));
        let yptr = b.alloca(MirType::Array(Box::new(MirType::F32), n as u32));
        let out = b.alloca(MirType::Array(Box::new(MirType::F32), n as u32));
        for (i, &(x, y, _)) in cases.iter().enumerate() {
            let x = b.build(MirType::F32, Op::ConstFloat(x as f64, MirType::F32));
            let y = b.build(MirType::F32, Op::ConstFloat(y as f64, MirType::F32));
            let ptr = element(&mut b, xptr, i);
            b.build_void(Op::Store { ptr, value: x });
            let ptr = element(&mut b, yptr, i);
            b.build_void(Op::Store { ptr, value: y });
            let value = b.build(
                MirType::F32,
                Op::Call {
                    func: scalar,
                    args: vec![x, y],
                },
            );
            print_bits(&mut b, print, value);
        }
        b.build_void(Op::Call {
            func: vector,
            args: vec![xptr, yptr, out],
        });
        for i in 0..n {
            let ptr = element(&mut b, out, i);
            let value = b.build(MirType::F32, Op::Load(ptr, MirType::F32));
            print_bits(&mut b, print, value);
        }
        let zero = b.build(MirType::I32, Op::ConstInt(0, MirType::I32));
        b.ret(Some(zero));
        program.funcs.push(b.finish());
        let expected: String = expected
            .iter()
            .chain(&expected)
            .map(|x| {
                format!(
                    "{}\n",
                    if x.is_nan() {
                        0x7fc0_0000u32
                    } else {
                        x.to_bits()
                    } as i32
                )
            })
            .collect();
        for level in [0, 2] {
            let mut optimized = program.clone();
            wukong_opt::optimize(&mut optimized, level);
            let errors = wukong_mir::verify::verify_program(&optimized);
            assert!(errors.is_empty(), "{name} O{level}: {errors:?}");
            assert_eq!(
                wukong_interp::run_with_output(&optimized, main, &interner).unwrap(),
                (0, expected.as_bytes().to_vec()),
                "{name} interpreter O{level}"
            );
            assert_eq!(
                wukong_codegen_cranelift::jit_run(&optimized, main, &interner).unwrap(),
                (0, expected.as_bytes().to_vec()),
                "{name} native O{level}"
            );
        }
    }
}
