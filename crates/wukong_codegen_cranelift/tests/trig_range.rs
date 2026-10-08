//! Full-range trig: independent f64 accuracy, exact implementation agreement, and scalar AD.
use wukong_mir::{MirType, Op, Program};
use wukong_runtime::{VM_COS, VM_SIN, VM_TAN};
use wukong_span::{Interner, SourceId};

const OPS: [(i64, &str); 3] = [(VM_SIN, "sin"), (VM_COS, "cos"), (VM_TAN, "tan")];

fn samples() -> Vec<f32> {
    let mut values = vec![0.0, -0.0, 0.125, -0.5, 1.0, -1.0];
    for bits in [4096.0f32.to_bits(), 0x4f6e_6b28, f32::MAX.to_bits()] {
        for delta in [-1i64, 0, 1] {
            let v = f32::from_bits((i64::from(bits) + delta) as u32);
            values.extend([v, -v]);
        }
    }
    // Exponent coverage, without constructing expectations from a Wukong implementation.
    for exponent in (140u32..255).step_by(11) {
        let v = f32::from_bits((exponent << 23) | 0x0055_5555);
        values.extend([v, -v]);
    }
    // Adjacent floats around poles and nearest-quadrant ties, including near the fast cutoff.
    for k in [0, 1, 17, 511, 1303] {
        for angle in [
            (k as f64 + 0.5) * std::f64::consts::PI,
            (k as f64 + 0.5) * std::f64::consts::FRAC_PI_2,
        ] {
            let bits = (angle as f32).to_bits();
            for offset in [-1i64, 0, 1] {
                let v = f32::from_bits((i64::from(bits) + offset) as u32);
                values.extend([v, -v]);
            }
        }
    }
    values.extend([
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::from_bits(0x7fc1_2345),
        f32::from_bits(0xffc2_3456),
        f32::from_bits(0x7fa1_2345),
    ]);
    values
}

fn reference(x: f32, op: i64) -> f64 {
    let x = x as f64;
    match op {
        VM_SIN => x.sin(),
        VM_COS => x.cos(),
        VM_TAN => x.tan(),
        _ => unreachable!(),
    }
}

fn assert_accuracy(actual: f32, x: f32, op: i64, label: &str) {
    if !x.is_finite() {
        assert!(
            actual.is_nan(),
            "{label}: nonfinite {x:?} produced {actual}"
        );
        return;
    }
    let expected = reference(x, op);
    // Existing runtime bounds: vmath_matches_libm sin/cos and vmath_inverse_trig tan.
    let tolerance = if op == VM_TAN { 5e-5 } else { 1e-4 };
    assert!(actual.is_finite(), "{label}: finite {x} produced {actual}");
    assert!(
        (actual as f64 - expected).abs() <= tolerance + tolerance * expected.abs(),
        "{label}: x={x:?}, got={actual:?}, f64={expected:?}"
    );
    if x == 0.0 {
        let expected = if op == VM_COS { 1.0 } else { x };
        assert_eq!(actual.to_bits(), expected.to_bits(), "{label}: signed zero");
    }
}

fn scalar(x: f32, op: i64) -> f32 {
    let mut out = 0.0;
    unsafe { wukong_runtime::wukong_vmath_f32(&x, &mut out, 1, op) };
    out
}

#[test]
fn full_range_runtime_lanes_match_scalar_and_f64() {
    for (op, name) in OPS {
        for x in samples() {
            let input = [x; 9];
            let mut output = [0.0; 9];
            unsafe { wukong_runtime::wukong_vmath_f32(input.as_ptr(), output.as_mut_ptr(), 9, op) };
            let one = scalar(x, op);
            assert_accuracy(one, x, op, name);
            for (lane, actual) in output.into_iter().enumerate() {
                assert_eq!(actual.to_bits(), one.to_bits(), "{name} {x:?} lane {lane}");
            }
            if op != VM_TAN {
                assert_eq!(
                    wukong_runtime::wukong_rt_trig_f32(x, op).to_bits(),
                    one.to_bits()
                );
                if x.is_nan() {
                    assert_eq!(one.to_bits(), x.to_bits() | 0x0040_0000, "quiet payload");
                } else if x.is_infinite() {
                    assert_eq!(one.to_bits(), 0x7fc0_0000, "canonical domain NaN");
                }
            }
            // Agent-local C mirror validation consumes these bit records, in addition to
            // the independent f64 assertions above; these are not mathematical expectations.
            println!("TRIG_BITS {op} {:08x} {:08x}", x.to_bits(), one.to_bits());
        }
    }
    for x in [0.0, -0.0, 4e9, f32::from_bits(0x7fa1_2345)] {
        assert_eq!(
            wukong_runtime::wukong_rt_trig_f32(x, -1).to_bits(),
            x.to_bits()
        );
    }
    // Check this exact FMA/nearest-quadrant implementation on a grid across the retained
    // interval; Cephes' related accuracy report alone does not certify our operation order.
    let input: Vec<f32> = (-8192..=8192).map(|i| i as f32 * 0.5).collect();
    for (op, name) in OPS {
        let mut output = vec![0.0; input.len()];
        unsafe {
            wukong_runtime::wukong_vmath_f32(
                input.as_ptr(),
                output.as_mut_ptr(),
                input.len() as i64,
                op,
            )
        };
        for (&x, actual) in input.iter().zip(output) {
            assert_accuracy(actual, x, op, name);
            assert_eq!(
                actual.to_bits(),
                scalar(x, op).to_bits(),
                "{name} cutoff interval x={x}"
            );
        }
    }
}

fn lower(source: &str, recognize: bool) -> (Program, Interner) {
    let mut interner = Interner::new();
    let (ast, diagnostics) = wukong_parser::parse_module(source, SourceId(0), &mut interner);
    assert!(diagnostics.is_empty(), "parse: {diagnostics:?}");
    let (sema, diagnostics) = wukong_sema::check(&ast, &interner);
    assert!(diagnostics.is_empty(), "sema: {diagnostics:?}");
    let (program, diagnostics) = wukong_mir_build::lower_program_with(
        &ast,
        &sema,
        &mut interner,
        wukong_mir_build::LowerOptions {
            recognize,
            vec_kernels: false,
            ..Default::default()
        },
    );
    assert!(diagnostics.is_empty(), "lower: {diagnostics:?}");
    (program, interner)
}

fn check_dispatch(program: &Program, interner: &Interner, recognize: bool, operation: i64) {
    let mut array_calls = 0;
    let mut scalar_calls = 0;
    let mut vector_fmas = 0;
    for function in &program.funcs {
        for inst in function.blocks.iter().flat_map(|block| &block.insts) {
            match &inst.op {
                Op::Call { func, args } if interner.resolve(*func) == "wukong_vmath_f32" => {
                    assert_eq!(args.len(), 4);
                    assert_eq!(constant(function, args[3]), Some(operation as i128));
                    array_calls += 1;
                }
                Op::Call { func, args } if interner.resolve(*func) == "wukong_rt_trig_f32" => {
                    assert_eq!(args.len(), 2);
                    assert_eq!(function.value_type(args[0]), &MirType::F32);
                    assert_eq!(function.value_type(inst.result.unwrap()), &MirType::F32);
                    let opcode = constant(function, args[1]).unwrap();
                    assert!(if operation == VM_TAN {
                        [15, 16].contains(&opcode)
                    } else {
                        opcode == operation as i128
                    });
                    scalar_calls += 1;
                }
                Op::Fma(..)
                    if matches!(
                        function.value_type(inst.result.unwrap()),
                        MirType::Vec(_, _)
                    ) =>
                {
                    vector_fmas += 1
                }
                _ => {}
            }
        }
    }
    assert_eq!(
        array_calls,
        usize::from(recognize),
        "array recognition switch"
    );
    if !recognize {
        assert!(
            scalar_calls > 0,
            "generic trig must retain full-range fallback"
        );
        assert!(
            vector_fmas > 0,
            "ordinary generic lanes must retain SIMD polynomial"
        );
    }
}

fn constant(function: &wukong_mir::Function, value: wukong_mir::ValueId) -> Option<i128> {
    function
        .blocks
        .iter()
        .flat_map(|block| &block.insts)
        .find_map(|inst| {
            if inst.result == Some(value) {
                if let Op::ConstInt(value, _) = &inst.op {
                    return Some(*value);
                }
            }
            None
        })
}

#[test]
fn source_recognition_and_generic_simd_match_at_o0_o2() {
    for (op, name) in OPS {
        let source = format!(
            "fn entry(x: [f32; 17], mut out: [f32; 17]) {{
            for i in 0..17 {{ out[i] = {name}(x[i]); }} }}"
        );
        for recognize in [false, true] {
            let (program, mut interner) = lower(&source, recognize);
            check_dispatch(&program, &interner, recognize, op);
            let entry = interner.intern("entry");
            for level in [0, 2] {
                let mut optimized = program.clone();
                wukong_opt::optimize_verified(&mut optimized, level).unwrap();
                check_dispatch(&optimized, &interner, recognize, op);
                if let Some(dir) = std::env::var_os("WUKONG_TRIG_ARTIFACT_DIR") {
                    let dir = std::path::PathBuf::from(dir);
                    std::fs::create_dir_all(&dir).unwrap();
                    let llvm =
                        wukong_codegen_llvm::try_emit_llvm_ir(&optimized, &interner).unwrap();
                    std::fs::write(
                        dir.join(format!("{name}-rec{}-O{level}.ll", u8::from(recognize))),
                        llvm,
                    )
                    .unwrap();
                }
                let module = wukong_codegen_cranelift::jit_module(&optimized, &interner).unwrap();
                let function: unsafe extern "C" fn(*const f32, *mut f32) =
                    unsafe { std::mem::transmute(module.func_ptr(entry).unwrap()) };
                for x in samples() {
                    // Mixed blocks force exceptional lanes beside ordinary polynomial lanes;
                    // positions 7/8/15/16 also straddle SIMD blocks and the scalar tail.
                    let mut input: Vec<f32> = (0..17).map(|i| (i as f32 - 8.0) / 8.0).collect();
                    input[0] = -0.0;
                    for lane in [1, 7, 8, 15, 16] {
                        input[lane] = x;
                    }
                    let mut ix = input.clone();
                    let mut interpreted = vec![1234.0; 17];
                    let mut native = interpreted.clone();
                    wukong_interp::run_kernel_f32(
                        &optimized,
                        entry,
                        &mut [&mut ix, &mut interpreted],
                        &interner,
                    )
                    .unwrap();
                    unsafe { function(input.as_ptr(), native.as_mut_ptr()) };
                    for lane in 0..17 {
                        assert_accuracy(native[lane], input[lane], op, name);
                        assert_eq!(
                            native[lane].to_bits(),
                            interpreted[lane].to_bits(),
                            "{name} recognize={recognize} O{level} x={x:?} lane={lane}"
                        );
                        assert_eq!(
                            native[lane].to_bits(),
                            scalar(input[lane], op).to_bits(),
                            "{name} scalar agreement lane={lane}"
                        );
                        assert_eq!(ix[lane].to_bits(), input[lane].to_bits());
                    }
                }
            }
        }
    }
}

#[test]
fn standalone_trig_ad_remains_single_block_at_o0_o2() {
    for (op, name) in OPS {
        let source = format!("fn loss(x: [f32; 1]) -> f32 {{ return {name}(x[0]); }}");
        let (program, mut interner) = lower(&source, false);
        let loss = interner.intern("loss");
        for level in [0, 2] {
            let mut optimized = program.clone();
            wukong_opt::optimize_verified(&mut optimized, level).unwrap();
            let forward = optimized.funcs.iter().find(|f| f.name == loss).unwrap();
            assert_eq!(
                forward.blocks.len(),
                1,
                "{name} O{level}: scalar AD candidate lost"
            );
            let gradient = wukong_autodiff::grad(forward, &[0], &mut interner).unwrap();
            let entry = gradient.name;
            optimized.funcs.push(gradient);
            assert!(wukong_mir::verify::verify_program(&optimized).is_empty());
            let module = wukong_codegen_cranelift::jit_module(&optimized, &interner).unwrap();
            let function: unsafe extern "C" fn(*const f32, *mut f32) =
                unsafe { std::mem::transmute(module.func_ptr(entry).unwrap()) };
            for x in samples().into_iter().filter(|x| x.is_finite()) {
                let mut ix = [x];
                let mut interpreted = [0.0];
                let mut native = [0.0];
                wukong_interp::run_kernel_f32(
                    &optimized,
                    entry,
                    &mut [&mut ix, &mut interpreted],
                    &interner,
                )
                .unwrap();
                unsafe { function(&x, native.as_mut_ptr()) };
                let xf = x as f64;
                let expected = match op {
                    VM_SIN => xf.cos(),
                    VM_COS => -xf.sin(),
                    VM_TAN => 1.0 + xf.tan().powi(2),
                    _ => unreachable!(),
                };
                // Existing autodiff closed-form bound (vmath_algebraic_activation_vjps).
                assert!(
                    (native[0] as f64 - expected).abs() <= 3e-3 + 3e-3 * expected.abs(),
                    "{name}'({x})={} vs {expected}",
                    native[0]
                );
                assert_eq!(native[0].to_bits(), interpreted[0].to_bits());
            }
        }
    }
}

#[test]
fn scalar_trig_llvm_and_object_imports_are_typed() {
    let (program, interner) = lower(
        "fn entry(x: [f32; 1], mut out: [f32; 3]) {
        out[0] = sin(x[0]); out[1] = cos(x[0]); out[2] = tan(x[0]); }",
        false,
    );
    let llvm = wukong_codegen_llvm::try_emit_llvm_ir(&program, &interner).unwrap();
    assert!(
        llvm.contains("declare float @wukong_rt_trig_f32(float, i64)"),
        "{llvm}"
    );
    let object = wukong_codegen_cranelift::emit_object(&program, &interner).unwrap();
    assert!(!object.is_empty());
    // Optional artifact export supports an actual C-fallback link in agent-local validation.
    // All compiler/assertion work above runs regardless of this export destination.
    if let Some(dir) = std::env::var_os("WUKONG_TRIG_ARTIFACT_DIR") {
        let dir = std::path::PathBuf::from(dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("trig-source.o"), object).unwrap();
        std::fs::write(dir.join("trig-source.ll"), llvm).unwrap();
    }
}

/// The number of representable doubles between `a` and `b` (±0 count as the same point).
fn ulp_diff64(a: f64, b: f64) -> u64 {
    fn ordered(x: f64) -> i64 {
        let bits = x.to_bits() as i64;
        if bits < 0 {
            i64::MIN - bits
        } else {
            bits
        }
    }
    (ordered(a) - ordered(b)).unsigned_abs()
}

/// An `f64` trig call is the runtime's double-precision library on both backends — no demotion
/// through the f32 polynomial, no `fptrunc` anywhere in the MIR — and the result is within 2 ULP
/// of the host libm on every finite argument, including the ones that need Payne–Hanek
/// reduction (`1e22`, `1e300`). Before this contract an `f64` `sin` was `(x as f32).sin() as
/// f64`: seven significant digits on a sixteen-digit type.
#[test]
fn f64_source_uses_the_double_precision_runtime() {
    let (program, mut interner) = lower(
        "fn entry(x: [f64; 1], mut out: [f64; 3]) {
        out[0] = sin(x[0]); out[1] = cos(x[0]); out[2] = tan(x[0]); }",
        false,
    );
    let mut codes = Vec::new();
    for function in &program.funcs {
        for inst in function.blocks.iter().flat_map(|block| &block.insts) {
            match &inst.op {
                Op::Call { func, args } if interner.resolve(*func) == "wukong_rt_math_f64" => {
                    assert_eq!(args.len(), 2);
                    assert_eq!(function.value_type(args[0]), &MirType::F64);
                    assert_eq!(function.value_type(inst.result.unwrap()), &MirType::F64);
                    codes.push(constant(function, args[1]).unwrap());
                }
                Op::Call { func, .. } if interner.resolve(*func) == "wukong_rt_trig_f32" => {
                    panic!("an f64 trig call reached the f32 runtime");
                }
                Op::Cast(wukong_mir::CastKind::FpTrunc, _, _) => {
                    panic!("an f64 trig call demoted its operand to f32");
                }
                _ => {}
            }
        }
    }
    assert_eq!(
        codes,
        vec![VM_SIN as i128, VM_COS as i128, VM_TAN as i128],
        "one f64 runtime call per intrinsic, in source order"
    );
    let entry = interner.intern("entry");
    for level in [0, 2] {
        let mut optimized = program.clone();
        wukong_opt::optimize_verified(&mut optimized, level).unwrap();
        let module = wukong_codegen_cranelift::jit_module(&optimized, &interner).unwrap();
        let function: unsafe extern "C" fn(*const f64, *mut f64) =
            unsafe { std::mem::transmute(module.func_ptr(entry).unwrap()) };
        for x in [
            0.0f64,
            -0.0,
            0.5,
            -1.5,
            4096.0001,
            4_000_000_017.0,
            -4_000_000_017.0,
            f32::MAX as f64,
            1e22,
            1e300,
            (1u64 << 60) as f64,
        ] {
            let mut input = [x];
            let mut interpreted = [0.0; 3];
            let mut native = [0.0; 3];
            wukong_interp::run_kernel_f64(
                &optimized,
                entry,
                &mut [&mut input, &mut interpreted],
                &interner,
            )
            .unwrap();
            unsafe { function(&x, native.as_mut_ptr()) };
            let expected = [
                wukong_runtime::math64::sin(x),
                wukong_runtime::math64::cos(x),
                wukong_runtime::math64::tan(x),
            ];
            let host = [x.sin(), x.cos(), x.tan()];
            for lane in 0..3 {
                assert_eq!(
                    native[lane].to_bits(),
                    expected[lane].to_bits(),
                    "f64 O{level} x={x} lane={lane}: native != runtime library"
                );
                assert_eq!(
                    native[lane].to_bits(),
                    interpreted[lane].to_bits(),
                    "f64 O{level} x={x} lane={lane}: native != interpreter"
                );
                // Two ≤1-ULP implementations (ours and the host libm) differ by at most 2 ULP.
                assert!(
                    ulp_diff64(native[lane], host[lane]) <= 2,
                    "f64 O{level} x={x} lane={lane}: {} vs host {}",
                    native[lane],
                    host[lane]
                );
            }
            if x == 0.0 {
                assert_eq!(
                    native[0].to_bits(),
                    x.to_bits(),
                    "sin keeps the sign of zero"
                );
                assert_eq!(
                    native[2].to_bits(),
                    x.to_bits(),
                    "tan keeps the sign of zero"
                );
            }
        }
    }
}

#[test]
fn interpreter_f32_buffer_transfer_preserves_nan_bits() {
    let bits = [
        0x7fa1_2345,
        0xffa2_3456,
        0x7fc1_2345,
        0xffc2_3456,
        0x0000_0000,
        0x8000_0000,
        0x3f80_0000,
        0xbf80_0000,
        0x7f80_0000,
        0xff80_0000,
        0x0080_0000,
        0x0000_0001,
        0x8000_0001,
    ];
    let (program, mut interner) = lower("fn entry(x: [f32; 13]) {}", false);
    let entry = interner.intern("entry");
    let mut input = bits.map(f32::from_bits);
    wukong_interp::run_kernel_f32(&program, entry, &mut [&mut input], &interner).unwrap();
    assert_eq!(input.map(f32::to_bits), bits, "no-op buffer transfer");

    let (program, mut interner) = lower(
        "fn entry(x: [f32; 4], mut out: [f32; 4]) {
            for i in 0..4 { out[i] = x[i] + 1.0; } }",
        false,
    );
    let entry = interner.intern("entry");
    let mut input = bits[..4]
        .iter()
        .copied()
        .map(f32::from_bits)
        .collect::<Vec<_>>();
    let mut output = [0.0; 4];
    wukong_interp::run_kernel_f32(&program, entry, &mut [&mut input, &mut output], &interner)
        .unwrap();
    for lane in 0..4 {
        assert_eq!(input[lane].to_bits(), bits[lane], "arithmetic input");
        assert_eq!(
            output[lane].to_bits(),
            bits[lane] | 0x0040_0000,
            "quiet result"
        );
    }
}

#[test]
fn rope_forward_backward_use_full_range_trig() {
    let (rows, half) = (3usize, 9usize);
    for frequency in [4096.0, 4e9, -4e9, f32::from_bits(0x6e55_5555)] {
        let frequencies = [frequency; 9];
        let mut input = vec![0.0; rows * 2 * half];
        for row in 0..rows {
            for lane in 0..half {
                input[row * 2 * half + lane] = 1.0;
            }
        }
        for backward in [false, true] {
            let mut output = vec![0.0; input.len()];
            unsafe {
                if backward {
                    wukong_runtime::wukong_rope_bwd_f32(
                        input.as_ptr(),
                        frequencies.as_ptr(),
                        output.as_mut_ptr(),
                        rows as i64,
                        half as i64,
                    );
                } else {
                    wukong_runtime::wukong_rope_f32(
                        input.as_ptr(),
                        frequencies.as_ptr(),
                        output.as_mut_ptr(),
                        rows as i64,
                        half as i64,
                    );
                }
            }
            for row in 0..rows {
                // The runtime multiplies in f32 before reduction; reference that rounded angle.
                let theta = ((row as f32) * frequency) as f64;
                for lane in 0..half {
                    let low = output[row * 2 * half + lane];
                    let high = output[row * 2 * half + lane + half];
                    let expected_high = if backward { -theta.sin() } else { theta.sin() };
                    // Existing rope/rope_bwd f64 accuracy bound.
                    assert!((low as f64 - theta.cos()).abs() < 1e-4);
                    assert!((high as f64 - expected_high).abs() < 1e-4);
                    assert_eq!(
                        low.to_bits(),
                        output[row * 2 * half + half - 1].to_bits(),
                        "vector/tail cosine"
                    );
                    assert_eq!(
                        high.to_bits(),
                        output[row * 2 * half + 2 * half - 1].to_bits(),
                        "vector/tail sine"
                    );
                }
            }
        }
    }
}
