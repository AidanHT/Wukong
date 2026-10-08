//! `f64` transcendentals are double precision on every backend (P15.2): each one lowers to one
//! call of the runtime's pure-Rust `math64` library, the interpreter and native produce the same
//! bits, the LLVM emitter declares the symbol, the composed activations inherit the precision, and
//! the scalar autodiff rules for every code match the closed-form derivatives.
use wukong_mir::{MirType, Op, Program};
use wukong_runtime::{
    VM2_ATAN2, VM2_HYPOT, VM2_POW, VM_ACOS, VM_ACOSH, VM_ASIN, VM_ASINH, VM_ATAN, VM_ATANH,
    VM_CBRT, VM_COS, VM_COSH, VM_ERF, VM_EXP, VM_EXP10, VM_EXP2, VM_EXPM1, VM_LOG, VM_LOG10,
    VM_LOG1P, VM_LOG2, VM_SIN, VM_SINH, VM_TAN, VM_TANH,
};
use wukong_span::{Interner, SourceId};

const LANES: usize = 9;

/// Every single-argument intrinsic that reaches the f64 library, with its source name.
const OPS1: [(i64, &str); 22] = [
    (VM_EXP, "exp"),
    (VM_LOG, "log"),
    (VM_TANH, "tanh"),
    (VM_SIN, "sin"),
    (VM_COS, "cos"),
    (VM_ERF, "erf"),
    (VM_EXP2, "exp2"),
    (VM_LOG2, "log2"),
    (VM_SINH, "sinh"),
    (VM_COSH, "cosh"),
    (VM_ASINH, "asinh"),
    (VM_ACOSH, "acosh"),
    (VM_ATANH, "atanh"),
    (VM_ATAN, "atan"),
    (VM_EXPM1, "expm1"),
    (VM_LOG1P, "log1p"),
    (VM_EXP10, "exp10"),
    (VM_LOG10, "log10"),
    (VM_TAN, "tan"),
    (VM_ASIN, "asin"),
    (VM_ACOS, "acos"),
    (VM_CBRT, "cbrt"),
];

const OPS2: [(i64, &str); 3] = [(VM2_POW, "pow"), (VM2_ATAN2, "atan2"), (VM2_HYPOT, "hypot")];

fn lower(source: &str) -> (Program, Interner) {
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
            recognize: true,
            vec_kernels: false,
            ..Default::default()
        },
    );
    assert!(diagnostics.is_empty(), "lower: {diagnostics:?}");
    (program, interner)
}

fn constant(function: &wukong_mir::Function, value: wukong_mir::ValueId) -> Option<i128> {
    function
        .blocks
        .iter()
        .flat_map(|block| &block.insts)
        .find_map(|inst| match (inst.result, &inst.op) {
            (Some(r), Op::ConstInt(v, _)) if r == value => Some(*v),
            _ => None,
        })
}

/// The `(symbol, op code)` of every f64 library call in `program`, in instruction order, and a
/// check that nothing was demoted: no `fptrunc`, no f32 runtime call.
fn library_calls(program: &Program, interner: &Interner) -> Vec<(String, i128)> {
    let mut calls = Vec::new();
    for function in &program.funcs {
        for inst in function.blocks.iter().flat_map(|block| &block.insts) {
            match &inst.op {
                Op::Call { func, args } => {
                    let name = interner.resolve(*func);
                    match name {
                        "wukong_rt_math_f64" => {
                            assert_eq!(args.len(), 2);
                            assert_eq!(function.value_type(args[0]), &MirType::F64);
                            assert_eq!(function.value_type(inst.result.unwrap()), &MirType::F64);
                            calls.push((name.to_string(), constant(function, args[1]).unwrap()));
                        }
                        "wukong_rt_math2_f64" => {
                            assert_eq!(args.len(), 3);
                            assert_eq!(function.value_type(args[0]), &MirType::F64);
                            assert_eq!(function.value_type(args[1]), &MirType::F64);
                            assert_eq!(function.value_type(inst.result.unwrap()), &MirType::F64);
                            calls.push((name.to_string(), constant(function, args[2]).unwrap()));
                        }
                        other if other.starts_with("wukong_") => {
                            panic!("f64 math reached an f32 runtime kernel: {other}")
                        }
                        _ => {}
                    }
                }
                Op::Cast(wukong_mir::CastKind::FpTrunc, _, _) => {
                    panic!("f64 math demoted a value to f32")
                }
                _ => {}
            }
        }
    }
    calls
}

fn ulp_diff(a: f64, b: f64) -> u64 {
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

/// Domain-appropriate arguments for each function: the general grid plus its own edge cases.
fn samples(op: i64) -> Vec<f64> {
    let general = [
        0.0, -0.0, 1e-300, -1e-300, 1e-9, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0, 1.5, 2.0, 3.0, 10.0,
        20.5, 100.0, 700.0, -0.1, -0.5, -1.0, -3.0, -20.5, -700.0,
    ];
    let mut values: Vec<f64> = general.to_vec();
    match op {
        VM_SIN | VM_COS | VM_TAN => values.extend([
            1e6,
            1e22,
            1e300,
            (1u64 << 60) as f64,
            std::f64::consts::PI,
            std::f64::consts::FRAC_PI_2,
            -12345.678,
        ]),
        VM_LOG | VM_LOG2 | VM_LOG10 => {
            values.retain(|v| *v > 0.0);
            values.extend([5e-324, 2.2250738585072014e-308, 1e300, f64::MAX]);
        }
        VM_ASIN | VM_ACOS | VM_ATANH => {
            values.retain(|v| v.abs() < 1.0);
            values.extend([0.999999, -0.999999]);
        }
        VM_ACOSH => {
            values.retain(|v| *v >= 1.0);
            values.extend([1.0001, 1.1, 1.25, 4.0, 7.5, 50.0, 1e10, 1e300]);
        }
        VM_LOG1P => {
            values.retain(|v| *v > -1.0);
            values.extend([-0.9999, 1e-20, 1e300]);
        }
        VM_EXP | VM_EXPM1 => values.extend([709.7, -745.0, 1e-20]),
        VM_EXP2 => values.extend([1023.9, -1074.0, -1074.5]),
        VM_EXP10 => values.extend([15.0, -15.0, 307.5, -307.5]),
        VM_CBRT => values.extend([27.0, -64.0, 1e300, 5e-324]),
        VM_ATAN | VM_ASINH | VM_SINH | VM_COSH | VM_TANH | VM_ERF => {
            values.extend([1e10, -1e10, 1e300, 0.027, -0.027, 0.2554, 0.5493]);
        }
        _ => {}
    }
    values.extend([f64::INFINITY, f64::NEG_INFINITY, f64::NAN]);
    values
}

/// Host libm reference for a sanity bound (the ULP contract itself is pinned against mpmath in
/// `wukong_runtime`). `None` where std has no counterpart.
fn host1(op: i64, x: f64) -> Option<f64> {
    Some(match op {
        VM_EXP => x.exp(),
        VM_LOG => x.ln(),
        VM_TANH => x.tanh(),
        VM_SIN => x.sin(),
        VM_COS => x.cos(),
        VM_EXP2 => x.exp2(),
        VM_LOG2 => x.log2(),
        VM_SINH => x.sinh(),
        VM_COSH => x.cosh(),
        VM_ASINH => x.asinh(),
        VM_ACOSH => x.acosh(),
        VM_ATANH => x.atanh(),
        VM_ATAN => x.atan(),
        VM_EXPM1 => x.exp_m1(),
        VM_LOG1P => x.ln_1p(),
        VM_LOG10 => x.log10(),
        VM_TAN => x.tan(),
        VM_ASIN => x.asin(),
        VM_ACOS => x.acos(),
        VM_CBRT => x.cbrt(),
        _ => return None,
    })
}

#[test]
fn f64_transcendentals_match_runtime_on_both_backends() {
    for (op, name) in OPS1 {
        let source = format!(
            "fn entry(x: [f64; {LANES}], mut out: [f64; {LANES}]) {{
                for i in 0..{LANES} {{ out[i] = {name}(x[i]); }} }}"
        );
        let (program, mut interner) = lower(&source);
        let calls = library_calls(&program, &interner);
        assert_eq!(
            calls,
            vec![("wukong_rt_math_f64".to_string(), op as i128)],
            "{name}: exactly one f64 library call with its own code"
        );
        let entry = interner.intern("entry");
        let samples = samples(op);
        for level in [0, 2] {
            let mut optimized = program.clone();
            wukong_opt::optimize_verified(&mut optimized, level).unwrap();
            let kept = library_calls(&optimized, &interner);
            assert!(
                !kept.is_empty() && kept.iter().all(|c| c.1 == op as i128),
                "{name} O{level}: the optimizer must keep the call(s): {kept:?}"
            );
            let module = wukong_codegen_cranelift::jit_module(&optimized, &interner).unwrap();
            let function: unsafe extern "C" fn(*const f64, *mut f64) =
                unsafe { std::mem::transmute(module.func_ptr(entry).unwrap()) };
            for chunk in samples.chunks(LANES) {
                let mut input = [0.0f64; LANES];
                input[..chunk.len()].copy_from_slice(chunk);
                let mut ix = input;
                let mut interpreted = [1234.0f64; LANES];
                let mut native = interpreted;
                wukong_interp::run_kernel_f64(
                    &optimized,
                    entry,
                    &mut [&mut ix, &mut interpreted],
                    &interner,
                )
                .unwrap();
                unsafe { function(input.as_ptr(), native.as_mut_ptr()) };
                for lane in 0..LANES {
                    let x = input[lane];
                    let expected = wukong_runtime::wukong_rt_math_f64(x, op);
                    assert_eq!(
                        native[lane].to_bits(),
                        expected.to_bits(),
                        "{name} O{level} x={x:e}: native {:e} vs library {expected:e}",
                        native[lane]
                    );
                    assert_eq!(
                        interpreted[lane].to_bits(),
                        expected.to_bits(),
                        "{name} O{level} x={x:e}: interpreter {:e} vs library {expected:e}",
                        interpreted[lane]
                    );
                    // The host libm's own large-argument trig reduction is not under test here.
                    let host_trig_out_of_scope =
                        matches!(op, VM_SIN | VM_COS | VM_TAN) && x.abs() > 1e6;
                    if let Some(host) = host1(op, x).filter(|_| !host_trig_out_of_scope) {
                        if host.is_finite() && expected.is_finite() {
                            // Two implementations within the documented bounds (ours ≤ 2 ULP
                            // on the hyperbolics, the host's own ≤ 1–2) stay within 4 ULP.
                            assert!(
                                ulp_diff(native[lane], host) <= 4,
                                "{name} x={x:e}: {:e} vs host {host:e}",
                                native[lane]
                            );
                        } else {
                            assert_eq!(
                                native[lane].is_nan(),
                                host.is_nan(),
                                "{name} x={x:e}: NaN-ness differs from the host"
                            );
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn f64_two_argument_transcendentals_match_runtime_on_both_backends() {
    let pairs: [(f64, f64); 14] = [
        (2.0, 0.5),
        (2.0, 10.0),
        (10.0, -3.0),
        (1.5, 2.5),
        (0.1, 0.1),
        (-2.0, 3.0),
        (-2.0, 4.0),
        (-2.0, 0.5),
        (0.0, -1.0),
        (-0.0, 3.0),
        (1.0, f64::NAN),
        (f64::NAN, 0.0),
        (f64::INFINITY, -1.0),
        (-1.0, f64::INFINITY),
    ];
    for (op, name) in OPS2 {
        let source = format!(
            "fn entry(a: [f64; {LANES}], b: [f64; {LANES}], mut out: [f64; {LANES}]) {{
                for i in 0..{LANES} {{ out[i] = {name}(a[i], b[i]); }} }}"
        );
        let (program, mut interner) = lower(&source);
        assert_eq!(
            library_calls(&program, &interner),
            vec![("wukong_rt_math2_f64".to_string(), op as i128)],
            "{name}: exactly one two-argument f64 library call"
        );
        let entry = interner.intern("entry");
        for level in [0, 2] {
            let mut optimized = program.clone();
            wukong_opt::optimize_verified(&mut optimized, level).unwrap();
            let module = wukong_codegen_cranelift::jit_module(&optimized, &interner).unwrap();
            let function: unsafe extern "C" fn(*const f64, *const f64, *mut f64) =
                unsafe { std::mem::transmute(module.func_ptr(entry).unwrap()) };
            for chunk in pairs.chunks(LANES) {
                let mut a = [1.0f64; LANES];
                let mut b = [1.0f64; LANES];
                for (lane, &(x, y)) in chunk.iter().enumerate() {
                    a[lane] = x;
                    b[lane] = y;
                }
                let (mut ia, mut ib) = (a, b);
                let mut interpreted = [1234.0f64; LANES];
                let mut native = interpreted;
                wukong_interp::run_kernel_f64(
                    &optimized,
                    entry,
                    &mut [&mut ia, &mut ib, &mut interpreted],
                    &interner,
                )
                .unwrap();
                unsafe { function(a.as_ptr(), b.as_ptr(), native.as_mut_ptr()) };
                for lane in 0..LANES {
                    let expected = wukong_runtime::wukong_rt_math2_f64(a[lane], b[lane], op);
                    assert_eq!(
                        native[lane].to_bits(),
                        expected.to_bits(),
                        "{name} O{level} ({:e}, {:e}): native {:e} vs library {expected:e}",
                        a[lane],
                        b[lane],
                        native[lane]
                    );
                    assert_eq!(
                        interpreted[lane].to_bits(),
                        expected.to_bits(),
                        "{name} O{level} ({:e}, {:e}): interpreter {:e} vs library {expected:e}",
                        a[lane],
                        b[lane],
                        interpreted[lane]
                    );
                    let host = match op {
                        VM2_POW => a[lane].powf(b[lane]),
                        VM2_ATAN2 => a[lane].atan2(b[lane]),
                        _ => a[lane].hypot(b[lane]),
                    };
                    if host.is_finite() && expected.is_finite() {
                        assert!(
                            ulp_diff(native[lane], host) <= 2,
                            "{name} ({:e}, {:e}): {:e} vs host {host:e}",
                            a[lane],
                            b[lane],
                            native[lane]
                        );
                    } else if host.is_nan() {
                        assert!(native[lane].is_nan(), "{name} special value must be NaN");
                    } else {
                        assert_eq!(
                            native[lane].to_bits(),
                            host.to_bits(),
                            "{name} special value"
                        );
                    }
                }
            }
        }
    }
}

/// The activations compose the library calls in MIR, so an `f64` `sigmoid`/`gelu`/`softplus`/…
/// is double precision too: both backends agree bit-for-bit, and the value is within a few ULP of
/// the host-libm composition (the old f32 demotion was ~2^27 ULP away).
/// A named host-libm reference composition for one `f64` activation.
type HostReference = (&'static str, fn(f64) -> f64);

#[test]
fn f64_composed_activations_are_double_precision() {
    let reference: [HostReference; 8] = [
        ("sigmoid", |x| 1.0 / (1.0 + (-x).exp())),
        ("silu", |x| x / (1.0 + (-x).exp())),
        ("softplus", |x| x.max(0.0) + (1.0 + (-x.abs()).exp()).ln()),
        ("logsigmoid", |x| {
            -((-x).max(0.0) + (1.0 + (-x.abs()).exp()).ln())
        }),
        ("elu", |x| if x > 0.0 { x } else { x.exp() - 1.0 }),
        ("tanhshrink", |x| x - x.tanh()),
        ("mish", |x| {
            x * (x.max(0.0) + (1.0 + (-x.abs()).exp()).ln()).tanh()
        }),
        ("gelu", |x| {
            0.5 * x * (1.0 + (0.7978845608 * (x + 0.044715 * x * x * x)).tanh())
        }),
    ];
    let inputs = [-3.0f64, -1.0, -0.5, -0.1, 0.0, 0.1, 0.5, 1.0, 2.5];
    for (name, host) in reference {
        let source = format!(
            "fn entry(x: [f64; {LANES}], mut out: [f64; {LANES}]) {{
                for i in 0..{LANES} {{ out[i] = {name}(x[i]); }} }}"
        );
        let (program, mut interner) = lower(&source);
        assert!(
            !library_calls(&program, &interner).is_empty(),
            "{name}: must be composed from the f64 library"
        );
        let entry = interner.intern("entry");
        for level in [0, 2] {
            let mut optimized = program.clone();
            wukong_opt::optimize_verified(&mut optimized, level).unwrap();
            let module = wukong_codegen_cranelift::jit_module(&optimized, &interner).unwrap();
            let function: unsafe extern "C" fn(*const f64, *mut f64) =
                unsafe { std::mem::transmute(module.func_ptr(entry).unwrap()) };
            let mut ix = inputs;
            let mut interpreted = [0.0f64; LANES];
            let mut native = interpreted;
            wukong_interp::run_kernel_f64(
                &optimized,
                entry,
                &mut [&mut ix, &mut interpreted],
                &interner,
            )
            .unwrap();
            unsafe { function(inputs.as_ptr(), native.as_mut_ptr()) };
            for lane in 0..LANES {
                let x = inputs[lane];
                assert_eq!(
                    native[lane].to_bits(),
                    interpreted[lane].to_bits(),
                    "{name} O{level} x={x}: native vs interpreter"
                );
                let expected = host(x);
                // Relative to the larger of the result and the argument: `x − tanh(x)` and the
                // like cancel near 0, where a 1-ULP difference in the library's tanh is a large
                // relative difference in a tiny result but a tiny absolute one.
                let tolerance = 8.0 * f64::EPSILON * expected.abs().max(x.abs()).max(1.0);
                assert!(
                    (native[lane] - expected).abs() <= tolerance,
                    "{name} O{level} x={x}: {:e} vs host composition {expected:e}",
                    native[lane]
                );
            }
        }
    }
}

#[test]
fn f64_transcendental_llvm_and_object_imports_are_typed() {
    let (program, interner) = lower(
        "fn entry(x: [f64; 1], mut out: [f64; 4]) {
            out[0] = exp(x[0]); out[1] = erf(x[0]); out[2] = pow(x[0], x[0]); out[3] = cbrt(x[0]); }",
    );
    let llvm = wukong_codegen_llvm::try_emit_llvm_ir(&program, &interner).unwrap();
    assert!(
        llvm.contains("declare double @wukong_rt_math_f64(double, i64)"),
        "{llvm}"
    );
    assert!(
        llvm.contains("declare double @wukong_rt_math2_f64(double, double, i64)"),
        "{llvm}"
    );
    assert!(!llvm.contains("fptrunc"), "{llvm}");
    let object = wukong_codegen_cranelift::emit_object(&program, &interner).unwrap();
    assert!(!object.is_empty());
}

/// Closed-form derivative of each single-argument code, from the host libm.
fn host_derivative(op: i64, x: f64) -> f64 {
    match op {
        VM_EXP => x.exp(),
        VM_LOG => 1.0 / x,
        VM_TANH => 1.0 - x.tanh() * x.tanh(),
        VM_SIN => x.cos(),
        VM_COS => -x.sin(),
        VM_ERF => std::f64::consts::FRAC_2_SQRT_PI * (-x * x).exp(),
        VM_EXP2 => x.exp2() * std::f64::consts::LN_2,
        VM_LOG2 => 1.0 / (x * std::f64::consts::LN_2),
        VM_SINH => x.cosh(),
        VM_COSH => x.sinh(),
        VM_ASINH => 1.0 / (x * x + 1.0).sqrt(),
        VM_ACOSH => 1.0 / (x * x - 1.0).sqrt(),
        VM_ATANH => 1.0 / (1.0 - x * x),
        VM_ATAN => 1.0 / (1.0 + x * x),
        VM_EXPM1 => x.exp(),
        VM_LOG1P => 1.0 / (1.0 + x),
        VM_EXP10 => 10f64.powf(x) * std::f64::consts::LN_10,
        VM_LOG10 => 1.0 / (x * std::f64::consts::LN_10),
        VM_TAN => 1.0 + x.tan() * x.tan(),
        VM_ASIN => 1.0 / (1.0 - x * x).sqrt(),
        VM_ACOS => -1.0 / (1.0 - x * x).sqrt(),
        VM_CBRT => 1.0 / (3.0 * x.cbrt() * x.cbrt()),
        _ => unreachable!(),
    }
}

/// A standalone `f64` transcendental stays a single-block scalar autodiff candidate, and its
/// VJP matches the closed-form derivative on both backends.
#[test]
fn f64_scalar_transcendental_ad_matches_closed_forms() {
    for (op, name) in OPS1 {
        let source = format!("fn loss(x: [f64; 1]) -> f64 {{ return {name}(x[0]); }}");
        let (program, mut interner) = lower(&source);
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
            let function: unsafe extern "C" fn(*const f64, *mut f64) =
                unsafe { std::mem::transmute(module.func_ptr(entry).unwrap()) };
            // Interior points only: the derivative is a closed form there, and a pole or a
            // domain edge would compare infinities. Normal arguments only, for the same reason:
            // at the subnormal 5e-324 the log family's `1/x` is 2^1074, which overflows to +∞
            // on both sides and `∞ − ∞` fails any tolerance.
            let xs: Vec<f64> = samples(op)
                .into_iter()
                .filter(|x| x.is_normal() && x.abs() < 100.0)
                .filter(|x| match op {
                    VM_ASIN | VM_ACOS | VM_ATANH => x.abs() < 0.99,
                    VM_ACOSH => *x > 1.01,
                    VM_LOG1P => *x > -0.99,
                    VM_TAN => (x.cos()).abs() > 0.1,
                    _ => true,
                })
                .collect();
            assert!(xs.len() >= 8, "{name}: too few interior samples");
            for x in xs {
                let mut ix = [x];
                let mut interpreted = [0.0];
                let mut native = [0.0];
                wukong_interp::run_kernel_f64(
                    &optimized,
                    entry,
                    &mut [&mut ix, &mut interpreted],
                    &interner,
                )
                .unwrap();
                unsafe { function(&x, native.as_mut_ptr()) };
                let expected = host_derivative(op, x);
                assert_eq!(
                    native[0].to_bits(),
                    interpreted[0].to_bits(),
                    "{name}'({x}) native vs interpreter"
                );
                assert!(
                    (native[0] - expected).abs() <= 1e-12 * expected.abs().max(1e-300) + 1e-300,
                    "{name}'({x}) = {} vs closed form {expected}",
                    native[0]
                );
            }
        }
    }
    // The two-argument codes: ∂pow/∂a, ∂pow/∂b, ∂atan2/∂y, ∂atan2/∂x, ∂hypot/∂a, ∂hypot/∂b.
    for (op, name) in OPS2 {
        let source =
            format!("fn loss(a: [f64; 1], b: [f64; 1]) -> f64 {{ return {name}(a[0], b[0]); }}");
        let (program, mut interner) = lower(&source);
        let loss = interner.intern("loss");
        let mut optimized = program.clone();
        wukong_opt::optimize_verified(&mut optimized, 2).unwrap();
        let forward = optimized.funcs.iter().find(|f| f.name == loss).unwrap();
        let gradient = wukong_autodiff::grad(forward, &[0, 1], &mut interner).unwrap();
        let entry = gradient.name;
        optimized.funcs.push(gradient);
        assert!(wukong_mir::verify::verify_program(&optimized).is_empty());
        let module = wukong_codegen_cranelift::jit_module(&optimized, &interner).unwrap();
        let function: unsafe extern "C" fn(*const f64, *const f64, *mut f64, *mut f64) =
            unsafe { std::mem::transmute(module.func_ptr(entry).unwrap()) };
        for (a, b) in [
            (2.0f64, 0.5),
            (1.5, 2.5),
            (0.3, 3.0),
            (3.0, -0.75),
            (-2.0, 3.0),
        ] {
            if op == VM2_POW && a < 0.0 {
                continue; // ∂/∂b needs ln(a)
            }
            let (mut ia, mut ib) = ([a], [b]);
            let mut interp_a = [0.0];
            let mut interp_b = [0.0];
            let mut native_a = [0.0];
            let mut native_b = [0.0];
            wukong_interp::run_kernel_f64(
                &optimized,
                entry,
                &mut [&mut ia, &mut ib, &mut interp_a, &mut interp_b],
                &interner,
            )
            .unwrap();
            unsafe { function(&a, &b, native_a.as_mut_ptr(), native_b.as_mut_ptr()) };
            let (da, db) = match op {
                VM2_POW => (b * a.powf(b - 1.0), a.ln() * a.powf(b)),
                VM2_ATAN2 => (b / (a * a + b * b), -a / (a * a + b * b)),
                _ => (a / a.hypot(b), b / a.hypot(b)),
            };
            assert_eq!(
                native_a[0].to_bits(),
                interp_a[0].to_bits(),
                "{name} ∂a backends"
            );
            assert_eq!(
                native_b[0].to_bits(),
                interp_b[0].to_bits(),
                "{name} ∂b backends"
            );
            assert!(
                (native_a[0] - da).abs() <= 1e-12 * da.abs(),
                "{name} ∂/∂a({a}, {b}) = {} vs {da}",
                native_a[0]
            );
            assert!(
                (native_b[0] - db).abs() <= 1e-12 * db.abs(),
                "{name} ∂/∂b({a}, {b}) = {} vs {db}",
                native_b[0]
            );
        }
    }
}
