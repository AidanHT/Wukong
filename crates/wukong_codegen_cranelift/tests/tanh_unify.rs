//! One `tanh` (P13.2): the dispatched `VM_TANH` kernel, the inlined scalar form, the inlined
//! 128-bit SIMD form and the `@generic`/no-recognition lowering are the same function
//! bit-for-bit, on both backends, at `-O0` and `-O2` — including −0, |x| < 2^-25, the crossover
//! at `TANH_XC`, the saturated band, ±∞ and NaN.
use wukong_mir::{MirType, Op, Program};
use wukong_runtime::VM_TANH;
use wukong_span::{Interner, SourceId};

const LANES: usize = 17;

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

/// The runtime kernel (`tanh1`/`tanh8`), the reference every spelling must equal.
fn kernel(x: f32) -> f32 {
    let mut out = 0.0;
    unsafe { wukong_runtime::wukong_vmath_f32(&x, &mut out, 1, VM_TANH) };
    out
}

/// Arguments that exercise every branch and every boundary of the kernel. Pairs ±x so the sign
/// handling is covered on both sides.
fn samples() -> Vec<f32> {
    let mut values = vec![0.0f32, 1.0, 2.0, 3.0];
    // Signed zero and the band below 2^-25 where the old composed form returned +0.0.
    for bits in [
        0x0000_0001u32, // smallest subnormal
        0x0080_0000,    // smallest normal
        0x3300_0000,    // 2^-25
        0x32ff_ffff,    // just below 2^-25
        0x3980_0000,    // 2^-12 (the fma returns x below 2^-13)
        0x3c00_0000,    // 2^-7
    ] {
        values.push(f32::from_bits(bits));
    }
    // The 80-ULP region of the old form, the crossover (0.75 is polynomial, the next float is exp),
    // the saturation point (tanh rounds to exactly 1 past ~9.0109), the clamp at 10, and beyond.
    for v in [
        0.027f32,
        0.1,
        0.25,
        0.5,
        0.7,
        0.75,
        0.8,
        0.9,
        1.5,
        4.0,
        8.0,
        9.0,
        9.0109,
        9.5,
        10.0,
        10.5,
        20.0,
        88.0,
        1e10,
        f32::MAX,
    ] {
        values.push(v);
        for delta in [-1i32, 1] {
            values.push(f32::from_bits((v.to_bits() as i32 + delta) as u32));
        }
    }
    // A fine grid across the crossover window and the saturated approach.
    for k in 0..=400 {
        values.push(0.5 + 0.001 * k as f32);
    }
    for k in 0..=100 {
        values.push(1.0 + 0.1 * k as f32);
    }
    let negatives: Vec<f32> = values.iter().map(|v| -v).collect();
    values.extend(negatives);
    values.extend([
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::from_bits(0x7fc1_2345),
        f32::from_bits(0xffc2_3456),
        f32::from_bits(0x7fa1_2345),
        f32::from_bits(0xffff_ffc0),
    ]);
    values
}

/// Count the kernel dispatches and the inlined tanh crossover compares in a program.
fn shape(program: &Program, interner: &Interner) -> (usize, usize) {
    let mut kernel_calls = 0;
    let mut vector_fmas = 0;
    for function in &program.funcs {
        for inst in function.blocks.iter().flat_map(|block| &block.insts) {
            match &inst.op {
                Op::Call { func, args } if interner.resolve(*func) == "wukong_vmath_f32" => {
                    assert_eq!(args.len(), 4);
                    kernel_calls += 1;
                }
                Op::Call { func, .. } if interner.resolve(*func) == "wukong_rt_trig_f32" => {
                    panic!("tanh must not route through the scalar trig runtime");
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
    (kernel_calls, vector_fmas)
}

#[test]
fn dispatched_tanh_equals_composed_tanh() {
    // Three spellings of the same elementwise map: the recognizer's dispatched loop (kernel),
    // the same loop with recognition off (`@generic`, the inlined form), and a composed
    // expression the recognizer declines but the vectorizer lowers as 128-bit inlined MIR.
    let dispatched = format!(
        "fn entry(x: [f32; {LANES}], mut out: [f32; {LANES}]) {{
            for i in 0..{LANES} {{ out[i] = tanh(x[i]); }} }}"
    );
    let composed = format!(
        "fn entry(x: [f32; {LANES}], mut out: [f32; {LANES}]) {{
            let s: f32 = 1.0;
            for i in 0..{LANES} {{ out[i] = tanh(x[i]) * s; }} }}"
    );
    let variants: [(&str, &str, bool); 3] = [
        ("dispatched", &dispatched, true),
        ("generic", &dispatched, false),
        ("composed", &composed, true),
    ];
    let samples = samples();
    for (label, source, recognize) in variants {
        let (program, mut interner) = lower(source, recognize);
        let (kernel_calls, vector_fmas) = shape(&program, &interner);
        if label == "dispatched" {
            assert_eq!(kernel_calls, 1, "{label}: the loop must dispatch VM_TANH");
        } else {
            assert_eq!(kernel_calls, 0, "{label}: no kernel dispatch");
            assert!(
                vector_fmas > 0,
                "{label}: the inlined kernel must vectorize"
            );
        }
        let entry = interner.intern("entry");
        for level in [0, 2] {
            let mut optimized = program.clone();
            wukong_opt::optimize_verified(&mut optimized, level).unwrap();
            let module = wukong_codegen_cranelift::jit_module(&optimized, &interner).unwrap();
            let function: unsafe extern "C" fn(*const f32, *mut f32) =
                unsafe { std::mem::transmute(module.func_ptr(entry).unwrap()) };
            // Pack the samples 17 at a time so vector blocks and the scalar tail both see every
            // value; pad the last block with zeros.
            for chunk in samples.chunks(LANES) {
                let mut input = [0.0f32; LANES];
                input[..chunk.len()].copy_from_slice(chunk);
                let mut ix = input;
                let mut interpreted = [1234.0f32; LANES];
                let mut native = interpreted;
                wukong_interp::run_kernel_f32(
                    &optimized,
                    entry,
                    &mut [&mut ix, &mut interpreted],
                    &interner,
                )
                .unwrap();
                unsafe { function(input.as_ptr(), native.as_mut_ptr()) };
                for lane in 0..LANES {
                    let x = input[lane];
                    let expected = kernel(x);
                    assert_eq!(
                        native[lane].to_bits(),
                        expected.to_bits(),
                        "{label} O{level} lane {lane} x={x:?} ({:#010x}): native {:?} vs kernel {expected:?}",
                        x.to_bits(),
                        native[lane]
                    );
                    assert_eq!(
                        interpreted[lane].to_bits(),
                        expected.to_bits(),
                        "{label} O{level} lane {lane} x={x:?}: interpreter {:?} vs kernel {expected:?}",
                        interpreted[lane]
                    );
                }
            }
        }
    }
}

/// The kernel's own contract on the values the old composed form got wrong: `tanh(x) == x`
/// (sign included) below 2^-13, −0 stays −0, the result saturates to exactly ±1, and a NaN
/// keeps its sign and payload.
#[test]
fn tanh_kernel_contract_at_the_edges() {
    for x in [0.0f32, -0.0, 1e-30, -1e-30, 1e-8, -1e-8, 1e-5, -1e-5] {
        assert_eq!(kernel(x).to_bits(), x.to_bits(), "{x:?}");
    }
    for x in [9.5f32, 10.0, 20.0, 1e10, f32::INFINITY] {
        assert_eq!(kernel(x), 1.0, "{x:?}");
        assert_eq!(kernel(-x), -1.0, "{x:?}");
    }
    assert_eq!(kernel(f32::from_bits(0x7fa1_2345)).to_bits(), 0x7fe1_2345);
    assert_eq!(kernel(f32::from_bits(0xffc2_3456)).to_bits(), 0xffc2_3456);
    // Composed activations keep their own (exp-composed) tanh building block so they stay equal
    // to their kernels; the public `tanh` is the piecewise one. Both are tanh to f32 accuracy.
    let host = 0.5f64.tanh() as f32;
    assert!((kernel(0.5) - host).abs() <= 2.0 * f32::EPSILON * host);
}
