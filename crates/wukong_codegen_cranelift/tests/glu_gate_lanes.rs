//! Source-level GLU extent and argument-marshalling checks. The runtime's `vmath2_gate`
//! test separately covers varied sigmoid inputs against its scalar twin and f64 reference.
//! Here sigmoid(0) = 1/2 gives exact independent expectations for every lane.

use wukong_mir::{Op, Program};
use wukong_span::{Interner, SourceId};

const GUARD: usize = 4;

fn lower(n: usize, reversed: bool, recognize: bool) -> (Program, Interner) {
    let capacity = n + 2 * GUARD;
    let expression = if reversed {
        "sigmoid(g[j]) * x[j]"
    } else {
        "x[j] * sigmoid(g[j])"
    };
    let source = format!(
        "module test.glu_gate_lanes
         fn entry(x: [f32; {capacity}], g: [f32; {capacity}], mut out: [f32; {capacity}]) {{
             for j in {GUARD}..{} {{ out[j] = {expression}; }}
         }}",
        GUARD + n
    );
    let mut interner = Interner::new();
    let (ast, diagnostics) = wukong_parser::parse_module(&source, SourceId(0), &mut interner);
    assert!(diagnostics.is_empty(), "parse: {diagnostics:?}\n{source}");
    let (sema, diagnostics) = wukong_sema::check(&ast, &interner);
    assert!(diagnostics.is_empty(), "sema: {diagnostics:?}\n{source}");
    let (program, diagnostics) = wukong_mir_build::lower_program_with(
        &ast,
        &sema,
        &mut interner,
        wukong_mir_build::LowerOptions {
            recognize,
            vec_kernels: false,
        },
    );
    assert!(diagnostics.is_empty(), "lower: {diagnostics:?}\n{source}");
    (program, interner)
}

fn assert_dispatch(program: &Program, interner: &Interner, recognize: bool, label: &str) {
    let mut calls = 0;
    for function in &program.funcs {
        for inst in function.blocks.iter().flat_map(|block| &block.insts) {
            let Op::Call { func, args } = &inst.op else {
                continue;
            };
            let name = interner.resolve(*func);
            if !name.starts_with("wukong_vmath2_") {
                continue;
            }
            calls += 1;
            assert_eq!(name, "wukong_vmath2_f32", "{label}");
            assert_eq!(args.len(), 5, "{label}");
            let opcode = function
                .blocks
                .iter()
                .flat_map(|block| &block.insts)
                .find_map(|definition| {
                    if definition.result == Some(args[4]) {
                        if let Op::ConstInt(value, _) = definition.op {
                            return Some(value);
                        }
                    }
                    None
                });
            // Pin the public operation number independently: another vmath2 operation
            // could otherwise masquerade as correct dispatch through the shared symbol.
            assert_eq!(opcode, Some(11), "{label}: expected VM2_SIGMOID_GATE");
        }
    }
    assert_eq!(calls, usize::from(recognize), "{label}: dispatch count");
}

fn check(reversed: bool) {
    for n in [7, 8, 9, 257] {
        let capacity = n + 2 * GUARD;
        let mut input: Vec<f32> = (0..capacity)
            .map(|lane| (lane as i32 * 5 - 41) as f32 / 8.0)
            .collect();
        input[GUARD] = 5.0; // Preserve the original fixture's exact 5 * sigmoid(0) case.
                            // Nonzero guard gates expose a gate-only pointer-offset error; the active
                            // range still has the exact sigmoid(0) = 1/2 oracle.
        let mut gate = vec![3.0f32; capacity];
        gate[GUARD..GUARD + n].fill(0.0);
        let initial: Vec<f32> = (0..capacity).map(|lane| (4096 + lane) as f32).collect();
        let mut expected = initial.clone();
        for lane in GUARD..GUARD + n {
            expected[lane] = input[lane] / 2.0;
        }
        for recognize in [false, true] {
            let label = format!("n={n} reversed={reversed} recognize={recognize}");
            let (program, mut interner) = lower(n, reversed, recognize);
            assert_dispatch(&program, &interner, recognize, &label);
            let entry = interner.intern("entry");
            for level in [0, 2] {
                let label = format!("{label} O{level}");
                let mut optimized = program.clone();
                wukong_opt::optimize_verified(&mut optimized, level).unwrap();
                assert_dispatch(&optimized, &interner, recognize, &label);

                let mut ix = input.clone();
                let mut ig = gate.clone();
                let mut interpreted = initial.clone();
                wukong_interp::run_kernel_f32(
                    &optimized,
                    entry,
                    &mut [&mut ix, &mut ig, &mut interpreted],
                    &interner,
                )
                .unwrap();

                let module = wukong_codegen_cranelift::jit_module(&optimized, &interner).unwrap();
                // entry takes three fixed f32 arrays by pointer and returns void. The
                // buffers are distinct, contain capacity lanes, and out is writable.
                let function: unsafe extern "C" fn(*const f32, *const f32, *mut f32) = unsafe {
                    std::mem::transmute(module.func_ptr(entry).expect("entry remains present"))
                };
                let nx = input.clone();
                let ng = gate.clone();
                let mut native = initial.clone();
                unsafe { function(nx.as_ptr(), ng.as_ptr(), native.as_mut_ptr()) };
                for (backend, actual, x, g) in [
                    ("interpreter", interpreted, ix, ig),
                    ("native", native, nx, ng),
                ] {
                    for (lane, (actual, expected)) in actual.iter().zip(&expected).enumerate() {
                        assert_eq!(
                            actual.to_bits(),
                            expected.to_bits(),
                            "{label} {backend} lane {lane}: {actual} != {expected}"
                        );
                        assert_eq!(
                            x[lane].to_bits(),
                            input[lane].to_bits(),
                            "{label} x[{lane}]"
                        );
                        assert_eq!(g[lane].to_bits(), gate[lane].to_bits(), "{label} g[{lane}]");
                    }
                }
            }
        }
    }
}

#[test]
fn glu_gate_preserves_all_lanes_and_guards() {
    check(false);
}

#[test]
fn glu_gate_reversed_factors_preserve_all_lanes_and_guards() {
    check(true);
}
