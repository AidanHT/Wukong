use std::fmt::Write;
use wukong_mir::{Op, Program};
use wukong_span::{Interner, SourceId};

const START: usize = 3;
const CAP: usize = 2064;
const SIZES: &[usize] = &[
    0, 1, 7, 8, 9, 15, 16, 17, 23, 24, 25, 31, 32, 33, 39, 40, 41, 2047, 2048, 2049, 2055, 2056,
    2057,
];

struct RestoreEnv(Option<std::ffi::OsString>);
impl Drop for RestoreEnv {
    fn drop(&mut self) {
        if let Some(value) = &self.0 {
            std::env::set_var("WUKONG_P4_NO_256", value);
        } else {
            std::env::remove_var("WUKONG_P4_NO_256");
        }
    }
}

fn source() -> String {
    let mut source = String::new();
    for op in ["sum", "dot", "max", "min", "pressure"] {
        for bound in SIZES
            .iter()
            .map(|n| n.to_string())
            .chain(std::iter::once("dynamic".into()))
        {
            let end = if bound == "dynamic" {
                "end".to_string()
            } else {
                (START + bound.parse::<usize>().unwrap()).to_string()
            };
            let prelude = if bound == "dynamic" {
                format!("let end: i64 = (out[1] as i64) + {START};")
            } else {
                String::new()
            };
            let rhs = match op {
                "sum" => "s + a[i]",
                "dot" => "s + a[i] * b[i]",
                "max" => "fmax(s, a[i])",
                "min" => "fmin(s, a[i])",
                // Four distinct hoisted constants and several simultaneously live operands
                // leave room for one register bank beside the four accumulators: the wide kernel
                // shares that bank between its unrolled copies, and the tree stays U=4.
                "pressure" => {
                    "s + a[i] * ((b[i] + 1.0) * (b[i] + 2.0) + (b[i] + 3.0) * (b[i] + 4.0))"
                }
                _ => unreachable!(),
            };
            writeln!(
                source,
                "@generic fn {op}_{bound}(a: [f32; {CAP}], b: [f32; {CAP}], mut out: [f32; {CAP}]) {{ \
                 {prelude} let mut s: f32 = out[0]; for i in {START}..{end} {{ s = {rhs}; }} out[0] = s; }}"
            )
            .unwrap();
        }
    }
    source
}

fn lower(source: &str, vec_kernels: bool) -> (Program, Interner) {
    let mut interner = Interner::new();
    let (ast, diagnostics) = wukong_parser::parse_module(source, SourceId(0), &mut interner);
    assert!(diagnostics.is_empty(), "{diagnostics:?}");
    let (sema, diagnostics) = wukong_sema::check(&ast, &interner);
    assert!(diagnostics.is_empty(), "{diagnostics:?}");
    let (program, diagnostics) = wukong_mir_build::lower_program_with(
        &ast,
        &sema,
        &mut interner,
        wukong_mir_build::LowerOptions {
            recognize: false,
            vec_kernels,
        },
    );
    assert!(diagnostics.is_empty(), "{diagnostics:?}");
    (program, interner)
}

struct Case {
    label: &'static str,
    a: Vec<f32>,
    b: Vec<f32>,
    seed: f32,
    expected: f32,
}

fn cases(op: &str, n: usize) -> Vec<Case> {
    let plain = |label, fill, seed, expected| {
        let mut a = vec![fill; CAP];
        // These must never contribute: every test reduction begins at a nonzero offset.
        a[..START].fill(f32::INFINITY);
        Case {
            label,
            a,
            b: vec![1.0; CAP],
            seed,
            expected,
        }
    };
    if op == "pressure" {
        let mut case = plain("exact pressure expression", 1.0, 7.0, 7.0 + 14.0 * n as f32);
        case.b.fill(0.0);
        // U=4 sends indices 0, 8, 16 to lane 0 of accumulators 0, 1, 2, combined in that order:
        // cancellation finishes before the final 14 is added. With b=0 the exact multiplier is 14.
        let expected = match n {
            0 => 0.0,
            1..=8 => f32::from_bits(0x7360_0000), // 14 * 2^100, exactly representable
            9..=16 => 0.0,
            _ => 14.0,
        };
        let mut cancellation = plain("pressure-limited cancellation", 0.0, 0.0, expected);
        cancellation.b.fill(0.0);
        cancellation.a[START] = f32::from_bits(0x7180_0000); // 2^100
        cancellation.a[START + 8] = -f32::from_bits(0x7180_0000);
        cancellation.a[START + 16] = 1.0;
        return vec![case, cancellation];
    }
    let sum = op == "sum" || op == "dot";
    let nan = f32::from_bits(0x7fc1_2345);
    let mut cases = vec![
        plain(
            "nonzero seed",
            5.0,
            7.0,
            if sum {
                7.0 + 5.0 * n as f32
            } else if n == 0 || op == "max" {
                7.0
            } else {
                5.0
            },
        ),
        plain(
            "signed zero",
            -0.0,
            -0.0,
            if sum && n >= 8 { 0.0 } else { -0.0 },
        ),
        plain("NaN seed", 5.0, nan, if sum || n == 0 { nan } else { 5.0 }),
    ];
    if sum {
        // In the canonical 32-partial main loop, index 32 joins index 0 before index 16
        // cancels it. With only 33..39 elements index 32 is a scalar tail instead.
        let expected = match n {
            0 => 0.0,
            1..=16 => 1e20,
            17..=32 => 0.0,
            33..=39 => 1.0,
            _ => 0.0,
        };
        let mut cancellation = plain("partition cancellation", 0.0, 0.0, expected);
        cancellation.a[START] = 1e20;
        cancellation.a[START + 16] = -1e20;
        cancellation.a[START + 32] = 1.0;
        cases.push(cancellation);
        // Horizontal cancellation must finish BEFORE combining the scalar seed. Folding the
        // seed into lane 0 first loses this 1.0 even with the correct count of partials.
        let mut seed = plain(
            "seed after horizontal fold",
            0.0,
            1.0,
            if n == 0 || n >= 8 {
                1.0
            } else if n == 1 {
                1e20
            } else {
                0.0
            },
        );
        seed.a[START] = 1e20;
        seed.a[START + 1] = -1e20;
        cases.push(seed);
    }
    if n >= 32 && n.is_multiple_of(32) {
        // Last accumulator, last lane: this NaN reaches the last horizontal operand, even for
        // max/min's RHS-on-unordered rule. Its quiet payload is an explicit bit-level oracle.
        let mut last = plain("last-lane NaN", 5.0, 7.0, nan);
        last.a[START + n - 1] = nan;
        cases.push(last);
    }
    cases
}

/// This integration binary owns its environment (one test); no other test can race the switch.
/// The explicit cancellation/seed/zero/NaN oracles are independent of either backend. Running
/// every configuration against them also pins width-on/off equality without tolerances.
#[test]
fn reduction_partition_is_independent_of_width_and_threshold() {
    let _restore = RestoreEnv(std::env::var_os("WUKONG_P4_NO_256"));
    let source = source();
    for mode in ["wide", "option-off", "switch-off"] {
        if mode == "switch-off" {
            std::env::set_var("WUKONG_P4_NO_256", "1");
        } else {
            std::env::remove_var("WUKONG_P4_NO_256");
        }
        let (base, mut interner) = lower(&source, mode != "option-off");
        let kernels: Vec<_> = base.funcs.iter().flat_map(|f| &f.vec_kernels).collect();
        if mode == "wide" && wukong_mir::host_supports_vec_kernels() {
            assert!(!kernels.is_empty(), "wide case must dispatch");
            assert!(
                kernels
                    .iter()
                    .any(|k| interner.resolve(k.name).starts_with("pressure_")
                        && k.reduction_banks() < k.reduction_unroll()),
                "must exercise shared register banks"
            );
            // The canonical tree: register pressure changes the schedule, never the accumulator
            // count, for every body.
            for k in &kernels {
                assert_eq!(k.reduction_unroll(), 4, "canonical reduction partition");
            }
        } else {
            assert!(kernels.is_empty(), "portable mode emitted raw kernels");
        }
        assert!(
            base.funcs
                .iter()
                .flat_map(|f| &f.blocks)
                .flat_map(|b| &b.insts)
                .any(|i| matches!(i.op, Op::Load(_, wukong_mir::MirType::Vec(_, 4)))),
            "small/portable cases must exercise four-lane SIMD"
        );
        for opt in [0, 2] {
            let mut program = base.clone();
            wukong_opt::optimize_verified(&mut program, opt).unwrap();
            let native = wukong_codegen_cranelift::jit_module(&program, &interner).unwrap();
            for op in ["sum", "dot", "max", "min", "pressure"] {
                for &n in SIZES {
                    for dynamic in [false, true] {
                        let name = if dynamic {
                            format!("{op}_dynamic")
                        } else {
                            format!("{op}_{n}")
                        };
                        let symbol = interner.intern(&name);
                        // The source declaration fixes three pointer parameters and a void return.
                        let function: unsafe extern "C" fn(*const f32, *const f32, *mut f32) = unsafe {
                            std::mem::transmute(
                                native.func_ptr(symbol).expect("entry remains present"),
                            )
                        };
                        for case in cases(op, n) {
                            let mut out = vec![19.0; CAP];
                            out[0] = case.seed;
                            out[1] = n as f32;
                            let mut interpreted = out.clone();
                            let initial = out.clone();
                            let mut a = case.a.clone();
                            let mut b = case.b.clone();
                            wukong_interp::run_kernel_f32(
                                &program,
                                symbol,
                                &mut [&mut a, &mut b, &mut interpreted],
                                &interner,
                            )
                            .unwrap();
                            unsafe {
                                function(case.a.as_ptr(), case.b.as_ptr(), out.as_mut_ptr());
                            }
                            for (backend, actual) in [("interp", &interpreted), ("native", &out)] {
                                assert_eq!(
                                    actual[0].to_bits(),
                                    case.expected.to_bits(),
                                    "{mode} O{opt} {backend} {name} n={n}: {}",
                                    case.label
                                );
                                assert!(
                                    actual[1..]
                                        .iter()
                                        .zip(&initial[1..])
                                        .all(|(a, b)| a.to_bits() == b.to_bits()),
                                    "{backend} wrote outside result slot"
                                );
                            }
                        }
                    }
                }
            }
        }
    }
}
