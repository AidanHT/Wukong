#[path = "support/alias_guard.rs"]
mod alias_guard;

use std::fmt::Write;
use wukong_mir::{Op, Program};
use wukong_span::{Interner, SourceId};

const ROWS: usize = 2;
const COLS: usize = 8;
const CAPACITY: usize = ROWS * COLS + 8;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Family {
    Softmax,
    RmsNorm,
    LayerNorm,
}

impl Family {
    fn kernel(self) -> &'static str {
        match self {
            Self::Softmax => "wukong_softmax_bwd_f32",
            Self::RmsNorm => "wukong_rmsnorm_bwd_f32",
            Self::LayerNorm => "wukong_layernorm_bwd_f32",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Alias {
    None,
    Activation,
    Gradient,
    Gamma,
}

fn input(family: Family, operand: Alias) -> Vec<f32> {
    (0..CAPACITY)
        .map(|lane| match operand {
            Alias::Activation if family == Family::Softmax => 0.125,
            Alias::Activation => {
                if lane % 2 == 0 {
                    -1.0
                } else {
                    1.0
                }
            }
            Alias::Gradient => (lane % 4 + 1) as f32,
            Alias::Gamma => [1.0, 2.0, 1.0, 2.0, 2.0, 1.0, 2.0, 1.0][lane % COLS],
            Alias::None => (100 + lane) as f32,
        })
        .collect()
}

/// Independent source-order evaluation, including writes through shared backing storage.
/// The norm inputs have mean 0 and variance/mean-square 1, so eps=0 gives rstd=rinv=1.
/// Every intermediate is a small dyadic rational: reassociation and FMA do not round differently.
/// No compiler/runtime output or runtime kernel is used to construct these expectations.
fn expected(family: Family, alias: Alias, freeze_gamma: bool) -> Vec<f32> {
    let activation = input(family, Alias::Activation);
    let gradient = input(family, Alias::Gradient);
    let gamma = input(family, Alias::Gamma);
    let mut output = input(family, alias);
    for row in 0..ROWS {
        let read = |operand, lane| {
            if alias == operand && !(operand == Alias::Gamma && freeze_gamma) {
                output[lane]
            } else {
                match operand {
                    Alias::Activation => activation[lane],
                    Alias::Gradient => gradient[lane],
                    Alias::Gamma => gamma[lane],
                    Alias::None => unreachable!(),
                }
            }
        };
        let mut dot = 0.0;
        let mut sum = 0.0;
        for col in 0..COLS {
            let lane = row * COLS + col;
            let x = read(Alias::Activation, lane);
            let dy = read(Alias::Gradient, lane);
            let g = if family == Family::Softmax {
                dy
            } else {
                dy * read(Alias::Gamma, col)
            };
            dot += g * x;
            sum += g;
        }
        for col in 0..COLS {
            let lane = row * COLS + col;
            // Read afresh for each store: gamma aliases row zero of output, so later rows
            // observe row zero's completed gradient, not gamma's initial values.
            let x = if alias == Alias::Activation {
                output[lane]
            } else {
                activation[lane]
            };
            let dy = if alias == Alias::Gradient {
                output[lane]
            } else {
                gradient[lane]
            };
            let scale = if alias == Alias::Gamma && !freeze_gamma {
                output[col]
            } else {
                gamma[col]
            };
            output[lane] = match family {
                Family::Softmax => x * (dy - dot),
                Family::RmsNorm => dy * scale - x * (dot / COLS as f32),
                Family::LayerNorm => dy * scale - sum / COLS as f32 - x * (dot / COLS as f32),
            };
        }
    }
    output
}

fn nest(family: Family) -> String {
    let body = match family {
        Family::Softmax => "
            let mut s: f32 = 0.0;
            for i in 0..COLS { s = s + x[r * COLS + i] * dy[r * COLS + i]; }
            for i in 0..COLS { output[r * COLS + i] = x[r * COLS + i] * (dy[r * COLS + i] - s); }",
        Family::RmsNorm => "
            let mut ms: f32 = 0.0;
            for i in 0..COLS { ms = ms + x[r * COLS + i] * x[r * COLS + i]; }
            let rinv: f32 = 1.0 / sqrt(ms / WIDTH + 0.0);
            let mut sg: f32 = 0.0;
            for i in 0..COLS { sg = sg + dy[r * COLS + i] * gamma[i] * x[r * COLS + i]; }
            let coef: f32 = rinv * rinv * sg / WIDTH;
            for i in 0..COLS { output[r * COLS + i] = rinv * (dy[r * COLS + i] * gamma[i] - x[r * COLS + i] * coef); }",
        Family::LayerNorm => "
            let mut sm: f32 = 0.0;
            for i in 0..COLS { sm = sm + x[r * COLS + i]; }
            let mean: f32 = sm / WIDTH;
            let mut vv: f32 = 0.0;
            for i in 0..COLS { vv = vv + (x[r * COLS + i] - mean) * (x[r * COLS + i] - mean); }
            let rstd: f32 = 1.0 / sqrt(vv / WIDTH + 0.0);
            let mut s1: f32 = 0.0;
            for i in 0..COLS { s1 = s1 + dy[r * COLS + i] * gamma[i]; }
            let mut s2: f32 = 0.0;
            for i in 0..COLS { s2 = s2 + dy[r * COLS + i] * gamma[i] * ((x[r * COLS + i] - mean) * rstd); }
            let m1: f32 = s1 / WIDTH;
            let m2: f32 = s2 / WIDTH;
            for i in 0..COLS { output[r * COLS + i] = rstd * (dy[r * COLS + i] * gamma[i] - m1 - ((x[r * COLS + i] - mean) * rstd) * m2); }",
    };
    format!("for r in 0..{ROWS} {{ {body} }}")
        .replace("COLS", &COLS.to_string())
        .replace("WIDTH", &format!("{COLS}.0"))
}

fn source(family: Family, alias: Alias, local_view: bool, parallel: bool) -> String {
    let attribute = if parallel { "@parallel" } else { "" };
    let mut bindings = String::new();
    let mut parameters = Vec::new();
    let mut arguments = Vec::new();
    for (name, operand) in [
        ("x", Alias::Activation),
        ("dy", Alias::Gradient),
        ("gamma", Alias::Gamma),
    ] {
        if family == Family::Softmax && operand == Alias::Gamma {
            continue;
        }
        parameters.push(format!("{name}: [f32; {CAPACITY}]"));
        if alias == operand {
            arguments.push("output");
            if local_view {
                writeln!(bindings, "let {name}: []f32 = output;").unwrap();
            }
        } else {
            arguments.push(name);
            let values = input(family, operand)
                .iter()
                .map(|v| format!("{v:.3}"))
                .collect::<Vec<_>>()
                .join(", ");
            writeln!(bindings, "let {name}: [f32; {CAPACITY}] = [{values}];").unwrap();
        }
    }
    let nest = nest(family);
    if local_view {
        format!("{attribute} fn entry(mut output: [f32; {CAPACITY}]) {{ {bindings} {nest} }}")
    } else {
        let parameters = parameters.join(", ");
        let arguments = arguments.join(", ");
        format!("{attribute} fn backward({parameters}, mut output: [f32; {CAPACITY}]) {{ {nest} }}
            fn entry(mut output: [f32; {CAPACITY}]) {{ {bindings} backward({arguments}, output); }}")
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

fn is_norm_bwd_kernel(name: &str) -> bool {
    [Family::Softmax, Family::RmsNorm, Family::LayerNorm]
        .iter()
        .any(|family| name.starts_with(family.kernel()))
}

fn dispatches(program: &Program, interner: &Interner) -> Vec<String> {
    program
        .funcs
        .iter()
        .flat_map(|f| &f.blocks)
        .flat_map(|b| &b.insts)
        .filter_map(|inst| match &inst.op {
            Op::Call { func, .. } => {
                let name = interner.resolve(*func);
                is_norm_bwd_kernel(name).then(|| name.to_string())
            }
            _ => None,
        })
        .collect()
}

fn unguarded_dispatches(program: &Program, interner: &Interner) -> Vec<String> {
    alias_guard::unguarded_dispatches(program, interner, &is_norm_bwd_kernel)
}

/// The `wukong_parallel_for` calls in `program` that are not an apply over the eight independent
/// columns, as (function index, block index, instruction index).
///
/// gamma[i] aliases output[i]: row zero must finish before row one reads it. An inner apply over
/// the eight independent columns is safe. Threading the two dependent rows is not, so a region
/// over them is accepted only behind the region outliner's run-time overlap guard
/// (`wukong_mir_build::par_region`: `wukong_parallel_for` in the `then` arm of a `ptrtoint` range
/// test, the outlined body called directly over every row on the calling thread otherwise). This
/// pins that structure before execution; [`assert_guard_takes_serial_arm`] then proves by
/// execution that the guard sends this aliasing input to the serial arm.
fn guarded_row_regions(
    program: &Program,
    interner: &Interner,
    label: &str,
) -> Vec<(usize, usize, usize)> {
    let mut regions = Vec::new();
    for (fi, function) in program.funcs.iter().enumerate() {
        for (bi, block) in function.blocks.iter().enumerate() {
            for (ii, inst) in block.insts.iter().enumerate() {
                let Op::Call { func, args } = &inst.op else {
                    continue;
                };
                if interner.resolve(*func) != "wukong_parallel_for" {
                    continue;
                }
                let extent = function
                    .blocks
                    .iter()
                    .flat_map(|block| &block.insts)
                    .find_map(|definition| {
                        if definition.result == Some(args[0]) {
                            if let Op::ConstInt(value, _) = &definition.op {
                                return Some(*value);
                            }
                        }
                        None
                    });
                if extent == Some(COLS as i128) {
                    continue;
                }
                assert!(
                    alias_guard::is_guarded_block(function, block.id),
                    "{label}: gamma alias outlined dependent rows (extent {extent:?}) with no \
                     run-time overlap guard"
                );
                regions.push((fi, bi, ii));
            }
        }
    }
    regions
}

/// Run `program`'s `entry` over a fresh copy of the input on the interpreter and on the native
/// backend.
fn execute(
    program: &Program,
    interner: &mut Interner,
    family: Family,
    alias: Alias,
) -> [(&'static str, Vec<f32>); 2] {
    let entry = interner.intern("entry");
    let mut interpreted = input(family, alias);
    wukong_interp::run_kernel_f32(program, entry, &mut [&mut interpreted], interner).unwrap();
    let module = wukong_codegen_cranelift::jit_module(program, interner).unwrap();
    // entry has one fixed f32 array parameter (pointer ABI) and no return.
    let function: unsafe extern "C" fn(*mut f32) =
        unsafe { std::mem::transmute(module.func_ptr(entry).expect("entry remains present")) };
    let mut native = input(family, alias);
    unsafe { function(native.as_mut_ptr()) };
    [("interpreter", interpreted), ("native", native)]
}

fn assert_every_lane(results: [(&str, Vec<f32>); 2], expected: &[f32], label: &str) {
    for (backend, actual) in results {
        assert_eq!(actual.len(), expected.len(), "{label} {backend}");
        for (lane, (actual, expected)) in actual.iter().zip(expected).enumerate() {
            assert_eq!(
                actual.to_bits(),
                expected.to_bits(),
                "{label} {backend} lane {lane}: {actual} != {expected}"
            );
        }
    }
}

/// The guard's verdict on this input, by execution: delete the threaded arm's `wukong_parallel_for`
/// of every guarded row region, so the rows run only if the guard chooses the serial arm. Had it
/// judged the gamma view and the output apart, nothing would write the rows and every lane would
/// keep its input value, which `expected` differs from. Unlike a timing-dependent race, this
/// cannot pass by luck.
fn assert_guard_takes_serial_arm(
    program: &Program,
    interner: &mut Interner,
    regions: &[(usize, usize, usize)],
    (family, alias): (Family, Alias),
    expected: &[f32],
    label: &str,
) {
    if regions.is_empty() {
        return;
    }
    let initial = input(family, alias);
    assert_ne!(
        initial[..ROWS * COLS],
        expected[..ROWS * COLS],
        "{label}: a region that never ran must be visible"
    );
    let mut serial_only = program.clone();
    // Highest instruction index first, so earlier indices in one block stay valid.
    let mut regions = regions.to_vec();
    regions.sort_unstable_by(|a, b| b.cmp(a));
    for (fi, bi, ii) in regions {
        serial_only.funcs[fi].blocks[bi].insts.remove(ii);
    }
    assert_every_lane(
        execute(&serial_only, interner, family, alias),
        expected,
        &format!("{label} threaded arm removed"),
    );
}

fn check(family: Family, local_view: bool) {
    let aliases: &[Alias] = if family == Family::Softmax {
        &[Alias::None, Alias::Activation, Alias::Gradient]
    } else {
        &[
            Alias::None,
            Alias::Activation,
            Alias::Gradient,
            Alias::Gamma,
        ]
    };
    for &alias in aliases {
        let expected = expected(family, alias, false);
        if alias == Alias::Gamma {
            let frozen = self::expected(family, alias, true);
            assert_ne!(
                expected[COLS..ROWS * COLS],
                frozen[COLS..ROWS * COLS],
                "the gamma fixture must expose live writes across rows"
            );
        }
        for parallel in [false, true] {
            let source = source(family, alias, local_view, parallel);
            for recognize in [true, false] {
                let label = format!("{family:?} alias={alias:?} local_view={local_view} parallel={parallel} recognize={recognize}");
                let (program, mut interner) = lower(&source, recognize);
                if alias == Alias::Gamma {
                    guarded_row_regions(&program, &interner, &label);
                }
                let calls = dispatches(&program, &interner);
                if recognize && alias == Alias::None {
                    let suffix = if parallel { "_parallel" } else { "" };
                    assert_eq!(calls, [format!("{}{suffix}", family.kernel())], "{label}");
                } else if recognize {
                    // Run-time alias versioning (`wukong_mir_build::kernel_guard`) dispatches a
                    // may-alias call site behind a `ptrtoint` overlap test whose other arm is
                    // the ORIGINAL source loop, so the kernel may be present -- but never
                    // unguarded. Prove that before executing a potentially racing kernel; the
                    // every-lane comparison below then proves the guard chose the source loop
                    // (or an exact in-place kernel, softmax/RMSNorm dy) on these aliases.
                    let unguarded = unguarded_dispatches(&program, &interner);
                    assert!(
                        unguarded.is_empty(),
                        "{label}: unguarded dispatch {unguarded:?}"
                    );
                } else {
                    assert!(calls.is_empty(), "{label}: unexpected dispatch {calls:?}");
                }
                for level in [0, 2] {
                    let mut optimized = program.clone();
                    wukong_opt::optimize_verified(&mut optimized, level).unwrap();
                    if alias == Alias::Gamma {
                        let label = format!("{label} O{level}");
                        let regions = guarded_row_regions(&optimized, &interner, &label);
                        assert_guard_takes_serial_arm(
                            &optimized,
                            &mut interner,
                            &regions,
                            (family, alias),
                            &expected,
                            &label,
                        );
                    }
                    let optimized_calls = dispatches(&optimized, &interner);
                    // Inlining can duplicate dispatch while keeping the original function.
                    assert_eq!(
                        optimized_calls.is_empty(),
                        calls.is_empty(),
                        "{label} O{level}"
                    );
                    if alias != Alias::None {
                        let unguarded = unguarded_dispatches(&optimized, &interner);
                        assert!(
                            unguarded.is_empty(),
                            "{label} O{level}: unguarded dispatch {unguarded:?}"
                        );
                    }
                    assert!(
                        optimized_calls.iter().all(|call| calls.contains(call)),
                        "{label} O{level}: unexpected dispatch {optimized_calls:?}"
                    );
                    assert_every_lane(
                        execute(&optimized, &mut interner, family, alias),
                        &expected,
                        &format!("{label} O{level}"),
                    );
                }
            }
        }
    }
}

#[test]
fn softmax_call_site_aliases_and_disjoint_control() {
    check(Family::Softmax, false);
}

#[test]
fn softmax_local_views_and_disjoint_control() {
    check(Family::Softmax, true);
}

#[test]
fn rmsnorm_call_site_aliases_and_disjoint_control() {
    check(Family::RmsNorm, false);
}

#[test]
fn rmsnorm_local_views_and_disjoint_control() {
    check(Family::RmsNorm, true);
}

#[test]
fn layernorm_call_site_aliases_and_disjoint_control() {
    check(Family::LayerNorm, false);
}

#[test]
fn layernorm_local_views_and_disjoint_control() {
    check(Family::LayerNorm, true);
}
