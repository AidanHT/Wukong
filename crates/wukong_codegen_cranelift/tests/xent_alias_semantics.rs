#[path = "support/alias_guard.rs"]
mod alias_guard;

use std::fmt::Write;
use wukong_mir::{Op, Program};
use wukong_span::{Interner, SourceId};

const ROWS: usize = 16;
const COLS: usize = 2;
const CAPACITY: usize = ROWS * COLS + 8;

#[derive(Clone, Copy, Debug)]
enum Direction {
    Forward,
    Backward,
}

fn initial() -> Vec<f32> {
    (0..CAPACITY)
        .map(|lane| {
            if lane < ROWS * COLS {
                (lane / COLS) as f32
            } else {
                (100 + lane) as f32
            }
        })
        .collect()
}

fn expected(direction: Direction) -> Vec<f32> {
    let mut out = initial();
    // Each input row consists of two equal finite logits: exp(x - max) is exactly 1,
    // the denominator is exactly 2, and its logarithm is the correctly rounded f32 ln(2).
    // Preserve the source's individual f32 addition/subtraction roundings for forward loss.
    for row in 0..ROWS {
        match direction {
            Direction::Forward => {
                let peak = row as f32;
                out[row] = (peak + std::f32::consts::LN_2) - peak;
            }
            Direction::Backward => {
                for column in 0..COLS {
                    out[row * COLS + column] = if column == row % COLS { -0.5 } else { 0.5 };
                }
            }
        }
    }
    out
}

fn source(direction: Direction, alias: bool, local_view: bool, parallel: bool) -> String {
    let labels = (0..ROWS)
        .map(|row| (row % COLS).to_string())
        .collect::<Vec<_>>()
        .join(", ");
    let logits = initial()
        .iter()
        .map(|value| format!("{value:.1}"))
        .collect::<Vec<_>>()
        .join(", ");
    let attribute = if parallel { "@parallel" } else { "" };
    let mut nest = format!(
        "for r in 0..{ROWS} {{
            let mut m: f32 = x[r * {COLS}];
            for i in 0..{COLS} {{ m = fmax(m, x[r * {COLS} + i]); }}
            let mut z: f32 = 0.0;
            for i in 0..{COLS} {{ z = z + exp(x[r * {COLS} + i] - m); }}"
    );
    match direction {
        Direction::Forward => writeln!(
            nest,
            "output[r] = m + log(z) - x[r * {COLS} + target[r]]; }}"
        ),
        Direction::Backward => writeln!(
            nest,
            "let invz: f32 = 1.0 / z;
             for i in 0..{COLS} {{ output[r * {COLS} + i] = exp(x[r * {COLS} + i] - m) * invz; }}
             output[r * {COLS} + target[r]] = output[r * {COLS} + target[r]] - 1.0; }}"
        ),
    }
    .unwrap();
    if local_view {
        let binding = if alias {
            "let x: []f32 = output;".to_string()
        } else {
            format!("let x: [f32; {CAPACITY}] = [{logits}];")
        };
        format!(
            "{attribute} fn entry(mut output: [f32; {CAPACITY}]) {{
                {binding} let target: [i32; {ROWS}] = [{labels}]; {nest}
             }}"
        )
    } else {
        let (binding, argument) = if alias {
            (String::new(), "output")
        } else {
            (format!("let x: [f32; {CAPACITY}] = [{logits}];"), "x")
        };
        format!(
            "{attribute} fn ce(x: [f32; {CAPACITY}], target: [i32; {ROWS}], mut output: [f32; {CAPACITY}]) {{
                {nest}
             }}
             fn entry(mut output: [f32; {CAPACITY}]) {{
                {binding} let target: [i32; {ROWS}] = [{labels}]; ce({argument}, target, output);
             }}"
        )
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

fn dispatches(program: &Program, interner: &Interner) -> Vec<String> {
    program
        .funcs
        .iter()
        .flat_map(|function| &function.blocks)
        .flat_map(|block| &block.insts)
        .filter_map(|inst| match &inst.op {
            Op::Call { func, .. } => {
                let name = interner.resolve(*func);
                name.starts_with("wukong_xent_").then(|| name.to_string())
            }
            _ => None,
        })
        .collect()
}

fn unguarded_dispatches(program: &Program, interner: &Interner) -> Vec<String> {
    alias_guard::unguarded_dispatches(program, interner, &|name| name.starts_with("wukong_xent_"))
}

fn check(direction: Direction, local_view: bool) {
    let expected = expected(direction);
    for alias in [true, false] {
        for parallel in [false, true] {
            let source = source(direction, alias, local_view, parallel);
            for recognize in [true, false] {
                let label = format!(
                    "{direction:?}, alias={alias}, local_view={local_view}, parallel={parallel}, recognize={recognize}"
                );
                let (program, mut interner) = lower(&source, recognize);
                let calls = dispatches(&program, &interner);
                if recognize && !alias {
                    let direction = match direction {
                        Direction::Forward => "fwd",
                        Direction::Backward => "bwd",
                    };
                    let suffix = if parallel { "_parallel" } else { "" };
                    assert_eq!(
                        calls,
                        [format!("wukong_xent_{direction}_f32{suffix}")],
                        "{label}"
                    );
                } else if recognize {
                    // Run-time alias versioning (`wukong_mir_build::kernel_guard`) dispatches a
                    // may-alias call site behind a `ptrtoint` overlap test whose other arm is the
                    // ORIGINAL source loop, so the kernel may be present -- but never unguarded.
                    // Check this before native execution: an unguarded aliased call would race.
                    // The every-lane comparison below proves the guard took the source loop.
                    let unguarded = unguarded_dispatches(&program, &interner);
                    assert!(
                        unguarded.is_empty(),
                        "{label}: unguarded dispatch {unguarded:?}"
                    );
                } else {
                    assert!(calls.is_empty(), "{label}: unexpected dispatch {calls:?}");
                }
                let entry = interner.intern("entry");
                for level in [0, 2] {
                    let mut optimized = program.clone();
                    wukong_opt::optimize_verified(&mut optimized, level).unwrap();
                    let optimized_calls = dispatches(&optimized, &interner);
                    // Inlining may duplicate a callee's call while retaining the original function.
                    assert_eq!(
                        optimized_calls.is_empty(),
                        calls.is_empty(),
                        "{label} O{level}"
                    );
                    if alias {
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
                    let mut interpreted = initial();
                    wukong_interp::run_kernel_f32(
                        &optimized,
                        entry,
                        &mut [&mut interpreted],
                        &interner,
                    )
                    .unwrap();
                    let module =
                        wukong_codegen_cranelift::jit_module(&optimized, &interner).unwrap();
                    // The source entry has one fixed f32 array parameter (pointer ABI) and no return.
                    let function: unsafe extern "C" fn(*mut f32) = unsafe {
                        std::mem::transmute(module.func_ptr(entry).expect("entry remains present"))
                    };
                    let mut native = initial();
                    unsafe { function(native.as_mut_ptr()) };
                    for (backend, actual) in [("interpreter", interpreted), ("native", native)] {
                        for (lane, (actual, expected)) in actual.iter().zip(&expected).enumerate() {
                            assert_eq!(
                                actual.to_bits(),
                                expected.to_bits(),
                                "{label} O{level} {backend} lane {lane}: {actual} != {expected}"
                            );
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn forward_call_site_alias_and_disjoint_control() {
    check(Direction::Forward, false);
}

#[test]
fn backward_call_site_alias_and_disjoint_control() {
    check(Direction::Backward, false);
}

#[test]
fn forward_local_slice_alias_and_fresh_array_control() {
    check(Direction::Forward, true);
}

#[test]
fn backward_local_slice_alias_and_fresh_array_control() {
    check(Direction::Backward, true);
}
