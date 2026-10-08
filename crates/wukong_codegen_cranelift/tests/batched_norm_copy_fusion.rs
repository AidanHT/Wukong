use wukong_mir::{Op, Program};
use wukong_span::{Interner, SourceId};

const ROWS: usize = 3;
const COLS: usize = 8;
const CAPACITY: usize = 32;

#[derive(Clone, Copy, Debug)]
enum Kind {
    Layer,
    Rms,
    Softmax,
    L2,
    AffineLayer,
    AffineRms,
}

impl Kind {
    fn affine(self) -> bool {
        matches!(self, Self::AffineLayer | Self::AffineRms)
    }

    fn body(self) -> String {
        let (statistics, normalized) = match self {
            Self::Layer | Self::AffineLayer => (
                "let mut s: f32 = 0.0;
                 for i in 0..8 { s = s + out[r*8+i]; }
                 let mean: f32 = s / 8.0;
                 let mut v: f32 = 0.0;
                 for i in 0..8 { v = v + (out[r*8+i]-mean) * (out[r*8+i]-mean); }
                 let inv: f32 = rsqrt(v / 8.0 + 0.0);",
                "(out[r*8+i]-mean) * inv",
            ),
            Self::Rms | Self::AffineRms => (
                "let mut s: f32 = 0.0;
                 for i in 0..8 { s = s + out[r*8+i] * out[r*8+i]; }
                 let inv: f32 = rsqrt(s / 8.0 + 0.0);",
                "out[r*8+i] * inv",
            ),
            Self::L2 => (
                "let mut s: f32 = 0.0;
                 for i in 0..8 { s = s + out[r*8+i] * out[r*8+i]; }
                 let inv: f32 = rsqrt(s + 0.0);",
                "out[r*8+i] * inv",
            ),
            Self::Softmax => (
                "let mut m: f32 = out[r*8];
                 for i in 0..8 { m = fmax(m, out[r*8+i]); }
                 for i in 0..8 { out[r*8+i] = exp(out[r*8+i]-m); }
                 let mut s: f32 = 0.0;
                 for i in 0..8 { s = s + out[r*8+i]; }
                 let inv: f32 = 1.0 / s;",
                "out[r*8+i] * inv",
            ),
        };
        let affine = if self.affine() { " * g[i] + b[i]" } else { "" };
        format!("{statistics} for i in 0..8 {{ out[r*8+i] = {normalized}{affine}; }}")
    }
}

#[derive(Clone, Copy, Debug)]
enum Shape {
    Exact,
    Folded,
    WiderCopy,
    ShorterCopy,
    Alias,
    ShadowSource,
    ShadowIndex,
    NamedRows,
    ZeroRows,
    NegativeRows,
    ObservableBody,
    SharedAffine,
}

impl Shape {
    fn fused(self) -> bool {
        matches!(self, Self::Exact | Self::Folded)
    }

    fn copy_count(self) -> usize {
        match self {
            Self::WiderCopy => CAPACITY,
            Self::ShorterCopy => 16,
            _ => ROWS * COLS,
        }
    }
}

fn source(kind: Kind, shape: Shape, parallel: bool) -> String {
    let attr = if parallel { "@parallel" } else { "" };
    let mut body = kind.body();
    match shape {
        Shape::Folded => {
            body = body
                .replace("i in 0..8", "i in 0..(1 + 7)")
                .replace("r*8", "r*(1 + 7)")
                .replace(" / 8.0", " / ((1 + 7) as f32)");
        }
        Shape::ShadowSource => {
            body = body
                .replace("mut s:", "mut x:")
                .replace("s = s +", "x = x +")
                .replace("= s /", "= x /");
        }
        Shape::ShadowIndex => {
            body = body.replace("for i in", "for x in").replace("+i]", "+x]");
        }
        Shape::ObservableBody => body.push_str(" out[31] = out[31] + 1.0;"),
        _ => {}
    }
    let (rows, copy, binding) = match shape {
        Shape::Folded => ("1 + 2".to_string(), "3 * 8".to_string(), ""),
        Shape::NamedRows => ("rows".to_string(), "24".to_string(), "let rows = 3;"),
        Shape::ZeroRows => ("0".to_string(), "24".to_string(), ""),
        Shape::NegativeRows => ("-1".to_string(), "24".to_string(), ""),
        _ => ("3".to_string(), shape.copy_count().to_string(), ""),
    };
    let input = if matches!(shape, Shape::Alias) {
        "output"
    } else {
        "input"
    };
    let beta = if matches!(shape, Shape::SharedAffine) {
        "g"
    } else {
        "b"
    };
    format!(
        "{attr} fn kernel(x: [f32; 32], g: [f32; 8], b: [f32; 8], mut out: [f32; 32]) {{
            {binding}
            for c in 0..{copy} {{ out[c] = x[c]; }}
            for r in 0..{rows} {{ {body} }}
         }}
         fn entry(input: [f32; 32], mut output: [f32; 32]) {{
            let g: [f32; 8] = [2.0; 8];
            let b: [f32; 8] = [1.0; 8];
            kernel({input}, g, {beta}, output);
         }}"
    )
}

fn initial(kind: Kind) -> Vec<f32> {
    (0..CAPACITY)
        .map(|lane| {
            if lane >= ROWS * COLS {
                return (100 + lane) as f32;
            }
            let row = lane / COLS;
            let sign = if lane % 2 == 0 { -1.0 } else { 1.0 };
            match kind {
                Kind::Layer | Kind::AffineLayer => 2.0 * row as f32 + sign,
                Kind::Rms | Kind::AffineRms => sign * (1 << row) as f32,
                Kind::Softmax => row as f32,
                Kind::L2 => {
                    if lane % COLS == 0 {
                        (1 << row) as f32
                    } else {
                        0.0
                    }
                }
            }
        })
        .collect()
}

fn destination(kind: Kind, shape: Shape) -> Vec<f32> {
    if matches!(shape, Shape::Alias) {
        initial(kind)
    } else {
        (0..CAPACITY)
            .map(|i| if i % 2 == 0 { 9.0 } else { 7.0 })
            .collect()
    }
}

fn expected(kind: Kind, shape: Shape) -> Vec<f32> {
    let input = initial(kind);
    let mut out = destination(kind, shape);
    out[..shape.copy_count()].copy_from_slice(&input[..shape.copy_count()]);
    if matches!(shape, Shape::ZeroRows | Shape::NegativeRows) {
        return out;
    }
    for row in 0..ROWS {
        for column in 0..COLS {
            let lane = row * COLS + column;
            // Chosen inputs give exact binary answers independently of reduction grouping:
            // LayerNorm has variance1, RMSNorm has power-of-two RMS, L2 has one nonzero
            // power of two, and softmax has eight equal logits. A short copy leaves the
            // last LayerNorm row [9,7,...], with mean8 and variance1.
            let mut value = match kind {
                Kind::Layer | Kind::AffineLayer => {
                    let sign = if column % 2 == 0 { -1.0 } else { 1.0 };
                    if matches!(shape, Shape::ShorterCopy) && row == 2 {
                        -sign
                    } else {
                        sign
                    }
                }
                Kind::Rms | Kind::AffineRms => {
                    if column % 2 == 0 {
                        -1.0
                    } else {
                        1.0
                    }
                }
                Kind::Softmax => 0.125,
                Kind::L2 => {
                    if column == 0 {
                        1.0
                    } else {
                        0.0
                    }
                }
            };
            if kind.affine() {
                value = value * 2.0
                    + if matches!(shape, Shape::SharedAffine) {
                        2.0
                    } else {
                        1.0
                    };
            }
            out[lane] = value;
        }
        if matches!(shape, Shape::ObservableBody) {
            out[31] += 1.0;
        }
    }
    out
}

fn lower(source: &str, recognize: bool) -> (Program, Interner) {
    let mut interner = Interner::new();
    let (ast, diagnostics) = wukong_parser::parse_module(source, SourceId(0), &mut interner);
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
            ..Default::default()
        },
    );
    assert!(diagnostics.is_empty(), "lower: {diagnostics:?}\n{source}");
    (program, interner)
}

fn check(kind: Kind, shape: Shape) {
    for parallel in [false, true] {
        for recognize in [false, true] {
            let label = format!("{kind:?} {shape:?} parallel={parallel} recognize={recognize}");
            let (program, mut interner) = lower(&source(kind, shape, parallel), recognize);
            let kernel = interner.intern("kernel");
            let function = program.funcs.iter().find(|f| f.name == kernel).unwrap();
            let calls: Vec<_> = function
                .blocks
                .iter()
                .flat_map(|b| &b.insts)
                .filter_map(|inst| {
                    if let Op::Call { func, args } = &inst.op {
                        let name = interner.resolve(*func);
                        name.starts_with("wukong_norm_").then_some((name, args))
                    } else {
                        None
                    }
                })
                .collect();
            if recognize && shape.fused() {
                let base = if kind.affine() {
                    "wukong_norm_affine_f32"
                } else {
                    "wukong_norm_f32"
                };
                let expected_call = format!("{base}{}", if parallel { "_parallel" } else { "" });
                assert_eq!(calls.len(), 1, "{label}");
                assert_eq!(calls[0].0, expected_call, "{label}");
                assert_ne!(calls[0].1[0], calls[0].1[1], "{label}: copy not fused");
                // This wrapper needs no writes of its own once both statements fuse.
                assert!(
                    !function
                        .blocks
                        .iter()
                        .flat_map(|b| &b.insts)
                        .any(|i| { matches!(i.op, Op::Store { .. }) }),
                    "{label}: copy stores remain"
                );
            } else {
                assert!(
                    calls.iter().all(|(_, args)| args[0] == args[1]),
                    "{label}: unsound fusion"
                );
                if !recognize {
                    assert!(calls.is_empty(), "{label}");
                }
            }
            let entry = interner.intern("entry");
            let expected = expected(kind, shape);
            for level in [0, 2] {
                let mut optimized = program.clone();
                wukong_opt::optimize_verified(&mut optimized, level).unwrap();
                let mut interpreted_input = initial(kind);
                let mut interpreted = destination(kind, shape);
                wukong_interp::run_kernel_f32(
                    &optimized,
                    entry,
                    &mut [&mut interpreted_input, &mut interpreted],
                    &interner,
                )
                .unwrap();
                let module = wukong_codegen_cranelift::jit_module(&optimized, &interner).unwrap();
                let function: unsafe extern "C" fn(*const f32, *mut f32) =
                    unsafe { std::mem::transmute(module.func_ptr(entry).unwrap()) };
                let native_input = initial(kind);
                let mut native = destination(kind, shape);
                unsafe { function(native_input.as_ptr(), native.as_mut_ptr()) };
                for (backend, actual, preserved) in [
                    ("interpreter", interpreted, interpreted_input),
                    ("native", native, native_input),
                ] {
                    for (lane, (actual, expected)) in actual.iter().zip(&expected).enumerate() {
                        assert_eq!(
                            actual.to_bits(),
                            expected.to_bits(),
                            "{label} O{level} {backend} lane{lane}: {actual} != {expected}"
                        );
                    }
                    assert_eq!(
                        preserved,
                        initial(kind),
                        "{label} O{level} {backend}: source changed"
                    );
                }
            }
        }
    }
}

#[test]
fn whole_batched_copy_fuses_with_exact_values() {
    for kind in [
        Kind::Layer,
        Kind::Rms,
        Kind::Softmax,
        Kind::L2,
        Kind::AffineLayer,
        Kind::AffineRms,
    ] {
        check(kind, Shape::Exact);
    }
    check(Kind::Layer, Shape::Folded);
}

#[test]
fn mismatched_or_unproved_windows_keep_the_copy() {
    for shape in [
        Shape::WiderCopy,
        Shape::ShorterCopy,
        Shape::Alias,
        Shape::ShadowSource,
        Shape::ShadowIndex,
        Shape::NamedRows,
        Shape::ZeroRows,
        Shape::NegativeRows,
        Shape::ObservableBody,
    ] {
        check(Kind::Layer, shape);
    }
    check(Kind::AffineLayer, Shape::SharedAffine);
}

#[test]
fn narrow_row_offsets_preserve_wrapping_and_safe_control() {
    for rows in [3, 33] {
        let count = rows * COLS;
        let body = Kind::Layer
            .body()
            .replace("i in 0..8", "i in 0u8..8u8")
            .replace("r*8", "r*8u8");
        let source = format!(
            "fn entry(input: [f32; {count}], mut out: [f32; {count}]) {{
                for c in 0..{count} {{ out[c] = input[c]; }}
                for r in 0u8..{rows}u8 {{ {body} }}
             }}"
        );
        let initial: Vec<f32> = (0..count)
            .map(|lane| 2.0 * (lane / COLS) as f32 + if lane % 2 == 0 { -1.0 } else { 1.0 })
            .collect();
        let expected: Vec<f32> = (0..count)
            .map(|lane| {
                // In u8 arithmetic, row32 starts at0 and normalizes row0 again.
                // Its own eight copied values63,65,... must remain untouched.
                if lane >= 256 {
                    initial[lane]
                } else if lane % 2 == 0 {
                    -1.0
                } else {
                    1.0
                }
            })
            .collect();
        for recognize in [false, true] {
            let (program, mut interner) = lower(&source, recognize);
            let entry = interner.intern("entry");
            let calls: Vec<_> = program
                .funcs
                .iter()
                .flat_map(|f| &f.blocks)
                .flat_map(|b| &b.insts)
                .filter_map(|inst| {
                    if let Op::Call { func, args } = &inst.op {
                        interner
                            .resolve(*func)
                            .starts_with("wukong_norm_")
                            .then_some(args)
                    } else {
                        None
                    }
                })
                .collect();
            if recognize && rows == 3 {
                assert_eq!(calls.len(), 1);
                assert_ne!(calls[0][0], calls[0][1]);
            } else {
                assert!(
                    calls.is_empty(),
                    "wrapping offsets must not redispatch in-place"
                );
            }
            for level in [0, 2] {
                let mut optimized = program.clone();
                wukong_opt::optimize_verified(&mut optimized, level).unwrap();
                let mut interpreted_input = initial.clone();
                let mut interpreted = vec![0.0; count];
                wukong_interp::run_kernel_f32(
                    &optimized,
                    entry,
                    &mut [&mut interpreted_input, &mut interpreted],
                    &interner,
                )
                .unwrap();
                let module = wukong_codegen_cranelift::jit_module(&optimized, &interner).unwrap();
                let function: unsafe extern "C" fn(*const f32, *mut f32) =
                    unsafe { std::mem::transmute(module.func_ptr(entry).unwrap()) };
                let native_input = initial.clone();
                let mut native = vec![0.0; count];
                unsafe { function(native_input.as_ptr(), native.as_mut_ptr()) };
                for (backend, actual, preserved) in [
                    ("interpreter", interpreted, interpreted_input),
                    ("native", native, native_input),
                ] {
                    for (lane, (actual, expected)) in actual.iter().zip(&expected).enumerate() {
                        assert_eq!(
                            actual.to_bits(),
                            expected.to_bits(),
                            "rows={rows} recognize={recognize} O{level} {backend} lane{lane}"
                        );
                    }
                    assert_eq!(preserved, initial);
                }
            }
        }
    }
}

#[test]
fn a_shadowed_row_binding_keeps_the_actual_access_pattern() {
    let source = "fn entry(input: [f32; 8], mut out: [f32; 8]) {
        for c in 0..6 { out[c] = input[c]; }
        for r in 0..3 {
            let mut s: f32 = 0.0;
            for r in 0..2 { s = s + out[r*2+r] * out[r*2+r]; }
            let inv: f32 = rsqrt(s / 2.0 + 0.0);
            for r in 0..2 { out[r*2+r] = out[r*2+r] * inv; }
        }
    }";
    // The inner binding addresses only lanes0 and3, both1: RMS=1 on every
    // repetition. Copied lanes1,2,4,5 and the uncopied suffix must remain intact.
    let initial = vec![1.0f32, 2.0, 3.0, 1.0, 4.0, 5.0, 6.0, 7.0];
    let expected = [1.0f32, 2.0, 3.0, 1.0, 4.0, 5.0, -9.0, -9.0];
    for recognize in [false, true] {
        let (program, mut interner) = lower(source, recognize);
        assert!(
            !program
                .funcs
                .iter()
                .flat_map(|f| &f.blocks)
                .flat_map(|b| &b.insts)
                .any(|inst| matches!(&inst.op, Op::Call { func, .. }
                if interner.resolve(*func).starts_with("wukong_norm_"))),
            "a rebound row must not dispatch either fused or in-place normalization"
        );
        let entry = interner.intern("entry");
        for level in [0, 2] {
            let mut optimized = program.clone();
            wukong_opt::optimize_verified(&mut optimized, level).unwrap();
            let mut interpreted_input = initial.clone();
            let mut interpreted = vec![-9.0; 8];
            wukong_interp::run_kernel_f32(
                &optimized,
                entry,
                &mut [&mut interpreted_input, &mut interpreted],
                &interner,
            )
            .unwrap();
            let module = wukong_codegen_cranelift::jit_module(&optimized, &interner).unwrap();
            let function: unsafe extern "C" fn(*const f32, *mut f32) =
                unsafe { std::mem::transmute(module.func_ptr(entry).unwrap()) };
            let native_input = initial.clone();
            let mut native = vec![-9.0; 8];
            unsafe { function(native_input.as_ptr(), native.as_mut_ptr()) };
            for (backend, actual, preserved) in [
                ("interpreter", interpreted, interpreted_input),
                ("native", native, native_input),
            ] {
                for (lane, (actual, expected)) in actual.iter().zip(&expected).enumerate() {
                    assert_eq!(
                        actual.to_bits(),
                        expected.to_bits(),
                        "recognize={recognize} O{level} {backend} lane{lane}"
                    );
                }
                assert_eq!(preserved, initial);
            }
        }
    }
}

#[test]
fn affine_struct_fields_preserve_dispatch_and_exact_lanes() {
    for slice in [false, true] {
        for copy in [false, true] {
            for parallel in [false, true] {
                let gamma_type = if slice { "[]f32" } else { "[f32; 4]" };
                let view = if slice { "let view: []f32 = g;" } else { "" };
                let gamma_value = if slice { "view" } else { "g" };
                let attribute = if parallel { "@parallel" } else { "" };
                let copy_loop = if copy {
                    "for c in 0..8 { out[c] = input[c]; }"
                } else {
                    ""
                };
                let source = format!(
                    "struct Weights {{ gamma: {gamma_type} }}
                     {attribute} fn entry(input: [f32; 8], mut out: [f32; 8]) {{
                        let g: [f32; 4] = [1.0, 2.0, 3.0, 4.0];
                        {view}
                        let weights: Weights = Weights {{ gamma: {gamma_value} }};
                        {copy_loop}
                        for r in 0..2 {{
                            let mut s: f32 = 0.0;
                            for i in 0..4 {{ s = s + out[r*4+i] * out[r*4+i]; }}
                            let inv: f32 = rsqrt(s / 4.0 + 0.0);
                            for i in 0..4 {{ out[r*4+i] = out[r*4+i] * inv * weights.gamma[i]; }}
                        }}
                     }}"
                );
                // Both rows have sum of squares16 and RMS2. Gamma is [1,2,3,4].
                let initial = vec![2.0f32, 2.0, 2.0, 2.0, 4.0, 0.0, 0.0, 0.0];
                let expected = [1.0f32, 2.0, 3.0, 4.0, 2.0, 0.0, 0.0, 0.0];
                for recognize in [false, true] {
                    let label = format!(
                        "slice={slice} copy={copy} parallel={parallel} recognize={recognize}"
                    );
                    let (program, mut interner) = lower(&source, recognize);
                    let entry = interner.intern("entry");
                    let calls: Vec<_> = program
                        .funcs
                        .iter()
                        .flat_map(|f| &f.blocks)
                        .flat_map(|b| &b.insts)
                        .filter_map(|inst| {
                            if let Op::Call { func, args } = &inst.op {
                                let name = interner.resolve(*func);
                                name.starts_with("wukong_norm_").then_some((name, args))
                            } else {
                                None
                            }
                        })
                        .collect();
                    if recognize {
                        assert_eq!(
                            calls.len(),
                            1,
                            "{label}: field access must not lose dispatch"
                        );
                        assert_eq!(
                            calls[0].0,
                            if parallel {
                                "wukong_norm_affine_f32_parallel"
                            } else {
                                "wukong_norm_affine_f32"
                            },
                            "{label}"
                        );
                        assert_eq!(calls[0].1[0] != calls[0].1[1], copy, "{label}: copy fusion");
                    } else {
                        assert!(calls.is_empty(), "{label}");
                    }
                    for level in [0, 2] {
                        let mut optimized = program.clone();
                        wukong_opt::optimize_verified(&mut optimized, level).unwrap();
                        let mut interpreted_input = initial.clone();
                        let mut interpreted = if copy { vec![-9.0; 8] } else { initial.clone() };
                        wukong_interp::run_kernel_f32(
                            &optimized,
                            entry,
                            &mut [&mut interpreted_input, &mut interpreted],
                            &interner,
                        )
                        .unwrap();
                        let module =
                            wukong_codegen_cranelift::jit_module(&optimized, &interner).unwrap();
                        let function: unsafe extern "C" fn(*const f32, *mut f32) =
                            unsafe { std::mem::transmute(module.func_ptr(entry).unwrap()) };
                        let native_input = initial.clone();
                        let mut native = if copy { vec![-9.0; 8] } else { initial.clone() };
                        unsafe { function(native_input.as_ptr(), native.as_mut_ptr()) };
                        for (backend, actual, preserved) in [
                            ("interpreter", interpreted, interpreted_input),
                            ("native", native, native_input),
                        ] {
                            for (lane, (actual, expected)) in
                                actual.iter().zip(&expected).enumerate()
                            {
                                assert_eq!(
                                    actual.to_bits(),
                                    expected.to_bits(),
                                    "{label} O{level} {backend} lane{lane}"
                                );
                            }
                            assert_eq!(
                                preserved, initial,
                                "{label} O{level} {backend}: source changed"
                            );
                        }
                    }
                }
            }
        }
    }
}
