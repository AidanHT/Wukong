//! Tests of the `@parallel` reduction tree (`par_reduce.rs`, P13.7): which lowering each reduction
//! spelling takes inside a `@parallel` function, that every one of them reaches the runtime's tree
//! (a kernel or the chunk-buffer fallback over `wukong_sreduce_f32`), and that nothing outside the
//! contract moves.
//!
//! Each test reads `par_reduce::take_trace`, so a reduction that silently fell to the vectorizer's
//! lane tree (no trace entry) fails by name rather than by a value some fixture may not exercise.
//! The VALUES — the tree's bits on cancellation data across interp, native, `-O0`/`-O2`, thread
//! counts and recognition on/off — are pinned by `tests/run/parallel_reduce_order*.wk` and
//! `crates/wukongc/tests/parallel_lane.rs`.

use std::collections::BTreeMap;
use std::path::Path;

use wukong_mir::verify::verify_function;
use wukong_mir::{CastKind, Function, MirType, Op, Program};
use wukong_span::{Interner, SourceId};

use crate::par_reduce::{self, TreePath};
use crate::{lower_program_with, LowerOptions};

fn lower_opts(src: &str, opts: LowerOptions) -> (Program, Interner, Vec<TreePath>) {
    let mut interner = Interner::new();
    let (module, pdiags) = wukong_parser::parse_module(src, SourceId(0), &mut interner);
    assert!(pdiags.is_empty(), "parse: {pdiags:?}");
    let (sema, sdiags) = wukong_sema::check(&module, &interner);
    assert!(sdiags.iter().all(|d| !d.is_error()), "sema: {sdiags:?}");
    par_reduce::take_trace();
    let (prog, diags) = lower_program_with(&module, &sema, &mut interner, opts);
    let trace = par_reduce::take_trace();
    assert!(diags.iter().all(|d| !d.is_error()), "lower: {diags:?}");
    for f in &prog.funcs {
        let errs = verify_function(f);
        assert!(errs.is_empty(), "{}: {errs:?}", interner.resolve(f.name));
    }
    (prog, interner, trace)
}

fn lower(src: &str) -> (Program, Interner, Vec<TreePath>) {
    lower_opts(src, LowerOptions::default())
}

fn func<'p>(prog: &'p Program, interner: &Interner, name: &str) -> &'p Function {
    prog.funcs
        .iter()
        .find(|f| interner.resolve(f.name) == name)
        .unwrap_or_else(|| panic!("no function `{name}`"))
}

fn calls(f: &Function, interner: &Interner, callee: &str) -> usize {
    f.blocks
        .iter()
        .flat_map(|b| b.insts.iter())
        .filter(|i| matches!(&i.op, Op::Call { func, .. } if interner.resolve(*func) == callee))
        .count()
}

/// Does `f` compute anything on `f32` vectors (the sequential vectorizer's lane tree)?
fn has_f32_vectors(f: &Function) -> bool {
    f.blocks.iter().flat_map(|b| b.insts.iter()).any(|i| {
        i.result
            .is_some_and(|v| matches!(f.value_type(v), MirType::Vec(e, _) if **e == MirType::F32))
    })
}

/// A program holding `body` as the whole of a `@parallel` function `r` over `x: []f32` (and a second
/// operand `y`), returning `s`.
fn par_fn(attrs: &str, body: &str) -> String {
    format!(
        "{attrs}\nfn r(x: []f32, y: []f32, n: i64) -> f32 {{\n    let mut s: f32 = 0.5;\n    {body}\n    return s;\n}}\n"
    )
}

/// The spellings no `f32` reduction kernel matches. Each must still compute the parallel tree.
const DECLINED: &[&str] = &[
    // An inclusive range.
    "for i in 0..=n - 1 { s += x[i]; }",
    // A `step`.
    "for i in 0..n step 2 { s += x[i]; }",
    // A non-zero start.
    "for i in 1..n { s += x[i]; }",
    // An addend with no kernel.
    "for i in 0..n { s += x[i] * 2.0; }",
    "for i in 0..n { s = s + x[i] * y[i] * 0.5; }",
    "for i in 0..n { s += exp(x[i]); }",
    "for i in 0..n { s += if x[i] > 0.0 { x[i] } else { 0.0 }; }",
    // An index that is not the bare loop variable.
    "for i in 0..n - 1 { s += x[i + 1]; }",
    "for i in 0..n { s += x[n - 1 - i] * y[i]; }",
    // A `+` chain holding the accumulator once.
    "for i in 0..n { s = s + x[i] + y[i]; }",
    // A running max / min of an expression.
    "for i in 0..n { s = fmax(s, x[i] - y[i]); }",
    "for i in 0..n { s = fmin(y[i] * 3.0, s); }",
    // An unsigned counter whose end is known only at run time.
    "let m: usize = n as usize; for i in 0..m { s += x[i]; }",
    // No loop variable at all.
    "for _ in 0..n { s += y[0]; }",
];

#[test]
fn a_kernel_shape_takes_the_multicore_kernel() {
    let (prog, it, trace) = lower(&par_fn("@parallel", "for i in 0..n { s += x[i] * y[i]; }"));
    assert_eq!(trace, [TreePath::Kernel]);
    let f = func(&prog, &it, "r");
    assert_eq!(calls(f, &it, "wukong_sreduce_f32_parallel"), 1);
    assert_eq!(calls(f, &it, "wukong_sreduce_f32"), 0);
}

#[test]
fn every_declined_spelling_takes_the_tree_fallback() {
    for body in DECLINED {
        let (prog, it, trace) = lower(&par_fn("@parallel", body));
        assert_eq!(trace, [TreePath::Fallback], "{body}");
        let f = func(&prog, &it, "r");
        // Two flushes: a full chunk inside the loop, the final partial one after it.
        assert_eq!(calls(f, &it, "wukong_sreduce_f32"), 2, "{body}");
        assert!(!has_f32_vectors(f), "{body}: vectorized with the lane tree");
    }
}

/// The same spellings in a function without `@parallel` are untouched: no trace, no reduction call.
#[test]
fn without_parallel_nothing_moves() {
    for body in DECLINED
        .iter()
        .chain(&["for i in 0..n { s += x[i] * y[i]; }"])
    {
        let (prog, it, trace) = lower(&par_fn("", body));
        assert!(trace.is_empty(), "{body}: {trace:?}");
        let f = func(&prog, &it, "r");
        assert_eq!(calls(f, &it, "wukong_sreduce_f32"), 0, "{body}");
    }
}

/// `@strict_fp` keeps the source-order fold, `@parallel` or not.
#[test]
fn strict_parallel_keeps_the_source_order() {
    for body in DECLINED
        .iter()
        .chain(&["for i in 0..n { s += x[i] * y[i]; }"])
    {
        let (prog, it, trace) = lower(&par_fn("@parallel\n@strict_fp", body));
        assert!(trace.is_empty(), "{body}: {trace:?}");
        let f = func(&prog, &it, "r");
        assert_eq!(calls(f, &it, "wukong_sreduce_f32"), 0, "{body}");
        assert_eq!(calls(f, &it, "wukong_sreduce_f32_parallel"), 0, "{body}");
    }
}

/// With recognition off (`WUKONG_NO_RECOGNIZE`, `@generic`) the kernel shape takes the fallback —
/// the same tree, so recognition cannot change a `@parallel` reduction's bits.
#[test]
fn recognition_off_still_takes_the_tree() {
    let src = par_fn("@parallel", "for i in 0..n { s += x[i] * y[i]; }");
    let opts = LowerOptions {
        recognize: false,
        ..LowerOptions::default()
    };
    let (prog, it, trace) = lower_opts(&src, opts);
    assert_eq!(trace, [TreePath::Fallback]);
    assert_eq!(calls(func(&prog, &it, "r"), &it, "wukong_sreduce_f32"), 2);
    let (prog, it, trace) = lower(&par_fn(
        "@parallel\n@generic",
        "for i in 0..n { s += x[i] * y[i]; }",
    ));
    assert_eq!(trace, [TreePath::Fallback]);
    let f = func(&prog, &it, "r");
    assert_eq!(calls(f, &it, "wukong_sreduce_f32"), 2);
    assert_eq!(calls(f, &it, "wukong_sreduce_f32_parallel"), 0);
}

/// The op code of every `wukong_sreduce_f32` call in `f` (its fourth argument, a constant).
fn sreduce_ops(f: &Function, interner: &Interner) -> Vec<i128> {
    let insts = || f.blocks.iter().flat_map(|b| b.insts.iter());
    insts()
        .filter_map(|i| match &i.op {
            Op::Call { func, args } if interner.resolve(*func) == "wukong_sreduce_f32" => insts()
                .find_map(|d| match &d.op {
                    Op::ConstInt(v, _) if d.result == Some(args[3]) => Some(*v),
                    _ => None,
                }),
            _ => None,
        })
        .collect()
}

/// A product addend keeps the rounding the source gives the product. `s = s + a * b` (either
/// operand order) contracts in the source at the `f32` of the `+` whatever the product's own type,
/// so the fallback fuses the widened factors (`RED_DOT`) — a `bf16 * bf16` product used to be
/// rounded to bf16 first, and an `f32 * i16` one to f32. The compound `s += a * b` fuses an `f32`
/// product, as the kernel does, but a half product is rounded to its half type by the source before
/// the `+`, so it stays a plain addend (`RED_SUM`). A narrow-integer addend widens into the fold
/// too, an unsigned one unsigned. Values: `tests/run/parallel_reduce_order_lowp.wk`.
#[test]
fn a_product_addend_keeps_the_sources_rounding() {
    let (dot, sum) = (i128::from(crate::RED_DOT), i128::from(crate::RED_SUM));
    let cases: &[(&str, &str, i128)] = &[
        ("x: []bf16, y: []bf16", "s = s + x[i] * y[i];", dot),
        ("x: []f16, y: []f16", "s = x[i] * y[i] + s;", dot),
        ("x: []bf16, c: i8", "s = s + x[i] * c;", dot),
        ("x: []f32, c: i16", "s = s + x[i] * c;", dot),
        ("x: []f32, c: i16", "s += x[i] * c;", dot),
        ("x: []bf16, y: []bf16", "s += x[i] * y[i];", sum),
        ("x: []f16, y: []f16", "s += x[i] * y[i];", sum),
        ("q: []i16", "s += q[i];", sum),
        ("u: []u16", "s = s + u[i];", sum),
    ];
    for &(params, body, op) in cases {
        let src = format!(
            "@parallel\nfn r({params}, n: i64) -> f32 {{\n    let mut s: f32 = 0.5;\n    for i in 0..n {{ {body} }}\n    return s;\n}}\n"
        );
        let (prog, it, trace) = lower(&src);
        assert_eq!(trace, [TreePath::Fallback], "{params}: {body}");
        let f = func(&prog, &it, "r");
        assert_eq!(sreduce_ops(f, &it), [op, op], "{params}: {body}");
        assert!(!has_f32_vectors(f), "{params}: {body}");
    }
    // The `u16` addend is converted unsigned, never sign-extended.
    let src = "@parallel\nfn r(u: []u16, n: i64) -> f32 {\n    let mut s: f32 = 0.5;\n    for i in 0..n { s += u[i]; }\n    return s;\n}\n";
    let (prog, it, _) = lower(src);
    let casts: Vec<CastKind> = func(&prog, &it, "r")
        .blocks
        .iter()
        .flat_map(|b| b.insts.iter())
        .filter_map(|i| match &i.op {
            Op::Cast(k, _, MirType::F32) => Some(*k),
            _ => None,
        })
        .collect();
    assert!(casts.contains(&CastKind::UiToFp), "{casts:?}");
    assert!(!casts.contains(&CastKind::SiToFp), "{casts:?}");
}

/// The chunk buffer is one chunk, or the whole trip when that is a smaller constant: an inner
/// 64-element reduction must not carry a 32 KB frame.
#[test]
fn the_chunk_buffer_is_sized_by_a_constant_trip() {
    let alloca_lens = |f: &Function| -> Vec<u32> {
        let mut v: Vec<u32> = f
            .blocks
            .iter()
            .flat_map(|b| b.insts.iter())
            .filter_map(|i| match &i.op {
                Op::Alloca {
                    ty: MirType::Array(e, n),
                    ..
                } if **e == MirType::F32 => Some(*n),
                _ => None,
            })
            .collect();
        v.sort_unstable();
        v
    };
    let src = "@parallel\nfn r(x: [f32; 100]) -> f32 {\n    let mut s: f32 = 0.0;\n    for i in 0..=63 { s += x[i + 1]; }\n    for i in 0..100 step 3 { s += x[i] * x[i]; }\n    return s;\n}\n";
    let (prog, it, trace) = lower(src);
    assert_eq!(trace, [TreePath::Fallback, TreePath::Fallback]);
    // 64 for `0..=63`; 34 (= ceil(100 / 3)) twice for the product's two factors.
    assert_eq!(alloca_lens(func(&prog, &it, "r")), [34, 34, 64]);
    let (prog, it, _) = lower(&par_fn("@parallel", "for i in 1..n { s += x[i]; }"));
    assert_eq!(alloca_lens(func(&prog, &it, "r")), [8192]);
}

/// A reduction inside an outlined `@parallel` region keeps the tree: the kernel shape calls the
/// serial twin, any other spelling the fallback — never the vectorizer's lane tree — so whether
/// the region outliner fires cannot change the bits either.
#[test]
fn a_reduction_inside_an_outlined_region_keeps_the_tree() {
    let src = "@parallel\nfn rows(x: [f32; 256], @noalias mut out: [f32; 4]) {\n    let c: f32 = 0.5;\n    for r in 0..4 {\n        let mut s: f32 = 0.0;\n        for k in 0..64 { s += x[r * 64 + k] * c; }\n        let mut t: f32 = 0.0;\n        for k in 0..256 { t += x[k]; }\n        out[r] = s + t;\n    }\n}\n";
    crate::par_region::take_trace();
    let (prog, it, trace) = lower(src);
    // The positive control: the row loop really is an outlined region.
    assert_eq!(crate::par_region::take_trace(), [Ok(())]);
    let mut sorted = trace.clone();
    sorted.sort();
    assert_eq!(sorted, [TreePath::Kernel, TreePath::Fallback], "{trace:?}");
    let body = prog
        .funcs
        .iter()
        .find(|f| it.resolve(f.name).starts_with("wukong$par$"))
        .expect("the row loop is outlined");
    assert_eq!(calls(body, &it, "wukong_sreduce_f32"), 3); // two fallback flushes + the kernel
    assert_eq!(calls(body, &it, "wukong_sreduce_f32_parallel"), 0);
    assert!(!has_f32_vectors(body));
}

/// The whole-function outliner's chunk body is a region body too.
#[test]
fn a_reduction_inside_a_whole_function_chunk_keeps_the_tree() {
    let src = "@parallel\nfn rows(x: [f32; 256], mut out: [f32; 4]) {\n    for r in 0..4 {\n        let mut s: f32 = 0.0;\n        for k in 0..64 { s = s + x[r * 64 + k]; }\n        out[r] = s;\n    }\n}\n";
    let (prog, it, trace) = lower(src);
    assert_eq!(trace, [TreePath::Fallback]);
    let total: usize = prog
        .funcs
        .iter()
        .map(|f| calls(f, &it, "wukong_sreduce_f32"))
        .sum();
    assert_eq!(total, 2);
}

/// A half reduction keeps the `_parallel` tree inside a region body.
#[test]
fn a_half_reduction_in_a_region_takes_the_parallel_twin() {
    let src = "@parallel\nfn rows(x: [bf16; 256], mut out: [f32; 4]) {\n    for r in 0..4 {\n        let mut s: f32 = 0.0;\n        for k in 0..256 { s = s + (x[k] as f32); }\n        out[r] = s;\n    }\n}\n";
    let (prog, it, trace) = lower(src);
    assert_eq!(trace, [TreePath::Lowp]);
    let total: usize = prog
        .funcs
        .iter()
        .map(|f| calls(f, &it, "wukong_sum_bf16_parallel"))
        .sum();
    assert_eq!(total, 1);
}

/// Outside the contract, and declining by construction: a non-`f32` accumulator, a body that is
/// not one fold statement, an addend that calls a user function or reads the accumulator, and a
/// bound that reads it.
#[test]
fn what_the_contract_does_not_cover_is_untouched() {
    let cases = [
        "fn r(x: []f64, n: i64) -> f64 { let mut s: f64 = 0.0; for i in 0..n { s += x[i]; } return s; }",
        "fn r(x: []i32, n: i64) -> i32 { let mut s: i32 = 0; for i in 0..n { s += x[i]; } return s; }",
        "fn r(x: []f32, n: i64) -> f32 { let mut s: f32 = 0.0; for i in 0..n { let v = x[i]; s += v; } return s; }",
        "fn g(v: f32) -> f32 { return v; }\n@parallel\nfn r(x: []f32, n: i64) -> f32 { let mut s: f32 = 0.0; for i in 0..n { s += g(x[i]); } return s; }",
        "fn r(x: []f32, n: i64) -> f32 { let mut s: f32 = 0.0; for i in 0..n { s += x[i] * s; } return s; }",
        "fn r(x: []f32, n: i64) -> f32 { let mut s: f32 = 1.0; for i in 0..(s as i64) { s += x[i]; } return s; }",
    ];
    for c in cases {
        let src = if c.contains("@parallel") {
            c.to_string()
        } else {
            format!("@parallel\n{c}")
        };
        let (prog, it, trace) = lower(&src);
        assert!(trace.is_empty(), "{c}: {trace:?}");
        let f = func(&prog, &it, "r");
        assert_eq!(calls(f, &it, "wukong_sreduce_f32"), 0, "{c}");
    }
}

/// The corpus census: every `@parallel` reduction in `tests/run` and `examples` (root files without
/// imports), by the lowering it took. Printed with `--nocapture`; the floors keep the scan honest.
#[test]
fn corpus_parallel_reduction_paths() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let mut by_path: BTreeMap<TreePath, Vec<String>> = BTreeMap::new();
    let mut programs = 0usize;
    for dir in ["tests/run", "examples"] {
        let mut files: Vec<_> = std::fs::read_dir(root.join(dir))
            .expect("corpus directory")
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().is_some_and(|x| x == "wk"))
            .collect();
        files.sort();
        for path in files {
            let src = std::fs::read_to_string(&path).expect("read corpus file");
            if !src.contains("@parallel")
                || src.lines().any(|l| l.trim_start().starts_with("import "))
            {
                continue;
            }
            let mut interner = Interner::new();
            let (module, pdiags) = wukong_parser::parse_module(&src, SourceId(0), &mut interner);
            if !pdiags.is_empty() {
                continue;
            }
            let (sema, sdiags) = wukong_sema::check(&module, &interner);
            if sdiags.iter().any(|d| d.is_error()) {
                continue;
            }
            par_reduce::take_trace();
            let _ = lower_program_with(&module, &sema, &mut interner, LowerOptions::default());
            programs += 1;
            let name = format!("{dir}/{}", path.file_name().unwrap().to_string_lossy());
            for p in par_reduce::take_trace() {
                by_path.entry(p).or_default().push(name.clone());
            }
        }
    }
    eprintln!("{programs} @parallel programs; reductions by lowering:");
    for (p, files) in &by_path {
        eprintln!("  {:>3}  {p:?}", files.len());
        for f in files {
            eprintln!("         {f}");
        }
    }
    assert!(programs >= 50, "the corpus scan found {programs} programs");
    assert!(
        by_path.get(&TreePath::Kernel).map_or(0, Vec::len) >= 5,
        "{by_path:?}"
    );
    assert!(
        by_path.get(&TreePath::Fallback).map_or(0, Vec::len) >= 3,
        "{by_path:?}"
    );
}
