//! The floating-point contract at the MIR level: which `+` contracts into an `Fma`, and what
//! `@strict_fp` / `--fp=strict` ([`LowerOptions::strict_fp`]) turn off.
//!
//! The run fixtures (`tests/run/fp_contract_rules.wk`, `tests/run/strict_fp.wk`) prove the
//! *values*; these prove the *mechanism* — that strict mode removes every contraction site and the
//! reassociating reduction while leaving lane-wise vectorization alone — so a regression is named
//! here instead of surfacing as a one-ULP difference in a printed number.

use crate::{lower_program_with, LowerOptions};
use wukong_mir::{BinOp, Function, MirType, Op, Program, ValueId, VecOp};
use wukong_span::{FxHashMap as HashMap, Interner, SourceId};

fn lower_with(src: &str, opts: LowerOptions) -> (Program, Interner) {
    let mut interner = Interner::new();
    let (module, pdiags) = wukong_parser::parse_module(src, SourceId(0), &mut interner);
    assert!(pdiags.is_empty(), "parse: {pdiags:?}");
    let (sema, sdiags) = wukong_sema::check(&module, &interner);
    assert!(sdiags.iter().all(|d| !d.is_error()), "sema: {sdiags:?}");
    let (prog, diags) = lower_program_with(&module, &sema, &mut interner, opts);
    assert!(diags.iter().all(|d| !d.is_error()), "lower: {diags:?}");
    for f in &prog.funcs {
        let errs = wukong_mir::verify::verify_function(f);
        assert!(errs.is_empty(), "MIR fails the verifier: {errs:?}");
    }
    (prog, interner)
}

fn func<'p>(prog: &'p Program, interner: &Interner, name: &str) -> &'p Function {
    prog.funcs
        .iter()
        .find(|f| interner.resolve(f.name) == name)
        .unwrap_or_else(|| panic!("no function `{name}`"))
}

/// Every `Fma` the function can execute: MIR `Op::Fma` plus `VecOp::Fma` inside its 256-bit
/// recipes (a recipe body is the other half of a vectorized loop).
fn fma_count(f: &Function) -> usize {
    let mir = f
        .blocks
        .iter()
        .flat_map(|b| &b.insts)
        .filter(|i| matches!(i.op, Op::Fma(..)))
        .count();
    let recipes = f
        .vec_kernels
        .iter()
        .flat_map(|k| &k.ops)
        .filter(|op| matches!(op, VecOp::Fma { .. }))
        .count();
    mir + recipes
}

/// Did the lowering vectorize anything in `f` — a SIMD-typed value or a 256-bit recipe?
fn is_vectorized(f: &Function) -> bool {
    !f.vec_kernels.is_empty() || f.value_types.iter().any(|t| matches!(t, MirType::Vec(..)))
}

fn kernel_calls(prog: &Program, interner: &Interner) -> Vec<String> {
    let mut out = Vec::new();
    for f in &prog.funcs {
        for b in &f.blocks {
            for i in &b.insts {
                if let Op::Call { func, .. } = &i.op {
                    let name = interner.resolve(*func);
                    if name.starts_with("wukong_") && name != "wukong_parallel_for" {
                        out.push(name.to_string());
                    }
                }
            }
        }
    }
    out
}

/// The defining op of every instruction result in `f`.
fn defs(f: &Function) -> HashMap<u32, &Op> {
    let mut m = HashMap::default();
    for b in &f.blocks {
        for i in &b.insts {
            if let Some(r) = i.result {
                m.insert(r.0, &i.op);
            }
        }
    }
    m
}

/// Is `v` a float literal, possibly behind the width casts `lower_fma_operand` inserts?
fn is_float_const(defs: &HashMap<u32, &Op>, v: ValueId) -> bool {
    match defs.get(&v.0) {
        Some(Op::ConstFloat(..)) => true,
        Some(Op::Cast(_, inner, _)) => is_float_const(defs, *inner),
        _ => false,
    }
}

/// Is `v` an `FMul` whose right operand is a float literal (`c*2.0`, rounded on its own)?
fn is_rounded_product_with_const(defs: &HashMap<u32, &Op>, v: ValueId) -> bool {
    matches!(defs.get(&v.0), Some(Op::Bin(BinOp::FMul, _, k)) if is_float_const(defs, *k))
}

/// The only `Op::Fma` in `f`.
fn the_fma(f: &Function) -> (ValueId, ValueId, ValueId) {
    let fmas: Vec<_> = f
        .blocks
        .iter()
        .flat_map(|b| &b.insts)
        .filter_map(|i| match i.op {
            Op::Fma(a, b, c) => Some((a, b, c)),
            _ => None,
        })
        .collect();
    assert_eq!(fmas.len(), 1, "expected exactly one contraction: {fmas:?}");
    fmas[0]
}

const SCALAR_SRC: &str = "module m
fn fused(x: f32, y: f32, z: f32) -> f32 { return x + y * z; }
@strict_fp
fn strict(x: f32, y: f32, z: f32) -> f32 { return x + y * z; }
";

#[test]
fn scalar_contraction_is_the_default_and_strict_fp_removes_it() {
    let (prog, interner) = lower_with(SCALAR_SRC, LowerOptions::default());
    assert_eq!(fma_count(func(&prog, &interner, "fused")), 1);
    assert_eq!(
        fma_count(func(&prog, &interner, "strict")),
        0,
        "@strict_fp must keep `x + y*z` as a rounded multiply and a rounded add"
    );

    // `--fp=strict` is the same switch for every function, attribute or not.
    let (prog, interner) = lower_with(
        SCALAR_SRC,
        LowerOptions {
            strict_fp: true,
            ..LowerOptions::default()
        },
    );
    assert_eq!(fma_count(func(&prog, &interner, "fused")), 0);
    assert_eq!(fma_count(func(&prog, &interner, "strict")), 0);
}

/// `a*b + c*d`: the LEFT product is fused and the right one is rounded first. Distinguished here
/// by making the right factor of one product a literal, so the rounded product is recognisable.
#[test]
fn two_product_sum_fuses_the_left_product() {
    let src = "module m
fn left_first(a: f32, b: f32, c: f32) -> f32 { return a * b + c * 2.0; }
fn right_first(a: f32, b: f32, c: f32) -> f32 { return c * 2.0 + a * b; }
";
    let (prog, interner) = lower_with(src, LowerOptions::default());

    let f = func(&prog, &interner, "left_first");
    let d = defs(f);
    let (y, z, x) = the_fma(f);
    assert!(
        is_rounded_product_with_const(&d, x),
        "a*b + c*2.0 must be fma(a, b, round(c*2.0))"
    );
    assert!(!is_float_const(&d, y) && !is_float_const(&d, z));

    let f = func(&prog, &interner, "right_first");
    let d = defs(f);
    let (_y, z, x) = the_fma(f);
    assert!(
        is_float_const(&d, z),
        "c*2.0 + a*b must be fma(c, 2.0, round(a*b)) — the left product wins"
    );
    assert!(matches!(d.get(&x.0), Some(Op::Bin(BinOp::FMul, ..))));
    assert!(!is_rounded_product_with_const(&d, x));
}

/// An elementwise loop stays vectorized under `@strict_fp` (lane `k` IS scalar iteration `i + k`),
/// but neither the vector body nor the scalar remainder may contract.
#[test]
fn strict_fp_keeps_lanewise_vectorization_without_contraction() {
    let body = "(x: [f32; 64], y: [f32; 64], z: [f32; 64], mut o: [f32; 64]) {
    for i in 0..64 { o[i] = x[i] + y[i] * z[i]; }
}";
    // `main` binds four distinct arrays, so the whole-program alias facts prove the operands
    // disjoint whatever an uncalled function's default is.
    let main = "fn main() -> i32 {
    let x: [f32; 64] = [1.0; 64];
    let y: [f32; 64] = [2.0; 64];
    let z: [f32; 64] = [3.0; 64];
    let mut o: [f32; 64] = [0.0; 64];
    let mut p: [f32; 64] = [0.0; 64];
    fast(x, y, z, o);
    strict(x, y, z, p);
    return 0;
}";
    let src = format!("module m\n@generic\nfn fast{body}\n@strict_fp\nfn strict{body}\n{main}\n");
    let (prog, interner) = lower_with(&src, LowerOptions::default());

    let fast = func(&prog, &interner, "fast");
    assert!(is_vectorized(fast), "the control loop must vectorize");
    assert!(fma_count(fast) > 0, "the control loop must contract");

    let strict = func(&prog, &interner, "strict");
    assert!(
        is_vectorized(strict),
        "@strict_fp must not turn off lane-wise vectorization"
    );
    assert_eq!(fma_count(strict), 0, "@strict_fp loop contracted somewhere");
}

/// A float reduction keeps its sequential fold under `@strict_fp`; an integer reduction (exact in
/// any order) is still vectorized.
#[test]
fn strict_fp_keeps_float_reductions_sequential() {
    let src = "module m
@generic
fn fast(x: [f32; 64]) -> f32 {
    let mut s: f32 = 0.0;
    for i in 0..64 { s = s + x[i]; }
    return s;
}
@strict_fp
fn strict(x: [f32; 64]) -> f32 {
    let mut s: f32 = 0.0;
    for i in 0..64 { s = s + x[i]; }
    return s;
}
@strict_fp
fn strict_int(x: [i32; 64]) -> i32 {
    let mut s: i32 = 0;
    for i in 0..64 { s = s + x[i]; }
    return s;
}
";
    let (prog, interner) = lower_with(src, LowerOptions::default());
    assert!(
        is_vectorized(func(&prog, &interner, "fast")),
        "the control reduction must vectorize, or this test is vacuous"
    );
    assert!(
        !is_vectorized(func(&prog, &interner, "strict")),
        "@strict_fp reassociated a float reduction"
    );
    assert!(
        is_vectorized(func(&prog, &interner, "strict_int")),
        "an integer reduction is exact in any order and must stay vectorized"
    );
}

/// A recognized kernel fixes its own blocking and accumulator tree, so `@strict_fp` implies
/// `@generic`: the matmul nest stays the loop the user wrote.
#[test]
fn strict_fp_turns_off_the_recognizers() {
    let mm = "(a: [f32; 16], b: [f32; 16], mut c: [f32; 16]) {
    for i in 0..4 { for j in 0..4 {
        let mut s: f32 = 0.0;
        for k in 0..4 { s = s + a[i*4+k] * b[k*4+j]; }
        c[i*4+j] = s;
    } }
}";
    let (prog, interner) = lower_with(&format!("module m\nfn mm{mm}\n"), LowerOptions::default());
    assert!(
        kernel_calls(&prog, &interner).contains(&"wukong_sgemm".to_string()),
        "the default must dispatch the GEMM, or this test is vacuous"
    );
    let (prog, interner) = lower_with(
        &format!("module m\n@strict_fp\nfn mm{mm}\n"),
        LowerOptions::default(),
    );
    let calls = kernel_calls(&prog, &interner);
    assert!(calls.is_empty(), "@strict_fp still dispatched {calls:?}");
    assert_eq!(fma_count(func(&prog, &interner, "mm")), 0);
}
