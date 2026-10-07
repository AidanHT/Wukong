//! Structural tests for the recognizer spelling-coverage modules (`canon::inline_value_lets`,
//! `elementwise`, `ucount`, `colacc`, `softmax_oop`, `detile`, `dispatch`).
//!
//! Recognized kernels are GATE-BLIND: they fire identically on both backends at every `-O` level, so
//! no differential gate can see whether a spelling dispatched. These tests assert the emitted symbol
//! (and, where it matters, the operands) on the MIR itself; the value half lives in the
//! `tests/run/*_spelling*.wk` fixtures, which print every lane.

use wukong_ast::{Block, ExprKind, ItemKind, StmtKind};
use wukong_diag::{Diagnostic, Severity};
use wukong_mir::verify::verify_function;
use wukong_mir::{CmpOp, Function, Op, Program, ValueId};
use wukong_span::{Interner, SourceId};

use crate::{
    lower_program, lower_program_reporting_dispatch, LowerOptions, NORM_LOGSOFTMAX, NORM_SOFTMAX,
    VMATH_GELU, VMATH_SILU,
};

fn lower_src(src: &str) -> (Program, Vec<Diagnostic>, Interner) {
    let mut interner = Interner::new();
    let (module, pdiags) = wukong_parser::parse_module(src, SourceId(0), &mut interner);
    assert!(pdiags.is_empty(), "parse: {pdiags:?}");
    let (sema, sdiags) = wukong_sema::check(&module, &interner);
    assert!(sdiags.iter().all(|d| !d.is_error()), "sema: {sdiags:?}");
    let (prog, diags) = lower_program(&module, &sema, &mut interner);
    assert!(diags.iter().all(|d| !d.is_error()), "lower: {diags:?}");
    for f in &prog.funcs {
        assert!(
            verify_function(f).is_empty(),
            "MIR verify failed in `{}`",
            interner.resolve(f.name)
        );
    }
    (prog, diags, interner)
}

fn func<'p>(prog: &'p Program, interner: &Interner, name: &str) -> &'p Function {
    prog.funcs
        .iter()
        .find(|f| interner.resolve(f.name) == name)
        .unwrap_or_else(|| panic!("function `{name}` was not lowered"))
}

/// Every recognized-kernel callee in `f`, in emission order (`wukong_parallel_for` excluded).
fn kernels(f: &Function, interner: &Interner) -> Vec<String> {
    let mut out = Vec::new();
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
    out
}

/// The argument lists of every call to `callee` in `f`.
fn call_args(f: &Function, interner: &Interner, callee: &str) -> Vec<Vec<ValueId>> {
    let mut out = Vec::new();
    for b in &f.blocks {
        for i in &b.insts {
            if let Op::Call { func, args } = &i.op {
                if interner.resolve(*func) == callee {
                    out.push(args.clone());
                }
            }
        }
    }
    out
}

/// The integer constant `v` is defined as, if it is one.
fn const_int(f: &Function, v: ValueId) -> Option<i128> {
    f.blocks
        .iter()
        .flat_map(|b| &b.insts)
        .find_map(|i| match &i.op {
            Op::ConstInt(k, _) if i.result == Some(v) => Some(*k),
            _ => None,
        })
}

// ---- P12.7: a hoisted value binding keeps the elementwise kernels ------------------------------

#[test]
fn hoisted_value_bindings_keep_the_elementwise_kernels() {
    let src = "module m
fn act(x: [f32; 64], mut o: [f32; 64]) {
    for i in 0..64 {
        let v: f32 = x[i];
        o[i] = gelu(v);
    }
}
fn axpy(x: [f32; 64], y: [f32; 64], mut o: [f32; 64]) {
    for i in 0..64 {
        let a: f32 = x[i];
        let b: f32 = y[i];
        o[i] = 2.0 * a + b;
    }
}
fn swish(x: [f32; 64], mut o: [f32; 64]) {
    for i in 0..64 {
        let v: f32 = x[i];
        o[i] = v * sigmoid(v);
    }
}
";
    let (prog, _, interner) = lower_src(src);
    assert_eq!(
        kernels(func(&prog, &interner, "act"), &interner),
        vec!["wukong_vmath_f32"]
    );
    assert_eq!(
        kernels(func(&prog, &interner, "axpy"), &interner),
        vec!["wukong_velem_f32"]
    );
    let swish = func(&prog, &interner, "swish");
    assert_eq!(kernels(swish, &interner), vec!["wukong_vmath_f32"]);
    let args = call_args(swish, &interner, "wukong_vmath_f32");
    assert_eq!(const_int(swish, args[0][3]), Some(VMATH_SILU as i128));
}

/// `canon::inline_value_lets` over the body of the first statement (a `for`) of `fn f` in `src`.
fn value_view(src: &str) -> Option<Block> {
    let mut interner = Interner::new();
    let (module, pdiags) = wukong_parser::parse_module(src, SourceId(0), &mut interner);
    assert!(pdiags.is_empty(), "parse: {pdiags:?}");
    let (sema, sdiags) = wukong_sema::check(&module, &interner);
    assert!(sdiags.iter().all(|d| !d.is_error()), "sema: {sdiags:?}");
    let body = module
        .items
        .iter()
        .find_map(|item| match &item.kind {
            ItemKind::Fn(f) if interner.resolve(f.name.sym) == "f" => f.body.as_ref(),
            _ => None,
        })
        .expect("`fn f` with a body");
    let StmtKind::For { body: lb, .. } = &body.stmts[0].kind else {
        panic!("the first statement of `f` is not a `for`");
    };
    crate::canon::inline_value_lets(lb, &sema, &interner)
}

/// The view's soundness conditions, one at a time. Each refused body is one the view would
/// otherwise get WRONG, not merely one no kernel matches: a bound float product is rounded at the
/// `let` but would be contracted into one FMA beside the `+` once inlined, and a binding that
/// shadows a name an earlier initializer reads would capture that name on substitution (`x[i]`
/// inside `a`'s initializer would become an index into the second binding).
#[test]
fn the_value_view_inlines_a_load_and_refuses_unsound_bindings() {
    let src = |body: &str| -> String {
        format!(
            "module m\nfn f(x: [f32; 64], y: [f32; 64], mut o: [f32; 64]) {{\n    \
             for i in 0..64 {{ {body} }}\n}}\n"
        )
    };
    let view = value_view(&src("let v: f32 = x[i]; o[i] = gelu(v);"))
        .expect("a loaded value bound once and read by the store is inlined");
    assert_eq!(view.stmts.len(), 1, "the `let` is gone from the view");
    let StmtKind::Assign { value, .. } = &view.stmts[0].kind else {
        panic!("the view's statement is the store");
    };
    assert!(
        matches!(&value.kind, ExprKind::Call { args, .. }
            if matches!(args[0].kind, ExprKind::Index { .. })),
        "`gelu(v)` became `gelu(x[i])`"
    );
    for refused in [
        "let t: f32 = x[i] * 2.0; o[i] = t + y[i];",
        "let a: f32 = x[i]; let x: f32 = a + 1.0; o[i] = a * x;",
        "let v: f32 = x[i]; o[i] = gelu(v); o[i] = v;",
        "let v: f32 = x[i]; o[i] = gelu(x[i]);",
    ] {
        assert!(value_view(&src(refused)).is_none(), "{refused}");
    }
}

/// A binding that is still read after the store is not a value the view may inline: dropping the
/// `let` would leave the later read unbound. The loop keeps its ordinary lowering.
#[test]
fn a_binding_read_after_the_store_is_not_inlined() {
    let src = "module m
fn keep(x: [f32; 64], mut o: [f32; 64], mut p: [f32; 64]) {
    for i in 0..64 {
        let v: f32 = x[i];
        o[i] = gelu(v);
        p[i] = v;
    }
}
";
    let (prog, _, interner) = lower_src(src);
    assert!(
        !kernels(func(&prog, &interner, "keep"), &interner).contains(&"wukong_vmath_f32".into())
    );
}

// ---- P12.10: the hand-expanded GELU ------------------------------------------------------------

#[test]
fn the_kernel_order_gelu_expansion_dispatches_vmath_gelu() {
    let src = "module m
fn inline_form(x: [f32; 64], mut o: [f32; 64]) {
    for i in 0..64 {
        o[i] = 0.5 * x[i] * (1.0 + tanh(0.7978845608 * (x[i] + 0.044715 * (x[i] * x[i] * x[i]))));
    }
}
fn hoisted_form(x: [f32; 64], mut o: [f32; 64]) {
    for i in 0..64 {
        let v: f32 = x[i];
        o[i] = (1.0 + tanh((v + (v * v * v) * 0.044715) * 0.7978845608)) * (v * 0.5);
    }
}
";
    let (prog, _, interner) = lower_src(src);
    for name in ["inline_form", "hoisted_form"] {
        let f = func(&prog, &interner, name);
        assert_eq!(kernels(f, &interner), vec!["wukong_vmath_f32"], "{name}");
        let args = call_args(f, &interner, "wukong_vmath_f32");
        assert_eq!(
            const_int(f, args[0][3]),
            Some(VMATH_GELU as i128),
            "{name}: the op must be VMATH_GELU"
        );
    }
}

/// HuggingFace's literal `0.044715 * v * v * v` rounds the cubic term differently from the kernel
/// (`((C1*v)*v)*v` contracted into the FMA vs the kernel's `fma(C1, (v*v)*v, v)`), and the erf form
/// is a different function. Neither may be substituted.
#[test]
fn gelu_spellings_that_are_not_the_kernel_do_not_dispatch() {
    let src = "module m
fn hf_literal(x: [f32; 64], mut o: [f32; 64]) {
    for i in 0..64 {
        let v: f32 = x[i];
        o[i] = 0.5 * v * (1.0 + tanh(0.7978845608 * (v + 0.044715 * v * v * v)));
    }
}
fn erf_form(x: [f32; 64], mut o: [f32; 64]) {
    for i in 0..64 {
        o[i] = 0.5 * x[i] * (1.0 + erf(x[i] * 0.70710678));
    }
}
fn regrouped(x: [f32; 64], mut o: [f32; 64]) {
    for i in 0..64 {
        o[i] = 0.5 * (x[i] * (1.0 + tanh(0.7978845608 * (x[i] + 0.044715 * (x[i] * x[i] * x[i])))));
    }
}
";
    let (prog, _, interner) = lower_src(src);
    for name in ["hf_literal", "erf_form", "regrouped"] {
        assert!(
            kernels(func(&prog, &interner, name), &interner).is_empty(),
            "{name} must not dispatch"
        );
    }
}

// ---- P12.6: unsigned 64-bit counters -----------------------------------------------------------

fn has_sge(f: &Function) -> bool {
    f.blocks
        .iter()
        .flat_map(|b| &b.insts)
        .any(|i| matches!(&i.op, Op::Cmp(CmpOp::Sge, ..)))
}

/// The symbolic-dim spelling — the language's headline shape-safe form — dispatches its kernel
/// behind the run-time signed-domain test, with the loop as written as the fallback.
#[test]
fn a_symbolic_tensor_dim_dispatches_behind_a_signed_domain_guard() {
    let src = "module m
fn scale1<N>(x: Tensor[f32, N], mut out: Tensor[f32, N], s: f32) {
    for i in 0..N {
        out[i] = x[i] * s;
    }
}
";
    let (prog, _, interner) = lower_src(src);
    let f = func(&prog, &interner, "scale1");
    assert_eq!(kernels(f, &interner), vec!["wukong_velem_f32"]);
    assert!(has_sge(f), "the dispatch must sit behind `end >= 0`");
}

/// The two-input map has no `@parallel`-chunk arm in `try_vectorize_ranged`, so the guarded path
/// probes it itself; a runtime `usize` bound must not cost it the kernel either.
#[test]
fn a_runtime_usize_bound_keeps_the_two_input_map_kernel() {
    let src = "module m
fn powmap(x: [f32; 64], y: [f32; 64], mut o: [f32; 64], n: usize) {
    for i in 0..n {
        o[i] = pow(x[i], y[i]);
    }
}
";
    let (prog, _, interner) = lower_src(src);
    let f = func(&prog, &interner, "powmap");
    assert_eq!(kernels(f, &interner), vec!["wukong_vmath2_f32"]);
    assert!(has_sge(f), "the dispatch must sit behind `end >= 0`");
}

/// A reduction over a `usize` counter keeps its source-order fold: vectorizing it would reassociate
/// the sum, which the unsigned spelling never did.
#[test]
fn an_unsigned_reduction_is_not_routed_through_the_fast_path() {
    let src = "module m
fn total(x: [f32; 64], n: usize) -> f32 {
    let mut s: f32 = 0.0;
    for i in 0..n {
        s = s + x[i];
    }
    return s;
}
";
    let (prog, _, interner) = lower_src(src);
    assert!(!has_sge(func(&prog, &interner, "total")));
}

// ---- P12.8: row-outer column reductions --------------------------------------------------------

#[test]
fn row_outer_column_reductions_dispatch_the_column_kernels() {
    let src = "module m
fn sum_loop_seed(x: [f32; 32], mut out: [f32; 8]) {
    for j in 0..8 { out[j] = 0.0; }
    for i in 0..4 { for j in 0..8 { out[j] = out[j] + x[i * 8 + j]; } }
}
fn sum_let_seed(x: [f32; 32]) -> f32 {
    let mut out: [f32; 8] = [0.0; 8];
    for i in 0..4 { for j in 0..8 { out[j] += x[i * 8 + j]; } }
    return out[3];
}
fn sumsq(x: [f32; 32], mut out: [f32; 8]) {
    for j in 0..8 { out[j] = 0.0; }
    for i in 0..4 { for j in 0..8 { out[j] += x[i * 8 + j] * x[i * 8 + j]; } }
}
fn colmax(x: [f32; 32], mut out: [f32; 8]) {
    for j in 0..8 { out[j] = x[j]; }
    for i in 1..4 { for j in 0..8 { out[j] = fmax(out[j], x[i * 8 + j]); } }
}
fn colmin(x: [f32; 32], mut out: [f32; 8]) {
    for j in 0..8 { out[j] = x[j]; }
    for i in 0..4 { for j in 0..8 { out[j] = fmin(out[j], x[i * 8 + j]); } }
}
fn colabsmax(x: [f32; 32], mut out: [f32; 8]) {
    for j in 0..8 { out[j] = abs(x[j]); }
    for i in 1..4 { for j in 0..8 { out[j] = fmax(out[j], abs(x[i * 8 + j])); } }
}
";
    let (prog, _, interner) = lower_src(src);
    for (name, kernel) in [
        ("sum_loop_seed", "wukong_colsum_f32"),
        ("sum_let_seed", "wukong_colsum_f32"),
        ("sumsq", "wukong_colsumsq_f32"),
        ("colmax", "wukong_colmax_f32"),
        ("colmin", "wukong_colmin_f32"),
        ("colabsmax", "wukong_colmaxabs_f32"),
    ] {
        assert_eq!(
            kernels(func(&prog, &interner, name), &interner),
            vec![kernel],
            "{name}"
        );
    }
}

/// The spellings the kernel would change a bit of: the contracted square (one FMA in the source,
/// mul-then-add in the kernel), the swapped `fmax` operands (a different zero/NaN), an accumulate
/// into data no seed initialized, and a zero seed under a fold that skips row 0.
#[test]
fn row_outer_spellings_the_kernel_does_not_compute_decline() {
    let src = "module m
fn contracted_sq(x: [f32; 32], mut out: [f32; 8]) {
    for j in 0..8 { out[j] = 0.0; }
    for i in 0..4 { for j in 0..8 { out[j] = out[j] + x[i * 8 + j] * x[i * 8 + j]; } }
}
fn swapped_max(x: [f32; 32], mut out: [f32; 8]) {
    for j in 0..8 { out[j] = x[j]; }
    for i in 1..4 { for j in 0..8 { out[j] = fmax(x[i * 8 + j], out[j]); } }
}
fn no_seed(x: [f32; 32], mut out: [f32; 8]) {
    for i in 0..4 { for j in 0..8 { out[j] = out[j] + x[i * 8 + j]; } }
}
fn skips_row0(x: [f32; 32], mut out: [f32; 8]) {
    for j in 0..8 { out[j] = 0.0; }
    for i in 1..4 { for j in 0..8 { out[j] = out[j] + x[i * 8 + j]; } }
}
";
    let (prog, _, interner) = lower_src(src);
    for name in ["contracted_sq", "swapped_max", "no_seed", "skips_row0"] {
        let ks = kernels(func(&prog, &interner, name), &interner);
        assert!(
            !ks.iter().any(|k| k.starts_with("wukong_col")),
            "{name} must not dispatch a column kernel, got {ks:?}"
        );
    }
}

// ---- P12.9: out-of-place softmax / log-softmax -------------------------------------------------

#[test]
fn out_of_place_softmax_and_log_softmax_dispatch_the_fused_norm() {
    let src = "module m
fn softmax_fused(x: [f32; 64], mut o: [f32; 64]) {
    let mut m: f32 = x[0];
    for i in 0..64 { m = fmax(m, x[i]); }
    let mut s: f32 = 0.0;
    for i in 0..64 { o[i] = exp(x[i] - m); s = s + o[i]; }
    for i in 0..64 { o[i] = o[i] / s; }
}
fn softmax_split(x: [f32; 64], mut o: [f32; 64]) {
    let mut m: f32 = x[0];
    for i in 0..64 { m = fmax(m, x[i]); }
    for i in 0..64 { o[i] = exp(x[i] - m); }
    let mut s: f32 = 0.0;
    for i in 0..64 { s = s + o[i]; }
    let inv: f32 = 1.0 / s;
    for i in 0..64 { o[i] = o[i] * inv; }
}
fn log_softmax(x: [f32; 64], mut o: [f32; 64]) {
    let mut m: f32 = x[0];
    for i in 0..64 { m = fmax(m, x[i]); }
    let mut s: f32 = 0.0;
    for i in 0..64 { s = s + exp(x[i] - m); }
    let ls: f32 = log(s);
    for i in 0..64 { o[i] = (x[i] - m) - ls; }
}
";
    let (prog, _, interner) = lower_src(src);
    for (name, op) in [
        ("softmax_fused", NORM_SOFTMAX),
        ("softmax_split", NORM_SOFTMAX),
        ("log_softmax", NORM_LOGSOFTMAX),
    ] {
        let f = func(&prog, &interner, name);
        assert_eq!(kernels(f, &interner), vec!["wukong_norm_f32"], "{name}");
        let calls = call_args(f, &interner, "wukong_norm_f32");
        let args = &calls[0];
        // `wukong_norm_f32(x, out, rows, cols, eps, op)`: source and destination are distinct.
        assert_ne!(
            args[0], args[1],
            "{name}: the kernel must read x and write o"
        );
        assert_eq!(const_int(f, args[5]), Some(op as i128), "{name}");
    }
}

/// `s = s + x[i]` after the exp store sums the RAW input, not the exponentials: not a softmax.
#[test]
fn an_out_of_place_window_reading_raw_x_after_the_store_declines() {
    let src = "module m
fn not_softmax(x: [f32; 64], mut o: [f32; 64]) {
    let mut m: f32 = x[0];
    for i in 0..64 { m = fmax(m, x[i]); }
    let mut s: f32 = 0.0;
    for i in 0..64 { o[i] = exp(x[i] - m); s = s + x[i]; }
    for i in 0..64 { o[i] = o[i] / s; }
}
";
    let (prog, _, interner) = lower_src(src);
    assert!(!kernels(func(&prog, &interner, "not_softmax"), &interner)
        .contains(&"wukong_norm_f32".to_string()));
}

// ---- P12.1: hand-tiled GEMM --------------------------------------------------------------------

#[test]
fn a_hand_tiled_gemm_dispatches_the_gemm_kernel() {
    let src = "module m
fn tiled_for(a: [f32; 1024], b: [f32; 1024], mut c: [f32; 1024]) {
    for i0 in 0..32 step 16 {
        for j0 in 0..32 step 16 {
            for i in i0..i0 + 16 {
                for j in j0..j0 + 16 {
                    let mut s: f32 = 0.0;
                    for k in 0..32 { s = s + a[i * 32 + k] * b[k * 32 + j]; }
                    c[i * 32 + j] = s;
                }
            }
        }
    }
}
fn tiled_while(a: [f32; 1024], b: [f32; 1024], mut c: [f32; 1024]) {
    let mut i0: i64 = 0;
    while i0 < 32 {
        let mut j0: i64 = 0;
        while j0 < 32 {
            for i in i0..(i0 + 16) {
                for j in j0..(j0 + 16) {
                    let mut s: f32 = 0.0;
                    for k in 0..32 { s = s + a[i * 32 + k] * b[k * 32 + j]; }
                    c[i * 32 + j] = s;
                }
            }
            j0 = j0 + 16;
        }
        i0 = i0 + 16;
    }
}
";
    let (prog, _, interner) = lower_src(src);
    for name in ["tiled_for", "tiled_while"] {
        assert_eq!(
            kernels(func(&prog, &interner, name), &interner),
            vec!["wukong_sgemm"],
            "{name}"
        );
    }
}

/// Tiles that do not exactly cover the extent visit a different iteration set (`0..30 step 16`
/// with `i in i0..i0+16` runs `i` up to 31); the nest must stay as written.
#[test]
fn tiles_that_do_not_cover_the_extent_exactly_decline() {
    let src = "module m
fn ragged(a: [f32; 1024], b: [f32; 1024], mut c: [f32; 1024]) {
    for i0 in 0..30 step 16 {
        for i in i0..i0 + 16 {
            for j in 0..32 {
                let mut s: f32 = 0.0;
                for k in 0..32 { s = s + a[i * 32 + k] * b[k * 32 + j]; }
                c[i * 32 + j] = s;
            }
        }
    }
}
";
    let (prog, _, interner) = lower_src(src);
    let ks = kernels(func(&prog, &interner, "ragged"), &interner);
    assert!(
        !ks.iter().any(|k| k.starts_with("wukong_sgemm")),
        "a ragged tiling must not become a GEMM, got {ks:?}"
    );
}

// ---- P12.14: per-loop `@generic` and the dispatch report ---------------------------------------

#[test]
fn generic_on_a_loop_opts_out_that_loop_only() {
    let src = "module m
fn two(x: [f32; 64], mut o: [f32; 64], mut p: [f32; 64]) {
    for i in 0..64 { o[i] = exp(x[i]); }
    @generic
    for i in 0..64 { p[i] = exp(x[i]); }
}
fn mm(a: [f32; 16], b: [f32; 16], mut c: [f32; 16]) {
    @generic
    for i in 0..4 { for j in 0..4 {
        let mut s: f32 = 0.0;
        for k in 0..4 { s = s + a[i*4+k] * b[k*4+j]; }
        c[i*4+j] = s;
    } }
}
";
    let (prog, _, interner) = lower_src(src);
    assert_eq!(
        kernels(func(&prog, &interner, "two"), &interner),
        vec!["wukong_vmath_f32"],
        "exactly the un-annotated loop dispatches"
    );
    assert!(
        kernels(func(&prog, &interner, "mm"), &interner).is_empty(),
        "a `@generic` loop is never claimed by a whole-function interception"
    );
}

/// A `@generic` loop nested inside a recognizable nest keeps the WHOLE nest from being replaced (the
/// GEMM recognizer probing `for i` would otherwise swallow the `@generic` k-loop), while a loop that
/// merely encloses one hands recognition back to its body, so the `@generic` loop's sibling still
/// dispatches.
#[test]
fn a_nested_generic_loop_keeps_its_enclosing_nest_and_not_its_siblings() {
    let src = "module m
fn nest(a: [f32; 16], b: [f32; 16], mut c: [f32; 16]) {
    for i in 0..4 { for j in 0..4 {
        let mut s: f32 = 0.0;
        @generic
        for k in 0..4 { s = s + a[i*4+k] * b[k*4+j]; }
        c[i*4+j] = s;
    } }
}
fn siblings(x: [f32; 64], mut o: [f32; 64], mut p: [f32; 64], n: i64) {
    for r in 0..n {
        for c in 0..64 { o[c] = exp(x[c]); }
        @generic
        for c in 0..64 { p[c] = exp(x[c]); }
    }
}
";
    let (prog, _, interner) = lower_src(src);
    let ks = kernels(func(&prog, &interner, "nest"), &interner);
    assert!(
        !ks.iter().any(|k| k.starts_with("wukong_sgemm")),
        "a nest holding a `@generic` loop must not become a GEMM, got {ks:?}"
    );
    assert_eq!(
        kernels(func(&prog, &interner, "siblings"), &interner),
        vec!["wukong_vmath_f32"],
        "the un-annotated sibling inside the enclosing loop still dispatches"
    );
}

/// [`lower_src`] through `lower_program_reporting_dispatch` (`--remarks=dispatch`).
fn lower_src_reporting(src: &str) -> (Program, Vec<Diagnostic>, Interner) {
    let mut interner = Interner::new();
    let (module, pdiags) = wukong_parser::parse_module(src, SourceId(0), &mut interner);
    assert!(pdiags.is_empty(), "parse: {pdiags:?}");
    let (sema, sdiags) = wukong_sema::check(&module, &interner);
    assert!(sdiags.iter().all(|d| !d.is_error()), "sema: {sdiags:?}");
    let (prog, diags) =
        lower_program_reporting_dispatch(&module, &sema, &mut interner, LowerOptions::default());
    assert!(diags.iter().all(|d| !d.is_error()), "lower: {diags:?}");
    (prog, diags, interner)
}

fn dispatch_notes(diags: &[Diagnostic]) -> Vec<&Diagnostic> {
    diags
        .iter()
        .filter(|d| {
            d.severity == Severity::Note && d.message.starts_with("recognized kernel dispatch")
        })
        .collect()
}

#[test]
fn the_dispatch_report_names_each_site_and_changes_no_mir() {
    let src = "module m
fn mm(a: [f32; 16], b: [f32; 16], mut c: [f32; 16]) {
    for i in 0..4 { for j in 0..4 {
        let mut s: f32 = 0.0;
        for k in 0..4 { s = s + a[i*4+k] * b[k*4+j]; }
        c[i*4+j] = s;
    } }
}
fn nested(x: [f32; 64], mut o: [f32; 64], mut t: [f32; 4]) {
    for r in 0..4 {
        t[r] = 1.0;
        for c in 0..64 { o[c] = exp(x[c]); }
    }
}
";
    let (quiet, quiet_diags, qi) = lower_src(src);
    assert!(
        dispatch_notes(&quiet_diags).is_empty(),
        "reporting is opt-in"
    );
    let (loud, diags, li) = lower_src_reporting(src);
    assert_eq!(
        wukong_mir::print::print_program(&quiet, &qi),
        wukong_mir::print::print_program(&loud, &li),
        "the report must not change the MIR"
    );
    let notes = dispatch_notes(&diags);
    assert_eq!(notes.len(), 2, "{notes:?}");
    // The whole-function GEMM interception is reported at the function, by symbol.
    assert!(notes.iter().any(|d| d.message.contains("`wukong_sgemm`")));
    // The activation is reported at the INNER loop, and only there (not again at `for r`).
    let act: Vec<_> = notes
        .iter()
        .filter(|d| d.message.contains("`wukong_vmath_f32`"))
        .collect();
    assert_eq!(act.len(), 1);
    let lo = act[0].primary_span().expect("a located note").lo as usize;
    assert!(
        lo >= src.find("for c in").unwrap(),
        "attributed to the inner loop"
    );
}
