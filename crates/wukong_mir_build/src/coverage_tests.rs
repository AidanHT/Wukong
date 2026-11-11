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
use wukong_mir::{BinOp, CmpOp, Function, Op, Program, ValueId};
use wukong_span::{Interner, SourceId};

use crate::spell::{
    VMATH_ELU_EXPM1, VMATH_GELU_ERF_MUL, VMATH_GELU_TANH, VMATH_GELU_TANH_HF, VMATH_SILU_DIV,
    VMATH_SILU_SX,
};
use crate::{
    lower_program_reporting_dispatch, lower_program_with, LowerOptions, NORM_LOGSOFTMAX,
    NORM_SOFTMAX, VMATH_GELU, VMATH_SIGMOID, VMATH_SILU,
};

fn lower_src(src: &str) -> (Program, Vec<Diagnostic>, Interner) {
    lower_src_with(src, LowerOptions::default())
}

fn lower_src_with(src: &str, opts: LowerOptions) -> (Program, Vec<Diagnostic>, Interner) {
    let mut interner = Interner::new();
    let (module, pdiags) = wukong_parser::parse_module(src, SourceId(0), &mut interner);
    assert!(pdiags.is_empty(), "parse: {pdiags:?}");
    let (sema, sdiags) = wukong_sema::check(&module, &interner);
    assert!(sdiags.iter().all(|d| !d.is_error()), "sema: {sdiags:?}");
    let (prog, diags) = lower_program_with(&module, &sema, &mut interner, opts);
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

/// The kernel-order expansion used to fold into `VMATH_GELU`, on the premise that the source's
/// `tanh` is the kernel's internal `tanh_exp1`. Since the language's `tanh` became the runtime's
/// piecewise kernel on every spelling (`5f423d1e`) while `gelu` keeps `tanh_exp1`, the expansion is
/// a different function in the last bits, and substituting the kernel would change the answer
/// (`tests/run/spelling_value_bindings.wk` pins the inline values lane by lane). So it must not
/// dispatch the GELU kernel — in either spelling.
#[test]
fn the_kernel_order_gelu_expansion_no_longer_dispatches_vmath_gelu() {
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
        for args in call_args(f, &interner, "wukong_vmath_f32") {
            assert_ne!(
                const_int(f, args[3]),
                Some(VMATH_GELU as i128),
                "{name}: the expansion must not be substituted by the GELU kernel"
            );
        }
    }
}

/// The single `wukong_vmath_f32` op code `name` dispatches, if it dispatches exactly one call.
fn vmath_code(prog: &Program, interner: &Interner, name: &str) -> Option<i128> {
    let f = func(prog, interner, name);
    match call_args(f, interner, "wukong_vmath_f32").as_slice() {
        [args] => const_int(f, args[3]),
        _ => None,
    }
}

/// HuggingFace's literal `0.044715 * v * v * v` rounds the cubic term differently from the GELU
/// kernel (`((C1*v)*v)*v` contracted into the FMA vs the kernel's `fma(C1, (v*v)*v, v)`), and the
/// erf form is a different function: neither may be substituted by `VMATH_GELU`. Each dispatches
/// the kernel of ITS OWN expression (P12.10, `spell.rs`) -- and a regrouping no kernel computes
/// (`0.5 * (x * (1 + tanh(..)))`) dispatches nothing.
#[test]
fn gelu_spellings_dispatch_their_own_kernels_never_the_gelu_kernel() {
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
    assert_eq!(
        vmath_code(&prog, &interner, "hf_literal"),
        Some(VMATH_GELU_TANH_HF as i128)
    );
    assert_eq!(
        vmath_code(&prog, &interner, "erf_form"),
        Some(VMATH_GELU_ERF_MUL as i128)
    );
    assert!(
        kernels(func(&prog, &interner, "regrouped"), &interner).is_empty(),
        "regrouped must not dispatch"
    );
}

/// The two operand orders of the gated SiLU product are different functions on a NaN input
/// (`x * sigmoid(x)` returns the input NaN, `sigmoid(x) * x` the sigmoid's sign-flipped one), so
/// only the first is `VMATH_SILU`; the second has its own kernel. Both used to dispatch `VMATH_SILU`.
#[test]
fn the_two_silu_product_orders_dispatch_different_kernels() {
    let src = "module m
fn xs(x: [f32; 64], mut o: [f32; 64]) {
    for i in 0..64 {
        o[i] = x[i] * sigmoid(x[i]);
    }
}
fn sx(x: [f32; 64], mut o: [f32; 64]) {
    for i in 0..64 {
        o[i] = sigmoid(x[i]) * x[i];
    }
}
";
    let (prog, _, interner) = lower_src(src);
    assert_eq!(vmath_code(&prog, &interner, "xs"), Some(VMATH_SILU as i128));
    assert_eq!(
        vmath_code(&prog, &interner, "sx"),
        Some(VMATH_SILU_SX as i128)
    );
}

/// A spelling inside a `@parallel` function dispatches per chunk of the outlined region, and a
/// batched `[R, C]` nest one flat call -- the spelling matcher sits under every `match_vmath_body`
/// caller.
#[test]
fn a_spelling_dispatches_on_the_parallel_and_batched_paths_too() {
    let src = "module m
@parallel
fn par(x: [f32; 4096], mut o: [f32; 4096]) {
    for i in 0..4096 {
        o[i] = x[i] / (1.0 + exp(-x[i]));
    }
}
fn batched(x: [f32; 256], mut o: [f32; 256]) {
    for r in 0..16 {
        for c in 0..16 {
            o[r * 16 + c] = if x[r * 16 + c] > 0.0 { x[r * 16 + c] } else { expm1(x[r * 16 + c]) };
        }
    }
}
";
    let (prog, _, interner) = lower_src(src);
    let mut codes: Vec<Option<i128>> = Vec::new();
    for f in &prog.funcs {
        if !interner.resolve(f.name).starts_with("par") {
            continue;
        }
        for k in ["wukong_vmath_f32", "wukong_vmath_f32_parallel"] {
            for args in call_args(f, &interner, k) {
                codes.push(const_int(f, args[3]));
            }
        }
    }
    assert!(
        !codes.is_empty() && codes.iter().all(|c| *c == Some(VMATH_SILU_DIV as i128)),
        "par: {codes:?}"
    );
    assert_eq!(
        vmath_code(&prog, &interner, "batched"),
        Some(VMATH_ELU_EXPM1 as i128)
    );
}

/// `LowerOptions::spell_kernels` cleared (what `--backend=gpu-native` gets, whose PTX implements
/// only the codes below `VM_SPELL_FIRST`): a spelling that only a spelling-only code computes lowers
/// as written -- no kernel call at all -- while a spelling that IS an existing kernel keeps it. The
/// default dispatches all three, or the clear proves nothing.
#[test]
fn clearing_spell_kernels_keeps_only_the_existing_kernels_a_spelling_is() {
    let src = "module m
fn gelu_spelled(x: [f32; 64], mut o: [f32; 64]) {
    for i in 0..64 {
        o[i] = 0.5 * x[i] * (1.0 + tanh(0.7978845608 * (x[i] + 0.044715 * (x[i] * x[i] * x[i]))));
    }
}
fn swish(x: [f32; 64], mut o: [f32; 64]) {
    for i in 0..64 {
        o[i] = x[i] * sigmoid(x[i]);
    }
}
fn bound(x: [f32; 64], mut o: [f32; 64]) {
    for i in 0..64 {
        let v: f32 = x[i];
        let e: f32 = exp(-v);
        o[i] = 1.0 / (1.0 + e);
    }
}
";
    let (on, _, on_it) = lower_src(src);
    assert_eq!(
        vmath_code(&on, &on_it, "gelu_spelled"),
        Some(VMATH_GELU_TANH as i128)
    );
    assert_eq!(vmath_code(&on, &on_it, "swish"), Some(VMATH_SILU as i128));
    assert_eq!(
        vmath_code(&on, &on_it, "bound"),
        Some(VMATH_SIGMOID as i128)
    );
    let (off, _, off_it) = lower_src_with(
        src,
        LowerOptions {
            spell_kernels: false,
            ..LowerOptions::default()
        },
    );
    assert!(
        kernels(func(&off, &off_it, "gelu_spelled"), &off_it).is_empty(),
        "a spelling-only code must not be emitted when the consumer cannot run it"
    );
    assert_eq!(vmath_code(&off, &off_it, "swish"), Some(VMATH_SILU as i128));
    assert_eq!(
        vmath_code(&off, &off_it, "bound"),
        Some(VMATH_SIGMOID as i128)
    );
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

/// Not only the elementwise maps: the proven side of the `end >= 0` test lowers the loop again with
/// its end known to fit `i64`, so every range-based recognizer a signed counter would get fires —
/// here the low-precision sum.
#[test]
fn a_runtime_usize_bound_keeps_the_low_precision_reduction_kernel() {
    let src = "module m
fn bsum(x: [bf16; 64], n: usize) -> f32 {
    let mut s: f32 = 0.0;
    for k in 0..n {
        s = s + (x[k] as f32);
    }
    return s;
}
";
    let (prog, _, interner) = lower_src(src);
    let f = func(&prog, &interner, "bsum");
    assert_eq!(kernels(f, &interner), vec!["wukong_sum_bf16"]);
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

/// Every op of `f`, in block order.
fn ops(f: &Function) -> Vec<&Op> {
    f.blocks
        .iter()
        .flat_map(|b| &b.insts)
        .map(|i| &i.op)
        .collect()
}

/// `min`/`max` on integers used to be `C0001`; they are a compare in the result type's signedness
/// and a select.
#[test]
fn min_and_max_lower_to_a_compare_and_a_select() {
    let src = "module m
fn f(a: usize, b: usize, c: i32, d: i32) -> usize {
    let lo: usize = min(a, b);
    let hi: i32 = max(c, d);
    return lo + (hi as usize);
}
";
    let (prog, _, interner) = lower_src(src);
    let ops = ops(func(&prog, &interner, "f"));
    assert!(
        ops.iter().any(|o| matches!(o, Op::Cmp(CmpOp::Ult, ..))),
        "usize min compares unsigned"
    );
    assert!(
        ops.iter().any(|o| matches!(o, Op::Cmp(CmpOp::Sgt, ..))),
        "i32 max compares signed"
    );
    assert!(ops.iter().filter(|o| matches!(o, Op::Select(..))).count() >= 2);
}

/// The tilings a blocked GEMM is really written with: counted tiles clamped with `min` over
/// symbolic dims (`examples/matmul.wk`'s structure), stepped tiles clamped over literal extents,
/// full tiles over a runtime extent, and a tiled contraction around a `C`-seeded accumulation.
#[test]
fn ragged_counted_and_runtime_tilings_dispatch_the_gemm_kernel() {
    let src = "module m
fn counted<M, N, K>(a: Tensor[f32, M, K], b: Tensor[f32, K, N], mut c: Tensor[f32, M, N]) {
    const MT: usize = 4;
    for it in 0..((M + MT - 1) / MT) {
        let i0 = it * MT;
        let i1 = min(i0 + MT, M);
        for i in i0..i1 {
            for j in 0..N {
                let mut s: f32 = 0.0;
                for k in 0..K { s = s + a[i, k] * b[k, j]; }
                c[i, j] = s;
            }
        }
    }
}
fn stepped_min(a: [f32; 100], b: [f32; 100], mut c: [f32; 100]) {
    for i0 in 0..10 step 4 {
        for i in i0..min(i0 + 4, 10) {
            for j in 0..10 {
                let mut s: f32 = 0.0;
                for k in 0..10 { s = s + a[i * 10 + k] * b[k * 10 + j]; }
                c[i * 10 + j] = s;
            }
        }
    }
}
fn runtime_rows(a: [f32; 64], b: [f32; 64], mut c: [f32; 64], m: i64) {
    for i0 in 0..m step 4 {
        for i in i0..i0 + 4 {
            for j in 0..8 {
                let mut s: f32 = 0.0;
                for k in 0..8 { s = s + a[i * 8 + k] * b[k * 8 + j]; }
                c[i * 8 + j] = s;
            }
        }
    }
}
fn ktiled_cseed(a: [f32; 64], b: [f32; 64], mut c: [f32; 64]) {
    for k0 in 0..8 step 4 {
        for i in 0..8 {
            for j in 0..8 {
                let mut s: f32 = c[i * 8 + j];
                for k in k0..k0 + 4 { s = s + a[i * 8 + k] * b[k * 8 + j]; }
                c[i * 8 + j] = s;
            }
        }
    }
}
";
    let (prog, _, interner) = lower_src(src);
    for name in ["counted", "stepped_min", "runtime_rows", "ktiled_cseed"] {
        assert_eq!(
            kernels(func(&prog, &interner, name), &interner),
            vec!["wukong_sgemm"],
            "{name}"
        );
    }
    // Symbolic dims: the kernel sits behind the run-time cover test `ceil(M/4)*4 >= M`.
    assert!(ops(func(&prog, &interner, "counted"))
        .iter()
        .any(|o| matches!(o, Op::Cmp(CmpOp::Uge, ..))));
    // A runtime extent under full tiles: behind `m % 4 == 0`.
    assert!(ops(func(&prog, &interner, "runtime_rows"))
        .iter()
        .any(|o| matches!(o, Op::Bin(BinOp::SRem, ..))));
    // Literal extents prove the cover statically: no run-time test.
    assert!(!ops(func(&prog, &interner, "stepped_min"))
        .iter()
        .any(|o| matches!(
            o,
            Op::Bin(BinOp::SRem | BinOp::URem, ..) | Op::Cmp(CmpOp::Uge, ..)
        )));
    // The `C`-seeded accumulation is `C += A·B`: beta = 1.
    let f = func(&prog, &interner, "ktiled_cseed");
    let args = call_args(f, &interner, "wukong_sgemm");
    assert_eq!(const_int(f, args[0][6]), Some(1), "beta");
}

/// A counted tile whose point loop is a full tile names no extent (the untiled nest would run to
/// `NT*T`, which nothing says is the data's extent), and a contraction tiled around a ZERO-seeded
/// scalar overwrites `C` with the last tile's partial sum - not a GEMM. Both stay as written.
#[test]
fn tilings_that_are_not_the_untiled_nest_decline() {
    let src = "module m
fn counted_exact(a: [f32; 64], b: [f32; 64], mut c: [f32; 64]) {
    for it in 0..2 {
        let i0 = it * 4;
        for i in i0..i0 + 4 {
            for j in 0..8 {
                let mut s: f32 = 0.0;
                for k in 0..8 { s = s + a[i * 8 + k] * b[k * 8 + j]; }
                c[i * 8 + j] = s;
            }
        }
    }
}
fn ktiled_zero_seed(a: [f32; 64], b: [f32; 64], mut c: [f32; 64]) {
    for k0 in 0..8 step 4 {
        for i in 0..8 {
            for j in 0..8 {
                let mut s: f32 = 0.0;
                for k in k0..k0 + 4 { s = s + a[i * 8 + k] * b[k * 8 + j]; }
                c[i * 8 + j] = s;
            }
        }
    }
}
";
    let (prog, _, interner) = lower_src(src);
    for name in ["counted_exact", "ktiled_zero_seed"] {
        let ks = kernels(func(&prog, &interner, name), &interner);
        assert!(
            !ks.iter().any(|k| k.starts_with("wukong_sgemm")),
            "{name} must not become a GEMM, got {ks:?}"
        );
    }
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

// ---- P12.4: a multi-statement `@parallel` function selects every row/column arm's `_parallel` twin

/// The vehicle every `tests/run/cov4_par_*.wk` fixture relies on, pinned structurally: a `@parallel`
/// function whose body is more than one statement is not outlined (`parallel_spec` wants exactly one
/// loop), so it is lowered whole with `parallel_fn` set and each recognizer arm in `lower_for`
/// emits its `_parallel` kernel. The same nests in a plain function emit the serial kernels, so the
/// twin choice — not the recognition — is what this pins. If the outliner ever starts taking
/// multi-statement bodies, or an arm hardcodes its serial symbol, this fails by family instead of
/// the corpus silently losing its only witnesses for those `_parallel` entries.
#[test]
fn a_multi_statement_parallel_fn_selects_the_parallel_twin_of_each_arm() {
    let body = "
    for j in 0..40 {
        let mut a: f32 = 0.0;
        for i in 0..3 { a = a + x[i * 40 + j]; }
        cs[j] = a;
    }
    for r in 0..3 {
        let mut acc: f32 = 0.0;
        for i in 0..40 { acc = acc + x[r * 40 + i]; sc[r * 40 + i] = acc; }
    }
    for r in 0..3 {
        let mut bv: f32 = x[r * 40];
        let mut bi: i32 = 0;
        for j in 1..40 { if x[r * 40 + j] > bv { bv = x[r * 40 + j]; bi = j; } }
        am[r] = bi;
    }
    for i in 0..3 {
        for j in 0..40 { tr[j * 3 + i] = x[i * 40 + j]; }
    }
";
    let sig = "(x: [f32; 120], mut cs: [f32; 40], mut sc: [f32; 120], mut am: [i32; 3], \
               mut tr: [f32; 120])";
    let src = format!("module m\n@parallel\nfn par{sig} {{{body}}}\nfn ser{sig} {{{body}}}\n");
    let (prog, _, interner) = lower_src(&src);
    let serial = [
        "wukong_colsum_f32",
        "wukong_cumsum_f32",
        "wukong_rowargmax_i32",
        "wukong_transpose_f32",
    ];
    let par = kernels(func(&prog, &interner, "par"), &interner);
    let ser = kernels(func(&prog, &interner, "ser"), &interner);
    for k in serial {
        assert!(
            par.contains(&format!("{k}_parallel")),
            "@parallel fn lost the `{k}_parallel` twin; got {par:?}"
        );
        assert!(
            !par.iter().any(|c| c == k),
            "@parallel fn emitted the SERIAL `{k}`; got {par:?}"
        );
        assert!(
            ser.iter().any(|c| c == k),
            "the plain twin no longer recognizes `{k}` (the comparison is vacuous); got {ser:?}"
        );
    }
}
