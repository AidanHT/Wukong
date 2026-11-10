//! The natural-spelling census for the elementwise activations (P12.10).
//!
//! Each row is one way real code writes an activation — HuggingFace's `NewGELUActivation`,
//! PyTorch's C++ GELU kernels, llm.c's `gelu_forward`, JAX's `jax.nn.gelu`, the textbook SiLU and
//! sigmoid, the `relu6(x + 3) / 6` hard activations, the select-spelled leaky ReLU and ELU — and the
//! census lowers it inside an elementwise loop over three containers (a fixed array, a `[]f32`
//! slice, a dimension-generic `Tensor[f32, N]`) in two forms (the element read inline at every use,
//! and bound once with `let v = x[i]`), then records which kernel the loop dispatched to.
//!
//! The expected column is what the compiler must do, not what it happens to do: a row expects a
//! kernel only where the kernel computes the spelling's own scalar semantics bit for bit (the
//! `tests/run/spelling_act_*.wk` fixtures check the values, lane by lane, against the same spelling
//! in a loop no recognizer sees), and `None` where the spelling is a different function from every
//! kernel (`pow(x, 3.0)` is `exp(3*log x)`, NaN for `x < 0`; `0.5 * (x * (..))` rounds differently
//! from every GELU kernel). A row whose dispatch moves fails here by name. `census_report` prints
//! the hit rate.
//!
//! Before P12.10 the census read **15 of 246 loops dispatched, 2 of 41 spellings in every form**:
//! `x * sigmoid(x)` and `sigmoid(x) * x` (the latter to `VMATH_SILU`, whose NaN is the other
//! operand's -- a recognition-on/off divergence on a NaN input), plus the inline ReLU select. After
//! it, **231 of 246 loops, 38 of 41 spellings**; the three that do not dispatch are the rows that
//! must not (`pow_cube`, `half_outer`, and the bound ReLU select, which `velem` does not match).

use wukong_mir::{Function, Op};
use wukong_span::{Interner, SourceId};

/// What one spelling must dispatch to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Want {
    /// Nothing: the loop lowers as written (vectorized, or scalar).
    None,
    /// `wukong_vmath_f32` (or its `_parallel` twin) with this op code.
    Vmath(i64),
    /// `wukong_velem_f32` (the streaming affine / ReLU kernel).
    Velem,
}

/// `(family, name, prelude lets, value)`, with `{x}` standing for the loaded element. The prelude
/// lets run before the store, in order. `hoisted` is what the `let v = x[i]` form must dispatch
/// (the same as `want` except where a kernel's own matcher cannot see through the binding).
struct Spelling {
    family: &'static str,
    name: &'static str,
    prelude: &'static str,
    value: &'static str,
    want: Want,
    hoisted: Want,
}

const fn sp(
    family: &'static str,
    name: &'static str,
    prelude: &'static str,
    value: &'static str,
    want: Want,
) -> Spelling {
    Spelling {
        family,
        name,
        prelude,
        value,
        want,
        hoisted: want,
    }
}

// The op codes the census names: `wukong_mir_build`'s own mirrors of the runtime's `VM_*` codes
// (the `tests/run/spelling_act_*.wk` fixtures are what pin those to the runtime's values).
use crate::spell::{
    VMATH_ELU_EXPM1, VMATH_GELU_ERF_CDF, VMATH_GELU_ERF_DIV, VMATH_GELU_ERF_MUL, VMATH_GELU_TANH,
    VMATH_GELU_TANH_CDF, VMATH_GELU_TANH_FAST, VMATH_GELU_TANH_HF, VMATH_GELU_TANH_NOFMA,
    VMATH_HARDSIGMOID_CLAMP, VMATH_HARDSIGMOID_MINMAX, VMATH_HARDSWISH_CLAMP,
    VMATH_HARDSWISH_MINMAX, VMATH_LEAKY_LT, VMATH_MISH_LOG1P, VMATH_MISH_NAIVE,
    VMATH_MISH_SOFTPLUS, VMATH_QUICK_GELU, VMATH_SIGMOID_RATIO, VMATH_SILU_DIV, VMATH_SILU_SX,
    VMATH_SOFTPLUS_LOG1P, VMATH_SOFTPLUS_NAIVE,
};
use crate::{VMATH_ELU, VMATH_LEAKYRELU, VMATH_SIGMOID, VMATH_SILU, VMATH_SOFTPLUS};

/// `Want::Vmath` of a `u32` mirror constant.
const fn vm(code: u32) -> Want {
    Want::Vmath(code as i64)
}

const SPELLINGS: &[Spelling] = &[
    // ---- GELU, tanh approximation ----
    // HuggingFace `NewGELUActivation` / minGPT / GPT-2 (`torch.pow(x, 3.0)` written as `x*x*x`).
    sp(
        "gelu-tanh",
        "hf_new_gelu",
        "",
        "0.5 * {x} * (1.0 + tanh(0.7978845608 * ({x} + 0.044715 * {x} * {x} * {x})))",
        vm(VMATH_GELU_TANH_HF),
    ),
    // PyTorch's C++ CPU kernel: `x_cube = x*x*x; inner = kBeta*(x + kKappa*x_cube)`.
    sp(
        "gelu-tanh",
        "torch_cube_grouped",
        "",
        "0.5 * {x} * (1.0 + tanh(0.7978845608 * ({x} + 0.044715 * ({x} * {x} * {x}))))",
        vm(VMATH_GELU_TANH),
    ),
    // The same with the half on the right (`x * 0.5 * ...`): the constant commutes exactly.
    sp(
        "gelu-tanh",
        "x_half_first",
        "",
        "{x} * 0.5 * (1.0 + tanh(0.7978845608 * ({x} + 0.044715 * {x} * {x} * {x})))",
        vm(VMATH_GELU_TANH_HF),
    ),
    // PyTorch's kernel with its two named intermediates: `C1*x3` is a product OF a binding, so the
    // `+` still contracts -- the same operations as the inline form.
    sp(
        "gelu-tanh",
        "torch_named_inner",
        "let x3: f32 = {x} * {x} * {x}; let inner: f32 = 0.7978845608 * ({x} + 0.044715 * x3);",
        "0.5 * {x} * (1.0 + tanh(inner))",
        vm(VMATH_GELU_TANH),
    ),
    // llm.c `gelu_forward`: `cube = 0.044715f*x*x*x; 0.5f*x*(1.0f + tanhf(S*(x + cube)))` -- a bound
    // product does not contract.
    sp(
        "gelu-tanh",
        "llmc_cube_binding",
        "let cube: f32 = 0.044715 * {x} * {x} * {x};",
        "0.5 * {x} * (1.0 + tanh(0.7978845608 * ({x} + cube)))",
        vm(VMATH_GELU_TANH_NOFMA),
    ),
    // HuggingFace `FastGELUActivation`: `0.5*x*(1 + tanh(x*0.7978845608*(1 + 0.044715*x*x)))`.
    sp(
        "gelu-tanh",
        "hf_fast_gelu",
        "",
        "0.5 * {x} * (1.0 + tanh({x} * 0.7978845608 * (1.0 + 0.044715 * {x} * {x})))",
        vm(VMATH_GELU_TANH_FAST),
    ),
    // JAX `jax.nn.gelu(approximate=True)`: `x * (0.5 * (1 + tanh(...)))` -- the cdf first.
    sp(
        "gelu-tanh",
        "jax_cdf",
        "",
        "{x} * (0.5 * (1.0 + tanh(0.7978845608 * ({x} + 0.044715 * ({x} * {x} * {x})))))",
        vm(VMATH_GELU_TANH_CDF),
    ),
    // `pow(x, 3.0)` literally: `exp(3*log x)`, NaN for every negative `x` -- not GELU at all.
    sp(
        "gelu-tanh",
        "pow_cube",
        "",
        "0.5 * {x} * (1.0 + tanh(0.7978845608 * ({x} + 0.044715 * pow({x}, 3.0))))",
        Want::None,
    ),
    // `0.5 * (x * (1 + tanh(..)))`: the half applied last rounds differently from every kernel.
    sp(
        "gelu-tanh",
        "half_outer",
        "",
        "0.5 * ({x} * (1.0 + tanh(0.7978845608 * ({x} + 0.044715 * ({x} * {x} * {x})))))",
        Want::None,
    ),
    // HuggingFace `QuickGELUActivation`: `x * sigmoid(1.702 * x)`.
    sp(
        "gelu-tanh",
        "hf_quick_gelu",
        "",
        "{x} * sigmoid(1.702 * {x})",
        vm(VMATH_QUICK_GELU),
    ),
    // ---- GELU, exact (erf) ----
    // BERT / HuggingFace `_gelu_python`: `x * 0.5 * (1 + erf(x / sqrt(2)))`.
    sp(
        "gelu-erf",
        "bert_div_sqrt2",
        "",
        "{x} * 0.5 * (1.0 + erf({x} / sqrt(2.0)))",
        vm(VMATH_GELU_ERF_DIV),
    ),
    // PyTorch's C++ CPU kernel: `x * 0.5 * (1 + erf(x * M_SQRT1_2))`.
    sp(
        "gelu-erf",
        "torch_mul_rsqrt2",
        "",
        "{x} * 0.5 * (1.0 + erf({x} * 0.7071067811865476))",
        vm(VMATH_GELU_ERF_MUL),
    ),
    // The reciprocal root computed in place: `1.0 / sqrt(2.0)` is the same `f32`.
    sp(
        "gelu-erf",
        "mul_reciprocal_expr",
        "",
        "{x} * 0.5 * (1.0 + erf({x} * (1.0 / sqrt(2.0))))",
        vm(VMATH_GELU_ERF_MUL),
    ),
    // The half first, the root as a literal.
    sp(
        "gelu-erf",
        "half_first_div",
        "",
        "0.5 * {x} * (1.0 + erf({x} / 1.4142135623730951))",
        vm(VMATH_GELU_ERF_DIV),
    ),
    // JAX `jax.nn.gelu(approximate=False)`: `x * (erf(x / sqrt(2)) + 1) / 2`.
    sp(
        "gelu-erf",
        "jax_exact",
        "",
        "{x} * (erf({x} / 1.4142135623730951) + 1.0) / 2.0",
        vm(VMATH_GELU_ERF_CDF),
    ),
    // ---- SiLU / swish ----
    sp(
        "silu",
        "div_one_plus_exp",
        "",
        "{x} / (1.0 + exp(-{x}))",
        vm(VMATH_SILU_DIV),
    ),
    sp(
        "silu",
        "x_times_sigmoid",
        "",
        "{x} * sigmoid({x})",
        vm(VMATH_SILU),
    ),
    // The other operand order: its NaN is the sigmoid's, sign flipped -- not `VMATH_SILU`'s.
    sp(
        "silu",
        "sigmoid_times_x",
        "",
        "sigmoid({x}) * {x}",
        vm(VMATH_SILU_SX),
    ),
    // `sigmoid` written out is the very operations `sigmoid()` lowers to.
    sp(
        "silu",
        "x_times_reciprocal",
        "",
        "{x} * (1.0 / (1.0 + exp(-{x})))",
        vm(VMATH_SILU),
    ),
    // ---- sigmoid ----
    // `1 / (1 + exp(-x))` IS the language's (and the kernel's) `sigmoid`.
    sp(
        "sigmoid",
        "reciprocal",
        "",
        "1.0 / (1.0 + exp(-{x}))",
        vm(VMATH_SIGMOID),
    ),
    sp(
        "sigmoid",
        "reciprocal_exp_first",
        "",
        "1.0 / (exp(-{x}) + 1.0)",
        vm(VMATH_SIGMOID),
    ),
    sp(
        "sigmoid",
        "exp_ratio",
        "",
        "exp({x}) / (1.0 + exp({x}))",
        vm(VMATH_SIGMOID_RATIO),
    ),
    // ---- softplus ----
    sp(
        "softplus",
        "log_one_plus_exp",
        "",
        "log(1.0 + exp({x}))",
        vm(VMATH_SOFTPLUS_NAIVE),
    ),
    sp(
        "softplus",
        "log1p_exp",
        "",
        "log1p(exp({x}))",
        vm(VMATH_SOFTPLUS_LOG1P),
    ),
    // The stable form IS the language's (and the kernel's) `softplus`.
    sp(
        "softplus",
        "stable_max_abs",
        "",
        "max({x}, 0.0) + log(1.0 + exp(-abs({x})))",
        vm(VMATH_SOFTPLUS),
    ),
    // ---- mish ----
    sp(
        "mish",
        "tanh_log_one_plus_exp",
        "",
        "{x} * tanh(log(1.0 + exp({x})))",
        vm(VMATH_MISH_NAIVE),
    ),
    sp(
        "mish",
        "tanh_softplus",
        "",
        "{x} * tanh(softplus({x}))",
        vm(VMATH_MISH_SOFTPLUS),
    ),
    // The stable softplus written out inside: the same operations as `softplus()`.
    sp(
        "mish",
        "tanh_stable_softplus",
        "",
        "{x} * tanh(max({x}, 0.0) + log(1.0 + exp(-abs({x}))))",
        vm(VMATH_MISH_SOFTPLUS),
    ),
    sp(
        "mish",
        "tanh_log1p_exp",
        "",
        "{x} * tanh(log1p(exp({x})))",
        vm(VMATH_MISH_LOG1P),
    ),
    // ---- hardsigmoid: `relu6(x + 3) / 6` ----
    sp(
        "hardsigmoid",
        "min_max_div6",
        "",
        "min(max({x} + 3.0, 0.0), 6.0) / 6.0",
        vm(VMATH_HARDSIGMOID_MINMAX),
    ),
    sp(
        "hardsigmoid",
        "clamp_div6",
        "",
        "clamp({x} + 3.0, 0.0, 6.0) / 6.0",
        vm(VMATH_HARDSIGMOID_CLAMP),
    ),
    sp(
        "hardsigmoid",
        "fmin_fmax_div6",
        "",
        "fmin(fmax({x} + 3.0, 0.0), 6.0) / 6.0",
        vm(VMATH_HARDSIGMOID_MINMAX),
    ),
    // ---- hardswish: `x * relu6(x + 3) / 6` ----
    sp(
        "hardswish",
        "min_max_div6",
        "",
        "{x} * min(max({x} + 3.0, 0.0), 6.0) / 6.0",
        vm(VMATH_HARDSWISH_MINMAX),
    ),
    sp(
        "hardswish",
        "clamp_div6",
        "",
        "{x} * clamp({x} + 3.0, 0.0, 6.0) / 6.0",
        vm(VMATH_HARDSWISH_CLAMP),
    ),
    // ---- leaky ReLU (slope 0.01) ----
    sp(
        "leaky-relu",
        "select_gt",
        "",
        "if {x} > 0.0 { {x} } else { 0.01 * {x} }",
        vm(VMATH_LEAKYRELU),
    ),
    sp(
        "leaky-relu",
        "select_ge",
        "",
        "if {x} >= 0.0 { {x} } else { {x} * 0.01 }",
        vm(VMATH_LEAKYRELU),
    ),
    sp(
        "leaky-relu",
        "select_lt",
        "",
        "if {x} < 0.0 { 0.01 * {x} } else { {x} }",
        vm(VMATH_LEAKY_LT),
    ),
    sp(
        "leaky-relu",
        "max_scaled",
        "",
        "max({x}, 0.01 * {x})",
        vm(VMATH_LEAKYRELU),
    ),
    // ---- ELU (alpha 1) ----
    sp(
        "elu",
        "select_exp_minus_one",
        "",
        "if {x} > 0.0 { {x} } else { exp({x}) - 1.0 }",
        vm(VMATH_ELU),
    ),
    sp(
        "elu",
        "select_expm1",
        "",
        "if {x} > 0.0 { {x} } else { expm1({x}) }",
        vm(VMATH_ELU_EXPM1),
    ),
    // ---- ReLU (control: dispatched to `wukong_velem_f32` before P12.10) ----
    // Bound first (`let v = x[i]`) it still lowers as written: `canon::inline_value_lets` does not
    // see through an `if`, and the select is velem's shape, not a `wukong_vmath_f32` spelling.
    Spelling {
        family: "relu",
        name: "select",
        prelude: "",
        value: "if {x} > 0.0 { {x} } else { 0.0 }",
        want: Want::Velem,
        hoisted: Want::None,
    },
];

/// The three containers each spelling is lowered over.
const CONTAINERS: &[(&str, &str, &str)] = &[
    ("array", "x: [f32; 64], mut o: [f32; 64]", "64"),
    ("slice", "x: []f32, mut o: []f32, n: i64", "n"),
    ("tensor", "x: Tensor[f32, N], mut o: Tensor[f32, N]", "N"),
];

/// One loop body: the element read inline (`hoisted == false`) or bound once to `v`.
fn body(s: &Spelling, hoisted: bool) -> String {
    let xv = if hoisted { "v" } else { "x[i]" };
    let mut b = String::new();
    if hoisted {
        b.push_str("let v: f32 = x[i]; ");
    }
    b.push_str(&s.prelude.replace("{x}", xv));
    b.push_str(&format!(" o[i] = {};", s.value.replace("{x}", xv)));
    b
}

/// The source of one census function.
fn function(name: &str, container: &(&str, &str, &str), body: &str) -> String {
    let (kind, params, end) = container;
    let generics = if *kind == "tensor" { "<N>" } else { "" };
    format!("fn {name}{generics}({params}) {{\n    for i in 0..{end} {{ {body} }}\n}}\n")
}

/// What `f` dispatched: the vmath op code, `velem`, another kernel's name, or nothing.
fn dispatched(f: &Function, interner: &Interner) -> (Want, Vec<String>) {
    let mut want = Want::None;
    let mut others = Vec::new();
    for b in &f.blocks {
        for i in &b.insts {
            let Op::Call { func, args } = &i.op else {
                continue;
            };
            let name = interner.resolve(*func);
            if !name.starts_with("wukong_") || name == "wukong_parallel_for" {
                continue;
            }
            match name {
                "wukong_vmath_f32" | "wukong_vmath_f32_parallel" => {
                    let op = f
                        .blocks
                        .iter()
                        .flat_map(|b| &b.insts)
                        .find_map(|d| match d.op {
                            Op::ConstInt(k, _) if d.result == Some(args[3]) => Some(k as i64),
                            _ => None,
                        })
                        .expect("vmath op code is a constant");
                    want = Want::Vmath(op);
                }
                "wukong_velem_f32" | "wukong_velem_f32_parallel" => want = Want::Velem,
                "wukong_rt_bounds_fail" | "wukong_rt_check_fail" => {}
                other => others.push(other.to_string()),
            }
        }
    }
    (want, others)
}

/// Lower every spelling in every container and form; return `(row label, want, got, others)`.
fn run_census() -> Vec<(String, Want, Want, Vec<String>)> {
    let mut src = String::from("module m\n");
    let mut rows = Vec::new();
    for (si, s) in SPELLINGS.iter().enumerate() {
        for (ci, c) in CONTAINERS.iter().enumerate() {
            for hoisted in [false, true] {
                let fname = format!("s{si}_c{ci}_h{}", hoisted as u8);
                src.push_str(&function(&fname, c, &body(s, hoisted)));
                let label = format!(
                    "{}/{} over {} ({})",
                    s.family,
                    s.name,
                    c.0,
                    if hoisted {
                        "let v = x[i]"
                    } else {
                        "x[i] inline"
                    }
                );
                rows.push((fname, label, if hoisted { s.hoisted } else { s.want }));
            }
        }
    }
    let mut interner = Interner::new();
    let (module, pdiags) = wukong_parser::parse_module(&src, SourceId(0), &mut interner);
    assert!(pdiags.is_empty(), "parse: {pdiags:?}\n{src}");
    let (sema, sdiags) = wukong_sema::check(&module, &interner);
    assert!(
        sdiags.iter().all(|d| !d.is_error()),
        "sema: {sdiags:?}\n{src}"
    );
    let (prog, diags) = crate::lower_program(&module, &sema, &mut interner);
    assert!(diags.iter().all(|d| !d.is_error()), "lower: {diags:?}");
    for f in &prog.funcs {
        let errs = wukong_mir::verify::verify_function(f);
        assert!(errs.is_empty(), "MIR verify: {errs:?}");
    }
    rows.into_iter()
        .map(|(fname, label, want)| {
            let f = prog
                .funcs
                .iter()
                .find(|f| interner.resolve(f.name) == fname)
                .unwrap_or_else(|| panic!("`{fname}` was not lowered"));
            let (got, others) = dispatched(f, &interner);
            (label, want, got, others)
        })
        .collect()
}

/// Every natural spelling dispatches exactly where its row says, in every container and form.
#[test]
fn every_natural_activation_spelling_dispatches_where_it_must() {
    let rows = run_census();
    let mut wrong = Vec::new();
    for (label, want, got, others) in &rows {
        if want != got || !others.is_empty() {
            wrong.push(format!(
                "{label}: wanted {want:?}, dispatched {got:?} (other kernels: {others:?})"
            ));
        }
    }
    assert!(
        wrong.is_empty(),
        "{} of {} census rows moved:\n{}",
        wrong.len(),
        rows.len(),
        wrong.join("\n")
    );
}

/// The census as a table, with the hit rate: loops that reach a kernel, and spellings whose every
/// loop (each container, each form) does. Run with `--nocapture` to read it; it asserts nothing the
/// test above does not.
#[test]
fn census_report() {
    let rows = run_census();
    let hits = rows
        .iter()
        .filter(|(_, _, got, _)| *got != Want::None)
        .count();
    let per = CONTAINERS.len() * 2;
    let spelling_hits = rows
        .chunks(per)
        .filter(|c| c.iter().all(|(_, _, got, _)| *got != Want::None))
        .count();
    for (label, _, got, _) in &rows {
        eprintln!("{got:?}\t{label}");
    }
    eprintln!(
        "natural-spelling census: {hits}/{} loops dispatched; {spelling_hits}/{} spellings in every form",
        rows.len(),
        SPELLINGS.len()
    );
}
