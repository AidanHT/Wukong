//! `tests/compare/` — timed (D25; master plan §7.1).
//!
//! The comparison corpus at `tests/compare/` ships committed C and CUDA peers that had never met a
//! clock: `run_cpu.ps1` prints program output only, and the master plan's §7.1 pass condition for
//! each CPU fixture — "CPU within the §3.3 band" — had never been evaluated by any run. This mode
//! is that timing surface.
//!
//! For each CPU fixture it times a TIMING-SCALE TWIN of the committed loop — the same algorithm,
//! the same spelling, the same `@generic` marking, with the toy extent (8 elements / 2x2x3)
//! replaced by a timing-sized one — against the committed C peer's kernel exported as `kbench`, on
//! the round-interleaved, pinned, twin-controlled protocol the general suite established. The
//! committed fixtures stay the evidence for line counts, shape safety and every-lane printing; the
//! twins are transcriptions, not includes, because the committed `.wk` is a whole program printing
//! its lanes and the committed `.c` keeps its kernel `static` behind a `main`, so neither can be
//! loaded as a library. Each twin cites its fixture in a header comment; keeping them in step is a
//! review obligation on any edit to either side.
//!
//! Recognition-off is load-bearing: every Wukong twin carries the fixtures' own `@generic`, the
//! census printed per row is the T2 witness that it held (`NOTHING` dispatched), and a row whose
//! census shows a `wukong_*` kernel is INADMISSIBLE under §3B1 and says so instead of printing a
//! ratio. Columns are Wukong, C (the committed peer's flags: honest `-O3 -march=native
//! -ffp-contract=fast`), and the `C(twin)` control; every column is checked per-lane against an
//! f64 scalar reference before any ratio is quoted.
//!
//! Before any of that, the mode runs [`premise_check`]: the recognition on/off premise at a
//! measurement size, over `tests/run/premise_*.wk` (P12.24). A failure there makes `compare` exit
//! with the numerics gate's code and `sota-check` fail.

use std::path::Path;

use crate::general::{
    compile_block, ix_of, labels_of, max_rel_vs_ref, peer_col, print_instrument, print_round_trace,
    time_round, twin_ratio, vs_c_cell, Col, Keep, Rounds, DIVIDE_WARNING, HONEST_FLAGS,
};
use crate::SotaRow;

/// The shared 3-pointer ABI every fixture twin uses (`x`/`a`, `y`/`b`, `out`/`c`; the stencil
/// leaves the middle slot unused, exactly like the battery's unary rows).
type CmpFn = unsafe extern "C" fn(*const f32, *const f32, *mut f32);

/// saxpy / stencil element count. 3 x 16 MiB of f32 — past any L3 on this machine, so the loops
/// are measured memory-bound, the regime the master plan §3.3 band is stated for.
const ELEM_N: usize = 1 << 22;
/// naive_gemm square extent: compute-bound at 2·256³ = 33.5 MFLOP per call, L2-resident operands —
/// the naive-nest regime the fixture exists to compare (the *recognized* GEMM is the battery's
/// `matmul` section, deliberately not this).
const GEMM_NS: usize = 256;

struct Fixture {
    /// FLOOR.md row name, `compare/<dir>`.
    program: &'static str,
    /// The committed fixture the twins transcribe.
    fixture: &'static str,
    wk: String,
    c: String,
    /// Element counts of the three buffers (x, y, out).
    dims: (usize, usize, usize),
    /// The f64 scalar reference for the whole output buffer, from the same input bits.
    reference: fn(&[f32], &[f32]) -> Vec<f64>,
}

fn wk_saxpy() -> String {
    // Twin of tests/compare/saxpy/saxpy.wk: out[i] = a*x[i] + y[i], a = 2.0, @generic.
    format!(
        "module bench\n\
         // Timing twin of tests/compare/saxpy/saxpy.wk — same loop, same spelling, N={ELEM_N}.\n\
         @generic\n\
         fn kbench(x: [f32; {ELEM_N}], y: [f32; {ELEM_N}], mut out: [f32; {ELEM_N}]) {{\n\
         \x20   let a: f32 = 2.0;\n\
         \x20   for i in 0..{ELEM_N} {{\n\
         \x20       out[i] = a * x[i] + y[i];\n\
         \x20   }}\n\
         }}\n"
    )
}

fn c_saxpy() -> String {
    // Twin of tests/compare/saxpy/saxpy.c's `saxpy` kernel, exported as `kbench`.
    format!(
        "#include <stddef.h>\n#define N {ELEM_N}\n\
         __declspec(dllexport) void kbench(const float *__restrict__ x, const float *__restrict__ y,\n\
         \x20                                float *__restrict__ out) {{\n\
         \x20   const float a = 2.0f;\n\
         \x20   for (size_t i = 0; i < N; ++i) {{\n\
         \x20       out[i] = a * x[i] + y[i];\n\
         \x20   }}\n\
         }}\n"
    )
}

fn ref_saxpy(x: &[f32], y: &[f32]) -> Vec<f64> {
    x.iter()
        .zip(y)
        .map(|(&x, &y)| 2.0f64 * x as f64 + y as f64)
        .collect()
}

fn wk_stencil() -> String {
    // Twin of tests/compare/unknown_op/unknown_op.wk: the three-point stencil no recognizer names.
    let last = ELEM_N - 1;
    format!(
        "module bench\n\
         // Timing twin of tests/compare/unknown_op/unknown_op.wk — same stencil, N={ELEM_N}.\n\
         @generic\n\
         fn kbench(x: [f32; {ELEM_N}], y: [f32; {ELEM_N}], mut out: [f32; {ELEM_N}]) {{\n\
         \x20   out[0] = x[0];\n\
         \x20   out[{last}] = x[{last}];\n\
         \x20   for i in 1..{last} {{\n\
         \x20       out[i] = x[i - 1] + 2.0 * x[i] + x[i + 1];\n\
         \x20   }}\n\
         }}\n"
    )
}

fn c_stencil() -> String {
    // Twin of tests/compare/unknown_op/unknown_op.c's `stencil` kernel, exported as `kbench`.
    format!(
        "#include <stddef.h>\n#define N {ELEM_N}\n\
         __declspec(dllexport) void kbench(const float *__restrict__ x, const float *__restrict__ y,\n\
         \x20                                float *__restrict__ out) {{\n\
         \x20   out[0] = x[0];\n\
         \x20   for (size_t i = 1; i + 1 < N; ++i) {{\n\
         \x20       out[i] = x[i - 1] + 2.0f * x[i] + x[i + 1];\n\
         \x20   }}\n\
         \x20   out[N - 1] = x[N - 1];\n\
         }}\n"
    )
}

fn ref_stencil(x: &[f32], _y: &[f32]) -> Vec<f64> {
    let n = x.len();
    let mut out = vec![0.0f64; n];
    out[0] = x[0] as f64;
    out[n - 1] = x[n - 1] as f64;
    for i in 1..n - 1 {
        out[i] = x[i - 1] as f64 + 2.0 * x[i] as f64 + x[i + 1] as f64;
    }
    out
}

fn wk_gemm() -> String {
    // Twin of tests/compare/naive_gemm/naive_gemm.wk: the same ikj order, hoisted a[i*K+k], square
    // at NS so the peer's natural loop order is identical.
    let nn = GEMM_NS * GEMM_NS;
    format!(
        "module bench\n\
         // Timing twin of tests/compare/naive_gemm/naive_gemm.wk — same ikj nest, {GEMM_NS}^3.\n\
         @generic\n\
         fn kbench(a: [f32; {nn}], b: [f32; {nn}], mut c: [f32; {nn}]) {{\n\
         \x20   for i in 0..{GEMM_NS} {{\n\
         \x20       for j0 in 0..{GEMM_NS} {{\n\
         \x20           c[i * {GEMM_NS} + j0] = 0.0;\n\
         \x20       }}\n\
         \x20       for k in 0..{GEMM_NS} {{\n\
         \x20           let aik: f32 = a[i * {GEMM_NS} + k];\n\
         \x20           for j in 0..{GEMM_NS} {{\n\
         \x20               c[i * {GEMM_NS} + j] = c[i * {GEMM_NS} + j] + aik * b[k * {GEMM_NS} + j];\n\
         \x20           }}\n\
         \x20       }}\n\
         \x20   }}\n\
         }}\n"
    )
}

fn c_gemm() -> String {
    // Twin of tests/compare/naive_gemm/naive_gemm.c's `gemm` kernel at m=n=k=NS, exported as
    // `kbench`.
    format!(
        "#include <stddef.h>\n#define NS {GEMM_NS}\n\
         __declspec(dllexport) void kbench(const float *__restrict__ a, const float *__restrict__ b,\n\
         \x20                                float *__restrict__ c) {{\n\
         \x20   for (size_t i = 0; i < NS; ++i) {{\n\
         \x20       for (size_t j = 0; j < NS; ++j) c[i * NS + j] = 0.0f;\n\
         \x20       for (size_t p = 0; p < NS; ++p) {{\n\
         \x20           const float aip = a[i * NS + p];\n\
         \x20           for (size_t j = 0; j < NS; ++j) {{\n\
         \x20               c[i * NS + j] += aip * b[p * NS + j];\n\
         \x20           }}\n\
         \x20       }}\n\
         \x20   }}\n\
         }}\n"
    )
}

fn ref_gemm(a: &[f32], b: &[f32]) -> Vec<f64> {
    let ns = GEMM_NS;
    let mut c = vec![0.0f64; ns * ns];
    for i in 0..ns {
        for p in 0..ns {
            let aip = a[i * ns + p] as f64;
            for j in 0..ns {
                c[i * ns + j] += aip * b[p * ns + j] as f64;
            }
        }
    }
    c
}

fn fixtures() -> Vec<Fixture> {
    vec![
        Fixture {
            program: "compare/saxpy",
            fixture: "tests/compare/saxpy",
            wk: wk_saxpy(),
            c: c_saxpy(),
            dims: (ELEM_N, ELEM_N, ELEM_N),
            reference: ref_saxpy,
        },
        Fixture {
            program: "compare/naive_gemm",
            fixture: "tests/compare/naive_gemm",
            wk: wk_gemm(),
            c: c_gemm(),
            dims: (GEMM_NS * GEMM_NS, GEMM_NS * GEMM_NS, GEMM_NS * GEMM_NS),
            reference: ref_gemm,
        },
        Fixture {
            program: "compare/unknown_op",
            fixture: "tests/compare/unknown_op",
            wk: wk_stencil(),
            c: c_stencil(),
            dims: (ELEM_N, ELEM_N, ELEM_N),
            reference: ref_stencil,
        },
    ]
}

/// Per-lane agreement bar against the f64 reference. The elementwise fixtures are a couple of
/// rounding steps from exact; the gemm accumulates a 256-term f32 dot per lane (c·√K·ε territory),
/// and gcc's `-ffp-contract=fast` FMA vs Wukong's own contraction round differently. 1e-3 is the
/// suite's standard relaxed-reassociation bar and is ~2 orders looser than either path measures.
const ORACLE_BAR: f64 = 1e-3;

/// Time every fixture; print the tables; return one [`SotaRow`] per fixture for `sota-check`.
pub(crate) fn bench_compare(cc: &str, dir: &Path) -> Vec<SotaRow> {
    println!("================================================================================");
    println!("tests/compare/ — TIMED (the master plan 7.1 CPU band, recognition-off)");
    println!("================================================================================");
    println!(
        "Timing-scale twins of the committed fixtures (same loop, same spelling, same @generic),\n\
         vs the committed C peer's kernel exported as kbench. Columns: Wukong | C | C(twin).\n\
         The committed fixtures remain the line-count / shape-safety / every-lane evidence;\n\
         this section owns their timing claim. {}",
        crate::model::power_status_line()
    );

    let mut rows = Vec::new();
    for fx in fixtures() {
        println!();
        rows.push(bench_fixture(&fx, cc, dir));
    }
    rows
}

fn bench_fixture(fx: &Fixture, cc: &str, dir: &Path) -> SotaRow {
    println!("--- {} (twin of {}) ---", fx.program, fx.fixture);

    // Deterministic inputs, moderate magnitudes (the battery's patterns).
    let (nx, ny, nout) = fx.dims;
    let x: Vec<f32> = (0..nx).map(|i| (i as f32 % 17.0) * 0.5 + 1.0).collect();
    let y: Vec<f32> = (0..ny).map(|i| (i as f32 % 13.0) * 0.25 - 0.5).collect();
    let mut out: Vec<f32> = vec![0.0; nout];
    let reference = (fx.reference)(&x, &y);

    // ---- build every column first, time nothing yet (the round protocol's precondition) ----
    let stem = fx.program.replace('/', "_");
    let mut cols: Vec<Col<CmpFn>> = Vec::new();
    // Recognizers stay ENABLED here on purpose: these twins carry `@generic` in the source, which is
    // the per-function opt-out, and the census check below is what proves it held. Passing `false`
    // would reach the same lowering by a different road and quietly make both the attribute and that
    // check vacuous — a fixture that lost its `@generic` would still read `NOTHING` and pass.
    match compile_block(&fx.wk, &stem, 3, true) {
        Some(blk) => {
            // Dump the exact compiled source so `wukongc --emit=mir -O2 <file>` reproduces the
            // census.
            let _ = std::fs::write(dir.join(format!("{stem}.wk")), &fx.wk);
            let census = crate::general::census_line(&blk.census);
            // SAFETY: `compile_block` arity-checked `kbench` as 3 pointers -> void = `CmpFn`; the
            // handle moves into `Keep` and outlives every call.
            let f: CmpFn = unsafe { std::mem::transmute(blk.ptr) };
            cols.push(Col {
                label: "Wukong".into(),
                note: "the fixture's loop, @generic".into(),
                census,
                compile: blk.compile,
                f,
                _keep: Keep::Jit(blk.handle),
            });
        }
        None => println!("  ! Wukong twin did not compile — row dropped"),
    }
    for (label, name) in [("C", stem.clone()), ("C(twin)", format!("{stem}_twin"))] {
        // SAFETY: the C twin declares `kbench` with the same 3-pointer ABI as `CmpFn`.
        let built = unsafe { peer_col::<CmpFn>("c", &fx.c, dir, &name, cc, &HONEST_FLAGS) };
        match built {
            Some((f, lib, compile)) => cols.push(Col {
                label: label.into(),
                note: if label == "C(twin)" {
                    "CONTROL: identical source+compiler+flags as C".into()
                } else {
                    "the committed peer's kernel, honest flags".into()
                },
                census: "-".into(),
                compile,
                f,
                _keep: Keep::Dll(lib),
            }),
            None => println!("  ! {label} peer unavailable — column dropped"),
        }
    }

    let labels = labels_of(&cols);
    let wk_ix = ix_of(&labels, "Wukong");
    let c_ix = ix_of(&labels, "C");

    // ---- rounds ----
    let (xp, yp): (*const f32, *const f32) = (x.as_ptr(), y.as_ptr());
    let mut outs: Vec<Vec<f32>> = vec![Vec::new(); cols.len()];
    let rr = Rounds::run(cols.len(), crate::gen_rounds(), |i, r| {
        let f = cols[i].f;
        out.iter_mut().for_each(|v| *v = 0.0);
        let op = out.as_mut_ptr(); // the ONE live pointer to `out` while a kernel runs
        let ns = time_round(|| unsafe { f(xp, yp, op) });
        if r == 0 {
            outs[i] = out.clone();
        }
        ns
    });

    // ---- report ----
    let reportable = rr.reportable();
    let twin = twin_ratio(&labels, &rr);
    let floor = twin.map(crate::floor_factor);
    println!();
    println!(
        "  {:<10} {:>12} {:<30}  note",
        "column", "ms (best)", "vs C (median [min-max])"
    );
    for (i, col) in cols.iter().enumerate() {
        println!(
            "  {:<10} {:>12.3} {:<30}  {}",
            col.label,
            rr.best(i) / 1e6,
            vs_c_cell(i, c_ix, &rr, floor, reportable),
            col.note,
        );
    }
    println!("{DIVIDE_WARNING}");
    println!();
    print_instrument(&rr, twin);
    println!();
    print_round_trace(&labels, &rr);

    // ---- the T2 witness and the oracle ----
    let census = cols
        .iter()
        .find(|c| c.label == "Wukong")
        .map(|c| c.census.clone())
        .unwrap_or_else(|| crate::sota::CENSUS_DID_NOT_COMPILE.into());
    println!();
    println!("  census (Wukong twin, must be NOTHING for the row to be admissible): {census}");
    let mut oracle_ok = true;
    println!(
        "  CORRECTNESS vs the f64 scalar reference (worst of all {} lanes; bar {ORACLE_BAR:.0e})",
        reference.len()
    );
    for (i, col) in cols.iter().enumerate() {
        let (dev, at) = max_rel_vs_ref(&outs[i], &reference);
        let ok = dev <= ORACLE_BAR;
        if !ok {
            oracle_ok = false;
        }
        println!(
            "    {:<10} max rel dev {dev:.3e} at lane {at}{}",
            col.label,
            if ok {
                ""
            } else {
                "  !!! OVER BAR — column wrong, ratios void"
            }
        );
    }
    let admissible = census.starts_with("NOTHING");
    if !admissible {
        println!(
            "  !!! the Wukong twin DISPATCHED a kernel — @generic did not hold. The row is\n\
             \x20   INADMISSIBLE under the recognition-off bar (campaign 3B1) and its ratio is void."
        );
    }

    SotaRow {
        program: fx.program.to_string(),
        peer: "C".into(),
        ratio: match (wk_ix, c_ix) {
            (Some(w), Some(c)) if admissible && oracle_ok => rr.ratio(c, w),
            _ => None,
        },
        twin,
        reportable,
        power: rr.state_key(),
        census,
        peer_missing: c_ix.is_none(),
        oracle_ok,
    }
}

// =================================================================================================
// The recognition on/off PREMISE check at measurement size (P12.24)
// =================================================================================================
//
// Everything this harness reads off a recognizer-on column rests on one premise: recognition changes
// HOW a program is compiled, never WHAT it computes beyond a declared tolerance. The only gate on that
// premise (`crates/wukongc/tests/recognize.rs`) runs fixtures of N <= 16 that print exact integers,
// where a reassociated reduction cannot differ from a sequential one - so it says nothing about the
// sizes this harness times. The three `tests/run/premise_*.wk` fixtures run a 4096-term reduction, a
// 768-wide RMSNorm row and a 64x64x64 GEMM over fractional data and print every lane at full
// precision against exact closed forms with a cited `c*sqrt(K)*eps` bound. This check compiles each
// one twice - recognizers on and off - runs both on the interpreter (the oracle) and natively, and
// requires: every run inside every one of the fixture's own `EXPECT-OUT-APPROX` lines; interpreter
// and native bit-identical within each lowering; and on vs off within `2t/(1-t)` lane by lane (two
// values each within `t` of one reference). It prints how far on and off ACTUALLY moved, which is
// the number a timed recognizer-on vs `wk(gen)` comparison has to be read against.

/// The premise fixtures, by file name under `tests/run/`.
pub(crate) const PREMISE_FIXTURES: [&str; 3] = [
    "premise_reduce_4096.wk",
    "premise_norm_wide.wk",
    "premise_gemm_64.wk",
];

/// `tests/run/` in the source tree this binary was built from.
fn premise_dir() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/run")
}

/// The fixture's expected stdout lines as `(value, relative tolerance)`: an `EXPECT-OUT-APPROX:
/// <value> <rel>` line as written, an exact `EXPECT-OUT:` number as tolerance 0. Anything else that
/// constrains stdout is refused - this check compares numbers only.
pub(crate) fn premise_expectations(src: &str) -> Result<Vec<(f64, f64)>, String> {
    let mut out = Vec::new();
    for (n, raw) in src.lines().enumerate() {
        let Some(body) = raw.trim_start().strip_prefix("//") else {
            continue;
        };
        let body = body.trim_start();
        if let Some(rest) = body.strip_prefix("EXPECT-OUT-APPROX:") {
            let f: Vec<&str> = rest.split_whitespace().collect();
            let [v, t] = f[..] else {
                return Err(format!(
                    "line {}: EXPECT-OUT-APPROX needs `<value> <rel>`",
                    n + 1
                ));
            };
            let v: f64 = v.parse().map_err(|e| format!("line {}: {e}", n + 1))?;
            let t: f64 = t.parse().map_err(|e| format!("line {}: {e}", n + 1))?;
            if !(t > 0.0 && t < 1.0) {
                return Err(format!("line {}: tolerance must be in (0, 1)", n + 1));
            }
            out.push((v, t));
        } else if let Some(rest) = body.strip_prefix("EXPECT-OUT:") {
            let v: f64 = rest
                .trim()
                .parse()
                .map_err(|e| format!("line {}: non-numeric EXPECT-OUT ({e})", n + 1))?;
            out.push((v, 0.0));
        } else if body.starts_with("EXPECT-OUT") {
            return Err(format!("line {}: unsupported directive {body:?}", n + 1));
        }
    }
    if out.is_empty() {
        return Err("no EXPECT-OUT / EXPECT-OUT-APPROX lines".into());
    }
    Ok(out)
}

/// `|a - e| <= rel * |e|`; NaN never matches, a zero or infinite expectation demands equality - the
/// comparator the corpus gate (`crates/wukongc/tests/common/mod.rs`) uses.
fn premise_approx_eq(a: f64, e: f64, rel: f64) -> bool {
    if a.is_nan() || e.is_nan() {
        return false;
    }
    if e == 0.0 || e.is_infinite() {
        return a == e;
    }
    (a - e).abs() <= rel * e.abs()
}

/// Compile `src` with recognizers on or off (the `WUKONG_NO_RECOGNIZE` knob, set the way the driver
/// sets it for an ordinary `--run`) at -O2 and run `main`, on the interpreter or natively. Returns
/// the exit code and stdout.
pub(crate) fn premise_run(
    src: &str,
    recognize: bool,
    native: bool,
) -> Result<(i64, String), String> {
    use wukong_span::{Interner, SourceId};
    let mut interner = Interner::new();
    let (module, pd) = wukong_parser::parse_module(src, SourceId(0), &mut interner);
    if pd.iter().any(|d| d.is_error()) {
        return Err(format!("parse error: {pd:?}"));
    }
    let (sema, sd) = wukong_sema::check(&module, &interner);
    if sd.iter().any(|d| d.is_error()) {
        return Err(format!("sema error: {sd:?}"));
    }
    let opts = wukong_mir_build::LowerOptions {
        unroll_for_autodiff: false,
        recognize,
        ..Default::default()
    };
    let (mut program, ld) =
        wukong_mir_build::lower_program_with(&module, &sema, &mut interner, opts);
    if ld.iter().any(|d| d.is_error()) {
        return Err(format!("lowering error: {ld:?}"));
    }
    wukong_opt::optimize(&mut program, 2);
    let errs = wukong_mir::verify::verify_program(&program);
    if !errs.is_empty() {
        return Err(format!("MIR verifier: {errs:?}"));
    }
    let main = interner.intern("main");
    let (code, out) = if native {
        wukong_codegen_cranelift::jit_run(&program, main, &interner)?
    } else {
        wukong_interp::run_with_output(&program, main, &interner)?
    };
    Ok((code, String::from_utf8_lossy(&out).into_owned()))
}

/// The verdict for one premise fixture.
#[derive(Debug)]
pub(crate) struct PremiseRow {
    pub(crate) name: String,
    pub(crate) lanes: usize,
    /// Worst relative deviation from the closed form, recognition on / off.
    pub(crate) worst_on: f64,
    pub(crate) worst_off: f64,
    /// Lanes whose printed value moved between on and off, and the largest relative move.
    pub(crate) moved: usize,
    pub(crate) max_move: f64,
    /// Every failure, as the `!` line it prints; empty means the premise held for this fixture.
    pub(crate) failures: Vec<String>,
}

/// One run of a premise fixture: `(recognize, native, result)`.
pub(crate) type PremiseRun = (bool, bool, Result<(i64, String), String>);

/// Judge one fixture's four runs (on/off x interpreter/native) against its expectations. Pure, so
/// the verdict logic is unit-testable without compiling anything.
pub(crate) fn judge_premise(name: &str, expect: &[(f64, f64)], runs: &[PremiseRun]) -> PremiseRow {
    let mut failures = Vec::new();
    // [recognition on, off][interp, native]
    let mut lanes: [[Option<Vec<f64>>; 2]; 2] = Default::default();
    let mut worst = [0.0f64; 2];
    for (recognize, native, r) in runs {
        let mode = format!(
            "recognition {}, {}",
            if *recognize { "on" } else { "off" },
            if *native { "native" } else { "interp" }
        );
        let (code, out) = match r {
            Ok(v) => v,
            Err(e) => {
                failures.push(format!("  ! premise {name} [{mode}]: did not run: {e}"));
                continue;
            }
        };
        if *code != 0 {
            failures.push(format!(
                "  ! premise {name} [{mode}]: exit {code}, expected 0"
            ));
            continue;
        }
        let printed: Vec<&str> = out.lines().collect();
        if printed.len() != expect.len() {
            failures.push(format!(
                "  ! premise {name} [{mode}]: {} stdout line(s), {} expected",
                printed.len(),
                expect.len()
            ));
            continue;
        }
        let mut vals = Vec::with_capacity(printed.len());
        for (i, (l, &(e, t))) in printed.iter().zip(expect).enumerate() {
            let Ok(a) = l.trim().parse::<f64>() else {
                failures.push(format!(
                    "  ! premise {name} [{mode}]: lane {i} {l:?} is not a number"
                ));
                break;
            };
            let dev = if e == 0.0 {
                a.abs()
            } else {
                ((a - e) / e).abs()
            };
            let w = &mut worst[usize::from(!*recognize)];
            *w = w.max(if dev.is_nan() { f64::INFINITY } else { dev });
            if !premise_approx_eq(a, e, t) {
                failures.push(format!(
                    "  ! premise {name} [{mode}]: lane {i} = {l} mismatch vs closed form {e} \
                     (rel {dev:.2e} > {t:e})"
                ));
            }
            vals.push(a);
        }
        // Keep the run's lanes only if every one parsed.
        if vals.len() == expect.len() {
            lanes[usize::from(!*recognize)][usize::from(*native)] = Some(vals);
        }
    }
    // The bit-exact differential, within each lowering.
    for (k, which) in [(0usize, "on"), (1, "off")] {
        if let [Some(i), Some(n)] = &lanes[k] {
            if let Some(at) = i
                .iter()
                .zip(n)
                .position(|(a, b)| a.to_bits() != b.to_bits())
            {
                failures.push(format!(
                    "  ! premise {name}: recognition {which}: interp and native disagree at lane \
                     {at} ({} vs {}) - bit-exact is the contract",
                    i[at], n[at]
                ));
            }
        }
    }
    // On vs off, lane by lane, on the interpreter (the oracle): each lane is within `t` of one
    // reference on both sides, so the two may differ by at most 2t/(1-t) relative to either.
    let (mut moved, mut max_move) = (0usize, 0.0f64);
    if let ([Some(on), _], [Some(off), _]) = (&lanes[0], &lanes[1]) {
        for (i, ((a, b), &(_, t))) in on.iter().zip(off).zip(expect).enumerate() {
            if a.to_bits() == b.to_bits() {
                continue;
            }
            moved += 1;
            let m = if *b == 0.0 {
                f64::INFINITY
            } else {
                ((a - b) / b).abs()
            };
            max_move = max_move.max(m);
            let bound = 2.0 * t / (1.0 - t);
            if m > bound {
                failures.push(format!(
                    "  ! premise {name}: lane {i} moved {m:.2e} between recognition on ({a}) and \
                     off ({b}), beyond the declared bound {bound:.2e}"
                ));
            }
        }
    } else {
        failures.push(format!(
            "  ! premise {name}: on/off comparison NOT RUN - an interpreter run is missing"
        ));
    }
    PremiseRow {
        name: name.to_string(),
        lanes: expect.len(),
        worst_on: worst[0],
        worst_off: worst[1],
        moved,
        max_move,
        failures,
    }
}

/// Compile and judge one committed premise fixture by file name.
fn premise_fixture(file: &str) -> Result<PremiseRow, String> {
    let path = premise_dir().join(file);
    let src = std::fs::read_to_string(&path)
        .map_err(|e| format!("cannot read {}: {e}", path.display()))?;
    let expect = premise_expectations(&src)?;
    let runs: Vec<PremiseRun> = [(true, false), (true, true), (false, false), (false, true)]
        .into_iter()
        .map(|(rec, nat)| (rec, nat, premise_run(&src, rec, nat)))
        .collect();
    Ok(judge_premise(file, &expect, &runs))
}

/// Run the premise check over every fixture and print its section. Returns `true` when the premise
/// held for all of them (a missing or unparsable fixture is a failure, never a skip).
pub(crate) fn premise_check() -> bool {
    println!("================================================================================");
    println!("recognition on/off PREMISE at measurement size (P12.24): tests/run/premise_*.wk");
    println!("================================================================================");
    println!(
        "Each fixture compiled twice (recognizers on / off) at -O2 and run on the interpreter and\n\
         natively: every lane within its own EXPECT-OUT-APPROX bound, interp == native bit for bit,\n\
         and on vs off within 2t/(1-t). `moved` counts the printed lanes recognition changed."
    );
    let mut ok = true;
    for file in PREMISE_FIXTURES {
        let row = match premise_fixture(file) {
            Ok(r) => r,
            Err(e) => {
                println!("  ! premise {file}: {e}");
                ok = false;
                continue;
            }
        };
        println!(
            "  {:<24} {:>5} lanes  worst vs closed form: on {:.2e} / off {:.2e}  moved {} lane(s), \
             max {:.2e}{}",
            row.name,
            row.lanes,
            row.worst_on,
            row.worst_off,
            row.moved,
            row.max_move,
            if row.failures.is_empty() {
                "  - PREMISE HOLDS"
            } else {
                ""
            }
        );
        for f in &row.failures {
            println!("{f}");
        }
        ok &= row.failures.is_empty();
    }
    if !ok {
        println!(
            "  ! PREMISE CHECK FAILED: recognition changed a measurement-size answer beyond its \
             declared bound, or a premise fixture did not run - a recognizer-on vs wk(gen) reading \
             is unsupported until it is fixed"
        );
    }
    println!();
    ok
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ok(lines: &[f64]) -> Result<(i64, String), String> {
        Ok((0, lines.iter().map(|v| format!("{v}\n")).collect()))
    }

    fn clean() -> Vec<PremiseRun> {
        vec![
            (true, false, ok(&[100.0, 200.0])),
            (true, true, ok(&[100.0, 200.0])),
            (false, false, ok(&[100.05, 200.0])),
            (false, true, ok(&[100.05, 200.0])),
        ]
    }

    #[test]
    fn premise_expectations_read_approx_and_exact_numbers_only() {
        let src = "// RUN: --run\n// EXPECT-EXIT: 0\n// EXPECT-OUT-APPROX: 1.5 1e-3\n\
                   // EXPECT-OUT: 2\nfn main() -> i32 { return 0; }\n";
        assert_eq!(
            premise_expectations(src).unwrap(),
            vec![(1.5, 1e-3), (2.0, 0.0)]
        );
        assert!(premise_expectations("// EXPECT-OUT: hello\n").is_err());
        assert!(premise_expectations("// EXPECT-OUT-APPROX: 1.0 2.0\n").is_err());
        assert!(premise_expectations("// EXPECT-OUT-NONE\n").is_err());
        assert!(premise_expectations("fn main() {}\n").is_err());
    }

    /// The verdict can fail on every axis it claims to check - the check's own mutation control.
    #[test]
    fn the_premise_verdict_fails_on_each_axis_and_passes_a_clean_run() {
        let expect = [(100.0, 1e-3), (200.0, 1e-3)];
        let row = judge_premise("x", &expect, &clean());
        assert!(row.failures.is_empty(), "{:?}", row.failures);
        assert_eq!(row.moved, 1);
        assert!((row.max_move - 0.05 / 100.05).abs() < 1e-12);

        // A lane outside its closed-form bound.
        let mut r = clean();
        r[0].2 = ok(&[100.2, 200.0]);
        r[1].2 = ok(&[100.2, 200.0]);
        assert!(judge_premise("x", &expect, &r)
            .failures
            .iter()
            .any(|f| f.contains("mismatch vs closed form")));

        // Interpreter and native disagree within one lowering (each still inside the bound).
        let mut r = clean();
        r[1].2 = ok(&[100.01, 200.0]);
        assert!(judge_premise("x", &expect, &r)
            .failures
            .iter()
            .any(|f| f.contains("interp and native disagree")));

        // On and off each inside an exact (t = 0) expectation cannot move; a move is caught. (With
        // t > 0, two lanes inside the bound cannot exceed 2t/(1-t), so this is the only way the
        // on/off arm can fire on its own.)
        let exact = [(100.0, 0.0)];
        let mut r = vec![
            (true, false, ok(&[100.0])),
            (true, true, ok(&[100.0])),
            (false, false, ok(&[100.0])),
            (false, true, ok(&[100.0])),
        ];
        assert!(judge_premise("x", &exact, &r).failures.is_empty());
        r[2].2 = ok(&[100.000001]);
        r[3].2 = ok(&[100.000001]);
        let row = judge_premise("x", &exact, &r);
        assert!(
            row.failures
                .iter()
                .any(|f| f.contains("beyond the declared bound")),
            "{:?}",
            row.failures
        );

        // A run that did not happen is a failure, never a pass.
        let mut r = clean();
        r[2].2 = Err("lowering error".into());
        r[3].2 = Err("lowering error".into());
        let row = judge_premise("x", &expect, &r);
        assert!(
            row.failures.iter().any(|f| f.contains("did not run")),
            "{:?}",
            row.failures
        );
        assert!(row.failures.iter().any(|f| f.contains("NOT RUN")));

        // A wrong line count, and a non-zero exit.
        let mut r = clean();
        r[0].2 = ok(&[100.0]);
        r[1].2 = Ok((3, "100\n200\n".into()));
        let row = judge_premise("x", &expect, &r);
        assert!(row.failures.iter().any(|f| f.contains("stdout line(s)")));
        assert!(row.failures.iter().any(|f| f.contains("exit 3")));
    }

    /// The committed fixtures, end to end, through the same `premise_check` that `wukong-xbench
    /// compare` and `sota-check` print before any timing - so `cargo test -p wukong_xbench` gates the
    /// premise too (its `! premise` lines are in the captured stdout when this fails).
    #[test]
    fn the_premise_holds_on_the_committed_fixtures() {
        assert!(
            premise_check(),
            "the recognition on/off premise failed - see the captured `! premise` lines"
        );
        // Each fixture is non-trivial: a premise over a handful of lanes would say nothing.
        for (file, min_lanes) in PREMISE_FIXTURES.into_iter().zip([8, 768, 4096]) {
            let src = std::fs::read_to_string(premise_dir().join(file)).unwrap();
            let lanes = premise_expectations(&src).unwrap().len();
            assert_eq!(lanes, min_lanes, "{file}");
        }
    }
}
