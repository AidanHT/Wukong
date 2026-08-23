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
        .unwrap_or_else(|| "did not compile".into());
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
