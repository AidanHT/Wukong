//! The forced-parallel lane: every `tests/run` program that dispatches a `_parallel` kernel is run
//! with the runtime's fork gates forced open and must print exactly what it prints without them, and
//! exactly what the interpreter prints.
//!
//! ## The hole this closes
//!
//! Every `_parallel` entry in `wukong_runtime` consults a WORK GATE before it forks (`par::fork_rows`
//! / `par::fork_units`, the GEMM's MAC threshold, the streaming map's element threshold, the packer's
//! byte threshold) and below it tail-calls its serial sibling. The gates are sized for real models:
//! 2^18 element-ops, 2^23 MACs, 256 Ki elements. Every corpus fixture is far below them — so on every
//! gate in this repo, a recognized `_parallel` kernel has been running its SERIAL fallback. The
//! chunking, the claim queue and the cross-core combine — the only code a `_parallel` twin has that
//! its serial twin does not — ran nowhere but in `wukong_runtime`'s own unit tests (where
//! `cfg(test)` zeroes the work gate). A `_parallel` kernel that split a row wrongly would pass
//! `native_matches_interpreter`, both invariance gates and the census, because none of them ever
//! made it split.
//!
//! The gates are scheduling-only by the runtime's standing law ("no setting can change a result":
//! rows/columns/channels are independent, partials fold in a fixed order), and each one reads an
//! environment override once. This lane sets every override to its smallest accepted value, so any
//! shape with two or more units forks, and asserts the law: native with the gates forced open ==
//! native with the defaults == the interpreter (which calls the serial entries), byte for byte, exit
//! code and stdout. A failure is a real serial != parallel defect, not a tolerance question.
//!
//! ## The forcing environment, and where each knob is defined
//!
//! | variable | parsed by | smallest accepted |
//! |---|---|---|
//! | `WUKONG_PAR_MIN_WORK` | `wukong_runtime/src/par.rs` `parse_min_work`: a positive ASCII decimal | `1` |
//! | `WUKONG_GEMM_PAR_MIN_MACS` | `gemm.rs` `par_min_macs`: `u64` parse (MAC count) | `1` (0 would also parse; 1 already admits every non-empty GEMM) |
//! | `WUKONG_VELEM_PAR_MIN` | `velem.rs` `velem_par_min`: `usize` parse (element count) | `1` |
//! | `WUKONG_PACK_PAR_MIN_KB` | `gemm.rs` `pack_min_bytes_from_kb`: `usize` KiB, zero allowed ("always pack in parallel") | `0` |
//!
//! Two gates are NOT environment-tunable and this lane cannot open them: `TRANSPOSE_PAR_MIN`
//! (65536 elements, transpose.rs) and `VMATH_PAR_MIN` (16 Ki elements, vmath.rs); the fixed-chunk
//! reductions (`RCHUNK` = 8192 in reduce.rs / lowp.rs) split by size alone. Those are reached by
//! fixtures sized above them rather than by this lane, one gate each: `TRANSPOSE_PAR_MIN` by
//! `cov4_par_transpose_wide.wk` (65792 elements), `VMATH_PAR_MIN` by `vmath_parallel.wk` (20000 >
//! 16384, the only corpus program whose `wukong_vmath_f32_parallel` call crosses it), and `RCHUNK`
//! by `cov4_par_lowp_dot.wk` and `parallel_argmax.wk` (20000 > 8192). [`FIXED_GATE_WITNESSES`] pins
//! that list, so a witness that is deleted, stops dispatching its kernel, or is shrunk below its
//! size fails a test instead of leaving this paragraph quietly wrong.
//!
//! The set of programs is read from `tests/recognizer-census.txt`, which `recognizer_census.rs`
//! holds byte-identical to what the compiler emits, so this lane cannot drift from the real
//! dispatch surface without that gate failing first.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

#[path = "common/mod.rs"]
mod common;

/// The gate overrides, each at its smallest accepted value (see the module table).
const FORCE_PARALLEL: &[(&str, &str)] = &[
    ("WUKONG_PAR_MIN_WORK", "1"),
    ("WUKONG_GEMM_PAR_MIN_MACS", "1"),
    ("WUKONG_VELEM_PAR_MIN", "1"),
    ("WUKONG_PACK_PAR_MIN_KB", "0"),
];

/// Variables that change what the compiler emits or how the runtime schedules. Every child starts
/// without them, so "the default env" means the defaults and not whatever the developer exported.
const AMBIENT: &[&str] = &[
    "WUKONG_NO_RECOGNIZE",
    "WUKONG_NO_VECTORIZE",
    "WUKONG_NO_UNROLL",
    "WUKONG_NO_IV_RAMP",
    "WUKONG_P4_NO_256",
    "WUKONG_VEC_TRACE",
    "WUKONG_MAX_ISA",
    "WUKONG_PAR_MIN_WORK",
    "WUKONG_GEMM_PAR_MIN_MACS",
    "WUKONG_VELEM_PAR_MIN",
    "WUKONG_PACK_PAR_MIN_KB",
    "WUKONG_NUM_THREADS",
    "RAYON_NUM_THREADS",
];

/// Floor on the number of programs this lane covers. 43 when it landed: 26 pre-existing
/// `_parallel` fixtures plus the `cov4_par_*` family. Raise it with the corpus; a census that stops
/// listing `_parallel` symbols must not turn this lane silently vacuous.
const MIN_PARALLEL_PROGRAMS: usize = 43;

/// A fork gate no environment knob opens, and the corpus program that is the only thing crossing it.
struct FixedGate {
    /// The runtime constant and where it is defined.
    gate: &'static str,
    /// The size that must be exceeded for the `_parallel` entry to split into two or more pieces.
    threshold: usize,
    /// The `tests/run` program sized above it.
    program: &'static str,
    /// The `_parallel` kernel its census line must list.
    symbol: &'static str,
    /// The element count the program passes, spelled as a decimal literal in its source.
    size: usize,
}

/// The witnesses for the gates this lane cannot force (see the module doc). Each threshold is the
/// runtime's own constant: `TRANSPOSE_PAR_MIN = 1 << 16` (transpose.rs: `r * c < it` runs serial),
/// `VMATH_PAR_MIN = VMATH_CHUNK = 1 << 14` (vmath.rs: `n < it` runs serial, and `n == it` is one
/// chunk), `RCHUNK = 8192` (reduce.rs, which lowp.rs reuses: fewer than two chunks runs serial).
const FIXED_GATE_WITNESSES: &[FixedGate] = &[
    FixedGate {
        gate: "TRANSPOSE_PAR_MIN (wukong_runtime/src/transpose.rs)",
        threshold: 1 << 16,
        program: "cov4_par_transpose_wide.wk",
        symbol: "wukong_transpose_f32_parallel",
        size: 65792,
    },
    FixedGate {
        gate: "VMATH_PAR_MIN (wukong_runtime/src/vmath.rs)",
        threshold: 1 << 14,
        program: "vmath_parallel.wk",
        symbol: "wukong_vmath_f32_parallel",
        size: 20000,
    },
    FixedGate {
        gate: "RCHUNK (wukong_runtime/src/reduce.rs, used by lowp.rs)",
        threshold: 8192,
        program: "cov4_par_lowp_dot.wk",
        symbol: "wukong_dot_f16_parallel",
        size: 20000,
    },
    FixedGate {
        gate: "RCHUNK (wukong_runtime/src/reduce.rs)",
        threshold: 8192,
        program: "parallel_argmax.wk",
        symbol: "wukong_argreduce_f32_parallel",
        size: 20000,
    },
];

/// Does `src` spell `n` as a standalone decimal literal (not a digit run inside a longer number)?
fn spells_literal(src: &str, n: usize) -> bool {
    let lit = n.to_string();
    let bytes = src.as_bytes();
    src.match_indices(&lit).any(|(i, _)| {
        let before = i.checked_sub(1).map(|j| bytes[j]);
        let after = bytes.get(i + lit.len()).copied();
        !before.is_some_and(|b| b.is_ascii_digit()) && !after.is_some_and(|b| b.is_ascii_digit())
    })
}

fn run_dir() -> PathBuf {
    common::repo_root().join("tests").join("run")
}

/// Is this census token a `_parallel` runtime kernel (multiplicity witnesses `sym*N` included)?
/// `wukong_parallel_for` is the generic outliner, not a `_parallel` twin, and has no work gate.
fn is_parallel_kernel(token: &str) -> bool {
    let sym = token.split('*').next().unwrap_or(token);
    sym.starts_with("wukong_") && sym.ends_with("_parallel")
}

/// The census fixtures whose dispatch includes a `_parallel` kernel, with the kernels.
fn parallel_programs() -> Vec<(String, Vec<String>)> {
    let path = common::repo_root()
        .join("tests")
        .join("recognizer-census.txt");
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
    let mut out = Vec::new();
    for line in text.lines() {
        if line.starts_with('#') || line.trim().is_empty() {
            continue;
        }
        let (name, rest) = line
            .split_once(':')
            .unwrap_or_else(|| panic!("malformed census line {line:?}"));
        let kernels: Vec<String> = rest
            .split_whitespace()
            .filter(|t| is_parallel_kernel(t) && !t.contains('*'))
            .map(str::to_string)
            .collect();
        if !kernels.is_empty() {
            out.push((name.to_string(), kernels));
        }
    }
    out
}

static WORK_SEQ: AtomicU64 = AtomicU64::new(0);

/// A fresh scratch CWD per run: some fixtures write relative-path blobs.
fn scratch() -> PathBuf {
    let seq = WORK_SEQ.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir()
        .join("wukongc_parallel_lane")
        .join(format!("{}_{seq}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("create scratch dir");
    dir
}

/// One `--run -O2` of `path` on `backend`, with `env` on top of a scrubbed environment.
fn run(path: &Path, backend: &str, env: &[(&str, &str)]) -> (Option<i32>, String, String) {
    let dir = scratch();
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_wukongc"));
    for k in AMBIENT {
        cmd.env_remove(k);
    }
    for (k, v) in env {
        cmd.env(k, v);
    }
    let out = cmd
        .args(["--run", "-O2", &format!("--backend={backend}")])
        .arg(path)
        .current_dir(&dir)
        .output()
        .expect("spawn wukongc");
    let _ = std::fs::remove_dir_all(&dir);
    (
        out.status.code(),
        String::from_utf8_lossy(&out.stdout).into_owned(),
        String::from_utf8_lossy(&out.stderr).into_owned(),
    )
}

/// The three columns for one program, or the reason they disagree.
fn check(name: &str, kernels: &[String]) -> Result<(), String> {
    let path = run_dir().join(name);
    let interp = run(&path, "interp", &[]);
    let native = run(&path, "native", &[]);
    let forced = run(&path, "native", FORCE_PARALLEL);
    if interp.0 != native.0 || interp.1 != native.1 {
        return Err(format!(
            "{name} (dispatches {kernels:?}): native with DEFAULT gates differs from the \
             interpreter\n    interp = {:?} {:?}\n    native = {:?} {:?}\n    stderr(native): {}",
            interp.0, interp.1, native.0, native.1, native.2
        ));
    }
    if forced.0 != native.0 || forced.1 != native.1 {
        return Err(format!(
            "{name} (dispatches {kernels:?}): native with the fork gates FORCED OPEN differs from \
             the default-gate run and the interpreter — a serial != parallel defect in one of those \
             kernels\n    default = {:?} {:?}\n    forced  = {:?} {:?}\n    stderr(forced): {}",
            native.0, native.1, forced.0, forced.1, forced.2
        ));
    }
    Ok(())
}

/// The census token filter: a `_parallel` twin counts, the generic outliner and a serial kernel do
/// not, and a multiplicity witness resolves to its symbol.
#[test]
fn the_lane_selects_parallel_twins_and_nothing_else() {
    assert!(is_parallel_kernel("wukong_colsum_f32_parallel"));
    assert!(is_parallel_kernel("wukong_sgemm_parallel*2"));
    assert!(!is_parallel_kernel("wukong_parallel_for"));
    assert!(!is_parallel_kernel("wukong_parallel_for*2"));
    assert!(!is_parallel_kernel("wukong_colsum_f32"));
    assert!(!is_parallel_kernel("veckernel*1"));
}

/// The gates this lane cannot force are crossed by named corpus programs, and only by them — so
/// each witness must still exist, still dispatch the gated `_parallel` kernel (per the census, which
/// `recognizer_census.rs` holds to the compiler's real output), and still spell a size above the
/// gate. Losing any of the three would leave that split executed by no gate in the repo.
#[test]
fn every_fixed_gate_has_a_witness_sized_above_it() {
    let census = parallel_programs();
    for w in FIXED_GATE_WITNESSES {
        assert!(
            w.size > w.threshold,
            "{}: the witness size {} does not exceed the gate {} it is meant to cross",
            w.gate,
            w.size,
            w.threshold
        );
        let path = run_dir().join(w.program);
        let src = std::fs::read_to_string(&path).unwrap_or_else(|e| {
            panic!(
                "{}: its witness tests/run/{} is unreadable ({e}) — no program crosses this gate",
                w.gate, w.program
            )
        });
        let dispatches = census
            .iter()
            .find(|(name, _)| name == w.program)
            .is_some_and(|(_, kernels)| kernels.iter().any(|k| k == w.symbol));
        assert!(
            dispatches,
            "{}: tests/recognizer-census.txt no longer lists {} for its witness {} — the gated \
             kernel is no longer dispatched there",
            w.gate, w.symbol, w.program
        );
        assert!(
            spells_literal(&src, w.size),
            "{}: its witness {} no longer spells the size {} that crosses the gate ({}); \
             re-derive the witness or name another",
            w.gate,
            w.program,
            w.size,
            w.threshold
        );
    }
    assert!(spells_literal("for k in 0..20000 {", 20000));
    assert!(!spells_literal("[f32; 200000]", 20000));
    assert!(!spells_literal("120000", 20000));
}

#[test]
fn forced_parallel_native_matches_default_native_and_the_interpreter() {
    let programs = parallel_programs();
    assert!(
        programs.len() >= MIN_PARALLEL_PROGRAMS,
        "the census lists only {} program(s) that dispatch a `_parallel` kernel (floor \
         {MIN_PARALLEL_PROGRAMS}); either the corpus lost its parallel fixtures or the census \
         parser stopped matching, and this lane would be checking nothing",
        programs.len()
    );
    for (name, _) in &programs {
        assert!(
            run_dir().join(name).exists(),
            "the census names tests/run/{name}, which does not exist (stale census)"
        );
    }

    // A small fixed worker pool, as in the census sweep: three spawns per program is the whole cost.
    const WORKERS: usize = 4;
    let mut failures: Vec<String> = Vec::new();
    std::thread::scope(|s| {
        let handles: Vec<_> = (0..WORKERS)
            .map(|w| {
                let mine: Vec<&(String, Vec<String>)> =
                    programs.iter().skip(w).step_by(WORKERS).collect();
                s.spawn(move || {
                    mine.into_iter()
                        .filter_map(|(name, kernels)| check(name, kernels).err())
                        .collect::<Vec<_>>()
                })
            })
            .collect();
        for h in handles {
            failures.extend(h.join().expect("parallel-lane worker panicked"));
        }
    });
    failures.sort();
    assert!(
        failures.is_empty(),
        "{} of {} `_parallel` program(s) broke the serial == parallel law:\n{}",
        failures.len(),
        programs.len(),
        failures.join("\n")
    );
    let kernels: std::collections::BTreeSet<&str> = programs
        .iter()
        .flat_map(|(_, k)| k.iter().map(String::as_str))
        .collect();
    eprintln!(
        "parallel lane: {} program(s), {} distinct `_parallel` kernel(s), each run on the \
         interpreter, native (default gates) and native (gates forced open): all identical",
        programs.len(),
        kernels.len()
    );
}
