//! The typed-MIR round trip and the `--passes` law, end to end through the real binary, over the
//! `tests/run` corpus.
//!
//! For each program `p` (the directory scan is non-recursive, as in every harness here; a program
//! that does not compile without its `// RUN:` flags is skipped), with `F` = `-O2
//! --target-cpu=x86-64-v2` (a level below x86-64-v3 builds no vector-kernel recipe, which has no
//! textual form):
//!
//! 1. `A` = `wukongc --emit=mir-text F p.wk`, written to `x.mir`;
//! 2. `wukongc --emit=mir-text x.mir` is `A`, byte for byte: the parser loses nothing the typed
//!    printer prints;
//! 3. `wukongc --emit=mir x.mir` is `wukongc --emit=mir F p.wk`: the parsed program is the
//!    program, as the plain dump sees it;
//! 4. `wukongc --emit=mir -O0 --passes=<the -O2 list> --target-cpu=x86-64-v2 p.wk` is that same
//!    `--emit=mir` output: the pass-list replica is the standard pipeline (lowering reads no `-O`
//!    level, so the two compiles start from the same MIR);
//! 5. `wukongc --emit=mir -O2 --target-cpu=x86-64-v2 y.mir`, where `y.mir` is the *unoptimized*
//!    typed dump, is that same output too: optimizing a parsed dump is optimizing the program,
//!    which is what makes a dump a reproducer for an optimizer bug (the optimizer builds 256-bit
//!    recipes of its own, so this compile names the same level as `F`);
//! 6. and so is the pipeline cut in two through a dump: `--passes=<the fixpoint half>
//!    --emit=mir-text p.wk` into `z.mir`, then `--passes=unroll,slp,form-switches --emit=mir
//!    z.mir`. A pass deletes values, so this dump's arena outlives its last value, and the
//!    `values N` line is what keeps the values the second half creates numbered as in one run.
//!
//! Checks 1-3 run on every program, on up to four worker threads. Checks 4-6 compile the program
//! three more times, so they run on every second program by name plus [`ALWAYS`] (about 60 s for
//! the 548-program corpus in a debug build here); `WUKONG_MIR_TEXT_ALL=1` runs them everywhere
//! (all six held on all 548 programs when this harness landed).

use std::path::{Path, PathBuf};
use std::process::Command;

/// The -O2 pass list (`wukong_driver::passes::standard_spec(2)`), spelled out so a drift between
/// it and the CLI is a failure here rather than a silent agreement.
const O2_LIST: &str =
    "inline,sroa,mem2reg,simplify,jump-thread,simplify-cfg,simplify-phis,dce,tre,\
                       cse,dse,loop-canon,loop-delete,fuse,licm,bce,unswitch,promote,vectorize,\
                       unroll,slp,form-switches";

/// The two halves of [`O2_LIST`] for law 6: every function pass, then the program steps after them.
const FIRST_HALF: &str = "inline,sroa,mem2reg,simplify,jump-thread,simplify-cfg,simplify-phis,\
                          dce,tre,cse,dse,loop-canon,loop-delete,fuse,licm,bce,unswitch,promote,\
                          vectorize";
const SECOND_HALF: &str = "unroll,slp,form-switches";

/// Programs that always get all six checks: between them, every optimizer stage, every MIR type
/// constructor and the statics, externs and generic-instance names only the typed form carries.
const ALWAYS: &[&str] = &[
    "tensor_matmul.wk",
    "vec5_slp_bundles.wk",
    "vec5_scatter.wk",
    "atomic_add_histogram.wk",
    "align_local.wk",
    "match_dense_enum_switch.wk",
    "loop_unroll.wk",
    "tail_recursion.wk",
];

/// The variables that change what `wukongc` emits, or print to stderr, cleared from every child so
/// the gate is hermetic (the list `determinism.rs` keeps, plus the tracing switches).
const COMPILER_ENV: &[&str] = &[
    "WUKONG_NO_RECOGNIZE",
    "WUKONG_P4_NO_256",
    "WUKONG_NO_VECTORIZE",
    "WUKONG_NO_UNROLL",
    "WUKONG_NO_IV_RAMP",
    "WUKONG_NO_SLP",
    "WUKONG_VEC256_MIN_TRIP",
    "WUKONG_VEC_TRACE",
    "WUKONG_SLP_TRACE",
];

const CPU: &str = "--target-cpu=x86-64-v2";

fn corpus() -> Vec<PathBuf> {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/run");
    let mut programs: Vec<PathBuf> = std::fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", dir.display()))
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "wk"))
        .collect();
    programs.sort();
    assert!(programs.len() > 100, "the run corpus is missing");
    programs
}

/// A scratch directory unique to this process and test, removed on drop.
struct Scratch(PathBuf);

impl Scratch {
    fn new(tag: &str) -> Scratch {
        let dir = std::env::temp_dir()
            .join("wukongc_mir_text")
            .join(format!("{}_{tag}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("create scratch dir");
        Scratch(dir)
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// Run `wukongc args`, returning (exit code, stdout, stderr).
fn wukongc(args: &[&str]) -> (i32, String, String) {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_wukongc"));
    for v in COMPILER_ENV {
        cmd.env_remove(v);
    }
    let out = cmd
        .args(args)
        .arg("--color=never")
        .output()
        .expect("failed to spawn wukongc");
    (
        out.status.code().unwrap_or(-1),
        String::from_utf8_lossy(&out.stdout).into_owned(),
        String::from_utf8_lossy(&out.stderr).into_owned(),
    )
}

/// Run `args`, which must succeed, returning stdout.
fn ok(args: &[&str]) -> String {
    let (code, out, err) = wukongc(args);
    assert_eq!(code, 0, "wukongc {args:?} failed:\n{err}");
    out
}

/// The checks of the module docs for one program: 1-3, and 4-6 too when `full`. `Ok(false)` =
/// skipped (does not compile without its RUN flags); `Err` = a failed law, described.
///
/// Whether to skip is decided by the plain `--emit=mir` compile, never by `--emit=mir-text`
/// itself: a typed-printer failure on a program that compiles is a broken law, not a skip, or a
/// regression confined to a rare op would hide under the skipped count.
fn check(program: &Path, scratch: &Path, full: bool) -> Result<bool, String> {
    let wk = program.to_str().expect("utf-8 path");
    let name = program.file_name().unwrap().to_string_lossy().into_owned();
    let (code, plain, _) = wukongc(&["--emit=mir", "-O2", CPU, wk]);
    if code != 0 {
        return Ok(false);
    }
    let (code, typed, err) = wukongc(&["--emit=mir-text", "-O2", CPU, wk]);
    if code != 0 {
        return Err(format!(
            "{name}: --emit=mir-text failed (exit {code}) where --emit=mir succeeded:\n{err}"
        ));
    }
    let x = scratch.join(format!("{name}.O2.mir"));
    std::fs::write(&x, &typed).expect("write the dump");
    let x = x.to_str().unwrap();
    let diff = |law: &str, want: &str, got: &str| {
        let line = want
            .lines()
            .zip(got.lines())
            .position(|(a, b)| a != b)
            .unwrap_or_else(|| want.lines().count().min(got.lines().count()));
        format!(
            "{name}: {law}: first difference at line {}\n  want: {:?}\n  got:  {:?}",
            line + 1,
            want.lines().nth(line),
            got.lines().nth(line)
        )
    };
    let reread = ok(&["--emit=mir-text", x]);
    if reread != typed {
        return Err(diff("--emit=mir-text of the dump", &typed, &reread));
    }
    let reparsed = ok(&["--emit=mir", x]);
    if reparsed != plain {
        return Err(diff("--emit=mir of the dump", &plain, &reparsed));
    }
    if !full {
        return Ok(true);
    }
    let listed = ok(&["--emit=mir", "-O0", &format!("--passes={O2_LIST}"), CPU, wk]);
    if listed != plain {
        return Err(diff("-O0 --passes=<-O2 list>", &plain, &listed));
    }
    let lowered = ok(&["--emit=mir-text", "-O0", CPU, wk]);
    let y = scratch.join(format!("{name}.O0.mir"));
    std::fs::write(&y, &lowered).expect("write the dump");
    let optimized = ok(&["--emit=mir", "-O2", CPU, y.to_str().unwrap()]);
    if optimized != plain {
        return Err(diff("-O2 of the unoptimized dump", &plain, &optimized));
    }
    let halfway = ok(&[
        "--emit=mir-text",
        &format!("--passes={FIRST_HALF}"),
        CPU,
        wk,
    ]);
    let z = scratch.join(format!("{name}.half.mir"));
    std::fs::write(&z, &halfway).expect("write the dump");
    let finished = ok(&[
        "--emit=mir",
        &format!("--passes={SECOND_HALF}"),
        z.to_str().unwrap(),
    ]);
    if finished != plain {
        return Err(diff(
            "the -O2 list cut in two through a dump",
            &plain,
            &finished,
        ));
    }
    Ok(true)
}

#[test]
fn typed_mir_round_trips_and_the_pass_list_is_the_pipeline() {
    assert_eq!(format!("{FIRST_HALF},{SECOND_HALF}"), O2_LIST);
    let programs = corpus();
    let all = std::env::var_os("WUKONG_MIR_TEXT_ALL").is_some();
    let full = |i: usize, p: &Path| {
        all || i.is_multiple_of(2) || ALWAYS.contains(&p.file_name().unwrap().to_str().unwrap())
    };
    let scratch = Scratch::new("corpus");
    let next = std::sync::atomic::AtomicUsize::new(0);
    let workers = std::thread::available_parallelism().map_or(2, |n| n.get().clamp(1, 4));
    let results: Vec<(PathBuf, Result<bool, String>)> = std::thread::scope(|s| {
        let handles: Vec<_> = (0..workers)
            .map(|_| {
                s.spawn(|| {
                    let mut mine = Vec::new();
                    loop {
                        let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        let Some(p) = programs.get(i) else { break };
                        mine.push((p.clone(), check(p, &scratch.0, full(i, p))));
                    }
                    mine
                })
            })
            .collect();
        handles
            .into_iter()
            .flat_map(|h| h.join().expect("worker panicked"))
            .collect()
    });
    let checked = results.iter().filter(|(_, r)| *r == Ok(true)).count();
    let skipped = results.iter().filter(|(_, r)| *r == Ok(false)).count();
    eprintln!(
        "mir_text: {checked} of {} programs checked ({skipped} skipped: they compile only with \
         their RUN flags)",
        programs.len(),
    );
    let failures: Vec<&String> = results
        .iter()
        .filter_map(|(_, r)| r.as_ref().err())
        .collect();
    assert!(
        failures.is_empty(),
        "{} of {} programs broke a law:\n{}",
        failures.len(),
        programs.len(),
        failures
            .iter()
            .map(|f| f.as_str())
            .collect::<Vec<_>>()
            .join("\n")
    );
    // A sweep that skipped everything proves nothing.
    assert!(
        checked * 10 >= programs.len() * 9,
        "only {checked} of {} programs compiled",
        programs.len()
    );
    for name in ALWAYS {
        assert!(
            results
                .iter()
                .any(|(p, r)| p.ends_with(name) && *r == Ok(true)),
            "{name} must be checked"
        );
    }
}

/// The `--print-after` acceptance: the snapshot is on stderr, stdout is untouched.
#[test]
fn print_after_writes_stderr_and_leaves_stdout_identical() {
    let wk = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/run/tensor_matmul.wk");
    let wk = wk.to_str().unwrap();
    let (code, plain, _) = wukongc(&["-O2", "--emit=mir", wk]);
    assert_eq!(code, 0);
    let (code, watched, err) = wukongc(&["-O2", "--print-after=licm", "--emit=mir", wk]);
    assert_eq!(code, 0, "{err}");
    assert_eq!(watched, plain);
    assert!(err.contains("; *** MIR after licm"), "{err}");
}

/// A `.mir` file's errors: a syntax error names the file and line (exit 1), an invalid program is
/// the verifier's ICE (exit 1), a stage other than the two MIR ones is a usage error (exit 2).
#[test]
fn a_bad_mir_file_is_reported_with_its_line() {
    let scratch = Scratch::new("errors");
    let bad = scratch.0.join("bad.mir");
    std::fs::write(
        &bad,
        "; wukong mir text v1\nfn main() -> i32 {\n  bb0:\n    v0: i32 = frob\n    ret v0\n}\n",
    )
    .unwrap();
    let bad = bad.to_str().unwrap();
    let (code, out, err) = wukongc(&["--emit=mir", bad]);
    assert_eq!(code, 1, "{err}");
    assert!(out.is_empty());
    assert!(
        err.contains(&format!("{bad}:4: invalid MIR: unknown operation `frob`")),
        "{err}"
    );

    let invalid = scratch.0.join("invalid.mir");
    std::fs::write(
        &invalid,
        "fn main() -> i32 {\n  bb0:\n    v0: i64 = const.i64 1\n    ret v0\n}\n",
    )
    .unwrap();
    let (code, _, err) = wukongc(&["--emit=mir-text", invalid.to_str().unwrap()]);
    assert_eq!(code, 1, "{err}");
    assert!(
        err.contains("internal compiler error (MIR verify)"),
        "{err}"
    );

    let (code, _, err) = wukongc(&["--run", bad]);
    assert_eq!(code, 2, "{err}");
    assert!(err.contains(".mir"), "{err}");

    let veck = scratch.0.join("veck.mir");
    std::fs::write(
        &veck,
        "fn k(v0 ptr, v1 i64) -> void {\n  bb0(v0 ptr, v1 i64):\n    \
         veckernel #0(ptrs=v0, scalars=v0, n=v1)\n    ret\n}\n",
    )
    .unwrap();
    let (code, _, err) = wukongc(&["--emit=mir", veck.to_str().unwrap()]);
    assert_eq!(code, 1, "{err}");
    assert!(
        err.contains(":3: invalid MIR: vector-kernel recipes"),
        "{err}"
    );
}
