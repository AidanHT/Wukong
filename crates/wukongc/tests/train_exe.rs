//! P4.3: `wukongc --train --emit=exe` builds the training loop into an executable whose stdout is
//! **byte-identical** to `wukongc --train` on the interpreter and on `--backend=native`.
//!
//! The equality holds by construction — the executable's synthesized `main` calls the runtime's
//! `wukong_rt_train_main`, which runs the same `wukong_runtime::train::run_training` and prints the
//! same `format_report` the in-process run does, over the same spec bytes — and this gate is what
//! checks the construction: the kernels inside the artifact (the forward, the generated backward,
//! the fused AdamW kernels and their dispatcher) are a different Cranelift configuration from the
//! JIT's (PIC, real relocations, the object's symbol policy) and link a different copy of the
//! runtime, and only running the linked executable can see a difference.
//!
//! Covered: the two hand-derived `tests/run/autodiff_train_*.wk` trajectories, the 2-layer MLP
//! fixture, the `@parallel` region fixture (whose backward runs `wukong_parallel_for` from inside
//! the artifact), the GEMM + bias model of `autodiff_cli.rs` and a loss that prints, at `-O0` and
//! `-O2`, under SGD and under AdamW with every update-policy field the spec carries (weight decay,
//! clip, warmup, cosine).
//!
//! **A skip must be earned**, exactly as in `exe.rs` (whose link probe this repeats): only a box
//! with no rustc + `wukong_runtime` rlib may skip, and the test panics if it linked nothing.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

#[path = "common/mod.rs"]
mod common;

/// Which link path `wukongc --emit=exe` takes here (see `exe.rs`'s `LinkPath`, which this mirrors:
/// the rlib half calls the driver's own `runtime_rlib_in`, so it cannot drift from the link).
#[derive(Debug, PartialEq, Eq)]
enum LinkPath {
    /// rustc + the runtime rlib: the path a training executable needs. A failure is a defect.
    Rustc,
    /// Only a C compiler: the C fallback runtime has no training loop, so the driver refuses the
    /// link with a named error. Every case is then an earned skip (its stderr is printed).
    CcFallback,
    /// No toolchain at all.
    None,
}

fn probe_link_path() -> LinkPath {
    let runs = |program: &str| {
        Command::new(program)
            .arg("--version")
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
    };
    let rlib = Path::new(env!("CARGO_BIN_EXE_wukongc"))
        .parent()
        .and_then(wukong_driver::runtime_rlib_in)
        .is_some();
    if runs("rustc") && rlib {
        return LinkPath::Rustc;
    }
    let cc = std::env::var("CC").unwrap_or_else(|_| "cc".to_string());
    if runs(&cc) {
        return LinkPath::CcFallback;
    }
    LinkPath::None
}

/// The GEMM + broadcast-bias least-squares model of `autodiff_cli.rs` (`h = x·Wᵀ + bias`,
/// `L = Σ (h - t)²`), the kernel-backed model whose interp/native agreement that file pins.
const LINEAR_BIAS: &str = "\
@parallel
fn loss(x: [f32; 4], w: [f32; 4], bias: [f32; 2], t: [f32; 4], mut out: [f32; 1]) -> f32 {
    let mut h: [f32; 4] = [0.0; 4];
    for i in 0..2 { for j in 0..2 { let mut s: f32 = 0.0;
        for k in 0..2 { s = s + x[i * 2 + k] * w[j * 2 + k]; } h[i * 2 + j] = s; } }
    for i in 0..2 { for j in 0..2 { h[i * 2 + j] = h[i * 2 + j] + bias[j]; } }
    let mut l: f32 = 0.0;
    for i in 0..4 { l = l + (h[i] - t[i]) * (h[i] - t[i]); }
    out[0] = l;
    return l;
}
fn main() -> i32 { return 0; }
";

/// A loss that prints. Every kernel run prints the loss it computes — the gradient kernel replays
/// the forward, so once per update, then once for the final forward — and those lines must come
/// out of the executable exactly where `--train` forwards them: before the report.
const PRINTING: &str = "\
fn loss(w: [f32; 2], mut out: [f32; 1]) -> f32 {
    let l: f32 = w[0] * w[0] + w[1] * w[1];
    print(l);
    out[0] = l;
    return l;
}
fn main() -> i32 { return 0; }
";

/// A loss that prints and then fails an assertion, on its very first call (every seeded weight is
/// far below 1). The executable has written the `7` before the assertion stops it; the in-process
/// run must print it too, before its error, on both backends.
const FAILING: &str = "\
fn loss(w: [f32; 1], mut out: [f32; 1]) -> f32 {
    print(7);
    assert(w[0] > 1.0);
    let l: f32 = w[0] * w[0];
    out[0] = l;
    return l;
}
fn main() -> i32 { return 0; }
";

/// One training configuration: a source file and the `--train` flags it runs with.
struct Case {
    name: String,
    source: PathBuf,
    args: Vec<String>,
}

fn run_dir() -> PathBuf {
    common::repo_root().join("tests").join("run")
}

/// A corpus fixture's own `// RUN:` flags (its hand-derived `--train` invocation).
fn fixture_case(file: &str) -> (PathBuf, Vec<String>) {
    let source = run_dir().join(file);
    let text = std::fs::read_to_string(&source).expect("read the training fixture");
    let directives = common::parse_directives(&text, file).unwrap_or_else(|e| panic!("{e}"));
    assert!(
        directives.run_args.iter().any(|a| a == "--train"),
        "{file} is no longer a --train fixture: {:?}",
        directives.run_args
    );
    (source, directives.run_args)
}

fn cases(scratch: &Path) -> Vec<Case> {
    let mut models: Vec<(String, PathBuf, Vec<String>)> = [
        "autodiff_train_sumsq.wk",
        "autodiff_train_helper.wk",
        "autodiff_train_mlp.wk",
        "autodiff_train_parallel_region.wk",
    ]
    .iter()
    .map(|file| {
        let (source, args) = fixture_case(file);
        (file.trim_end_matches(".wk").to_string(), source, args)
    })
    .collect();
    let linear = scratch.join("linear_bias.wk");
    std::fs::write(&linear, LINEAR_BIAS).expect("write the GEMM + bias model");
    models.push((
        "linear_bias".into(),
        linear,
        [
            "--train",
            "--grad-wrt=1,2",
            "--train-steps=4",
            "--train-lr=0.05",
            "--train-seed=11",
        ]
        .map(String::from)
        .to_vec(),
    ));
    let printing = scratch.join("printing.wk");
    std::fs::write(&printing, PRINTING).expect("write the printing loss");
    models.push((
        "printing".into(),
        printing,
        [
            "--train",
            "--grad-wrt=0",
            "--train-steps=3",
            "--train-lr=0.25",
            "--train-seed=5",
        ]
        .map(String::from)
        .to_vec(),
    ));
    let mut out = Vec::new();
    for (name, source, args) in models {
        for level in ["-O0", "-O2"] {
            // SGD as the fixture states it, then AdamW with every update-policy field set.
            let sgd: &[&str] = &["--train-opt=sgd"];
            let adamw: &[&str] = &[
                "--train-opt=adamw",
                "--train-weight-decay=0.0625",
                "--train-clip=0.75",
                "--train-warmup=1",
                "--train-schedule=cosine",
            ];
            for (opt, extra) in [("sgd", sgd), ("adamw", adamw)] {
                let mut a = args.clone();
                a.push(level.to_string());
                a.extend(extra.iter().map(|s| s.to_string()));
                out.push(Case {
                    name: format!("{name}{level}-{opt}"),
                    source: source.clone(),
                    args: a,
                });
            }
        }
    }
    out
}

fn wukongc(dir: &Path, args: &[String], extra: &[&str], source: &Path) -> Output {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_wukongc"));
    // Autodiff differentiates the recognized kernels; the instrument would make every model refuse.
    cmd.env_remove(wukong_driver::NO_RECOGNIZE_ENV);
    // Link through a runtime cache under the target directory, never the user's own.
    cmd.env(
        wukong_driver::CACHE_DIR_ENV,
        Path::new(env!("CARGO_TARGET_TMPDIR")).join("wukong-runtime-cache"),
    );
    cmd.current_dir(dir)
        .args(args)
        .args(extra)
        .arg(source)
        .output()
        .expect("spawn wukongc")
}

/// What one case established.
enum Verdict {
    /// The executable printed exactly the interpreter's and the JIT's trajectory.
    Matched,
    /// The executable could not be linked on the C fallback (reported, never silent).
    Skipped(String),
    /// A disagreement, or a failure on the preferred link path.
    Failed(String),
}

fn check(case: &Case, scratch: &Path, link: &LinkPath) -> Verdict {
    let dir = scratch.join(&case.name);
    std::fs::create_dir_all(&dir).expect("create the case directory");
    let text = |o: &Output| String::from_utf8_lossy(&o.stdout).into_owned();
    let interp = wukongc(&dir, &case.args, &[], &case.source);
    let native = wukongc(&dir, &case.args, &["--backend=native"], &case.source);
    for (what, o) in [("interp", &interp), ("native", &native)] {
        if !o.status.success() {
            return Verdict::Failed(format!(
                "--train ({what}) failed: {}",
                String::from_utf8_lossy(&o.stderr)
            ));
        }
    }
    if text(&interp) != text(&native) {
        return Verdict::Failed(format!(
            "--train interp and native differ:\n{}\n---\n{}",
            text(&interp),
            text(&native)
        ));
    }
    let exe = dir.join(format!("train{}", std::env::consts::EXE_SUFFIX));
    let exe_arg = exe.to_string_lossy().into_owned();
    let built = wukongc(
        &dir,
        &case.args,
        &["--emit=exe", "-o", &exe_arg],
        &case.source,
    );
    if !built.status.success() || !exe.exists() {
        let stderr = String::from_utf8_lossy(&built.stderr).into_owned();
        return match link {
            LinkPath::CcFallback if stderr.contains("no usable `rustc` + rlib") => {
                Verdict::Skipped(stderr)
            }
            _ => Verdict::Failed(format!(
                "--train --emit=exe failed (exit {:?}):\n{stderr}",
                built.status.code()
            )),
        };
    }
    if !built.stdout.is_empty() {
        return Verdict::Failed(format!(
            "--train --emit=exe printed a trajectory instead of only building: {}",
            text(&built)
        ));
    }
    let ran = Command::new(&exe)
        .current_dir(&dir)
        .output()
        .expect("run the training executable");
    if ran.status.code() != Some(0) || !ran.stderr.is_empty() {
        return Verdict::Failed(format!(
            "the training executable exited {:?}: {}",
            ran.status.code(),
            String::from_utf8_lossy(&ran.stderr)
        ));
    }
    if ran.stdout != interp.stdout {
        return Verdict::Failed(format!(
            "the training executable's stdout differs from --train:\nexe:\n{}\n--train:\n{}",
            text(&ran),
            text(&interp)
        ));
    }
    Verdict::Matched
}

#[test]
fn train_exe_matches_train_on_interp_and_native() {
    let link = probe_link_path();
    if link == LinkPath::None {
        eprintln!(
            "train_exe: neither `rustc` + a `wukong_runtime` rlib nor a C compiler runs on this \
             box - skipped"
        );
        return;
    }
    // One directory per process: the driver's link intermediates and each executable live here.
    let scratch = std::env::temp_dir()
        .join("wukong_train_exe")
        .join(std::process::id().to_string());
    std::fs::create_dir_all(&scratch).expect("create the scratch directory");
    let cases = cases(&scratch);
    let workers = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(2)
        .clamp(1, 4);
    let next = std::sync::atomic::AtomicUsize::new(0);
    let verdicts = std::sync::Mutex::new(Vec::new());
    std::thread::scope(|s| {
        for _ in 0..workers {
            s.spawn(|| loop {
                let i = next.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                let Some(case) = cases.get(i) else { break };
                let verdict = check(case, &scratch, &link);
                verdicts.lock().unwrap().push((case.name.clone(), verdict));
            });
        }
    });
    let mut verdicts = verdicts.into_inner().unwrap();
    verdicts.sort_by(|a, b| a.0.cmp(&b.0));
    let (mut linked, mut skipped, mut failures) = (0usize, 0usize, Vec::new());
    for (name, verdict) in verdicts {
        match verdict {
            Verdict::Matched => linked += 1,
            Verdict::Skipped(why) => {
                skipped += 1;
                eprintln!("train_exe: skipped {name} (C fallback, no training loop):\n{why}");
            }
            Verdict::Failed(why) => failures.push(format!("--- {name} ---\n{why}")),
        }
    }
    assert!(
        failures.is_empty(),
        "{} of {} training executables disagree with `--train` (or failed to build on the \
         preferred link path):\n{}",
        failures.len(),
        cases.len(),
        failures.join("\n")
    );
    eprintln!(
        "train_exe: {linked} of {} training executables linked and matched `--train` (interp and \
         native) byte for byte, {skipped} skipped ({link:?})",
        cases.len()
    );
    if link == LinkPath::Rustc {
        assert_eq!(
            linked,
            cases.len(),
            "every case must link on the rustc path"
        );
    }
    if linked == 0 {
        panic!(
            "train_exe compared ZERO linked training executables against `--train` - the gate is \
             dark. A gate that cannot fail is not a gate."
        );
    }
    let _ = std::fs::remove_dir_all(&scratch);
}

/// `--train --emit=exe` refuses the file options by name (exit 2) and writes nothing: a training
/// executable reads and writes no files, and dropping the option silently would train on other
/// weights (or lose the trained ones) with exit 0.
#[test]
fn train_exe_refuses_file_options_by_name() {
    let scratch = std::env::temp_dir()
        .join("wukong_train_exe_refusals")
        .join(std::process::id().to_string());
    std::fs::create_dir_all(&scratch).expect("create the scratch directory");
    let (source, args) = fixture_case("autodiff_train_sumsq.wk");
    let exe = scratch.join(format!("refused{}", std::env::consts::EXE_SUFFIX));
    let exe_arg = exe.to_string_lossy().into_owned();
    let data = scratch.join("w.bin");
    std::fs::write(&data, [0u8; 32]).unwrap();
    for flag in ["--train-init", "--train-save"] {
        let option = format!("{flag}=0={}", data.display());
        let out = wukongc(
            &scratch,
            &args,
            &[&option, "--emit=exe", "-o", &exe_arg],
            &source,
        );
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert_eq!(out.status.code(), Some(2), "{flag}: {stderr}");
        assert!(stderr.contains(flag), "{flag}: {stderr}");
        assert!(out.stdout.is_empty() && !exe.exists(), "{flag}");
    }
    let _ = std::fs::remove_dir_all(&scratch);
}

/// `--depfile` names the training artifact as its rule's target, exactly as for any other
/// `--emit=obj|exe` build — a Make rule naming the depfile would never rebuild the artifact when an
/// input changes, and ninja (`deps = gcc`) rejects a depfile that does not mention its output. An
/// in-process `--train` writes no artifact, so its rule names the depfile itself, as `--run` does.
#[test]
fn train_artifact_depfile_names_the_artifact() {
    let scratch = std::env::temp_dir()
        .join("wukong_train_exe_depfile")
        .join(std::process::id().to_string());
    std::fs::create_dir_all(&scratch).expect("create the scratch directory");
    let (source, args) = fixture_case("autodiff_train_sumsq.wk");
    let rule = |extra: &[&str], depfile: &str| {
        let flag = format!("--depfile={depfile}");
        let mut all: Vec<&str> = extra.to_vec();
        all.push(&flag);
        let out = wukongc(&scratch, &args, &all, &source);
        assert!(
            out.status.success(),
            "{extra:?}: {}",
            String::from_utf8_lossy(&out.stderr)
        );
        std::fs::read_to_string(scratch.join(depfile)).expect("read the depfile")
    };
    // `--emit=obj` writes only the object (no link), so this needs no toolchain.
    let named = rule(&["--emit=obj", "-o", "train.o"], "named.d");
    assert!(named.starts_with("train.o:"), "{named}");
    let default = rule(&["--emit=obj"], "default.d");
    assert!(default.starts_with("autodiff_train_sumsq.o:"), "{default}");
    let in_process = rule(&[], "run.d");
    assert!(in_process.starts_with("run.d:"), "{in_process}");
    let _ = std::fs::remove_dir_all(&scratch);
}

/// A kernel that fails at run time keeps what it printed before the failure, on every path: the
/// training executable has written it to stdout as it happened, and `wukongc --train` (interp and
/// native) forwards the failing call's output before it reports the error. Exit code 1 and the
/// assertion message on all three.
#[test]
fn train_failure_keeps_what_the_kernel_printed() {
    let scratch = std::env::temp_dir()
        .join("wukong_train_exe_failure")
        .join(std::process::id().to_string());
    std::fs::create_dir_all(&scratch).expect("create the scratch directory");
    let source = scratch.join("failing.wk");
    std::fs::write(&source, FAILING).expect("write the failing loss");
    let args: Vec<String> = ["--train", "--train-steps=2", "--train-seed=3"]
        .map(String::from)
        .to_vec();
    let expect_failure = |what: &str, o: &Output| {
        let stderr = String::from_utf8_lossy(&o.stderr);
        assert_eq!(o.status.code(), Some(1), "{what}: {stderr}");
        assert!(stderr.contains("assertion failed"), "{what}: {stderr}");
        assert_eq!(
            String::from_utf8_lossy(&o.stdout),
            "7\n",
            "{what}: stdout before the failure"
        );
    };
    expect_failure("--train (interp)", &wukongc(&scratch, &args, &[], &source));
    expect_failure(
        "--train (native)",
        &wukongc(&scratch, &args, &["--backend=native"], &source),
    );
    if probe_link_path() == LinkPath::Rustc {
        let exe = scratch.join(format!("failing{}", std::env::consts::EXE_SUFFIX));
        let exe_arg = exe.to_string_lossy().into_owned();
        let built = wukongc(&scratch, &args, &["--emit=exe", "-o", &exe_arg], &source);
        assert!(
            built.status.success(),
            "{}",
            String::from_utf8_lossy(&built.stderr)
        );
        let ran = Command::new(&exe)
            .current_dir(&scratch)
            .output()
            .expect("run the training executable");
        expect_failure("the training executable", &ran);
    }
    let _ = std::fs::remove_dir_all(&scratch);
}

/// Without the rustc + runtime-rlib link (here: a copy of wukongc with no rlib beside it), a
/// training executable cannot be built, and the refusal is by name (exit 2, nothing written). Its
/// advice must be followable: an object from `--train --emit=obj` imports `wukong_rt_run_main` and
/// the print/assert entries, which only wukongc's generated runtime shim defines, so "link the object
/// against `wukong_runtime`" would end in an unresolved symbol.
#[test]
fn train_exe_without_the_rlib_link_is_refused_with_followable_advice() {
    let scratch = std::env::temp_dir()
        .join("wukong_train_exe_no_rlib")
        .join(std::process::id().to_string());
    let bin = scratch.join("bin");
    std::fs::create_dir_all(&bin).expect("create the scratch directory");
    let lone = bin.join(format!("wukongc{}", std::env::consts::EXE_SUFFIX));
    std::fs::copy(env!("CARGO_BIN_EXE_wukongc"), &lone).expect("copy wukongc");
    assert!(
        wukong_driver::runtime_rlib_in(&bin).is_none(),
        "the copy must have no runtime rlib beside it"
    );
    let (source, args) = fixture_case("autodiff_train_sumsq.wk");
    let exe = scratch.join(format!("refused{}", std::env::consts::EXE_SUFFIX));
    let out = Command::new(&lone)
        .current_dir(&scratch)
        .args(&args)
        .arg("--emit=exe")
        .arg("-o")
        .arg(&exe)
        .arg(&source)
        .env_remove(wukong_driver::NO_RECOGNIZE_ENV)
        .output()
        .expect("spawn the copied wukongc");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(2), "{stderr}");
    assert!(out.stdout.is_empty() && !exe.exists(), "{stderr}");
    for needle in [
        "no usable `rustc` + rlib",
        "run `wukongc --train` in process",
        "wukong_rt_run_main",
    ] {
        assert!(stderr.contains(needle), "missing `{needle}`: {stderr}");
    }
    assert!(
        !stderr.contains("link the object against"),
        "unfollowable advice: {stderr}"
    );
    let _ = std::fs::remove_dir_all(&scratch);
}
