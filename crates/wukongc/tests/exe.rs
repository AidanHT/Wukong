//! Gated end-to-end **executable** tests: compile each curated program to a native executable via
//! `--emit=exe`, then assert its stdout + exit code match `--run` (the interpreter oracle). This
//! extends the differential gate to the *linked exe* — the AOT path — covering both the read-only
//! string `.rodata` section (`Op::GlobalAddr`) and the `wukong_runtime` kernel linking.
//!
//! **A skip must be earned.** This gate used to treat *any* non-zero `--emit=exe` exit as "no
//! linker toolchain on this box" and pass green with the stderr thrown away — so a genuine failure
//! of the compiler's own link invocation (which `wukong_driver` deliberately reports as
//! `LinkOutcome::Failed` → `exit::COMPILE_ERROR`, precisely so it is *not* swallowed) was
//! indistinguishable from a missing toolchain, and the AOT gate could go permanently dark while
//! printing a benign note. It therefore now probes, up front and independently of any fixture,
//! which of the driver's two link paths is actually available here (see [`LinkPath`]), and only the
//! genuinely-absent-toolchain case may skip. Every skip prints the captured stderr.

use std::path::{Path, PathBuf};
use std::process::Command;

#[path = "common/mod.rs"]
mod common;

/// `<repo>/tests/run` (CARGO_MANIFEST_DIR = crates/wukongc).
fn run_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("tests")
        .join("run")
}

/// Which link path `wukongc --emit=exe` will take here. The rlib half of the decision is not
/// mirrored but *shared*: it calls the driver's own `runtime_rlib_in`, so the probe cannot drift
/// from `wukong_driver::rustc_link`. (It drifted once already — the probe hardcoded
/// `libwukong_runtime.rlib` next to the binary, which only a `cargo build` tree has, so under CI's
/// `cargo test --workspace` the probe reported `CcFallback`, every fixture skipped, and the gate
/// went dark.)
#[derive(Debug, PartialEq, Eq)]
enum LinkPath {
    /// `rustc` runs *and* the `wukong_runtime` rlib is reachable from the compiler binary's
    /// directory — the driver's preferred path, the one that is supposed to link every fixture here
    /// (it pulls in the `wukong_*` microkernels and links the string `.rodata` relocations). A
    /// failure on this path is a COMPILER bug, never an environment one.
    Rustc,
    /// No rlib (or no rustc), but a C compiler runs: the driver falls back to its small C runtime,
    /// which by construction resolves only `wukong_rt_*` — a program needing a `wukong_*` kernel or
    /// MSVC-object data relocations legitimately cannot be linked this way.
    CcFallback(String),
    /// Neither path is available: the whole gate is genuinely inapplicable here.
    None,
}

/// Probe the toolchain ONCE, before any fixture, so no fixture's own failure can be misread as the
/// toolchain being absent.
fn probe_link_path() -> LinkPath {
    let rustc_runs = Command::new("rustc")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false);
    // The driver locates the runtime rlib from the directory holding its own executable
    // (`current_exe().parent()`), which for the binary these tests spawn is `target/<profile>/`.
    // Ask the driver itself rather than re-deriving the rule: `runtime_rlib_in` knows that cargo
    // leaves the rlib in `deps/` under a hash-suffixed name unless a plain `cargo build` uplifted
    // it. NOTE this must be the *compiler binary's* directory, not the test binary's — the test
    // binary lives one level down in `deps/`.
    let rlib = Path::new(env!("CARGO_BIN_EXE_wukongc"))
        .parent()
        .and_then(wukong_driver::runtime_rlib_in)
        .is_some();
    if rustc_runs && rlib {
        return LinkPath::Rustc;
    }
    let cc = std::env::var("CC").unwrap_or_else(|_| "cc".to_string());
    let cc_runs = Command::new(&cc)
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false);
    if cc_runs {
        return LinkPath::CcFallback(cc);
    }
    LinkPath::None
}

/// Compile `src` to an executable via `--emit=exe` (intermediates land in `tmp`, not the source
/// tree) and run it. `Err` carries the compiler's captured stderr — the caller decides whether that
/// is a failure or an earned skip, and either way the text is reported.
fn build_and_run_exe(tmp: &Path, src: &Path, stem: &str) -> Result<(Option<i32>, String), String> {
    // `<stem>` plus the *host's* executable suffix — `.exe` on Windows, empty on unix — matching what
    // `wukong_driver::emit_native` names an unqualified `--emit=exe` output. Hardcoding `.exe` here
    // happened to work on Linux (we pass `-o`, so the driver writes exactly this path) but named the
    // ELF file a lie, and the gate would have stopped mirroring the driver the moment it stopped
    // passing `-o`.
    let exe = tmp.join(format!("{stem}{}", std::env::consts::EXE_SUFFIX));
    let _ = std::fs::remove_file(&exe);
    let emit = Command::new(env!("CARGO_BIN_EXE_wukongc"))
        .current_dir(tmp) // the `.o` / `_shim.rs` / `_rt.c` intermediates land here
        .arg("--emit=exe")
        .arg("-o")
        .arg(&exe)
        .arg(src)
        .output()
        .expect("failed to spawn wukongc --emit=exe");
    if !emit.status.success() || !exe.exists() {
        return Err(String::from_utf8_lossy(&emit.stderr).into_owned());
    }
    let out = Command::new(&exe)
        .output()
        .expect("failed to run the linked executable");
    Ok((
        out.status.code(),
        String::from_utf8_lossy(&out.stdout).into_owned(),
    ))
}

/// Run `src` on the interpreter (the oracle) for the comparison baseline.
fn run_interp(src: &Path) -> (Option<i32>, String) {
    let out = Command::new(env!("CARGO_BIN_EXE_wukongc"))
        .arg("--run")
        .arg(src)
        .output()
        .expect("failed to spawn wukongc --run");
    (
        out.status.code(),
        String::from_utf8_lossy(&out.stdout).into_owned(),
    )
}

/// The linked executable must produce byte-identical stdout + the same exit code as `--run`. Covers:
/// a returned/threaded string literal (`.rodata` via `Op::GlobalAddr` — the dangling-`*u8` fix), a
/// recognized transcendental kernel and a GEMM (both link the `wukong_runtime` microkernels into the
/// exe).
///
/// When the driver's preferred `rustc` link path is available (rustc + the runtime rlib), *every*
/// fixture must link: a failure there is the compiler's own link invocation breaking, and it fails
/// this test with the captured stderr. Only the C-runtime fallback — which cannot resolve the
/// `wukong_*` kernels by construction — may skip individual fixtures, and only with its stderr
/// printed. With no toolchain at all the whole gate skips.
#[test]
fn exe_matches_run() {
    let fixtures = [
        "string_return.wk",         // read-only string `.rodata`, returned/threaded `*u8`
        "transcendental_kernel.wk", // `wukong_vmath_f32` linked into the exe
        "tensor_matmul.wk",         // `wukong_sgemm` linked into the exe
        "heap_alloc.wk",            // `wukong_rt_alloc`/`wukong_rt_free` linked into the exe
        "print_float_display.wk",   // every float print path (`Display` digits, inf/NaN/-0, f16)
    ];

    let path = probe_link_path();
    if path == LinkPath::None {
        eprintln!(
            "exe_matches_run: neither `rustc` + a `wukong_runtime` rlib nor a C compiler runs on \
             this box — AOT gate skipped (this is not a failure; install rustc and build the \
             runtime rlib, or a C compiler, to exercise it)"
        );
        return;
    }

    // One scratch directory per PROCESS, not one shared name: the driver writes `{stem}.o` and
    // `{stem}_shim.rs` next to the link and DELETES the shim after linking, so two concurrent
    // `cargo test -p wukongc` invocations (the documented multi-agent workflow does exactly this)
    // would clobber and unlink each other's intermediates. `tests/run.rs` solved the identical
    // collision the same way.
    let tmp = std::env::temp_dir()
        .join("wukong_exe_gate")
        .join(std::process::id().to_string());
    std::fs::create_dir_all(&tmp).expect("create temp dir for exe gate");

    let mut failures: Vec<String> = Vec::new();
    let mut linked = 0usize;
    let mut skipped = 0usize;
    for name in fixtures {
        let src = run_dir().join(name);
        if !src.exists() {
            continue;
        }
        let stem = Path::new(name)
            .file_stem()
            .unwrap()
            .to_string_lossy()
            .into_owned();
        match build_and_run_exe(&tmp, &src, &stem) {
            Ok(exe_result) => {
                linked += 1;
                let run_result = run_interp(&src);
                assert_eq!(
                    exe_result, run_result,
                    "{name}: linked-exe output/exit differs from `--run`"
                );
            }
            Err(stderr) => match &path {
                LinkPath::Rustc => failures.push(format!("--- {name} ---\n{}", stderr.trim_end())),
                LinkPath::CcFallback(cc) => {
                    skipped += 1;
                    eprintln!(
                        "exe_matches_run: skipping {name} — no `wukong_runtime` rlib is reachable \
                         from the compiler binary (neither the uplifted `libwukong_runtime.rlib` \
                         nor a hash-suffixed one in `deps/`), so the driver fell back to the \
                         `{cc}` C runtime, which cannot resolve the `wukong_*` kernels. Compiler \
                         stderr:\n{}",
                        stderr.trim_end()
                    );
                }
                LinkPath::None => unreachable!("handled above"),
            },
        }
    }

    assert!(
        failures.is_empty(),
        "`wukongc --emit=exe` failed for {} of the AOT fixtures while its PREFERRED link path is \
         available here (rustc runs and `wukong_driver::runtime_rlib_in` found the runtime rlib), \
         so this is the compiler's own link invocation breaking — not a missing toolchain. Fix \
         `rustc_link`/`WUKONG_RT_SHIM` in crates/wukong_driver/src/lib.rs. If the stderr below \
         reports `LNK1136`/`LNK1107: invalid or corrupt file` on libwukong_runtime.rlib followed by \
         EVERY `wukong_*` symbol unresolved, the rlib holds LLVM bitcode rather than native objects \
         (the workspace `[profile.release]` sets `lto = \"thin\"`, so cargo builds the runtime with \
         `-C linker-plugin-lto`) and `rlib_holds_bitcode_members` failed to classify it — that \
         probe is what makes `rustc_link` add `-C lto=thin -C opt-level=3`, and without it no \
         release-profile executable links at all. If instead exactly one or two `wukong_*` symbols \
         are unresolved, the rlib `runtime_rlib_in` picked is simply STALE — rebuild the workspace. \
         (This gate previously reported every one of these states as \"no linker toolchain to \
         produce an exe\" and passed green, which is why the break went unreported. Note it must be \
         run under `--release` too: the profiles hand the linker archives of different shapes, and \
         the release one was 0/4 for the project's whole history while debug was 4/4.)\n{}",
        failures.len(),
        failures.join("\n")
    );

    eprintln!("exe_matches_run: {linked} fixture(s) linked and matched `--run`, {skipped} skipped");
    if linked == 0 {
        panic!(
            "exe_matches_run compared ZERO linked executables against `--run` — the AOT gate is \
             dark. A gate that cannot fail is not a gate."
        );
    }

    // Best effort: a lingering temp dir must never fail the test.
    let _ = std::fs::remove_dir_all(&tmp);
}

/// One process's observable result: exit code, stdout, stderr.
type Outcome = (Option<i32>, String, String);

fn outcome(cmd: &mut Command) -> Outcome {
    let out = cmd.output().expect("failed to spawn");
    (
        out.status.code(),
        String::from_utf8_lossy(&out.stdout).into_owned(),
        String::from_utf8_lossy(&out.stderr).into_owned(),
    )
}

/// Every `DEFAULT_STRIDE`-th fixture of the sorted corpus is linked by default (about 50 of ~395),
/// a deterministic slice that keeps the always-on cost bounded. `WUKONG_EXE_CORPUS=all` links the
/// whole corpus (the integration gate's setting), `WUKONG_EXE_CORPUS=<n>` every n-th.
const DEFAULT_STRIDE: usize = 8;

fn corpus_slice() -> (Vec<PathBuf>, usize) {
    let all = common::collect_wk(&run_dir());
    let stride = match std::env::var("WUKONG_EXE_CORPUS").ok().as_deref() {
        Some("all") => 1,
        Some(n) => n
            .parse::<usize>()
            .ok()
            .filter(|&n| n > 0)
            .unwrap_or(DEFAULT_STRIDE),
        None => DEFAULT_STRIDE,
    };
    let total = all.len();
    (all.into_iter().step_by(stride).collect(), total)
}

/// What linking and running one corpus program established.
enum Verdict {
    /// The linked executable produced `--run`'s stdout and exit code.
    Matched,
    /// A documented capability decline, not an AOT defect (the reason is reported).
    Declined(String),
    /// The AOT path disagreed with the oracle, or failed to link a program it should link.
    Mismatch(String),
}

fn short(s: &str) -> String {
    const MAX: usize = 400;
    if s.len() <= MAX {
        s.to_string()
    } else {
        let mut end = MAX;
        while !s.is_char_boundary(end) {
            end -= 1;
        }
        format!("{}... ({} bytes)", &s[..end], s.len())
    }
}

/// `--emit=exe` one corpus program, run it, and compare with `--run`. Each process gets its own
/// working directory, so the file-I/O fixtures (relative `io_test_*.bin` paths) neither collide
/// nor see each other's files.
fn check_corpus_program(src: &Path, tmp: &Path, link: &LinkPath) -> Verdict {
    let stem = src.file_stem().unwrap().to_string_lossy().into_owned();
    let dir = tmp.join(&stem);
    let (emit_dir, exe_dir, oracle_dir, native_dir) = (
        dir.join("emit"),
        dir.join("exe"),
        dir.join("run"),
        dir.join("native"),
    );
    for d in [&emit_dir, &exe_dir, &oracle_dir, &native_dir] {
        std::fs::create_dir_all(d).expect("create per-program scratch dir");
    }
    let exe = emit_dir.join(format!("{stem}{}", std::env::consts::EXE_SUFFIX));
    let emitted = outcome(
        Command::new(env!("CARGO_BIN_EXE_wukongc"))
            .current_dir(&emit_dir)
            .arg("--emit=exe")
            .arg("-o")
            .arg(&exe)
            .arg(src),
    );
    if emitted.0 != Some(0) || !exe.exists() {
        // The JIT is the same Cranelift codegen as the object: a program it cannot compile either
        // is a native-backend limitation (with its own diagnostic), not an AOT defect.
        let native = outcome(
            Command::new(env!("CARGO_BIN_EXE_wukongc"))
                .current_dir(&native_dir)
                .arg("--error-format=json")
                .arg("--run")
                .arg("--backend=native")
                .arg(src),
        );
        // Only a native *compile* failure is a decline: it is rendered as a catalogued `C0xxx`
        // diagnostic or an internal-compiler-error (`bug`) record. A program that compiles and then
        // exits non-zero with no stdout (a failed assertion, an `EXPECT-EXIT: 1` fixture) must not
        // excuse an AOT link failure that happens to share its exit code.
        let native_declined =
            native.2.contains("\"code\":\"C0") || native.2.contains("\"severity\":\"bug\"");
        if native_declined && native.0 == emitted.0 && native.1.is_empty() {
            return Verdict::Declined(format!(
                "the native backend declines it too (exit {:?}): {}",
                native.0,
                native.2.lines().next().unwrap_or("")
            ));
        }
        if let LinkPath::CcFallback(cc) = link {
            return Verdict::Declined(format!(
                "the `{cc}` C fallback cannot link it (no `wukong_*` kernels): {}",
                emitted.2.lines().last().unwrap_or("")
            ));
        }
        return Verdict::Mismatch(format!(
            "`--emit=exe` failed (exit {:?}) while `--run --backend=native` did not:\n{}",
            emitted.0,
            short(&emitted.2)
        ));
    }
    let ran = outcome(Command::new(&exe).current_dir(&exe_dir));
    let oracle = outcome(
        Command::new(env!("CARGO_BIN_EXE_wukongc"))
            .current_dir(&oracle_dir)
            .arg("--run")
            .arg(src),
    );
    // `--run` reports a failed assertion as its result and discards the stdout the program had
    // produced; the executable has already written it. Exit code, message and the prefix agree.
    if oracle.0 == Some(1) && oracle.2.contains("assertion failed") {
        return if ran.0 == Some(1)
            && ran.1.starts_with(&oracle.1)
            && ran.2.contains("assertion failed")
        {
            Verdict::Matched
        } else {
            Verdict::Mismatch(format!(
                "assertion failure differs: exe exit {:?} stdout {:?} stderr {:?}",
                ran.0,
                short(&ran.1),
                short(&ran.2)
            ))
        };
    }
    if ran.0 == oracle.0 && ran.1 == oracle.1 {
        Verdict::Matched
    } else {
        Verdict::Mismatch(format!(
            "exe = (exit {:?}, stdout {:?})\n    run = (exit {:?}, stdout {:?})",
            ran.0,
            short(&ran.1),
            oracle.0,
            short(&oracle.1)
        ))
    }
}

/// P7.19 / P11.9: the linked-executable path against the oracle on a deterministic slice of the
/// whole corpus, not four curated programs. The object path is a different Cranelift configuration
/// from the JIT (PIC, real relocations for `.rodata` strings and runtime symbols, the installed
/// raw-AVX2 kernel bytes) and a different runtime binding (the rustc shim or the C runtime instead
/// of in-process Rust functions), so `native_matches_interpreter` does not cover it.
///
/// Only documented capability declines may skip a program: one the native backend cannot compile
/// at all (the JIT twin fails identically), and on the C fallback a program that needs a
/// `wukong_*` kernel. On the preferred rustc path at least 90% of the slice must link and match,
/// so a decline cannot quietly grow into the norm.
#[test]
fn exe_matches_run_on_a_corpus_slice() {
    let link = probe_link_path();
    if link == LinkPath::None {
        eprintln!("exe_matches_run_on_a_corpus_slice: no link toolchain on this box - skipped");
        return;
    }
    let (programs, total) = corpus_slice();
    assert!(!programs.is_empty(), "the corpus slice is empty");
    let tmp = std::env::temp_dir()
        .join("wukong_exe_corpus")
        .join(std::process::id().to_string());
    std::fs::create_dir_all(&tmp).expect("create corpus scratch dir");

    let workers = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(2)
        .clamp(1, 6);
    let next = std::sync::atomic::AtomicUsize::new(0);
    let verdicts: std::sync::Mutex<Vec<(String, Verdict)>> = std::sync::Mutex::new(Vec::new());
    std::thread::scope(|s| {
        for _ in 0..workers {
            s.spawn(|| loop {
                let i = next.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                let Some(src) = programs.get(i) else { break };
                let name = src.file_name().unwrap().to_string_lossy().into_owned();
                let verdict = check_corpus_program(src, &tmp, &link);
                verdicts.lock().unwrap().push((name, verdict));
            });
        }
    });
    let mut verdicts = verdicts.into_inner().unwrap();
    verdicts.sort_by(|a, b| a.0.cmp(&b.0));

    let (mut matched, mut declined, mut mismatches) = (0usize, Vec::new(), Vec::new());
    for (name, verdict) in &verdicts {
        match verdict {
            Verdict::Matched => matched += 1,
            Verdict::Declined(why) => declined.push(format!("{name}: {why}")),
            Verdict::Mismatch(why) => mismatches.push(format!("--- {name} ---\n    {why}")),
        }
    }
    for d in &declined {
        eprintln!("exe_matches_run_on_a_corpus_slice: declined {d}");
    }
    eprintln!(
        "exe_matches_run_on_a_corpus_slice: {matched} of {} sliced programs ({total} in the \
         corpus) linked and matched `--run`, {} declined",
        verdicts.len(),
        declined.len()
    );
    assert!(
        mismatches.is_empty(),
        "the linked executable disagrees with `--run` (or failed to link) for {} program(s):\n{}",
        mismatches.len(),
        mismatches.join("\n")
    );
    assert!(matched > 0, "the AOT corpus slice linked nothing");
    if link == LinkPath::Rustc {
        assert!(
            matched * 10 >= verdicts.len() * 9,
            "only {matched} of {} programs linked and matched on the preferred rustc path; the \
             declines above are no longer the exception",
            verdicts.len()
        );
    }
    let _ = std::fs::remove_dir_all(&tmp);
}

/// P13.19: the C fallback reached the way a user reaches it - a `wukongc` with no runtime rlib
/// beside it - not by calling the driver's private link function. A copy of the compiler in an
/// empty directory has no `libwukong_runtime.rlib` and no `deps/`, so its rustc link is
/// unavailable and `--emit=exe` must announce and take the `cc` path, and the executable must
/// print exactly what `--run` prints: floats in Rust's `Display` form (not `%g`), integers,
/// unsigned integers, and the exit code.
#[test]
fn cc_fallback_discovered_from_the_cli_matches_run() {
    let cc = std::env::var("CC").unwrap_or_else(|_| "cc".to_string());
    let cc_runs = Command::new(&cc)
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false);
    if !cc_runs {
        eprintln!("cc_fallback_discovered_from_the_cli_matches_run: no `{cc}` - skipped");
        return;
    }
    let tmp = std::env::temp_dir()
        .join("wukong_exe_cc_fallback")
        .join(std::process::id().to_string());
    std::fs::create_dir_all(&tmp).expect("create cc-fallback scratch dir");
    let bin = tmp.join(format!("wukongc{}", std::env::consts::EXE_SUFFIX));
    std::fs::copy(env!("CARGO_BIN_EXE_wukongc"), &bin).expect("copy wukongc");
    let src = tmp.join("cc_floats.wk");
    std::fs::write(
        &src,
        "fn main() -> i32 {
    let x: f64 = 7.25;
    print(x % 2.0);
    let a: f32 = 0.1;
    print(a);
    let third: f64 = 1.0 / 3.0;
    print(third);
    print(-7);
    let big: u64 = 18446744073709551615;
    print(big);
    assert(x > 7.0);
    return 3;
}
",
    )
    .unwrap();
    let exe = tmp.join(format!("cc_floats{}", std::env::consts::EXE_SUFFIX));
    let emitted = outcome(
        Command::new(&bin)
            .current_dir(&tmp)
            .arg("--emit=exe")
            .arg("-o")
            .arg(&exe)
            .arg(&src),
    );
    assert_eq!(
        emitted.0,
        Some(0),
        "cc fallback link failed:\n{}",
        emitted.2
    );
    assert!(
        emitted.2.contains("note: no usable `rustc`"),
        "the fallback must say it is being taken:\n{}",
        emitted.2
    );
    let ran = outcome(Command::new(&exe).current_dir(&tmp));
    // Hand-derived: 7.25 % 2.0 = 1.25 exactly; f32 0.1 widens to 0.100000001490116119384765625,
    // whose shortest round-trip f64 digits are 0.10000000149011612; 1/3 rounds to the double whose
    // shortest digits are 0.3333333333333333; u64::MAX prints unsigned.
    let expected = "1.25\n0.10000000149011612\n0.3333333333333333\n-7\n18446744073709551615\n";
    assert_eq!((ran.0, ran.1.as_str()), (Some(3), expected), "{}", ran.2);
    let oracle = outcome(
        Command::new(env!("CARGO_BIN_EXE_wukongc"))
            .current_dir(&tmp)
            .arg("--run")
            .arg(&src),
    );
    assert_eq!((oracle.0, oracle.1.as_str()), (Some(3), expected));
    let _ = std::fs::remove_dir_all(&tmp);
}
