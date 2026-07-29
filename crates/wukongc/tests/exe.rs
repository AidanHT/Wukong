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

/// `<repo>/tests/run` (CARGO_MANIFEST_DIR = crates/wukongc).
fn run_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("tests")
        .join("run")
}

/// Which link path `wukongc --emit=exe` will take here — mirroring the driver's own decision
/// procedure in `wukong_driver::emit_native`/`rustc_link` so the gate can tell a real break from a
/// missing toolchain.
#[derive(Debug, PartialEq, Eq)]
enum LinkPath {
    /// `rustc` runs *and* `libwukong_runtime.rlib` sits next to the compiler binary — the driver's
    /// preferred path, the one that is supposed to link every fixture here (it pulls in the
    /// `wukong_*` microkernels and links the string `.rodata` relocations). A failure on this path
    /// is a COMPILER bug, never an environment one.
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
    // The driver locates the runtime rlib next to its own executable (`current_exe().parent()`),
    // which for these tests is `target/<profile>/`.
    let rlib = Path::new(env!("CARGO_BIN_EXE_wukongc"))
        .parent()
        .map(|d| d.join("libwukong_runtime.rlib"))
        .map(|p| p.exists())
        .unwrap_or(false);
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
fn build_and_run_exe(
    tmp: &Path,
    src: &Path,
    stem: &str,
) -> Result<(Option<i32>, String), String> {
    let exe = tmp.join(format!("{stem}.exe"));
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
        "string_return.wk",        // read-only string `.rodata`, returned/threaded `*u8`
        "transcendental_kernel.wk", // `wukong_vmath_f32` linked into the exe
        "tensor_matmul.wk",        // `wukong_sgemm` linked into the exe
        "heap_alloc.wk",           // `wukong_rt_alloc`/`wukong_rt_free` linked into the exe
    ];

    let path = probe_link_path();
    if path == LinkPath::None {
        eprintln!(
            "exe_matches_run: neither `rustc` + libwukong_runtime.rlib nor a C compiler runs on \
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
                LinkPath::Rustc => failures.push(format!(
                    "--- {name} ---\n{}",
                    stderr.trim_end()
                )),
                LinkPath::CcFallback(cc) => {
                    skipped += 1;
                    eprintln!(
                        "exe_matches_run: skipping {name} — libwukong_runtime.rlib is not next to \
                         the compiler binary, so the driver fell back to the `{cc}` C runtime, \
                         which cannot resolve the `wukong_*` kernels. Compiler stderr:\n{}",
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
         available here (rustc runs and libwukong_runtime.rlib sits next to the compiler binary), \
         so this is the compiler's own link invocation breaking — not a missing toolchain. Fix \
         `rustc_link`/`WUKONG_RT_SHIM` in crates/wukong_driver/src/lib.rs. If the stderr below \
         reports `LNK1107: invalid or corrupt file` on libwukong_runtime.rlib, the rlib holds \
         LLVM bitcode rather than native objects (the workspace `[profile.release]` sets \
         `lto = \"thin\"`), so `rustc_link` has to be told to consume it accordingly — that is the \
         release-profile shape of this break, and every `wukong_*` symbol then reads as \
         unresolved. (This gate previously reported exactly this state as \"no linker toolchain to \
         produce an exe\" and passed green, which is why the break went unreported.)\n{}",
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
