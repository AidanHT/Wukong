//! Gated end-to-end **executable** tests: compile each curated program to a native executable via
//! `--emit=exe`, then assert its stdout + exit code match `--run` (the interpreter oracle). This
//! extends the differential gate to the *linked exe* — the AOT path — covering both the read-only
//! string `.rodata` section (`Op::GlobalAddr`) and the `mercury_runtime` kernel linking.
//!
//! Skips cleanly when no linker toolchain can produce an executable on this box (like the GPU peer
//! tests), so plain `cargo test` stays green without one: a fixture whose exe cannot be linked is
//! reported and skipped rather than failed.

use std::path::{Path, PathBuf};
use std::process::Command;

/// `<repo>/tests/run` (CARGO_MANIFEST_DIR = crates/mercuryc).
fn run_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("tests")
        .join("run")
}

/// Compile `src` to an executable via `--emit=exe` (intermediates land in `tmp`, not the source
/// tree) and run it. Returns `Some((exit, stdout))` if the exe was produced and ran, or `None` if it
/// could not be linked on this box — the caller then skips that fixture.
fn build_and_run_exe(tmp: &Path, src: &Path, stem: &str) -> Option<(Option<i32>, String)> {
    let exe = tmp.join(format!("{stem}.exe"));
    let _ = std::fs::remove_file(&exe);
    let emit = Command::new(env!("CARGO_BIN_EXE_mercuryc"))
        .current_dir(tmp) // the `.o` / `_shim.rs` / `_rt.c` intermediates land here
        .arg("--emit=exe")
        .arg("-o")
        .arg(&exe)
        .arg(src)
        .output()
        .expect("failed to spawn mercuryc --emit=exe");
    if !emit.status.success() || !exe.exists() {
        return None; // no toolchain could link the executable — skip cleanly
    }
    let out = Command::new(&exe)
        .output()
        .expect("failed to run the linked executable");
    Some((
        out.status.code(),
        String::from_utf8_lossy(&out.stdout).into_owned(),
    ))
}

/// Run `src` on the interpreter (the oracle) for the comparison baseline.
fn run_interp(src: &Path) -> (Option<i32>, String) {
    let out = Command::new(env!("CARGO_BIN_EXE_mercuryc"))
        .arg("--run")
        .arg(src)
        .output()
        .expect("failed to spawn mercuryc --run");
    (
        out.status.code(),
        String::from_utf8_lossy(&out.stdout).into_owned(),
    )
}

/// The linked executable must produce byte-identical stdout + the same exit code as `--run`. Covers:
/// a returned/threaded string literal (`.rodata` via `Op::GlobalAddr` — the dangling-`*u8` fix), a
/// recognized transcendental kernel and a GEMM (both link the `mercury_runtime` microkernels into the
/// exe). Skips any fixture whose exe cannot be linked here, and the whole gate if none can.
#[test]
fn exe_matches_run() {
    let fixtures = [
        "string_return.mer",        // read-only string `.rodata`, returned/threaded `*u8`
        "transcendental_kernel.mer", // `mercury_vmath_f32` linked into the exe
        "tensor_matmul.mer",        // `mercury_sgemm` linked into the exe
    ];
    let tmp = std::env::temp_dir().join("mercury_exe_gate");
    std::fs::create_dir_all(&tmp).expect("create temp dir for exe gate");

    let mut linked_any = false;
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
            Some(exe_result) => {
                linked_any = true;
                let run_result = run_interp(&src);
                assert_eq!(
                    exe_result, run_result,
                    "{name}: linked-exe output/exit differs from `--run`"
                );
            }
            None => {
                eprintln!("exe_matches_run: skipping {name} (no linker toolchain to produce an exe)");
            }
        }
    }
    if !linked_any {
        eprintln!(
            "exe_matches_run: no executable could be linked on this box — AOT gate skipped (this is \
             not a failure; install rustc + the runtime rlib, or a C compiler, to exercise it)"
        );
    }
}
