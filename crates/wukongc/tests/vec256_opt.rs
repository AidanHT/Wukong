//! The optimizer's 256-bit kernels through the real binary: `tests/run/vec256_opt_lanes.wk`'s
//! three loops call helpers, so the front end's vectorizer leaves them scalar (`--emit=mir-high`
//! holds no `veckernel #`), and `wukong_opt::vectorize` hands each to a raw-AVX2 recipe at `-O2`
//! (`--emit=mir`). Every switch that turns the path off must turn it off for the optimizer too:
//! `--backend=gpu-native` (whose PTX lowering declines a `VecKernelCall`), a `--target-cpu` below
//! x86-64-v3 (the recipe is AVX2+FMA machine code with no runtime dispatch) and the
//! `WUKONG_P4_NO_256` kill-switch.
//!
//! The recipe exists only where `wukong_mir::host_supports_vec_kernels()` is true (Win64 with
//! AVX2+FMA3); elsewhere every count below is zero, which the test asserts instead.

use std::path::{Path, PathBuf};
use std::process::Command;

/// Variables that change what the compiler emits; the child inherits none of them unless a case
/// sets one itself.
const COMPILER_ENV: &[&str] = &[
    "WUKONG_NO_RECOGNIZE",
    "WUKONG_NO_VECTORIZE",
    "WUKONG_NO_UNROLL",
    "WUKONG_NO_IV_RAMP",
    "WUKONG_NO_SLP",
    "WUKONG_P4_NO_256",
    "WUKONG_VEC256_MIN_TRIP",
    "WUKONG_VEC_TRACE",
    "WUKONG_SLP_TRACE",
    "WUKONG_MAX_ISA",
];

fn fixture() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/run/vec256_opt_lanes.wk")
}

/// `veckernel #` calls in the MIR `wukongc args… <fixture>` prints, with `env` set.
fn kernels(args: &[&str], env: &[(&str, &str)]) -> usize {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_wukongc"));
    for v in COMPILER_ENV {
        cmd.env_remove(v);
    }
    cmd.envs(env.iter().copied()).args(args).arg(fixture());
    let out = cmd.output().expect("spawn wukongc");
    assert!(
        out.status.success(),
        "wukongc {args:?} failed:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8_lossy(&out.stdout)
        .matches("veckernel #")
        .count()
}

/// Can the recipe exist in a child of this process? The kill-switch is scrubbed from the child, so
/// only the host decides.
fn host() -> bool {
    wukong_mir::host_supports_vec_kernels()
}

#[test]
fn the_optimizer_builds_the_kernels_the_front_end_declined() {
    assert_eq!(
        kernels(&["--emit=mir-high"], &[]),
        0,
        "the front end now vectorizes these loops itself, so the fixture no longer tests the \
         optimizer's kernels"
    );
    let want = if host() { 3 } else { 0 };
    assert_eq!(kernels(&["--emit=mir", "-O2"], &[]), want);
    // `-O3` is `-O2`.
    assert_eq!(kernels(&["--emit=mir", "-O3"], &[]), want);
    // `-O1` runs no vectorizer.
    assert_eq!(kernels(&["--emit=mir", "-O1"], &[]), 0);
}

#[test]
fn every_switch_that_turns_the_recipes_off_turns_off_the_optimizers() {
    for (what, args, env) in [
        (
            "gpu-native",
            &["--emit=mir", "-O2", "--backend=gpu-native"][..],
            &[][..],
        ),
        (
            "x86-64-v2",
            &["--emit=mir", "-O2", "--target-cpu=x86-64-v2"][..],
            &[][..],
        ),
        (
            "WUKONG_P4_NO_256",
            &["--emit=mir", "-O2"][..],
            &[("WUKONG_P4_NO_256", "1")][..],
        ),
    ] {
        assert_eq!(kernels(args, env), 0, "{what}");
    }
}

/// A `--passes` list builds the vectorizer the backend asked for, as the standard pipeline does.
#[test]
fn a_pass_list_follows_the_backend() {
    let list = "--passes=inline,sroa,mem2reg,simplify,simplify-cfg,simplify-phis,dce,cse,\
                loop-canon,licm,bce,vectorize";
    let want = if host() { 3 } else { 0 };
    assert_eq!(kernels(&["--emit=mir", list], &[]), want);
    assert_eq!(
        kernels(&["--emit=mir", list, "--backend=gpu-native"], &[]),
        0
    );
}
