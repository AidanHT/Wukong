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
    "WUKONG_NO_LANE_SERIAL",
    "WUKONG_NO_OUTER_VEC",
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
    kernels_in(&fixture(), args, env)
}

/// [`kernels`] for the program at `path`.
fn kernels_in(path: &Path, args: &[&str], env: &[(&str, &str)]) -> usize {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_wukongc"));
    for v in COMPILER_ENV {
        cmd.env_remove(v);
    }
    cmd.envs(env.iter().copied()).args(args).arg(path);
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

/// A loss whose forward is a recognized 4096-element SUM: its backward is the tape's
/// `fill(dx, 4096, g)`, a plain f32 splat-store loop the optimizer hands to a raw-AVX2 recipe.
const SUM_LOSS: &str = "module sum_loss\n\
    \n\
    @parallel\n\
    fn loss(x: [f32; 4096], mut out: [f32; 1]) -> f32 {\n\
        let mut l: f32 = 0.0;\n\
        for i in 0..4096 { l = l + x[i]; }\n\
        out[0] = l;\n\
        return l;\n\
    }\n\
    \n\
    fn main() -> i32 { return 0; }\n";

/// Run `wukongc args… <path>` with every compiler variable scrubbed.
fn wukongc(args: &[&str], path: &Path) -> std::process::Output {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_wukongc"));
    for v in COMPILER_ENV {
        cmd.env_remove(v);
    }
    cmd.args(args).arg(path).output().expect("spawn wukongc")
}

/// The backward `--emit=grad`, `--grad-of` and `--train` generate is optimized by a call of its
/// own, after the forward, and it follows the same switch. It used to run the optimizer's default
/// pipeline, which has the kernels on, so under `--target-cpu=x86-64-v2` the backward carried one.
/// `--emit=grad` printed it, and `--emit=obj --grad-of` and `--train --emit=obj` failed Cranelift's
/// target check (`function 'loss_grad' carries raw AVX2+FMA vector kernels`).
#[test]
fn the_generated_backward_follows_the_switch() {
    let dir = std::env::temp_dir().join(format!("wukong-vec256-opt-grad-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("create scratch dir");
    let src = dir.join("sum_loss.wk");
    std::fs::write(&src, SUM_LOSS).expect("write source");

    let grad_kernels = |extra: &[&str]| -> usize {
        let mut args = vec!["--emit=grad", "-O2"];
        args.extend_from_slice(extra);
        let out = wukongc(&args, &src);
        assert!(
            out.status.success(),
            "wukongc {args:?} failed:\n{}",
            String::from_utf8_lossy(&out.stderr)
        );
        String::from_utf8_lossy(&out.stdout)
            .matches("veckernel #")
            .count()
    };
    assert_eq!(grad_kernels(&["--target-cpu=x86-64-v2"]), 0, "x86-64-v2");
    assert_eq!(grad_kernels(&["--backend=gpu-native"]), 0, "gpu-native");
    // Where the recipe can exist the same backward takes one, so the zeros are the switch's.
    if host() {
        assert!(
            grad_kernels(&[]) > 0,
            "the backward's fill no longer takes a kernel; the zeros above prove nothing"
        );
    }

    for (what, args) in [
        ("--grad-of", &["--emit=obj", "--grad-of=loss"][..]),
        ("--train", &["--emit=obj", "--train"][..]),
    ] {
        let obj = dir.join(format!("{}.o", what.trim_start_matches('-')));
        let obj_arg = obj.to_str().expect("utf-8 scratch path");
        let mut full = vec!["-O2", "--target-cpu=x86-64-v2", "-o", obj_arg];
        full.extend_from_slice(args);
        let out = wukongc(&full, &src);
        assert!(
            out.status.success(),
            "{what}: wukongc {full:?} failed:\n{}",
            String::from_utf8_lossy(&out.stderr)
        );
        assert!(obj.is_file(), "{what}: no object at {}", obj.display());
    }
    let _ = std::fs::remove_dir_all(&dir);
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

/// `tests/run/vec256_f16_lanes.wk`'s five f16 loops take F16C kernels at `-O2` (an f16 element is
/// an `f32` lane between its conversions), the front end builds none, and every switch that turns
/// the recipes off — gpu-native, a target below x86-64-v3, the kill-switch — leaves them on the
/// 128-bit `<4 x f16>` loop. A host without F16C builds none either.
#[test]
fn f16_loops_take_f16c_kernels_and_follow_every_switch() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/run/vec256_f16_lanes.wk");
    let want = if host() && wukong_mir::host_supports_f16c_kernels() {
        5
    } else {
        0
    };
    assert_eq!(kernels_in(&path, &["--emit=mir-high"], &[]), 0);
    assert_eq!(kernels_in(&path, &["--emit=mir", "-O2"], &[]), want);
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
        assert_eq!(kernels_in(&path, args, env), 0, "{what}");
    }
}

/// The 128-bit form of the same loops through the real binary: `tests/run/vec_f16_lanes.wk`'s five
/// f16 loops print as `<4 x f16>` at `-O2` — with the 256-bit kernels off (`x86-64-v2`), where the
/// vector loop is all there is, and with them on, where it is the kernel's remainder — and as no
/// vector at all before optimization.
#[test]
fn f16_loops_print_as_four_f16_lanes() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/run/vec_f16_lanes.wk");
    let mir = |args: &[&str]| -> String {
        let out = wukongc(args, &path);
        assert!(
            out.status.success(),
            "wukongc {args:?} failed:\n{}",
            String::from_utf8_lossy(&out.stderr)
        );
        String::from_utf8_lossy(&out.stdout).into_owned()
    };
    assert!(!mir(&["--emit=mir-high"]).contains("x f16>"));
    for args in [
        &["--emit=mir", "-O2", "--target-cpu=x86-64-v2"][..],
        &["--emit=mir", "-O2"][..],
    ] {
        let text = mir(args);
        assert!(
            text.contains("load <4 x f16>") && text.contains("fptrunc"),
            "{args:?}: no <4 x f16> loop"
        );
    }
}
