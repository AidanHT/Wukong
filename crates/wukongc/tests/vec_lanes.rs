//! The loop vectorizer's per-lane shapes through the real binary, at `--emit=mir -O2`: a call to a
//! side-effect-free function replicated per lane beside widened work (`tests/run/vec7_call_lanes.wk`),
//! a checked gather tested per group (`vec7_checked_gather.wk`), the lane-serial regions of a
//! read-modify-write histogram, two scatters into one object, a scatter beside a store to its object
//! and a gather beside a store it may read (`vec7_lane_serial.wk`), and two-deep nests widened
//! across their rows (`vec7_outer_loops.wk`). Each fixture prints every lane
//! against hand-derived values, and the differential gates prove `-O0 == -O2` and interp ==
//! native on them; what only this file proves is that the vector path is the one that ran — that
//! each kernel holds `<4 x ..>` values at all, which those gates cannot see.

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

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/run")
        .join(name)
}

/// The MIR `wukongc --emit=mir -O2 <fixture>` prints, with `env` set.
fn mir(name: &str, env: &[(&str, &str)]) -> String {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_wukongc"));
    for v in COMPILER_ENV {
        cmd.env_remove(v);
    }
    cmd.envs(env.iter().copied())
        .args(["--emit=mir", "-O2"])
        .arg(fixture(name));
    let out = cmd.output().expect("spawn wukongc");
    assert!(
        out.status.success(),
        "wukongc --emit=mir -O2 {name} failed:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8_lossy(&out.stdout).into_owned()
}

/// The text of function `name` in a MIR listing: from its `fn name(` line to the next `fn `.
fn function<'a>(listing: &'a str, name: &str) -> &'a str {
    let head = format!("fn {name}(");
    let start = listing
        .find(&head)
        .unwrap_or_else(|| panic!("no fn {name} in the listing"));
    let rest = &listing[start + head.len()..];
    let end = rest.find("\nfn ").map_or(rest.len(), |e| e + 1);
    &listing[start..start + head.len() + end]
}

#[test]
fn a_pure_call_is_called_once_per_lane_beside_widened_work() {
    let m = mir("vec7_call_lanes.wk", &[]);
    let f32k = function(&m, "call_f32");
    assert!(f32k.contains("<4 x f32>"), "call_f32 was not widened");
    // Four lane calls in the vector body, one in the scalar epilogue.
    assert!(
        f32k.matches("call tri(").count() >= 5,
        "call_f32 does not call `tri` once per lane:\n{f32k}"
    );
    let i64k = function(&m, "call_i64");
    assert!(i64k.contains("<2 x i64>"), "call_i64 was not widened");
    assert!(
        i64k.matches("call h(").count() >= 3,
        "call_i64 does not call `h` once per lane"
    );
    // One argument value for the group: one call per group, beside the widened multiply-add.
    let uni = function(&m, "call_uniform");
    assert!(uni.contains("<4 x f32>"), "call_uniform was not widened");
    assert!(uni.matches("call tri(").count() >= 2);
    // A widened compare as a `bool` argument: each lane's call takes that lane's truth, so the
    // listing verifies (an `i32` mask lane passed to `sel`'s `bool` would be a MIR-verify ICE).
    let cb = function(&m, "call_bool");
    assert!(cb.contains("<4 x f32>"), "call_bool was not widened");
    assert!(
        cb.matches("call sel(").count() >= 5,
        "call_bool does not call `sel` once per lane:
{cb}"
    );
}

#[test]
fn a_checked_gather_is_tested_per_group() {
    let m = mir("vec7_checked_gather.wk", &[]);
    for k in ["gather_f32", "gather_i32"] {
        let f = function(&m, k);
        assert!(f.contains("<4 x"), "{k} was not widened:\n{f}");
        // The group's index check folds its lanes with a shuffle before the branch.
        assert!(
            f.contains("shuffle"),
            "{k} has no per-group lane test:\n{f}"
        );
        // The scalar loop keeps the check, and its failure path.
        assert!(f.contains("wukong_rt_bounds_fail"), "{k} lost its check");
    }
}

#[test]
fn lane_serial_regions_widen_around_the_serial_accesses() {
    let m = mir("vec7_lane_serial.wk", &[]);
    // `scat_hdr` is a plain scatter beside its slice's in-loop data-pointer read, which only the
    // lane-serial consumer's no-punning exemption lets widen.
    for k in [
        "hist", "two_scat", "beside", "alias", "checked", "hist_sel", "scat_hdr",
    ] {
        assert!(
            function(&m, k).contains("<4 x"),
            "{k} was not widened around its lane-serial region"
        );
    }
    // The histogram's update is four scalar load/store pairs in the vector body.
    let hist = function(&m, "hist");
    assert!(hist.matches("store ").count() >= 5, "{hist}");
    // The kill switch restores the decline, region by region, and the header-read exemption with
    // them.
    let off = mir("vec7_lane_serial.wk", &[("WUKONG_NO_LANE_SERIAL", "1")]);
    for k in [
        "hist", "two_scat", "beside", "alias", "checked", "hist_sel", "scat_hdr",
    ] {
        assert!(
            !function(&off, k).contains("<4 x"),
            "{k} widened with WUKONG_NO_LANE_SERIAL set"
        );
    }
}

/// The block parameters a function's loop headers carry as vectors, one string per parameter list.
fn vector_params(f: &str) -> Vec<&str> {
    f.lines()
        .filter(|l| l.trim_start().starts_with("bb") && l.contains("(") && l.contains(" x "))
        .collect()
}

#[test]
fn nests_widen_across_their_rows() {
    let m = mir("vec7_outer_loops.wk", &[]);
    // Each row's accumulator is one lane of a vector the inner loop carries.
    for (k, ty) in [
        ("mv", "<4 x f32>"),
        ("horner", "<4 x f32>"),
        ("rowmax", "<2 x f64>"),
    ] {
        let f = function(&m, k);
        assert!(
            vector_params(f).iter().any(|l| l.contains(ty)),
            "{k} carries no {ty} accumulator:\n{f}"
        );
    }
    // The kill switch leaves every inner loop to the inner path, whose float accumulators stay
    // scalar.
    let off = mir("vec7_outer_loops.wk", &[("WUKONG_NO_OUTER_VEC", "1")]);
    for k in ["mv", "horner", "rowmax"] {
        let f = function(&off, k);
        assert!(
            vector_params(f).is_empty(),
            "{k} carries a vector with WUKONG_NO_OUTER_VEC set:\n{f}"
        );
    }
}
