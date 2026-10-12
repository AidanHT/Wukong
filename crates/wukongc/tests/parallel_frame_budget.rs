//! A `@parallel` loop whose body declares more scratch than a runtime worker's stack holds runs
//! serially instead of aborting natively.
//!
//! Outlined, a loop body's locals live in the outlined function's frame, and that function runs on
//! the runtime's pool workers, whose stacks are 16 MiB (`wukong_runtime`'s `stack_size(16 * 1024 *
//! 1024)`). A body-local `[f32; 5000000]` is a 20 MB frame: threaded, `--run --backend=native`
//! aborted with "has overflowed its stack" on a `wukong-gemm-<n>` worker, while the interpreter --
//! which runs a region in order on heap-backed slots -- printed the right answer, so no
//! differential gate could tell. Both outliners now keep such a loop serial
//! (`wukong_mir_build::par_grain::OUTLINED_FRAME_BUDGET`, half a worker stack), where its scratch
//! sits in the enclosing function's frame on the calling thread (256 MiB under `--run`), as it
//! would without `@parallel`. `par_region_tests.rs` pins the declines in MIR; this runs them.
//!
//! The interpreter is not run here: allocating and zeroing 5,000,000 slots per iteration takes it
//! over ten seconds in a debug build. The expected output is the scalar loop by hand instead:
//! `out[r*8 + j] = x[r*8 + j] + 1.0` with `x[i] = i`, so lane `i` prints `i + 1`.

use std::process::Command;

/// The region outliner's shape: `[]f32` parameters and a run-time trip count.
const SLICE_REGION: &str = "module frame.budget.slices

@parallel
fn rows(x: []f32, mut out: []f32, n: i64) {
    for r in 0..n {
        let mut t: [f32; 5000000] = [0.0; 5000000];
        for j in 0..8 {
            t[j] = x[r * 8 + j];
        }
        for j in 0..8 {
            out[r * 8 + j] = t[j] + 1.0;
        }
    }
}

fn main() -> i32 {
    let mut x: []f32 = alloc_f32(32);
    let mut o: []f32 = alloc_f32(32);
    for i in 0..32 {
        x[i] = i as f32;
    }
    rows(x, o, 4);
    for i in 0..32 {
        print(o[i]);
    }
    free(x);
    free(o);
    return 0;
}
";

/// The whole-function outliner's shape: fixed-size array parameters, one loop as the whole body.
const ARRAY_FUNCTION: &str = "module frame.budget.arrays

@parallel
fn rows(x: [f32; 32], mut out: [f32; 32]) {
    for r in 0..4 {
        let mut t: [f32; 5000000] = [0.0; 5000000];
        for j in 0..8 {
            t[j] = x[r * 8 + j];
        }
        for j in 0..8 {
            out[r * 8 + j] = t[j] + 1.0;
        }
    }
}

fn main() -> i32 {
    let mut x: [f32; 32] = [0.0; 32];
    let mut o: [f32; 32] = [0.0; 32];
    for i in 0..32 {
        x[i] = i as f32;
    }
    rows(x, o);
    for i in 0..32 {
        print(o[i]);
    }
    return 0;
}
";

fn check(name: &str, src: &str) {
    let path = std::env::temp_dir().join(format!(
        "wukong-frame-budget-{name}-{}.wk",
        std::process::id()
    ));
    std::fs::write(&path, src).unwrap();
    let expected: String = (1..=32).map(|v| format!("{v}\n")).collect();
    for opt in ["-O0", "-O2"] {
        // Four workers, so a threaded body would really run on a pool worker's stack.
        let output = Command::new(env!("CARGO_BIN_EXE_wukongc"))
            .args(["--run", "--backend=native", "--threads=4", opt])
            .arg(&path)
            .output()
            .expect("spawn wukongc");
        let stdout = String::from_utf8_lossy(&output.stdout);
        assert_eq!(
            output.status.code(),
            Some(0),
            "{name} {opt}\nstdout: {stdout}\nstderr: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert_eq!(stdout, expected, "{name} {opt}");
    }
    std::fs::remove_file(path).unwrap();
}

#[test]
fn a_slice_region_over_the_frame_budget_runs_serially_natively() {
    check("slices", SLICE_REGION);
}

#[test]
fn a_whole_function_loop_over_the_frame_budget_runs_serially_natively() {
    check("arrays", ARRAY_FUNCTION);
}
