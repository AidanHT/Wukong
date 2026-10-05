//! Empty GEMV/GEVM contractions still store their seed and apply the source finalizer.
use std::path::Path;
use std::process::Command;

const ROWS: usize = 17;
const COLS: usize = 9;
const LEN: usize = ROWS * COLS;

fn source(parallel: bool) -> String {
    let attr = if parallel { "@parallel\n" } else { "" };
    let mut src = String::from("module gemv.empty\nconst SCALE: f32 = -2.0;\n");
    for family in ["gemv", "gevm"] {
        for scaled in [false, true] {
            let name = if scaled { "scaled" } else { "plain" };
            let finalizer = if scaled { "acc * SCALE" } else { "acc" };
            for (suffix, bound, param) in [
                ("dynamic", "n", ", n: i32"),
                ("zero", "0", ""),
                ("negative", "-1", ""),
                ("positive", "9", ""),
            ] {
                let product = if family == "gemv" {
                    format!("a[i*({bound})+j]*x[j]")
                } else {
                    format!("x[j]*a[j*{ROWS}+i]")
                };
                src.push_str(&format!(
                "{attr}fn {family}_{name}_{suffix}(a: [f32; {LEN}], x: [f32; {COLS}], mut y: [f32; {ROWS}]{param}) {{\n\
                for i in 0..{ROWS} {{ let mut acc: f32 = 0.0;\n\
                for j in 0..{bound} {{ acc = acc + {product}; }}\n\
                y[i] = {finalizer}; }} }}\n"
            ));
            }
        }
    }
    src.push_str(&format!(
        "fn main() -> i32 {{ let mut a: [f32; {LEN}] = [0.0; {LEN}];\n\
        let x: [f32; {COLS}] = [1.0; {COLS}];\n\
        for k in 0..{LEN} {{ a[k] = (k+1) as f32; }}\n"
    ));
    for family in ["gemv", "gevm"] {
        for scale in ["plain", "scaled"] {
            let name = format!("{family}_{scale}");
            for call in [
                format!("{name}_dynamic(a, x, y, 0)"),
                format!("{name}_dynamic(a, x, y, -1)"),
                format!("{name}_dynamic(a, x, y, 9)"),
                format!("{name}_zero(a, x, y)"),
                format!("{name}_negative(a, x, y)"),
                format!("{name}_positive(a, x, y)"),
            ] {
                // A no-op kernel would leave the nonzero sentinel visible.
                src.push_str(&format!(
                    "{{ let mut y: [f32; {ROWS}] = [123.0; {ROWS}]; {call};\n\
                for i in 0..{ROWS} {{ print(y[i]); }} }}\n"
                ));
            }
        }
    }
    src.push_str("return 0; }\n");
    src
}

fn oracle() -> String {
    let mut expected = String::new();
    for family in ["gemv", "gevm"] {
        for scaled in [false, true] {
            for positive in [false, false, true, false, false, true] {
                for row in 0..ROWS {
                    if positive {
                        // GEMV sums nine consecutive values in a row. GEVM sums
                        // nine column values at stride 17. All arithmetic is exact.
                        let dot = if family == "gemv" {
                            COLS * COLS * row + COLS * (COLS + 1) / 2
                        } else {
                            COLS * (row + 1) + ROWS * COLS * (COLS - 1) / 2
                        } as i32;
                        let value = if scaled { -2 * dot } else { dot };
                        expected.push_str(&format!("{value}\n"));
                    } else {
                        // Empty +0 seed, followed by multiplication by -2 if scaled.
                        expected.push_str(if scaled { "-0\n" } else { "0\n" });
                    }
                }
            }
        }
    }
    expected
}

fn invoke(path: &Path, args: &[&str], no_recognize: bool) -> String {
    let mut command = Command::new(env!("CARGO_BIN_EXE_wukongc"));
    command.args(args).arg(path).env("RAYON_NUM_THREADS", "2");
    if no_recognize {
        command.env(wukong_driver::NO_RECOGNIZE_ENV, "1");
    } else {
        command.env_remove(wukong_driver::NO_RECOGNIZE_ENV);
    }
    let output = command.output().expect("spawn wukongc");
    assert_eq!(
        output.status.code(),
        Some(0),
        "{args:?}, recognition off={no_recognize}\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8(output.stdout).unwrap()
}

fn check(parallel: bool) {
    let path = std::env::temp_dir().join(format!(
        "wukong-gemv-empty-{}-{parallel}.wk",
        std::process::id()
    ));
    std::fs::write(&path, source(parallel)).unwrap();
    let expected = oracle();
    assert_eq!(expected.lines().count(), 24 * ROWS);
    let mir = invoke(&path, &["--emit=mir", "-O0"], false);
    for family in ["gemv", "gevm"] {
        for name in ["plain", "scaled"] {
            for suffix in ["dynamic", "positive"] {
                let function = format!("{family}_{name}_{suffix}");
                let (_, tail) = mir
                    .split_once(&format!("fn {function}("))
                    .unwrap_or_else(|| panic!("missing {function}:\n{mir}"));
                let body = tail.split("\n}\n").next().unwrap();
                let base = if family == "gevm" {
                    "wukong_sgevm_f32"
                } else if name == "scaled" {
                    "wukong_sgemv_alpha"
                } else {
                    "wukong_sgemv"
                };
                let symbol = format!("{base}{}", if parallel { "_parallel" } else { "" });
                assert_eq!(
                    body.matches(&format!("call {symbol}(")).count(),
                    1,
                    "{function}:\n{body}"
                );
            }
        }
    }
    for opt in ["-O0", "-O2"] {
        let off = invoke(&path, &["--emit=mir", opt], true);
        assert!(
            !off.contains("call wukong_sgemv"),
            "recognition off:\n{off}"
        );
        assert!(
            !off.contains("call wukong_sgevm"),
            "recognition off:\n{off}"
        );
        for backend in ["--backend=interp", "--backend=native"] {
            for no_recognize in [false, true] {
                assert_eq!(
                    invoke(&path, &["--run", backend, opt], no_recognize),
                    expected,
                    "parallel={parallel} {backend} {opt} recognition off={no_recognize}"
                );
            }
        }
    }
    std::fs::remove_file(path).unwrap();
}

#[test]
fn serial_gemv_empty_contraction_stores_seed() {
    check(false);
}

#[test]
fn parallel_gemv_empty_contraction_stores_seed() {
    check(true);
}
