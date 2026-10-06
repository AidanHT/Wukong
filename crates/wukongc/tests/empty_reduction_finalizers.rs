//! Empty scans retain observable seeds/finalizers even when runtime kernels are no-ops.
use std::path::Path;
use std::process::Command;

struct Fixture {
    source: String,
    expected: String,
    kernels: Vec<(String, String, bool)>,
    declines: Vec<String>,
}

impl Fixture {
    fn new() -> Self {
        Self {
            source: "module empty.finalizers\n".into(),
            expected: String::new(),
            kernels: Vec::new(),
            declines: Vec::new(),
        }
    }

    fn dispatch(&mut self, name: &str, symbol: &str, parallel: bool, dynamic: bool) {
        self.kernels.push((
            name.into(),
            format!("{symbol}{}", if parallel { "_parallel" } else { "" }),
            dynamic,
        ));
    }
}

fn arg_fixture(parallel: bool) -> Fixture {
    let attr = if parallel { "@parallel\n" } else { "" };
    let mut fixture = Fixture::new();
    let mut calls = String::from(
        "fn main() -> i32 { let x: [f32; 9] = [2.0,9.0,3.0,5.0,4.0,3.0,5.0,1.0,8.0];\n",
    );
    for axis in ["col", "row"] {
        for maximum in [false, true] {
            for start in [0, 1] {
                let kind = if maximum { "max" } else { "min" };
                let cmp = if maximum { ">" } else { "<" };
                let stem = format!("{axis}_{kind}_{start}");
                for (suffix, bound, param) in [
                    ("dynamic", "n", ", n: i32"),
                    ("zero", "0", ""),
                    ("negative", "-1", ""),
                    ("one", "1", ""),
                    ("positive", "3", ""),
                ] {
                    let name = format!("{stem}_{suffix}");
                    // Column seeds x[j] exist even for a nonpositive row count.
                    // Row tests use ONE row: r*n is zero even when n is negative.
                    let (outer, seed, index) = if axis == "col" {
                        (3, "x[r]".into(), "x[j*3+r]".into())
                    } else {
                        (1, format!("x[r*({bound})]"), format!("x[r*({bound})+j]"))
                    };
                    fixture.source.push_str(&format!(
                        "{attr}fn {name}(x: [f32; 9], mut out: [i32; 3]{param}) {{\n\
                        for r in 0..{outer} {{ let mut bv: f32 = {seed}; let mut bi: i32 = 0;\n\
                        for j in {start}..{bound} {{ if {index} {cmp} bv {{ bv = {index}; bi = j; }} }}\n\
                        out[r] = bi; }} }}\n"
                    ));
                    if matches!(suffix, "dynamic" | "positive" | "one") {
                        fixture.dispatch(
                            &name,
                            &format!("wukong_{axis}arg{kind}_i32"),
                            parallel,
                            suffix == "dynamic",
                        );
                    }
                }
                for (suffix, argument, full) in [
                    ("dynamic", ", 0", false),
                    ("dynamic", ", -1", false),
                    ("dynamic", ", 1", false),
                    ("dynamic", ", 3", true),
                    ("zero", "", false),
                    ("negative", "", false),
                    ("one", "", false),
                    ("positive", "", true),
                ] {
                    calls.push_str(&format!(
                        "{{ let mut out: [i32; 3] = [41,43,47]; {stem}_{suffix}(x,out{argument});\n\
                        for k in 0..3 {{ print(out[k]); }} }}\n"
                    ));
                    // Independent extrema of the declared 3x3 matrix. Strict ties
                    // retain the first row: column 0 max at 1, column 2 min at 0.
                    let values = if axis == "col" {
                        if !full {
                            [0, 0, 0]
                        } else if maximum {
                            [1, 0, 2]
                        } else {
                            [0, 2, 0]
                        }
                    } else {
                        [i32::from(full && maximum), 43, 47]
                    };
                    for value in values {
                        fixture.expected.push_str(&format!("{value}\n"));
                    }
                }
            }
        }
    }
    calls.push_str("return 0; }\n");
    fixture.source.push_str(&calls);
    assert_eq!(fixture.expected.lines().count(), 8 * 8 * 3);
    fixture
}

fn loss_fixture(parallel: bool) -> Fixture {
    let attr = if parallel { "@parallel\n" } else { "" };
    let mut fixture = Fixture::new();
    let mut calls = String::from(
        "fn main() -> i32 { let x: [f32; 3] = [3.0,5.0,7.0];\n\
        let p: [f32; 3] = [1.0;3]; let q: [f32; 3] = [1.0;3]; let target: [i32; 1] = [0];\n",
    );
    for family in [
        "kldiv",
        "entropy",
        "kd_loss",
        "logsumexp",
        "xent",
        "xent_bwd",
    ] {
        for (suffix, bound, param) in [
            ("dynamic", "n", ", n: i32"),
            ("zero", "0", ""),
            ("negative", "-1", ""),
            ("positive", "1", ""),
        ] {
            let name = format!("{family}_{suffix}");
            let rows = if matches!(family, "kldiv" | "entropy") {
                3
            } else {
                1
            };
            let index = format!("r*({bound})+i");
            // Seed/gather families use one row and target[0]=0. Every source
            // read addresses x[0], even for n=-1; all buffers really have 3 lanes.
            let body = match family {
                "kldiv" => format!(
                    "let mut s: f32 = 0.0; for i in 0..{bound} {{\n\
                    s = s + p[{index}]*(log(p[{index}])-log(q[{index}])); }} out[r] = s;"
                ),
                "entropy" => format!(
                    "let mut s: f32 = 0.0; for i in 0..{bound} {{\n\
                    s = s + p[{index}]*log(p[{index}]); }} out[r] = -s;"
                ),
                _ => {
                    let prefix = format!(
                        "let mut m: f32 = x[r*({bound})];\n\
                        for i in 0..{bound} {{ m = fmax(m,x[{index}]); }}\n\
                        let mut z: f32 = 0.0; for i in 0..{bound} {{ z = z + exp(x[{index}]-m); }}\n"
                    );
                    let finalizer = match family {
                        "kd_loss" => format!(
                            "let lse: f32 = m + log(z); let mut s: f32 = 0.0;\n\
                            for i in 0..{bound} {{ s = s + q[{index}]*(lse-x[{index}]); }} out[r] = s;"
                        ),
                        "logsumexp" => "out[r] = m + log(z);".into(),
                        "xent" => format!("out[r] = m + log(z) - x[r*({bound})+target[r]];"),
                        "xent_bwd" => format!(
                            "let invz: f32 = 1.0/z;\n\
                            for i in 0..{bound} {{ out[{index}] = exp(x[{index}]-m)*invz; }}\n\
                            out[r*({bound})+target[r]] = out[r*({bound})+target[r]] - 1.0;"
                        ),
                        _ => unreachable!(),
                    };
                    prefix + &finalizer
                }
            };
            fixture.source.push_str(&format!(
                "{attr}fn {name}(x: [f32;3], p: [f32;3], q: [f32;3], target: [i32;1], mut out: [f32;3]{param}) {{\n\
                for r in 0..{rows} {{ {body} }} }}\n"
            ));
            if matches!(suffix, "dynamic" | "positive") {
                let kernel = if family == "xent" { "xent_fwd" } else { family };
                fixture.dispatch(
                    &name,
                    &format!("wukong_{kernel}_f32"),
                    parallel,
                    suffix == "dynamic",
                );
            }
        }
        for (suffix, argument, positive) in [
            ("dynamic", ", 0", false),
            ("dynamic", ", -1", false),
            ("dynamic", ", 1", true),
            ("zero", "", false),
            ("negative", "", false),
            ("positive", "", true),
        ] {
            calls.push_str(&format!(
                "{{ let mut out: [f32;3] = [41.0,43.0,47.0];\n\
                {family}_{suffix}(x,p,q,target,out{argument}); for k in 0..3 {{ print(out[k]); }} }}\n"
            ));
            // log(1)=0 and exp(0)=1 make every positive oracle exact. Empty
            // reductions retain +0; entropy negates it. log(+0) is -infinity.
            // The backward scatter is outside the empty loop: 41-1 = 40.
            fixture.expected.push_str(match (family, positive) {
                ("kldiv", _) => "0\n0\n0\n",
                ("entropy", _) => "-0\n-0\n-0\n",
                ("kd_loss", _) | ("xent", true) | ("xent_bwd", true) => "0\n43\n47\n",
                ("logsumexp", true) => "3\n43\n47\n",
                ("logsumexp" | "xent", false) => "-inf\n43\n47\n",
                ("xent_bwd", false) => "40\n43\n47\n",
                _ => unreachable!(),
            });
        }
    }
    calls.push_str("return 0; }\n");
    fixture.source.push_str(&calls);
    assert_eq!(fixture.expected.lines().count(), 6 * 6 * 3);
    fixture
}

fn pool_fixture(parallel: bool) -> Fixture {
    let attr = if parallel { "@parallel\n" } else { "" };
    let mut fixture = Fixture::new();
    let mut calls = String::from("fn main() -> i32 { let x: [f32;3] = [3.0,5.0,7.0];\n");
    for kind in ["max", "avg"] {
        for axis in ["height", "width"] {
            let stem = format!("pool_{kind}_{axis}");
            for (suffix, window, output, param) in [
                ("dynamic", "n", "o", ", n: i32, o: i32"),
                ("zero", "0", "2", ""),
                ("negative", "-1", "3", ""),
                ("positive", "1", "1", ""),
            ] {
                let name = format!("{stem}_{suffix}");
                let (kh, kw, oh, ow) = if axis == "height" {
                    (window, "1", output, "1")
                } else {
                    ("1", window, "1", output)
                };
                // H=W=1 and strides=1. For nonpositive KH/KW the source's
                // computed output extent grows to 2/3; the full backing array
                // intentionally contains all max-pool seeds x[oy+ox].
                let seed = if kind == "max" {
                    "x[c*1*1+(oy*1)*1+(ox*1)]"
                } else {
                    "0.0"
                };
                let fold = if kind == "max" { "fmax(s,v)" } else { "s+v" };
                let finalizer = if kind == "max" {
                    "s".into()
                } else {
                    format!("s / ((({kh}) as f32)*(({kw}) as f32))")
                };
                fixture.source.push_str(&format!(
                    "{attr}fn {name}(x: [f32;3], mut out: [f32;3]{param}) {{\n\
                    for c in 0..1 {{ for oy in 0..{oh} {{ for ox in 0..{ow} {{\n\
                    let mut s: f32 = {seed}; for dy in 0..{kh} {{ for dx in 0..{kw} {{\n\
                    let v = x[c*1*1+(oy*1+dy)*1+(ox*1+dx)]; s = {fold}; }} }}\n\
                    out[c*({oh})*({ow})+oy*({ow})+ox] = {finalizer}; }} }} }} }}\n"
                ));
                if suffix == "positive" {
                    fixture.dispatch(&name, &format!("wukong_{kind}pool2d_f32"), parallel, false);
                } else {
                    // Pooling already declines symbolic shapes; this fix retains
                    // that boundary and additionally declines empty literals.
                    fixture.declines.push(name);
                }
            }
            for (suffix, argument, window) in [
                ("dynamic", ", 0, 2", 0),
                ("dynamic", ", -1, 3", -1),
                ("dynamic", ", 1, 1", 1),
                ("zero", "", 0),
                ("negative", "", -1),
                ("positive", "", 1),
            ] {
                calls.push_str(&format!(
                    "{{ let mut out: [f32;3] = [41.0,43.0,47.0]; {stem}_{suffix}(x,out{argument});\n\
                    for k in 0..3 {{ if out[k] != out[k] {{ print(\"true\"); }} else {{ print(\"false\"); }}\n\
                    if out[k] == out[k] {{ print(out[k]); }} else {{ print(0); }} }} }}\n"
                ));
                for lane in 0..3 {
                    // Each lane asserts NaN classification plus its exact finite
                    // value/sign. NaN payload propagation is not a pooling contract.
                    let value = if lane >= (2 - window) {
                        ["41", "43", "47"][lane as usize]
                    } else if kind == "max" {
                        ["3", "5", "7"][lane as usize]
                    } else if window < 0 {
                        "-0"
                    } else if window == 0 {
                        fixture.expected.push_str("true\n0\n");
                        continue;
                    } else {
                        "3"
                    };
                    fixture.expected.push_str(&format!("false\n{value}\n"));
                }
            }
        }
    }
    calls.push_str("return 0; }\n");
    fixture.source.push_str(&calls);
    assert_eq!(fixture.expected.lines().count(), 4 * 6 * 3 * 2);
    fixture
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

fn function<'a>(mir: &'a str, name: &str) -> &'a str {
    mir.split_once(&format!("fn {name}("))
        .unwrap_or_else(|| panic!("missing {name}:\n{mir}"))
        .1
        .split("\n}\n")
        .next()
        .unwrap()
}

fn check(group: &str, parallel: bool, fixture: Fixture) {
    let path = std::env::temp_dir().join(format!(
        "wukong-empty-finalizers-{}-{group}-{parallel}.wk",
        std::process::id()
    ));
    std::fs::write(&path, fixture.source).unwrap();
    // Prove each recognizer's actual positive dispatch and dynamic guard before
    // any native execution. Whole-function @parallel paths must use the twin.
    let mir = invoke(&path, &["--emit=mir", "-O0"], false);
    for (name, symbol, dynamic) in &fixture.kernels {
        let body = function(&mir, name);
        assert_eq!(
            body.matches(&format!("call {symbol}(")).count(),
            1,
            "{name}:\n{body}"
        );
        if *dynamic {
            assert!(
                body.contains("cmp.sgt") && body.contains("cond_br"),
                "unguarded {name}:\n{body}"
            );
        }
    }
    for name in &fixture.declines {
        let body = function(&mir, name);
        assert!(
            !body.contains("pool2d_f32"),
            "empty/symbolic pooling dispatch in {name}:\n{body}"
        );
    }
    for opt in ["-O0", "-O2"] {
        let off = invoke(&path, &["--emit=mir", opt], true);
        for (_, symbol, _) in &fixture.kernels {
            assert!(
                !off.contains(&format!("call {symbol}(")),
                "recognition off:\n{off}"
            );
        }
        for backend in ["--backend=interp", "--backend=native"] {
            for no_recognize in [false, true] {
                assert_eq!(
                    invoke(&path, &["--run", backend, opt], no_recognize),
                    fixture.expected,
                    "{group} parallel={parallel} {backend} {opt} recognition off={no_recognize}"
                );
            }
        }
    }
    std::fs::remove_file(path).unwrap();
}

#[test]
fn serial_arg_empty_scan_stores_index() {
    check("arg", false, arg_fixture(false));
}
#[test]
fn parallel_arg_empty_scan_stores_index() {
    check("arg", true, arg_fixture(true));
}
#[test]
fn serial_loss_empty_reduction_runs_finalizer() {
    check("loss", false, loss_fixture(false));
}
#[test]
fn parallel_loss_empty_reduction_runs_finalizer() {
    check("loss", true, loss_fixture(true));
}
#[test]
fn serial_pool_empty_window_runs_finalizer() {
    check("pool", false, pool_fixture(false));
}
#[test]
fn parallel_pool_empty_window_runs_finalizer() {
    check("pool", true, pool_fixture(true));
}
