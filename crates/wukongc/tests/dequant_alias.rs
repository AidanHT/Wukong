//! Per-channel dequantization must preserve scales reread after output stores.
use std::path::Path;
use std::process::Command;

const ROWS: usize = 17; // Cross the runtime's eight-row parallel band.
const COLS: usize = 9; // An AVX vector and a scalar tail on disjoint controls.
const LEN: usize = ROWS * COLS;
const TYPES: [&str; 3] = ["i8", "u8", "i32"];

fn source(parallel: bool, aliases: bool) -> String {
    let attr = if parallel { "@parallel\n" } else { "" };
    let nest = format!("for i in 0..{ROWS} {{ for j in 0..{COLS} {{ out[i*{COLS}+j] = (q[i*{COLS}+j] as f32) * scale[j]; }} }}");
    let mut src = String::from("module dequant.alias\n");
    for ty in TYPES {
        if aliases {
            src.push_str(&format!(
                "{attr}fn argument_{ty}(q: [{ty}; {LEN}], scale: [f32; {LEN}], mut out: [f32; {LEN}]) {{ {nest} }}\n\
                {attr}fn local_{ty}(q: [{ty}; {LEN}], mut storage: [f32; {LEN}]) {{\n\
                let scale: []f32 = storage; let mut out: []f32 = storage; {nest} }}\n\
                {attr}fn same_{ty}(q: [{ty}; {LEN}], mut out: [f32; {LEN}]) {{ {} }}\n",
                nest.replace("scale[j]", "out[j]")
            ));
        } else {
            src.push_str(&format!(
                "{attr}fn disjoint_{ty}(q: [{ty}; {LEN}], scale: [f32; {LEN}], mut out: [f32; {LEN}]) {{ {nest} }}\n"
            ));
        }
    }
    src.push_str("fn main() -> i32 {\n");
    for ty in TYPES {
        let calls = if aliases {
            vec![
                format!("argument_{ty}(q, out, out)"),
                format!("local_{ty}(q, out)"),
                format!("same_{ty}(q, out)"),
            ]
        } else {
            vec![format!("disjoint_{ty}(q, scale, out)")]
        };
        for call in calls {
            src.push_str(&format!(
                "{{ let mut q: [{ty}; {LEN}] = [2; {LEN}];\n\
                let mut scale: [f32; {LEN}] = [0.0; {LEN}];\n\
                let mut out: [f32; {LEN}] = [0.0; {LEN}];\n\
                for k in 0..{LEN} {{ scale[k] = (k+1) as f32; out[k] = (k+1) as f32; }}\n\
                {call}; for k in 0..{LEN} {{ print(out[k]); }} }}\n"
            ));
        }
    }
    src.push_str("return 0; }\n");
    src
}

fn oracle(aliases: bool) -> String {
    let mut expected = String::new();
    for _ in TYPES {
        for _ in 0..if aliases { 3 } else { 1 } {
            for k in 0..LEN {
                // q is exactly 2. Row zero doubles each live scale; later rows
                // read that doubled scale. Disjoint scales stay at column + 1.
                // All products are small integers, represented exactly in f32.
                let factor = if aliases && k >= COLS { 4 } else { 2 };
                expected.push_str(&format!("{}\n", factor * (k % COLS + 1)));
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
    for aliases in [true, false] {
        let path = std::env::temp_dir().join(format!(
            "wukong-dequant-alias-{}-{parallel}-{aliases}.wk",
            std::process::id()
        ));
        std::fs::write(&path, source(parallel, aliases)).unwrap();
        let expected = oracle(aliases);
        assert_eq!(
            expected.lines().count(),
            TYPES.len() * LEN * if aliases { 3 } else { 1 }
        );
        // Check both recognition settings and optimization levels BEFORE native
        // execution. O0 additionally pins each function, so a different function
        // cannot satisfy the positive dispatch obligation or hide an unsafe call.
        for no_recognize in [false, true] {
            for opt in ["-O0", "-O2"] {
                let mir = invoke(&path, &["--emit=mir", opt], no_recognize);
                if aliases || no_recognize {
                    assert!(
                        !mir.contains("call wukong_dequant"),
                        "unsafe or disabled dispatch:\n{mir}"
                    );
                }
                if aliases {
                    assert!(
                        !mir.contains("call wukong_parallel_for"),
                        "aliased rows/columns must remain ordered:\n{mir}"
                    );
                }
                if opt == "-O0" {
                    let names: &[&str] = if aliases {
                        &["argument", "local", "same"]
                    } else {
                        &["disjoint"]
                    };
                    for ty in TYPES {
                        for name in names {
                            let name = format!("{name}_{ty}");
                            let (_, tail) = mir
                                .split_once(&format!("fn {name}("))
                                .unwrap_or_else(|| panic!("missing {name}:\n{mir}"));
                            let body = tail.split("\n}\n").next().unwrap();
                            let symbol = if parallel {
                                "wukong_dequant_perchan_f32_parallel"
                            } else {
                                "wukong_dequant_perchan_f32"
                            };
                            assert_eq!(
                                body.matches(&format!("call {symbol}(")).count(),
                                usize::from(!aliases && !no_recognize),
                                "{name}:\n{body}"
                            );
                            if aliases {
                                assert!(!body.contains("call wukong_dequant"), "{name}:\n{body}");
                                assert!(
                                    !body.contains("call wukong_parallel_for"),
                                    "{name}:\n{body}"
                                );
                            }
                        }
                    }
                }
            }
        }
        for opt in ["-O0", "-O2"] {
            for backend in ["--backend=interp", "--backend=native"] {
                for no_recognize in [false, true] {
                    assert_eq!(
                        invoke(&path, &["--run", backend, opt], no_recognize),
                        expected,
                        "parallel={parallel} aliases={aliases} {backend} {opt} recognition off={no_recognize}"
                    );
                }
            }
        }
        std::fs::remove_file(path).unwrap();
    }
}

#[test]
fn serial_dequant_alias_preserves_source_reads() {
    check(false);
}

#[test]
fn parallel_dequant_alias_preserves_source_reads() {
    check(true);
}
