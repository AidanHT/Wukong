//! Broadcast-bias recognition must preserve live source reads under aliasing.
use std::path::Path;
use std::process::Command;

const ROWS: usize = 17; // Cross the runtime's eight-row parallel band.
const COLS: usize = 9; // One AVX vector and a scalar tail on disjoint controls.
const LEN: usize = ROWS * COLS;

fn source(parallel: bool) -> String {
    let attr = if parallel { "@parallel\n" } else { "" };
    let nest = format!(
        "for i in 0..{ROWS} {{ for j in 0..{COLS} {{ out[i*{COLS}+j] = x[i*{COLS}+j] + b[j]; }} }}"
    );
    let mut src = format!(
        "module bias.alias\n\
        {attr}fn aliased(x: [f32; {LEN}], b: [f32; {LEN}], mut out: [f32; {LEN}]) {{ {nest} }}\n\
        {attr}fn local_alias(x: [f32; {LEN}], mut storage: [f32; {LEN}]) {{\n\
        let b: []f32 = storage; let mut out: []f32 = storage; {nest} }}\n\
        {attr}fn disjoint(x: [f32; {LEN}], b: [f32; {LEN}], mut out: [f32; {LEN}]) {{ {nest} }}\n\
        {attr}fn inplace(mut out: [f32; {LEN}], b: [f32; {LEN}]) {{ {} }}\n\
        {attr}fn read_alias(x: [f32; {LEN}], mut out: [f32; {LEN}]) {{ {} }}\n\
        fn main() -> i32 {{\n",
        nest.replace("x[i*", "out[i*"),
        nest.replace("b[j]", "x[j]")
    );
    for (i, call) in [
        "aliased(x, out, out)",
        "local_alias(x, out)",
        "disjoint(x, b, out)",
        "inplace(out, b)",
        "read_alias(x, out)",
    ]
    .iter()
    .enumerate()
    {
        // Separate scopes keep every input fresh; setup itself is unrecognized scalar work.
        src.push_str(&format!("{{ let mut x: [f32; {LEN}] = [0.0; {LEN}]; let mut b: [f32; {LEN}] = [0.0; {LEN}]; let mut out: [f32; {LEN}] = [0.0; {LEN}];\n\
            for k in 0..{LEN} {{ x[k] = (k+1) as f32; b[k] = (k+1) as f32; out[k] = (10*(k+1)) as f32; }}\n\
            {call}; for k in 0..{LEN} {{ print(out[k]); }} }} // case {i}\n"));
    }
    src.push_str("return 0; }\n");
    src
}

fn oracle() -> String {
    let mut expected = String::new();
    for case in 0..5 {
        for k in 0..LEN {
            let col = k % COLS;
            let value = match case {
                // Row zero overwrites the bias. Every later row sees that new bias.
                0 | 1 if k < COLS => 11 * (col + 1),
                0 | 1 => k + 1 + 11 * (col + 1),
                2 | 4 => k + col + 2,
                3 => 10 * (k + 1) + col + 1,
                _ => unreachable!(),
            };
            expected.push_str(&format!("{value}\n"));
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
        "{args:?}, recognition off={no_recognize}: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8(output.stdout).unwrap()
}

fn check(parallel: bool) {
    let path = std::env::temp_dir().join(format!(
        "wukong-bias-alias-{}-{parallel}.wk",
        std::process::id()
    ));
    std::fs::write(&path, source(parallel)).unwrap();
    let expected = oracle();
    assert_eq!(expected.lines().count(), 5 * LEN);
    // O0 retains function names, so another function cannot satisfy a dispatch obligation.
    let mir = invoke(&path, &["--emit=mir", "-O0"], false);
    let symbol = if parallel {
        "wukong_bias_bcast_f32_parallel"
    } else {
        "wukong_bias_bcast_f32"
    };
    for (name, count) in [
        ("aliased", 0),
        ("local_alias", 0),
        ("disjoint", 1),
        ("inplace", 1),
        ("read_alias", 1),
    ] {
        let (_, tail) = mir
            .split_once(&format!("fn {name}("))
            .unwrap_or_else(|| panic!("missing {name}:\n{mir}"));
        let body = tail.split("\n}\n").next().unwrap();
        assert_eq!(
            body.matches(&format!("call {symbol}(")).count(),
            count,
            "{name}:\n{body}"
        );
        if count == 0 {
            assert!(
                !body.contains("call wukong_bias_bcast"),
                "unsafe bias dispatch in {name}:\n{body}"
            );
            assert!(
                !body.contains("call wukong_parallel_for"),
                "aliased rows must remain ordered:\n{body}"
            );
        }
    }
    for opt in ["-O0", "-O2"] {
        let off = invoke(&path, &["--emit=mir", opt], true);
        assert!(
            !off.contains("call wukong_bias_bcast"),
            "recognition off: {off}"
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
fn serial_bias_alias_preserves_source_reads() {
    check(false);
}

#[test]
fn parallel_bias_alias_preserves_source_reads() {
    check(true);
}
