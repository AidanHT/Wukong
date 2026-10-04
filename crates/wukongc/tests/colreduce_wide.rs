//! P7.18: wide column folds must dispatch and match independently derived exact values.

use std::process::Command;

#[path = "common/mod.rs"]
mod common;

const SYMBOLS: [&str; 7] = [
    "wukong_colsum_f32",
    "wukong_colmax_f32",
    "wukong_colmin_f32",
    "wukong_colmaxabs_f32",
    "wukong_colsumsq_f32",
    "wukong_colargmax_i32",
    "wukong_colargmin_i32",
];

// Closed forms, not a second implementation of the source reduction loops. All numerators
// are small integers and all denominators powers of two, so f64 and f32 are exact here.
fn oracle() -> String {
    let mut lines = Vec::new();
    for cols in [75, 11] {
        for kind in 0..7 {
            for j in 0..cols {
                let k = f64::from(j + 1);
                let value = match kind {
                    0 => -k / 2.0,
                    1 => 5.0 * k / 4.0,
                    2 => -6.0 * k / 4.0,
                    3 => 6.0 * k / 4.0,
                    4 => (if cols == 75 { 156.0 } else { 136.0 }) * k * k / 16.0,
                    5 if cols == 11 => f64::from((3_i32 - j).rem_euclid(10)),
                    6 if cols == 11 => f64::from((4_i32 - j).rem_euclid(10)),
                    5 => f64::from((3_i32 - j).rem_euclid(10).min((6 - j).rem_euclid(10))),
                    6 => f64::from((4_i32 - j).rem_euclid(10).min((7 - j).rem_euclid(10))),
                    _ => unreachable!(),
                };
                lines.push(value.to_string());
            }
        }
    }
    assert_eq!(lines.len(), 602);
    lines.join("\n") + "\n"
}

fn invoke(args: &[&str], no_recognize: bool) -> String {
    let mut command = Command::new(env!("CARGO_BIN_EXE_wukongc"));
    command
        .args(args)
        .arg(common::repo_root().join("tests/run/colreduce_wide.wk"));
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
    String::from_utf8(output.stdout).expect("UTF-8 stdout")
}

#[test]
fn wide_column_dispatch_and_exact_values() {
    let source = include_str!("../../../tests/run/colreduce_wide.wk");
    let directives = common::parse_directives(source, "colreduce_wide.wk").unwrap();
    assert_eq!(directives.exit, Some(0));
    let expected = oracle();
    let exact: Vec<_> = expected
        .lines()
        .map(|s| common::Expectation::Exact(s.to_owned()))
        .collect();
    assert_eq!(
        directives.out, exact,
        "fixture must pin every exact oracle lane"
    );

    // These APIs encode the reduction kind in the symbol, with no numeric kind argument.
    // At O0 both source functions remain separate, so each symbol must occur once per shape.
    for opt in ["-O0", "-O2"] {
        let on = invoke(&["--emit=mir", opt], false);
        if opt == "-O0" {
            for function in ["wide_75", "wide_11"] {
                let (_, tail) = on
                    .split_once(&format!("fn {function}("))
                    .unwrap_or_else(|| panic!("missing {function}:\n{on}"));
                let body = tail.split("\n}\n").next().unwrap();
                for symbol in SYMBOLS {
                    assert_eq!(
                        body.matches(&format!("call {symbol}(")).count(),
                        1,
                        "{function} must dispatch {symbol} exactly once:\n{body}"
                    );
                }
            }
        }
        for symbol in SYMBOLS {
            let count = on.matches(&format!("call {symbol}(")).count();
            assert!(
                count >= 2,
                "{opt}: expected both widths to dispatch {symbol}:\n{on}"
            );
        }
        let off = invoke(&["--emit=mir", opt], true);
        assert!(
            !off.contains("call wukong_"),
            "{opt}: recognition off retained runtime kernel calls:\n{off}"
        );
        for backend in ["--backend=interp", "--backend=native"] {
            for no_recognize in [false, true] {
                let actual = invoke(&["--run", backend, opt], no_recognize);
                assert_eq!(
                    actual.as_bytes(),
                    expected.as_bytes(),
                    "{backend} {opt}, recognition off={no_recognize}"
                );
                common::check_stdout(&actual, &directives).unwrap();
            }
        }
    }
}
