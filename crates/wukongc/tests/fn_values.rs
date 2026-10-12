//! Function values end to end through the binary: what a call through one lowers to, and what the
//! optimizer makes of it when it can see the function (`wukong_opt::devirtualize_program`).
//!
//! `tests/run/fn_values_devirt.wk` holds every function value in a local the optimizer can follow:
//! at `-O0` each call through one is an `Op::CallIndirect` (`call_indirect`); at `-O1` mem2reg has
//! forwarded the `func_addr` each local holds and every such call is a direct `call`; at `-O2`
//! those direct calls are inlined like any other. `tests/run/fn_values.wk` keeps run-time choices
//! (an element picked by a loop counter) that stay indirect at every level.

use std::path::{Path, PathBuf};
use std::process::Command;

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join("..")
}

/// `--emit=mir -O<level>` of the run-suite program `name`.
fn mir(name: &str, level: u8) -> String {
    let path = repo_root().join("tests").join("run").join(name);
    let out = Command::new(env!("CARGO_BIN_EXE_wukongc"))
        .arg("--emit=mir")
        .arg(format!("-O{level}"))
        .arg(&path)
        .output()
        .expect("spawn wukongc");
    assert!(
        out.status.success(),
        "{name} -O{level}: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8_lossy(&out.stdout).into_owned()
}

/// The text of function `name` in a `--emit=mir` dump.
fn function<'a>(dump: &'a str, name: &str) -> &'a str {
    let head = format!("fn {name}(");
    let start = dump
        .find(&head)
        .unwrap_or_else(|| panic!("no `fn {name}` in:\n{dump}"));
    let rest = &dump[start..];
    &rest[..rest.find("\n}\n").map_or(rest.len(), |e| e + 3)]
}

#[test]
fn a_known_callee_is_called_directly_once_optimized() {
    let o0 = mir("fn_values_devirt.wk", 0);
    let main0 = function(&o0, "main");
    assert_eq!(main0.matches("call_indirect").count(), 5, "{main0}");

    let o1 = mir("fn_values_devirt.wk", 1);
    let main1 = function(&o1, "main");
    assert!(
        !main1.contains("call_indirect"),
        "-O1 left an indirect call:\n{main1}"
    );
    assert_eq!(main1.matches("call scale3(").count(), 3, "{main1}");
    assert_eq!(main1.matches("call shift(").count(), 2, "{main1}");

    let o2 = mir("fn_values_devirt.wk", 2);
    let main2 = function(&o2, "main");
    assert!(!main2.contains("call_indirect"), "{main2}");
    assert!(
        !main2.contains("call scale3(") && !main2.contains("call shift("),
        "-O2 did not inline the devirtualized calls:\n{main2}"
    );
}

#[test]
fn a_run_time_choice_stays_an_indirect_call() {
    let o0 = mir("fn_values_dispatch.wk", 0);
    // The dispatch table is indexed by each instruction's op, read from memory.
    let exec = function(&o0, "execute");
    assert!(exec.contains("call_indirect"), "{exec}");
    // A function-typed parameter given a run-time value is compiled once more, taking it as a
    // pointer and calling through it.
    let map6 = function(&o0, "map6$fn1$i32$to$i32");
    assert!(map6.contains("call_indirect"), "{map6}");
    // A `@parallel` loop around a call through a run-time value is not outlined: the region
    // proof cannot see what the callee writes.
    let fill8 = function(&o0, "fill8$fn1$i32$to$i32");
    assert!(fill8.contains("call_indirect"), "{fill8}");
    assert!(!fill8.contains("wukong_parallel_for"), "{fill8}");
    // At -O2 everything is inlined into `main`, and the run-time choices are still indirect.
    let o2 = mir("fn_values_dispatch.wk", 2);
    let main2 = function(&o2, "main");
    assert!(main2.contains("call_indirect"), "{main2}");
}
