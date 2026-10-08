//! `--time-passes`, `--time-trace` and `--depfile` observe a compile without changing it: the same
//! stdout, exit code and object bytes with and without them, through the real binary.
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

fn repo() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn wukongc(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_wukongc"))
        .args(args)
        .output()
        .unwrap()
}

fn scratch(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("wukong-{tag}-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn text(bytes: &[u8]) -> String {
    String::from_utf8_lossy(bytes).into_owned()
}

fn path(p: &Path) -> &str {
    p.to_str().expect("UTF-8 test path")
}

/// A path as a Make rule spells it (the scratch directory may contain a space).
fn make_target(p: &Path) -> String {
    path(p)
        .replace('$', "$$")
        .replace(' ', "\\ ")
        .replace('#', "\\#")
}

#[test]
fn timing_flags_leave_the_run_unchanged_and_report_every_stage() {
    let dir = scratch("time-passes");
    let fib = repo().join("examples/fib.wk");
    for (k, backend) in ["--backend=interp", "--backend=native"].iter().enumerate() {
        for level in ["-O0", "-O2"] {
            let plain = wukongc(&["--run", backend, level, path(&fib)]);
            assert!(plain.status.success(), "{}", text(&plain.stderr));
            assert!(!text(&plain.stderr).contains("time-passes"));

            let trace = dir.join(format!("trace-{k}{level}.json"));
            let trace_flag = format!("--time-trace={}", path(&trace));
            let timed = wukongc(&[
                "--run",
                backend,
                level,
                "--time-passes",
                &trace_flag,
                path(&fib),
            ]);
            assert_eq!(timed.status.code(), plain.status.code());
            assert_eq!(timed.stdout, plain.stdout, "{backend} {level}");

            let report = text(&timed.stderr);
            for stage in [
                "time-passes:",
                "  lex",
                "  parse",
                "  sema",
                "  lower",
                "  optimize",
            ] {
                assert!(report.contains(stage), "missing `{stage}`:\n{report}");
            }
            for line in ["  run", "  total", "mir: "] {
                assert!(report.contains(line), "missing `{line}`:\n{report}");
            }
            if *backend == "--backend=native" {
                assert!(report.contains("  jit-compile"), "{report}");
            }
            if level == "-O2" {
                assert!(report.contains("mem2reg"), "per-pass breakdown:\n{report}");
            }

            let json = std::fs::read_to_string(&trace).unwrap();
            assert!(json.starts_with("{\"traceEvents\":["), "{json}");
            assert!(json.contains("\"name\":\"optimize\""), "{json}");
            assert!(json.contains("\"ph\":\"X\""), "{json}");
        }
    }
}

#[test]
fn timed_object_is_byte_identical() {
    let dir = scratch("time-passes-obj");
    let fib = repo().join("examples/fib.wk");
    for level in ["-O0", "-O2"] {
        let plain_obj = dir.join(format!("plain{level}.o"));
        let timed_obj = dir.join(format!("timed{level}.o"));
        let plain = wukongc(&["--emit=obj", level, "-o", path(&plain_obj), path(&fib)]);
        assert!(plain.status.success(), "{}", text(&plain.stderr));
        let timed = wukongc(&[
            "--emit=obj",
            level,
            "--time-passes",
            "-o",
            path(&timed_obj),
            path(&fib),
        ]);
        assert!(timed.status.success(), "{}", text(&timed.stderr));
        let report = text(&timed.stderr);
        for line in ["  codegen", "isel+regalloc", "object-write"] {
            assert!(report.contains(line), "missing `{line}`:\n{report}");
        }
        assert_eq!(
            std::fs::read(&plain_obj).unwrap(),
            std::fs::read(&timed_obj).unwrap(),
            "{level}: --time-passes changed the object"
        );
    }
}

#[test]
fn depfile_names_the_root_and_every_import() {
    let dir = scratch("depfile");
    let root = repo().join("tests/run/import_multi.wk");

    let dep = dir.join("run.d");
    let dep_flag = format!("--depfile={}", path(&dep));
    let plain = wukongc(&["--run", path(&root)]);
    let with_dep = wukongc(&["--run", &dep_flag, path(&root)]);
    assert!(with_dep.status.success(), "{}", text(&with_dep.stderr));
    assert_eq!(with_dep.stdout, plain.stdout);
    let rule = std::fs::read_to_string(&dep).unwrap();
    for file in ["import_multi.wk", "mathlib.wk", "deep.wk"] {
        assert!(rule.contains(file), "missing {file}:\n{rule}");
    }
    // `--run` writes no artifact, so the rule's target is the dependency file itself.
    assert!(
        rule.starts_with(&format!("{}:", make_target(&dep))),
        "{rule}"
    );
    // Each import also gets an empty rule, so deleting it does not break the build.
    assert!(rule.contains("mathlib.wk:\n"), "{rule}");

    let obj = dir.join("prog.o");
    let dep = dir.join("obj.d");
    let dep_flag = format!("--depfile={}", path(&dep));
    let out = wukongc(&["--emit=obj", "-o", path(&obj), &dep_flag, path(&root)]);
    assert!(out.status.success(), "{}", text(&out.stderr));
    let rule = std::fs::read_to_string(&dep).unwrap();
    assert!(
        rule.starts_with(&format!("{}:", make_target(&obj))),
        "{rule}"
    );
}

#[test]
fn json_error_format_keeps_stderr_one_object_per_line() {
    let fib = repo().join("examples/fib.wk");
    let out = wukongc(&["--run", "--error-format=json", "--time-passes", path(&fib)]);
    assert!(out.status.success(), "{}", text(&out.stderr));
    let stderr = text(&out.stderr);
    let lines: Vec<&str> = stderr.lines().collect();
    assert!(
        lines.iter().any(|l| l.starts_with("{\"time_passes\":{")),
        "no time_passes object:\n{stderr}"
    );
    for line in lines {
        assert!(
            line.starts_with('{') && line.ends_with('}'),
            "not a JSON object line: {line:?}"
        );
    }
}
