//! Smoke tests for every `--emit` stage.
//!
//! Front-end stages (`tokens`, `ast`) must succeed for every example and run-suite program. The
//! lowering stages (`mir-high`, `mir`) must succeed for the run-suite programs, which are all known
//! to lower end to end. This guards the pretty-printers and the MIR verifier (which runs under
//! `--emit=mir`). The textual LLVM-IR emitter may refuse a construct it cannot express faithfully
//! (`C0001`, exit 2, one of `wukong_codegen_llvm::DECLINED`), and how many corpus programs it
//! refuses is a ratchet: [`LLVM_DECLINE_RATCHET`].

use std::path::{Path, PathBuf};
use std::process::Command;

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join("..")
}

fn wk_files(dir: &Path) -> Vec<PathBuf> {
    let mut v: Vec<PathBuf> = std::fs::read_dir(dir)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", dir.display()))
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().map(|x| x == "wk").unwrap_or(false))
        .collect();
    v.sort();
    v
}

fn emit(path: &Path, stage: &str) -> (bool, String, String) {
    emit_opt(path, stage, 0)
}

fn emit_opt(path: &Path, stage: &str, opt: u8) -> (bool, String, String) {
    let out = Command::new(env!("CARGO_BIN_EXE_wukongc"))
        .arg(format!("--emit={stage}"))
        .arg(format!("-O{opt}"))
        .arg(path)
        .output()
        .expect("spawn wukongc");
    (
        out.status.success(),
        String::from_utf8_lossy(&out.stdout).into_owned(),
        String::from_utf8_lossy(&out.stderr).into_owned(),
    )
}

#[test]
fn frontend_stages_succeed_for_all_sources() {
    let root = repo_root();
    let mut files = wk_files(&root.join("examples"));
    files.extend(wk_files(&root.join("tests").join("run")));
    assert!(!files.is_empty());

    for f in &files {
        for stage in ["tokens", "ast"] {
            let (ok, stdout, stderr) = emit(f, stage);
            let name = f.file_name().unwrap().to_string_lossy();
            assert!(ok, "{name}: --emit={stage} failed:\n{stderr}");
            assert!(
                !stdout.trim().is_empty(),
                "{name}: --emit={stage} produced no output"
            );
        }
    }
}

/// The most `(program, -O level)` pairs of the run corpus and the examples that `--emit=llvm-ir`
/// may refuse. It is a ratchet: lowering a refused construct lowers it, and raising it needs a
/// `DECLINED` row that says why the construct has no faithful LLVM spelling. At 0, every program
/// that compiles at all also emits a module.
const LLVM_DECLINE_RATCHET: usize = 0;

/// Did `--emit=llvm-ir` refuse (the backend's checked decline), as opposed to fail?
fn llvm_declined(stderr: &str) -> bool {
    stderr.contains("C0001") && stderr.contains("LLVM emission [")
}

#[test]
// The ratchet compare stays a `<=` even while its bound is 0: raising the bound must not require
// rewriting the check.
#[allow(clippy::absurd_extreme_comparisons)]
fn lowering_stages_succeed_for_run_suite() {
    let dir = repo_root().join("tests").join("run");
    let mut declined: Vec<String> = Vec::new();
    // Exercise both unoptimized and fully optimized output: -O2 turns on mem2reg/LICM, so the MIR
    // verifier (run under --emit=mir) and the phi-emitting LLVM-IR backend are checked on the
    // optimized, block-parameter form too — not just the alloca-based -O0 form.
    for f in &wk_files(&dir) {
        for opt in [0u8, 2] {
            for stage in ["mir-high", "mir", "llvm-ir"] {
                let (ok, stdout, stderr) = emit_opt(f, stage, opt);
                let name = f.file_name().unwrap().to_string_lossy();
                if stage == "llvm-ir" && !ok && llvm_declined(&stderr) {
                    declined.push(format!("{name} -O{opt}: {}", stderr.trim()));
                    continue;
                }
                assert!(ok, "{name}: --emit={stage} -O{opt} failed:\n{stderr}");
                assert!(
                    !stdout.trim().is_empty(),
                    "{name}: --emit={stage} -O{opt} produced no output"
                );
                // The MIR verifier prints internal-compiler-error lines to stderr; there must be none.
                assert!(
                    !stderr.contains("internal compiler error"),
                    "{name}: --emit={stage} -O{opt} reported a verifier ICE:\n{stderr}"
                );
            }
        }
    }
    // The examples too, wherever the front end lowers them (a few use surface the native lowering
    // does not support yet; those fail with exit 1 before the LLVM stage and are not declines).
    for f in &wk_files(&repo_root().join("examples")) {
        let (ok, _, stderr) = emit_opt(f, "llvm-ir", 2);
        if !ok && llvm_declined(&stderr) {
            let name = f.file_name().unwrap().to_string_lossy();
            declined.push(format!("{name} -O2: {}", stderr.trim()));
        }
    }
    assert!(
        declined.len() <= LLVM_DECLINE_RATCHET,
        "{} --emit=llvm-ir refusals exceed the ratchet of {LLVM_DECLINE_RATCHET}:\n{}",
        declined.len(),
        declined.join("\n")
    );
}

/// Run `wukongc args…`; (success, stdout, stderr), stdout as bytes.
fn wukongc(args: &[&std::ffi::OsStr]) -> (bool, Vec<u8>, String) {
    let out = Command::new(env!("CARGO_BIN_EXE_wukongc"))
        .args(args)
        .output()
        .expect("spawn wukongc");
    (
        out.status.success(),
        out.stdout,
        String::from_utf8_lossy(&out.stderr).into_owned(),
    )
}

/// Split a multi-artifact stdout at its `==> <stage> <==` header lines: (stage, artifact).
fn split_artifacts(stdout: &str) -> Vec<(String, String)> {
    let mut parts: Vec<(String, String)> = Vec::new();
    for line in stdout.split_inclusive('\n') {
        let header = line
            .strip_prefix("==> ")
            .and_then(|l| l.trim_end().strip_suffix(" <=="));
        match (header, parts.last_mut()) {
            (Some(stage), _) => parts.push((stage.to_string(), String::new())),
            (None, Some((_, text))) => text.push_str(line),
            (None, None) => panic!("artifact text before the first header: {line:?}"),
        }
    }
    parts
}

/// A multi-artifact `--emit=a,b,..` is ONE compile, and without `exe` in the list every artifact
/// it prints is byte for byte what `--emit=<that stage>` prints alone: the same program, the same
/// optimizer, the same printers. The stages come out in pipeline order whatever order they were
/// written in, each under its header. Programs: a plain one, one with imports, a recognized kernel
/// and a GEMM, at -O0 and -O2.
#[test]
fn multi_artifact_emit_prints_each_single_stage_artifact() {
    let root = repo_root();
    let stages = ["tokens", "ast", "mir-high", "mir", "mir-text", "llvm-ir"];
    // Written backwards: the order on the command line is not the order of the artifacts.
    let list = format!(
        "--emit={}",
        stages.iter().rev().copied().collect::<Vec<_>>().join(",")
    );
    for program in [
        root.join("examples").join("fib.wk"),
        root.join("tests").join("run").join("import_multi.wk"),
        root.join("tests")
            .join("run")
            .join("transcendental_kernel.wk"),
        root.join("tests").join("run").join("tensor_matmul.wk"),
    ] {
        let name = program.file_name().unwrap().to_string_lossy().into_owned();
        for opt in ["-O0", "-O2"] {
            let (ok, stdout, stderr) = wukongc(&[list.as_ref(), opt.as_ref(), program.as_os_str()]);
            assert!(ok, "{name} {list} {opt}:\n{stderr}");
            let stdout = String::from_utf8(stdout).expect("text artifacts are UTF-8");
            let parts = split_artifacts(&stdout);
            assert_eq!(
                parts.iter().map(|(s, _)| s.as_str()).collect::<Vec<_>>(),
                stages,
                "{name} {opt}: the artifacts, in pipeline order"
            );
            for (stage, text) in &parts {
                let single = format!("--emit={stage}");
                let (ok, alone, stderr) =
                    wukongc(&[single.as_ref(), opt.as_ref(), program.as_os_str()]);
                assert!(ok, "{name} {single} {opt}:\n{stderr}");
                assert_eq!(
                    text.as_bytes(),
                    &alone[..],
                    "{name} {opt}: the `{stage}` artifact of {list} differs from {single}"
                );
            }
        }
    }
}

/// With `-o <stem>` a multi-artifact compile writes every artifact to `<stem>.<ext>` — the text
/// ones byte-identical to their single-stage stdout, the object byte-identical to `--emit=obj`'s —
/// prints nothing on stdout, says `wrote <file>` for each, and its `--depfile` rule names every
/// file it wrote.
#[test]
fn multi_artifact_emit_writes_every_artifact_at_the_stem() {
    let dir = std::env::temp_dir().join(format!("wukong-emit-multi-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let program = repo_root()
        .join("tests")
        .join("run")
        .join("transcendental_kernel.wk");
    let stem = dir.join("k");
    let depfile = dir.join("k.d");
    let dep_flag = format!("--depfile={}", depfile.display());
    let (ok, stdout, stderr) = wukongc(&[
        "--emit=obj,llvm-ir,mir-text,ast".as_ref(),
        "-O2".as_ref(),
        "-o".as_ref(),
        stem.as_os_str(),
        dep_flag.as_ref(),
        program.as_os_str(),
    ]);
    assert!(ok, "{stderr}");
    assert!(stdout.is_empty(), "with -o nothing goes to stdout");
    let written: Vec<_> = ["ast", "mir", "ll", "o"]
        .iter()
        .map(|ext| stem.with_extension(ext))
        .collect();
    for file in &written {
        assert!(
            stderr.contains(&format!("wrote {}", file.display())),
            "{}: {stderr}",
            file.display()
        );
    }
    for (stage, file) in ["ast", "mir-text", "llvm-ir"].iter().zip(&written) {
        let single = format!("--emit={stage}");
        let (ok, alone, stderr) = wukongc(&[single.as_ref(), "-O2".as_ref(), program.as_os_str()]);
        assert!(ok, "{single}: {stderr}");
        assert_eq!(std::fs::read(file).unwrap(), alone, "{}", file.display());
    }
    let single_obj = dir.join("single.o");
    let (ok, _, stderr) = wukongc(&[
        "--emit=obj".as_ref(),
        "-O2".as_ref(),
        "-o".as_ref(),
        single_obj.as_os_str(),
        program.as_os_str(),
    ]);
    assert!(ok, "{stderr}");
    assert_eq!(
        std::fs::read(&written[3]).unwrap(),
        std::fs::read(&single_obj).unwrap(),
        "the list's object is --emit=obj's object"
    );
    // One rule, every written file a target: `<targets>: \` then one source per line.
    let rule = std::fs::read_to_string(&depfile).unwrap();
    let targets = rule.lines().next().unwrap().strip_suffix(": \\").unwrap();
    let expected: Vec<String> = written.iter().map(|f| f.display().to_string()).collect();
    assert_eq!(targets, expected.join(" "), "{rule}");
    let _ = std::fs::remove_dir_all(&dir);
}

/// A list that ends in `obj` with `--grad-of=<fn>` is the gradient export's compile, as `exe` makes
/// a list the executable's: its object is `--emit=obj --grad-of`'s, byte for byte, and the MIR and
/// IR it also writes are the program that object is built from — the generated `<fn>_grad`
/// included — not what `--emit=mir` prints alone (a lone `mir` or `llvm-ir` ignores `--grad-of`).
#[test]
fn multi_artifact_emit_ending_in_a_gradient_export_shows_that_compile() {
    let dir = std::env::temp_dir().join(format!("wukong-emit-grad-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let program = dir.join("cubic.wk");
    std::fs::write(
        &program,
        "fn loss(x: [f32; 1], mut out: [f32; 1]) -> f32 {\n    let a: f32 = x[0];\n    \
         let l: f32 = a * a * a;\n    out[0] = l;\n    return l;\n}\n",
    )
    .unwrap();
    // Autodiff refuses to run with kernel recognition off; this test is about the artifacts.
    let wukongc = |args: &[&std::ffi::OsStr]| {
        let out = Command::new(env!("CARGO_BIN_EXE_wukongc"))
            .env_remove(wukong_driver::NO_RECOGNIZE_ENV)
            .args(args)
            .output()
            .expect("spawn wukongc");
        assert!(
            out.status.success(),
            "{args:?}: {}",
            String::from_utf8_lossy(&out.stderr)
        );
        String::from_utf8_lossy(&out.stdout).into_owned()
    };
    let stem = dir.join("k");
    wukongc(&[
        "--emit=mir,llvm-ir,obj".as_ref(),
        "--grad-of=loss".as_ref(),
        "-O0".as_ref(),
        "-o".as_ref(),
        stem.as_os_str(),
        program.as_os_str(),
    ]);
    let mir = std::fs::read_to_string(stem.with_extension("mir-low")).unwrap();
    let ll = std::fs::read_to_string(stem.with_extension("ll")).unwrap();
    assert!(mir.contains("fn loss_grad("), "{mir}");
    assert!(ll.contains("@loss_grad("), "{ll}");
    let single = dir.join("single.o");
    wukongc(&[
        "--emit=obj".as_ref(),
        "--grad-of=loss".as_ref(),
        "-O0".as_ref(),
        "-o".as_ref(),
        single.as_os_str(),
        program.as_os_str(),
    ]);
    assert_eq!(
        std::fs::read(stem.with_extension("o")).unwrap(),
        std::fs::read(&single).unwrap(),
        "the list's object is --emit=obj --grad-of's object"
    );
    for stage in ["mir", "llvm-ir"] {
        let single = format!("--emit={stage}");
        let alone = wukongc(&[
            single.as_ref(),
            "--grad-of=loss".as_ref(),
            "-O0".as_ref(),
            program.as_os_str(),
        ]);
        assert!(!alone.contains("loss_grad"), "{single} --grad-of: {alone}");
    }
    let _ = std::fs::remove_dir_all(&dir);
}

/// `docs/llvm-setup.md` shows the module `--emit=llvm-ir -O2 examples/fib.wk` prints, for readers
/// to compile by hand; it must be exactly what the compiler emits. The `target` lines name the
/// host, so they are compared only for shape.
#[test]
fn llvm_setup_doc_shows_the_emitted_fib_module() {
    let doc = std::fs::read_to_string(repo_root().join("docs").join("llvm-setup.md")).unwrap();
    let marker = "<!-- emitted: wukongc --emit=llvm-ir -O2 examples/fib.wk -->";
    let start = doc
        .find(marker)
        .expect("the fib module block in docs/llvm-setup.md");
    let block = doc[start + marker.len()..].trim_start();
    let block = block.strip_prefix("```llvm").expect("an llvm fence");
    let block = &block[..block.find("```").expect("a closing fence")];
    let fib = repo_root().join("examples").join("fib.wk");
    let (ok, emitted, stderr) = emit_opt(&fib, "llvm-ir", 2);
    assert!(ok, "{stderr}");
    let body = |text: &str| -> Vec<String> {
        text.lines()
            .filter(|l| !l.starts_with("target ") && !l.trim().is_empty())
            .map(|l| l.trim_end().to_string())
            .collect()
    };
    assert_eq!(
        body(block),
        body(&emitted),
        "docs/llvm-setup.md's fib module is stale; paste the current output"
    );
    // The doc shows the module an x86-64 Windows host prints: layout, then triple.
    assert!(
        block.contains("\ntarget datalayout = \"e-m:w-")
            && block.contains("\ntarget triple = \"x86_64-pc-windows-msvc\""),
        "{block}"
    );
    // The emitted module names THIS host, whatever it is; the exact lines per host are pinned by
    // `wukong_codegen_llvm`'s `module_names_the_host_target`. An aarch64 host prints its own triple
    // and no layout, so only the shape is host-independent: the `target` lines come first, the
    // layout (if any) before the triple, and the triple's architecture is the host's.
    let target: Vec<&str> = emitted
        .lines()
        .skip(1)
        .take_while(|l| l.starts_with("target "))
        .collect();
    let layout = target
        .iter()
        .position(|l| l.starts_with("target datalayout = \""));
    let triple = target
        .iter()
        .position(|l| l.starts_with("target triple = \""));
    assert_eq!(
        emitted.matches("\ntarget ").count(),
        target.len(),
        "{emitted}"
    );
    if let (Some(layout), Some(triple)) = (layout, triple) {
        assert!(layout < triple, "{emitted}");
    }
    if let Some(triple) = triple {
        let arch = format!("target triple = \"{}-", std::env::consts::ARCH);
        assert!(target[triple].starts_with(&arch), "{emitted}");
    }
    // The Tier 1 hosts (x86-64 Windows MSVC, x86-64 Linux GNU) carry both lines.
    if cfg!(all(
        target_arch = "x86_64",
        any(
            all(target_os = "windows", target_env = "msvc"),
            all(target_os = "linux", target_env = "gnu")
        )
    )) {
        assert!(layout.is_some() && triple.is_some(), "{emitted}");
    }
}
