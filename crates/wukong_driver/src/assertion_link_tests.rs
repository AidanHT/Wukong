use super::*;
use std::process::Command;

/// Lower `src` as the file `t.wk`, with the driver's source locator, so every runtime failure
/// carries the `t.wk:line:col` site a linked executable reports.
fn lower_assertion_fixture(src: &str) -> (wukong_mir::Program, wukong_span::Interner) {
    let mut sm = SourceMap::new();
    let id = sm.add("t.wk".to_string(), src.to_string());
    let mut interner = wukong_span::Interner::new();
    let (module, pd) = wukong_parser::parse_module(src, id, &mut interner);
    assert!(pd.iter().all(|d| !d.is_error()), "{pd:?}");
    let (sema, sd) = wukong_sema::check(&module, &interner);
    assert!(sd.iter().all(|d| !d.is_error()), "{sd:?}");
    let locate = |span: wukong_span::Span| {
        let loc = sm.span_location(span);
        format!("{}:{}:{}", sm.name(span.source), loc.line, loc.col)
    };
    let (program, ld) = wukong_mir_build::lower_program_located(
        &module,
        &sema,
        &mut interner,
        wukong_mir_build::LowerOptions::default(),
        &locate,
    );
    assert!(ld.iter().all(|d| !d.is_error()), "{ld:?}");
    assert!(wukong_mir::verify::verify_program(&program).is_empty());
    (program, interner)
}

#[test]
fn assertion_aot_rust_and_c_shims_stop_before_trailing_effects() {
    let scratch = ScratchDir::new("assertion_aot").unwrap();
    let dir = &scratch.path;
    let rustc = link_rustc();
    // Unit-test executables live under <profile>/deps. Use the real runtime and production
    // archive selection: COFF carries all declared imports, including unused runtime symbols.
    let current = std::env::current_exe().unwrap();
    let deps = current.parent().unwrap();
    let profile = deps.parent().unwrap();
    let rust_shim = dir.join("shim.rs");
    std::fs::write(&rust_shim, WUKONG_RT_SHIM).unwrap();
    let c_shim = dir.join("shim.c");
    std::fs::write(&c_shim, WUKONG_RT_C).unwrap();
    // Every site below is the `assert` call's (or the slice access's) line and column in its
    // source text: `fn main() -> i32 { print(10); ` is 30 characters, so `assert` opens column 31.
    const ASSERTION: &str = "error: assertion failed at t.wk:1:31";
    const MESSAGE: &str = "error: assertion failed: ten is not twenty at t.wk:1:31";
    // `fn get(s: []i32, i: i64) -> i32 { print(i); return ` is 51 characters: `s[i]` is column 52.
    const BOUNDS: &str = "error: index out of bounds: the len is 2 but the index is 2 at t.wk:1:52";
    for (name, source, status, stdout, diagnostic_line) in [
        (
            "direct",
            "fn main() -> i32 { print(10); assert(false); print(99); return 0; }",
            1,
            "10\n",
            ASSERTION,
        ),
        (
            "message",
            "fn main() -> i32 { print(10); assert(10 == 20, \"ten is not twenty\"); print(99); \
             return 0; }",
            1,
            "10\n",
            MESSAGE,
        ),
        (
            "nested",
            "fn leaf() -> i32 { print(20); assert(false); print(21); return 5; }
                    fn middle() -> i32 { let x: i32 = leaf(); print(30); return x; }
                    fn main() -> i32 { print(10); let x: i32 = middle(); print(40); return x; }",
            1,
            "10\n20\n",
            ASSERTION,
        ),
        (
            "success",
            "fn leaf() -> i32 { print(20); assert(true); print(21); return 5; }
                     fn main() -> i32 { print(10); let x: i32 = leaf(); print(40); return x; }",
            5,
            "10\n20\n21\n40\n",
            "",
        ),
        (
            // A slice bounds failure in a callee: the executable resolves the failure edge to
            // `wukong_runtime::wukong_rt_bounds_fail` (Rust link) or the C shim's twin, both of
            // which print the shared diagnostic and exit 1 at the access.
            "bounds",
            "fn get(s: []i32, i: i64) -> i32 { print(i); return s[i]; }
                     fn main() -> i32 { let a: [i32; 2] = [3, 4]; print(get(a, 1));
                     print(get(a, 2)); print(99); return 0; }",
            1,
            "1\n4\n2\n",
            BOUNDS,
        ),
    ] {
        let (program, interner) = lower_assertion_fixture(source);
        let object = dir.join(format!("{name}.o"));
        std::fs::write(
            &object,
            wukong_codegen_cranelift::emit_object(&program, &interner).unwrap(),
        )
        .unwrap();
        for flavor in ["rust", "c"] {
            let executable = dir.join(format!("{name}_{flavor}{}", std::env::consts::EXE_SUFFIX));
            let linked = if flavor == "rust" {
                try_runtime_rlibs(runtime_rlib_candidates(profile), |runtime| {
                    let mut command = Command::new(&rustc);
                    command
                        .arg(&rust_shim)
                        .args(["--edition", "2021", "-C", "panic=abort"]);
                    if rlib_holds_bitcode_members(runtime) {
                        command.args(["-C", "lto=thin", "-C", "opt-level=3"]);
                    }
                    command
                        .arg("--extern")
                        .arg(format!("wukong_runtime={}", runtime.display()))
                        .arg("-L")
                        .arg(format!("dependency={}", deps.display()))
                        .arg("-C")
                        .arg(format!("link-arg={}", object.display()))
                        .arg("-o")
                        .arg(&executable)
                        .output()
                })
                .unwrap()
                .expect("the test build must provide a runtime archive")
                .1
            } else {
                Command::new(std::env::var_os("CC").unwrap_or_else(|| "cc".into()))
                    .args(cc_link_args(&object, &c_shim, &executable))
                    .output()
                    .expect("C assertion-shim regression requires the configured CC or cc")
            };
            assert!(
                linked.status.success(),
                "{name}/{flavor}: {}",
                String::from_utf8_lossy(&linked.stderr)
            );
            let ran = Command::new(&executable).output().unwrap();
            assert_eq!(ran.status.code(), Some(status), "{name}/{flavor}");
            assert_eq!(
                ran.stdout,
                stdout.as_bytes(),
                "{name}/{flavor}: trailing effects ran"
            );
            let diagnostic = String::from_utf8(ran.stderr).unwrap();
            if status == 1 {
                assert_eq!(
                    diagnostic.lines().collect::<Vec<_>>(),
                    [diagnostic_line],
                    "{name}/{flavor}"
                );
            } else {
                assert_eq!(diagnostic, "", "{name}/{flavor}");
            }
        }
    }
}
