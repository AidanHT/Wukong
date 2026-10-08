use super::*;
use std::process::Command;

fn lower_assertion_fixture(src: &str) -> (wukong_mir::Program, wukong_span::Interner) {
    let mut interner = wukong_span::Interner::new();
    let (module, pd) = wukong_parser::parse_module(src, wukong_span::SourceId(0), &mut interner);
    assert!(pd.iter().all(|d| !d.is_error()), "{pd:?}");
    let (sema, sd) = wukong_sema::check(&module, &interner);
    assert!(sd.iter().all(|d| !d.is_error()), "{sd:?}");
    let (program, ld) = wukong_mir_build::lower_program(&module, &sema, &mut interner);
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
    const ASSERTION: &str = "assertion failed";
    const BOUNDS: &str = "error: index out of bounds: the len is 2 but the index is 2";
    for (name, source, status, stdout, diagnostic_line) in [
        (
            "direct",
            "fn main() -> i32 { print(10); assert(false); print(99); return 0; }",
            1,
            "10\n",
            ASSERTION,
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
