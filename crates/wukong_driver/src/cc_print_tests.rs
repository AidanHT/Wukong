use super::*;
use std::process::Command;
use wukong_mir::{Builder, MirType, Op, Program};

fn lower(src: &str) -> (Program, Interner) {
    let mut interner = Interner::new();
    let (module, diagnostics) =
        wukong_parser::parse_module(src, wukong_span::SourceId(0), &mut interner);
    assert!(diagnostics.iter().all(|d| !d.is_error()), "{diagnostics:?}");
    let (sema, diagnostics) = wukong_sema::check(&module, &interner);
    assert!(diagnostics.iter().all(|d| !d.is_error()), "{diagnostics:?}");
    let (program, diagnostics) = wukong_mir_build::lower_program(&module, &sema, &mut interner);
    assert!(diagnostics.iter().all(|d| !d.is_error()), "{diagnostics:?}");
    assert!(wukong_mir::verify::verify_program(&program).is_empty());
    (program, interner)
}

#[test]
fn gate_covers_every_float_type_and_preserves_internal_call_resolution() {
    for ty in [MirType::F16, MirType::BF16, MirType::F32, MirType::F64] {
        for name in ["print", "println"] {
            let mut interner = Interner::new();
            let print = interner.intern(name);
            let mut helper = Builder::new(interner.intern("helper"), MirType::Void);
            // Parameters, not just constants, must trigger the capability check.
            let arg = helper.add_param(ty.clone());
            helper.build_void(Op::Call {
                func: print,
                args: vec![arg],
            });
            helper.ret(None);
            let mut program = Program::new();
            program.funcs.push(helper.finish());
            assert!(wukong_mir::verify::verify_program(&program).is_empty());
            assert_eq!(
                cc_float_print_requirement(&program, &interner),
                Some(format!("{name}({}) in `helper`", ty.display()))
            );

            // Cranelift resolves a defined function before looking for an intrinsic. A user
            // function named print/println with a float parameter need not print any float.
            let mut internal = Builder::new(print, MirType::Void);
            internal.add_param(ty.clone());
            internal.ret(None);
            program.funcs.push(internal.finish());
            assert!(wukong_mir::verify::verify_program(&program).is_empty());
            assert_eq!(cc_float_print_requirement(&program, &interner), None);
        }
    }
}

#[test]
fn forced_c_link_declines_float_print_before_writing_link_inputs_or_output() {
    let scratch = ScratchDir::new("cc_print_decline").unwrap();
    for ty in ["f32", "f64"] {
        for name in ["print", "println"] {
            let source = format!(
                "fn helper(x: {ty}) {{ {name}(x); }}
                 fn main() -> i32 {{ helper(1.23456789012345); return 0; }}"
            );
            let (program, interner) = lower(&source);
            let stem = format!("declined_{name}_{ty}");
            let out = scratch
                .path
                .join(format!("{stem}{}", std::env::consts::EXE_SUFFIX));
            let object = scratch.path.join(format!("{stem}.o"));
            let opts = Options {
                input: PathBuf::from(format!("{stem}.wk")),
                ..Options::default()
            };
            // Deliberately no object file: refusal must precede writing C or invoking CC.
            assert_eq!(
                cc_link(
                    &program,
                    &interner,
                    &opts,
                    Some(&scratch),
                    &stem,
                    &object,
                    &out
                ),
                exit::UNIMPLEMENTED
            );
            assert!(!out.exists());
            assert!(!scratch.path.join(format!("{stem}_rt.c")).exists());
        }
    }
}

#[test]
fn forced_c_link_preserves_integer_output_and_float_computation() {
    let (program, mut interner) = lower(
        "fn main() -> i32 {
             let lo: i64 = -9223372036854775807 - 1;
             let hi: u64 = 18446744073709551615;
             print(lo);
             println(hi);
             print(true);
             println(false);
             let x: f64 = 7.5;
             let y: f64 = x % 2.0;
             assert(y == 1.5);
             let z: i32 = y as i32;
             print(z);
             return 7;
         }",
    );
    let expected = b"-9223372036854775808\n18446744073709551615\n1\n0\n1\n";
    let main = interner.intern("main");
    let oracle = wukong_interp::run_with_output(&program, main, &interner).unwrap();
    assert_eq!(oracle, (7, expected.to_vec()));
    assert_eq!(cc_float_print_requirement(&program, &interner), None);
    let scratch = ScratchDir::new("cc_print_supported").unwrap();
    let object = scratch.path.join("supported.o");
    std::fs::write(
        &object,
        wukong_codegen_cranelift::emit_object(&program, &interner).unwrap(),
    )
    .unwrap();
    let out = scratch
        .path
        .join(format!("supported{}", std::env::consts::EXE_SUFFIX));
    let opts = Options {
        input: PathBuf::from("supported.wk"),
        ..Options::default()
    };
    assert_eq!(
        cc_link(
            &program,
            &interner,
            &opts,
            Some(&scratch),
            "supported",
            &object,
            &out
        ),
        exit::OK,
        "this regression requires the configured CC or cc"
    );
    let ran = Command::new(&out).output().unwrap();
    assert_eq!(ran.status.code(), Some(7));
    assert_eq!(ran.stdout, expected);
    assert!(ran.stderr.is_empty());
}

#[test]
fn bypassed_gate_float_symbol_fails_without_printing_an_approximation() {
    let (program, interner) =
        lower("fn main() -> i32 { let x: f64 = 1.23456789012345; print(x); print(99); return 0; }");
    let scratch = ScratchDir::new("cc_print_guard").unwrap();
    let object = scratch.path.join("guard.o");
    std::fs::write(
        &object,
        wukong_codegen_cranelift::emit_object(&program, &interner).unwrap(),
    )
    .unwrap();
    let runtime = scratch.path.join("guard_rt.c");
    std::fs::write(&runtime, c_runtime_source()).unwrap();
    let out = scratch
        .path
        .join(format!("guard{}", std::env::consts::EXE_SUFFIX));
    // Bypass ONLY the capability gate to prove the generated runtime's last defense. This uses
    // a real Cranelift object, including the float ABI, rather than a hand-written C caller.
    let linked = Command::new(std::env::var_os("CC").unwrap_or_else(|| "cc".into()))
        .args(cc_link_args(&object, &runtime, &out))
        .output()
        .expect("this regression requires the configured CC or cc");
    assert!(
        linked.status.success(),
        "{}",
        String::from_utf8_lossy(&linked.stderr)
    );
    let ran = Command::new(&out).output().unwrap();
    assert_eq!(ran.status.code(), Some(exit::UNIMPLEMENTED));
    assert!(
        ran.stdout.is_empty(),
        "lossy output or trailing print executed"
    );
    assert_eq!(
        String::from_utf8(ran.stderr).unwrap().lines().collect::<Vec<_>>(),
        ["error: floating-point printing requires the Rust runtime link; C fallback cannot preserve output"]
    );
}

#[test]
fn float_print_object_emission_remains_available() {
    let (program, interner) = lower("fn main() -> i32 { print(1.23456789012345); return 0; }");
    let scratch = ScratchDir::new("cc_print_object").unwrap();
    let object = scratch.path.join("float.o");
    let opts = Options {
        input: PathBuf::from("float.wk"),
        output: Some(object.clone()),
        emit: EmitStage::Obj,
        ..Options::default()
    };
    let mut timer = report::PassTimer::new(&opts);
    assert_eq!(
        emit_native(&program, &interner, &opts, &mut timer),
        exit::OK
    );
    assert!(std::fs::metadata(object).unwrap().len() > 0);
}
