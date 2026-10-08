use super::*;
use std::process::Command;
use wukong_mir::Program;

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

/// Link `program` through the forced C fallback (the production `cc_link` boundary) and run it.
fn link_with_cc_and_run(program: &Program, interner: &Interner, stem: &str) -> (i32, Vec<u8>) {
    let scratch = ScratchDir::new(&format!("cc_print_{stem}")).unwrap();
    let object = scratch.path.join(format!("{stem}.o"));
    std::fs::write(
        &object,
        wukong_codegen_cranelift::emit_object(program, interner).unwrap(),
    )
    .unwrap();
    let out = scratch
        .path
        .join(format!("{stem}{}", std::env::consts::EXE_SUFFIX));
    let opts = Options {
        input: PathBuf::from(format!("{stem}.wk")),
        ..Options::default()
    };
    assert_eq!(
        cc_link(&opts, Some(&scratch), stem, &object, &out),
        exit::OK,
        "this regression requires the configured CC or cc"
    );
    let ran = Command::new(&out).output().unwrap();
    assert!(
        ran.stderr.is_empty(),
        "{}",
        String::from_utf8_lossy(&ran.stderr)
    );
    (ran.status.code().unwrap(), ran.stdout)
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
    assert_eq!(
        link_with_cc_and_run(&program, &interner, "supported"),
        (7, expected.to_vec())
    );
}

/// P15.26 / P13.19: the C fallback prints floats exactly as the oracle does — Rust `Display`'s
/// shortest round-trip digits, laid out positionally — instead of `%g`'s six significant digits.
/// The expected lines are hand-derived from that rule (the shortest decimal that parses back to
/// the double, closest when several do), then the linked executable is also compared with the
/// interpreter byte for byte. The values cover each layout branch (integral, fractional, leading
/// zeros, trailing zeros past the digits), the f32-widened 0.1, signed zero, NaN, both infinities,
/// the smallest subnormal and a power of two whose shortest form needs zero padding.
#[test]
fn forced_c_link_prints_floats_exactly_like_the_oracle() {
    let (program, mut interner) = lower(
        "fn main() -> i32 {
             let z: f64 = 0.0;
             print(0.1);
             let a: f32 = 0.1;
             print(a);
             let mut e21: f64 = 1.0;
             let mut k: i32 = 0;
             while k < 21 { e21 = e21 * 10.0; k = k + 1; }
             print(e21);
             print(0.0000001);
             print(-z);
             print(z / z);
             print(1.0 / z);
             print(-1.0 / z);
             print(123456789.125);
             print(2.5);
             print(1.0);
             let third: f64 = 1.0 / 3.0;
             print(third);
             let mut tiny: f64 = 1.0;
             k = 0;
             while k < 1074 { tiny = tiny / 2.0; k = k + 1; }
             print(tiny);
             let mut big: f64 = 1.0;
             k = 0;
             while k < 60 { big = big * 2.0; k = k + 1; }
             print(big);
             let w: f32 = 16777217.0;
             print(w);
             return 0;
         }",
    );
    let mut expected = String::new();
    for line in [
        "0.1",
        "0.10000000149011612",
        "1000000000000000000000",
        "0.0000001",
        "-0",
        "NaN",
        "inf",
        "-inf",
        "123456789.125",
        "2.5",
        "1",
        "0.3333333333333333",
    ] {
        expected.push_str(line);
        expected.push('\n');
    }
    // 2^-1074: 323 zeros after the point, then the single shortest digit 5.
    expected.push_str(&format!("0.{}5\n", "0".repeat(323)));
    // 2^60 = 1152921504606846976; its shortest round-trip digits are 1152921504606847.
    expected.push_str("1152921504606847000\n");
    // 16777217 is not an f32; the literal rounds to 16777216.
    expected.push_str("16777216\n");

    let main = interner.intern("main");
    let oracle = wukong_interp::run_with_output(&program, main, &interner).unwrap();
    assert_eq!(
        String::from_utf8(oracle.1.clone()).unwrap(),
        expected,
        "the hand-derived lines must be the oracle's"
    );
    let (code, stdout) = link_with_cc_and_run(&program, &interner, "floats");
    assert_eq!(code, 0);
    assert_eq!(String::from_utf8(stdout).unwrap(), expected);
}

/// `print` of a string literal links and prints through the C runtime's `wukong_rt_print_str`. Not
/// on Windows, where the MinGW linker cannot consume the MSVC object's `.rodata` data relocation
/// that a string literal needs (a link failure there, never different output).
#[cfg(not(windows))]
#[test]
fn forced_c_link_prints_strings_like_the_oracle() {
    let (program, mut interner) =
        lower("fn main() -> i32 { print(\"hello, world\"); print(\"\"); print(3); return 0; }");
    let main = interner.intern("main");
    let oracle = wukong_interp::run_with_output(&program, main, &interner).unwrap();
    assert_eq!(oracle.1, b"hello, world\n\n3\n".to_vec());
    assert_eq!(
        link_with_cc_and_run(&program, &interner, "strings"),
        (0, oracle.1)
    );
}

#[test]
fn generated_c_runtime_defines_every_print_symbol() {
    let source = c_runtime_source();
    for symbol in [
        "void wukong_rt_print_i64(",
        "void wukong_rt_print_u64(",
        "void wukong_rt_print_f64(",
        "void wukong_rt_print_str(",
        "void wukong_rt_assert(",
    ] {
        assert!(source.contains(symbol), "{symbol}");
    }
    // `%g` (six significant digits) is exactly the approximation this runtime must never print.
    assert!(!source.contains("%g"), "{source}");
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
    assert_eq!(emit_native(&program, &interner, &opts), exit::OK);
    assert!(std::fs::metadata(object).unwrap().len() > 0);
}
