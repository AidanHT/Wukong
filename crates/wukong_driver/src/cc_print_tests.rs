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
        cc_link(
            &aot_cache::RuntimeCache::from_env(),
            &opts,
            stem,
            &object,
            &out,
            &mut report::PassTimer::new(&opts),
        ),
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
    // A `bool` prints `true`/`false` (`wukong_rt_print_bool`), on every path.
    let expected = b"-9223372036854775808\n18446744073709551615\ntrue\nfalse\n1\n";
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
/// the value at its own precision, closest when several do), then the linked executable is also
/// compared with the interpreter byte for byte. The values cover each layout branch (integral,
/// fractional, leading zeros, trailing zeros past the digits), an f32 0.1 (which prints `0.1` at
/// f32 precision, not the `0.10000000149011612` of the f64 it widens to), signed zero, NaN, both
/// infinities, the smallest subnormal and a power of two whose shortest form needs zero padding.
/// The f64 cases carry an explicit `f64` suffix: an unsuffixed float literal defaults to `f32`
/// (language guide, Literals).
#[test]
fn forced_c_link_prints_floats_exactly_like_the_oracle() {
    let (program, mut interner) = lower(
        "fn main() -> i32 {
             let z: f64 = 0.0;
             print(0.1f64);
             let a: f32 = 0.1;
             print(a);
             let mut e21: f64 = 1.0;
             let mut k: i32 = 0;
             while k < 21 { e21 = e21 * 10.0; k = k + 1; }
             print(e21);
             print(0.0000001f64);
             print(-z);
             print(z / z);
             print(1.0 / z);
             print(-1.0 / z);
             print(123456789.125f64);
             print(2.5f64);
             print(1.0f64);
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
        // The f32 0.1 (13421773 * 2^-27): "0.1" is the shortest decimal that rounds back to it.
        "0.1",
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

/// The per-type scalar printers through the C fallback: an `f32` prints its shortest round-trip
/// digits at f32 precision (the C runtime derives them by round-tripping `%e` candidates through
/// `strtof`), a `bool` prints `true`/`false`, a `char` its UTF-8 glyph. Every expected line is
/// hand-derived from those rules: f32 1/3 is 11184811 * 2^-25, shortest "0.33333334"; f32::MAX's
/// shortest digits are 34028235 then 31 zeros; 1e-7 and 2^-149 (the smallest subnormal, shortest
/// digit "1" at 1e-45) need leading zeros; f32 0.1 + 0.2 is the f32 0.3; `0xD800 as char` is a
/// surrogate, which prints U+FFFD.
#[test]
fn forced_c_link_prints_f32_bool_and_char_like_the_oracle() {
    let (program, mut interner) = lower(
        "fn main() -> i32 {
             let one: f32 = 1.0;
             let three: f32 = 3.0;
             print(one / three);
             let big: f32 = 3.4028235e38;
             print(big);
             let small: f32 = 1e-7;
             print(small);
             let tiny: f32 = 1e-45;
             println(tiny);
             let p: f32 = 0.1;
             let q: f32 = 0.2;
             print(p + q);
             let nz: f32 = -0.0;
             print(nz);
             print(1 < 2);
             println(2 < 1);
             let t: bool = true;
             print(t);
             print('A');
             print('\\u{E9}');
             println('\\u{20AC}');
             print('\\u{1F600}');
             let s: u32 = 55296;
             print(s as char);
             return 0;
         }",
    );
    let mut expected = String::new();
    for line in [
        "0.33333334",
        "340282350000000000000000000000000000000",
        "0.0000001",
        "0.000000000000000000000000000000000000000000001",
        "0.3",
        "-0",
        "true",
        "false",
        "true",
        "A",
        "\u{E9}",
        "\u{20AC}",
        "\u{1F600}",
        "\u{FFFD}",
    ] {
        expected.push_str(line);
        expected.push('\n');
    }
    let main = interner.intern("main");
    let oracle = wukong_interp::run_with_output(&program, main, &interner).unwrap();
    assert_eq!(
        String::from_utf8(oracle.1.clone()).unwrap(),
        expected,
        "the hand-derived lines must be the oracle's"
    );
    let (code, stdout) = link_with_cc_and_run(&program, &interner, "scalars");
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
        "void wukong_rt_print_f32(",
        "void wukong_rt_print_bool(",
        "void wukong_rt_print_char(",
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
    let mut timer = report::PassTimer::new(&opts);
    assert_eq!(
        emit_native(&program, &interner, &opts, &mut timer, None),
        exit::OK
    );
    assert!(std::fs::metadata(object).unwrap().len() > 0);
}
