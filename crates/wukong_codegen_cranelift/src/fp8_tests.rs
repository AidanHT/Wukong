//! The first-class 8-bit floats (`f8e4m3`, `f8e5m2`) on the native backend and the interpreter,
//! against `wukong_runtime::fp8` — the reference codecs and the bit-level codec they pin.
//!
//! Each program prints bit patterns, never a value a printer could round, and the expectation is
//! computed here from the runtime's functions, not recorded from a run. Every program runs on the
//! interpreter and natively at `-O0` and `-O2`.

use wukong_runtime::fp8::{self, decode_f32, encode_f32, encode_f64, Fp8Format};
use wukong_span::{Interner, SourceId};

fn lower(src: &str, opt: u8) -> (wukong_mir::Program, Interner) {
    let mut interner = Interner::new();
    let (module, pd) = wukong_parser::parse_module(src, SourceId(0), &mut interner);
    assert!(pd.iter().all(|d| !d.is_error()), "parse: {pd:?}");
    let (sema, sd) = wukong_sema::check(&module, &interner);
    assert!(sd.iter().all(|d| !d.is_error()), "sema: {sd:?}");
    let (mut program, ld) = wukong_mir_build::lower_program(&module, &sema, &mut interner);
    assert!(ld.iter().all(|d| !d.is_error()), "lower: {ld:?}");
    wukong_opt::optimize(&mut program, opt);
    (program, interner)
}

/// Run `src` on both backends at `-O0` and `-O2`; each must print exactly `want` and exit 0.
fn check_everywhere(src: &str, want: &str) {
    for opt in [0u8, 2] {
        let (program, mut interner) = lower(src, opt);
        let main = interner.intern("main");
        for (backend, run) in [
            (
                "interp",
                wukong_interp::run_with_output(&program, main, &interner),
            ),
            ("native", crate::jit_run(&program, main, &interner)),
        ] {
            let (code, out) = run.unwrap_or_else(|e| panic!("{backend} -O{opt}: {e}"));
            assert_eq!(code, 0, "{backend} exit at -O{opt}");
            let got = String::from_utf8(out).unwrap();
            if got != want {
                let (g, w): (Vec<&str>, Vec<&str>) =
                    (got.lines().collect(), want.lines().collect());
                let at = g.iter().zip(&w).position(|(a, b)| a != b);
                panic!(
                    "{backend} -O{opt} disagrees with the fp8 reference: {}",
                    match at {
                        Some(i) => format!("line {i}: got {}, want {}", g[i], w[i]),
                        None => format!("{} lines vs {} expected", g.len(), w.len()),
                    }
                );
            }
        }
    }
}

/// The code a value of the format is held as after any rounding: a NaN code becomes the format's
/// canonical NaN (with its sign), every other code is itself.
fn canonical(c: u8, fmt: Fp8Format) -> u8 {
    encode_f32(decode_f32(c, fmt), fmt)
}

/// Every code of both formats through the type: `*_from_bits` then `bits` (the code, NaNs
/// canonical), the `f32` it widens to, and a store and reload through a `[f8; 256]` array (the
/// 1-byte storage both ways). The decoded `f32` is checked against the REFERENCE decoder
/// (`f8e4m3_to_f32` / `f8e5m2_to_f32`), so the bit-level codec is pinned too.
#[test]
fn every_code_round_trips_through_the_type() {
    let src = "fn main() -> i32 { \
        let mut a4: [f8e4m3; 256] = [0.0; 256]; \
        let mut a5: [f8e5m2; 256] = [0.0; 256]; \
        let mut i: i32 = 0; \
        while i < 256 { \
            let c: u8 = i as u8; \
            let a = f8e4m3_from_bits(c); \
            let b = f8e5m2_from_bits(c); \
            print(bits(a)); \
            print(bits(a as f32)); \
            print(bits(b)); \
            print(bits(b as f32)); \
            a4[i] = a; \
            a5[i] = b; \
            i = i + 1; \
        } \
        i = 0; \
        while i < 256 { \
            print(bits(a4[i])); \
            print(bits(a5[i] as f64)); \
            i = i + 1; \
        } \
        return 0; }";
    let mut want = String::new();
    for c in 0u8..=255 {
        let (k4, k5) = (canonical(c, Fp8Format::E4M3), canonical(c, Fp8Format::E5M2));
        want.push_str(&format!("{k4}\n{}\n", fp8::f8e4m3_to_f32(k4).to_bits()));
        want.push_str(&format!("{k5}\n{}\n", fp8::f8e5m2_to_f32(k5).to_bits()));
    }
    for c in 0u8..=255 {
        let k5 = canonical(c, Fp8Format::E5M2);
        want.push_str(&format!(
            "{}\n{}\n",
            canonical(c, Fp8Format::E4M3),
            f64::from(fp8::f8e5m2_to_f32(k5)).to_bits()
        ));
    }
    check_everywhere(src, &want);
}

/// `x as f8e4m3` / `x as f8e5m2` from an `f32`: every midpoint between neighbouring codes, one
/// `f32` ULP either side and its negation, and 4096 bit patterns over the whole `f32` range
/// (subnormals, infinities and NaNs included), against the reference encoders.
#[test]
fn f32_conversion_matches_the_reference_encoder() {
    let src = "fn main() -> i32 { \
        let mut c: i32 = 0; \
        while c < 126 { \
            let mid: f32 = (f8e4m3_to_f32(c as u8) + f8e4m3_to_f32((c + 1) as u8)) / 2.0; \
            print(bits(mid as f8e4m3)); \
            print(bits(f32_from_bits(bits(mid) - 1) as f8e4m3)); \
            print(bits(f32_from_bits(bits(mid) + 1) as f8e4m3)); \
            print(bits(-mid as f8e4m3)); \
            c = c + 1; \
        } \
        c = 0; \
        while c < 123 { \
            let mid: f32 = (f8e5m2_to_f32(c as u8) + f8e5m2_to_f32((c + 1) as u8)) / 2.0; \
            print(bits(mid as f8e5m2)); \
            print(bits(f32_from_bits(bits(mid) - 1) as f8e5m2)); \
            print(bits(f32_from_bits(bits(mid) + 1) as f8e5m2)); \
            print(bits(-mid as f8e5m2)); \
            c = c + 1; \
        } \
        let mut k: u32 = 0; \
        while k < 4096 { \
            let x: f32 = f32_from_bits(k * 1048573); \
            print(bits(x as f8e4m3)); \
            print(bits(x as f8e5m2)); \
            k = k + 1; \
        } \
        return 0; }";
    let mut want = String::new();
    type Codec = (fn(u8) -> f32, fn(f32) -> u8, u8);
    let codecs: [Codec; 2] = [
        (fp8::f8e4m3_to_f32, fp8::f32_to_f8e4m3, 126),
        (fp8::f8e5m2_to_f32, fp8::f32_to_f8e5m2, 123),
    ];
    for (dec, enc, top) in codecs {
        for c in 0..top {
            let mid = (dec(c) + dec(c + 1)) / 2.0;
            for x in [
                mid,
                f32::from_bits(mid.to_bits() - 1),
                f32::from_bits(mid.to_bits() + 1),
                -mid,
            ] {
                want.push_str(&format!("{}\n", enc(x)));
            }
        }
    }
    for k in 0u32..4096 {
        let b = k.wrapping_mul(1048573);
        let x = f32::from_bits(b);
        want.push_str(&format!(
            "{}\n{}\n",
            fp8::f32_to_f8e4m3(x),
            fp8::f32_to_f8e5m2(x)
        ));
    }
    check_everywhere(src, &want);
}

/// `x as f8e4m3` from an `f64` rounds ONCE: on each side of every E4M3FN midpoint by a relative
/// 1e-12 (which `f32` cannot see, so a conversion through `f32` would tie to the even code) and on
/// the midpoint itself. The expectation is the runtime's `encode_f64`, which its own tests pin to
/// the exact reference.
#[test]
fn f64_conversion_rounds_once() {
    let src = "fn main() -> i32 { \
        let mut c: i32 = 0; \
        while c < 126 { \
            let lo: f64 = f8e4m3_from_bits(c as u8) as f64; \
            let hi: f64 = f8e4m3_from_bits((c + 1) as u8) as f64; \
            let mid: f64 = (lo + hi) / 2.0; \
            let k: f64 = 0.000000000001; \
            let d: f64 = mid * k; \
            print(bits(mid as f8e4m3)); \
            print(bits((mid + d) as f8e4m3)); \
            print(bits((mid - d) as f8e4m3)); \
            print(bits((mid + d) as f8e5m2)); \
            c = c + 1; \
        } \
        return 0; }";
    let mut want = String::new();
    for c in 0u8..126 {
        let lo = f64::from(fp8::f8e4m3_to_f32(c));
        let hi = f64::from(fp8::f8e4m3_to_f32(c + 1));
        let mid = (lo + hi) / 2.0;
        let d = mid * 0.000000000001;
        for (x, fmt) in [
            (mid, Fp8Format::E4M3),
            (mid + d, Fp8Format::E4M3),
            (mid - d, Fp8Format::E4M3),
            (mid + d, Fp8Format::E5M2),
        ] {
            want.push_str(&format!("{}\n", encode_f64(x, fmt)));
        }
    }
    check_everywhere(src, &want);
}

/// Arithmetic computes in `f32` and rounds every result onto the operands' grid: `a op b` is
/// `encode(decode(a) op decode(b))` for `+ - * /`, over a strided grid of code pairs of each
/// format (both signs, subnormals, the saturating and overflowing top, NaNs).
#[test]
fn arithmetic_rounds_each_result() {
    let src = "fn main() -> i32 { \
        let mut i: i32 = 0; \
        while i < 256 { \
            let mut j: i32 = 0; \
            while j < 256 { \
                let a = f8e4m3_from_bits(i as u8); \
                let b = f8e4m3_from_bits(j as u8); \
                print(bits(a + b)); print(bits(a - b)); print(bits(a * b)); print(bits(a / b)); \
                let x = f8e5m2_from_bits(i as u8); \
                let y = f8e5m2_from_bits(j as u8); \
                print(bits(x + y)); print(bits(x - y)); print(bits(x * y)); print(bits(x / y)); \
                j = j + 7; \
            } \
            i = i + 5; \
        } \
        return 0; }";
    let mut want = String::new();
    for i in (0u32..256).step_by(5) {
        for j in (0u32..256).step_by(7) {
            for fmt in [Fp8Format::E4M3, Fp8Format::E5M2] {
                let a = decode_f32(i as u8, fmt);
                let b = decode_f32(j as u8, fmt);
                for r in [a + b, a - b, a * b, a / b] {
                    want.push_str(&format!("{}\n", encode_f32(r, fmt)));
                }
            }
        }
    }
    check_everywhere(src, &want);
}
