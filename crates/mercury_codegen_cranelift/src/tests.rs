//! JIT correctness tests, and a differential check that the native backend agrees with the
//! tree-walking interpreter (the reference oracle) on stdout and exit code.

use mercury_span::{Interner, SourceId};

/// Compile `src` to optimized MIR and run it natively, returning (exit_code, stdout).
fn jit(src: &str, opt: u8) -> Result<(i64, Vec<u8>), String> {
    let mut interner = Interner::new();
    let (module, pd) = mercury_parser::parse_module(src, SourceId(0), &mut interner);
    assert!(pd.iter().all(|d| !d.is_error()), "parse: {pd:?}");
    let (sema, sd) = mercury_sema::check(&module, &interner);
    assert!(sd.iter().all(|d| !d.is_error()), "sema: {sd:?}");
    let (mut program, ld) = mercury_mir_build::lower_program(&module, &sema, &mut interner);
    assert!(ld.iter().all(|d| !d.is_error()), "lower: {ld:?}");
    mercury_opt::optimize(&mut program, opt);
    let main = interner.intern("main");
    crate::jit_run(&program, main, &interner)
}

/// Run `src` through the interpreter for differential comparison.
fn interp(src: &str, opt: u8) -> Result<(i64, Vec<u8>), String> {
    let mut interner = Interner::new();
    let (module, _) = mercury_parser::parse_module(src, SourceId(0), &mut interner);
    let (sema, _) = mercury_sema::check(&module, &interner);
    let (mut program, _) = mercury_mir_build::lower_program(&module, &sema, &mut interner);
    mercury_opt::optimize(&mut program, opt);
    let main = interner.intern("main");
    mercury_interp::run_with_output(&program, main, &interner)
}

fn jit_ok(src: &str) -> (i64, String) {
    let (code, out) = jit(src, 2).expect("jit run");
    (code, String::from_utf8(out).unwrap())
}

#[test]
fn loop_sum() {
    let src = "fn main() -> i32 { let mut s: i32 = 0; let mut i: i32 = 0; \
               while i < 10 { s += i; i += 1; } return s; }";
    assert_eq!(jit_ok(src).0, 45);
}

#[test]
fn recursive_fib() {
    let src = "fn fib(n: i32) -> i32 { if n < 2 { return n; } \
               return fib(n - 1) + fib(n - 2); } \
               fn main() -> i32 { return fib(10); }";
    assert_eq!(jit_ok(src).0, 55);
}

#[test]
fn print_captures_stdout() {
    let src = "fn main() -> i32 { print(42); print(7 * 6); return 0; }";
    let (code, out) = jit_ok(src);
    assert_eq!(code, 0);
    assert_eq!(out, "42\n42\n");
}

#[test]
fn array_index_and_reduction() {
    let src = "fn main() -> i32 { let mut xs: [i32; 4] = [0,0,0,0]; \
               let mut i: i32 = 0; while i < 4 { xs[i] = i * i; i = i + 1; } \
               let mut s: i32 = 0; let mut j: i32 = 0; \
               while j < 4 { s = s + xs[j]; j = j + 1; } return s; }";
    assert_eq!(jit_ok(src).0, 14);
}

#[test]
fn for_step_loop() {
    let src = "fn main() -> i32 { let mut s: i32 = 0; \
               for i in 0..10 step 2 { s += i; } return s; }";
    assert_eq!(jit_ok(src).0, 20);
}

#[test]
fn casts_and_signed_modulo() {
    assert_eq!(jit_ok("fn main() -> i32 { return -7 % 3; }").0, -1);
    assert_eq!(
        jit_ok("fn main() -> i32 { let a: i64 = 300; return a as i32; }").0,
        300
    );
    assert_eq!(
        jit_ok("fn main() -> i32 { let f: f32 = 9.9; return f as i32; }").0,
        9
    );
}

#[test]
fn failed_assert_is_error() {
    let src = "fn main() -> i32 { assert(1 > 2); return 0; }";
    assert!(jit(src, 2).is_err());
}

/// The native backend must produce the same exit code and stdout as the interpreter across opt
/// levels — this is the soundness contract that lets either back the language.
#[test]
fn differential_against_interpreter() {
    let programs = [
        "fn main() -> i32 { let mut s: i32 = 0; for i in 0..100 { s += i * i; } return s % 1000; }",
        "fn fib(n: i32) -> i32 { if n < 2 { return n; } return fib(n-1) + fib(n-2); } \
         fn main() -> i32 { return fib(15); }",
        "fn main() -> i32 { let mut a: [i32; 8] = [0;8]; let mut i: i32 = 0; \
         while i < 8 { a[i] = i; i += 1; } let mut s: i32 = 0; let mut j: i32 = 0; \
         while j < 8 { s += a[j] * a[j]; j += 1; } return s; }",
        "fn main() -> i32 { let mut g: i32 = 48; let mut b: i32 = 18; \
         while b != 0 { let t: i32 = b; b = g % b; g = t; } return g; }",
        "fn main() -> i32 { print(123); print(0 - 5); let mut x: i64 = 7; \
         x = x * x * x; print(x as i32); return 0; }",
    ];
    for src in programs {
        for opt in [0u8, 1, 2, 3] {
            let n = jit(src, opt).expect("jit");
            let i = interp(src, opt).expect("interp");
            assert_eq!(n, i, "native vs interp mismatch at -O{opt} for:\n{src}");
        }
    }
}

/// Float kernels must agree too: the interpreter computes `f32` ops in `f32`, so its printed
/// results are bit-identical to native (including division, which would otherwise double-round).
#[test]
fn differential_floats() {
    let programs = [
        // f32 dot-product-ish reduction with a final divide (a mean).
        "fn main() -> i32 { let mut xs: [f32; 16] = [0.0; 16]; let mut i: i32 = 0; \
         while i < 16 { xs[i] = (i as f32) * 0.5; i += 1; } \
         let mut s: f32 = 0.0; let mut j: i32 = 0; \
         while j < 16 { s = s + xs[j] * xs[j]; j += 1; } \
         print(s / 16.0); return 0; }",
        // f32 SAXPY then print a few elements.
        "fn main() -> i32 { let mut y: [f32; 8] = [1.0; 8]; let a: f32 = 2.5; \
         let mut i: i32 = 0; while i < 8 { y[i] = a * (i as f32) + y[i]; i += 1; } \
         print(y[0]); print(y[3]); print(y[7]); return 0; }",
        // f64 path stays full precision.
        "fn main() -> i32 { let mut s: f64 = 0.0; let mut i: i32 = 0; \
         while i < 100 { s = s + 1.0 / ((i as f64) + 1.0); i += 1; } print(s); return 0; }",
    ];
    for src in programs {
        for opt in [0u8, 2, 3] {
            let n = jit(src, opt).expect("jit");
            let i = interp(src, opt).expect("interp");
            assert_eq!(n, i, "float native vs interp mismatch at -O{opt} for:\n{src}");
        }
    }
}

/// `if`/`else` used as a value (block tail and `let`-bound), including a branchy ReLU loop.
#[test]
fn if_as_expression() {
    // `let m = if a > b { a } else { b }` — value-producing if.
    let max = "fn main() -> i32 { let a: i32 = 7; let b: i32 = 12; \
               let m: i32 = if a > b { a } else { b }; return m; }";
    assert_eq!(jit_ok(max).0, 12);

    // if as the tail of a loop body (the relu shape that previously failed to lower).
    let relu = "fn main() -> i32 { let mut xs: [i32; 6] = [-2, 5, -1, 3, 0, -4]; \
                let mut i: i32 = 0; \
                while i < 6 { let v: i32 = xs[i]; if v > 0 { xs[i] = v; } else { xs[i] = 0; } i += 1; } \
                let mut s: i32 = 0; let mut j: i32 = 0; \
                while j < 6 { s += xs[j]; j += 1; } return s; }";
    // max(-2,0)+max(5,0)+... = 0+5+0+3+0+0 = 8
    assert_eq!(jit_ok(relu).0, 8);

    // Differential vs interpreter.
    for src in [max, relu] {
        for opt in [0u8, 1, 2, 3] {
            assert_eq!(jit(src, opt).unwrap(), interp(src, opt).unwrap());
        }
    }
}

/// Compile `src` to MIR and return the optimized program plus interner (for inspecting whether the
/// vectorizer fired).
fn lowered(src: &str, opt: u8) -> (mercury_mir::Program, Interner) {
    let mut interner = Interner::new();
    let (module, _) = mercury_parser::parse_module(src, SourceId(0), &mut interner);
    let (sema, _) = mercury_sema::check(&module, &interner);
    let (mut program, _) = mercury_mir_build::lower_program(&module, &sema, &mut interner);
    mercury_opt::optimize(&mut program, opt);
    (program, interner)
}

/// The SIMD loop vectorizer: a saxpy `for` loop must (a) actually lower to vector ops, (b) agree
/// between interpreter and native, and (c) compute the same result the scalar loop would — checked
/// across sizes that hit the vector body only, the remainder only, and both. `sum(2*i+1) == N*N`.
#[test]
fn vectorized_saxpy_is_correct_across_sizes() {
    let kernel = |n: usize| {
        format!(
            "fn main() -> i32 {{ let mut x: [f32; {n}] = [0.0; {n}]; let mut y: [f32; {n}] = [1.0; {n}]; \
             let mut o: [f32; {n}] = [0.0; {n}]; let mut i: i32 = 0; \
             while i < {n} {{ x[i] = (i as f32); i += 1; }} \
             let a: f32 = 2.0; for k in 0..{n} {{ o[k] = a * x[k] + y[k]; }} \
             let mut s: f32 = 0.0; let mut j: i32 = 0; while j < {n} {{ s = s + o[j]; j += 1; }} \
             return s as i32; }}"
        )
    };

    // The vectorizer must have fired at least once on a representative size.
    let (prog, interner) = lowered(&kernel(64), 2);
    let mir = mercury_mir::print::print_program(&prog, &interner);
    assert!(
        mir.contains("splat") && mir.contains("x f32>"),
        "saxpy loop should have vectorized to SIMD ops"
    );

    // 2 (remainder only), 4 (one vector, no remainder), 7/13 (vector + remainder), 1024 (many).
    for n in [2usize, 4, 7, 8, 13, 64, 1024] {
        let src = kernel(n);
        let native = jit(&src, 3).expect("jit");
        let interp = interp(&src, 3).expect("interp");
        assert_eq!(native, interp, "vectorized native vs interp mismatch at n={n}");
        assert_eq!(native.0, (n * n) as i64, "wrong saxpy result at n={n}");
    }
}

/// Hand-built SIMD MIR (vector load + splat + vector `fadd` + vector store) must execute
/// identically on the interpreter (lane-wise over its side arena) and the native backend (real
/// SSE vectors). This is the contract the loop vectorizer relies on.
#[test]
fn vector_ops_interp_matches_native() {
    use mercury_mir::{BinOp, Builder, CastKind, MirLevel, Op, Program};

    let mut interner = Interner::new();
    let mut b = Builder::new(interner.intern("main"), mercury_mir::MirType::I32);
    use mercury_mir::MirType;
    let f32t = MirType::F32;
    let vty = MirType::Vec(Box::new(MirType::F32), 4);

    // arr: [f32; 4] = [1, 2, 3, 4]
    let arr = b.alloca(MirType::Array(Box::new(f32t.clone()), 4));
    for i in 0..4i128 {
        let idx = b.build(MirType::I64, Op::ConstInt(i, MirType::I64));
        let slot = b.build(
            MirType::Ptr,
            Op::Gep { ptr: arr, index: idx, elem: f32t.clone() },
        );
        let v = b.build(f32t.clone(), Op::ConstFloat((i + 1) as f64, f32t.clone()));
        b.build_void(Op::Store { ptr: slot, value: v });
    }

    // base = &arr[0]; v = load <4 x f32>; v += splat(10.0); store back
    let zero = b.build(MirType::I64, Op::ConstInt(0, MirType::I64));
    let base = b.build(
        MirType::Ptr,
        Op::Gep { ptr: arr, index: zero, elem: f32t.clone() },
    );
    let v = b.build(vty.clone(), Op::Load(base, vty.clone()));
    let ten = b.build(f32t.clone(), Op::ConstFloat(10.0, f32t.clone()));
    let sp = b.build(vty.clone(), Op::Splat(ten));
    let sum = b.build(vty.clone(), Op::Bin(BinOp::FAdd, v, sp));
    b.build_void(Op::Store { ptr: base, value: sum });

    // return (i32) arr[2]  ==  3 + 10  ==  13
    let two = b.build(MirType::I64, Op::ConstInt(2, MirType::I64));
    let slot2 = b.build(
        MirType::Ptr,
        Op::Gep { ptr: arr, index: two, elem: f32t.clone() },
    );
    let e2 = b.build(f32t.clone(), Op::Load(slot2, f32t.clone()));
    let r = b.build(MirType::I32, Op::Cast(CastKind::FpToSi, e2, MirType::I32));
    b.ret(Some(r));

    let prog = Program {
        funcs: vec![b.finish()],
        level: MirLevel::Low,
    };
    for f in &prog.funcs {
        assert!(
            mercury_mir::verify::verify_function(f).is_empty(),
            "vector MIR should verify: {:?}",
            mercury_mir::verify::verify_function(f)
        );
    }
    let main = interner.intern("main");
    let native = crate::jit_run(&prog, main, &interner).expect("jit");
    let interp = mercury_interp::run_with_output(&prog, main, &interner).expect("interp");
    assert_eq!(native, interp, "vector native vs interp mismatch");
    assert_eq!(native.0, 13);
}

/// A `@parallel for` kernel: the native backend runs it across CPU cores via the runtime, the
/// interpreter runs the whole range sequentially, and the observable result must be identical.
#[test]
fn parallel_for_matches_interpreter() {
    let src = "@parallel fn scale(x: [i32; 4096], out: [i32; 4096]) { \
               for i in 0..4096 { out[i] = x[i] * 3; } } \
               fn main() -> i32 { let x: [i32; 4096] = [2; 4096]; let out: [i32; 4096] = [0; 4096]; \
               scale(x, out); let mut s: i32 = 0; let mut i: i32 = 0; \
               while i < 4096 { s += out[i]; i += 1; } return s % 100000; }";
    // 4096 * (2*3) = 24576
    assert_eq!(jit_ok(src).0, 24576);
    for opt in [0u8, 1, 2, 3] {
        assert_eq!(
            jit(src, opt).unwrap(),
            interp(src, opt).unwrap(),
            "parallel native vs interp mismatch at -O{opt}"
        );
    }
}
