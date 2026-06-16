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

/// Tripwire documenting *why* the vectorizer caps at 128-bit (`VEC_REG_BYTES = 16`): Cranelift
/// 0.124 cannot legalize a 256-bit `f32x8` value and rejects it at `define_function`. We therefore
/// get true 256-bit AVX throughput on the width-sensitive kernels (GEMM, etc.) via runtime
/// microkernels, not via wider CLIF vectors. If a future Cranelift starts accepting `f32x8`, this
/// test flips to passing — a signal to widen `VEC_REG_BYTES` and revisit the dispatch story.
#[test]
fn cranelift_still_rejects_f32x8() {
    use cranelift_codegen::ir::{types, AbiParam, InstBuilder, MemFlags, Signature};
    use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};
    use cranelift_jit::{JITBuilder, JITModule};
    use cranelift_module::{Linkage, Module};

    let isa = crate::make_isa(false).expect("isa");
    let builder = JITBuilder::with_isa(isa, cranelift_module::default_libcall_names());
    let mut module = JITModule::new(builder);
    let ptr = module.target_config().pointer_type();
    let mut sig = Signature::new(module.target_config().default_call_conv);
    sig.params.push(AbiParam::new(ptr));
    sig.params.push(AbiParam::new(ptr));
    sig.params.push(AbiParam::new(ptr));
    let fid = module.declare_function("probe", Linkage::Export, &sig).unwrap();
    let mut ctx = module.make_context();
    ctx.func.signature = sig;
    let mut fbctx = FunctionBuilderContext::new();
    let f32x8 = types::F32.by(8).expect("f32x8 type exists");
    {
        let mut b = FunctionBuilder::new(&mut ctx.func, &mut fbctx);
        let blk = b.create_block();
        b.append_block_params_for_function_params(blk);
        b.switch_to_block(blk);
        let a = b.block_params(blk)[0];
        let bb = b.block_params(blk)[1];
        let out = b.block_params(blk)[2];
        let va = b.ins().load(f32x8, MemFlags::trusted(), a, 0);
        let vb = b.ins().load(f32x8, MemFlags::trusted(), bb, 0);
        let s = b.ins().fadd(va, vb);
        b.ins().store(MemFlags::trusted(), s, out, 0);
        b.ins().return_(&[]);
        b.seal_all_blocks();
        b.finalize();
    }
    let define = module.define_function(fid, &mut ctx);
    let finalize = define.and_then(|_| {
        module.finalize_definitions().map_err(|e| {
            cranelift_module::ModuleError::Compilation(
                cranelift_codegen::CodegenError::Unsupported(e.to_string()),
            )
        })
    });
    assert!(
        finalize.is_err(),
        "Cranelift now accepts f32x8 — widen VEC_REG_BYTES and revisit the AVX dispatch"
    );
    unsafe { module.free_memory() };
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

    // The vectorizer must have fired at least once on a representative size, and `a*x[k] + y[k]`
    // must have contracted to a lane-wise fused multiply-add.
    let (prog, interner) = lowered(&kernel(64), 2);
    let mir = mercury_mir::print::print_program(&prog, &interner);
    assert!(
        mir.contains("splat") && mir.contains("x f32>"),
        "saxpy loop should have vectorized to SIMD ops"
    );
    assert!(
        mir.contains("fma") && mir.contains("x f32>"),
        "saxpy `a*x + y` should contract to a vector fma:\n{mir}"
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

/// Reduction vectorization: `s = s + x[k]*y[k]` (a dot product) and `s += x[k]` (a sum) must
/// vectorize to lane accumulators + a horizontal reduce, agree between interpreter and native, and
/// compute the right value. Inputs are chosen so every partial sum is an exact f32 integer, so the
/// reassociated (lane-parallel) order gives the identical result as strict left-to-right.
#[test]
fn vectorized_reductions_are_correct() {
    // dot: x[k]=k+1, y[k]=2  =>  sum 2*(k+1) = n*(n+1).
    let dot = |n: usize| {
        format!(
            "fn main() -> i32 {{ let mut x: [f32; {n}] = [0.0; {n}]; let mut y: [f32; {n}] = [2.0; {n}]; \
             let mut i: i32 = 0; while i < {n} {{ x[i] = ((i + 1) as f32); i += 1; }} \
             let mut s: f32 = 0.0; for k in 0..{n} {{ s = s + x[k] * y[k]; }} \
             return s as i32; }}"
        )
    };
    // dot via `+=`, same result.
    let dot_pluseq = |n: usize| {
        format!(
            "fn main() -> i32 {{ let mut x: [f32; {n}] = [0.0; {n}]; let mut y: [f32; {n}] = [2.0; {n}]; \
             let mut i: i32 = 0; while i < {n} {{ x[i] = ((i + 1) as f32); i += 1; }} \
             let mut s: f32 = 0.0; for k in 0..{n} {{ s += x[k] * y[k]; }} \
             return s as i32; }}"
        )
    };
    // sum: x[k]=1  =>  sum = n (non-product addend, so the FAdd path, not the fma path).
    let sum = |n: usize| {
        format!(
            "fn main() -> i32 {{ let mut x: [f32; {n}] = [1.0; {n}]; \
             let mut s: f32 = 0.0; for k in 0..{n} {{ s = s + x[k]; }} \
             return s as i32; }}"
        )
    };

    // The dot reduction must lower to a vector fma accumulator; the sum to a vector add.
    let (prog, interner) = lowered(&dot(64), 2);
    let mir = mercury_mir::print::print_program(&prog, &interner);
    assert!(
        mir.contains("fma") && mir.contains("x f32>"),
        "dot reduction should vectorize to a vector fma:\n{mir}"
    );

    for n in [1usize, 2, 3, 4, 7, 8, 15, 16, 17, 31, 64, 100, 257, 1000] {
        for (src, expect) in [
            (dot(n), (n as i64) * (n as i64 + 1)),
            (dot_pluseq(n), (n as i64) * (n as i64 + 1)),
            (sum(n), n as i64),
        ] {
            for opt in [0u8, 2, 3] {
                let native = jit(&src, opt).expect("jit");
                let interpd = interp(&src, opt).expect("interp");
                assert_eq!(native, interpd, "reduction native vs interp at n={n} -O{opt}");
                assert_eq!(native.0, expect, "reduction wrong at n={n} -O{opt}");
            }
        }
    }
}

/// The hard case for reduction vectorization: *fractional* inputs, where the lane-parallel
/// (reassociated) sum genuinely differs in the low bits from a strict left-to-right sum. There is no
/// clean closed-form expected value — the contract is only that the interpreter and the native
/// backend, both executing the same reassociated MIR, agree *bit-for-bit*, at every opt level and
/// across sizes that hit the unrolled body, the single-vector cleanup, and the scalar remainder.
#[test]
fn reduction_reassociation_is_backend_consistent() {
    let dot = |n: usize| {
        format!(
            "fn main() -> i32 {{ let mut x: [f32; {n}] = [0.0; {n}]; let mut y: [f32; {n}] = [0.0; {n}]; \
             let mut i: i32 = 0; while i < {n} {{ x[i] = ((i % 17) as f32) * 0.5 + 1.0; \
             y[i] = ((i % 13) as f32) * 0.25 - 0.5; i += 1; }} \
             let mut s: f32 = 0.0; for k in 0..{n} {{ s = s + x[k] * y[k]; }} \
             print(s); return 0; }}"
        )
    };
    for n in [5usize, 16, 17, 33, 50, 257, 1000, 4096] {
        let src = dot(n);
        for opt in [0u8, 1, 2, 3] {
            assert_eq!(
                jit(&src, opt).unwrap(),
                interp(&src, opt).unwrap(),
                "fractional dot interp vs native mismatch at n={n} -O{opt}"
            );
        }
    }
}

/// Integer reductions vectorize too, and (unlike float) integer addition is associative, so the
/// lane-parallel form is bit-identical to the strict scalar one — no reassociation caveat at all.
#[test]
fn vectorized_int_reduction() {
    // sum of a[k]=k+1 over 0..n = n(n+1)/2; int dot of a[k]=k+1, b[k]=2 = 2*sum = n(n+1).
    let isum = |n: usize| {
        format!(
            "fn main() -> i32 {{ let mut a: [i32; {n}] = [0; {n}]; \
             let mut i: i32 = 0; while i < {n} {{ a[i] = i + 1; i += 1; }} \
             let mut s: i32 = 0; for k in 0..{n} {{ s += a[k]; }} \
             return s; }}"
        )
    };
    let idot = |n: usize| {
        format!(
            "fn main() -> i32 {{ let mut a: [i32; {n}] = [0; {n}]; let mut b: [i32; {n}] = [2; {n}]; \
             let mut i: i32 = 0; while i < {n} {{ a[i] = i + 1; i += 1; }} \
             let mut s: i32 = 0; for k in 0..{n} {{ s = s + a[k] * b[k]; }} \
             return s; }}"
        )
    };
    let (prog, interner) = lowered(&isum(64), 2);
    let mir = mercury_mir::print::print_program(&prog, &interner);
    assert!(
        mir.contains("x i32>") && mir.contains("add"),
        "int sum reduction should vectorize to a vector add:\n{mir}"
    );
    for n in [1usize, 4, 7, 8, 16, 31, 64, 200] {
        let nn = n as i64;
        for (src, expect) in [(isum(n), nn * (nn + 1) / 2), (idot(n), nn * (nn + 1))] {
            for opt in [0u8, 2, 3] {
                let native = jit(&src, opt).expect("jit");
                let interpd = interp(&src, opt).expect("interp");
                assert_eq!(native, interpd, "int reduction native vs interp n={n} -O{opt}");
                assert_eq!(native.0, expect, "int reduction wrong n={n} -O{opt}");
            }
        }
    }
}

/// The reduction vectorizer generalizes past dot: any vectorizable addend works. A sum of squared
/// differences `s += (x[k]-y[k])*(x[k]-y[k])` (the core of an L2 loss) must vectorize to a vector
/// fma over the squared term and stay correct + differentially equal.
#[test]
fn vectorized_ssd_reduction() {
    // x[k]=k+1, y[k]=1  =>  (x-y)^2 = k^2; sum_{k<n} k^2 = (n-1)n(2n-1)/6. Exact in f32 for our n.
    let kernel = |n: usize| {
        format!(
            "fn main() -> i32 {{ let mut x: [f32; {n}] = [0.0; {n}]; let mut y: [f32; {n}] = [1.0; {n}]; \
             let mut i: i32 = 0; while i < {n} {{ x[i] = ((i + 1) as f32); i += 1; }} \
             let mut s: f32 = 0.0; for k in 0..{n} {{ s += (x[k] - y[k]) * (x[k] - y[k]); }} \
             return s as i32; }}"
        )
    };
    let (prog, interner) = lowered(&kernel(64), 2);
    let mir = mercury_mir::print::print_program(&prog, &interner);
    assert!(
        mir.contains("fma") && mir.contains("x f32>"),
        "ssd reduction should vectorize to a vector fma:\n{mir}"
    );
    for n in [1usize, 4, 7, 8, 16, 31, 64, 128] {
        let src = kernel(n);
        let nn = n as i64;
        let expect = (nn - 1) * nn * (2 * nn - 1) / 6; // sum of squares 0..n-1
        for opt in [0u8, 2, 3] {
            let native = jit(&src, opt).expect("jit");
            let interpd = interp(&src, opt).expect("interp");
            assert_eq!(native, interpd, "ssd native vs interp at n={n} -O{opt}");
            assert_eq!(native.0, expect, "ssd wrong at n={n} -O{opt}");
        }
    }
}

/// ReLU via if-conversion: `out[i] = if x[i] > 0 { x[i] } else { 0 }` must vectorize to a vector
/// compare + blend, agree between interpreter and native, and match the scalar reference across
/// sizes that exercise the vector body and the remainder. x[i] = i - n/2 spans negatives/positives.
#[test]
fn vectorized_relu_is_correct() {
    let kernel = |n: usize| {
        format!(
            "fn main() -> i32 {{ let mut x: [f32; {n}] = [0.0; {n}]; let mut o: [f32; {n}] = [0.0; {n}]; \
             let mut i: i32 = 0; while i < {n} {{ x[i] = ((i - {n}/2) as f32); i += 1; }} \
             for k in 0..{n} {{ o[k] = if x[k] > 0.0 {{ x[k] }} else {{ 0.0 }}; }} \
             let mut s: f32 = 0.0; let mut j: i32 = 0; while j < {n} {{ s = s + o[j]; j += 1; }} \
             return s as i32; }}"
        )
    };
    // Reference: sum of max(i - n/2, 0) for i in 0..n.
    let reference = |n: i64| -> i64 { (0..n).map(|i| (i - n / 2).max(0)).sum() };

    let (prog, interner) = lowered(&kernel(64), 2);
    let mir = mercury_mir::print::print_program(&prog, &interner);
    assert!(mir.contains("select") && mir.contains("x f32>"), "relu should vectorize");

    for n in [3usize, 4, 8, 13, 64, 257] {
        let src = kernel(n);
        let native = jit(&src, 3).expect("jit");
        let interp = interp(&src, 3).expect("interp");
        assert_eq!(native, interp, "relu native vs interp at n={n}");
        assert_eq!(native.0, reference(n as i64), "relu wrong at n={n}");
    }
}

/// Float remainder (`%` on floats) used to lower to an integer `urem` and crash the MIR verifier.
/// It now lowers to `FRem` (`x - trunc(x/y)*y` natively) and agrees with the interpreter.
#[test]
fn float_modulo() {
    // 7.5 % 2.0 = 1.5, scaled to an int so the exit code carries it.
    assert_eq!(jit_ok("fn main() -> i32 { let a: f32 = 7.5; let b: f32 = 2.0; return (a % b) as i32 * 10 + (a % b > 1.0) as i32; }").0, 11);
    let prog = "fn main() -> i32 { let a: f64 = 10.0; let b: f64 = 3.0; return (a % b) as i32; }";
    assert_eq!(jit_ok(prog).0, 1);
    for src in [
        "fn main() -> i32 { let a: f32 = 7.5; let b: f32 = 2.0; print(a % b); return 0; }",
        "fn main() -> i32 { let a: f64 = -10.5; let b: f64 = 3.0; print(a % b); return 0; }",
    ] {
        for opt in [0u8, 2, 3] {
            assert_eq!(jit(src, opt).unwrap(), interp(src, opt).unwrap(), "frem mismatch at -O{opt}");
        }
    }
}

/// Float `x + y*z` contracts to a fused multiply-add. This is the go/no-go check that the native
/// `fma` and the interpreter's `mul_add` agree *bit for bit* on this host (true only with hardware
/// FMA3) — the whole contraction rests on it. Also confirms the contraction actually fired in MIR
/// and that a rounding-sensitive case (where one rounding ≠ two roundings) still matches.
#[test]
fn fma_contraction_is_bit_exact() {
    // (a) The front-end emitted an `fma`, not a separate `fmul`/`fadd`.
    let (prog, interner) = lowered(
        "fn main() -> i32 { let a: f32 = 2.0; let b: f32 = 3.0; let c: f32 = 4.0; \
         return (a + b * c) as i32; }",
        0,
    );
    let mir = mercury_mir::print::print_program(&prog, &interner);
    assert!(mir.contains("fma "), "x + y*z should contract to fma:\n{mir}");
    // 2 + 3*4 = 14.
    assert_eq!(
        jit_ok("fn main() -> i32 { let a: f32 = 2.0; let b: f32 = 3.0; let c: f32 = 4.0; return (a + b * c) as i32; }").0,
        14
    );

    // (b) Differential interp-vs-native across opt levels, including a rounding-sensitive f32 case:
    // 16777217 = 2^24 + 1 is not representable in f32, so `huge*huge` rounds; the fused form keeps
    // the residual that the unfused form drops. Both backends must still produce identical bytes.
    for src in [
        "fn main() -> i32 { let a: f32 = 1.0; let b: f32 = 0.1; let c: f32 = 0.2; print(a + b * c); return 0; }",
        "fn main() -> i32 { let h: f32 = 16777217.0; let n: f32 = -281474976710656.0; print(n + h * h); return 0; }",
        "fn main() -> i32 { let a: f64 = 1.0; let b: f64 = 0.1; let c: f64 = 0.2; print(b * c + a); return 0; }",
        "fn main() -> i32 { let a: f64 = 3.0; let b: f64 = 7.0; let c: f64 = 11.0; print(a * b + c); return 0; }",
    ] {
        for opt in [0u8, 1, 2, 3] {
            assert_eq!(
                jit(src, opt).unwrap(),
                interp(src, opt).unwrap(),
                "fma mismatch at -O{opt} for {src:?}"
            );
        }
    }
}

/// Operator fusion: two adjacent same-range elementwise loops (a linear map then ReLU) must fuse
/// into one loop. Since fusion concatenates the exact statements, the two-loop form lowers to the
/// *same* MIR as the hand-written single loop — an exact structural check — and stays correct.
#[test]
fn fusion_collapses_adjacent_loops() {
    let prelude = |n: usize| {
        format!(
            "let mut x: [f32; {n}] = [0.0; {n}]; let mut t: [f32; {n}] = [0.0; {n}]; \
             let mut o: [f32; {n}] = [0.0; {n}]; let mut i: i32 = 0; \
             while i < {n} {{ x[i] = ((i - {n}/2) as f32); i += 1; }} "
        )
    };
    let epilogue = |n: usize| {
        format!(
            " let mut s: f32 = 0.0; let mut j: i32 = 0; while j < {n} {{ s = s + o[j]; j += 1; }} return s as i32;"
        )
    };
    let n = 100usize;
    let two = format!(
        "fn main() -> i32 {{ {}for k in 0..{n} {{ t[k] = 2.0 * x[k] + 1.0; }} \
         for k in 0..{n} {{ o[k] = if t[k] > 0.0 {{ t[k] }} else {{ 0.0 }}; }}{} }}",
        prelude(n),
        epilogue(n)
    );
    let one = format!(
        "fn main() -> i32 {{ {}for k in 0..{n} {{ t[k] = 2.0 * x[k] + 1.0; \
         o[k] = if t[k] > 0.0 {{ t[k] }} else {{ 0.0 }}; }}{} }}",
        prelude(n),
        epilogue(n)
    );

    let count_ops = |src: &str| -> usize {
        let (p, _) = lowered(src, 3);
        p.funcs
            .iter()
            .flat_map(|f| &f.blocks)
            .map(|b| b.insts.len() + 1)
            .sum()
    };
    // Fusion rewrites the two-loop form into the one-loop form before lowering ⇒ identical MIR.
    assert_eq!(
        count_ops(&two),
        count_ops(&one),
        "two adjacent loops should fuse to the same MIR as one combined loop"
    );

    // sum of relu(2*(k-50)+1) over k in 0..100 = sum of the first 50 positive odd numbers = 2500.
    let native = jit(&two, 3).expect("jit");
    assert_eq!(native, interp(&two, 3).expect("interp"), "fused native vs interp");
    assert_eq!(native.0, 2500);
}

/// Nested if-conversion: relu6 `clamp(x, 0, 6)` written as nested value-ifs must vectorize (nested
/// vector blends) and stay correct. Exercises the recursive `else`/`then` handling in the vectorizer.
#[test]
fn vectorized_relu6_nested_if() {
    let kernel = |n: usize| {
        format!(
            "fn main() -> i32 {{ let mut x: [f32; {n}] = [0.0; {n}]; let mut o: [f32; {n}] = [0.0; {n}]; \
             let mut i: i32 = 0; while i < {n} {{ x[i] = ((i % 11) as f32 - 2.0); i += 1; }} \
             for k in 0..{n} {{ o[k] = if x[k] < 6.0 {{ if x[k] > 0.0 {{ x[k] }} else {{ 0.0 }} }} else {{ 6.0 }}; }} \
             let mut s: f32 = 0.0; let mut j: i32 = 0; while j < {n} {{ s = s + o[j]; j += 1; }} \
             return s as i32; }}"
        )
    };
    let reference = |n: i64| -> i64 {
        (0..n).map(|i| ((i % 11) - 2).clamp(0, 6)).sum::<i64>()
    };
    let (prog, interner) = lowered(&kernel(40), 2);
    let mir = mercury_mir::print::print_program(&prog, &interner);
    assert!(mir.contains("select"), "relu6 should vectorize via nested blends");
    for n in [5usize, 8, 13, 40] {
        let src = kernel(n);
        let native = jit(&src, 3).expect("jit");
        assert_eq!(native, interp(&src, 3).expect("interp"), "relu6 native vs interp n={n}");
        assert_eq!(native.0, reference(n as i64), "relu6 wrong n={n}");
    }
}

/// Matmul (the headline ML kernel): its vectorized inner loop must agree between interpreter and
/// native AND match an independent scalar reference, for both the serial and `@parallel` forms, at
/// a side length that is a multiple of the vector width (8) and one that is not (10, exercising the
/// remainder). A[i]=i%3, B[i]=i%2 keep every product exact in f32.
#[test]
fn matmul_is_correct() {
    fn reference(ns: usize) -> i64 {
        let a: Vec<f32> = (0..ns * ns).map(|i| (i % 3) as f32).collect();
        let b: Vec<f32> = (0..ns * ns).map(|i| (i % 2) as f32).collect();
        let mut sum = 0.0f32;
        for i in 0..ns {
            for j in 0..ns {
                let mut acc = 0.0f32;
                for k in 0..ns {
                    acc += a[i * ns + k] * b[k * ns + j];
                }
                sum += acc;
            }
        }
        sum as i64
    }

    let kernel = |ns: usize, parallel: bool| {
        let attr = if parallel { "@parallel\n" } else { "" };
        let n2 = ns * ns;
        format!(
            "module m\n{attr}fn mm(a: [f32; {n2}], b: [f32; {n2}], c: [f32; {n2}]) {{\n\
             for i in 0..{ns} {{ for k in 0..{ns} {{ let aik: f32 = a[i*{ns}+k]; \
             for j in 0..{ns} {{ c[i*{ns}+j] = c[i*{ns}+j] + aik * b[k*{ns}+j]; }} }} }} }}\n\
             fn main() -> i32 {{ let mut a: [f32; {n2}] = [0.0; {n2}]; let mut b: [f32; {n2}] = [0.0; {n2}]; \
             let mut c: [f32; {n2}] = [0.0; {n2}]; let mut i: i32 = 0; \
             while i < {n2} {{ a[i] = ((i % 3) as f32); b[i] = ((i % 2) as f32); i += 1; }} \
             mm(a, b, c); let mut s: f32 = 0.0; let mut j: i32 = 0; \
             while j < {n2} {{ s = s + c[j]; j += 1; }} return s as i32; }}"
        )
    };

    // Sizes that exercise the microkernel's MR=6 / NR=16 remainders and small macro-blocks.
    for ns in [6usize, 7, 16, 17, 32] {
        for parallel in [false, true] {
            let src = kernel(ns, parallel);
            let native = jit(&src, 3).expect("jit");
            let interp = interp(&src, 3).expect("interp");
            assert_eq!(native, interp, "matmul native vs interp (ns={ns}, par={parallel})");
            assert_eq!(
                native.0,
                reference(ns),
                "matmul wrong result (ns={ns}, par={parallel})"
            );
        }
    }
}

/// Lower `src` and report whether any function calls the named runtime symbol — used to prove the
/// matmul recognizer fired (and picked the serial vs parallel variant), not merely that a scalar
/// fallback happened to compute the right answer.
fn lowered_calls(src: &str, callee: &str) -> bool {
    let mut interner = Interner::new();
    let (module, _) = mercury_parser::parse_module(src, SourceId(0), &mut interner);
    let (sema, _) = mercury_sema::check(&module, &interner);
    let (program, _) = mercury_mir_build::lower_program(&module, &sema, &mut interner);
    let target = interner.intern(callee);
    program.funcs.iter().any(|f| {
        f.blocks.iter().any(|b| {
            b.insts.iter().any(|ins| {
                matches!(&ins.op, mercury_mir::Op::Call { func, .. } if *func == target)
            })
        })
    })
}

/// The canonical f32 matmul nest must lower to the tuned `mercury_sgemm` microkernel (and the
/// `@parallel` form to the parallel variant), in both the accumulate and zero-init shapes.
#[test]
fn matmul_nest_lowers_to_sgemm() {
    // Accumulate form (beta = 1): no per-row zero-init.
    let acc = |attr: &str| {
        format!(
            "module m\n{attr}fn mm(a:[f32;64],b:[f32;64],c:[f32;64]) {{ \
             for i in 0..8 {{ for k in 0..8 {{ let aik: f32 = a[i*8+k]; \
             for j in 0..8 {{ c[i*8+j] = c[i*8+j] + aik * b[k*8+j]; }} }} }} }}"
        )
    };
    // Overwrite form (beta = 0): a per-row zero-init loop precedes the K loop.
    let ovr = |attr: &str| {
        format!(
            "module m\n{attr}fn mm(a:[f32;64],b:[f32;64],c:[f32;64]) {{ \
             for i in 0..8 {{ for j0 in 0..8 {{ c[i*8+j0] = 0.0; }} \
             for k in 0..8 {{ let aik: f32 = a[i*8+k]; \
             for j in 0..8 {{ c[i*8+j] = c[i*8+j] + aik * b[k*8+j]; }} }} }} }}"
        )
    };
    assert!(lowered_calls(&acc(""), "mercury_sgemm"), "accumulate -> sgemm");
    assert!(lowered_calls(&ovr(""), "mercury_sgemm"), "overwrite -> sgemm");
    assert!(
        lowered_calls(&acc("@parallel\n"), "mercury_sgemm_parallel"),
        "@parallel -> sgemm_parallel"
    );
    // A non-matmul triple loop (wrong B stride) must NOT be misrecognized.
    let not_mm = "module m\nfn f(a:[f32;64],b:[f32;64],c:[f32;64]) {{ \
        for i in 0..8 { for k in 0..8 { let aik: f32 = a[i*8+k]; \
        for j in 0..8 { c[i*8+j] = c[i*8+j] + aik * b[j*8+k]; } } } }";
    assert!(!lowered_calls(not_mm, "mercury_sgemm"), "transposed-B is not a row-major matmul");
}

/// The nn.Linear form `C = A·Bᵀ` (B indexed `[j*K+k]`) must lower to `mercury_sgemm_nt`, run
/// correctly, and stay native==interp. A plain `C = A·B` must NOT pick the transposed kernel.
#[test]
fn linear_nt_matmul_lowers_and_runs() {
    let nt = |attr: &str| {
        format!(
            "module m\n{attr}fn lin(a:[f32;48],b:[f32;32],c:[f32;24]) {{ \
             for i in 0..6 {{ for j0 in 0..4 {{ c[i*4+j0] = 0.0; }} \
             for k in 0..8 {{ let aik: f32 = a[i*8+k]; \
             for j in 0..4 {{ c[i*4+j] = c[i*4+j] + aik * b[j*8+k]; }} }} }} }}"
        )
    };
    assert!(lowered_calls(&nt(""), "mercury_sgemm_nt"), "A·Bᵀ -> sgemm_nt");
    assert!(
        lowered_calls(&nt("@parallel\n"), "mercury_sgemm_nt_parallel"),
        "@parallel A·Bᵀ -> sgemm_nt_parallel"
    );
    // C=A·B (b indexed [k*4+j]) must use the non-transposed kernel, never the nt one.
    let normal = "module m\nfn mm(a:[f32;48],b:[f32;32],c:[f32;24]) { \
        for i in 0..6 { for k in 0..8 { let aik: f32 = a[i*8+k]; \
        for j in 0..4 { c[i*4+j] = c[i*4+j] + aik * b[k*4+j]; } } } }";
    assert!(lowered_calls(normal, "mercury_sgemm"), "C=A·B -> sgemm");
    assert!(!lowered_calls(normal, "mercury_sgemm_nt"), "C=A·B is not transposed");

    // End to end: A is 6x8, B is 4x8 (so Bᵀ is 8x4), C is 6x4. Native must equal interp.
    let src = "module m\nfn lin(a:[f32;48],b:[f32;32],c:[f32;24]) { \
        for i in 0..6 { for j0 in 0..4 { c[i*4+j0] = 0.0; } \
        for k in 0..8 { let aik: f32 = a[i*8+k]; \
        for j in 0..4 { c[i*4+j] = c[i*4+j] + aik * b[j*8+k]; } } } }\n\
        fn main() -> i32 { let mut a:[f32;48]=[0.0;48]; let mut b:[f32;32]=[0.0;32]; \
        let mut c:[f32;24]=[0.0;24]; let mut i: i32 = 0; \
        while i < 48 { a[i] = ((i % 5) as f32) * 0.25; i += 1; } \
        let mut j: i32 = 0; while j < 32 { b[j] = ((j % 7) as f32) - 2.0; j += 1; } \
        lin(a, b, c); let mut s: f32 = 0.0; let mut t: i32 = 0; \
        while t < 24 { s = s + c[t]; t += 1; } return (s * 1000.0) as i32; }";
    assert!(lowered_calls(src, "mercury_sgemm_nt"));
    assert_eq!(jit(src, 3).expect("jit"), interp(src, 3).expect("interp"));
}

/// The textbook `ijk` dot-product matmul (`let s=0; for k s+=a*b; c=s`) must be recognized too —
/// both `C=A·B` and the `C=A·Bᵀ` (nn.Linear) spelling — and stay native==interp.
#[test]
fn ijk_dot_product_matmul_recognized() {
    // C = A·Bᵀ (b[j*K+k]) — the natural nn.Linear spelling.
    let nt = "module m\nfn lin(a:[f32;48],b:[f32;32],c:[f32;24]) { \
        for i in 0..6 { for j in 0..4 { let mut s: f32 = 0.0; \
        for k in 0..8 { s = s + a[i*8+k] * b[j*8+k]; } c[i*4+j] = s; } } }";
    assert!(lowered_calls(nt, "mercury_sgemm_nt"), "ijk A·Bᵀ -> sgemm_nt");
    // C = A·B (b[k*N+j]).
    let normal = "module m\nfn mm(a:[f32;48],b:[f32;32],c:[f32;24]) { \
        for i in 0..6 { for j in 0..4 { let mut s: f32 = 0.0; \
        for k in 0..8 { s = s + a[i*8+k] * b[k*4+j]; } c[i*4+j] = s; } } }";
    assert!(lowered_calls(normal, "mercury_sgemm"), "ijk A·B -> sgemm");
    assert!(!lowered_calls(normal, "mercury_sgemm_nt"));

    // End to end (A·Bᵀ): native must equal interp.
    let src = "module m\nfn lin(a:[f32;48],b:[f32;32],c:[f32;24]) { \
        for i in 0..6 { for j in 0..4 { let mut s: f32 = 0.0; \
        for k in 0..8 { s = s + a[i*8+k] * b[j*8+k]; } c[i*4+j] = s; } } }\n\
        fn main() -> i32 { let mut a:[f32;48]=[0.0;48]; let mut b:[f32;32]=[0.0;32]; \
        let mut c:[f32;24]=[0.0;24]; let mut i: i32 = 0; \
        while i < 48 { a[i] = ((i % 5) as f32) * 0.25; i += 1; } \
        let mut j: i32 = 0; while j < 32 { b[j] = ((j % 7) as f32) - 2.0; j += 1; } \
        lin(a, b, c); let mut s: f32 = 0.0; let mut t: i32 = 0; \
        while t < 24 { s = s + c[t]; t += 1; } return (s * 1000.0) as i32; }";
    assert!(lowered_calls(src, "mercury_sgemm_nt"));
    assert_eq!(jit(src, 3).expect("jit"), interp(src, 3).expect("interp"));
}

/// The zero-init (beta = 0) matmul, end to end: native and interpreter agree bit-for-bit (both run
/// the same kernel) and match an independent reference.
#[test]
fn matmul_overwrite_differential() {
    let ns = 9usize; // not a multiple of MR/NR, exercises the microkernel remainders
    let n2 = ns * ns;
    let src = format!(
        "module m\nfn mm(a: [f32; {n2}], b: [f32; {n2}], c: [f32; {n2}]) {{ \
         for i in 0..{ns} {{ for j0 in 0..{ns} {{ c[i*{ns}+j0] = 0.0; }} \
         for k in 0..{ns} {{ let aik: f32 = a[i*{ns}+k]; \
         for j in 0..{ns} {{ c[i*{ns}+j] = c[i*{ns}+j] + aik * b[k*{ns}+j]; }} }} }} }}\n\
         fn main() -> i32 {{ let mut a: [f32; {n2}] = [0.0; {n2}]; let mut b: [f32; {n2}] = [0.0; {n2}]; \
         let mut c: [f32; {n2}] = [0.0; {n2}]; let mut i: i32 = 0; \
         while i < {n2} {{ a[i] = ((i % 5) as f32) * 0.5; b[i] = ((i % 3) as f32) - 1.0; i += 1; }} \
         mm(a, b, c); let mut s: f32 = 0.0; let mut j: i32 = 0; \
         while j < {n2} {{ s = s + c[j]; j += 1; }} return (s * 100.0) as i32; }}"
    );
    assert!(lowered_calls(&src, "mercury_sgemm"), "recognizer must fire");
    let native = jit(&src, 3).expect("jit");
    let interp = interp(&src, 3).expect("interp");
    assert_eq!(native, interp, "overwrite matmul native vs interp");
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
