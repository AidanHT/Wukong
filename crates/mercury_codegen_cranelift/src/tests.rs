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

/// Compile `src` to a callable native handle (no run), for timing. Honors `MERCURY_P4_NO_256`.
fn compile_native(src: &str, opt: u8) -> crate::JitProgram {
    let mut interner = Interner::new();
    let (module, _) = mercury_parser::parse_module(src, SourceId(0), &mut interner);
    let (sema, _) = mercury_sema::check(&module, &interner);
    let (mut program, _) = mercury_mir_build::lower_program(&module, &sema, &mut interner);
    mercury_opt::optimize(&mut program, opt);
    let main = interner.intern("main");
    crate::jit_compile(&program, main, &interner).expect("jit compile")
}

/// A same-run A/B of the 256-bit AVX2 recipe vs the 128-bit CLIF vectorizer on one compute-heavy
/// elementwise kernel. Reports the best-of-N wall-clock ratio (best-of controls for this laptop's
/// clock/thermal drift; alternating A/B/A/B keeps the two measurements adjacent). Ignored by default
/// — timing is noisy in CI; run with `--ignored --nocapture`. Not a correctness gate (the differential
/// tests are); a sanity check that widening to 256 bits actually buys throughput.
#[test]
#[ignore]
fn p4_bench_256_vs_128() {
    // Compute-bound, L1-resident body: a high arithmetic-intensity FMA chain (no sqrt, no memory
    // spill) over a 512-element array (3×2KB ≪ L1), repeated so wall-clock dominates setup. This
    // isolates the SIMD-width win; memory-bound bodies (large arrays, few flops/elem) see less
    // because both widths saturate the same bandwidth.
    let src = "\
        fn main() -> i32 { \
          let mut a: [f32; 512] = [0.0; 512]; let mut b: [f32; 512] = [0.0; 512]; \
          let mut o: [f32; 512] = [0.0; 512]; \
          for i in 0..512 { a[i] = (i as f32) * 0.001; } \
          for i in 0..512 { b[i] = (i as f32) * 0.002 + 1.0; } \
          let mut r: i32 = 0; \
          while r < 200000 { \
            for i in 0..512 { o[i] = a[i]*b[i] + a[i]*a[i] - b[i]*b[i] + a[i]*b[i]*a[i] - b[i]; } r += 1; \
          } \
          print(o[100] as i32); return 0; }";

    let best = |p: &crate::JitProgram| -> std::time::Duration {
        (0..5)
            .map(|_| {
                let t = std::time::Instant::now();
                std::hint::black_box(p.call());
                t.elapsed()
            })
            .min()
            .unwrap()
    };

    std::env::set_var("MERCURY_P4_NO_256", "1");
    let p128 = compile_native(src, 3);
    std::env::remove_var("MERCURY_P4_NO_256");
    let p256 = compile_native(src, 3);

    // Warm, then alternate to keep the two adjacent under one clock state.
    best(&p128);
    best(&p256);
    let (t128, t256) = (best(&p128), best(&p256));
    eprintln!(
        "p4 256-vs-128: 128-bit {:?}, 256-bit {:?}  =>  {:.2}x",
        t128,
        t256,
        t128.as_secs_f64() / t256.as_secs_f64()
    );
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
#[allow(clippy::result_large_err)] // Cranelift's ModuleError is large; irrelevant in a test.
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
    let fid = module
        .declare_function("probe", Linkage::Export, &sig)
        .unwrap();
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

/// P4 exploratory probe: which 256-bit vector ops does Cranelift 0.124.3 legalize on THIS host
/// (AVX2/FMA on)? Each op is built in a fresh module inside `catch_unwind`, so a panic in one does
/// not stop the others. Prints OK / ERR(msg) / PANIC per op. Run with `--nocapture`. Not a gate —
/// pure fact-finding for the raw-AVX2-vs-CLIF-widen architecture decision. Deleted before commit.
#[test]
fn p4_probe_vec256_ops() {
    use cranelift_codegen::ir::{types, AbiParam, InstBuilder, MemFlags, Signature, Value};
    use cranelift_codegen::ir::condcodes::{FloatCC, IntCC};
    use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};
    use cranelift_jit::{JITBuilder, JITModule};
    use cranelift_module::{Linkage, Module};
    use std::panic::{catch_unwind, AssertUnwindSafe};

    // Build a function with 4 ptr params, run `body(&mut b, &params, vty)`, define+finalize.
    // Returns Ok(()) if it compiled to machine code, Err(msg) otherwise.
    fn probe<F>(vty: types::Type, body: F) -> Result<(), String>
    where
        F: Fn(&mut FunctionBuilder, &[Value], types::Type),
    {
        let isa = crate::make_isa(false).map_err(|e| format!("isa: {e}"))?;
        let builder = JITBuilder::with_isa(isa, cranelift_module::default_libcall_names());
        let mut module = JITModule::new(builder);
        let ptr = module.target_config().pointer_type();
        let mut sig = Signature::new(module.target_config().default_call_conv);
        for _ in 0..4 {
            sig.params.push(AbiParam::new(ptr));
        }
        let fid = module
            .declare_function("probe", Linkage::Export, &sig)
            .map_err(|e| format!("declare: {e}"))?;
        let mut ctx = module.make_context();
        ctx.func.signature = sig;
        let mut fbctx = FunctionBuilderContext::new();
        {
            let mut b = FunctionBuilder::new(&mut ctx.func, &mut fbctx);
            let blk = b.create_block();
            b.append_block_params_for_function_params(blk);
            b.switch_to_block(blk);
            let params: Vec<Value> = b.block_params(blk).to_vec();
            body(&mut b, &params, vty);
            b.ins().return_(&[]);
            b.seal_all_blocks();
            b.finalize();
        }
        module
            .define_function(fid, &mut ctx)
            .map_err(|e| format!("define: {e}"))?;
        module
            .finalize_definitions()
            .map_err(|e| format!("finalize: {e}"))?;
        unsafe { module.free_memory() };
        Ok(())
    }

    let f32x8 = types::F32.by(8).unwrap();
    let f32x4 = types::F32.by(4).unwrap();
    let i32x8 = types::I32.by(8).unwrap();

    // (name, vty, builder). Each loads from params, applies the op, stores to params[last].
    type B = Box<dyn Fn(&mut FunctionBuilder, &[Value], types::Type)>;
    let cases: Vec<(&str, types::Type, B)> = vec![
        ("f32x4 fadd (control)", f32x4, Box::new(|b: &mut FunctionBuilder, p: &[Value], vt| {
            let va = b.ins().load(vt, MemFlags::trusted(), p[0], 0);
            let vb = b.ins().load(vt, MemFlags::trusted(), p[1], 0);
            let r = b.ins().fadd(va, vb);
            b.ins().store(MemFlags::trusted(), r, p[3], 0);
        })),
        ("f32x8 load+store", f32x8, Box::new(|b: &mut FunctionBuilder, p: &[Value], vt| {
            let va = b.ins().load(vt, MemFlags::trusted(), p[0], 0);
            b.ins().store(MemFlags::trusted(), va, p[3], 0);
        })),
        ("f32x8 fadd", f32x8, Box::new(|b: &mut FunctionBuilder, p: &[Value], vt| {
            let va = b.ins().load(vt, MemFlags::trusted(), p[0], 0);
            let vb = b.ins().load(vt, MemFlags::trusted(), p[1], 0);
            let r = b.ins().fadd(va, vb);
            b.ins().store(MemFlags::trusted(), r, p[3], 0);
        })),
        ("f32x8 fsub", f32x8, Box::new(|b: &mut FunctionBuilder, p: &[Value], vt| {
            let va = b.ins().load(vt, MemFlags::trusted(), p[0], 0);
            let vb = b.ins().load(vt, MemFlags::trusted(), p[1], 0);
            let r = b.ins().fsub(va, vb);
            b.ins().store(MemFlags::trusted(), r, p[3], 0);
        })),
        ("f32x8 fmul", f32x8, Box::new(|b: &mut FunctionBuilder, p: &[Value], vt| {
            let va = b.ins().load(vt, MemFlags::trusted(), p[0], 0);
            let vb = b.ins().load(vt, MemFlags::trusted(), p[1], 0);
            let r = b.ins().fmul(va, vb);
            b.ins().store(MemFlags::trusted(), r, p[3], 0);
        })),
        ("f32x8 fdiv", f32x8, Box::new(|b: &mut FunctionBuilder, p: &[Value], vt| {
            let va = b.ins().load(vt, MemFlags::trusted(), p[0], 0);
            let vb = b.ins().load(vt, MemFlags::trusted(), p[1], 0);
            let r = b.ins().fdiv(va, vb);
            b.ins().store(MemFlags::trusted(), r, p[3], 0);
        })),
        ("f32x8 fma", f32x8, Box::new(|b: &mut FunctionBuilder, p: &[Value], vt| {
            let va = b.ins().load(vt, MemFlags::trusted(), p[0], 0);
            let vb = b.ins().load(vt, MemFlags::trusted(), p[1], 0);
            let vc = b.ins().load(vt, MemFlags::trusted(), p[2], 0);
            let r = b.ins().fma(va, vb, vc);
            b.ins().store(MemFlags::trusted(), r, p[3], 0);
        })),
        ("f32x8 fneg", f32x8, Box::new(|b: &mut FunctionBuilder, p: &[Value], vt| {
            let va = b.ins().load(vt, MemFlags::trusted(), p[0], 0);
            let r = b.ins().fneg(va);
            b.ins().store(MemFlags::trusted(), r, p[3], 0);
        })),
        ("f32x8 sqrt", f32x8, Box::new(|b: &mut FunctionBuilder, p: &[Value], vt| {
            let va = b.ins().load(vt, MemFlags::trusted(), p[0], 0);
            let r = b.ins().sqrt(va);
            b.ins().store(MemFlags::trusted(), r, p[3], 0);
        })),
        ("f32x8 fmin/fmax", f32x8, Box::new(|b: &mut FunctionBuilder, p: &[Value], vt| {
            let va = b.ins().load(vt, MemFlags::trusted(), p[0], 0);
            let vb = b.ins().load(vt, MemFlags::trusted(), p[1], 0);
            let r = b.ins().fmax(va, vb);
            b.ins().store(MemFlags::trusted(), r, p[3], 0);
        })),
        ("f32x8 fcmp->bitselect", f32x8, Box::new(|b: &mut FunctionBuilder, p: &[Value], vt| {
            let va = b.ins().load(vt, MemFlags::trusted(), p[0], 0);
            let vb = b.ins().load(vt, MemFlags::trusted(), p[1], 0);
            let mask = b.ins().fcmp(FloatCC::GreaterThan, va, vb);
            let r = b.ins().bitselect(mask, va, vb);
            b.ins().store(MemFlags::trusted(), r, p[3], 0);
        })),
        ("f32x8 splat(scalar)", f32x8, Box::new(|b: &mut FunctionBuilder, p: &[Value], vt| {
            let s = b.ins().load(types::F32, MemFlags::trusted(), p[0], 0);
            let r = b.ins().splat(vt, s);
            b.ins().store(MemFlags::trusted(), r, p[3], 0);
        })),
        ("i32x8 iadd", i32x8, Box::new(|b: &mut FunctionBuilder, p: &[Value], vt| {
            let va = b.ins().load(vt, MemFlags::trusted(), p[0], 0);
            let vb = b.ins().load(vt, MemFlags::trusted(), p[1], 0);
            let r = b.ins().iadd(va, vb);
            b.ins().store(MemFlags::trusted(), r, p[3], 0);
        })),
        ("i32x8 icmp->bitselect", i32x8, Box::new(|b: &mut FunctionBuilder, p: &[Value], vt| {
            let va = b.ins().load(vt, MemFlags::trusted(), p[0], 0);
            let vb = b.ins().load(vt, MemFlags::trusted(), p[1], 0);
            let mask = b.ins().icmp(IntCC::SignedGreaterThan, va, vb);
            let r = b.ins().bitselect(mask, va, vb);
            b.ins().store(MemFlags::trusted(), r, p[3], 0);
        })),
    ];

    println!("\n=== P4 vec256 legalization probe (Cranelift 0.124.3, AVX2/FMA host) ===");
    let mut any_f32x8_ok = false;
    for (name, vty, f) in &cases {
        let res = catch_unwind(AssertUnwindSafe(|| probe(*vty, f)));
        let verdict = match res {
            Ok(Ok(())) => {
                if name.starts_with("f32x8") {
                    any_f32x8_ok = true;
                }
                "OK".to_string()
            }
            Ok(Err(e)) => format!("ERR: {}", e.lines().next().unwrap_or("").trim()),
            Err(_) => "PANIC".to_string(),
        };
        println!("  {name:<28} -> {verdict}");
    }
    println!("=== any f32x8 op legalized: {any_f32x8_ok} ===\n");
    // The whole reason the P4 raw-AVX2 emitter (`avx2.rs`) exists. If a future Cranelift starts
    // legalizing *any* 256-bit op, revisit whether CLIF vectors can replace the raw emitter.
    assert!(
        !any_f32x8_ok,
        "Cranelift now legalizes an f32x8 op — revisit the raw-AVX2 vs CLIF-vector decision"
    );
}

/// A function with a stack frame larger than a page (here a ~200 KB local array — the kind an
/// im2col/conv scratch buffer needs) must touch each guard page in its prologue instead of jumping
/// past it. Without `enable_probestack` the JIT'd entry faults when called at a shallow stack depth
/// (e.g. directly from the bench harness). This asserts it runs and still matches the interpreter.
#[test]
fn large_local_array_does_not_smash_stack() {
    let src = "fn main() -> i32 { let mut buf: [f32; 50000] = [3.0; 50000]; \
               let mut s: f32 = 0.0; for i in 0..50000 { s = s + buf[i]; } \
               return (s as i32) / 50000; }";
    for opt in [0u8, 2, 3] {
        let n = jit(src, opt).expect("jit run (large frame must not fault)");
        let i = interp(src, opt).expect("interp");
        assert_eq!(
            n, i,
            "native vs interp mismatch at -O{opt} for large-frame fn"
        );
        assert_eq!(n.0, 3, "sum(50000 * 3.0)/50000 == 3 at -O{opt}");
    }
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
            assert_eq!(
                n, i,
                "float native vs interp mismatch at -O{opt} for:\n{src}"
            );
        }
    }
}

/// The transcendental suite (built from primitive ops) must agree between native and interpreter
/// across the full input range — including `exp`'s overflow/underflow clamp, `log` over many
/// magnitudes, and `pow` — on both the scalar and the vectorized (unit-stride loop) paths. Since
/// every step is a primitive both backends already match on, equality is by construction; this
/// pins it across opt levels and extreme inputs.
#[test]
fn differential_transcendentals() {
    let programs = [
        // vectorized exp over [-8, 7.5] (the loop vectorizes; mid-range, all finite).
        "fn main() -> i32 { let mut xs: [f32; 32] = [0.0; 32]; \
         for i in 0..32 { xs[i] = (i as f32) * 0.5 - 8.0; } \
         let mut ys: [f32; 32] = [0.0; 32]; for i in 0..32 { ys[i] = exp(xs[i]); } \
         print((ys[0] * 1000000.0) as i32); print((ys[16] * 1000.0) as i32); \
         print((ys[31] * 100.0) as i32); return 0; }",
        // exp clamp at the extremes (x>88 -> huge-finite, x<-88 -> ~0): native and interp must
        // take the same select branches.
        "fn main() -> i32 { print((exp(100.0) * 0.0 + 7.0) as i32); \
         print((exp(-100.0) * 1000000.0) as i32); return 0; }",
        // vectorized log over a wide magnitude range (10 .. 320).
        "fn main() -> i32 { let mut xs: [f32; 32] = [0.0; 32]; \
         for i in 0..32 { xs[i] = ((i as f32) + 1.0) * 10.0; } \
         let mut ys: [f32; 32] = [0.0; 32]; for i in 0..32 { ys[i] = log(xs[i]); } \
         print((ys[0] * 1000.0) as i32); print((ys[31] * 1000.0) as i32); return 0; }",
        // scalar log across magnitudes incl. sub-1 (negative result).
        "fn main() -> i32 { print((log(0.001) * 1000.0) as i32); \
         print((log(1000000.0) * 1000.0) as i32); return 0; }",
        // vectorized pow (square) and a scalar large exponent.
        "fn main() -> i32 { let mut xs: [f32; 16] = [0.0; 16]; \
         for i in 0..16 { xs[i] = (i as f32) + 1.0; } \
         let mut ys: [f32; 16] = [0.0; 16]; for i in 0..16 { ys[i] = pow(xs[i], 2.0); } \
         print((ys[3] * 100.0) as i32); print((ys[15] * 100.0) as i32); \
         print((pow(2.0, 20.0) + 0.5) as i32); return 0; }",
    ];
    for src in programs {
        for opt in [0u8, 2, 3] {
            let n = jit(src, opt).expect("jit");
            let i = interp(src, opt).expect("interp");
            assert_eq!(
                n, i,
                "transcendental native vs interp mismatch at -O{opt} for:\n{src}"
            );
        }
    }
}

/// A pure elementwise `out[i] = f(x[i])` loop for f(∈ exp/log/tanh/sigmoid) is dispatched to the
/// 256-bit AVX2 `mercury_vmath_f32` kernel (the width Cranelift can't emit). The interpreter marshals
/// through the identical kernel, so native and interp must agree across opt levels — and the result
/// must stay ≈1 ULP of the true function (golden checks below catch an accuracy regression that the
/// native-vs-interp comparison alone would not, since both call the same kernel).
#[test]
fn differential_vmath_dispatch() {
    let programs = [
        // tanh over [-4, 3.75] — saturates toward ±1 at the ends.
        "fn main() -> i32 { let mut xs: [f32; 32] = [0.0; 32]; \
         for i in 0..32 { xs[i] = (i as f32) * 0.25 - 4.0; } \
         let mut ys: [f32; 32] = [0.0; 32]; for i in 0..32 { ys[i] = tanh(xs[i]); } \
         print((ys[0] * 1000.0) as i32); print((ys[20] * 1000.0) as i32); return 0; }",
        // sigmoid over the same range.
        "fn main() -> i32 { let mut xs: [f32; 32] = [0.0; 32]; \
         for i in 0..32 { xs[i] = (i as f32) * 0.25 - 4.0; } \
         let mut ys: [f32; 32] = [0.0; 32]; for i in 0..32 { ys[i] = sigmoid(xs[i]); } \
         print((ys[0] * 1000.0) as i32); print((ys[31] * 1000.0) as i32); return 0; }",
        // a length that is not a multiple of 8 exercises the kernel's scalar tail.
        "fn main() -> i32 { let mut xs: [f32; 13] = [0.0; 13]; \
         for i in 0..13 { xs[i] = (i as f32) * 0.5; } \
         let mut ys: [f32; 13] = [0.0; 13]; for i in 0..13 { ys[i] = exp(xs[i]); } \
         print((ys[12] * 100.0) as i32); return 0; }",
        // silu and gelu (the Llama / BERT activations), dispatched to the fused AVX2 kernels.
        "fn main() -> i32 { let mut xs: [f32; 32] = [0.0; 32]; \
         for i in 0..32 { xs[i] = (i as f32) * 0.25 - 4.0; } \
         let mut ys: [f32; 32] = [0.0; 32]; for i in 0..32 { ys[i] = silu(xs[i]); } \
         print((ys[24] * 1000.0) as i32); return 0; }",
        "fn main() -> i32 { let mut xs: [f32; 32] = [0.0; 32]; \
         for i in 0..32 { xs[i] = (i as f32) * 0.25 - 4.0; } \
         let mut ys: [f32; 32] = [0.0; 32]; for i in 0..32 { ys[i] = gelu(xs[i]); } \
         print((ys[24] * 1000.0) as i32); return 0; }",
        // adjacent activation loops the fusion pass merges into one multi-statement body: each
        // statement must still dispatch to its own kernel call (a full-range pass, in source order).
        "fn main() -> i32 { let mut xs: [f32; 16] = [0.0; 16]; \
         for i in 0..16 { xs[i] = (i as f32) * 0.5 - 4.0; } \
         let mut a: [f32; 16] = [0.0; 16]; let mut b: [f32; 16] = [0.0; 16]; \
         for i in 0..16 { a[i] = silu(xs[i]); } for i in 0..16 { b[i] = gelu(xs[i]); } \
         print((a[10] * 1000.0) as i32); print((b[10] * 1000.0) as i32); return 0; }",
    ];
    for src in programs {
        for opt in [0u8, 2, 3] {
            let n = jit(src, opt).expect("jit");
            let i = interp(src, opt).expect("interp");
            assert_eq!(
                n, i,
                "vmath native vs interp mismatch at -O{opt} for:\n{src}"
            );
        }
    }
    // Golden accuracy (≈1 ULP of libm, truncating `as i32`): exp(3.5)=33.1154→33115,
    // tanh(1)=0.761594→76159, sigmoid(2)=0.880797→88079, silu(2)=1.76159→176159,
    // gelu(2)=1.95460→195459.
    let golden = "fn main() -> i32 { \
        let a: [f32; 4] = [3.5; 4]; let b: [f32; 4] = [1.0; 4]; let c: [f32; 4] = [2.0; 4]; \
        let oa: [f32; 4] = [0.0; 4]; let ob: [f32; 4] = [0.0; 4]; let oc: [f32; 4] = [0.0; 4]; \
        let od: [f32; 4] = [0.0; 4]; let oe: [f32; 4] = [0.0; 4]; \
        for i in 0..4 { oa[i] = exp(a[i]); } for i in 0..4 { ob[i] = tanh(b[i]); } \
        for i in 0..4 { oc[i] = sigmoid(c[i]); } for i in 0..4 { od[i] = silu(c[i]); } \
        for i in 0..4 { oe[i] = gelu(c[i]); } \
        print((oa[0] * 1000.0) as i32); print((ob[0] * 100000.0) as i32); \
        print((oc[0] * 100000.0) as i32); print((od[0] * 100000.0) as i32); \
        print((oe[0] * 100000.0) as i32); return 0; }";
    let (_, out) = jit(golden, 3).expect("jit golden");
    assert_eq!(
        String::from_utf8(out).unwrap(),
        "33115\n76159\n88079\n176159\n195459\n",
        "vmath kernel drifted from libm accuracy"
    );
}

/// An `@parallel` activation dispatches each per-thread chunk to the AVX2 kernel (multicore ×
/// 256-bit). The native run splits the range across cores; the interpreter runs the whole range in
/// one pass — both must agree, since the kernel is elementwise (chunk boundaries don't change any
/// per-element result). Exercised for gelu (a fused activation) at a size that spans many chunks.
#[test]
fn differential_parallel_vmath() {
    let src = "@parallel fn act(x: [f32; 4096], out: [f32; 4096]) { \
         for i in 0..4096 { out[i] = gelu(x[i]); } } \
         fn main() -> i32 { let mut x: [f32; 4096] = [0.0; 4096]; \
         let mut o: [f32; 4096] = [0.0; 4096]; \
         for i in 0..4096 { x[i] = (i as f32) * 0.001 - 2.0; } act(x, o); \
         print((o[3000] * 100000.0) as i32); print((o[0] * 100000.0) as i32); return 0; }";
    for opt in [0u8, 2, 3] {
        let n = jit(src, opt).expect("jit");
        let i = interp(src, opt).expect("interp");
        assert_eq!(n, i, "parallel vmath native vs interp mismatch at -O{opt}");
    }
}

/// An int8 quantized `nn.Linear` nest (`u8`×`i8`→`i32`, `C = A·Bᵀ`) dispatches to the int8 GEMM
/// kernel (`mercury_i8gemm_nt`; the `@parallel` form to `_parallel`). Integer arithmetic is exact and
/// order-independent (associative mod 2³²), so the fused kernel equals the scalar nest bit-for-bit —
/// native and interp must agree at every opt level. `K = 40` exercises the AVX2 32/16-wide chunks and
/// the scalar tail; inputs span the full `u8`/`i8` ranges.
#[test]
fn differential_i8gemm() {
    let body = |attr: &str| {
        format!(
            "{attr}fn lin(a: [u8; 160], b: [i8; 160], c: [i32; 16]) {{ \
             for i in 0..4 {{ for j in 0..4 {{ let mut s: i32 = 0; \
             for k in 0..40 {{ s = s + (a[i * 40 + k] as i32) * (b[j * 40 + k] as i32); }} \
             c[i * 4 + j] = s; }} }} }} \
             fn main() -> i32 {{ let mut a: [u8; 160] = [0 as u8; 160]; \
             let mut b: [i8; 160] = [0 as i8; 160]; let mut c: [i32; 16] = [0; 16]; \
             for i in 0..160 {{ a[i] = ((i * 7 + 3) % 256) as u8; \
             b[i] = (((i * 5 + 1) % 256) - 128) as i8; }} \
             lin(a, b, c); \
             let mut acc: i32 = 0; for i in 0..16 {{ acc = acc + c[i]; }} \
             print(acc); print(c[0]); print(c[15]); return 0; }}"
        )
    };
    for src in [body(""), body("@parallel ")] {
        for opt in [0u8, 2, 3] {
            let n = jit(&src, opt).expect("jit");
            let i = interp(&src, opt).expect("interp");
            assert_eq!(
                n, i,
                "i8gemm native vs interp mismatch at -O{opt} for:\n{src}"
            );
        }
    }
}

/// The int8 `nn.Linear` nest dispatches to `mercury_i8gemm_nt` (the `@parallel` whole-function form
/// to `_parallel`, checked before the @parallel outliner so it reaches the multicore kernel rather
/// than a scalar loop). The signedness guard is correctness-critical (the kernel zero-extends A,
/// sign-extends B): an `i8×i8` nest must NOT pick the `u8×i8` kernel.
#[test]
fn i8_linear_nest_lowers_to_i8gemm() {
    let lin = |attr: &str| {
        format!(
            "module m\n{attr}fn lin(a:[u8;160],b:[i8;160],c:[i32;16]) {{ \
             for i in 0..4 {{ for j in 0..4 {{ let mut s: i32 = 0; \
             for k in 0..40 {{ s = s + (a[i*40+k] as i32) * (b[j*40+k] as i32); }} \
             c[i*4+j] = s; }} }} }}"
        )
    };
    assert!(
        lowered_calls(&lin(""), "mercury_i8gemm_nt"),
        "u8×i8 nn.Linear -> i8gemm_nt"
    );
    assert!(
        lowered_calls(&lin("@parallel\n"), "mercury_i8gemm_nt_parallel"),
        "@parallel u8×i8 nn.Linear -> i8gemm_nt_parallel"
    );
    // Signedness mismatch (A is i8, not u8): the kernel's zero/sign-extend split would miscompile, so
    // the recognizer must bail and never emit either int8 kernel symbol.
    let signed = "module m\nfn lin(a:[i8;160],b:[i8;160],c:[i32;16]) { \
        for i in 0..4 { for j in 0..4 { let mut s: i32 = 0; \
        for k in 0..40 { s = s + (a[i*40+k] as i32) * (b[j*40+k] as i32); } \
        c[i*4+j] = s; } } }";
    assert!(
        !lowered_calls(signed, "mercury_i8gemm_nt"),
        "i8×i8 must not pick the u8×i8 kernel"
    );
    assert!(
        !lowered_calls(signed, "mercury_i8gemm_nt_parallel"),
        "i8×i8 must not pick the parallel u8×i8 kernel"
    );
}

/// The embedding lookup `out[t,:] = weight[ids[t],:]` (the first layer of every LLM) dispatches to
/// `mercury_embedding_f32`. The native run gathers via the AVX2 kernel; the interpreter marshals the
/// identical serial kernel — pure data movement (a row copy), so they are bit-identical by
/// construction. Covers the serial form and a large-T `@parallel` form (T past the kernel's multicore
/// threshold so the rayon row-split actually runs, still bit-equal to serial).
#[test]
fn differential_embedding() {
    // Small serial gather: T=4, V=4, H=3, ids hitting row 0, the last row, and a repeat.
    let serial = "fn embed(ids: [i32; 4], weight: [f32; 12], out: [f32; 12]) { \
         for t in 0..4 { for d in 0..3 { out[t * 3 + d] = weight[ids[t] * 3 + d]; } } } \
         fn main() -> i32 { let weight: [f32; 12] = [0.0,1.0,2.0,10.0,11.0,12.0,\
         20.0,21.0,22.0,30.0,31.0,32.0]; let ids: [i32; 4] = [2,0,3,0]; \
         let mut out: [f32; 12] = [0.0; 12]; embed(ids, weight, out); \
         let mut acc: f32 = 0.0; for i in 0..12 { acc = acc + out[i]; } \
         print(acc); print(out[0]); print(out[11]); return 0; }"
        .to_string();
    // Large @parallel gather: T=80 (> EMBEDDING_PAR_MIN=64, so the multicore split runs), V=8, H=4.
    let parallel = "@parallel\nfn embed(ids: [i32; 80], weight: [f32; 32], out: [f32; 320]) { \
         for t in 0..80 { for d in 0..4 { out[t * 4 + d] = weight[ids[t] * 4 + d]; } } } \
         fn main() -> i32 { let mut weight: [f32; 32] = [0.0; 32]; \
         for i in 0..32 { weight[i] = (i as f32) * 0.5 - 3.0; } \
         let mut ids: [i32; 80] = [0; 80]; for i in 0..80 { ids[i] = ((i * 3 + 1) % 8) as i32; } \
         let mut out: [f32; 320] = [0.0; 320]; embed(ids, weight, out); \
         let mut acc: f32 = 0.0; for i in 0..320 { acc = acc + out[i]; } \
         print(acc); print(out[0]); print(out[319]); return 0; }"
        .to_string();
    for src in [serial, parallel] {
        for opt in [0u8, 1, 2, 3] {
            let n = jit(&src, opt).expect("jit");
            let i = interp(&src, opt).expect("interp");
            assert_eq!(
                n, i,
                "embedding native vs interp mismatch at -O{opt} for:\n{src}"
            );
        }
    }
}

/// The embedding nest must lower to `mercury_embedding_f32` (the `@parallel` whole-function form to
/// `_parallel`, intercepted before the outliner so it reaches the multicore kernel). A nest whose
/// load is NOT a data-dependent gather — `weight[t * H + d]` with the *outer var* as the row, i.e. a
/// plain elementwise copy — must NOT pick the embedding kernel (no `ids[t]` indirection).
#[test]
fn embedding_nest_lowers_to_kernel() {
    let embed = |attr: &str| {
        format!(
            "module m\n{attr}fn embed(ids: [i32; 4], weight: [f32; 12], out: [f32; 12]) {{ \
             for t in 0..4 {{ for d in 0..3 {{ out[t * 3 + d] = weight[ids[t] * 3 + d]; }} }} }}"
        )
    };
    assert!(
        lowered_calls(&embed(""), "mercury_embedding_f32"),
        "embedding nest -> mercury_embedding_f32"
    );
    assert!(
        lowered_calls(&embed("@parallel\n"), "mercury_embedding_f32_parallel"),
        "@parallel embedding nest -> mercury_embedding_f32_parallel"
    );
    // A direct copy `out[t*3+d] = weight[t*3+d]` (the row index is the loop var, not a gathered id) is
    // not an embedding lookup — the recognizer must bail (no `ids[t]` indirection to fold).
    let copy = "module m\nfn cp(ids: [i32; 4], weight: [f32; 12], out: [f32; 12]) { \
        for t in 0..4 { for d in 0..3 { out[t * 3 + d] = weight[t * 3 + d]; } } }";
    assert!(
        !lowered_calls(copy, "mercury_embedding_f32"),
        "a plain elementwise copy must not pick the embedding kernel"
    );
}

/// A `@parallel` reduction (`s += f(x[k], y[k])`, `m = fmax(m, x[k])`) dispatches to the multicore
/// reduction kernel (`mercury_sreduce_f32_parallel`). The native run folds across cores; the
/// interpreter calls the *serial* kernel — both are bit-identical by construction (fixed chunking,
/// ascending combine), so native and interp must agree at every opt level. Covers dot, ssd, the
/// unary sum, and the running max/min (the per-tensor max/absmax for softmax / int8 quantization).
#[test]
fn differential_parallel_reduce() {
    let programs = [
        // dot product Σ x·y
        "@parallel fn dotp(x: [f32; 4096], y: [f32; 4096], o: [f32; 1]) { \
         let mut s: f32 = 0.0; for k in 0..4096 { s = s + x[k] * y[k]; } o[0] = s; } \
         fn main() -> i32 { let mut x: [f32; 4096] = [0.0; 4096]; \
         let mut y: [f32; 4096] = [0.0; 4096]; let mut o: [f32; 1] = [0.0; 1]; \
         for i in 0..4096 { x[i] = (i as f32) * 0.001; y[i] = 2.0; } dotp(x, y, o); \
         print((o[0] * 100.0) as i32); return 0; }",
        // sum of squared differences Σ (x−y)² (an L2 loss)
        "@parallel fn ssd(x: [f32; 4096], y: [f32; 4096], o: [f32; 1]) { \
         let mut s: f32 = 0.0; for k in 0..4096 { s += (x[k] - y[k]) * (x[k] - y[k]); } o[0] = s; } \
         fn main() -> i32 { let mut x: [f32; 4096] = [0.0; 4096]; \
         let mut y: [f32; 4096] = [0.0; 4096]; let mut o: [f32; 1] = [0.0; 1]; \
         for i in 0..4096 { x[i] = (i as f32) * 0.001; y[i] = 1.0; } ssd(x, y, o); \
         print((o[0] * 10.0) as i32); return 0; }",
        // unary sum Σ x (LayerNorm-style accumulation; the recognizer passes y == x)
        "@parallel fn sumv(x: [f32; 4096], o: [f32; 1]) { \
         let mut s: f32 = 0.0; for k in 0..4096 { s += x[k]; } o[0] = s; } \
         fn main() -> i32 { let mut x: [f32; 4096] = [0.0; 4096]; \
         let mut o: [f32; 1] = [0.0; 1]; \
         for i in 0..4096 { x[i] = (i as f32) * 0.01; } sumv(x, o); \
         print((o[0]) as i32); return 0; }",
        // running max (fold by fmax; mixed-sign fractional inputs)
        "@parallel fn maxv(x: [f32; 4096], o: [f32; 1]) { \
         let mut m: f32 = x[0]; for k in 0..4096 { m = fmax(m, x[k]); } o[0] = m; } \
         fn main() -> i32 { let mut x: [f32; 4096] = [0.0; 4096]; \
         let mut o: [f32; 1] = [0.0; 1]; \
         for i in 0..4096 { x[i] = (i as f32) * 0.001 - 2.0; } maxv(x, o); \
         print((o[0] * 1000.0) as i32); return 0; }",
        // running min (fold by fmin, operand order m second)
        "@parallel fn minv(x: [f32; 4096], o: [f32; 1]) { \
         let mut m: f32 = x[0]; for k in 0..4096 { m = fmin(x[k], m); } o[0] = m; } \
         fn main() -> i32 { let mut x: [f32; 4096] = [0.0; 4096]; \
         let mut o: [f32; 1] = [0.0; 1]; \
         for i in 0..4096 { x[i] = 5.0 - (i as f32) * 0.001; } minv(x, o); \
         print((o[0] * 1000.0) as i32); return 0; }",
        // running absmax (fmax fold over abs(x[k]) → RED_MAXABS; negative-dominant tail)
        "@parallel fn absmaxv(x: [f32; 4096], o: [f32; 1]) { \
         let mut m: f32 = 0.0; for k in 0..4096 { m = fmax(m, abs(x[k])); } o[0] = m; } \
         fn main() -> i32 { let mut x: [f32; 4096] = [0.0; 4096]; \
         let mut o: [f32; 1] = [0.0; 1]; \
         for i in 0..4096 { x[i] = 3.0 - (i as f32) * 0.002; } absmaxv(x, o); \
         print((o[0] * 1000.0) as i32); return 0; }",
    ];
    for src in programs {
        for opt in [0u8, 2, 3] {
            let n = jit(src, opt).expect("jit");
            let i = interp(src, opt).expect("interp");
            assert_eq!(
                n, i,
                "parallel reduce native vs interp mismatch at -O{opt} for:\n{src}"
            );
        }
    }
    // Golden, using f32-exact values so the reassociated sum is unambiguous: dot = 2·3·4096 = 24576,
    // sum = 2·4096 = 8192.
    let golden = "@parallel fn dotp(x: [f32; 4096], y: [f32; 4096], o: [f32; 1]) { \
         let mut s: f32 = 0.0; for k in 0..4096 { s = s + x[k] * y[k]; } o[0] = s; } \
         @parallel fn sumv(x: [f32; 4096], o: [f32; 1]) { \
         let mut s: f32 = 0.0; for k in 0..4096 { s += x[k]; } o[0] = s; } \
         fn main() -> i32 { let mut x: [f32; 4096] = [2.0; 4096]; \
         let mut y: [f32; 4096] = [3.0; 4096]; let mut o: [f32; 1] = [0.0; 1]; \
         dotp(x, y, o); print((o[0]) as i32); sumv(x, o); print((o[0]) as i32); return 0; }";
    let (_, out) = jit(golden, 3).expect("jit golden");
    assert_eq!(
        String::from_utf8(out).unwrap(),
        "24576\n8192\n",
        "parallel reduction produced the wrong value"
    );
    // Max/min golden over a ramp (exact integer elements): max(i−1000) = 3095, min = −1000.
    let golden_mm = "@parallel fn maxv(x: [f32; 4096], o: [f32; 1]) { \
         let mut m: f32 = x[0]; for k in 0..4096 { m = fmax(m, x[k]); } o[0] = m; } \
         @parallel fn minv(x: [f32; 4096], o: [f32; 1]) { \
         let mut m: f32 = x[0]; for k in 0..4096 { m = fmin(m, x[k]); } o[0] = m; } \
         fn main() -> i32 { let mut x: [f32; 4096] = [0.0; 4096]; let mut o: [f32; 1] = [0.0; 1]; \
         for i in 0..4096 { x[i] = (i as f32) - 1000.0; } \
         maxv(x, o); print((o[0]) as i32); minv(x, o); print((o[0]) as i32); return 0; }";
    let (_, out) = jit(golden_mm, 3).expect("jit golden_mm");
    assert_eq!(
        String::from_utf8(out).unwrap(),
        "3095\n-1000\n",
        "parallel max/min produced the wrong value"
    );
    // Absmax golden over a ramp straddling zero (exact integer elements): max(|i−3000|) = 3000
    // (the negative end |−3000| beats the positive end |1095|).
    let golden_abs = "@parallel fn absmaxv(x: [f32; 4096], o: [f32; 1]) { \
         let mut m: f32 = 0.0; for k in 0..4096 { m = fmax(m, abs(x[k])); } o[0] = m; } \
         fn main() -> i32 { let mut x: [f32; 4096] = [0.0; 4096]; let mut o: [f32; 1] = [0.0; 1]; \
         for i in 0..4096 { x[i] = (i as f32) - 3000.0; } absmaxv(x, o); print((o[0]) as i32); return 0; }";
    let (_, out) = jit(golden_abs, 3).expect("jit golden_abs");
    assert_eq!(
        String::from_utf8(out).unwrap(),
        "3000\n",
        "parallel absmax produced the wrong value"
    );
}

/// bf16 mixed-precision reductions (`mercury_dot_bf16` / `mercury_sum_bf16`): a `s += (x[k] as f32)
/// [* (y[k] as f32)]` loop over `[bf16; _]` arrays with an f32 accumulator. The native backend rounds
/// to bf16 on store (`round_to_bf16`, the identical integer arithmetic as the interpreter's
/// `round_bf16`) and both call the identical reassociated kernel, so they must agree bit-for-bit even
/// for *non*-bf16-exact, fractional inputs that genuinely exercise the rounding and the 8-lane sum.
#[test]
fn differential_bf16_reduce() {
    let programs = [
        // bf16 dot Σ (x·y) with fractional, non-bf16-exact elements (real rounding + reassociation).
        "fn dotbf(x: [bf16; 4096], y: [bf16; 4096], o: [f32; 1]) { \
         let mut s: f32 = 0.0; for k in 0..4096 { s = s + (x[k] as f32) * (y[k] as f32); } o[0] = s; } \
         fn main() -> i32 { let mut x: [bf16; 4096] = [0.0 as bf16; 4096]; \
         let mut y: [bf16; 4096] = [0.0 as bf16; 4096]; let mut o: [f32; 1] = [0.0; 1]; \
         for i in 0..4096 { x[i] = ((i as f32) * 0.001) as bf16; y[i] = 1.5 as bf16; } dotbf(x, y, o); \
         print((o[0] * 100.0) as i32); return 0; }",
        // bf16 unary sum Σ x with fractional elements.
        "fn sumbf(x: [bf16; 4096], o: [f32; 1]) { \
         let mut s: f32 = 0.0; for k in 0..4096 { s += (x[k] as f32); } o[0] = s; } \
         fn main() -> i32 { let mut x: [bf16; 4096] = [0.0 as bf16; 4096]; \
         let mut o: [f32; 1] = [0.0; 1]; \
         for i in 0..4096 { x[i] = ((i as f32) * 0.003 - 1.7) as bf16; } sumbf(x, o); \
         print((o[0] * 10.0) as i32); return 0; }",
        // the reduction directly in main (no helper fn), accumulator reused by `s = s + …` form.
        "fn main() -> i32 { let mut x: [bf16; 1000] = [0.0 as bf16; 1000]; \
         for i in 0..1000 { x[i] = ((i as f32) * 0.01) as bf16; } \
         let mut s: f32 = 0.0; for k in 0..1000 { s = s + (x[k] as f32); } \
         print((s * 10.0) as i32); return 0; }",
    ];
    for src in programs {
        for opt in [0u8, 2, 3] {
            let n = jit(src, opt).expect("jit");
            let i = interp(src, opt).expect("interp");
            assert_eq!(
                n, i,
                "bf16 reduce native vs interp mismatch at -O{opt} for:\n{src}"
            );
        }
    }
    // Golden, bf16-exact small integers so the reassociated sum is unambiguous: dot = 2·(1+…+8) =
    // 72, sum = 36. Proves the dispatch produces the right value, not just self-consistency.
    let golden = "fn dotbf(x: [bf16; 8], y: [bf16; 8], o: [f32; 1]) { \
         let mut s: f32 = 0.0; for k in 0..8 { s = s + (x[k] as f32) * (y[k] as f32); } o[0] = s; } \
         fn sumbf(x: [bf16; 8], o: [f32; 1]) { \
         let mut s: f32 = 0.0; for k in 0..8 { s += (x[k] as f32); } o[0] = s; } \
         fn main() -> i32 { let mut x: [bf16; 8] = [0.0 as bf16; 8]; \
         let mut y: [bf16; 8] = [0.0 as bf16; 8]; let mut o: [f32; 1] = [0.0; 1]; \
         for i in 0..8 { x[i] = ((i + 1) as f32) as bf16; y[i] = 2.0 as bf16; } \
         dotbf(x, y, o); print((o[0]) as i32); sumbf(x, o); print((o[0]) as i32); return 0; }";
    let (_, out) = jit(golden, 3).expect("jit golden");
    assert_eq!(
        String::from_utf8(out).unwrap(),
        "72\n36\n",
        "bf16 reduction produced the wrong value"
    );
}

/// The bf16→f32 streaming axpby `out[k] = a*(x[k] as f32) + b*(y[k] as f32)` must dispatch to
/// `mercury_axpby_bf16` (bf16 in, f32 out, f32 math) and stay native==interp across opt levels (both
/// marshal the identical kernel). A 1-term scale must NOT dispatch (it would force a `0*y` the source
/// lacks). Golden small-integer case pins the value.
#[test]
fn differential_bf16_axpby() {
    let axpby = "module m\nfn ax(x:[bf16;64], y:[bf16;64], o:[f32;64]) { \
        for k in 0..64 { o[k] = 1.5 * (x[k] as f32) + 2.0 * (y[k] as f32); } }";
    assert!(
        lowered_calls(axpby, "mercury_axpby_bf16"),
        "bf16 axpby -> mercury_axpby_bf16"
    );
    let saxpy = "module m\nfn ax(x:[bf16;64], y:[bf16;64], o:[f32;64]) { \
        for k in 0..64 { o[k] = 3.0 * (x[k] as f32) + (y[k] as f32); } }";
    assert!(
        lowered_calls(saxpy, "mercury_axpby_bf16"),
        "bf16 saxpy (implicit b=1) -> mercury_axpby_bf16"
    );
    let add = "module m\nfn ax(x:[bf16;64], y:[bf16;64], o:[f32;64]) { \
        for k in 0..64 { o[k] = (x[k] as f32) + (y[k] as f32); } }";
    assert!(
        lowered_calls(add, "mercury_axpby_bf16"),
        "bf16 add -> mercury_axpby_bf16"
    );
    // A 1-term scale has no second additive term, so it must decline (avoids a 0*inf the source lacks).
    let scale = "module m\nfn ax(x:[bf16;64], o:[f32;64]) { \
        for k in 0..64 { o[k] = 2.0 * (x[k] as f32); } }";
    assert!(
        !lowered_calls(scale, "mercury_axpby_bf16"),
        "1-term scale must not dispatch to the axpby kernel"
    );

    // native == interp across opt levels, fractional non-bf16-exact inputs.
    let prog = "fn ax(x:[bf16;4096], y:[bf16;4096], o:[f32;4096]) { \
         for k in 0..4096 { o[k] = 1.5 * (x[k] as f32) + 2.0 * (y[k] as f32); } } \
         fn main() -> i32 { let mut x:[bf16;4096]=[0.0 as bf16;4096]; \
         let mut y:[bf16;4096]=[0.0 as bf16;4096]; let mut o:[f32;4096]=[0.0;4096]; \
         for i in 0..4096 { x[i]=((i as f32)*0.001) as bf16; y[i]=((i as f32)*0.002-1.3) as bf16; } \
         ax(x,y,o); let mut s:f32=0.0; for t in 0..4096 { s = s + o[t]; } \
         print((s*10.0) as i32); return 0; }";
    for opt in [0u8, 2, 3] {
        assert_eq!(
            jit(prog, opt).expect("jit"),
            interp(prog, opt).expect("interp"),
            "bf16 axpby native vs interp mismatch at -O{opt}"
        );
    }
    // Golden bf16-exact: o[k] = 2·(k+1) + 3·2 = 2(k+1)+6; Σ_{k=0..7} = 2·36 + 48 = 120.
    let golden = "fn ax(x:[bf16;8], y:[bf16;8], o:[f32;8]) { \
        for k in 0..8 { o[k] = 2.0*(x[k] as f32) + 3.0*(y[k] as f32); } } \
        fn main() -> i32 { let mut x:[bf16;8]=[0.0 as bf16;8]; let mut y:[bf16;8]=[0.0 as bf16;8]; \
        let mut o:[f32;8]=[0.0;8]; for i in 0..8 { x[i]=((i+1) as f32) as bf16; y[i]=2.0 as bf16; } \
        ax(x,y,o); let mut s:f32=0.0; for t in 0..8 { s = s + o[t]; } print((s) as i32); return 0; }";
    let (_, out) = jit(golden, 3).expect("jit golden");
    assert_eq!(
        String::from_utf8(out).unwrap(),
        "120\n",
        "bf16 axpby produced the wrong value"
    );
}

/// `erf` (and thus exact GELU) is built from primitive ops + the exp polynomial, so the native
/// backend must match the interpreter bit-for-bit across opt levels — scalar and vectorized,
/// including the odd-function sign (`erf(-x) = -erf(x)`) and saturation toward ±1 for large |x|.
#[test]
fn differential_erf() {
    let programs = [
        // scalar erf across sign and magnitude (odd symmetry + saturation at |x|=4).
        "fn main() -> i32 { print((erf(0.25) * 100000.0) as i32); \
         print((erf(1.5) * 100000.0) as i32); print((erf(-2.0) * 100000.0) as i32); \
         print((erf(4.0) * 100000.0) as i32); return 0; }",
        // vectorized erf over [-4, 3.5] (the loop vectorizes; f32 lane).
        "fn main() -> i32 { let mut xs: [f32; 16] = [0.0; 16]; \
         for i in 0..16 { xs[i] = (i as f32) * 0.5 - 4.0; } \
         let mut ys: [f32; 16] = [0.0; 16]; for i in 0..16 { ys[i] = erf(xs[i]); } \
         print((ys[0] * 100000.0) as i32); print((ys[8] * 100000.0) as i32); \
         print((ys[15] * 100000.0) as i32); return 0; }",
        // exact GELU = 0.5·x·(1 + erf(x/√2)), vectorized.
        "fn main() -> i32 { let mut xs: [f32; 16] = [0.0; 16]; \
         for i in 0..16 { xs[i] = (i as f32) * 0.5 - 4.0; } \
         let mut ys: [f32; 16] = [0.0; 16]; \
         for i in 0..16 { ys[i] = 0.5 * xs[i] * (1.0 + erf(xs[i] * 0.70710678)); } \
         print((ys[10] * 10000.0) as i32); print((ys[2] * 10000.0) as i32); return 0; }",
    ];
    for src in programs {
        for opt in [0u8, 2, 3] {
            let n = jit(src, opt).expect("jit");
            let i = interp(src, opt).expect("interp");
            assert_eq!(n, i, "erf native vs interp mismatch at -O{opt} for:\n{src}");
        }
    }
}

/// `sin`/`cos` (which enable RoPE) reduce the argument by quadrant and select ±sin/±cos, all from
/// primitive ops — so the native backend must match the interpreter bit-for-bit across opt levels,
/// scalar and vectorized, and across quadrants (incl. negative and out-of-first-period angles).
#[test]
fn differential_trig() {
    let programs = [
        // scalar sin/cos across several quadrants and signs (0, π/3, π, -1, 5, 10).
        "fn main() -> i32 { print((sin(1.0471976) * 100000.0) as i32); \
         print((cos(3.1415927) * 100000.0) as i32); print((sin(0.0 - 1.0) * 100000.0) as i32); \
         print((cos(5.0) * 100000.0) as i32); print((sin(10.0) * 100000.0) as i32); return 0; }",
        // vectorized sin over [0, ~4.7] (crosses π/2, π, 3π/2 — exercises all quadrant branches).
        "fn main() -> i32 { let mut xs: [f32; 16] = [0.0; 16]; \
         for i in 0..16 { xs[i] = (i as f32) * 0.3; } \
         let mut ys: [f32; 16] = [0.0; 16]; for i in 0..16 { ys[i] = sin(xs[i]); } \
         print((ys[3] * 100000.0) as i32); print((ys[8] * 100000.0) as i32); \
         print((ys[15] * 100000.0) as i32); return 0; }",
        // a RoPE rotation of (q0,q1) by per-element angles, vectorized cos/sin.
        "fn main() -> i32 { let mut a: [f32; 16] = [0.0; 16]; \
         for i in 0..16 { a[i] = (i as f32) * 0.2; } \
         let mut r: [f32; 16] = [0.0; 16]; \
         for i in 0..16 { r[i] = 2.0 * cos(a[i]) - 3.0 * sin(a[i]); } \
         print((r[4] * 10000.0) as i32); print((r[11] * 10000.0) as i32); return 0; }",
    ];
    for src in programs {
        for opt in [0u8, 2, 3] {
            let n = jit(src, opt).expect("jit");
            let i = interp(src, opt).expect("interp");
            assert_eq!(
                n, i,
                "trig native vs interp mismatch at -O{opt} for:\n{src}"
            );
        }
    }
}

/// A *batched* matmul — H independent matmuls over one buffer, each index carrying a per-head base
/// offset `h*STRIDE`. This is the multi-head-attention shape (`scores[h] = Q[h]·K[h]ᵀ`). The
/// recognizer peels the offset and GEPs each base pointer per head before the shared GEMM kernel.
/// Asserts native == interp across opt levels *and* the hand-computed values — the value check is
/// essential because a recognizer offset bug would be shared by both backends (both call the same
/// dispatched kernel), so the differential gate alone could not catch it.
#[test]
fn differential_batched_matmul() {
    // 2 heads of a 2x2 matmul. head0: [[1,2],[3,4]], head1: I=[[1,0],[0,1]];
    // b head0: [[5,6],[7,8]], head1: [[9,10],[11,12]]. Prints c[0], c[3], c[4], c[7].
    let head = "let a: [f32; 8] = [1.0,2.0,3.0,4.0,1.0,0.0,0.0,1.0]; \
        let b: [f32; 8] = [5.0,6.0,7.0,8.0,9.0,10.0,11.0,12.0]; let mut c: [f32; 8] = [0.0; 8]; \
        bmm(a, b, c); print(c[0] as i32); print(c[3] as i32); print(c[4] as i32); print(c[7] as i32); \
        return 0; }";
    // C[h] = A[h]·B[h]:    head0 [[19,22],[43,50]], head1 = I·B1 = [[9,10],[11,12]].
    let normal = format!(
        "fn bmm(a: [f32; 8], b: [f32; 8], c: [f32; 8]) {{ \
        for h in 0..2 {{ for i in 0..2 {{ for j in 0..2 {{ let mut s: f32 = 0.0; \
        for k in 0..2 {{ s = s + a[h*4 + i*2 + k] * b[h*4 + k*2 + j]; }} \
        c[h*4 + i*2 + j] = s; }} }} }} }} fn main() -> i32 {{ {head}"
    );
    // C[h] = A[h]·B[h]ᵀ (attention Q·Kᵀ): head0 [[17,23],[39,53]], head1 = I·B1ᵀ = B1ᵀ [[9,11],[10,12]].
    let transposed = format!(
        "fn bmm(a: [f32; 8], b: [f32; 8], c: [f32; 8]) {{ \
        for h in 0..2 {{ for i in 0..2 {{ for j in 0..2 {{ let mut s: f32 = 0.0; \
        for k in 0..2 {{ s = s + a[h*4 + i*2 + k] * b[h*4 + j*2 + k]; }} \
        c[h*4 + i*2 + j] = s; }} }} }} }} fn main() -> i32 {{ {head}"
    );
    for (src, expect) in [
        (&normal, "19\n50\n9\n12\n"),
        (&transposed, "17\n53\n9\n12\n"),
    ] {
        for opt in [0u8, 2, 3] {
            let n = jit(src, opt).expect("jit");
            let i = interp(src, opt).expect("interp");
            assert_eq!(n, i, "batched-matmul native vs interp mismatch at -O{opt}");
            assert_eq!(
                String::from_utf8(n.1).unwrap(),
                expect,
                "batched-matmul value mismatch at -O{opt} for:\n{src}"
            );
        }
    }
}

/// Fused `nn.Linear` epilogue (`out = act(x·Wᵀ [+ bias])`): native ≡ interp, plus hand-computed
/// values. A matmul followed by a bias-add / activation loop folds to one `mercury_sgemm_nt_epi`; the
/// differential gate can't catch a recognizer misfire (both backends run the fused kernel), so the
/// values are computed independently. x=[[1,2],[3,4]], W=I, bias=[-10,-1] ⇒ pre-act [-9,1,-7,3]
/// (with bias) or [1,2,3,4] (bias-free). The GELU/SiLU cases cover the transformer FFN forms; the
/// bias-free `silu(x·Wᵀ)` is the LLaMA SwiGLU shape (null bias through the same kernel).
#[test]
fn differential_linear_epilogue() {
    let body = "fn lin(x: [f32; 4], w: [f32; 4], bias: [f32; 2], out: [f32; 4]) { \
        for i in 0..2 { for j in 0..2 { let mut s: f32 = 0.0; \
        for k in 0..2 { s = s + x[i*2+k] * w[j*2+k]; } out[i*2+j] = s; } } \
        for i in 0..2 { for j in 0..2 { out[i*2+j] = EPI; } } } \
        fn main() -> i32 { let x: [f32; 4] = [1.0,2.0,3.0,4.0]; \
        let w: [f32; 4] = [1.0,0.0,0.0,1.0]; let bias: [f32; 2] = [-10.0,-1.0]; \
        let mut out: [f32; 4] = [0.0,0.0,0.0,0.0]; lin(x, w, bias, out); \
        print(out[0] as i32); print(out[1] as i32); print(out[2] as i32); print(out[3] as i32); \
        return 0; }";
    // Bias-free variant (LLaMA SwiGLU `silu(x·Wᵀ)`): no bias param, pre-act = [1,2,3,4].
    let nobias = "fn lin(x: [f32; 4], w: [f32; 4], out: [f32; 4]) { \
        for i in 0..2 { for j in 0..2 { let mut s: f32 = 0.0; \
        for k in 0..2 { s = s + x[i*2+k] * w[j*2+k]; } out[i*2+j] = s; } } \
        for i in 0..2 { for j in 0..2 { out[i*2+j] = EPI; } } } \
        fn main() -> i32 { let x: [f32; 4] = [1.0,2.0,3.0,4.0]; \
        let w: [f32; 4] = [1.0,0.0,0.0,1.0]; \
        let mut out: [f32; 4] = [0.0,0.0,0.0,0.0]; lin(x, w, out); \
        print(out[0] as i32); print(out[1] as i32); print(out[2] as i32); print(out[3] as i32); \
        return 0; }";
    let relu = body.replace("EPI", "fmax(out[i*2+j] + bias[j], 0.0)");
    let ident = body.replace("EPI", "out[i*2+j] + bias[j]");
    let gelu = body.replace("EPI", "gelu(out[i*2+j] + bias[j])");
    let silu = body.replace("EPI", "silu(out[i*2+j] + bias[j])");
    let silu_nobias = nobias.replace("EPI", "silu(out[i*2+j])");
    for (src, expect) in [
        (&relu, "0\n1\n0\n3\n"),        // relu([-9,1,-7,3])
        (&ident, "-9\n1\n-7\n3\n"),     // x·Wᵀ + bias
        (&gelu, "0\n0\n0\n2\n"),        // gelu([-9,1,-7,3]) as i32: gelu(1)=0.84→0, gelu(3)=3.0→2
        (&silu, "0\n0\n0\n2\n"),        // silu([-9,1,-7,3]) as i32: silu(1)=0.73→0, silu(3)=2.86→2
        (&silu_nobias, "0\n1\n2\n3\n"), // silu([1,2,3,4]) as i32: 0.73,1.76,2.86,3.93
    ] {
        for opt in [0u8, 2, 3] {
            let n = jit(src, opt).expect("jit");
            let i = interp(src, opt).expect("interp");
            assert_eq!(n, i, "linear-epilogue native vs interp at -O{opt}");
            assert_eq!(
                String::from_utf8(n.1).unwrap(),
                expect,
                "linear-epilogue value at -O{opt} for:\n{src}"
            );
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

/// The streaming saxpy dispatch: a `for k { o[k] = a*x[k] + y[k] }` loop must (a) lower to one
/// `mercury_velem_f32` call (the 256-bit AVX2 + non-temporal-store kernel — wider than, and store-
/// cheaper than, the generic 128-bit vectorizer), (b) agree between interpreter and native, and (c)
/// compute the same result the scalar loop would — across sizes that hit the kernel's vector body
/// only, the scalar tail only, and both. `sum(2*i+1) == N*N`.
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

    // The streaming recognizer must have fired: `a*x[k] + y[k]` is one `mercury_velem_f32` call (the
    // fused multiply-add and the lane work now live inside that kernel, not as inline MIR vector ops).
    let (prog, interner) = lowered(&kernel(64), 2);
    let mir = mercury_mir::print::print_program(&prog, &interner);
    assert!(
        mir.contains("mercury_velem_f32"),
        "saxpy `a*x + y` should dispatch to the streaming velem kernel:\n{mir}"
    );

    // 2 (remainder only), 4 (one vector, no remainder), 7/13 (vector + remainder), 1024 (many).
    for n in [2usize, 4, 7, 8, 13, 64, 1024] {
        let src = kernel(n);
        let native = jit(&src, 3).expect("jit");
        let interp = interp(&src, 3).expect("interp");
        assert_eq!(
            native, interp,
            "vectorized native vs interp mismatch at n={n}"
        );
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
                assert_eq!(
                    native, interpd,
                    "reduction native vs interp at n={n} -O{opt}"
                );
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
                assert_eq!(
                    native, interpd,
                    "int reduction native vs interp n={n} -O{opt}"
                );
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

/// ReLU streaming dispatch: `out[i] = if x[i] > 0 { x[i] } else { 0 }` (= `max(x, 0)`) must lower to
/// one `mercury_velem_f32` call (the 256-bit AVX2 + non-temporal-store kernel, `VE_RELU`), agree
/// between interpreter and native, and match the scalar reference across sizes that exercise the
/// kernel's vector body and its tail. x[i] = i - n/2 spans negatives/positives.
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
    assert!(
        mir.contains("mercury_velem_f32"),
        "relu should dispatch to the streaming velem kernel:\n{mir}"
    );

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
            assert_eq!(
                jit(src, opt).unwrap(),
                interp(src, opt).unwrap(),
                "frem mismatch at -O{opt}"
            );
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
    assert!(
        mir.contains("fma "),
        "x + y*z should contract to fma:\n{mir}"
    );
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
    assert_eq!(
        native,
        interp(&two, 3).expect("interp"),
        "fused native vs interp"
    );
    assert_eq!(native.0, 2500);
}

/// The general 256-bit AVX2 recipe path (`Op::VecKernelCall`, raw-AVX2 `avx2.rs`) must agree with the
/// interpreter oracle bit-for-bit on the f32 elementwise bodies it claims — a stream×stream product,
/// an in-place ReLU (load-before-store), `sqrt` composed with arithmetic, and negation over a third
/// stream — across trip counts spanning the vector part and the non-multiple-of-8 scalar tail. Also
/// pins native -O0 == -O3 (the recipe is opaque to the optimizer, so both must match).
#[test]
fn p4_vec256_general_matches_interp() {
    let prog = |n: usize| {
        format!(
            "fn main() -> i32 {{ \
               let mut a: [f32; {n}] = [0.0; {n}]; let mut b: [f32; {n}] = [0.0; {n}]; \
               let mut c: [f32; {n}] = [0.0; {n}]; let mut d: [f32; {n}] = [0.0; {n}]; \
               let mut k: i32 = 0; \
               while k < {n} {{ a[k] = ((k - {n}/2) as f32) * 0.5; b[k] = (k as f32) + 1.0; k += 1; }} \
               for i in 0..{n} {{ c[i] = a[i] * b[i]; }} \
               for i in 0..{n} {{ c[i] = if c[i] > 0.0 {{ c[i] }} else {{ 0.0 }}; }} \
               for i in 0..{n} {{ d[i] = sqrt(a[i] * a[i]) + (-b[i]); }} \
               let mut s: f32 = 0.0; let mut j: i32 = 0; \
               while j < {n} {{ s = s + c[j] + d[j]; j += 1; }} \
               print(s as i32); return (s as i32) & 255; }}"
        )
    };
    for n in [1usize, 7, 8, 9, 15, 16, 17, 64, 100, 257] {
        let src = prog(n);
        // The product loop must actually reach the recipe (else the test is vacuous).
        let (p, _) = lowered(&src, 3);
        assert!(
            p.funcs.iter().any(|f| !f.vec_kernels.is_empty()),
            "n={n}: expected a synthesized vector kernel in the MIR"
        );
        let native = jit(&src, 3).expect("jit -O3");
        assert_eq!(native, interp(&src, 3).expect("interp"), "n={n}: native vs interp");
        assert_eq!(jit(&src, 0).expect("jit -O0"), native, "n={n}: native -O0 vs -O3");
    }
}

/// A counting `while i < N { …; i += 1 }` with a vectorizable body normalizes to the for-range
/// vectorizer (`try_normalize_counting_while`). The rewrite must (a) match the interpreter oracle
/// bit-for-bit, (b) preserve the while's post-loop counter — `N` if it ran, else the untouched
/// start (the `start >= N` empty-run case), and (c) leave `i` reachable for code after the loop.
#[test]
fn p4_counting_while_normalizes_and_matches_interp() {
    // `lo` lets us cover both the ran case (lo < N) and the empty case (lo == N ⇒ never runs). The
    // counting-while body is a stream×stream product plus a third stream (`a*b + e`) — velem can't
    // claim that shape, so it reaches the general recipe (a real `vec_kernels` entry).
    let prog = |n: usize, lo: usize| {
        format!(
            "fn main() -> i32 {{ \
               let mut a: [f32; {n}] = [0.0; {n}]; let mut b: [f32; {n}] = [0.0; {n}]; \
               let mut c: [f32; {n}] = [0.0; {n}]; let mut e: [f32; {n}] = [0.0; {n}]; \
               let mut p: i32 = 0; \
               while p < {n} {{ a[p] = (p as f32) - 3.0; b[p] = (p as f32) + 1.0; e[p] = (p as f32); p += 1; }} \
               let mut k: i32 = {lo}; \
               while k < {n} {{ c[k] = a[k] * b[k] + e[k]; k += 1; }} \
               let mut s: f32 = 0.0; let mut t: i32 = 0; while t < {n} {{ s = s + c[t]; t += 1; }} \
               print(k); print(s as i32); return ((k + (s as i32)) & 255); }}"
        )
    };
    for n in [8usize, 9, 16, 33, 100] {
        for lo in [0usize, n] {
            // lo < n runs (final k == n); lo == n never runs (final k == n == lo). Both must hold.
            let src = prog(n, lo);
            if lo < n {
                let (p, _) = lowered(&src, 3);
                assert!(
                    p.funcs.iter().any(|f| !f.vec_kernels.is_empty()),
                    "n={n} lo={lo}: counting while should normalize to a vector kernel"
                );
            }
            let native = jit(&src, 3).expect("jit -O3");
            assert_eq!(
                native,
                interp(&src, 3).expect("interp"),
                "n={n} lo={lo}: native vs interp"
            );
            assert_eq!(jit(&src, 0).expect("jit -O0"), native, "n={n} lo={lo}: -O0 vs -O3");
        }
    }
}

/// ReLU6 streaming dispatch: `clamp(x, 0, 6)` written as the nested value-ifs `if x < 6 { if x > 0 {
/// x } else { 0 } } else { 6 }` must lower to one `mercury_velem_f32` call (`VE_RELU6`) and stay
/// correct across sizes. Exercises the recursive ReLU6 peel (`peel_velem_act`) that matches the outer
/// `< 6` guard around an inner ReLU over the same value.
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
    let reference = |n: i64| -> i64 { (0..n).map(|i| ((i % 11) - 2).clamp(0, 6)).sum::<i64>() };
    let (prog, interner) = lowered(&kernel(40), 2);
    let mir = mercury_mir::print::print_program(&prog, &interner);
    assert!(
        mir.contains("mercury_velem_f32"),
        "relu6 should dispatch to the streaming velem kernel:\n{mir}"
    );
    for n in [5usize, 8, 13, 40] {
        let src = kernel(n);
        let native = jit(&src, 3).expect("jit");
        assert_eq!(
            native,
            interp(&src, 3).expect("interp"),
            "relu6 native vs interp n={n}"
        );
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
            assert_eq!(
                native, interp,
                "matmul native vs interp (ns={ns}, par={parallel})"
            );
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
            b.insts
                .iter()
                .any(|ins| matches!(&ins.op, mercury_mir::Op::Call { func, .. } if *func == target))
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
    assert!(
        lowered_calls(&acc(""), "mercury_sgemm"),
        "accumulate -> sgemm"
    );
    assert!(
        lowered_calls(&ovr(""), "mercury_sgemm"),
        "overwrite -> sgemm"
    );
    assert!(
        lowered_calls(&acc("@parallel\n"), "mercury_sgemm_parallel"),
        "@parallel -> sgemm_parallel"
    );
    // A non-matmul triple loop (wrong B stride) must NOT be misrecognized.
    let not_mm = "module m\nfn f(a:[f32;64],b:[f32;64],c:[f32;64]) {{ \
        for i in 0..8 { for k in 0..8 { let aik: f32 = a[i*8+k]; \
        for j in 0..8 { c[i*8+j] = c[i*8+j] + aik * b[j*8+k]; } } } }";
    assert!(
        !lowered_calls(not_mm, "mercury_sgemm"),
        "transposed-B is not a row-major matmul"
    );
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
    assert!(
        lowered_calls(&nt(""), "mercury_sgemm_nt"),
        "A·Bᵀ -> sgemm_nt"
    );
    assert!(
        lowered_calls(&nt("@parallel\n"), "mercury_sgemm_nt_parallel"),
        "@parallel A·Bᵀ -> sgemm_nt_parallel"
    );
    // C=A·B (b indexed [k*4+j]) must use the non-transposed kernel, never the nt one.
    let normal = "module m\nfn mm(a:[f32;48],b:[f32;32],c:[f32;24]) { \
        for i in 0..6 { for k in 0..8 { let aik: f32 = a[i*8+k]; \
        for j in 0..4 { c[i*4+j] = c[i*4+j] + aik * b[k*4+j]; } } } }";
    assert!(lowered_calls(normal, "mercury_sgemm"), "C=A·B -> sgemm");
    assert!(
        !lowered_calls(normal, "mercury_sgemm_nt"),
        "C=A·B is not transposed"
    );

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
    assert!(
        lowered_calls(nt, "mercury_sgemm_nt"),
        "ijk A·Bᵀ -> sgemm_nt"
    );
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

/// Affine LayerNorm/RMSNorm (a per-column scale `g[i]` and optional shift `b[i]`) must dispatch to
/// `mercury_norm_affine_f32`, while plain (gamma=1) norms keep using `mercury_norm_f32`, and a softmax
/// with a trailing per-column scale must decline both (it has no affine parameters — the soundness
/// guard makes it fall back to the generic vectorizer rather than silently drop the scale).
#[test]
fn affine_norm_dispatch() {
    // Affine LayerNorm: (x-mean)*inv*g[i] + b[i]
    let ln_affine = "module m\nfn f(x:[f32;8], g:[f32;8], b:[f32;8]) { \
        let mut s: f32 = 0.0; for i in 0..8 { s = s + x[i]; } let mean: f32 = s / 8.0; \
        let mut v: f32 = 0.0; for i in 0..8 { v = v + (x[i] - mean) * (x[i] - mean); } \
        let inv: f32 = rsqrt(v / 8.0 + 0.00001); \
        for i in 0..8 { x[i] = (x[i] - mean) * inv * g[i] + b[i]; } }";
    assert!(
        lowered_calls(ln_affine, "mercury_norm_affine_f32"),
        "affine LayerNorm -> mercury_norm_affine_f32"
    );
    assert!(
        !lowered_calls(ln_affine, "mercury_norm_f32"),
        "affine LayerNorm must NOT use the plain kernel"
    );

    // Affine RMSNorm: x[i]*inv*g[i] (scale only, no shift)
    let rn_affine = "module m\nfn f(x:[f32;8], g:[f32;8]) { \
        let mut s: f32 = 0.0; for i in 0..8 { s = s + x[i] * x[i]; } \
        let inv: f32 = rsqrt(s / 8.0 + 0.00001); \
        for i in 0..8 { x[i] = x[i] * inv * g[i]; } }";
    assert!(
        lowered_calls(rn_affine, "mercury_norm_affine_f32"),
        "affine RMSNorm -> mercury_norm_affine_f32"
    );

    // Plain LayerNorm (gamma=1, beta=0) still uses the plain kernel, not the affine one.
    let ln_plain = "module m\nfn f(x:[f32;8]) { \
        let mut s: f32 = 0.0; for i in 0..8 { s = s + x[i]; } let mean: f32 = s / 8.0; \
        let mut v: f32 = 0.0; for i in 0..8 { v = v + (x[i] - mean) * (x[i] - mean); } \
        let inv: f32 = rsqrt(v / 8.0 + 0.00001); \
        for i in 0..8 { x[i] = (x[i] - mean) * inv; } }";
    assert!(
        lowered_calls(ln_plain, "mercury_norm_f32"),
        "plain LayerNorm -> mercury_norm_f32"
    );
    assert!(
        !lowered_calls(ln_plain, "mercury_norm_affine_f32"),
        "plain LayerNorm must NOT use the affine kernel"
    );

    // softmax with a trailing scale has no affine semantics; the guard makes it decline BOTH norm
    // kernels (falls back to the vectorizer) rather than dropping the scale.
    let sm_scaled = "module m\nfn f(x:[f32;8], g:[f32;8]) { \
        let mut m: f32 = x[0]; for i in 0..8 { m = fmax(m, x[i]); } \
        for i in 0..8 { x[i] = exp(x[i] - m); } \
        let mut s: f32 = 0.0; for i in 0..8 { s = s + x[i]; } let inv: f32 = 1.0 / s; \
        for i in 0..8 { x[i] = x[i] * inv * g[i]; } }";
    assert!(
        !lowered_calls(sm_scaled, "mercury_norm_f32")
            && !lowered_calls(sm_scaled, "mercury_norm_affine_f32"),
        "softmax+scale must decline both norm kernels (no affine softmax)"
    );
}

/// A batched norm — `for r in 0..R { <RMSNorm over x[r*C + i]> }` — must dispatch to the fused
/// `mercury_norm_f32` kernel with `rows = R` (the real `[batch*seq, hidden]` transformer shape) rather
/// than fall back to the scalar vectorizer; and the refactor that threaded the batch offset through the
/// recognizer must not have broken the single-row (`rows = 1`) form. The differential gate
/// (native == interp over rows > 1, adversarial inputs) lives in the fuzzer's `rmsnorm_batched` kernel,
/// and the independent per-row reference in `tests/run/batched_rmsnorm.mer`; this just pins that the
/// recognizer keeps *firing* (a silent fallback to scalar would pass both of those yet regress speed).
#[test]
fn batched_norm_dispatch() {
    // R = 3 rows, C = 4 cols, normalized in place over the flat [12] buffer via the `r*4 + i` offset.
    let batched = "module m\nfn f(x:[f32;12]) { \
        for r in 0..3 { \
        let mut s: f32 = 0.0; for i in 0..4 { s = s + x[r*4+i] * x[r*4+i]; } \
        let inv: f32 = rsqrt(s / 4.0 + 0.00001); \
        for i in 0..4 { x[r*4+i] = x[r*4+i] * inv; } } }";
    assert!(
        lowered_calls(batched, "mercury_norm_f32"),
        "batched RMSNorm (for r {{ <row r> }}) must dispatch to mercury_norm_f32"
    );

    // Batched LayerNorm (two reductions: mean, then variance) over the `r*4 + i` offset must dispatch too.
    let batched_ln = "module m\nfn f(x:[f32;12]) { \
        for r in 0..3 { \
        let mut s: f32 = 0.0; for i in 0..4 { s = s + x[r*4+i]; } let mean: f32 = s / 4.0; \
        let mut v: f32 = 0.0; for i in 0..4 { v = v + (x[r*4+i] - mean) * (x[r*4+i] - mean); } \
        let inv: f32 = rsqrt(v / 4.0 + 0.00001); \
        for i in 0..4 { x[r*4+i] = (x[r*4+i] - mean) * inv; } } }";
    assert!(
        lowered_calls(batched_ln, "mercury_norm_f32"),
        "batched LayerNorm (for r {{ <row r> }}) must dispatch to mercury_norm_f32"
    );

    // Batched softmax (max/exp/sum/normalize, with the row-local `x[r*4]` max-seed) over the offset.
    let batched_sm = "module m\nfn f(x:[f32;12]) { \
        for r in 0..3 { \
        let mut m: f32 = x[r*4]; for i in 0..4 { m = fmax(m, x[r*4+i]); } \
        for i in 0..4 { x[r*4+i] = exp(x[r*4+i] - m); } \
        let mut s: f32 = 0.0; for i in 0..4 { s = s + x[r*4+i]; } let inv: f32 = 1.0 / s; \
        for i in 0..4 { x[r*4+i] = x[r*4+i] * inv; } } }";
    assert!(
        lowered_calls(batched_sm, "mercury_norm_f32"),
        "batched softmax (for r {{ <row r> }}) must dispatch to mercury_norm_f32"
    );

    // Batched AFFINE RMSNorm (per-column scale g[i]) must route to the affine kernel, not the plain one
    // — the data is offset-indexed x[r*4+i] while gamma stays column-indexed g[i].
    let batched_affine = "module m\nfn f(x:[f32;12], g:[f32;4]) { \
        for r in 0..3 { \
        let mut s: f32 = 0.0; for i in 0..4 { s = s + x[r*4+i] * x[r*4+i]; } \
        let inv: f32 = rsqrt(s / 4.0 + 0.00001); \
        for i in 0..4 { x[r*4+i] = x[r*4+i] * inv * g[i]; } } }";
    assert!(
        lowered_calls(batched_affine, "mercury_norm_affine_f32"),
        "batched affine RMSNorm -> mercury_norm_affine_f32"
    );
    assert!(
        !lowered_calls(batched_affine, "mercury_norm_f32"),
        "batched affine RMSNorm must NOT use the plain kernel"
    );

    // The single-row form (no outer loop) must still dispatch — rows = 1 is the `batch = None` path.
    let single = "module m\nfn f(x:[f32;4]) { \
        let mut s: f32 = 0.0; for i in 0..4 { s = s + x[i] * x[i]; } \
        let inv: f32 = rsqrt(s / 4.0 + 0.00001); \
        for i in 0..4 { x[i] = x[i] * inv; } }";
    assert!(
        lowered_calls(single, "mercury_norm_f32"),
        "single-row RMSNorm must still dispatch to mercury_norm_f32"
    );
}

/// A `@parallel` batched RMSNorm normalizes its rows across CPU cores via `mercury_norm_f32_parallel`
/// (intercepted before the generic `@parallel` outliner, mirroring the sgemm/int8 whole-function
/// interceptions). Rows are independent — no cross-row combine — so the multicore kernel is bit-
/// identical to the serial one the interpreter marshals, and native must equal interp at every opt
/// level. (The runtime's own `serial_matches_parallel_bit_for_bit` test pins the kernel equality; this
/// pins the end-to-end dispatch + the rows>1 marshalling across the rayon boundary.)
#[test]
fn differential_parallel_batched_norm() {
    // 64 rows × 64 cols = 4096; many rows so the kernel genuinely spreads across cores.
    let src = "@parallel fn rmsnorm_batch(x: [f32; 4096]) { \
         for r in 0..64 { \
         let mut s: f32 = 0.0; for i in 0..64 { s = s + x[r*64+i] * x[r*64+i]; } \
         let inv: f32 = rsqrt(s / 64.0 + 0.00001); \
         for i in 0..64 { x[r*64+i] = x[r*64+i] * inv; } } } \
         fn main() -> i32 { let mut x: [f32; 4096] = [0.0; 4096]; \
         for i in 0..4096 { x[i] = (i as f32) * 0.001 - 2.0; } rmsnorm_batch(x); \
         let mut s: f32 = 0.0; for i in 0..4096 { s = s + x[i]; } \
         print((s * 1000.0) as i32); return 0; }";
    assert!(
        lowered_calls(src, "mercury_norm_f32_parallel"),
        "@parallel batched RMSNorm must dispatch to the multicore norm kernel"
    );
    for opt in [0u8, 2, 3] {
        let n = jit(src, opt).expect("jit");
        let i = interp(src, opt).expect("interp");
        assert_eq!(n, i, "parallel batched norm native vs interp at -O{opt}");
    }

    // The affine form (a learned per-column gamma) maps rows across cores via the multicore *affine*
    // kernel `mercury_norm_affine_f32_parallel`, and must stay bit-exact vs the serial kernel the
    // interpreter marshals (rows independent, no cross-row combine).
    let src_affine = "@parallel fn rmsnorm_affine_batch(x: [f32; 4096], g: [f32; 64]) { \
         for r in 0..64 { \
         let mut s: f32 = 0.0; for i in 0..64 { s = s + x[r*64+i] * x[r*64+i]; } \
         let inv: f32 = rsqrt(s / 64.0 + 0.00001); \
         for i in 0..64 { x[r*64+i] = x[r*64+i] * inv * g[i]; } } } \
         fn main() -> i32 { let mut x: [f32; 4096] = [0.0; 4096]; let mut g: [f32; 64] = [0.0; 64]; \
         for i in 0..4096 { x[i] = (i as f32) * 0.001 - 2.0; } \
         for i in 0..64 { g[i] = (i as f32) * 0.01 + 0.5; } rmsnorm_affine_batch(x, g); \
         let mut s: f32 = 0.0; for i in 0..4096 { s = s + x[i]; } \
         print((s * 1000.0) as i32); return 0; }";
    assert!(
        lowered_calls(src_affine, "mercury_norm_affine_f32_parallel"),
        "@parallel batched affine RMSNorm must dispatch to the multicore affine kernel"
    );
    for opt in [0u8, 2, 3] {
        let n = jit(src_affine, opt).expect("jit");
        let i = interp(src_affine, opt).expect("interp");
        assert_eq!(
            n, i,
            "parallel batched affine norm native vs interp at -O{opt}"
        );
    }
}

/// The accumulate (beta = 1) matmul — `C += A·B` with `C` pre-initialized — end to end. Guards the
/// beta=1 dispatch (a real pattern: accumulating a matmul into a bias-initialized output). Unlike the
/// overwrite test, this also checks the value against an independent reference, so a *beta
/// misclassification* (silently overwriting and dropping `C`'s initial values) is caught — native ==
/// interp alone would not notice that. Inputs are exact multiples of 0.5 so the kernel's reassociated
/// sum equals the naive reference bit-for-bit.
#[test]
fn matmul_accumulate_differential() {
    let ns = 7usize; // not a multiple of MR/NR — exercises the microkernel remainders
    let n2 = ns * ns;
    // No per-row zero-init in `mm` => the recognizer reads it as the accumulate (beta = 1) form.
    let src = format!(
        "module m\nfn mm(a: [f32; {n2}], b: [f32; {n2}], c: [f32; {n2}]) {{ \
         for i in 0..{ns} {{ for k in 0..{ns} {{ let aik: f32 = a[i*{ns}+k]; \
         for j in 0..{ns} {{ c[i*{ns}+j] = c[i*{ns}+j] + aik * b[k*{ns}+j]; }} }} }} }}\n\
         fn main() -> i32 {{ let mut a: [f32; {n2}] = [0.0; {n2}]; let mut b: [f32; {n2}] = [0.0; {n2}]; \
         let mut c: [f32; {n2}] = [0.0; {n2}]; let mut i: i32 = 0; \
         while i < {n2} {{ a[i] = ((i % 5) as f32) * 0.5; b[i] = ((i % 3) as f32) - 1.0; \
         c[i] = ((i % 7) as f32) - 2.0; i += 1; }} \
         mm(a, b, c); let mut s: f32 = 0.0; let mut j: i32 = 0; \
         while j < {n2} {{ s = s + c[j]; j += 1; }} return (s * 100.0) as i32; }}"
    );
    assert!(lowered_calls(&src, "mercury_sgemm"), "recognizer must fire");
    let native = jit(&src, 3).expect("jit");
    let interp = interp(&src, 3).expect("interp");
    assert_eq!(native, interp, "accumulate matmul native vs interp");

    // Independent reference (C starts at its init values, then accumulates A·B).
    let (mut a, mut b, mut c) = (vec![0f32; n2], vec![0f32; n2], vec![0f32; n2]);
    for i in 0..n2 {
        a[i] = ((i % 5) as f32) * 0.5;
        b[i] = ((i % 3) as f32) - 1.0;
        c[i] = ((i % 7) as f32) - 2.0;
    }
    for i in 0..ns {
        for k in 0..ns {
            let aik = a[i * ns + k];
            for j in 0..ns {
                c[i * ns + j] += aik * b[k * ns + j];
            }
        }
    }
    let mut s = 0f32;
    for &cj in &c {
        s += cj;
    }
    assert_eq!(
        native.0,
        (s * 100.0) as i32 as i64,
        "accumulate matmul value vs reference"
    );
}

/// Regression: `(negative float) as iN` must lower to a *signed* fp→int conversion (fptosi), not
/// fptoui. With fptoui the native backend saturates a negative to 0 while the interpreter keeps the
/// signed value, so the two diverged — exactly what broke the `linear_f32` differential gate, where
/// `C[0,0] = -0.5` printed as 0 on native and -5 on the interpreter. Check native == interp == the
/// signed value, for both an i32 and an i64 target and a negative literal-ish operand.
#[test]
fn negative_float_to_signed_int_cast_is_signed() {
    // a = (1.0 - 3.5) * 4.0 = -10.0 -> -10 ; b = (0.0 - 7.0) -> -7 ; stdout "-10\n-7\n", ret -17.
    let src = "module m\nfn main() -> i32 { \
        let mut x: f32 = 1.0; x = x - 3.5; let a: i32 = (x * 4.0) as i32; \
        let mut y: f32 = 0.0; y = y - 7.0; let b: i32 = y as i32; \
        print(a); print(b); return a + b; }";
    let native = jit(src, 3).expect("jit");
    let interp = interp(src, 3).expect("interp");
    assert_eq!(native, interp, "negative float->int cast: native vs interp");
    assert_eq!(native.0, -17, "negative float->int cast value");
    assert_eq!(String::from_utf8(native.1).unwrap(), "-10\n-7\n");
}

/// Out-of-range and negative→unsigned fp→int casts must saturate identically on both backends
/// (Cranelift uses the `_sat` fcvt variants). The interpreter previously bit-masked an i128 cast,
/// which diverged: `1e12 as i32` should clamp to i32::MAX, and `(-5.0) as u32` to 0 — on *both*.
#[test]
fn float_to_int_saturation_matches_native() {
    let src = "module m\nfn main() -> i32 { \
        let mut big: f32 = 1000000.0; big = big * big; let a: i32 = big as i32; \
        let mut neg: f32 = 1.0; neg = neg - 6.0; let b: u32 = neg as u32; \
        print(a); print(b as i32); return 0; }";
    let native = jit(src, 3).expect("jit");
    let interp = interp(src, 3).expect("interp");
    assert_eq!(native, interp, "saturating fp->int: native vs interp");
    assert_eq!(String::from_utf8(native.1).unwrap(), "2147483647\n0\n");
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
            Op::Gep {
                ptr: arr,
                index: idx,
                elem: f32t.clone(),
            },
        );
        let v = b.build(f32t.clone(), Op::ConstFloat((i + 1) as f64, f32t.clone()));
        b.build_void(Op::Store {
            ptr: slot,
            value: v,
        });
    }

    // base = &arr[0]; v = load <4 x f32>; v += splat(10.0); store back
    let zero = b.build(MirType::I64, Op::ConstInt(0, MirType::I64));
    let base = b.build(
        MirType::Ptr,
        Op::Gep {
            ptr: arr,
            index: zero,
            elem: f32t.clone(),
        },
    );
    let v = b.build(vty.clone(), Op::Load(base, vty.clone()));
    let ten = b.build(f32t.clone(), Op::ConstFloat(10.0, f32t.clone()));
    let sp = b.build(vty.clone(), Op::Splat(ten));
    let sum = b.build(vty.clone(), Op::Bin(BinOp::FAdd, v, sp));
    b.build_void(Op::Store {
        ptr: base,
        value: sum,
    });

    // return (i32) arr[2]  ==  3 + 10  ==  13
    let two = b.build(MirType::I64, Op::ConstInt(2, MirType::I64));
    let slot2 = b.build(
        MirType::Ptr,
        Op::Gep {
            ptr: arr,
            index: two,
            elem: f32t.clone(),
        },
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
