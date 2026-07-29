//! JIT correctness tests, and a differential check that the native backend agrees with the
//! tree-walking interpreter (the reference oracle) on stdout and exit code.

use wukong_span::{Interner, SourceId};

/// Compile `src` to optimized MIR and run it natively, returning (exit_code, stdout).
fn jit(src: &str, opt: u8) -> Result<(i64, Vec<u8>), String> {
    let mut interner = Interner::new();
    let (module, pd) = wukong_parser::parse_module(src, SourceId(0), &mut interner);
    assert!(pd.iter().all(|d| !d.is_error()), "parse: {pd:?}");
    let (sema, sd) = wukong_sema::check(&module, &interner);
    assert!(sd.iter().all(|d| !d.is_error()), "sema: {sd:?}");
    let (mut program, ld) = wukong_mir_build::lower_program(&module, &sema, &mut interner);
    assert!(ld.iter().all(|d| !d.is_error()), "lower: {ld:?}");
    wukong_opt::optimize(&mut program, opt);
    let main = interner.intern("main");
    crate::jit_run(&program, main, &interner)
}

/// Compile `src` to a callable native handle (no run), for timing. Honors `WUKONG_P4_NO_256`.
fn compile_native(src: &str, opt: u8) -> crate::JitProgram {
    let mut interner = Interner::new();
    let (module, _) = wukong_parser::parse_module(src, SourceId(0), &mut interner);
    let (sema, _) = wukong_sema::check(&module, &interner);
    let (mut program, _) = wukong_mir_build::lower_program(&module, &sema, &mut interner);
    wukong_opt::optimize(&mut program, opt);
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

    std::env::set_var("WUKONG_P4_NO_256", "1");
    let p128 = compile_native(src, 3);
    std::env::remove_var("WUKONG_P4_NO_256");
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

/// Same-run A/B of the 256-bit AVX2 *reduction* kernel vs the 128-bit CLIF reduction. Ignored by
/// default; run with `--ignored --nocapture`. Sanity check that widening the reduction buys
/// throughput (not a correctness gate — the differential + f64-reference tests are). Uses a
/// *compute-bound* addend `a·(a²+b²)`: its top `*` is fused into the accumulate (`vfmadd`, matching
/// the 128-bit path), and the several muls/adds per pair of loads keep it FP-bound, not L1-bandwidth
/// bound — a memory-bound dot (`a·b`) merely ties, since both widths saturate the same L1 ports.
#[test]
#[ignore]
fn p4_bench_reduction_256_vs_128() {
    // One `+ addend` off `s` (so `reduction_of` claims it); the addend's top `*` fuses via FMA. N is
    // ≥ `VEC256_REDUCTION_MIN_TRIP` so the gated 256-bit path actually fires (below it, both compile to
    // the inlined 128-bit reduction and this would compare 128 against 128).
    let src = "\
        fn main() -> i32 { \
          let mut a: [f32; 2048] = [0.0; 2048]; let mut b: [f32; 2048] = [0.0; 2048]; \
          for i in 0..2048 { a[i] = (i as f32) * 0.001; b[i] = (i as f32) * 0.002 + 1.0; } \
          let mut acc: f32 = 0.0; let mut r: i32 = 0; \
          while r < 50000 { \
            let mut s: f32 = 0.0; \
            for i in 0..2048 { s = s + a[i] * (a[i]*a[i] + b[i]*b[i]); } \
            acc = acc + s; r += 1; \
          } \
          print(acc as i32); return 0; }";
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
    std::env::set_var("WUKONG_P4_NO_256", "1");
    let p128 = compile_native(src, 3);
    std::env::remove_var("WUKONG_P4_NO_256");
    let p256 = compile_native(src, 3);
    best(&p128);
    best(&p256);
    let (t128, t256) = (best(&p128), best(&p256));
    eprintln!(
        "p4 reduction 256-vs-128: 128-bit {:?}, 256-bit {:?}  =>  {:.2}x",
        t128,
        t256,
        t128.as_secs_f64() / t256.as_secs_f64()
    );
}

/// Broad differential sweep of the 256-bit vectorizer: many f32 elementwise body shapes (every
/// recipe op — +−×÷, neg, sqrt, nested if/else blends, multi-stream, invariant scalars/literals,
/// in-place output=input) crossed with trip counts that straddle the 8-lane boundary and the scalar
/// tail, in both `for` and normalized-`while` form. Each must agree native-vs-interp AND -O0-vs-O3.
/// This is the regression lock for the whole feature — if a lane, a tail, or an aliasing case ever
/// drifts, one of these fails.
#[test]
fn p4_vec256_coverage_sweep() {
    // Each body computes `o[i]` (or updates a stream in place) from streams a,b,c and scalar `k`.
    let bodies: &[&str] = &[
        "o[i] = a[i] + b[i] - c[i]",
        "o[i] = a[i] * b[i] * c[i]",
        "o[i] = a[i] / (b[i] + 1.0)",
        "o[i] = -a[i] + b[i] * c[i]",
        "o[i] = sqrt(a[i] * a[i] + b[i] * b[i])",
        "o[i] = a[i] * k + b[i]",                                    // invariant scalar
        "o[i] = if a[i] > b[i] { a[i] } else { b[i] }",              // max via blend
        "o[i] = if a[i] > 0.0 { a[i] } else { 0.0 }",                // relu
        "o[i] = if a[i] < 6.0 { if a[i] > 0.0 { a[i] } else { 0.0 } } else { 6.0 }", // relu6 (nested)
        "o[i] = a[i] * b[i] + a[i] * c[i] - b[i] * c[i] + a[i]",     // load reuse (CSE)
        "a[i] = a[i] * a[i] + 1.0",                                  // in-place, output=input
        "o[i] = sqrt(a[i]) * k - b[i] / c[i] + a[i] * b[i]",         // mixed, 3 streams + scalar
    ];
    // Trip counts around the 8-lane group boundary, its multiples, and non-multiples (tail).
    let sizes: &[usize] = &[1, 2, 7, 8, 9, 15, 16, 17, 24, 63, 64, 65, 100, 255, 256, 257];

    for body in bodies {
        for &n in sizes {
            for form in ["for", "while"] {
                let loop_src = if form == "for" {
                    format!("for i in 0..{n} {{ {body}; }}")
                } else {
                    format!("let mut i: i32 = 0; while i < {n} {{ {body}; i += 1; }}")
                };
                let src = format!(
                    "fn main() -> i32 {{ \
                       let mut a: [f32; {m}] = [0.0; {m}]; let mut b: [f32; {m}] = [0.0; {m}]; \
                       let mut c: [f32; {m}] = [0.0; {m}]; let mut o: [f32; {m}] = [0.0; {m}]; \
                       let k: f32 = 1.5; \
                       for j in 0..{n} {{ a[j] = (j as f32) * 0.5 - 3.0; b[j] = (j as f32) * 0.25 + 1.0; c[j] = (j as f32) - 7.0; }} \
                       {loop_src} \
                       let mut s: f32 = 0.0; for j in 0..{n} {{ s = s + a[j] + o[j]; }} \
                       print(s as i32); return ((s as i32) & 255); }}",
                    m = n.max(1),
                );
                let native = jit(&src, 3)
                    .unwrap_or_else(|e| panic!("jit -O3 [{form} n={n}] `{body}`: {e}"));
                let oracle = interp(&src, 3)
                    .unwrap_or_else(|e| panic!("interp [{form} n={n}] `{body}`: {e}"));
                assert_eq!(native, oracle, "native vs interp [{form} n={n}] `{body}`");
                let o0 = jit(&src, 0)
                    .unwrap_or_else(|e| panic!("jit -O0 [{form} n={n}] `{body}`: {e}"));
                assert_eq!(o0, native, "-O0 vs -O3 [{form} n={n}] `{body}`");
            }
        }
    }
}

/// Run `src` through the interpreter for differential comparison.
fn interp(src: &str, opt: u8) -> Result<(i64, Vec<u8>), String> {
    let mut interner = Interner::new();
    let (module, _) = wukong_parser::parse_module(src, SourceId(0), &mut interner);
    let (sema, _) = wukong_sema::check(&module, &interner);
    let (mut program, _) = wukong_mir_build::lower_program(&module, &sema, &mut interner);
    wukong_opt::optimize(&mut program, opt);
    let main = interner.intern("main");
    wukong_interp::run_with_output(&program, main, &interner)
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

/// An entry point that returns a float must be read out of XMM0 with *its own* width. The whole
/// corpus had no float-returning `main` before this test, which is how `invoke_code` came to fold
/// `f32`/`f16`/`bf16` into the `f64` ABI: the narrow floats return a single-precision XMM0, so
/// reading 64 bits gave a denormal that `as i64` saturated to 0 while the interpreter exited 42.
/// The three exits also pin the rounding each width applies to the same literal — 42.9 is 42 in
/// f32 and f16, but 43 in bf16 (0x422b_999a rounds *up* to 0x422c) — so a fix that read the right
/// register but dropped the narrowing would still fail.
#[test]
fn differential_float_return_type() {
    let programs = [
        ("fn main() -> f32 { print(1); return 42.9; }", 42),
        ("fn main() -> f64 { print(1); return 42.9; }", 42),
        ("fn main() -> f16 { print(1); return 42.9; }", 42),
        ("fn main() -> bf16 { print(1); return 42.9; }", 43),
        // Negative and fractional-magnitude values: `as i64` truncates toward zero on both sides.
        ("fn main() -> f32 { return 0 as f32 - 7.75; }", -7),
        ("fn main() -> f32 { return 0.5; }", 0),
    ];
    for (src, want) in programs {
        for opt in [0u8, 1, 2, 3] {
            let n = jit(src, opt).expect("jit");
            let i = interp(src, opt).expect("interp");
            assert_eq!(n, i, "native vs interp mismatch at -O{opt} for:\n{src}");
            assert_eq!(n.0, want, "exit code at -O{opt} for:\n{src}");
        }
    }
}

/// Heap allocation (`alloc_<T>(n)` / `free(s)`): an alloc'd slice must behave identically on
/// native and the interpreter at every opt level — zero-initialized contents (read before any
/// write), fill/reduce over a vectorizer-territory buffer, mutation across a fn boundary through
/// the `mut` slice param, the negative-length clamp to an empty slice, fat-pointer re-binding of a
/// length past 127 (the I8-byte-copy masking regression), and a store immediately followed by
/// `free` with no intervening read (the pointer escapes into the call, so DSE must keep the
/// store — deleting it would be observable only through the allocator, but the conservatism is
/// what this pins).
#[test]
fn differential_heap_alloc() {
    let programs = [
        // Fill + reduce past the vectorizer thresholds, plus len() and read-before-write.
        "fn main() -> i32 { let n: i64 = 5000; let mut s: []f32 = alloc_f32(n); \
         print(s.len()); print(s[4999]); \
         for i in 0..n { s[i] = (i % 7i64) as f32; } \
         let mut acc: f32 = 0.0; for i in 0..n { acc = acc + s[i]; } \
         print(acc); free(s); return 0; }",
        // Mutation through a fn boundary + zero-init tail + i64/u8 elements.
        "fn fill(mut s: []i64, v: i64) { for i in 0..s.len() { s[i] = v; } } \
         fn main() -> i32 { let s: []i64 = alloc_i64(200); fill(s, 41); \
         print(s[0]); print(s[199]); let b: []u8 = alloc_u8(3); print(b[2]); \
         free(b); free(s); return 0; }",
        // Negative length clamps to an empty slice; iterating it runs zero times.
        "fn main() -> i32 { let e: []f32 = alloc_f32(-8); \
         print(e.len()); let mut hits: i32 = 0; for x in e { hits += 1; } \
         print(hits); free(e); return 0; }",
        // Fat-pointer re-binding with a length past 127, and store->free with no read between.
        "fn main() -> i32 { let big: []f32 = alloc_f32(300); let view: []f32 = big; \
         print(view.len()); let mut d: []f32 = alloc_f32(10); d[3] = 9.5; free(d); \
         free(big); return 0; }",
    ];
    for src in programs {
        for opt in [0u8, 1, 2, 3] {
            let n = jit(src, opt).expect("jit");
            let i = interp(src, opt).expect("interp");
            assert_eq!(
                n, i,
                "heap-alloc native vs interp mismatch at -O{opt} for:\n{src}"
            );
        }
    }
}

/// Tuples lower to a padded byte buffer with byte-offset field GEPs (no aggregate MIR type). The
/// native backend writes the real packed layout; the interpreter indexes the byte offset as a slot.
/// Distinct field offsets never alias, so both must agree on the observed field values — including a
/// heterogeneous `(f32, i32)` whose `i32` field sits at a padded offset, and a field assignment.
#[test]
fn differential_tuple() {
    let programs = [
        "fn main() -> i32 { let t = (3, 4); return t.0 + t.1; }",
        "fn main() -> i32 { let mut u = (1.5, 2); u.0 = u.0 + 0.5; \
         print(u.0); print(t_dummy(u.1)); return 0; } fn t_dummy(x: i32) -> i32 { return x * 3; }",
        "fn main() -> i32 { let v = (10, 20, 30); let w = (v.0 + v.1, v.2); \
         return w.0 + w.1; }",
        "fn main() -> i32 { let p = (1, 2.5, 7); print(p.0); print(p.2); \
         let q: f32 = p.1 * 2.0; print(q as i32); return p.0 + p.2; }",
    ];
    for src in programs {
        for opt in [0u8, 1, 2, 3] {
            let n = jit(src, opt).expect("jit");
            let i = interp(src, opt).expect("interp");
            assert_eq!(n, i, "tuple native vs interp mismatch at -O{opt} for:\n{src}");
        }
    }
}

/// Structs lower to the same byte-buffer/byte-GEP representation as tuples, with named fields
/// resolved through the declared layout. Native and interp must agree, including a heterogeneous
/// struct whose float field sits at a padded offset and a literal with fields out of declaration
/// order (each value routed to its named offset, not its position).
#[test]
fn differential_struct() {
    let programs = [
        "struct P { x: i32, y: i32 } \
         fn main() -> i32 { let p = P { x: 3, y: 4 }; return p.x + p.y; }",
        "struct M { a: i32, b: f32, c: i32 } \
         fn main() -> i32 { let mut m = M { a: 10, b: 3.5, c: 20 }; m.a = m.a + m.c; \
         print(m.a); print(m.b); return m.a + (m.b as i32); }",
        "struct P { x: f32, y: f32 } \
         fn main() -> i32 { let q = P { y: 100.0, x: 1.0 }; print(q.x); print(q.y); \
         return (q.x + q.y) as i32; }",
    ];
    for src in programs {
        for opt in [0u8, 1, 2, 3] {
            let n = jit(src, opt).expect("jit");
            let i = interp(src, opt).expect("interp");
            assert_eq!(n, i, "struct native vs interp mismatch at -O{opt} for:\n{src}");
        }
    }
}

/// Nested aggregates: a struct field that is itself a struct (and a tuple field that is a struct)
/// lays out recursively, reads as a *pointer* (the by-pointer convention, so `o.inner.a` GEPs twice
/// rather than loading a whole buffer through a register), and initializes either by recursing into
/// a nested literal or by a leaf-precise deep copy of a non-literal aggregate value. Native and
/// interp must agree across all four `-O` levels, including a 3-deep chain and an out-of-order
/// initializer.
#[test]
fn differential_nested_struct() {
    let programs = [
        // struct-in-struct, nested literal.
        "struct Inner { a: i32, b: i32 } struct Outer { inner: Inner, c: i32 } \
         fn main() -> i32 { let o = Outer { inner: Inner { a: 10, b: 20 }, c: 5 }; \
         print(o.inner.a); print(o.inner.b); return o.inner.a + o.inner.b + o.c; }",
        // aggregate field initialized from a *variable* (exercises the leaf-precise deep copy).
        "struct Inner { a: i32, b: i32 } struct Outer { inner: Inner, c: i32 } \
         fn main() -> i32 { let src = Inner { a: 10, b: 20 }; \
         let o = Outer { inner: src, c: 5 }; return o.inner.a + o.inner.b + o.c; }",
        // 3-deep nesting + a mixed-precision inner field at a padded offset.
        "struct A { v: f32 } struct B { a: A, w: i32 } struct C { b: B, x: i32 } \
         fn main() -> i32 { let c = C { b: B { a: A { v: 2.5 }, w: 2 }, x: 4 }; \
         print(c.b.a.v); return (c.b.a.v as i32) + c.b.w + c.x; }",
        // tuple whose first field is a struct.
        "struct A { v: i32, w: i32 } \
         fn main() -> i32 { let t = (A { v: 100, w: 1 }, 8); return t.0.v + t.0.w + t.1; }",
        // write through a nested field path.
        "struct Inner { a: i32 } struct Outer { inner: Inner, c: i32 } \
         fn main() -> i32 { let mut o = Outer { inner: Inner { a: 1 }, c: 2 }; \
         o.inner.a = o.inner.a + 40; return o.inner.a + o.c; }",
    ];
    for src in programs {
        for opt in [0u8, 1, 2, 3] {
            let n = jit(src, opt).expect("jit");
            let i = interp(src, opt).expect("interp");
            assert_eq!(n, i, "nested-struct native vs interp mismatch at -O{opt} for:\n{src}");
        }
    }
}

/// An array whose element is an aggregate (`[Struct; N]` / `[(..); N]`): indexing then a field
/// access, a runtime-index field write, a by-value array parameter, and an `[agg; n]` repeat. The
/// element is wider than one slot, so the element GEP must stride the whole element on *both*
/// backends — native scales the index by `size_of(elem)` bytes; the interpreter scales by the
/// element's `slot_count`. A scalar `Op::Load` of an aggregate element (the prior bug) verifier-
/// rejected native for a struct element and segfaulted it for a wider tuple element while the
/// interpreter mis-strided by a single slot — so this must agree across all four `-O` levels.
#[test]
fn differential_array_of_aggregate() {
    let programs = [
        // array of 2-field structs: const + runtime index, field read + write, by-value param.
        "struct P { x: i32, y: i32 } \
         fn psum(a: [P; 2]) -> i32 { return a[0].x + a[0].y + a[1].x + a[1].y; } \
         fn main() -> i32 { let mut a: [P; 2] = [P { x: 7, y: 8 }, P { x: 9, y: 10 }]; \
         let i: i32 = 1; a[i].x = 100; print(a[0].x); print(a[1].x); return psum(a); }",
        // array of tuples (a wider, >4-byte element — the prior segfault case).
        "fn main() -> i32 { let a: [(i32, i32); 2] = [(7, 8), (9, 10)]; \
         print(a[0].0); print(a[1].1); return a[0].0 + a[1].1 + a[1].0 + a[0].1; }",
        // mixed-precision struct element at a padded offset, read through a runtime index.
        "struct M { a: i32, b: f32 } \
         fn main() -> i32 { let arr: [M; 3] = [M { a: 1, b: 1.5 }, M { a: 2, b: 2.5 }, \
         M { a: 3, b: 3.5 }]; let k: i32 = 2; print(arr[k].a); print(arr[k].b); \
         return arr[0].a + arr[1].a + arr[k].a; }",
        // `[agg; n]` repeat: every element is an independent deep copy.
        "struct P { x: i32, y: i32 } \
         fn main() -> i32 { let mut a: [P; 4] = [P { x: 5, y: 6 }; 4]; a[2].x = 50; \
         return a[0].x + a[1].x + a[2].x + a[3].y; }",
    ];
    for src in programs {
        for opt in [0u8, 1, 2, 3] {
            let n = jit(src, opt).expect("jit");
            let i = interp(src, opt).expect("interp");
            assert_eq!(
                n, i,
                "array-of-aggregate native vs interp mismatch at -O{opt} for:\n{src}"
            );
        }
    }
}

/// Deep recursion: JIT'd native code recurses on the host call stack (one machine frame per Wukong
/// call), so without a big-stack worker it overflows the default ~8 MiB stack and *aborts* the
/// process, while the interpreter (on its 512 MiB stack) completes — a divergence the opt-invariance
/// harness misses (it only re-runs the interpreter). Both backends now run on matching 512 MiB
/// stacks, so a depth that overflows the default stack must agree. ~30k frames is well past the
/// default stack but fast.
#[test]
fn differential_deep_recursion() {
    let src = "fn sum(n: i32) -> i32 { if n == 0 { return 0; } return n + sum(n - 1); } \
               fn main() -> i32 { print(sum(30000)); return 0; }";
    for opt in [0u8, 2] {
        let n = jit(src, opt).expect("jit");
        let i = interp(src, opt).expect("interp");
        assert_eq!(n, i, "deep-recursion native vs interp mismatch at -O{opt}");
    }
}

/// Printing an unsigned integer renders its magnitude, not the signed two's-complement
/// reinterpretation: a high-bit-set `u32`/`u64` (a quantization scale, a hash) would otherwise
/// print negative. mir_build routes an unsigned `print` argument to `print_u`/`rt_print_u64` after
/// zero-extending to 64 bits; the interpreter's `print_u` formats the same bits as `u64`, so the
/// captured stdout must agree native==interp (a signed argument is unaffected).
#[test]
fn differential_unsigned_print() {
    let src = "fn main() -> i32 { \
               let g: u32 = 3221225472; print(g); \
               let h: u64 = 18446744073709551615; print(h); \
               let s: u8 = 200; print(s); \
               let i: i32 = -5; print(i); \
               return 0; }";
    for opt in [0u8, 1, 2, 3] {
        let n = jit(src, opt).expect("jit");
        let i = interp(src, opt).expect("interp");
        assert_eq!(n, i, "unsigned-print native vs interp mismatch at -O{opt}");
    }
    // The captured stdout is the actual evidence (the return value is 0 either way).
    let (_, out) = jit(src, 0).expect("jit");
    assert_eq!(
        String::from_utf8_lossy(&out),
        "3221225472\n18446744073709551615\n200\n-5\n"
    );
}

/// `if`/`match` used as a *value* whose arms have different numeric types: each arm is coerced to
/// the expression's joined type so all arms pass the merge-block parameter the same MIR type.
/// Without the coercion an arm of a different width/kind (`if c { 1 } else { 2.5 }`) passes a
/// mismatched value the native verifier rejects while the interpreter runs loosely — a divergence on
/// a program the front-end accepted. Must agree across all four `-O` levels.
#[test]
fn differential_mixed_branch_types() {
    let programs = [
        // if arms: i32 vs f32 -> f32 (1.0 and 3.5).
        "fn main() -> i32 { let c: bool = true; let x = if c { 1 } else { 2.5 }; \
         let d: bool = false; let y = if d { 7 } else { 3.5 }; \
         return (x as i32) + ((y * 10.0) as i32); }",
        // match arms: i32 vs f32 -> f32.
        "fn pick(n: i32) -> i32 { let r = match n { 0 => 10, _ => 2.5 }; return (r * 2.0) as i32; } \
         fn main() -> i32 { return pick(0) + pick(9); }",
        // homogeneous arms are unaffected (the coercion is a no-op, no spurious rounding).
        "fn main() -> i32 { let c: bool = false; let x = if c { 100 } else { 200 }; \
         let r = match x { 100 => 1, 200 => 2, _ => 3 }; return x + r; }",
    ];
    for src in programs {
        for opt in [0u8, 1, 2, 3] {
            let n = jit(src, opt).expect("jit");
            let i = interp(src, opt).expect("interp");
            assert_eq!(n, i, "mixed-branch native vs interp mismatch at -O{opt} for:\n{src}");
        }
    }
}

/// `&&` / `||` short-circuit: the RHS runs only when the LHS doesn't decide the result. The captured
/// stdout (the `jit`/`interp` helpers return it) is the side-effect evidence, so a regression to a
/// bitwise `and`/`or` of both operands would change the printed trace AND must still agree
/// native==interp. Also proves the guard idiom `n != 0 && 100/n > 0` does NOT divide by zero.
#[test]
fn differential_short_circuit() {
    let programs = [
        // side(t) prints t and returns true; falseside prints t and returns false.
        "fn side(t: i32) -> bool { print(t); return true; } \
         fn fs(t: i32) -> bool { print(t); return false; } \
         fn main() -> i32 { \
           if false && side(1) { print(91); } \
           if true || side(2) { print(92); } \
           if true && side(3) { print(93); } \
           if false || side(4) { print(94); } \
           if fs(5) && side(6) { print(95); } \
           if side(7) || side(8) { print(97); } \
           return 0; }",
        // short-circuit guards an unsafe RHS: n==0 so 100/n must never be evaluated.
        "fn main() -> i32 { let n: i32 = 0; let mut hit: i32 = 0; \
         if n != 0 && 100 / n > 0 { hit = 1; } print(hit); return 0; }",
        // nested / chained short-circuit with mixed operators.
        "fn t(x: i32) -> bool { print(x); return x > 0; } \
         fn main() -> i32 { if t(1) && (t(0) || t(2)) && t(3) { print(99); } return 0; }",
    ];
    for src in programs {
        for opt in [0u8, 1, 2, 3] {
            let n = jit(src, opt).expect("jit");
            let i = interp(src, opt).expect("interp");
            assert_eq!(n, i, "short-circuit native vs interp mismatch at -O{opt} for:\n{src}");
        }
    }
}

/// `continue` inside a range `for` must run the loop step (the step lives at the body tail, so the
/// lowering routes `continue` to a latch that increments first) — otherwise the loop never
/// terminates. Native must agree with interp and actually halt (the test would hang on a regression).
/// Covers a plain `for`, a nested `for`, and a `continue` reached on the first iteration.
#[test]
fn differential_for_continue() {
    let programs = [
        // skip odds: 0+2+4+6+8 = 20.
        "fn main() -> i32 { let mut s: i32 = 0; \
         for i in 0..10 { if i % 2 == 1 { continue; } s = s + i; } return s; }",
        // continue on the very first iteration (i==0) then proceed.
        "fn main() -> i32 { let mut s: i32 = 0; \
         for i in 0..5 { if i == 0 { continue; } s = s + i; } return s; }",
        // nested: inner continue skips j==1; sum over i in 0..3, j in 0..3, j!=1 -> per i (0+2)=2, *3 rows,
        // plus i*3 added once per inner pass that isn't skipped (2 passes) -> handled by direct sum.
        "fn main() -> i32 { let mut s: i32 = 0; \
         for i in 0..3 { for j in 0..3 { if j == 1 { continue; } s = s + i * 10 + j; } } return s; }",
        // continue interacts with a break in the same loop.
        "fn main() -> i32 { let mut s: i32 = 0; \
         for i in 0..100 { if i == 5 { break; } if i % 2 == 0 { continue; } s = s + i; } return s; }",
    ];
    for src in programs {
        for opt in [0u8, 1, 2, 3] {
            let n = jit(src, opt).expect("jit");
            let i = interp(src, opt).expect("interp");
            assert_eq!(n, i, "for+continue native vs interp mismatch at -O{opt} for:\n{src}");
        }
    }
}

/// A labeled `break`/`continue` `'l` targets the enclosing loop named `'l` (not just the innermost),
/// so it can escape or restart an *outer* loop from inside a nested one. Native must agree with interp
/// that the branch goes to the labeled loop's break/continue block. Covers labeled break/continue out
/// of a nested `for`, a labeled `break` out of a `while`, a labeled `loop`, and the interaction with an
/// unlabeled inner break (the inner one stays innermost-scoped).
#[test]
fn differential_labeled_loops() {
    let programs = [
        // continue 'outer skips the rest of the inner loop and resumes the outer for.
        "fn main() -> i32 { let mut c: i32 = 0; \
         'o: for i in 0..4 { for j in 0..4 { if j == 2 { continue 'o; } c = c + 1; } } return c; }",
        // break 'outer escapes both loops at once.
        "fn main() -> i32 { let mut c: i32 = 0; \
         'o: for i in 0..5 { for j in 0..5 { if i + j >= 3 { break 'o; } c = c + 1; } } return c; }",
        // labeled break out of a while from inside a for.
        "fn main() -> i32 { let mut k: i32 = 0; \
         'w: while k < 1000 { for m in 0..10 { if m == 4 { break 'w; } k = k + 1; } } return k; }",
        // labeled loop {} with a labeled break.
        "fn main() -> i32 { let mut n: i32 = 0; \
         'l: loop { n = n + 1; for _q in 0..3 { if n >= 7 { break 'l; } n = n + 1; } } return n; }",
        // unlabeled inner break coexists with an outer label: inner break exits only the inner loop.
        "fn main() -> i32 { let mut c: i32 = 0; \
         'o: for i in 0..3 { for j in 0..9 { if j == 2 { break; } c = c + 1; } if i == 1 { break 'o; } } \
         return c; }",
    ];
    for src in programs {
        for opt in [0u8, 1, 2, 3] {
            let n = jit(src, opt).expect("jit");
            let i = interp(src, opt).expect("interp");
            assert_eq!(n, i, "labeled-loop native vs interp mismatch at -O{opt} for:\n{src}");
        }
    }
}

/// Structs cross function boundaries: a struct passed **by value** (`fn f(p: Pt)`) is passed by base
/// pointer (its registry-aware ABI is a buffer pointer, not the `I32` the free `mir_ty` gave — which
/// is what made the native verifier reject the field GEP and `mem2reg` panic at -O2), and field
/// access **through a pointer/reference** (`p.x` on a `&Pt` / `*mut Pt`) auto-derefs. Native == interp
/// across -O0..-O3, including a mutation written through a `*mut Pt` that the caller observes.
#[test]
fn differential_struct_across_fns() {
    let programs = [
        // struct by value: param passed by pointer, fields read in the callee.
        "struct Pt { x: i32, y: i32 } fn f(p: Pt) -> i32 { return p.x + p.y; } \
         fn main() -> i32 { let p = Pt { x: 8, y: 9 }; return f(p); }",
        // field through &Pt (read) and *mut Pt (write the caller observes).
        "struct Pt { x: i32, y: i32 } \
         fn rd(p: &Pt) -> i32 { return p.x + p.y; } \
         fn setx(p: *mut Pt, v: i32) { p.x = v; } \
         fn main() -> i32 { let mut s = Pt { x: 1, y: 2 }; let a = rd(&s); \
         setx(&mut s, 40); return a + s.x + s.y; }",
        // a struct param alongside scalar params (ABI ordering) + a nested struct field through a ref.
        "struct Inner { a: i32 } struct Outer { inner: Inner, b: i32 } \
         fn sum(o: &Outer, k: i32) -> i32 { return o.inner.a + o.b + k; } \
         fn main() -> i32 { let o = Outer { inner: Inner { a: 10 }, b: 5 }; return sum(&o, 100); }",
    ];
    for src in programs {
        for opt in [0u8, 1, 2, 3] {
            let n = jit(src, opt).expect("jit");
            let i = interp(src, opt).expect("interp");
            assert_eq!(n, i, "struct-across-fns native vs interp mismatch at -O{opt} for:\n{src}");
        }
    }
}

/// Whole-aggregate **assignment** (not just initialization): storing a struct *value* into a place
/// whose type is an aggregate must deep-copy the buffer, not store the RHS buffer's base pointer
/// into the destination's first slot. Before the fix this silently miscompiled on *both* backends
/// (the interpreter wrote a `Ptr` into slot 0, native wrote a 32-bit-truncated address), so the
/// assignment path is routed through the same leaf-precise `init_field`/`emit_copy` as a `let`.
/// Covers `*p = Struct{..}`, `*p = struct_var`, a nested struct field `s.f = Struct{..}`/`= var`.
/// (Array-*of*-struct element assignment is excluded — that hits the documented array-of-aggregate
/// limitation where the interpreter's slot-indexed and native's byte-indexed memory can't agree.)
#[test]
fn differential_struct_assign() {
    let programs = [
        // store a struct *literal* through a *mut pointer (the core C7 case).
        "struct Pt { x: i32, y: i32 } \
         fn put(p: *mut Pt) { *p = Pt { x: 3, y: 4 }; } \
         fn main() -> i32 { let mut s = Pt { x: 1, y: 2 }; put(&mut s); return s.x + s.y; }",
        // store a non-literal struct *value* (a by-value param) through a pointer.
        "struct Pt { x: i32, y: i32 } \
         fn cp(dst: *mut Pt, src: Pt) { *dst = src; } \
         fn main() -> i32 { let mut a = Pt { x: 1, y: 1 }; let b = Pt { x: 10, y: 20 }; \
         cp(&mut a, b); return a.x + a.y; }",
        // assign a whole struct literal into a nested struct field.
        "struct Pt { x: i32, y: i32 } struct Box { lo: Pt, hi: Pt } \
         fn main() -> i32 { let mut bx = Box { lo: Pt { x: 0, y: 0 }, hi: Pt { x: 0, y: 0 } }; \
         bx.hi = Pt { x: 5, y: 6 }; return bx.hi.x + bx.hi.y; }",
        // assign a struct *variable* into a nested struct field.
        "struct Pt { x: i32, y: i32 } struct Box { lo: Pt, hi: Pt } \
         fn main() -> i32 { let b = Pt { x: 10, y: 20 }; \
         let mut bx = Box { lo: Pt { x: 0, y: 0 }, hi: Pt { x: 0, y: 0 } }; \
         bx.lo = b; return bx.lo.x + bx.lo.y; }",
    ];
    for src in programs {
        for opt in [0u8, 1, 2, 3] {
            let n = jit(src, opt).expect("jit");
            let i = interp(src, opt).expect("interp");
            assert_eq!(n, i, "struct-assign native vs interp mismatch at -O{opt} for:\n{src}");
        }
    }
}

/// Returning an aggregate **by value** (`fn f() -> Struct`). Modeled with an sret ABI entirely in
/// `mir_build`: an aggregate-returning function gets a hidden leading pointer parameter and returns
/// void; `return <agg>` deep-copies into that pointer, and a call site allocates the destination
/// buffer, prepends it, and uses it as the call's value. Both backends only ever pass/copy pointers,
/// so no aggregate rides in a register — they agree at every `-O`. Covers an explicit `return`, a
/// tail expression, a tuple return, a nested struct, a non-literal (`return p`) return, returned
/// values used as arguments, and a field read off a returned temporary.
#[test]
fn differential_struct_return() {
    let programs = [
        // explicit `return Struct{..}`, let-bound and field-read off a temporary.
        "struct Pt { x: i32, y: i32 } \
         fn make(a: i32, b: i32) -> Pt { return Pt { x: a, y: b }; } \
         fn main() -> i32 { let p = make(3, 4); return p.x + p.y + make(10, 20).y; }",
        // tail-expression return (no `return` keyword).
        "struct Pt { x: i32, y: i32 } \
         fn make(a: i32, b: i32) -> Pt { Pt { x: a, y: b } } \
         fn main() -> i32 { let p = make(4, 5); return p.x + p.y; }",
        // tuple return.
        "fn mk(a: i32, b: i32) -> (i32, i32) { return (a, b); } \
         fn main() -> i32 { let t = mk(3, 4); return t.0 + t.1; }",
        // nested struct return.
        "struct Inner { a: i32 } struct Outer { inner: Inner, b: i32 } \
         fn mk(a: i32, b: i32) -> Outer { return Outer { inner: Inner { a: a }, b: b }; } \
         fn main() -> i32 { let o = mk(10, 5); return o.inner.a + o.b; }",
        // non-literal return (`return p`) + returned values flowing into another call's args.
        "struct Pt { x: i32, y: i32 } \
         fn id(p: Pt) -> Pt { return p; } \
         fn add(p: Pt, q: Pt) -> Pt { return Pt { x: p.x + q.x, y: p.y + q.y }; } \
         fn main() -> i32 { let s = add(id(Pt { x: 1, y: 2 }), Pt { x: 3, y: 4 }); \
         return s.x + s.y; }",
        // multiple return paths (conditional), each an aggregate.
        "struct Pt { x: i32, y: i32 } \
         fn pick(c: i32) -> Pt { if c > 0 { return Pt { x: 1, y: 1 }; } Pt { x: 5, y: 5 } } \
         fn main() -> i32 { return pick(1).x + pick(0).x; }",
    ];
    for src in programs {
        for opt in [0u8, 1, 2, 3] {
            let n = jit(src, opt).expect("jit");
            let i = interp(src, opt).expect("interp");
            assert_eq!(n, i, "struct-return native vs interp mismatch at -O{opt} for:\n{src}");
        }
    }
}

/// `match` expressions: integer/bool/negative-integer literal patterns, a wildcard / identifier
/// catch-all (the latter binding the scrutinee), and `if` guards, in both value and statement
/// position. Lowered to an if-else chain over the arms (scrutinee evaluated once), so the
/// interpreter oracle and native backend agree at every `-O`. A non-exhaustive fallthrough yields a
/// zero default (lenient, identical on both backends — no trap).
#[test]
fn differential_match() {
    let programs = [
        // integer literal arms + wildcard default.
        "fn f(n: i32) -> i32 { match n { 0 => 10, 1 => 20, 2 => 30, _ => 99 } } \
         fn main() -> i32 { return f(0) + f(1) + f(2) + f(5); }", // 10+20+30+99 = 159
        // identifier binding catch-all.
        "fn f(n: i32) -> i32 { match n { 0 => 0, x => x * 2 } } \
         fn main() -> i32 { return f(21) + f(0); }", // 42 + 0
        // guard arms (negative result exercises signed compare).
        "fn sign(n: i32) -> i32 { match n { 0 => 0, x if x > 0 => 1, _ => 0 - 1 } } \
         fn main() -> i32 { return sign(5) * 100 + sign(0) * 10 + sign(0 - 9) + 1000; }",
        // negative literal pattern.
        "fn f(n: i32) -> i32 { match n { -1 => 100, 0 => 0, _ => 1 } } \
         fn main() -> i32 { return f(0 - 1) + f(0) + f(7); }", // 100 + 0 + 1
        // bool scrutinee.
        "fn f(b: bool) -> i32 { match b { true => 7, false => 9 } } \
         fn main() -> i32 { return f(true) * 10 + f(false); }", // 79
        // match as a statement with block arms and a side effect.
        "fn main() -> i32 { let mut a: i32 = 0; let n: i32 = 2; \
         match n { 1 => { a = 3; } 2 => { a = 5; } _ => { a = 0; } } return a; }",
        // match bound in a `let`, and a match whose scrutinee is itself an expression.
        "fn main() -> i32 { let n: i32 = 3; let r: i32 = match n + 0 { 3 => 30, _ => 0 }; return r; }",
        // nested match (an arm body is itself a match).
        "fn f(a: i32, b: i32) -> i32 { match a { 0 => match b { 0 => 1, _ => 2 }, _ => 3 } } \
         fn main() -> i32 { return f(0,0)*100 + f(0,9)*10 + f(9,9); }", // 123
    ];
    for src in programs {
        for opt in [0u8, 1, 2, 3] {
            let n = jit(src, opt).expect("jit");
            let i = interp(src, opt).expect("interp");
            assert_eq!(n, i, "match native vs interp mismatch at -O{opt} for:\n{src}");
        }
    }
}

/// Radix integer literals — hex `0xFF`, octal `0o17`, binary `0b1010`, with `_` separators and type
/// suffixes — must evaluate to their real value, not `0`. The literal value is a compile-time MIR
/// constant, so both backends agree by construction; these assertions pin the *value* (the bug was
/// that `parse_int` kept only the leading decimal-digit run, so every non-decimal literal was `0`).
#[test]
fn differential_radix_literals() {
    // (program, expected exit code) — `main` returns the literal-derived value directly.
    let cases = [
        ("fn main() -> i32 { return 0xFF; }", 255),
        ("fn main() -> i32 { return 0o17; }", 15),
        ("fn main() -> i32 { return 0b1010; }", 10),
        ("fn main() -> i32 { return 0xFF & 0x0F; }", 15),
        ("fn main() -> i32 { return 0xFF_FF; }", 65535),
        ("fn main() -> i32 { return 1_000 + 0x10; }", 1016),
        // hex array index and a decimal literal sanity check.
        ("fn main() -> i32 { let a: [i32; 4] = [10,20,30,40]; return a[0x2]; }", 30),
        ("fn main() -> i32 { return 1_000_000; }", 1_000_000),
    ];
    for (src, want) in cases {
        for opt in [0u8, 1, 2, 3] {
            let n = jit(src, opt).expect("jit");
            let i = interp(src, opt).expect("interp");
            assert_eq!(n, i, "radix native vs interp mismatch at -O{opt} for:\n{src}");
            assert_eq!(n.0, want, "radix literal wrong value at -O{opt} for:\n{src}");
        }
    }
}

/// Float → narrow integer (`i8`/`i16`/`u8`/`u16`) casts. Cranelift's `fcvt_to_{sint,uint}_sat`
/// cannot target a sub-32-bit result on x64 and used to panic the backend (an ICE on a perfectly
/// valid program). The fix converts to `i32` saturating, clamps to the narrow type's range, and
/// `ireduce`s — reproducing Rust `as` / the interpreter's saturating semantics, so in-range,
/// out-of-range, negative-to-unsigned, and truncating inputs all agree with the oracle.
#[test]
fn differential_float_narrow_int_cast() {
    let cases = [
        ("fn main() -> i32 { let a: f32 = 7.0;          return (a as u8) as i32; }", 7),
        ("fn main() -> i32 { let a: f32 = 7.0;          return (a as i8) as i32; }", 7),
        ("fn main() -> i32 { let a: f32 = 7.0;          return (a as u16) as i32; }", 7),
        ("fn main() -> i32 { let a: f32 = 7.0;          return (a as i16) as i32; }", 7),
        ("fn main() -> i32 { let a: f32 = 300.0;        return (a as u8) as i32; }", 255),
        ("fn main() -> i32 { let a: f32 = 300.0;        return (a as i8) as i32; }", 127),
        ("fn main() -> i32 { let a: f32 = 0.0 - 1.0;    return (a as u8) as i32; }", 0),
        ("fn main() -> i32 { let a: f32 = 0.0 - 300.0;  return (a as i8) as i32; }", -128),
        ("fn main() -> i32 { let a: f32 = 70000.0;      return (a as u16) as i32; }", 65535),
        ("fn main() -> i32 { let a: f32 = 0.0 - 70000.0; return (a as i16) as i32; }", -32768),
        ("fn main() -> i32 { let a: f64 = 3.9;          return (a as u8) as i32; }", 3),
        // a >= 32-bit target still uses the direct path.
        ("fn main() -> i32 { let a: f32 = 1000000.0;    return (a as i32); }", 1_000_000),
    ];
    for (src, want) in cases {
        for opt in [0u8, 1, 2, 3] {
            let n = jit(src, opt).expect("jit");
            let i = interp(src, opt).expect("interp");
            assert_eq!(n, i, "float->narrow-int native vs interp mismatch at -O{opt} for:\n{src}");
            assert_eq!(n.0, want, "float->narrow-int wrong value at -O{opt} for:\n{src}");
        }
    }
}

/// Top-level `const` references. The def map records only a const's *type*, so reading one as a
/// value used to fail with C0001 ("value reference … not yet supported"). Sema now type-checks each
/// const initializer and records it; `mir_build` inlines the initializer at every use site. Covers
/// a bare reference, use in arithmetic / as an array index / as a loop bound, an `f32` const, and a
/// const that references another const (recursive inlining). Both backends agree by construction.
#[test]
fn differential_top_level_const() {
    let cases = [
        ("const N: i32 = 64; fn main() -> i32 { return N; }", 64),
        ("const N: i32 = 64; fn main() -> i32 { return N * 2 + 1; }", 129),
        ("const I: i32 = 2; fn main() -> i32 { let a: [i32; 4] = [10,20,30,40]; return a[I]; }", 30),
        ("const A: i32 = 64; const B: i32 = A + 1; fn main() -> i32 { return B; }", 65),
        ("const LIM: i32 = 5; fn main() -> i32 { let mut c: i32 = 0; \
          for i in 0..LIM { c += 1; } return c; }", 5),
        ("const PI: f32 = 3.5; fn main() -> i32 { return PI as i32; }", 3),
        ("const BIG: i64 = 1000; fn main() -> i32 { return BIG as i32; }", 1000),
    ];
    for (src, want) in cases {
        for opt in [0u8, 1, 2, 3] {
            let n = jit(src, opt).expect("jit");
            let i = interp(src, opt).expect("interp");
            assert_eq!(n, i, "const native vs interp mismatch at -O{opt} for:\n{src}");
            assert_eq!(n.0, want, "const wrong value at -O{opt} for:\n{src}");
        }
    }
}

/// Nested tuple-field access `t.0.0`. The lexer glues the trailing `0.0` into a single float token,
/// so the parser now splits a plain `N.M` float into two consecutive tuple-field accesses. Covers a
/// read, an assignment target, and a deeper `t.0.0.0`; both backends agree (this is the existing
/// tuple-field lowering, just reachable now).
#[test]
fn differential_nested_tuple_field() {
    let cases = [
        ("fn main() -> i32 { let t = ((1, 2), 3); return t.0.0; }", 1),
        ("fn main() -> i32 { let t = ((1, 2), 3); return t.0.1; }", 2),
        ("fn main() -> i32 { let t = (9, (7, 8)); return t.1.0; }", 7),
        ("fn main() -> i32 { let mut t = ((1, 2), 3); t.0.0 = 50; return t.0.0 + t.0.1; }", 52),
        ("fn main() -> i32 { let t = (((5, 6), 7), 8); return t.0.0.0; }", 5),
    ];
    for (src, want) in cases {
        for opt in [0u8, 1, 2, 3] {
            let n = jit(src, opt).expect("jit");
            let i = interp(src, opt).expect("interp");
            assert_eq!(n, i, "nested-tuple-field native vs interp mismatch at -O{opt} for:\n{src}");
            assert_eq!(n.0, want, "nested-tuple-field wrong value at -O{opt} for:\n{src}");
        }
    }
}

/// C-style enums: a variant `E::B` lowers to its integer discriminant (auto-incremented from 0, or
/// set by an explicit `= <int>` and continuing from there). Covers an explicit discriminant, plain
/// auto-increment, continuation after an explicit value, a `let x: E = E::A` binding, an equality
/// comparison, and arithmetic via `as i32`. The value is a compile-time constant, so both backends
/// agree.
#[test]
fn differential_enum() {
    let cases = [
        ("enum E { A = 10, B = 20 } fn main() -> i32 { return E::B as i32; }", 20),
        ("enum Color { Red, Green, Blue } fn main() -> i32 { return Color::Blue as i32; }", 2),
        ("enum E { A = 5, B, C } fn main() -> i32 { return E::C as i32; }", 7),
        ("enum E { A = 10, B = 20 } fn main() -> i32 { let x: E = E::A; return x as i32; }", 10),
        ("enum E { A = 10, B = 20 } \
          fn main() -> i32 { let x: E = E::B; if x == E::B { return 1; } return 0; }", 1),
        ("enum E { A = 10, B = 20 } \
          fn main() -> i32 { return (E::A as i32) + (E::B as i32); }", 30),
    ];
    for (src, want) in cases {
        for opt in [0u8, 1, 2, 3] {
            let n = jit(src, opt).expect("jit");
            let i = interp(src, opt).expect("interp");
            assert_eq!(n, i, "enum native vs interp mismatch at -O{opt} for:\n{src}");
            assert_eq!(n.0, want, "enum wrong value at -O{opt} for:\n{src}");
        }
    }
}

/// Tuple-destructuring `let (a, b) = …`. The `let` lowering binds each sub-pattern to its field's
/// place within the initialized tuple buffer (a scalar reads via a `Load`, an aggregate field by
/// pointer; a nested tuple pattern recurses; `_` skips). Covers a literal tuple, a triple, a
/// struct-returning… (here a tuple-returning) call, a nested pattern, a wildcard, and mutating a
/// destructured binding. Both backends agree.
#[test]
fn differential_let_destructure() {
    let cases = [
        ("fn main() -> i32 { let (a, b) = (3, 4); return a + b; }", 7),
        ("fn main() -> i32 { let (a, b, c) = (1, 2, 3); return a + b + c; }", 6),
        ("fn mk() -> (i32, i32) { return (10, 20); } \
          fn main() -> i32 { let (x, y) = mk(); return x + y; }", 30),
        ("fn main() -> i32 { let ((a, b), c) = ((1, 2), 3); return a + b + c; }", 6),
        ("fn main() -> i32 { let (a, _) = (5, 99); return a; }", 5),
        ("fn main() -> i32 { let (a, b) = (10, 20); a = 30; return a + b; }", 50),
    ];
    for (src, want) in cases {
        for opt in [0u8, 1, 2, 3] {
            let n = jit(src, opt).expect("jit");
            let i = interp(src, opt).expect("interp");
            assert_eq!(n, i, "let-destructure native vs interp mismatch at -O{opt} for:\n{src}");
            assert_eq!(n.0, want, "let-destructure wrong value at -O{opt} for:\n{src}");
        }
    }
}

/// Math intrinsics on integer operands. `abs`/`round`/`floor`/`ceil`/`trunc` are type-preserving
/// (`abs(-5): i32`): integer abs lowers to `select(x<0,-x,x)` and the roundings to the identity;
/// `sqrt` and the transcendentals promote an int operand to `f32`. Previously these emitted float
/// ops on an int SSA value — MIR the verifier rejected on native while the interpreter ran it lossily
/// (a silent divergence). Pins native==interp at every `-O`, plus that float abs is unchanged.
#[test]
fn differential_int_math_intrinsics() {
    let cases = [
        ("fn main() -> i32 { return abs(-5); }", 5),
        ("fn main() -> i32 { return abs(7); }", 7),
        ("fn main() -> i32 { return abs(-2147483647); }", 2147483647),
        // round/floor/ceil/trunc on integers are the identity.
        ("fn main() -> i32 { return round(5) + floor(-9) + ceil(3) + trunc(8); }", 7),
        // sqrt promotes the int operand to f32 (16 -> 16.0 -> 4.0 -> 4).
        ("fn main() -> i32 { return sqrt(16) as i32; }", 4),
        // float abs still works (the float path is unchanged).
        ("fn main() -> i32 { return abs(-3.5) as i32; }", 3),
    ];
    for (src, want) in cases {
        for opt in [0u8, 1, 2, 3] {
            let n = jit(src, opt).expect("jit");
            let i = interp(src, opt).expect("interp");
            assert_eq!(n, i, "int-math native vs interp mismatch at -O{opt} for:\n{src}");
            assert_eq!(n.0, want, "int-math wrong value at -O{opt} for:\n{src}");
        }
    }
    // i64 abs returns i64 (no precision loss, unlike the old via-f32 detour).
    let src64 = "fn main() -> i64 { let x: i64 = -5000000000; return abs(x); }";
    for opt in [0u8, 1, 2, 3] {
        let n = jit(src64, opt).expect("jit");
        let i = interp(src64, opt).expect("interp");
        assert_eq!(n, i, "i64-abs native vs interp mismatch at -O{opt}");
        assert_eq!(n.0, 5000000000i64, "i64-abs wrong value at -O{opt}");
    }
}

/// `fmax`/`fmin` on integer *variables*. Like `sqrt`/`exp`/`pow`, they promote an int operand to
/// `f32` (the float compare-and-select). Previously their lowering used a bare `lower_expr` (no
/// coercion), so the float `Cmp(Fogt/Folt)` ran on an `i32` operand: MIR the verifier and Cranelift
/// reject on native, while the interpreter computed an integer max and returned silently — a backend
/// divergence on `fmax(int, int)`. Pins native==interp at every `-O`; the float forms are unchanged.
#[test]
fn differential_fmax_fmin_int() {
    let cases = [
        ("fn main() -> i32 { let a: i32 = 5; let b: i32 = 3; return fmax(a, b) as i32; }", 5),
        ("fn main() -> i32 { let a: i32 = 5; let b: i32 = 3; return fmin(a, b) as i32; }", 3),
        ("fn main() -> i32 { let a: i32 = -7; let b: i32 = 2; return fmax(a, b) as i32; }", 2),
        ("fn main() -> i32 { let a: i32 = -7; let b: i32 = 2; return fmin(a, b) as i32; }", -7),
        // float forms unchanged (the operands were already f32).
        ("fn main() -> i32 { let a: f32 = 5.0; let b: f32 = 3.0; return fmax(a, b) as i32; }", 5),
        ("fn main() -> i32 { let a: f32 = 5.0; let b: f32 = 3.0; return fmin(a, b) as i32; }", 3),
        // an integer literal already coerced before; still does.
        ("fn main() -> i32 { return fmax(5, 3) as i32; }", 5),
    ];
    for (src, want) in cases {
        for opt in [0u8, 1, 2, 3] {
            let n = jit(src, opt).expect("jit");
            let i = interp(src, opt).expect("interp");
            assert_eq!(n, i, "fmax/fmin-int native vs interp mismatch at -O{opt} for:\n{src}");
            assert_eq!(n.0, want, "fmax/fmin-int wrong value at -O{opt} for:\n{src}");
        }
    }
}

/// Richer `match` patterns: or-patterns `1 | 2 | 3`, half-open `lo..hi` and inclusive `lo..=hi`
/// ranges, and enum-variant patterns `Color::Red` (compared by discriminant). Each lowers to a
/// pure value test (an OR of equalities / a range conjunction / a discriminant equality), so native
/// and interp agree at every `-O`. Also covers an or-pattern nested in a tuple field and a negative
/// range bound.
#[test]
fn differential_match_patterns() {
    const FIZZ: &str = "fn fizz(n: i32) -> i32 { return match n \
        { 0 | 1 | 2 => 100, 3..10 => 200, 10..=20 => 300, _ => 400 }; } \
        fn main() -> i32 { return fizz";
    const NAME: &str = "enum Color { Red, Green, Blue } \
        fn name(c: Color) -> i32 { return match c \
        { Color::Red => 1, Color::Green => 2, Color::Blue => 3 }; } \
        fn main() -> i32 { return name";
    let cases = [
        (format!("{FIZZ}(0); }}"), 100),
        (format!("{FIZZ}(2); }}"), 100),
        (format!("{FIZZ}(3); }}"), 200),
        (format!("{FIZZ}(9); }}"), 200),
        (format!("{FIZZ}(10); }}"), 300),
        (format!("{FIZZ}(20); }}"), 300),
        (format!("{FIZZ}(21); }}"), 400),
        (format!("{FIZZ}(-5); }}"), 400),
        (format!("{NAME}(Color::Red); }}"), 1),
        (format!("{NAME}(Color::Green); }}"), 2),
        (format!("{NAME}(Color::Blue); }}"), 3),
        // an or-pattern nested in a tuple field.
        ("fn main() -> i32 { return match (1, 7) { (0 | 1, y) => y, _ => 0 }; }".to_string(), 7),
        // a negative range bound, signed comparison.
        ("fn f(n: i32) -> i32 { return match n { -5..0 => 1, 0..=5 => 2, _ => 3 }; } \
          fn main() -> i32 { return f(-3); }".to_string(), 1),
    ];
    for (src, want) in &cases {
        for opt in [0u8, 1, 2, 3] {
            let n = jit(src, opt).expect("jit");
            let i = interp(src, opt).expect("interp");
            assert_eq!(n, i, "match-patterns native vs interp mismatch at -O{opt} for:\n{src}");
            assert_eq!(n.0, *want, "match-patterns wrong value at -O{opt} for:\n{src}");
        }
    }
}

/// Char-literal `match` patterns: a single char `'a'`, an inclusive char range `'0'..='9'`, and a
/// char or-pattern `'x' | 'y' | 'z'`. A char decodes to its code point and matches by an integer
/// equality / range test, so native and interp agree at every `-O`. The parser previously rejected a
/// char literal in pattern position (E0206) even though char works in `let`/arithmetic/comparison.
#[test]
fn differential_char_patterns() {
    const C: &str = "fn classify(d: char) -> i32 { return match d \
        { 'a' => 1, 'b' => 2, '0'..='9' => 3, 'x' | 'y' | 'z' => 4, _ => 0 }; } \
        fn main() -> i32 { return classify";
    let cases = [
        (format!("{C}('a'); }}"), 1),
        (format!("{C}('b'); }}"), 2),
        (format!("{C}('0'); }}"), 3),
        (format!("{C}('9'); }}"), 3),
        (format!("{C}('x'); }}"), 4),
        (format!("{C}('z'); }}"), 4),
        (format!("{C}('q'); }}"), 0),
    ];
    for (src, want) in &cases {
        for opt in [0u8, 1, 2, 3] {
            let n = jit(src, opt).expect("jit");
            let i = interp(src, opt).expect("interp");
            assert_eq!(n, i, "char-pattern native vs interp mismatch at -O{opt} for:\n{src}");
            assert_eq!(n.0, *want, "char-pattern wrong value at -O{opt} for:\n{src}");
        }
    }
}

/// A 1-D `Tensor[..]` parameter fed to a recognized 1-D runtime kernel — streaming-elementwise
/// (velem), a vmath activation, a reduction, and an `@parallel` reduction. A `Tensor` param binds to
/// a slot holding the base pointer, so the recognizers must load the base out of the slot (as the
/// regular index path and the GEMM path do) before GEPing. They used to GEP off the slot *address*,
/// so the kernel read past it — the interpreter trapped while native segfaulted / returned wrong
/// values, a backend divergence on a documented feature (only 2-D tensor params were ever tested).
#[test]
fn differential_tensor_1d_kernels() {
    let cases = [
        // velem: out[i] = a[i] * 3  -> 1*3 + 4*3 = 15
        ("fn f(a: Tensor[f32,4], mut o: Tensor[f32,4]) { for i in 0..4 { o[i] = a[i] * 3.0; } } \
          fn main() -> i32 { let a:[f32;4]=[1.0,2.0,3.0,4.0]; let o:[f32;4]=[0.0,0.0,0.0,0.0]; \
          f(a, o); return (o[0] + o[3]) as i32; }", 15),
        // vmath: out[i] = exp(a[i])  -> exp(0)*4 = 4
        ("fn f(a: Tensor[f32,4], mut o: Tensor[f32,4]) { for i in 0..4 { o[i] = exp(a[i]); } } \
          fn main() -> i32 { let a:[f32;4]=[0.0,0.0,0.0,0.0]; let o:[f32;4]=[0.0,0.0,0.0,0.0]; \
          f(a, o); return (o[0] + o[1] + o[2] + o[3]) as i32; }", 4),
        // reduction: sum -> 10
        ("fn f(a: Tensor[f32,4]) -> f32 { let mut s:f32=0.0; for i in 0..4 { s = s + a[i]; } return s; } \
          fn main() -> i32 { let a:[f32;4]=[1.0,2.0,3.0,4.0]; return f(a) as i32; }", 10),
        // @parallel reduction: sum of eight 1.0 -> 8
        ("@parallel fn f(a: Tensor[f32,8]) -> f32 { let mut s:f32=0.0; for i in 0..8 { s = s + a[i]; } return s; } \
          fn main() -> i32 { let a:[f32;8]=[1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0]; return f(a) as i32; }", 8),
    ];
    for (src, want) in cases {
        for opt in [0u8, 1, 2, 3] {
            let n = jit(src, opt).expect("jit");
            let i = interp(src, opt).expect("interp");
            assert_eq!(n, i, "tensor-1d-kernel native vs interp mismatch at -O{opt} for:\n{src}");
            assert_eq!(n.0, want, "tensor-1d-kernel wrong value at -O{opt} for:\n{src}");
        }
    }
}

/// `for i in 0..n` where the END bound is wider than the literal-`0` start (`i64`/`usize`). The
/// counter must be driven by the wider of the two bound types: a literal `0` lowers to `i32`, so an
/// `i64` end produced `cmp.i32 i32, i64` — verifier-invalid MIR that crashed the native backend
/// while the interpreter trapped. Only the *scalar* loop hit it (an array loop vectorizes the
/// counter away), so it's exercised over a `Tensor` param (vectorizer declines) plus the array
/// variant, with both `i64` and `usize` bounds. Regression guard for the loop-counter widening.
#[test]
fn differential_loop_wide_bound() {
    let cases = [
        // Tensor param, i64 bound: sum of 1..=6 -> 21.
        ("fn f(a: Tensor[f32,6], n: i64) -> f32 { let mut s:f32=0.0; for i in 0..n { s = s + a[i]; } return s; } \
          fn main() -> i32 { let a:[f32;6]=[1.0,2.0,3.0,4.0,5.0,6.0]; return f(a, 6) as i32; }", 21),
        // Tensor param, usize bound -> 21.
        ("fn f(a: Tensor[f32,6], n: usize) -> f32 { let mut s:f32=0.0; for i in 0..n { s = s + a[i]; } return s; } \
          fn main() -> i32 { let a:[f32;6]=[1.0,2.0,3.0,4.0,5.0,6.0]; return f(a, 6) as i32; }", 21),
        // array param (vectorized counter path), i64 bound: sum of 1..=5 -> 15.
        ("fn f(a: [f32;5], n: i64) -> f32 { let mut s:f32=0.0; for i in 0..n { s = s + a[i]; } return s; } \
          fn main() -> i32 { let a:[f32;5]=[1.0,2.0,3.0,4.0,5.0]; return f(a, 5) as i32; }", 15),
        // integer reduction, i64 bound, over a Tensor[i32] param -> 6.
        ("fn f(a: Tensor[i32,4], n: i64) -> i32 { let mut s:i32=0; for i in 0..n { s = s + a[i]; } return s; } \
          fn main() -> i32 { let a:[i32;4]=[0,1,2,3]; return f(a, 4); }", 6),
    ];
    for (src, want) in cases {
        for opt in [0u8, 1, 2, 3] {
            let n = jit(src, opt).expect("jit");
            let i = interp(src, opt).expect("interp");
            assert_eq!(n, i, "wide-bound loop native vs interp mismatch at -O{opt} for:\n{src}");
            assert_eq!(n.0, want, "wide-bound loop wrong value at -O{opt} for:\n{src}");
        }
    }
}

/// A transcendental written INLINE inside an autovectorizable elementwise `for` loop (so the generic
/// vectorizer — not the vmath recognizer — lifts the inlined polynomial to `<N x f32>`). This used to
/// panic the interpreter: the polynomial's range-reduction has an internal `Cmp`/`Select`, and a
/// scalar operand (e.g. the literal `1.0` in `1.0 + exp(-z)`) was not splatted to N lanes, so the
/// vectorized `Op::Cmp` indexed past the 1-lane operand. Now scalars broadcast to N lanes and the
/// whole thing is bit-exact on both backends. Covers the exact in-place-SiLU trigger, a `Cmp+Select`
/// branch with an inline `exp`, and a couple of compound transcendentals. `want == -1` means assert
/// only the native==interp differential (transcendental f32 result not worth pinning).
#[test]
fn differential_vectorized_inline_transcendental() {
    let cases: [(&str, i64); 5] = [
        // The exact documented trigger: in-place SiLU with a scalar `1.0` operand. silu(0.1)*1000 = 52.
        ("fn main() -> i32 { let g:[f32;8]=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8]; \
          for i in 0..8 { let z = g[i]; g[i] = z / (1.0 + exp(0.0 - z)); } return (g[0]*1000.0) as i32; }", 52),
        // inline exp in a compound (vmath declines) form, x=0 → exp(0)*2 = 2.
        ("fn main() -> i32 { let a:[f32;8]=[0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7]; \
          let b:[f32;8]=[2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0]; let o:[f32;8]=[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]; \
          for i in 0..8 { o[i] = exp(a[i]) * b[i]; } return (o[0]) as i32; }", 2),
        // sigmoid inline, x=0 → sigmoid(0)*2 = 1.
        ("fn main() -> i32 { let a:[f32;8]=[0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7]; \
          let b:[f32;8]=[2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0]; let o:[f32;8]=[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]; \
          for i in 0..8 { o[i] = sigmoid(a[i]) * b[i]; } return (o[0]) as i32; }", 1),
        // a vectorized Cmp+Select with an inline exp on the true branch. a[0]=0.2 ≤ 0.3 → 0.2*100 = 20.
        ("fn main() -> i32 { let a:[f32;8]=[0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]; \
          let o:[f32;8]=[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]; \
          for i in 0..8 { if a[i] > 0.3 { o[i] = exp(a[i]); } else { o[i] = a[i] * 100.0; } } return (o[0]*1.0) as i32; }", 20),
        // sin+cos compound with a nonzero input (exercises range reduction); differential-only.
        ("fn main() -> i32 { let a:[f32;8]=[0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]; \
          let o:[f32;8]=[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]; \
          for i in 0..8 { o[i] = sin(a[i]) + cos(a[i]); } return (o[0]*100.0) as i32; }", -1),
    ];
    for (src, want) in cases {
        for opt in [0u8, 1, 2, 3] {
            let n = jit(src, opt).expect("jit");
            let i = interp(src, opt).expect("interp");
            assert_eq!(n, i, "inline-transcendental native vs interp mismatch at -O{opt} for:\n{src}");
            if want >= 0 {
                assert_eq!(n.0, want, "inline-transcendental wrong value at -O{opt} for:\n{src}");
            }
        }
    }
}

/// An EXHAUSTIVE `match` with no `_` catch-all (every enum variant / both bool cases). Each arm is a
/// conditional discriminant test, so mir_build emits the "no arm matched" fallthrough block
/// structurally — but it is dynamically dead and now terminates in `Unreachable` (was a zero default).
/// Pins that the dead block stays well-typed at every opt level and both backends agree. Companion to
/// the E0405 sema rejection of *non*-exhaustive value matches (a compile-fail, covered by tests/fail).
#[test]
fn differential_match_exhaustiveness() {
    let cases = [
        // exhaustive enum, no `_`: rank(Blue) = 3.
        ("enum Color { Red, Green, Blue } \
          fn rank(c: Color) -> i32 { return match c { Color::Red => 1, Color::Green => 2, Color::Blue => 3 }; } \
          fn main() -> i32 { return rank(Color::Blue); }", 3),
        // exhaustive enum, first variant: rank(Red) = 1.
        ("enum Color { Red, Green, Blue } \
          fn rank(c: Color) -> i32 { return match c { Color::Red => 1, Color::Green => 2, Color::Blue => 3 }; } \
          fn main() -> i32 { return rank(Color::Red); }", 1),
        // exhaustive bool, no `_`: pick(false) = 20.
        ("fn pick(b: bool) -> i32 { return match b { true => 10, false => 20 }; } \
          fn main() -> i32 { return pick(false); }", 20),
        // an enum match folded into an arithmetic expression (the dead fallthrough is still emitted).
        ("enum Dir { N, S } fn dval(d: Dir) -> i32 { return match d { Dir::N => 1, Dir::S => -1 }; } \
          fn main() -> i32 { return dval(Dir::N) * 7 + dval(Dir::S); }", 6),
    ];
    for (src, want) in cases {
        for opt in [0u8, 1, 2, 3] {
            let n = jit(src, opt).expect("jit");
            let i = interp(src, opt).expect("interp");
            assert_eq!(n, i, "exhaustive-match native vs interp mismatch at -O{opt} for:\n{src}");
            assert_eq!(n.0, want, "exhaustive-match wrong value at -O{opt} for:\n{src}");
        }
    }
}

/// Casting a numeric value to `bool` is C-like truthiness (`x != 0`), not a low-bit truncation: `2
/// as bool` used to be `false` (low bit 0) while `if 2` is `true`. Now both compare `!= 0`. Lowered
/// to an `Op::Cmp`, so both backends run it identically. Pins nonzero→true, zero→false, NaN/float→0,
/// and the bool→int direction (regression).
#[test]
fn differential_cast_to_bool() {
    let cases = [
        ("fn main() -> i32 { let x: i32 = 0; return (x as bool) as i32; }", 0),
        ("fn main() -> i32 { let x: i32 = 2; return (x as bool) as i32; }", 1), // was 0 under truncation
        ("fn main() -> i32 { let x: i32 = 255; return (x as bool) as i32; }", 1),
        ("fn main() -> i32 { let x: i32 = 0 - 4; return (x as bool) as i32; }", 1), // negative is nonzero
        ("fn main() -> i32 { let x: f32 = 0.5; return (x as bool) as i32; }", 1),
        ("fn main() -> i32 { let x: f32 = 0.0; return (x as bool) as i32; }", 0),
        // the cast now agrees with the condition path on the same value.
        ("fn main() -> i32 { let x: i32 = 2; if (x as bool) { return 7; } else { return 0; } }", 7),
        // regression: bool -> int is unchanged.
        ("fn main() -> i32 { let b: bool = true; return (b as i32) + 10; }", 11),
    ];
    for (src, want) in cases {
        for opt in [0u8, 1, 2, 3] {
            let n = jit(src, opt).expect("jit");
            let i = interp(src, opt).expect("interp");
            assert_eq!(n, i, "cast-to-bool native vs interp mismatch at -O{opt} for:\n{src}");
            assert_eq!(n.0, want, "cast-to-bool wrong value at -O{opt} for:\n{src}");
        }
    }
}

/// Bitwise complement `~x`. The parser now accepts `~` (previously E0202 "found ~") as a prefix
/// operator aliased to `UnOp::Not` — bitwise on an integer, logical on a `bool` (masked to width).
/// Both backends share `Op::Not`, so it is bit-exact; pins the values and that `!`/`~` agree.
#[test]
fn differential_bitwise_not() {
    let cases = [
        ("fn main() -> i32 { let x: i32 = 5; return ~x; }", -6),
        ("fn main() -> i32 { let x: i32 = 0; return ~x; }", -1),
        // De Morgan: ~(a & b) == (~a) | (~b).
        ("fn main() -> i32 { let a: i32 = 12; let b: i32 = 10; \
          if ~(a & b) == (~a) | (~b) { return 1; } else { return 0; } }", 1),
        // width masking: ~5 as u8 = 0xFA = 250.
        ("fn main() -> i32 { let x: u8 = 5; let y: u8 = ~x; return y as i32; }", 250),
        // composed with arithmetic: ~3 + 10 = -4 + 10 = 6.
        ("fn main() -> i32 { let x: i32 = 3; return ~x + 10; }", 6),
        // `~` and `!` are the same op on an integer.
        ("fn main() -> i32 { let x: i32 = 42; if ~x == !x { return 7; } else { return 0; } }", 7),
    ];
    for (src, want) in cases {
        for opt in [0u8, 1, 2, 3] {
            let n = jit(src, opt).expect("jit");
            let i = interp(src, opt).expect("interp");
            assert_eq!(n, i, "bitwise-not native vs interp mismatch at -O{opt} for:\n{src}");
            assert_eq!(n.0, want, "bitwise-not wrong value at -O{opt} for:\n{src}");
        }
    }
}

/// An unsuffixed integer literal that overflows i32 must keep its value, not silently wrap to its
/// low 32 bits. `9000000000` used to default to i32 and bake a `const.i32` (= 410065408) into the
/// shared MIR, so BOTH backends agreed on the wrong value — the interp-vs-native gate was blind to
/// it. The default now widens to i64 when the value does not fit i32. Asserted via *stdout* (the i64
/// exit code is i32-narrow): the printed value distinguishes the true value from the old wrap.
#[test]
fn differential_int_literal_widening() {
    let cases = [
        // return position: `return 9000000000` from `-> i64`.
        ("fn f() -> i64 { return 9000000000; } fn main() -> i32 { print(f()); return 0; }", "9000000000\n"),
        // bare literal, type inferred (no annotation) -> i64, not a truncated i32.
        ("fn main() -> i32 { let x = 9000000000; print(x); return 0; }", "9000000000\n"),
        // cast operand: the literal is i64 *before* the (no-op) cast, not an i32 wrap widened after.
        ("fn main() -> i32 { let x = 9000000000 as i64; print(x); return 0; }", "9000000000\n"),
        // struct-field init of a wider field.
        ("struct B { v: i64 } fn main() -> i32 { let b = B { v: 12345678901 }; print(b.v); return 0; }", "12345678901\n"),
        // assignment to a wider field.
        ("struct B { v: i64 } fn main() -> i32 { let mut b = B { v: 0 }; b.v = 12345678901; print(b.v); return 0; }", "12345678901\n"),
        // unsigned return: an i64-typed literal reinterpreted as u64 (same width, positive) — the old
        // path truncated to i32 then sign-extended, printing 18446744072414584320.
        ("fn f() -> u64 { return 3000000000; } fn main() -> i32 { print(f()); return 0; }", "3000000000\n"),
        // a value that fits i32 is unaffected (stays i32).
        ("fn main() -> i32 { let x = 5; print(x); return 0; }", "5\n"),
    ];
    for (src, want) in cases {
        for opt in [0u8, 1, 2, 3] {
            let n = jit(src, opt).expect("jit");
            let i = interp(src, opt).expect("interp");
            assert_eq!(n, i, "int-literal-widening native vs interp mismatch at -O{opt} for:\n{src}");
            assert_eq!(
                String::from_utf8_lossy(&n.1),
                want,
                "int-literal-widening wrong value at -O{opt} for:\n{src}"
            );
        }
    }
}

/// An array / tuple literal adapts its elements to a matching aggregate annotation, so a typed buffer
/// can be built straight from literals (`let a: [i8; 2] = [127, 0]`). Before, each was rejected as an
/// `[i32; 2]` / `(i32, i32)` mismatch (E0401); now the annotation's element type is threaded into each
/// element (recursively, through nested aggregates), so the elements lower at the right width and
/// out-of-range elements are range-checked. native ≡ interp at every opt level.
#[test]
fn differential_aggregate_literal_adapt() {
    let cases = [
        // narrow element type.
        ("fn main() -> i32 { let xs: [i8; 2] = [127, 0]; return xs[0] as i32; }", 127),
        // negative element literal (peels the unary minus).
        ("fn main() -> i32 { let xs: [i8; 2] = [-56, 5]; return xs[0] as i32; }", -56),
        // tuple element adaptation.
        ("fn main() -> i32 { let t: (u8, u8) = (200, 1); return t.0 as i32; }", 200),
        // array-repeat adaptation.
        ("fn main() -> i32 { let a: [i8; 3] = [5; 3]; return (a[0] + a[1] + a[2]) as i32; }", 15),
        // wide element type: no truncation (9000000000 / 1e9 = 9).
        ("fn main() -> i32 { let a: [i64; 2] = [9000000000, 0]; return (a[0] / 1000000000) as i32; }", 9),
        // nested: array of tuples.
        ("fn main() -> i32 { let a: [(i8, i8); 2] = [(1, 2), (3, 4)]; return (a[1].0 + a[1].1) as i32; }", 7),
        // nested: array of structs with a narrow field.
        ("struct S { x: i8 } fn main() -> i32 { let a: [S; 2] = [S { x: 100 }, S { x: 27 }]; return (a[0].x + a[1].x) as i32; }", 127),
    ];
    for (src, want) in cases {
        for opt in [0u8, 1, 2, 3] {
            let n = jit(src, opt).expect("jit");
            let i = interp(src, opt).expect("interp");
            assert_eq!(n, i, "aggregate-literal native vs interp mismatch at -O{opt} for:\n{src}");
            assert_eq!(n.0, want, "aggregate-literal wrong value at -O{opt} for:\n{src}");
        }
    }
}

/// A top-level `const` used as an array length resolves to its value. Both sema's `eval_usize` and
/// mir_build's `const_usize_expr` were integer-literal-only, silently sizing the slot to 0 (a
/// zero-length array, then a spurious E0501). They now both resolve a single-segment path naming a
/// `const` to its checked initializer, MIRRORED exactly so the alloca'd length agrees with sema's
/// index-bounds checks (a disagreement would re-introduce an interp/native slot-size divergence).
/// Guards a plain const, a const-references-const chain, an array-repeat count, a struct field, and
/// order-independence (consts are populated in `collect`, before any body is checked).
#[test]
fn differential_const_array_length() {
    let cases = [
        // plain const length: a[0] + a[3] = 10 + 40.
        ("const N: usize = 4; fn main() -> i32 { let a: [i32; N] = [10, 20, 30, 40]; return a[0] + a[3]; }", 50),
        // const-references-const chain: C = B = A = 2.
        ("const A: usize = 2; const B: usize = A; const C: usize = B; \
          fn main() -> i32 { let c: [i32; C] = [7, 8]; return c[0] + c[1]; }", 15),
        // array-repeat count via const: [3; N] with N = 4 -> a[0] + a[3] = 6.
        ("const N: usize = 4; fn main() -> i32 { let a: [i32; N] = [3; N]; return a[0] + a[3]; }", 6),
        // struct field const-length array.
        ("const K: usize = 3; struct Buf { data: [i32; K] } \
          fn main() -> i32 { let b = Buf { data: [10, 20, 30] }; return b.data[2]; }", 30),
        // const defined AFTER use (order-independent).
        ("fn main() -> i32 { let a: [i32; M] = [1, 2, 3, 4, 5]; return a[4]; } const M: usize = 5;", 5),
    ];
    for (src, want) in cases {
        for opt in [0u8, 1, 2, 3] {
            let n = jit(src, opt).expect("jit");
            let i = interp(src, opt).expect("interp");
            assert_eq!(n, i, "const-array-length native vs interp mismatch at -O{opt} for:\n{src}");
            assert_eq!(n.0, want, "const-array-length wrong value at -O{opt} for:\n{src}");
        }
    }
}

/// `char` is a usable type: a char literal types as `char`, the `char` annotation resolves to the
/// same scalar (so `let c: char = 'A'` checks), and char <-> int casts are valid both ways. char
/// lowers to a 32-bit int in MIR, so the interpreter and native backend agree bit-for-bit.
#[test]
fn differential_char_type() {
    let cases = [
        // char literal annotated, cast to int.
        ("fn main() -> i32 { let c: char = 'A'; return c as i32; }", 65),
        // int -> char -> int.
        ("fn main() -> i32 { let d: char = 66 as char; return d as i32; }", 66),
        // ordered comparison on char (not bool — must NOT be rejected).
        ("fn main() -> i32 { let a: char = 'a'; let b: char = 'b'; if a < b { return 1; } return 0; }", 1),
        // char as a struct field.
        ("struct G { code: char } fn main() -> i32 { let g = G { code: 'Z' }; return g.code as i32; }", 90),
        // char arithmetic: 'z' - 'a' = 25.
        ("fn main() -> i32 { return ('z' as i32) - ('a' as i32); }", 25),
    ];
    for (src, want) in cases {
        for opt in [0u8, 1, 2, 3] {
            let n = jit(src, opt).expect("jit");
            let i = interp(src, opt).expect("interp");
            assert_eq!(n, i, "char native vs interp mismatch at -O{opt} for:\n{src}");
            assert_eq!(n.0, want, "char wrong value at -O{opt} for:\n{src}");
        }
    }
}

/// A binary arithmetic/bitwise expression of constant literals adapts to a wider annotated scalar
/// type (`let v: i64 = 0 - 16`), like the unary-minus and bare-literal forms. Both operands are
/// retyped to the annotation so the whole expression is lowered at one width — ill-typed MIR (i32
/// operands, i64 result) would otherwise diverge / be rejected. Only all-constant expressions adapt.
#[test]
fn differential_binary_const_adapt() {
    let cases = [
        // binary const adapts to i64 (was an i32-vs-i64 mismatch); -16.
        ("fn main() -> i32 { let v: i64 = 0 - 16; return v as i32; }", -16),
        // multiply + add fold, wide target: 1_000_001 / 1000 = 1000.
        ("fn main() -> i32 { let v: i64 = 1000 * 1000 + 1; return (v / 1000) as i32; }", 1000),
        // float binary adapts to f64: (1.5 - 0.5) * 10 = 10.
        ("fn main() -> i32 { let v: f64 = 1.5 - 0.5; return (v * 10.0) as i32; }", 10),
        // bitwise const adapts: 0xF0 | 0x0F = 255.
        ("fn main() -> i32 { let v: i64 = 0xF0 | 0x0F; return v as i32; }", 255),
        // nested binary, all literals: 3 + 2*2 = 7.
        ("fn main() -> i32 { let v: i64 = 3 + 2 * 2; return v as i32; }", 7),
    ];
    for (src, want) in cases {
        for opt in [0u8, 1, 2, 3] {
            let n = jit(src, opt).expect("jit");
            let i = interp(src, opt).expect("interp");
            assert_eq!(n, i, "binary-const-adapt native vs interp mismatch at -O{opt} for:\n{src}");
            assert_eq!(n.0, want, "binary-const-adapt wrong value at -O{opt} for:\n{src}");
        }
    }
}

/// An unsuffixed integer literal in `(i64::MAX, u64::MAX]` defaults to `u64`, not `i32`. It used to
/// default to i32 in a cast-source / bare-expression position and TRUNCATE to its low 32 bits — a
/// gate-blind silent miscompile (both backends baked the same `const.i32`). The annotated path
/// (`let x: u64 = ...`) was already correct; mir_build's `parse_int` already uses i128, so only the
/// scalar width was wrong. Verifies the cast path now matches the annotated path, bit-for-bit.
#[test]
fn differential_u64_range_literal() {
    let cases = [
        // u64-range literal in a cast source must not truncate: 2^63 / 1e18 = 9.
        ("fn main() -> i32 { return (9223372036854775808 as u64 / 1000000000000000000 as u64) as i32; }", 9),
        // cast path == annotated path (both 2^63): difference 0.
        ("fn main() -> i32 { let x: u64 = 9223372036854775808; return ((9223372036854775808 as u64) - x) as i32; }", 0),
        // u64::MAX low byte via bitand = 255.
        ("fn main() -> i32 { return (18446744073709551615 as u64 & 255 as u64) as i32; }", 255),
        // bare u64-range literals (no cast), magnitude-typed u64: 1e19 - (1e19 - 10) = 10.
        ("fn main() -> i32 { return (10000000000000000000 as u64 - 9999999999999999990 as u64) as i32; }", 10),
    ];
    for (src, want) in cases {
        for opt in [0u8, 1, 2, 3] {
            let n = jit(src, opt).expect("jit");
            let i = interp(src, opt).expect("interp");
            assert_eq!(n, i, "u64-range-literal native vs interp mismatch at -O{opt} for:\n{src}");
            assert_eq!(n.0, want, "u64-range-literal wrong value at -O{opt} for:\n{src}");
        }
    }
}

/// Field / element access through an EXPLICIT pointer deref — `(*p).field`, `(*p)[i]`, `(*p).0`,
/// `&mut (*p).field`. The base `*p` was lowered as a *value* (a `Load` of the whole aggregate), and
/// GEPing a field off the loaded buffer is invalid MIR (`gep base [N x i8]`): interp and native-O{1,2,3}
/// all errored on it, but native-O0 codegen'd the wild gep into a SIGSEGV. The base is now lowered as a
/// place (the pointer), so the explicit spelling lowers like the implicit auto-deref `p.field`. Also
/// guards the already-working `*p = Struct{..}` whole-store and `**pp` double-deref against regression.
#[test]
fn differential_deref_field_access() {
    let cases = [
        // (*p).field read: 7 + 5 = 12.
        ("struct S { a: i32, b: i32 } fn rd(p: *mut S) -> i32 { return (*p).a + (*p).b; } \
          fn main() -> i32 { let mut s: S = S { a: 7, b: 5 }; return rd(&mut s); }", 12),
        // (*p).field write: set to 99.
        ("struct S { a: i32 } fn wr(p: *mut S) { (*p).a = 99; } \
          fn main() -> i32 { let mut s: S = S { a: 1 }; wr(&mut s); return s.a; }", 99),
        // (*p)[i] on a *mut array: xs[2] = 30.
        ("fn el(p: *mut [i32; 4]) -> i32 { return (*p)[2]; } \
          fn main() -> i32 { let mut xs: [i32; 4] = [10, 20, 30, 40]; return el(&mut xs); }", 30),
        // &mut (*p).field, written through the borrow: 42.
        ("struct S { a: i32 } fn bump(p: *mut S) { let q = &mut (*p).a; *q = 42; } \
          fn main() -> i32 { let mut s: S = S { a: 0 }; bump(&mut s); return s.a; }", 42),
        // explicit deref of a tuple pointer: (*p).0 = 5.
        ("fn first(p: *mut (i32, i32)) -> i32 { return (*p).0; } \
          fn main() -> i32 { let mut t: (i32, i32) = (5, 6); return first(&mut t); }", 5),
        // regression: whole-struct store through a pointer still works.
        ("struct S { a: i32, b: i32 } fn set(p: *mut S) { *p = S { a: 3, b: 4 }; } \
          fn main() -> i32 { let mut s: S = S { a: 0, b: 0 }; set(&mut s); return s.a * 10 + s.b; }", 34),
        // regression: scalar double-deref still works.
        ("fn rd(pp: *mut *mut i32) -> i32 { return **pp; } \
          fn main() -> i32 { let mut x: i32 = 77; let mut p: *mut i32 = &mut x; return rd(&mut p); }", 77),
    ];
    for (src, want) in cases {
        for opt in [0u8, 1, 2, 3] {
            let n = jit(src, opt).expect("jit");
            let i = interp(src, opt).expect("interp");
            assert_eq!(n, i, "deref-field-access native vs interp mismatch at -O{opt} for:\n{src}");
            assert_eq!(n.0, want, "deref-field-access wrong value at -O{opt} for:\n{src}");
        }
    }
}

/// An `if`/`match` used as a VALUE whose result is an aggregate (tuple/struct). The value flows
/// through the control-flow merge block as the aggregate's base POINTER, so the merge param must be
/// typed `Ptr`, not the byte-buffer `Array` type. It used to be typed `Array`, which the arm's pointer
/// arg matched only under the interpreter's loose -O0 typing — `mem2reg`'s verifier rejected it, so
/// these compiled at -O0 but panicked at -O2/-O3. Pins that the merge param is now `Ptr` at every level
/// on both backends.
#[test]
fn differential_aggregate_value_merge() {
    let cases = [
        // if → tuple, then-arm taken: (1,2) -> 1+2 = 3.
        ("fn main() -> i32 { let c = true; let t = if c { (1, 2) } else { (3, 4) }; return t.0 + t.1; }", 3),
        // if → tuple, else-arm taken: (3,4) -> 3*4 = 12.
        ("fn main() -> i32 { let c = false; let t = if c { (1, 2) } else { (3, 4) }; return t.0 * t.1; }", 12),
        // match → tuple (exhaustive enum): pair(C) = (2,3) -> 2*3 = 6.
        ("enum Op { A, B, C } fn pair(o: Op) -> (i32, i32) { return match o { Op::A => (1, 2), Op::B => (3, 1), Op::C => (2, 3) }; } \
          fn main() -> i32 { let m = pair(Op::C); return m.0 * m.1; }", 6),
        // if → struct, else-arm taken: P{5,7} -> 5+7 = 12.
        ("struct P { x: i32, y: i32 } \
          fn main() -> i32 { let c = false; let p = if c { P { x: 1, y: 2 } } else { P { x: 5, y: 7 } }; return p.x + p.y; }", 12),
        // match → struct used in arithmetic (the dead fallthrough is an aggregate merge param).
        ("struct V { a: i32, b: i32 } enum K { L, R } \
          fn mk(k: K) -> V { return match k { K::L => V { a: 2, b: 3 }, K::R => V { a: 4, b: 5 } }; } \
          fn main() -> i32 { let v = mk(K::R); return v.a * 10 + v.b; }", 45),
    ];
    for (src, want) in cases {
        for opt in [0u8, 1, 2, 3] {
            let n = jit(src, opt).expect("jit");
            let i = interp(src, opt).expect("interp");
            assert_eq!(n, i, "aggregate-value-merge native vs interp mismatch at -O{opt} for:\n{src}");
            assert_eq!(n.0, want, "aggregate-value-merge wrong value at -O{opt} for:\n{src}");
        }
    }
}

/// Tuple-scrutinee `match`: each field's sub-pattern is tested (literals compare, `_`/identifiers
/// match anything, nested tuples recurse) and identifier sub-patterns bind to the tuple's fields.
/// Regression guard — a tuple pattern was previously treated as always-matching, a *silent*
/// miscompile both backends shared (so the differential gate couldn't catch it). Pins the real
/// per-field semantics, including a binding arm, a tuple-pattern guard, a nested pattern, and bools.
#[test]
fn differential_tuple_match() {
    const F: &str = "fn f(a: i32, b: i32) -> i32 { return match (a, b) \
        { (0, 0) => 1, (0, _) => 2, (x, y) => x + y }; } fn main() -> i32 { return f";
    const N: &str = "fn f(a: i32, b: i32, c: i32) -> i32 { return match ((a, b), c) \
        { ((0, 0), 0) => 1, ((0, y), _) => y, ((x, _), z) => x + z }; } fn main() -> i32 { return f";
    let cases = [
        (format!("{F}(0, 0); }}"), 1),
        (format!("{F}(0, 5); }}"), 2),
        (format!("{F}(7, 3); }}"), 10),
        (format!("{F}(4, 0); }}"), 4),
        (format!("{N}(0, 9, 1); }}"), 9),
        (format!("{N}(3, 4, 5); }}"), 8),
        // a tuple-pattern guard (the binding precedes the guard, so the guard reads it).
        ("fn f(a: i32, b: i32) -> i32 { return match (a, b) { (x, y) if x > y => 1, _ => 0 }; } \
          fn main() -> i32 { return f(5, 2); }".to_string(), 1),
        ("fn f(a: i32, b: i32) -> i32 { return match (a, b) { (x, y) if x > y => 1, _ => 0 }; } \
          fn main() -> i32 { return f(2, 5); }".to_string(), 0),
        // bool fields.
        ("fn f(a: bool, b: bool) -> i32 { return match (a, b) { (true, _) => 1, (_, true) => 2, _ => 3 }; } \
          fn main() -> i32 { return f(false, true); }".to_string(), 2),
    ];
    for (src, want) in &cases {
        for opt in [0u8, 1, 2, 3] {
            let n = jit(src, opt).expect("jit");
            let i = interp(src, opt).expect("interp");
            assert_eq!(n, i, "tuple-match native vs interp mismatch at -O{opt} for:\n{src}");
            assert_eq!(n.0, *want, "tuple-match wrong value at -O{opt} for:\n{src}");
        }
    }
}

/// Char literals lower to their Unicode scalar value (sema types a char as `u32`, which MIR carries
/// as `i32`). Covers ASCII, the one-character escapes, `\xHH` hex, `\u{…}` Unicode, ordering, and
/// arithmetic. The value is a compile-time constant, so native == interp at every `-O`.
#[test]
fn differential_char_literals() {
    let cases = [
        ("fn main() -> i32 { return 'A' as i32; }", 65),
        ("fn main() -> i32 { return '0' as i32; }", 48),
        ("fn main() -> i32 { return '\\n' as i32; }", 10),
        ("fn main() -> i32 { return '\\t' as i32; }", 9),
        ("fn main() -> i32 { return '\\\\' as i32; }", 92),
        ("fn main() -> i32 { return '\\'' as i32; }", 39),
        ("fn main() -> i32 { return '\\0' as i32; }", 0),
        ("fn main() -> i32 { return '\\x41' as i32; }", 65),
        ("fn main() -> i32 { return '\\u{1F600}' as i32; }", 128512),
        ("fn main() -> i32 { let z = 'Z'; return z as i32; }", 90),
        ("fn main() -> i32 { if 'a' < 'b' { return 1; } return 0; }", 1),
        ("fn main() -> i32 { return ('z' as i32) - ('a' as i32); }", 25),
    ];
    for (src, want) in cases {
        for opt in [0u8, 1, 2, 3] {
            let n = jit(src, opt).expect("jit");
            let i = interp(src, opt).expect("interp");
            assert_eq!(n, i, "char native vs interp mismatch at -O{opt} for:\n{src}");
            assert_eq!(n.0, want, "char wrong value at -O{opt} for:\n{src}");
        }
    }
}

/// A `*u8` string argument to `print`/`println` renders its bytes; native (`rt_print_str` over the
/// materialized NUL-terminated buffer) must produce byte-identical stdout to the interpreter (which
/// walks its slot memory). Covers a literal argument, a `let`-bound string, the `\t`/`\n`/`\\`/`\"`
/// escapes, an empty string, a multi-byte `\u{…}`, and strings interleaved with numeric prints (so
/// the type-directed dispatch is exercised both ways). `assert_eq!(n, i)` compares exit code AND the
/// whole stdout buffer.
#[test]
fn differential_string_literals() {
    let programs = [
        "fn main() -> i32 { println(\"hello\"); return 0; }",
        // a let-bound string, then printed
        "fn main() -> i32 { let s = \"world\"; print(s); return 0; }",
        // escapes: tab, newline-in-string, backslash, quote
        "fn main() -> i32 { print(\"a\\tb\"); print(\"x\\\\y\"); print(\"q\\\"r\"); return 0; }",
        // empty string prints just a newline
        "fn main() -> i32 { print(\"\"); print(\"after\"); return 0; }",
        // multi-byte UTF-8 via \u{…}
        "fn main() -> i32 { println(\"caf\\u{e9}\"); return 0; }",
        // interleave string and numeric prints — type-directed dispatch both ways
        "fn main() -> i32 { print(\"n=\"); print(42); print(\"done\"); return 0; }",
        // a string returned through a helper that takes/returns *u8
        "fn id(p: *u8) -> *u8 { return p; } \
         fn main() -> i32 { print(id(\"piped\")); return 0; }",
    ];
    for src in programs {
        for opt in [0u8, 1, 2, 3] {
            let n = jit(src, opt).expect("jit");
            let i = interp(src, opt).expect("interp");
            assert_eq!(n, i, "string native vs interp mismatch at -O{opt} for:\n{src}");
        }
    }
}

/// A `return`/tail value must be coerced to the function's declared return type, or `fn f() -> i64
/// { return 0; }` emits a `Ret` of the `i32` literal `0` from an `i64` function — MIR the native
/// verifier and `mem2reg` reject (a panic / exit-1 at -O2 and natively) while the interpreter
/// silently runs it. Both the explicit `return` and the implicit tail are covered; native must agree
/// with interp at every opt level (the previous corpus only ever returned type-matched literals).
#[test]
fn differential_return_type_coercion() {
    let cases = [
        // int literal from a wider return type (stays 0).
        ("fn f() -> i64 { return 0; } fn main() -> i64 { return f(); }", 0i64),
        // implicit tail value coerced to the return type.
        ("fn g() -> i64 { 5 } fn main() -> i64 { return g(); }", 5),
        // negative i32 literal sign-extends to i64.
        (
            "fn n() -> i64 { return 0 - 1; } fn main() -> i64 { return n(); }",
            -1,
        ),
        // float literal promotes f32 -> f64 (1.5 is exact), then *4 = 6.
        (
            "fn h() -> f64 { return 1.5; } fn main() -> i64 { return (h() * 4.0) as i64; }",
            6,
        ),
        // narrow (u8) return type.
        ("fn b() -> u8 { return 5; } fn main() -> i64 { return b() as i64; }", 5),
        // a recursive i64 function whose base case is `return 0;` (the field pattern the hunt hit).
        (
            "fn sum(n: i32) -> i64 { if n <= 0 { return 0; } return (n as i64) + sum(n - 1); } \
             fn main() -> i64 { return sum(5); }",
            15,
        ),
    ];
    for (src, want) in cases {
        for opt in [0u8, 1, 2, 3] {
            let n = jit(src, opt).expect("jit");
            let i = interp(src, opt).expect("interp");
            assert_eq!(n, i, "return-coercion native vs interp mismatch at -O{opt} for:\n{src}");
            assert_eq!(n.0, want, "return-coercion wrong value at -O{opt} for:\n{src}");
        }
    }
}

/// Constant folding must stay at f32 precision for an f32 chain. Folding in f64 (and only narrowing
/// at the final store) makes `-O2` disagree with `-O0` — `(2^24 + 1) - 2^24` is `0` in f32 but `1`
/// in f64, and `0.1 + 0.2 == 0.3` is true in f32 but false in f64. Each (backend, opt) pair must
/// agree AND equal the f32-correct value, which also pins `-O0 == -O2`.
#[test]
fn differential_f32_const_fold() {
    let cases = [
        (
            "fn main() -> i32 { let b: f32 = 16777216.0; let r: f32 = (b + 1.0) - b; return r as i32; }",
            0i64,
        ),
        (
            "fn main() -> i32 { if 0.1 + 0.2 == 0.3 { return 1; } return 0; }",
            1,
        ),
        (
            "fn main() -> i64 { let m: f32 = (0.1 + 0.2) * 100000000.0; return m as i64; }",
            30000002,
        ),
    ];
    for (src, want) in cases {
        for opt in [0u8, 1, 2, 3] {
            let n = jit(src, opt).expect("jit");
            let i = interp(src, opt).expect("interp");
            assert_eq!(n, i, "f32-fold native vs interp mismatch at -O{opt} for:\n{src}");
            assert_eq!(n.0, want, "f32-fold wrong value at -O{opt} for:\n{src}");
        }
    }
}

/// Integer → `f32` conversion must round in a single IEEE step. The interpreter used to go int→f64→
/// f32 (two roundings) while native does one `fcvt_from_{sint,uint}(F32)`, so they disagreed for
/// magnitudes above 2^53 — the differential oracle was silently wrong. Pin both to the correctly
/// rounded value (`9007199791611905 as f32 == 9007200328482816`), for signed and unsigned sources.
#[test]
fn differential_int_to_f32_rounding() {
    let cases = [
        // 2^53 + 2^29 + 1, just past where f32 (and the double-round) diverge.
        ("fn main() -> i64 { let n: i64 = 9007199791611905; let f: f32 = n as f32; return f as i64; }",
         9007200328482816i64),
        ("fn main() -> i64 { let n: u64 = 9007199791611905; let f: f32 = n as f32; return f as i64; }",
         9007200328482816),
        // a small value is exact and unchanged.
        ("fn main() -> i64 { let n: i64 = 1234567; let f: f32 = n as f32; return f as i64; }",
         1234567),
    ];
    for (src, want) in cases {
        for opt in [0u8, 1, 2, 3] {
            let n = jit(src, opt).expect("jit");
            let i = interp(src, opt).expect("interp");
            assert_eq!(n, i, "int->f32 native vs interp mismatch at -O{opt} for:\n{src}");
            assert_eq!(n.0, want, "int->f32 wrong value at -O{opt} for:\n{src}");
        }
    }
}

/// Pointers/references: `&mut x` takes an address, `*p` loads/stores through it, and a pointer
/// threads through a function call. An address-taken local must stay in memory (mem2reg refuses to
/// promote a slot whose address escapes), so native == interp at every `-O`. Also pins the
/// cast-precedence fix: `*p as T` is `(*p) as T`, not `*(p as T)`.
#[test]
fn differential_pointer() {
    let programs = [
        // &mut + store through pointer + read back.
        "fn main() -> i32 { let mut x: i32 = 3; let p: *mut i32 = &mut x; *p = 7; return *p; }",
        // pointer threaded through a call mutates the caller's local.
        "fn setit(p: *mut i32, v: i32) { *p = v; } \
         fn main() -> i32 { let mut n: i32 = 0; setit(&mut n, 99); return n; }",
        // f32 through a pointer + the cast-precedence case `(*pf) as i32`.
        "fn main() -> i32 { let mut x: f32 = 3.0; let pf: *mut f32 = &mut x; *pf = 7.5; \
         print(*pf as i32); return (*pf as i32) - 7; }",
    ];
    for src in programs {
        for opt in [0u8, 1, 2, 3] {
            let n = jit(src, opt).expect("jit");
            let i = interp(src, opt).expect("interp");
            assert_eq!(n, i, "pointer native vs interp mismatch at -O{opt} for:\n{src}");
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
        // GLU 2-array gate `out[j] = x[j] * sigmoid(g[j])` (distinct arrays) -> VM2_SIGMOID_GATE:
        // native and interp both marshal the identical wukong_vmath2_f32 kernel, so they must agree.
        "fn main() -> i32 { let mut x: [f32; 32] = [0.0; 32]; let mut g: [f32; 32] = [0.0; 32]; \
         for i in 0..32 { x[i] = (i as f32) * 0.5 - 8.0; g[i] = (i as f32) * 0.25 - 4.0; } \
         let mut o: [f32; 32] = [0.0; 32]; for j in 0..32 { o[j] = x[j] * sigmoid(g[j]); } \
         print((o[3] * 1000.0) as i32); print((o[20] * 1000.0) as i32); return 0; }",
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
/// 256-bit AVX2 `wukong_vmath_f32` kernel (the width Cranelift can't emit). The interpreter marshals
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
    let src = "@parallel fn act(x: [f32; 4096], mut out: [f32; 4096]) { \
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
/// kernel (`wukong_i8gemm_nt`; the `@parallel` form to `_parallel`). Integer arithmetic is exact and
/// order-independent (associative mod 2³²), so the fused kernel equals the scalar nest bit-for-bit —
/// native and interp must agree at every opt level. `K = 40` exercises the AVX2 32/16-wide chunks and
/// the scalar tail; inputs span the full `u8`/`i8` ranges.
#[test]
fn differential_i8gemm() {
    let body = |attr: &str| {
        format!(
            "{attr}fn lin(a: [u8; 160], b: [i8; 160], mut c: [i32; 16]) {{ \
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

/// The int8 `nn.Linear` nest dispatches to `wukong_i8gemm_nt` (the `@parallel` whole-function form
/// to `_parallel`, checked before the @parallel outliner so it reaches the multicore kernel rather
/// than a scalar loop). The signedness guard is correctness-critical (the kernel zero-extends A,
/// sign-extends B): an `i8×i8` nest must NOT pick the `u8×i8` kernel.
#[test]
fn i8_linear_nest_lowers_to_i8gemm() {
    let lin = |attr: &str| {
        format!(
            "module m\n{attr}fn lin(a:[u8;160],b:[i8;160],mut c:[i32;16]) {{ \
             for i in 0..4 {{ for j in 0..4 {{ let mut s: i32 = 0; \
             for k in 0..40 {{ s = s + (a[i*40+k] as i32) * (b[j*40+k] as i32); }} \
             c[i*4+j] = s; }} }} }}"
        )
    };
    assert!(
        lowered_calls(&lin(""), "wukong_i8gemm_nt"),
        "u8×i8 nn.Linear -> i8gemm_nt"
    );
    assert!(
        lowered_calls(&lin("@parallel\n"), "wukong_i8gemm_nt_parallel"),
        "@parallel u8×i8 nn.Linear -> i8gemm_nt_parallel"
    );
    // Signedness mismatch (A is i8, not u8): the kernel's zero/sign-extend split would miscompile, so
    // the recognizer must bail and never emit either int8 kernel symbol.
    let signed = "module m\nfn lin(a:[i8;160],b:[i8;160],mut c:[i32;16]) { \
        for i in 0..4 { for j in 0..4 { let mut s: i32 = 0; \
        for k in 0..40 { s = s + (a[i*40+k] as i32) * (b[j*40+k] as i32); } \
        c[i*4+j] = s; } } }";
    assert!(
        !lowered_calls(signed, "wukong_i8gemm_nt"),
        "i8×i8 must not pick the u8×i8 kernel"
    );
    assert!(
        !lowered_calls(signed, "wukong_i8gemm_nt_parallel"),
        "i8×i8 must not pick the parallel u8×i8 kernel"
    );
}

/// An int-input dequant `out[j] = act((q[j] as f32)·scale)` over an `[i8]`/`[u8]`/`[i32]` array
/// dispatches to `wukong_dequant_f32` (256-bit widen+scale). Integer→f32 is exact, so the kernel
/// equals the scalar loop bit-for-bit — native and interp must agree at every opt level, for every
/// input width, activation (identity / `fmax` ReLU / GELU / SiLU), and the serial *and* `@parallel`
/// forms. The width sweep pins signedness (`i8` sign-extends, `u8` zero-extends): the fill spans past
/// the byte boundary in both directions, so a wrong extension would diverge — but *consistently* in
/// both backends (gate-blind), which is why the hand-computed golden fixture `tests/run/dequant*.wk`
/// is the independent signedness gate; this pins the two backends to each other across the matrix.
#[test]
fn differential_dequant() {
    // `V` is the dequant value `(q[j] as f32)·s`; each `act` wraps it.
    let make = |attr: &str, ty: &str, act: &str| {
        let val = "((q[j] as f32) * s)";
        let body = act.replace('V', val);
        format!(
            "{attr}fn deq(q: [{ty}; 64], mut out: [f32; 64]) {{ let s: f32 = 0.0125; \
             for j in 0..64 {{ out[j] = {body}; }} }} \
             fn main() -> i32 {{ let mut q: [{ty}; 64] = [0 as {ty}; 64]; \
             let mut out: [f32; 64] = [0.0; 64]; \
             for j in 0..64 {{ q[j] = ((j * 37 + 5) % 251 - 100) as {ty}; }} \
             deq(q, out); \
             let mut acc: f32 = 0.0; for j in 0..64 {{ acc = acc + out[j]; }} \
             print((acc * 1000.0) as i32); print((out[3] * 1000.0) as i32); \
             print((out[63] * 1000.0) as i32); return 0; }}"
        )
    };
    for ty in ["i8", "u8", "i32"] {
        for act in ["V", "fmax(V, 0.0)", "gelu(V)", "silu(V)"] {
            for attr in ["", "@parallel "] {
                let src = make(attr, ty, act);
                for opt in [0u8, 2, 3] {
                    let n = jit(&src, opt).expect("jit");
                    let i = interp(&src, opt).expect("interp");
                    assert_eq!(
                        n, i,
                        "dequant native vs interp mismatch ty={ty} act={act} attr={attr:?} at -O{opt}\n{src}"
                    );
                }
            }
        }
    }
}

/// An int-input dequant loop dispatches to `wukong_dequant_f32` for every supported width and
/// activation (the `@parallel` form reaches it via the outliner's per-chunk `try_vectorize_ranged`).
/// An f32-array elementwise map must NOT reach the dequant kernel (that is velem's job — dequant reads
/// an *int* array through an `as f32` cast, which velem declines and vice-versa).
#[test]
fn dequant_loop_lowers_to_dequant_kernel() {
    let deq = |attr: &str, ty: &str, act: &str| {
        format!(
            "module m\n{attr}fn d(q: [{ty}; 64], mut out: [f32; 64]) {{ let s: f32 = 0.1; \
             for j in 0..64 {{ out[j] = {act}; }} }}"
        )
    };
    for ty in ["i8", "u8", "i32"] {
        assert!(
            lowered_calls(&deq("", ty, "(q[j] as f32) * s"), "wukong_dequant_f32"),
            "{ty} pure dequant -> wukong_dequant_f32"
        );
    }
    for act in ["fmax((q[j] as f32) * s, 0.0)", "gelu((q[j] as f32) * s)", "silu((q[j] as f32) * s)"] {
        assert!(
            lowered_calls(&deq("", "i8", act), "wukong_dequant_f32"),
            "activated dequant `{act}` -> wukong_dequant_f32"
        );
    }
    // A `@parallel` dequant whose body has a leading `let s` (so it is not a single-loop body the
    // outliner claims) is lowered `parallel_fn = true` and dispatches to the rayon parallel kernel.
    assert!(
        lowered_calls(
            &deq("@parallel\n", "i32", "(q[j] as f32) * s"),
            "wukong_dequant_f32_parallel"
        ),
        "@parallel dequant (mixed body) -> wukong_dequant_f32_parallel"
    );
    // An f32-array scale map is velem's, not dequant's — the dequant kernel must not claim it.
    let f32_map = "module m\nfn d(x: [f32; 64], mut out: [f32; 64]) { let s: f32 = 0.1; \
                   for j in 0..64 { out[j] = x[j] * s; } }";
    assert!(
        !lowered_calls(f32_map, "wukong_dequant_f32"),
        "f32 elementwise map must not reach the dequant kernel"
    );
}

/// A *mixed* `@parallel` function (not a single elementwise loop, so the outliner declines it and it is
/// lowered with `parallel_fn = true`) dispatches its dequant loops to the rayon
/// `wukong_dequant_f32_parallel` — bit-identical to serial (elementwise), so native and interp agree.
/// This exercises the parallel-symbol emission path the outlined common case skips.
#[test]
fn differential_dequant_mixed_parallel() {
    // Two dequant loops in one @parallel fn → not a single-loop body → the parallel-symbol path.
    let src = "@parallel fn deq2(q: [i32; 128], mut out: [f32; 128]) { let s: f32 = 0.03125; \
         for j in 0..64 { out[j] = (q[j] as f32) * s; } \
         for j in 64..128 { out[j] = fmax((q[j] as f32) * s, 0.0); } } \
         fn main() -> i32 { let mut q: [i32; 128] = [0; 128]; let mut out: [f32; 128] = [0.0; 128]; \
         for j in 0..128 { q[j] = (j - 64) * 3; } deq2(q, out); \
         let mut acc: f32 = 0.0; for j in 0..128 { acc = acc + out[j]; } \
         print((acc * 100.0) as i32); print((out[0] * 100.0) as i32); \
         print((out[127] * 100.0) as i32); return 0; }";
    assert!(
        lowered_calls(src, "wukong_dequant_f32_parallel"),
        "mixed @parallel dequant -> wukong_dequant_f32_parallel"
    );
    for opt in [0u8, 2, 3] {
        let n = jit(src, opt).expect("jit");
        let i = interp(src, opt).expect("interp");
        assert_eq!(n, i, "mixed-parallel dequant native vs interp mismatch at -O{opt}");
    }
}

/// A per-channel dequant nest `out[i*C+j] = act((q[i*C+j] as f32)·scale[j])` over a `[R,C]` matrix
/// dispatches to `wukong_dequant_perchan_f32` (the `@parallel` whole-function form to `_parallel`,
/// intercepted before the outliner so rows map across cores rather than being scalarized). Integer→f32
/// is exact, so native and interp agree at every opt level, for every input width, activation, and the
/// serial *and* `@parallel` forms. `C = 12` (not a multiple of 8) exercises the kernel's column tail.
#[test]
fn differential_dequant_perchan() {
    let make = |attr: &str, ty: &str, act: &str| {
        let val = "((q[i * 12 + j] as f32) * scale[j])";
        let body = act.replace('V', val);
        format!(
            "{attr}fn deq(q: [{ty}; 60], scale: [f32; 12], mut out: [f32; 60]) {{ \
             for i in 0..5 {{ for j in 0..12 {{ out[i * 12 + j] = {body}; }} }} }} \
             fn main() -> i32 {{ let mut q: [{ty}; 60] = [0 as {ty}; 60]; \
             let mut scale: [f32; 12] = [0.0; 12]; let mut out: [f32; 60] = [0.0; 60]; \
             for j in 0..12 {{ scale[j] = ((j % 4) as f32) * 0.01 + 0.005; }} \
             for t in 0..60 {{ q[t] = ((t * 29 + 7) % 233 - 110) as {ty}; }} \
             deq(q, scale, out); \
             let mut acc: f32 = 0.0; for t in 0..60 {{ acc = acc + out[t]; }} \
             print((acc * 10000.0) as i32); print((out[13] * 10000.0) as i32); \
             print((out[59] * 10000.0) as i32); return 0; }}"
        )
    };
    // Dispatch: serial and @parallel both reach the (parallel) kernel.
    assert!(
        lowered_calls(&make("", "i32", "V"), "wukong_dequant_perchan_f32"),
        "per-channel dequant -> wukong_dequant_perchan_f32"
    );
    assert!(
        lowered_calls(&make("@parallel ", "i8", "V"), "wukong_dequant_perchan_f32_parallel"),
        "@parallel per-channel dequant -> wukong_dequant_perchan_f32_parallel"
    );
    for ty in ["i8", "u8", "i32"] {
        for act in ["V", "fmax(V, 0.0)", "gelu(V)", "silu(V)"] {
            for attr in ["", "@parallel "] {
                let src = make(attr, ty, act);
                for opt in [0u8, 2, 3] {
                    let n = jit(&src, opt).expect("jit");
                    let i = interp(&src, opt).expect("interp");
                    assert_eq!(
                        n, i,
                        "per-channel dequant native vs interp mismatch ty={ty} act={act} attr={attr:?} at -O{opt}\n{src}"
                    );
                }
            }
        }
    }
}

/// The embedding lookup `out[t,:] = weight[ids[t],:]` (the first layer of every LLM) dispatches to
/// `wukong_embedding_f32`. The native run gathers via the AVX2 kernel; the interpreter marshals the
/// identical serial kernel — pure data movement (a row copy), so they are bit-identical by
/// construction. Covers the serial form and a large-T `@parallel` form (T past the kernel's multicore
/// threshold so the rayon row-split actually runs, still bit-equal to serial).
#[test]
fn differential_embedding() {
    // Small serial gather: T=4, V=4, H=3, ids hitting row 0, the last row, and a repeat.
    let serial = "fn embed(ids: [i32; 4], weight: [f32; 12], mut out: [f32; 12]) { \
         for t in 0..4 { for d in 0..3 { out[t * 3 + d] = weight[ids[t] * 3 + d]; } } } \
         fn main() -> i32 { let weight: [f32; 12] = [0.0,1.0,2.0,10.0,11.0,12.0,\
         20.0,21.0,22.0,30.0,31.0,32.0]; let ids: [i32; 4] = [2,0,3,0]; \
         let mut out: [f32; 12] = [0.0; 12]; embed(ids, weight, out); \
         let mut acc: f32 = 0.0; for i in 0..12 { acc = acc + out[i]; } \
         print(acc); print(out[0]); print(out[11]); return 0; }"
        .to_string();
    // Large @parallel gather: T=80 (> EMBEDDING_PAR_MIN=64, so the multicore split runs), V=8, H=4.
    let parallel = "@parallel\nfn embed(ids: [i32; 80], weight: [f32; 32], mut out: [f32; 320]) { \
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

/// The embedding nest must lower to `wukong_embedding_f32` (the `@parallel` whole-function form to
/// `_parallel`, intercepted before the outliner so it reaches the multicore kernel). A nest whose
/// load is NOT a data-dependent gather — `weight[t * H + d]` with the *outer var* as the row, i.e. a
/// plain elementwise copy — must NOT pick the embedding kernel (no `ids[t]` indirection).
#[test]
fn embedding_nest_lowers_to_kernel() {
    let embed = |attr: &str| {
        format!(
            "module m\n{attr}fn embed(ids: [i32; 4], weight: [f32; 12], mut out: [f32; 12]) {{ \
             for t in 0..4 {{ for d in 0..3 {{ out[t * 3 + d] = weight[ids[t] * 3 + d]; }} }} }}"
        )
    };
    assert!(
        lowered_calls(&embed(""), "wukong_embedding_f32"),
        "embedding nest -> wukong_embedding_f32"
    );
    assert!(
        lowered_calls(&embed("@parallel\n"), "wukong_embedding_f32_parallel"),
        "@parallel embedding nest -> wukong_embedding_f32_parallel"
    );
    // A direct copy `out[t*3+d] = weight[t*3+d]` (the row index is the loop var, not a gathered id) is
    // not an embedding lookup — the recognizer must bail (no `ids[t]` indirection to fold).
    let copy = "module m\nfn cp(ids: [i32; 4], weight: [f32; 12], mut out: [f32; 12]) { \
        for t in 0..4 { for d in 0..3 { out[t * 3 + d] = weight[t * 3 + d]; } } }";
    assert!(
        !lowered_calls(copy, "wukong_embedding_f32"),
        "a plain elementwise copy must not pick the embedding kernel"
    );
}

/// A `@parallel` reduction (`s += f(x[k], y[k])`, `m = fmax(m, x[k])`) dispatches to the multicore
/// reduction kernel (`wukong_sreduce_f32_parallel`). The native run folds across cores; the
/// interpreter calls the *serial* kernel — both are bit-identical by construction (fixed chunking,
/// ascending combine), so native and interp must agree at every opt level. Covers dot, ssd, the
/// unary sum, the running max/min (the per-tensor max/absmax for softmax / int8 quantization), and
/// the L1 folds abssum `Σ|x|` (RED_SUMABS) / MAE `Σ|x−y|` (RED_ABSDIFF).
#[test]
fn differential_parallel_reduce() {
    let programs = [
        // dot product Σ x·y
        "@parallel fn dotp(x: [f32; 4096], y: [f32; 4096], mut o: [f32; 1]) { \
         let mut s: f32 = 0.0; for k in 0..4096 { s = s + x[k] * y[k]; } o[0] = s; } \
         fn main() -> i32 { let mut x: [f32; 4096] = [0.0; 4096]; \
         let mut y: [f32; 4096] = [0.0; 4096]; let mut o: [f32; 1] = [0.0; 1]; \
         for i in 0..4096 { x[i] = (i as f32) * 0.001; y[i] = 2.0; } dotp(x, y, o); \
         print((o[0] * 100.0) as i32); return 0; }",
        // sum of squared differences Σ (x−y)² (an L2 loss)
        "@parallel fn ssd(x: [f32; 4096], y: [f32; 4096], mut o: [f32; 1]) { \
         let mut s: f32 = 0.0; for k in 0..4096 { s += (x[k] - y[k]) * (x[k] - y[k]); } o[0] = s; } \
         fn main() -> i32 { let mut x: [f32; 4096] = [0.0; 4096]; \
         let mut y: [f32; 4096] = [0.0; 4096]; let mut o: [f32; 1] = [0.0; 1]; \
         for i in 0..4096 { x[i] = (i as f32) * 0.001; y[i] = 1.0; } ssd(x, y, o); \
         print((o[0] * 10.0) as i32); return 0; }",
        // unary sum Σ x (LayerNorm-style accumulation; the recognizer passes y == x)
        "@parallel fn sumv(x: [f32; 4096], mut o: [f32; 1]) { \
         let mut s: f32 = 0.0; for k in 0..4096 { s += x[k]; } o[0] = s; } \
         fn main() -> i32 { let mut x: [f32; 4096] = [0.0; 4096]; \
         let mut o: [f32; 1] = [0.0; 1]; \
         for i in 0..4096 { x[i] = (i as f32) * 0.01; } sumv(x, o); \
         print((o[0]) as i32); return 0; }",
        // running max (fold by fmax; mixed-sign fractional inputs)
        "@parallel fn maxv(x: [f32; 4096], mut o: [f32; 1]) { \
         let mut m: f32 = x[0]; for k in 0..4096 { m = fmax(m, x[k]); } o[0] = m; } \
         fn main() -> i32 { let mut x: [f32; 4096] = [0.0; 4096]; \
         let mut o: [f32; 1] = [0.0; 1]; \
         for i in 0..4096 { x[i] = (i as f32) * 0.001 - 2.0; } maxv(x, o); \
         print((o[0] * 1000.0) as i32); return 0; }",
        // running min (fold by fmin, operand order m second)
        "@parallel fn minv(x: [f32; 4096], mut o: [f32; 1]) { \
         let mut m: f32 = x[0]; for k in 0..4096 { m = fmin(x[k], m); } o[0] = m; } \
         fn main() -> i32 { let mut x: [f32; 4096] = [0.0; 4096]; \
         let mut o: [f32; 1] = [0.0; 1]; \
         for i in 0..4096 { x[i] = 5.0 - (i as f32) * 0.001; } minv(x, o); \
         print((o[0] * 1000.0) as i32); return 0; }",
        // running absmax (fmax fold over abs(x[k]) → RED_MAXABS; negative-dominant tail)
        "@parallel fn absmaxv(x: [f32; 4096], mut o: [f32; 1]) { \
         let mut m: f32 = 0.0; for k in 0..4096 { m = fmax(m, abs(x[k])); } o[0] = m; } \
         fn main() -> i32 { let mut x: [f32; 4096] = [0.0; 4096]; \
         let mut o: [f32; 1] = [0.0; 1]; \
         for i in 0..4096 { x[i] = 3.0 - (i as f32) * 0.002; } absmaxv(x, o); \
         print((o[0] * 1000.0) as i32); return 0; }",
        // abssum Σ|x| (L1 norm → RED_SUMABS, y == x; ramp straddling zero exercises the sign clear)
        "@parallel fn abssumv(x: [f32; 4096], mut o: [f32; 1]) { \
         let mut s: f32 = 0.0; for k in 0..4096 { s += abs(x[k]); } o[0] = s; } \
         fn main() -> i32 { let mut x: [f32; 4096] = [0.0; 4096]; \
         let mut o: [f32; 1] = [0.0; 1]; \
         for i in 0..4096 { x[i] = (i as f32) * 0.001 - 2.0; } abssumv(x, o); \
         print((o[0] * 100.0) as i32); return 0; }",
        // MAE Σ|x−y| (two-array L1 distance → RED_ABSDIFF, like ssd but abs not square)
        "@parallel fn maev(x: [f32; 4096], y: [f32; 4096], mut o: [f32; 1]) { \
         let mut s: f32 = 0.0; for k in 0..4096 { s += abs(x[k] - y[k]); } o[0] = s; } \
         fn main() -> i32 { let mut x: [f32; 4096] = [0.0; 4096]; \
         let mut y: [f32; 4096] = [0.0; 4096]; let mut o: [f32; 1] = [0.0; 1]; \
         for i in 0..4096 { x[i] = (i as f32) * 0.001; y[i] = 1.0; } maev(x, y, o); \
         print((o[0] * 10.0) as i32); return 0; }",
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
    let golden = "@parallel fn dotp(x: [f32; 4096], y: [f32; 4096], mut o: [f32; 1]) { \
         let mut s: f32 = 0.0; for k in 0..4096 { s = s + x[k] * y[k]; } o[0] = s; } \
         @parallel fn sumv(x: [f32; 4096], mut o: [f32; 1]) { \
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
    let golden_mm = "@parallel fn maxv(x: [f32; 4096], mut o: [f32; 1]) { \
         let mut m: f32 = x[0]; for k in 0..4096 { m = fmax(m, x[k]); } o[0] = m; } \
         @parallel fn minv(x: [f32; 4096], mut o: [f32; 1]) { \
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
    let golden_abs = "@parallel fn absmaxv(x: [f32; 4096], mut o: [f32; 1]) { \
         let mut m: f32 = 0.0; for k in 0..4096 { m = fmax(m, abs(x[k])); } o[0] = m; } \
         fn main() -> i32 { let mut x: [f32; 4096] = [0.0; 4096]; let mut o: [f32; 1] = [0.0; 1]; \
         for i in 0..4096 { x[i] = (i as f32) - 3000.0; } absmaxv(x, o); print((o[0]) as i32); return 0; }";
    let (_, out) = jit(golden_abs, 3).expect("jit golden_abs");
    assert_eq!(
        String::from_utf8(out).unwrap(),
        "3000\n",
        "parallel absmax produced the wrong value"
    );
    // Abssum / MAE golden (constant arrays, exact): Σ|−2| = 2·4096 = 8192; Σ|2−5| = 3·4096 = 12288
    // (a negative diff, so it also confirms the sign-bit clear; RED_SSD would give 9·4096 = 36864).
    let golden_l1 = "@parallel fn abssumv(x: [f32; 4096], mut o: [f32; 1]) { \
         let mut s: f32 = 0.0; for k in 0..4096 { s += abs(x[k]); } o[0] = s; } \
         @parallel fn maev(x: [f32; 4096], y: [f32; 4096], mut o: [f32; 1]) { \
         let mut s: f32 = 0.0; for k in 0..4096 { s += abs(x[k] - y[k]); } o[0] = s; } \
         fn main() -> i32 { let mut x: [f32; 4096] = [0.0; 4096]; let mut o: [f32; 1] = [0.0; 1]; \
         for i in 0..4096 { x[i] = 0.0 - 2.0; } abssumv(x, o); print((o[0]) as i32); \
         let mut a: [f32; 4096] = [2.0; 4096]; let mut b: [f32; 4096] = [5.0; 4096]; \
         maev(a, b, o); print((o[0]) as i32); return 0; }";
    let (_, out) = jit(golden_l1, 3).expect("jit golden_l1");
    assert_eq!(
        String::from_utf8(out).unwrap(),
        "8192\n12288\n",
        "parallel abssum/MAE produced the wrong value"
    );
}

/// bf16 mixed-precision reductions (`wukong_dot_bf16` / `wukong_sum_bf16`): a `s += (x[k] as f32)
/// [* (y[k] as f32)]` loop over `[bf16; _]` arrays with an f32 accumulator. The native backend rounds
/// to bf16 on store (`round_to_bf16`, the identical integer arithmetic as the interpreter's
/// `round_bf16`) and both call the identical reassociated kernel, so they must agree bit-for-bit even
/// for *non*-bf16-exact, fractional inputs that genuinely exercise the rounding and the 8-lane sum.
#[test]
fn differential_bf16_reduce() {
    let programs = [
        // bf16 dot Σ (x·y) with fractional, non-bf16-exact elements (real rounding + reassociation).
        "fn dotbf(x: [bf16; 4096], y: [bf16; 4096], mut o: [f32; 1]) { \
         let mut s: f32 = 0.0; for k in 0..4096 { s = s + (x[k] as f32) * (y[k] as f32); } o[0] = s; } \
         fn main() -> i32 { let mut x: [bf16; 4096] = [0.0 as bf16; 4096]; \
         let mut y: [bf16; 4096] = [0.0 as bf16; 4096]; let mut o: [f32; 1] = [0.0; 1]; \
         for i in 0..4096 { x[i] = ((i as f32) * 0.001) as bf16; y[i] = 1.5 as bf16; } dotbf(x, y, o); \
         print((o[0] * 100.0) as i32); return 0; }",
        // bf16 unary sum Σ x with fractional elements.
        "fn sumbf(x: [bf16; 4096], mut o: [f32; 1]) { \
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
    let golden = "fn dotbf(x: [bf16; 8], y: [bf16; 8], mut o: [f32; 1]) { \
         let mut s: f32 = 0.0; for k in 0..8 { s = s + (x[k] as f32) * (y[k] as f32); } o[0] = s; } \
         fn sumbf(x: [bf16; 8], mut o: [f32; 1]) { \
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
/// `wukong_axpby_bf16` (bf16 in, f32 out, f32 math) and stay native==interp across opt levels (both
/// marshal the identical kernel). A 1-term scale must NOT dispatch (it would force a `0*y` the source
/// lacks). Golden small-integer case pins the value.
#[test]
fn differential_bf16_axpby() {
    let axpby = "module m\nfn ax(x:[bf16;64], y:[bf16;64], mut o:[f32;64]) { \
        for k in 0..64 { o[k] = 1.5 * (x[k] as f32) + 2.0 * (y[k] as f32); } }";
    assert!(
        lowered_calls(axpby, "wukong_axpby_bf16"),
        "bf16 axpby -> wukong_axpby_bf16"
    );
    let saxpy = "module m\nfn ax(x:[bf16;64], y:[bf16;64], mut o:[f32;64]) { \
        for k in 0..64 { o[k] = 3.0 * (x[k] as f32) + (y[k] as f32); } }";
    assert!(
        lowered_calls(saxpy, "wukong_axpby_bf16"),
        "bf16 saxpy (implicit b=1) -> wukong_axpby_bf16"
    );
    let add = "module m\nfn ax(x:[bf16;64], y:[bf16;64], mut o:[f32;64]) { \
        for k in 0..64 { o[k] = (x[k] as f32) + (y[k] as f32); } }";
    assert!(
        lowered_calls(add, "wukong_axpby_bf16"),
        "bf16 add -> wukong_axpby_bf16"
    );
    // A 1-term scale has no second additive term, so it must decline (avoids a 0*inf the source lacks).
    let scale = "module m\nfn ax(x:[bf16;64], mut o:[f32;64]) { \
        for k in 0..64 { o[k] = 2.0 * (x[k] as f32); } }";
    assert!(
        !lowered_calls(scale, "wukong_axpby_bf16"),
        "1-term scale must not dispatch to the axpby kernel"
    );

    // native == interp across opt levels, fractional non-bf16-exact inputs.
    let prog = "fn ax(x:[bf16;4096], y:[bf16;4096], mut o:[f32;4096]) { \
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
    let golden = "fn ax(x:[bf16;8], y:[bf16;8], mut o:[f32;8]) { \
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

/// The **all-half** streaming axpby `out[k] = (a*(x[k] as f32) + b*(y[k] as f32)) as {bf16,f16}` —
/// half in AND a narrowing half store — must dispatch to `wukong_axpby_{bf16,f16}_out` and stay
/// native==interp across opt levels. The narrowing round goes through the shared `f32_to_*_bits` shim
/// (== a scalar `as bf16`/`as f16` store), so `-O0`==`-O3` and both backends agree bit-for-bit.
#[test]
fn differential_half_out_axpby() {
    // bf16 in, bf16 out -> the narrowing kernel.
    let bf = "module m\nfn ax(x:[bf16;64], y:[bf16;64], mut o:[bf16;64]) { \
        for k in 0..64 { o[k] = (1.5 * (x[k] as f32) + 2.0 * (y[k] as f32)) as bf16; } }";
    assert!(
        lowered_calls(bf, "wukong_axpby_bf16_out"),
        "bf16-in/bf16-out axpby -> wukong_axpby_bf16_out"
    );
    // f16 in, f16 out -> the f16 twin.
    let hf = "module m\nfn ax(x:[f16;64], y:[f16;64], mut o:[f16;64]) { \
        for k in 0..64 { o[k] = (1.5 * (x[k] as f32) + 2.0 * (y[k] as f32)) as f16; } }";
    assert!(
        lowered_calls(hf, "wukong_axpby_f16_out"),
        "f16-in/f16-out axpby -> wukong_axpby_f16_out"
    );
    // A **f32** output must NOT take the narrowing path (it's the plain bf16-in/f32-out kernel).
    let f32out = "module m\nfn ax(x:[bf16;64], y:[bf16;64], mut o:[f32;64]) { \
        for k in 0..64 { o[k] = 1.5 * (x[k] as f32) + 2.0 * (y[k] as f32); } }";
    assert!(
        !lowered_calls(f32out, "wukong_axpby_bf16_out"),
        "f32-output axpby must not take the narrowing store path"
    );

    // native == interp across opt levels, fractional non-half-exact inputs, both halves.
    for (ty, k) in [("bf16", "bf16"), ("f16", "f16")] {
        let prog = format!(
            "fn ax(x:[{ty};4096], y:[{ty};4096], mut o:[{k};4096]) {{ \
             for j in 0..4096 {{ o[j] = (1.5 * (x[j] as f32) + 2.0 * (y[j] as f32)) as {k}; }} }} \
             fn main() -> i32 {{ let mut x:[{ty};4096]=[0.0 as {ty};4096]; \
             let mut y:[{ty};4096]=[0.0 as {ty};4096]; let mut o:[{k};4096]=[0.0 as {k};4096]; \
             for i in 0..4096 {{ x[i]=((i as f32)*0.001) as {ty}; y[i]=((i as f32)*0.002-1.3) as {ty}; }} \
             ax(x,y,o); let mut s:f32=0.0; for t in 0..4096 {{ s = s + (o[t] as f32); }} \
             print((s*10.0) as i32); return 0; }}"
        );
        for opt in [0u8, 2, 3] {
            assert_eq!(
                jit(&prog, opt).expect("jit"),
                interp(&prog, opt).expect("interp"),
                "{ty}-out axpby native vs interp mismatch at -O{opt}"
            );
        }
    }
    // Golden half-exact: o[k] = 2·(k+1) + 3·2 = 2(k+1)+6 (all bf16-exact); Σ_{k=0..7} = 2·36 + 48 = 120.
    let golden = "fn ax(x:[bf16;8], y:[bf16;8], mut o:[bf16;8]) { \
        for k in 0..8 { o[k] = (2.0*(x[k] as f32) + 3.0*(y[k] as f32)) as bf16; } } \
        fn main() -> i32 { let mut x:[bf16;8]=[0.0 as bf16;8]; let mut y:[bf16;8]=[0.0 as bf16;8]; \
        let mut o:[bf16;8]=[0.0 as bf16;8]; for i in 0..8 { x[i]=((i+1) as f32) as bf16; y[i]=2.0 as bf16; } \
        ax(x,y,o); let mut s:f32=0.0; for t in 0..8 { s = s + (o[t] as f32); } print((s) as i32); return 0; }";
    let (_, out) = jit(golden, 3).expect("jit golden");
    assert_eq!(
        String::from_utf8(out).unwrap(),
        "120\n",
        "bf16-out axpby produced the wrong value"
    );
}

/// The **half-output** activation `out[k] = (f((x[k] as f32))) as {bf16,f16}` — half in AND a narrowing
/// half store — must dispatch to `wukong_vmath_{bf16,f16}_out` and stay native==interp across opt
/// levels. The activation is the shared `vmath` kernel and the narrowing store the shared shim, so the
/// half-out result equals the half-in/f32-out activation narrowed with `as {bf16,f16}` bit-for-bit.
#[test]
fn differential_half_out_vmath() {
    // bf16 in, bf16 out -> the narrowing activation kernel.
    let bf = "module m\nfn a(x:[bf16;64], mut o:[bf16;64]) { \
        for k in 0..64 { o[k] = (silu((x[k] as f32))) as bf16; } }";
    assert!(
        lowered_calls(bf, "wukong_vmath_bf16_out"),
        "bf16-in/bf16-out activation -> wukong_vmath_bf16_out"
    );
    // f16 in, f16 out -> the f16 twin.
    let hf = "module m\nfn a(x:[f16;64], mut o:[f16;64]) { \
        for k in 0..64 { o[k] = (gelu((x[k] as f32))) as f16; } }";
    assert!(
        lowered_calls(hf, "wukong_vmath_f16_out"),
        "f16-in/f16-out activation -> wukong_vmath_f16_out"
    );
    // An **f32** output must NOT take the narrowing path (it's the plain half-in/f32-out kernel).
    let f32out = "module m\nfn a(x:[bf16;64], mut o:[f32;64]) { \
        for k in 0..64 { o[k] = silu((x[k] as f32)); } }";
    assert!(
        !lowered_calls(f32out, "wukong_vmath_bf16_out"),
        "f32-output activation must not take the narrowing store path"
    );

    // native == interp across opt levels, over a sign/magnitude spread, both halves and two activations.
    for (ty, act) in [("bf16", "silu"), ("f16", "gelu")] {
        let prog = format!(
            "fn a(x:[{ty};4096], mut o:[{ty};4096]) {{ \
             for j in 0..4096 {{ o[j] = ({act}((x[j] as f32))) as {ty}; }} }} \
             fn main() -> i32 {{ let mut x:[{ty};4096]=[0.0 as {ty};4096]; \
             let mut o:[{ty};4096]=[0.0 as {ty};4096]; \
             for i in 0..4096 {{ x[i]=(((i as f32)-2048.0)*0.01) as {ty}; }} \
             a(x,o); let mut s:f32=0.0; for t in 0..4096 {{ s = s + (o[t] as f32); }} \
             print((s*100.0) as i32); return 0; }}"
        );
        for opt in [0u8, 2, 3] {
            assert_eq!(
                jit(&prog, opt).expect("jit"),
                interp(&prog, opt).expect("interp"),
                "{ty}-out {act} native vs interp mismatch at -O{opt}"
            );
        }
    }
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
        "fn bmm(a: [f32; 8], b: [f32; 8], mut c: [f32; 8]) {{ \
        for h in 0..2 {{ for i in 0..2 {{ for j in 0..2 {{ let mut s: f32 = 0.0; \
        for k in 0..2 {{ s = s + a[h*4 + i*2 + k] * b[h*4 + k*2 + j]; }} \
        c[h*4 + i*2 + j] = s; }} }} }} }} fn main() -> i32 {{ {head}"
    );
    // C[h] = A[h]·B[h]ᵀ (attention Q·Kᵀ): head0 [[17,23],[39,53]], head1 = I·B1ᵀ = B1ᵀ [[9,11],[10,12]].
    let transposed = format!(
        "fn bmm(a: [f32; 8], b: [f32; 8], mut c: [f32; 8]) {{ \
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
/// values. A matmul followed by a bias-add / activation loop folds to one `wukong_sgemm_nt_epi`; the
/// differential gate can't catch a recognizer misfire (both backends run the fused kernel), so the
/// values are computed independently. x=[[1,2],[3,4]], W=I, bias=[-10,-1] ⇒ pre-act [-9,1,-7,3]
/// (with bias) or [1,2,3,4] (bias-free). The GELU/SiLU cases cover the transformer FFN forms; the
/// bias-free `silu(x·Wᵀ)` is the LLaMA SwiGLU shape (null bias through the same kernel).
#[test]
fn differential_linear_epilogue() {
    let body = "fn lin(x: [f32; 4], w: [f32; 4], bias: [f32; 2], mut out: [f32; 4]) { \
        for i in 0..2 { for j in 0..2 { let mut s: f32 = 0.0; \
        for k in 0..2 { s = s + x[i*2+k] * w[j*2+k]; } out[i*2+j] = s; } } \
        for i in 0..2 { for j in 0..2 { out[i*2+j] = EPI; } } } \
        fn main() -> i32 { let x: [f32; 4] = [1.0,2.0,3.0,4.0]; \
        let w: [f32; 4] = [1.0,0.0,0.0,1.0]; let bias: [f32; 2] = [-10.0,-1.0]; \
        let mut out: [f32; 4] = [0.0,0.0,0.0,0.0]; lin(x, w, bias, out); \
        print(out[0] as i32); print(out[1] as i32); print(out[2] as i32); print(out[3] as i32); \
        return 0; }";
    // Bias-free variant (LLaMA SwiGLU `silu(x·Wᵀ)`): no bias param, pre-act = [1,2,3,4].
    let nobias = "fn lin(x: [f32; 4], w: [f32; 4], mut out: [f32; 4]) { \
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
fn lowered(src: &str, opt: u8) -> (wukong_mir::Program, Interner) {
    let mut interner = Interner::new();
    let (module, _) = wukong_parser::parse_module(src, SourceId(0), &mut interner);
    let (sema, _) = wukong_sema::check(&module, &interner);
    let (mut program, _) = wukong_mir_build::lower_program(&module, &sema, &mut interner);
    wukong_opt::optimize(&mut program, opt);
    (program, interner)
}

/// The streaming saxpy dispatch: a `for k { o[k] = a*x[k] + y[k] }` loop must (a) lower to one
/// `wukong_velem_f32` call (the 256-bit AVX2 + non-temporal-store kernel — wider than, and store-
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

    // The streaming recognizer must have fired: `a*x[k] + y[k]` is one `wukong_velem_f32` call (the
    // fused multiply-add and the lane work now live inside that kernel, not as inline MIR vector ops).
    let (prog, interner) = lowered(&kernel(64), 2);
    let mir = wukong_mir::print::print_program(&prog, &interner);
    assert!(
        mir.contains("wukong_velem_f32"),
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

/// The broadcast-bias dispatch: a `for i { for j { out[i*C+j] = x[i*C+j] + b[j] } }` nest must (a)
/// lower to one `wukong_bias_bcast_f32` call (the 256-bit AVX2 broadcast-bias kernel — the row-
/// broadcast `b[j]` is what the affine velem recognizer declines), (b) agree between interpreter and
/// native bit-for-bit, and (c) compute the same result the scalar nest would. `x=0`, `b[j]=j` ⇒
/// `sum = rows * C*(C-1)/2`. Sizes hit the kernel's vector body, its scalar tail, and both.
#[test]
fn bias_bcast_is_correct_across_sizes() {
    let kernel = |rows: usize, cols: usize| {
        let n = rows * cols;
        format!(
            "fn main() -> i32 {{ let x: [f32; {n}] = [0.0; {n}]; \
             let mut b: [f32; {cols}] = [0.0; {cols}]; \
             let mut j0: i32 = 0; while j0 < {cols} {{ b[j0] = (j0 as f32); j0 += 1; }} \
             let mut out: [f32; {n}] = [0.0; {n}]; \
             for i in 0..{rows} {{ for j in 0..{cols} {{ out[i * {cols} + j] = x[i * {cols} + j] + b[j]; }} }} \
             let mut s: f32 = 0.0; let mut k: i32 = 0; while k < {n} {{ s = s + out[k]; k += 1; }} \
             return s as i32; }}"
        )
    };

    // The broadcast-bias recognizer must have fired.
    let (prog, interner) = lowered(&kernel(8, 12), 2);
    let mir = wukong_mir::print::print_program(&prog, &interner);
    assert!(
        mir.contains("wukong_bias_bcast_f32"),
        "broadcast bias `x[i*C+j] + b[j]` should dispatch to the bias kernel:\n{mir}"
    );

    // (3,4) tail-only, (2,8) one vector no tail, (5,13)/(7,3) vector+tail, (8,8)/(7,16) larger.
    for (rows, cols) in [(3usize, 4usize), (2, 8), (5, 13), (7, 3), (8, 8), (7, 16)] {
        let src = kernel(rows, cols);
        let native = jit(&src, 3).expect("jit");
        let interp = interp(&src, 3).expect("interp");
        assert_eq!(native, interp, "bias native vs interp mismatch at {rows}x{cols}");
        let want = (rows * (cols * (cols - 1) / 2)) as i64;
        assert_eq!(native.0, want, "wrong bias-bcast sum at {rows}x{cols}");
    }
}

/// The `@parallel` broadcast-bias must dispatch to `wukong_bias_bcast_f32_parallel` (via the
/// whole-function interceptor, before the generic outliner) and stay bit-identical to the serial
/// kernel the interpreter calls — interp == native across opt levels.
#[test]
fn bias_bcast_parallel_matches_interp() {
    let src = "@parallel fn addb(x: [f32; 96], b: [f32; 12], mut out: [f32; 96]) { \
               for i in 0..8 { for j in 0..12 { out[i * 12 + j] = x[i * 12 + j] + b[j]; } } } \
               fn main() -> i32 { let mut x: [f32; 96] = [0.0; 96]; \
               let mut i0: i32 = 0; while i0 < 96 { x[i0] = (i0 as f32); i0 += 1; } \
               let mut b: [f32; 12] = [0.0; 12]; \
               let mut j0: i32 = 0; while j0 < 12 { b[j0] = (j0 as f32); j0 += 1; } \
               let mut out: [f32; 96] = [0.0; 96]; addb(x, b, out); \
               let mut s: f32 = 0.0; let mut k: i32 = 0; while k < 96 { s = s + out[k]; k += 1; } \
               return s as i32; }";
    let (prog, interner) = lowered(src, 2);
    let mir = wukong_mir::print::print_program(&prog, &interner);
    assert!(
        mir.contains("wukong_bias_bcast_f32_parallel"),
        "@parallel bias should dispatch to the multicore kernel:\n{mir}"
    );
    for opt in [0u8, 3] {
        let native = jit(src, opt).expect("jit");
        let interp = interp(src, opt).expect("interp");
        assert_eq!(native, interp, "parallel bias native vs interp mismatch at O{opt}");
        // sum_k k (0..96) + 8 * sum_j j (0..12) = 4560 + 8*66 = 5088.
        assert_eq!(native.0, 5088, "wrong parallel bias sum at O{opt}");
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

    // At this (small) trip the f32 dot reduction takes the inlined 128-bit CLIF path: reassociated
    // lane accumulators with a fused multiply-add (the 256-bit AVX2 kernel only fires at large trips —
    // `VEC256_REDUCTION_MIN_TRIP` — and is covered by `p4_reduction256_gate_and_differential`).
    let (prog, interner) = lowered(&dot(64), 2);
    let mir = wukong_mir::print::print_program(&prog, &interner);
    assert!(
        mir.contains("fma") && mir.contains("x f32>"),
        "dot reduction should vectorize to a 128-bit fma accumulator:\n{mir}"
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

/// The reassociated-reduction gate: the 256-bit AVX2 reduction reassociates the f32 sum, so beyond
/// interp==native (which `reduction_reassociation_is_backend_consistent` covers) the *value* must
/// stay within a `c·√K·ε` tolerance of an f64 reference computed from the same f32 inputs. This is
/// the documented check for reassociated float reductions — it catches a wrong horizontal fold or a
/// dropped lane that a bit-for-bit backend match alone would miss (both backends could agree on a
/// wrong reassociation). Fractional inputs so the low bits genuinely differ from a strict sum.
#[test]
fn p4_reduction_f64_reference() {
    for n in [64usize, 257, 1000, 4096] {
        let src = format!(
            "fn main() -> i32 {{ let mut x: [f32; {n}] = [0.0; {n}]; let mut y: [f32; {n}] = [0.0; {n}]; \
             let mut i: i32 = 0; while i < {n} {{ x[i] = ((i % 17) as f32) * 0.5 + 1.0; \
             y[i] = ((i % 13) as f32) * 0.25 - 0.5; i += 1; }} \
             let mut s: f32 = 0.0; for k in 0..{n} {{ s = s + x[k] * y[k]; }} \
             print(s); return 0; }}"
        );
        let (_, out) = jit(&src, 3).expect("jit");
        let got: f64 = String::from_utf8(out)
            .unwrap()
            .trim()
            .parse()
            .expect("parse printed f32 sum");
        // f64 reference over the identical f32 inputs (widened exactly).
        let mut refsum = 0.0f64;
        for k in 0..n {
            let x = (((k % 17) as f32) * 0.5 + 1.0) as f64;
            let y = (((k % 13) as f32) * 0.25 - 0.5) as f64;
            refsum += x * y;
        }
        let tol = 8.0 * (n as f64).sqrt() * f64::from(f32::EPSILON) * refsum.abs().max(1.0);
        assert!(
            (got - refsum).abs() <= tol,
            "n={n}: reassociated 256-bit dot {got} vs f64 ref {refsum} exceeds tol {tol}"
        );
    }
}

/// The 256-bit reduction is gated on a compile-time-known trip ≥ `VEC256_REDUCTION_MIN_TRIP` (=2048):
/// below the threshold its out-of-line call loses to the inlined 128-bit reduction, so a small (or
/// runtime-unknown) trip must keep the 128-bit path. Verifies BOTH the gate (a `veckernel` appears only
/// at/above the threshold) and that the gated 256-bit path is bit-exact interp-vs-native and -O0-vs-O3
/// across the fold kinds: fused-`fma` dot/composite, a plain (non-product) addend, and non-fused
/// `fmax`/`fmin`.
#[test]
fn p4_reduction256_gate_and_differential() {
    // Full program: init streams a,b (deterministic, both signs), then the reduction `red`, print s.
    let mk = |n: usize, red: &str| {
        format!(
            "fn main() -> i32 {{ let mut a: [f32; {n}] = [0.0; {n}]; let mut b: [f32; {n}] = [0.0; {n}]; \
             let mut i: i32 = 0; while i < {n} {{ a[i] = ((i % 23) as f32) * 0.5 - 3.0; \
             b[i] = ((i % 19) as f32) * 0.25 + 0.5; i += 1; }} \
             {red} print(s); return ((s as i32) & 1023); }}"
        )
    };
    // The fold bodies, parameterized by trip `n`. Two fuse a product into the accumulate (`Add` of an
    // `X*Y`), one has a non-product addend (no fma), two are max/min (never fma).
    let folds = |n: usize| -> Vec<(&'static str, String)> {
        vec![
            ("fma dot", format!("let mut s: f32 = 0.0; for k in 0..{n} {{ s = s + a[k]*b[k]; }}")),
            ("fma composite", format!("let mut s: f32 = 0.0; for k in 0..{n} {{ s = s + a[k]*(a[k]+b[k]); }}")),
            ("plain", format!("let mut s: f32 = 0.0; for k in 0..{n} {{ s = s + (a[k] - b[k]); }}")),
            ("fmax", format!("let mut s: f32 = -1000.0; for k in 0..{n} {{ s = fmax(s, a[k]*b[k]); }}")),
            ("fmin", format!("let mut s: f32 = 1000.0; for k in 0..{n} {{ s = fmin(s, a[k]*b[k]); }}")),
        ]
    };

    // Gate — below the threshold (1024 < 2048), no 256-bit kernel: the reduction stays 128-bit.
    for (name, red) in folds(1024) {
        let (prog, interner) = lowered(&mk(1024, &red), 2);
        assert!(
            !wukong_mir::print::print_program(&prog, &interner).contains("veckernel"),
            "N=1024 (< 2048) must keep the inlined 128-bit reduction [{name}]"
        );
    }
    // At/above the threshold — the 256-bit kernel fires and stays bit-exact on both backends and opts.
    for n in [2048usize, 4096] {
        for (name, red) in folds(n) {
            let src = mk(n, &red);
            let (prog, interner) = lowered(&src, 2);
            assert!(
                wukong_mir::print::print_program(&prog, &interner).contains("veckernel"),
                "N={n} (≥ 2048) must use the 256-bit reduction [{name}]"
            );
            for opt in [0u8, 3] {
                let native = jit(&src, opt).expect("jit");
                let interpd = interp(&src, opt).expect("interp");
                assert_eq!(
                    native, interpd,
                    "256-bit reduction native vs interp N={n} -O{opt} [{name}]"
                );
            }
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
    let mir = wukong_mir::print::print_program(&prog, &interner);
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
    // Small trip → the inlined 128-bit reduction (the 256-bit kernel is gated to large trips, covered
    // by `p4_reduction256_gate_and_differential`); the squared-difference addend fuses to an `fma`.
    let (prog, interner) = lowered(&kernel(64), 2);
    let mir = wukong_mir::print::print_program(&prog, &interner);
    assert!(
        mir.contains("fma") && mir.contains("x f32>"),
        "ssd reduction should vectorize to a 128-bit fma accumulator:\n{mir}"
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
/// one `wukong_velem_f32` call (the 256-bit AVX2 + non-temporal-store kernel, `VE_RELU`), agree
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
    let mir = wukong_mir::print::print_program(&prog, &interner);
    assert!(
        mir.contains("wukong_velem_f32"),
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
    let mir = wukong_mir::print::print_program(&prog, &interner);
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
/// x } else { 0 } } else { 6 }` must lower to one `wukong_velem_f32` call (`VE_RELU6`) and stay
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
    let mir = wukong_mir::print::print_program(&prog, &interner);
    assert!(
        mir.contains("wukong_velem_f32"),
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
            "module m\n{attr}fn mm(a: [f32; {n2}], b: [f32; {n2}], mut c: [f32; {n2}]) {{\n\
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

/// The idiomatic shape-typed matmul — `c[i,j] = Σ a[i,k]·b[k,j]` written with multi-index tensor
/// accesses on `Tensor[f32,N,N]` params — must dispatch to the SAME tuned `wukong_sgemm` kernel as
/// the flat `a[i*N+k]` spelling (the 2-index access supplies the row stride from the tensor's inner
/// dim), agree native==interp, and compute the right result. The transposed `b[j,k]` spelling is the
/// `nn.Linear` `A·Bᵀ` form and must reach `wukong_sgemm_nt`. Arrays are passed into the tensor
/// params (sema's lenient array↔tensor unify) and both decay to a base pointer.
#[test]
fn tensor_matmul_is_correct() {
    fn reference(ns: usize, transposed_b: bool) -> i64 {
        let a: Vec<f32> = (0..ns * ns).map(|i| (i % 3) as f32).collect();
        let b: Vec<f32> = (0..ns * ns).map(|i| (i % 2) as f32).collect();
        let mut sum = 0.0f32;
        for i in 0..ns {
            for j in 0..ns {
                let mut acc = 0.0f32;
                for k in 0..ns {
                    let bkj = if transposed_b { b[j * ns + k] } else { b[k * ns + j] };
                    acc += a[i * ns + k] * bkj;
                }
                sum += acc;
            }
        }
        sum as i64
    }

    // `ijk` dot form in shape-typed tensor notation; `b_idx` is `k,j` (normal) or `j,k` (nn.Linear).
    let kernel = |ns: usize, transposed_b: bool, parallel: bool| {
        let attr = if parallel { "@parallel\n" } else { "" };
        let n2 = ns * ns;
        let b_idx = if transposed_b { "j, k" } else { "k, j" };
        format!(
            "module m\n{attr}fn mm(a: Tensor[f32, {ns}, {ns}], b: Tensor[f32, {ns}, {ns}], \
             mut c: Tensor[f32, {ns}, {ns}]) {{\n\
             for i in 0..{ns} {{ for j in 0..{ns} {{ let mut s: f32 = 0.0; \
             for k in 0..{ns} {{ s = s + a[i, k] * b[{b_idx}]; }} c[i, j] = s; }} }} }}\n\
             fn main() -> i32 {{ let mut a: [f32; {n2}] = [0.0; {n2}]; \
             let mut b: [f32; {n2}] = [0.0; {n2}]; let mut c: [f32; {n2}] = [0.0; {n2}]; \
             let mut i: i32 = 0; \
             while i < {n2} {{ a[i] = ((i % 3) as f32); b[i] = ((i % 2) as f32); i += 1; }} \
             mm(a, b, c); let mut s: f32 = 0.0; let mut j: i32 = 0; \
             while j < {n2} {{ s = s + c[j]; j += 1; }} return s as i32; }}"
        )
    };

    // The 2-index tensor spelling must reach the kernel, not fall to a scalar nest.
    assert!(
        lowered_calls(&kernel(8, false, false), "wukong_sgemm"),
        "tensor a[i,k]*b[k,j] -> wukong_sgemm"
    );
    assert!(
        lowered_calls(&kernel(8, true, false), "wukong_sgemm_nt"),
        "tensor a[i,k]*b[j,k] -> wukong_sgemm_nt"
    );

    for ns in [6usize, 7, 16, 17, 32] {
        for transposed_b in [false, true] {
            for parallel in [false, true] {
                let src = kernel(ns, transposed_b, parallel);
                let native = jit(&src, 3).expect("jit");
                let interp = interp(&src, 3).expect("interp");
                assert_eq!(
                    native, interp,
                    "tensor matmul native vs interp (ns={ns}, tb={transposed_b}, par={parallel})"
                );
                assert_eq!(
                    native.0,
                    reference(ns, transposed_b),
                    "tensor matmul wrong result (ns={ns}, tb={transposed_b}, par={parallel})"
                );
            }
        }
    }
}

/// The `ikj` *accumulate* matmul (`let aik = a[i,k]; for j { c[i,j] = c[i,j] + aik*b[k,j] }`, the
/// other canonical spelling, with a per-row zero-init for beta=0) must ALSO dispatch from tensor
/// notation — the `aik` binding, the C read-modify-write store, and the zero-init all accept the
/// 2-index form. Proves dispatch + native==interp==reference.
#[test]
fn tensor_matmul_accumulate_form() {
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
            "module m\n{attr}fn mm(a: Tensor[f32, {ns}, {ns}], b: Tensor[f32, {ns}, {ns}], \
             mut c: Tensor[f32, {ns}, {ns}]) {{\n\
             for i in 0..{ns} {{ for j0 in 0..{ns} {{ c[i, j0] = 0.0; }} \
             for k in 0..{ns} {{ let aik: f32 = a[i, k]; \
             for j in 0..{ns} {{ c[i, j] = c[i, j] + aik * b[k, j]; }} }} }} }}\n\
             fn main() -> i32 {{ let mut a: [f32; {n2}] = [0.0; {n2}]; \
             let mut b: [f32; {n2}] = [0.0; {n2}]; let mut c: [f32; {n2}] = [0.0; {n2}]; \
             let mut i: i32 = 0; \
             while i < {n2} {{ a[i] = ((i % 3) as f32); b[i] = ((i % 2) as f32); i += 1; }} \
             mm(a, b, c); let mut s: f32 = 0.0; let mut j: i32 = 0; \
             while j < {n2} {{ s = s + c[j]; j += 1; }} return s as i32; }}"
        )
    };
    assert!(
        lowered_calls(&kernel(8, false), "wukong_sgemm"),
        "tensor accumulate matmul -> wukong_sgemm"
    );
    for ns in [6usize, 7, 16, 17, 32] {
        for parallel in [false, true] {
            let src = kernel(ns, parallel);
            let native = jit(&src, 3).expect("jit");
            let interp = interp(&src, 3).expect("interp");
            assert_eq!(native, interp, "tensor acc matmul native vs interp (ns={ns}, par={parallel})");
            assert_eq!(native.0, reference(ns), "tensor acc matmul wrong (ns={ns}, par={parallel})");
        }
    }
}

/// Lower `src` and report whether any function calls the named runtime symbol — used to prove the
/// matmul recognizer fired (and picked the serial vs parallel variant), not merely that a scalar
/// fallback happened to compute the right answer.
fn lowered_calls(src: &str, callee: &str) -> bool {
    let mut interner = Interner::new();
    let (module, _) = wukong_parser::parse_module(src, SourceId(0), &mut interner);
    let (sema, _) = wukong_sema::check(&module, &interner);
    let (program, _) = wukong_mir_build::lower_program(&module, &sema, &mut interner);
    let target = interner.intern(callee);
    program.funcs.iter().any(|f| {
        f.blocks.iter().any(|b| {
            b.insts
                .iter()
                .any(|ins| matches!(&ins.op, wukong_mir::Op::Call { func, .. } if *func == target))
        })
    })
}

/// The canonical f32 matmul nest must lower to the tuned `wukong_sgemm` microkernel (and the
/// `@parallel` form to the parallel variant), in both the accumulate and zero-init shapes.
#[test]
fn matmul_nest_lowers_to_sgemm() {
    // Accumulate form (beta = 1): no per-row zero-init.
    let acc = |attr: &str| {
        format!(
            "module m\n{attr}fn mm(a:[f32;64],b:[f32;64],mut c:[f32;64]) {{ \
             for i in 0..8 {{ for k in 0..8 {{ let aik: f32 = a[i*8+k]; \
             for j in 0..8 {{ c[i*8+j] = c[i*8+j] + aik * b[k*8+j]; }} }} }} }}"
        )
    };
    // Overwrite form (beta = 0): a per-row zero-init loop precedes the K loop.
    let ovr = |attr: &str| {
        format!(
            "module m\n{attr}fn mm(a:[f32;64],b:[f32;64],mut c:[f32;64]) {{ \
             for i in 0..8 {{ for j0 in 0..8 {{ c[i*8+j0] = 0.0; }} \
             for k in 0..8 {{ let aik: f32 = a[i*8+k]; \
             for j in 0..8 {{ c[i*8+j] = c[i*8+j] + aik * b[k*8+j]; }} }} }} }}"
        )
    };
    assert!(
        lowered_calls(&acc(""), "wukong_sgemm"),
        "accumulate -> sgemm"
    );
    assert!(
        lowered_calls(&ovr(""), "wukong_sgemm"),
        "overwrite -> sgemm"
    );
    assert!(
        lowered_calls(&acc("@parallel\n"), "wukong_sgemm_parallel"),
        "@parallel -> sgemm_parallel"
    );
    // A non-matmul triple loop (wrong B stride) must NOT be misrecognized.
    let not_mm = "module m\nfn f(a:[f32;64],b:[f32;64],mut c:[f32;64]) {{ \
        for i in 0..8 { for k in 0..8 { let aik: f32 = a[i*8+k]; \
        for j in 0..8 { c[i*8+j] = c[i*8+j] + aik * b[j*8+k]; } } } }";
    assert!(
        !lowered_calls(not_mm, "wukong_sgemm"),
        "transposed-B is not a row-major matmul"
    );
}

/// An embedded matmul nest inside a MULTI-statement `@parallel` fn must pick the multicore
/// kernel from the statement path (`lower_for` passes `self.parallel_fn`, like its i8/lowp/norm
/// siblings) — it was hardcoded serial, so a `@parallel` transformer block ran its six plain
/// GEMMs on one core while the norms went multicore (measured ≈serial to ~2× slower overall via
/// the package-clock penalty). The parallel kernel is bit-identical to serial (fixed chunking),
/// pinned by the native-vs-interp differential across opt levels.
#[test]
fn embedded_parallel_matmul_dispatches_multicore() {
    // Two top-level statements (a zero-fill loop + the matmul nest), so the single-statement
    // whole-function interceptor declines and lower_for's statement path does the recognition.
    let mm = "@parallel\nfn mm(a:[f32;64],b:[f32;64],mut c:[f32;64]) { \
         for s in 0..64 { c[s] = 0.0; } \
         for i in 0..8 { for k in 0..8 { let aik: f32 = a[i*8+k]; \
         for j in 0..8 { c[i*8+j] = c[i*8+j] + aik * b[k*8+j]; } } } }";
    let src = format!("module m\n{mm}");
    assert!(
        lowered_calls(&src, "wukong_sgemm_parallel"),
        "embedded matmul in a multi-statement @parallel fn -> sgemm_parallel"
    );
    let full = format!(
        "module m\n{mm}\n\
         fn main() -> i32 {{ let mut a:[f32;64]=[0.0;64]; let mut b:[f32;64]=[0.0;64]; \
         let mut c:[f32;64]=[0.0;64]; \
         for i in 0..64 {{ a[i] = ((i % 7) as f32) * 0.31; b[i] = ((i % 5) as f32) * 0.17 - 0.4; }} \
         mm(a,b,c); let mut s: f32 = 0.0; \
         for i in 0..64 {{ s = s + c[i] * ((i % 3) as f32); }} \
         return (s * 100.0) as i32; }}"
    );
    for opt in [0u8, 2, 3] {
        let n = jit(&full, opt).expect("jit");
        let i = interp(&full, opt).expect("interp");
        assert_eq!(
            n, i,
            "embedded @parallel matmul: native vs interp mismatch at -O{opt}"
        );
    }
}

/// The nn.Linear form `C = A·Bᵀ` (B indexed `[j*K+k]`) must lower to `wukong_sgemm_nt`, run
/// correctly, and stay native==interp. A plain `C = A·B` must NOT pick the transposed kernel.
#[test]
fn linear_nt_matmul_lowers_and_runs() {
    let nt = |attr: &str| {
        format!(
            "module m\n{attr}fn lin(a:[f32;48],b:[f32;32],mut c:[f32;24]) {{ \
             for i in 0..6 {{ for j0 in 0..4 {{ c[i*4+j0] = 0.0; }} \
             for k in 0..8 {{ let aik: f32 = a[i*8+k]; \
             for j in 0..4 {{ c[i*4+j] = c[i*4+j] + aik * b[j*8+k]; }} }} }} }}"
        )
    };
    assert!(
        lowered_calls(&nt(""), "wukong_sgemm_nt"),
        "A·Bᵀ -> sgemm_nt"
    );
    assert!(
        lowered_calls(&nt("@parallel\n"), "wukong_sgemm_nt_parallel"),
        "@parallel A·Bᵀ -> sgemm_nt_parallel"
    );
    // C=A·B (b indexed [k*4+j]) must use the non-transposed kernel, never the nt one.
    let normal = "module m\nfn mm(a:[f32;48],b:[f32;32],mut c:[f32;24]) { \
        for i in 0..6 { for k in 0..8 { let aik: f32 = a[i*8+k]; \
        for j in 0..4 { c[i*4+j] = c[i*4+j] + aik * b[k*4+j]; } } } }";
    assert!(lowered_calls(normal, "wukong_sgemm"), "C=A·B -> sgemm");
    assert!(
        !lowered_calls(normal, "wukong_sgemm_nt"),
        "C=A·B is not transposed"
    );

    // End to end: A is 6x8, B is 4x8 (so Bᵀ is 8x4), C is 6x4. Native must equal interp.
    let src = "module m\nfn lin(a:[f32;48],b:[f32;32],mut c:[f32;24]) { \
        for i in 0..6 { for j0 in 0..4 { c[i*4+j0] = 0.0; } \
        for k in 0..8 { let aik: f32 = a[i*8+k]; \
        for j in 0..4 { c[i*4+j] = c[i*4+j] + aik * b[j*8+k]; } } } }\n\
        fn main() -> i32 { let mut a:[f32;48]=[0.0;48]; let mut b:[f32;32]=[0.0;32]; \
        let mut c:[f32;24]=[0.0;24]; let mut i: i32 = 0; \
        while i < 48 { a[i] = ((i % 5) as f32) * 0.25; i += 1; } \
        let mut j: i32 = 0; while j < 32 { b[j] = ((j % 7) as f32) - 2.0; j += 1; } \
        lin(a, b, c); let mut s: f32 = 0.0; let mut t: i32 = 0; \
        while t < 24 { s = s + c[t]; t += 1; } return (s * 1000.0) as i32; }";
    assert!(lowered_calls(src, "wukong_sgemm_nt"));
    assert_eq!(jit(src, 3).expect("jit"), interp(src, 3).expect("interp"));
}

/// The textbook `ijk` dot-product matmul (`let s=0; for k s+=a*b; c=s`) must be recognized too —
/// both `C=A·B` and the `C=A·Bᵀ` (nn.Linear) spelling — and stay native==interp.
#[test]
fn ijk_dot_product_matmul_recognized() {
    // C = A·Bᵀ (b[j*K+k]) — the natural nn.Linear spelling.
    let nt = "module m\nfn lin(a:[f32;48],b:[f32;32],mut c:[f32;24]) { \
        for i in 0..6 { for j in 0..4 { let mut s: f32 = 0.0; \
        for k in 0..8 { s = s + a[i*8+k] * b[j*8+k]; } c[i*4+j] = s; } } }";
    assert!(
        lowered_calls(nt, "wukong_sgemm_nt"),
        "ijk A·Bᵀ -> sgemm_nt"
    );
    // C = A·B (b[k*N+j]).
    let normal = "module m\nfn mm(a:[f32;48],b:[f32;32],mut c:[f32;24]) { \
        for i in 0..6 { for j in 0..4 { let mut s: f32 = 0.0; \
        for k in 0..8 { s = s + a[i*8+k] * b[k*4+j]; } c[i*4+j] = s; } } }";
    assert!(lowered_calls(normal, "wukong_sgemm"), "ijk A·B -> sgemm");
    assert!(!lowered_calls(normal, "wukong_sgemm_nt"));

    // End to end (A·Bᵀ): native must equal interp.
    let src = "module m\nfn lin(a:[f32;48],b:[f32;32],mut c:[f32;24]) { \
        for i in 0..6 { for j in 0..4 { let mut s: f32 = 0.0; \
        for k in 0..8 { s = s + a[i*8+k] * b[j*8+k]; } c[i*4+j] = s; } } }\n\
        fn main() -> i32 { let mut a:[f32;48]=[0.0;48]; let mut b:[f32;32]=[0.0;32]; \
        let mut c:[f32;24]=[0.0;24]; let mut i: i32 = 0; \
        while i < 48 { a[i] = ((i % 5) as f32) * 0.25; i += 1; } \
        let mut j: i32 = 0; while j < 32 { b[j] = ((j % 7) as f32) - 2.0; j += 1; } \
        lin(a, b, c); let mut s: f32 = 0.0; let mut t: i32 = 0; \
        while t < 24 { s = s + c[t]; t += 1; } return (s * 1000.0) as i32; }";
    assert!(lowered_calls(src, "wukong_sgemm_nt"));
    assert_eq!(jit(src, 3).expect("jit"), interp(src, 3).expect("interp"));
}

/// The zero-init (beta = 0) matmul, end to end: native and interpreter agree bit-for-bit (both run
/// the same kernel) and match an independent reference.
#[test]
fn matmul_overwrite_differential() {
    let ns = 9usize; // not a multiple of MR/NR, exercises the microkernel remainders
    let n2 = ns * ns;
    let src = format!(
        "module m\nfn mm(a: [f32; {n2}], b: [f32; {n2}], mut c: [f32; {n2}]) {{ \
         for i in 0..{ns} {{ for j0 in 0..{ns} {{ c[i*{ns}+j0] = 0.0; }} \
         for k in 0..{ns} {{ let aik: f32 = a[i*{ns}+k]; \
         for j in 0..{ns} {{ c[i*{ns}+j] = c[i*{ns}+j] + aik * b[k*{ns}+j]; }} }} }} }}\n\
         fn main() -> i32 {{ let mut a: [f32; {n2}] = [0.0; {n2}]; let mut b: [f32; {n2}] = [0.0; {n2}]; \
         let mut c: [f32; {n2}] = [0.0; {n2}]; let mut i: i32 = 0; \
         while i < {n2} {{ a[i] = ((i % 5) as f32) * 0.5; b[i] = ((i % 3) as f32) - 1.0; i += 1; }} \
         mm(a, b, c); let mut s: f32 = 0.0; let mut j: i32 = 0; \
         while j < {n2} {{ s = s + c[j]; j += 1; }} return (s * 100.0) as i32; }}"
    );
    assert!(lowered_calls(&src, "wukong_sgemm"), "recognizer must fire");
    let native = jit(&src, 3).expect("jit");
    let interp = interp(&src, 3).expect("interp");
    assert_eq!(native, interp, "overwrite matmul native vs interp");
}

/// Affine LayerNorm/RMSNorm (a per-column scale `g[i]` and optional shift `b[i]`) must dispatch to
/// `wukong_norm_affine_f32`, while plain (gamma=1) norms keep using `wukong_norm_f32`, and a softmax
/// with a trailing per-column scale must decline both (it has no affine parameters — the soundness
/// guard makes it fall back to the generic vectorizer rather than silently drop the scale).
#[test]
fn affine_norm_dispatch() {
    // Affine LayerNorm: (x-mean)*inv*g[i] + b[i]
    let ln_affine = "module m\nfn f(mut x:[f32;8], g:[f32;8], b:[f32;8]) { \
        let mut s: f32 = 0.0; for i in 0..8 { s = s + x[i]; } let mean: f32 = s / 8.0; \
        let mut v: f32 = 0.0; for i in 0..8 { v = v + (x[i] - mean) * (x[i] - mean); } \
        let inv: f32 = rsqrt(v / 8.0 + 0.00001); \
        for i in 0..8 { x[i] = (x[i] - mean) * inv * g[i] + b[i]; } }";
    assert!(
        lowered_calls(ln_affine, "wukong_norm_affine_f32"),
        "affine LayerNorm -> wukong_norm_affine_f32"
    );
    assert!(
        !lowered_calls(ln_affine, "wukong_norm_f32"),
        "affine LayerNorm must NOT use the plain kernel"
    );

    // Affine RMSNorm: x[i]*inv*g[i] (scale only, no shift)
    let rn_affine = "module m\nfn f(mut x:[f32;8], g:[f32;8]) { \
        let mut s: f32 = 0.0; for i in 0..8 { s = s + x[i] * x[i]; } \
        let inv: f32 = rsqrt(s / 8.0 + 0.00001); \
        for i in 0..8 { x[i] = x[i] * inv * g[i]; } }";
    assert!(
        lowered_calls(rn_affine, "wukong_norm_affine_f32"),
        "affine RMSNorm -> wukong_norm_affine_f32"
    );

    // Plain LayerNorm (gamma=1, beta=0) still uses the plain kernel, not the affine one.
    let ln_plain = "module m\nfn f(mut x:[f32;8]) { \
        let mut s: f32 = 0.0; for i in 0..8 { s = s + x[i]; } let mean: f32 = s / 8.0; \
        let mut v: f32 = 0.0; for i in 0..8 { v = v + (x[i] - mean) * (x[i] - mean); } \
        let inv: f32 = rsqrt(v / 8.0 + 0.00001); \
        for i in 0..8 { x[i] = (x[i] - mean) * inv; } }";
    assert!(
        lowered_calls(ln_plain, "wukong_norm_f32"),
        "plain LayerNorm -> wukong_norm_f32"
    );
    assert!(
        !lowered_calls(ln_plain, "wukong_norm_affine_f32"),
        "plain LayerNorm must NOT use the affine kernel"
    );

    // softmax with a trailing scale has no affine semantics; the guard makes it decline BOTH norm
    // kernels (falls back to the vectorizer) rather than dropping the scale.
    let sm_scaled = "module m\nfn f(mut x:[f32;8], g:[f32;8]) { \
        let mut m: f32 = x[0]; for i in 0..8 { m = fmax(m, x[i]); } \
        for i in 0..8 { x[i] = exp(x[i] - m); } \
        let mut s: f32 = 0.0; for i in 0..8 { s = s + x[i]; } let inv: f32 = 1.0 / s; \
        for i in 0..8 { x[i] = x[i] * inv * g[i]; } }";
    assert!(
        !lowered_calls(sm_scaled, "wukong_norm_f32")
            && !lowered_calls(sm_scaled, "wukong_norm_affine_f32"),
        "softmax+scale must decline both norm kernels (no affine softmax)"
    );
}

/// A batched norm — `for r in 0..R { <RMSNorm over x[r*C + i]> }` — must dispatch to the fused
/// `wukong_norm_f32` kernel with `rows = R` (the real `[batch*seq, hidden]` transformer shape) rather
/// than fall back to the scalar vectorizer; and the refactor that threaded the batch offset through the
/// recognizer must not have broken the single-row (`rows = 1`) form. The differential gate
/// (native == interp over rows > 1, adversarial inputs) lives in the fuzzer's `rmsnorm_batched` kernel,
/// and the independent per-row reference in `tests/run/batched_rmsnorm.wk`; this just pins that the
/// recognizer keeps *firing* (a silent fallback to scalar would pass both of those yet regress speed).
#[test]
fn batched_norm_dispatch() {
    // R = 3 rows, C = 4 cols, normalized in place over the flat [12] buffer via the `r*4 + i` offset.
    let batched = "module m\nfn f(mut x:[f32;12]) { \
        for r in 0..3 { \
        let mut s: f32 = 0.0; for i in 0..4 { s = s + x[r*4+i] * x[r*4+i]; } \
        let inv: f32 = rsqrt(s / 4.0 + 0.00001); \
        for i in 0..4 { x[r*4+i] = x[r*4+i] * inv; } } }";
    assert!(
        lowered_calls(batched, "wukong_norm_f32"),
        "batched RMSNorm (for r {{ <row r> }}) must dispatch to wukong_norm_f32"
    );

    // Batched LayerNorm (two reductions: mean, then variance) over the `r*4 + i` offset must dispatch too.
    let batched_ln = "module m\nfn f(mut x:[f32;12]) { \
        for r in 0..3 { \
        let mut s: f32 = 0.0; for i in 0..4 { s = s + x[r*4+i]; } let mean: f32 = s / 4.0; \
        let mut v: f32 = 0.0; for i in 0..4 { v = v + (x[r*4+i] - mean) * (x[r*4+i] - mean); } \
        let inv: f32 = rsqrt(v / 4.0 + 0.00001); \
        for i in 0..4 { x[r*4+i] = (x[r*4+i] - mean) * inv; } } }";
    assert!(
        lowered_calls(batched_ln, "wukong_norm_f32"),
        "batched LayerNorm (for r {{ <row r> }}) must dispatch to wukong_norm_f32"
    );

    // Batched softmax (max/exp/sum/normalize, with the row-local `x[r*4]` max-seed) over the offset.
    let batched_sm = "module m\nfn f(mut x:[f32;12]) { \
        for r in 0..3 { \
        let mut m: f32 = x[r*4]; for i in 0..4 { m = fmax(m, x[r*4+i]); } \
        for i in 0..4 { x[r*4+i] = exp(x[r*4+i] - m); } \
        let mut s: f32 = 0.0; for i in 0..4 { s = s + x[r*4+i]; } let inv: f32 = 1.0 / s; \
        for i in 0..4 { x[r*4+i] = x[r*4+i] * inv; } } }";
    assert!(
        lowered_calls(batched_sm, "wukong_norm_f32"),
        "batched softmax (for r {{ <row r> }}) must dispatch to wukong_norm_f32"
    );

    // Batched AFFINE RMSNorm (per-column scale g[i]) must route to the affine kernel, not the plain one
    // — the data is offset-indexed x[r*4+i] while gamma stays column-indexed g[i].
    let batched_affine = "module m\nfn f(mut x:[f32;12], g:[f32;4]) { \
        for r in 0..3 { \
        let mut s: f32 = 0.0; for i in 0..4 { s = s + x[r*4+i] * x[r*4+i]; } \
        let inv: f32 = rsqrt(s / 4.0 + 0.00001); \
        for i in 0..4 { x[r*4+i] = x[r*4+i] * inv * g[i]; } } }";
    assert!(
        lowered_calls(batched_affine, "wukong_norm_affine_f32"),
        "batched affine RMSNorm -> wukong_norm_affine_f32"
    );
    assert!(
        !lowered_calls(batched_affine, "wukong_norm_f32"),
        "batched affine RMSNorm must NOT use the plain kernel"
    );

    // The single-row form (no outer loop) must still dispatch — rows = 1 is the `batch = None` path.
    let single = "module m\nfn f(mut x:[f32;4]) { \
        let mut s: f32 = 0.0; for i in 0..4 { s = s + x[i] * x[i]; } \
        let inv: f32 = rsqrt(s / 4.0 + 0.00001); \
        for i in 0..4 { x[i] = x[i] * inv; } }";
    assert!(
        lowered_calls(single, "wukong_norm_f32"),
        "single-row RMSNorm must still dispatch to wukong_norm_f32"
    );
}

/// A `@parallel` batched RMSNorm normalizes its rows across CPU cores via `wukong_norm_f32_parallel`
/// (intercepted before the generic `@parallel` outliner, mirroring the sgemm/int8 whole-function
/// interceptions). Rows are independent — no cross-row combine — so the multicore kernel is bit-
/// identical to the serial one the interpreter marshals, and native must equal interp at every opt
/// level. (The runtime's own `serial_matches_parallel_bit_for_bit` test pins the kernel equality; this
/// pins the end-to-end dispatch + the rows>1 marshalling across the rayon boundary.)
#[test]
fn differential_parallel_batched_norm() {
    // 64 rows × 64 cols = 4096; many rows so the kernel genuinely spreads across cores.
    let src = "@parallel fn rmsnorm_batch(mut x: [f32; 4096]) { \
         for r in 0..64 { \
         let mut s: f32 = 0.0; for i in 0..64 { s = s + x[r*64+i] * x[r*64+i]; } \
         let inv: f32 = rsqrt(s / 64.0 + 0.00001); \
         for i in 0..64 { x[r*64+i] = x[r*64+i] * inv; } } } \
         fn main() -> i32 { let mut x: [f32; 4096] = [0.0; 4096]; \
         for i in 0..4096 { x[i] = (i as f32) * 0.001 - 2.0; } rmsnorm_batch(x); \
         let mut s: f32 = 0.0; for i in 0..4096 { s = s + x[i]; } \
         print((s * 1000.0) as i32); return 0; }";
    assert!(
        lowered_calls(src, "wukong_norm_f32_parallel"),
        "@parallel batched RMSNorm must dispatch to the multicore norm kernel"
    );
    for opt in [0u8, 2, 3] {
        let n = jit(src, opt).expect("jit");
        let i = interp(src, opt).expect("interp");
        assert_eq!(n, i, "parallel batched norm native vs interp at -O{opt}");
    }

    // The affine form (a learned per-column gamma) maps rows across cores via the multicore *affine*
    // kernel `wukong_norm_affine_f32_parallel`, and must stay bit-exact vs the serial kernel the
    // interpreter marshals (rows independent, no cross-row combine).
    let src_affine = "@parallel fn rmsnorm_affine_batch(mut x: [f32; 4096], g: [f32; 64]) { \
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
        lowered_calls(src_affine, "wukong_norm_affine_f32_parallel"),
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
        "module m\nfn mm(a: [f32; {n2}], b: [f32; {n2}], mut c: [f32; {n2}]) {{ \
         for i in 0..{ns} {{ for k in 0..{ns} {{ let aik: f32 = a[i*{ns}+k]; \
         for j in 0..{ns} {{ c[i*{ns}+j] = c[i*{ns}+j] + aik * b[k*{ns}+j]; }} }} }} }}\n\
         fn main() -> i32 {{ let mut a: [f32; {n2}] = [0.0; {n2}]; let mut b: [f32; {n2}] = [0.0; {n2}]; \
         let mut c: [f32; {n2}] = [0.0; {n2}]; let mut i: i32 = 0; \
         while i < {n2} {{ a[i] = ((i % 5) as f32) * 0.5; b[i] = ((i % 3) as f32) - 1.0; \
         c[i] = ((i % 7) as f32) - 2.0; i += 1; }} \
         mm(a, b, c); let mut s: f32 = 0.0; let mut j: i32 = 0; \
         while j < {n2} {{ s = s + c[j]; j += 1; }} return (s * 100.0) as i32; }}"
    );
    assert!(lowered_calls(&src, "wukong_sgemm"), "recognizer must fire");
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
    use wukong_mir::{BinOp, Builder, CastKind, MirLevel, Op, Program};

    let mut interner = Interner::new();
    let mut b = Builder::new(interner.intern("main"), wukong_mir::MirType::I32);
    use wukong_mir::MirType;
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
        statics: Vec::new(),
        level: MirLevel::Low,
    };
    for f in &prog.funcs {
        assert!(
            wukong_mir::verify::verify_function(f).is_empty(),
            "vector MIR should verify: {:?}",
            wukong_mir::verify::verify_function(f)
        );
    }
    let main = interner.intern("main");
    let native = crate::jit_run(&prog, main, &interner).expect("jit");
    let interp = wukong_interp::run_with_output(&prog, main, &interner).expect("interp");
    assert_eq!(native, interp, "vector native vs interp mismatch");
    assert_eq!(native.0, 13);
}

/// A `@parallel for` kernel: the native backend runs it across CPU cores via the runtime, the
/// interpreter runs the whole range sequentially, and the observable result must be identical.
#[test]
fn parallel_for_matches_interpreter() {
    let src = "@parallel fn scale(x: [i32; 4096], mut out: [i32; 4096]) { \
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

/// The α-scaled GEMV store `y[i] = s * c` (`c` a literal, a prior non-`mut` `let`, or a top-level
/// `const`) must dispatch to `wukong_sgemv_alpha` (the `@parallel` form to its `_parallel` twin),
/// while the plain store keeps the exact pre-α `wukong_sgemv` call (its MIR is byte-identical —
/// verified against a pre-change dump); a store that is NOT `s * c` must not dispatch at all.
/// Differential: fractional inputs across -O0/-O2/-O3 must be native == interp (both backends
/// marshal the identical `wukong_sgemv_alpha` kernel), and an exact-arithmetic golden case pins
/// the value (all inputs halves/quarters, so the dot is exact under any association).
#[test]
fn scaled_gemv_dispatches_and_matches() {
    let scaled = |attr: &str, store: &str| {
        format!(
            "module m\n{attr}fn scores(k: [f32; 256], q: [f32; 32], mut y: [f32; 8]) {{ \
             for i in 0..8 {{ let mut acc: f32 = 0.0; \
             for d in 0..32 {{ acc = acc + k[i*32+d]*q[d]; }} \
             y[i] = {store}; }} }}"
        )
    };
    assert!(
        lowered_calls(&scaled("", "acc * 0.125"), "wukong_sgemv_alpha"),
        "literal-scaled store -> wukong_sgemv_alpha"
    );
    assert!(
        lowered_calls(
            &scaled("@parallel\n", "acc * 0.125"),
            "wukong_sgemv_alpha_parallel"
        ),
        "@parallel scaled store -> wukong_sgemv_alpha_parallel"
    );
    // The plain store still emits the exact unscaled call.
    assert!(
        lowered_calls(&scaled("", "acc"), "wukong_sgemv"),
        "plain store -> wukong_sgemv"
    );
    assert!(
        !lowered_calls(&scaled("", "acc"), "wukong_sgemv_alpha"),
        "plain store must not pick the alpha kernel"
    );
    // A store that is not `acc * c` must not dispatch (falls to the scalar/vectorized nest, which
    // computes it correctly) — never an unscaled kernel that would silently drop the `+ 0.125`.
    assert!(
        !lowered_calls(&scaled("", "acc + 0.125"), "wukong_sgemv")
            && !lowered_calls(&scaled("", "acc + 0.125"), "wukong_sgemv_alpha"),
        "an additive store is not a scaled GEMV"
    );
    // A per-row (loop-variant) scale must not dispatch either.
    assert!(
        !lowered_calls(&scaled("", "acc * q[i]"), "wukong_sgemv_alpha"),
        "an indexed scale is not loop-invariant"
    );

    // Differential with fractional inputs (M=9, N=33 straddle the kernel's 32/8 unroll edges).
    let src = "fn scores(k: [f32; 297], q: [f32; 33], mut y: [f32; 9]) { \
         for i in 0..9 { let mut acc: f32 = 0.0; \
         for d in 0..33 { acc = acc + k[i*33+d]*q[d]; } \
         y[i] = acc * 0.37; } } \
         fn main() -> i32 { let mut k: [f32; 297] = [0.0; 297]; let mut q: [f32; 33] = [0.0; 33]; \
         let mut y: [f32; 9] = [0.0; 9]; \
         for t in 0..297 { k[t] = ((t as f32) * 0.013) - 1.7; } \
         for t in 0..33 { q[t] = ((t as f32) * 0.11) - 1.3; } \
         scores(k, q, y); \
         let mut s: f32 = 0.0; for t in 0..9 { s = s + y[t]; } \
         print((s * 1000.0) as i32); return 0; }";
    for opt in [0u8, 2, 3] {
        assert_eq!(
            jit(src, opt).expect("jit"),
            interp(src, opt).expect("interp"),
            "scaled gemv native vs interp at -O{opt}"
        );
    }

    // @parallel differential: M=300 exceeds GEMV_PAR_MIN_ROWS, so the real multicore kernel runs
    // natively while the interpreter marshals the serial one — bit-identical (rows independent).
    let par = "@parallel\nfn scores(k: [f32; 4800], q: [f32; 16], mut y: [f32; 300]) { \
         for i in 0..300 { let mut acc: f32 = 0.0; \
         for d in 0..16 { acc = acc + k[i*16+d]*q[d]; } \
         y[i] = acc * 0.37; } } \
         fn main() -> i32 { let mut k: [f32; 4800] = [0.0; 4800]; let mut q: [f32; 16] = [0.0; 16]; \
         let mut y: [f32; 300] = [0.0; 300]; \
         for t in 0..4800 { k[t] = (((t % 37) as f32) * 0.31) - 1.9; } \
         for t in 0..16 { q[t] = ((t as f32) * 0.17) - 0.9; } \
         scores(k, q, y); \
         let mut s: f32 = 0.0; for t in 0..300 { s = s + y[t]; } \
         print((s * 100.0) as i32); return 0; }";
    for opt in [0u8, 2] {
        assert_eq!(
            jit(par, opt).expect("jit"),
            interp(par, opt).expect("interp"),
            "@parallel scaled gemv native vs interp at -O{opt}"
        );
    }

    // Exact-arithmetic golden value: dots 18 and 3.5, x0.125 -> 2.25 and 0.4375.
    let golden = "fn scores(k: [f32; 8], q: [f32; 4], mut y: [f32; 2]) { \
         for i in 0..2 { let mut acc: f32 = 0.0; \
         for d in 0..4 { acc = acc + k[i*4+d]*q[d]; } \
         y[i] = acc * 0.125; } } \
         fn main() -> i32 { let k: [f32; 8] = [2.0, 3.0, 4.0, 5.0, 1.0, 1.0, 1.0, 1.0]; \
         let q: [f32; 4] = [1.0, 0.0 - 2.0, 0.5, 4.0]; let mut y: [f32; 2] = [0.0; 2]; \
         scores(k, q, y); print((y[0] * 100.0) as i32); print((y[1] * 100.0) as i32); return 0; }";
    let (_, out) = jit(golden, 2).expect("jit golden");
    assert_eq!(
        String::from_utf8(out).unwrap(),
        "225\n43\n",
        "scaled gemv produced the wrong value"
    );
}

/// The vector·matrix nest `for j { let s=0; for i { s += w[i]*a[i*N+j] }; out[j] = s [* c] }` (the
/// KV-decode attention read-out `out = scoresᵀ·V`) must dispatch to `wukong_sgevm_f32` (the
/// `@parallel` form to its `_parallel` twin), for both the plain and the α-scaled store. Negatives:
/// an output aliasing an input, a non-scale store, and an outer-indexed weight must NOT dispatch.
/// Differential: fractional inputs across -O0/-O2/-O3 must be native == interp (both marshal the
/// identical kernel — which is itself bit-exact vs the FMA-contracted scalar nest: same per-element
/// ascending-i chain), plus an exact-arithmetic golden value.
#[test]
fn gevm_dispatches_and_matches() {
    let gevm = |attr: &str, store: &str| {
        format!(
            "module m\n{attr}fn readout(w: [f32; 16], a: [f32; 512], mut out: [f32; 32]) {{ \
             for j in 0..32 {{ let mut s: f32 = 0.0; \
             for i in 0..16 {{ s = s + w[i]*a[i*32+j]; }} \
             out[j] = {store}; }} }}"
        )
    };
    assert!(
        lowered_calls(&gevm("", "s"), "wukong_sgevm_f32"),
        "vector-matrix nest -> wukong_sgevm_f32"
    );
    assert!(
        lowered_calls(&gevm("", "s * 0.125"), "wukong_sgevm_f32"),
        "alpha-scaled vector-matrix store -> wukong_sgevm_f32"
    );
    assert!(
        lowered_calls(&gevm("@parallel\n", "s"), "wukong_sgevm_f32_parallel"),
        "@parallel vector-matrix -> wukong_sgevm_f32_parallel"
    );
    // The output aliasing the matrix input must NOT dispatch (the kernel accumulates INTO out
    // across the i sweep, which would read the aliased input mid-update).
    let aliased = "module m\nfn f(w: [f32; 16], mut a: [f32; 512]) { \
         for j in 0..32 { let mut s: f32 = 0.0; \
         for i in 0..16 { s = s + w[i]*a[i*32+j]; } \
         a[j] = s; } }";
    assert!(
        !lowered_calls(aliased, "wukong_sgevm_f32"),
        "an output aliasing the matrix must not dispatch"
    );
    // A store that is not `s` or `s * c` must not dispatch.
    assert!(
        !lowered_calls(&gevm("", "s + 1.0"), "wukong_sgevm_f32"),
        "an additive store is not a vector-matrix product"
    );
    // A weight indexed by the OUTER var (`w[j]*a[i*32+j]` — an elementwise scale of a column sum,
    // not a contraction over i) must not dispatch.
    let outer_w = "module m\nfn f(w: [f32; 32], a: [f32; 512], mut out: [f32; 32]) { \
         for j in 0..32 { let mut s: f32 = 0.0; \
         for i in 0..16 { s = s + w[j]*a[i*32+j]; } \
         out[j] = s; } }";
    assert!(
        !lowered_calls(outer_w, "wukong_sgevm_f32"),
        "an outer-indexed weight is not the contraction"
    );

    // Differential with fractional inputs (rows=37, cols=65 straddle the 8-lane edge and stripe
    // boundaries); the store carries the alpha scale so the fold + finalize are both exercised.
    let src = "fn readout(w: [f32; 37], a: [f32; 2405], mut out: [f32; 65]) { \
         for j in 0..65 { let mut s: f32 = 0.0; \
         for i in 0..37 { s = s + w[i]*a[i*65+j]; } \
         out[j] = s * 0.37; } } \
         fn main() -> i32 { let mut w: [f32; 37] = [0.0; 37]; let mut a: [f32; 2405] = [0.0; 2405]; \
         let mut out: [f32; 65] = [0.0; 65]; \
         for t in 0..37 { w[t] = ((t as f32) * 0.13) - 1.1; } \
         for t in 0..2405 { a[t] = (((t % 41) as f32) * 0.29) - 2.3; } \
         readout(w, a, out); \
         let mut s: f32 = 0.0; for t in 0..65 { s = s + out[t]; } \
         print((s * 100.0) as i32); return 0; }";
    for opt in [0u8, 2, 3] {
        assert_eq!(
            jit(src, opt).expect("jit"),
            interp(src, opt).expect("interp"),
            "gevm native vs interp at -O{opt}"
        );
    }

    // @parallel differential: cols=300 exceeds GEVM_PAR_MIN, so the real column-striped multicore
    // kernel runs natively while the interpreter marshals the serial one — bit-identical (each
    // column's ascending-i chain is self-contained regardless of the stripe split).
    let par = "@parallel\nfn readout(w: [f32; 16], a: [f32; 4800], mut out: [f32; 300]) { \
         for j in 0..300 { let mut s: f32 = 0.0; \
         for i in 0..16 { s = s + w[i]*a[i*300+j]; } \
         out[j] = s; } } \
         fn main() -> i32 { let mut w: [f32; 16] = [0.0; 16]; let mut a: [f32; 4800] = [0.0; 4800]; \
         let mut out: [f32; 300] = [0.0; 300]; \
         for t in 0..16 { w[t] = ((t as f32) * 0.21) - 1.5; } \
         for t in 0..4800 { a[t] = (((t % 23) as f32) * 0.17) - 1.9; } \
         readout(w, a, out); \
         let mut s: f32 = 0.0; for t in 0..300 { s = s + out[t]; } \
         print((s * 100.0) as i32); return 0; }";
    for opt in [0u8, 2] {
        assert_eq!(
            jit(par, opt).expect("jit"),
            interp(par, opt).expect("interp"),
            "@parallel gevm native vs interp at -O{opt}"
        );
    }

    // Exact-arithmetic golden: w=[0.5,-1,2] over rows [1,-2,3,.5]/[2,1,-1,4]/[1.5,-1,4,1.25]
    // -> out = [1.5, -4, 10.5, -1.25].
    let golden = "fn readout(w: [f32; 3], v: [f32; 12], mut out: [f32; 4]) { \
         for d in 0..4 { let mut acc: f32 = 0.0; \
         for s in 0..3 { acc = acc + w[s]*v[s*4+d]; } \
         out[d] = acc; } } \
         fn main() -> i32 { let w: [f32; 3] = [0.5, 0.0 - 1.0, 2.0]; \
         let v: [f32; 12] = [1.0, 0.0 - 2.0, 3.0, 0.5, 2.0, 1.0, 0.0 - 1.0, 4.0, 1.5, 0.0 - 1.0, 4.0, 1.25]; \
         let mut out: [f32; 4] = [0.0; 4]; readout(w, v, out); \
         print((out[0] * 100.0) as i32); print((out[2] * 100.0) as i32); return 0; }";
    let (_, out) = jit(golden, 2).expect("jit golden");
    assert_eq!(
        String::from_utf8(out).unwrap(),
        "150\n1050\n",
        "gevm produced the wrong value"
    );
}

/// The full single-token KV-decode attention body (q[D], K[S,D], V[S,D]; S=128, D=64 — the CPU LLM
/// serving hot path) must dispatch ALL THREE stages: the scaled score GEMV -> wukong_sgemv_alpha,
/// the in-place softmax window -> wukong_norm_f32, and the read-out -> wukong_sgevm_f32.
#[test]
fn decode_attention_dispatches_all_kernels() {
    let src = "module m\n\
         fn decode_attn(q: [f32; 64], k: [f32; 8192], v: [f32; 8192], \
                        mut scores: [f32; 128], mut out: [f32; 64]) { \
         for s in 0..128 { let mut acc: f32 = 0.0; \
         for d in 0..64 { acc = acc + k[s*64+d]*q[d]; } \
         scores[s] = acc * 0.125; } \
         let mut m: f32 = scores[0]; \
         for i in 0..128 { m = fmax(m, scores[i]); } \
         for i in 0..128 { scores[i] = exp(scores[i] - m); } \
         let mut ssum: f32 = 0.0; \
         for i in 0..128 { ssum = ssum + scores[i]; } \
         let inv: f32 = 1.0 / ssum; \
         for i in 0..128 { scores[i] = scores[i] * inv; } \
         for d in 0..64 { let mut acc: f32 = 0.0; \
         for s in 0..128 { acc = acc + scores[s]*v[s*64+d]; } \
         out[d] = acc; } }";
    assert!(
        lowered_calls(src, "wukong_sgemv_alpha"),
        "decode scores -> wukong_sgemv_alpha"
    );
    assert!(
        lowered_calls(src, "wukong_norm_f32"),
        "decode softmax -> wukong_norm_f32"
    );
    assert!(
        lowered_calls(src, "wukong_sgevm_f32"),
        "decode read-out -> wukong_sgevm_f32"
    );
}

/// A mid-function `@parallel` head-attention loop (per-head scratch declared INSIDE the loop body,
/// disjoint hh-sliced output writes) must outline into a `wukong_parallel_for` region, and the
/// result must be bit-exact three ways: the @parallel fn against its serial twin in the same
/// program (G4 — the fixture compares elementwise and prints the mismatch count), the native
/// backend against the interpreter oracle (G1), and every opt level against -O0 (G2). A ragged
/// head count (3 heads across the core pool's uneven chunks) runs the same gate.
#[test]
fn parallel_head_region_matches_interpreter() {
    let body = |d: usize, hd: usize, h: usize, s: usize| {
        let (shd, ss) = (s * hd, s * s);
        format!(
            "for hh in 0..{h} {{ \
               let mut qh: [f32; {shd}] = [0.0; {shd}]; \
               let mut kh: [f32; {shd}] = [0.0; {shd}]; \
               let mut scores: [f32; {ss}] = [0.0; {ss}]; \
               for i in 0..{s} {{ for p in 0..{hd} {{ qh[i*{hd}+p] = q[i*{d} + hh*{hd} + p]; }} }} \
               for i in 0..{s} {{ for p in 0..{hd} {{ kh[i*{hd}+p] = k[i*{d} + hh*{hd} + p]; }} }} \
               for i in 0..{s} {{ for j in 0..{s} {{ \
                 let mut acc: f32 = 0.0; \
                 for p in 0..{hd} {{ acc = acc + qh[i*{hd}+p] * kh[j*{hd}+p]; }} \
                 scores[i*{s}+j] = acc; \
               }} }} \
               for i in 0..{s} {{ for j in 0..{hd} {{ \
                 attn[i*{d} + hh*{hd} + j] = scores[i*{s}] + v[i*{d} + hh*{hd} + j]; \
               }} }} \
             }}"
        )
    };
    let src = |d: usize, hd: usize, h: usize, s: usize| {
        let sd = s * d;
        let inner = body(d, hd, h, s);
        format!(
            "module m \
             fn heads_serial(q: [f32; {sd}], k: [f32; {sd}], v: [f32; {sd}], mut attn: [f32; {sd}]) {{ {inner} }} \
             @parallel \
             fn heads_par(q: [f32; {sd}], k: [f32; {sd}], v: [f32; {sd}], mut attn: [f32; {sd}], mut y: [f32; {sd}]) {{ \
               for i in 0..{sd} {{ y[i] = q[i]; }} \
               {inner} \
             }} \
             fn main() -> i32 {{ \
               let mut q: [f32; {sd}] = [0.0; {sd}]; \
               let mut k: [f32; {sd}] = [0.0; {sd}]; \
               let mut v: [f32; {sd}] = [0.0; {sd}]; \
               for i in 0..{sd} {{ \
                 q[i] = (i as f32) * 0.25; \
                 k[i] = 8.0 - (i as f32) * 0.5; \
                 v[i] = (i as f32) - 16.0; \
               }} \
               let mut a_s: [f32; {sd}] = [0.0; {sd}]; \
               let mut a_p: [f32; {sd}] = [0.0; {sd}]; \
               let mut y: [f32; {sd}] = [0.0; {sd}]; \
               heads_serial(q, k, v, a_s); \
               heads_par(q, k, v, a_p, y); \
               let mut bad: i32 = 0; \
               for i in 0..{sd} {{ if a_p[i] != a_s[i] {{ bad = bad + 1; }} }} \
               print(bad); \
               let mut cs: f32 = 0.0; \
               for i in 0..{sd} {{ cs = cs + a_p[i]; }} \
               return (cs as i32) % 100000; \
             }}"
        )
    };
    // Even (2 heads) and ragged (3 heads) trip counts.
    for (d, hd, h, s) in [(8usize, 4usize, 2usize, 4usize), (6, 2, 3, 4)] {
        let program = src(d, hd, h, s);
        assert!(
            lowered_calls(&program, "wukong_parallel_for"),
            "H={h}: the @parallel head loop must outline into a parallel_for region"
        );
        let (_, out) = jit_ok(&program);
        assert!(
            out.starts_with("0\n"),
            "H={h}: serial vs @parallel must be bit-exact, got mismatches: {out}"
        );
        for opt in [0u8, 1, 2, 3] {
            assert_eq!(
                jit(&program, opt).unwrap(),
                interp(&program, opt).unwrap(),
                "H={h}: head-region native vs interp mismatch at -O{opt}"
            );
        }
    }
}

// ---------------------------------------------------------------------------------------------------
// Backend compile-time levers: byte-identity gates
//
// Every `emit_object` compile-time lever (IR verifier on/off, serial vs parallel per-function
// codegen) MUST leave the emitted object bytes bit-for-bit identical. These tests pin that: they
// compile multi-function programs (parallel codegen only engages with >1 function) every which way
// and diff the raw object bytes. If any of these ever fails, a lever changed observable output and
// must not ship.
// ---------------------------------------------------------------------------------------------------

/// Full front-end + `-O2` to an optimized MIR program (+ its interner) for the byte-identity gates.
fn program_o2(src: &str) -> (wukong_mir::Program, Interner) {
    let mut interner = Interner::new();
    let (module, pd) = wukong_parser::parse_module(src, SourceId(0), &mut interner);
    assert!(pd.iter().all(|d| !d.is_error()), "parse: {pd:?}");
    let (sema, sd) = wukong_sema::check(&module, &interner);
    assert!(sd.iter().all(|d| !d.is_error()), "sema: {sd:?}");
    let (mut program, ld) = wukong_mir_build::lower_program(&module, &sema, &mut interner);
    assert!(ld.iter().all(|d| !d.is_error()), "lower: {ld:?}");
    wukong_opt::optimize(&mut program, 2);
    (program, interner)
}

/// A handful of multi-function programs with calls, statics (string literals), recursion, and a
/// recognized GEMM kernel — so relocations, imported symbols, and vector kernels are all exercised.
/// The two `examples/*.wk` models compile to two functions each (parallel path engages).
const BYTE_ID_SRCS: &[&str] = &[
    // Calls + recursion + a string static.
    r#"
fn fib(n: i64) -> i64 { if n < 2 { return n; } return fib(n - 1) + fib(n - 2); }
fn twice(n: i64) -> i64 { return n + n; }
fn main() {
    let a: i64 = fib(10);
    let b: i64 = twice(a);
    println("fib+twice:");
    print(b);
}
"#,
    // Two helpers feeding main; mixed float/int arithmetic.
    r#"
fn sq(x: f64) -> f64 { return x * x; }
fn poly(x: f64) -> f64 { return sq(x) + sq(x) * x - 1.0; }
fn main() { let r: f64 = poly(3.0); print(r); }
"#,
    include_str!("../../../examples/gpt2.wk"),
    include_str!("../../../examples/llama_block.wk"),
    include_str!("../../../examples/gemm.wk"),
];

#[test]
fn serial_parallel_object_bytes_identical() {
    for (i, src) in BYTE_ID_SRCS.iter().enumerate() {
        let (program, interner) = program_o2(src);
        let serial = crate::emit_object_ex(
            &program,
            &interner,
            crate::EmitOptions { verify: false, parallel: false },
        )
        .unwrap_or_else(|e| panic!("src {i} serial: {e}"))
        .0;
        let parallel = crate::emit_object_ex(
            &program,
            &interner,
            crate::EmitOptions { verify: false, parallel: true },
        )
        .unwrap_or_else(|e| panic!("src {i} parallel: {e}"))
        .0;
        assert_eq!(
            serial, parallel,
            "src {i}: parallel per-function codegen changed the object bytes ({} vs {} bytes)",
            serial.len(),
            parallel.len()
        );
    }
}

#[test]
fn verifier_toggle_object_bytes_identical() {
    for (i, src) in BYTE_ID_SRCS.iter().enumerate() {
        let (program, interner) = program_o2(src);
        let verified = crate::emit_object_ex(
            &program,
            &interner,
            crate::EmitOptions { verify: true, parallel: false },
        )
        .unwrap_or_else(|e| panic!("src {i} verify: {e}"))
        .0;
        let unverified = crate::emit_object_ex(
            &program,
            &interner,
            crate::EmitOptions { verify: false, parallel: false },
        )
        .unwrap_or_else(|e| panic!("src {i} noverify: {e}"))
        .0;
        assert_eq!(
            verified, unverified,
            "src {i}: disabling the IR verifier changed the object bytes"
        );
    }
}

/// Determinism: the parallel path must emit identical bytes on every run (worker scheduling must not
/// leak into output). Compile the same program several times in parallel and diff.
#[test]
fn parallel_object_bytes_deterministic() {
    let (program, interner) = program_o2(include_str!("../../../examples/gpt2.wk"));
    let first = crate::emit_object_ex(
        &program,
        &interner,
        crate::EmitOptions { verify: false, parallel: true },
    )
    .unwrap()
    .0;
    for _ in 0..8 {
        let again = crate::emit_object_ex(
            &program,
            &interner,
            crate::EmitOptions { verify: false, parallel: true },
        )
        .unwrap()
        .0;
        assert_eq!(first, again, "parallel codegen produced nondeterministic object bytes");
    }
}

/// Determinism of the FAILURE path, not just the success path: when two functions both fail to
/// lower, the reported diagnostic must be the lowest-source-order one every time, and must match
/// what the serial path reports. Collecting the rayon results straight into `Result<Vec<_>, _>`
/// short-circuits on whichever worker lost the race, so this program alternated between the
/// `[... x i32]` and `[... x i64]` messages across identical invocations of the same binary.
#[test]
fn parallel_codegen_error_is_source_ordered() {
    // Enough functions to give rayon a real split tree, each with a stack slot over the 4 GiB
    // layout limit and a distinct extent so the message identifies which one was reported. `f0`
    // reaches its oversized slot only after a long body, so in wall-clock order it is the LAST of
    // the batch to fail — exactly the case where "first worker to report" and "first in source
    // order" disagree.
    const N: u64 = 32;
    let mut src = String::new();
    for k in 0..N {
        let n = 2_000_000_000u64 + k;
        let mut body = String::new();
        if k == 0 {
            // Seeded from a parameter so `-O2` cannot constant-fold the chain away.
            for j in 0..3000 {
                body.push_str(&format!("acc = acc * 3 + {j};"));
            }
        }
        src.push_str(&format!(
            "fn f{k}(mut acc: i32) -> i32 {{ {body} let x: [i32; {n}] = [0; {n}]; \
             return x[0] + acc; }}\n"
        ));
    }
    src.push_str("fn main() -> i32 {");
    for k in 0..N {
        src.push_str(&format!(" print(f{k}({k}));"));
    }
    src.push_str(" return 0; }");
    let (program, interner) = program_o2(&src);
    let serial = crate::emit_object_ex(
        &program,
        &interner,
        crate::EmitOptions { verify: false, parallel: false },
    )
    .map(|_| ())
    .expect_err("an oversized stack slot must not compile");
    assert!(
        serial.contains("2000000000"),
        "serial reported a later function than the first: {serial}"
    );
    for _ in 0..24 {
        let parallel = crate::emit_object_ex(
            &program,
            &interner,
            crate::EmitOptions { verify: false, parallel: true },
        )
        .map(|_| ())
        .expect_err("an oversized stack slot must not compile");
        assert_eq!(
            serial, parallel,
            "parallel codegen reported a different failing function than serial"
        );
    }
}

// ---------------------------------------------------------------------------------------------------
// Backend compile-time A/B harness (ignored; run in RELEASE with --nocapture).
//
//   cargo test -p wukong_codegen_cranelift --release -- --ignored --nocapture backend_compile_ab
//
// Same-run interleaved best-of-N ratios only (this laptop's clock is not reportable in absolutes; a
// ratio measured A/B/A/B adjacently survives clock/thermal drift). Reports, over the corpus:
//   * verifier off vs on  (the config-audit lever)
//   * parallel vs serial per-function codegen  (the big lever)
//   * the codegen-vs-object-write split (attack 3) and the object-container floor (attack 6)
// It also re-asserts byte-identity per file, so it doubles as a corpus-wide gate.
// ---------------------------------------------------------------------------------------------------

#[cfg(test)]
fn corpus_files() -> Vec<std::path::PathBuf> {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..");
    let mut out = Vec::new();
    for dir in ["tests/run", "examples", "bench/kernels"] {
        if let Ok(rd) = std::fs::read_dir(root.join(dir)) {
            for e in rd.flatten() {
                let p = e.path();
                if p.extension().and_then(|s| s.to_str()) == Some("wk") {
                    out.push(p);
                }
            }
        }
    }
    out.sort();
    out
}

/// Best-of-N minimum wall time of `run`, ~`budget_ms` total, min 5 reps. The minimum is the run least
/// perturbed by scheduler noise.
#[cfg(test)]
fn best_of_ms(budget_ms: u64, mut run: impl FnMut() -> std::time::Duration) -> f64 {
    use std::time::Instant;
    run();
    let mut best = f64::MAX;
    let start = Instant::now();
    let mut reps = 0u32;
    loop {
        best = best.min(run().as_secs_f64() * 1e3);
        reps += 1;
        if reps >= 5 && start.elapsed().as_millis() as u64 >= budget_ms {
            break;
        }
        if reps >= 400 {
            break;
        }
    }
    best
}

#[test]
#[ignore]
fn backend_compile_ab() {
    use std::time::{Duration, Instant};

    let files = corpus_files();
    // Compile the front-end once per file; hold the optimized programs in memory so the A/B times
    // only the backend.
    let mut progs: Vec<(String, wukong_mir::Program, Interner)> = Vec::new();
    let mut multi = 0usize;
    for path in &files {
        let Ok(src) = std::fs::read_to_string(path) else {
            continue;
        };
        // Skip files that don't cleanly reach an object (mirrors compile-profile).
        let ok = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let mut interner = Interner::new();
            let (m, pd) = wukong_parser::parse_module(&src, SourceId(0), &mut interner);
            if pd.iter().any(|d| d.is_error()) {
                return None;
            }
            let (s, sd) = wukong_sema::check(&m, &interner);
            if sd.iter().any(|d| d.is_error()) {
                return None;
            }
            let (mut p, ld) = wukong_mir_build::lower_program(&m, &s, &mut interner);
            if ld.iter().any(|d| d.is_error()) {
                return None;
            }
            wukong_opt::optimize(&mut p, 2);
            crate::emit_object_ex(&p, &interner, crate::EmitOptions { verify: false, parallel: false })
                .ok()?;
            Some((p, interner))
        }))
        .ok()
        .flatten();
        if let Some((p, interner)) = ok {
            if p.funcs.len() > 1 {
                multi += 1;
            }
            progs.push((
                path.file_name().unwrap().to_string_lossy().into_owned(),
                p,
                interner,
            ));
        }
    }

    let ser = crate::EmitOptions { verify: false, parallel: false };
    let ver = crate::EmitOptions { verify: true, parallel: false };
    let par = crate::EmitOptions { verify: false, parallel: true };
    let emit = |p: &wukong_mir::Program, i: &Interner, o: crate::EmitOptions| {
        crate::emit_object_ex(p, i, o).unwrap()
    };

    // Geomean accumulators (sum of ln ratio).
    let (mut ln_ver, mut ln_par, mut n_ver, mut n_par) = (0.0f64, 0.0f64, 0u32, 0u32);
    let mut tot_cg = Duration::ZERO;
    let mut tot_ow = Duration::ZERO;
    let mut tot_obj = 0usize;
    let mut n_funcs = 0usize;

    // Per-file heavy table.
    struct Row {
        name: String,
        funcs: usize,
        obj: usize,
        ver_ratio: f64,
        par_ratio: f64,
        ow_share: f64,
        base_ms: f64,
    }
    let mut rows: Vec<Row> = Vec::new();

    for (name, p, i) in &progs {
        // Byte-identity gate (corpus-wide).
        let b_ser = emit(p, i, ser).0;
        let b_ver = emit(p, i, ver).0;
        let b_par = emit(p, i, par).0;
        assert_eq!(b_ser, b_ver, "{name}: verifier changed bytes");
        assert_eq!(b_ser, b_par, "{name}: parallel changed bytes");
        tot_obj += b_ser.len();
        n_funcs += p.funcs.len();

        // codegen vs object-write split (verify off = shipping config).
        let t = emit(p, i, ser).1;
        tot_cg += t.codegen;
        tot_ow += t.object_write;

        // A/B: verifier ON vs OFF, adjacent best-of.
        let t_ver = best_of_ms(30, || {
            let s = Instant::now();
            let _ = crate::emit_object_ex(p, i, ver);
            s.elapsed()
        });
        let t_ser = best_of_ms(30, || {
            let s = Instant::now();
            let _ = crate::emit_object_ex(p, i, ser);
            s.elapsed()
        });
        // A/B: parallel vs serial (verify off), adjacent best-of.
        let t_par = best_of_ms(30, || {
            let s = Instant::now();
            let _ = crate::emit_object_ex(p, i, par);
            s.elapsed()
        });
        let t_ser2 = best_of_ms(30, || {
            let s = Instant::now();
            let _ = crate::emit_object_ex(p, i, ser);
            s.elapsed()
        });

        let ver_ratio = t_ser / t_ver; // <1 means verify-off is faster (ser faster than ver)
        let par_ratio = t_par / t_ser2; // <1 means parallel faster
        ln_ver += (t_ver / t_ser).ln();
        n_ver += 1;
        ln_par += (t_ser2 / t_par).ln();
        n_par += 1;
        let ow_share = t.object_write.as_secs_f64()
            / (t.codegen.as_secs_f64() + t.object_write.as_secs_f64()).max(1e-12);
        rows.push(Row {
            name: name.clone(),
            funcs: p.funcs.len(),
            obj: b_ser.len(),
            ver_ratio,
            par_ratio,
            ow_share,
            base_ms: t_ser2,
        });
    }

    // ISA-build fixed cost (attack 1: is the ISA worth caching across emit_object calls?).
    let t_isa = best_of_ms(200, || {
        let s = Instant::now();
        let _ = crate::make_isa_verify(true, false).unwrap();
        s.elapsed()
    });
    // A representative small single-function file's full backend time, to size the ISA share of the
    // fixed floor.
    let small_ms = rows
        .iter()
        .filter(|r| r.funcs == 1)
        .map(|r| r.base_ms)
        .fold(f64::MAX, f64::min);
    println!(
        "ISA build (make_isa) fixed cost: {t_isa:.4} ms  |  smallest 1-func backend: {small_ms:.4} ms  => ISA is {:.1}% of that floor",
        100.0 * t_isa / small_ms.max(1e-9)
    );

    let g_ver = (ln_ver / n_ver.max(1) as f64).exp(); // verify-on / verify-off (>1 = verifier costs)
    let g_par = (ln_par / n_par.max(1) as f64).exp(); // serial / parallel (>1 = parallel wins)
    println!("\n================ backend compile A/B (same-run best-of-N ratios) ================");
    println!(
        "corpus: {} programs ({} multi-function), {} total funcs, {} object bytes",
        progs.len(),
        multi,
        n_funcs,
        tot_obj
    );
    println!(
        "codegen vs object-write (verify off): codegen {:.1}% | object-write {:.1}%  (obj-write is the COFF-container floor)",
        100.0 * tot_cg.as_secs_f64() / (tot_cg + tot_ow).as_secs_f64(),
        100.0 * tot_ow.as_secs_f64() / (tot_cg + tot_ow).as_secs_f64(),
    );
    println!(
        "VERIFIER lever  (geomean verify-on / verify-off): {:.3}x   => disabling it is {:.1}% faster",
        g_ver,
        100.0 * (1.0 - 1.0 / g_ver)
    );
    println!(
        "PARALLEL lever  (geomean serial / parallel):      {:.3}x   => parallel is {:.1}% faster",
        g_par,
        100.0 * (1.0 - 1.0 / g_par)
    );

    rows.sort_by(|a, b| b.base_ms.partial_cmp(&a.base_ms).unwrap());
    println!(
        "\n{:<26} {:>5} {:>7} {:>9} {:>9} {:>9} {:>10}",
        "heaviest file", "funcs", "obj", "base(ms)", "ver x", "par x", "objwr%"
    );
    println!("{}", "-".repeat(80));
    for r in rows.iter().take(14) {
        println!(
            "{:<26} {:>5} {:>7} {:>9.3} {:>9.3} {:>9.3} {:>9.1}%",
            r.name,
            r.funcs,
            r.obj,
            r.base_ms,
            r.ver_ratio,
            r.par_ratio,
            100.0 * r.ow_share
        );
    }
    println!("(ver x = serial/verify: <1 verifier-off faster; par x = parallel/serial: <1 parallel faster)");
}
