//! `--target-cpu`: an explicit ISA level must change the code, never the answer, and must never let
//! an object embed instructions its level cannot execute.

use super::*;
use wukong_mir::Builder;

fn lower(src: &str, opt: u8) -> (Program, Interner, Symbol) {
    let mut interner = Interner::new();
    let (module, parsed) =
        wukong_parser::parse_module(src, wukong_span::SourceId(0), &mut interner);
    assert!(parsed.iter().all(|d| !d.is_error()), "{parsed:?}");
    let (sema, checked) = wukong_sema::check(&module, &interner);
    assert!(checked.iter().all(|d| !d.is_error()), "{checked:?}");
    // Exactly what the driver builds for a level below x86-64-v3: no raw-AVX2 recipes.
    let options = wukong_mir_build::LowerOptions {
        vec_kernels: false,
        ..wukong_mir_build::LowerOptions::default()
    };
    let (mut program, lowered) =
        wukong_mir_build::lower_program_with(&module, &sema, &mut interner, options);
    assert!(lowered.iter().all(|d| !d.is_error()), "{lowered:?}");
    wukong_opt::optimize(&mut program, opt);
    assert!(wukong_mir::verify::verify_program(&program).is_empty());
    let main = interner.intern("main");
    (program, interner, main)
}

const LEVELS: [TargetCpu; 4] = [
    TargetCpu::X86_64,
    TargetCpu::X86_64V2,
    TargetCpu::X86_64V3,
    TargetCpu::X86_64V4,
];

/// Scalar FMA contraction, every rounding mode, sqrt, saturating conversions and a 128-bit
/// vectorized loop: the operations whose machine code depends on SSE4.1/AVX/FMA availability.
/// Below those levels Cranelift calls the C runtime's `fmaf`/`nearbyintf`/`floorf`/... instead,
/// which must agree bit-for-bit with the hardware instructions. (The vectorized loops avoid a
/// multiply feeding an add, so the 128-bit path exercised here is plain SSE2 arithmetic.)
const PROGRAM: &str = "
fn main() -> i32 {
    let mut acc: f32 = 0.0;
    let mut i: i32 = 0;
    while i < 23 {
        let x: f32 = (i as f32) * 0.37 - 4.1;
        acc = acc + x * 1.0000001;
        print(round(x));
        print(floor(x));
        print(ceil(x));
        print(trunc(x));
        print(sqrt(x * x + 0.5));
        print((x * 1000000000.0) as i32);
        i = i + 1;
    }
    print(acc);
    let mut a: [f32; 37] = [0.0; 37];
    let mut b: [f32; 37] = [0.0; 37];
    for k in 0..37 { a[k] = (k as f32) / 4.0 - 3.0; }
    for k in 0..37 { b[k] = (a[k] * a[k]) / 7.0 - a[k]; }
    for k in 0..37 { print(b[k]); }
    return 0;
}
";

#[test]
fn target_cpu_names_round_trip() {
    for name in TargetCpu::NAMES {
        assert_eq!(TargetCpu::parse(name).map(TargetCpu::name), Some(*name));
    }
    assert_eq!(TargetCpu::parse("haswell"), None);
    assert_eq!(TargetCpu::default(), TargetCpu::Native);
    assert!(!TargetCpu::X86_64.allows_vec_kernels());
    assert!(!TargetCpu::X86_64V2.allows_vec_kernels());
    assert!(TargetCpu::X86_64V3.allows_vec_kernels());
}

/// Every level the host can execute computes exactly the interpreter's output; a level the host
/// lacks is a classified refusal, never a fault.
#[cfg(target_arch = "x86_64")]
#[test]
fn every_level_runs_the_same_program_as_the_oracle() {
    for opt in [0, 2] {
        let (program, interner, main) = lower(PROGRAM, opt);
        let oracle = wukong_interp::run_with_output(&program, main, &interner).unwrap();
        let native = jit_compile(&program, main, &interner)
            .unwrap()
            .run()
            .unwrap();
        assert_eq!(native, oracle, "-O{opt}: host level");
        for level in LEVELS {
            match jit_compile_for(&program, main, &interner, level) {
                Ok(compiled) => {
                    assert_eq!(compiled.run().unwrap(), oracle, "-O{opt}: {level:?}");
                }
                Err(e) => {
                    assert!(!target::missing_on_host(level).is_empty(), "{level:?}: {e}");
                    assert_eq!(classify_error(&e), NativeErrorKind::Unsupported, "{e}");
                }
            }
        }
    }
}

/// The level is part of the object: a baseline object routes FMA through a call where an
/// x86-64-v3 object uses the instruction, and a level is deterministic.
#[cfg(target_arch = "x86_64")]
#[test]
fn object_code_depends_on_the_level() {
    let (program, interner, _) = lower(PROGRAM, 2);
    let base = emit_object_for(&program, &interner, TargetCpu::X86_64).unwrap();
    let v3 = emit_object_for(&program, &interner, TargetCpu::X86_64V3).unwrap();
    assert_ne!(base, v3);
    assert_eq!(
        base,
        emit_object_for(&program, &interner, TargetCpu::X86_64).unwrap()
    );
}

/// A program carrying a raw AVX2+FMA recipe cannot be emitted for a level without AVX2: the bytes
/// would `#UD` on exactly the machines the level exists for.
#[test]
fn baseline_levels_refuse_raw_avx2_kernels() {
    let mut i = Interner::new();
    let mut b = Builder::new(i.intern("main"), MirType::Void);
    let ptrs = b.add_param(MirType::Ptr);
    let n = b.add_param(MirType::I64);
    let kernel = b.add_vec_kernel(wukong_mir::VecKernel {
        name: i.intern("copy"),
        streams: 2,
        scalars: 0,
        ops: vec![
            wukong_mir::VecOp::Load { stream: 0 },
            wukong_mir::VecOp::Store { stream: 1, val: 0 },
        ],
        unroll: 1,
        reduce: None,
    });
    b.build_void(Op::VecKernelCall {
        kernel,
        ptrs,
        scalars: ptrs,
        n,
    });
    b.ret(None);
    let mut program = Program::new();
    program.funcs.push(b.finish());
    for level in [TargetCpu::X86_64, TargetCpu::X86_64V2] {
        let e = emit_object_for(&program, &i, level).expect_err("AVX2 bytes for a baseline level");
        assert_eq!(classify_error(&e), NativeErrorKind::Unsupported, "{e}");
        assert!(e.contains("x86-64-v3") && e.contains("`main`"), "{e}");
    }
}
