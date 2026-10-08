//! The backend configuration knobs that trade compile time for code speed or concurrency —
//! [`BackendOpt`] (Cranelift `opt_level`) and parallel JIT population — must never change what a
//! program computes. The interpreter is the oracle: every configuration has to agree with it
//! bit-for-bit on stdout and exit code, at the MIR the driver hands over at `-O0` and at `-O2`.

use super::*;

fn lower(src: &str, opt: u8) -> (Program, Interner, Symbol) {
    let mut interner = Interner::new();
    let (module, parsed) =
        wukong_parser::parse_module(src, wukong_span::SourceId(0), &mut interner);
    assert!(parsed.iter().all(|d| !d.is_error()), "{parsed:?}");
    let (sema, checked) = wukong_sema::check(&module, &interner);
    assert!(checked.iter().all(|d| !d.is_error()), "{checked:?}");
    let (mut program, lowered) = wukong_mir_build::lower_program(&module, &sema, &mut interner);
    assert!(lowered.iter().all(|d| !d.is_error()), "{lowered:?}");
    wukong_opt::optimize(&mut program, opt);
    assert!(wukong_mir::verify::verify_program(&program).is_empty());
    let main = interner.intern("main");
    (program, interner, main)
}

/// Programs that exercise what an optimizing mid-end could plausibly disturb: recursion, float
/// recurrences printed at every step (any reassociation or contraction would show in the low
/// bits), arrays with loops and branches spread over several functions, and aggregate copies.
const SOURCES: &[&str] = &[
    "fn fib(n: i64) -> i64 { if n < 2 { return n; } return fib(n - 1) + fib(n - 2); }
     fn main() -> i32 { print(fib(20)); return 7; }",
    "fn main() -> i32 {
        let mut x: f32 = 0.1;
        let mut acc: f64 = 0.0;
        for i in 0..40 {
            x = x * 1.7 - 0.3 / (x + 2.0);
            acc = acc + (x as f64) * 0.5;
            print(x);
        }
        print(acc);
        return 0;
    }",
    "fn sq(v: f32) -> f32 { return v * v; }
     fn clampi(v: i32, lo: i32, hi: i32) -> i32 {
        if v < lo { return lo; }
        if v > hi { return hi; }
        return v;
     }
     fn main() -> i32 {
        let mut a: [f32; 32] = [0.0; 32];
        for i in 0..32 { a[i] = sq((i as f32) * 0.25 - 3.0); }
        let mut s: f32 = 0.0;
        for i in 0..32 { s = s + a[i]; print(a[i]); }
        print(s);
        print(clampi(-5, 0, 9));
        print(clampi(15, 0, 9));
        print(clampi(4, 0, 9));
        return clampi(12, 0, 9);
     }",
    "struct P { x: i32, y: f32 }
     fn main() -> i32 {
        let a: [P; 12] = [P { x: 3, y: 0.5 }; 12];
        let mut b = a;
        b[5].x = 9;
        for i in 0..12 { print(b[i].x); print(b[i].y); }
        return b[5].x + a[5].x;
     }",
];

#[test]
fn backend_opt_level_maps_to_cranelift_opt_level() {
    use cranelift_codegen::settings::OptLevel;
    assert_eq!(BackendOpt::for_level(0).cranelift_opt_level(), "none");
    for level in 1..=3 {
        assert_eq!(BackendOpt::for_level(level).cranelift_opt_level(), "speed");
    }
    assert_eq!(BackendOpt::default().cranelift_opt_level(), "speed");
    let none = make_isa_with(false, false, BackendOpt::for_level(0)).unwrap();
    assert_eq!(none.flags().opt_level(), OptLevel::None);
    let speed = make_isa_with(true, false, BackendOpt::default()).unwrap();
    assert_eq!(speed.flags().opt_level(), OptLevel::Speed);
}

/// `-O0` code compiled with `opt_level=none` computes exactly what the interpreter and the `speed`
/// backend compute, at both MIR shapes.
#[test]
fn every_backend_opt_level_matches_the_interpreter() {
    for (i, src) in SOURCES.iter().enumerate() {
        for mir_opt in [0, 2] {
            let (program, interner, main) = lower(src, mir_opt);
            let oracle = wukong_interp::run_with_output(&program, main, &interner);
            assert!(oracle.is_ok(), "src {i}: {oracle:?}");
            for level in [0, 1, 2, 3] {
                let native = jit_run_with(&program, main, &interner, BackendOpt::for_level(level));
                assert_eq!(
                    native, oracle,
                    "src {i}, MIR -O{mir_opt}, backend level {level}: native differs from the interpreter"
                );
            }
            let object = emit_object_with(&program, &interner, BackendOpt::for_level(0));
            assert!(object.is_ok(), "src {i}: {object:?}");
        }
    }
}

/// The JIT twin of `serial_parallel_object_bytes_identical`: populating the JIT module from
/// concurrently compiled functions runs exactly like the serial population.
#[test]
fn parallel_jit_population_matches_serial() {
    for (i, src) in SOURCES.iter().enumerate() {
        let (program, interner, main) = lower(src, 2);
        let run = |parallel: bool| {
            jit_compile_entry(
                &program,
                main,
                &interner,
                false,
                crate::TargetCpu::Native,
                BackendOpt::default(),
                parallel,
            )
            .unwrap_or_else(|e| panic!("src {i}, parallel = {parallel}: {e}"))
            .run()
        };
        let serial = run(false);
        assert!(serial.is_ok(), "src {i}: {serial:?}");
        assert_eq!(
            run(true),
            serial,
            "src {i}: parallel JIT population changed the run"
        );
    }
}

/// Callers listed before their callees: the worklist must still mark the whole chain (the old
/// sweep needed one pass per level here), and nothing that cannot reach `assert`.
#[test]
fn assertion_propagation_reaches_the_whole_chain_in_any_order() {
    let (program, mut interner, _) = lower(
        "fn a() -> i32 { return b(); }
         fn b() -> i32 { return c(); }
         fn c() -> i32 { return d(); }
         fn d() -> i32 { assert(true); return 1; }
         fn e() -> i32 { return 2; }
         fn main() -> i32 { return a() + e(); }",
        0,
    );
    let actual = asserting_functions(&program, &interner);
    let expected: HashSet<Symbol> = ["a", "b", "c", "d", "main"]
        .into_iter()
        .map(|name| interner.intern(name))
        .collect();
    assert_eq!(actual, expected);
}
