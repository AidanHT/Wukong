//! Per-instruction source locations (`FnTranslator::mark`): every Cranelift instruction carries
//! the `ValueId` of the located MIR value it was lowered for as its `SourceLoc`, so a machine-code
//! range maps back to a span through `Function::value_loc` — what a DWARF line table consumes.
//!
//! The invariant that matters most is that locations never change code: an object built from a
//! program with its location table is byte-identical to the one built without it.

use super::*;
use std::cell::RefCell;
use wukong_span::{SourceId, SourceMap};

/// One defined function: its name and its machine code's sorted `(start, end, srcloc bits)` ranges.
type Captured = (Symbol, Vec<(u32, u32, u32)>);

thread_local! {
    /// While a test has it `Some`, [`define_functions_serial`] records each function it defines on
    /// this thread.
    static CAPTURED: RefCell<Option<Vec<Captured>>> = const { RefCell::new(None) };
}

/// The capture hook [`define_functions_serial`] calls between defining a function and clearing
/// its context.
pub(super) fn capture(name: Symbol, ctx: &cranelift_codegen::Context) {
    CAPTURED.with(|c| {
        if let Some(funcs) = c.borrow_mut().as_mut() {
            let code = ctx
                .compiled_code()
                .expect("a defined function has machine code");
            let ranges = code
                .buffer
                .get_srclocs_sorted()
                .iter()
                .map(|r| (r.start, r.end, r.loc.bits()))
                .collect();
            funcs.push((name, ranges));
        }
    });
}

/// Parse, check, lower and optimize `src` at `opt`, keeping the source map that locates it.
fn lower(src: &str, opt: u8) -> (Program, Interner, SourceMap) {
    let mut sm = SourceMap::new();
    let id = sm.add("t.wk", src);
    let mut interner = Interner::new();
    let (module, parsed) = wukong_parser::parse_module(src, id, &mut interner);
    assert!(parsed.iter().all(|d| !d.is_error()), "{parsed:?}");
    let (sema, checked) = wukong_sema::check(&module, &interner);
    assert!(checked.iter().all(|d| !d.is_error()), "{checked:?}");
    let (mut program, lowered) = wukong_mir_build::lower_program(&module, &sema, &mut interner);
    assert!(lowered.iter().all(|d| !d.is_error()), "{lowered:?}");
    wukong_opt::optimize(&mut program, opt);
    assert!(wukong_mir::verify::verify_program(&program).is_empty());
    assert_eq!(id, SourceId(0));
    (program, interner, sm)
}

/// Loops and float recurrences, calls and recursion, branches, and a checked slice access.
const SOURCES: &[&str] = &[
    "fn fib(n: i64) -> i64 {
    if n < 2 {
        return n;
    }
    return fib(n - 1) + fib(n - 2);
}

fn main() -> i32 {
    print(fib(15));
    return 3;
}
",
    "fn main() -> i32 {
    let mut x: f32 = 0.1;
    let mut acc: f64 = 0.0;
    for i in 0..40 {
        x = x * 1.7 - 0.3 / (x + 2.0);
        acc = acc + (x as f64) * 0.5;
    }
    print(acc);
    return 0;
}
",
    "fn pick(s: []i32, i: i32) -> i32 {
    return s[i] * 3;
}

fn clampi(v: i32, lo: i32, hi: i32) -> i32 {
    if v < lo {
        return lo;
    }
    if v > hi {
        return hi;
    }
    return v;
}

fn main() -> i32 {
    let arr: [i32; 6] = [4, 8, 15, 16, 23, 42];
    let s: []i32 = arr;
    let mut t: i32 = 0;
    for i in 0..6 {
        t = t + clampi(pick(s, i), 10, 60);
    }
    print(t);
    return 0;
}
",
    "struct P { x: i32, y: f32 }

fn main() -> i32 {
    let a: [P; 12] = [P { x: 3, y: 0.5 }; 12];
    let mut b = a;
    b[5].x = 9;
    for i in 0..12 {
        print(b[i].y);
    }
    return b[5].x + a[5].x;
}
",
];

fn object(program: &Program, interner: &Interner, level: u8, parallel: bool) -> Vec<u8> {
    emit_object_full(
        program,
        interner,
        EmitOptions {
            verify: true,
            parallel,
        },
        TargetCpu::Native,
        BackendOpt::for_level(level),
        ObjectKind::Library,
    )
    .unwrap_or_else(|e| panic!("emit: {e}"))
    .0
}

/// (e) The location table changes no emitted byte — at both backend opt levels, serial and
/// parallel. If it ever did, `-O0` objects would depend on whitespace and comments.
#[test]
fn source_locations_never_change_object_bytes() {
    for (n, src) in SOURCES.iter().enumerate() {
        for level in [0, 2] {
            let (program, interner, _) = lower(src, level);
            let located = program
                .funcs
                .iter()
                .filter(|f| f.locs.values.iter().any(|s| !s.is_dummy()))
                .count();
            assert!(located > 0, "src {n}: nothing to strip");
            let mut stripped = program.clone();
            for f in &mut stripped.funcs {
                f.locs = wukong_mir::SourceLocs::default();
            }
            for parallel in [false, true] {
                assert_eq!(
                    object(&program, &interner, level, parallel),
                    object(&stripped, &interner, level, parallel),
                    "src {n}, -O{level}, parallel {parallel}: source locations changed the object"
                );
            }
        }
    }
}

/// (f) The machine code of a located program carries srcloc ranges, each naming a located MIR
/// value, and a chosen statement's line is among the lines they map back to.
#[test]
fn machine_code_ranges_map_back_to_source_lines() {
    for level in [0, 2] {
        let (program, interner, sm) = lower(SOURCES[1], level);
        CAPTURED.with(|c| *c.borrow_mut() = Some(Vec::new()));
        let emitted = emit_object_full(
            &program,
            &interner,
            EmitOptions {
                verify: false,
                parallel: false,
            },
            TargetCpu::Native,
            BackendOpt::for_level(level),
            ObjectKind::Library,
        );
        let captured = CAPTURED.with(|c| c.borrow_mut().take()).unwrap();
        assert!(emitted.is_ok(), "{:?}", emitted.err());

        let main = interner.get("main").unwrap();
        let mut lines = Vec::new();
        for (name, ranges) in &captured {
            let f = program.function(*name).expect("a captured function exists");
            // Code emitted before the first mark (the prologue) carries the default location.
            for &(start, end, bits) in ranges.iter().filter(|r| r.2 != u32::MAX) {
                assert!(start <= end);
                let span = f.value_loc(ValueId(bits)).unwrap_or_else(|| {
                    panic!("-O{level}: srcloc {bits} is not a located value of `{name:?}`")
                });
                lines.push(sm.span_location(span).line);
            }
        }
        let main_ranges = captured
            .iter()
            .find(|(name, _)| *name == main)
            .map(|(_, r)| r.iter().filter(|r| r.2 != u32::MAX).count())
            .expect("main was defined");
        assert!(main_ranges > 0, "-O{level}: main has no srcloc ranges");
        // `acc = acc + (x as f64) * 0.5;`
        assert!(
            lines.contains(&6),
            "-O{level}: line 6 is not in the machine code's lines {lines:?}"
        );
    }
}
