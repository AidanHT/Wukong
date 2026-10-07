use super::*;
use wukong_mir::Builder;

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

#[test]
fn failed_assert_stops_side_effects_in_current_and_nested_functions() {
    for (src, before) in [
        (
            "fn main() -> i32 { print(1); assert(false); print(99); return 3; }",
            "1\n",
        ),
        (
            "fn leaf() -> f64 { print(7); assert(false); print(98); return 2.0; }
          fn middle() -> i64 { let x: f64 = leaf(); print(99); return x as i64; }
          fn main() -> i32 { print(1); let x: i64 = middle(); print(100); return x as i32; }",
            "1\n7\n",
        ),
        (
            "fn recurse(n: i32) { if n == 0 { assert(false); } else { recurse(n-1); } print(99); }
          fn main() -> i32 { print(1); recurse(4); print(100); return 3; }",
            "1\n",
        ),
    ] {
        for opt in [0, 2] {
            let (program, interner, main) = lower(src, opt);
            let native = jit_compile(&program, main, &interner).unwrap();
            // The public error API discards stdout. Inspect the actual side-effect buffer under
            // the same run lock so continuing after failure cannot hide behind that discard.
            {
                let _guard = RUN_LOCK.lock().unwrap_or_else(|e| e.into_inner());
                OUTPUT.lock().unwrap().clear();
                ASSERT_FAILED.store(false, Ordering::SeqCst);
                unsafe {
                    native.invoke();
                }
                assert!(ASSERT_FAILED.load(Ordering::SeqCst));
                assert_eq!(
                    *OUTPUT.lock().unwrap(),
                    before.as_bytes(),
                    "opt={opt}: {src}"
                );
            }
            let expected = Err("assertion failed".to_owned());
            assert_eq!(native.run(), expected);
            assert_eq!(
                wukong_interp::run_with_output(&program, main, &interner),
                expected
            );
            // The capturing APIs `wukongc --run` uses keep exactly the output printed before the
            // failure, on both backends -- what a linked executable leaves on stdout too.
            let failed: Result<i64, String> = Err("assertion failed".to_owned());
            let printed = before.as_bytes().to_vec();
            assert_eq!(
                native.run_capturing(),
                (failed.clone(), printed.clone()),
                "opt={opt}: {src}"
            );
            assert_eq!(
                wukong_interp::run_capturing(&program, main, &interner),
                (failed, printed),
                "opt={opt}: {src}"
            );
            assert_eq!(native.call(), 1);
        }
    }
}

#[test]
fn successful_assertions_and_subsequent_runs_keep_working() {
    let (bad, bad_interner, bad_main) = lower("fn main() -> i32 { assert(false); return 0; }", 0);
    let bad = jit_compile(&bad, bad_main, &bad_interner).unwrap();
    let (good, interner, main) = lower(
        "fn leaf() -> i32 { assert(true); return 42; }
         fn main() -> i32 { let x: i32 = leaf(); print(x); return x; }",
        0,
    );
    let good = jit_compile(&good, main, &interner).unwrap();
    assert!(bad.run().is_err());
    assert_eq!(
        good.call(),
        42,
        "timing entry must reset a previous failure flag"
    );
    assert_eq!(good.run().unwrap(), (42, b"42\n".to_vec()));
}

#[test]
fn a_failing_parallel_region_reports_its_first_failing_iteration_and_stops_its_caller() {
    // `body` is a parallel-region body in the runtime's `(start, end, env)` ABI, and `drive` --
    // built by hand below, since no `.wk` spelling reaches `func_addr` with a failure-capable body
    // today -- runs it over [0, 64) through `wukong_parallel_for`. `get(a, i)` fails its bounds
    // check for every i >= 40 against the 40-element array, so natively several spans fail at
    // once, in any order. The interpreter runs the range in order and stops at i = 40, so the
    // cancellation ABI must report exactly that failure on every run (not whichever span's thread
    // got there first), and nothing after the region may run: stdout is "7\n", never "99".
    let src = "fn get(s: []i32, i: i64) -> i32 { return s[i]; }
               fn body(start: i64, end: i64, env: *mut u8) {
                   let a: [i32; 40] = [1; 40];
                   let mut i: i64 = start;
                   let mut t: i32 = 0;
                   while i < end { t = t + get(a, i); i = i + 1; }
               }
               fn drive() { }
               fn main() -> i32 { print(7); drive(); print(99); return 0; }";
    for opt in [0, 2] {
        let (mut program, mut interner, main) = lower(src, 0);
        let drive = interner.intern("drive");
        let mut region = Builder::new(drive, MirType::Void);
        let n = region.build(MirType::I64, Op::ConstInt(64, MirType::I64));
        let body = region.build(MirType::Ptr, Op::FuncAddr(interner.intern("body")));
        let env = region.alloca(MirType::I64);
        region.build_void(Op::Call {
            func: interner.intern("wukong_parallel_for"),
            args: vec![n, body, env],
        });
        region.ret(None);
        let placeholder = program.funcs.iter_mut().find(|f| f.name == drive).unwrap();
        *placeholder = region.finish();
        wukong_opt::optimize(&mut program, opt);
        assert!(wukong_mir::verify::verify_program(&program).is_empty());

        let failed: Result<i64, String> =
            Err("index out of bounds: the len is 40 but the index is 40".to_owned());
        let printed = b"7\n".to_vec();
        assert_eq!(
            wukong_interp::run_capturing(&program, main, &interner),
            (failed.clone(), printed.clone()),
            "opt={opt}"
        );
        let native = jit_compile(&program, main, &interner).unwrap();
        for _ in 0..16 {
            assert_eq!(
                native.run_capturing(),
                (failed.clone(), printed.clone()),
                "opt={opt}"
            );
        }
        assert_eq!(native.call(), 1, "opt={opt}");
        // A linked executable exits at the failure itself, so the region needs no propagation
        // there and the object is emitted like any other.
        emit_object(&program, &interner).expect("a failure-capable region body emits an object");
    }
}

#[test]
fn a_region_body_that_cannot_fail_adds_no_failure_status_to_its_caller() {
    // The propagation set grows through a region only when its body can fail: `safe` cannot, so
    // neither can `drive`, and `main` keeps no status poll.
    let (mut program, mut interner, _) = lower(
        "fn safe(start: i64, end: i64, env: *mut u8) { }
         fn drive() { }
         fn main() -> i32 { drive(); return 0; }",
        0,
    );
    let drive = interner.intern("drive");
    let mut region = Builder::new(drive, MirType::Void);
    let n = region.build(MirType::I64, Op::ConstInt(8, MirType::I64));
    let body = region.build(MirType::Ptr, Op::FuncAddr(interner.intern("safe")));
    let env = region.alloca(MirType::I64);
    region.build_void(Op::Call {
        func: interner.intern("wukong_parallel_for"),
        args: vec![n, body, env],
    });
    region.ret(None);
    let placeholder = program.funcs.iter_mut().find(|f| f.name == drive).unwrap();
    *placeholder = region.finish();
    assert!(asserting_functions(&program, &interner).is_empty());
}

#[test]
fn assertion_propagation_only_marks_transitively_asserting_functions() {
    let (program, mut interner, _) = lower(
        "fn safe(x: i32) -> i32 { return x+1; }
         fn leaf() { assert(true); }
         fn middle() { leaf(); }
         fn main() -> i32 { middle(); return safe(2); }",
        0,
    );
    let actual = asserting_functions(&program, &interner);
    let expected = [
        interner.intern("leaf"),
        interner.intern("middle"),
        interner.intern("main"),
    ]
    .into_iter()
    .collect();
    assert_eq!(
        actual, expected,
        "safe calls must not receive assertion status polls"
    );
    let (safe, interner, _) = lower(
        "fn f() -> i32 { return 2; } fn main() -> i32 { return f(); }",
        0,
    );
    assert!(asserting_functions(&safe, &interner).is_empty());
}

#[test]
fn raw_jit_accepts_assertions_and_never_reads_a_guarded_runs_failure() {
    let (bad, bad_interner, bad_main) = lower("fn main() -> i32 { assert(false); return 0; }", 0);
    let bad = jit_compile(&bad, bad_main, &bad_interner).unwrap();
    assert_eq!(bad.run(), Err("assertion failed".to_owned()));

    // A raw entry cannot reset or inspect a guarded run's failure flag -- the run above may have
    // left it raised -- so raw code never reads it: its status poll is bound to "not failed" and a
    // failed assertion aborts instead. A passing assertion therefore returns the real value, not
    // the dummy zero a stale flag would force out of `leaf`'s caller.
    let (good, interner, main) = lower(
        "fn leaf() -> i32 { assert(true); return 42; }
         fn main() -> i32 { return leaf(); }",
        0,
    );
    let module = jit_module(&good, &interner).expect("raw modules accept assertions");
    let entry: unsafe extern "C" fn() -> i32 =
        unsafe { std::mem::transmute(module.func_ptr(main).unwrap()) };
    assert_eq!(unsafe { entry() }, 42);

    // An assertion-free raw entry has no status polls and remains valid even after failure.
    let (safe, interner, main) = lower("fn main() -> i32 { return 42; }", 0);
    let module = jit_module(&safe, &interner).unwrap();
    let entry: unsafe extern "C" fn() -> i32 =
        unsafe { std::mem::transmute(module.func_ptr(main).unwrap()) };
    assert_eq!(unsafe { entry() }, 42);
}

#[test]
fn slice_bounds_failure_stops_like_an_assert_and_reports_the_shared_message() {
    // `get(a, 3)` on a 3-element slice fails its bounds check. Everything before the access ran
    // (`2`, `6`, `3`); nothing after it may (`99` from main, the return value). The native run must
    // report the very message the interpreter does, at -O0 and after -O2 inlined `get` into main.
    let src = "fn get(s: []i32, i: i64) -> i32 { print(i); return s[i]; }
               fn main() -> i32 { let a: [i32; 3] = [4, 5, 6]; print(get(a, 2)); \
               print(get(a, 3)); print(99); return 0; }";
    for opt in [0, 2] {
        let (program, interner, main) = lower(src, opt);
        let native = jit_compile(&program, main, &interner).unwrap();
        {
            let _guard = RUN_LOCK.lock().unwrap_or_else(|e| e.into_inner());
            OUTPUT.lock().unwrap().clear();
            reset_failure();
            unsafe {
                native.invoke();
            }
            assert!(ASSERT_FAILED.load(Ordering::SeqCst), "opt={opt}");
            assert_eq!(*OUTPUT.lock().unwrap(), b"2\n6\n3\n", "opt={opt}");
        }
        let expected = Err("index out of bounds: the len is 3 but the index is 3".to_owned());
        assert_eq!(native.run(), expected, "opt={opt}");
        assert_eq!(
            wukong_interp::run_with_output(&program, main, &interner),
            expected,
            "opt={opt}"
        );
        assert_eq!(native.call(), 1, "opt={opt}");
    }
}

#[test]
fn raw_jit_accepts_bounds_checked_code_and_runs_it_in_range() {
    // A raw caller gets bounds-checked code with an abort-on-failure binding. In range it runs.
    let (program, mut interner, _) = lower(
        "fn get(s: []i32, i: i64) -> i32 { return s[i] * 10; }
         fn main() -> i32 { return 0; }",
        0,
    );
    let module = jit_module(&program, &interner).expect("bounds checks are raw-callable");
    let get = interner.intern("get");
    let entry: unsafe extern "C" fn(*const u64, i64) -> i32 =
        unsafe { std::mem::transmute(module.func_ptr(get).unwrap()) };
    let data = [7i32, 8, 9];
    // A `[]i32` parameter is a pointer to its 16-byte fat pointer: { data, len }.
    let fat = [data.as_ptr() as u64, data.len() as u64];
    for (i, want) in [(0i64, 70), (1, 80), (2, 90)] {
        assert_eq!(unsafe { entry(fat.as_ptr(), i) }, want);
    }
}
