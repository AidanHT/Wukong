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
fn assertion_capable_callbacks_decline_before_native_execution() {
    let (mut program, mut interner, main) = lower(
        "fn leaf() { assert(false); }
         fn callback() { leaf(); }
         fn main() -> i32 { return 0; }",
        0,
    );
    let callback = interner.intern("callback");
    let mut address = Builder::new(interner.intern("address"), MirType::Ptr);
    let value = address.build(MirType::Ptr, Op::FuncAddr(callback));
    address.ret(Some(value));
    program.funcs.push(address.finish());
    assert!(wukong_mir::verify::verify_program(&program).is_empty());
    for error in [
        jit_compile(&program, main, &interner).err().unwrap(),
        emit_object(&program, &interner).unwrap_err(),
    ] {
        assert!(
            error.contains("assertion-capable callback `callback`"),
            "{error}"
        );
        assert!(error.contains("cancellation ABI"), "{error}");
    }
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
    let actual = asserting_functions(&program, &interner).unwrap();
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
    assert!(asserting_functions(&safe, &interner).unwrap().is_empty());
}

#[test]
fn raw_jit_declines_assertions_and_preserves_assertion_free_calls() {
    let (bad, bad_interner, bad_main) = lower("fn main() -> i32 { assert(false); return 0; }", 0);
    let bad = jit_compile(&bad, bad_main, &bad_interner).unwrap();
    assert_eq!(bad.run(), Err("assertion failed".to_owned()));

    // A raw entry cannot reset or inspect the previous failure flag. Even this passing
    // assertion would return a dummy zero from its caller under the stale flag.
    let (good, interner, _) = lower(
        "fn leaf() -> i32 { assert(true); return 42; }
         fn main() -> i32 { return leaf(); }",
        0,
    );
    let error = jit_module(&good, &interner)
        .err()
        .expect("raw assertion API must decline");
    assert!(error.contains("raw JIT function pointers"), "{error}");
    assert!(error.contains("JitProgram::run"), "{error}");

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
    // A bounds check alone does not make a program assertion-capable for the raw API: a raw
    // caller gets the checked code with an abort-on-failure binding. In range it just runs.
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
