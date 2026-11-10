//! Regression tests for [`Op::CallIndirect`] across the optimizer.
//!
//! An indirect call reaches code no pass can see. Every pass must treat it as a call with every
//! effect ([`wukong_mir::Effects`]): never value-numbered, deleted, hoisted, sunk or reordered
//! against memory it may touch, and a function that makes one is never summarized as side-effect
//! free. Each program below runs at -O0, -O1 and -O2; its -O0 answer (the interpreter on
//! unoptimized MIR) is what a pass that took the op for pure would get wrong. Where the indirect
//! call is in a function nothing inlines (`run`, `twice`, ...), the test also checks the optimized
//! function still holds every indirect call, in order.
//!
//! The callee here is a parameter, so nothing can devirtualize it inside the function under test;
//! [`a_known_callee_is_devirtualized`] covers the case where it can.

use wukong_mir::{BinOp, BlockId, Builder, CmpOp, Function, MirType, Op, Program, ValueId};
use wukong_span::{Interner, Symbol};

/// Run `program` from `entry` at -O0, -O1 and -O2 (each from a fresh clone, verified as a whole
/// program), returning `(exit, stdout)` per level.
fn run_levels(program: &Program, entry: Symbol, it: &Interner) -> [(i64, String); 3] {
    let mut out: [(i64, String); 3] = Default::default();
    for (slot, level) in [0u8, 1, 2].into_iter().enumerate() {
        let mut p = program.clone();
        if level > 0 {
            crate::optimize(&mut p, level);
        }
        let errors = wukong_mir::verify::verify_program(&p);
        assert!(errors.is_empty(), "-O{level}: {errors:?}");
        let (exit, stdout) = wukong_interp::run_with_output(&p, entry, it)
            .unwrap_or_else(|e| panic!("-O{level}: {e}"));
        out[slot] = (exit, String::from_utf8(stdout).unwrap());
    }
    out
}

/// `program` optimized at -O2, and the function named `name` in it.
fn optimized(program: &Program, name: Symbol) -> Function {
    let mut p = program.clone();
    crate::optimize(&mut p, 2);
    let errors = wukong_mir::verify::verify_program(&p);
    assert!(errors.is_empty(), "{errors:?}");
    p.function(name).expect("the function survives").clone()
}

fn program_of(funcs: Vec<Function>) -> Program {
    let mut p = Program::new();
    p.funcs = funcs;
    p
}

fn i64c(b: &mut Builder, v: i128) -> ValueId {
    b.build(MirType::I64, Op::ConstInt(v, MirType::I64))
}

fn indirect(b: &mut Builder, callee: ValueId, args: Vec<ValueId>, ret: MirType) -> ValueId {
    b.build(ret.clone(), Op::CallIndirect { callee, args, ret })
}

fn indirect_void(b: &mut Builder, callee: ValueId, args: Vec<ValueId>, ret: MirType) {
    b.build_void(Op::CallIndirect { callee, args, ret });
}

/// The operands of every indirect call of `f`, in block-then-instruction order.
fn indirect_calls(f: &Function) -> Vec<Vec<ValueId>> {
    f.blocks
        .iter()
        .flat_map(|b| &b.insts)
        .filter_map(|i| match &i.op {
            Op::CallIndirect { args, .. } => Some(args.clone()),
            _ => None,
        })
        .collect()
}

fn count_ops(f: &Function, pred: impl Fn(&Op) -> bool) -> usize {
    f.blocks
        .iter()
        .flat_map(|b| &b.insts)
        .filter(|i| pred(&i.op))
        .count()
}

/// `say(x) = { print(x); return x }`.
fn say(it: &mut Interner) -> Function {
    let mut b = Builder::new(it.intern("say"), MirType::I64);
    let x = b.add_param(MirType::I64);
    b.build_void(Op::Call {
        func: it.intern("print"),
        args: vec![x],
    });
    b.ret(Some(x));
    b.finish()
}

/// `tick(p) = { *p = *p + 1; return *p }`.
fn tick(it: &mut Interner) -> Function {
    let mut b = Builder::new(it.intern("tick"), MirType::I64);
    let p = b.add_param(MirType::Ptr);
    let v = b.build(MirType::I64, Op::Load(p, MirType::I64));
    let one = i64c(&mut b, 1);
    let w = b.build(MirType::I64, Op::Bin(BinOp::Add, v, one));
    b.build_void(Op::Store { ptr: p, value: w });
    b.ret(Some(w));
    b.finish()
}

/// A counted loop `for i in 0..n` in `b`: returns `(header, exit, i)`, the builder left in the
/// body. The caller ends the body with [`close_loop`].
fn open_loop(b: &mut Builder, n: ValueId) -> (BlockId, BlockId, ValueId) {
    let header = b.new_block();
    let body = b.new_block();
    let exit = b.new_block();
    let zero = i64c(b, 0);
    b.br(header, vec![zero]);
    b.switch_to(header);
    let i = b.block_param(header, MirType::I64);
    let c = b.build(MirType::I1, Op::Cmp(CmpOp::Slt, i, n));
    b.cond_br(c, body, vec![], exit, vec![]);
    b.switch_to(body);
    (header, exit, i)
}

fn close_loop(b: &mut Builder, header: BlockId, exit: BlockId, i: ValueId) {
    let one = i64c(b, 1);
    let next = b.build(MirType::I64, Op::Bin(BinOp::Add, i, one));
    b.br(header, vec![next]);
    b.switch_to(exit);
}

/// Two identical indirect calls of a printer, a third whose result is unused: `cse` must not
/// number the second as the first, `dce` must not drop the third, and they print in order.
#[test]
fn identical_indirect_calls_are_all_kept_and_ordered() {
    let mut it = Interner::new();
    let say = say(&mut it);
    // run(fp, x) = { a = fp(x); b = fp(x); fp(x + 1); return a + b }
    let run_name = it.intern("run");
    let mut r = Builder::new(run_name, MirType::I64);
    let fp = r.add_param(MirType::Ptr);
    let x = r.add_param(MirType::I64);
    let a = indirect(&mut r, fp, vec![x], MirType::I64);
    let b = indirect(&mut r, fp, vec![x], MirType::I64);
    let one = i64c(&mut r, 1);
    let x1 = r.build(MirType::I64, Op::Bin(BinOp::Add, x, one));
    indirect_void(&mut r, fp, vec![x1], MirType::I64);
    let s = r.build(MirType::I64, Op::Bin(BinOp::Add, a, b));
    r.ret(Some(s));
    let run = r.finish();

    let main_name = it.intern("main");
    let mut m = Builder::new(main_name, MirType::I64);
    let f = m.build(MirType::Ptr, Op::FuncAddr(say.name));
    let seven = i64c(&mut m, 7);
    let got = m.build(
        MirType::I64,
        Op::Call {
            func: run_name,
            args: vec![f, seven],
        },
    );
    m.ret(Some(got));

    let program = program_of(vec![say.clone(), run.clone(), m.finish()]);
    for (exit, out) in run_levels(&program, main_name, &it) {
        assert_eq!((exit, out.as_str()), (14, "7\n7\n8\n"));
    }
    // `run` on its own, where nothing can see its callee.
    let alone = program_of(vec![say, run]);
    let f = optimized(&alone, run_name);
    assert_eq!(
        indirect_calls(&f).len(),
        3,
        "an indirect call was merged or deleted"
    );
}

/// `*p = 1; fp(p); return *p` with a callee that adds 10: the load after the call must not be
/// forwarded the stored 1 (`cse`), and the store must not be dropped as overwritten-or-dead
/// (`dse`) — the callee reads it.
#[test]
fn memory_is_not_forwarded_or_dropped_across_an_indirect_call() {
    let mut it = Interner::new();
    // add10(p) = { *p = *p + 10 }
    let add10_name = it.intern("add10");
    let mut a = Builder::new(add10_name, MirType::Void);
    let p = a.add_param(MirType::Ptr);
    let v = a.build(MirType::I64, Op::Load(p, MirType::I64));
    let ten = i64c(&mut a, 10);
    let w = a.build(MirType::I64, Op::Bin(BinOp::Add, v, ten));
    a.build_void(Op::Store { ptr: p, value: w });
    a.ret(None);
    let add10 = a.finish();

    // run(fp, p) = { *p = 1; fp(p); let r = *p; *p = 5; return r }
    let run_name = it.intern("run");
    let mut r = Builder::new(run_name, MirType::I64);
    let fp = r.add_param(MirType::Ptr);
    let p = r.add_param(MirType::Ptr);
    let one = i64c(&mut r, 1);
    r.build_void(Op::Store { ptr: p, value: one });
    indirect_void(&mut r, fp, vec![p], MirType::Void);
    let back = r.build(MirType::I64, Op::Load(p, MirType::I64));
    let five = i64c(&mut r, 5);
    r.build_void(Op::Store {
        ptr: p,
        value: five,
    });
    r.ret(Some(back));
    let run = r.finish();

    // main = { let s; let r = run(add10, &s); return r * 100 + s }
    let main_name = it.intern("main");
    let mut m = Builder::new(main_name, MirType::I64);
    let s = m.alloca(MirType::I64);
    let f = m.build(MirType::Ptr, Op::FuncAddr(add10_name));
    let got = m.build(
        MirType::I64,
        Op::Call {
            func: run_name,
            args: vec![f, s],
        },
    );
    let hundred = i64c(&mut m, 100);
    let scaled = m.build(MirType::I64, Op::Bin(BinOp::Mul, got, hundred));
    let after = m.build(MirType::I64, Op::Load(s, MirType::I64));
    let sum = m.build(MirType::I64, Op::Bin(BinOp::Add, scaled, after));
    m.ret(Some(sum));

    let program = program_of(vec![add10.clone(), run.clone(), m.finish()]);
    for (exit, _) in run_levels(&program, main_name, &it) {
        assert_eq!(exit, 1105);
    }
    let alone = program_of(vec![add10, run]);
    let f = optimized(&alone, run_name);
    assert_eq!(
        count_ops(&f, |op| matches!(op, Op::Store { .. })),
        2,
        "the store the callee reads was dropped"
    );
    assert_eq!(count_ops(&f, |op| matches!(op, Op::Load(..))), 1);
}

/// An indirect call with loop-invariant arguments inside a loop is not hoisted (`licm`): every
/// iteration ticks the counter.
#[test]
fn an_indirect_call_with_invariant_arguments_stays_in_its_loop() {
    let mut it = Interner::new();
    let tick = tick(&mut it);
    // sum3(fp, p) = { let s = 0; for i in 0..3 { s += fp(p) }; return s }
    let sum_name = it.intern("sum3");
    let mut b = Builder::new(sum_name, MirType::I64);
    let fp = b.add_param(MirType::Ptr);
    let p = b.add_param(MirType::Ptr);
    let acc = b.alloca(MirType::I64);
    let zero = i64c(&mut b, 0);
    b.build_void(Op::Store {
        ptr: acc,
        value: zero,
    });
    let n = i64c(&mut b, 3);
    let (header, exit, i) = open_loop(&mut b, n);
    let t = indirect(&mut b, fp, vec![p], MirType::I64);
    let cur = b.build(MirType::I64, Op::Load(acc, MirType::I64));
    let next = b.build(MirType::I64, Op::Bin(BinOp::Add, cur, t));
    b.build_void(Op::Store {
        ptr: acc,
        value: next,
    });
    close_loop(&mut b, header, exit, i);
    let r = b.build(MirType::I64, Op::Load(acc, MirType::I64));
    b.ret(Some(r));
    let sum3 = b.finish();

    let main_name = it.intern("main");
    let mut m = Builder::new(main_name, MirType::I64);
    let s = m.alloca(MirType::I64);
    let zero = i64c(&mut m, 0);
    m.build_void(Op::Store {
        ptr: s,
        value: zero,
    });
    let f = m.build(MirType::Ptr, Op::FuncAddr(tick.name));
    let got = m.build(
        MirType::I64,
        Op::Call {
            func: sum_name,
            args: vec![f, s],
        },
    );
    m.ret(Some(got));

    let program = program_of(vec![tick.clone(), sum3.clone(), m.finish()]);
    for (exit, _) in run_levels(&program, main_name, &it) {
        assert_eq!(exit, 6, "1 + 2 + 3");
    }
    let alone = program_of(vec![tick, sum3]);
    let f = optimized(&alone, sum_name);
    let in_loop = crate::inline::loop_blocks(&f);
    let calls_in_loop = f
        .blocks
        .iter()
        .enumerate()
        .filter(|(b, _)| in_loop[*b])
        .flat_map(|(_, blk)| &blk.insts)
        .filter(|i| matches!(i.op, Op::CallIndirect { .. }))
        .count();
    assert!(calls_in_loop >= 1, "the indirect call left its loop");
}

/// A function whose only effect is an indirect call is not side-effect free: summarizing it as
/// pure would let `dce` delete its unused calls at -O1 (which does not inline).
#[test]
fn a_caller_of_an_indirect_call_is_not_summarized_pure() {
    let mut it = Interner::new();
    let tick = tick(&mut it);
    // poke(fp, p) = { fp(p); return 0 }
    let poke_name = it.intern("poke");
    let mut b = Builder::new(poke_name, MirType::I64);
    let fp = b.add_param(MirType::Ptr);
    let p = b.add_param(MirType::Ptr);
    indirect_void(&mut b, fp, vec![p], MirType::I64);
    let zero = i64c(&mut b, 0);
    b.ret(Some(zero));
    let poke = b.finish();

    // main = { let s = 0; poke(tick, &s); poke(tick, &s); return s }  (both results unused)
    let main_name = it.intern("main");
    let mut m = Builder::new(main_name, MirType::I64);
    let s = m.alloca(MirType::I64);
    let zero = i64c(&mut m, 0);
    m.build_void(Op::Store {
        ptr: s,
        value: zero,
    });
    let f = m.build(MirType::Ptr, Op::FuncAddr(tick.name));
    for _ in 0..2 {
        m.build(
            MirType::I64,
            Op::Call {
                func: poke_name,
                args: vec![f, s],
            },
        );
    }
    let r = m.build(MirType::I64, Op::Load(s, MirType::I64));
    m.ret(Some(r));

    let program = program_of(vec![tick, poke, m.finish()]);
    let table = crate::effects::CallEffects::of(&program);
    assert!(
        table.get(poke_name).is_none(),
        "an indirect call is a side effect; its caller must stay opaque"
    );
    for (exit, _) in run_levels(&program, main_name, &it) {
        assert_eq!(exit, 2);
    }
}

/// `for i in 0..3 { *p = *p + 1; fp(p) }` with a callee that doubles `*p`: keeping `*p` in a
/// register across the loop (`promote`) would hide every write from the callee.
#[test]
fn a_location_an_indirect_call_can_reach_is_not_promoted() {
    let mut it = Interner::new();
    // dbl(p) = { *p = *p * 2 }
    let dbl_name = it.intern("dbl");
    let mut d = Builder::new(dbl_name, MirType::Void);
    let p = d.add_param(MirType::Ptr);
    let v = d.build(MirType::I64, Op::Load(p, MirType::I64));
    let two = i64c(&mut d, 2);
    let w = d.build(MirType::I64, Op::Bin(BinOp::Mul, v, two));
    d.build_void(Op::Store { ptr: p, value: w });
    d.ret(None);
    let dbl = d.finish();

    let run_name = it.intern("run");
    let mut b = Builder::new(run_name, MirType::I64);
    let fp = b.add_param(MirType::Ptr);
    let p = b.add_param(MirType::Ptr);
    let n = i64c(&mut b, 3);
    let (header, exit, i) = open_loop(&mut b, n);
    let v = b.build(MirType::I64, Op::Load(p, MirType::I64));
    let one = i64c(&mut b, 1);
    let w = b.build(MirType::I64, Op::Bin(BinOp::Add, v, one));
    b.build_void(Op::Store { ptr: p, value: w });
    indirect_void(&mut b, fp, vec![p], MirType::Void);
    close_loop(&mut b, header, exit, i);
    let r = b.build(MirType::I64, Op::Load(p, MirType::I64));
    b.ret(Some(r));
    let run = b.finish();

    let main_name = it.intern("main");
    let mut m = Builder::new(main_name, MirType::I64);
    let s = m.alloca(MirType::I64);
    let zero = i64c(&mut m, 0);
    m.build_void(Op::Store {
        ptr: s,
        value: zero,
    });
    let f = m.build(MirType::Ptr, Op::FuncAddr(dbl_name));
    let got = m.build(
        MirType::I64,
        Op::Call {
            func: run_name,
            args: vec![f, s],
        },
    );
    m.ret(Some(got));

    let program = program_of(vec![dbl, run, m.finish()]);
    for (exit, _) in run_levels(&program, main_name, &it) {
        assert_eq!(exit, 14, "((0+1)*2+1)*2+1)*2");
    }
}

/// A loop whose body is nothing but an indirect call of a printer has an effect on every
/// iteration: loop deletion and the unroller must keep every call, in order, and the vectorizer
/// must not widen a loop around one.
#[test]
fn a_loop_of_indirect_calls_is_neither_deleted_nor_reordered() {
    let mut it = Interner::new();
    let say = say(&mut it);
    // each(fp) = { for i in 0..4 { fp(i) } }
    let each_name = it.intern("each");
    let mut b = Builder::new(each_name, MirType::Void);
    let fp = b.add_param(MirType::Ptr);
    let n = i64c(&mut b, 4);
    let (header, exit, i) = open_loop(&mut b, n);
    indirect_void(&mut b, fp, vec![i], MirType::I64);
    close_loop(&mut b, header, exit, i);
    b.ret(None);
    let each = b.finish();

    let main_name = it.intern("main");
    let mut m = Builder::new(main_name, MirType::I64);
    let f = m.build(MirType::Ptr, Op::FuncAddr(say.name));
    m.build_void(Op::Call {
        func: each_name,
        args: vec![f],
    });
    let zero = i64c(&mut m, 0);
    m.ret(Some(zero));

    let program = program_of(vec![say.clone(), each.clone(), m.finish()]);
    for (exit, out) in run_levels(&program, main_name, &it) {
        assert_eq!((exit, out.as_str()), (0, "0\n1\n2\n3\n"));
    }
    let alone = program_of(vec![say, each]);
    let f = optimized(&alone, each_name);
    assert!(
        !indirect_calls(&f).is_empty(),
        "the loop of calls was deleted"
    );
    assert!(
        f.value_types.iter().all(|t| !t.is_vector()),
        "a loop around an indirect call was vectorized"
    );
}

/// A callee the optimizer can see — `func_addr say` flowing into the call through a stack slot —
/// becomes a direct call at -O1 (`devirt`), and at -O2 the direct call is then inlined like any
/// other: no indirect call is left either way.
#[test]
fn a_known_callee_is_devirtualized() {
    let mut it = Interner::new();
    let say = say(&mut it);
    // main = { let slot = say; let f = *slot; return f(3) + f(4) }
    let main_name = it.intern("main");
    let mut m = Builder::new(main_name, MirType::I64);
    let slot = m.alloca(MirType::Ptr);
    let addr = m.build(MirType::Ptr, Op::FuncAddr(say.name));
    m.build_void(Op::Store {
        ptr: slot,
        value: addr,
    });
    let f = m.build(MirType::Ptr, Op::Load(slot, MirType::Ptr));
    let three = i64c(&mut m, 3);
    let a = indirect(&mut m, f, vec![three], MirType::I64);
    let four = i64c(&mut m, 4);
    let b = indirect(&mut m, f, vec![four], MirType::I64);
    let s = m.build(MirType::I64, Op::Bin(BinOp::Add, a, b));
    m.ret(Some(s));

    let program = program_of(vec![say.clone(), m.finish()]);
    for (exit, out) in run_levels(&program, main_name, &it) {
        assert_eq!((exit, out.as_str()), (7, "3\n4\n"));
    }
    for level in [1u8, 2] {
        let mut p = program.clone();
        crate::optimize(&mut p, level);
        let main = p.function(main_name).unwrap();
        assert!(indirect_calls(main).is_empty(), "-O{level}: still indirect");
        let direct = count_ops(
            main,
            |op| matches!(op, Op::Call { func, .. } if *func == say.name),
        );
        assert_eq!(direct, if level == 1 { 2 } else { 0 }, "-O{level}");
    }
}

/// Inlining a call that discards a value-returning callee's result (legal MIR — and what a
/// devirtualized result-less indirect call is) gives the continuation a parameter for the
/// returned value instead of branching to it with an argument it does not take.
#[test]
fn an_inlined_call_may_discard_its_result() {
    let mut it = Interner::new();
    let say = say(&mut it);
    let main_name = it.intern("main");
    let mut m = Builder::new(main_name, MirType::I64);
    let five = i64c(&mut m, 5);
    m.build_void(Op::Call {
        func: say.name,
        args: vec![five],
    });
    m.ret(Some(five));
    let mut p = program_of(vec![say, m.finish()]);
    assert!(crate::inline_program(&mut p));
    let errors = wukong_mir::verify::verify_program(&p);
    assert!(errors.is_empty(), "{errors:?}");
    let (exit, out) = wukong_interp::run_with_output(&p, main_name, &it).unwrap();
    assert_eq!((exit, out.as_slice()), (5, &b"5\n"[..]));
}

/// A `func_addr` of a function whose signature differs from the call's is left indirect: the
/// rewrite would be a direct call of the wrong ABI.
#[test]
fn a_mismatched_signature_is_not_devirtualized() {
    let mut it = Interner::new();
    let say = say(&mut it);
    let mut f = Builder::new(it.intern("main"), MirType::I32);
    let addr = f.build(MirType::Ptr, Op::FuncAddr(say.name));
    let x = f.build(MirType::I32, Op::ConstInt(1, MirType::I32));
    // `say` takes an i64; this call passes an i32.
    let r = indirect(&mut f, addr, vec![x], MirType::I32);
    f.ret(Some(r));
    let mut p = program_of(vec![say, f.finish()]);
    assert!(crate::devirtualize_program(&mut p).is_empty());
    assert_eq!(indirect_calls(&p.funcs[1]).len(), 1);
}
