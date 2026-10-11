//! Regression tests for the memory-effect audit of [`Op::AtomicAdd`].
//!
//! An atomic add has no result, reads *and* writes its location, and may race other threads' adds
//! to it. Every pass that walks memory must therefore treat it the way it treats
//! [`Op::VecKernelCall`] — an opaque effect — and never as the "no result, nothing to see" op its
//! shape suggests. Five tests below run a program at -O0, -O1 and -O2 whose -O0 answer (the
//! interpreter on unoptimized MIR) differs from what a pass that missed the op would compute;
//! `an_atomic_add_publishes_its_location` checks the escape scan's verdicts directly.
//!
//! What each test can see, measured by removing one `AtomicAdd` arm at a time:
//! - `effects` fails `an_atomics_only_callee_is_not_summarized_pure`;
//! - `cse` fails `a_load_is_not_forwarded_past_an_atomic_add` and the callee test;
//! - `licm` fails `a_load_of_an_atomically_updated_location_stays_in_the_loop`;
//! - `has_side_effects` fails `a_loop_of_atomic_adds_survives_loop_deletion`;
//! - `loop_info::dependence` fails `a_loop_with_an_atomic_add_is_not_vectorized`.
//!
//! The escape scan names no `AtomicAdd` arm: its fallback marks every operand of an op it does not
//! model escaped, and the alias test pins that (an arm that skipped the op would fail it). The
//! three `vectorize.rs` arms are defense in depth behind `loop_info` and the planner's catch-all,
//! and no test sees one of them alone.

use wukong_mir::{
    BinOp, BlockId, Builder, CmpOp, Function, MirType, Op, ParamAttrs, Program, ValueId,
};
use wukong_span::{Interner, Symbol};

use crate::alias::AliasInfo;
use crate::loop_info::MemDep;

/// Run `program` from `entry` at -O0, -O1 and -O2 (each from a fresh clone, verified), and return
/// the three results.
fn run_levels(program: &Program, entry: Symbol, it: &Interner) -> [i64; 3] {
    let mut out = [0; 3];
    for (slot, level) in [0u8, 1, 2].into_iter().enumerate() {
        let mut p = program.clone();
        if level > 0 {
            crate::optimize(&mut p, level);
        }
        let errors = wukong_mir::verify::verify_program(&p);
        assert!(errors.is_empty(), "-O{level}: {errors:?}");
        out[slot] = wukong_interp::run(&p, entry, it).unwrap_or_else(|e| panic!("-O{level}: {e}"));
    }
    out
}

fn program_of(funcs: Vec<Function>) -> Program {
    let mut p = Program::new();
    p.funcs = funcs;
    p
}

fn i64c(b: &mut Builder, v: i128) -> ValueId {
    b.build(MirType::I64, Op::ConstInt(v, MirType::I64))
}

fn count_atomics(f: &Function) -> usize {
    f.blocks
        .iter()
        .flat_map(|b| &b.insts)
        .filter(|i| matches!(i.op, Op::AtomicAdd { .. }))
        .count()
}

/// A counted loop `for i in 0..n` in `b`: returns `(header, body, exit, i)`, with the builder left
/// in `body`. The caller ends the body with [`close_loop`].
fn open_loop(b: &mut Builder, n: ValueId) -> (BlockId, BlockId, BlockId, ValueId) {
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
    (header, body, exit, i)
}

fn close_loop(b: &mut Builder, header: BlockId, exit: BlockId, i: ValueId) {
    let one = i64c(b, 1);
    let next = b.build(MirType::I64, Op::Bin(BinOp::Add, i, one));
    b.br(header, vec![next]);
    b.switch_to(exit);
}

/// `store 1, p; atomic.add 2, p; load p` is 3: no pass may forward the stored 1 past the atomic,
/// whether `p` is a slot of this frame or a pointer handed in.
#[test]
fn a_load_is_not_forwarded_past_an_atomic_add() {
    let mut it = Interner::new();
    // f(p) = { *p = 1; atomic_add(*p, 2); return *p }
    let f_name = it.intern("f");
    let mut f = Builder::new(f_name, MirType::I64);
    let p = f.add_param(MirType::Ptr);
    let one = i64c(&mut f, 1);
    f.build_void(Op::Store { ptr: p, value: one });
    let two = i64c(&mut f, 2);
    f.atomic_add(p, two);
    let r = f.build(MirType::I64, Op::Load(p, MirType::I64));
    f.ret(Some(r));

    // main = { let s; f(&s) * 100 + local(s') } where the second half is the same sequence on a
    // slot of main's own frame.
    let main_name = it.intern("main");
    let mut m = Builder::new(main_name, MirType::I64);
    let s = m.alloca(MirType::I64);
    let through_call = m.build(
        MirType::I64,
        Op::Call {
            func: f_name,
            args: vec![s],
        },
    );
    let t = m.alloca(MirType::I64);
    let one = i64c(&mut m, 1);
    m.build_void(Op::Store { ptr: t, value: one });
    let two = i64c(&mut m, 2);
    m.atomic_add(t, two);
    let local = m.build(MirType::I64, Op::Load(t, MirType::I64));
    let hundred = i64c(&mut m, 100);
    let scaled = m.build(MirType::I64, Op::Bin(BinOp::Mul, through_call, hundred));
    let sum = m.build(MirType::I64, Op::Bin(BinOp::Add, scaled, local));
    m.ret(Some(sum));

    let program = program_of(vec![f.finish(), m.finish()]);
    assert_eq!(run_levels(&program, main_name, &it), [303; 3]);
}

/// A callee whose only effect is an atomic add is *not* side-effect free: summarizing it as pure
/// (what `effects.rs`'s `_` arm did before the op was named there) lets `dce` delete a call whose
/// result is unused, and the counter never moves. -O1 does not inline, so the summary decides it.
#[test]
fn an_atomics_only_callee_is_not_summarized_pure() {
    let mut it = Interner::new();
    // bump(p) = { atomic_add(*p, 5); return 0 }
    let bump_name = it.intern("bump");
    let mut bump = Builder::new(bump_name, MirType::I64);
    let p = bump.add_param(MirType::Ptr);
    let five = i64c(&mut bump, 5);
    bump.atomic_add(p, five);
    let zero = i64c(&mut bump, 0);
    bump.ret(Some(zero));
    let bump = bump.finish();

    // main = { let s = 0; bump(&s); bump(&s); return s }  (both results unused)
    let main_name = it.intern("main");
    let mut m = Builder::new(main_name, MirType::I64);
    let s = m.alloca(MirType::I64);
    let zero = i64c(&mut m, 0);
    m.build_void(Op::Store {
        ptr: s,
        value: zero,
    });
    for _ in 0..2 {
        m.build(
            MirType::I64,
            Op::Call {
                func: bump_name,
                args: vec![s],
            },
        );
    }
    let r = m.build(MirType::I64, Op::Load(s, MirType::I64));
    m.ret(Some(r));

    let program = program_of(vec![bump.clone(), m.finish()]);
    let table = crate::effects::CallEffects::of(&program);
    assert!(
        table.get(bump_name).is_none(),
        "an atomic add is a side effect; the callee must stay opaque"
    );
    assert_eq!(run_levels(&program, main_name, &it), [10; 3]);
}

/// A loop whose body is nothing but an atomic add has a side effect on every iteration: loop
/// deletion (and DCE) must keep it.
#[test]
fn a_loop_of_atomic_adds_survives_loop_deletion() {
    let mut it = Interner::new();
    let main_name = it.intern("main");
    let mut m = Builder::new(main_name, MirType::I64);
    let s = m.alloca(MirType::I64);
    let zero = i64c(&mut m, 0);
    m.build_void(Op::Store {
        ptr: s,
        value: zero,
    });
    let n = i64c(&mut m, 10);
    let (header, _body, exit, i) = open_loop(&mut m, n);
    let three = i64c(&mut m, 3);
    m.atomic_add(s, three);
    close_loop(&mut m, header, exit, i);
    let r = m.build(MirType::I64, Op::Load(s, MirType::I64));
    m.ret(Some(r));
    let program = program_of(vec![m.finish()]);
    assert_eq!(run_levels(&program, main_name, &it), [30; 3]);

    let mut o2 = program.clone();
    crate::optimize(&mut o2, 2);
    assert!(
        count_atomics(&o2.funcs[0]) > 0,
        "the atomic adds were deleted:\n{}",
        wukong_mir::print::print_function(&o2.funcs[0], &it)
    );
}

/// `acc += *p; atomic_add(*p, 1)` in a loop reads a different `*p` every iteration. LICM may not
/// hoist the load (nor promotion carry `*p` in a register) as if nothing in the loop wrote it.
/// From `*p = 10` over four iterations the reads are 10, 11, 12, 13: 46, not the hoisted 40.
#[test]
fn a_load_of_an_atomically_updated_location_stays_in_the_loop() {
    let mut it = Interner::new();
    // sum(p, n) = { acc = 0; for i in 0..n { acc += *p; atomic_add(*p, 1) }; return acc }
    let sum_name = it.intern("sum");
    let mut f = Builder::new(sum_name, MirType::I64);
    let p = f.add_param(MirType::Ptr);
    let n = f.add_param(MirType::I64);
    let header = f.new_block();
    let body = f.new_block();
    let exit = f.new_block();
    let zero = i64c(&mut f, 0);
    f.br(header, vec![zero, zero]);
    f.switch_to(header);
    let i = f.block_param(header, MirType::I64);
    let acc = f.block_param(header, MirType::I64);
    let c = f.build(MirType::I1, Op::Cmp(CmpOp::Slt, i, n));
    f.cond_br(c, body, vec![], exit, vec![]);
    f.switch_to(body);
    let v = f.build(MirType::I64, Op::Load(p, MirType::I64));
    let acc2 = f.build(MirType::I64, Op::Bin(BinOp::Add, acc, v));
    let one = i64c(&mut f, 1);
    f.atomic_add(p, one);
    let i2 = f.build(MirType::I64, Op::Bin(BinOp::Add, i, one));
    f.br(header, vec![i2, acc2]);
    f.switch_to(exit);
    f.ret(Some(acc));

    let main_name = it.intern("main");
    let mut m = Builder::new(main_name, MirType::I64);
    let s = m.alloca(MirType::I64);
    let ten = i64c(&mut m, 10);
    m.build_void(Op::Store { ptr: s, value: ten });
    let four = i64c(&mut m, 4);
    let r = m.build(
        MirType::I64,
        Op::Call {
            func: sum_name,
            args: vec![s, four],
        },
    );
    // 46 * 100 + the final *p (14).
    let after = m.build(MirType::I64, Op::Load(s, MirType::I64));
    let hundred = i64c(&mut m, 100);
    let scaled = m.build(MirType::I64, Op::Bin(BinOp::Mul, r, hundred));
    let out = m.build(MirType::I64, Op::Bin(BinOp::Add, scaled, after));
    m.ret(Some(out));
    let program = program_of(vec![f.finish(), m.finish()]);
    assert_eq!(run_levels(&program, main_name, &it), [4614; 3]);
}

/// The escape scan's fallback marks an atomic's pointer escaped, so an atomically updated slot is
/// never private and a `@noalias` parameter reached by one loses its promise. That is what keeps
/// Cranelift's untagged (`None` alias region) atomic sound: every other access to the location is
/// untagged too.
#[test]
fn an_atomic_add_publishes_its_location() {
    let mut it = Interner::new();
    let mut b = Builder::new(it.intern("g"), MirType::Void);
    let q = b.add_param(MirType::Ptr);
    b.set_param_attrs(
        q,
        ParamAttrs {
            noalias: true,
            ..ParamAttrs::default()
        },
    );
    let untouched = b.add_param(MirType::Ptr);
    b.set_param_attrs(
        untouched,
        ParamAttrs {
            noalias: true,
            ..ParamAttrs::default()
        },
    );
    let slot = b.alloca(MirType::I32);
    let quiet = b.alloca(MirType::I32);
    let one = b.build(MirType::I32, Op::ConstInt(1, MirType::I32));
    b.build_void(Op::Store {
        ptr: quiet,
        value: one,
    });
    b.atomic_add(slot, one);
    let k = i64c(&mut b, 2);
    let elem = b.build(
        MirType::Ptr,
        Op::Gep {
            ptr: q,
            index: k,
            elem: MirType::I32,
        },
    );
    b.atomic_add(elem, one);
    let _ = b.build(MirType::I32, Op::Load(untouched, MirType::I32));
    b.ret(None);
    let f = b.finish();
    assert!(wukong_mir::verify::verify_function(&f).is_empty());
    let alias = AliasInfo::analyze(&f);
    assert!(
        !alias.is_private_stack(slot),
        "an atomically written slot is not private"
    );
    assert!(
        alias.is_private_stack(quiet),
        "control: a plainly stored slot stays private"
    );
    assert_eq!(alias.promised_storage(q), None);
    assert_eq!(alias.promised_storage(elem), None);
    assert!(
        alias.promised_storage(untouched).is_some(),
        "control: an untouched @noalias parameter keeps its promise"
    );
}

/// A loop that would otherwise widen does not once it contains an atomic add: lane `k` of a
/// widened group would add once for `n` iterations' worth of adds, or race. Plain
/// `a[i] = a[i] + 1` widens (the control); with an atomic counter beside it the loop stays scalar
/// and the counter equals the trip count.
///
/// Two independent layers keep it scalar, and this test pins each one. `loop_info::dependence`
/// names the op and calls the loop's memory `Carried`, so the first assertion fails when that arm
/// is removed. The vectorizer's per-instruction planner (`plan_body`) also ends in
/// `_ => return None`, and the end-to-end check catches the planner admitting the op only when the
/// `dependence` arm is gone too: then `-O2` runs one atomic per four-lane group and the counter
/// reads 16. The three `vectorize.rs` arms (a predicated arm in `linearize_region` and the tree
/// walk, and `header_is_test_only`) sit behind both layers, so no test can see one of them alone.
#[test]
fn a_loop_with_an_atomic_add_is_not_vectorized() {
    fn kernel(it: &mut Interner, name: &str, with_atomic: bool) -> Function {
        let mut b = Builder::new(it.intern(name), MirType::Void);
        let a = b.add_param(MirType::Ptr);
        let cnt = b.add_param(MirType::Ptr);
        let n = b.add_param(MirType::I64);
        let (header, _body, exit, i) = open_loop(&mut b, n);
        let at = b.build(
            MirType::Ptr,
            Op::Gep {
                ptr: a,
                index: i,
                elem: MirType::I32,
            },
        );
        let x = b.build(MirType::I32, Op::Load(at, MirType::I32));
        let one = b.build(MirType::I32, Op::ConstInt(1, MirType::I32));
        let y = b.build(MirType::I32, Op::Bin(BinOp::Add, x, one));
        b.build_void(Op::Store { ptr: at, value: y });
        if with_atomic {
            b.atomic_add(cnt, one);
        }
        close_loop(&mut b, header, exit, i);
        b.ret(None);
        b.finish()
    }
    let widened = |f: &Function| {
        f.value_types.iter().any(|t| t.is_vector())
            || !f.vec_kernels.is_empty()
            || f.blocks
                .iter()
                .flat_map(|b| &b.insts)
                .any(|i| matches!(i.op, Op::VecKernelCall { .. }))
    };
    let mut it = Interner::new();
    let plain = kernel(&mut it, "plain", false);
    let counted = kernel(&mut it, "counted", true);
    let counted_name = counted.name;

    // The first layer: the atomic makes the loop's memory dependence carried, so no loop pass may
    // reorder its iterations. The control's same-element `a[i] = a[i] + 1` is not carried.
    let verdict = |f: &Function| {
        let forest = crate::loop_info::analyze_function(f);
        assert_eq!(forest.loops.len(), 1);
        forest.loops[0].dep.memory
    };
    assert_ne!(verdict(&plain), MemDep::Carried, "control");
    assert_eq!(
        verdict(&counted),
        MemDep::Carried,
        "loop_info::dependence must not see past an atomic add"
    );

    // main ={ a = [0; 64]; cnt = 0; counted(a, &cnt, 64); return cnt * 1000 + a[0] + a[63] }
    let main_name = it.intern("main");
    let mut m = Builder::new(main_name, MirType::I64);
    let a = m.alloca(MirType::Array(Box::new(MirType::I32), 64));
    let cnt = m.alloca(MirType::I32);
    let zero32 = m.build(MirType::I32, Op::ConstInt(0, MirType::I32));
    m.build_void(Op::Store {
        ptr: cnt,
        value: zero32,
    });
    let n = i64c(&mut m, 64);
    let (header, _body, exit, i) = open_loop(&mut m, n);
    let at = m.build(
        MirType::Ptr,
        Op::Gep {
            ptr: a,
            index: i,
            elem: MirType::I32,
        },
    );
    let i32v = m.build(
        MirType::I32,
        Op::Cast(wukong_mir::CastKind::Trunc, i, MirType::I32),
    );
    m.build_void(Op::Store {
        ptr: at,
        value: i32v,
    });
    close_loop(&mut m, header, exit, i);
    m.build_void(Op::Call {
        func: counted_name,
        args: vec![a, cnt, n],
    });
    let c = m.build(MirType::I32, Op::Load(cnt, MirType::I32));
    let first = m.build(MirType::I32, Op::Load(a, MirType::I32));
    let k63 = i64c(&mut m, 63);
    let last_at = m.build(
        MirType::Ptr,
        Op::Gep {
            ptr: a,
            index: k63,
            elem: MirType::I32,
        },
    );
    let last = m.build(MirType::I32, Op::Load(last_at, MirType::I32));
    let thousand = m.build(MirType::I32, Op::ConstInt(1000, MirType::I32));
    let scaled = m.build(MirType::I32, Op::Bin(BinOp::Mul, c, thousand));
    let s1 = m.build(MirType::I32, Op::Bin(BinOp::Add, scaled, first));
    let s2 = m.build(MirType::I32, Op::Bin(BinOp::Add, s1, last));
    let wide = m.build(
        MirType::I64,
        Op::Cast(wukong_mir::CastKind::SExt, s2, MirType::I64),
    );
    m.ret(Some(wide));

    let program = program_of(vec![plain, counted, m.finish()]);
    // cnt = 64; a[0] = 0 + 1; a[63] = 63 + 1.
    assert_eq!(run_levels(&program, main_name, &it), [64065; 3]);

    // Keep both kernels as roots: `plain` has no caller, and dead-function elimination would
    // otherwise take it before it could serve as the control.
    let roots = [it.intern("plain"), counted_name];
    let mut o2 = program.clone();
    crate::optimize_preserving(&mut o2, 2, &roots);
    let plain = by_name(&o2, &mut it, "plain");
    let counted = by_name(&o2, &mut it, "counted");
    assert!(
        widened(&plain),
        "control: the loop without the atomic widens\n{}",
        wukong_mir::print::print_function(&plain, &it)
    );
    assert!(
        !widened(&counted),
        "a loop containing an atomic add was widened\n{}",
        wukong_mir::print::print_function(&counted, &it)
    );
    assert_eq!(count_atomics(&counted), 1);
}

fn by_name(p: &Program, it: &mut Interner, name: &str) -> Function {
    let sym = it.intern(name);
    p.funcs.iter().find(|f| f.name == sym).unwrap().clone()
}
