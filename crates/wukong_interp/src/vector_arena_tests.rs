use super::*;
use wukong_mir::{verify::verify_program, Builder};

fn interpreter<'a>(program: &'a Program, interner: &'a Interner) -> Interp<'a, 'a> {
    Interp {
        program,
        interner,
        memory: Vec::new(),
        stdout: Vec::new(),
        scratch_arena: ScratchSlots::default(),
        slots: SlotReuse::default(),
        frames: Vec::new(),
        call_args: Vec::new(),
        scratch: Vec::new(),
        vecs: Vec::new(),
        vector_peak: 0,
        data_addrs: HashMap::new(),
        function_indices: None,
        static_indices: None,
        accel: None,
        depth: 0,
    }
}

fn vector() -> MirType {
    MirType::Vec(Box::new(MirType::I64), 4)
}
fn constant(b: &mut Builder, n: i128) -> ValueId {
    b.build(MirType::I64, Op::ConstInt(n, MirType::I64))
}
fn verified(funcs: Vec<Function>) -> Program {
    let program = Program {
        funcs,
        ..Program::new()
    };
    assert!(
        verify_program(&program).is_empty(),
        "{:?}",
        verify_program(&program)
    );
    program
}

#[test]
fn vector_arena_loop_growth_is_bounded_and_lanes_survive() {
    let mut interner = Interner::new();
    let name = interner.intern("loop");
    let mut b = Builder::new(name, MirType::I64);
    let n = b.add_param(MirType::I64);
    let out = b.add_param(MirType::Ptr);
    let zero = constant(&mut b, 0);
    let one = constant(&mut b, 1);
    let initial = b.build(vector(), Op::Iota(vector()));
    let increment = b.build(vector(), Op::Splat(one));
    let head = b.new_block();
    let body = b.new_block();
    let done = b.new_block();
    let i = b.block_param(head, MirType::I64);
    let acc = b.block_param(head, vector());
    b.br(head, vec![zero, initial]);
    b.switch_to(head);
    let condition = b.build(MirType::I1, Op::Cmp(CmpOp::Slt, i, n));
    b.cond_br(condition, body, vec![], done, vec![]);
    b.switch_to(body);
    b.build(vector(), Op::Neg(acc)); // A dead result retained only until its SSA slot is overwritten.
    let next = b.build(vector(), Op::Bin(BinOp::Add, acc, increment));
    let next_i = b.build(MirType::I64, Op::Bin(BinOp::Add, i, one));
    b.br(head, vec![next_i, next]);
    b.switch_to(done);
    b.build_void(Op::Store {
        ptr: out,
        value: acc,
    });
    b.ret(Some(i));
    let program = verified(vec![b.finish()]);
    let function = &program.funcs[0];
    let vector_registers = function
        .value_types
        .iter()
        .filter(|t| matches!(t, MirType::Vec(..)))
        .count();
    let vector_results_per_block = function
        .blocks
        .iter()
        .map(|block| {
            block
                .insts
                .iter()
                .filter(|inst| {
                    inst.result
                        .is_some_and(|r| matches!(function.value_type(r), MirType::Vec(..)))
                })
                .count()
        })
        .max()
        .unwrap();
    let mut peaks = Vec::new();
    for iterations in [100, 10_000] {
        let mut interp = interpreter(&program, &interner);
        interp.memory = vec![Value::Int(-99); 4];
        assert_eq!(
            interp
                .run_function(function, &[Value::Int(iterations), Value::Ptr(0)])
                .unwrap(),
            Value::Int(iterations)
        );
        assert_eq!(
            interp.memory,
            (0..4)
                .map(|lane| Value::Int(iterations + lane))
                .collect::<Vec<_>>()
        );
        assert!(interp.vecs.is_empty());
        assert!(
            interp.vector_peak <= vector_registers + vector_results_per_block,
            "peak {} exceeds {} vector registers + {} block results",
            interp.vector_peak,
            vector_registers,
            vector_results_per_block
        );
        peaks.push(interp.vector_peak);
    }
    assert_eq!(
        peaks[0], peaks[1],
        "retention must depend on MIR size, not iteration count"
    );
}

#[test]
fn vector_arena_calls_promote_returns_and_preserve_caller_aliases() {
    let mut interner = Interner::new();
    let neg_name = interner.intern("neg");
    let id_name = interner.intern("identity");
    let caller_name = interner.intern("caller");
    let mut neg = Builder::new(neg_name, vector());
    let arg = neg.add_param(vector());
    let result = neg.build(vector(), Op::Neg(arg));
    neg.build(vector(), Op::Iota(vector())); // Returned vector is not the last arena entry.
    neg.ret(Some(result));
    let mut id = Builder::new(id_name, vector());
    let arg = id.add_param(vector());
    id.build(
        vector(),
        Op::Call {
            func: neg_name,
            args: vec![arg],
        },
    );
    id.ret(Some(arg)); // This handle belongs to the caller, despite a nested callee's new result.
    let mut caller = Builder::new(caller_name, MirType::Void);
    let left = caller.add_param(MirType::Ptr);
    let right = caller.add_param(MirType::Ptr);
    let original = caller.build(vector(), Op::Iota(vector()));
    let changed = caller.build(
        vector(),
        Op::Call {
            func: neg_name,
            args: vec![original],
        },
    );
    let unchanged = caller.build(
        vector(),
        Op::Call {
            func: id_name,
            args: vec![original],
        },
    );
    caller.build_void(Op::Store {
        ptr: left,
        value: unchanged,
    });
    caller.build_void(Op::Store {
        ptr: right,
        value: changed,
    });
    caller.ret(None);
    let program = verified(vec![neg.finish(), id.finish(), caller.finish()]);
    let mut interp = interpreter(&program, &interner);
    interp.memory = vec![Value::Unit; 8];
    for _ in 0..64 {
        assert_eq!(
            interp
                .run_function(&program.funcs[2], &[Value::Ptr(0), Value::Ptr(4)])
                .unwrap(),
            Value::Unit
        );
        assert_eq!(interp.memory, [0, 1, 2, 3, 0, -1, -2, -3].map(Value::Int));
        assert!(interp.vecs.is_empty());
    }
    assert_eq!(interp.vector_peak, 4); // Original, promoted negative, nested identity/neg temporaries.
}

#[test]
fn vector_arena_compaction_preserves_branch_swaps_and_duplicate_handles() {
    let mut interner = Interner::new();
    let mut b = Builder::new(interner.intern("swap"), MirType::Void);
    let out = b.add_param(MirType::Ptr);
    let zero = constant(&mut b, 0);
    let one = constant(&mut b, 1);
    let three = constant(&mut b, 3);
    b.build(vector(), Op::Splat(three)); // Unused leading entry creates a hole on later passes.
    let initial = b.build(vector(), Op::Iota(vector()));
    let negative = b.build(vector(), Op::Neg(initial));
    let head = b.new_block();
    let body = b.new_block();
    let done = b.new_block();
    let i = b.block_param(head, MirType::I64);
    let a = b.block_param(head, vector());
    let other = b.block_param(head, vector());
    let alias = b.block_param(head, vector());
    b.br(head, vec![zero, initial, negative, initial]);
    b.switch_to(head);
    let condition = b.build(MirType::I1, Op::Cmp(CmpOp::Slt, i, three));
    b.cond_br(condition, body, vec![], done, vec![]);
    b.switch_to(body);
    let next = b.build(MirType::I64, Op::Bin(BinOp::Add, i, one));
    b.build(vector(), Op::Neg(a));
    b.br(head, vec![next, other, a, other]); // Parallel-copy swap plus duplicated destination roots.
    b.switch_to(done);
    let sum = b.build(vector(), Op::Bin(BinOp::Add, a, alias));
    b.build_void(Op::Store {
        ptr: out,
        value: sum,
    });
    b.ret(None);
    let program = verified(vec![b.finish()]);
    let mut interp = interpreter(&program, &interner);
    interp.memory = vec![Value::Unit; 4];
    interp
        .run_function(&program.funcs[0], &[Value::Ptr(0)])
        .unwrap();
    assert_eq!(interp.memory, [0, -2, -4, -6].map(Value::Int));
    assert!(interp.vecs.is_empty());

    // Directly pin relocation when two registers alias a vector after dead entries.
    let parent = interp.push_vec(vec![Value::Int(7); 4]).unwrap();
    let base = interp.vecs.len();
    interp.push_vec(vec![Value::Int(99); 4]).unwrap();
    let live = interp.push_vec(vec![Value::Int(3); 4]).unwrap();
    let mut regs = [live, parent, live];
    interp
        .compact_vectors(base, &mut regs, &mut Vec::new())
        .unwrap();
    assert_eq!(regs, [Value::VecRef(base), parent, Value::VecRef(base)]);
    assert_eq!(interp.vec_lanes(regs[0], 4).unwrap(), [Value::Int(3); 4]);
    assert_eq!(interp.vec_lanes(parent, 4).unwrap(), [Value::Int(7); 4]);
    assert_eq!(interp.vecs.len(), base + 1);
}

#[test]
fn vector_arena_error_cleanup_and_reuse() {
    let mut interner = Interner::new();
    let mut bad = Builder::new(interner.intern("bad"), MirType::Void);
    bad.build(vector(), Op::Iota(vector()));
    // Default terminator is unreachable: execution fails after allocating the vector.
    let mut good = Builder::new(interner.intern("good"), MirType::I64);
    let v = good.build(vector(), Op::Iota(vector()));
    let lane = good.build(MirType::I64, Op::ExtractLane(v, 3));
    good.ret(Some(lane));
    let program = verified(vec![bad.finish(), good.finish()]);
    let mut interp = interpreter(&program, &interner);
    let caller_value = interp.push_vec(vec![Value::Int(17); 4]).unwrap();
    for _ in 0..32 {
        assert!(interp
            .run_function(&program.funcs[0], &[])
            .unwrap_err()
            .contains("unreachable"));
        assert_eq!(interp.vecs.len(), 1);
        assert_eq!(interp.depth, 0);
        assert!(interp.frames.iter().all(Vec::is_empty));
        assert!(interp.scratch.is_empty());
        assert_eq!(
            interp.vec_lanes(caller_value, 4).unwrap(),
            [Value::Int(17); 4]
        );
        assert_eq!(
            interp.run_function(&program.funcs[1], &[]).unwrap(),
            Value::Int(3)
        );
        assert_eq!(interp.vecs.len(), 1);
    }
}

#[test]
fn vector_arena_rejects_nested_lanes_and_invalid_handles() {
    let mut interner = Interner::new();
    let mut b = Builder::new(interner.intern("nested"), MirType::Void);
    let scalar = constant(&mut b, 1);
    let v = b.build(vector(), Op::Splat(scalar));
    b.build(MirType::Vec(Box::new(vector()), 2), Op::Splat(v));
    b.ret(None);
    let program = verified(vec![b.finish()]);
    let mut interp = interpreter(&program, &interner);
    assert!(interp
        .run_function(&program.funcs[0], &[])
        .unwrap_err()
        .contains("nested vectors"));
    assert!(interp.vecs.is_empty());
    assert!(interp.vec_lanes(Value::VecRef(usize::MAX), 4).is_err());
    interp.memory = vec![Value::Int(0); 4];
    let err = interp
        .eval(
            &Op::Store {
                ptr: ValueId(0),
                value: ValueId(1),
            },
            &program.funcs[0],
            &[Value::Ptr(0), Value::VecRef(usize::MAX)],
            None,
        )
        .unwrap_err();
    assert!(err.contains("invalid SIMD vector handle"));
    assert_eq!(interp.memory, [Value::Int(0); 4]);
}

#[test]
fn vector_arena_capacity_overflow_is_fallible_without_allocating() {
    // usize::MAX Values exceed Vec's representable allocation size on every target.
    let error = collect_lanes(std::iter::repeat_n(Value::Int(0), usize::MAX)).unwrap_err();
    assert!(error.contains("allocation failed"));
}
