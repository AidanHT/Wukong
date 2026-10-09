//! The interpreter's model of the scratch allocator (`ScratchSlots`): the mark is the live-allocation
//! count native reports, released regions are reused at their depth and re-zeroed with the right
//! typed zero, and a balanced `mark; alloc; release` loop does not grow the run's memory.

use super::*;
use Value::{Float as F, Int as I, Ptr as P};

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

fn call(interp: &mut Interp<'_, '_>, name: &str, args: &[Value]) -> Value {
    interp.intrinsic(name, args).unwrap()
}

/// `scratch_<T>(count)` of 4-byte elements (`float` picks the typed zero); returns the base slot.
fn alloc(interp: &mut Interp<'_, '_>, count: i128, float: bool) -> usize {
    match call(
        interp,
        "wukong_rt_scratch_alloc",
        &[I(count), I(4), I(i128::from(float))],
    ) {
        P(base) => base,
        other => panic!("scratch alloc returned {other:?}"),
    }
}

fn mark(interp: &mut Interp<'_, '_>) -> Value {
    call(interp, "wukong_rt_scratch_mark", &[])
}

fn release(interp: &mut Interp<'_, '_>, to: i128) {
    assert_eq!(
        call(interp, "wukong_rt_scratch_release", &[I(to)]),
        Value::Unit
    );
}

#[test]
fn marks_count_live_allocations_and_release_is_lifo() {
    let program = Program::new();
    let interner = Interner::new();
    let mut it = interpreter(&program, &interner);
    assert_eq!(mark(&mut it), I(0));
    let a = alloc(&mut it, 4, false);
    assert!(it.memory[a..a + 4].iter().all(|v| *v == I(0)));
    assert_eq!(mark(&mut it), I(1));
    alloc(&mut it, 0, false); // a zero-length allocation still counts
    alloc(&mut it, 2, true);
    assert_eq!(mark(&mut it), I(3));
    release(&mut it, 7); // past the count: nothing
    assert_eq!(mark(&mut it), I(3));
    release(&mut it, 1);
    assert_eq!(mark(&mut it), I(1));
    release(&mut it, -4); // negative: everything
    assert_eq!(mark(&mut it), I(0));
}

#[test]
fn a_released_region_is_reused_rezeroed_with_the_new_type_and_memory_stays_bounded() {
    let program = Program::new();
    let interner = Interner::new();
    let mut it = interpreter(&program, &interner);
    let _outer = alloc(&mut it, 3, false);
    let m = 1;
    let first = alloc(&mut it, 100, false);
    it.memory[first] = I(77);
    let grown = it.memory.len();
    for round in 0..1000 {
        release(&mut it, m);
        // Alternate the element type: the reused slots must hold the NEW type's zero.
        let float = round % 2 == 0;
        let again = alloc(&mut it, 100, float);
        assert_eq!(
            again, first,
            "the region released at this depth is handed out again"
        );
        let zero = if float { F(0.0) } else { I(0) };
        assert!(
            it.memory[again..again + 100].iter().all(|v| *v == zero),
            "round {round}: reused slots must be re-zeroed as {zero:?}"
        );
        it.memory[again + 99] = if float { F(1.5) } else { I(5) };
    }
    assert_eq!(
        it.memory.len(),
        grown,
        "a balanced mark/alloc/release loop must not grow the run's memory"
    );
}

#[test]
fn a_region_too_small_for_the_next_request_is_replaced_not_overrun() {
    let program = Program::new();
    let interner = Interner::new();
    let mut it = interpreter(&program, &interner);
    let small = alloc(&mut it, 2, false);
    release(&mut it, 0);
    let big = alloc(&mut it, 5, false);
    assert_ne!(big, small, "a 2-slot region cannot hold 5 elements");
    assert!(it.memory[big..big + 5].iter().all(|v| *v == I(0)));
    // The larger region is the one reused from now on.
    release(&mut it, 0);
    assert_eq!(alloc(&mut it, 3, false), big);
}
