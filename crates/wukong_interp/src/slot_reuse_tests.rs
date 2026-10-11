//! The interpreter's reuse of its slot memory (`SlotReuse`): a returning frame releases its
//! allocas, growth a frame made that must outlive it (a heap block) is never cut, and `free` feeds a
//! size-bucketed free list whose blocks come back re-zeroed with the new element type's zero.

use super::*;
use wukong_span::SourceId;
use Value::{Float as F, Int as I, Ptr as P};

fn interpreter<'a>(program: &'a Program, interner: &'a Interner) -> Interp<'a, 'a> {
    Interp {
        program,
        interner,
        memory: null_reserved_memory(),
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

fn lower(src: &str, interner: &mut Interner) -> Program {
    let (module, pd) = wukong_parser::parse_module(src, SourceId(0), interner);
    assert!(pd.iter().all(|d| !d.is_error()), "parse: {pd:?}");
    let (sema, sd) = wukong_sema::check(&module, interner);
    assert!(sd.iter().all(|d| !d.is_error()), "sema: {sd:?}");
    let (program, ld) = wukong_mir_build::lower_program(&module, &sema, interner);
    assert!(ld.iter().all(|d| !d.is_error()), "lower: {ld:?}");
    program
}

/// `alloc_<T>(count)` of 4-byte elements (`float` picks the typed zero); returns the base slot.
fn heap_alloc(interp: &mut Interp<'_, '_>, count: i128, float: bool) -> usize {
    match interp
        .intrinsic("wukong_rt_alloc", &[I(count), I(4), I(i128::from(float))])
        .unwrap()
    {
        P(base) => base,
        other => panic!("alloc returned {other:?}"),
    }
}

fn heap_free(interp: &mut Interp<'_, '_>, base: usize) {
    assert_eq!(
        interp.intrinsic("wukong_rt_free", &[P(base)]).unwrap(),
        Value::Unit
    );
}

#[test]
fn a_freed_block_is_reused_rezeroed_with_the_new_type_and_memory_stays_bounded() {
    let program = Program::new();
    let interner = Interner::new();
    let mut it = interpreter(&program, &interner);
    let other = heap_alloc(&mut it, 3, false);
    let first = heap_alloc(&mut it, 100, false);
    assert_ne!(other, first);
    it.memory[first] = I(77);
    let grown = it.memory.len();
    for round in 0..1000 {
        heap_free(&mut it, first);
        // Alternate the element type: the reused slots must hold the NEW type's zero.
        let float = round % 2 == 0;
        let again = heap_alloc(&mut it, 100, float);
        assert_eq!(again, first, "the freed block is handed out again");
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
        "an alloc/free loop must not grow the run's memory"
    );
}

#[test]
fn the_free_list_fits_requests_and_ignores_foreign_and_double_frees() {
    let program = Program::new();
    let interner = Interner::new();
    let mut it = interpreter(&program, &interner);
    let a = heap_alloc(&mut it, 8, true);
    heap_free(&mut it, a);
    heap_free(&mut it, a); // a double free: ignored, so `a` is handed out once below
    heap_free(&mut it, NULL_SLOT); // not a heap block: ignored

    // 8 slots for 3 would waste more than half the block: a fresh block instead.
    let tiny = heap_alloc(&mut it, 3, true);
    assert_ne!(tiny, a);
    // 8 slots for 5 fits: `a` again, re-zeroed with the integer zero.
    let b = heap_alloc(&mut it, 5, false);
    assert_eq!(b, a);
    assert!(it.memory[b..b + 5].iter().all(|v| *v == I(0)));
    let c = heap_alloc(&mut it, 5, false);
    assert_ne!(c, a, "a block freed twice is still handed out only once");
    // Freeing `b` returns the block's whole 8-slot capacity, not the 5 slots requested.
    heap_free(&mut it, b);
    assert_eq!(heap_alloc(&mut it, 8, true), a);
    // A zero-length allocation is never a block, so it is neither recorded nor reused.
    let empty = heap_alloc(&mut it, 0, false);
    heap_free(&mut it, empty);
    assert!(it.slots.heap_free.is_empty());
}

/// `main` calls `cell` (a 64-lane local array) `n` times and then allocates one heap block. Without
/// frame release every call's 64 slots stay in memory, so the block's base grows with `n`; with
/// it, the base is the same for every `n`. `cell(k)` sets a[k] = k and returns a[k] + a[63 - k]
/// = k (63 - k never equals k), so the result is the sum of i % 64 over i < n: 10 * (0 + ... +
/// 63) = 20160 for n = 640 and 0 + ... + 63 = 2016 for n = 64.
#[test]
fn a_returning_frame_releases_its_allocas() {
    let mut bases = Vec::new();
    for (n, expected) in [(64, 2016), (640, 20160)] {
        let mut interner = Interner::new();
        let src = format!(
            "fn cell(k: i32) -> i32 {{ let mut a: [i32; 64] = [0; 64]; a[k] = k; \
             return a[k] + a[63 - k]; }} \
             fn main() -> i32 {{ let mut s: i32 = 0; let mut i: i32 = 0; \
             while i < {n} {{ s = s + cell(i % 64); i = i + 1; }} \
             let h: []i32 = alloc_i32(1); return s; }}"
        );
        let program = lower(&src, &mut interner);
        let main = interner.intern("main");
        let func = program.function(main).unwrap();
        let mut it = interpreter(&program, &interner);
        assert_eq!(it.run_function(func, &[]).unwrap(), I(expected));
        let live: Vec<usize> = it.slots.heap_live.keys().copied().collect();
        assert_eq!(
            live.len(),
            1,
            "exactly the one `alloc_i32(1)` block is live"
        );
        bases.push(live[0]);
    }
    assert_eq!(
        bases[0], bases[1],
        "memory at the end of main must not depend on how many calls returned before it"
    );
}

/// A heap block a callee allocates after its own locals outlives the callee's frame: the release
/// stops at it. `make` fills h = 1, 4, 7, 10, 13 (h[i] = 3i + 1) behind a local array; `noise`
/// then pushes and writes a 32-lane local where a wrongly released `h` would have been. Folding h
/// as t = t*100 + h[i] gives 1 04 07 10 13 = 104071013.
#[test]
fn heap_growth_inside_a_frame_survives_its_release() {
    let mut interner = Interner::new();
    let src = "fn make(n: i64) -> []i32 { let mut buf: [i32; 8] = [0; 8]; \
               let mut h: []i32 = alloc_i32(n); \
               for i in 0..n { h[i] = (i as i32) * 3 + 1; buf[i] = h[i]; } return h; } \
               fn noise() -> i32 { let mut a: [i32; 32] = [0; 32]; \
               for i in 0..32 { a[i] = 555; } return a[31]; } \
               fn main() -> i32 { let h: []i32 = make(5); let x: i32 = noise(); \
               let mut t: i32 = 0; for i in 0..5 { t = t * 100 + h[i]; } \
               return t + x - 555; }";
    let program = lower(src, &mut interner);
    let main = interner.intern("main");
    let func = program.function(main).unwrap();
    let mut it = interpreter(&program, &interner);
    assert_eq!(it.run_function(func, &[]).unwrap(), I(104071013));
}

fn scratch(interp: &mut Interp<'_, '_>, name: &str, args: &[Value]) -> Value {
    interp.intrinsic(name, args).unwrap()
}

/// The live-allocation table kernel operands are checked against (`marshal::LiveBlocks`) follows
/// the reuse above exactly: a heap block is live until `free` and comes back as a block of the
/// size asked for, not of its capacity; a scratch region is live until `scratch_release` (an
/// empty one never is); a double free and a foreign free change nothing.
#[test]
fn the_live_table_follows_alloc_free_and_scratch_release() {
    let program = Program::new();
    let interner = Interner::new();
    let mut it = interpreter(&program, &interner);
    it.slots = SlotReuse::tracked();
    let a = heap_alloc(&mut it, 8, true);
    let b = heap_alloc(&mut it, 2, false);
    assert_eq!(it.slots.live.blocks(), [(a, a + 8), (b, b + 2)]);
    heap_free(&mut it, a);
    heap_free(&mut it, a);
    heap_free(&mut it, NULL_SLOT);
    assert_eq!(it.slots.live.blocks(), [(b, b + 2)]);
    assert_eq!(
        heap_alloc(&mut it, 5, false),
        a,
        "the freed block is reused"
    );
    assert_eq!(it.slots.live.blocks(), [(a, a + 5), (b, b + 2)]);
    assert!(
        it.slots.live.check("k", 0, a, 6).is_err(),
        "not its capacity"
    );

    let mark = scratch(&mut it, "wukong_rt_scratch_mark", &[]);
    let P(s) = scratch(&mut it, "wukong_rt_scratch_alloc", &[I(4), I(4), I(1)]) else {
        panic!("scratch alloc")
    };
    let P(empty) = scratch(&mut it, "wukong_rt_scratch_alloc", &[I(0), I(4), I(1)]) else {
        panic!("scratch alloc")
    };
    assert_eq!(it.slots.live.blocks(), [(a, a + 5), (b, b + 2), (s, s + 4)]);
    assert!(it.slots.live.check("k", 0, empty, 1).is_err());
    scratch(&mut it, "wukong_rt_scratch_release", &[mark]);
    assert_eq!(it.slots.live.blocks(), [(a, a + 5), (b, b + 2)]);
    // Reused at its depth as a block of the new request's size.
    let again = scratch(&mut it, "wukong_rt_scratch_alloc", &[I(3), I(4), I(1)]);
    assert_eq!(again, P(s));
    assert_eq!(it.slots.live.blocks(), [(a, a + 5), (b, b + 2), (s, s + 3)]);
}

/// Every alloca is a live block until its frame's memory is released, and only then. `cell`
/// returns `k` as in `a_returning_frame_releases_its_allocas` (the result is the sum of `i % 64`),
/// and none of its 64 calls' 64-lane arrays survives in the table. What does survive is exactly
/// what survives in memory: the heap block, and `main`'s own locals, which sit below that
/// persistent growth and are therefore never cut (the conservative `SlotReuse` floor).
#[test]
fn a_returning_frame_drops_its_allocas_from_the_live_table() {
    let mut interner = Interner::new();
    let src = "fn cell(k: i32) -> i32 { let mut a: [i32; 64] = [0; 64]; a[k] = k; \
               return a[k] + a[63 - k]; } \
               fn main() -> i32 { let mut s: i32 = 0; let mut i: i32 = 0; \
               while i < 64 { s = s + cell(i % 64); i = i + 1; } \
               let h: []i32 = alloc_i32(3); return s; }";
    let program = lower(src, &mut interner);
    let main = interner.intern("main");
    let func = program.function(main).unwrap();
    let mut it = interpreter(&program, &interner);
    it.slots = SlotReuse::tracked();
    assert_eq!(it.run_function(func, &[]).unwrap(), I(2016));
    let heap: Vec<usize> = it.slots.heap_live.keys().copied().collect();
    assert_eq!(heap.len(), 1);
    let blocks = it.slots.live.blocks();
    assert_eq!(blocks.last(), Some(&(heap[0], heap[0] + 3)));
    assert_eq!(blocks.last().unwrap().1, it.memory.len());
    assert!(
        blocks.iter().all(|&(start, end)| end - start != 64),
        "a returned `cell` frame's array is still live: {blocks:?}"
    );
    assert!(blocks[..blocks.len() - 1]
        .iter()
        .all(|&(_, end)| end <= heap[0]));
}
