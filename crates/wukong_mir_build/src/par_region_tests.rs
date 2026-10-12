//! Tests of the mid-function `@parallel` region outliner's reach past fixed arrays (`par_region`):
//! `[]T` slice captures and their run-time overlap guard, run-time trip counts, and kernel builtin
//! calls inside a region — each with its accept AND its reject side, plus the corpus decline census.
//!
//! A region that is outlined when it should not be is a data race the differential gates can miss
//! (it is nondeterministic on native and invisible on the interpreter, which runs regions serially),
//! so every reject test below names the REASON it expects (`par_region::take_trace`): a test that
//! only asserted "no `wukong_parallel_for`" would stay green if the loop were refused for an
//! unrelated, unmodeled construct, and would then prove nothing about the proof.

use std::collections::BTreeMap;
use std::path::Path;

use wukong_mir::verify::verify_function;
use wukong_mir::{CastKind, Function, Op, Program};
use wukong_span::{Interner, SourceId};

use crate::{lower_program_with, par_region, LowerOptions};

fn lower(src: &str) -> (Program, Interner, Vec<Result<(), &'static str>>) {
    let mut interner = Interner::new();
    let (module, pdiags) = wukong_parser::parse_module(src, SourceId(0), &mut interner);
    assert!(pdiags.is_empty(), "parse: {pdiags:?}");
    let (sema, sdiags) = wukong_sema::check(&module, &interner);
    assert!(sdiags.iter().all(|d| !d.is_error()), "sema: {sdiags:?}");
    par_region::take_trace();
    let (prog, diags) = lower_program_with(&module, &sema, &mut interner, LowerOptions::default());
    let trace = par_region::take_trace();
    assert!(diags.iter().all(|d| !d.is_error()), "lower: {diags:?}");
    for f in &prog.funcs {
        let errs = verify_function(f);
        assert!(errs.is_empty(), "{}: {errs:?}", interner.resolve(f.name));
    }
    (prog, interner, trace)
}

fn func<'p>(prog: &'p Program, interner: &Interner, name: &str) -> &'p Function {
    prog.funcs
        .iter()
        .find(|f| interner.resolve(f.name) == name)
        .unwrap_or_else(|| panic!("no function `{name}`"))
}

fn calls(f: &Function, interner: &Interner, callee: &str) -> usize {
    f.blocks
        .iter()
        .flat_map(|b| b.insts.iter())
        .filter(|i| matches!(&i.op, Op::Call { func, .. } if interner.resolve(*func) == callee))
        .count()
}

fn pfor(prog: &Program, interner: &Interner) -> usize {
    prog.funcs
        .iter()
        .map(|f| calls(f, interner, "wukong_parallel_for"))
        .sum()
}

/// The `ptrtoint` range ends of `f`: a nonzero count is a run-time overlap guard (none of these
/// sources takes an address any other way).
fn guard_ends(f: &Function) -> usize {
    f.blocks
        .iter()
        .flat_map(|b| b.insts.iter())
        .filter(|i| matches!(i.op, Op::Cast(CastKind::PtrToInt, _, _)))
        .count()
}

/// The one outlined region body of a program.
fn region_body<'p>(prog: &'p Program, interner: &Interner) -> &'p Function {
    let mut bodies = prog
        .funcs
        .iter()
        .filter(|f| interner.resolve(f.name).starts_with("wukong$par$"));
    let body = bodies.next().expect("an outlined region body");
    assert!(bodies.next().is_none(), "exactly one region body");
    body
}

/// A region over `[]f32` parameters, written per iteration (`out[h*8 + j]`) and read at the same
/// partition (`x[h*8 + j]`). The leading statement keeps the loop a mid-function region.
const HEADS: &str = "module m
@parallel
fn heads(x: []f32, mut out: []f32) {
    let mut guard: i32 = 0;
    guard = guard + 1;
    for h in 0..4 {
        let mut s: f32 = 0.0;
        for j in 0..8 { s = s + x[h * 8 + j]; }
        for j in 0..8 { out[h * 8 + j] = x[h * 8 + j] - s; }
    }
}
";

/// Slices are outlined, behind a run-time guard: `x` and `out` may be one buffer at any offset,
/// which no name can tell, so the call is versioned on their address ranges and the serial arm
/// calls the same outlined body directly.
#[test]
fn a_slice_region_is_outlined_behind_a_run_time_overlap_guard() {
    let (prog, it, trace) = lower(HEADS);
    assert_eq!(
        trace,
        vec![Ok(())],
        "the head loop is the one region candidate"
    );
    assert_eq!(pfor(&prog, &it), 1);
    let heads = func(&prog, &it, "heads");
    assert_eq!(
        guard_ends(heads),
        4,
        "one range (two ends) per guarded slice"
    );
    let body = region_body(&prog, &it);
    let body_name = it.resolve(body.name).to_string();
    assert_eq!(
        calls(heads, &it, &body_name),
        1,
        "the may-overlap arm runs the outlined body directly, over every chunk"
    );
}

/// The accesses of `x` and `out` share one partition, so the union proof passes and the guard
/// also admits ONE start address (`heads(x, x)`, in place): an `eq` of the two range starts.
#[test]
fn a_guard_admits_one_start_address_when_the_union_proof_passes() {
    let (prog, it, _) = lower(HEADS);
    let heads = func(&prog, &it, "heads");
    let eqs = heads
        .blocks
        .iter()
        .flat_map(|b| b.insts.iter())
        .filter(|i| matches!(i.op, Op::Cmp(wukong_mir::CmpOp::Eq, _, _)))
        .count();
    assert_eq!(eqs, 1, "the in-place disjunct");
    // A read at another iteration's partition fails the union proof: the pair may then only be
    // disjoint, never the same buffer.
    let shifted = HEADS.replace("x[h * 8 + j] - s", "x[(h * 8 + j + 8) % 32] - s");
    let (prog, it, trace) = lower(&shifted);
    assert_eq!(trace, vec![Ok(())]);
    let heads = func(&prog, &it, "heads");
    let eqs = heads
        .blocks
        .iter()
        .flat_map(|b| b.insts.iter())
        .filter(|i| matches!(i.op, Op::Cmp(wukong_mir::CmpOp::Eq, _, _)))
        .count();
    assert_eq!(eqs, 0, "no in-place disjunct without the union proof");
}

/// Slices of different element scalars are never one buffer, so no guard is needed.
#[test]
fn slices_of_different_element_types_need_no_guard() {
    let src = "module m
@parallel
fn heads(x: []i32, mut out: []f32) {
    let mut guard: i32 = 0;
    guard = guard + 1;
    for h in 0..4 {
        let mut s: i32 = 0;
        for j in 0..8 { s = s + x[h * 8 + j]; }
        for j in 0..8 { out[h * 8 + j] = (x[h * 8 + j] - s) as f32; }
    }
}
";
    let (prog, it, trace) = lower(src);
    assert_eq!(trace, vec![Ok(())]);
    assert_eq!(pfor(&prog, &it), 1);
    assert_eq!(guard_ends(func(&prog, &it, "heads")), 0);
}

/// The per-buffer proof binds a slice exactly as it binds an array: a write that reaches the next
/// iteration's partition (`out[h*8 + j]` with `j` up to 9 > C = 8) stays serial, for the proof's
/// reason.
#[test]
fn an_overlapping_slice_write_stays_serial() {
    let src = HEADS.replace("for j in 0..8 { out", "for j in 0..10 { out");
    let (prog, it, trace) = lower(&src);
    assert_eq!(
        trace,
        vec![Err(
            "a written buffer's accesses are not per-iteration disjoint"
        )]
    );
    assert_eq!(pfor(&prog, &it), 0);
}

/// A slice written at an index that does not carry the loop variable as a digit is a race.
#[test]
fn a_slice_scatter_stays_serial() {
    let src = "module m
@parallel
fn scatter(ids: []i32, mut out: []f32) {
    let mut guard: i32 = 0;
    guard = guard + 1;
    for t in 0..8 { out[ids[t]] = 1.0; }
}
";
    let (prog, it, trace) = lower(src);
    assert_eq!(
        trace,
        vec![Err(
            "a written buffer's accesses are not per-iteration disjoint"
        )]
    );
    assert_eq!(pfor(&prog, &it), 0);
}

/// A slice VALUE handed on whole (to a user function, a `let`) still declines: the callee could
/// write anything through it.
#[test]
fn a_slice_passed_whole_to_a_user_function_stays_serial() {
    let src = "module m
fn touch(mut s: []f32, i: i32) { s[i] = 0.0; }
@parallel
fn f(mut out: []f32) {
    let mut guard: i32 = 0;
    guard = guard + 1;
    for t in 0..8 { touch(out, t); }
}
";
    let (prog, it, trace) = lower(src);
    assert_eq!(trace.len(), 1);
    assert!(trace[0].is_err(), "{trace:?}");
    assert_eq!(pfor(&prog, &it), 0);
}

/// A region over a slice LOCAL (`alloc_f32`) is a capture too, guarded against the parameter it
/// may view.
#[test]
fn a_slice_local_is_captured_and_guarded() {
    let src = "module m
@parallel
fn f(x: []f32, n: i64) {
    let mut tmp: []f32 = alloc_f32(32);
    for h in 0..4 {
        for j in 0..8 { tmp[h * 8 + j] = x[h * 8 + j] * 2.0; }
    }
    free(tmp);
}
";
    let (prog, it, trace) = lower(src);
    assert_eq!(trace, vec![Ok(())]);
    assert_eq!(pfor(&prog, &it), 1);
    assert_eq!(guard_ends(func(&prog, &it, "f")), 4);
}

/// The single-digit proof places every access of a buffer in ONE band `[k*C, k*C + C)` past
/// `hh*C`: a partition shifted by a constant (`out[(h + 1)*8 + j]`, band 1) is still one row per
/// iteration, while a buffer written at one band and read at another is the loop-carried shift.
#[test]
fn a_constant_shift_of_the_partition_is_one_band() {
    let src = "module m
@parallel
fn f(x: [i32; 32], mut out: [f32; 40]) {
    let mut guard: i32 = 0;
    guard = guard + 1;
    for h in 0..4 {
        for j in 0..8 { out[(h + 1) * 8 + j] = BODY; }
    }
}
";
    let (prog, it, trace) = lower(&src.replace("BODY", "x[h * 8 + j] as f32"));
    assert_eq!(trace, vec![Ok(())]);
    assert_eq!(pfor(&prog, &it), 1);
    // Read at band 0, written at band 1: iteration h + 1 reads what iteration h wrote, so the `h`
    // loop stays serial. (Its inner `j` loop is then a candidate of its own, and a legal one: for
    // one `h`, lane `j` reads row h and writes row h + 1 at lane `j` only.)
    let (_, _, trace) = lower(&src.replace("BODY", "out[h * 8 + j] + 1.0"));
    assert_eq!(
        trace,
        vec![
            Err("a written buffer's accesses are not per-iteration disjoint"),
            Ok(())
        ]
    );
}

/// A kernel-builtin dimension held by a binding (`matmul::<m, 2, 2>`) is read by the lowering
/// through that binding, so the region captures it: the outlined body lowers clean (`lower`
/// rejects any lowering error) and reads `m` from its env.
#[test]
fn a_named_kernel_dimension_is_captured() {
    let src = "module m
@parallel
fn heads(a: []f32, b: []f32, mut c: []f32, m: i64) {
    for h in 0..3 {
        matmul::<m, 2, 2>(a[h * 4..h * 4 + 4], b[h * 4..h * 4 + 4], c[h * 4..h * 4 + 4]);
    }
}
";
    let (prog, it, trace) = lower(src);
    assert_eq!(trace, vec![Ok(())]);
    assert_eq!(pfor(&prog, &it), 1);
    assert!(calls(region_body(&prog, &it), &it, "wukong_sgemm") >= 1);
}

/// A run-time trip count: the index of iteration `r` must lie in `[r*C, r*C + C)` with no help from
/// a modulus, so an `outer`-looking term that a constant trip count would have erased now
/// declines.
#[test]
fn a_run_time_trip_count_outlines_without_outer_terms() {
    let rows = "module m
@parallel
fn rows(x: [f32; 64], mut out: [f32; 64], n: i64) {
    let mut guard: i32 = 0;
    guard = guard + 1;
    for r in 0..n {
        let mut s: f32 = 0.0;
        for j in 0..8 { s = s + x[r * 8 + j]; }
        for j in 0..8 { out[r * 8 + j] = x[r * 8 + j] - s; }
    }
}
";
    let (prog, it, trace) = lower(rows);
    assert_eq!(trace, vec![Ok(())]);
    assert_eq!(pfor(&prog, &it), 1);
    // `b*32` vanishes mod C·N = 8·4 for the constant trip 4, but no modulus exists for `n`.
    let outer = "module m
@parallel
fn rows(x: [f32; 64], mut out: [f32; 64], n: i64, b: i64) {
    let mut guard: i32 = 0;
    guard = guard + 1;
    for r in 0..TRIP {
        for j in 0..8 { out[b * 32 + r * 8 + j] = x[b * 32 + r * 8 + j]; }
    }
}
";
    let (prog, it, trace) = lower(&outer.replace("TRIP", "4"));
    assert_eq!(trace, vec![Ok(())]);
    assert_eq!(pfor(&prog, &it), 1);
    let (prog, it, trace) = lower(&outer.replace("TRIP", "n"));
    assert_eq!(
        trace,
        vec![Err(
            "a written buffer's accesses are not per-iteration disjoint"
        )]
    );
    assert_eq!(pfor(&prog, &it), 0);
}

/// A run-time trip count must be a pure bound of a signed counter: the call site evaluates it once
/// where the serial loop re-evaluates it per iteration.
#[test]
fn a_run_time_trip_count_must_be_a_pure_signed_bound() {
    let src = "module m
fn next() -> i64 { print(1); return 4; }
@parallel
fn rows(mut out: [f32; 64], n: i64, u: u64) {
    let mut guard: i32 = 0;
    guard = guard + 1;
    for r in 0..END {
        for j in 0..8 { out[r * 8 + j] = 1.0; }
    }
}
";
    let (_, _, trace) = lower(&src.replace("END", "next()"));
    assert_eq!(
        trace,
        vec![Err("a run-time trip count that is not a pure bound")]
    );
    let (_, _, trace) = lower(&src.replace("END", "n / 2"));
    assert_eq!(trace, vec![Ok(())]);
}

/// The two-digit form under a run-time trip count keeps `t / C` as the high digit (its range is
/// unbounded) and declines the reverse order.
#[test]
fn a_run_time_divmod_needs_the_quotient_as_the_high_digit() {
    let src = "module m
@parallel
fn tiles(mut out: [f32; 64], n: i64) {
    let mut guard: i32 = 0;
    guard = guard + 1;
    for t in 0..n {
        let q = t / 2;
        let r = t % 2;
        for j in 0..4 { out[INDEX] = 1.0; }
    }
}
";
    // q*8 + r*4 + j: r's slot (4) holds j < 4; q's (8) holds r*4 + j <= 7.
    let (_, _, trace) = lower(&src.replace("INDEX", "q * 8 + r * 4 + j"));
    assert_eq!(trace, vec![Ok(())]);
    // r as the high digit needs q bounded, which only a constant trip count gives.
    let (_, _, trace) = lower(&src.replace("INDEX", "r * 32 + q * 4 + j"));
    assert_eq!(
        trace,
        vec![Err(
            "a written buffer's accesses are not per-iteration disjoint"
        )]
    );
}

/// A row kernel on the iteration's own row: the output range view `out[r*8..r*8 + 8]` is the
/// partition, the input view is read at the same rows, and the outlined body calls the SERIAL
/// kernel.
#[test]
fn a_row_kernel_on_its_own_partition_is_outlined() {
    let src = "module m
@parallel
fn rows(x: []f32, mut out: []f32) {
    let mut guard: i32 = 0;
    guard = guard + 1;
    for r in 0..4 {
        softmax(x[r * 8..r * 8 + 8], out[r * 8..r * 8 + 8]);
    }
}
";
    let (prog, it, trace) = lower(src);
    assert_eq!(trace, vec![Ok(())]);
    assert_eq!(pfor(&prog, &it), 1);
    let body = region_body(&prog, &it);
    assert!(
        calls(body, &it, "wukong_norm_f32") >= 1,
        "the serial row kernel"
    );
    assert_eq!(calls(body, &it, "wukong_norm_f32_parallel"), 0);
}

/// `sdpa` carries its extents as values: its four buffer operands are range views of the head's
/// own rows, and the outlined body lowers it clean (the static form, or the overlap-guarded one).
#[test]
fn a_per_head_sdpa_is_outlined() {
    let src = "module m
@parallel
fn attn(q: []f32, k: []f32, v: []f32, mut out: []f32) {
    for h in 0..4 {
        sdpa(q[h * 4..h * 4 + 4], k[h * 4..h * 4 + 4], v[h * 4..h * 4 + 4], out[h * 4..h * 4 + 4], 2, 2, 1.0, 0);
    }
}
";
    let (prog, it, trace) = lower(src);
    assert_eq!(trace, vec![Ok(())]);
    assert_eq!(pfor(&prog, &it), 1);
    assert!(calls(region_body(&prog, &it), &it, "wukong_attention_f32") >= 1);
    // An output view shifted by one lane straddles two rows: offsets 1..=4 past `h*4` leave the
    // band `[0, 4)`, so iteration h writes the first lane iteration h + 1 owns.
    let shifted = src.replace("out[h * 4..h * 4 + 4]", "out[h * 4 + 1..h * 4 + 5]");
    let (prog, it, trace) = lower(&shifted);
    assert_eq!(
        trace,
        vec![Err(
            "a written buffer's accesses are not per-iteration disjoint"
        )]
    );
    assert_eq!(pfor(&prog, &it), 0);
}

/// A `const` trip count is a constant like a literal: the modulus exists, so an outer term is
/// erased exactly as with `0..4`.
#[test]
fn a_const_trip_count_is_a_constant() {
    let src = "module m
const HEADS: i32 = 4;
@parallel
fn rows(x: [f32; 64], mut out: [f32; 64], b: i32) {
    let mut guard: i32 = 0;
    guard = guard + 1;
    for r in 0..HEADS {
        for j in 0..8 { out[b * 32 + r * 8 + j] = x[b * 32 + r * 8 + j]; }
    }
}
";
    let (prog, it, trace) = lower(src);
    assert_eq!(trace, vec![Ok(())]);
    assert_eq!(pfor(&prog, &it), 1);
}

/// Every reject side of the kernel-builtin rule, each for the proof's reason.
#[test]
fn a_kernel_writing_outside_its_partition_stays_serial() {
    let base = "module m
@parallel
fn rows(x: []f32, mut out: []f32, mut whole: [f32; 8]) {
    let mut guard: i32 = 0;
    guard = guard + 1;
    for r in 0..4 {
        CALL;
    }
}
";
    for (call, why) in [
        // A row of 9 reaches the next iteration's first element.
        (
            "softmax(x[r * 8..r * 8 + 9], out[r * 8..r * 8 + 9])",
            "a written buffer's accesses are not per-iteration disjoint",
        ),
        // Every iteration writes the same captured buffer whole.
        (
            "softmax(x[r * 8..r * 8 + 8], whole)",
            "a written buffer's accesses are not per-iteration disjoint",
        ),
        // `out[lo..]` runs to the run-time length: an extent no proof can place.
        (
            "gelu(x[r * 8..r * 8 + 8], out[r * 8..])",
            "a written buffer's accesses are not per-iteration disjoint",
        ),
        // A run-time length.
        (
            "gelu(x[r * 8..r * 8 + 8], out[r * 8..r * 8 + r])",
            "a written buffer's accesses are not per-iteration disjoint",
        ),
    ] {
        let (prog, it, trace) = lower(&base.replace("CALL", call));
        assert_eq!(trace, vec![Err(why)], "{call}");
        assert_eq!(pfor(&prog, &it), 0, "{call}");
    }
}

/// A kernel whose output is a body-local array writes the iteration's own storage, and a reduction
/// builtin only reads; a captured buffer read whole is fine while nothing in the region writes it.
#[test]
fn a_kernel_into_private_scratch_and_a_reduction_are_outlined() {
    let src = "module m
@parallel
fn rows(x: []f32, gamma: [f32; 8], mut out: []f32) {
    let mut guard: i32 = 0;
    guard = guard + 1;
    for r in 0..4 {
        let mut t: [f32; 8] = [0.0; 8];
        rmsnorm(x[r * 8..r * 8 + 8], gamma, t, 0.00001);
        let s: f32 = sum(t);
        for j in 0..8 { out[r * 8 + j] = t[j] * s; }
    }
}
";
    let (prog, it, trace) = lower(src);
    assert_eq!(trace, vec![Ok(())]);
    assert_eq!(pfor(&prog, &it), 1);
    // `gamma` (an f32 array parameter) and `x` (an f32 slice) may each be `out`'s storage: both
    // pairs are guarded.
    assert_eq!(guard_ends(func(&prog, &it, "rows")), 6);
}

/// Atomic adds into a slice: the adds commute, so no digit is needed; nothing else may observe the
/// counters, and a may-alias slice read is settled at run time.
#[test]
fn atomic_adds_into_a_slice_are_outlined_behind_the_guard() {
    let src = "module m
@parallel
fn hist(x: []i32, mut h: []i64) {
    let mut guard: i32 = 0;
    guard = guard + 1;
    for i in 0..64 { atomic_add(h[x[i] & 3], 1); }
}
";
    let (prog, it, trace) = lower(src);
    assert_eq!(trace, vec![Ok(())]);
    assert_eq!(pfor(&prog, &it), 1);
    assert_eq!(
        guard_ends(func(&prog, &it, "hist")),
        0,
        "i32 and i64 never share"
    );
    let same = src.replace("x: []i32", "x: []i64");
    let (prog, it, trace) = lower(&same);
    assert_eq!(trace, vec![Ok(())]);
    assert_eq!(guard_ends(func(&prog, &it, "hist")), 4);
    let read = src.replace(
        "atomic_add(h[x[i] & 3], 1);",
        "atomic_add(h[x[i] & 3], 1); let c: i64 = h[0];",
    );
    let (_, _, trace) = lower(&read);
    assert_eq!(
        trace,
        vec![Err(
            "an atomically updated buffer is also read or written plainly"
        )]
    );
}

/// The region verdicts of one corpus program, or `None` when it does not lower clean on its own
/// (the driver's front end does more than this harness).
fn corpus_trace(src: &str) -> Option<Vec<Result<(), &'static str>>> {
    let mut interner = Interner::new();
    let (module, pdiags) = wukong_parser::parse_module(src, SourceId(0), &mut interner);
    if !pdiags.is_empty() {
        return None;
    }
    let (sema, sdiags) = wukong_sema::check(&module, &interner);
    if sdiags.iter().any(|d| d.is_error()) {
        return None;
    }
    par_region::take_trace();
    let (_, diags) = lower_program_with(&module, &sema, &mut interner, LowerOptions::default());
    let trace = par_region::take_trace();
    (!diags.iter().any(|d| d.is_error())).then_some(trace)
}

/// The decline census over the corpus: every `@parallel` program in `tests/run` and `examples`
/// (root files only — the loader is the driver's), every region verdict, grouped by reason.
/// `cargo test -p wukong_mir_build --lib corpus_region_declines -- --nocapture` prints the table.
///
/// What it pins: a floor on the regions outlined, and that no corpus region stays serial for one
/// of the slice / kernel-builtin corners this outliner leaves unmodeled — so a regression that
/// quietly narrows the scan shows up here by name, not as a missing speedup. (Whole-function
/// `@parallel` loops outlined by `parallel_spec` are not region verdicts and are not counted.)
#[test]
fn corpus_region_declines() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let mut by_reason: BTreeMap<&'static str, Vec<String>> = BTreeMap::new();
    let mut outlined = 0usize;
    let mut programs = 0usize;
    let mut skipped: Vec<String> = Vec::new();
    for dir in ["tests/run", "examples"] {
        let mut files: Vec<_> = std::fs::read_dir(root.join(dir))
            .expect("corpus directory")
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().is_some_and(|x| x == "wk"))
            .collect();
        files.sort();
        for path in files {
            let src = std::fs::read_to_string(&path).expect("read corpus file");
            if !src.contains("@parallel")
                || src.lines().any(|l| l.trim_start().starts_with("import "))
            {
                continue;
            }
            let name = format!("{dir}/{}", path.file_name().unwrap().to_string_lossy());
            let Some(trace) = corpus_trace(&src) else {
                skipped.push(name);
                continue;
            };
            programs += 1;
            for verdict in trace {
                match verdict {
                    Ok(()) => outlined += 1,
                    Err(why) => by_reason.entry(why).or_default().push(name.clone()),
                }
            }
        }
    }
    eprintln!(
        "{programs} @parallel programs, {outlined} regions outlined ({} skipped: {skipped:?});          declines by reason:",
        skipped.len()
    );
    for (why, files) in &by_reason {
        eprintln!("  {:>3}  {why}", files.len());
        for f in files {
            eprintln!("         {f}");
        }
    }
    assert!(programs >= 50, "the corpus scan found {programs} programs");
    assert!(outlined >= 16, "only {outlined} regions outlined");
    // The corners of the slice and kernel-builtin rules this outliner leaves unmodeled. None is
    // reached by the corpus today; a region that starts declining for one of them is a narrowing
    // to look at, not noise.
    for unmodeled in [
        "an indexed slice of non-scalar elements",
        "a captured buffer whose storage cannot be identified",
        "a range view of a computed base",
        "a range view of a name that is not a captured slice",
        "a kernel operand that is not a name or a range view",
        "a kernel operand body-local that views shared storage",
        "a kernel operand that is not a runtime binding",
        "the length of a slice whose storage cannot be identified",
        "a run-time trip count that is not a pure bound",
    ] {
        assert!(
            !by_reason.contains_key(unmodeled),
            "a corpus region declines for `{unmodeled}`: {:?}",
            by_reason[unmodeled]
        );
    }
}
