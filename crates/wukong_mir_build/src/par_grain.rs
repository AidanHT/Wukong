//! `@parallel(grain = G)` — the minimum number of consecutive iterations each worker claims at a
//! time from a function's parallel loop, applied entirely in the compiler.
//!
//! The runtime entry `wukong_parallel_for(n, body, env)` splits the index range `[0, n)` into spans
//! and calls `body(start, end, env)` once per span; how big a span is depends on the pool width and
//! the claim queue, so a span can be as small as one index. A grain `G` is applied without touching
//! that ABI (so the interpreter, the native backend and the GPU lowering all see the same call
//! shape): the call site passes the **chunk count** `ceil(n / G)` instead of `n`, and the outlined
//! body maps the chunk span it receives back to iterations,
//!
//! ```text
//! [start, end)  ->  [start·G, end >= chunks ? n : end·G)
//! ```
//!
//! so every worker gets whole runs of `G` consecutive iterations (the last chunk may be shorter),
//! and a loop of at most `G` iterations becomes a one-chunk call, which the runtime runs without
//! forking. `start·G` is always below `n`, and `end·G` is *selected* only below the last chunk,
//! where it is below `n` too; the last chunk takes `n` itself. (The unselected `end·G` of the last
//! chunk is at most `n + G - 1`, so even it cannot wrap for any trip count below
//! `i64::MAX - 2^31`.)
//!
//! Iterations of an outlined loop are independent (the outliner's legality proof), so regrouping
//! them changes scheduling only: the interpreter, which runs `body(0, chunks)` in one call, executes
//! exactly `[0, n)` in order, and the result is bit-identical at every grain. `G = 1` emits exactly
//! the MIR it always did.
//!
//! **The implicit grain (P12.21).** A `@parallel` without a `grain` argument used to mean `G = 1`,
//! so an 8-iteration region forked eight claimable indices -- a fork-join that costs more than the
//! loop. With [`crate::LowerOptions::implicit_grain`] set (the driver sets it for every CPU-executing
//! backend and clears it for `--backend=gpu-native`, whose one-thread-per-index mapping wants
//! `G = 1`), such a loop gets `G = max(1, ceil(MIN_CHUNK_OPS / body_ops))` from the static size of
//! its outlined body ([`implicit_grain`]): each claimed chunk then carries at least
//! [`MIN_CHUNK_OPS`] instructions of work, and a loop whose whole trip count is below one chunk's
//! worth becomes a one-chunk call, which `wukong_parallel_for` runs serially. A body whose static
//! size is not its per-iteration cost -- one with an inner loop or any call -- keeps `G = 1`. An
//! explicit `@parallel(grain = G)`, `G = 1` included, always wins.

use wukong_ast as ast;
use wukong_diag::Diagnostic;
use wukong_mir::{BinOp, BlockId, Builder, CmpOp, Function, MirType, Op, Terminator, ValueId};
use wukong_span::Interner;

/// The minimum static MIR instruction count one claimed chunk of an implicit-grain `@parallel` loop
/// carries: `2^14`. At one to a few nanoseconds per pre-optimization MIR instruction that is a few
/// microseconds of work -- the order of one fork-join wake on the runtime's unified pool -- so a
/// loop whose WHOLE trip count is below one chunk runs as a single serial call instead of forking,
/// and a long loop is claimed in chunks big enough that the per-claim atomic is noise. A starting
/// point anchored to that cost, not a measurement (like `wukong_runtime::par`'s fork threshold),
/// and a compile-time constant on purpose: it shapes the MIR, and the compiler is a pure function
/// of its input, so no environment read may move it.
pub(crate) const MIN_CHUNK_OPS: u64 = 1 << 14;

/// The user's `@parallel(grain = G)` on `item`: `Some(G)` when the attribute names a grain, `None`
/// when it carries no argument (or an invalid one, which sema reports as `E0306`).
pub(crate) fn user_grain(item: &ast::Item, interner: &Interner) -> Option<u64> {
    let attr = item
        .attrs
        .iter()
        .find(|a| interner.resolve(a.name.sym) == "parallel")?;
    if attr.args().is_empty() {
        return None;
    }
    wukong_sema::parallel_grain(attr, interner).ok()
}

/// The implicit grain of a loop whose outlined body, built at grain 1, is `outlined`:
/// `max(1, ceil(MIN_CHUNK_OPS / body_ops))` with `body_ops` the body's static instruction count.
/// That count is the per-iteration cost only for a straight-line body (one back edge: the
/// outlined loop's own); an inner loop or any call -- a runtime kernel, a user function, a raw-AVX2
/// recipe -- does work no static count sees, so such a body keeps grain 1 rather than risk
/// collapsing a heavy loop (a head-parallel attention region is a dozen iterations of GEMM calls)
/// into one serial chunk.
pub(crate) fn implicit_grain(outlined: &Function) -> u64 {
    match straight_line_ops(outlined) {
        Some(ops) => MIN_CHUNK_OPS.div_ceil(ops.max(1)).max(1),
        None => 1,
    }
}

/// The static instruction count of `f`, or `None` when it is not a straight-line loop body.
fn straight_line_ops(f: &Function) -> Option<u64> {
    let mut ops = 0u64;
    for b in &f.blocks {
        for i in &b.insts {
            if matches!(
                i.op,
                Op::Call { .. } | Op::VecKernelCall { .. } | Op::Barrier
            ) {
                return None;
            }
            ops += 1;
        }
    }
    (back_edges(f) <= 1).then_some(ops)
}

fn successors(t: &Terminator) -> Vec<BlockId> {
    match t {
        Terminator::Br { target, .. } => vec![*target],
        Terminator::CondBr {
            then_blk, else_blk, ..
        } => vec![*then_blk, *else_blk],
        Terminator::Switch { cases, default, .. } => cases
            .iter()
            .map(|c| c.target)
            .chain(std::iter::once(*default))
            .collect(),
        Terminator::Ret(_) | Terminator::Unreachable => Vec::new(),
    }
}

/// Retreating edges of a depth-first walk from the entry: one per natural loop in reducible MIR.
fn back_edges(f: &Function) -> usize {
    let index: std::collections::HashMap<BlockId, usize> = f
        .blocks
        .iter()
        .enumerate()
        .map(|(i, b)| (b.id, i))
        .collect();
    let Some(&entry) = index.get(&f.entry) else {
        return 0;
    };
    // 0 = unvisited, 1 = on the DFS stack, 2 = finished.
    let mut state = vec![0u8; f.blocks.len()];
    let mut stack: Vec<(usize, Vec<BlockId>)> = vec![(entry, successors(&f.blocks[entry].term))];
    state[entry] = 1;
    let mut back = 0;
    while let Some((node, succs)) = stack.last_mut() {
        let node = *node;
        match succs.pop() {
            Some(s) => {
                let Some(&si) = index.get(&s) else { continue };
                match state[si] {
                    0 => {
                        state[si] = 1;
                        stack.push((si, successors(&f.blocks[si].term)));
                    }
                    1 => back += 1,
                    _ => {}
                }
            }
            None => {
                state[node] = 2;
                stack.pop();
            }
        }
    }
    back
}

/// Build an outlined `@parallel` body at its grain and return it with that grain. `build(g, diags)`
/// lowers the body at grain `g`. The user's grain wins; without one, the implicit grain (when
/// `implicit`) is read off a grain-1 build, which is kept as-is when that grain is 1 and rebuilt
/// otherwise -- lowering is a pure function of its inputs, and only the kept build's diagnostics
/// are reported, so the result is exactly what a one-shot build at the chosen grain gives.
pub(crate) fn outline_with_grain(
    user: Option<u64>,
    implicit: bool,
    diags: &mut Vec<Diagnostic>,
    mut build: impl FnMut(u64, &mut Vec<Diagnostic>) -> Function,
) -> (Function, u64) {
    if let Some(g) = user {
        return (build(g, diags), g);
    }
    if !implicit {
        return (build(1, diags), 1);
    }
    let mut probe_diags = Vec::new();
    let probe = build(1, &mut probe_diags);
    let g = implicit_grain(&probe);
    if g <= 1 {
        diags.extend(probe_diags);
        return (probe, 1);
    }
    (build(g, diags), g)
}

/// The chunk count for a compile-time trip count: `ceil(n / g)` for `n > 0`, else `n` unchanged
/// (a non-positive trip count runs nothing either way).
pub(crate) fn const_chunks(n: i64, g: u64) -> i64 {
    if n <= 0 || g <= 1 {
        return n;
    }
    // `g <= MAX_PARALLEL_GRAIN` (2^31) and `n <= i64::MAX`: exact in i128.
    ((n as i128 + g as i128 - 1) / g as i128) as i64
}

fn i64_const(b: &mut Builder, v: i128) -> ValueId {
    b.build(MirType::I64, Op::ConstInt(v, MirType::I64))
}

/// Emit the chunk count `n / g + (n % g > 0)` for a runtime trip count `n` (i64). Never overflows;
/// for `n <= 0` the result is `<= 0`, so the runtime runs nothing, exactly as it would for `n`.
pub(crate) fn emit_chunks(b: &mut Builder, n: ValueId, g: u64) -> ValueId {
    let gv = i64_const(b, g as i128);
    let q = b.build(MirType::I64, Op::Bin(BinOp::SDiv, n, gv));
    let r = b.build(MirType::I64, Op::Bin(BinOp::SRem, n, gv));
    let zero = i64_const(b, 0);
    let one = i64_const(b, 1);
    let has_rem = b.build(MirType::I1, Op::Cmp(CmpOp::Sgt, r, zero));
    let bump = b.build(MirType::I64, Op::Select(has_rem, one, zero));
    b.build(MirType::I64, Op::Bin(BinOp::Add, q, bump))
}

/// Inside an outlined body: map the chunk span `[start, end)` the runtime hands over to the
/// iteration span `[start·g, end >= chunks ? n : end·g)`. `n` is the loop's trip count (i64) and
/// `chunks` the chunk count the call site passed.
pub(crate) fn emit_iteration_bounds(
    b: &mut Builder,
    start: ValueId,
    end: ValueId,
    n: ValueId,
    chunks: ValueId,
    g: u64,
) -> (ValueId, ValueId) {
    let gv = i64_const(b, g as i128);
    let lo = b.build(MirType::I64, Op::Bin(BinOp::Mul, start, gv));
    let scaled = b.build(MirType::I64, Op::Bin(BinOp::Mul, end, gv));
    let last = b.build(MirType::I1, Op::Cmp(CmpOp::Sge, end, chunks));
    let hi = b.build(MirType::I64, Op::Select(last, n, scaled));
    (lo, hi)
}

#[cfg(test)]
mod tests {
    use super::*;
    use wukong_mir::Function;
    use wukong_span::SourceId;

    fn lower(src: &str) -> (wukong_mir::Program, Interner) {
        lower_opts(src, crate::LowerOptions::default())
    }

    /// [`lower`] with the implicit grain on, as the driver lowers for a CPU backend.
    fn lower_implicit(src: &str) -> (wukong_mir::Program, Interner) {
        lower_opts(
            src,
            crate::LowerOptions {
                implicit_grain: true,
                ..crate::LowerOptions::default()
            },
        )
    }

    fn lower_opts(src: &str, opts: crate::LowerOptions) -> (wukong_mir::Program, Interner) {
        let mut interner = Interner::new();
        let (module, pdiags) = wukong_parser::parse_module(src, SourceId(0), &mut interner);
        assert!(pdiags.is_empty(), "parse: {pdiags:?}");
        let (sema, sdiags) = wukong_sema::check(&module, &interner);
        assert!(sdiags.iter().all(|d| !d.is_error()), "sema: {sdiags:?}");
        let (prog, diags) = crate::lower_program_with(&module, &sema, &mut interner, opts);
        assert!(diags.iter().all(|d| !d.is_error()), "lower: {diags:?}");
        for f in &prog.funcs {
            let errs = wukong_mir::verify::verify_function(f);
            assert!(errs.is_empty(), "{errs:?}");
        }
        (prog, interner)
    }

    /// The first argument of every `wukong_parallel_for` call in `f`, as a constant when it is one.
    fn pfor_trip_consts(f: &Function, interner: &Interner) -> Vec<Option<i128>> {
        let consts: std::collections::HashMap<ValueId, i128> = f
            .blocks
            .iter()
            .flat_map(|b| b.insts.iter())
            .filter_map(|i| match (&i.op, i.result) {
                (Op::ConstInt(v, _), Some(r)) => Some((r, *v)),
                _ => None,
            })
            .collect();
        f.blocks
            .iter()
            .flat_map(|b| b.insts.iter())
            .filter_map(|i| match &i.op {
                Op::Call { func, args } if interner.resolve(*func) == "wukong_parallel_for" => {
                    Some(args.first().and_then(|a| consts.get(a).copied()))
                }
                _ => None,
            })
            .collect()
    }

    fn func<'p>(prog: &'p wukong_mir::Program, interner: &Interner, name: &str) -> &'p Function {
        prog.funcs
            .iter()
            .find(|f| interner.resolve(f.name) == name)
            .unwrap_or_else(|| panic!("no function `{name}`"))
    }

    /// The single outlined region body of a one-region program.
    fn region_body<'p>(prog: &'p wukong_mir::Program, interner: &Interner) -> &'p Function {
        let mut bodies = prog
            .funcs
            .iter()
            .filter(|f| interner.resolve(f.name).starts_with("wukong$par$"));
        let body = bodies.next().expect("one outlined region");
        assert!(bodies.next().is_none(), "exactly one outlined region");
        body
    }

    /// An integer region whose trip count is `n` (integer data, so no f32 recipe claims the loop
    /// before the region outliner sees it -- the `lang_parallel_int_regions.wk` shape).
    fn int_region(attr: &str, n: usize) -> String {
        format!(
            "{attr}\nfn main() -> i32 {{ let mut a: [i64; {n}] = [0; {n}]; \
             for i in 0..{n} {{ let v = (i as i64) + 1; a[i] = v * v - (v % 3); }} \
             print(a[{last}]); return 0; }}",
            last = n - 1
        )
    }

    /// P12.21: without a user grain, a straight-line region's grain is
    /// `ceil(MIN_CHUNK_OPS / body_ops)` of its grain-1 body. A 10-iteration region is then ONE
    /// chunk (the runtime runs it serially, no fork); a long region is claimed in chunks of that
    /// grain; and the flag off -- what the driver passes for `--backend=gpu-native` -- keeps one
    /// index per iteration.
    #[test]
    fn implicit_grain_sizes_chunks_from_the_body() {
        // The grain the rule assigns, read off the grain-1 build.
        let (g1, interner) = lower(&int_region("@parallel", 10));
        let g = implicit_grain(region_body(&g1, &interner));
        assert!(
            g > 10,
            "a tiny straight-line body must get a grain past 10 (got {g})"
        );
        assert_eq!(
            pfor_trip_consts(func(&g1, &interner, "main"), &interner),
            vec![Some(10)]
        );
        let (prog, interner) = lower_implicit(&int_region("@parallel", 10));
        assert_eq!(
            pfor_trip_consts(func(&prog, &interner, "main"), &interner),
            vec![Some(1)],
            "a region below one chunk of work is a one-chunk call"
        );
        let n = 100_000usize;
        let (g1, interner) = lower(&int_region("@parallel", n));
        let g = implicit_grain(region_body(&g1, &interner));
        let (prog, interner) = lower_implicit(&int_region("@parallel", n));
        let want = const_chunks(n as i64, g) as i128;
        assert!(want > 1 && want < n as i128, "chunks {want}");
        assert_eq!(
            pfor_trip_consts(func(&prog, &interner, "main"), &interner),
            vec![Some(want)]
        );
    }

    /// An explicit grain always wins over the implicit one -- `grain = 1` included, which is how a
    /// user opts a loop back into one index per iteration.
    #[test]
    fn a_user_grain_overrides_the_implicit_grain() {
        for (attr, want) in [("@parallel(grain = 1)", 10), ("@parallel(grain = 4)", 3)] {
            let (prog, interner) = lower_implicit(&int_region(attr, 10));
            assert_eq!(
                pfor_trip_consts(func(&prog, &interner, "main"), &interner),
                vec![Some(want)],
                "{attr}"
            );
        }
    }

    /// A body whose static size is not its per-iteration cost -- an inner loop here -- keeps grain
    /// 1: the rule never guesses a heavy body light.
    #[test]
    fn a_body_with_an_inner_loop_keeps_grain_one() {
        let src = "@parallel\nfn main() -> i32 { let mut a: [i64; 10] = [0; 10]; \
                   for i in 0..10 { let mut s: i64 = 0; for j in 0..4 { s = s + (j as i64) * (i as i64); } \
                   a[i] = s; } print(a[9]); return 0; }";
        let (prog, interner) = lower_implicit(src);
        assert_eq!(implicit_grain(region_body(&prog, &interner)), 1);
        assert_eq!(
            pfor_trip_consts(func(&prog, &interner, "main"), &interner),
            vec![Some(10)]
        );
    }

    /// The whole-function form (`lower_parallel`): the wrapper divides the runtime trip count by
    /// the implicit grain -- an `SDiv` by that constant -- and only when the flag is on.
    #[test]
    fn whole_function_parallel_gets_the_implicit_grain() {
        let src = "@parallel\nfn bump(a: [i64; 64], mut out: [i64; 64]) { \
                   for i in 0..64 { out[i] = a[i] * 3 + (i as i64); } }\n\
                   fn main() -> i32 { let a: [i64; 64] = [2; 64]; let mut b: [i64; 64] = [0; 64]; \
                   bump(a, b); print(b[63]); return 0; }";
        let sdivs = |f: &Function| -> Vec<i128> {
            let consts: std::collections::HashMap<ValueId, i128> = f
                .blocks
                .iter()
                .flat_map(|b| b.insts.iter())
                .filter_map(|i| match (&i.op, i.result) {
                    (Op::ConstInt(v, _), Some(r)) => Some((r, *v)),
                    _ => None,
                })
                .collect();
            f.blocks
                .iter()
                .flat_map(|b| b.insts.iter())
                .filter_map(|i| match &i.op {
                    Op::Bin(BinOp::SDiv, _, d) => consts.get(d).copied(),
                    _ => None,
                })
                .collect()
        };
        let (off, interner) = lower(src);
        let body = func(&off, &interner, "bump$par");
        let g = implicit_grain(body);
        assert!(g > 1, "a straight-line elementwise body gets a grain");
        assert!(
            sdivs(func(&off, &interner, "bump")).is_empty(),
            "flag off: no chunking"
        );
        let (on, interner) = lower_implicit(src);
        assert_eq!(sdivs(func(&on, &interner, "bump")), vec![g as i128]);
    }

    #[test]
    fn chunk_count_rounds_up_and_preserves_empty_ranges() {
        assert_eq!(const_chunks(10, 3), 4);
        assert_eq!(const_chunks(9, 3), 3);
        assert_eq!(const_chunks(2, 64), 1);
        assert_eq!(const_chunks(10, 1), 10);
        assert_eq!(const_chunks(0, 4), 0);
        assert_eq!(const_chunks(-5, 4), -5);
        assert_eq!(const_chunks(i64::MAX, 1 << 31), i64::MAX / (1 << 31) + 1);
    }

    /// P12.29: more mid-function regions than the initial symbol pool (32) all outline. The 33rd
    /// onward used to stay serial with no diagnostic; the lowering now redoes itself with a larger
    /// pool, so all 40 loops below reach `wukong_parallel_for`.
    #[test]
    fn more_regions_than_the_initial_pool_all_outline() {
        let mut src =
            String::from("@parallel\nfn main() -> i32 {\n    let mut a: [i64; 8] = [0; 8];\n");
        for k in 0..40 {
            src.push_str(&format!(
                "    for i in 0..8 {{ let v = (i as i64) + 1; a[i] = v * v - (v % 3) + {k}; }}\n"
            ));
        }
        src.push_str("    print(a[7]);\n    return 0;\n}\n");
        let (prog, interner) = lower(&src);
        let main = prog
            .funcs
            .iter()
            .find(|f| interner.resolve(f.name) == "main")
            .unwrap();
        assert_eq!(pfor_trip_consts(main, &interner), vec![Some(8); 40]);
        let bodies = prog
            .funcs
            .iter()
            .filter(|f| interner.resolve(f.name).starts_with("wukong$par$"))
            .count();
        assert_eq!(bodies, 40, "one outlined body per region");
    }

    /// A mid-function `@parallel` region honors the function's grain: 10 iterations at grain 4 is
    /// a 3-chunk call, and the default grain keeps the call at the trip count. Integer data, so no
    /// f32 vector recipe claims the loop before the region outliner sees it (the
    /// `lang_parallel_int_regions.wk` shape).
    #[test]
    fn region_call_passes_the_chunk_count() {
        let body = "{ let mut a: [i64; 10] = [0; 10]; \
                    for i in 0..10 { let v = (i as i64) + 1; a[i] = v * v - (v % 3); } \
                    print(a[9]); return 0; }";
        let (prog, interner) = lower(&format!("@parallel(grain = 4)\nfn main() -> i32 {body}"));
        let main = prog
            .funcs
            .iter()
            .find(|f| interner.resolve(f.name) == "main")
            .unwrap();
        assert_eq!(pfor_trip_consts(main, &interner), vec![Some(3)]);
        let (prog, interner) = lower(&format!("@parallel\nfn main() -> i32 {body}"));
        let main = prog
            .funcs
            .iter()
            .find(|f| interner.resolve(f.name) == "main")
            .unwrap();
        assert_eq!(pfor_trip_consts(main, &interner), vec![Some(10)]);
    }
}
