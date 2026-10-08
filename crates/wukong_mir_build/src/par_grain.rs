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
//! exactly `[0, n)` in order, and the result is bit-identical at every grain. `G = 1` (the default,
//! and every `@parallel` without an argument) emits exactly the MIR it always did.

use wukong_ast as ast;
use wukong_mir::{BinOp, Builder, CmpOp, MirType, Op, ValueId};
use wukong_span::Interner;

/// The grain on `item`'s `@parallel` attribute, or 1 when there is none. An invalid argument is
/// `E0306` in sema and reads as 1 here.
pub(crate) fn item_grain(item: &ast::Item, interner: &Interner) -> u64 {
    item.attrs
        .iter()
        .find(|a| interner.resolve(a.name.sym) == "parallel")
        .and_then(|a| wukong_sema::parallel_grain(a, interner).ok())
        .unwrap_or(1)
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
        let mut interner = Interner::new();
        let (module, pdiags) = wukong_parser::parse_module(src, SourceId(0), &mut interner);
        assert!(pdiags.is_empty(), "parse: {pdiags:?}");
        let (sema, sdiags) = wukong_sema::check(&module, &interner);
        assert!(sdiags.iter().all(|d| !d.is_error()), "sema: {sdiags:?}");
        let (prog, diags) = crate::lower_program(&module, &sema, &mut interner);
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
