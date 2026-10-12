//! The GEMM bias-alias preflight's cost (N65, see `bias_memo.rs`): its recognizer probes grow
//! linearly with nest depth, and the memo that makes them do so changes no MIR anywhere in the
//! corpus.
//!
//! `cargo test -p wukong_mir_build --lib bias_memo -- --nocapture` prints the probe counts (memo
//! on and off) for the synthetic nests and the corpus's deepest nests.

use std::path::Path;

use crate::{bias_memo, lower_program_with, LowerOptions};
use wukong_ast::{Block, Expr, ExprKind, ItemKind, Module, StmtKind};
use wukong_mir::print::print_program;
use wukong_span::{Interner, SourceId};

/// Lower `src` and return its printed MIR and the guard probes it took, with the memo on or off.
/// `None` when the program does not lower clean on its own (the driver's front end does more than
/// this harness: imports, type aliases).
fn probe(src: &str, opts: LowerOptions, memo: bool) -> Option<(String, u64)> {
    let mut interner = Interner::new();
    let (module, pdiags) = wukong_parser::parse_module(src, SourceId(0), &mut interner);
    if !pdiags.is_empty() {
        return None;
    }
    let (sema, sdiags) = wukong_sema::check(&module, &interner);
    if sdiags.iter().any(|d| d.is_error()) {
        return None;
    }
    let was = bias_memo::set_disabled(!memo);
    bias_memo::take_probes();
    let (prog, diags) = lower_program_with(&module, &sema, &mut interner, opts);
    let probes = bias_memo::take_probes();
    bias_memo::set_disabled(was);
    if diags.iter().any(|d| d.is_error()) {
        return None;
    }
    Some((print_program(&prog, &interner), probes))
}

/// A `depth`-deep `for` nest. Each level's body is the next loop followed by a store, so every
/// level has a loop for the guard to walk and a two-statement window for it to probe. `@parallel`
/// with a leading `let` keeps the function out of the whole-function outliner, so every loop is
/// lowered with `parallel_fn` set; its eligibility check (`parallel_spec`) walks the whole body
/// once more, with no memo, which is why that variant takes twice the probes per level.
fn nest(depth: usize, parallel: bool) -> String {
    let mut src = String::from("module m\n");
    if parallel {
        src.push_str("@parallel\n");
    }
    src.push_str("fn f(x: [f32; 64], mut out: [f32; 64]) {\n    let base = 0;\n");
    let idx = |k: usize| {
        let mut e = String::from("base");
        for v in 0..=k {
            e.push_str(&format!(" + v{v}"));
        }
        e
    };
    for k in 0..depth {
        src.push_str(&format!("{}for v{k} in 0..2 {{\n", "    ".repeat(k + 1)));
    }
    for k in (0..depth).rev() {
        let pad = "    ".repeat(k + 2);
        let i = idx(k);
        src.push_str(&format!("{pad}out[{i}] = out[{i}] + x[{i}];\n"));
        src.push_str(&format!("{}}}\n", "    ".repeat(k + 1)));
    }
    src.push_str("}\nfn main() -> i32 {\n    let x: [f32; 64] = [1.0; 64];\n");
    src.push_str("    let mut out: [f32; 64] = [0.0; 64];\n    f(x, out);\n    return 0;\n}\n");
    src
}

/// The probe counts of the depth-1..=8 nests, memo on (`.0`) and off (`.1`).
fn nest_probes(parallel: bool, opts: impl Fn() -> LowerOptions) -> Vec<(u64, u64)> {
    (1..=8)
        .map(|d| {
            let src = nest(d, parallel);
            let (on_mir, on) = probe(&src, opts(), true).expect("the nest lowers clean");
            let (off_mir, off) = probe(&src, opts(), false).expect("the nest lowers clean");
            assert_eq!(on_mir, off_mir, "depth {d}: the memo changed the MIR");
            (on, off)
        })
        .collect()
}

/// Every level past the second adds the same number of probes as the one before it: linear growth.
/// (The step from depth 1 to 2 may differ: in the `@parallel` nest the one-deep loop never reaches
/// `lower_for`'s guard, so only the whole-function eligibility walk in `parallel_spec` probes it.)
fn assert_linear(what: &str, counts: &[u64]) {
    let steps: Vec<u64> = counts.windows(2).map(|w| w[1] - w[0]).collect();
    assert!(
        steps[1..].windows(2).all(|s| s[0] == s[1]),
        "{what}: probes {counts:?} grow by {steps:?} per level, not by a constant"
    );
}

#[test]
fn bias_alias_probes_grow_linearly_with_nest_depth() {
    for (what, parallel, recognize) in [
        ("recognition on", false, true),
        ("@parallel", true, true),
        ("@parallel, recognition off", true, false),
    ] {
        let counts = nest_probes(parallel, || LowerOptions {
            recognize,
            ..LowerOptions::default()
        });
        let on: Vec<u64> = counts.iter().map(|c| c.0).collect();
        let off: Vec<u64> = counts.iter().map(|c| c.1).collect();
        eprintln!("{what}: depth 1..=8 probes, memo on {on:?}, off {off:?}");
        assert_linear(what, &on);
        // The quadratic it replaces, pinned so the test cannot pass vacuously: without the memo a
        // loop at depth k is walked by each of its ancestors as well.
        let off_steps: Vec<u64> = off.windows(2).map(|w| w[1] - w[0]).collect();
        assert!(
            off_steps.windows(2).all(|s| s[1] > s[0]),
            "{what}: memo off, probes {off:?} should grow faster than linearly"
        );
        // A 6-deep nest -- the size the tracker names -- is walked once per loop and window.
        assert!(
            on[5] * 2 < off[5],
            "{what}: depth 6 takes {} probes memoized vs {} unmemoized",
            on[5],
            off[5]
        );
    }
    // Recognition off and no `@parallel`: the per-loop guard never runs (`lower_for` asks only
    // when something could be substituted or outlined), so only the windows are probed, once each.
    let counts = nest_probes(false, || LowerOptions {
        recognize: false,
        ..LowerOptions::default()
    });
    assert!(counts.iter().all(|c| c.0 == c.1), "{counts:?}");
    assert_linear(
        "recognition off",
        &counts.iter().map(|c| c.0).collect::<Vec<_>>(),
    );
}

/// The deepest `for` nest in `b` (loops inside any statement or expression position).
fn block_depth(b: &Block) -> usize {
    let stmts = b.stmts.iter().map(|s| match &s.kind {
        StmtKind::For { body, .. } => 1 + block_depth(body),
        StmtKind::While { cond, body, .. } => expr_depth(cond).max(block_depth(body)),
        StmtKind::Let { init, .. } => init.as_ref().map_or(0, expr_depth),
        StmtKind::Assign { target, value, .. } => expr_depth(target).max(expr_depth(value)),
        StmtKind::Expr(e) | StmtKind::Defer(e) => expr_depth(e),
        StmtKind::Return(e) | StmtKind::Break(_, e) => e.as_ref().map_or(0, expr_depth),
        StmtKind::Continue(_) => 0,
    });
    stmts
        .chain(b.tail.as_deref().map(expr_depth))
        .max()
        .unwrap_or(0)
}

fn expr_depth(e: &Expr) -> usize {
    match &e.kind {
        ExprKind::Block(b) | ExprKind::Loop { body: b, .. } => block_depth(b),
        ExprKind::If {
            then_branch,
            else_branch,
            ..
        } => block_depth(then_branch).max(else_branch.as_deref().map_or(0, expr_depth)),
        ExprKind::Match { arms, .. } => arms.iter().map(|a| expr_depth(&a.body)).max().unwrap_or(0),
        _ => 0,
    }
}

fn module_depth(m: &Module) -> usize {
    m.items
        .iter()
        .filter_map(|it| match &it.kind {
            ItemKind::Fn(f) => f.body.as_ref().map(block_depth),
            _ => None,
        })
        .max()
        .unwrap_or(0)
}

/// The corpus, lowered with the memo on and off: identical MIR for every program (the memo is a
/// cache, not a decision), never more probes with it, and the deepest nests' counts printed.
#[test]
fn corpus_mir_is_identical_with_and_without_the_bias_alias_memo() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let mut rows: Vec<(usize, String, u64, u64)> = Vec::new();
    for dir in ["tests/run", "examples"] {
        let mut files: Vec<_> = std::fs::read_dir(root.join(dir))
            .expect("corpus directory")
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().is_some_and(|x| x == "wk"))
            .collect();
        files.sort();
        for path in files {
            let src = std::fs::read_to_string(&path).expect("read corpus file");
            if src.lines().any(|l| l.trim_start().starts_with("import ")) {
                continue;
            }
            let name = format!("{dir}/{}", path.file_name().unwrap().to_string_lossy());
            let Some((on_mir, on)) = probe(&src, LowerOptions::default(), true) else {
                continue;
            };
            let (off_mir, off) =
                probe(&src, LowerOptions::default(), false).expect("lowered clean with the memo");
            assert!(on_mir == off_mir, "{name}: the memo changed the MIR");
            assert!(on <= off, "{name}: {on} probes memoized vs {off} without");
            let mut interner = Interner::new();
            let (module, _) = wukong_parser::parse_module(&src, SourceId(0), &mut interner);
            rows.push((module_depth(&module), name, on, off));
        }
    }
    assert!(
        rows.len() >= 400,
        "the corpus scan lowered {} programs",
        rows.len()
    );
    rows.sort_by(|a, b| b.0.cmp(&a.0).then(b.3.cmp(&a.3)));
    let (on, off) = rows.iter().fold((0, 0), |(a, b), r| (a + r.2, b + r.3));
    eprintln!(
        "{} programs: {on} guard probes memoized, {off} without; deepest nests:",
        rows.len()
    );
    for (depth, name, on, off) in rows.iter().take(12) {
        eprintln!("  depth {depth}  {on:>5} vs {off:>5}  {name}");
    }
}
