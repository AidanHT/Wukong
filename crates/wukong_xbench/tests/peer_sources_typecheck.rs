//! Every `.wk` peer source `wukong_xbench` ships must parse, sema-check and lower cleanly.
//!
//! CLAUDE.md lists as a cross-cutting landmine that `wukong_xbench` compiles its `.wk` peer programs
//! at RUNTIME, so a language / sema / `mir_build` / recognizer change can reject a spelling used only
//! by an xbench peer and break every generated kernel — and the measured peer-comparison numbers the
//! campaign depends on — while `cargo test --workspace` stays green, because nothing in the crate
//! front-ends those sources at test time. The only corpus test, `every_placeholder_is_substituted`,
//! asserts `!out.contains('$')` and nothing more; no test runs the real pipeline over them.
//!
//! This file closes that for the on-disk peer corpus: the five structure-tax block spellings, the
//! fused focal-loss program, and the Mamba/S6 scan. For each it performs the same `$TOKEN$`
//! substitution the benchmark does, then `parse_module` -> `sema::check` -> `lower_program` ->
//! `optimize(3)`, and asserts zero error diagnostics at every stage. No execution, no timing.
//!
//! `wukong_xbench` is a bin-only crate, so this integration test cannot reach its internal
//! `subst`/`Dims`/`kernels()` items; it mirrors the substitution instead, using the defaults from
//! `general.rs` (`Dims` s=256,d=256,h=4,f=512; `LossDims` r=8192,c=1024,eps=0.125; `ScanDims`
//! t=2048,d=256,n=16) computed with the identical arithmetic. The mirror is drift-guarded two ways:
//! the token tables below are the exact set present in the sources (any token the benchmark adds
//! leaves a `$` behind, which [`typecheck`] rejects), and the substitution exercises the SPELLINGS,
//! which is what a parser / sema / lowering regression would reject regardless of the dimension
//! values. The inline `kernels()` / `ABLATIONS` sources remain out of reach from a test file and
//! would need a library target to cover; the on-disk peers are the ones with the exotic spellings
//! (struct weights, shape-typed `Tensor[..]`, `[]f32` helper functions).

use wukong_span::{Interner, SourceId};

const VARIANT_SOURCES: &[(&str, &str)] = &[
    (
        "block_a_recognizer",
        include_str!("../wk/general/block_a_recognizer.wk"),
    ),
    (
        "block_b_natural",
        include_str!("../wk/general/block_b_natural.wk"),
    ),
    (
        "block_c_helpers",
        include_str!("../wk/general/block_c_helpers.wk"),
    ),
    (
        "block_d_struct",
        include_str!("../wk/general/block_d_struct.wk"),
    ),
    (
        "block_e_tensor",
        include_str!("../wk/general/block_e_tensor.wk"),
    ),
];
const LOSS_SOURCE: &str = include_str!("../wk/general/loss_focal.wk");
const SCAN_SOURCE: &str = include_str!("../wk/general/scan_s6.wk");

/// Substitute the 15 block placeholders exactly as `general.rs::subst` does under the default
/// `Dims` (s=256, d=256, h=4, f=512 => hd=64, hd2=32, scale=1/sqrt(64)=0.125).
fn subst_variant(src: &str) -> String {
    let (s, d, h, f) = (256usize, 256usize, 4usize, 512usize);
    let hd = d / h;
    let hd2 = hd / 2;
    let scale = 1.0f64 / (hd as f64).sqrt();
    let pairs: [(&str, String); 15] = [
        ("$SD$", (s * d).to_string()),
        ("$SS$", (s * s).to_string()),
        ("$SHD$", (s * hd).to_string()),
        ("$SH2$", (s * hd2).to_string()),
        ("$SF$", (s * f).to_string()),
        ("$DD$", (d * d).to_string()),
        ("$FD$", (f * d).to_string()),
        ("$DF$", (d * f).to_string()),
        ("$HD2$", hd2.to_string()),
        ("$HD$", hd.to_string()),
        ("$SCALE$", scale.to_string()),
        ("$S$", s.to_string()),
        ("$D$", d.to_string()),
        ("$H$", h.to_string()),
        ("$F$", f.to_string()),
    ];
    apply(src, &pairs)
}

/// Substitute the 5 loss placeholders exactly as `general.rs::subst_loss` under the default
/// `LossDims` (r=8192, c=1024, eps=0.125, all f32-exact by construction).
fn subst_loss(src: &str) -> String {
    let (r, c) = (8192usize, 1024usize);
    let eps = 0.125f32;
    let qo = eps / c as f32;
    let qt = 1.0f32 - eps + qo;
    let pairs: [(&str, String); 5] = [
        ("$RC$", (r * c).to_string()),
        ("$QT$", qt.to_string()),
        ("$QO$", qo.to_string()),
        ("$R$", r.to_string()),
        ("$C$", c.to_string()),
    ];
    apply(src, &pairs)
}

/// Substitute the 6 scan placeholders exactly as `general.rs::subst_scan` under the default
/// `ScanDims` (t=2048, d=256, n=16).
fn subst_scan(src: &str) -> String {
    let (t, d, n) = (2048usize, 256usize, 16usize);
    let pairs: [(&str, String); 6] = [
        ("$TD$", (t * d).to_string()),
        ("$DN$", (d * n).to_string()),
        ("$TN$", (t * n).to_string()),
        ("$T$", t.to_string()),
        ("$D$", d.to_string()),
        ("$N$", n.to_string()),
    ];
    apply(src, &pairs)
}

/// The `$`-delimited tokens are order-independent (`$S$` cannot match inside `$SD$`), so a single
/// pass in any order is faithful to the benchmark's own `replace` loop.
fn apply(src: &str, pairs: &[(&str, String)]) -> String {
    let mut out = src.to_string();
    for (tok, val) in pairs {
        out = out.replace(tok, val);
    }
    out
}

/// Front-end a fully-substituted peer source and assert it is accepted at every stage.
fn typecheck(label: &str, src: &str) {
    // Drift guard: an unsubstituted placeholder means the source grew a token this mirror does not
    // know. Every `$` in these sources is the delimiter of a known `$TOKEN$`, so a surviving one is
    // a real signal, not prose.
    if let Some(pos) = src.find('$') {
        let ctx: String = src[pos..].chars().take(24).collect();
        panic!(
            "{label}: an unsubstituted placeholder remains near {ctx:?} — the token set in \
             general.rs changed and this mirror must be updated in the same change"
        );
    }

    let mut interner = Interner::new();

    let (module, pd) = wukong_parser::parse_module(src, SourceId(0), &mut interner);
    let perrs: Vec<_> = pd.iter().filter(|d| d.is_error()).collect();
    assert!(
        perrs.is_empty(),
        "{label}: the parser rejected a peer source spelling — a language-surface change broke an \
         xbench peer:\n{perrs:?}"
    );

    let (sema, sd) = wukong_sema::check(&module, &interner);
    let serrs: Vec<_> = sd.iter().filter(|d| d.is_error()).collect();
    assert!(
        serrs.is_empty(),
        "{label}: sema rejected a peer source spelling — a resolution/type/shape change broke an \
         xbench peer:\n{serrs:?}"
    );

    let (mut program, ld) = wukong_mir_build::lower_program(&module, &sema, &mut interner);
    let lerrs: Vec<_> = ld.iter().filter(|d| d.is_error()).collect();
    assert!(
        lerrs.is_empty(),
        "{label}: mir_build rejected a peer source spelling — a lowering/recognizer change broke an \
         xbench peer:\n{lerrs:?}"
    );

    // The benchmark lowers at -O3; run the same so a pass that panics or asserts on these shapes is
    // caught here rather than only in a measured round.
    wukong_opt::optimize(&mut program, 3);
}

#[test]
fn every_block_variant_peer_source_typechecks() {
    for (label, src) in VARIANT_SOURCES {
        typecheck(label, &subst_variant(src));
    }
}

#[test]
fn the_focal_loss_peer_source_typechecks() {
    typecheck("loss_focal", &subst_loss(LOSS_SOURCE));
}

#[test]
fn the_s6_scan_peer_source_typechecks() {
    typecheck("scan_s6", &subst_scan(SCAN_SOURCE));
}
