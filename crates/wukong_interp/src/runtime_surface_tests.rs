//! A census of the `wukong_runtime` dispatch surface the interpreter shares with the native
//! backend — an explicit record of the exact set of symbols for which no differential,
//! opt-invariance or determinism gate can tell "the kernel is right" from "the kernel is wrong in
//! the same way on both sides".
//!
//! CLAUDE.md states that recognized kernels are gate-blind; this pins the *size and membership* of
//! that blind spot so it cannot grow or silently shrink unnoticed. It does **not** prove any kernel
//! is correct: per-symbol value correctness is owned by each `wukong_runtime` module's own
//! scalar-twin / f64-reference tests, and every-lane corpus fixtures. What this catches is a
//! refactor that quietly drops a kernel's dispatch (so the surface shrinks and the lost kernel's
//! only remaining caller is gone) — the interpreter is the oracle, so losing a dispatch here means
//! the native backend loses its only differential peer for that kernel.
//!
//! The scan is over this crate's own source, embedded at compile time so there is no runtime path
//! to get wrong.

/// Every distinct `wukong_runtime::wukong_*` callee named in `lib.rs`, extracted textually.
fn dispatched_runtime_symbols() -> std::collections::BTreeSet<String> {
    const SRC: &str = include_str!("lib.rs");
    const PREFIX: &str = "wukong_runtime::wukong_";
    let mut out = std::collections::BTreeSet::new();
    let mut rest = SRC;
    while let Some(i) = rest.find(PREFIX) {
        // Keep the `wukong_` of the symbol, dropping the `wukong_runtime::` module path.
        let sym_start = i + "wukong_runtime::".len();
        let tail = &rest[sym_start..];
        let end = tail
            .find(|c: char| !c.is_ascii_alphanumeric() && c != '_')
            .unwrap_or(tail.len());
        out.insert(tail[..end].to_string());
        rest = &tail[end..];
    }
    out
}

/// The dispatch surface must not silently shrink, and must still contain the load-bearing kernels.
///
/// `FLOOR` is a floor, not an exact pin (matching `run.rs::MIN_FIXTURES` and
/// `recognize.rs::PROGRAM_FLOOR`): adding a kernel never trips it, losing a chunk of the surface
/// does. The observed surface is ~81 symbols (80 kernels + `wukong_now_ns`); the floor sits below
/// that so minor churn does not false-fail, while a refactor that drops a family of kernels does.
#[test]
fn interpreter_runtime_dispatch_surface_is_pinned() {
    const FLOOR: usize = 75;
    // Symbols whose loss would be a silent correctness hole — each is a recognizer cascade head
    // verified elsewhere to dispatch (see `crates/wukongc/tests/recognize.rs`).
    const MUST_HAVE: &[&str] = &[
        "wukong_sgemm",
        "wukong_sgemm_nt_epi",
        "wukong_vmath_f32",
        "wukong_norm_f32",
        "wukong_cumsum_f32",
        "wukong_velem_f32",
    ];

    let surface = dispatched_runtime_symbols();
    assert!(
        surface.len() >= FLOOR,
        "the interpreter now dispatches only {} distinct wukong_runtime symbols (floor {FLOOR}) — a \
         kernel family lost its dispatch, and with it the native backend's only differential peer:\n{:#?}",
        surface.len(),
        surface
    );
    for want in MUST_HAVE {
        assert!(
            surface.contains(*want),
            "{want} is no longer dispatched by the interpreter — its value correctness is no longer \
             reachable from the differential oracle"
        );
    }
    eprintln!(
        "interpreter runtime dispatch surface: {} distinct wukong_runtime symbols",
        surface.len()
    );
}
