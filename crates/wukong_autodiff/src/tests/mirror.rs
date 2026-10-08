//! P12.15: the recognizer ↔ autodiff coupling, enforced. Every `"wukong_*"` runtime symbol
//! `wukong_mir_build` can emit must be a forward kernel the tape differentiates
//! (`tape::FORWARD_KERNELS`), a routine the scalar core differentiates (`coverage::SCALAR_CORE`), or a
//! named, reasoned entry in `coverage::NOT_DIFFERENTIABLE` — and nothing else. A recognizer arm that
//! lands without a mirror (the historical `norm_f32_parallel` hole, where a model differentiated in
//! its serial spelling and not in its `@parallel` one) now fails here instead of in a user's
//! `--train`.

use crate::coverage::{NOT_DIFFERENTIABLE, SCALAR_CORE};
use crate::tape::FORWARD_KERNELS;
use std::collections::{BTreeSet, HashSet};

/// Every `"wukong_<ident>"` string literal in `wukong_mir_build`'s sources — the symbols lowering
/// can intern and emit.
fn mir_build_symbols() -> BTreeSet<String> {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../wukong_mir_build/src");
    let mut out = BTreeSet::new();
    let entries =
        std::fs::read_dir(&dir).unwrap_or_else(|e| panic!("cannot read {}: {e}", dir.display()));
    for entry in entries {
        let path = entry.expect("dir entry").path();
        if path.extension().and_then(|e| e.to_str()) != Some("rs") {
            continue;
        }
        let src = std::fs::read_to_string(&path).expect("read mir_build source");
        let mut rest = src.as_str();
        while let Some(at) = rest.find("\"wukong_") {
            let tail = &rest[at + 1..];
            let len = tail
                .find(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
                .unwrap_or(tail.len());
            // A bare `"wukong_"` (a prefix test in lowering) is not a symbol, and neither is any
            // literal handed to `starts_with` (`dispatch.rs`'s `"wukong_rt_"` service-prefix test, a
            // test's `"wukong_col"` family check): a prefix names a set, never one emitted call.
            let prefix_test = rest[..at].trim_end().ends_with("starts_with(");
            if len > "wukong_".len() && tail[len..].starts_with('"') && !prefix_test {
                out.insert(tail[..len].to_string());
            }
            rest = &tail[len..];
        }
    }
    out
}

#[test]
fn every_emitted_kernel_symbol_is_differentiable_or_declined_by_name() {
    let emitted = mir_build_symbols();
    // Vacuity floor: the scan itself must be finding the recognizer surface (157 symbols when this
    // test landed), or every assertion below passes over an empty set.
    assert!(
        emitted.len() >= 100,
        "the mir_build symbol scan found only {} symbols — the scan is broken",
        emitted.len()
    );
    let forward: HashSet<&str> = FORWARD_KERNELS.iter().map(|&(n, _, _)| n).collect();
    let scalar: HashSet<&str> = SCALAR_CORE.iter().copied().collect();
    let declined: HashSet<&str> = NOT_DIFFERENTIABLE.iter().map(|&(n, _)| n).collect();
    let mut problems = Vec::new();
    for s in &emitted {
        let homes = [
            ("FORWARD_KERNELS", forward.contains(s.as_str())),
            ("SCALAR_CORE", scalar.contains(s.as_str())),
            ("NOT_DIFFERENTIABLE", declined.contains(s.as_str())),
        ];
        let found: Vec<&str> = homes.iter().filter(|h| h.1).map(|h| h.0).collect();
        match found.len() {
            1 => {}
            0 => problems.push(format!(
                "`{s}` is emitted by wukong_mir_build but has no autodiff disposition: give it a \
                 VJP rule (tape::FORWARD_KERNELS) or list it in coverage::NOT_DIFFERENTIABLE \
                 with a reason"
            )),
            _ => problems.push(format!("`{s}` is listed in more than one table: {found:?}")),
        }
    }
    // No stale entries: a declined or scalar symbol mir_build no longer emits is dead weight that
    // would hide a renamed recognizer.
    for name in declined.iter().chain(scalar.iter()) {
        if !emitted.contains(*name) {
            problems.push(format!(
                "`{name}` is listed as non-differentiable but wukong_mir_build no longer emits it"
            ));
        }
    }
    for &(name, why) in NOT_DIFFERENTIABLE {
        if why.trim().len() < 10 {
            problems.push(format!("`{name}` is declined without a usable reason"));
        }
    }
    assert!(problems.is_empty(), "{}", problems.join("\n"));
}

/// The forward table is the single source of truth for recognition, arity and dispatch, so it must
/// not list a symbol twice (the later entry would be silently unreachable), and every `@parallel`
/// spelling must have its serial twin under the same rule.
#[test]
fn forward_kernel_table_is_well_formed() {
    let mut seen = HashSet::new();
    for &(name, kind, par) in FORWARD_KERNELS {
        assert!(seen.insert(name), "`{name}` is listed twice");
        if par {
            let serial = name
                .strip_suffix("_parallel")
                .unwrap_or_else(|| panic!("`{name}` is marked parallel but is not `*_parallel`"));
            assert!(
                FORWARD_KERNELS
                    .iter()
                    .any(|&(n, k, p)| n == serial && k == kind && !p),
                "`{name}` has no serial twin `{serial}` under the same rule"
            );
        } else {
            assert!(
                !name.ends_with("_parallel"),
                "`{name}` is spelled parallel but marked serial"
            );
        }
    }
}
