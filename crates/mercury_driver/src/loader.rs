//! The multi-file front end: resolve, load, and splice `import`ed source files.
//!
//! v1 semantics (deliberately simple and correct):
//!
//! - **Resolution**: `import a.b.c` — in *any* file of the program — resolves to
//!   `<root-dir>/a/b/c.mer`, where `<root-dir>` is the directory of the ROOT source file handed to
//!   the compiler and dots are path separators. A single segment `import util` is
//!   `<root-dir>/util.mer`.
//! - **Loading**: imports are walked depth-first and deduplicated by canonical path. An import
//!   cycle (`a` imports `b` imports `a`, or a self-import) is NOT an error — an already-loaded
//!   file is simply not reloaded, so every file's items land exactly once.
//! - **Namespace**: the program is ONE merged flat namespace. Every loaded file's items are
//!   spliced into the root [`Module`] before sema, so cross-file calls and types resolve with the
//!   ordinary single-module machinery. A top-level name defined in two files is the ordinary
//!   duplicate-name error (E0300), which points at both definitions. `import x as y` aliases and
//!   `import x.{a, b}` item lists parse but neither rename nor restrict anything in v1.
//! - An imported file's own `module` header is informational, exactly like the root's — it is not
//!   required to match the import path (no error on mismatch in v1).
//! - A missing/unreadable import file is **E0305** at the import site. Lex/parse diagnostics from
//!   imported files carry that file's own [`mercury_span::SourceId`], so they render with the
//!   right path/line/column through the shared [`SourceMap`].
//!
//! `NodeId` uniqueness across files is guaranteed by threading a watermark through
//! [`mercury_parser::parse_module_tokens_from`] — see its doc comment.

use std::collections::HashSet;
use std::path::{Path, PathBuf};

use mercury_ast::{Import, Item, ItemKind, Module};
use mercury_diag::{Diagnostic, DiagnosticSink};
use mercury_span::{Interner, SourceMap};

/// Recursively load every file `module` imports (directly or transitively) and splice their items
/// into `module.items`. `root_path` is the root source file (its directory anchors resolution and
/// it is pre-seeded as "loaded" so a cycle back to the root is a no-op). Lex/parse diagnostics of
/// imported files and E0305 for unresolvable imports are emitted into `sink`; `next_node` is the
/// `NodeId` watermark from the root parse, advanced per imported file.
pub(crate) fn load_imports(
    module: &mut Module,
    root_path: &Path,
    sm: &mut SourceMap,
    interner: &mut Interner,
    sink: &mut DiagnosticSink,
    next_node: &mut u32,
) {
    let root_dir = root_path
        .parent()
        // `parent()` is `Some("")` for a bare relative filename like `prog.mer`; both that and a
        // pathless root resolve imports against the current directory.
        .filter(|p| !p.as_os_str().is_empty())
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."));

    let mut loaded: HashSet<PathBuf> = HashSet::new();
    loaded.insert(canonical(root_path));

    let mut extra: Vec<Item> = Vec::new();
    for item in &module.items {
        if let ItemKind::Import(imp) = &item.kind {
            load_one(
                imp, &root_dir, sm, interner, sink, next_node, &mut loaded, &mut extra,
            );
        }
    }
    module.items.append(&mut extra);
}

/// Load the file `imp` names (unless already loaded), recurse into its own imports depth-first,
/// and push its non-import items onto `out`.
#[allow(clippy::too_many_arguments)] // internal DFS; a state struct would just rename the args
fn load_one(
    imp: &Import,
    root_dir: &Path,
    sm: &mut SourceMap,
    interner: &mut Interner,
    sink: &mut DiagnosticSink,
    next_node: &mut u32,
    loaded: &mut HashSet<PathBuf>,
    out: &mut Vec<Item>,
) {
    // `a.b.c` -> `<root-dir>/a/b/c.mer` (+ the dotted form for diagnostics).
    let mut file = root_dir.to_path_buf();
    let mut dotted = String::new();
    for (i, seg) in imp.path.segments.iter().enumerate() {
        let s = interner.resolve(seg.sym);
        if i > 0 {
            dotted.push('.');
        }
        dotted.push_str(s);
        file.push(s);
    }
    file.set_extension("mer");

    // Dedup / cycle break by canonical path identity — inserted BEFORE parsing the file, so a
    // cycle back through this file (or a second report for the same missing file) is a no-op.
    if !loaded.insert(canonical(&file)) {
        return;
    }

    let src = match std::fs::read_to_string(&file) {
        Ok(s) => s,
        Err(e) => {
            sink.emit(
                Diagnostic::error(format!(
                    "unresolved import `{dotted}`: cannot read `{}`: {e}",
                    file.display()
                ))
                .with_code("E0305")
                .primary(imp.path.span, "this import does not resolve to a source file")
                .help(format!(
                    "`import a.b` resolves to `a/b.mer` under the root source file's directory — \
                     expected `{}`",
                    file.display()
                )),
            );
            return;
        }
    };

    let id = sm.add(file.display().to_string(), src);
    let (tokens, lex_diags) = mercury_lexer::tokenize(sm.source(id), id);
    for d in lex_diags {
        sink.emit(d);
    }
    let (m, parse_diags, next) =
        mercury_parser::parse_module_tokens_from(&tokens, sm.source(id), interner, *next_node);
    *next_node = next;
    for d in parse_diags {
        sink.emit(d);
    }

    // Depth-first, in source order: an import recurses immediately; every other item is spliced.
    // (The imported file's `module` header — `m.name` — is informational and intentionally unused.)
    for item in m.items {
        if let ItemKind::Import(nested) = &item.kind {
            load_one(nested, root_dir, sm, interner, sink, next_node, loaded, out);
        } else {
            out.push(item);
        }
    }
}

/// Canonical path identity for dedup: two spellings of the same on-disk file (case, `..`, symlink)
/// compare equal. A path that cannot be canonicalized (it does not exist) falls back to itself —
/// good enough, since it then fails to read with E0305 anyway.
fn canonical(p: &Path) -> PathBuf {
    std::fs::canonicalize(p).unwrap_or_else(|_| p.to_path_buf())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `<repo>/tests/imports` — fixtures used only by these unit tests (the e2e harnesses scan
    /// `tests/run` / `tests/fail` and ignore this directory).
    fn imports_dir() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/imports")
    }

    /// Lex+parse `root` (a file under `tests/imports/`), run the loader, and return the merged
    /// module's fn names, the sink, and how many files the SourceMap ended up with.
    fn load(root: &str) -> (Vec<String>, DiagnosticSink, usize) {
        let path = imports_dir().join(root);
        let src = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
        let mut sm = SourceMap::new();
        let id = sm.add(path.display().to_string(), src);
        let (tokens, ld) = mercury_lexer::tokenize(sm.source(id), id);
        assert!(ld.is_empty(), "unexpected lexer diags: {ld:?}");
        let mut interner = Interner::new();
        let (mut module, pd, mut next) =
            mercury_parser::parse_module_tokens_from(&tokens, sm.source(id), &mut interner, 0);
        assert!(pd.is_empty(), "unexpected parser diags: {pd:?}");
        let mut sink = DiagnosticSink::new();
        load_imports(&mut module, &path, &mut sm, &mut interner, &mut sink, &mut next);
        let fns = module
            .items
            .iter()
            .filter_map(|i| match &i.kind {
                ItemKind::Fn(f) => Some(interner.resolve(f.name.sym).to_string()),
                _ => None,
            })
            .collect();
        (fns, sink, sm.file_count())
    }

    fn count(fns: &[String], name: &str) -> usize {
        fns.iter().filter(|f| *f == name).count()
    }

    /// `cycle_a.mer` imports `cycle_b.mer`, which imports `cycle_a.mer` back. The cycle must not
    /// error and must not reload either file: each file's items land exactly once.
    #[test]
    fn import_cycle_loads_each_file_once() {
        let (fns, sink, files) = load("cycle_a.mer");
        assert!(!sink.has_errors(), "cycle must not be an error: {:?}", sink.diagnostics());
        assert_eq!(files, 2, "cycle_b's import of cycle_a must not reload it");
        assert_eq!(count(&fns, "from_a"), 1);
        assert_eq!(count(&fns, "from_b"), 1);
    }

    /// `diamond_root.mer` imports `diamond_b` and `diamond_c`, which both import `diamond_d`
    /// (also spelled through a redundant self-import): `d`'s items must land exactly once.
    #[test]
    fn diamond_import_dedups_by_canonical_path() {
        let (fns, sink, files) = load("diamond_root.mer");
        assert!(!sink.has_errors(), "diamond must not be an error: {:?}", sink.diagnostics());
        assert_eq!(files, 4, "root + b + c + d, d loaded once through two paths");
        assert_eq!(count(&fns, "shared_leaf"), 1, "diamond leaf must be spliced exactly once");
        assert_eq!(count(&fns, "from_diamond_b"), 1);
        assert_eq!(count(&fns, "from_diamond_c"), 1);
    }

    /// An import that resolves to no file is a clean E0305 at the import site (and is reported
    /// once, not once per mention).
    #[test]
    fn missing_import_is_e0305() {
        let src = "import does.not.exist;\nimport does.not.exist;\nfn main() -> i32 { return 0; }\n";
        let mut sm = SourceMap::new();
        // The root itself is synthetic (not on disk): the loader only needs its directory.
        let root = imports_dir().join("virtual_root.mer");
        let id = sm.add(root.display().to_string(), src.to_string());
        let (tokens, _) = mercury_lexer::tokenize(sm.source(id), id);
        let mut interner = Interner::new();
        let (mut module, pd, mut next) =
            mercury_parser::parse_module_tokens_from(&tokens, sm.source(id), &mut interner, 0);
        assert!(pd.is_empty(), "unexpected parser diags: {pd:?}");
        let mut sink = DiagnosticSink::new();
        load_imports(&mut module, &root, &mut sm, &mut interner, &mut sink, &mut next);
        assert!(sink.has_errors());
        let diags = sink.diagnostics();
        assert_eq!(diags.len(), 1, "one E0305 per missing file, not per import: {diags:?}");
        assert_eq!(diags[0].code, Some("E0305"));
    }
}
