//! The multi-file front end: resolve, load, and namespace `import`ed source files.
//!
//! Semantics:
//!
//! - **Resolution**: `import a.b.c` — in *any* file of the program — resolves to
//!   `<root-dir>/a/b/c.wk`, where `<root-dir>` is the directory of the ROOT source file handed to
//!   the compiler and dots are path separators. A single segment `import util` is
//!   `<root-dir>/util.wk`.
//! - **Loading**: imports are walked depth-first and deduplicated by canonical path. An import
//!   cycle (`a` imports `b` imports `a`, or a self-import) is NOT an error — an already-loaded
//!   file is simply not reloaded, so every file's items land exactly once.
//! - **Namespaces**: every file is a module with its own namespace (see [`resolve`]). A file sees
//!   its own items plus what its OWN imports bring in — imports are not transitive — and another
//!   module's item is reachable only if it is declared `pub`:
//!     * `import a.b;` brings every `pub` item of `a.b` into scope unqualified (a local item or an
//!       explicit import shadows it; two such imports that disagree on a name make any USE of that
//!       name ambiguous, E0313) and binds the qualifier `b`, so `b::item` also works;
//!     * `import a.b as c;` binds only the qualifier: `c::item(..)`, `c::Enum::Variant`,
//!       `c::Struct { .. }`, and `c.Struct` / `c::Struct` in a type;
//!     * `import a.b.{x, y};` brings exactly `x` and `y` into scope (each must exist and be `pub`).
//!
//!   Naming a private item through any of these is E0312. The namespaces are implemented by
//!   resolution, not by a new symbol table downstream: each imported module's items are renamed
//!   to the program-unique symbol `<import.path>.<name>` (the root file's items keep their own
//!   names, so a single-file program is untouched) and every reference is rewritten to the symbol
//!   it resolves to in ITS file. Two modules may therefore both define a private `helper`. An
//!   `extern` function keeps its C name — it is the link symbol — and identical `extern`
//!   declarations repeated in several modules are merged.
//! - An imported file's own `module` header is informational, exactly like the root's — it is not
//!   required to match the import path (no error on mismatch).
//! - A missing/unreadable import file is **E0305** at the import site. Lex/parse diagnostics from
//!   imported files carry that file's own [`wukong_span::SourceId`], so they render with the
//!   right path/line/column through the shared [`SourceMap`].
//!
//! `NodeId` uniqueness across files is guaranteed by threading a watermark through
//! [`wukong_parser::parse_module_tokens_from`] — see its doc comment.

mod resolve;

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use wukong_ast::{Import, Item, ItemKind, Module};
use wukong_diag::{Diagnostic, DiagnosticSink};
use wukong_span::{Interner, SourceMap};

/// One source file of the program, indexed in load order (`0` is the root file).
pub(crate) struct FileInfo {
    /// The dotted import path the file was first loaded through (`lib.mathlib`); empty for the
    /// root. It prefixes the file's item symbols and names the module in diagnostics.
    pub(crate) path: String,
    /// The file's `import` items in source order, each with the index of the file it resolved
    /// to (`None` when it did not resolve — E0305 was reported).
    pub(crate) imports: Vec<(Import, Option<usize>)>,
}

/// The DFS work stack: a file index and the not-yet-visited remainder of its items.
type Pending = Vec<(usize, std::vec::IntoIter<Item>)>;

/// Load every file `module` imports (directly or transitively) depth-first, resolve every name
/// against its own file's namespace, and splice the imported items into `module.items`.
/// `root_path` is the root source file (its directory anchors resolution and it is pre-seeded as
/// "loaded" so a cycle back to the root is a no-op). Lex/parse diagnostics of imported files,
/// E0305 for unresolvable imports and the resolution errors (E0300/E0301/E0312/E0313) are emitted
/// into `sink`; `next_node` is the `NodeId` watermark from the root parse, advanced per file.
pub(crate) fn load_imports(
    module: &mut Module,
    root_path: &Path,
    sm: &mut SourceMap,
    interner: &mut Interner,
    sink: &mut DiagnosticSink,
    next_node: &mut u32,
) {
    // Most single-file compilations have no imports. Avoid filesystem identity lookup and
    // loader allocations entirely in that case.
    let root_imports: Vec<Import> = module
        .items
        .iter()
        .filter_map(|item| match &item.kind {
            ItemKind::Import(imp) => Some(imp.clone()),
            _ => None,
        })
        .collect();
    if root_imports.is_empty() {
        return;
    }
    let root_dir = root_path
        .parent()
        // `parent()` is `Some("")` for a bare relative filename like `prog.wk`; both that and a
        // pathless root resolve imports against the current directory.
        .filter(|p| !p.as_os_str().is_empty())
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."));

    let errors_before = sink.error_count();
    let mut loader = Loader {
        root_dir,
        sm,
        interner,
        sink,
        next_node,
        by_path: HashMap::new(),
        files: vec![FileInfo {
            path: String::new(),
            imports: Vec::new(),
        }],
        pending: Vec::new(),
    };
    loader.by_path.insert(canonical(root_path), Some(0));

    // Imported items in DFS order, each tagged with the index of the file that defines it.
    let mut extra: Vec<(usize, Item)> = Vec::new();
    for imp in root_imports {
        let target = loader.load_file(&imp);
        loader.files[0].imports.push((imp, target));
        // Iterator frames retain only unvisited items. Each child's token buffer is released
        // by load_file before descending, and import depth does not consume the call stack.
        // The parent frame stays in place so traversal and item order match recursive DFS.
        while let Some((file, items)) = loader.pending.last_mut() {
            let file = *file;
            let Some(item) = items.next() else {
                loader.pending.pop();
                continue;
            };
            if let ItemKind::Import(nested) = &item.kind {
                let nested = nested.clone();
                let target = loader.load_file(&nested);
                loader.files[file].imports.push((nested, target));
            } else {
                extra.push((file, item));
            }
        }
    }

    let Loader {
        files,
        interner,
        sink,
        ..
    } = loader;
    // A file that failed to load or parse leaves the program incomplete: resolving it would only
    // add cascading "not found" noise to the real error, and the driver stops here anyway.
    if sink.error_count() == errors_before {
        resolve::resolve_program(&mut module.items, &mut extra, &files, interner, sink);
    }
    module.items.extend(extra.into_iter().map(|(_, item)| item));
}

/// The state of one DFS walk over the import graph.
struct Loader<'a> {
    root_dir: PathBuf,
    sm: &'a mut SourceMap,
    interner: &'a mut Interner,
    sink: &'a mut DiagnosticSink,
    next_node: &'a mut u32,
    /// Canonical path -> file index; `None` for a file that failed to read (reported once).
    by_path: HashMap<PathBuf, Option<usize>>,
    files: Vec<FileInfo>,
    pending: Pending,
}

impl Loader<'_> {
    /// Resolve `imp` to a file index, parsing the file (and queueing its items on `pending`) the
    /// first time it is seen. `None` means the import does not resolve (E0305, reported once per
    /// file, not once per mention).
    fn load_file(&mut self, imp: &Import) -> Option<usize> {
        // `a.b.c` -> `<root-dir>/a/b/c.wk`.
        let mut file = self.root_dir.clone();
        let mut dotted = String::new();
        for (i, seg) in imp.path.segments.iter().enumerate() {
            let s = self.interner.resolve(seg.sym);
            file.push(s);
            if i > 0 {
                dotted.push('.');
            }
            dotted.push_str(s);
        }
        file.set_extension("wk");

        // Dedup / cycle break by canonical path identity — recorded BEFORE parsing the file, so
        // a cycle back through this file (or a second report for the same missing file) is a
        // lookup, not a reload.
        let key = canonical(&file);
        if let Some(&known) = self.by_path.get(&key) {
            return known;
        }

        let src = match std::fs::read_to_string(&file) {
            Ok(s) => s,
            Err(e) => {
                self.by_path.insert(key, None);
                self.sink.emit(
                    Diagnostic::error(format!(
                        "unresolved import `{dotted}`: cannot read `{}`: {e}",
                        file.display()
                    ))
                    .with_code("E0305")
                    .primary(
                        imp.path.span,
                        "this import does not resolve to a source file",
                    )
                    .help(format!(
                        "`import a.b` resolves to `a/b.wk` under the root source file's directory \
                         — expected `{}`",
                        file.display()
                    )),
                );
                return None;
            }
        };

        let idx = self.files.len();
        self.by_path.insert(key, Some(idx));
        self.files.push(FileInfo {
            path: dotted,
            imports: Vec::new(),
        });

        let id = self.sm.add(file.display().to_string(), src);
        let (tokens, lex_diags) = wukong_lexer::tokenize(self.sm.source(id), id);
        for d in lex_diags {
            self.sink.emit(d);
        }
        let (m, parse_diags, next) = wukong_parser::parse_module_tokens_from(
            &tokens,
            self.sm.source(id),
            self.interner,
            *self.next_node,
        );
        *self.next_node = next;
        for d in parse_diags {
            self.sink.emit(d);
        }

        // The imported file's `module` header — `m.name` — remains informational and unused.
        self.pending.push((idx, m.items.into_iter()));
        Some(idx)
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

    /// Lex+parse `src` as a root file living in `tests/imports/` (`name` is its file name — it
    /// need not exist on disk unless something imports it back), then run the loader.
    fn load_src(name: &str, src: &str) -> (Module, DiagnosticSink, Interner, usize) {
        let path = imports_dir().join(name);
        let mut sm = SourceMap::new();
        let id = sm.add(path.display().to_string(), src.to_string());
        let (tokens, ld) = wukong_lexer::tokenize(sm.source(id), id);
        assert!(ld.is_empty(), "unexpected lexer diags: {ld:?}");
        let mut interner = Interner::new();
        let (mut module, pd, mut next) =
            wukong_parser::parse_module_tokens_from(&tokens, sm.source(id), &mut interner, 0);
        assert!(pd.is_empty(), "unexpected parser diags: {pd:?}");
        let mut sink = DiagnosticSink::new();
        load_imports(
            &mut module,
            &path,
            &mut sm,
            &mut interner,
            &mut sink,
            &mut next,
        );
        (module, sink, interner, sm.file_count())
    }

    /// Load the on-disk root `tests/imports/<root>`.
    fn load(root: &str) -> (Module, DiagnosticSink, Interner, usize) {
        let path = imports_dir().join(root);
        let src = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
        load_src(root, &src)
    }

    /// The merged module's fn names (after resolution renamed the imported ones).
    fn fn_names(module: &Module, interner: &Interner) -> Vec<String> {
        module
            .items
            .iter()
            .filter_map(|i| match &i.kind {
                ItemKind::Fn(f) => Some(interner.resolve(f.name.sym).to_string()),
                _ => None,
            })
            .collect()
    }

    /// Every expression path in the merged module, as the AST printer renders it (`path a::b`).
    fn expr_paths(module: &Module, interner: &Interner) -> Vec<String> {
        wukong_ast::print::print_module(module, interner)
            .lines()
            .filter_map(|l| l.trim().strip_prefix("path ").map(str::to_string))
            .collect()
    }

    fn codes(sink: &DiagnosticSink) -> Vec<&'static str> {
        sink.diagnostics().iter().filter_map(|d| d.code).collect()
    }

    fn count(names: &[String], name: &str) -> usize {
        names.iter().filter(|f| *f == name).count()
    }

    /// `cycle_a.wk` imports `cycle_b.wk`, which imports `cycle_a.wk` back. The cycle must not
    /// error and must not reload either file: each file's items land exactly once, the imported
    /// one under its module-qualified symbol.
    #[test]
    fn import_cycle_loads_each_file_once() {
        let (module, sink, interner, files) = load("cycle_a.wk");
        assert!(
            !sink.has_errors(),
            "cycle must not be an error: {:?}",
            sink.diagnostics()
        );
        assert_eq!(files, 2, "cycle_b's import of cycle_a must not reload it");
        let fns = fn_names(&module, &interner);
        assert_eq!(count(&fns, "from_a"), 1, "root items keep their names");
        assert_eq!(count(&fns, "cycle_b.from_b"), 1, "{fns:?}");
        assert!(expr_paths(&module, &interner).contains(&"cycle_b.from_b".to_string()));
    }

    /// `diamond_root.wk` imports `diamond_b` and `diamond_c`, which both import `diamond_d`
    /// (also spelled through a redundant self-import): `d`'s items must land exactly once.
    #[test]
    fn diamond_import_dedups_by_canonical_path() {
        let (module, sink, interner, files) = load("diamond_root.wk");
        assert!(
            !sink.has_errors(),
            "diamond must not be an error: {:?}",
            sink.diagnostics()
        );
        assert_eq!(
            files, 4,
            "root + b + c + d, d loaded once through two paths"
        );
        let fns = fn_names(&module, &interner);
        assert_eq!(
            count(&fns, "diamond_d.shared_leaf"),
            1,
            "diamond leaf must be spliced exactly once: {fns:?}"
        );
        assert_eq!(count(&fns, "diamond_b.from_diamond_b"), 1);
        assert_eq!(count(&fns, "diamond_c.from_diamond_c"), 1);
    }

    /// An import that resolves to no file is a clean E0305 at the import site (and is reported
    /// once, not once per mention).
    #[test]
    fn missing_import_is_e0305() {
        let (_, sink, _, _) = load_src(
            "virtual_root.wk",
            "import does.not.exist;\nimport does.not.exist;\nfn main() -> i32 { return 0; }\n",
        );
        assert!(sink.has_errors());
        assert_eq!(
            codes(&sink),
            vec!["E0305"],
            "one E0305 per missing file, not per import: {:?}",
            sink.diagnostics()
        );
    }

    /// The tentpole: `ns_a` and `ns_b` both define a PRIVATE `helper` — in the old flat namespace
    /// that was E0300 — and each module's own call resolves to its own. A glob import reaches
    /// `a_val`, the `as` alias reaches `b::b_val`, the implicit qualifier reaches `ns_a::Pt { .. }`
    /// and an enum through `Mode::Fast`, and the types are rewritten with the values.
    #[test]
    fn modules_have_private_namespaces_and_qualified_access() {
        let (module, sink, interner, _) = load_src(
            "virtual_root.wk",
            "import ns_a;\nimport ns_b as b;\n\
             fn main() -> i32 {\n\
                 let p = ns_a::Pt { x: 1, y: 2 };\n\
                 let m: Mode = Mode::Fast;\n\
                 let q: b.Unit = b::Unit::One;\n\
                 return a_val() + b::b_val() + p.x;\n\
             }\n",
        );
        assert!(
            !sink.has_errors(),
            "namespaced program must resolve: {:?}",
            sink.diagnostics()
        );
        let fns = fn_names(&module, &interner);
        for f in [
            "main",
            "ns_a.a_val",
            "ns_a.helper",
            "ns_b.b_val",
            "ns_b.helper",
            "ns_b.a_val",
        ] {
            assert_eq!(count(&fns, f), 1, "`{f}` in {fns:?}");
        }
        let paths = expr_paths(&module, &interner);
        for p in [
            "ns_a.a_val",
            "ns_b.b_val",
            "ns_a.helper",
            "ns_b.helper",
            "ns_a.Mode",
            "ns_b.Unit",
        ] {
            assert!(paths.contains(&p.to_string()), "`{p}` in {paths:?}");
        }
        assert!(
            !paths
                .iter()
                .any(|p| p == "helper" || p == "b" || p == "a_val"),
            "no reference may stay unresolved: {paths:?}"
        );
        let printed = wukong_ast::print::print_module(&module, &interner);
        assert!(printed.contains("struct-lit ns_a.Pt"), "{printed}");
    }

    /// Two glob imports that both export `a_val` make a USE of `a_val` ambiguous (E0313) — but
    /// importing both is fine as long as the clashing name is not used unqualified.
    #[test]
    fn glob_ambiguity_is_reported_at_use() {
        let (_, sink, _, _) = load_src(
            "virtual_root.wk",
            "import ns_a;\nimport ns_b;\nfn main() -> i32 { return a_val(); }\n",
        );
        assert_eq!(codes(&sink), vec!["E0313"], "{:?}", sink.diagnostics());

        let (module, sink, interner, _) = load_src(
            "virtual_root.wk",
            "import ns_a;\nimport ns_b;\n\
             fn main() -> i32 { return b_val() + ns_a::a_val() + ns_b::a_val(); }\n",
        );
        assert!(!sink.has_errors(), "{:?}", sink.diagnostics());
        let paths = expr_paths(&module, &interner);
        assert!(paths.contains(&"ns_a.a_val".to_string()), "{paths:?}");
        assert!(paths.contains(&"ns_b.a_val".to_string()), "{paths:?}");
    }

    /// A private item is unreachable from another file however it is named: by selection, by
    /// qualified path, or by a glob (which simply does not bring it in).
    #[test]
    fn private_items_are_e0312() {
        let (_, sink, _, _) = load_src(
            "virtual_root.wk",
            "import ns_a.{helper};\nfn main() -> i32 { return 0; }\n",
        );
        assert_eq!(codes(&sink), vec!["E0312"], "{:?}", sink.diagnostics());

        let (_, sink, _, _) = load_src(
            "virtual_root.wk",
            "import ns_a as a;\nfn main() -> i32 { return a::helper(); }\n",
        );
        assert_eq!(codes(&sink), vec!["E0312"], "{:?}", sink.diagnostics());

        // A glob does not import the private `helper`: the reference is left for sema's E0301.
        let (module, sink, interner, _) = load_src(
            "virtual_root.wk",
            "import ns_a;\nfn main() -> i32 { return helper(); }\n",
        );
        assert!(!sink.has_errors(), "{:?}", sink.diagnostics());
        assert!(expr_paths(&module, &interner).contains(&"helper".to_string()));
    }

    /// `import m.{x}` binds exactly `x`: `Pt` from the same module stays unresolved (sema's E0301),
    /// and a missing selection is E0301 at the import.
    #[test]
    fn selective_import_binds_only_the_listed_names() {
        let (module, sink, interner, _) = load_src(
            "virtual_root.wk",
            "import ns_a.{a_val};\n\
             fn main() -> i32 { let s = Pt { x: 1, y: 2 }; return a_val(); }\n",
        );
        assert!(!sink.has_errors(), "{:?}", sink.diagnostics());
        assert!(expr_paths(&module, &interner).contains(&"ns_a.a_val".to_string()));
        let printed = wukong_ast::print::print_module(&module, &interner);
        assert!(
            !printed.contains("struct-lit ns_a.Pt"),
            "an unselected item must not resolve: {printed}"
        );

        let (_, sink, _, _) = load_src(
            "virtual_root.wk",
            "import ns_a.{nope};\nfn main() -> i32 { return 0; }\n",
        );
        assert_eq!(codes(&sink), vec!["E0301"], "{:?}", sink.diagnostics());
    }

    /// Locals, parameters and generics shadow imported items exactly as in sema, so resolution
    /// must never rewrite them.
    #[test]
    fn locals_and_generics_shadow_imported_items() {
        let (module, sink, interner, _) = load_src(
            "virtual_root.wk",
            "import ns_a;\n\
             fn f(SIZE: i64) -> i64 { return SIZE; }\n\
             fn g<Pt>(x: Pt) -> Pt { return x; }\n\
             fn main() -> i32 {\n\
                 let a_val = 3;\n\
                 for helper in 0..2 { let z = helper; }\n\
                 return match a_val { Mode => Mode };\n\
             }\n",
        );
        assert!(!sink.has_errors(), "{:?}", sink.diagnostics());
        let paths = expr_paths(&module, &interner);
        for p in ["ns_a.a_val", "ns_a.SIZE", "ns_a.Mode"] {
            assert!(!paths.contains(&p.to_string()), "`{p}` leaked: {paths:?}");
        }
        assert!(paths.contains(&"a_val".to_string()));
        assert!(paths.contains(&"SIZE".to_string()));
        let printed = wukong_ast::print::print_module(&module, &interner);
        assert_eq!(
            printed.matches("ns_a.Pt").count(),
            1,
            "only ns_a's own declaration may carry the qualified name: {printed}"
        );
    }

    /// A selected name colliding with a local definition is the ordinary duplicate (E0300); a glob
    /// is silently shadowed by the local item instead.
    #[test]
    fn selection_colliding_with_a_local_item_is_e0300() {
        let (_, sink, _, _) = load_src(
            "virtual_root.wk",
            "import ns_a.{a_val};\nfn a_val() -> i32 { return 1; }\n\
             fn main() -> i32 { return a_val(); }\n",
        );
        assert_eq!(codes(&sink), vec!["E0300"], "{:?}", sink.diagnostics());

        let (module, sink, interner, _) = load_src(
            "virtual_root.wk",
            "import ns_a;\nfn a_val() -> i32 { return 1; }\n\
             fn main() -> i32 { return a_val(); }\n",
        );
        assert!(!sink.has_errors(), "{:?}", sink.diagnostics());
        let paths = expr_paths(&module, &interner);
        assert!(paths.contains(&"a_val".to_string()), "{paths:?}");
        assert!(!paths.contains(&"ns_a.a_val".to_string()), "{paths:?}");
    }

    /// Imports are per file: `ns_leak` calls the root's `root_only` without importing the root,
    /// which the flat namespace used to allow.
    #[test]
    fn root_items_are_not_visible_to_modules_that_do_not_import_them() {
        let (_, sink, _, _) = load_src(
            "virtual_root.wk",
            "import ns_leak;\nfn root_only() -> i32 { return 1; }\n\
             fn main() -> i32 { return leak(); }\n",
        );
        assert_eq!(codes(&sink), vec!["E0301"], "{:?}", sink.diagnostics());
    }

    /// The same C function declared `extern` in two modules is one link symbol: the identical
    /// redeclaration is merged (sema would otherwise report E0300), and both files can call it.
    #[test]
    fn identical_extern_declarations_are_merged() {
        let (module, sink, interner, _) = load_src(
            "virtual_root.wk",
            "import ns_ext;\nextern \"C\" { fn cos(x: f64) -> f64; }\n\
             fn main() -> i32 { return (cos(0.0) + twice_cos(0.0)) as i32; }\n",
        );
        assert!(!sink.has_errors(), "{:?}", sink.diagnostics());
        let cos = interner.get("cos").expect("interned");
        let decls = module
            .items
            .iter()
            .filter_map(|i| match &i.kind {
                ItemKind::Extern(blk) => Some(blk.items.iter().filter(|f| f.name.sym == cos)),
                _ => None,
            })
            .flatten()
            .count();
        assert_eq!(decls, 1, "the repeated `cos` declaration must be merged");
        let paths = expr_paths(&module, &interner);
        assert!(paths.contains(&"ns_ext.twice_cos".to_string()), "{paths:?}");
        assert_eq!(
            paths.iter().filter(|p| *p == "cos").count(),
            2,
            "both files call the C symbol by its own name: {paths:?}"
        );
    }
}
