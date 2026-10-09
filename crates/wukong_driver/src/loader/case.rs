//! Exact-case import resolution (E0330).
//!
//! `import a.b` names the file `a/b.wk` byte for byte. Opening that path asks the file system,
//! which folds case on Windows and (by default) macOS but not on Linux, so a program importing
//! `Util` for a file `util.wk` compiled on the first two and was E0305 on the third: the same
//! source meant different things on different hosts, and the project's own development host is
//! the forgiving one. Resolution now compares each path component against the directory listing
//! itself, as Go and the TypeScript compiler's `forceConsistentCasingInFileNames` do, so the
//! answer is the same everywhere: a component that matches an entry only up to case is E0330
//! naming the on-disk spelling, whatever the host's file system would have done.

use std::path::Path;

/// The on-disk spelling of the path `components` under `base` when at least one component matches
/// a directory entry only up to letter case. `None` when every component matches an entry byte
/// for byte (the import is spelled correctly), when no casing of the path exists (E0305 reports
/// that), or when a directory cannot be listed (nothing can be said, so nothing is reported).
pub(super) fn case_mismatch(base: &Path, components: &[String]) -> Option<Vec<String>> {
    let mut dir = base.to_path_buf();
    let mut on_disk = Vec::with_capacity(components.len());
    let mut mismatched = false;
    for want in components {
        let mut exact = false;
        let mut folded: Option<String> = None;
        for entry in std::fs::read_dir(&dir).ok()?.flatten() {
            let name = entry.file_name();
            let Some(name) = name.to_str() else {
                continue;
            };
            if name == want {
                exact = true;
                break;
            }
            if folded.is_none() && name.to_lowercase() == want.to_lowercase() {
                folded = Some(name.to_string());
            }
        }
        let actual = if exact {
            want.clone()
        } else {
            mismatched = true;
            folded?
        };
        dir.push(&actual);
        on_disk.push(actual);
    }
    mismatched.then_some(on_disk)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn imports_dir() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/imports")
    }

    fn parts(p: &[&str]) -> Vec<String> {
        p.iter().map(|s| s.to_string()).collect()
    }

    /// A byte comparison against the directory listing, so the answer does not depend on whether
    /// this host's file system folds case.
    #[test]
    fn a_case_only_difference_names_the_on_disk_spelling() {
        assert_eq!(
            case_mismatch(&imports_dir(), &parts(&["NS_A.wk"])),
            Some(parts(&["ns_a.wk"]))
        );
        assert_eq!(
            case_mismatch(&imports_dir(), &parts(&["Ns_b.WK"])),
            Some(parts(&["ns_b.wk"]))
        );
    }

    #[test]
    fn an_exact_or_missing_path_is_not_a_case_mismatch() {
        assert_eq!(case_mismatch(&imports_dir(), &parts(&["ns_a.wk"])), None);
        assert_eq!(case_mismatch(&imports_dir(), &parts(&["no_such.wk"])), None);
        assert_eq!(
            case_mismatch(&imports_dir().join("no_such_dir"), &parts(&["ns_a.wk"])),
            None
        );
    }
}
