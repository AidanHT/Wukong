//! The AOT runtime cache's key, as the **production** key builders assemble it:
//! [`aot_runtime_material`] for the two rustc flavours and [`cc_runtime_material`] for the `cc`
//! fallback. `aot_cache`'s own unit tests prove that `Material` + `key_of` turn any changed line
//! into a different key; these prove the lines are there at all. Without them, dropping an input
//! from a builder (the shim hash, the recipe, the C runtime source) would leave every test green
//! while an edited shim or recipe silently reused a stale cached runtime.

use super::*;
use std::time::{Duration, SystemTime};

/// A real `rustc -vV` shape: every line is key material.
const RUSTC_VV: &str = "rustc 1.97.1 (8bab26f4f 2026-07-14)
binary: rustc
commit-hash: 8bab26f4f68e0e26f0bb7960be334d5b520ea452
commit-date: 2026-07-14
host: x86_64-pc-windows-msvc
release: 1.97.1
LLVM version: 22.1.6
";

/// A real `cc --version` shape (two non-empty lines and a blank one).
const CC_VERSION: &str = "gcc.exe (Rev2, Built by MSYS2 project) 14.2.0
Copyright (C) 2024 Free Software Foundation, Inc.

";

fn scratch(name: &str) -> PathBuf {
    let dir =
        std::env::temp_dir().join(format!("wukong-aot-material-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn stamp(path: &Path, secs: u64) {
    std::fs::File::options()
        .write(true)
        .open(path)
        .unwrap()
        .set_modified(SystemTime::UNIX_EPOCH + Duration::from_secs(secs))
        .unwrap();
}

/// Every `expected` line is a whole line of `material`.
fn assert_lines(what: &str, material: &str, expected: &[String]) {
    let lines: Vec<&str> = material.lines().collect();
    for want in expected {
        assert!(
            lines.contains(&want.as_str()),
            "{what}: the key material lacks the line `{want}`:\n{material}"
        );
    }
}

#[test]
fn the_rustc_runtime_key_names_every_input() {
    let dir = scratch("rustc-lines");
    let rlib = dir.join("libwukong_runtime-0123456789abcdef.rlib");
    std::fs::write(&rlib, b"rlib bytes").unwrap();
    stamp(&rlib, 1_700_000_000);
    for flavor in [aot_cache::Flavor::RustNative, aot_cache::Flavor::RustLto] {
        let material = aot_runtime_material(flavor, RUSTC_VV, &rlib).unwrap();
        let mut expected = vec![
            format!("flavor {}", flavor.name()),
            format!("host {}", aot_cache::host_material()),
            format!("runtime {}", aot_cache::file_identity(&rlib).unwrap()),
            // The exact text `build_aot_runtime` writes and compiles.
            format!("shim {}", aot_cache::hash_bytes(WUKONG_RT_SHIM.as_bytes())),
            // The exact arguments `build_aot_runtime` passes.
            format!("recipe {}", aot_runtime_recipe(flavor).join(" ")),
        ];
        expected.extend(RUSTC_VV.lines().map(|l| format!("rustc {l}")));
        assert_lines(&format!("{flavor:?}"), &material, &expected);
    }
    // The two flavours' recipes really differ (fat LTO at opt-level 3 versus a plain rlib), so the
    // recipe line - not only the flavour's name - separates their entries.
    let lto = aot_runtime_recipe(aot_cache::Flavor::RustLto).join(" ");
    assert!(
        lto.contains("lto=fat") && lto.contains("opt-level=3"),
        "{lto}"
    );
    assert_ne!(
        aot_runtime_recipe(aot_cache::Flavor::RustNative).join(" "),
        lto
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn the_rustc_runtime_key_changes_with_each_input() {
    let dir = scratch("rustc-keys");
    let rlib = dir.join("libwukong_runtime.rlib");
    std::fs::write(&rlib, b"one").unwrap();
    stamp(&rlib, 1_700_000_000);
    let key = |flavor: aot_cache::Flavor, vv: &str, rlib: &Path| {
        aot_cache::key_of(flavor, &aot_runtime_material(flavor, vv, rlib).unwrap())
    };
    let native = aot_cache::Flavor::RustNative;
    let base = key(native, RUSTC_VV, &rlib);
    assert_eq!(
        base,
        key(native, RUSTC_VV, &rlib),
        "the key is a pure function of its inputs"
    );

    let mut changed: Vec<(&str, String)> = vec![
        (
            "the flavour (and its recipe)",
            key(aot_cache::Flavor::RustLto, RUSTC_VV, &rlib),
        ),
        (
            "the toolchain release",
            key(native, &RUSTC_VV.replace("1.97.1", "1.98.0"), &rlib),
        ),
        (
            "the toolchain commit",
            key(
                native,
                &RUSTC_VV.replace("8bab26f4f68e0e26", "0000000000000000"),
                &rlib,
            ),
        ),
        (
            "the toolchain host",
            key(
                native,
                &RUSTC_VV.replace(
                    "host: x86_64-pc-windows-msvc",
                    "host: x86_64-pc-windows-gnu",
                ),
                &rlib,
            ),
        ),
        (
            "the LLVM version",
            key(native, &RUSTC_VV.replace("22.1.6", "22.1.7"), &rlib),
        ),
    ];
    // The runtime rlib's identity: its modification time, its length, its path - each alone.
    stamp(&rlib, 1_700_000_001);
    changed.push(("the rlib's modification time", key(native, RUSTC_VV, &rlib)));
    std::fs::write(&rlib, b"three").unwrap();
    stamp(&rlib, 1_700_000_000);
    changed.push(("the rlib's length", key(native, RUSTC_VV, &rlib)));
    let elsewhere = dir.join("libwukong_runtime-feedface.rlib");
    std::fs::write(&elsewhere, b"one").unwrap();
    stamp(&elsewhere, 1_700_000_000);
    changed.push(("the rlib's path", key(native, RUSTC_VV, &elsewhere)));

    for (what, other) in &changed {
        assert_ne!(other, &base, "{what} did not change the key");
    }
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn the_cc_runtime_key_names_and_changes_with_every_input() {
    let source = c_runtime_source();
    let material = cc_runtime_material("gcc", CC_VERSION, &source);
    let recipe =
        cc_runtime_compile_args(Path::new(CC_RUNTIME_SOURCE), Path::new(CC_RUNTIME_OBJECT))
            .iter()
            .map(|a| a.to_string_lossy().into_owned())
            .collect::<Vec<_>>()
            .join(" ");
    // No FMA contraction is part of what the cached object means.
    assert!(recipe.contains("-ffp-contract=off"), "{recipe}");
    let mut expected = vec![
        "flavor cc".to_string(),
        format!("host {}", aot_cache::host_material()),
        "cc gcc".to_string(),
        // The exact text the build step writes and compiles.
        format!("source {}", aot_cache::hash_bytes(source.as_bytes())),
        format!("recipe {recipe}"),
    ];
    expected.extend(
        CC_VERSION
            .lines()
            .filter(|l| !l.trim().is_empty())
            .map(|l| format!("cc-version {l}")),
    );
    assert_lines("cc", &material, &expected);

    let key = |cc: &str, version: &str, source: &str| {
        aot_cache::key_of(
            aot_cache::Flavor::Cc,
            &cc_runtime_material(cc, version, source),
        )
    };
    let base = key("gcc", CC_VERSION, &source);
    assert_eq!(base, key("gcc", CC_VERSION, &source));
    let changed = [
        ("the compiler", key("clang", CC_VERSION, &source)),
        (
            "the compiler's version",
            key("gcc", &CC_VERSION.replace("14.2.0", "14.2.1"), &source),
        ),
        (
            "the C runtime source",
            key("gcc", CC_VERSION, &format!("{source}/* edited */\n")),
        ),
    ];
    for (what, other) in &changed {
        assert_ne!(other, &base, "{what} did not change the key");
    }
}
