//! Records the toolchain cargo is building this crate with, so the `--emit=exe` AOT link can run
//! THAT `rustc` instead of whatever a bare name resolves to on the user's `PATH`.
//!
//! The rlib the link consumes (`libwukong_runtime.rlib`, located by
//! `wukong_driver::runtime_rlib_in`) was produced by cargo's `rustc`. Under rustup, the `rustc` on
//! `PATH` is a shim that picks a toolchain from the *current directory* - a `rust-toolchain.toml`
//! beside it or a `rustup override` on it - so running `wukongc --emit=exe` from outside this
//! workspace can hand the link a different compiler and earn
//! `E0514: found crate wukong_runtime compiled by an incompatible version of rustc`.
//!
//! Cargo sets `RUSTC` for build scripts to an absolute path to the toolchain binary itself, not to
//! the shim, so recording it side-steps directory-based resolution entirely. `wukong_driver`'s
//! `link_rustc()` reads it through `option_env!` and falls back to `PATH` when the recorded path no
//! longer exists (a moved or installed binary), which is the pre-existing behaviour.

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    // Without this, a `rustup update` (which replaces the binary at the same path) or a toolchain
    // switch would not be seen until something else invalidated the crate.
    println!("cargo:rerun-if-env-changed=RUSTC");
    if let Ok(rustc) = std::env::var("RUSTC") {
        // A `cargo:` directive is line-oriented, so a multi-line value would be misread as further
        // directives. Every other shape - spaces, drive letters, backslashes - passes through to
        // `option_env!` verbatim.
        if !rustc.is_empty() && rustc.lines().count() == 1 {
            println!("cargo:rustc-env=WUKONG_BUILD_RUSTC={rustc}");
        }
    }
}
