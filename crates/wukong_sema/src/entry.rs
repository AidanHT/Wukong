//! The program entry's signature (E0331).
//!
//! `fn main` is called by the runtime, never by the program, so nothing else checked its shape.
//! The interpreter and the JIT accept an integer, float or void result, but a `main` with
//! parameters compiled cleanly and then failed at run time ("native entry point must take no
//! parameters" on the JIT, missing arguments in the interpreter), and an aggregate result (an array,
//! a struct, a tensor) had no exit-code meaning at all: the interpreter reported whatever its value
//! slot read as an integer while an executable could not be linked. Like Rust's E0580 and C's
//! "`main` must return `int`", the entry's signature is now checked where it is written: no
//! parameters, no generic parameters, and a scalar result (an integer, a float, `bool`, `char`) or
//! none. The process exit code is the low 8 bits of that value.

use wukong_ast::{ItemKind, Module};
use wukong_diag::Diagnostic;
use wukong_span::Interner;
use wukong_types::Ty;

use crate::{DefKind, DefMap};

/// Report E0331 for a root `fn main` whose signature cannot be a program entry. Imported modules'
/// items carry `<path>.<name>` symbols, so only the root file's `main` is named exactly `main`.
pub(crate) fn check(
    module: &Module,
    interner: &Interner,
    defs: &DefMap,
    diags: &mut Vec<Diagnostic>,
) {
    for item in &module.items {
        let ItemKind::Fn(f) = &item.kind else {
            continue;
        };
        if f.body.is_none() || interner.resolve(f.name.sym) != "main" {
            continue;
        }
        let Some(DefKind::Fn(sig)) = defs.lookup(f.name.sym).map(|d| &d.kind) else {
            continue;
        };
        if let Some(param) = f.params.first() {
            diags.push(
                Diagnostic::error("`main` cannot take parameters")
                    .with_code("E0331")
                    .primary(param.span, "the program entry is called with no arguments")
                    .help(
                        "read command-line arguments with `arg_count()` and `arg(i)` instead; \
                         `main` is declared `fn main() -> i32`",
                    ),
            );
        } else if !sig.generics.is_empty() || !f.generics.is_empty() {
            diags.push(
                Diagnostic::error("`main` cannot be generic")
                    .with_code("E0331")
                    .primary(
                        f.name.span,
                        "the program entry has nothing to instantiate it with",
                    )
                    .help("declare `fn main() -> i32`"),
            );
        }
        let scalar = matches!(sig.ret, Ty::Unit | Ty::Scalar(_) | Ty::Error | Ty::Unknown);
        if !scalar {
            let span = f.ret.as_ref().map_or(f.name.span, |t| t.span);
            diags.push(
                Diagnostic::error(format!(
                    "`main` cannot return `{}`",
                    sig.ret.display(interner)
                ))
                .with_code("E0331")
                .primary(span, "the exit code must come from a scalar")
                .help(
                    "return an integer (the process exit code is its low 8 bits), a float, \
                     `bool`, `char` or nothing; print or write an aggregate result instead",
                ),
            );
        }
    }
}
