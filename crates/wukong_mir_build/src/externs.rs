//! `extern "C"` declarations -> [`wukong_mir::Program::externs`].
//!
//! Sema already registers every function of an `extern` block as an ordinary callable, so a call to
//! one lowers through the normal user-call path to an `Op::Call` naming the C symbol. What was
//! missing is the declaration itself: no MIR function carries that name, so the interpreter
//! reported an unknown function at run time and native codegen had no signature to import. This
//! pass turns each declaration into an [`ExternFn`] — the C-ABI view of its sema signature — which
//! every backend resolves the call through.
//!
//! Only C-ABI-representable types cross: integers (with their signedness, which the C ABI needs to
//! extend a narrow argument), `bool`, `f32`, `f64`, raw pointers and references (as pointers), and
//! `()` as a result. A slice, array, tuple, struct, enum, tensor, SIMD vector, `f16`/`bf16` or a
//! generic signature has no stable C mapping here and is a C0001 at the declaration — pass a
//! pointer instead.

use wukong_ast::{ItemKind, Module};
use wukong_diag::Diagnostic;
use wukong_mir::{ExternFn, ExternParam, MirType};
use wukong_sema::{DefKind, SemaResult};
use wukong_span::{Interner, Span};
use wukong_types::{Scalar, Ty};

/// Symbols the lowering itself emits for builtins (`print` of a string / an unsigned value lowers to
/// `print_str` / `print_u`, …). An `extern` of the same name would hijack those calls.
const RESERVED: &[&str] = &[
    "print",
    "println",
    "assert",
    "print_str",
    "println_str",
    "print_u",
    "println_u",
    "print_bool",
    "println_bool",
    "print_char",
    "println_char",
];

/// The C-ABI declarations of every `extern "C"` function in `module`, in declaration order and
/// unique by name. Unrepresentable declarations are reported into `diags` and left out.
pub(crate) fn lower_externs(
    module: &Module,
    sema: &SemaResult,
    interner: &Interner,
    diags: &mut Vec<Diagnostic>,
) -> Vec<ExternFn> {
    let mut out: Vec<ExternFn> = Vec::new();
    for item in &module.items {
        let ItemKind::Extern(blk) = &item.kind else {
            continue;
        };
        let abi = interner.resolve(blk.abi);
        if abi != "\"C\"" {
            diags.push(
                Diagnostic::error(format!("unsupported ABI {abi} on an `extern` block"))
                    .with_code("C0001")
                    .primary(item.span, "only `extern \"C\"` is supported")
                    .help("declare the functions in an `extern \"C\" { … }` block"),
            );
            continue;
        }
        for f in &blk.items {
            let name = interner.resolve(f.name.sym);
            // A repeated name is either an identical declaration the module loader merged or a
            // conflicting one sema already reported (E0300); either way the first one stands.
            if out.iter().any(|e| e.name == f.name.sym) {
                continue;
            }
            // The `wukong_` prefix names the runtime kernels the recognizers and the builtins lower
            // to (`wukong_sgemm`, `wukong_rt_alloc`, ...): an `extern` of one would redeclare it
            // under a second signature (native) or shadow the interpreter's kernel (oracle).
            if RESERVED.contains(&name) || name.starts_with("wukong_") {
                diags.push(
                    Diagnostic::error(format!(
                        "`{name}` is a Wukong builtin and cannot be declared `extern`"
                    ))
                    .with_code("E0300")
                    .primary(f.name.span, "reserved name")
                    .help("the builtin's lowering emits calls to this symbol itself"),
                );
                continue;
            }
            if !f.generics.is_empty() {
                diags.push(
                    Diagnostic::error(format!(
                        "`extern` function `{name}` is generic, and a generic signature has no C \
                         ABI"
                    ))
                    .with_code("C0001")
                    .primary(f.name.span, "declared with generic parameters"),
                );
                continue;
            }
            let Some(DefKind::Fn(sig)) = sema.defs.lookup(f.name.sym).map(|d| &d.kind) else {
                continue;
            };
            let mut ok = true;
            let mut params = Vec::with_capacity(sig.params.len());
            for (p, ty) in f.params.iter().zip(&sig.params) {
                match c_abi(ty, false) {
                    Some(cp) => params.push(cp),
                    None => {
                        ok = false;
                        diags.push(no_c_abi(name, ty, p.ty.span, interner));
                    }
                }
            }
            let ret = match c_abi(&sig.ret, true) {
                Some(cp) => cp,
                None => {
                    let span = f.ret.as_ref().map(|t| t.span).unwrap_or(f.name.span);
                    diags.push(no_c_abi(name, &sig.ret, span, interner));
                    continue;
                }
            };
            if ok {
                out.push(ExternFn {
                    name: f.name.sym,
                    params,
                    ret,
                });
            }
        }
    }
    out
}

/// The C-ABI view of a sema type, or `None` when it has no C mapping.
fn c_abi(ty: &Ty, is_ret: bool) -> Option<ExternParam> {
    let (ty, signed) = match ty {
        Ty::Unit if is_ret => (MirType::Void, false),
        // A half or an 8-bit float has no C type (C23's `_Float16` is not `bf16`, and the f8
        // formats have none at all).
        Ty::Scalar(Scalar::F16 | Scalar::Bf16 | Scalar::F8E4M3 | Scalar::F8E5M2) => return None,
        // C has no standard 128-bit integer, and the compilers that offer `__int128` disagree on
        // how it is passed (a register pair, or by reference on Windows).
        Ty::Scalar(Scalar::I128 | Scalar::U128) => return None,
        Ty::Scalar(s) => (MirType::from_scalar(*s), s.is_signed()),
        Ty::Ptr { .. } | Ty::Ref { .. } => (MirType::Ptr, false),
        _ => return None,
    };
    Some(ExternParam { ty, signed })
}

fn no_c_abi(name: &str, ty: &Ty, span: Span, interner: &Interner) -> Diagnostic {
    Diagnostic::error(format!(
        "`extern \"C\"` function `{name}` uses the type `{}`, which has no C ABI mapping",
        ty.display(interner)
    ))
    .with_code("C0001")
    .primary(span, "not C-ABI representable")
    .help(
        "C functions take and return integers of at most 64 bits, `bool`, `f32`, `f64` and \
         pointers; pass an aggregate, a buffer or a 128-bit integer by pointer (`*T`)",
    )
}
