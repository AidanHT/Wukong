//! The `@noalias` parameter promise, as lowering consumes it.
//!
//! Sema accepts `fn k(@noalias mut out: []f32, a: []f32)` and rejects the call sites and
//! reassignments that provably break the promise (`E0316`, `wukong_sema`'s `noalias` module).
//! Lowering records it on the MIR parameter ([`record_param`]) as [`wukong_mir::ParamAttrs`], where
//! `wukong_opt::alias`, the loop dependence test, the vectorizer's runtime alias guard and both
//! native backends read it. Those consumers reason about MIR *values* and track provenance, so they
//! handle every parameter spelling soundly — a slice whose fat pointer gets rewritten simply loses
//! its element provenance there. Nothing is recorded for a parameter without the attribute, so a
//! program that never writes `@noalias` lowers to byte-identical MIR.

use wukong_ast::Param;
use wukong_mir::{Builder, MirType, ParamAttrs, ValueId};
use wukong_span::Interner;
use wukong_types::Ty;

/// Record the `@noalias` promise of source parameter `p` (declared type `ty`, lowered to the entry
/// parameter `val`) on the function being built. A promise is only meaningful on a pointer the
/// caller supplies — a generic parameter instantiated at a scalar arrives as a copy and carries
/// nothing — and `slice` is recorded only alongside it, so an unannotated parameter adds no entry.
pub(crate) fn record_param(
    builder: &mut Builder,
    p: &Param,
    ty: &Ty,
    val: ValueId,
    interner: &Interner,
) {
    if !matches!(builder.value_type(val), MirType::Ptr)
        || !wukong_sema::is_noalias_param(p, interner)
    {
        return;
    }
    builder.set_param_attrs(
        val,
        ParamAttrs {
            noalias: true,
            slice: matches!(ty, Ty::Slice(_)),
        },
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use wukong_mir::Program;
    use wukong_span::SourceId;

    fn lower(src: &str) -> (Program, Interner) {
        let mut interner = Interner::new();
        let (module, pdiags) = wukong_parser::parse_module(src, SourceId(0), &mut interner);
        assert!(pdiags.is_empty(), "parse: {pdiags:?}");
        let (sema, sdiags) = wukong_sema::check(&module, &interner);
        assert!(sdiags.iter().all(|d| !d.is_error()), "sema: {sdiags:?}");
        let (prog, diags) = crate::lower_program(&module, &sema, &mut interner);
        assert!(diags.iter().all(|d| !d.is_error()), "lower: {diags:?}");
        (prog, interner)
    }

    /// The attribute lands on the MIR parameter it was written on (marking a slice as one), and a
    /// program that never writes it records nothing at all.
    #[test]
    fn the_promise_is_recorded_on_its_mir_parameter() {
        let src = "module m
fn k(@noalias mut o: []f32, x: []f32, @noalias mut w: [f32; 4], n: i64) {
    o[0] = x[0];
    w[0] = x[1];
}
fn plain(mut o: []f32, x: []f32) { o[0] = x[0]; }
fn main() -> i32 { return 0; }
";
        let (prog, mut it) = lower(src);
        let k = prog.function(it.intern("k")).expect("k");
        assert_eq!(
            k.param_attr(0),
            ParamAttrs {
                noalias: true,
                slice: true
            }
        );
        assert_eq!(k.param_attr(1), ParamAttrs::default());
        assert_eq!(
            k.param_attr(2),
            ParamAttrs {
                noalias: true,
                slice: false
            }
        );
        assert_eq!(k.param_attr(3), ParamAttrs::default());
        let plain = prog.function(it.intern("plain")).expect("plain");
        assert!(plain.param_attrs.is_empty());
        assert!(
            prog.funcs
                .iter()
                .filter(|f| f.name != k.name)
                .all(|f| f.param_attrs.is_empty()),
            "only the annotated function records anything"
        );
    }
}
