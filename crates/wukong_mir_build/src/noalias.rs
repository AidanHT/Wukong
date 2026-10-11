//! The `@noalias` parameter promise, as lowering consumes it.
//!
//! Sema accepts `fn k(@noalias mut out: []f32, a: []f32)` and rejects the call sites and
//! reassignments that provably break the promise (`E0316`, `wukong_sema`'s `noalias` module). This
//! module does the two things lowering owes it:
//!
//! * **Record it on the MIR parameter** ([`record_param`]) as [`wukong_mir::ParamAttrs`], where
//!   `wukong_opt::alias`, the loop dependence test, the vectorizer's runtime alias guard and both
//!   native backends read it. Those consumers reason about MIR *values* and track provenance, so
//!   they handle every parameter spelling soundly — a slice whose fat pointer gets rewritten simply
//!   loses its element provenance there. Without the attribute no promise is recorded: a parameter
//!   carries only the facts of its own type (`ParamAttrs::slice`, `ParamAttrs::extent`, recorded by
//!   `lower_fn`), which no consumer reads as a promise, and the printed MIR is unchanged.
//! * **Answer "are these two operands disjoint?"** for the parts of lowering that reason about
//!   operands by *name* rather than by MIR value — the kernel recognizers' disjointness guard
//!   (`FnLowerer::all_provably_disjoint`) and the `@parallel` region outliner's grouping of
//!   parameters that may be one buffer ([`FnLowerer::promised_disjoint`], [`could_share_storage`]).
//!
//! The name-level answer is deliberately narrower than the value-level one, because a name is not
//! a value. A `[]T` parameter is bound to the caller's 16-byte fat pointer, and that fat pointer can
//! be re-pointed after entry — by `b = a;` on a `mut` slice parameter, or by ANY callee that receives
//! the slice and reassigns its own `mut` parameter, since an aggregate parameter is passed by
//! reference and the reassignment is caller-visible. Neither is a broken promise in C's terms (the
//! new pointer is derived from the promised one), yet after it two parameter NAMES denote one
//! buffer. So a name is trusted only when it is bound to its storage itself and can never be
//! re-pointed: a fixed-size array or statically-shaped tensor parameter, which every assignment —
//! in this body or in a callee — copies into rather than redirects.

use wukong_ast::Param;
use wukong_mir::{Builder, MirType, ParamAttrs, ValueId};
use wukong_span::{Interner, Symbol};
use wukong_types::{Scalar, Ty};

use crate::FnLowerer;

/// Record the `@noalias` promise of source parameter `p` (declared type `ty`, lowered to the entry
/// parameter `val`) on the function being built. A promise is only meaningful on a pointer the
/// caller supplies — a generic parameter instantiated at a scalar arrives as a copy and carries
/// nothing. (`slice` is set here too, and `lower_fn` sets it again on every `[]T` parameter with or
/// without the promise: it is a fact of the type, which alias analysis reads only beside `noalias`.)
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
            extent: None,
        },
    );
}

/// Can a parameter of type `a` and one of type `b` be bound to overlapping storage by a call the
/// front end accepts? Conservatively yes unless both are buffers over *different* element scalars:
/// the declared length is no evidence at all (P1.20 — the front end once accepted a `[f32; 16]`
/// argument for a `[f32; 4]` parameter, and a slice and an array of one element type are two
/// spellings of the same storage), while Wukong has no reinterpreting view, so an `f32` buffer and
/// an `i32` buffer are never one object.
pub(crate) fn could_share_storage(a: &Ty, b: &Ty) -> bool {
    match (scalar_leaf(a), scalar_leaf(b)) {
        (Some(x), Some(y)) => x == y,
        _ => true,
    }
}

/// The element scalar a buffer type bottoms out in, or `None` for anything else (an aggregate, a
/// generic, an unknown type), which [`could_share_storage`] treats as "may share".
fn scalar_leaf(t: &Ty) -> Option<Scalar> {
    match t {
        Ty::Scalar(s) => Some(*s),
        Ty::Array { elem, .. } | Ty::Slice(elem) => scalar_leaf(elem),
        Ty::Ptr { pointee, .. } | Ty::Ref { pointee, .. } => scalar_leaf(pointee),
        // A dtype-generic tensor's `elem` is a placeholder: it may share with anything.
        Ty::Tensor {
            elem,
            elem_var: None,
            ..
        }
        | Ty::Vector { elem, .. } => Some(*elem),
        _ => None,
    }
}

impl FnLowerer<'_> {
    /// Are the operands named `a` and `b` two different parameters of the function being lowered,
    /// each bound to its own storage for good, with a `@noalias` promise between them?
    ///
    /// The per-pair proof the recognizers and the region outliner may use when the whole-program
    /// call-site fact (`GemmSyms::alias_free`) cannot help. `false` — no claim — for any name that is
    /// not such a parameter (a local, a view, a slice or pointer parameter, a struct-field operand,
    /// an `env` capture in an outlined body), so every lowerer that never sees a promise answers
    /// exactly as before. See the module note for why only storage-bound parameters qualify.
    pub(crate) fn promised_disjoint(&self, a: Symbol, b: Symbol) -> bool {
        let (Some((va, _)), Some((vb, _))) = (self.lookup(a), self.lookup(b)) else {
            return false;
        };
        if va == vb || !self.param_is_its_storage(va) || !self.param_is_its_storage(vb) {
            return false;
        }
        let (Some(pa), Some(pb)) = (
            self.builder.param_attrs_of(va),
            self.builder.param_attrs_of(vb),
        ) else {
            return false;
        };
        pa.noalias || pb.noalias
    }

    /// Is `v` an entry parameter bound to its storage itself — a fixed-size array or a
    /// statically-shaped tensor, whose binding no assignment anywhere can redirect? (A name only
    /// resolves to the parameter VALUE when lowering bound it directly; a parameter copied into a
    /// slot resolves to the slot and never gets here.) Also the `sdpa` builtin's static
    /// disjointness proof (`builtins.rs`), which needs the same "this name IS that storage" fact.
    pub(crate) fn param_is_its_storage(&self, v: ValueId) -> bool {
        self.param_arrays
            .iter()
            .find(|(p, _)| *p == v)
            .is_some_and(|(_, t)| matches!(t, Ty::Array { .. } | Ty::Tensor { .. }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use wukong_mir::{CastKind, Op, Program};
    use wukong_span::SourceId;

    fn arr(s: Scalar, len: u64) -> Ty {
        Ty::Array {
            elem: Box::new(Ty::Scalar(s)),
            len,
            shared: false,
        }
    }

    /// P1.20: the declared length says nothing about storage, and a slice and an array of one
    /// element are one buffer's two spellings; only a different element scalar rules sharing out.
    #[test]
    fn storage_sharing_ignores_length_and_spelling_but_not_the_element() {
        assert!(could_share_storage(
            &arr(Scalar::F32, 16),
            &arr(Scalar::F32, 4)
        ));
        assert!(could_share_storage(
            &arr(Scalar::F32, 16),
            &Ty::Slice(Box::new(Ty::Scalar(Scalar::F32)))
        ));
        assert!(!could_share_storage(
            &arr(Scalar::F32, 16),
            &arr(Scalar::I32, 16)
        ));
        // Anything that is not a plain scalar buffer may share.
        assert!(could_share_storage(&arr(Scalar::F32, 4), &Ty::Unknown));
    }

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

    fn calls(prog: &Program, interner: &Interner, callee: &str) -> usize {
        prog.funcs
            .iter()
            .flat_map(|f| f.blocks.iter())
            .flat_map(|b| b.insts.iter())
            .filter(|i| matches!(&i.op, Op::Call { func, .. } if interner.resolve(*func) == callee))
            .count()
    }

    /// The `ptrtoint` range ends of the run-time alias guards `kernel_guard.rs` builds in function
    /// `f` (`lo = ptrtoint(base)`, `hi = ptrtoint(gep(base, n))` per range-tested operand). The
    /// sources below take no pointer's address anywhere else, so a nonzero count IS a versioned
    /// dispatch and zero is an unguarded one.
    fn guard_range_ends(prog: &Program, interner: &Interner, f: &str) -> usize {
        prog.funcs
            .iter()
            .filter(|func| interner.resolve(func.name) == f)
            .flat_map(|func| func.blocks.iter())
            .flat_map(|b| b.insts.iter())
            .filter(|i| matches!(i.op, Op::Cast(CastKind::PtrToInt, _, _)))
            .count()
    }

    /// The attribute lands on the MIR parameter it was written on, and a program that never writes
    /// it records no promise. The facts of a parameter's own type are recorded either way: every
    /// `[]T` is marked a slice and every fixed-size buffer carries its extent, while a raw pointer
    /// records nothing (`0 as *T` is null, so nothing about its target is known).
    #[test]
    fn the_promise_is_recorded_on_its_mir_parameter() {
        let src = "module m
fn k(@noalias mut o: []f32, x: []f32, @noalias mut w: [f32; 4], n: i64) {
    o[0] = x[0];
    w[0] = x[1];
}
fn plain(mut o: []f32, x: []f32, q: *mut f32) { o[0] = x[0]; }
fn main() -> i32 { return 0; }
";
        let (prog, mut it) = lower(src);
        let k = prog.function(it.intern("k")).expect("k");
        let slice = ParamAttrs {
            noalias: false,
            slice: true,
            extent: None,
        };
        assert_eq!(
            k.param_attr(0),
            ParamAttrs {
                noalias: true,
                slice: true,
                extent: None,
            }
        );
        assert_eq!(k.param_attr(1), slice);
        assert_eq!(
            k.param_attr(2),
            ParamAttrs {
                noalias: true,
                slice: false,
                extent: Some(wukong_mir::ParamExtent {
                    elems: 4,
                    elem_bytes: 4,
                }),
            }
        );
        assert_eq!(k.param_attr(3), ParamAttrs::default());
        let plain = prog.function(it.intern("plain")).expect("plain");
        assert_eq!(plain.param_attr(0), slice);
        assert_eq!(plain.param_attr(1), slice);
        assert_eq!(plain.param_attr(2), ParamAttrs::default());
        assert!(
            prog.funcs
                .iter()
                .filter(|f| f.name != k.name)
                .all(|f| !f.has_noalias_params()),
            "only the annotated function records a promise"
        );
    }

    const MATMUL: &str = "module m
fn mm(a: [f32; 16], b: [f32; 16], CPARAM: [f32; 16]) {
    for i in 0..4 { for j in 0..4 {
        let mut s: f32 = 0.0;
        for k in 0..4 { s = s + a[i*4+k] * b[k*4+j]; }
        c[i*4+j] = s;
    } }
}
fn go(p: [f32; 16], q: [f32; 16], mut r: [f32; 16]) { mm(p, q, r); }
fn main() -> i32 {
    let mut x: [f32; 16] = [0.0; 16];
    let mut z: [f32; 16] = [0.0; 16];
    go(x, x, z);
    return 0;
}
";

    /// `go(x, x, z)` legitimately hands `mm` one buffer as both A and B (it squares `x`), so the
    /// whole-program call-site fact says `mm`'s parameters may alias, so the GEMM can only be
    /// dispatched behind a run-time alias test (`kernel_guard.rs`) whose may-overlap arm is the
    /// scalar nest. The GEMM needs only C apart from A and B — exactly what `@noalias` on `c`
    /// states — so with the promise both pairs are settled at compile time and the kernel
    /// dispatches with no guard at all.
    #[test]
    fn a_promise_lets_the_gemm_dispatch_where_the_call_site_fact_cannot() {
        let (prog, it) = lower(&MATMUL.replace("CPARAM", "mut c"));
        assert_eq!(calls(&prog, &it, "wukong_sgemm"), 1);
        assert!(
            guard_range_ends(&prog, &it, "mm") > 0,
            "without a promise the may-alias call site must keep the scalar nest behind a \
             run-time alias guard"
        );
        let (prog, it) = lower(&MATMUL.replace("CPARAM", "@noalias mut c"));
        assert_eq!(
            calls(&prog, &it, "wukong_sgemm"),
            1,
            "the promise on C is the disjointness the kernel needs"
        );
        assert_eq!(
            guard_range_ends(&prog, &it, "mm"),
            0,
            "the promise settles both C pairs at compile time: no run-time guard"
        );
    }

    const GATHER: &str = "module m
@generic
@parallel
fn gather(w: WTYPE, OUTPARAM: [f32; 16], ids: [i32; 8]) {
    let mut guard: i32 = 0;
    guard = guard + 1;
    for t in 0..8 {
        for d in 0..2 { out[t * 2 + d] = w[ids[t] * 2 + d]; }
    }
}
fn main() -> i32 {
    let mut ids: [i32; 8] = [0; 8];
    let mut w: WTYPE = [0.0; WLEN];
    let mut out: [f32; 16] = [0.0; 16];
    gather(w, out, ids);
    return 0;
}
";

    fn gather(wlen: u32, out_param: &str) -> String {
        GATHER
            .replace("WTYPE", &format!("[f32; {wlen}]"))
            .replace("WLEN", &wlen.to_string())
            .replace("OUTPARAM", out_param)
    }

    /// P1.20: the region outliner proves parameters the caller may bind to one buffer as ONE buffer.
    /// It used to decide "may" by `Ty` equality, so a `[f32; 32]` source and a `[f32; 16]`
    /// destination were two buffers by fiat and this gather — whose index `ids[t]` carries no region
    /// digit — was threaded. A declared length is no such evidence; the region must stay serial.
    #[test]
    fn region_parameters_of_different_lengths_are_still_one_possible_buffer() {
        let (prog, it) = lower(&gather(32, "mut out"));
        assert_eq!(calls(&prog, &it, "wukong_parallel_for"), 0);
    }

    /// The same gather with the destination promised apart from its source: nothing the region
    /// reads can be another iteration's write, so it may thread — the capability `@noalias` adds.
    /// (Without the promise the same-typed spelling declines:
    /// `a_region_over_maybe_aliasing_parameters_declines`.)
    #[test]
    fn a_promise_separates_a_region_destination_from_its_source() {
        let (prog, it) = lower(&gather(16, "mut out"));
        assert_eq!(calls(&prog, &it, "wukong_parallel_for"), 0);
        let (prog, it) = lower(&gather(16, "@noalias mut out"));
        assert_eq!(calls(&prog, &it, "wukong_parallel_for"), 1);
        let (prog, it) = lower(&gather(32, "@noalias mut out"));
        assert_eq!(calls(&prog, &it, "wukong_parallel_for"), 1);
    }

    /// Slices are never trusted by NAME: a callee that receives one and reassigns its `mut`
    /// parameter re-points the caller's fat pointer, so two slice parameter names can come to denote
    /// one buffer without either being written here. The GEMM over promised slices therefore still
    /// rests on the call-site fact — under a may-alias call site it keeps its run-time alias guard
    /// (and the value-level analysis keeps every slice fact it can).
    #[test]
    fn slice_parameters_are_not_trusted_by_name() {
        let src = MATMUL.replace(
            "a: [f32; 16], b: [f32; 16], CPARAM: [f32; 16]",
            "a: []f32, b: []f32, @noalias mut c: []f32",
        );
        let (prog, it) = lower(&src);
        assert_eq!(calls(&prog, &it, "wukong_sgemm"), 1);
        assert!(
            guard_range_ends(&prog, &it, "mm") > 0,
            "a promise on a slice NAME must not remove the run-time alias guard"
        );
        // Control: the same slice GEMM dispatches unguarded when no call site aliases anything, so
        // the guard above is the name-level refusal and not a slice GEMM that is always guarded.
        let disjoint = src.replace(
            "go(x, x, z);",
            "let mut y: [f32; 16] = [0.0; 16];\n    go(x, y, z);",
        );
        let (prog, it) = lower(&disjoint);
        assert_eq!(calls(&prog, &it, "wukong_sgemm"), 1);
        assert_eq!(guard_range_ends(&prog, &it, "mm"), 0);
    }
}
