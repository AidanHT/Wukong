//! Array patterns over a `[]T` slice, and `rest @ ..` in any array pattern.
//!
//! A fixed-size array `[T; N]` pattern needs no test of its own: sema checked the element count,
//! and each listed element is a constant-index place (`FnLowerer::slice_pattern_elems`). A slice's
//! length is known only at run time, so its array pattern is a **length test** first —
//! `len == listed` without `..`, `len >= listed` with one — and only then the listed elements'
//! own tests, at `data[i]` for the prefix and `data[len - suffix + j]` for the suffix. The element
//! tests are lowered in a block only the passing length test reaches
//! ([`FnLowerer::slice_view_pattern_cond`]): reading `data[0]` of an empty slice, whose data
//! pointer may be null, would fault natively and is a load past the end on the interpreter. For
//! the same reason a match binds a pattern's names only after its test passed (`lower_match`).
//!
//! `rest @ ..` binds the elements the `..` stands for, in place — never a copy of them:
//!
//! * on a slice, a fresh fat pointer `{ data + prefix, len - listed }` over the same storage (the
//!   sub-slice `s[prefix..len - suffix]` would build, without its bounds check, which the length
//!   test already implies);
//! * on an array `[T; N]`, the address of element `prefix` typed as the sub-array
//!   `[T; N - listed]` — the by-pointer convention every aggregate part of a pattern follows. In a
//!   `match` it is therefore a view of the scrutinee's elements, exactly as an aggregate element
//!   binding is; in a `let` the initializer was first copied into the `let`'s own slot, so `rest`
//!   views that copy and never sees a later write to the source array.

use wukong_ast::{self as ast, Pattern};
use wukong_mir::{BinOp, CmpOp, MirType, Op, ValueId};
use wukong_types::Ty;

use crate::{FnLowerer, SLICE_LEN_OFF, SLICE_PTR_OFF, SLICE_SIZE};

impl FnLowerer<'_> {
    /// The i1 test of the array pattern `[prefix.., (..), suffix..]` against the `[]T` slice whose
    /// fat pointer is at `scrut`, element type `elem`: the length test, then — in a block only a
    /// passing length test reaches — the listed elements' tests. The builder is left in the join
    /// block, whose parameter is the result.
    pub(crate) fn slice_view_pattern_cond(
        &mut self,
        prefix: &[Pattern],
        rest: bool,
        suffix: &[Pattern],
        scrut: ValueId,
        elem: &Ty,
    ) -> ValueId {
        let (data, len) = self.text_slice_parts(scrut);
        let listed = (prefix.len() + suffix.len()) as i128;
        let need = self
            .builder
            .build(MirType::I64, Op::ConstInt(listed, MirType::I64));
        let op = if rest { CmpOp::Sge } else { CmpOp::Eq };
        let len_ok = self.builder.build(MirType::I1, Op::Cmp(op, len, need));
        if !prefix.iter().chain(suffix).any(pattern_tests) {
            return len_ok;
        }
        let elems_bb = self.builder.new_block();
        let join = self.builder.new_block();
        let result = self.builder.block_param(join, MirType::I1);
        let no = self
            .builder
            .build(MirType::I1, Op::ConstInt(0, MirType::I1));
        self.builder
            .cond_br(len_ok, elems_bb, vec![], join, vec![no]);
        self.builder.switch_to(elems_bb);
        self.terminated = false;
        let mut acc: Option<ValueId> = None;
        for (eptr, sub) in self.slice_view_elems(data, len, prefix, suffix, elem) {
            if let Some(c) = self.field_pattern_cond(sub, eptr, elem) {
                acc = Some(match acc {
                    Some(a) => self.builder.build(MirType::I1, Op::Bin(BinOp::And, a, c)),
                    None => c,
                });
            }
        }
        let all = acc.unwrap_or_else(|| {
            self.builder
                .build(MirType::I1, Op::ConstInt(1, MirType::I1))
        });
        self.builder.br(join, vec![all]);
        self.builder.switch_to(join);
        self.terminated = false;
        result
    }

    /// Bind the names of the array pattern `[prefix.., (rest @ ..), suffix..]` matched against the
    /// `[]T` slice whose fat pointer is at `scrut`. Only called once the pattern's length test has
    /// passed (or on a pattern sema proved always matches), so every listed element is in bounds.
    pub(crate) fn bind_slice_view_pattern(
        &mut self,
        prefix: &[Pattern],
        rest_binding: Option<&Pattern>,
        suffix: &[Pattern],
        scrut: ValueId,
        elem: &Ty,
    ) {
        let binds_element = prefix.iter().chain(suffix).any(crate::pattern_has_binding);
        if !binds_element && rest_binding.is_none() {
            return;
        }
        let (data, len) = self.text_slice_parts(scrut);
        for (eptr, sub) in self.slice_view_elems(data, len, prefix, suffix, elem) {
            self.bind_field_pattern(sub, eptr, elem);
        }
        let Some(b) = rest_binding else {
            return;
        };
        let emir = self.mir_ty_of(elem);
        let start = self.gep_elem(data, &emir, prefix.len() as i128);
        let listed = self.builder.build(
            MirType::I64,
            Op::ConstInt((prefix.len() + suffix.len()) as i128, MirType::I64),
        );
        let n = self
            .builder
            .build(MirType::I64, Op::Bin(BinOp::Sub, len, listed));
        let fat = MirType::Array(Box::new(MirType::I8), SLICE_SIZE as u32);
        let sub = self.builder.alloca(fat.clone());
        let dp = self.field_ptr(sub, SLICE_PTR_OFF);
        self.builder.build_void(Op::Store {
            ptr: dp,
            value: start,
        });
        let lp = self.field_ptr(sub, SLICE_LEN_OFF);
        self.builder.build_void(Op::Store { ptr: lp, value: n });
        self.bind_match_ident(b, sub, &fat, &Ty::Slice(Box::new(elem.clone())));
    }

    /// Bind `rest @ ..` of an array pattern over the fixed-size array at `scrut` (type
    /// `scrut_ty`, `[T; N]`) whose prefix lists `prefix` elements and suffix `suffix`: the
    /// address of element `prefix`, typed `[T; N - prefix - suffix]`.
    pub(crate) fn bind_array_rest(
        &mut self,
        b: &Pattern,
        scrut: ValueId,
        scrut_ty: &Ty,
        prefix: usize,
        suffix: usize,
    ) {
        let Ty::Array { elem, len, .. } = scrut_ty else {
            return;
        };
        let rest_ty = Ty::Array {
            elem: elem.clone(),
            len: len.saturating_sub((prefix + suffix) as u64),
            shared: false,
        };
        let emir = self.mir_ty_of(elem);
        let at = self.gep_elem(scrut, &emir, prefix as i128);
        let rmir = self.mir_ty_of(&rest_ty);
        self.bind_match_ident(b, at, &rmir, &rest_ty);
    }

    /// Each listed element's place in a slice with data pointer `data` and length `len`, with its
    /// sub-pattern: the prefix at constant indices from 0, the suffix ending at `len - 1`.
    fn slice_view_elems<'p>(
        &mut self,
        data: ValueId,
        len: ValueId,
        prefix: &'p [Pattern],
        suffix: &'p [Pattern],
        elem: &Ty,
    ) -> Vec<(ValueId, &'p Pattern)> {
        let emir = self.mir_ty_of(elem);
        let mut out = Vec::with_capacity(prefix.len() + suffix.len());
        for (i, p) in prefix.iter().enumerate() {
            out.push((self.gep_elem(data, &emir, i as i128), p));
        }
        for (j, p) in suffix.iter().enumerate() {
            let back = self.builder.build(
                MirType::I64,
                Op::ConstInt((suffix.len() - j) as i128, MirType::I64),
            );
            let idx = self
                .builder
                .build(MirType::I64, Op::Bin(BinOp::Sub, len, back));
            out.push((self.gep_elem_value(data, &emir, idx), p));
        }
        out
    }
}

/// Whether `p` may test the value it matches — `false` only for the forms that match everything
/// (`_`, a name, `()`, `name @ _`, and the tuples and structs built only from those), so a slice
/// pattern of those needs no element block and a variant pattern of those no payload block
/// (`FnLowerer::pattern_cond`). Conservative: an array pattern always counts, since on a `[]T` it
/// is a length test whatever it lists. Wildcard-free over `PatKind`, so a new pattern form is a
/// compile error here rather than a test lowered where nothing guards it.
pub(crate) fn pattern_tests(p: &Pattern) -> bool {
    match &p.kind {
        ast::PatKind::Wildcard | ast::PatKind::Ident(_) | ast::PatKind::Unit => false,
        ast::PatKind::Binding { sub, .. } => pattern_tests(sub),
        ast::PatKind::Tuple(subs) => subs.iter().any(pattern_tests),
        ast::PatKind::Struct { fields, .. } => fields.iter().any(|f| pattern_tests(&f.pat)),
        ast::PatKind::Slice { .. }
        | ast::PatKind::Variant { .. }
        | ast::PatKind::Or(_)
        | ast::PatKind::Int { .. }
        | ast::PatKind::Char(_)
        | ast::PatKind::Bool(_)
        | ast::PatKind::Path(_)
        | ast::PatKind::Range { .. } => true,
    }
}

#[cfg(test)]
mod tests {
    use wukong_mir::{BasicBlock, CmpOp, Function, MirType, Op, Program, Terminator};
    use wukong_span::{Interner, SourceId};

    /// Parse, check and lower `src` (no optimization), which must be clean.
    fn lowered(src: &str) -> (Program, Interner) {
        let mut interner = Interner::new();
        let (module, pd) = wukong_parser::parse_module(src, SourceId(0), &mut interner);
        assert!(pd.is_empty(), "parse errors: {pd:?}");
        let (sema, sd) = wukong_sema::check(&module, &interner);
        assert!(sd.iter().all(|d| !d.is_error()), "sema errors: {sd:?}");
        let (program, diags) = crate::lower_program(&module, &sema, &mut interner);
        assert!(diags.iter().all(|d| !d.is_error()), "lowering: {diags:?}");
        (program, interner)
    }

    fn func<'p>(p: &'p Program, interner: &Interner, name: &str) -> &'p Function {
        p.funcs
            .iter()
            .find(|f| interner.resolve(f.name) == name)
            .unwrap_or_else(|| panic!("no function `{name}`"))
    }

    fn has(b: &BasicBlock, pred: impl Fn(&Op) -> bool) -> bool {
        b.insts.iter().any(|i| pred(&i.op))
    }

    fn loads_i32(b: &BasicBlock) -> bool {
        has(b, |o| matches!(o, Op::Load(_, MirType::I32)))
    }

    /// The block that compares the slice's length against `op`, which must end in the branch on
    /// that compare.
    fn length_test(f: &Function, op: CmpOp) -> &BasicBlock {
        let b = f
            .blocks
            .iter()
            .find(|b| has(b, |o| matches!(o, Op::Cmp(c, ..) if *c == op)))
            .unwrap_or_else(|| panic!("no length compare {op:?}"));
        assert!(matches!(b.term, Terminator::CondBr { .. }), "{:?}", b.term);
        b
    }

    /// `[0, ..]` must not read `data[0]` of an empty slice (whose data pointer may be null): the
    /// element is loaded and compared only in a block the passing length test reaches.
    #[test]
    fn element_tests_run_only_after_the_length_test() {
        let (p, i) = lowered(
            "fn f(s: []i32) -> i32 { return match s { [0, ..] => 1, _ => 2 }; }
             fn main() -> i32 { return 0; }",
        );
        let f = func(&p, &i, "f");
        let test = length_test(f, CmpOp::Sge);
        assert!(
            !loads_i32(test),
            "the element is read before the length is known"
        );
        assert!(
            f.blocks.iter().any(loads_i32),
            "the element is read somewhere"
        );
    }

    /// `[a, b]` binds its names only once `len == 2` held: binding first loaded `data[1]` of a
    /// one-element slice.
    #[test]
    fn names_bind_only_after_the_length_test() {
        let (p, i) = lowered(
            "fn f(s: []i32) -> i32 { return match s { [a, b] => a + b, _ => 0 }; }
             fn main() -> i32 { return 0; }",
        );
        let f = func(&p, &i, "f");
        let test = length_test(f, CmpOp::Eq);
        assert!(!loads_i32(test), "a name is bound before the length test");
        assert!(f.blocks.iter().any(loads_i32));
    }

    /// `rest @ ..` on a slice is a new fat pointer `{ data + 1, len - 1 }` over the same storage:
    /// no element is read or copied.
    #[test]
    fn a_rest_binding_is_a_view_not_a_copy() {
        let (p, i) = lowered(
            "fn f(s: []i32) -> i64 { return match s { [_, t @ ..] => t.len(), _ => 0 }; }
             fn main() -> i32 { return 0; }",
        );
        let f = func(&p, &i, "f");
        assert!(!f.blocks.iter().any(loads_i32), "an element was read");
        let subs = f
            .blocks
            .iter()
            .filter(|b| has(b, |o| matches!(o, Op::Bin(wukong_mir::BinOp::Sub, ..))))
            .count();
        assert_eq!(subs, 1, "one `len - listed`");
    }

    /// A `[]T` array pattern inside a variant's payload reads the fat pointer, its length and the
    /// element behind it only once the discriminant said the value IS that variant: for any other
    /// variant those payload bytes are its own fields (`V::P(4096, 3)` read as `data = 4096`,
    /// `len = 3`). The discriminant is tested in the entry block; nothing of the slice is.
    #[test]
    fn a_payload_slice_is_read_only_after_its_discriminant_matched() {
        for arm in [
            "V::L([7, ..]) => 1",
            "V::L([7, ..]) | V::P(0, _) => 1",
            "V::L([x, 7]) | V::P(_, x) => x as i32",
        ] {
            let (p, i) = lowered(&format!(
                "enum V {{ L([]i64), P(i64, i64) }}
                 fn f(v: V) -> i32 {{ return match v {{ {arm}, _ => 0 }}; }}
                 fn main() -> i32 {{ return 0; }}"
            ));
            let f = func(&p, &i, "f");
            let entry = &f.blocks[0];
            assert!(
                has(entry, |o| matches!(o, Op::Load(_, MirType::I32))),
                "{arm}: the discriminant is tested first"
            );
            assert!(
                !has(entry, |o| matches!(o, Op::Load(_, MirType::Ptr))),
                "{arm}: the payload's data pointer is read before the discriminant is known"
            );
            assert!(
                !has(entry, |o| matches!(o, Op::Load(_, MirType::I64))),
                "{arm}: a payload field is read before the discriminant is known"
            );
            assert!(f
                .blocks
                .iter()
                .any(|b| has(b, |o| matches!(o, Op::Load(_, MirType::Ptr)))));
        }
    }

    /// `rest @ ..` of an array is the address of its first unlisted element, typed as the
    /// sub-array: passing it on copies nothing either.
    #[test]
    fn an_array_rest_binding_is_an_address() {
        let (p, i) = lowered(
            "fn g(m: [i32; 2]) -> i32 { return m[0] * 10 + m[1]; }
             fn f(a: [i32; 4]) -> i32 { return match a { [_, mid @ .., _] => g(mid) }; }
             fn main() -> i32 { return 0; }",
        );
        let f = func(&p, &i, "f");
        assert!(!f.blocks.iter().any(loads_i32), "the sub-array was copied");
        assert!(f
            .blocks
            .iter()
            .any(|b| has(b, |o| matches!(o, Op::Gep { .. }))));
    }
}
