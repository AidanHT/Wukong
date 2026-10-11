//! Lowering of the `str` operations that are not already slice operations.
//!
//! Sema publishes every `str` as the `[]u8` slice it is laid out as (`wukong_sema::strs`), so a
//! `str` reaches this crate as an ordinary 16-byte fat pointer `{ data @ 0, len @ 8 }`: indexing
//! (with its bounds check), `.len()`, `for b in s`, passing, returning and storing all go through
//! the existing slice machinery unchanged. Four things are new, and they are all here or call here:
//!
//! * **A string literal sema adapted to `str`** ([`FnLowerer::str_literal_value`]) is a fat pointer
//!   over the literal's `.rodata` bytes, its length the byte count without the NUL.
//! * **`==` / `!=` of two `str`s** ([`FnLowerer::lower_str_eq`]) compare the lengths, then the bytes
//!   in a block-parameter loop. Sema admits `==` on a slice only when it is a `str`.
//! * **A range index** `s[a..b]` (the parser's `$slice` / `$slice_from` call, on a `str` or any
//!   `[]T`) is a sub-slice ([`FnLowerer::lower_range_slice`]): checked `a <= b <= len`
//!   (`bounds.rs`), then `{ data + a, b - a }` in a fresh fat pointer — no element copy.
//! * **Printing** a `str` prints its text (`text.rs`, keyed on `SemaResult::str_print_args`).
//!
//! The annotation readers (`mir_ty_of_ann`, `annotation_storage_ty`, `let_binding_ty`) also map the
//! type name `str` to the slice storage; sema reserves the name, so no program type can shadow it.

use wukong_ast::{Expr, ExprKind};
use wukong_mir::{BinOp, CmpOp, MirType, Op, ValueId};
use wukong_span::Symbol;
use wukong_types::{Scalar, Ty};

use crate::{FnLowerer, SLICE_LEN_OFF, SLICE_PTR_OFF, SLICE_SIZE};

impl FnLowerer<'_> {
    /// Whether the annotation name `sym` is the built-in `str`.
    pub(crate) fn is_str_name(&self, sym: Symbol) -> bool {
        self.interner.resolve(sym) == "str"
    }

    /// The storage of a `str`: a `[]u8` fat pointer, as sema publishes it.
    pub(crate) fn str_ty() -> Ty {
        Ty::Slice(Box::new(Ty::Scalar(Scalar::U8)))
    }

    /// A string literal `e` whose bytes start at `data` (a `.rodata` address, or the frame-local
    /// fallback buffer) and number `len` (without the NUL): the `*u8` itself, or — when sema
    /// adapted the literal to `str` — a fresh fat pointer over those bytes.
    pub(crate) fn str_literal_value(&mut self, e: &Expr, data: ValueId, len: u64) -> ValueId {
        if !matches!(self.expr_ty(e), Ty::Slice(_)) {
            return data;
        }
        let sb = self
            .builder
            .alloca(MirType::Array(Box::new(MirType::I8), SLICE_SIZE as u32));
        self.store_slice_fat_ptr(sb, data, len);
        sb
    }

    /// `lhs == rhs` (or `!=` when `ne`) for two `str`s: equal lengths, then equal bytes.
    ///
    /// ```text
    ///     cond_br (llen == rlen), header(0), exit(differ)
    /// header(i):
    ///     cond_br (i <s llen), body, exit(same)
    /// body:
    ///     cond_br (l[i] == r[i]), header(i + 1), exit(differ)
    /// exit(result):
    /// ```
    pub(crate) fn lower_str_eq(&mut self, lhs: &Expr, rhs: &Expr, ne: bool) -> ValueId {
        let l = self.place_base_ptr(lhs);
        let r = self.place_base_ptr(rhs);
        let (ld, ll) = self.text_slice_parts(l);
        let (rd, rl) = self.text_slice_parts(r);
        let i1 = |b: bool| Op::ConstInt(i128::from(b), MirType::I1);
        let differ = self.builder.build(MirType::I1, i1(ne));
        let same = self.builder.build(MirType::I1, i1(!ne));
        let zero = self
            .builder
            .build(MirType::I64, Op::ConstInt(0, MirType::I64));
        let header = self.builder.new_block();
        let body = self.builder.new_block();
        let exit = self.builder.new_block();
        let i = self.builder.block_param(header, MirType::I64);
        let result = self.builder.block_param(exit, MirType::I1);
        let len_eq = self.builder.build(MirType::I1, Op::Cmp(CmpOp::Eq, ll, rl));
        self.builder
            .cond_br(len_eq, header, vec![zero], exit, vec![differ]);

        self.builder.switch_to(header);
        let more = self.builder.build(MirType::I1, Op::Cmp(CmpOp::Slt, i, ll));
        self.builder.cond_br(more, body, vec![], exit, vec![same]);

        self.builder.switch_to(body);
        let byte = |this: &mut Self, data: ValueId| {
            let p = this.builder.build(
                MirType::Ptr,
                Op::Gep {
                    ptr: data,
                    index: i,
                    elem: MirType::I8,
                },
            );
            this.builder.build(MirType::I8, Op::Load(p, MirType::I8))
        };
        let a = byte(self, ld);
        let b = byte(self, rd);
        let eq = self.builder.build(MirType::I1, Op::Cmp(CmpOp::Eq, a, b));
        let one = self
            .builder
            .build(MirType::I64, Op::ConstInt(1, MirType::I64));
        let next = self
            .builder
            .build(MirType::I64, Op::Bin(BinOp::Add, i, one));
        self.builder
            .cond_br(eq, header, vec![next], exit, vec![differ]);

        self.builder.switch_to(exit);
        self.terminated = false;
        result
    }

    /// A range index — the call `$slice(base, lo, hi)` / `$slice_from(base, lo)` the parser
    /// desugars `base[lo..hi]` / `base[lo..]` to — or `None` when `callee` is not one. `base` is a
    /// `str` or a `[]T` (sema rejects anything else); the result is a fresh fat pointer
    /// `{ data + lo, hi - lo }` viewing the same storage, after the run-time check
    /// `lo <= hi <= len` (`bounds.rs`). Evaluation order is base, `lo`, `hi`.
    pub(crate) fn lower_range_slice(
        &mut self,
        callee: &Expr,
        args: &[Expr],
        e: &Expr,
    ) -> Option<ValueId> {
        let ExprKind::Path(p) = &callee.kind else {
            return None;
        };
        if !p.is_single() || !wukong_sema::is_range_slice(self.interner.resolve(p.first().sym)) {
            return None;
        }
        let out = self
            .builder
            .alloca(MirType::Array(Box::new(MirType::I8), SLICE_SIZE as u32));
        let (Some(base), Some(lo)) = (args.first(), args.get(1)) else {
            self.unsupported(e.span, "a range index without a base and a lower bound");
            return Some(out);
        };
        let Ty::Slice(elem) = self.expr_ty(base) else {
            self.unsupported(base.span, "a range index of a value that is not a slice");
            return Some(out);
        };
        let elem = self.mir_ty_of(&elem);
        let sbuf = self.place_base_ptr(base);
        let (data, len) = self.text_slice_parts(sbuf);
        let lo = self.lower_index_i64(lo);
        let hi = match args.get(2) {
            Some(h) => self.lower_index_i64(h),
            None => len,
        };
        self.emit_range_bounds_check(lo, hi, len, e.span);
        let start = self.builder.build(
            MirType::Ptr,
            Op::Gep {
                ptr: data,
                index: lo,
                elem,
            },
        );
        let n = self
            .builder
            .build(MirType::I64, Op::Bin(BinOp::Sub, hi, lo));
        let dp = self.field_ptr(out, SLICE_PTR_OFF);
        self.builder.build_void(Op::Store {
            ptr: dp,
            value: start,
        });
        let lp = self.field_ptr(out, SLICE_LEN_OFF);
        self.builder.build_void(Op::Store { ptr: lp, value: n });
        Some(out)
    }
}

#[cfg(test)]
mod tests {
    use wukong_mir::{CmpOp, Function, MirType, Op, Program};
    use wukong_span::{Interner, SourceId};

    /// Parse, check and lower `src` (no optimization), which must be clean.
    fn lower(src: &str) -> (Program, Interner) {
        let mut interner = Interner::new();
        let (module, pd) = wukong_parser::parse_module(src, SourceId(0), &mut interner);
        assert!(pd.is_empty(), "parse errors: {pd:?}");
        let (sema, sd) = wukong_sema::check(&module, &interner);
        assert!(sd.iter().all(|d| !d.is_error()), "sema errors: {sd:?}");
        let (program, diags) = crate::lower_program(&module, &sema, &mut interner);
        assert!(diags.iter().all(|d| !d.is_error()), "lowering: {diags:?}");
        (program, interner)
    }

    /// Function `name` of `p`.
    fn func<'p>(p: &'p Program, interner: &Interner, name: &str) -> &'p Function {
        p.funcs
            .iter()
            .find(|f| interner.resolve(f.name) == name)
            .unwrap_or_else(|| panic!("no function `{name}`"))
    }

    fn ops(f: &Function) -> Vec<&Op> {
        f.blocks
            .iter()
            .flat_map(|b| b.insts.iter().map(|i| &i.op))
            .collect()
    }

    #[test]
    fn a_str_literal_is_a_rodata_fat_pointer() {
        let (p, i) = lower("fn main() -> i32 { let s: str = \"wukong\"; return s.len() as i32; }");
        let main_ops = ops(func(&p, &i, "main"));
        assert!(main_ops.iter().any(|o| matches!(o, Op::GlobalAddr(_))));
        // The length stored into the fat pointer is the byte count, without the NUL.
        assert!(main_ops
            .iter()
            .any(|o| matches!(o, Op::ConstInt(6, MirType::I64))));
        // An unannotated literal is still the bare `*u8`: no fat pointer is built for it.
        let (p, i) = lower("fn main() -> i32 { let s = \"wukong\"; print(s); return 0; }");
        let main_ops = ops(func(&p, &i, "main"));
        assert!(!main_ops
            .iter()
            .any(|o| matches!(o, Op::ConstInt(6, MirType::I64))));
    }

    #[test]
    fn str_equality_is_a_byte_loop() {
        let (p, i) = lower(
            "fn eq(a: str, b: str) -> bool { return a == b; } fn main() -> i32 { return 0; }",
        );
        let f = func(&p, &i, "eq");
        let ops = ops(f);
        let loads_i8 = ops
            .iter()
            .filter(|o| matches!(o, Op::Load(_, MirType::I8)))
            .count();
        assert_eq!(loads_i8, 2, "one byte from each side per iteration");
        assert!(ops.iter().any(|o| matches!(o, Op::Cmp(CmpOp::Eq, ..))));
        assert!(ops.iter().any(|o| matches!(o, Op::Cmp(CmpOp::Slt, ..))));
        assert!(f.blocks.len() >= 4, "entry, header, body, exit");
    }

    #[test]
    fn a_range_index_checks_both_bounds_unless_unchecked() {
        let src = |attr: &str| {
            format!(
                "{attr}fn mid(s: str, a: i64, b: i64) -> str {{ return s[a..b]; }}\n\
                 fn main() -> i32 {{ return 0; }}"
            )
        };
        let ule = |src: String| -> usize {
            let (p, i) = lower(&src);
            ops(func(&p, &i, "mid"))
                .into_iter()
                .filter(|o| matches!(o, Op::Cmp(CmpOp::Ule, ..)))
                .count()
        };
        assert_eq!(ule(src("")), 2, "hi <= len, then lo <= hi");
        assert_eq!(ule(src("@unchecked\n")), 0);
    }
}
