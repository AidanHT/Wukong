//! Lowering of the `str` operations that are not already slice operations.
//!
//! Sema publishes every `str` as the `[]u8` slice it is laid out as (`wukong_sema::strs`), so a
//! `str` reaches this crate as an ordinary 16-byte fat pointer `{ data @ 0, len @ 8 }`: indexing
//! (with its bounds check), `.len()`, `for b in s`, passing, returning and storing all go through
//! the existing slice machinery unchanged. What is new is all here or calls here:
//!
//! * **A string literal sema adapted to `str`** ([`FnLowerer::str_literal_value`]) is a fat pointer
//!   over the literal's `.rodata` bytes, its length the byte count without the NUL.
//! * **Comparisons of two `str`s**: `==` / `!=` ([`FnLowerer::lower_str_eq`]) compare the lengths,
//!   then the bytes in a block-parameter loop; `<` `<=` `>` `>=` and `.cmp()`
//!   ([`FnLowerer::lower_str_cmp`]) are the three-way loop. Bytes are compared zero-extended to
//!   `i32` ([`FnLowerer::load_byte`]). Sema admits a comparison of slices only for `str`s.
//! * **A range index** `s[a..b]` (the parser's `$slice` / `$slice_from` call, on a `str` or any
//!   `[]T`) is a sub-slice ([`FnLowerer::lower_range_slice`]): checked `a <= b <= len`
//!   (`bounds.rs`), then `{ data + a, b - a }` in a fresh fat pointer — no element copy.
//! * **The `str` methods and builtins** sema recorded in `SemaResult::str_info`
//!   ([`FnLowerer::lower_str_op`]): the zero-copy views `as_bytes` / `as_ptr`, `char_count` and the
//!   checked `str_from_utf8` (through `wukong_runtime::utf8`), and `str_concat`, which allocates
//!   from the `alloc_<T>` heap. `str_format` is lowered with the other formatting builtins
//!   (`text.rs`).
//! * **`for c in s.chars()`** ([`FnLowerer::lower_for_chars`]) and **string-literal patterns**
//!   ([`FnLowerer::str_pattern_cond`]).
//! * **Printing** a `str` prints its text (`text.rs`, keyed on `SemaResult::str_print_args`, and on
//!   `StrInfo::print_types` for a `str` inside an aggregate).
//!
//! The annotation readers (`mir_ty_of_ann`, `annotation_storage_ty`, `let_binding_ty`) also map the
//! type name `str` to the slice storage; sema reserves the name, so no program type can shadow it.

use wukong_ast::{self as ast, Block, Expr, ExprKind, Pattern};
use wukong_mir::{BinOp, CmpOp, MirType, Op, ValueId};
use wukong_sema::StrOp;
use wukong_span::{Interner, Symbol};
use wukong_types::{Scalar, Ty};

use crate::bounds::CHECK_UTF8;
use crate::{FnLowerer, SLICE_LEN_OFF, SLICE_PTR_OFF, SLICE_SIZE};

/// The runtime entries of the `str` operations (`wukong_runtime::utf8`), interned once at the top
/// of `lower_program` and carried on `GemmSyms` (`FnLowerer` holds a shared interner).
#[derive(Clone, Copy)]
pub(crate) struct StrSyms {
    /// `wukong_rt_utf8_check(data, len) -> i64`: `-1` when valid, else the packed fault.
    utf8_check: Symbol,
    /// `wukong_rt_utf8_decode(data, len, i) -> i64`: `code point | width << 32`.
    utf8_decode: Symbol,
    /// `wukong_rt_utf8_count(data, len) -> i64`.
    utf8_count: Symbol,
}

impl StrSyms {
    pub(crate) fn new(interner: &mut Interner) -> StrSyms {
        StrSyms {
            utf8_check: interner.intern("wukong_rt_utf8_check"),
            utf8_decode: interner.intern("wukong_rt_utf8_decode"),
            utf8_count: interner.intern("wukong_rt_utf8_count"),
        }
    }
}

/// The low 56 bits of a packed UTF-8 fault: its `valid_up_to` (`wukong_runtime::utf8::pack_fault`).
const FAULT_INDEX_MASK: i128 = (1 << 56) - 1;

/// Collect the raw symbol of every string-literal pattern inside `p`, for the `.rodata` table
/// (`collect_static_strings`): a `match` arm `"add" =>` compares against the literal's static
/// bytes, shared with every equal literal in the program.
pub(crate) fn collect_str_pat(p: &Pattern, out: &mut Vec<Symbol>) {
    match &p.kind {
        ast::PatKind::Str(s) => out.push(*s),
        ast::PatKind::Tuple(subs) | ast::PatKind::Or(subs) => {
            subs.iter().for_each(|s| collect_str_pat(s, out))
        }
        ast::PatKind::Binding { sub, .. } => collect_str_pat(sub, out),
        ast::PatKind::Struct { fields, .. } => {
            fields.iter().for_each(|f| collect_str_pat(&f.pat, out))
        }
        ast::PatKind::Variant { fields, .. } => match fields {
            ast::VariantPat::Tuple(subs) => subs.iter().for_each(|s| collect_str_pat(s, out)),
            ast::VariantPat::Struct(fps) => fps.iter().for_each(|f| collect_str_pat(&f.pat, out)),
        },
        ast::PatKind::Slice { prefix, suffix, .. } => prefix
            .iter()
            .chain(suffix)
            .for_each(|s| collect_str_pat(s, out)),
        ast::PatKind::Wildcard
        | ast::PatKind::Ident(_)
        | ast::PatKind::Unit
        | ast::PatKind::Int { .. }
        | ast::PatKind::Char(_)
        | ast::PatKind::Bool(_)
        | ast::PatKind::Path(_)
        | ast::PatKind::Range { .. } => {}
    }
}

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
        self.str_eq_parts((ld, ll), (rd, rl), ne)
    }

    /// [`Self::lower_str_eq`] over already-loaded `(data, len)` pairs.
    fn str_eq_parts(
        &mut self,
        (ld, ll): (ValueId, ValueId),
        (rd, rl): (ValueId, ValueId),
        ne: bool,
    ) -> ValueId {
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
        let a = self.load_byte(ld, i);
        let b = self.load_byte(rd, i);
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

    /// The byte `data[i]` (`i` an `i64`), zero-extended to `i32`. Every byte comparison goes
    /// through this: the widened value is the byte's unsigned value on every backend, whatever
    /// representation the interpreter's memory slot holds it in.
    fn load_byte(&mut self, data: ValueId, i: ValueId) -> ValueId {
        let p = self.builder.build(
            MirType::Ptr,
            Op::Gep {
                ptr: data,
                index: i,
                elem: MirType::I8,
            },
        );
        let b = self.builder.build(MirType::I8, Op::Load(p, MirType::I8));
        self.coerce_to(b, &MirType::I8, &MirType::I32, false)
    }

    fn i64_const(&mut self, v: i128) -> ValueId {
        self.builder
            .build(MirType::I64, Op::ConstInt(v, MirType::I64))
    }

    fn i32_const(&mut self, v: i128) -> ValueId {
        self.builder
            .build(MirType::I32, Op::ConstInt(v, MirType::I32))
    }

    /// A fresh 16-byte fat pointer `{ data, len }` — a `str` or a `[]u8` — whose fields are run-time
    /// values.
    pub(crate) fn fat_ptr(&mut self, data: ValueId, len: ValueId) -> ValueId {
        let out = self
            .builder
            .alloca(MirType::Array(Box::new(MirType::I8), SLICE_SIZE as u32));
        let dp = self.field_ptr(out, SLICE_PTR_OFF);
        self.builder.build_void(Op::Store {
            ptr: dp,
            value: data,
        });
        let lp = self.field_ptr(out, SLICE_LEN_OFF);
        self.builder.build_void(Op::Store {
            ptr: lp,
            value: len,
        });
        out
    }

    /// The bytes of the string literal whose raw source text is `sym`, without the NUL: its
    /// `.rodata` blob's address (every equal literal shares one), or — for a lowerer without the
    /// table — a frame-local copy; and its byte length.
    pub(crate) fn str_literal_data(&mut self, sym: Symbol) -> (ValueId, u64) {
        let mut key = crate::decode_string_literal(self.interner.resolve(sym));
        key.push(0);
        let data = if let Some(sym) = self.mono.and_then(|m| m.strings.get(&key)).copied() {
            self.builder.build(MirType::Ptr, Op::GlobalAddr(sym))
        } else {
            let bytes = &key[..key.len() - 1];
            let elem = MirType::I8;
            let n = key.len() as u32; // bytes + NUL terminator
            let base = self
                .builder
                .alloca(MirType::Array(Box::new(elem.clone()), n));
            for (i, b) in bytes.iter().enumerate() {
                let v = self
                    .builder
                    .build(MirType::I8, Op::ConstInt(*b as i128, MirType::I8));
                self.store_element(base, &elem, i as i128, v);
            }
            let nul = self
                .builder
                .build(MirType::I8, Op::ConstInt(0, MirType::I8));
            self.store_element(base, &elem, bytes.len() as i128, nul);
            base
        };
        (data, key.len() as u64 - 1)
    }

    /// The test of a string-literal pattern `"…"` (raw source text `sym`) against the `str` whose
    /// fat pointer is at `scrut`: [`Self::lower_str_eq`]'s length-then-bytes loop against the
    /// literal's static bytes. The decoder is the one sema's usefulness analysis keys the literal's
    /// constructor by (`wukong_sema::text::decode_string_literal`), so both agree on which literals
    /// are the same pattern.
    pub(crate) fn str_pattern_cond(&mut self, sym: Symbol, scrut: ValueId) -> ValueId {
        let parts = self.text_slice_parts(scrut);
        let (ld, n) = self.str_literal_data(sym);
        let ll = self.i64_const(i128::from(n));
        self.str_eq_parts(parts, (ld, ll), false)
    }

    /// The `str` operation sema recorded for `callee`, if any.
    fn str_op(&self, callee: &Expr) -> Option<StrOp> {
        self.sema.str_info.ops.get(&callee.id).copied()
    }

    /// Lower a call sema typed as a `str` operation (`wukong_sema::StrOp`: a `str` method or a
    /// `str` builtin), or return `None` for any other callee. `e` is the call.
    pub(crate) fn lower_str_op(
        &mut self,
        callee: &Expr,
        args: &[Expr],
        e: &Expr,
    ) -> Option<ValueId> {
        let op = self.str_op(callee)?;
        let recv = match &callee.kind {
            ExprKind::Field { base, .. } => Some(&**base),
            _ => None,
        };
        Some(match op {
            // The same bytes, in a fresh fat pointer: no copy of the text.
            StrOp::AsBytes | StrOp::Chars => {
                let s = self.place_base_ptr(recv?);
                let (data, len) = self.text_slice_parts(s);
                self.fat_ptr(data, len)
            }
            StrOp::AsPtr => {
                let s = self.place_base_ptr(recv?);
                self.text_slice_parts(s).0
            }
            StrOp::CharCount => {
                let s = self.place_base_ptr(recv?);
                let (data, len) = self.text_slice_parts(s);
                let func = self.gemm.str_syms.utf8_count;
                self.builder.build(
                    MirType::I64,
                    Op::Call {
                        func,
                        args: vec![data, len],
                    },
                )
            }
            StrOp::Cmp => self.lower_str_cmp(recv?, args.first()?),
            // Checked, then a view of the same bytes. The check is the conversion's contract, not
            // a bounds check, so `@unchecked` keeps it.
            StrOp::FromUtf8 => {
                let (data, len) = self.text_u8_buffer(args.first()?)?;
                let fault = self.utf8_fault(data, len);
                let valid = self.i64_const(-1);
                let ok = self
                    .builder
                    .build(MirType::I1, Op::Cmp(CmpOp::Eq, fault, valid));
                self.emit_check_fail_unless(ok, CHECK_UTF8, fault, len, e.span);
                self.fat_ptr(data, len)
            }
            StrOp::ValidUpTo => {
                let (data, len) = self.text_u8_buffer(args.first()?)?;
                let fault = self.utf8_fault(data, len);
                let zero = self.i64_const(0);
                let valid = self
                    .builder
                    .build(MirType::I1, Op::Cmp(CmpOp::Slt, fault, zero));
                let mask = self.i64_const(FAULT_INDEX_MASK);
                let at = self
                    .builder
                    .build(MirType::I64, Op::Bin(BinOp::And, fault, mask));
                self.builder.build(MirType::I64, Op::Select(valid, len, at))
            }
            StrOp::Concat => self.lower_str_concat(args),
        })
    }

    /// `wukong_rt_utf8_check(data, len)`: `-1`, or the packed first fault.
    fn utf8_fault(&mut self, data: ValueId, len: ValueId) -> ValueId {
        let func = self.gemm.str_syms.utf8_check;
        self.builder.build(
            MirType::I64,
            Op::Call {
                func,
                args: vec![data, len],
            },
        )
    }

    /// `a.cmp(b)`: `-1`, `0` or `1` as an `i32`, comparing the bytes lexicographically (see
    /// [`Self::str_cmp_parts`]). Evaluation order is `a`, then `b`.
    pub(crate) fn lower_str_cmp(&mut self, lhs: &Expr, rhs: &Expr) -> ValueId {
        let l = self.place_base_ptr(lhs);
        let r = self.place_base_ptr(rhs);
        let lp = self.text_slice_parts(l);
        let rp = self.text_slice_parts(r);
        self.str_cmp_parts(lp, rp)
    }

    /// `<`, `<=`, `>`, `>=` of two `str`s: their [`Self::lower_str_cmp`] against zero.
    pub(crate) fn lower_str_order(&mut self, pred: CmpOp, lhs: &Expr, rhs: &Expr) -> ValueId {
        let c = self.lower_str_cmp(lhs, rhs);
        let zero = self.i32_const(0);
        self.builder.build(MirType::I1, Op::Cmp(pred, c, zero))
    }

    /// The three-way comparison of two `(data, len)` byte strings: the first differing byte
    /// decides (as unsigned values), and when one is a prefix of the other the shorter is less —
    /// `memcmp` over the common length, then the lengths, which is Rust's `Ord for str`.
    ///
    /// ```text
    ///     n = min(llen, rlen); br header(0)
    /// header(i):  cond_br (i <s n), body, tail
    /// body:       a = l[i], b = r[i]; cond_br (a == b), header(i + 1), differ
    /// differ:     br exit(a <u b ? -1 : 1)
    /// tail:       br exit(llen <s rlen ? -1 : llen == rlen ? 0 : 1)
    /// exit(result: i32):
    /// ```
    fn str_cmp_parts(
        &mut self,
        (ld, ll): (ValueId, ValueId),
        (rd, rl): (ValueId, ValueId),
    ) -> ValueId {
        let shorter = self.builder.build(MirType::I1, Op::Cmp(CmpOp::Slt, ll, rl));
        let n = self
            .builder
            .build(MirType::I64, Op::Select(shorter, ll, rl));
        let zero = self.i64_const(0);
        let minus = self.i32_const(-1);
        let equal = self.i32_const(0);
        let plus = self.i32_const(1);
        let header = self.builder.new_block();
        let body = self.builder.new_block();
        let differ = self.builder.new_block();
        let tail = self.builder.new_block();
        let exit = self.builder.new_block();
        let i = self.builder.block_param(header, MirType::I64);
        let result = self.builder.block_param(exit, MirType::I32);
        self.builder.br(header, vec![zero]);

        self.builder.switch_to(header);
        let more = self.builder.build(MirType::I1, Op::Cmp(CmpOp::Slt, i, n));
        self.builder.cond_br(more, body, vec![], tail, vec![]);

        self.builder.switch_to(body);
        let a = self.load_byte(ld, i);
        let b = self.load_byte(rd, i);
        let eq = self.builder.build(MirType::I1, Op::Cmp(CmpOp::Eq, a, b));
        let one = self.i64_const(1);
        let next = self
            .builder
            .build(MirType::I64, Op::Bin(BinOp::Add, i, one));
        self.builder.cond_br(eq, header, vec![next], differ, vec![]);

        self.builder.switch_to(differ);
        let lt = self.builder.build(MirType::I1, Op::Cmp(CmpOp::Slt, a, b));
        let v = self
            .builder
            .build(MirType::I32, Op::Select(lt, minus, plus));
        self.builder.br(exit, vec![v]);

        self.builder.switch_to(tail);
        let same = self.builder.build(MirType::I1, Op::Cmp(CmpOp::Eq, ll, rl));
        let ge = self
            .builder
            .build(MirType::I32, Op::Select(same, equal, plus));
        let v = self
            .builder
            .build(MirType::I32, Op::Select(shorter, minus, ge));
        self.builder.br(exit, vec![v]);

        self.builder.switch_to(exit);
        self.terminated = false;
        result
    }

    /// `str_concat(a, b, …)`: every operand evaluated once, in order; then one zeroed heap
    /// allocation of the total length plus a NUL (`wukong_rt_alloc`, released by `free`), each
    /// operand's bytes copied in after the previous one's, and a `str` over the total length. The
    /// NUL makes the result's `as_ptr()` a C string too, and keeps a zero-length result a non-null
    /// allocation on every backend.
    fn lower_str_concat(&mut self, args: &[Expr]) -> ValueId {
        let mut parts = Vec::with_capacity(args.len());
        for a in args {
            let s = self.place_base_ptr(a);
            parts.push(self.text_slice_parts(s));
        }
        let mut total = self.i64_const(0);
        for &(_, n) in &parts {
            total = self
                .builder
                .build(MirType::I64, Op::Bin(BinOp::Add, total, n));
        }
        let data = self.heap_bytes(total);
        let mut off = self.i64_const(0);
        for (src, n) in parts {
            self.emit_byte_copy(data, off, src, n);
            off = self
                .builder
                .build(MirType::I64, Op::Bin(BinOp::Add, off, n));
        }
        self.fat_ptr(data, total)
    }

    /// A zeroed heap allocation of `len + 1` bytes (`wukong_rt_alloc`): room for `len` bytes of
    /// text and a NUL after them.
    pub(crate) fn heap_bytes(&mut self, len: ValueId) -> ValueId {
        let one = self.i64_const(1);
        let cap = self
            .builder
            .build(MirType::I64, Op::Bin(BinOp::Add, len, one));
        let zero = self.i64_const(0);
        let func = self.gemm.rt_alloc;
        self.builder.build(
            MirType::Ptr,
            Op::Call {
                func,
                args: vec![cap, one, zero],
            },
        )
    }

    /// Copy `n` bytes from `src` to `dst + off` (`n`, `off` `i64`s), one byte per iteration.
    fn emit_byte_copy(&mut self, dst: ValueId, off: ValueId, src: ValueId, n: ValueId) {
        let zero = self.i64_const(0);
        let header = self.builder.new_block();
        let body = self.builder.new_block();
        let exit = self.builder.new_block();
        let k = self.builder.block_param(header, MirType::I64);
        self.builder.br(header, vec![zero]);

        self.builder.switch_to(header);
        let more = self.builder.build(MirType::I1, Op::Cmp(CmpOp::Slt, k, n));
        self.builder.cond_br(more, body, vec![], exit, vec![]);

        self.builder.switch_to(body);
        let sp = self.builder.build(
            MirType::Ptr,
            Op::Gep {
                ptr: src,
                index: k,
                elem: MirType::I8,
            },
        );
        let b = self.builder.build(MirType::I8, Op::Load(sp, MirType::I8));
        let at = self
            .builder
            .build(MirType::I64, Op::Bin(BinOp::Add, off, k));
        let dp = self.builder.build(
            MirType::Ptr,
            Op::Gep {
                ptr: dst,
                index: at,
                elem: MirType::I8,
            },
        );
        self.builder.build_void(Op::Store { ptr: dp, value: b });
        let one = self.i64_const(1);
        let next = self
            .builder
            .build(MirType::I64, Op::Bin(BinOp::Add, k, one));
        self.builder.br(header, vec![next]);

        self.builder.switch_to(exit);
        self.terminated = false;
    }

    /// `for c in s.chars() { body }` (`iter` is the `s.chars()` call sema recorded), or `false`
    /// when `iter` is not one. The text is decoded in place, one character per iteration:
    ///
    /// ```text
    ///     data, len = s; i = 0
    /// header:  cond_br (i <s len), body, exit
    /// body:    b = data[i]; cond_br (b <u 0x80), join(b, 1), slow
    /// slow:    p = wukong_rt_utf8_decode(data, len, i); br join(p as i32, p >> 32)
    /// join(c, w): next = i + w; c is the loop variable; <body>; br latch
    /// latch:   i = next; br header
    /// ```
    ///
    /// ASCII never leaves the loop; any other character is one runtime call, which decodes an
    /// ill-formed sequence as U+FFFD (`wukong_runtime::utf8::decode_at`) — the bytes of a `str`
    /// can be rewritten through `as_bytes()` or the buffer `str_from_utf8` viewed, so the loop
    /// must never assume they are still valid. `len` and `data` are read once, before the loop.
    pub(crate) fn lower_for_chars(
        &mut self,
        label: Option<Symbol>,
        pat: &Pattern,
        iter: &Expr,
        body: &Block,
    ) -> bool {
        let ExprKind::Call { callee, .. } = &iter.kind else {
            return false;
        };
        if self.str_op(callee) != Some(StrOp::Chars) {
            return false;
        }
        let ExprKind::Field { base, .. } = &callee.kind else {
            return false;
        };
        let name = match &pat.kind {
            ast::PatKind::Ident(n) => Some(*n),
            ast::PatKind::Wildcard => None,
            _ => return false,
        };
        let char_ty = self.mir_ty_of(&Ty::Scalar(Scalar::Char));
        let s = self.place_base_ptr(base);
        let (data, len) = self.text_slice_parts(s);
        let slot = self.builder.alloca(MirType::I64);
        let next_slot = self.builder.alloca(MirType::I64);
        let zero = self.i64_const(0);
        self.builder.build_void(Op::Store {
            ptr: slot,
            value: zero,
        });

        self.push_scope();
        let header = self.builder.new_block();
        let body_bb = self.builder.new_block();
        let slow = self.builder.new_block();
        let join = self.builder.new_block();
        let latch = self.builder.new_block();
        let exit = self.builder.new_block();
        let cp = self.builder.block_param(join, char_ty.clone());
        let width = self.builder.block_param(join, MirType::I64);
        self.builder.br(header, vec![]);

        self.builder.switch_to(header);
        self.terminated = false;
        let i = self
            .builder
            .build(MirType::I64, Op::Load(slot, MirType::I64));
        let more = self.builder.build(MirType::I1, Op::Cmp(CmpOp::Slt, i, len));
        self.builder.cond_br(more, body_bb, vec![], exit, vec![]);

        self.builder.switch_to(body_bb);
        let b = self.load_byte(data, i);
        let ascii_end = self.i32_const(0x80);
        let ascii = self
            .builder
            .build(MirType::I1, Op::Cmp(CmpOp::Slt, b, ascii_end));
        let b = self.coerce_to(b, &MirType::I32, &char_ty, false);
        let one = self.i64_const(1);
        self.builder
            .cond_br(ascii, join, vec![b, one], slow, vec![]);

        self.builder.switch_to(slow);
        let func = self.gemm.str_syms.utf8_decode;
        let packed = self.builder.build(
            MirType::I64,
            Op::Call {
                func,
                args: vec![data, len, i],
            },
        );
        let c = self.coerce_to(packed, &MirType::I64, &char_ty, false);
        let shift = self.i64_const(32);
        let w = self
            .builder
            .build(MirType::I64, Op::Bin(BinOp::LShr, packed, shift));
        self.builder.br(join, vec![c, w]);

        self.builder.switch_to(join);
        let next = self
            .builder
            .build(MirType::I64, Op::Bin(BinOp::Add, i, width));
        self.builder.build_void(Op::Store {
            ptr: next_slot,
            value: next,
        });
        if let Some(name) = name {
            let xslot = self.builder.alloca(char_ty.clone());
            self.builder.build_void(Op::Store {
                ptr: xslot,
                value: cp,
            });
            self.bind(name, xslot, char_ty);
        }
        self.loops.push((label, latch, exit));
        self.lower_block(body);
        self.loops.pop();
        if !self.terminated {
            self.builder.br(latch, vec![]);
        }

        // latch: i = next (reached by fall-through and by every `continue`).
        self.builder.switch_to(latch);
        self.terminated = false;
        let n = self
            .builder
            .build(MirType::I64, Op::Load(next_slot, MirType::I64));
        self.builder.build_void(Op::Store {
            ptr: slot,
            value: n,
        });
        self.builder.br(header, vec![]);

        self.pop_scope();
        self.builder.switch_to(exit);
        self.terminated = false;
        true
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

    /// The callees of every `Op::Call` in `f`, by name, in instruction order.
    fn callees(f: &Function, i: &Interner) -> Vec<String> {
        ops(f)
            .into_iter()
            .filter_map(|o| match o {
                Op::Call { func, .. } => Some(i.resolve(*func).to_string()),
                _ => None,
            })
            .collect()
    }

    /// A range index fails through `wukong_rt_check_fail` with its own kinds — the end against
    /// the length first, then the start against the end — never the element-access entry.
    #[test]
    fn a_range_index_fails_through_the_check_entry() {
        let (p, i) = lower(
            "fn mid(s: str, a: i64, b: i64) -> str { return s[a..b]; }\n\
             fn main() -> i32 { return 0; }",
        );
        let f = func(&p, &i, "mid");
        assert_eq!(
            callees(f, &i),
            ["wukong_rt_check_fail", "wukong_rt_check_fail"]
        );
        let kinds: Vec<i128> =
            f.blocks
                .iter()
                .flat_map(|b| b.insts.iter())
                .filter_map(|inst| match &inst.op {
                    Op::Call { args, .. } => args.first().copied(),
                    _ => None,
                })
                .filter_map(|k| {
                    f.blocks.iter().flat_map(|b| b.insts.iter()).find_map(|d| {
                        match (d.result, &d.op) {
                            (Some(r), Op::ConstInt(v, MirType::I64)) if r == k => Some(*v),
                            _ => None,
                        }
                    })
                })
                .collect();
        assert_eq!(
            kinds,
            [
                i128::from(crate::bounds::CHECK_RANGE_END),
                i128::from(crate::bounds::CHECK_RANGE_ORDER)
            ]
        );
    }

    /// `str_from_utf8`'s check is the conversion's contract, not a bounds check: it stays under
    /// `@unchecked`, which removes the range index's.
    #[test]
    fn str_from_utf8_is_checked_even_unchecked() {
        for attr in ["", "@unchecked\n"] {
            let (p, i) = lower(&format!(
                "{attr}fn conv(b: []u8) -> str {{ return str_from_utf8(b); }}\n\
                 fn main() -> i32 {{ return 0; }}"
            ));
            assert_eq!(
                callees(func(&p, &i, "conv"), &i),
                ["wukong_rt_utf8_check", "wukong_rt_check_fail"],
                "{attr:?}"
            );
        }
    }

    /// A string-literal pattern compares against the literal's `.rodata` blob, shared with an
    /// equal literal elsewhere: one static, bytes and NUL.
    #[test]
    fn a_string_pattern_compares_against_the_shared_static() {
        let (p, i) = lower(
            "fn k(s: str) -> i32 { match s { \"abc\" => 1, \"a\\x62c\" => 2, _ => 0 } }\n\
             fn main() -> i32 { let t: str = \"abc\"; return k(t); }",
        );
        let abc: Vec<&wukong_mir::StaticData> =
            p.statics.iter().filter(|s| s.bytes == b"abc\0").collect();
        assert_eq!(abc.len(), 1, "one deduplicated blob");
        let f = func(&p, &i, "k");
        let addrs = ops(f)
            .into_iter()
            .filter(|o| matches!(o, Op::GlobalAddr(sym) if *sym == abc[0].name))
            .count();
        assert_eq!(addrs, 2, "both arms test against the one blob");
    }

    /// `for c in s.chars()` decodes ASCII inline (a compare against 0x80) and calls the runtime
    /// decoder once, on the other path.
    #[test]
    fn a_chars_loop_decodes_ascii_inline() {
        let (p, i) = lower(
            "fn n(s: str) -> i64 { let mut k: i64 = 0; for c in s.chars() { k = k + (c as i64); } return k; }\n\
             fn main() -> i32 { return 0; }",
        );
        let f = func(&p, &i, "n");
        assert_eq!(callees(f, &i), ["wukong_rt_utf8_decode"]);
        assert!(ops(f)
            .iter()
            .any(|o| matches!(o, Op::ConstInt(0x80, MirType::I32))));
    }

    /// `str_concat` allocates the total length plus a NUL from the `alloc_<T>` heap, once.
    #[test]
    fn str_concat_allocates_once() {
        let (p, i) = lower(
            "fn j(a: str, b: str) -> str { return str_concat(a, \"-\", b); }\n\
             fn main() -> i32 { return 0; }",
        );
        assert_eq!(callees(func(&p, &i, "j"), &i), ["wukong_rt_alloc"]);
    }
}
