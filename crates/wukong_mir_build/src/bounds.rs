//! Slice bounds checks.
//!
//! A `[]T` slice is a fat pointer whose length sits next to its data pointer, so every runtime-sized
//! access can be checked for one load and one compare. `s[i]` lowers to
//!
//! ```text
//!     len  = load i64 (s + 8)
//!     ok   = cmp ult i, len          ; a negative i is huge unsigned, so it fails too
//!     cond_br ok, cont, fail
//! fail:
//!     call wukong_rt_bounds_fail(i, len, site)
//!     unreachable
//! cont:
//!     gep data, i …
//! ```
//!
//! — the shape Rust gives `slice[i]`. The failure edge is a dedicated dead-end block, which is what
//! `wukong_opt::bce` keys on: it proves the comparison true for a whole loop (one guard in the
//! preheader, the loop versioned on it) so a hot loop pays nothing and stays vectorizable, while
//! the out-of-bounds iteration of the original loop still stops at exactly the same point.
//!
//! Arrays and constant-shape tensors need no runtime check here — their extents are static and sema
//! rejects a provably out-of-range constant index. A function marked `@unchecked` opts out (the
//! explicit, visible counterpart of Rust's `get_unchecked`); an out-of-bounds access there is
//! undefined behaviour on the native backend, as every access was before this check existed.
//!
//! `site` is the access's `file:line:col`. The lowerer cannot mint the `.rodata` blob that holds it
//! (it has only a shared `&Interner`), so it emits a placeholder `i64` constant packing the span's
//! source id and start offset, and [`attach_sites`] — run once by `lower_program_located`, which
//! owns the interner and the caller's source locator — rewrites each placeholder into the
//! `GlobalAddr` of a deduplicated static string.

use wukong_mir::{CmpOp, MirType, Op, Program, StaticData, ValueId};
use wukong_span::{FxHashMap as HashMap, FxHashSet as HashSet, Interner, SourceId, Span, Symbol};

use crate::{FnLowerer, SLICE_LEN_OFF};

/// The placeholder site operand for `span`: source id in the high half, start offset in the low.
/// A compiler-synthesized node (dummy span) gets `-1`, which [`attach_sites`] renders as no site.
fn site_key(span: Span) -> i128 {
    if span.is_dummy() {
        -1
    } else {
        (i128::from(span.source.0) << 32) | i128::from(span.lo)
    }
}

/// The span start a [`site_key`] encoded, or `None` for the dummy key.
fn site_span(key: i128) -> Option<Span> {
    if key < 0 {
        return None;
    }
    let source = SourceId((key >> 32) as u32);
    let lo = (key & 0xffff_ffff) as u32;
    Some(Span::new(source, lo, lo))
}

impl FnLowerer<'_> {
    /// Emit the bounds check for a slice access whose fat pointer is at `sbuf` and whose already
    /// lowered `i64` index is `idx`, leaving the builder in the in-bounds continuation block. A
    /// no-op inside an `@unchecked` function.
    pub(crate) fn emit_slice_bounds_check(&mut self, sbuf: ValueId, idx: ValueId, span: Span) {
        if !self.gemm.bounds_checks {
            return;
        }
        let lp = self.field_ptr(sbuf, SLICE_LEN_OFF);
        let len = self.builder.build(MirType::I64, Op::Load(lp, MirType::I64));
        let ok = self
            .builder
            .build(MirType::I1, Op::Cmp(CmpOp::Ult, idx, len));
        let cont = self.builder.new_block();
        let fail = self.builder.new_block();
        self.builder.cond_br(ok, cont, vec![], fail, vec![]);
        self.builder.switch_to(fail);
        let site = self
            .builder
            .build(MirType::I64, Op::ConstInt(site_key(span), MirType::I64));
        let func = self.gemm.bounds_fail;
        self.builder.build_void(Op::Call {
            func,
            args: vec![idx, len, site],
        });
        // A fresh block's terminator is already `Unreachable`: the call never returns normally.
        self.builder.switch_to(cont);
    }

    /// Emit the bounds check of a range index `s[lo..hi]` over a slice of length `len` (all three
    /// already lowered `i64`s): `hi <= len`, then `lo <= hi`, each compared unsigned (a negative
    /// bound is huge, so it fails) with its own dead-end failure block, as
    /// [`Self::emit_slice_bounds_check`] builds one. A failure reports the bound and the limit it
    /// passed: `the len is 6 but the index is 9` for `hi`, and `hi` in the role of the length for
    /// `lo`. Leaves the builder in the in-bounds continuation; a no-op inside an `@unchecked`
    /// function.
    ///
    /// The comparisons are `ule`, never the `ult` of an element access, so `wukong_opt::bce` (which
    /// keys on `ult`) never hoists or removes one: a range check stays exactly where it was written.
    pub(crate) fn emit_range_bounds_check(
        &mut self,
        lo: ValueId,
        hi: ValueId,
        len: ValueId,
        span: Span,
    ) {
        if !self.gemm.bounds_checks {
            return;
        }
        for (bound, limit) in [(hi, len), (lo, hi)] {
            let ok = self
                .builder
                .build(MirType::I1, Op::Cmp(CmpOp::Ule, bound, limit));
            let cont = self.builder.new_block();
            let fail = self.builder.new_block();
            self.builder.cond_br(ok, cont, vec![], fail, vec![]);
            self.builder.switch_to(fail);
            let site = self
                .builder
                .build(MirType::I64, Op::ConstInt(site_key(span), MirType::I64));
            let func = self.gemm.bounds_fail;
            self.builder.build_void(Op::Call {
                func,
                args: vec![bound, limit, site],
            });
            self.builder.switch_to(cont);
        }
    }
}

/// Rewrite every bounds-failure call's placeholder site operand into the address of a static
/// `file:line:col` string, deduplicated by content. `locate` renders a span start; an empty string
/// means "no site" and the runtime message then omits it.
pub(crate) fn attach_sites(
    program: &mut Program,
    interner: &mut Interner,
    bounds_fail: Symbol,
    locate: &dyn Fn(Span) -> String,
) {
    let mut by_text: HashMap<String, Symbol> = HashMap::default();
    let mut statics: Vec<StaticData> = Vec::new();
    for f in &mut program.funcs {
        // The placeholder site operand of every bounds-failure call in this function.
        let mut placeholders: HashSet<u32> = HashSet::default();
        for b in &f.blocks {
            for inst in &b.insts {
                if let Op::Call { func, args } = &inst.op {
                    if *func == bounds_fail && args.len() == 3 {
                        placeholders.insert(args[2].0);
                    }
                }
            }
        }
        if placeholders.is_empty() {
            continue;
        }
        for b in &mut f.blocks {
            for inst in &mut b.insts {
                let Some(r) = inst.result else { continue };
                if !placeholders.contains(&r.0) {
                    continue;
                }
                let Op::ConstInt(key, MirType::I64) = inst.op else {
                    continue;
                };
                let text = site_span(key).map(locate).unwrap_or_default();
                let sym = *by_text.entry(text).or_insert_with_key(|text| {
                    let name = interner.intern(&format!("wukong$loc${}", statics.len()));
                    let mut bytes = text.as_bytes().to_vec();
                    bytes.push(0);
                    statics.push(StaticData { name, bytes });
                    name
                });
                inst.op = Op::GlobalAddr(sym);
                f.value_types[r.0 as usize] = MirType::Ptr;
            }
        }
    }
    program.statics.extend(statics);
}
