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
//!
//! A failed `assert` reports its site the same way: with a source locator the intrinsic call is
//! `assert(cond, site)`, or `assert(cond, site, message)` for `assert(cond, "message")`, whose
//! message operand is the literal's own `.rodata` blob. Every CPU backend hands both to
//! `wukong_runtime::assert_failure_message`. A kernel builtin's run-time extent check traps through
//! the same intrinsic ([`FnLowerer::emit_builtin_checks`]), so it names the builtin call, and so
//! does a value `match` no arm of which matched ([`FnLowerer::emit_match_fallthrough_trap`]),
//! naming the `match`.

use wukong_mir::{CmpOp, MirType, Op, Program, StaticData, ValueId};
use wukong_span::{FxHashMap as HashMap, FxHashSet as HashSet, Interner, SourceId, Span, Symbol};

use crate::{DefKind, FnLowerer, SLICE_LEN_OFF};

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
    /// The placeholder site operand of a runtime failure at `span` (an `i64` constant
    /// [`attach_sites`] rewrites into the address of its `file:line:col` string).
    pub(crate) fn site_placeholder(&mut self, span: Span) -> ValueId {
        self.builder
            .build(MirType::I64, Op::ConstInt(site_key(span), MirType::I64))
    }

    /// The operands after the condition of an `assert` call at `span`: its site placeholder when
    /// the lowering has a source locator (or a message needs the slot), then `message` — the
    /// address of the call's message literal — when it has one. Empty for a bare `assert(cond)`
    /// lowered without a locator, which keeps the call's historical one-operand form.
    pub(crate) fn assert_site_operands(
        &mut self,
        span: Span,
        message: Option<ValueId>,
    ) -> Vec<ValueId> {
        let mut ops = Vec::new();
        if self.gemm.assert_sites || message.is_some() {
            ops.push(self.site_placeholder(span));
        }
        ops.extend(message);
        ops
    }

    /// The block a value `match` falls into when no arm matched (one sema could not prove
    /// exhaustive, so it keeps every test -- a zero-filled `let e: E;` of an enum with no
    /// discriminant 0 reaches it): stop the run through the `assert` intrinsic, naming the `match`
    /// at `span` (`assertion failed: no `match` arm matched its value at file:line:col`) before the
    /// block's `unreachable`. That terminator alone was `execution reached unreachable` on the
    /// interpreter and a hardware trap natively (`SIGILL`: the process killed, its output lost).
    /// Without a source locator the call is a bare `assert(false)`; when a user function named
    /// `assert` shadows the intrinsic, nothing is emitted and the block keeps its bare trap.
    pub(crate) fn emit_match_fallthrough_trap(&mut self, span: Span) {
        let assert_fn = self.gemm.assert_fn;
        if matches!(
            self.sema.defs.lookup(assert_fn).map(|d| &d.kind),
            Some(DefKind::Fn(_))
        ) {
            return;
        }
        let never = self
            .builder
            .build(MirType::I1, Op::ConstInt(0, MirType::I1));
        let mut args = vec![never];
        if self.gemm.assert_sites {
            args.push(self.site_placeholder(span));
            args.push(self.builder.build(
                MirType::I64,
                Op::ConstInt(MATCH_FALLTHROUGH_KEY, MirType::I64),
            ));
        }
        self.builder.build_void(Op::Call {
            func: assert_fn,
            args,
        });
    }

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
        let site = self.site_placeholder(span);
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
    /// [`Self::emit_slice_bounds_check`] builds one. A failure calls `wukong_rt_check_fail` with
    /// Rust's wording for each: `range end index 9 out of range for slice of length 6` for `hi`,
    /// and `slice index starts at 5 but ends at 3` for `lo` (it used to reuse the element-access
    /// wording, with `hi` in the role of the length). Leaves the builder in the in-bounds
    /// continuation; a no-op inside an `@unchecked` function.
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
        for (kind, bound, limit) in [(CHECK_RANGE_END, hi, len), (CHECK_RANGE_ORDER, lo, hi)] {
            let ok = self
                .builder
                .build(MirType::I1, Op::Cmp(CmpOp::Ule, bound, limit));
            self.emit_check_fail_unless(ok, kind, bound, limit, span);
        }
    }

    /// Branch on `ok`: continue when it holds, otherwise call
    /// `wukong_rt_check_fail(kind, a, b, site)` in a dead-end block — the shape of every bounds
    /// failure edge, so the optimizer and the backends treat it the same way. Leaves the builder in
    /// the continuation.
    pub(crate) fn emit_check_fail_unless(
        &mut self,
        ok: ValueId,
        kind: i64,
        a: ValueId,
        b: ValueId,
        span: Span,
    ) {
        let cont = self.builder.new_block();
        let fail = self.builder.new_block();
        self.builder.cond_br(ok, cont, vec![], fail, vec![]);
        self.builder.switch_to(fail);
        let k = self
            .builder
            .build(MirType::I64, Op::ConstInt(i128::from(kind), MirType::I64));
        let site = self.site_placeholder(span);
        let func = self.gemm.check_fail;
        self.builder.build_void(Op::Call {
            func,
            args: vec![k, a, b, site],
        });
        // A fresh block's terminator is already `Unreachable`: the call never returns normally.
        self.builder.switch_to(cont);
    }
}

/// The message placeholder of a `match` that no arm matched ([`attach_sites`] rewrites it into the
/// address of [`MATCH_FALLTHROUGH_MESSAGE`]). Negative and below the dummy site key (-1), so no
/// span's key is ever mistaken for it.
const MATCH_FALLTHROUGH_KEY: i128 = -2;

/// The message a `match` that no arm matched fails with.
const MATCH_FALLTHROUGH_MESSAGE: &str = "no `match` arm matched its value";

/// `wukong_rt_check_fail` kinds — the values of `wukong_runtime::safety::CHECK_*`, which this crate
/// does not depend on; the run fixtures that print each message pin the pairing end to end
/// (`tests/run/str_slice_bounds.wk`, `str_range_order.wk`, `str_from_utf8_invalid.wk`).
pub(crate) const CHECK_RANGE_ORDER: i64 = 1;
pub(crate) const CHECK_RANGE_END: i64 = 2;
pub(crate) const CHECK_UTF8: i64 = 3;

/// Rewrite every failure call's placeholder site operand into the address of a static
/// `file:line:col` string, deduplicated by content. `failures` names each failure entry as
/// `(callee, arity, site operand index)`: `wukong_rt_bounds_fail(index, len, site)`,
/// `wukong_rt_check_fail(kind, a, b, site)`, and the `assert` intrinsic's `assert(cond, site)` /
/// `assert(cond, site, message)`. `locate` renders a span start; an empty string means "no site"
/// and the runtime message then omits it. The operand after an `assert`'s site is its message: a
/// literal's blob already, or the compiler's own [`MATCH_FALLTHROUGH_KEY`] placeholder, rewritten
/// here into a blob of [`MATCH_FALLTHROUGH_MESSAGE`].
///
/// A callee the program itself defines is skipped: a user function named `assert` shadows the
/// intrinsic, and its integer arguments are the program's values, not placeholders.
pub(crate) fn attach_sites(
    program: &mut Program,
    interner: &mut Interner,
    failures: &[(Symbol, usize, usize)],
    locate: &dyn Fn(Span) -> String,
) {
    let defined: HashSet<Symbol> = program.funcs.iter().map(|f| f.name).collect();
    let failures: Vec<(Symbol, usize, usize)> = failures
        .iter()
        .copied()
        .filter(|(callee, _, _)| !defined.contains(callee))
        .collect();
    let mut by_text: HashMap<String, Symbol> = HashMap::default();
    let mut statics: Vec<StaticData> = Vec::new();
    for f in &mut program.funcs {
        // The placeholder site operand of every runtime-failure call in this function, and the
        // operand after it (an `assert`'s message) where there is one.
        let mut placeholders: HashSet<u32> = HashSet::default();
        let mut messages: HashSet<u32> = HashSet::default();
        for b in &f.blocks {
            for inst in &b.insts {
                if let Op::Call { func, args } = &inst.op {
                    for &(callee, arity, site) in &failures {
                        if *func == callee && args.len() == arity {
                            placeholders.insert(args[site].0);
                            if let Some(m) = args.get(site + 1) {
                                messages.insert(m.0);
                            }
                        }
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
                let Op::ConstInt(key, MirType::I64) = inst.op else {
                    continue;
                };
                let text = if placeholders.contains(&r.0) {
                    site_span(key).map(locate).unwrap_or_default()
                } else if messages.contains(&r.0) && key == MATCH_FALLTHROUGH_KEY {
                    MATCH_FALLTHROUGH_MESSAGE.to_string()
                } else {
                    continue;
                };
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
