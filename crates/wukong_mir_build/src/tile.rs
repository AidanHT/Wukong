//! `@tile` strip-mining: expand one range `for` into tiled inner loops plus a remainder loop.

use wukong_ast::{self as ast, Attr, AttrArg, Block, Expr, ForIter, Pattern};
use wukong_diag::Diagnostic;
use wukong_mir::{BinOp, CmpOp, MirType, Op};
use wukong_span::{Interner, Span, Symbol};

use crate::FnLowerer;

/// The help line every `E0309` carries: one sentence stating both rules the attribute has.
const TILE_HELP: &str = "`@tile` takes one or more positive integer widths (`@tile(64)`, \
                         `@tile(64, 32)`) and applies them, in source order, to loops written \
                         `for <i> in <start>..<end>`";

/// One `E0309`. A `@tile` the compiler cannot apply is an error and never silence — the width used
/// to be dropped where it stood, which is the silent-discard failure `E0306` was introduced to end.
fn tile_error(span: Span, message: &str, label: &str) -> Diagnostic {
    Diagnostic::error(message.to_string())
        .with_code("E0309")
        .primary(span, label)
        .help(TILE_HELP)
}

/// Push `d` unless the same message is already recorded at the same span. One statement's
/// attributes can be read more than once — a generic function is lowered once per instantiation —
/// and the second reading carries nothing the first did not.
fn push_unique(diags: &mut Vec<Diagnostic>, d: Diagnostic) {
    let dup = diags.iter().any(|x| {
        x.code == d.code
            && x.message == d.message
            && x.labels.first().map(|l| l.span) == d.labels.first().map(|l| l.span)
    });
    if !dup {
        diags.push(d);
    }
}

/// Where to point `E0309` for one malformed argument.
fn arg_span(arg: &AttrArg) -> Span {
    match arg {
        AttrArg::Word(_, s) | AttrArg::Int(_, s) | AttrArg::Str(_, s) => *s,
        AttrArg::KeyValue { key, .. } => key.span,
    }
}

/// Every tile width on a `@tile(...)` attribute, in source order, with each argument that is not a
/// positive integer literal reported as `E0309`.
///
/// `@tile(0)`, `@tile(rows)` and a bare `@tile` used to be dropped here without a word, so the
/// program compiled with none of the requested tiling and no diagnostic at all.
pub fn tile_sizes_from_attrs(
    attrs: &[Attr],
    interner: &Interner,
    diags: &mut Vec<Diagnostic>,
) -> Vec<u64> {
    let mut out = Vec::new();
    for a in attrs {
        if interner.resolve(a.name.sym) != "tile" {
            continue;
        }
        if a.args().is_empty() {
            push_unique(
                diags,
                tile_error(
                    a.span,
                    "`@tile` needs a tile width",
                    "no tile width was given",
                ),
            );
            continue;
        }
        for arg in a.args() {
            let AttrArg::Int(sym, span) = arg else {
                push_unique(
                    diags,
                    tile_error(
                        arg_span(arg),
                        "a `@tile` width must be an integer literal",
                        "not an integer literal",
                    ),
                );
                continue;
            };
            match interner.resolve(*sym).parse::<u64>() {
                Ok(n) if n > 0 => out.push(n),
                Ok(_) => push_unique(
                    diags,
                    tile_error(
                        *span,
                        "a `@tile` width must be at least 1",
                        "a tile of zero iterations",
                    ),
                ),
                Err(_) => push_unique(
                    diags,
                    tile_error(
                        *span,
                        "this `@tile` width does not fit in 64 bits",
                        "tile width out of range",
                    ),
                ),
            }
        }
    }
    out
}

/// Report function-item `@tile` widths that the lowering this function took can never consume:
/// a queue left non-empty at the end of the body, or a whole-function interception that builds
/// its lowerer with no queue at all. Silence here was the D4 failure one level up -- the
/// attribute was read, validated, and then dropped because nothing ever asked for it.
pub(super) fn report_unapplied_tiles(
    diags: &mut Vec<Diagnostic>,
    span: Span,
    widths: &[u64],
    label: &str,
) {
    if widths.is_empty() {
        return;
    }
    let list = widths
        .iter()
        .map(|w| w.to_string())
        .collect::<Vec<_>>()
        .join(", ");
    let (noun, verb) = if widths.len() == 1 {
        ("width", "was")
    } else {
        ("widths", "were")
    };
    push_unique(
        diags,
        tile_error(
            span,
            &format!("`@tile` {noun} {list} {verb} never applied"),
            label,
        ),
    );
}

/// The `(start, end)` of a `for` loop [`FnLowerer::lower_tiled_range_for`] can rewrite: a single
/// identifier bound over an exclusive, step-less range with a known end. Every other form — `0..=n`,
/// a stepped or open range, iteration over an array, a destructuring pattern — reaches the
/// strip-miner only to be declined, which is exactly why the caller must ask this *before* it
/// consumes a width.
pub(super) fn tileable_range<'e>(pat: &Pattern, iter: &'e ForIter) -> Option<(&'e Expr, &'e Expr)> {
    if !matches!(pat.kind, ast::PatKind::Ident(_)) {
        return None;
    }
    crate::range_bounds(iter)
}

impl FnLowerer<'_> {
    /// Is a tile width waiting for the next `for` loop — an explicit `@tile(B)` on the statement, or
    /// the head of the function-item `@tile(BM, BN, …)` queue? Reads without consuming, and
    /// validates the statement's own arguments, so a malformed `@tile` is still reported on a loop
    /// the strip-miner would decline anyway.
    pub(super) fn tile_requested(&mut self, stmt_attrs: &[Attr]) -> bool {
        if self.has_stmt_tile(stmt_attrs) {
            let sizes = tile_sizes_from_attrs(stmt_attrs, self.interner, &mut *self.diags);
            return !sizes.is_empty();
        }
        !self.tile_queue.is_empty()
    }

    /// Does this statement carry a `@tile` of its own? Asked before reading the widths so that a
    /// statement whose every `@tile` argument is malformed still takes the statement path and
    /// reports there, instead of silently falling through to the function-item queue.
    fn has_stmt_tile(&self, stmt_attrs: &[Attr]) -> bool {
        stmt_attrs
            .iter()
            .any(|a| self.interner.resolve(a.name.sym) == "tile")
    }

    /// Consume the tile width for this `for` loop: an explicit `@tile(B)` on the statement wins;
    /// else the next size from the function-item `@tile(BM, BN, …)` queue.
    ///
    /// Call this only once [`tileable_range`] has accepted the loop. It used to pop the queue before
    /// anything looked at the loop form, so an unsupported form discarded its width in silence
    /// **and** shifted the queue — the NEXT loop was strip-mined at the wrong width.
    pub(super) fn take_tile_size(&mut self, stmt_attrs: &[Attr]) -> Option<u64> {
        if self.has_stmt_tile(stmt_attrs) {
            // EVERY `@tile` on the statement, not just the first. Both readers used to `return`
            // inside the loop on the first one, so `@tile(4) @tile(0)` applied the 4 and discarded
            // the 0 without a word -- while `tile_sizes_from_attrs` on a function ITEM iterates all
            // of them, so the two positions disagreed about the same spelling. They now agree:
            // `@tile(4) @tile(8)` means the widths 4 and 8 in source order, exactly like
            // `@tile(4, 8)`, and a malformed argument in any of them is reported.
            let sizes = tile_sizes_from_attrs(stmt_attrs, self.interner, &mut *self.diags);
            if sizes.is_empty() {
                return None;
            }
            if sizes.len() > 1 {
                self.tile_queue.splice(0..0, sizes[1..].iter().copied());
            }
            return Some(sizes[0]);
        }
        if self.tile_queue.is_empty() {
            return None;
        }
        Some(self.tile_queue.remove(0))
    }

    /// Report the `@tile` width that has nowhere to go: `E0309`, pointed at the statement's own
    /// `@tile` when it has one and at the loop variable when the width came from the function item.
    /// Nothing is consumed, so the width stays at the head of the queue for the next loop rather
    /// than shifting onto it.
    pub(super) fn tile_not_applicable(
        &mut self,
        stmt_attrs: &[Attr],
        pat: &Pattern,
        loop_span: Span,
    ) {
        let span = self.tile_span(stmt_attrs, loop_span);
        // Two different rules reach this decline and they used to share one label, so a loop that
        // WAS written `for <i> in <start>..<end>` was told that only that form can be strip-mined.
        // Name the rule the loop actually broke.
        let label = if matches!(pat.kind, ast::PatKind::Ident(_)) {
            "only `for <i> in <start>..<end>` can be strip-mined"
        } else {
            "the loop variable must be a name, not a pattern"
        };
        push_unique(
            &mut *self.diags,
            tile_error(span, "`@tile` cannot strip-mine this loop", label),
        );
    }

    /// Where an `E0309` about this loop points: the statement's own `@tile`, or the loop itself when
    /// the width came from the function item's queue.
    fn tile_span(&self, stmt_attrs: &[Attr], loop_span: Span) -> Span {
        stmt_attrs
            .iter()
            .find(|a| self.interner.resolve(a.name.sym) == "tile")
            .map_or(loop_span, |a| a.span)
    }

    /// Report a `@tile` on a `@parallel` `f32` reduction loop (P13.7, `par_reduce.rs`). That loop's
    /// value is the parallel reduction tree, whichever lowering computes it; the strip-miner lowers
    /// the fold body statement by statement, which is the source's left fold. Tiling it would change
    /// the result bits, not just the schedule, and `@parallel` and `@strict_fp` are the only switches
    /// that choose such a loop's order. So the width has nowhere to go: `E0309`, nothing consumed,
    /// exactly as for a loop form the strip-miner cannot take.
    pub(super) fn tile_on_parallel_reduction(&mut self, stmt_attrs: &[Attr], loop_span: Span) {
        let span = self.tile_span(stmt_attrs, loop_span);
        push_unique(
            &mut *self.diags,
            Diagnostic::error("`@tile` cannot strip-mine this loop".to_string())
                .with_code("E0309")
                .primary(
                    span,
                    "a `@parallel` `f32` reduction folds by the parallel reduction tree; strip-mining \
                     would fold it in source order",
                )
                .help(
                    "drop the `@tile` (the tree already walks the loop in fixed 8,192-element \
                     chunks), or mark the function `@strict_fp`, whose reductions fold in source \
                     order and can be strip-mined",
                ),
        );
    }

    /// Strip-mine `for pat in start..end` with tile `tile` into a full-tile nest plus a remainder
    /// loop. Returns `false` when the pattern is not a single identifier.
    pub(super) fn lower_tiled_range_for(
        &mut self,
        label: Option<Symbol>,
        pat: &Pattern,
        start: &Expr,
        end: &Expr,
        body: &Block,
        tile: u64,
    ) -> bool {
        let ast::PatKind::Ident(name) = pat.kind else {
            return false;
        };
        let b = tile as i128;
        if b <= 0 {
            return false;
        }

        let (ity, signed) = self.range_counter_type(start, end);
        if !ity.is_int() || !self.expr_mir(start).is_int() || !self.expr_mir(end).is_int() {
            push_unique(
                self.diags,
                tile_error(
                    start.span,
                    "`@tile` requires integer range bounds",
                    "this range has a non-integer counter",
                ),
            );
            return false;
        }

        // A tile is a u64 count, independent of the source counter width. Narrowing it to that
        // width could turn @tile(256) on u8 into division by zero, or a positive width into a
        // negative divisor. Keep the full count and the inner offset unsigned through division.
        let tile_ty = MirType::I64;
        let tile_const = self
            .builder
            .build(tile_ty.clone(), Op::ConstInt(b, tile_ty.clone()));
        let tile_offset = self.coerce_to(tile_const, &tile_ty, &ity, false);

        let start_val = self.lower_range_bound(start, &ity, signed);
        let end_val = self.lower_range_bound(end, &ity, signed);
        let exit = self.builder.new_block();
        let nonempty = self.builder.new_block();
        let ordered = self.builder.build(
            MirType::I1,
            Op::Cmp(
                if signed { CmpOp::Slt } else { CmpOp::Ult },
                start_val,
                end_val,
            ),
        );
        self.builder
            .cond_br(ordered, nonempty, vec![], exit, vec![]);
        self.builder.switch_to(nonempty);
        self.terminated = false;

        // Once start < end in the counter's ordering, the modular difference interpreted as
        // unsigned is their exact positive distance, including signed ranges spanning zero.
        // A signed difference could overflow even when both endpoints are representable.
        let len_bits = self
            .builder
            .build(ity.clone(), Op::Bin(BinOp::Sub, end_val, start_val));
        let len = self.coerce_to(len_bits, &ity, &tile_ty, false);
        let quot = self
            .builder
            .build(tile_ty.clone(), Op::Bin(BinOp::UDiv, len, tile_const));
        let full_wide = self
            .builder
            .build(tile_ty.clone(), Op::Bin(BinOp::Mul, quot, tile_const));
        let full = self.coerce_to(full_wide, &tile_ty, &ity, false);
        let full_end = self
            .builder
            .build(ity.clone(), Op::Bin(BinOp::Add, start_val, full));

        let after_tiles = self.builder.new_block();

        // --- Full tiles ---------------------------------------------------------
        let io_slot = self.builder.alloca(ity.clone());
        self.builder.build_void(Op::Store {
            ptr: io_slot,
            value: start_val,
        });
        let ii_slot = self.builder.alloca(tile_ty.clone());

        let outer_header = self.builder.new_block();
        let inner_init = self.builder.new_block();
        let inner_header = self.builder.new_block();
        let inner_body = self.builder.new_block();
        let inner_step = self.builder.new_block();
        let outer_step = self.builder.new_block();

        self.builder.br(outer_header, vec![]);

        // outer: io < full_end ?
        self.builder.switch_to(outer_header);
        self.terminated = false;
        let io = self
            .builder
            .build(ity.clone(), Op::Load(io_slot, ity.clone()));
        let outer_ok = self.builder.build(
            MirType::I1,
            Op::Cmp(if signed { CmpOp::Slt } else { CmpOp::Ult }, io, full_end),
        );
        self.builder
            .cond_br(outer_ok, inner_init, vec![], after_tiles, vec![]);

        // inner init: ii = 0
        self.builder.switch_to(inner_init);
        self.terminated = false;
        let zero = self
            .builder
            .build(tile_ty.clone(), Op::ConstInt(0, tile_ty.clone()));
        self.builder.build_void(Op::Store {
            ptr: ii_slot,
            value: zero,
        });
        self.builder.br(inner_header, vec![]);

        // inner header: ii < tile ?
        self.builder.switch_to(inner_header);
        self.terminated = false;
        let ii = self
            .builder
            .build(tile_ty.clone(), Op::Load(ii_slot, tile_ty.clone()));
        let inner_ok = self
            .builder
            .build(MirType::I1, Op::Cmp(CmpOp::Ult, ii, tile_const));
        self.builder
            .cond_br(inner_ok, inner_body, vec![], outer_step, vec![]);

        // inner body
        self.builder.switch_to(inner_body);
        self.terminated = false;
        self.push_scope();
        let io_v = self
            .builder
            .build(ity.clone(), Op::Load(io_slot, ity.clone()));
        let ii_v = self
            .builder
            .build(tile_ty.clone(), Op::Load(ii_slot, tile_ty.clone()));
        let ii_offset = self.coerce_to(ii_v, &tile_ty, &ity, false);
        let i_val = self
            .builder
            .build(ity.clone(), Op::Bin(BinOp::Add, io_v, ii_offset));
        let i_slot = self.builder.alloca(ity.clone());
        self.builder.build_void(Op::Store {
            ptr: i_slot,
            value: i_val,
        });
        self.bind(name, i_slot, ity.clone());
        self.loops.push((label, inner_step, exit));
        // The body is lowered TWICE -- once for the full-tile nest and once for the remainder --
        // but it is ONE source loop, so a nested `for` inside it must not pop the function-item
        // `@tile(BM, BN, ...)` queue twice. `@tile(2, 4, 8)` over `for i { for j { } }` used to
        // give the inner loop 4 in the tiled copy and 8 in the remainder copy: two widths for one
        // loop the user wrote once, silently, and MIR byte-identical to the pre-E0309 parent.
        // Snapshot before the first copy, rewind for the second, and keep what the first left.
        let tiles_before = self.tile_queue.clone();
        self.lower_block(body);
        let tiles_after = std::mem::replace(&mut self.tile_queue, tiles_before);
        self.loops.pop();
        self.pop_scope();
        if !self.terminated {
            self.builder.br(inner_step, vec![]);
        }

        // inner step: ii += 1
        self.builder.switch_to(inner_step);
        self.terminated = false;
        let ii_cur = self
            .builder
            .build(tile_ty.clone(), Op::Load(ii_slot, tile_ty.clone()));
        let one = self
            .builder
            .build(tile_ty.clone(), Op::ConstInt(1, tile_ty.clone()));
        let ii_next = self
            .builder
            .build(tile_ty.clone(), Op::Bin(BinOp::Add, ii_cur, one));
        self.builder.build_void(Op::Store {
            ptr: ii_slot,
            value: ii_next,
        });
        self.builder.br(inner_header, vec![]);

        // outer step: io += tile
        self.builder.switch_to(outer_step);
        self.terminated = false;
        let io_cur = self
            .builder
            .build(ity.clone(), Op::Load(io_slot, ity.clone()));
        let io_next = self
            .builder
            .build(ity.clone(), Op::Bin(BinOp::Add, io_cur, tile_offset));
        self.builder.build_void(Op::Store {
            ptr: io_slot,
            value: io_next,
        });
        self.builder.br(outer_header, vec![]);

        // --- Remainder: for i in full_end..end ----------------------------------
        self.builder.switch_to(after_tiles);
        self.terminated = false;

        let rem_slot = self.builder.alloca(ity.clone());
        self.builder.build_void(Op::Store {
            ptr: rem_slot,
            value: full_end,
        });

        let rem_header = self.builder.new_block();
        let rem_body = self.builder.new_block();
        let rem_step = self.builder.new_block();

        self.builder.br(rem_header, vec![]);

        self.builder.switch_to(rem_header);
        self.terminated = false;
        let ri = self
            .builder
            .build(ity.clone(), Op::Load(rem_slot, ity.clone()));
        let rem_ok = self.builder.build(
            MirType::I1,
            Op::Cmp(if signed { CmpOp::Slt } else { CmpOp::Ult }, ri, end_val),
        );
        self.builder.cond_br(rem_ok, rem_body, vec![], exit, vec![]);

        self.builder.switch_to(rem_body);
        self.terminated = false;
        self.push_scope();
        self.bind(name, rem_slot, ity.clone());
        self.loops.push((label, rem_step, exit));
        self.lower_block(body);
        self.tile_queue = tiles_after;
        self.loops.pop();
        self.pop_scope();
        if !self.terminated {
            self.builder.br(rem_step, vec![]);
        }

        self.builder.switch_to(rem_step);
        self.terminated = false;
        let ri_cur = self
            .builder
            .build(ity.clone(), Op::Load(rem_slot, ity.clone()));
        let one_i = self
            .builder
            .build(ity.clone(), Op::ConstInt(1, ity.clone()));
        let ri_next = self
            .builder
            .build(ity.clone(), Op::Bin(BinOp::Add, ri_cur, one_i));
        self.builder.build_void(Op::Store {
            ptr: rem_slot,
            value: ri_next,
        });
        self.builder.br(rem_header, vec![]);

        self.builder.switch_to(exit);
        self.terminated = false;
        true
    }
}
