//! `@tile` strip-mining: expand one range `for` into tiled inner loops plus a remainder loop.

use wukong_ast::{self as ast, Attr, AttrArg, Block, Expr, Pattern};
use wukong_mir::{BinOp, CmpOp, MirType, Op};
use wukong_span::{Interner, Symbol};

use crate::FnLowerer;

/// Every integer argument on a `@tile(...)` attribute, in source order.
pub fn tile_sizes_from_attrs(attrs: &[Attr], interner: &Interner) -> Vec<u64> {
    let mut out = Vec::new();
    for a in attrs {
        if interner.resolve(a.name.sym) != "tile" {
            continue;
        }
        for arg in a.args() {
            if let AttrArg::Int(sym, _) = arg {
                if let Ok(n) = interner.resolve(*sym).parse::<u64>() {
                    if n > 0 {
                        out.push(n);
                    }
                }
            }
        }
    }
    out
}

impl FnLowerer<'_> {
    /// The tile width for the next `for` loop: an explicit `@tile(B)` on the statement wins; else
    /// the next size from the function-item `@tile(BM, BN, …)` queue.
    pub(super) fn take_tile_size(&mut self, stmt_attrs: &[Attr]) -> Option<u64> {
        for a in stmt_attrs {
            if self.interner.resolve(a.name.sym) != "tile" {
                continue;
            }
            let sizes = tile_sizes_from_attrs(std::slice::from_ref(a), self.interner);
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

        let sty = self.expr_mir(start);
        let ety = self.expr_mir(end);
        let (ity, signed) =
            if ety.is_int() && sty.is_int() && crate::mir_byte_size(&ety) > crate::mir_byte_size(&sty)
            {
                (ety.clone(), self.signed(end))
            } else {
                (
                    sty.clone(),
                    self.signed(start) && !self.forces_unsigned_cmp(end, &sty),
                )
            };

        let tile_ty = ity.clone();
        let tile_const = self
            .builder
            .build(tile_ty.clone(), Op::ConstInt(b, tile_ty.clone()));

        let start_raw = self.lower_expr(start);
        let start_val = self.coerce_to(start_raw, &sty, &ity, signed);
        let end_raw = self.lower_expr(end);
        let end_val = self.coerce_to(end_raw, &ety, &ity, signed);

        let len = self
            .builder
            .build(ity.clone(), Op::Bin(BinOp::Sub, end_val, start_val));
        let div_op = if signed {
            BinOp::SDiv
        } else {
            BinOp::UDiv
        };
        let quot = self
            .builder
            .build(ity.clone(), Op::Bin(div_op, len, tile_const));
        let full = self
            .builder
            .build(ity.clone(), Op::Bin(BinOp::Mul, quot, tile_const));
        let full_end = self
            .builder
            .build(ity.clone(), Op::Bin(BinOp::Add, start_val, full));

        let exit = self.builder.new_block();
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
        let io = self.builder.build(ity.clone(), Op::Load(io_slot, ity.clone()));
        let outer_ok = self.builder.build(
            MirType::I1,
            Op::Cmp(
                if signed {
                    CmpOp::Slt
                } else {
                    CmpOp::Ult
                },
                io,
                full_end,
            ),
        );
        self.builder
            .cond_br(outer_ok, inner_init, vec![], after_tiles, vec![]);

        // inner init: ii = 0
        self.builder.switch_to(inner_init);
        self.terminated = false;
        let zero = self.builder.build(tile_ty.clone(), Op::ConstInt(0, tile_ty.clone()));
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
        let inner_ok = self.builder.build(
            MirType::I1,
            Op::Cmp(
                if signed {
                    CmpOp::Slt
                } else {
                    CmpOp::Ult
                },
                ii,
                tile_const,
            ),
        );
        self.builder
            .cond_br(inner_ok, inner_body, vec![], outer_step, vec![]);

        // inner body
        self.builder.switch_to(inner_body);
        self.terminated = false;
        self.push_scope();
        let io_v = self.builder.build(ity.clone(), Op::Load(io_slot, ity.clone()));
        let ii_v = self
            .builder
            .build(tile_ty.clone(), Op::Load(ii_slot, tile_ty.clone()));
        let i_val = self
            .builder
            .build(ity.clone(), Op::Bin(BinOp::Add, io_v, ii_v));
        let i_slot = self.builder.alloca(ity.clone());
        self.builder.build_void(Op::Store {
            ptr: i_slot,
            value: i_val,
        });
        self.bind(name, i_slot, ity.clone());
        self.loops.push((label, inner_step, exit));
        self.lower_block(body);
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
        let io_cur = self.builder.build(ity.clone(), Op::Load(io_slot, ity.clone()));
        let io_next = self
            .builder
            .build(ity.clone(), Op::Bin(BinOp::Add, io_cur, tile_const));
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
        let ri = self.builder.build(ity.clone(), Op::Load(rem_slot, ity.clone()));
        let rem_ok = self.builder.build(
            MirType::I1,
            Op::Cmp(
                if signed {
                    CmpOp::Slt
                } else {
                    CmpOp::Ult
                },
                ri,
                end_val,
            ),
        );
        self.builder
            .cond_br(rem_ok, rem_body, vec![], exit, vec![]);

        self.builder.switch_to(rem_body);
        self.terminated = false;
        self.push_scope();
        self.bind(name, rem_slot, ity.clone());
        self.loops.push((label, rem_step, exit));
        self.lower_block(body);
        self.loops.pop();
        self.pop_scope();
        if !self.terminated {
            self.builder.br(rem_step, vec![]);
        }

        self.builder.switch_to(rem_step);
        self.terminated = false;
        let ri_cur = self.builder.build(ity.clone(), Op::Load(rem_slot, ity.clone()));
        let one_i = self.builder.build(ity.clone(), Op::ConstInt(1, ity.clone()));
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
