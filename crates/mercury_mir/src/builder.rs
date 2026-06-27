//! A convenience builder for constructing MIR functions imperatively.

use crate::{BasicBlock, BlockId, Function, Inst, MirType, Op, Terminator, ValueId};
use mercury_span::Symbol;

/// Builds one [`Function`]. Create it, add parameters, carve out blocks, append instructions to
/// the "current" block, and finish.
pub struct Builder {
    name: Symbol,
    ret: MirType,
    value_types: Vec<MirType>,
    blocks: Vec<BasicBlock>,
    entry: BlockId,
    current: BlockId,
}

impl Builder {
    pub fn new(name: Symbol, ret: MirType) -> Builder {
        let mut b = Builder {
            name,
            ret,
            value_types: Vec::new(),
            blocks: Vec::new(),
            entry: BlockId(0),
            current: BlockId(0),
        };
        let entry = b.new_block();
        b.entry = entry;
        b.current = entry;
        b
    }

    pub fn entry(&self) -> BlockId {
        self.entry
    }

    pub fn current(&self) -> BlockId {
        self.current
    }

    pub fn new_value(&mut self, ty: MirType) -> ValueId {
        let id = ValueId(self.value_types.len() as u32);
        self.value_types.push(ty);
        id
    }

    pub fn new_block(&mut self) -> BlockId {
        let id = BlockId(self.blocks.len() as u32);
        self.blocks.push(BasicBlock {
            id,
            params: Vec::new(),
            insts: Vec::new(),
            term: Terminator::Unreachable,
        });
        id
    }

    fn block_mut(&mut self, b: BlockId) -> &mut BasicBlock {
        &mut self.blocks[b.0 as usize]
    }

    /// Add a function parameter (also an entry-block parameter).
    pub fn add_param(&mut self, ty: MirType) -> ValueId {
        let v = self.new_value(ty);
        let entry = self.entry;
        self.block_mut(entry).params.push(v);
        v
    }

    /// Add a typed parameter to a block (an SSA merge point).
    pub fn block_param(&mut self, blk: BlockId, ty: MirType) -> ValueId {
        let v = self.new_value(ty);
        self.block_mut(blk).params.push(v);
        v
    }

    pub fn switch_to(&mut self, blk: BlockId) {
        self.current = blk;
    }

    /// Append an instruction to the current block, allocating a result value if `result_ty` is set.
    pub fn push(&mut self, result_ty: Option<MirType>, op: Op) -> Option<ValueId> {
        let result = result_ty.map(|t| self.new_value(t));
        let cur = self.current;
        self.block_mut(cur).insts.push(Inst { result, op });
        result
    }

    /// Append an instruction that produces a value of `ty` (the common case).
    pub fn build(&mut self, ty: MirType, op: Op) -> ValueId {
        self.push(Some(ty), op).unwrap()
    }

    /// Append an instruction with no result (e.g. `Store`).
    pub fn build_void(&mut self, op: Op) {
        self.push(None, op);
    }

    /// Allocate a stack slot in the entry block, returning the pointer value. Putting allocas in
    /// the entry block keeps them dominating all uses and out of loop bodies.
    pub fn alloca(&mut self, ty: MirType) -> ValueId {
        let v = self.new_value(MirType::Ptr);
        let entry = self.entry.0 as usize;
        self.blocks[entry].insts.push(Inst {
            result: Some(v),
            op: Op::Alloca(ty),
        });
        v
    }

    pub fn set_term(&mut self, term: Terminator) {
        let cur = self.current;
        self.block_mut(cur).term = term;
    }

    pub fn ret(&mut self, val: Option<ValueId>) {
        self.set_term(Terminator::Ret(val));
    }

    /// The function's declared return type (the `ret` passed to [`Builder::new`]). Lets the lowerer
    /// coerce a `return`/tail value to it so the emitted `Ret` is well-typed.
    pub fn ret_type(&self) -> &MirType {
        &self.ret
    }

    pub fn br(&mut self, target: BlockId, args: Vec<ValueId>) {
        self.set_term(Terminator::Br { target, args });
    }

    pub fn cond_br(
        &mut self,
        cond: ValueId,
        then_blk: BlockId,
        then_args: Vec<ValueId>,
        else_blk: BlockId,
        else_args: Vec<ValueId>,
    ) {
        self.set_term(Terminator::CondBr {
            cond,
            then_blk,
            then_args,
            else_blk,
            else_args,
        });
    }

    pub fn finish(self) -> Function {
        let params = self.blocks[self.entry.0 as usize].params.clone();
        Function {
            name: self.name,
            params,
            ret: self.ret,
            blocks: self.blocks,
            value_types: self.value_types,
            entry: self.entry,
        }
    }
}
