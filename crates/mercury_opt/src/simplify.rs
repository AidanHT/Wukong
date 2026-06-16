//! Constant folding and algebraic simplification.

use std::collections::HashMap;

use mercury_mir::{BinOp, CmpOp, Function, MirType, Op, ValueId};

use crate::{map_op_uses, map_term_uses, Pass};

pub struct Simplify;

#[derive(Clone, Copy)]
enum CV {
    Int(i128),
    Float(f64),
}

enum Alg {
    Replace(ValueId),
    Const(CV),
}

impl Pass for Simplify {
    fn name(&self) -> &'static str {
        "simplify"
    }

    fn run_function(&self, f: &mut Function) -> bool {
        let mut changed = false;
        let mut consts: HashMap<u32, CV> = HashMap::new();
        let mut subst: HashMap<u32, ValueId> = HashMap::new();

        let nblocks = f.blocks.len();
        for bi in 0..nblocks {
            let ninsts = f.blocks[bi].insts.len();
            for ii in 0..ninsts {
                // Rewrite operands through substitutions discovered so far.
                {
                    let s = &subst;
                    map_op_uses(&mut f.blocks[bi].insts[ii].op, |v| resolve(s, v));
                }
                let Some(res) = f.blocks[bi].insts[ii].result else {
                    continue;
                };
                let rty = f.value_types[res.0 as usize].clone();
                let op = f.blocks[bi].insts[ii].op.clone();

                match op {
                    Op::ConstInt(v, _) => {
                        consts.insert(res.0, CV::Int(v));
                    }
                    Op::ConstFloat(v, _) => {
                        consts.insert(res.0, CV::Float(v));
                    }
                    Op::Bin(b, l, r) => {
                        let lc = consts.get(&l.0).copied();
                        let rc = consts.get(&r.0).copied();
                        if let (Some(a), Some(c)) = (lc, rc) {
                            if let Some(folded) = fold_bin(b, a, c, &rty) {
                                set_const(f, bi, ii, folded, &rty);
                                consts.insert(res.0, folded);
                                changed = true;
                                continue;
                            }
                        }
                        if let Some(alg) = algebra(b, l, lc, r, rc) {
                            match alg {
                                Alg::Replace(v) => {
                                    subst.insert(res.0, v);
                                    changed = true;
                                }
                                Alg::Const(cv) => {
                                    set_const(f, bi, ii, cv, &rty);
                                    consts.insert(res.0, cv);
                                    changed = true;
                                }
                            }
                        }
                    }
                    Op::Cmp(c, l, r) => {
                        if let (Some(a), Some(bv)) =
                            (consts.get(&l.0).copied(), consts.get(&r.0).copied())
                        {
                            let val = fold_cmp(c, a, bv);
                            set_const(f, bi, ii, CV::Int(val), &MirType::I1);
                            consts.insert(res.0, CV::Int(val));
                            changed = true;
                        }
                    }
                    Op::Neg(v) => {
                        if let Some(cv) = consts.get(&v.0).copied() {
                            let nv = match cv {
                                CV::Int(i) => CV::Int(mask(i.wrapping_neg(), &rty)),
                                CV::Float(fl) => CV::Float(-fl),
                            };
                            set_const(f, bi, ii, nv, &rty);
                            consts.insert(res.0, nv);
                            changed = true;
                        }
                    }
                    Op::Not(v) => {
                        if let Some(CV::Int(i)) = consts.get(&v.0).copied() {
                            let nv = CV::Int(mask(!i, &rty));
                            set_const(f, bi, ii, nv, &rty);
                            consts.insert(res.0, nv);
                            changed = true;
                        }
                    }
                    _ => {}
                }
            }
        }

        if !subst.is_empty() {
            for b in &mut f.blocks {
                for inst in &mut b.insts {
                    map_op_uses(&mut inst.op, |v| resolve(&subst, v));
                }
                map_term_uses(&mut b.term, |v| resolve(&subst, v));
            }
        }
        changed
    }
}

fn resolve(subst: &HashMap<u32, ValueId>, v: ValueId) -> ValueId {
    let mut cur = v;
    while let Some(&next) = subst.get(&cur.0) {
        if next == cur {
            break;
        }
        cur = next;
    }
    cur
}

fn set_const(f: &mut Function, bi: usize, ii: usize, cv: CV, ty: &MirType) {
    f.blocks[bi].insts[ii].op = match cv {
        CV::Int(i) => Op::ConstInt(i, ty.clone()),
        CV::Float(fl) => Op::ConstFloat(fl, ty.clone()),
    };
}

fn fold_bin(b: BinOp, a: CV, c: CV, ty: &MirType) -> Option<CV> {
    match (a, c) {
        (CV::Int(x), CV::Int(y)) if !b.is_float() => fold_int(b, x, y, ty).map(CV::Int),
        (CV::Float(x), CV::Float(y)) if b.is_float() => Some(CV::Float(fold_float(b, x, y))),
        _ => None,
    }
}

fn fold_int(b: BinOp, x: i128, y: i128, ty: &MirType) -> Option<i128> {
    use BinOp::*;
    let r = match b {
        Add => x.wrapping_add(y),
        Sub => x.wrapping_sub(y),
        Mul => x.wrapping_mul(y),
        SDiv => {
            if y == 0 {
                return None;
            }
            x.wrapping_div(y)
        }
        UDiv => {
            if y == 0 {
                return None;
            }
            ((x as u128) / (y as u128)) as i128
        }
        SRem => {
            if y == 0 {
                return None;
            }
            x.wrapping_rem(y)
        }
        URem => {
            if y == 0 {
                return None;
            }
            ((x as u128) % (y as u128)) as i128
        }
        And => x & y,
        Or => x | y,
        Xor => x ^ y,
        Shl => x.wrapping_shl(y as u32),
        LShr => ((x as u128) >> (y as u32)) as i128,
        AShr => x >> (y as u32),
        FAdd | FSub | FMul | FDiv => return None,
    };
    Some(mask(r, ty))
}

fn fold_float(b: BinOp, x: f64, y: f64) -> f64 {
    use BinOp::*;
    match b {
        FAdd => x + y,
        FSub => x - y,
        FMul => x * y,
        FDiv => x / y,
        _ => 0.0,
    }
}

fn fold_cmp(c: CmpOp, a: CV, b: CV) -> i128 {
    use CmpOp::*;
    let res = match (a, b) {
        (CV::Float(x), CV::Float(y)) => match c {
            Foeq => x == y,
            Fone => x != y,
            Folt => x < y,
            Fole => x <= y,
            Fogt => x > y,
            Foge => x >= y,
            _ => false,
        },
        (CV::Int(x), CV::Int(y)) => {
            let (ux, uy) = (x as u128, y as u128);
            match c {
                Eq => x == y,
                Ne => x != y,
                Slt => x < y,
                Sle => x <= y,
                Sgt => x > y,
                Sge => x >= y,
                Ult => ux < uy,
                Ule => ux <= uy,
                Ugt => ux > uy,
                Uge => ux >= uy,
                _ => false,
            }
        }
        _ => false,
    };
    res as i128
}

fn algebra(b: BinOp, l: ValueId, lc: Option<CV>, r: ValueId, rc: Option<CV>) -> Option<Alg> {
    use BinOp::*;
    let is0 = |c: Option<CV>| matches!(c, Some(CV::Int(0)));
    let is1 = |c: Option<CV>| matches!(c, Some(CV::Int(1)));
    Some(match b {
        Add => {
            if is0(rc) {
                Alg::Replace(l)
            } else if is0(lc) {
                Alg::Replace(r)
            } else {
                return None;
            }
        }
        Sub => {
            if is0(rc) {
                Alg::Replace(l)
            } else if l == r {
                Alg::Const(CV::Int(0))
            } else {
                return None;
            }
        }
        Mul => {
            if is1(rc) {
                Alg::Replace(l)
            } else if is1(lc) {
                Alg::Replace(r)
            } else if is0(rc) || is0(lc) {
                Alg::Const(CV::Int(0))
            } else {
                return None;
            }
        }
        SDiv | UDiv => {
            if is1(rc) {
                Alg::Replace(l)
            } else {
                return None;
            }
        }
        Or | Xor => {
            if is0(rc) {
                Alg::Replace(l)
            } else if is0(lc) {
                Alg::Replace(r)
            } else {
                return None;
            }
        }
        And => {
            if is0(rc) || is0(lc) {
                Alg::Const(CV::Int(0))
            } else {
                return None;
            }
        }
        Shl | LShr | AShr => {
            if is0(rc) {
                Alg::Replace(l)
            } else {
                return None;
            }
        }
        _ => return None,
    })
}

fn mask(v: i128, ty: &MirType) -> i128 {
    let bits = match ty {
        MirType::I1 => 1,
        MirType::I8 => 8,
        MirType::I16 => 16,
        MirType::I32 => 32,
        MirType::I64 => 64,
        _ => 128,
    };
    if bits >= 128 {
        return v;
    }
    let shift = 128 - bits;
    (v << shift) >> shift
}
