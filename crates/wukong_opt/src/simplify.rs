//! Constant folding and algebraic simplification.

use crate::fxhash::FxHashMap;

use wukong_mir::{BinOp, CastKind, CmpOp, Function, MirType, Op, ValueId};

use crate::{map_op_uses, map_term_uses, CfgAnalyses, Pass};

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

/// A foldable instruction reduced to its Copy-sized operands, so classifying it borrows the op only
/// briefly (no whole-`Op` clone — which for `Call`/`Gep` would copy heap fields on every pass).
enum Act {
    Bin(BinOp, ValueId, ValueId),
    Cmp(CmpOp, ValueId, ValueId),
    Neg(ValueId),
    Not(ValueId),
    Cast(CastKind, ValueId),
}

impl Pass for Simplify {
    fn name(&self) -> &'static str {
        "simplify"
    }

    fn run_function(&self, f: &mut Function, _cache: &mut CfgAnalyses) -> bool {
        // Rewrites operands and folds constants — never changes the block graph, so the CFG cache
        // stays valid (untouched).
        let mut changed = false;
        let mut consts: FxHashMap<u32, CV> = FxHashMap::default();
        let mut subst: FxHashMap<u32, ValueId> = FxHashMap::default();

        let nblocks = f.blocks.len();
        for bi in 0..nblocks {
            let ninsts = f.blocks[bi].insts.len();
            for ii in 0..ninsts {
                // Rewrite operands through substitutions discovered so far. Skipped while `subst` is
                // empty (the common early-pass case), where every rewrite is a no-op v -> v.
                if !subst.is_empty() {
                    let s = &subst;
                    map_op_uses(&mut f.blocks[bi].insts[ii].op, |v| resolve(s, v));
                }
                let Some(res) = f.blocks[bi].insts[ii].result else {
                    continue;
                };

                // Classify by *borrowing* the op and copying out only its Copy operands, so the
                // borrow of `f` ends here — no whole-`Op` clone, and the result type is cloned only
                // in the arms that actually fold (below), not for every instruction.
                let act = match &f.blocks[bi].insts[ii].op {
                    Op::ConstInt(v, _) => {
                        consts.insert(res.0, CV::Int(*v));
                        continue;
                    }
                    Op::ConstFloat(v, _) => {
                        // Round to the constant's own precision (an f32 literal may carry f64 bits)
                        // so it enters the fold chain at the right precision. (`rty` was per-inst on
                        // the language branch; main folds lazily, so read the result type here.)
                        let rty = f.value_types[res.0 as usize].clone();
                        consts.insert(res.0, CV::Float(round_float_to_ty(*v, &rty)));
                        continue;
                    }
                    Op::Bin(b, l, r) => Act::Bin(*b, *l, *r),
                    Op::Cmp(c, l, r) => Act::Cmp(*c, *l, *r),
                    Op::Neg(v) => Act::Neg(*v),
                    Op::Not(v) => Act::Not(*v),
                    Op::Cast(k, v, _) => Act::Cast(*k, *v),
                    _ => continue,
                };

                match act {
                    Act::Bin(b, l, r) => {
                        let lc = consts.get(&l.0).copied();
                        let rc = consts.get(&r.0).copied();
                        if let (Some(a), Some(c)) = (lc, rc) {
                            let rty = f.value_types[res.0 as usize].clone();
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
                                    let rty = f.value_types[res.0 as usize].clone();
                                    // `x - x`/`x ^ x`/`x & 0` on a *vector* result would need a
                                    // splatted zero, not the scalar `ConstInt` `set_const` writes —
                                    // which is malformed MIR the verifier rejects. Only the
                                    // `l == r` identities reach here with a vector result (a lane
                                    // operand can never be a scalar const, so `is0`/`is1` cannot
                                    // fire), and declining the fold is correct and conservative.
                                    if !rty.is_vector() {
                                        set_const(f, bi, ii, cv, &rty);
                                        consts.insert(res.0, cv);
                                        changed = true;
                                    }
                                }
                            }
                        }
                    }
                    Act::Cmp(c, l, r) => {
                        if let (Some(a), Some(bv)) =
                            (consts.get(&l.0).copied(), consts.get(&r.0).copied())
                        {
                            // The operands share a type (you compare like-typed values); fold at
                            // that width so a high-bit-set constant compares the same as the runtime
                            // register. Without the width, a signed compare of an unsigned literal
                            // >= 2^(w-1) (or an `as iW` reinterpret) folds against the raw i128 and
                            // -O2 disagrees with -O0.
                            let oty = f.value_types[l.0 as usize].clone();
                            // Do not fold a bf16/f16 comparison, for the same reason `fold_bin`
                            // refuses bf16/f16 arithmetic: `round_float_to_ty` rounds these to the
                            // f32 grid, not the narrow grid the backends round to at runtime. Two
                            // distinct f32 literals sharing one bf16 grid point compare EQUAL when
                            // the op runs and "not equal" when folded here — an -O0/-O2 branch flip.
                            if !matches!(oty, MirType::BF16 | MirType::F16) {
                                let val = fold_cmp(c, a, bv, &oty);
                                set_const(f, bi, ii, CV::Int(val), &MirType::I1);
                                consts.insert(res.0, CV::Int(val));
                                changed = true;
                            }
                        } else if l == r && !f.value_types[res.0 as usize].is_vector() {
                            // Integer self-comparison is constant. Float self-comparison is NOT
                            // (NaN != NaN), so only fold the integer predicates. A lane-wise
                            // compare has a vector-typed result, which cannot hold the scalar `i1`
                            // const this writes — decline it, as the `Alg::Const` arm does.
                            if let Some(val) = fold_cmp_self(c) {
                                set_const(f, bi, ii, CV::Int(val), &MirType::I1);
                                consts.insert(res.0, CV::Int(val));
                                changed = true;
                            }
                        }
                    }
                    Act::Neg(v) => {
                        if let Some(cv) = consts.get(&v.0).copied() {
                            let rty = f.value_types[res.0 as usize].clone();
                            let nv = match cv {
                                CV::Int(i) => CV::Int(mask(i.wrapping_neg(), &rty)),
                                CV::Float(fl) => CV::Float(round_float_to_ty(-fl, &rty)),
                            };
                            set_const(f, bi, ii, nv, &rty);
                            consts.insert(res.0, nv);
                            changed = true;
                        }
                    }
                    Act::Not(v) => {
                        if let Some(CV::Int(i)) = consts.get(&v.0).copied() {
                            let rty = f.value_types[res.0 as usize].clone();
                            let nv = CV::Int(mask(!i, &rty));
                            set_const(f, bi, ii, nv, &rty);
                            consts.insert(res.0, nv);
                            changed = true;
                        }
                    }
                    Act::Cast(k, v) => {
                        if let Some(CV::Int(i)) = consts.get(&v.0).copied() {
                            let from = f.value_types[v.0 as usize].clone();
                            let rty = f.value_types[res.0 as usize].clone();
                            if let Some(nv) = fold_int_cast(k, i, &from, &rty) {
                                set_const(f, bi, ii, CV::Int(nv), &rty);
                                consts.insert(res.0, CV::Int(nv));
                                changed = true;
                            }
                        }
                    }
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

fn resolve(subst: &FxHashMap<u32, ValueId>, v: ValueId) -> ValueId {
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

/// Round a folded float constant to the precision of its MIR type, so constant folding matches the
/// per-op rounding the backends do at runtime. Without this an f32 *chain* is folded entirely in f64
/// and only narrowed at the final store, so `-O0` (real f32 arithmetic) and `-O2` (folded) disagree
/// — e.g. `(2^24 + 1) - 2^24` is `0` in f32 but `1` in f64. (f16/bf16 are computed as f32 here, like
/// the backends.) Applied both when a `ConstFloat` is read into the fold table and on every fold
/// result, so an f32 operand and the running value stay at f32 precision through the whole chain.
fn round_float_to_ty(v: f64, ty: &MirType) -> f64 {
    match ty {
        MirType::F32 | MirType::F16 | MirType::BF16 => v as f32 as f64,
        _ => v,
    }
}

/// Fold an **integer-to-integer** cast of a constant, mirroring `wukong_interp::apply_cast` exactly:
/// `SExt`/`Trunc` re-mask the sign-extended value to the target width, `ZExt` first reads the source
/// in its own width as unsigned. The interpreter is the language spec, so this table is a copy of
/// its arms, not a re-derivation.
///
/// Only int -> int is folded. The other ten `CastKind`s are declined on purpose:
/// * anything touching a float would have to reproduce the backends' *single* rounding step, and
///   `round_float_to_ty` rounds bf16/f16 to the f32 grid rather than the narrow grid the backends
///   use at runtime — the same reason `fold_bin` refuses bf16/f16 arithmetic and `Act::Cmp` refuses
///   to compare them (see `narrow_float_constant_compares_are_not_folded`);
/// * `Bitcast`, `IntToPtr` and `PtrToInt` are representation reinterpretations, and a folded pointer
///   constant is not something the rest of the pipeline is prepared for.
///
/// A vector operand or result is declined too: `set_const` writes a *scalar* `ConstInt`, which is
/// malformed MIR for a `<N x iW>` result, exactly as in the `Alg::Const` arm above.
fn fold_int_cast(k: CastKind, x: i128, from: &MirType, to: &MirType) -> Option<i128> {
    if !from.is_int() || !to.is_int() {
        return None;
    }
    Some(match k {
        CastKind::SExt | CastKind::Trunc => mask(x, to),
        CastKind::ZExt => mask(uval(x, int_bits(from)) as i128, to),
        _ => return None,
    })
}

fn fold_bin(b: BinOp, a: CV, c: CV, ty: &MirType) -> Option<CV> {
    match (a, c) {
        (CV::Int(x), CV::Int(y)) if !b.is_float() => fold_int(b, x, y, ty).map(CV::Int),
        // Do not fold bf16/f16 arithmetic. The backends compute these in f32 and round to the narrow
        // grid only at the store/cast, so a *runtime* `a + b` of two grid consts rounds correctly — but
        // folding here would combine the UNROUNDED `ConstFloat` operands (the MIR const carries the
        // literal f32, not its bf16-grid value) and bake a wrong-grid constant, making -O2 disagree
        // with -O0. Leaving the op to run keeps both levels identical (and bf16/f16 are storage
        // formats — seldom compile-time-constant operands, so the lost fold barely matters).
        (CV::Float(x), CV::Float(y))
            if b.is_float() && !matches!(ty, MirType::BF16 | MirType::F16) =>
        {
            Some(CV::Float(round_float_to_ty(fold_float(b, x, y), ty)))
        }
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
            let w = int_bits(ty);
            let (xu, yu) = (uval(x, w), uval(y, w));
            if yu == 0 {
                return None;
            }
            (xu / yu) as i128
        }
        SRem => {
            if y == 0 {
                return None;
            }
            x.wrapping_rem(y)
        }
        URem => {
            let w = int_bits(ty);
            let (xu, yu) = (uval(x, w), uval(y, w));
            if yu == 0 {
                return None;
            }
            (xu % yu) as i128
        }
        And => x & y,
        Or => x | y,
        Xor => x ^ y,
        // Shift count masked to the result width (x86/Cranelift + interpreter semantics):
        // `1i32 << 32 == 1`. `LShr` is logical, so shift the value's own unsigned width window;
        // `AShr` is arithmetic, so sign-extend the value to its width first. Mirrors the interpreter.
        Shl => {
            let w = int_bits(ty);
            x.wrapping_shl((y as u32) & (w - 1))
        }
        LShr => {
            let w = int_bits(ty);
            (uval(x, w) >> ((y as u32) & (w - 1))) as i128
        }
        AShr => {
            let w = int_bits(ty);
            mask(x, ty) >> ((y as u32) & (w - 1))
        }
        FAdd | FSub | FMul | FDiv | FRem => return None,
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
        FRem => x % y,
        _ => 0.0,
    }
}

fn fold_cmp(c: CmpOp, a: CV, b: CV, ty: &MirType) -> i128 {
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
            // Reproduce the width-correct register compare both backends do at runtime. A constant
            // can enter the fold table wider than its type (an unsigned literal >= 2^(w-1), or an
            // `as iW` reinterpret keeps the source bits), so signed predicates compare the
            // sign-extended value at the operand width (`mask`) and unsigned predicates the
            // zero-extended value (`uval`) — mirroring `fold_int`. Comparing the raw i128 instead
            // folds e.g. `(4000000000 as i32) < 0` to false while the runtime i32 register is
            // negative, so -O2 would disagree with -O0.
            let w = int_bits(ty);
            let (sx, sy) = (mask(x, ty), mask(y, ty));
            let (ux, uy) = (uval(x, w), uval(y, w));
            match c {
                Eq => sx == sy,
                Ne => sx != sy,
                Slt => sx < sy,
                Sle => sx <= sy,
                Sgt => sx > sy,
                Sge => sx >= sy,
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

/// Fold `x <cmp> x` for integer predicates (reflexive comparisons). Returns `None` for the
/// float predicates, where `x == x` is false when `x` is NaN.
fn fold_cmp_self(c: CmpOp) -> Option<i128> {
    use CmpOp::*;
    Some(match c {
        Eq | Sle | Sge | Ule | Uge => 1,
        Ne | Slt | Sgt | Ult | Ugt => 0,
        // Float predicates: not safe to fold on a possibly-NaN value.
        Foeq | Fone | Folt | Fole | Fogt | Foge => return None,
    })
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
        SRem | URem => {
            // x % 1 == 0 for any x.
            if is1(rc) {
                Alg::Const(CV::Int(0))
            } else {
                return None;
            }
        }
        Or => {
            if is0(rc) {
                Alg::Replace(l)
            } else if is0(lc) {
                Alg::Replace(r)
            } else if l == r {
                // x | x == x
                Alg::Replace(l)
            } else {
                return None;
            }
        }
        Xor => {
            if is0(rc) {
                Alg::Replace(l)
            } else if is0(lc) {
                Alg::Replace(r)
            } else if l == r {
                // x ^ x == 0
                Alg::Const(CV::Int(0))
            } else {
                return None;
            }
        }
        And => {
            if is0(rc) || is0(lc) {
                Alg::Const(CV::Int(0))
            } else if l == r {
                // x & x == x
                Alg::Replace(l)
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
    // `i1` is a boolean: keep the low bit unsigned (true == 1), never sign-extend to -1.
    if matches!(ty, MirType::I1) {
        return v & 1;
    }
    let bits = match ty {
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

fn int_bits(ty: &MirType) -> u32 {
    match ty {
        MirType::I1 => 1,
        MirType::I8 => 8,
        MirType::I16 => 16,
        MirType::I32 => 32,
        MirType::I64 => 64,
        _ => 64,
    }
}

/// Reinterpret the low `bits` of a (sign-extended) value as unsigned. Constants are folded in the
/// same sign-extended `i128` the interpreter uses, so unsigned folds (`UDiv`/`URem`/`LShr`) must read
/// only the value's own width as unsigned — matching `wukong_interp` so `-O1` equals `-O0`.
fn uval(v: i128, bits: u32) -> u128 {
    if bits >= 128 {
        v as u128
    } else {
        (v as u128) & ((1u128 << bits) - 1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use wukong_mir::CastKind::*;
    use wukong_span::{Interner, SourceId};

    /// Compile, optimize at `opt`, and run `main` in the interpreter — the language spec.
    fn run(src: &str, opt: u8) -> i64 {
        let mut interner = Interner::new();
        let (module, pd) = wukong_parser::parse_module(src, SourceId(0), &mut interner);
        assert!(pd.iter().all(|d| !d.is_error()), "parse: {pd:?}");
        let (sema, sd) = wukong_sema::check(&module, &interner);
        assert!(sd.iter().all(|d| !d.is_error()), "sema: {sd:?}");
        let (mut program, _) = wukong_mir_build::lower_program(&module, &sema, &mut interner);
        crate::optimize(&mut program, opt);
        for f in &program.funcs {
            assert!(
                wukong_mir::verify::verify_function(f).is_empty(),
                "verify after -O{opt}"
            );
        }
        let main = interner.intern("main");
        wukong_interp::run(&program, main, &interner).unwrap()
    }

    #[test]
    fn int_cast_folding_matches_the_interpreter() {
        // Every arm of `fold_int_cast`, at the widths where a wrong rule shows: `zext` of a
        // high-bit-set narrow value (the case that would come out negative if it were a `sext`),
        // `trunc` that drops a sign bit, and `sext` of a negative.
        let cases: [(CastKind, i128, MirType, MirType, i128); 8] = [
            (SExt, -1, MirType::I32, MirType::I64, -1),
            (SExt, 1, MirType::I32, MirType::I64, 1),
            (SExt, i32::MIN as i128, MirType::I32, MirType::I64, i32::MIN as i128),
            // `u32::MAX` is stored sign-extended as -1; zero-extending reads its own 32 bits.
            (ZExt, -1, MirType::I32, MirType::I64, u32::MAX as i128),
            (ZExt, -1, MirType::I8, MirType::I32, u8::MAX as i128),
            (ZExt, 1, MirType::I1, MirType::I32, 1),
            // 0x1_0000_0080 truncated to i8 is 0x80 == -128, not +128.
            (Trunc, 0x1_0000_0080, MirType::I64, MirType::I8, -128),
            (Trunc, 0x1_0000_0000, MirType::I64, MirType::I32, 0),
        ];
        for (k, x, from, to, want) in cases {
            assert_eq!(
                fold_int_cast(k, x, &from, &to),
                Some(want),
                "{} {x} : {} -> {}",
                k.name(),
                from.display(),
                to.display()
            );
        }
        // Float, pointer and representation casts are declined outright.
        for k in [FpToSi, SiToFp, FpExt, FpTrunc, Bitcast, IntToPtr, PtrToInt] {
            assert_eq!(fold_int_cast(k, 1, &MirType::I32, &MirType::I64), None, "{}", k.name());
        }
        // A vector operand or result would need a splatted const, which `set_const` cannot write.
        let v4 = MirType::Vec(Box::new(MirType::I32), 4);
        assert_eq!(fold_int_cast(SExt, 1, &v4, &MirType::I64), None);
        assert_eq!(fold_int_cast(SExt, 1, &MirType::I32, &v4), None);
    }

    #[test]
    fn folded_casts_agree_with_the_unoptimized_program() {
        // The differential that matters: whatever the fold computes must equal what `-O0` (a real
        // `Op::Cast` executed by the interpreter) computes. Sign/zero-extension of a high-bit-set
        // value and a truncation that flips a sign bit are where a wrong rule diverges.
        let cases = [
            // u32::MAX widened: zero-extension, so 4294967295, not -1.
            ("fn main() -> i32 { let a: u32 = 4294967295; let b: i64 = a as i64; \
              if b == 4294967295 { return 1; } return 0; }", 1),
            // i32 -1 widened: sign-extension.
            ("fn main() -> i32 { let a: i32 = -1; let b: i64 = a as i64; \
              if b == -1 { return 1; } return 0; }", 1),
            // Truncation to i8 keeps the low byte, sign-extended.
            ("fn main() -> i32 { let a: i64 = 4294967424; let b: i8 = a as i8; return b as i32; }", -128),
            // u8 widened to i32 is unsigned.
            ("fn main() -> i32 { let a: u8 = 255; return a as i32; }", 255),
            // Round trip through a narrow type inside an expression the folder now sees through.
            ("fn main() -> i32 { let a: i32 = 300; return ((a as i8) as i32) + 1; }", 45),
            // bool widening.
            ("fn main() -> i32 { let a: bool = true; return a as i32; }", 1),
        ];
        for (src, want) in cases {
            for lvl in [0, 1, 2, 3] {
                assert_eq!(run(src, lvl), want as i64, "-O{lvl}: {src}");
            }
        }
    }

    #[test]
    fn a_widened_loop_step_folds_to_a_literal() {
        // The canonicalization this fold exists for: `while i < n { ...; i = i + 1; }` with an
        // `i64` counter lowers the literal `1` as `sext (const.i32 1) to i64`, so the loop's step
        // is an opaque value and `loop_info` reports `IvStep::Invariant`, not `Const(1)`. After
        // folding it is a plain `const.i64 1` and the `while` spelling describes the same induction
        // variable as `for i in 0..n`.
        let src = "fn t(n: i64) -> i64 { let mut s: i64 = 0; let mut i: i64 = 0; \
                   while i < n { s = s + i; i = i + 1; } return s; } \
                   fn main() -> i32 { return t(10) as i32; }";
        let mut interner = Interner::new();
        let (module, _) = wukong_parser::parse_module(src, SourceId(0), &mut interner);
        let (sema, sd) = wukong_sema::check(&module, &interner);
        assert!(sd.iter().all(|d| !d.is_error()), "sema: {sd:?}");
        let (mut program, _) = wukong_mir_build::lower_program(&module, &sema, &mut interner);
        crate::PassManager::standard(2).run(&mut program);
        let sym = interner.intern("t");
        let f = program.function(sym).expect("fn t");
        let forest = crate::loop_info::analyze_function(f);
        assert_eq!(forest.loops.len(), 1);
        let iv = forest.loops[0].primary().expect("a primary IV");
        assert_eq!(
            iv.step,
            crate::loop_info::IvStep::Const(1),
            "the widened literal step must fold to a constant\n{}",
            crate::loop_info::dump_function_loops(f, "t")
        );
        assert_eq!(run(src, 2), 45);
        assert_eq!(run(src, 0), 45);
    }
}
