//! `wukong_types` — the semantic type vocabulary, shared by sema and MIR.
//!
//! These types describe *layout and shape*, the two things Wukong cares most about. A [`Ty`]
//! resolves to a concrete size/alignment (for sized types), and a [`Tensor`](Ty::Tensor) carries
//! its [`Shape`] — the dimensions checked at compile time, which is what makes shape mismatches
//! type errors rather than runtime crashes.
//!
//! This crate is the **layout authority**: [`Ty::size_of`], [`Ty::align_of`] and
//! [`Ty::tuple_offsets`] are the answer, and a 64-bit target is assumed throughout (`usize`/`isize`
//! and every `Ptr`/`Ref` are hard-coded 8/8). Do not recompute a size downstream — the one
//! sanctioned extension is `wukong_mir_build`'s registry-aware `ty_size`/`ty_align`/
//! `aggregate_layout`, which resolves `Ty::Named` through the sema defs and delegates here for
//! everything else.

use wukong_span::{Interner, Symbol};

pub mod dim;

pub use dim::{DimAtom, DimError, DimOp, DimPoly, Monomial};

/// A primitive scalar type.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Scalar {
    I8,
    I16,
    I32,
    I64,
    /// A 128-bit two's-complement integer: 16 bytes, 16-byte aligned (the x86-64 psABI's
    /// `__int128` and Rust's `i128`). One `i128` MIR value; the native backend legalizes the
    /// operations x64 has no instruction for (division, float conversion) into runtime calls.
    I128,
    U8,
    U16,
    U32,
    U64,
    /// The unsigned 128-bit integer (see [`Scalar::I128`]).
    U128,
    Usize,
    Isize,
    F16,
    Bf16,
    F32,
    F64,
    Bool,
    /// A Unicode scalar value (the type of a `'c'` literal). 4 bytes, unsigned; lowers to a 32-bit
    /// integer in MIR, so it is interconvertible with the integer types via `as`.
    Char,
}

impl Scalar {
    pub fn from_name(s: &str) -> Option<Scalar> {
        use Scalar::*;
        Some(match s {
            "i8" => I8,
            "i16" => I16,
            "i32" => I32,
            "i64" => I64,
            "i128" => I128,
            "u8" => U8,
            "u16" => U16,
            "u32" => U32,
            "u64" => U64,
            "u128" => U128,
            "usize" => Usize,
            "isize" => Isize,
            "f16" => F16,
            "bf16" => Bf16,
            "f32" => F32,
            "f64" => F64,
            "bool" => Bool,
            "char" => Char,
            _ => return None,
        })
    }

    pub fn name(self) -> &'static str {
        use Scalar::*;
        match self {
            I8 => "i8",
            I16 => "i16",
            I32 => "i32",
            I64 => "i64",
            I128 => "i128",
            U8 => "u8",
            U16 => "u16",
            U32 => "u32",
            U64 => "u64",
            U128 => "u128",
            Usize => "usize",
            Isize => "isize",
            F16 => "f16",
            Bf16 => "bf16",
            F32 => "f32",
            F64 => "f64",
            Bool => "bool",
            Char => "char",
        }
    }

    /// Size in bytes (usize/isize assume a 64-bit target).
    pub fn size(self) -> u64 {
        use Scalar::*;
        match self {
            I8 | U8 | Bool => 1,
            I16 | U16 | F16 | Bf16 => 2,
            I32 | U32 | F32 | Char => 4,
            I64 | U64 | F64 | Usize | Isize => 8,
            I128 | U128 => 16,
        }
    }

    pub fn align(self) -> u64 {
        self.size()
    }

    pub fn is_float(self) -> bool {
        matches!(self, Scalar::F16 | Scalar::Bf16 | Scalar::F32 | Scalar::F64)
    }

    /// Every non-float, non-`Bool` scalar — so `Char` counts as an integer here (it lowers to a
    /// 32-bit integer in MIR), and so does `Usize`/`Isize`. `is_signed()` is the separate axis.
    pub fn is_int(self) -> bool {
        !self.is_float() && self != Scalar::Bool
    }

    pub fn is_signed(self) -> bool {
        use Scalar::*;
        matches!(self, I8 | I16 | I32 | I64 | I128 | Isize)
    }

    /// The width in bits of an integer scalar (`char` is 32, `bool` 1), `None` for a float.
    pub fn int_bits(self) -> Option<u32> {
        use Scalar::*;
        Some(match self {
            Bool => 1,
            I8 | U8 => 8,
            I16 | U16 => 16,
            I32 | U32 | Char => 32,
            I64 | U64 | Usize | Isize => 64,
            I128 | U128 => 128,
            F16 | Bf16 | F32 | F64 => return None,
        })
    }

    /// The 128-bit integers, `i128` and `u128`.
    pub fn is_128(self) -> bool {
        matches!(self, Scalar::I128 | Scalar::U128)
    }
}

/// One extent of a tensor shape: a compile-time constant, a bound symbolic variable, a runtime-dynamic
/// dimension, or a derived extent computed from other dimensions.
///
/// **Invariant:** a `Dim` is always in normal form. A derived extent that folds to a constant is a
/// `Const`, one that is exactly a single variable is a `Var`, and only a genuinely compound
/// expression is an `Expr` (built through [`Dim::from_poly`] / [`Dim::binary`], never by hand). That
/// is what makes the derived `==` *semantic* equality for everything the normal form can prove:
/// `Tensor[f32, N + N]` and `Tensor[f32, 2*N]` are the same type.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub enum Dim {
    Const(u64),
    Var(Symbol),
    Dynamic,
    /// A derived extent — `3*D`, `H*HD`, `(L - K)/S + 1` — in the canonical polynomial normal form of
    /// [`dim`]. Never a bare constant or a bare variable.
    Expr(Box<DimPoly>),
}

impl Dim {
    /// The extent as a polynomial, or `None` for a runtime `?`.
    pub fn to_poly(&self) -> Option<DimPoly> {
        match self {
            Dim::Const(n) => Some(DimPoly::constant(*n as i128)),
            Dim::Var(s) => Some(DimPoly::var(*s)),
            Dim::Expr(p) => Some((**p).clone()),
            Dim::Dynamic => None,
        }
    }

    /// The normal-form `Dim` of a polynomial: a `Const` when it folds to a constant (which must be a
    /// valid extent — non-negative and within `u64`), a `Var` when it is one variable, else `Expr`.
    pub fn from_poly(p: DimPoly) -> Result<Dim, DimError> {
        if let Some(c) = p.as_constant() {
            if c < 0 {
                return Err(DimError::Negative(c));
            }
            return u64::try_from(c)
                .map(Dim::Const)
                .map_err(|_| DimError::Overflow);
        }
        if let Some(s) = p.as_var() {
            return Ok(Dim::Var(s));
        }
        Ok(Dim::Expr(Box::new(p)))
    }

    /// `a <op> b` in normal form. A runtime `?` operand makes the result `?`.
    pub fn binary(op: DimOp, a: &Dim, b: &Dim) -> Result<Dim, DimError> {
        let (Some(x), Some(y)) = (a.to_poly(), b.to_poly()) else {
            return Ok(Dim::Dynamic);
        };
        let r = match op {
            DimOp::Add => x.add(&y)?,
            DimOp::Sub => x.sub(&y)?,
            DimOp::Mul => x.mul(&y)?,
            DimOp::Div => x.floor_div(&y)?,
            DimOp::Rem => x.rem(&y)?,
        };
        Dim::from_poly(r)
    }

    /// The product of `dims` — the element count of a shape — in normal form (`1` for an empty
    /// list). `?` if any factor is `?`.
    pub fn product(dims: &[Dim]) -> Result<Dim, DimError> {
        let mut acc = Dim::Const(1);
        for d in dims {
            acc = Dim::binary(DimOp::Mul, &acc, d)?;
        }
        Ok(acc)
    }

    /// Substitute each variable `s` bound by `f`, re-normalizing a derived extent. A variable bound
    /// to `?` makes a derived extent that mentions it `?` as well.
    pub fn subst(&self, f: &dyn Fn(Symbol) -> Option<Dim>) -> Result<Dim, DimError> {
        match self {
            Dim::Var(s) => Ok(f(*s).unwrap_or(Dim::Var(*s))),
            Dim::Const(_) | Dim::Dynamic => Ok(self.clone()),
            Dim::Expr(p) => {
                if p.vars().iter().any(|s| matches!(f(*s), Some(Dim::Dynamic))) {
                    return Ok(Dim::Dynamic);
                }
                let q = p.substitute(&|s| f(s).and_then(|d| d.to_poly()))?;
                Dim::from_poly(q)
            }
        }
    }

    /// Every dimension variable this extent mentions, once each, in deterministic order.
    pub fn vars(&self) -> Vec<Symbol> {
        match self {
            Dim::Var(s) => vec![*s],
            Dim::Expr(p) => p.vars(),
            Dim::Const(_) | Dim::Dynamic => Vec::new(),
        }
    }

    /// A symbolic extent: a variable or a derived expression (known per call, not per program).
    pub fn is_symbolic(&self) -> bool {
        matches!(self, Dim::Var(_) | Dim::Expr(_))
    }

    /// Render for a diagnostic or a type display: `512`, `N`, `?`, `2*N + 1`.
    pub fn display(&self, interner: &Interner) -> String {
        match self {
            Dim::Const(n) => n.to_string(),
            Dim::Var(s) => interner.resolve(*s).to_string(),
            Dim::Dynamic => "?".to_string(),
            Dim::Expr(p) => p.display(interner),
        }
    }
}

/// A tensor shape — an ordered list of dimensions; its length is the rank.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct Shape(pub Vec<Dim>);

impl Shape {
    pub fn rank(&self) -> usize {
        self.0.len()
    }

    /// The symbolic (`Var`) dimensions of this shape, in axis order (repeats preserved — a shape
    /// like `[N, N]` yields `[N, N]`; de-duplication across a whole signature is the caller's job).
    /// This feeds the hidden runtime-dim ABI that lets a symbolic-generic tensor function execute —
    /// each such dim is threaded to the callee as a hidden `i64` parameter — but the ABI *order* is
    /// decided by `wukong_mir_build::symbolic_dim_params`, which de-duplicates across the parameter
    /// list. Both sides must keep it a `Vec`: an unordered container here is a nondeterministic ABI.
    ///
    /// A derived extent (`Dim::Expr`, e.g. `2*N`) contributes every variable it mentions, in its
    /// canonical order: the variables are what a caller must supply, the expression is recomputed
    /// from them inside the callee.
    pub fn symbolic_dims(&self) -> Vec<Symbol> {
        self.0.iter().flat_map(|d| d.vars()).collect()
    }
}

/// Physical memory layout of a tensor.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub enum Layout {
    Contiguous,
    ColMajor,
    Strided,
    Tiled(Vec<u64>),
}

/// A semantic type.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub enum Ty {
    Scalar(Scalar),
    Unit,
    Ptr {
        mutable: bool,
        pointee: Box<Ty>,
    },
    Ref {
        mutable: bool,
        pointee: Box<Ty>,
    },
    Slice(Box<Ty>),
    /// `str`: an immutable UTF-8 byte string, laid out exactly like a `[]u8` slice — a 16-byte fat
    /// pointer `(data, byte length)`. It exists only inside `wukong_sema`, which enforces that its
    /// bytes are never written; every type sema publishes has been passed through
    /// [`Ty::erase_str`], so lowering, the optimizer and the backends see a plain `[]u8`.
    Str,
    Array {
        elem: Box<Ty>,
        len: u64,
        /// `true` for `shared [T; N]` — CTA shared memory on GPU, ordinary local on CPU.
        shared: bool,
    },
    Tuple(Vec<Ty>),
    /// A SIMD vector: a power-of-two count of scalar lanes.
    Vector {
        elem: Scalar,
        lanes: u32,
    },
    /// A shape-typed tensor view.
    Tensor {
        elem: Scalar,
        shape: Shape,
        layout: Layout,
        /// A **dtype-generic** element: `Some(T)` for `Tensor[T, M, N]` where `T` is a type
        /// parameter of the enclosing function. `elem` is then a placeholder that means nothing; a
        /// substitution of `T` replaces the pair with the concrete element (and `None`), so every
        /// type a backend lowers carries `None`. Keeping the scalar field a `Scalar` means only the
        /// code that sees a generic body (the checker's opaque-`T` pass and the monomorphizer's
        /// substitution) has to know this field exists.
        elem_var: Option<Symbol>,
    },
    /// A named (struct/enum) type, identified by its interned name.
    Named(Symbol),
    Fn {
        params: Vec<Ty>,
        ret: Box<Ty>,
    },
    /// An as-yet-undetermined type (lenient inference for unmodeled builtins). Does not error.
    Unknown,
    /// A type produced after an error; suppresses cascading diagnostics.
    Error,
}

impl Ty {
    pub fn scalar(s: Scalar) -> Ty {
        Ty::Scalar(s)
    }

    pub fn is_error(&self) -> bool {
        matches!(self, Ty::Error)
    }

    pub fn is_unknown(&self) -> bool {
        matches!(self, Ty::Unknown)
    }

    /// `true` when this type is numeric (a scalar number or a SIMD vector of numbers).
    pub fn is_numeric(&self) -> bool {
        match self {
            Ty::Scalar(s) => s.is_int() || s.is_float(),
            Ty::Vector { elem, .. } => elem.is_int() || elem.is_float(),
            _ => false,
        }
    }

    /// Size in bytes; `None` for an overflowing layout or an unsized/unknown type.
    pub fn size_of(&self) -> Option<u64> {
        match self {
            Ty::Scalar(s) => Some(s.size()),
            Ty::Unit => Some(0),
            Ty::Ptr { .. } | Ty::Ref { .. } => Some(8),
            Ty::Vector { elem, lanes } => Some(elem.size() * *lanes as u64),
            // A slice is a fat pointer — a `(data: *T, len: usize)` view — so it is a sized 16-byte
            // aggregate (8-byte pointer + 8-byte length), even though the *pointee* is unsized. This
            // is what lets a slice be a struct/tuple field and be passed/returned by value.
            Ty::Slice(_) => Some(16),
            // A `str` is a `[]u8` fat pointer.
            Ty::Str => Some(16),
            Ty::Array { elem, len, .. } => elem.size_of()?.checked_mul(*len),
            Ty::Tuple(fields) => {
                let mut size = 0u64;
                let mut align = 1u64;
                for f in fields {
                    let fa = f.align_of()?;
                    let fs = f.size_of()?;
                    size = round_up(size, fa)?.checked_add(fs)?;
                    align = align.max(fa);
                }
                round_up(size, align)
            }
            _ => None,
        }
    }

    /// Alignment in bytes for sized types; `None` for unsized or unknown types.
    /// Size overflow does not affect alignment. Every returned alignment is a power of two.
    pub fn align_of(&self) -> Option<u64> {
        match self {
            Ty::Scalar(s) => Some(s.align()),
            Ty::Unit => Some(1),
            Ty::Ptr { .. } | Ty::Ref { .. } => Some(8),
            // An *alignment*, not the vector's size: `lanes` is documented as a power of two but
            // nothing rejects `f32x3`, and a 12-byte "alignment" makes the two aggregate-layout
            // authorities disagree. Identity for every power-of-two lane count.
            Ty::Vector { elem, lanes } => Some((elem.size() * *lanes as u64).next_power_of_two()),
            // A slice's fat pointer is 8-byte aligned (its data pointer and length are both 8 bytes).
            Ty::Slice(_) | Ty::Str => Some(8),
            Ty::Array { elem, .. } => elem.align_of(),
            Ty::Tuple(fields) => fields
                .iter()
                .try_fold(1u64, |a, f| Some(a.max(f.align_of()?))),
            _ => None,
        }
    }

    /// The padded byte offset and type of each field of a `Tuple`, in declaration order — the
    /// reference definition of aggregate layout (the same `round_up` accumulation as
    /// [`size_of`](Ty::size_of)). `None` for a non-tuple, an unsized field, or an overflowing
    /// complete layout.
    ///
    /// It has **no caller in the workspace**: `wukong_mir_build`'s `FnLowerer::aggregate_layout`
    /// repeats this accumulation because it must resolve `Ty::Named` struct/enum fields through the
    /// sema def registry, which this leaf crate cannot see. The two must therefore agree field-for-
    /// field; `tuple_offsets_are_alignment_correct` pins this side so they cannot drift.
    pub fn tuple_offsets(&self) -> Option<Vec<(u64, Ty)>> {
        let Ty::Tuple(fields) = self else {
            return None;
        };
        let mut out = Vec::with_capacity(fields.len());
        let mut off = 0u64;
        for f in fields {
            off = round_up(off, f.align_of()?)?;
            out.push((off, f.clone()));
            off = off.checked_add(f.size_of()?)?;
        }
        // Tail padding is part of the layout even when every field offset fits.
        round_up(off, self.align_of()?)?;
        Some(out)
    }

    /// A human-readable rendering for diagnostics (resolves interned symbols).
    pub fn display(&self, interner: &Interner) -> String {
        match self {
            Ty::Scalar(s) => s.name().to_string(),
            Ty::Unit => "()".to_string(),
            Ty::Ptr { mutable, pointee } => {
                format!(
                    "*{}{}",
                    if *mutable { "mut " } else { "" },
                    pointee.display(interner)
                )
            }
            Ty::Ref { mutable, pointee } => {
                format!(
                    "&{}{}",
                    if *mutable { "mut " } else { "" },
                    pointee.display(interner)
                )
            }
            Ty::Slice(e) => format!("[]{}", e.display(interner)),
            Ty::Str => "str".to_string(),
            Ty::Array { elem, len, shared } => {
                let inner = format!("[{}; {}]", elem.display(interner), len);
                if *shared {
                    format!("shared {inner}")
                } else {
                    inner
                }
            }
            Ty::Tuple(fields) => {
                let parts: Vec<_> = fields.iter().map(|f| f.display(interner)).collect();
                format!("({})", parts.join(", "))
            }
            Ty::Vector { elem, lanes } => format!("{}x{}", elem.name(), lanes),
            Ty::Tensor {
                elem,
                shape,
                elem_var,
                ..
            } => {
                let mut parts = vec![match elem_var {
                    Some(v) => interner.resolve(*v).to_string(),
                    None => elem.name().to_string(),
                }];
                for d in &shape.0 {
                    parts.push(d.display(interner));
                }
                format!("Tensor[{}]", parts.join(", "))
            }
            Ty::Named(s) => interner.resolve(*s).to_string(),
            Ty::Fn { params, ret } => {
                let ps: Vec<_> = params.iter().map(|p| p.display(interner)).collect();
                format!("fn({}) -> {}", ps.join(", "), ret.display(interner))
            }
            Ty::Unknown => "_".to_string(),
            Ty::Error => "<error>".to_string(),
        }
    }
}

impl Ty {
    /// Whether `str` occurs anywhere in this type: itself, behind a pointer or reference, as a
    /// slice/array element, a tuple field, or a function type's parameter or result. A nominal
    /// type's fields are not looked into (a `Named` is a name here).
    pub fn mentions_str(&self) -> bool {
        match self {
            Ty::Str => true,
            Ty::Ptr { pointee, .. } | Ty::Ref { pointee, .. } | Ty::Slice(pointee) => {
                pointee.mentions_str()
            }
            Ty::Array { elem, .. } => elem.mentions_str(),
            Ty::Tuple(ts) => ts.iter().any(Ty::mentions_str),
            Ty::Fn { params, ret } => params.iter().any(Ty::mentions_str) || ret.mentions_str(),
            _ => false,
        }
    }

    /// This type with every `str` replaced by the `[]u8` slice it is laid out as — recursing
    /// through pointers, references, slices, arrays, tuples and function types. Sema applies it
    /// to every type it publishes, so `str` never reaches lowering: the immutability rules are
    /// sema's alone, and the representation is the slice machinery every backend already runs.
    pub fn erase_str(&self) -> Ty {
        if !self.mentions_str() {
            return self.clone();
        }
        match self {
            Ty::Str => Ty::Slice(Box::new(Ty::Scalar(Scalar::U8))),
            Ty::Ptr { mutable, pointee } => Ty::Ptr {
                mutable: *mutable,
                pointee: Box::new(pointee.erase_str()),
            },
            Ty::Ref { mutable, pointee } => Ty::Ref {
                mutable: *mutable,
                pointee: Box::new(pointee.erase_str()),
            },
            Ty::Slice(e) => Ty::Slice(Box::new(e.erase_str())),
            Ty::Array { elem, len, shared } => Ty::Array {
                elem: Box::new(elem.erase_str()),
                len: *len,
                shared: *shared,
            },
            Ty::Tuple(ts) => Ty::Tuple(ts.iter().map(Ty::erase_str).collect()),
            Ty::Fn { params, ret } => Ty::Fn {
                params: params.iter().map(Ty::erase_str).collect(),
                ret: Box::new(ret.erase_str()),
            },
            other => other.clone(),
        }
    }
}

/// Whether `name` is a built-in type name: a scalar (`f32`, `u8`, …) or `str`. A program may not
/// declare a struct, enum, trait, type alias or generic parameter with one of these names.
pub fn is_builtin_type_name(name: &str) -> bool {
    Scalar::from_name(name).is_some() || name == "str"
}

fn round_up(x: u64, align: u64) -> Option<u64> {
    if align <= 1 {
        Some(x)
    } else {
        x.div_ceil(align).checked_mul(align)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aggregate_layout_overflow_declines() {
        fn array(elem: Ty, len: u64) -> Ty {
            Ty::Array {
                elem: Box::new(elem),
                len,
                shared: false,
            }
        }
        let nested = array(array(Ty::Scalar(Scalar::I64), 4_000_000_000), 4_000_000_000);
        assert_eq!(nested.size_of(), None);
        assert_eq!(
            Ty::Tuple(vec![nested, Ty::Scalar(Scalar::I32)]).tuple_offsets(),
            None
        );
        let full = array(Ty::Scalar(Scalar::U8), u64::MAX);
        // Addition, field padding, and tail padding each overflow independently.
        for fields in [
            vec![full.clone(), Ty::Scalar(Scalar::U8)],
            vec![full.clone(), Ty::Scalar(Scalar::I64)],
            vec![
                array(Ty::Scalar(Scalar::I64), u64::MAX / 8),
                array(Ty::Scalar(Scalar::U8), 7),
            ],
        ] {
            let tuple = Ty::Tuple(fields);
            assert_eq!(tuple.size_of(), None);
            assert_eq!(tuple.tuple_offsets(), None);
        }
        assert_eq!(full.size_of(), Some(u64::MAX));
        assert_eq!(array(Ty::Unit, u64::MAX).size_of(), Some(0));
        assert_eq!(
            array(array(Ty::Scalar(Scalar::I64), 0), u64::MAX).size_of(),
            Some(0)
        );
        assert_eq!(
            Ty::Tuple(vec![Ty::Scalar(Scalar::U8), Ty::Scalar(Scalar::I64)]).size_of(),
            Some(16)
        );
        assert_eq!(Ty::Unknown.size_of(), None);
        assert_eq!(
            Ty::Slice(Box::new(Ty::Scalar(Scalar::I32))).size_of(),
            Some(16)
        );
    }

    #[test]
    fn str_is_a_u8_fat_pointer() {
        let interner = Interner::new();
        assert_eq!(Ty::Str.size_of(), Some(16));
        assert_eq!(Ty::Str.align_of(), Some(8));
        assert_eq!(Ty::Str.display(&interner), "str");
        let u8_slice = Ty::Slice(Box::new(Ty::Scalar(Scalar::U8)));
        assert_eq!(Ty::Str.erase_str(), u8_slice);
        // A struct-like tuple holding a `str` keeps its fat-pointer layout through erasure.
        let t = Ty::Tuple(vec![Ty::Scalar(Scalar::I8), Ty::Str]);
        assert_eq!(t.size_of(), Some(24));
        assert_eq!(t.erase_str().size_of(), Some(24));
        // Erasure recurses through every structural position.
        let nested = Ty::Fn {
            params: vec![
                Ty::Ptr {
                    mutable: false,
                    pointee: Box::new(Ty::Str),
                },
                Ty::Ref {
                    mutable: true,
                    pointee: Box::new(Ty::Slice(Box::new(Ty::Str))),
                },
                Ty::Array {
                    elem: Box::new(Ty::Str),
                    len: 3,
                    shared: false,
                },
                Ty::Tuple(vec![Ty::Str, Ty::Scalar(Scalar::I32)]),
            ],
            ret: Box::new(Ty::Str),
        };
        assert!(nested.mentions_str());
        let erased = nested.erase_str();
        assert!(!erased.mentions_str());
        assert_eq!(
            erased.display(&interner),
            "fn(*[]u8, &mut [][]u8, [[]u8; 3], ([]u8, i32)) -> []u8"
        );
        // A type without `str` is returned unchanged.
        let plain = Ty::Array {
            elem: Box::new(Ty::Scalar(Scalar::F32)),
            len: 4,
            shared: true,
        };
        assert!(!plain.mentions_str());
        assert_eq!(plain.erase_str(), plain);
        assert!(is_builtin_type_name("str"));
        assert!(is_builtin_type_name("u8"));
        assert!(!is_builtin_type_name("String"));
    }

    #[test]
    fn scalar_sizes() {
        assert_eq!(Scalar::F32.size(), 4);
        assert_eq!(Scalar::F64.align(), 8);
        assert_eq!(Scalar::Bf16.size(), 2);
        assert!(Scalar::F32.is_float());
        assert!(Scalar::I32.is_signed());
        assert!(!Scalar::U32.is_signed());
    }

    #[test]
    fn vector_and_array_sizes() {
        let v = Ty::Vector {
            elem: Scalar::F32,
            lanes: 8,
        };
        assert_eq!(v.size_of(), Some(32));
        assert_eq!(v.align_of(), Some(32));
        let a = Ty::Array {
            elem: Box::new(Ty::Scalar(Scalar::I32)),
            len: 4,
            shared: false,
        };
        assert_eq!(a.size_of(), Some(16));
    }

    #[test]
    fn tuple_layout_has_padding() {
        // (i8, i32) -> 1 byte + 3 pad + 4 = 8, align 4.
        let t = Ty::Tuple(vec![Ty::Scalar(Scalar::I8), Ty::Scalar(Scalar::I32)]);
        assert_eq!(t.align_of(), Some(4));
        assert_eq!(t.size_of(), Some(8));
    }

    #[test]
    fn vector_align_is_a_power_of_two() {
        // `f32x3` parses today (wukong_parser's `split_vector_ident` accepts any digit run) and
        // nothing rejects a non-power-of-two lane count, so `align_of` must not hand one out:
        // every consumer of an alignment assumes a power of two.
        for lanes in 1u32..=17 {
            for elem in [Scalar::I8, Scalar::F16, Scalar::F32, Scalar::F64] {
                let a = Ty::Vector { elem, lanes }.align_of().unwrap();
                assert!(
                    a.is_power_of_two(),
                    "{}x{lanes} reported align {a}",
                    elem.name()
                );
            }
        }
    }

    #[test]
    fn align_of_satisfies_the_mir_build_round_up_mirror() {
        // `wukong_mir_build::round_up` is documented as a mirror of the `round_up` here, but it is
        // the bitmask form `(x + align - 1) & !(align - 1)`, which agrees with `div_ceil` only when
        // `align` is a power of two. This pins the precondition from our side: every alignment
        // `align_of` can produce must make the two forms identical, so the two layout authorities
        // (`Ty::size_of`/`tuple_offsets` here, `FnLowerer::aggregate_layout` there) cannot diverge.
        fn bitmask_round_up(x: u64, align: u64) -> u64 {
            if align <= 1 {
                x
            } else {
                (x + align - 1) & !(align - 1)
            }
        }
        let tys = [
            Ty::Scalar(Scalar::I8),
            Ty::Scalar(Scalar::F64),
            Ty::Slice(Box::new(Ty::Scalar(Scalar::F32))),
            Ty::Vector {
                elem: Scalar::F32,
                lanes: 3,
            },
            Ty::Vector {
                elem: Scalar::F32,
                lanes: 8,
            },
            Ty::Vector {
                elem: Scalar::Bf16,
                lanes: 6,
            },
        ];
        for t in &tys {
            let a = t.align_of().unwrap();
            for x in 0u64..64 {
                assert_eq!(
                    round_up(x, a),
                    Some(bitmask_round_up(x, a)),
                    "round_up({x}, {a})"
                );
            }
        }
    }

    #[test]
    fn scalar_from_name_round_trips() {
        for s in [
            Scalar::Bool,
            Scalar::I8,
            Scalar::U8,
            Scalar::I32,
            Scalar::U64,
            Scalar::I128,
            Scalar::U128,
            Scalar::Usize,
            Scalar::F16,
            Scalar::Bf16,
            Scalar::F32,
            Scalar::F64,
        ] {
            let name = s.name();
            assert_eq!(Scalar::from_name(name), Some(s), "round-trip {name}");
        }
        assert_eq!(Scalar::from_name("not_a_type"), None);
    }

    #[test]
    fn tuple_offsets_are_alignment_correct() {
        // `tuple_offsets` is `pub` and documented as the aggregate-layout authority, but it has no
        // callers in the workspace and had no test; pin it against `size_of`/`align_of` so it
        // cannot drift from the accumulation `wukong_mir_build`'s `aggregate_layout` performs.
        let t = Ty::Tuple(vec![
            Ty::Scalar(Scalar::I8),
            Ty::Scalar(Scalar::I32),
            Ty::Vector {
                elem: Scalar::F32,
                lanes: 4,
            },
        ]);
        let offs: Vec<u64> = t.tuple_offsets().unwrap().iter().map(|(o, _)| *o).collect();
        assert_eq!(offs, vec![0, 4, 16]);
        assert_eq!(t.align_of(), Some(16));
        assert_eq!(t.size_of(), Some(32));

        // The non-power-of-two lane count that motivated the `align_of` correction.
        let odd = Ty::Tuple(vec![
            Ty::Scalar(Scalar::I32),
            Ty::Vector {
                elem: Scalar::F32,
                lanes: 3,
            },
        ]);
        assert_eq!(odd.tuple_offsets().unwrap()[1].0, 16);
        assert_eq!(odd.size_of(), Some(32));

        // Every field sits at a multiple of its own alignment and fits inside the aggregate.
        for ty in [&t, &odd] {
            let size = ty.size_of().unwrap();
            for (off, f) in ty.tuple_offsets().unwrap() {
                assert_eq!(off % f.align_of().unwrap(), 0, "field at {off}");
                assert!(
                    off + f.size_of().unwrap() <= size,
                    "field at {off} overruns"
                );
            }
        }
        assert!(Ty::Scalar(Scalar::I32).tuple_offsets().is_none());
    }

    #[test]
    fn nested_array_size_and_pointer_layout() {
        // [[i32; 4]; 3] -> 3 * (4 * 4) = 48 bytes.
        let inner = Ty::Array {
            elem: Box::new(Ty::Scalar(Scalar::I32)),
            len: 4,
            shared: false,
        };
        let outer = Ty::Array {
            elem: Box::new(inner),
            len: 3,
            shared: false,
        };
        assert_eq!(outer.size_of(), Some(48));
    }

    #[test]
    fn display_tensor() {
        let mut i = Interner::new();
        let m = i.intern("M");
        let n = i.intern("N");
        let t = Ty::Tensor {
            elem: Scalar::F32,
            shape: Shape(vec![Dim::Var(m), Dim::Var(n)]),
            layout: Layout::Contiguous,
            elem_var: None,
        };
        assert_eq!(t.display(&i), "Tensor[f32, M, N]");
        // A dtype-generic element displays as its type parameter.
        let tv = i.intern("T");
        let g = Ty::Tensor {
            elem: Scalar::F32,
            shape: Shape(vec![Dim::Var(m)]),
            layout: Layout::Contiguous,
            elem_var: Some(tv),
        };
        assert_eq!(g.display(&i), "Tensor[T, M]");
    }

    fn v(s: Symbol) -> Dim {
        Dim::Var(s)
    }

    fn c(n: u64) -> Dim {
        Dim::Const(n)
    }

    fn bin(op: DimOp, a: &Dim, b: &Dim) -> Dim {
        Dim::binary(op, a, b).expect("valid dimension expression")
    }

    #[test]
    fn dimension_expressions_have_one_normal_form() {
        let mut i = Interner::new();
        let (n, h, hd) = (i.intern("N"), i.intern("H"), i.intern("HD"));
        // `N + N` and `2*N` are one type; so are `H*HD` and `HD*H`.
        assert_eq!(bin(DimOp::Add, &v(n), &v(n)), bin(DimOp::Mul, &c(2), &v(n)));
        assert_eq!(
            bin(DimOp::Mul, &v(h), &v(hd)),
            bin(DimOp::Mul, &v(hd), &v(h))
        );
        // Folding never leaves a compound form for a constant or a single variable.
        assert_eq!(bin(DimOp::Mul, &c(3), &c(4)), c(12));
        assert_eq!(bin(DimOp::Sub, &bin(DimOp::Add, &v(n), &c(1)), &c(1)), v(n));
        assert_eq!(bin(DimOp::Div, &bin(DimOp::Mul, &c(3), &v(n)), &c(3)), v(n));
        // `(2*N + 3) / 2` is `N + 1` for every integer `N` (the residual 3 splits as 2*1 + 1).
        let two_n_3 = bin(DimOp::Add, &bin(DimOp::Mul, &c(2), &v(n)), &c(3));
        assert_eq!(
            bin(DimOp::Div, &two_n_3, &c(2)),
            bin(DimOp::Add, &v(n), &c(1))
        );
        assert_eq!(bin(DimOp::Rem, &two_n_3, &c(2)), c(1));
        // A residual that is not constant stays an atom, and prints as written.
        let half = bin(DimOp::Div, &bin(DimOp::Add, &v(n), &c(1)), &c(2));
        assert!(matches!(half, Dim::Expr(_)));
        assert_eq!(half.display(&i), "(N + 1)/2");
        // A non-constant divisor is never cancelled: `(H*HD)/H` is not `HD` at `H = 0`.
        let hhd = bin(DimOp::Mul, &v(h), &v(hd));
        assert!(matches!(bin(DimOp::Div, &hhd, &v(h)), Dim::Expr(_)));
        assert_eq!(bin(DimOp::Sub, &v(n), &c(1)).display(&i), "N - 1");
        assert_eq!(bin(DimOp::Mul, &c(3), &v(n)).display(&i), "3*N");
    }

    #[test]
    fn dimension_expression_errors_are_reported_not_wrapped() {
        assert_eq!(
            Dim::binary(DimOp::Sub, &c(2), &c(5)),
            Err(DimError::Negative(-3))
        );
        assert_eq!(
            Dim::binary(DimOp::Div, &c(8), &c(0)),
            Err(DimError::BadDivisor(0))
        );
        assert_eq!(
            Dim::binary(DimOp::Mul, &c(u64::MAX), &c(2)),
            Err(DimError::Overflow)
        );
        // A `?` operand is unknown, not an error.
        assert_eq!(
            Dim::binary(DimOp::Mul, &Dim::Dynamic, &c(2)),
            Ok(Dim::Dynamic)
        );
    }

    #[test]
    fn substitution_and_linear_solving() {
        let mut i = Interner::new();
        let (n, m) = (i.intern("N"), i.intern("M"));
        let e = bin(DimOp::Add, &bin(DimOp::Mul, &c(2), &v(n)), &v(m));
        // Bind N := 3 (M stays symbolic), then M := 4.
        let partial = e
            .subst(&|s| (s == n).then_some(c(3)))
            .expect("substitution");
        assert_eq!(partial, bin(DimOp::Add, &v(m), &c(6)));
        let full = partial
            .subst(&|s| (s == m).then_some(c(4)))
            .expect("substitution");
        assert_eq!(full, c(10));
        // A binding to `?` makes the derived extent `?`.
        assert_eq!(
            e.subst(&|s| (s == m).then_some(Dim::Dynamic)),
            Ok(Dim::Dynamic)
        );
        // `2*N + M` is linear in N with coefficient 2 and remainder M.
        let p = e.to_poly().unwrap();
        let (a, rest) = p.linear_in(n).expect("linear in N");
        assert_eq!(a, 2);
        assert_eq!(Dim::from_poly(rest), Ok(v(m)));
        // `N*M` is not linear in N with a constant coefficient.
        assert!(bin(DimOp::Mul, &v(n), &v(m))
            .to_poly()
            .unwrap()
            .linear_in(n)
            .is_none());
        // Evaluation agrees with the folded value.
        let three_n = bin(DimOp::Mul, &c(3), &v(n)).to_poly().unwrap();
        assert_eq!(three_n.eval(&|s| (s == n).then_some(7)), Some(21));
        // The symbolic dims of a shape include the variables of derived extents.
        let sh = Shape(vec![e.clone(), v(n)]);
        assert_eq!(sh.symbolic_dims(), vec![n, m, n]);
    }
}
