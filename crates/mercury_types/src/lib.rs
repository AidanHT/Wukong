//! `mercury_types` — the semantic type vocabulary, shared by sema and MIR.
//!
//! These types describe *layout and shape*, the two things Mercury cares most about. A [`Ty`]
//! resolves to a concrete size/alignment (for sized types), and a [`Tensor`](Ty::Tensor) carries
//! its [`Shape`] — the dimensions checked at compile time, which is what makes shape mismatches
//! type errors rather than runtime crashes.

use mercury_span::{Interner, Symbol};

/// A primitive scalar type.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Scalar {
    I8,
    I16,
    I32,
    I64,
    U8,
    U16,
    U32,
    U64,
    Usize,
    Isize,
    F16,
    Bf16,
    F32,
    F64,
    Bool,
}

impl Scalar {
    pub fn from_name(s: &str) -> Option<Scalar> {
        use Scalar::*;
        Some(match s {
            "i8" => I8,
            "i16" => I16,
            "i32" => I32,
            "i64" => I64,
            "u8" => U8,
            "u16" => U16,
            "u32" => U32,
            "u64" => U64,
            "usize" => Usize,
            "isize" => Isize,
            "f16" => F16,
            "bf16" => Bf16,
            "f32" => F32,
            "f64" => F64,
            "bool" => Bool,
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
            U8 => "u8",
            U16 => "u16",
            U32 => "u32",
            U64 => "u64",
            Usize => "usize",
            Isize => "isize",
            F16 => "f16",
            Bf16 => "bf16",
            F32 => "f32",
            F64 => "f64",
            Bool => "bool",
        }
    }

    /// Size in bytes (usize/isize assume a 64-bit target).
    pub fn size(self) -> u64 {
        use Scalar::*;
        match self {
            I8 | U8 | Bool => 1,
            I16 | U16 | F16 | Bf16 => 2,
            I32 | U32 | F32 => 4,
            I64 | U64 | F64 | Usize | Isize => 8,
        }
    }

    pub fn align(self) -> u64 {
        self.size()
    }

    pub fn is_float(self) -> bool {
        matches!(self, Scalar::F16 | Scalar::Bf16 | Scalar::F32 | Scalar::F64)
    }

    pub fn is_int(self) -> bool {
        !self.is_float() && self != Scalar::Bool
    }

    pub fn is_signed(self) -> bool {
        use Scalar::*;
        matches!(self, I8 | I16 | I32 | I64 | Isize)
    }
}

/// One extent of a tensor shape: a compile-time constant, a bound symbolic variable, or a
/// runtime-dynamic dimension.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Dim {
    Const(u64),
    Var(Symbol),
    Dynamic,
}

/// A tensor shape — an ordered list of dimensions; its length is the rank.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct Shape(pub Vec<Dim>);

impl Shape {
    pub fn rank(&self) -> usize {
        self.0.len()
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
    Array {
        elem: Box<Ty>,
        len: u64,
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

    /// Size in bytes for sized types; `None` for slices/tensors/unsized or unknown types.
    pub fn size_of(&self) -> Option<u64> {
        match self {
            Ty::Scalar(s) => Some(s.size()),
            Ty::Unit => Some(0),
            Ty::Ptr { .. } | Ty::Ref { .. } => Some(8),
            Ty::Vector { elem, lanes } => Some(elem.size() * *lanes as u64),
            Ty::Array { elem, len } => Some(elem.size_of()? * len),
            Ty::Tuple(fields) => {
                let mut size = 0u64;
                let mut align = 1u64;
                for f in fields {
                    let fa = f.align_of()?;
                    let fs = f.size_of()?;
                    size = round_up(size, fa) + fs;
                    align = align.max(fa);
                }
                Some(round_up(size, align))
            }
            _ => None,
        }
    }

    pub fn align_of(&self) -> Option<u64> {
        match self {
            Ty::Scalar(s) => Some(s.align()),
            Ty::Unit => Some(1),
            Ty::Ptr { .. } | Ty::Ref { .. } => Some(8),
            Ty::Vector { elem, lanes } => Some(elem.size() * *lanes as u64),
            Ty::Array { elem, .. } => elem.align_of(),
            Ty::Tuple(fields) => fields
                .iter()
                .try_fold(1u64, |a, f| Some(a.max(f.align_of()?))),
            _ => None,
        }
    }

    /// The padded byte offset and type of each field of a `Tuple`, in declaration order — the
    /// single source of truth for aggregate layout (the same `round_up` accumulation as
    /// [`size_of`](Ty::size_of)). `None` for a non-tuple or a tuple with an unsized field. The MIR
    /// builder uses this to lower tuple construction/field-access as byte-offset GEPs.
    pub fn tuple_offsets(&self) -> Option<Vec<(u64, Ty)>> {
        let Ty::Tuple(fields) = self else {
            return None;
        };
        let mut out = Vec::with_capacity(fields.len());
        let mut off = 0u64;
        for f in fields {
            off = round_up(off, f.align_of()?);
            out.push((off, f.clone()));
            off += f.size_of()?;
        }
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
            Ty::Array { elem, len } => format!("[{}; {}]", elem.display(interner), len),
            Ty::Tuple(fields) => {
                let parts: Vec<_> = fields.iter().map(|f| f.display(interner)).collect();
                format!("({})", parts.join(", "))
            }
            Ty::Vector { elem, lanes } => format!("{}x{}", elem.name(), lanes),
            Ty::Tensor { elem, shape, .. } => {
                let mut parts = vec![elem.name().to_string()];
                for d in &shape.0 {
                    parts.push(dim_str(*d, interner));
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

fn dim_str(d: Dim, interner: &Interner) -> String {
    match d {
        Dim::Const(n) => n.to_string(),
        Dim::Var(s) => interner.resolve(s).to_string(),
        Dim::Dynamic => "?".to_string(),
    }
}

fn round_up(x: u64, align: u64) -> u64 {
    if align == 0 {
        x
    } else {
        x.div_ceil(align) * align
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn scalar_from_name_round_trips() {
        for s in [
            Scalar::Bool,
            Scalar::I8,
            Scalar::U8,
            Scalar::I32,
            Scalar::U64,
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
    fn nested_array_size_and_pointer_layout() {
        // [[i32; 4]; 3] -> 3 * (4 * 4) = 48 bytes.
        let inner = Ty::Array {
            elem: Box::new(Ty::Scalar(Scalar::I32)),
            len: 4,
        };
        let outer = Ty::Array {
            elem: Box::new(inner),
            len: 3,
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
        };
        assert_eq!(t.display(&i), "Tensor[f32, M, N]");
    }
}
