//! Typing for the **decode builtins** — the kernels an autoregressive decode step is made of:
//!
//! | builtin | operands | runtime |
//! |---|---|---|
//! | `sdpa_kv(q, k, v, out, sq, skv, d, q_pos0, n_heads, n_kv_heads, q_stride, kv_stride, scale, causal)` | `f32` buffers; integers; a float; an integer or `bool` | `wukong_attention_kv_f32` |
//! | `kv_append(cache, row, pos, stride)` | `f32` buffers; integers | `wukong_kv_append_f32` |
//! | `gemv_bf16(w, x, y)` / `gemv_f16(w, x, y)` | `bf16`/`f16` weight `[N·K]`; `f32` `x[K]`, `y[N]` | `wukong_sgemv_{bf16,f16}` |
//! | `quantize_q8_0(x, q, k)` / `quantize_q4_0(x, q, k)` | `f32` `x`; `u8` blocks `q`; row length | `wukong_quantize_{q8_0,q4_0}_f32` |
//! | `gemv_q8_0(w, x, y)` / `gemv_q4_0(w, x, y)` | `u8` blocks `w`; `f32` `x[K]`, `y[N]` | `wukong_gemv_{q8_0,q4_0}_f32` |
//!
//! Each is typed nominally here and returns `()`: a wrong arity or a turbofish is `E0503`, an
//! operand of the wrong kind or element type `E0401`, and writing the output through a non-`mut`
//! aggregate parameter `E0304` — the same codes and the same buffer rules (`[]T`, `[T; N]`, or a
//! contiguous `Tensor[T, …]` with known dimensions) as the ML builtins in `ml_builtins.rs`, whose
//! operand checks these reuse. The *shapes* are values (lengths, head counts, strides), so they are
//! checked where they are known: the runtime validates every extent against the operand lengths
//! before it touches memory, and `wukong_mir_build` turns a refusal into a trap exactly like a
//! failed `assert` — on every backend, since the interpreter runs the same validation.
//!
//! [`decode_builtin`] is the one name table; `mir_build` matches the same enum. A user function or a
//! local of the same name shadows the builtin (the caller, `type_kernel_builtin`, checks first).

use wukong_ast::{Expr, TypeExpr};
use wukong_types::{Scalar, Ty};

use crate::Sema;

/// One decode builtin.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DecodeBuiltin {
    /// `sdpa_kv(q, k, v, out, sq, skv, d, q_pos0, n_heads, n_kv_heads, q_stride, kv_stride, scale,
    /// causal)` — attention of `sq` query rows against a KV cache of `skv` rows (GQA, offset).
    SdpaKv,
    /// `kv_append(cache, row, pos, stride)` — `cache[pos·stride ..][..len(row)] = row`.
    KvAppend,
    /// `gemv_bf16(w, x, y)` / `gemv_f16(w, x, y)` — `y = W·x` with half-precision weights.
    GemvHalf(Scalar),
    /// `gemv_q8_0(w, x, y)` — `y = W·x` with GGML `Q8_0` weights.
    GemvQ8,
    /// `gemv_q4_0(w, x, y)` — `y = W·x` with GGML `Q4_0` weights.
    GemvQ4,
    /// `quantize_q8_0(x, q, k)` — pack rows of `k` values into `Q8_0` blocks.
    QuantizeQ8,
    /// `quantize_q4_0(x, q, k)` — pack rows of `k` values into `Q4_0` blocks.
    QuantizeQ4,
}

/// The decode builtin `name` denotes, or `None`.
pub fn decode_builtin(name: &str) -> Option<DecodeBuiltin> {
    use DecodeBuiltin::*;
    Some(match name {
        "sdpa_kv" => SdpaKv,
        "kv_append" => KvAppend,
        "gemv_bf16" => GemvHalf(Scalar::Bf16),
        "gemv_f16" => GemvHalf(Scalar::F16),
        "gemv_q8_0" => GemvQ8,
        "gemv_q4_0" => GemvQ4,
        "quantize_q8_0" => QuantizeQ8,
        "quantize_q4_0" => QuantizeQ4,
        _ => return None,
    })
}

/// One parameter of a decode builtin's signature.
#[derive(Clone, Copy)]
enum Param {
    /// A buffer of this element type; `true` when the builtin writes it.
    Buf(Scalar, bool),
    /// An integer (`i64` ABI).
    Int(&'static str),
    /// A floating-point scalar (`f32` ABI).
    Float(&'static str),
    /// An integer or `bool` flag.
    Flag(&'static str),
}

/// The parameter list of `b`, in argument order.
fn signature(b: DecodeBuiltin) -> Vec<Param> {
    use Param::{Buf, Flag, Float, Int};
    let f32b = |written| Buf(Scalar::F32, written);
    match b {
        DecodeBuiltin::SdpaKv => vec![
            f32b(false),
            f32b(false),
            f32b(false),
            f32b(true),
            Int("query row count"),
            Int("cache length"),
            Int("head dimension"),
            Int("query position"),
            Int("query head count"),
            Int("KV head count"),
            Int("query row stride"),
            Int("KV row stride"),
            Float("scale"),
            Flag("causal flag"),
        ],
        DecodeBuiltin::KvAppend => {
            vec![f32b(true), f32b(false), Int("position"), Int("row stride")]
        }
        DecodeBuiltin::GemvHalf(t) => vec![Buf(t, false), f32b(false), f32b(true)],
        DecodeBuiltin::GemvQ8 | DecodeBuiltin::GemvQ4 => {
            vec![Buf(Scalar::U8, false), f32b(false), f32b(true)]
        }
        DecodeBuiltin::QuantizeQ8 | DecodeBuiltin::QuantizeQ4 => {
            vec![f32b(false), Buf(Scalar::U8, true), Int("row length")]
        }
    }
}

/// The spelling of one parameter in an arity diagnostic.
fn describe(p: Param) -> String {
    match p {
        Param::Buf(t, _) => format!("a `{}` buffer", t.name()),
        Param::Int(what) => format!("an integer {what}"),
        Param::Float(what) => format!("a float {what}"),
        Param::Flag(what) => format!("an integer or `bool` {what}"),
    }
}

impl Sema<'_> {
    /// Type a call to a decode builtin, or return `None` when the callee is not one (the caller
    /// keeps looking). `name` is the callee's single-segment path, already known not to be shadowed.
    /// Every argument is typed exactly once, and misuse is diagnosed here; the result is `()`.
    pub(crate) fn type_decode_builtin(
        &mut self,
        name: &str,
        callee: &Expr,
        generic_args: &[TypeExpr],
        args: &[Expr],
    ) -> Option<Ty> {
        let b = decode_builtin(name)?;
        self.types.insert(callee.id, Ty::Unknown);
        let mut arg_tys: Vec<Ty> = args.iter().map(|a| self.type_expr(a)).collect();
        if let Some(first) = generic_args.first() {
            self.error(
                first.span,
                "E0503",
                format!(
                    "`{name}` takes no generic arguments; its extents come from its operands and \
                     integer arguments"
                ),
            );
        }
        let sig = signature(b);
        if args.len() != sig.len() {
            let want: Vec<String> = sig.iter().map(|&p| describe(p)).collect();
            self.error(
                callee.span,
                "E0503",
                format!(
                    "`{name}` takes exactly {} arguments ({}), but {} were supplied",
                    sig.len(),
                    want.join(", "),
                    args.len()
                ),
            );
            return Some(Ty::Unit);
        }
        for (i, &p) in sig.iter().enumerate() {
            match p {
                Param::Buf(t, written) => {
                    self.ml_buffer_arg(name, &args[i], &arg_tys[i], t, i, written)
                }
                Param::Int(what) => {
                    self.ml_scalar_arg(name, &args[i], &mut arg_tys[i], false, what)
                }
                Param::Float(what) => {
                    self.ml_scalar_arg(name, &args[i], &mut arg_tys[i], true, what)
                }
                Param::Flag(what) => {
                    if arg_tys[i] != Ty::Scalar(Scalar::Bool) {
                        self.ml_scalar_arg(name, &args[i], &mut arg_tys[i], false, what);
                    }
                }
            }
        }
        Some(Ty::Unit)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use wukong_span::{Interner, SourceId};

    fn errors(src: &str) -> Vec<&'static str> {
        let mut interner = Interner::new();
        let (module, pd) = wukong_parser::parse_module(src, SourceId(0), &mut interner);
        assert!(pd.is_empty(), "parse errors: {pd:?}");
        let (_res, diags) = crate::check(&module, &interner);
        diags
            .into_iter()
            .filter(|d| d.is_error())
            .filter_map(|d| d.code)
            .collect()
    }

    #[test]
    fn the_name_table_is_closed() {
        assert_eq!(decode_builtin("sdpa_kv"), Some(DecodeBuiltin::SdpaKv));
        assert_eq!(
            decode_builtin("gemv_f16"),
            Some(DecodeBuiltin::GemvHalf(Scalar::F16))
        );
        assert_eq!(decode_builtin("gemv_q4_0"), Some(DecodeBuiltin::GemvQ4));
        for not_one in [
            "sdpa",
            "gemv",
            "gemv_f32",
            "gemv_q5_0",
            "quantize_q8",
            "kv_cache",
        ] {
            assert_eq!(decode_builtin(not_one), None, "{not_one}");
        }
    }

    #[test]
    fn well_formed_calls_type_clean() {
        let ok = "fn step(q: []f32, k: []f32, v: []f32, mut o: []f32, t: i64) { \
                  sdpa_kv(q, k, v, o, 1, t + 1, 8, t, 4, 2, 32, 16, 0.25, true); \
                  sdpa_kv(q, k, v, o, 1, t + 1, 8, t, 4, 2, 32, 16, 1.0, 0); } \
                  fn grow(mut kc: [f32; 64], row: [f32; 16], t: i32) { kv_append(kc, row, t, 16); } \
                  fn proj(w: []bf16, h: [f16; 6], x: []f32, mut y: []f32, mut z: [f32; 3]) { \
                  gemv_bf16(w, x, y); gemv_f16(h, [1.0, 2.0], z); } \
                  fn quant(x: []f32, mut q: []u8, mut y: []f32, n: u32) { \
                  quantize_q8_0(x, q, n); gemv_q8_0(q, x, y); \
                  quantize_q4_0(x, q, 40); gemv_q4_0(q, x, y); }";
        let codes = errors(ok);
        assert!(codes.is_empty(), "unexpected: {codes:?}");
    }

    #[test]
    fn misuse_is_rejected_with_the_standard_codes() {
        let cases: &[(&str, &str)] = &[
            // Arity.
            (
                "fn f(mut c: []f32, r: []f32) { kv_append(c, r, 0); }",
                "E0503",
            ),
            // A turbofish.
            (
                "fn f(w: []bf16, x: []f32, mut y: []f32) { gemv_bf16::<2>(w, x, y); }",
                "E0503",
            ),
            // The weight's element type is exact: f16 storage is not bf16.
            (
                "fn f(w: []f16, x: []f32, mut y: []f32) { gemv_bf16(w, x, y); }",
                "E0401",
            ),
            // Quantized blocks are bytes.
            (
                "fn f(w: []i8, x: []f32, mut y: []f32) { gemv_q8_0(w, x, y); }",
                "E0401",
            ),
            // The output is written: a non-`mut` aggregate parameter refuses it.
            (
                "fn f(w: []bf16, x: []f32, y: [f32; 4]) { gemv_bf16(w, x, y); }",
                "E0304",
            ),
            // A stride is an integer, the scale a float.
            (
                "fn f(q: []f32, mut o: []f32) { sdpa_kv(q, q, q, o, 1, 1, 8, 0, 1, 1, 8.0, 8, \
                 1.0, 0); }",
                "E0401",
            ),
            (
                "fn f(q: []f32, mut o: []f32, s: *u8) { sdpa_kv(q, q, q, o, 1, 1, 8, 0, 1, 1, 8, \
                 8, s, 0); }",
                "E0401",
            ),
            // The row length of a packer.
            (
                "fn f(x: []f32, mut q: []u8) { quantize_q4_0(x, q, 2.5); }",
                "E0401",
            ),
        ];
        for &(src, code) in cases {
            assert_eq!(errors(src), [code], "{src}");
        }
    }

    #[test]
    fn a_user_function_shadows_a_decode_builtin() {
        let src = "fn gemv_bf16(a: i32) -> i32 { return a; } \
                   fn main() -> i32 { return gemv_bf16(3); }";
        assert!(errors(src).is_empty(), "{:?}", errors(src));
    }
}
