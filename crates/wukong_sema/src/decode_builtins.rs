//! Typing for the **decode builtins** — the kernels an autoregressive decode step is made of:
//!
//! | builtin | operands | runtime |
//! |---|---|---|
//! | `sdpa_kv(q, k, v, out, sq, skv, d, q_pos0, n_heads, n_kv_heads, q_stride, kv_stride, scale, causal)` | `f32` buffers; integers; a float; an integer or `bool` | `wukong_attention_kv_f32` |
//! | `sdpa_kv_batch::<B, SQ, SKV, H, HKV, D>(q, k, v, out, kv_lens, q_pos, scale, causal)` | `f32` `q, out: [B, SQ, H, D]`, `k, v: [B, SKV, HKV, D]`; `i32` `kv_lens, q_pos: [B]`; a float; an integer or `bool` | `wukong_attention_kv_batch_f32` |
//! | `kv_append(cache, row, pos, stride)` | `f32` buffers; integers | `wukong_kv_append_f32` |
//! | `gemv_bf16(w, x, y)` / `gemv_f16(w, x, y)` | `bf16`/`f16` weight `[N·K]`; `f32` `x[K]`, `y[N]` | `wukong_sgemv_{bf16,f16}` |
//! | `quantize_q8_0(x, q, k)` / `quantize_q4_0(x, q, k)` | `f32` `x`; `u8` blocks `q`; row length | `wukong_quantize_{q8_0,q4_0}_f32` |
//! | `gemv_q8_0(w, x, y)` / `gemv_q4_0(w, x, y)` | `u8` blocks `w`; `f32` `x[K]`, `y[N]` | `wukong_gemv_{q8_0,q4_0}_f32` |
//! | `dequant_i4(q, scale, group, out)` / `dequant_u4(q, scale, zero, group, out)` | packed `u8` nibbles `q`; `f32` `scale`, `zero`, `out`; a group size | `wukong_dequant_{i4,u4}_f32` |
//!
//! Each is typed nominally here and returns `()`: a wrong arity or a turbofish is `E0503`, an
//! operand of the wrong kind or element type `E0437`, and writing the output through a non-`mut`
//! aggregate parameter `E0304` — the same codes and the same buffer rules (`[]T`, `[T; N]`, or a
//! contiguous `Tensor[T, …]` with known dimensions) as the ML builtins in `ml_builtins.rs`, whose
//! operand checks these reuse. The *shapes* are values (lengths, head counts, strides), so they are
//! checked where they are known: the runtime validates every extent against the operand lengths
//! before it touches memory, and `wukong_mir_build` turns a refusal into a trap exactly like a
//! failed `assert` — on every backend, since the interpreter runs the same validation.
//!
//! **`sdpa_kv_batch` has a real shape signature.** Its six dimensions `<B, SQ, SKV, H, HKV, D>` are
//! resolved like a named kernel builtin's (`builtins.rs`): an explicit turbofish (literals, `const`s,
//! generic dimensions, or in-scope integer variables, which are runtime extents), the shape of every
//! `Tensor[T, ..]` operand unified axis by axis (a conflict is `E0502`, a rank other than the
//! parameter's `E0501`), and the length of a fixed array operand, which fixes the one dimension of its
//! shape still free (not a whole number of rows: `E0501`). A dimension none of those fixes is `E0409`
//! — a `[]T` slice's length fixes nothing here. A fully static array must hold exactly its shape's
//! elements (`E0501`); a slice, or an array under a runtime dimension, must too, which `mir_build`
//! asserts at run time like a `matmul` operand's length. Constant head counts must group (`H` a
//! multiple of `HKV`, else `E0502`). The resolution is recorded in `builtin_calls` as
//! [`crate::KernelBuiltin::SdpaKvBatch`], so `mir_build` lowers the call from the dimensions decided
//! here (and a copied or deferred call keeps them). The per-sequence `kv_lens` values are data: the
//! runtime checks each lies in `0..=SKV` and traps otherwise, like every other decode builtin.
//!
//! [`decode_builtin`] is the one name table; `mir_build` matches the same enum. A user function or a
//! local of the same name shadows the builtin (the caller, `type_kernel_builtin`, checks first).

use wukong_ast::{Expr, TypeExpr};
use wukong_span::{Span, Symbol};
use wukong_types::{Dim, Layout, Scalar, Ty};

use crate::builtins::dims_same;
use crate::{builtin_dims, builtin_operand_dims, BuiltinCall, BuiltinDim, KernelBuiltin, Sema};

/// One decode builtin.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DecodeBuiltin {
    /// `sdpa_kv(q, k, v, out, sq, skv, d, q_pos0, n_heads, n_kv_heads, q_stride, kv_stride, scale,
    /// causal)` — attention of `sq` query rows against a KV cache of `skv` rows (GQA, offset).
    SdpaKv,
    /// `sdpa_kv_batch(q, k, v, out, kv_lens, q_pos, scale, causal)` — `B` padded sequences in one
    /// call, shape-checked over `<B, SQ, SKV, H, HKV, D>`.
    SdpaKvBatch,
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
    /// `dequant_i4(q, scale, group, out)` — `out[i] = q_i · scale[i / group]` over packed signed
    /// 4-bit values (two per byte, low nibble first).
    DequantI4,
    /// `dequant_u4(q, scale, zero, group, out)` — `out[i] = (q_i − zero[g]) · scale[g]`,
    /// `g = i / group`, over packed unsigned 4-bit values.
    DequantU4,
}

/// The decode builtin `name` denotes, or `None`.
pub fn decode_builtin(name: &str) -> Option<DecodeBuiltin> {
    use DecodeBuiltin::*;
    Some(match name {
        "sdpa_kv" => SdpaKv,
        "sdpa_kv_batch" => SdpaKvBatch,
        "kv_append" => KvAppend,
        "gemv_bf16" => GemvHalf(Scalar::Bf16),
        "gemv_f16" => GemvHalf(Scalar::F16),
        "gemv_q8_0" => GemvQ8,
        "gemv_q4_0" => GemvQ4,
        "quantize_q8_0" => QuantizeQ8,
        "quantize_q4_0" => QuantizeQ4,
        "dequant_i4" => DequantI4,
        "dequant_u4" => DequantU4,
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
        DecodeBuiltin::SdpaKvBatch => vec![
            f32b(false),
            f32b(false),
            f32b(false),
            f32b(true),
            Buf(Scalar::I32, false),
            Buf(Scalar::I32, false),
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
        DecodeBuiltin::DequantI4 => {
            vec![
                Buf(Scalar::U8, false),
                f32b(false),
                Int("group size"),
                f32b(true),
            ]
        }
        DecodeBuiltin::DequantU4 => vec![
            Buf(Scalar::U8, false),
            f32b(false),
            f32b(false),
            Int("group size"),
            f32b(true),
        ],
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
        call: &Expr,
        callee: &Expr,
        generic_args: &[TypeExpr],
        args: &[Expr],
    ) -> Option<Ty> {
        let b = decode_builtin(name)?;
        self.types.insert(callee.id, Ty::Unknown);
        let mut arg_tys: Vec<Ty> = args.iter().map(|a| self.type_expr(a)).collect();
        let errors_before = self.diags.len();
        // `sdpa_kv_batch`'s turbofish is its dimension list (checked with its shapes below).
        let first = generic_args
            .first()
            .filter(|_| b != DecodeBuiltin::SdpaKvBatch);
        if let Some(first) = first {
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
        if b == DecodeBuiltin::SdpaKvBatch {
            self.sdpa_kv_batch_shapes(call, generic_args, args, &arg_tys, errors_before);
        }
        Some(Ty::Unit)
    }

    /// Resolve and check `sdpa_kv_batch`'s dimensions `<B, SQ, SKV, H, HKV, D>` (see the module
    /// docs), recording the call when no error was reported since `errors_before` (the operand kind
    /// checks included).
    fn sdpa_kv_batch_shapes(
        &mut self,
        call: &Expr,
        generic_args: &[TypeExpr],
        args: &[Expr],
        arg_tys: &[Ty],
        errors_before: usize,
    ) {
        const NAME: &str = "sdpa_kv_batch";
        const OPERANDS: [(&str, Scalar); 6] = [
            ("q", Scalar::F32),
            ("k", Scalar::F32),
            ("v", Scalar::F32),
            ("out", Scalar::F32),
            ("kv_lens", Scalar::I32),
            ("q_pos", Scalar::I32),
        ];
        let kind = KernelBuiltin::SdpaKvBatch;
        let names = builtin_dims(kind);
        let operand_dims = builtin_operand_dims(kind);
        let mut state: Vec<Option<Dim>> = vec![None; names.len()];
        let mut unsigned: Vec<Symbol> = Vec::new();

        // (1) An explicit turbofish, positional over the dimension list.
        if !generic_args.is_empty() {
            if generic_args.len() != names.len() {
                self.error(
                    call.span,
                    "E0503",
                    format!(
                        "`{NAME}` takes {} dimension arguments `<{}>`, but {} were supplied",
                        names.len(),
                        names.join(", "),
                        generic_args.len()
                    ),
                );
                return;
            }
            for (i, ga) in generic_args.iter().enumerate() {
                if let Some((d, is_unsigned)) = self.builtin_turbofish_dim(ga, NAME) {
                    if let (Dim::Var(n), true) = (&d, is_unsigned) {
                        unsigned.push(*n);
                    }
                    self.bind_batch_dim(&mut state, i, d, "turbofish", ga.span);
                }
            }
        }

        // (2) Tensor operands of the right element: their shapes, axis by axis. (A wrong element,
        // a strided layout or a `?` dimension was reported by the operand check.)
        for (i, &(pname, elem)) in OPERANDS.iter().enumerate() {
            let Some(dims) = operand_dims[i] else {
                continue;
            };
            let Ty::Tensor {
                elem: e,
                shape,
                layout: Layout::Contiguous,
                elem_var: None,
            } = &arg_tys[i]
            else {
                continue;
            };
            if *e != elem || shape.0.iter().any(|d| matches!(d, Dim::Dynamic)) {
                continue;
            }
            if shape.0.len() != dims.len() {
                let want: Vec<&str> = dims.iter().map(|&d| names[d]).collect();
                let found = self.batch_shape_str(&shape.0);
                self.error(
                    args[i].span,
                    "E0501",
                    format!(
                        "tensor rank mismatch: operand `{pname}` of `{NAME}` expects rank {} [{}], \
                         found rank {} {found}",
                        dims.len(),
                        want.join(", "),
                        shape.0.len()
                    ),
                );
                continue;
            }
            for (axis, (&di, d)) in dims.iter().zip(&shape.0).enumerate() {
                let what = format!("axis {axis} of `{pname}`");
                self.bind_batch_dim(&mut state, di, d.clone(), &what, args[i].span);
            }
        }

        // (3) A fixed array fixes the one dimension of its shape that is still free, to a fixed
        // point.
        let array_len = |i: usize| match &arg_tys[i] {
            Ty::Array { elem, len, .. } if **elem == Ty::Scalar(OPERANDS[i].1) => Some(*len),
            _ => None,
        };
        loop {
            let mut progress = false;
            for i in 0..OPERANDS.len() {
                let (Some(dims), Some(len)) = (operand_dims[i], array_len(i)) else {
                    continue;
                };
                let free: Vec<usize> = dims
                    .iter()
                    .copied()
                    .filter(|&d| state[d].is_none())
                    .collect();
                let [free] = free[..] else {
                    continue;
                };
                let mut prod: Option<u64> = Some(1);
                for &d in dims.iter().filter(|&&d| d != free) {
                    prod = match (prod, &state[d]) {
                        (Some(acc), Some(Dim::Const(n))) => acc.checked_mul(*n),
                        _ => None,
                    };
                }
                let Some(prod) = prod.filter(|&p| p > 0) else {
                    continue;
                };
                if !len.is_multiple_of(prod) {
                    self.error(
                        args[i].span,
                        "E0501",
                        format!(
                            "extent mismatch: operand `{}` of `{NAME}` is an array of {len} \
                             element(s), which is not a whole number of rows of {prod} (dimension \
                             `{}` cannot be inferred from it)",
                            OPERANDS[i].0, names[free]
                        ),
                    );
                }
                // Pinned even after an error, so it is reported once.
                state[free] = Some(Dim::Const(len / prod));
                progress = true;
            }
            if !progress {
                break;
            }
        }

        // (4) Every dimension must now be fixed.
        let mut unresolved = false;
        for (i, st) in state.iter().enumerate() {
            if st.is_none() {
                unresolved = true;
                self.error(
                    call.span,
                    "E0409",
                    format!(
                        "cannot infer dimension `{}` of `{NAME}`: no tensor shape, array length or \
                         turbofish fixes it (a slice's length does not); supply it explicitly, \
                         e.g. `{NAME}::<{}>(…)`",
                        names[i],
                        names.join(", ")
                    ),
                );
            }
        }
        if unresolved {
            return;
        }

        // (5) A fully static array holds exactly the elements its shape names.
        for i in 0..OPERANDS.len() {
            let (Some(dims), Some(len)) = (operand_dims[i], array_len(i)) else {
                continue;
            };
            let all_const = dims
                .iter()
                .all(|&d| matches!(state[d], Some(Dim::Const(_))));
            if !all_const {
                // A runtime dimension: `mir_build` asserts the exact length at run time.
                continue;
            }
            let mut need: Option<u64> = Some(1);
            for &d in dims {
                if let Some(Dim::Const(n)) = state[d] {
                    need = need.and_then(|acc| acc.checked_mul(n));
                }
            }
            if need != Some(len) {
                let shape: Vec<Dim> = dims.iter().filter_map(|&d| state[d].clone()).collect();
                let shape_s = self.batch_shape_str(&shape);
                let need_s =
                    need.map_or_else(|| "more than u64::MAX".to_string(), |n| n.to_string());
                self.error(
                    args[i].span,
                    "E0501",
                    format!(
                        "extent mismatch: operand `{}` of `{NAME}` is an array of {len} \
                         element(s), but its shape {shape_s} holds {need_s}",
                        OPERANDS[i].0
                    ),
                );
            }
        }

        // (6) Constant head counts must group: every KV head serves H / HKV query heads.
        if let (Some(Dim::Const(h)), Some(Dim::Const(hkv))) = (&state[3], &state[4]) {
            if *h != 0 && (*hkv == 0 || h % hkv != 0) {
                self.error(
                    call.span,
                    "E0502",
                    format!(
                        "dimension mismatch on `H` of `{NAME}`: the query heads are grouped over \
                         the KV heads, so `H` must be a multiple of `HKV`, found H = {h}, HKV = \
                         {hkv}"
                    ),
                );
            }
        }

        if self.diags[errors_before..].iter().any(|d| d.is_error()) {
            return;
        }
        let dims = state
            .into_iter()
            .map(|st| match st {
                Some(Dim::Const(n)) => BuiltinDim::Const(n),
                Some(Dim::Var(s)) => BuiltinDim::Named {
                    name: s,
                    signed: !unsigned.contains(&s),
                },
                Some(d) => BuiltinDim::Derived(d),
                // Unreachable: every dimension was fixed above.
                None => BuiltinDim::Const(0),
            })
            .collect();
        self.builtin_calls
            .insert(call.id, BuiltinCall { kind, dims });
    }

    /// Bind dimension `idx` of `sdpa_kv_batch` to `d`, or check `d` against its binding (`E0502`).
    fn bind_batch_dim(
        &mut self,
        state: &mut [Option<Dim>],
        idx: usize,
        d: Dim,
        what: &str,
        span: Span,
    ) {
        if matches!(d, Dim::Dynamic) {
            return;
        }
        match &state[idx] {
            None => state[idx] = Some(d),
            Some(b) if dims_same(b, &d) => {}
            Some(b) => {
                let extra = match (b, &d) {
                    (Dim::Const(_), Dim::Var(_) | Dim::Expr(_))
                    | (Dim::Var(_) | Dim::Expr(_), Dim::Const(_)) => {
                        " (a generic or runtime dimension cannot be assumed equal to a different \
                         extent)"
                    }
                    _ => "",
                };
                let (bs, ds) = (b.display(self.interner), d.display(self.interner));
                let dim = builtin_dims(KernelBuiltin::SdpaKvBatch)[idx];
                self.error(
                    span,
                    "E0502",
                    format!(
                        "dimension mismatch on `{dim}` of `sdpa_kv_batch` ({what}): expected {bs}, \
                         found {ds}{extra}"
                    ),
                );
            }
        }
    }

    fn batch_shape_str(&self, dims: &[Dim]) -> String {
        let parts: Vec<String> = dims.iter().map(|d| d.display(self.interner)).collect();
        format!("[{}]", parts.join(", "))
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
        let ok = "fn decode_step(q: []f32, k: []f32, v: []f32, mut o: []f32, t: i64) { \
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
                "E0437",
            ),
            // Quantized blocks are bytes.
            (
                "fn f(w: []i8, x: []f32, mut y: []f32) { gemv_q8_0(w, x, y); }",
                "E0437",
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
                "E0437",
            ),
            (
                "fn f(q: []f32, mut o: []f32, s: *u8) { sdpa_kv(q, q, q, o, 1, 1, 8, 0, 1, 1, 8, \
                 8, s, 0); }",
                "E0437",
            ),
            // The row length of a packer.
            (
                "fn f(x: []f32, mut q: []u8) { quantize_q4_0(x, q, 2.5); }",
                "E0437",
            ),
        ];
        for &(src, code) in cases {
            assert_eq!(errors(src), [code], "{src}");
        }
    }

    /// The recorded `sdpa_kv_batch` resolutions of `src` (which must type-check), in source order.
    fn batch_dims(src: &str) -> Vec<Vec<String>> {
        let mut interner = Interner::new();
        let (module, pd) = wukong_parser::parse_module(src, SourceId(0), &mut interner);
        assert!(pd.is_empty(), "parse errors: {pd:?}");
        let (res, diags) = crate::check(&module, &interner);
        let errs: Vec<_> = diags.iter().filter(|d| d.is_error()).collect();
        assert!(errs.is_empty(), "{src}: {errs:?}");
        let mut calls: Vec<_> = res
            .builtin_calls
            .iter()
            .filter(|(_, c)| c.kind == KernelBuiltin::SdpaKvBatch)
            .collect();
        calls.sort_by_key(|(id, _)| id.0);
        calls
            .into_iter()
            .map(|(_, c)| {
                c.dims
                    .iter()
                    .map(|d| match d {
                        BuiltinDim::Const(n) => n.to_string(),
                        BuiltinDim::Named { name, signed } => {
                            format!(
                                "{}{}",
                                interner.resolve(*name),
                                if *signed { "" } else { "u" }
                            )
                        }
                        BuiltinDim::Derived(d) => d.display(&interner),
                        BuiltinDim::SliceLen { .. } => "slice".to_string(),
                    })
                    .collect()
            })
            .collect()
    }

    #[test]
    fn sdpa_kv_batch_resolves_its_dimensions_like_a_kernel_builtin() {
        // Every dimension from the tensor shapes.
        let tensors = "fn f(q: Tensor[f32, 2, 1, 4, 8], k: Tensor[f32, 2, 16, 2, 8], \
                       v: Tensor[f32, 2, 16, 2, 8], mut o: Tensor[f32, 2, 1, 4, 8], \
                       l: Tensor[i32, 2], p: [i32; 2]) { \
                       sdpa_kv_batch(q, k, v, o, l, p, 0.35, true); }";
        assert_eq!(batch_dims(tensors), [["2", "1", "16", "4", "2", "8"]]);
        // Generic dimensions stay symbolic (the hidden dimension parameters at run time), and a
        // derived extent is recorded as such.
        let generic = "fn f<B, S, G>(q: Tensor[f32, B, 1, 2*G, 8], k: Tensor[f32, B, S, G, 8], \
                       v: Tensor[f32, B, S, G, 8], mut o: Tensor[f32, B, 1, 2*G, 8], \
                       l: Tensor[i32, B], p: Tensor[i32, B]) { \
                       sdpa_kv_batch(q, k, v, o, l, p, 0.35, 1); }";
        assert_eq!(batch_dims(generic), [["B", "1", "S", "2*G", "G", "8"]]);
        // A turbofish over flat arrays, with runtime extents from integer locals (an unsigned one
        // zero-extends), and the one free dimension an array fixes.
        let flat = "fn f(q: []f32, k: []f32, v: []f32, mut o: []f32, l: []i32, p: []i32, b: i64, \
                    s: u32) { sdpa_kv_batch::<b, 1, s, 4, 2, 8>(q, k, v, o, l, p, 0.35, true); } \
                    fn g(q: [f32; 64], k: [f32; 512], v: [f32; 512], mut o: [f32; 64], \
                    l: [i32; 2], p: [i32; 2]) { \
                    sdpa_kv_batch::<2, 1, 16, 4, 2, 8>(q, k, v, o, l, p, 0.35, true); }";
        assert_eq!(
            batch_dims(flat),
            [
                ["b", "1", "su", "4", "2", "8"],
                ["2", "1", "16", "4", "2", "8"]
            ]
        );
        // A fixed array fixes the one dimension of its shape left free by the tensors: SKV from k.
        let array = "fn f(q: Tensor[f32, 2, 1, 4, 8], k: [f32; 512], v: Tensor[f32, 2, 16, 2, 8], \
                     mut o: Tensor[f32, 2, 1, 4, 8], l: [i32; 2], p: [i32; 2]) { \
                     sdpa_kv_batch(q, k, v, o, l, p, 0.35, true); }";
        assert_eq!(batch_dims(array), [["2", "1", "16", "4", "2", "8"]]);
    }

    #[test]
    fn sdpa_kv_batch_shape_misuse_is_rejected() {
        let sig = |q: &str, k: &str, lens: &str| {
            format!(
                "fn f(q: {q}, k: {k}, mut o: Tensor[f32, 2, 1, 4, 8], l: {lens}, p: [i32; 2]) {{ \
                 sdpa_kv_batch(q, k, k, o, l, p, 0.35, true); }}"
            )
        };
        let t4 = "Tensor[f32, 2, 1, 4, 8]";
        let k4 = "Tensor[f32, 2, 16, 2, 8]";
        let cases: Vec<(String, &str)> =
            vec![
            // The head dimension disagrees between the queries and the keys (the call passes the
            // keys as the values too: one mismatch per operand).
            (sig(t4, "Tensor[f32, 2, 16, 2, 4]", "[i32; 2]"), "E0502 E0502"),
            // The batch disagrees with the length vector.
            (sig(t4, k4, "[i32; 3]"), "E0501"),
            // A rank-3 query.
            (sig("Tensor[f32, 2, 4, 8]", k4, "[i32; 2]"), "E0501"),
            // Six query heads cannot group over four KV heads.
            (
                "fn f(q: Tensor[f32, 1, 1, 6, 8], k: Tensor[f32, 1, 4, 4, 8], \
                 mut o: Tensor[f32, 1, 1, 6, 8], l: [i32; 1], p: [i32; 1]) { \
                 sdpa_kv_batch(q, k, k, o, l, p, 0.35, true); }"
                    .to_string(),
                "E0502",
            ),
            // The sequence vectors are `i32`.
            (sig(t4, k4, "[i64; 2]"), "E0437"),
            // A fully static array must hold exactly its shape.
            (
                "fn f(q: [f32; 63], k: [f32; 512], mut o: [f32; 64], l: [i32; 2], p: [i32; 2]) { \
                 sdpa_kv_batch::<2, 1, 16, 4, 2, 8>(q, k, k, o, l, p, 0.35, true); }"
                    .to_string(),
                "E0501",
            ),
            // The turbofish lists all six dimensions.
            (
                "fn f(q: []f32, mut o: []f32, l: []i32) { \
                 sdpa_kv_batch::<2, 1, 16>(q, q, q, o, l, l, 0.35, true); }"
                    .to_string(),
                "E0503",
            ),
            // Arity.
            (
                "fn f(q: []f32, mut o: []f32, l: []i32) { sdpa_kv_batch(q, q, q, o, l, 0.35, \
                 true); }"
                    .to_string(),
                "E0503",
            ),
            // The output is written.
            (
                "fn f(q: Tensor[f32, 2, 1, 4, 8], k: Tensor[f32, 2, 16, 2, 8], \
                 o: Tensor[f32, 2, 1, 4, 8], l: [i32; 2], p: [i32; 2]) { \
                 sdpa_kv_batch(q, k, k, o, l, p, 0.35, true); }"
                    .to_string(),
                "E0304",
            ),
        ];
        for (src, codes) in &cases {
            let want: Vec<&str> = codes.split(' ').collect();
            assert_eq!(errors(src), want, "{src}");
        }
        // Slices fix no dimension: every one left free is `E0409`.
        let codes = errors(
            "fn f(q: []f32, mut o: []f32, l: []i32) { \
             sdpa_kv_batch(q, q, q, o, l, l, 0.35, true); }",
        );
        assert_eq!(codes, ["E0409"; 6]);
    }

    #[test]
    fn a_user_function_shadows_a_decode_builtin() {
        let src = "fn gemv_bf16(a: i32) -> i32 { return a; } \
                   fn main() -> i32 { return gemv_bf16(3); }";
        assert!(errors(src).is_empty(), "{:?}", errors(src));
    }
}
