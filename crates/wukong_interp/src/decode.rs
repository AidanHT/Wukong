//! The interpreter twins of the decode builtins' runtime entries: KV-cache attention
//! (`wukong_attention_kv_f32`), the KV append, the half-weight GEMV (`wukong_sgemv_{bf16,f16}`) and
//! the GGML Q8_0 / Q4_0 packers and GEMV.
//!
//! Every arm first validates the call with the runtime's own plan function (`kv_attention_plan`,
//! `gemv_half_plan`, `quantize_plan`, …), so a malformed shape returns exactly the status the native
//! entry returns, with memory untouched. Then it marshals the operands out of slot memory, calls the
//! same **serial** runtime entry the native backend calls (the `_parallel` names too — every one is
//! pinned bit-identical to its serial twin), and writes back only the elements that entry defines.
//! Inputs are copied before any output is written, which is the value semantics the native entries
//! implement for an overlapping output.
//!
//! Byte buffers (the quantized weights) live one slot per byte in the canonical sign-extended form
//! (`0xC8` is `Int(-56)`, see `ml_io.rs`), so they are read `as u8` and written back `as i8`.

use wukong_runtime::attention_kv::{kv_append_plan, kv_attention_plan, KV_OK};
use wukong_runtime::gemv_half::{gemv_half_plan, GEMV_OK};
use wukong_runtime::quant::{gemv_q_plan, quantize_plan, QFormat, QUANT_OK};

use crate::{kernel_buffer, kernel_values, marshal, Interp, Intrinsic, Value};

/// An `i64` ABI argument as the native entry receives it (MIR passes these as `i64` values).
fn arg_i64(args: &[Value], i: usize) -> i64 {
    args.get(i)
        .map(|v| v.as_int().clamp(i64::MIN as i128, i64::MAX as i128) as i64)
        .unwrap_or(0)
}

/// The pointer argument at `i`.
fn ptr_arg(args: &[Value], i: usize, ctx: &str, what: &str) -> Result<usize, String> {
    match args.get(i).copied() {
        Some(Value::Ptr(p)) => Ok(p),
        _ => Err(format!("{ctx}: {what} is not a pointer")),
    }
}

/// A runtime status as the call's value.
fn status(code: i64) -> Value {
    Value::Int(i128::from(code))
}

/// A validated call the kernel nevertheless refused: an internal inconsistency, never a user error.
fn refused(ctx: &str, code: i64) -> String {
    format!("{ctx}: the runtime refused a call its own plan accepted (status {code})")
}

impl Interp<'_, '_> {
    /// Dispatch a decode-builtin runtime entry, or `None` when `id` is not one of them.
    pub(crate) fn decode_intrinsic(
        &mut self,
        id: Intrinsic,
        args: &[Value],
    ) -> Option<Result<Value, String>> {
        use Intrinsic as I;
        let name = id.name();
        Some(match id {
            I::AttentionKvF32 | I::AttentionKvF32Parallel => self.dec_attention_kv(name, args),
            I::KvAppendF32 => self.dec_kv_append(name, args),
            I::SgemvBf16 | I::SgemvBf16Parallel | I::SgemvF16 | I::SgemvF16Parallel => {
                self.dec_sgemv_half(name, args)
            }
            I::QuantizeQ80F32 => self.dec_quantize(name, args, QFormat::Q8),
            I::QuantizeQ40F32 => self.dec_quantize(name, args, QFormat::Q4),
            I::GemvQ80F32 | I::GemvQ80F32Parallel => self.dec_gemv_q(name, args, QFormat::Q8),
            I::GemvQ40F32 | I::GemvQ40F32Parallel => self.dec_gemv_q(name, args, QFormat::Q4),
            _ => return None,
        })
    }

    /// `wukong_attention_kv_f32(q, nq, k, nk, v, nv, o, no, sq, skv, d, q_pos0, n_heads, n_kv_heads,
    /// q_stride, kv_stride, scale, causal) -> i64`.
    fn dec_attention_kv(&mut self, name: &str, args: &[Value]) -> Result<Value, String> {
        let mut geo = [0i64; 8];
        for (g, slot) in geo.iter_mut().enumerate() {
            *slot = arg_i64(args, 8 + g);
        }
        let lens = [
            arg_i64(args, 1),
            arg_i64(args, 3),
            arg_i64(args, 5),
            arg_i64(args, 7),
        ];
        let plan = match kv_attention_plan(lens[0], lens[1], lens[2], lens[3], geo) {
            Ok(Some(plan)) => plan,
            Ok(None) => return Ok(status(KV_OK)),
            Err(code) => return Ok(status(code)),
        };
        let scale = args.get(16).map(|v| v.as_float() as f32).unwrap_or(0.0);
        let causal = arg_i64(args, 17);
        let q = marshal::read_f32(
            arena!(self),
            0,
            ptr_arg(args, 0, name, "q")?,
            plan.q_extent,
            name,
        )?;
        let (k, v) = if plan.kv_extent == 0 {
            (Vec::new(), Vec::new())
        } else {
            let kb = ptr_arg(args, 2, name, "k")?;
            let vb = ptr_arg(args, 4, name, "v")?;
            (
                marshal::read_f32(arena!(self), 2, kb, plan.kv_extent, name)?,
                marshal::read_f32(arena!(self), 4, vb, plan.kv_extent, name)?,
            )
        };
        let o_base = ptr_arg(args, 6, name, "out")?;
        let mut o = marshal::output_f32(arena!(self), 6, o_base, plan.q_extent, name)?;
        // SAFETY: every host buffer holds exactly the extent the plan validated, and they are
        // disjoint.
        let code = unsafe {
            wukong_runtime::wukong_attention_kv_f32(
                q.as_ptr(),
                q.len() as i64,
                k.as_ptr(),
                k.len() as i64,
                v.as_ptr(),
                v.len() as i64,
                o.as_mut_ptr(),
                o.len() as i64,
                geo[0],
                geo[1],
                geo[2],
                geo[3],
                geo[4],
                geo[5],
                geo[6],
                geo[7],
                scale,
                causal,
            )
        };
        if code != KV_OK {
            return Err(refused(name, code));
        }
        // Only the heads' elements are defined; a stride gap keeps whatever it held.
        for i in 0..plan.sq {
            for h in 0..plan.n_heads {
                let off = plan.q_off(i, h);
                let head = &o[off..off + plan.d];
                marshal::write_f32(arena_mut!(self), 6, o_base + off, head, name)?;
            }
        }
        marshal::recycle_f32(q);
        marshal::recycle_f32(k);
        marshal::recycle_f32(v);
        marshal::recycle_f32(o);
        Ok(status(KV_OK))
    }

    /// `wukong_kv_append_f32(cache, ncache, row, nrow, pos, stride) -> i64`.
    fn dec_kv_append(&mut self, name: &str, args: &[Value]) -> Result<Value, String> {
        let row_len = arg_i64(args, 3);
        let off = match kv_append_plan(
            arg_i64(args, 1),
            row_len,
            arg_i64(args, 4),
            arg_i64(args, 5),
        ) {
            Ok(Some(off)) => off,
            Ok(None) => return Ok(status(KV_OK)),
            Err(code) => return Ok(status(code)),
        };
        // The cache operand is the whole `[cache, cache + ncache)` the entry may write (the plan
        // proved the row lies inside it), so it must lie inside ONE allocation before anything is
        // copied. Checking only the destination row, against whichever allocation held it, let a
        // cache whose stated length overran its allocation write a row into the next allocation.
        let cache = ptr_arg(args, 0, name, "cache")?;
        let ncache = usize::try_from(arg_i64(args, 1))
            .map_err(|_| format!("{name}: cache extent overflow"))?;
        kernel_values(arena!(self), 0, cache, 1, ncache, name)?;
        let row = marshal::read_f32(
            arena!(self),
            2,
            ptr_arg(args, 2, name, "row")?,
            row_len as usize,
            name,
        )?;
        let dst = cache
            .checked_add(off)
            .ok_or_else(|| format!("{name}: cache out of bounds"))?;
        marshal::write_f32(arena_mut!(self), 0, dst, &row, name)?;
        marshal::recycle_f32(row);
        Ok(status(KV_OK))
    }

    /// `wukong_sgemv_{bf16,f16}[_parallel](w, nw, x, nx, y, ny) -> i64`.
    fn dec_sgemv_half(&mut self, name: &str, args: &[Value]) -> Result<Value, String> {
        let is_f16 = name.contains("_f16");
        let (n, k) = match gemv_half_plan(arg_i64(args, 1), arg_i64(args, 3), arg_i64(args, 5)) {
            Ok(nk) => nk,
            Err(code) => return Ok(status(code)),
        };
        if n == 0 {
            return Ok(status(GEMV_OK));
        }
        let (w, x) = if k == 0 {
            (Vec::new(), Vec::new())
        } else {
            let wb = ptr_arg(args, 0, name, "weight")?;
            let xb = ptr_arg(args, 2, name, "x")?;
            (
                marshal::read_half_bits(
                    arena!(self),
                    0,
                    wb,
                    marshal::span(n, k, name)?,
                    name,
                    is_f16,
                )?,
                marshal::read_f32(arena!(self), 2, xb, k, name)?,
            )
        };
        let y_base = ptr_arg(args, 4, name, "y")?;
        let mut y = marshal::output_f32(arena!(self), 4, y_base, n, name)?;
        let (wl, xl, yl) = (w.len() as i64, x.len() as i64, y.len() as i64);
        // SAFETY: the host buffers hold exactly the validated extents and are disjoint.
        let code = unsafe {
            if is_f16 {
                wukong_runtime::wukong_sgemv_f16(w.as_ptr(), wl, x.as_ptr(), xl, y.as_mut_ptr(), yl)
            } else {
                wukong_runtime::wukong_sgemv_bf16(
                    w.as_ptr(),
                    wl,
                    x.as_ptr(),
                    xl,
                    y.as_mut_ptr(),
                    yl,
                )
            }
        };
        if code != GEMV_OK {
            return Err(refused(name, code));
        }
        marshal::write_f32(arena_mut!(self), 4, y_base, &y, name)?;
        marshal::recycle_f32(x);
        marshal::recycle_f32(y);
        Ok(status(GEMV_OK))
    }

    /// `wukong_quantize_{q8_0,q4_0}_f32(x, nx, q, nq, k) -> i64`.
    fn dec_quantize(&mut self, name: &str, args: &[Value], fmt: QFormat) -> Result<Value, String> {
        let (rows, k) =
            match quantize_plan(arg_i64(args, 1), arg_i64(args, 3), arg_i64(args, 4), fmt) {
                Ok(rk) => rk,
                Err(code) => return Ok(status(code)),
            };
        if rows == 0 {
            return Ok(status(QUANT_OK));
        }
        let x_base = ptr_arg(args, 0, name, "x")?;
        let x = marshal::read_f32(arena!(self), 0, x_base, marshal::span(rows, k, name)?, name)?;
        let q_base = ptr_arg(args, 2, name, "q")?;
        let q_len = marshal::span(rows, fmt.row_bytes(k), name)?;
        let mut q: Vec<u8> = marshal::output(arena!(self), 2, q_base, q_len, name)?;
        let (xl, ql) = (x.len() as i64, q.len() as i64);
        // SAFETY: the host buffers hold exactly the validated extents and are disjoint.
        let code = unsafe {
            match fmt {
                QFormat::Q8 => wukong_runtime::wukong_quantize_q8_0_f32(
                    x.as_ptr(),
                    xl,
                    q.as_mut_ptr(),
                    ql,
                    k as i64,
                ),
                QFormat::Q4 => wukong_runtime::wukong_quantize_q4_0_f32(
                    x.as_ptr(),
                    xl,
                    q.as_mut_ptr(),
                    ql,
                    k as i64,
                ),
            }
        };
        if code != QUANT_OK {
            return Err(refused(name, code));
        }
        marshal::write_back(arena_mut!(self), 2, q_base, &q, name, |b| {
            Value::Int(i128::from(b as i8))
        })?;
        marshal::recycle_f32(x);
        Ok(status(QUANT_OK))
    }

    /// `wukong_gemv_{q8_0,q4_0}_f32[_parallel](w, nw, x, nx, y, ny) -> i64`.
    fn dec_gemv_q(&mut self, name: &str, args: &[Value], fmt: QFormat) -> Result<Value, String> {
        let (n, k) = match gemv_q_plan(arg_i64(args, 1), arg_i64(args, 3), arg_i64(args, 5), fmt) {
            Ok(nk) => nk,
            Err(code) => return Ok(status(code)),
        };
        if n == 0 {
            return Ok(status(QUANT_OK));
        }
        let (w, x) = if k == 0 {
            (Vec::new(), Vec::new())
        } else {
            let wb = ptr_arg(args, 0, name, "weight")?;
            let vals = kernel_values(arena!(self), 0, wb, n, fmt.row_bytes(k), name)?;
            let w: Vec<u8> = kernel_buffer(vals, name, |v| v.kernel_int(name).map(|b| b as u8))?;
            let xb = ptr_arg(args, 2, name, "x")?;
            (w, marshal::read_f32(arena!(self), 2, xb, k, name)?)
        };
        let y_base = ptr_arg(args, 4, name, "y")?;
        let mut y = marshal::output_f32(arena!(self), 4, y_base, n, name)?;
        let (wl, xl, yl) = (w.len() as i64, x.len() as i64, y.len() as i64);
        // SAFETY: the host buffers hold exactly the validated extents and are disjoint.
        let code = unsafe {
            match fmt {
                QFormat::Q8 => wukong_runtime::wukong_gemv_q8_0_f32(
                    w.as_ptr(),
                    wl,
                    x.as_ptr(),
                    xl,
                    y.as_mut_ptr(),
                    yl,
                ),
                QFormat::Q4 => wukong_runtime::wukong_gemv_q4_0_f32(
                    w.as_ptr(),
                    wl,
                    x.as_ptr(),
                    xl,
                    y.as_mut_ptr(),
                    yl,
                ),
            }
        };
        if code != QUANT_OK {
            return Err(refused(name, code));
        }
        marshal::write_f32(arena_mut!(self), 4, y_base, &y, name)?;
        marshal::recycle_f32(x);
        marshal::recycle_f32(y);
        Ok(status(QUANT_OK))
    }
}
