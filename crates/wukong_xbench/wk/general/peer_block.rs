// Rust peer for the STRUCTURE-TAX program — same block, same loop order, same arithmetic as
// `peer_block.c`, written the way a Rust systems programmer would write a kernel behind a C ABI.
//
// ALIASING (corrected 2026-08-06). `peer_block.c` declares all 27 buffers `__restrict__` and says
// why: they are 27 genuinely distinct allocations, so the qualifier is TRUE and it is what lets gcc
// keep accumulators in registers across the stores. This file used to take 27 bare raw pointers and
// argue that slices would be UB — but the C peer's `__restrict__` already asserts the same
// non-overlap, so the two languages were NOT being given the same information: rustc emits
// `noalias` on reference parameters only, never on raw pointers, so every loop here was compiled
// under may-alias assumptions its C twin was not.
//
// The fix is the shim below and nothing else: `kbench` builds the 27 slices and hands them to
// `kbody` as PARAMETERS (a slice built as a local inside `kbench` grants no aliasing information at
// all — measured: byte-identical asm to raw pointers, and no `noalias` metadata in the IR), and
// `kbody` immediately re-derives the raw pointers so every loop below is textually unchanged. No
// bounds check appears, no accumulation order moves; only the aliasing facts change, which is
// exactly what `__restrict__` does for the C column.
//
// rustc has no `-ffast-math` on stable, so this column is strict-IEEE-ordered like the plain C
// column, never like C(fast). That asymmetry is disclosed in the report rather than hidden.
#![allow(clippy::too_many_arguments)]

const S: usize = $S$;
const D: usize = $D$;
const H: usize = $H$;
const HD: usize = $HD$;
const HD2: usize = $HD2$;
const F: usize = $F$;
const SCALE: f32 = $SCALE$;

#[inline(always)]
fn silu(x: f32) -> f32 {
    x * (1.0 / (1.0 + (-x).exp()))
}

#[inline(always)]
unsafe fn rp<'a>(p: *const f32, n: usize) -> &'a [f32] {
    core::slice::from_raw_parts(p, n)
}
#[inline(always)]
unsafe fn rmp<'a>(p: *mut f32, n: usize) -> &'a mut [f32] {
    core::slice::from_raw_parts_mut(p, n)
}

/// The C-ABI entry. Its only job is to turn the 27 pointers into 27 slice PARAMETERS, which is the
/// spelling that actually carries `noalias` into LLVM — the Rust equivalent of `peer_block.c`'s
/// `__restrict__` on the same 27 buffers. Lengths are the ones `Bufs::new` allocates.
#[no_mangle]
pub unsafe extern "C" fn kbench(
    x: *const f32,
    g1: *const f32,
    g2: *const f32,
    wq: *const f32,
    wk: *const f32,
    wv: *const f32,
    wo: *const f32,
    w1: *const f32,
    w3: *const f32,
    w2: *const f32,
    b1: *const f32,
    rc: *const f32,
    rs: *const f32,
    nrm: *mut f32,
    q: *mut f32,
    k: *mut f32,
    v: *mut f32,
    sc: *mut f32,
    qh: *mut f32,
    kh: *mut f32,
    vt: *mut f32,
    ah: *mut f32,
    ctx: *mut f32,
    h: *mut f32,
    f1: *mut f32,
    f3: *mut f32,
    out: *mut f32,
) {
    kbody(
        rp(x, S * D), rp(g1, D), rp(g2, D),
        rp(wq, D * D), rp(wk, D * D), rp(wv, D * D), rp(wo, D * D),
        rp(w1, F * D), rp(w3, F * D), rp(w2, D * F), rp(b1, F),
        rp(rc, S * HD2), rp(rs, S * HD2),
        rmp(nrm, S * D),
        rmp(q, S * D), rmp(k, S * D), rmp(v, S * D),
        rmp(sc, S * S),
        rmp(qh, S * HD), rmp(kh, S * HD), rmp(vt, HD * S), rmp(ah, S * HD),
        rmp(ctx, S * D),
        rmp(h, S * D),
        rmp(f1, S * F), rmp(f3, S * F),
        rmp(out, S * D),
    )
}

#[inline(always)]
unsafe fn kbody(
    xs: &[f32],
    g1s: &[f32],
    g2s: &[f32],
    wqs: &[f32],
    wks: &[f32],
    wvs: &[f32],
    wos: &[f32],
    w1s: &[f32],
    w3s: &[f32],
    w2s: &[f32],
    b1s: &[f32],
    rcs: &[f32],
    rss: &[f32],
    nrms: &mut [f32],
    qs: &mut [f32],
    ks: &mut [f32],
    vs: &mut [f32],
    scs: &mut [f32],
    qhs: &mut [f32],
    khs: &mut [f32],
    vts: &mut [f32],
    ahs: &mut [f32],
    ctxs: &mut [f32],
    hs: &mut [f32],
    f1s: &mut [f32],
    f3s: &mut [f32],
    outs: &mut [f32],
) {
    // Back to raw pointers, so every loop below is byte-for-byte the code that was measured before
    // the aliasing fix. The `noalias` now rides on the parameters above.
    let (x, g1, g2) = (xs.as_ptr(), g1s.as_ptr(), g2s.as_ptr());
    let (wq, wk, wv, wo) = (wqs.as_ptr(), wks.as_ptr(), wvs.as_ptr(), wos.as_ptr());
    let (w1, w3, w2, b1) = (w1s.as_ptr(), w3s.as_ptr(), w2s.as_ptr(), b1s.as_ptr());
    let (rc, rs) = (rcs.as_ptr(), rss.as_ptr());
    let nrm = nrms.as_mut_ptr();
    let (q, k, v) = (qs.as_mut_ptr(), ks.as_mut_ptr(), vs.as_mut_ptr());
    let sc = scs.as_mut_ptr();
    let (qh, kh, vt, ah) = (qhs.as_mut_ptr(), khs.as_mut_ptr(), vts.as_mut_ptr(), ahs.as_mut_ptr());
    let ctx = ctxs.as_mut_ptr();
    let h = hs.as_mut_ptr();
    let (f1, f3) = (f1s.as_mut_ptr(), f3s.as_mut_ptr());
    let out = outs.as_mut_ptr();

    // 1. RMSNorm(x) * g1 -> nrm
    for r in 0..S {
        let rb = r * D;
        let mut ss = 0.0f32;
        for i in 0..D {
            ss += *x.add(rb + i) * *x.add(rb + i);
        }
        let inv = 1.0f32 / (ss / D as f32 + 0.00001).sqrt();
        for i in 0..D {
            *nrm.add(rb + i) = *x.add(rb + i) * inv * *g1.add(i);
        }
    }

    // 2. Q/K/V projections, NT layout. Spelled out three times rather than iterated over an array
    // of (weight, dest) pairs: rustc is invoked from the harness without `--edition`, so this is
    // edition 2015, where an array is not `IntoIterator` by value.
    for i in 0..S {
        let ib = i * D;
        for j in 0..D {
            let jb = j * D;
            let mut acc = 0.0f32;
            for p in 0..D {
                acc += *nrm.add(ib + p) * *wq.add(jb + p);
            }
            *q.add(ib + j) = acc;
        }
    }
    for i in 0..S {
        let ib = i * D;
        for j in 0..D {
            let jb = j * D;
            let mut acc = 0.0f32;
            for p in 0..D {
                acc += *nrm.add(ib + p) * *wk.add(jb + p);
            }
            *k.add(ib + j) = acc;
        }
    }
    for i in 0..S {
        let ib = i * D;
        for j in 0..D {
            let jb = j * D;
            let mut acc = 0.0f32;
            for p in 0..D {
                acc += *nrm.add(ib + p) * *wv.add(jb + p);
            }
            *v.add(ib + j) = acc;
        }
    }

    // 3. RoPE (rotate-half) on q and k
    for r in 0..S {
        let tb = r * HD2;
        let rb = r * D;
        for e in 0..H {
            let hb = rb + e * HD;
            for t in 0..HD2 {
                let (cc, sn) = (*rc.add(tb + t), *rs.add(tb + t));
                let (q0, q1) = (*q.add(hb + t), *q.add(hb + t + HD2));
                *q.add(hb + t) = q0 * cc - q1 * sn;
                *q.add(hb + t + HD2) = q0 * sn + q1 * cc;
                let (k0, k1) = (*k.add(hb + t), *k.add(hb + t + HD2));
                *k.add(hb + t) = k0 * cc - k1 * sn;
                *k.add(hb + t + HD2) = k0 * sn + k1 * cc;
            }
        }
    }

    // 4. Causal multi-head attention
    for e in 0..H {
        let ho = e * HD;
        for i in 0..S {
            let (src, dst) = (i * D + ho, i * HD);
            for p in 0..HD {
                *qh.add(dst + p) = *q.add(src + p);
                *kh.add(dst + p) = *k.add(src + p);
            }
        }
        for j in 0..HD {
            let dst = j * S;
            for p in 0..S {
                *vt.add(dst + p) = *v.add(p * D + ho + j);
            }
        }
        for i in 0..S {
            let (ib, so) = (i * HD, i * S);
            for j in 0..S {
                let jb = j * HD;
                let mut acc = 0.0f32;
                for p in 0..HD {
                    acc += *qh.add(ib + p) * *kh.add(jb + p);
                }
                *sc.add(so + j) = SCALE * acc;
            }
        }
        for i in 0..S {
            let so = i * S;
            for j in 0..S {
                if j > i {
                    *sc.add(so + j) = -1.0e30;
                }
            }
        }
        for r in 0..S {
            let so = r * S;
            let mut m = *sc.add(so);
            for i in 0..S {
                let c = *sc.add(so + i);
                m = if c > m { c } else { m };
            }
            for i in 0..S {
                *sc.add(so + i) = (*sc.add(so + i) - m).exp();
            }
            let mut sm = 0.0f32;
            for i in 0..S {
                sm += *sc.add(so + i);
            }
            let inv = 1.0f32 / sm;
            for i in 0..S {
                *sc.add(so + i) = *sc.add(so + i) * inv;
            }
        }
        for i in 0..S {
            let (so, ib) = (i * S, i * HD);
            for j in 0..HD {
                let jb = j * S;
                let mut acc = 0.0f32;
                for p in 0..S {
                    acc += *sc.add(so + p) * *vt.add(jb + p);
                }
                *ah.add(ib + j) = acc;
            }
        }
        for i in 0..S {
            let (dst, ib) = (i * D + ho, i * HD);
            for j in 0..HD {
                *ctx.add(dst + j) = *ah.add(ib + j);
            }
        }
    }

    // 5. Output projection + residual
    for i in 0..S {
        let ib = i * D;
        for j in 0..D {
            let jb = j * D;
            let mut acc = 0.0f32;
            for p in 0..D {
                acc += *ctx.add(ib + p) * *wo.add(jb + p);
            }
            *h.add(ib + j) = *x.add(ib + j) + acc;
        }
    }

    // 6. RMSNorm(h) * g2 -> nrm
    for r in 0..S {
        let rb = r * D;
        let mut ss = 0.0f32;
        for i in 0..D {
            ss += *h.add(rb + i) * *h.add(rb + i);
        }
        let inv = 1.0f32 / (ss / D as f32 + 0.00001).sqrt();
        for i in 0..D {
            *nrm.add(rb + i) = *h.add(rb + i) * inv * *g2.add(i);
        }
    }

    // 7. SwiGLU MLP
    for i in 0..S {
        let (ib, ob) = (i * D, i * F);
        for j in 0..F {
            let jb = j * D;
            let mut acc = 0.0f32;
            for p in 0..D {
                acc += *nrm.add(ib + p) * *w1.add(jb + p);
            }
            *f1.add(ob + j) = silu(*b1.add(j) + acc);
        }
    }
    for i in 0..S {
        let (ib, ob) = (i * D, i * F);
        for j in 0..F {
            let jb = j * D;
            let mut acc = 0.0f32;
            for p in 0..D {
                acc += *nrm.add(ib + p) * *w3.add(jb + p);
            }
            *f3.add(ob + j) = acc;
            *f1.add(ob + j) = *f1.add(ob + j) * acc;
        }
    }

    // 8. Down projection + residual
    for i in 0..S {
        let (ib, fb) = (i * D, i * F);
        for j in 0..D {
            let jb = j * F;
            let mut acc = 0.0f32;
            for p in 0..F {
                acc += *f1.add(fb + p) * *w2.add(jb + p);
            }
            *out.add(ib + j) = *h.add(ib + j) + acc;
        }
    }
}
