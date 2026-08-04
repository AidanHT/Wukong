// Rust peer for the STRUCTURE-TAX program — same block, same loop order, same arithmetic as
// `peer_block.c`, written the way a Rust systems programmer would write a kernel behind a C ABI.
//
// Raw pointers rather than slices: the entry point IS a C ABI taking 27 distinct buffers, and
// reconstructing 27 `&mut [f32]` from them would be UB the moment two of them were ever the same
// allocation. `#[inline(always)]` helpers keep the shape readable without costing a call.
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
