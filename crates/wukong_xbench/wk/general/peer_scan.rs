// Rust peer for program 3 — same loop order, same association order as `peer_scan.c`.
// Edition 2015, strict IEEE order (no -ffast-math on stable): pairs with the plain C column.
//
// ALIASING (corrected 2026-08-06): `peer_scan.c` declares all eight buffers `__restrict__`; this
// file took eight bare raw pointers, which carry no aliasing information in LLVM at all — and this
// kernel is the one where it bites hardest, since `h[db+n]` is stored inside the inner loop and
// without `noalias` every later load of `a`/`bmat`/`cmat` must be assumed clobbered by it. `kbench`
// now hands the slices to `kbody` as PARAMETERS (the only spelling that produces `noalias`) and
// `kbody` re-derives the raw pointers, so every loop below is textually unchanged.

const T: usize = $T$;
const D: usize = $D$;
const N: usize = $N$;

#[no_mangle]
pub unsafe extern "C" fn kbench(
    x: *const f32,
    dt: *const f32,
    a: *const f32,
    bmat: *const f32,
    cmat: *const f32,
    dskip: *const f32,
    h: *mut f32,
    y: *mut f32,
) {
    kbody(
        core::slice::from_raw_parts(x, T * D),
        core::slice::from_raw_parts(dt, T * D),
        core::slice::from_raw_parts(a, D * N),
        core::slice::from_raw_parts(bmat, T * N),
        core::slice::from_raw_parts(cmat, T * N),
        core::slice::from_raw_parts(dskip, D),
        core::slice::from_raw_parts_mut(h, D * N),
        core::slice::from_raw_parts_mut(y, T * D),
    )
}

#[inline(always)]
unsafe fn kbody(
    xs: &[f32],
    dts: &[f32],
    as_: &[f32],
    bmats: &[f32],
    cmats: &[f32],
    dskips: &[f32],
    hs: &mut [f32],
    ys: &mut [f32],
) {
    let (x, dt, a) = (xs.as_ptr(), dts.as_ptr(), as_.as_ptr());
    let (bmat, cmat, dskip) = (bmats.as_ptr(), cmats.as_ptr(), dskips.as_ptr());
    let (h, y) = (hs.as_mut_ptr(), ys.as_mut_ptr());
    for i in 0..D * N {
        *h.add(i) = 0.0;
    }
    for t in 0..T {
        let tb = t * D;
        let nb = t * N;
        for d in 0..D {
            let db = d * N;
            let dtv = *dt.add(tb + d);
            let xv = *x.add(tb + d);
            let gate = dtv * xv;
            let mut acc = 0.0f32;
            for n in 0..N {
                let decay = (dtv * *a.add(db + n)).exp();
                let hn = decay * *h.add(db + n) + gate * *bmat.add(nb + n);
                *h.add(db + n) = hn;
                acc += *cmat.add(nb + n) * hn;
            }
            *y.add(tb + d) = acc + *dskip.add(d) * xv;
        }
    }
}
