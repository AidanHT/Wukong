// Rust peer for program 3 — same loop order, same association order as `peer_scan.c`.
// Edition 2015, strict IEEE order (no -ffast-math on stable): pairs with the plain C column.

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
