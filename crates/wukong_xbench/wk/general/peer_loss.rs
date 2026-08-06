// Rust peer for program 2 — same passes, same association order as `peer_loss.c`.
// Edition 2015 (the harness invokes rustc without `--edition`), strict IEEE order (no -ffast-math
// on stable), so this column pairs with the plain C column, never with C(fast).
//
// ALIASING (corrected 2026-08-06): `peer_loss.c` declares all six buffers `__restrict__`; this file
// took six bare raw pointers, which carry no aliasing information in LLVM at all. `kbench` now hands
// the slices to `kbody` as PARAMETERS — the only spelling that produces `noalias` — and `kbody`
// re-derives the raw pointers so every loop below is textually unchanged. See `peer_block.rs`.

const R: usize = $R$;
const C: usize = $C$;
const QT: f32 = $QT$;
const QO: f32 = $QO$;

#[no_mangle]
pub unsafe extern "C" fn kbench(
    z: *const f32,
    alpha: *const f32,
    tgt: *const i32,
    p: *mut f32,
    loss: *mut f32,
    dz: *mut f32,
) {
    kbody(
        core::slice::from_raw_parts(z, R * C),
        core::slice::from_raw_parts(alpha, C),
        core::slice::from_raw_parts(tgt, R),
        core::slice::from_raw_parts_mut(p, R * C),
        core::slice::from_raw_parts_mut(loss, R),
        core::slice::from_raw_parts_mut(dz, R * C),
    )
}

#[inline(always)]
unsafe fn kbody(
    zs: &[f32],
    alphas: &[f32],
    tgts: &[i32],
    ps: &mut [f32],
    losses: &mut [f32],
    dzs: &mut [f32],
) {
    let (z, alpha, tgt) = (zs.as_ptr(), alphas.as_ptr(), tgts.as_ptr());
    let (p, loss, dz) = (ps.as_mut_ptr(), losses.as_mut_ptr(), dzs.as_mut_ptr());
    for r in 0..R {
        let rb = r * C;

        let mut m = *z.add(rb);
        for c in 0..C {
            let v = *z.add(rb + c);
            m = if v > m { v } else { m };
        }
        let mut sm = 0.0f32;
        for c in 0..C {
            let e = (*z.add(rb + c) - m).exp();
            *p.add(rb + c) = e;
            sm += e;
        }
        let inv = 1.0f32 / sm;
        for c in 0..C {
            *p.add(rb + c) = *p.add(rb + c) * inv;
        }

        let t = *tgt.add(r);
        let mut lr = 0.0f32;
        let mut sg = 0.0f32;
        for c in 0..C {
            let q = if c as i32 == t { QT } else { QO };
            let pc = *p.add(rb + c);
            let om = 1.0f32 - pc;
            let lg = if pc > 1.0e-30 { pc.ln() } else { (1.0e-30f32).ln() };
            let aq = *alpha.add(c) * q;
            lr = lr - aq * om * om * lg;
            let gc = aq * (2.0 * om * lg - om * om / pc);
            *dz.add(rb + c) = gc;
            sg += gc * pc;
        }
        *loss.add(r) = lr;

        for c in 0..C {
            *dz.add(rb + c) = *p.add(rb + c) * (*dz.add(rb + c) - sg);
        }
    }
}
