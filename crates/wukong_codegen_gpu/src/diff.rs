//! Tolerance differential harness for the GPU backend.
//!
//! The CPU differential gate is bit-exact (interpreter and Cranelift call the *identical* runtime
//! kernel). That trick cannot cross the CPU↔GPU boundary: the GPU reassociates reductions and uses
//! its own rounding / transcendental approximations. So GPU correctness is a **tolerance**
//! differential — GPU output vs the interpreter oracle (the same `wukong_runtime` kernel the
//! interpreter marshals) within a bound that scales with the reduction length, plus an independent
//! f64 accuracy check. Grids are fixed so a GPU result is deterministic run-to-run.

/// **§3A P3 — a gate that cannot run must never report green silently.**
///
/// Every device-gated test in this crate early-`return`s when the GPU (or the thing under test) is
/// unreachable. libtest captures stderr on a *pass*, so that early return is invisible: the suite
/// prints `test result: ok` having executed zero device instructions — the exact "green without the
/// hardware" failure this campaign already tripped over on the peer DLLs. Call this immediately
/// before every such early return.
///
/// By default it only prints the `[skip]` marker, so a genuinely GPU-less box (CI) still passes.
/// With `WUKONG_GPU_REQUIRED=1` — the campaign's own invocation, and any box that is *supposed* to
/// have a device — it fails the test instead. Mirrors `gpu::tests::with_gpu`'s escape hatch so one
/// env var governs every skip in the crate.
#[track_caller]
pub fn skip_or_fail(what: &str, why: &str) {
    assert!(
        !crate::gpu::gpu_required(),
        "{what}: WUKONG_GPU_REQUIRED is set but this gate cannot run ({why}) — it would have \
         reported a green pass having tested NOTHING"
    );
    eprintln!("[skip] {what}: {why}");
}

/// Deterministic, dependency-free PRNG (SplitMix64) for reproducible random test buffers — avoids a
/// `rand` dep and `Math.random`, and gives identical inputs every run so a tolerance is stable.
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Rng(seed)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform `f32` in `[lo, hi)`.
    pub fn f32_range(&mut self, lo: f32, hi: f32) -> f32 {
        let u = ((self.next_u64() >> 40) as f32) / ((1u64 << 24) as f32); // [0,1)
        lo + (hi - lo) * u
    }

    /// `n` uniform `f32` in `[lo, hi)`.
    pub fn vec(&mut self, n: usize, lo: f32, hi: f32) -> Vec<f32> {
        (0..n).map(|_| self.f32_range(lo, hi)).collect()
    }
}

/// Worst-case absolute / relative error between two equal-length buffers (computed in f64).
#[derive(Debug, Clone, Copy)]
pub struct ErrStats {
    pub max_abs: f64,
    pub max_rel: f64,
    pub at: usize,
}

/// Compare element-wise in f64; `reference` is the denominator of the relative error.
pub fn err_stats(got: &[f32], reference: &[f32]) -> ErrStats {
    assert_eq!(got.len(), reference.len());
    let mut s = ErrStats {
        max_abs: 0.0,
        max_rel: 0.0,
        at: 0,
    };
    for (i, (&g, &r)) in got.iter().zip(reference).enumerate() {
        let abs = (g as f64 - r as f64).abs();
        let rel = abs / (r.abs() as f64).max(f32::MIN_POSITIVE as f64);
        if abs > s.max_abs {
            s.max_abs = abs;
            s.at = i;
        }
        s.max_rel = s.max_rel.max(rel);
    }
    s
}

/// Assert `got` is within `abs_tol` OR `rel_tol` of `reference` at every lane; returns the stats so
/// callers can report them. Panics (failing the test) with the worst lane on violation.
pub fn assert_close(
    label: &str,
    got: &[f32],
    reference: &[f32],
    abs_tol: f64,
    rel_tol: f64,
) -> ErrStats {
    let s = err_stats(got, reference);
    // A lane passes if EITHER bound holds (rel for big values, abs for near-zero).
    for (i, (&g, &r)) in got.iter().zip(reference).enumerate() {
        let abs = (g as f64 - r as f64).abs();
        let rel = abs / (r.abs() as f64).max(f32::MIN_POSITIVE as f64);
        assert!(
            abs <= abs_tol || rel <= rel_tol,
            "{label}: lane {i} out of tolerance: got {g}, ref {r} (abs {abs:.3e} > {abs_tol:.1e} \
             and rel {rel:.3e} > {rel_tol:.1e})"
        );
    }
    s
}

/// Assert a single scalar (reduction result) is within tolerance of an f64 reference; returns the
/// relative error. The bound for an N-term reduction is `c·√N·ε` (the `sgemm_matches_naive` shape).
pub fn assert_scalar_close(
    label: &str,
    got: f32,
    reference: f64,
    abs_tol: f64,
    rel_tol: f64,
) -> f64 {
    let abs = (got as f64 - reference).abs();
    let rel = abs / reference.abs().max(f32::MIN_POSITIVE as f64);
    assert!(
        abs <= abs_tol || rel <= rel_tol,
        "{label}: scalar out of tolerance: got {got}, ref {reference} (abs {abs:.3e} > {abs_tol:.1e} \
         and rel {rel:.3e} > {rel_tol:.1e})"
    );
    rel
}
