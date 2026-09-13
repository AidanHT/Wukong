//! Single-row fused-norm **cost decomposition** probe (dev instrument, not a shipped bench).
//!
//! The 2026-09 CPU evaluation recorded the `1x768` / `1x4096` fused-norm rows at 0.345x-0.49x of
//! `C(fast)` (gcc `-O3 -march=native -ffast-math`), the worst peer losses in the suite. A single
//! wall time per row cannot say *where* those nanoseconds go, and the row is small enough (3 KB /
//! 16 KB, L1/L2-resident) that per-call overhead is a first-class term rather than a rounding
//! error. This probe splits the benchmarked program into the pieces it is actually made of and
//! times each in the same process, adjacently:
//!
//!   * `copy` - `wukong_velem_f32` in its identity-affine mode, which is what the recognizer
//!     emits for the benchmark's leading `for c { out[c] = x[c] }`;
//!   * `norm` - `wukong_norm_f32` alone, the fused reduction + normalize;
//!   * `both` - the two back to back, i.e. everything the JIT'd `kbench` does except its own
//!     prologue, so `measured_kbench - both` is the JIT wrapper's share;
//!   * `C(fast)` - the identical C source the suite compiles for the peer column, built by `$CC`
//!     at `C_FAST_FLAGS` and called through its DLL export **in this same process**, so the
//!     Wukong/peer ratio is a same-run ratio and not a cross-process comparison.
//!
//! Sweeping several `n` and reading off the intercept separates the fixed per-call cost from the
//! per-element cost, which is the question the evaluation left open.
//!
//! Run: `cargo run -p wukong_runtime --release --example norm_var [-- <n,n,..>]`
//! Env: `CC` to pick the C compiler (default `gcc`).

use std::hint::black_box;
use std::path::PathBuf;
use std::process::Command;
use std::time::Instant;
use wukong_runtime::{wukong_norm_f32, wukong_velem_f32};

/// `kbench(x, y, out)` — the 3-pointer signature every xbench peer exports.
type KbenchFn = unsafe extern "C" fn(*const f32, *const f32, *mut f32);

const NORM_LAYERNORM: i64 = 1;
const NORM_RMSNORM: i64 = 2;
const NORM_L2NORM: i64 = 4;
const EPS: f32 = 1e-5;

/// The exact C body `wukong_xbench::c_norm` renders, for the ops this probe covers.
fn c_norm(cols: usize, op: &str) -> String {
    let body = match op {
        "layernorm" => {
            "float s=0.0f; for(long i=0;i<C;i++) s+=out[i]; \
             float mean=s/(float)C; float v=0.0f; \
             for(long i=0;i<C;i++){ float d=out[i]-mean; v+=d*d; } \
             float inv=1.0f/sqrtf(v/(float)C+1e-5f); \
             for(long i=0;i<C;i++) out[i]=(out[i]-mean)*inv;"
        }
        "l2norm" => {
            "float s=0.0f; for(long i=0;i<C;i++) s+=out[i]*out[i]; \
             float inv=1.0f/sqrtf(s+1e-5f); \
             for(long i=0;i<C;i++) out[i]*=inv;"
        }
        _ => {
            "float s=0.0f; for(long i=0;i<C;i++) s+=out[i]*out[i]; \
             float inv=1.0f/sqrtf(s/(float)C+1e-5f); \
             for(long i=0;i<C;i++) out[i]*=inv;"
        }
    };
    format!(
        "#include <math.h>\n#define C {cols}\n__declspec(dllexport) void kbench(const float* __restrict__ x, const float* __restrict__ y, float* __restrict__ out) {{ \
         for(long i=0;i<C;i++) out[i]=x[i]; {body} }}\n"
    )
}

/// Build the C(fast) peer for one (op, n) and load it. Returns the library (kept alive by the
/// caller — dropping it unloads the DLL out from under the function pointer) and the entry point.
fn build_peer(
    dir: &std::path::Path,
    op: &str,
    n: usize,
) -> Option<(libloading::Library, KbenchFn)> {
    let cc = std::env::var("CC").unwrap_or_else(|_| "gcc".to_string());
    let src: PathBuf = dir.join(format!("norm_{op}_{n}.c"));
    let dll: PathBuf = dir.join(format!("norm_{op}_{n}.dll"));
    std::fs::write(&src, c_norm(n, op)).ok()?;
    let st = Command::new(&cc)
        .args(["-O3", "-march=native", "-ffast-math", "-shared"])
        .arg("-o")
        .arg(&dll)
        .arg(&src)
        .status()
        .ok()?;
    if !st.success() {
        eprintln!("  ! {cc} failed to build the C(fast) peer for {op} n={n}");
        return None;
    }
    // SAFETY: a DLL this process just produced from its own source, exporting exactly `kbench`
    // with the 3-pointer signature above; the library is returned so it outlives every call.
    unsafe {
        let lib = libloading::Library::new(&dll).ok()?;
        let sym: libloading::Symbol<KbenchFn> = lib.get(b"kbench").ok()?;
        let f = *sym;
        Some((lib, f))
    }
}

/// A bare 256-bit `out[i] = x[i]` loop — the floor `wukong_velem_f32`'s identity-affine mode is
/// competing against, so the `copy` column can be read as "is the recognizer's copy kernel at the
/// hardware's copy speed, or is it paying for its generality?".
///
/// # Safety
/// `x` and `out` valid for `n` `f32`, non-overlapping.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn copy256(x: *const f32, out: *mut f32, n: usize) {
    use std::arch::x86_64::*;
    let mut i = 0;
    while i + 32 <= n {
        for t in 0..4 {
            _mm256_storeu_ps(out.add(i + 8 * t), _mm256_loadu_ps(x.add(i + 8 * t)));
        }
        i += 32;
    }
    while i + 8 <= n {
        _mm256_storeu_ps(out.add(i), _mm256_loadu_ps(x.add(i)));
        i += 8;
    }
    while i < n {
        *out.add(i) = *x.add(i);
        i += 1;
    }
}

/// Best-of-`rounds` mean-per-call over rep-blocks, the xbench `time_ns` discipline: warm up, then
/// keep the fastest block. `run` is called `reps` times per block.
fn block(reps: u64, mut run: impl FnMut()) -> f64 {
    let t = Instant::now();
    for _ in 0..reps {
        run();
    }
    t.elapsed().as_secs_f64() * 1e9 / reps as f64
}

fn main() {
    let sizes: Vec<usize> = std::env::args()
        .nth(1)
        .map(|s| s.split(',').filter_map(|t| t.trim().parse().ok()).collect())
        .unwrap_or_else(|| vec![768, 1536, 3072, 4096]);
    let dir = std::env::temp_dir().join("wukong_norm_var");
    std::fs::create_dir_all(&dir).expect("scratch dir");

    println!(
        "single-row fused-norm cost decomposition (ns/call, same process, best-of-9 rep-blocks)"
    );
    println!("  copy = wukong_velem_f32 identity-affine; norm = wukong_norm_f32; both = copy+norm");
    println!(
        "\n  {:<10} {:>6} {:>9} {:>9} {:>9} {:>9} {:>9} {:>9}",
        "op", "n", "cp256", "copy", "norm", "both", "C(fast)", "both/Cf"
    );

    for op in ["rmsnorm", "l2norm", "layernorm"] {
        let code = match op {
            "layernorm" => NORM_LAYERNORM,
            "l2norm" => NORM_L2NORM,
            _ => NORM_RMSNORM,
        };
        for &n in &sizes {
            let x: Vec<f32> = (0..n).map(|i| (i % 17) as f32 * 0.5 + 1.0).collect();
            let mut out = vec![0.0f32; n];
            let peer = build_peer(&dir, op, n);
            // Enough reps that a block is milliseconds, not nanoseconds.
            let reps: u64 = (2_000_000 / n as u64).max(200);
            let (xp, op_ptr) = (x.as_ptr(), out.as_mut_ptr());
            let e = EPS.to_bits() as i64;

            // SAFETY (all four closures): `x` and `out` are live `n`-element f32 buffers for the
            // whole of `main`; every entry below is the documented `(x, out, ..)` contract, and the
            // peer's `kbench` has the signature its own source declares.
            let mut f_copy =
                || unsafe { wukong_velem_f32(xp, xp, op_ptr, n as i64, 1.0, 0.0, 0.0, 0) };
            // `black_box` on the pointers: `copy256` is visible to LLVM here, and without it the
            // rep-loop's repeated identical write is hoistable, which would report a fictitious floor.
            let mut f_c256 = || unsafe { copy256(black_box(xp), black_box(op_ptr), n) };
            let mut f_norm = || unsafe { wukong_norm_f32(op_ptr, op_ptr, 1, n as i64, e, code) };
            let mut f_both = || unsafe {
                wukong_velem_f32(xp, xp, op_ptr, n as i64, 1.0, 0.0, 0.0, 0);
                wukong_norm_f32(op_ptr, op_ptr, 1, n as i64, e, code);
            };

            let (mut bc, mut bn, mut bb, mut bp) = (f64::MAX, f64::MAX, f64::MAX, f64::MAX);
            let mut b2 = f64::MAX;
            for r in 0..10 {
                let c2 = block(reps, &mut f_c256);
                let c = block(reps, &mut f_copy);
                let nn = block(reps, &mut f_norm);
                let b = block(reps, &mut f_both);
                let p = peer
                    .as_ref()
                    .map(|(_, f)| block(reps, || unsafe { f(xp, xp, op_ptr) }));
                if r == 0 {
                    continue; // warm-up round, discarded
                }
                b2 = b2.min(c2);
                bc = bc.min(c);
                bn = bn.min(nn);
                bb = bb.min(b);
                if let Some(p) = p {
                    bp = bp.min(p);
                }
            }
            let ratio = if bp.is_finite() { bp / bb } else { f64::NAN };
            println!(
                "  {op:<10} {n:>6} {b2:>9.1} {bc:>9.1} {bn:>9.1} {bb:>9.1} {bp:>9.1} {ratio:>9.3}"
            );
        }
    }
    println!("\n  ratio column is peer_ns/wukong_ns = Wukong speed over C(fast); >= 1.00 is at or past the peer.");
}
