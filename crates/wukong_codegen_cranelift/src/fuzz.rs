//! Randomized **full-buffer** differential fuzzer: interpreter vs native (Cranelift).
//!
//! The older differential tests reduced a kernel's output to a single scalar (`(s*1000.0) as i32`)
//! and compared exit codes, and the cross-language harness cross-checked only a 3-point checksum —
//! so a kernel wrong at every index *except* the few sampled ones slipped through. This fuzzer
//! closes that hole: for a battery of kernels it generates random input buffers across many sizes
//! (deliberately hitting the SIMD remainder boundaries) and adversarial value regimes (NaN, ±Inf,
//! denormals, ±0, huge magnitudes), runs the *same* kernel through both backends over identical
//! buffers via the typed kernel-entry ABI (`wukong_interp::run_kernel_f32`/`run_kernel_i8` /
//! `func_ptr`), and asserts the **entire** output buffer is bit-for-bit identical. The int8
//! `u8×i8→i32` GEMM gets its own pass (`fuzz_full_buffer_i8_interp_vs_native`) over the kernel's
//! K-chunk boundaries.
//!
//! Both backends execute the same lowered MIR and marshal through the same runtime microkernels, so
//! bit-exactness is the correct (strongest) bar — not a tolerance. A divergence is a real miscompile.
//!
//! That same fact makes the interp-vs-native comparison blind to a WRONG kernel: where a kernel is
//! recognized, both sides call one runtime function. So `fuzz_full_buffer_interp_vs_native` also
//! lowers every kernel with recognition OFF (`LowerOptions { recognize: false, .. }`, the loop as the
//! user wrote it) and holds the recognized native output to that second opinion lane by lane: bit-exact
//! for the kernels documented bit-exact with their source loop, within a declared, derived bound for
//! the reassociating ones (`OffBound`). The split is printed.
//!
//! The file also carries `vmath_kernels_match_f64_reference`, which is deliberately *not* a
//! differential test: it checks the transcendental vmath kernels against an f64 reference under a
//! per-op relative-error bound, since interp-vs-native cannot catch a kernel that both backends
//! compute identically wrong. Its bounds are recorded measurements — do not "tidy" them.

use wukong_span::{Interner, SourceId, Symbol};

/// SplitMix64 — a tiny, dependency-free, fully deterministic PRNG so the fuzz corpus is identical
/// on every machine and every run (a failure reproduces exactly).
struct Rng(u64);
impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    /// A float in `[0, 1)` with 24 bits of entropy.
    fn unit(&mut self) -> f32 {
        ((self.next_u64() >> 40) as f32) / ((1u64 << 24) as f32)
    }
}

/// Which numeric regime to draw inputs from.
#[derive(Clone, Copy)]
enum Regime {
    /// Well-behaved values in roughly `[-2, 2)`.
    Normal,
    /// Strictly positive (domain of `log`/`sqrt`/`rsqrt`).
    Positive,
    /// A grab-bag of IEEE corner cases: ±0, ±1, ±Inf, NaN, denormals, and huge magnitudes.
    Adversarial,
}

fn gen(rng: &mut Rng, regime: Regime) -> f32 {
    match regime {
        Regime::Normal => rng.unit() * 4.0 - 2.0,
        Regime::Positive => rng.unit() * 8.0 + f32::MIN_POSITIVE,
        Regime::Adversarial => match rng.next_u64() % 12 {
            0 => 0.0,
            1 => -0.0,
            2 => 1.0,
            3 => -1.0,
            4 => f32::INFINITY,
            5 => f32::NEG_INFINITY,
            6 => f32::NAN,
            7 => f32::MIN_POSITIVE * rng.unit(), // a denormal
            8 => 1e30,
            9 => -1e30,
            10 => rng.unit() * 200.0 - 100.0,
            _ => rng.unit() * 4.0 - 2.0,
        },
    }
}

/// Two f32 results are "the same" if their bits match exactly, or both are NaN — payloads may differ
/// between an `f64`-rounding interpreter and a real-`f32` JIT, and no Wukong program can read one.
///
/// `+0.0` and `-0.0` are *not* interchangeable and used to be accepted here. The sign of zero is
/// fully observable: `print(-0.0)` emits `-0` and `1.0 / (-1.0 * 0.0)` emits `-inf`, identically on
/// both backends. A fold like `x * 0.0 -> 0.0` (wrong for `x < 0`, `-0.0`, `±Inf`, NaN) would make
/// every multiplying kernel here produce `+0.0` natively against the interpreter's `-0.0`, and this
/// fuzzer would pass every run.
fn same(a: f32, b: f32) -> bool {
    (a.is_nan() && b.is_nan()) || a.to_bits() == b.to_bits()
}

/// A kernel under test: a source template (parameterized by element count `n`), its element count,
/// and the regimes its inputs may take.
struct Kernel {
    name: &'static str,
    /// `n` -> Wukong source defining `fn kbench(x:[f32;LEN], y:[f32;LEN], out:[f32;LEN])`.
    src: fn(usize) -> String,
    /// Buffer length given the size parameter `n` (== `n` for elementwise, `n*n` for matmul).
    len: fn(usize) -> usize,
    regimes: &'static [Regime],
}

/// `fn kbench(x:[f32;LEN], y:[f32;LEN], out:[f32;LEN]) { <body over 0..N> }`
fn ew(n: usize, body: &str) -> String {
    let len = n;
    format!("module f\nfn kbench(x:[f32;{len}], y:[f32;{len}], mut out:[f32;{len}]) {{ for i in 0..{n} {{ {body} }} }}\n")
}

fn kernels() -> Vec<Kernel> {
    use Regime::*;
    const ANY: &[Regime] = &[Normal, Adversarial];
    const POS: &[Regime] = &[Positive, Adversarial];
    vec![
        Kernel {
            name: "saxpy",
            src: |n| ew(n, "out[i] = 2.0 * x[i] + y[i];"),
            len: |n| n,
            regimes: ANY,
        },
        Kernel {
            name: "mul",
            src: |n| ew(n, "out[i] = x[i] * y[i];"),
            len: |n| n,
            regimes: ANY,
        },
        Kernel {
            name: "fma3",
            src: |n| ew(n, "out[i] = x[i] * y[i] + x[i];"),
            len: |n| n,
            regimes: ANY,
        },
        Kernel {
            name: "relu",
            src: |n| ew(n, "out[i] = if x[i] > 0.0 { x[i] } else { 0.0 };"),
            len: |n| n,
            regimes: ANY,
        },
        Kernel {
            name: "relu6",
            src: |n| {
                ew(
                    n,
                    "out[i] = if x[i] < 6.0 { if x[i] > 0.0 { x[i] } else { 0.0 } } else { 6.0 };",
                )
            },
            len: |n| n,
            regimes: ANY,
        },
        Kernel {
            name: "poly",
            src: |n| {
                ew(n, "let v: f32 = x[i]; let mut r: f32 = 0.001; r = r*v + 0.01; r = r*v + 0.1; r = r*v + 1.0; out[i] = r;")
            },
            len: |n| n,
            regimes: ANY,
        },
        Kernel {
            name: "sqrt",
            src: |n| ew(n, "out[i] = sqrt(x[i]);"),
            len: |n| n,
            regimes: POS,
        },
        // abs = max(x, -x), a compare+select that vectorizes; the adversarial regime (negatives,
        // ±0, NaN, inf) is exactly where the two backends must still agree lane-for-lane.
        Kernel {
            name: "abs",
            src: |n| ew(n, "out[i] = abs(x[i]);"),
            len: |n| n,
            regimes: ANY,
        },
        // round (ties-to-even) / floor / ceil / trunc — one hardware round-to-integral each. The
        // adversarial regime (huge magnitudes already integral, .5 ties, NaN, inf) pins native
        // `nearest`/`floor`/`ceil`/`trunc` == interp `round_ties_even`/`floor`/`ceil`/`trunc`.
        Kernel {
            name: "round",
            src: |n| ew(n, "out[i] = round(x[i]);"),
            len: |n| n,
            regimes: ANY,
        },
        Kernel {
            name: "floor",
            src: |n| ew(n, "out[i] = floor(x[i]);"),
            len: |n| n,
            regimes: ANY,
        },
        Kernel {
            name: "ceil",
            src: |n| ew(n, "out[i] = ceil(x[i]);"),
            len: |n| n,
            regimes: ANY,
        },
        Kernel {
            name: "trunc",
            src: |n| ew(n, "out[i] = trunc(x[i]);"),
            len: |n| n,
            regimes: ANY,
        },
        Kernel {
            name: "exp",
            src: |n| ew(n, "out[i] = exp(x[i]);"),
            len: |n| n,
            regimes: ANY,
        },
        Kernel {
            name: "log",
            src: |n| ew(n, "out[i] = log(x[i]);"),
            len: |n| n,
            regimes: POS,
        },
        Kernel {
            name: "tanh",
            src: |n| ew(n, "out[i] = tanh(x[i]);"),
            len: |n| n,
            regimes: ANY,
        },
        Kernel {
            name: "sigmoid",
            src: |n| ew(n, "out[i] = sigmoid(x[i]);"),
            len: |n| n,
            regimes: ANY,
        },
        Kernel {
            name: "gelu",
            src: |n| ew(n, "out[i] = gelu(x[i]);"),
            len: |n| n,
            regimes: ANY,
        },
        Kernel {
            name: "silu",
            src: |n| ew(n, "out[i] = silu(x[i]);"),
            len: |n| n,
            regimes: ANY,
        },
        Kernel {
            name: "elu",
            src: |n| ew(n, "out[i] = elu(x[i]);"),
            len: |n| n,
            regimes: ANY,
        },
        Kernel {
            name: "leaky_relu",
            src: |n| ew(n, "out[i] = leaky_relu(x[i]);"),
            len: |n| n,
            regimes: ANY,
        },
        Kernel {
            name: "softplus",
            src: |n| ew(n, "out[i] = softplus(x[i]);"),
            len: |n| n,
            regimes: ANY,
        },
        Kernel {
            name: "mish",
            src: |n| ew(n, "out[i] = mish(x[i]);"),
            len: |n| n,
            regimes: ANY,
        },
        Kernel {
            name: "selu",
            src: |n| ew(n, "out[i] = selu(x[i]);"),
            len: |n| n,
            regimes: ANY,
        },
        Kernel {
            name: "tanhshrink",
            src: |n| ew(n, "out[i] = tanhshrink(x[i]);"),
            len: |n| n,
            regimes: ANY,
        },
        Kernel {
            name: "hardsigmoid",
            src: |n| ew(n, "out[i] = hardsigmoid(x[i]);"),
            len: |n| n,
            regimes: ANY,
        },
        Kernel {
            name: "hardswish",
            src: |n| ew(n, "out[i] = hardswish(x[i]);"),
            len: |n| n,
            regimes: ANY,
        },
        // Reductions (result lands in out[0]); the rest of `out` stays 0 on both backends.
        Kernel {
            name: "dot",
            src: |n| ew(n, "out[0] = out[0] + x[i] * y[i];"),
            len: |n| n,
            regimes: ANY,
        },
        Kernel {
            name: "ssd",
            src: |n| ew(n, "out[0] = out[0] + (x[i] - y[i]) * (x[i] - y[i]);"),
            len: |n| n,
            regimes: ANY,
        },
        // Fused softmax: the recognizer collapses the max/exp/sum/normalize passes into one
        // wukong_norm_f32 call (in place on `out`). The interpreter marshals the identical kernel,
        // so the full probability buffer must be bit-exact against the native JIT.
        Kernel {
            name: "softmax",
            src: |n| {
                format!(
                    "module f\nfn kbench(x:[f32;{n}], y:[f32;{n}], mut out:[f32;{n}]) {{ \
             for c in 0..{n} {{ out[c] = x[c]; }} \
             let mut m: f32 = out[0]; \
             for i in 0..{n} {{ m = fmax(m, out[i]); }} \
             for i in 0..{n} {{ out[i] = exp(out[i] - m); }} \
             let mut s: f32 = 0.0; \
             for i in 0..{n} {{ s = s + out[i]; }} \
             let inv: f32 = 1.0 / s; \
             for i in 0..{n} {{ out[i] = out[i] * inv; }} }}\n"
                )
            },
            len: |n| n,
            regimes: ANY,
        },
        // Fused LayerNorm (gamma=1, beta=0): the recognizer collapses mean / variance / normalize
        // into one wukong_norm_f32(.., NORM_LAYERNORM) call, in place on `out`. The `/ {n}.0`
        // divisor matches the trip count for every fuzz size, so the kernel fires across all sizes
        // and adversarial regimes; both backends marshal the identical kernel → full-buffer exact.
        Kernel {
            name: "layernorm",
            src: |n| {
                format!(
                    "module f\nfn kbench(x:[f32;{n}], y:[f32;{n}], mut out:[f32;{n}]) {{ \
             for c in 0..{n} {{ out[c] = x[c]; }} \
             let mut s: f32 = 0.0; \
             for i in 0..{n} {{ s = s + out[i]; }} \
             let mean: f32 = s / {n}.0; \
             let mut v: f32 = 0.0; \
             for i in 0..{n} {{ v = v + (out[i] - mean) * (out[i] - mean); }} \
             let inv: f32 = rsqrt(v / {n}.0 + 0.00001); \
             for i in 0..{n} {{ out[i] = (out[i] - mean) * inv; }} }}\n"
                )
            },
            len: |n| n,
            regimes: ANY,
        },
        // Fused RMSNorm (gamma=1): mean-square / rsqrt / scale → one NORM_RMSNORM call, in place.
        Kernel {
            name: "rmsnorm",
            src: |n| {
                format!(
                    "module f\nfn kbench(x:[f32;{n}], y:[f32;{n}], mut out:[f32;{n}]) {{ \
             for c in 0..{n} {{ out[c] = x[c]; }} \
             let mut s: f32 = 0.0; \
             for i in 0..{n} {{ s = s + out[i] * out[i]; }} \
             let inv: f32 = rsqrt(s / {n}.0 + 0.00001); \
             for i in 0..{n} {{ out[i] = out[i] * inv; }} }}\n"
                )
            },
            len: |n| n,
            regimes: ANY,
        },
        // Fused affine RMSNorm with a learned per-column scale gamma (= y): the normalize step
        // `out[i]*inv*gamma[i]` dispatches to wukong_norm_affine_f32 (gamma non-null, beta null —
        // exercises the absent-beta marshalling). The real transformer RMSNorm form.
        Kernel {
            name: "rmsnorm_affine",
            src: |n| {
                format!(
                    "module f\nfn kbench(x:[f32;{n}], y:[f32;{n}], mut out:[f32;{n}]) {{ \
             for c in 0..{n} {{ out[c] = x[c]; }} \
             let mut s: f32 = 0.0; \
             for i in 0..{n} {{ s = s + out[i] * out[i]; }} \
             let inv: f32 = rsqrt(s / {n}.0 + 0.00001); \
             for i in 0..{n} {{ out[i] = out[i] * inv * y[i]; }} }}\n"
                )
            },
            len: |n| n,
            regimes: ANY,
        },
        // Fused affine LayerNorm with BOTH scale gamma and shift beta (both = y, so both pointers are
        // non-null): `(out[i]-mean)*inv*gamma[i] + beta[i]` → one wukong_norm_affine_f32 call. Covers
        // the gamma+beta marshalling on both backends; bit-exact full-buffer.
        Kernel {
            name: "layernorm_affine",
            src: |n| {
                format!(
                    "module f\nfn kbench(x:[f32;{n}], y:[f32;{n}], mut out:[f32;{n}]) {{ \
             for c in 0..{n} {{ out[c] = x[c]; }} \
             let mut s: f32 = 0.0; \
             for i in 0..{n} {{ s = s + out[i]; }} \
             let mean: f32 = s / {n}.0; \
             let mut v: f32 = 0.0; \
             for i in 0..{n} {{ v = v + (out[i] - mean) * (out[i] - mean); }} \
             let inv: f32 = rsqrt(v / {n}.0 + 0.00001); \
             for i in 0..{n} {{ out[i] = (out[i] - mean) * inv * y[i] + y[i]; }} }}\n"
                )
            },
            len: |n| n,
            regimes: ANY,
        },
        // Batched RMSNorm: a `for r in 0..3 { <RMSNorm over out[r*C + i]> }` outer loop normalizes
        // each of 3 rows independently → one wukong_norm_f32(.., rows=3, cols=C, RMSNORM) call. This
        // is the ONLY kernel exercising the rows>1 path on the native backend: the kernel loops rows
        // and the interp marshals rows*cols (not just cols) floats. C = n straddles the kernel's
        // internal column vectorization across every fuzz size — the real [batch*seq, hidden] shape.
        Kernel {
            name: "rmsnorm_batched",
            src: |n| {
                format!(
                    "module f\nfn kbench(x:[f32;{l}], y:[f32;{l}], mut out:[f32;{l}]) {{ \
             for c in 0..{l} {{ out[c] = x[c]; }} \
             for r in 0..3 {{ \
             let mut s: f32 = 0.0; \
             for i in 0..{n} {{ s = s + out[r*{n}+i] * out[r*{n}+i]; }} \
             let inv: f32 = rsqrt(s / {n}.0 + 0.00001); \
             for i in 0..{n} {{ out[r*{n}+i] = out[r*{n}+i] * inv; }} }} }}\n",
                    l = 3 * n
                )
            },
            len: |n| 3 * n,
            regimes: ANY,
        },
        // Batched AFFINE RMSNorm — the real transformer form: a learned per-COLUMN scale `y[i]`
        // (length cols, shared across rows) rides the writeback → wukong_norm_affine_f32(.., rows=3).
        // Confirms the batched data offset `out[r*C+i]` and the column-indexed gamma `y[i]` compose.
        Kernel {
            name: "rmsnorm_affine_batched",
            src: |n| {
                format!(
                    "module f\nfn kbench(x:[f32;{l}], y:[f32;{l}], mut out:[f32;{l}]) {{ \
             for c in 0..{l} {{ out[c] = x[c]; }} \
             for r in 0..3 {{ \
             let mut s: f32 = 0.0; \
             for i in 0..{n} {{ s = s + out[r*{n}+i] * out[r*{n}+i]; }} \
             let inv: f32 = rsqrt(s / {n}.0 + 0.00001); \
             for i in 0..{n} {{ out[r*{n}+i] = out[r*{n}+i] * inv * y[i]; }} }} }}\n",
                    l = 3 * n
                )
            },
            len: |n| 3 * n,
            regimes: ANY,
        },
        // Batched softmax: `for r in 0..3 { <softmax over out[r*C + i]> }` → one
        // wukong_norm_f32(.., rows=3, cols=C, SOFTMAX) call. Exercises the rows>1 path for the
        // max/exp/sum/normalize window — including the row-local `out[r*C]` max-seed (the bare
        // `row*cols` offset) — over the vectorized `exp`. The [heads*seq, seq] attention-score shape.
        Kernel {
            name: "softmax_batched",
            src: |n| {
                format!(
                    "module f\nfn kbench(x:[f32;{l}], y:[f32;{l}], mut out:[f32;{l}]) {{ \
             for c in 0..{l} {{ out[c] = x[c]; }} \
             for r in 0..3 {{ \
             let mut m: f32 = out[r*{n}]; \
             for i in 0..{n} {{ m = fmax(m, out[r*{n}+i]); }} \
             for i in 0..{n} {{ out[r*{n}+i] = exp(out[r*{n}+i] - m); }} \
             let mut s: f32 = 0.0; \
             for i in 0..{n} {{ s = s + out[r*{n}+i]; }} \
             let inv: f32 = 1.0 / s; \
             for i in 0..{n} {{ out[r*{n}+i] = out[r*{n}+i] * inv; }} }} }}\n",
                    l = 3 * n
                )
            },
            len: |n| 3 * n,
            regimes: ANY,
        },
        // Batched LayerNorm: `for r in 0..3 { <LayerNorm over out[r*C + i]> }` → one
        // wukong_norm_f32(.., rows=3, cols=C, LAYERNORM) call. Exercises the rows>1 path for the
        // two-reduction (mean, then variance) norm — the offset machinery threaded through the
        // sum/centered-variance/center-scale bodies. Same [batch*seq, hidden] transformer shape.
        Kernel {
            name: "layernorm_batched",
            src: |n| {
                format!(
                    "module f\nfn kbench(x:[f32;{l}], y:[f32;{l}], mut out:[f32;{l}]) {{ \
             for c in 0..{l} {{ out[c] = x[c]; }} \
             for r in 0..3 {{ \
             let mut s: f32 = 0.0; \
             for i in 0..{n} {{ s = s + out[r*{n}+i]; }} \
             let mean: f32 = s / {n}.0; \
             let mut v: f32 = 0.0; \
             for i in 0..{n} {{ v = v + (out[r*{n}+i] - mean) * (out[r*{n}+i] - mean); }} \
             let inv: f32 = rsqrt(v / {n}.0 + 0.00001); \
             for i in 0..{n} {{ out[r*{n}+i] = (out[r*{n}+i] - mean) * inv; }} }} }}\n",
                    l = 3 * n
                )
            },
            len: |n| 3 * n,
            regimes: ANY,
        },
        // matmul C=A·B (ikj) and nn.Linear C=A·Bᵀ (ijk) — both dispatch to the GEMM microkernel.
        Kernel {
            name: "matmul",
            src: |n| {
                format!(
                    "module f\nfn kbench(x:[f32;{l}], y:[f32;{l}], mut out:[f32;{l}]) {{ \
             for i in 0..{n} {{ for j0 in 0..{n} {{ out[i*{n}+j0] = 0.0; }} \
             for k in 0..{n} {{ let aik: f32 = x[i*{n}+k]; \
             for j in 0..{n} {{ out[i*{n}+j] = out[i*{n}+j] + aik * y[k*{n}+j]; }} }} }} }}\n",
                    l = n * n
                )
            },
            len: |n| n * n,
            regimes: &[Normal],
        },
        Kernel {
            name: "linear",
            src: |n| {
                format!(
                    "module f\nfn kbench(x:[f32;{l}], y:[f32;{l}], mut out:[f32;{l}]) {{ \
             for i in 0..{n} {{ for j in 0..{n} {{ let mut s: f32 = 0.0; \
             for k in 0..{n} {{ s = s + x[i*{n}+k] * y[j*{n}+k]; }} out[i*{n}+j] = s; }} }} }}\n",
                    l = n * n
                )
            },
            len: |n| n * n,
            regimes: &[Normal],
        },
        // The shape-typed tensor surface: `Tensor[f32, M, N]` params + multi-dimensional indexing
        // `t[i, j]` now lower and EXECUTE (row-major flat offset), bit-exact on both backends.
        Kernel {
            name: "tensor_scale",
            src: |n| {
                format!(
                    "module f\nfn kbench(x:Tensor[f32,{n},{n}], y:Tensor[f32,{n},{n}], mut out:Tensor[f32,{n},{n}]) {{ \
             for i in 0..{n} {{ for j in 0..{n} {{ out[i, j] = x[i, j] * 2.0 + y[i, j]; }} }} }}\n"
                )
            },
            len: |n| n * n,
            regimes: ANY,
        },
        Kernel {
            name: "tensor_matmul",
            src: |n| {
                format!(
                    "module f\nfn kbench(x:Tensor[f32,{n},{n}], y:Tensor[f32,{n},{n}], mut out:Tensor[f32,{n},{n}]) {{ \
             for i in 0..{n} {{ for j in 0..{n} {{ let mut s: f32 = 0.0; \
             for k in 0..{n} {{ s = s + x[i, k] * y[k, j]; }} out[i, j] = s; }} }} }}\n"
                )
            },
            len: |n| n * n,
            regimes: &[Normal],
        },
        // Rank-3: a batched elementwise op over `Tensor[f32, 2, N, N]`, exercising a 3-index flatten.
        Kernel {
            name: "tensor3d",
            src: |n| {
                format!(
                    "module f\nfn kbench(x:Tensor[f32,2,{n},{n}], y:Tensor[f32,2,{n},{n}], mut out:Tensor[f32,2,{n},{n}]) {{ \
             for b in 0..2 {{ for i in 0..{n} {{ for j in 0..{n} {{ out[b, i, j] = x[b, i, j] - y[b, i, j]; }} }} }} }}\n"
                )
            },
            len: |n| 2 * n * n,
            regimes: ANY,
        },
    ]
}

type K3 = unsafe extern "C" fn(*const f32, *const f32, *mut f32);

/// How a kernel's RECOGNIZED native output must relate to the same source lowered with recognition
/// OFF and run on the interpreter (the language's own definition of the loop).
#[derive(Clone, Copy, Debug)]
enum OffBound {
    /// Bit-identical (`same`) in every regime: the kernel is documented bit-exact with its source
    /// loop. The elementwise `vmath`/`velem`/`vhorner` kernels mirror the generic lowering's own
    /// inline polynomials and FMA contraction (`emit_exp`/`emit_gelu`/... in `wukong_mir_build`,
    /// whose doc says the inlined form "is bit-identical to the kernel"), and data movement
    /// (transpose, copy) and every kernel that does not dispatch are bit-exact by construction. The
    /// two IEEE-corner lane classes where that documentation is wrong today are recorded, with
    /// reproducers, in [`KNOWN_OFF_DIVERGENCES`].
    Exact,
    /// A reassociating fold (the reduction and GEMM kernels: 8-wide partial sums, KC blocking). Two
    /// summation orders of the same terms each lie within `gamma_n * S` of the exact sum (Higham,
    /// *Accuracy and Stability of Numerical Algorithms*, 2nd ed., eq. 3.5, `gamma_n = n*u/(1-n*u)`,
    /// `u = 2^-24`), `S` the sum of the terms' magnitudes; so they differ by at most `2 * gamma_n * S`.
    /// Declared per lane as `2 * (n + 1) * 2^-24 * S`, `S` computed in f64 from the inputs
    /// ([`fold_terms`]); the `+ 1` covers the accumulator's starting `+ 0.0` and an FMA-versus-mul
    /// contraction difference. Non-finite lanes and zero lanes must match exactly (`same`, sign of
    /// zero included — [`off_tol_lane`]).
    Fold,
    /// A fused row normalization (softmax / LayerNorm / RMSNorm, plain and affine, single and
    /// batched): `|on - off| <= tol` absolute, twice the kernel's documented f64-reference bound on
    /// unit-scale rows — softmax `< 1e-5` (`wukong_runtime/src/norm.rs` `matches_f64_reference`),
    /// LayerNorm / RMSNorm `< 1e-3` (the same test), affine `< 2e-3`
    /// (`affine_matches_f64_reference`). Compared in the `Normal` regime only: those bounds are
    /// stated for unit-scale data, and under +-1e30 / inf / NaN inputs the kernel and the generic
    /// loop overflow their (differently ordered) mean / variance / exp sums at different points.
    /// Non-finite and zero lanes are held to `same`, sign of zero included ([`off_tol_lane`]).
    Abs(f64),
}

/// A lane class where a recognized kernel is KNOWN to disagree with its own source loop — an open
/// recognizer defect, recorded with its reproducer so it cannot hide. This is not a tolerance: the
/// lane must match `matches` EXACTLY (input, recognized output, recognition-off output), every other
/// lane of the kernel is still held bit-exact, and the run fails if an entry stops reproducing (so a
/// fix deletes the entry and the lane goes back under the bit-exact bar, the way recognize.rs's
/// `NATIVE_AXIS_WITHHELD` works).
struct KnownDivergence {
    kernel: &'static str,
    /// `(x[idx], recognized native out[idx], recognition-off interp out[idx])` -> is this the defect?
    matches: fn(f32, f32, f32) -> bool,
    defect: &'static str,
}

fn saxpy_loses_negative_zero(_x: f32, on: f32, off: f32) -> bool {
    on.to_bits() == 0 && off.to_bits() == 0x8000_0000
}

fn relu6_nan_clamps_to_zero(x: f32, on: f32, off: f32) -> bool {
    x.is_nan() && on.to_bits() == 0 && off == 6.0
}

const KNOWN_OFF_DIVERGENCES: &[KnownDivergence] = &[
    KnownDivergence {
        kernel: "saxpy",
        matches: saxpy_loses_negative_zero,
        defect: "`out[i] = 2.0 * x[i] + y[i]` with x = y = -0.0 prints -0 as written \
                 (WUKONG_NO_RECOGNIZE=1) but 0 when recognized: wukong_velem_f32 evaluates \
                 `a.mul_add(x, b.mul_add(y, c))` with the affine constant c = +0.0, and a +0 addend \
                 absorbs the -0 (wukong_runtime/src/velem.rs `elem1` and `velem_avx2`)",
    },
    KnownDivergence {
        kernel: "relu6",
        matches: relu6_nan_clamps_to_zero,
        defect: "`out[i] = if x[i] < 6.0 { if x[i] > 0.0 { x[i] } else { 0.0 } } else { 6.0 }` with \
                 x = NaN prints 6 as written (the `< 6` test is false) but 0 when recognized: the \
                 recognizer maps this min-outer spelling to wukong_velem_f32's VE_RELU6, which clamps \
                 max-first (`v > 0 ? v : 0`, then `< 6`), and the two orders differ only on NaN",
    },
];

/// The known divergence for `kernel`, if this lane is one.
fn known_divergence(kernel: &str, x: f32, on: f32, off: f32) -> Option<usize> {
    KNOWN_OFF_DIVERGENCES
        .iter()
        .position(|d| d.kernel == kernel && (d.matches)(x, on, off))
}

/// The recognition-off bound for each fuzzer kernel (see [`OffBound`]). Anything not listed is held
/// bit-exact.
fn off_bound(name: &str) -> OffBound {
    match name {
        "dot" | "ssd" | "matmul" | "linear" | "tensor_matmul" => OffBound::Fold,
        "softmax" | "softmax_batched" => OffBound::Abs(2e-5),
        "layernorm" | "rmsnorm" | "rmsnorm_batched" | "layernorm_batched" => OffBound::Abs(2e-3),
        "rmsnorm_affine" | "layernorm_affine" | "rmsnorm_affine_batched" => OffBound::Abs(4e-3),
        _ => OffBound::Exact,
    }
}

/// One lane of a tolerance-class kernel ([`OffBound::Fold`] / [`OffBound::Abs`]): `Ok(|on - off|)`
/// when the recognized value `on` stands within the absolute bound `tol` of the recognition-off
/// value `off`, or `Err(why)`.
///
/// A tolerance licenses a difference in MAGNITUDE and nothing else, so two kinds of lane are held to
/// [`same`] exactly whatever `tol` is. A non-finite lane carries no magnitude to bound. And a zero
/// lane: `|(+0) - (-0)|` is `0`, inside every bound, so a bare `err <= tol` would accept exactly
/// the sign-of-zero divergence `same` was tightened to refuse (an affine norm writeback that adds
/// `+0.0` for an absent beta, a fold that drops its `+0.0` seed). Reassociation is not a source of
/// one: a sum seeded with `+0.0` is `+0` whenever it is exactly zero, under any association (an
/// exact cancellation of nonzero terms rounds to `+0` and `+0 + -0 = +0`, so no subtree holding the
/// seed can be `-0`). The one IEEE route is a contracted product that underflows — `fma(a, b, +0)`
/// rounds a negative `a*b` below `2^-150` to `-0` where `a*b + 0` gives `+0` — which needs a product
/// of two of this fuzzer's inputs to vanish below the smallest subnormal; the seeded corpus does not
/// generate one, and if a change ever made it, the lane belongs in [`KNOWN_OFF_DIVERGENCES`] with
/// its reproducer like any other, not under a magnitude bound. The norm bounds (`Abs`) are stated
/// for magnitudes.
fn off_tol_lane(on: f32, off: f32, tol: f64) -> Result<f64, String> {
    if same(on, off) {
        return Ok(0.0);
    }
    if !(on.is_finite() && off.is_finite()) {
        return Err("a non-finite lane must match exactly".to_string());
    }
    if on == 0.0 && off == 0.0 {
        // Both zero and `same` failed: the bits differ, so the signs do.
        return Err("the sign of zero differs, and no magnitude bound covers it".to_string());
    }
    let err = (on as f64 - off as f64).abs();
    if err > tol {
        return Err(format!("|diff| {err:e} exceeds {tol:e}"));
    }
    Ok(err)
}

/// Per output lane of a [`OffBound::Fold`] kernel: `(terms in the fold, sum of their magnitudes)`,
/// in f64, from the f32 inputs. A lane no fold writes gets `(0, 0.0)`, so its bound is `0` and it is
/// held exact: any magnitude difference exceeds `0`, and a zero lane is held to `same` (sign
/// included) by [`off_tol_lane`] whatever the bound.
fn fold_terms(name: &str, n: usize, x: &[f32], y: &[f32], len: usize) -> Vec<(usize, f64)> {
    let mut out = vec![(0usize, 0.0f64); len];
    let (xd, yd) = (|i: usize| x[i] as f64, |i: usize| y[i] as f64);
    match name {
        "dot" => out[0] = (n, (0..n).map(|i| (xd(i) * yd(i)).abs()).sum()),
        "ssd" => {
            out[0] = (n, (0..n).map(|i| (xd(i) - yd(i)) * (xd(i) - yd(i))).sum());
        }
        // out[i, j] = sum_k x[i, k] * y[k, j]   (the ikj matmul and the Tensor[f32, n, n] matmul)
        "matmul" | "tensor_matmul" => {
            for i in 0..n {
                for j in 0..n {
                    let s = (0..n).map(|k| (xd(i * n + k) * yd(k * n + j)).abs()).sum();
                    out[i * n + j] = (n, s);
                }
            }
        }
        // out[i, j] = sum_k x[i, k] * y[j, k]   (nn.Linear, C = A * B^T)
        "linear" => {
            for i in 0..n {
                for j in 0..n {
                    let s = (0..n).map(|k| (xd(i * n + k) * yd(j * n + k)).abs()).sum();
                    out[i * n + j] = (n, s);
                }
            }
        }
        _ => unreachable!("{name} is not a Fold kernel"),
    }
    out
}

/// The tolerance classes license a difference in MAGNITUDE, and nothing else: a lane outside its
/// bound, a non-finite lane that differs, and a zero whose sign differs are all refused. The last is
/// the case a bare `|on - off| <= tol` cannot see -- `|(+0) - (-0)|` is `0`, inside every bound --
/// and it is exactly the sign-of-zero divergence [`same`] exists to refuse.
#[test]
fn off_tolerance_lanes_bound_magnitude_and_hold_the_sign_of_zero() {
    // Equal values (and NaN against NaN, whose payloads no program can read) need no bound.
    assert_eq!(off_tol_lane(1.5, 1.5, 0.0), Ok(0.0));
    assert_eq!(off_tol_lane(f32::NAN, -f32::NAN, 0.0), Ok(0.0));
    assert_eq!(off_tol_lane(0.0, 0.0, 0.0), Ok(0.0));
    assert_eq!(off_tol_lane(-0.0, -0.0, 0.0), Ok(0.0));
    // A magnitude difference is judged by the bound, on either side of it.
    let one_ulp = f32::from_bits(1.0f32.to_bits() + 1);
    assert!(off_tol_lane(1.0, one_ulp, 1e-6).is_ok());
    assert!(off_tol_lane(1.0, 1.001, 1e-6).is_err());
    // A zero against a tiny nonzero is still a magnitude question.
    assert!(off_tol_lane(0.0, -1e-30, 1e-20).is_ok());
    // A non-finite lane must match exactly, whatever the bound.
    assert!(off_tol_lane(f32::INFINITY, f32::MAX, f64::INFINITY).is_err());
    assert!(off_tol_lane(f32::NAN, 0.0, f64::INFINITY).is_err());
    // The sign of zero is observable (`print(-0.0)` is `-0`, `1/-0` is `-inf`) and no bound covers
    // it: `+0` against `-0` is refused at a zero bound, at a generous one, and in both orders.
    assert!(off_tol_lane(0.0, -0.0, 0.0).is_err());
    assert!(off_tol_lane(-0.0, 0.0, 0.0).is_err());
    assert!(off_tol_lane(0.0, -0.0, 4e-3).is_err());
    assert!(off_tol_lane(-0.0, 0.0, f64::INFINITY).is_err());
}

#[test]
fn fuzz_full_buffer_interp_vs_native() {
    // Sizes chosen to straddle the vectorizer's 4×128-bit unroll group and single-vector/scalar
    // remainder boundaries; matmul sizes straddle the 6×16 microkernel tile remainders.
    const EW_SIZES: &[usize] = &[1, 2, 3, 7, 8, 9, 15, 16, 17, 31, 33, 64, 100];
    const MM_SIZES: &[usize] = &[1, 2, 5, 6, 7, 8, 9, 16, 17];
    let mut rng = Rng(0xC0FFEE123);
    let mut runs = 0u64;

    // What this fuzzer actually proves depends on whether a kernel is recognized. For a kernel whose
    // lowered MIR calls a `wukong_*` symbol, interp and native marshal into the SAME runtime function,
    // so the bit-exact comparison is a check of the recognizer's base pointers and extents at every
    // remainder boundary (the class that produced the slice-param base-pointer defect) — valuable,
    // but NOT a check that the kernel computes the right value. For a kernel that does NOT dispatch,
    // the two sides are independent codegen and the comparison is a genuine kernel-correctness check.
    // The `vmath_kernels_match_f64_reference` test covers the transcendental dispatching kernels
    // against an independent oracle; this split records which class each kernel fell in and refuses
    // to let the non-dispatching class go empty — the exact way a refactor that made every kernel
    // recognized would turn this whole test vacuous while leaving it green.
    let mut all_names: std::collections::BTreeSet<&'static str> = std::collections::BTreeSet::new();
    let mut dispatching_names: std::collections::BTreeSet<&'static str> =
        std::collections::BTreeSet::new();

    // P15.11, the recognition-OFF arm. The interp-vs-native comparison above is blind to a wrong
    // kernel: for a dispatching kernel both sides call the SAME runtime function. So every kernel is
    // also lowered with `LowerOptions { recognize: false, .. }` — the loop exactly as written, no
    // `wukong_*` substitution — and run on the interpreter, and the recognized NATIVE output is held
    // lane by lane to that second opinion under the kernel's `OffBound`. This is the fuzzer-scale
    // twin of recognize.rs's corpus-wide on/off differential.
    let mut off_exact: std::collections::BTreeSet<&'static str> = std::collections::BTreeSet::new();
    // Per tolerance-class kernel: the worst observed |on - off| / declared bound, and the lanes
    // compared.
    let mut off_tol: std::collections::BTreeMap<&'static str, (f64, u64)> =
        std::collections::BTreeMap::new();
    let mut off_skipped = 0u64;
    let mut off_failures: Vec<String> = Vec::new();
    let mut known_hits = vec![0u64; KNOWN_OFF_DIVERGENCES.len()];

    for k in kernels() {
        all_names.insert(k.name);
        let sizes = if matches!(k.name, "matmul" | "linear") || k.name.starts_with("tensor") {
            MM_SIZES
        } else {
            EW_SIZES
        };
        for &n in sizes {
            let len = (k.len)(n);
            let src = (k.src)(n);
            let mut interner = Interner::new();
            let (module, pd) = wukong_parser::parse_module(&src, SourceId(0), &mut interner);
            assert!(pd.iter().all(|d| !d.is_error()), "{}: parse {pd:?}", k.name);
            let (sema, sd) = wukong_sema::check(&module, &interner);
            assert!(sd.iter().all(|d| !d.is_error()), "{}: sema {sd:?}", k.name);
            let kbench = interner.intern("kbench");

            for &opt in &[0u8, 3u8] {
                let (mut program, ld) =
                    wukong_mir_build::lower_program(&module, &sema, &mut interner);
                assert!(ld.iter().all(|d| !d.is_error()), "{}: lower {ld:?}", k.name);
                wukong_opt::optimize(&mut program, opt);
                // Classify once per kernel, from the un-reordered opt=0 lowering: a recognizer fires
                // in mir_build (before the optimizer), so its call is already present here.
                if opt == 0 {
                    let mir = wukong_mir::print::print_program(&program, &interner);
                    if mir.contains("call wukong_") {
                        dispatching_names.insert(k.name);
                    }
                }
                let handle = crate::jit_module(&program, &interner).expect("jit_module");
                let ptr = handle.func_ptr(kbench).expect("func_ptr kbench");
                let native: K3 = unsafe { std::mem::transmute(ptr) };

                // The same source, recognition OFF, at the same opt level.
                let (mut generic, gd) = wukong_mir_build::lower_program_with(
                    &module,
                    &sema,
                    &mut interner,
                    wukong_mir_build::LowerOptions {
                        recognize: false,
                        ..Default::default()
                    },
                );
                assert!(
                    gd.iter().all(|d| !d.is_error()),
                    "{}: lower (recognition off) {gd:?}",
                    k.name
                );
                wukong_opt::optimize(&mut generic, opt);
                let generic_mir = wukong_mir::print::print_program(&generic, &interner);
                assert!(
                    !generic_mir
                        .match_indices("call wukong_")
                        .any(|(i, _)| !generic_mir[i..].starts_with("call wukong_rt_")),
                    "{} n={n} opt={opt}: recognition OFF still dispatched a runtime kernel, so the \
                     off arm would compare the kernel with itself",
                    k.name
                );
                let bound = off_bound(k.name);

                for &regime in k.regimes {
                    for _ in 0..4 {
                        let x: Vec<f32> = (0..len).map(|_| gen(&mut rng, regime)).collect();
                        let y: Vec<f32> = (0..len).map(|_| gen(&mut rng, regime)).collect();

                        // Interpreter over its own buffer copies.
                        let (mut xi, mut yi, mut oi) = (x.clone(), y.clone(), vec![0f32; len]);
                        wukong_interp::run_kernel_f32(
                            &program,
                            kbench,
                            &mut [&mut xi, &mut yi, &mut oi],
                            &interner,
                        )
                        .expect("interp kernel");

                        // Native over its own buffer copies.
                        let (xn, yn, mut on) = (x.clone(), y.clone(), vec![0f32; len]);
                        unsafe { native(xn.as_ptr(), yn.as_ptr(), on.as_mut_ptr()) };

                        for idx in 0..len {
                            assert!(
                                same(oi[idx], on[idx]),
                                "{} n={n} opt={opt} out[{idx}]: interp={} (0x{:08x}) native={} (0x{:08x})",
                                k.name,
                                oi[idx],
                                oi[idx].to_bits(),
                                on[idx],
                                on[idx].to_bits(),
                            );
                        }

                        // Recognition OFF on the interpreter, over its own copies of the inputs.
                        let (mut xg, mut yg, mut og) = (x.clone(), y.clone(), vec![0f32; len]);
                        wukong_interp::run_kernel_f32(
                            &generic,
                            kbench,
                            &mut [&mut xg, &mut yg, &mut og],
                            &interner,
                        )
                        .expect("interp kernel (recognition off)");
                        let off_msg = |idx: usize, why: &str| {
                            format!(
                                "{} n={n} opt={opt} out[{idx}]: recognized native={} (0x{:08x}) vs \
                                 recognition-off interp={} (0x{:08x}) — {why}",
                                k.name,
                                on[idx],
                                on[idx].to_bits(),
                                og[idx],
                                og[idx].to_bits(),
                            )
                        };
                        match bound {
                            OffBound::Exact => {
                                for idx in 0..len {
                                    if same(on[idx], og[idx]) {
                                        continue;
                                    }
                                    if let Some(d) =
                                        known_divergence(k.name, x[idx], on[idx], og[idx])
                                    {
                                        known_hits[d] += 1;
                                    } else {
                                        off_failures.push(off_msg(
                                            idx,
                                            "documented bit-exact with its source loop",
                                        ));
                                    }
                                }
                                off_exact.insert(k.name);
                            }
                            OffBound::Fold => {
                                let terms = fold_terms(k.name, n, &x, &y, len);
                                let entry = off_tol.entry(k.name).or_insert((0.0, 0));
                                for idx in 0..len {
                                    let (a, b) = (on[idx], og[idx]);
                                    entry.1 += 1;
                                    let (cnt, mag) = terms[idx];
                                    let tol = 2.0 * (cnt as f64 + 1.0) * f64::powi(2.0, -24) * mag;
                                    match off_tol_lane(a, b, tol) {
                                        Ok(err) if tol > 0.0 => entry.0 = entry.0.max(err / tol),
                                        Ok(_) => {}
                                        Err(why) => {
                                            if let Some(d) = known_divergence(k.name, x[idx], a, b)
                                            {
                                                known_hits[d] += 1;
                                            } else {
                                                off_failures.push(off_msg(
                                                    idx,
                                                    &format!("{why} (bound 2*gamma_n*S)"),
                                                ));
                                            }
                                        }
                                    }
                                }
                            }
                            OffBound::Abs(tol) => {
                                let entry = off_tol.entry(k.name).or_insert((0.0, 0));
                                if !matches!(regime, Regime::Normal) {
                                    off_skipped += len as u64;
                                } else {
                                    for idx in 0..len {
                                        let (a, b) = (on[idx], og[idx]);
                                        entry.1 += 1;
                                        match off_tol_lane(a, b, tol) {
                                            Ok(err) => entry.0 = entry.0.max(err / tol),
                                            Err(why) => {
                                                if let Some(d) =
                                                    known_divergence(k.name, x[idx], a, b)
                                                {
                                                    known_hits[d] += 1;
                                                } else {
                                                    off_failures.push(off_msg(
                                                        idx,
                                                        &format!("{why} (declared absolute bound)"),
                                                    ));
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        runs += 1;
                    }
                }
                drop(handle);
            }
        }
    }
    // A floor so an accidental empty corpus (e.g. a refactor that drops every kernel) fails loudly.
    assert!(runs > 1000, "fuzzer ran too few cases: {runs}");

    let dispatching = dispatching_names.len();
    let non_dispatching = all_names.len() - dispatching;
    eprintln!(
        "fuzzer classification: {} kernels total — {dispatching} dispatch a wukong_* symbol \
         (interp-vs-native checks the recognizer's base pointers/extents), {non_dispatching} do not \
         (interp-vs-native checks kernel values). Dispatching: {dispatching_names:?}",
        all_names.len()
    );
    // The interp-vs-native arm only PROVES a kernel's value where neither side dispatches (otherwise
    // both marshal into one runtime symbol). If that class ever empties, this test still runs >1000
    // cases and stays green while proving nothing about any kernel — the precise vacuity this gate
    // exists to refuse. (The transcendental dispatching kernels keep their own oracle in
    // `vmath_kernels_match_f64_reference`.)
    assert!(
        non_dispatching > 0,
        "every fuzzer kernel now lowers to a recognized wukong_* call, so the interp-vs-native \
         bit-exact comparison is two callers of one runtime function for every case and proves \
         nothing about kernel values — add or restore a non-recognized kernel to the corpus"
    );
    assert!(
        dispatching > 0,
        "no fuzzer kernel dispatches a recognized wukong_* call — the recognizer base-pointer/extent \
         coverage this fuzzer is relied on for is gone"
    );

    // The recognition-off split. Only a DISPATCHING kernel is a real second opinion (a kernel that
    // never dispatches lowers to the same loop either way), so report the dispatching ones by class.
    let exact_dispatching: Vec<&str> = off_exact
        .iter()
        .filter(|n| dispatching_names.contains(*n))
        .copied()
        .collect();
    let tol_report: Vec<String> = off_tol
        .iter()
        .map(|(name, (worst, lanes))| {
            format!(
                "{name} ({:?}: {lanes} lanes, worst {:.3}% of the bound)",
                off_bound(name),
                worst * 100.0
            )
        })
        .collect();
    eprintln!(
        "recognition-off arm: {} dispatching kernel(s) bit-exact against their source loop {:?}; {} \
         within a declared reassociation bound: {}; {off_skipped} norm lane(s) outside the Normal \
         regime not compared",
        exact_dispatching.len(),
        exact_dispatching,
        off_tol.len(),
        tol_report.join(", ")
    );
    for (d, hits) in KNOWN_OFF_DIVERGENCES.iter().zip(&known_hits) {
        eprintln!(
            "  KNOWN recognizer != source divergence, {hits} lane(s) in {}: {}",
            d.kernel, d.defect
        );
    }
    let stale: Vec<&str> = KNOWN_OFF_DIVERGENCES
        .iter()
        .zip(&known_hits)
        .filter(|(_, hits)| **hits == 0)
        .map(|(d, _)| d.kernel)
        .collect();
    assert!(
        stale.is_empty(),
        "KNOWN_OFF_DIVERGENCES entries for {stale:?} no longer reproduce — the defect looks fixed: \
         delete the entry so those lanes are held bit-exact again"
    );
    assert!(
        off_failures.is_empty(),
        "recognition changed {} lane(s) beyond their declared bound; first 40:
{}",
        off_failures.len(),
        off_failures
            .iter()
            .take(40)
            .cloned()
            .collect::<Vec<_>>()
            .join(
                "
"
            )
    );
    assert!(
        !exact_dispatching.is_empty() && !off_tol.is_empty(),
        "the recognition-off arm lost one of its two classes (bit-exact dispatching kernels: {}, \
         tolerance kernels: {}) — it is no longer checking what it claims to",
        exact_dispatching.len(),
        off_tol.len()
    );
}

type K3i8 = unsafe extern "C" fn(*const u8, *const i8, *mut i32);

/// Full-buffer int8 differential fuzzer: the `u8×i8→i32` quantized `nn.Linear` (`C = A·Bᵀ`) kernel
/// run through both backends over identical random buffers, with the **entire** `C` buffer compared
/// bit-for-bit. The square sizes straddle the int8 microkernel's K-chunking — the AVX2 widen+`vpmaddwd`
/// 32-wide (two chains) and 16-wide steps plus the scalar tail — and odd tails (17/33/49). Inputs span
/// the full `u8`/`i8` ranges. Integer arithmetic is exact and order-independent (i32 add is associative
/// mod 2³²), so bit-exact equality is the correct bar with no tolerance and no reassociation caveat.
#[test]
fn fuzz_full_buffer_i8_interp_vs_native() {
    // M = N = K = n; n hits the 16/32/48 K-chunk boundaries and their odd neighbours.
    const SIZES: &[usize] = &[1, 2, 3, 7, 8, 9, 15, 16, 17, 31, 32, 33, 48, 49, 64];
    let mut rng = Rng(0x_1248_ADDE);
    let mut runs = 0u64;

    for &n in SIZES {
        let len = n * n;
        let src = format!(
            "module f\nfn lin(a:[u8;{len}], b:[i8;{len}], mut c:[i32;{len}]) {{ \
             for i in 0..{n} {{ for j in 0..{n} {{ let mut s: i32 = 0; \
             for k in 0..{n} {{ s = s + (a[i*{n}+k] as i32) * (b[j*{n}+k] as i32); }} \
             c[i*{n}+j] = s; }} }} }}\n"
        );
        let mut interner = Interner::new();
        let (module, pd) = wukong_parser::parse_module(&src, SourceId(0), &mut interner);
        assert!(pd.iter().all(|d| !d.is_error()), "i8 n={n}: parse {pd:?}");
        let (sema, sd) = wukong_sema::check(&module, &interner);
        assert!(sd.iter().all(|d| !d.is_error()), "i8 n={n}: sema {sd:?}");
        let lin = interner.intern("lin");

        for &opt in &[0u8, 3u8] {
            let (mut program, ld) = wukong_mir_build::lower_program(&module, &sema, &mut interner);
            assert!(ld.iter().all(|d| !d.is_error()), "i8 n={n}: lower {ld:?}");
            wukong_opt::optimize(&mut program, opt);
            let handle = crate::jit_module(&program, &interner).expect("jit_module");
            let ptr = handle.func_ptr(lin).expect("func_ptr lin");
            let native: K3i8 = unsafe { std::mem::transmute(ptr) };

            for _ in 0..6 {
                // Full u8 / i8 ranges (b via a wrapping cast of the low byte).
                let a: Vec<u8> = (0..len).map(|_| (rng.next_u64() & 0xFF) as u8).collect();
                let b: Vec<i8> = (0..len)
                    .map(|_| (rng.next_u64() & 0xFF) as u8 as i8)
                    .collect();

                let mut ci = vec![0i32; len];
                wukong_interp::run_kernel_i8(&program, lin, &a, &b, &mut ci, &interner)
                    .expect("interp i8 kernel");

                let mut cn = vec![0i32; len];
                unsafe { native(a.as_ptr(), b.as_ptr(), cn.as_mut_ptr()) };

                for idx in 0..len {
                    assert_eq!(
                        ci[idx], cn[idx],
                        "i8 n={n} opt={opt} c[{idx}]: interp={} native={}",
                        ci[idx], cn[idx]
                    );
                }
                runs += 1;
            }
            drop(handle);
        }
    }
    assert!(runs > 100, "i8 fuzzer ran too few cases: {runs}");
}

/// ReLU has an exact independent reference: retain positive x, otherwise return positive zero.
/// This also specifies the bounded activation's intended NaN behavior; it is not IEEE maximum.
/// Keep this corpus separate from the f64 table so adding coverage cannot shift its RNG stream.
#[test]
fn vmath_relu_matches_exact_reference() {
    // Explicit encodings retain both zero signs and quiet/signaling NaN payloads. Neither the
    // reference nor the expected bits calls a runtime helper or floating-point max intrinsic.
    const INPUT_BITS: &[u32] = &[
        0x0000_0000,
        0x8000_0000, // +/- zero
        0x3f80_0000,
        0xbf80_0000, // +/- one
        0x3e00_0000,
        0xbe00_0000, // +/- 0.125
        0x0000_0001,
        0x8000_0001, // smallest +/- subnormal
        0x007f_ffff,
        0x807f_ffff, // largest +/- subnormal
        0x0080_0000,
        0x8080_0000, // smallest +/- normal
        0x7f7f_ffff,
        0xff7f_ffff, // largest finite magnitudes
        0x7f80_0000,
        0xff80_0000, // +/- infinity
        0x7fc0_1234,
        0xffc0_5678, // signed quiet NaNs
        0x7f80_0001,
        0xff80_0123, // signed signaling NaNs
    ];
    // Short lengths exercise scalar-only and vector/tail boundaries. 16,385 exceeds the current
    // parallel threshold (16,384) and spans two chunks when the runtime pool has multiple workers.
    // The test still checks both entry points on a width-one pool or a non-AVX host; it never skips.
    const LENGTHS: &[usize] = &[0, 1, 7, 8, 9, 15, 16, 17, 31, 32, 33, 48, 49, 16_385];
    type Kernel = unsafe extern "C" fn(*const f32, *mut f32, i64, i64);
    let kernels: [(&str, Kernel); 2] = [
        ("serial", wukong_runtime::wukong_vmath_f32),
        ("parallel", wukong_runtime::wukong_vmath_f32_parallel),
    ];
    const SENTINEL: u32 = 0xc2f6_0000; // -123, distinct from every valid ReLU output
    for &n in LENGTHS {
        // Rotate every input class through every lane and scalar-tail position, even at n=1.
        for offset in 0..INPUT_BITS.len() {
            let input: Vec<f32> = (0..n)
                .map(|i| f32::from_bits(INPUT_BITS[(i + offset) % INPUT_BITS.len()]))
                .collect();
            let expected: Vec<u32> = input
                .iter()
                .map(|&x| {
                    let x = f64::from(x);
                    (if x > 0.0 { x } else { 0.0 } as f32).to_bits()
                })
                .collect();
            for &(name, kernel) in &kernels {
                let mut output = vec![f32::from_bits(SENTINEL); n + 2];
                // SAFETY: input has n elements; the output interior has n writable elements and
                // is disjoint. Both guards remain outside the runtime's declared output range.
                unsafe {
                    kernel(
                        input.as_ptr(),
                        output.as_mut_ptr().add(1),
                        n as i64,
                        wukong_runtime::VM_RELU,
                    );
                }
                assert_eq!(output[0].to_bits(), SENTINEL, "{name} n={n} prefix");
                assert_eq!(output[n + 1].to_bits(), SENTINEL, "{name} n={n} suffix");
                for i in 0..n {
                    assert_eq!(
                        output[i + 1].to_bits(),
                        expected[i],
                        "{name} n={n} offset={offset} lane={i} input={:#010x}",
                        input[i].to_bits()
                    );
                }
            }
        }
    }
}

/// Verify the *value* of the shared transcendental kernels against an independent `f64` reference
/// within a tight relative tolerance — this catches the case both backends are wrong the *same*
/// way (a bug in the shared `wukong_vmath_f32` kernel that interp-vs-native equality cannot see).
///
/// The tolerance is per-function and honest about the algorithm, measured as **relative error**
/// (machine-independent, and the right metric near saturation where one ULP is a vanishing absolute
/// error): `exp`/`log` are *direct* minimax polynomials and `sqrt` is the hardware instruction, so
/// they hold ≈1e-6; `tanh`/`sigmoid`/`silu` are *composed* from `exp` (e.g. `tanh(x) = 1 −
/// 2/(exp(2x)+1)`), so the division amplifies exp's error to ≈1e-4 — still far tighter than the
/// bf16/f16 these activations usually feed. The bound documents that reality and catches gross errors.
#[test]
fn vmath_kernels_match_f64_reference() {
    // Coverage: 32 distinct VM_* operations plus hardware sqrt (not a VM_* operation).
    // ReLU has a separate exact reference above; elu, tanhshrink and tan remain omitted here.
    // (name, true-function f64 reference, input regime, max relative error).
    #[allow(clippy::type_complexity)] // the row IS that 4-tuple; naming it would obscure the table
    let cases: &[(&str, fn(f64) -> f64, Regime, f64)] = &[
        ("sqrt", |v| v.sqrt(), Regime::Positive, 1e-6), // hardware fsqrt
        ("exp", |v| v.exp(), Regime::Normal, 1e-6),     // direct minimax poly
        ("log", |v| v.ln(), Regime::Positive, 1e-5), // direct minimax poly (log near 1 is touchy)
        ("tanh", |v| v.tanh(), Regime::Normal, 1e-4), // composed from exp
        (
            "sigmoid",
            |v| 1.0 / (1.0 + (-v).exp()),
            Regime::Normal,
            1e-4,
        ),
        ("silu", |v| v / (1.0 + (-v).exp()), Regime::Normal, 1e-4),
        // leaky_relu is exact (a branch + one multiply), so it holds to f32 rounding. `elu` is omitted
        // here: its only nonlinearity is the same `exp8` the "exp" case already value-checks, and
        // `exp(x)−1` cancels catastrophically near 0⁻ (a *relative*-error artifact of ELU's definition,
        // not a kernel bug) — the full-buffer fuzzer still gates its interp-vs-native equality.
        (
            "leaky_relu",
            |v| if v > 0.0 { v } else { 0.01 * v },
            Regime::Normal,
            1e-6,
        ),
        // --- Activations composed from exp/tanh (inherit exp's error, ≈1e-4) -----------------
        // gelu (tanh approximation): 0.5x(1 + tanh(√(2/π)(x + 0.044715x³))).
        (
            "gelu",
            |v| {
                let x3 = v * v * v;
                0.5 * v * (1.0 + (0.797_884_560_802_865_4 * (0.044_715 * x3 + v)).tanh())
            },
            Regime::Normal,
            1e-4,
        ),
        // softplus = max(x,0) + ln(1 + e^−|x|) (the stable form the kernel uses).
        (
            "softplus",
            |v| v.max(0.0) + (1.0 + (-v.abs()).exp()).ln(),
            Regime::Normal,
            1e-4,
        ),
        (
            "mish",
            |v| {
                let sp = v.max(0.0) + (1.0 + (-v.abs()).exp()).ln();
                v * sp.tanh()
            },
            Regime::Normal,
            1e-4,
        ),
        (
            "selu",
            |v| {
                let (lam, alpha) = (1.050_700_987_355_480_5_f64, 1.673_263_242_354_377_3_f64);
                lam * if v > 0.0 { v } else { alpha * (v.exp() - 1.0) }
            },
            Regime::Normal,
            1e-4,
        ),
        ("softsign", |v| v / (1.0 + v.abs()), Regime::Normal, 1e-4),
        // logsigmoid(x) = ln σ(x) = −softplus(−x) = −(max(−x,0) + ln(1 + e^−|x|)).
        (
            "logsigmoid",
            |v| -((-v).max(0.0) + (1.0 + (-v.abs()).exp()).ln()),
            Regime::Normal,
            1e-4,
        ),
        // hardsigmoid / hardswish are piecewise-linear clamps — exact bar f32 rounding.
        (
            "hardsigmoid",
            |v| (v + 3.0).clamp(0.0, 6.0) / 6.0,
            Regime::Normal,
            1e-4,
        ),
        (
            "hardswish",
            |v| v * ((v + 3.0).clamp(0.0, 6.0) / 6.0),
            Regime::Normal,
            1e-4,
        ),
        // --- Trig / inverse-trig (Cephes ≈1 ULP; |x|>1 → NaN want is skipped by the loop) -----
        ("sin", |v| v.sin(), Regime::Normal, 1e-4),
        ("cos", |v| v.cos(), Regime::Normal, 1e-4),
        ("atan", |v| v.atan(), Regime::Normal, 1e-4),
        ("asin", |v| v.asin(), Regime::Normal, 1e-2), // derivative → ∞ at ±1
        ("acos", |v| v.acos(), Regime::Normal, 1e-2),
        // --- Hyperbolic + inverse (composed from exp; acosh needs x≥1 → Positive regime) -------
        ("sinh", |v| v.sinh(), Regime::Normal, 1e-4),
        ("cosh", |v| v.cosh(), Regime::Normal, 1e-4),
        ("asinh", |v| v.asinh(), Regime::Normal, 1e-4),
        ("acosh", |v| v.acosh(), Regime::Positive, 1e-2),
        ("atanh", |v| v.atanh(), Regime::Normal, 1e-2), // blows up near ±1
        // --- Base-2 / base-10 exp & log (exp/log compositions, ≈1e-4) -------------------------
        ("exp2", |v| v.exp2(), Regime::Normal, 1e-4),
        ("log2", |v| v.log2(), Regime::Positive, 1e-4),
        ("exp10", |v| 10f64.powf(v), Regime::Normal, 1e-4),
        ("log10", |v| v.log10(), Regime::Positive, 1e-4),
        // --- Stable forms (log1p needs x>−1; near −1 it is ill-conditioned) -------------------
        ("expm1", |v| v.exp_m1(), Regime::Normal, 1e-4),
        ("log1p", |v| v.ln_1p(), Regime::Normal, 1e-3),
        // erf: Abramowitz–Stegun 7.1.26 high-precision f64 reference (std f64 has no erf). Its own
        // ≈1.5e-7 absolute error dominates near the x=0 zero, so the bound is 1e-3 (still catches
        // any gross kernel error; the full-buffer fuzzer gates interp==native exactly).
        (
            "erf",
            |v| {
                let (a1, a2, a3, a4, a5, p) = (
                    0.254_829_592_f64,
                    -0.284_496_736_f64,
                    1.421_413_741_f64,
                    -1.453_152_027_f64,
                    1.061_405_429_f64,
                    0.327_591_1_f64,
                );
                let sign = if v >= 0.0 { 1.0 } else { -1.0 };
                let x = v.abs();
                let t = 1.0 / (1.0 + p * x);
                let y = 1.0 - (((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t) * (-x * x).exp();
                sign * y
            },
            Regime::Normal,
            1e-3,
        ),
        ("cbrt", |v| v.cbrt(), Regime::Normal, 1e-4),
        // tan (pole at π/2 ∈ [−2,2)), tanhshrink and elu (catastrophic cancellation near 0) are
        // intentionally omitted — their relative error is unbounded by construction, not by a kernel
        // bug; the full-buffer fuzzer still gates their interp-vs-native equality.
    ];
    let n = 256usize;
    let mut rng = Rng(0x_5EED_2024);
    for &(name, fref, regime, max_allowed) in cases {
        let src = ew(n, &format!("out[i] = {name}(x[i]);"));
        let mut interner = Interner::new();
        let (module, _) = wukong_parser::parse_module(&src, SourceId(0), &mut interner);
        let (sema, _) = wukong_sema::check(&module, &interner);
        let (mut program, _) = wukong_mir_build::lower_program(&module, &sema, &mut interner);
        wukong_opt::optimize(&mut program, 3);
        let kbench: Symbol = interner.intern("kbench");

        let x: Vec<f32> = (0..n).map(|_| gen(&mut rng, regime)).collect();
        let (mut xi, mut yi, mut oi) = (x.clone(), vec![0f32; n], vec![0f32; n]);
        wukong_interp::run_kernel_f32(
            &program,
            kbench,
            &mut [&mut xi, &mut yi, &mut oi],
            &interner,
        )
        .unwrap();

        let mut max_rel = 0f64;
        for i in 0..n {
            let want = fref(x[i] as f64);
            let got = oi[i] as f64;
            if !want.is_finite() {
                continue;
            }
            // A non-finite result where the f64 reference is finite must fail loudly. `max_rel.max(rel)`
            // would otherwise *drop* it: `rel` is NaN, `f64::max` returns the other operand, and a
            // kernel returning NaN on a whole regime (say `log` on (0,1), ~1/8 of these samples) leaves
            // `max_rel` at whatever the finite samples produced and the test reports PASS. The
            // full-buffer fuzzer cannot catch that either — the interpreter marshals the same kernel.
            assert!(
                got.is_finite(),
                "{name}: kernel returned {got} where the f64 reference is finite ({want}) at x={}",
                x[i]
            );
            // Relative error with a small absolute floor so values near zero don't blow up the ratio.
            let rel = (got - want).abs() / want.abs().max(1e-6);
            max_rel = max_rel.max(rel);
        }
        assert!(
            max_rel <= max_allowed,
            "{name}: max relative error {max_rel:.2e} from f64 reference (bound {max_allowed:.0e})"
        );
    }
}
