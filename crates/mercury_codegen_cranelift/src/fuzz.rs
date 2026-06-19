//! Randomized **full-buffer** differential fuzzer: interpreter vs native (Cranelift).
//!
//! The older differential tests reduced a kernel's output to a single scalar (`(s*1000.0) as i32`)
//! and compared exit codes, and the cross-language harness cross-checked only a 3-point checksum —
//! so a kernel wrong at every index *except* the few sampled ones slipped through. This fuzzer
//! closes that hole: for a battery of kernels it generates random input buffers across many sizes
//! (deliberately hitting the SIMD remainder boundaries) and adversarial value regimes (NaN, ±Inf,
//! denormals, ±0, huge magnitudes), runs the *same* kernel through both backends over identical
//! buffers via the typed kernel-entry ABI (`mercury_interp::run_kernel_f32`/`run_kernel_i8` /
//! `func_ptr`), and asserts the **entire** output buffer is bit-for-bit identical. The int8
//! `u8×i8→i32` GEMM gets its own pass (`fuzz_full_buffer_i8_interp_vs_native`) over the kernel's
//! K-chunk boundaries.
//!
//! Both backends execute the same lowered MIR and marshal through the same runtime microkernels, so
//! bit-exactness is the correct (strongest) bar — not a tolerance. A divergence is a real miscompile.

use mercury_span::{Interner, SourceId, Symbol};

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

/// Two f32 results are "the same" if both are NaN (payloads may differ across an `f64`-rounding
/// interpreter and a real-`f32` JIT), both are zero (±0 compare equal numerically), or their bits
/// match exactly. Anything else is a divergence.
fn same(a: f32, b: f32) -> bool {
    (a.is_nan() && b.is_nan()) || (a == 0.0 && b == 0.0) || a.to_bits() == b.to_bits()
}

/// A kernel under test: a source template (parameterized by element count `n`), its element count,
/// and the regimes its inputs may take.
struct Kernel {
    name: &'static str,
    /// `n` -> Mercury source defining `fn kbench(x:[f32;LEN], y:[f32;LEN], out:[f32;LEN])`.
    src: fn(usize) -> String,
    /// Buffer length given the size parameter `n` (== `n` for elementwise, `n*n` for matmul).
    len: fn(usize) -> usize,
    regimes: &'static [Regime],
}

/// `fn kbench(x:[f32;LEN], y:[f32;LEN], out:[f32;LEN]) { <body over 0..N> }`
fn ew(n: usize, body: &str) -> String {
    let len = n;
    format!("module f\nfn kbench(x:[f32;{len}], y:[f32;{len}], out:[f32;{len}]) {{ for i in 0..{n} {{ {body} }} }}\n")
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
        // mercury_norm_f32 call (in place on `out`). The interpreter marshals the identical kernel,
        // so the full probability buffer must be bit-exact against the native JIT.
        Kernel {
            name: "softmax",
            src: |n| {
                format!(
                    "module f\nfn kbench(x:[f32;{n}], y:[f32;{n}], out:[f32;{n}]) {{ \
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
        // into one mercury_norm_f32(.., NORM_LAYERNORM) call, in place on `out`. The `/ {n}.0`
        // divisor matches the trip count for every fuzz size, so the kernel fires across all sizes
        // and adversarial regimes; both backends marshal the identical kernel → full-buffer exact.
        Kernel {
            name: "layernorm",
            src: |n| {
                format!(
                    "module f\nfn kbench(x:[f32;{n}], y:[f32;{n}], out:[f32;{n}]) {{ \
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
                    "module f\nfn kbench(x:[f32;{n}], y:[f32;{n}], out:[f32;{n}]) {{ \
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
        // `out[i]*inv*gamma[i]` dispatches to mercury_norm_affine_f32 (gamma non-null, beta null —
        // exercises the absent-beta marshalling). The real transformer RMSNorm form.
        Kernel {
            name: "rmsnorm_affine",
            src: |n| {
                format!(
                    "module f\nfn kbench(x:[f32;{n}], y:[f32;{n}], out:[f32;{n}]) {{ \
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
        // non-null): `(out[i]-mean)*inv*gamma[i] + beta[i]` → one mercury_norm_affine_f32 call. Covers
        // the gamma+beta marshalling on both backends; bit-exact full-buffer.
        Kernel {
            name: "layernorm_affine",
            src: |n| {
                format!(
                    "module f\nfn kbench(x:[f32;{n}], y:[f32;{n}], out:[f32;{n}]) {{ \
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
        // matmul C=A·B (ikj) and nn.Linear C=A·Bᵀ (ijk) — both dispatch to the GEMM microkernel.
        Kernel {
            name: "matmul",
            src: |n| {
                format!(
                    "module f\nfn kbench(x:[f32;{l}], y:[f32;{l}], out:[f32;{l}]) {{ \
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
                    "module f\nfn kbench(x:[f32;{l}], y:[f32;{l}], out:[f32;{l}]) {{ \
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
                    "module f\nfn kbench(x:Tensor[f32,{n},{n}], y:Tensor[f32,{n},{n}], out:Tensor[f32,{n},{n}]) {{ \
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
                    "module f\nfn kbench(x:Tensor[f32,{n},{n}], y:Tensor[f32,{n},{n}], out:Tensor[f32,{n},{n}]) {{ \
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
                    "module f\nfn kbench(x:Tensor[f32,2,{n},{n}], y:Tensor[f32,2,{n},{n}], out:Tensor[f32,2,{n},{n}]) {{ \
             for b in 0..2 {{ for i in 0..{n} {{ for j in 0..{n} {{ out[b, i, j] = x[b, i, j] - y[b, i, j]; }} }} }} }}\n"
                )
            },
            len: |n| 2 * n * n,
            regimes: ANY,
        },
    ]
}

type K3 = unsafe extern "C" fn(*const f32, *const f32, *mut f32);

#[test]
fn fuzz_full_buffer_interp_vs_native() {
    // Sizes chosen to straddle the vectorizer's 4×128-bit unroll group and single-vector/scalar
    // remainder boundaries; matmul sizes straddle the 6×16 microkernel tile remainders.
    const EW_SIZES: &[usize] = &[1, 2, 3, 7, 8, 9, 15, 16, 17, 31, 33, 64, 100];
    const MM_SIZES: &[usize] = &[1, 2, 5, 6, 7, 8, 9, 16, 17];
    let mut rng = Rng(0x_C0FF_EE_123);
    let mut runs = 0u64;

    for k in kernels() {
        let sizes = if matches!(k.name, "matmul" | "linear") || k.name.starts_with("tensor") {
            MM_SIZES
        } else {
            EW_SIZES
        };
        for &n in sizes {
            let len = (k.len)(n);
            let src = (k.src)(n);
            let mut interner = Interner::new();
            let (module, pd) = mercury_parser::parse_module(&src, SourceId(0), &mut interner);
            assert!(pd.iter().all(|d| !d.is_error()), "{}: parse {pd:?}", k.name);
            let (sema, sd) = mercury_sema::check(&module, &interner);
            assert!(sd.iter().all(|d| !d.is_error()), "{}: sema {sd:?}", k.name);
            let kbench = interner.intern("kbench");

            for &opt in &[0u8, 3u8] {
                let (mut program, ld) =
                    mercury_mir_build::lower_program(&module, &sema, &mut interner);
                assert!(ld.iter().all(|d| !d.is_error()), "{}: lower {ld:?}", k.name);
                mercury_opt::optimize(&mut program, opt);
                let handle = crate::jit_module(&program, &interner).expect("jit_module");
                let ptr = handle.func_ptr(kbench).expect("func_ptr kbench");
                let native: K3 = unsafe { std::mem::transmute(ptr) };

                for &regime in k.regimes {
                    for _ in 0..4 {
                        let x: Vec<f32> = (0..len).map(|_| gen(&mut rng, regime)).collect();
                        let y: Vec<f32> = (0..len).map(|_| gen(&mut rng, regime)).collect();

                        // Interpreter over its own buffer copies.
                        let (mut xi, mut yi, mut oi) = (x.clone(), y.clone(), vec![0f32; len]);
                        mercury_interp::run_kernel_f32(
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
                        runs += 1;
                    }
                }
                drop(handle);
            }
        }
    }
    // A floor so an accidental empty corpus (e.g. a refactor that drops every kernel) fails loudly.
    assert!(runs > 1000, "fuzzer ran too few cases: {runs}");
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
            "module f\nfn lin(a:[u8;{len}], b:[i8;{len}], c:[i32;{len}]) {{ \
             for i in 0..{n} {{ for j in 0..{n} {{ let mut s: i32 = 0; \
             for k in 0..{n} {{ s = s + (a[i*{n}+k] as i32) * (b[j*{n}+k] as i32); }} \
             c[i*{n}+j] = s; }} }} }}\n"
        );
        let mut interner = Interner::new();
        let (module, pd) = mercury_parser::parse_module(&src, SourceId(0), &mut interner);
        assert!(pd.iter().all(|d| !d.is_error()), "i8 n={n}: parse {pd:?}");
        let (sema, sd) = mercury_sema::check(&module, &interner);
        assert!(sd.iter().all(|d| !d.is_error()), "i8 n={n}: sema {sd:?}");
        let lin = interner.intern("lin");

        for &opt in &[0u8, 3u8] {
            let (mut program, ld) = mercury_mir_build::lower_program(&module, &sema, &mut interner);
            assert!(ld.iter().all(|d| !d.is_error()), "i8 n={n}: lower {ld:?}");
            mercury_opt::optimize(&mut program, opt);
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
                mercury_interp::run_kernel_i8(&program, lin, &a, &b, &mut ci, &interner)
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

/// Verify the *value* of the shared transcendental kernels against an independent `f64` reference
/// within a tight relative tolerance — this catches the case both backends are wrong the *same*
/// way (a bug in the shared `mercury_vmath_f32` kernel that interp-vs-native equality cannot see).
///
/// The tolerance is per-function and honest about the algorithm, measured as **relative error**
/// (machine-independent, and the right metric near saturation where one ULP is a vanishing absolute
/// error): `exp`/`log` are *direct* minimax polynomials and `sqrt` is the hardware instruction, so
/// they hold ≈1e-6; `tanh`/`sigmoid`/`silu` are *composed* from `exp` (e.g. `tanh(x) = 1 −
/// 2/(exp(2x)+1)`), so the division amplifies exp's error to ≈1e-4 — still far tighter than the
/// bf16/f16 these activations usually feed. The bound documents that reality and catches gross errors.
#[test]
fn vmath_kernels_match_f64_reference() {
    // (name, true-function f64 reference, input regime, max relative error).
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
    ];
    let n = 256usize;
    let mut rng = Rng(0x_5EED_2024);
    for &(name, fref, regime, max_allowed) in cases {
        let src = ew(n, &format!("out[i] = {name}(x[i]);"));
        let mut interner = Interner::new();
        let (module, _) = mercury_parser::parse_module(&src, SourceId(0), &mut interner);
        let (sema, _) = mercury_sema::check(&module, &interner);
        let (mut program, _) = mercury_mir_build::lower_program(&module, &sema, &mut interner);
        mercury_opt::optimize(&mut program, 3);
        let kbench: Symbol = interner.intern("kbench");

        let x: Vec<f32> = (0..n).map(|_| gen(&mut rng, regime)).collect();
        let (mut xi, mut yi, mut oi) = (x.clone(), vec![0f32; n], vec![0f32; n]);
        mercury_interp::run_kernel_f32(
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
