//! `bench_model` — an honest END-TO-END transformer-layer-stack CPU inference benchmark:
//! Mercury vs a strong idiomatic C implementation of the *same* 12-layer GPT-2-class forward.
//!
//! What is measured
//! ----------------
//! One full inference forward over a GPT-2 124M-shaped decoder stack: 12 pre-LayerNorm
//! transformer blocks (multi-head causal attention + GELU MLP, both with residuals) plus the
//! final LayerNorm, at d_model=768, heads=12, d_ff=3072, and seq lengths S=128 / S=512.
//! Reported as ms per forward, ms per layer, and tokens/sec (S tokens per forward). Absolute
//! numbers are clock/thermal-bound on this box — the Mercury-vs-C **ratio** is the stable metric.
//!
//! How the two columns are built (the honesty contract)
//! ----------------------------------------------------
//! * **Mercury** goes through the real pipeline: parse → sema → mir_build → optimize(-O3) →
//!   Cranelift JIT, exactly like every other kernel in this suite (`bench_mercury`). The block is
//!   ordinary Mercury source (adapted from `examples/gpt2.mer` to the full 768/3072 config): the
//!   LayerNorms / GEMM nests / softmax / GELU are spelled in the idiomatic recognized op-forms, so
//!   `mir_build` dispatches them to the tuned runtime kernels (`mercury_norm_affine_f32`,
//!   `mercury_sgemm_nt[_alpha|_epi]`, `mercury_norm_f32`, `mercury_velem_f32`). The optimized MIR
//!   is scanned and the dispatched kernel set is printed, so the mechanism is transparent — and if
//!   a recognizer ever regresses, the bench says so instead of silently timing scalar loops.
//! * **C** is the same computation, hand-written as a competent single-TU implementation
//!   (contiguous loops, per-head slice extraction, its own scaled-QKᵀ / masked-softmax / PV
//!   attention, two-pass LayerNorm, tanh-approx GELU matching Mercury's), compiled at runtime with
//!   `gcc -O3 -march=native -ffp-contract=fast` — the suite's standard basis (no `-ffast-math`, so
//!   its f32 dot reductions stay serial, the same disclosed convention as `bench_linear`/`dot`).
//!   **C(fast)** is the identical source at `-O3 -march=native -ffast-math` (the `llama2.c
//!   -Ofast` basis), which lets gcc reassociate + vectorize the dot products — the strongest
//!   flags-only C column.
//! * **PyTorch (the industry peer)** — when `python` + `torch` import (probed gracefully; a
//!   printed note + skipped columns otherwise), the harness dumps the *exact* weight/input
//!   buffers as little-endian f32 blobs and generates a self-contained eager-PyTorch script
//!   that rebuilds the identical forward: `F.linear` computes `x·Wᵀ` over the same `[out,in]`
//!   row-major weights Mercury/C use (the layouts coincide — no transposition), `F.layer_norm`
//!   at the same eps=1e-5, the same tanh-approx GELU via `F.gelu(approximate="tanh")`, and
//!   multi-head causal attention two ways — `F.scaled_dot_product_attention(is_causal=True)`
//!   (the fused industry path) *and* a manual matmul+softmax variant — under
//!   `torch.inference_mode()`, float32, **eager only** (`torch.compile` is not attempted on
//!   Windows). Each variant warms ≥3 then times ≥10 forwards (min + median; the timing loop is
//!   one bare forward per iteration — no per-iteration allocation/IO beyond what eager torch
//!   does inside the forward), at `torch.set_num_threads(1)` and at the default all-threads.
//!   Torch runs in the same bench invocation immediately after the Mercury columns (same-run
//!   adjacency); its single-thread variants run first inside the script and the all-core one
//!   last, so multicore heat pollutes no single-thread torch number. Torch's SDPA output is
//!   cross-checked against Mercury's with the suite's magnitude-normalized metric at the same
//!   1e-3 tolerance (the GELU flavor matches exactly, so no loosening is needed).
//! * **The layer loop lives in this harness** for both languages: one JIT'd/compiled block
//!   function is called 12× per forward with per-layer weight pointers (ping-ponging two
//!   activation buffers), then the final LayerNorm — the way a real runtime drives a layer stack.
//!   Scratch buffers are host-allocated and passed as pointers (same buffers to both languages).
//! * **Correctness**: the Mercury and C final outputs are cross-checked elementwise over the whole
//!   `[S, 768]` buffer with a magnitude-normalized relative tolerance (float reassociation and
//!   poly-vs-libm transcendentals differ, compounding over 12 layers). Additionally the
//!   interpreter oracle runs the identical 12-layer forward at a reduced config and must match the
//!   JIT bit-for-bit (the differential gate that keeps the recognizer dispatches honest).
//!
//! Known asymmetries (disclosed, not hidden)
//! -----------------------------------------
//! * Mercury's GEMMs go to the tuned AVX2/FMA microkernel; C's stay whatever gcc makes of the
//!   idiomatic nests. That *is* the product claim being measured (a shape-safe tensor language
//!   whose compiler lowers to tuned kernels), the same basis as `bench_matmul`/`bench_linear`.
//! * The `@parallel` Mercury column is fully multicore: the batched norms, the fused-GELU FFN
//!   GEMM, *and* the embedded plain matmul nests (Q/K/V/O/PV/down-proj) all dispatch `_parallel`
//!   kernels (each bit-identical to its serial twin) — `lower_for`'s statement path honors the
//!   enclosing `@parallel` for embedded GEMMs. The run-time dispatch scan prints the `@parallel`
//!   variant's kernel set (and a unit test pins it), so a regression back to serial is visible,
//!   not silent. Both C columns are single-threaded idiomatic code, as everywhere in this suite;
//!   the torch `Tn(sdpa)` column is PyTorch's own all-thread path — the only other multicore
//!   column, disclosed as such.

use std::cell::Cell;
use std::io::Write;
use std::path::Path;
use std::process::Command;
use std::time::{Duration, Instant};

use mercury_span::{Interner, SourceId};

use crate::max_rel_err;

/// GPT-2 124M stacks 12 identical decoder blocks.
const LAYERS: usize = 12;

#[derive(Clone, Copy)]
struct Cfg {
    s: usize,
    d: usize,
    h: usize,
    dff: usize,
}

impl Cfg {
    fn hd(&self) -> usize {
        self.d / self.h
    }
    /// MAC-convention FLOPs of one block (2 per multiply-accumulate; norms/softmax/GELU excluded):
    /// QKV+WO projections `8·S·D²`, attention scores+PV `4·S²·D`, MLP up+down `4·S·D·Dff`.
    fn flops_per_layer(&self) -> f64 {
        let (s, d, dff) = (self.s as f64, self.d as f64, self.dff as f64);
        8.0 * s * d * d + 4.0 * s * s * d + 4.0 * s * d * dff
    }
}

/// The block ABI: 11 input pointers (activations + the layer's weights), 12 host-allocated scratch
/// buffers, and the output activations. Mercury array params pass by base pointer, so the JIT'd
/// function and the C `kbench` share this signature exactly.
type BlockFn = unsafe extern "C" fn(
    *const f32, // x        [S,D]
    *const f32, // ln1g     [D]
    *const f32, // ln1b     [D]
    *const f32, // wq       [D,D]
    *const f32, // wk       [D,D]
    *const f32, // wv       [D,D]
    *const f32, // wo       [D,D]
    *const f32, // ln2g     [D]
    *const f32, // ln2b     [D]
    *const f32, // w1       [Dff,D]
    *const f32, // w2       [D,Dff]
    *mut f32,   // nrm      [S,D]
    *mut f32,   // q        [S,D]
    *mut f32,   // k        [S,D]
    *mut f32,   // v        [S,D]
    *mut f32,   // qh       [S,hd]
    *mut f32,   // kh       [S,hd]
    *mut f32,   // vt       [hd,S]
    *mut f32,   // scores   [S,S]
    *mut f32,   // ah       [S,hd]
    *mut f32,   // attn     [S,D]
    *mut f32,   // a        [S,D]
    *mut f32,   // ff1      [S,Dff]
    *mut f32,   // out      [S,D]
);

/// The final-LayerNorm ABI: `(x, gamma, beta, out)`.
type LnFn = unsafe extern "C" fn(*const f32, *const f32, *const f32, *mut f32);

struct LayerW {
    ln1g: Vec<f32>,
    ln1b: Vec<f32>,
    wq: Vec<f32>,
    wk: Vec<f32>,
    wv: Vec<f32>,
    wo: Vec<f32>,
    ln2g: Vec<f32>,
    ln2b: Vec<f32>,
    w1: Vec<f32>,
    w2: Vec<f32>,
}

struct Scratch {
    nrm: Vec<f32>,
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    qh: Vec<f32>,
    kh: Vec<f32>,
    vt: Vec<f32>,
    scores: Vec<f32>,
    ah: Vec<f32>,
    attn: Vec<f32>,
    a: Vec<f32>,
    ff1: Vec<f32>,
}

impl Scratch {
    fn new(cfg: Cfg) -> Self {
        let (sd, shd, ss, sdff) = (
            cfg.s * cfg.d,
            cfg.s * cfg.hd(),
            cfg.s * cfg.s,
            cfg.s * cfg.dff,
        );
        Scratch {
            nrm: vec![0.0; sd],
            q: vec![0.0; sd],
            k: vec![0.0; sd],
            v: vec![0.0; sd],
            qh: vec![0.0; shd],
            kh: vec![0.0; shd],
            vt: vec![0.0; shd],
            scores: vec![0.0; ss],
            ah: vec![0.0; shd],
            attn: vec![0.0; sd],
            a: vec![0.0; sd],
            ff1: vec![0.0; sdff],
        }
    }
}

/// Deterministic pseudo-random fill in `[-1, 1)` (LCG; same weights every run and for every
/// language column — the fill happens once in the harness, the kernels just read the buffers).
fn lcg(s: &mut u64) -> f32 {
    *s = s
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*s >> 40) as u32 as f32 / (1u32 << 23) as f32) - 1.0
}

fn fill(n: usize, scale: f32, off: f32, seed: &mut u64) -> Vec<f32> {
    (0..n).map(|_| off + scale * lcg(seed)).collect()
}

/// GPT-2-ish init: projections ~ ±0.02, gains ~ 1±0.1, biases ~ ±0.02 — the LayerNorms
/// re-normalize every sublayer input, so activations stay bounded through all 12 layers.
fn make_weights(cfg: Cfg, seed: &mut u64) -> Vec<LayerW> {
    let (d, dd, dffd) = (cfg.d, cfg.d * cfg.d, cfg.dff * cfg.d);
    (0..LAYERS)
        .map(|_| LayerW {
            ln1g: fill(d, 0.1, 1.0, seed),
            ln1b: fill(d, 0.02, 0.0, seed),
            wq: fill(dd, 0.02, 0.0, seed),
            wk: fill(dd, 0.02, 0.0, seed),
            wv: fill(dd, 0.02, 0.0, seed),
            wo: fill(dd, 0.02, 0.0, seed),
            ln2g: fill(d, 0.1, 1.0, seed),
            ln2b: fill(d, 0.02, 0.0, seed),
            w1: fill(dffd, 0.02, 0.0, seed),
            w2: fill(dffd, 0.02, 0.0, seed),
        })
        .collect()
}

// -------------------------------------------------------------------------------------------
// Sources
// -------------------------------------------------------------------------------------------

/// One GPT-2 block in Mercury, adapted from `examples/gpt2.mer` to the full config, with every
/// sub-op spelled in its recognized form (verified by the dispatch scan printed at run time):
///  * batched affine LayerNorm  -> `mercury_norm_affine_f32`
///  * `nn.Linear` NT dot nests  -> `mercury_sgemm_nt`
///  * scaled scores `α·(Q·Kᵀ)`  -> `mercury_sgemm_nt_alpha`
///  * batched masked softmax    -> `mercury_norm_f32`
///  * GEMM + `gelu` epilogue    -> `mercury_sgemm_nt_epi`
///  * residual adds             -> `mercury_velem_f32`
/// Attention runs per head over contiguous slices (extract Qh/Kh, V transposed) so each head's
/// scores / PV products are plain NT GEMMs — the same structure the C implementation uses.
fn mer_block(cfg: Cfg, parallel: bool) -> String {
    let attr = if parallel { "@parallel\n" } else { "" };
    let (s, d, h, dff, hd) = (cfg.s, cfg.d, cfg.h, cfg.dff, cfg.hd());
    let (sd, ss, shd, sdff, dd, dffd) = (s * d, s * s, s * hd, s * dff, d * d, dff * d);
    let scale = 1.0 / (hd as f64).sqrt(); // hd is a power of 4 here, so this is exact in f32
    format!(
        "module bench
{attr}fn kbench(
    x: [f32; {sd}],
    ln1g: [f32; {d}], ln1b: [f32; {d}],
    wq: [f32; {dd}], wk: [f32; {dd}], wv: [f32; {dd}], wo: [f32; {dd}],
    ln2g: [f32; {d}], ln2b: [f32; {d}],
    w1: [f32; {dffd}], w2: [f32; {dffd}],
    mut nrm: [f32; {sd}],
    mut q: [f32; {sd}], mut k: [f32; {sd}], mut v: [f32; {sd}],
    mut qh: [f32; {shd}], mut kh: [f32; {shd}], mut vt: [f32; {shd}],
    mut scores: [f32; {ss}],
    mut ah: [f32; {shd}],
    mut attn: [f32; {sd}],
    mut a: [f32; {sd}],
    mut ff1: [f32; {sdff}],
    mut out: [f32; {sd}],
) {{
    // 1. LayerNorm1(x) -> nrm
    for i in 0..{sd} {{ nrm[i] = x[i]; }}
    for r in 0..{s} {{
        let mut sm: f32 = 0.0;
        for i in 0..{d} {{ sm = sm + nrm[r*{d}+i]; }}
        let mean: f32 = sm / {d}.0;
        let mut vv: f32 = 0.0;
        for i in 0..{d} {{ vv = vv + (nrm[r*{d}+i] - mean) * (nrm[r*{d}+i] - mean); }}
        let inv: f32 = rsqrt(vv / {d}.0 + 0.00001);
        for i in 0..{d} {{ nrm[r*{d}+i] = (nrm[r*{d}+i] - mean) * inv * ln1g[i] + ln1b[i]; }}
    }}
    // 2. Q/K/V projections (nn.Linear)
    for i in 0..{s} {{ for j in 0..{d} {{
        let mut acc: f32 = 0.0;
        for p in 0..{d} {{ acc = acc + nrm[i*{d}+p] * wq[j*{d}+p]; }}
        q[i*{d}+j] = acc;
    }} }}
    for i in 0..{s} {{ for j in 0..{d} {{
        let mut acc: f32 = 0.0;
        for p in 0..{d} {{ acc = acc + nrm[i*{d}+p] * wk[j*{d}+p]; }}
        k[i*{d}+j] = acc;
    }} }}
    for i in 0..{s} {{ for j in 0..{d} {{
        let mut acc: f32 = 0.0;
        for p in 0..{d} {{ acc = acc + nrm[i*{d}+p] * wv[j*{d}+p]; }}
        v[i*{d}+j] = acc;
    }} }}
    // 3. Multi-head causal attention (H = {h}, hd = {hd})
    for hh in 0..{h} {{
        for i in 0..{s} {{ for p in 0..{hd} {{ qh[i*{hd}+p] = q[i*{d} + hh*{hd} + p]; }} }}
        for i in 0..{s} {{ for p in 0..{hd} {{ kh[i*{hd}+p] = k[i*{d} + hh*{hd} + p]; }} }}
        for j in 0..{hd} {{ for p in 0..{s} {{ vt[j*{s}+p] = v[p*{d} + hh*{hd} + j]; }} }}
        // scores = (Qh . KhT) * scale
        for i in 0..{s} {{ for j in 0..{s} {{
            let mut acc: f32 = 0.0;
            for p in 0..{hd} {{ acc = acc + qh[i*{hd}+p] * kh[j*{hd}+p]; }}
            scores[i*{s}+j] = {scale} * acc;
        }} }}
        // causal mask
        for i in 0..{s} {{ for j in 0..{s} {{ if j > i {{ scores[i*{s}+j] = 0.0 - 1.0e30; }} }} }}
        // numerically-stable row softmax, in place
        for r in 0..{s} {{
            let mut m: f32 = scores[r*{s}];
            for i in 0..{s} {{ m = fmax(m, scores[r*{s}+i]); }}
            for i in 0..{s} {{ scores[r*{s}+i] = exp(scores[r*{s}+i] - m); }}
            let mut sm: f32 = 0.0;
            for i in 0..{s} {{ sm = sm + scores[r*{s}+i]; }}
            let inv: f32 = 1.0 / sm;
            for i in 0..{s} {{ scores[r*{s}+i] = scores[r*{s}+i] * inv; }}
        }}
        // head output = scores . Vh
        for i in 0..{s} {{ for j in 0..{hd} {{
            let mut acc: f32 = 0.0;
            for p in 0..{s} {{ acc = acc + scores[i*{s}+p] * vt[j*{s}+p]; }}
            ah[i*{hd}+j] = acc;
        }} }}
        for i in 0..{s} {{ for j in 0..{hd} {{ attn[i*{d} + hh*{hd} + j] = ah[i*{hd}+j]; }} }}
    }}
    // 4. Output projection + residual: a = x + attn . WoT
    for i in 0..{s} {{ for j in 0..{d} {{
        let mut acc: f32 = 0.0;
        for p in 0..{d} {{ acc = acc + attn[i*{d}+p] * wo[j*{d}+p]; }}
        a[i*{d}+j] = acc;
    }} }}
    for i in 0..{sd} {{ a[i] = x[i] + a[i]; }}
    // 5. LayerNorm2(a) -> nrm
    for i in 0..{sd} {{ nrm[i] = a[i]; }}
    for r in 0..{s} {{
        let mut sm: f32 = 0.0;
        for i in 0..{d} {{ sm = sm + nrm[r*{d}+i]; }}
        let mean: f32 = sm / {d}.0;
        let mut vv: f32 = 0.0;
        for i in 0..{d} {{ vv = vv + (nrm[r*{d}+i] - mean) * (nrm[r*{d}+i] - mean); }}
        let inv: f32 = rsqrt(vv / {d}.0 + 0.00001);
        for i in 0..{d} {{ nrm[r*{d}+i] = (nrm[r*{d}+i] - mean) * inv * ln2g[i] + ln2b[i]; }}
    }}
    // 6. MLP up-projection with fused GELU epilogue
    for i in 0..{s} {{ for j in 0..{dff} {{
        let mut acc: f32 = 0.0;
        for p in 0..{d} {{ acc = acc + nrm[i*{d}+p] * w1[j*{d}+p]; }}
        ff1[i*{dff}+j] = acc;
    }} }}
    for i in 0..{s} {{ for j in 0..{dff} {{ ff1[i*{dff}+j] = gelu(ff1[i*{dff}+j]); }} }}
    // 7. MLP down-projection + residual: out = a + ff1 . W2T
    for i in 0..{s} {{ for j in 0..{d} {{
        let mut acc: f32 = 0.0;
        for p in 0..{dff} {{ acc = acc + ff1[i*{dff}+p] * w2[j*{dff}+p]; }}
        out[i*{d}+j] = acc;
    }} }}
    for i in 0..{sd} {{ out[i] = a[i] + out[i]; }}
}}
"
    )
}

/// The final LayerNorm (`ln_f` in GPT-2) as its own Mercury module — the batched affine norm form,
/// dispatching one `mercury_norm_affine_f32` call.
fn mer_final_ln(cfg: Cfg) -> String {
    let (s, d) = (cfg.s, cfg.d);
    let sd = s * d;
    format!(
        "module bench
fn kbench(x: [f32; {sd}], g: [f32; {d}], b: [f32; {d}], mut out: [f32; {sd}]) {{
    for i in 0..{sd} {{ out[i] = x[i]; }}
    for r in 0..{s} {{
        let mut sm: f32 = 0.0;
        for i in 0..{d} {{ sm = sm + out[r*{d}+i]; }}
        let mean: f32 = sm / {d}.0;
        let mut vv: f32 = 0.0;
        for i in 0..{d} {{ vv = vv + (out[r*{d}+i] - mean) * (out[r*{d}+i] - mean); }}
        let inv: f32 = rsqrt(vv / {d}.0 + 0.00001);
        for i in 0..{d} {{ out[r*{d}+i] = (out[r*{d}+i] - mean) * inv * g[i] + b[i]; }}
    }}
}}
"
    )
}

/// The same 12-layer block + final LayerNorm as one competent C translation unit. Same loop
/// structure, same per-head slice extraction, same scratch (passed in), tanh-approx GELU with
/// Mercury's constants. Exported as `kbench` (block) and `kfinal` (final LayerNorm).
fn c_model(cfg: Cfg) -> String {
    let (s, d, h, dff, hd) = (cfg.s, cfg.d, cfg.h, cfg.dff, cfg.hd());
    let scale = 1.0 / (hd as f64).sqrt();
    format!(
        "#include <math.h>
#define S {s}
#define D {d}
#define H {h}
#define DFF {dff}
#define HD {hd}
#define SCALE {scale}f

static void layernorm_affine(float* t, const float* g, const float* b) {{
  for (long r = 0; r < S; r++) {{
    float sm = 0.0f;
    for (long i = 0; i < D; i++) sm += t[r*D+i];
    float mean = sm / (float)D;
    float vv = 0.0f;
    for (long i = 0; i < D; i++) {{ float d0 = t[r*D+i] - mean; vv += d0*d0; }}
    float inv = 1.0f / sqrtf(vv / (float)D + 1e-5f);
    for (long i = 0; i < D; i++) t[r*D+i] = (t[r*D+i] - mean) * inv * g[i] + b[i];
  }}
}}

/* nn.Linear: out[M,N] = in[M,K] . w[N,K]^T (contiguous dot over K for both operands) */
static void linear_nt(const float* in, const float* w, float* out, long m, long kk, long n) {{
  for (long i = 0; i < m; i++)
    for (long j = 0; j < n; j++) {{
      float acc = 0.0f;
      for (long p = 0; p < kk; p++) acc += in[i*kk+p] * w[j*kk+p];
      out[i*n+j] = acc;
    }}
}}

__declspec(dllexport) void kbench(
    const float* x,
    const float* ln1g, const float* ln1b,
    const float* wq, const float* wk, const float* wv, const float* wo,
    const float* ln2g, const float* ln2b,
    const float* w1, const float* w2,
    float* nrm, float* q, float* k, float* v,
    float* qh, float* kh, float* vt,
    float* scores, float* ah, float* attn, float* a, float* ff1,
    float* out) {{
  /* 1. LayerNorm1(x) -> nrm */
  for (long i = 0; i < S*D; i++) nrm[i] = x[i];
  layernorm_affine(nrm, ln1g, ln1b);
  /* 2. Q/K/V projections */
  linear_nt(nrm, wq, q, S, D, D);
  linear_nt(nrm, wk, k, S, D, D);
  linear_nt(nrm, wv, v, S, D, D);
  /* 3. multi-head causal attention over contiguous head slices */
  for (long hh = 0; hh < H; hh++) {{
    for (long i = 0; i < S; i++)
      for (long p = 0; p < HD; p++) qh[i*HD+p] = q[i*D + hh*HD + p];
    for (long i = 0; i < S; i++)
      for (long p = 0; p < HD; p++) kh[i*HD+p] = k[i*D + hh*HD + p];
    for (long j = 0; j < HD; j++)
      for (long p = 0; p < S; p++) vt[j*S+p] = v[p*D + hh*HD + j];
    for (long i = 0; i < S; i++)
      for (long j = 0; j < S; j++) {{
        float acc = 0.0f;
        for (long p = 0; p < HD; p++) acc += qh[i*HD+p] * kh[j*HD+p];
        scores[i*S+j] = SCALE * acc;
      }}
    for (long i = 0; i < S; i++)
      for (long j = i+1; j < S; j++) scores[i*S+j] = -1.0e30f;
    for (long r = 0; r < S; r++) {{
      float m = scores[r*S];
      for (long i = 0; i < S; i++) if (scores[r*S+i] > m) m = scores[r*S+i];
      float sm = 0.0f;
      for (long i = 0; i < S; i++) {{ float e = expf(scores[r*S+i] - m); scores[r*S+i] = e; sm += e; }}
      float inv = 1.0f / sm;
      for (long i = 0; i < S; i++) scores[r*S+i] *= inv;
    }}
    for (long i = 0; i < S; i++)
      for (long j = 0; j < HD; j++) {{
        float acc = 0.0f;
        for (long p = 0; p < S; p++) acc += scores[i*S+p] * vt[j*S+p];
        ah[i*HD+j] = acc;
      }}
    for (long i = 0; i < S; i++)
      for (long j = 0; j < HD; j++) attn[i*D + hh*HD + j] = ah[i*HD+j];
  }}
  /* 4. output projection + residual */
  linear_nt(attn, wo, a, S, D, D);
  for (long i = 0; i < S*D; i++) a[i] = x[i] + a[i];
  /* 5. LayerNorm2(a) -> nrm */
  for (long i = 0; i < S*D; i++) nrm[i] = a[i];
  layernorm_affine(nrm, ln2g, ln2b);
  /* 6. MLP up-projection + tanh-approx GELU (Mercury's gelu constants) */
  linear_nt(nrm, w1, ff1, S, D, DFF);
  for (long i = 0; i < S*DFF; i++) {{
    float t = ff1[i];
    float inner = 0.7978846f * (0.044715f * t*t*t + t);
    ff1[i] = 0.5f * t * (1.0f + tanhf(inner));
  }}
  /* 7. MLP down-projection + residual */
  linear_nt(ff1, w2, out, S, DFF, D);
  for (long i = 0; i < S*D; i++) out[i] = a[i] + out[i];
}}

__declspec(dllexport) void kfinal(const float* x, const float* g, const float* b, float* out) {{
  for (long i = 0; i < S*D; i++) out[i] = x[i];
  layernorm_affine(out, g, b);
}}
"
    )
}

// -------------------------------------------------------------------------------------------
// Compilation
// -------------------------------------------------------------------------------------------

struct MerModule {
    handle: mercury_codegen_cranelift::JitModuleHandle,
    compile: Duration,
    /// Optimized-MIR text, for the recognized-kernel dispatch scan.
    mir: String,
}

impl MerModule {
    fn func(&self, interner: &mut Interner, name: &str) -> Option<*const u8> {
        let sym = interner.intern(name);
        self.handle.func_ptr(sym)
    }
}

/// The real pipeline, same as `bench_mercury`: parse → sema → mir_build → optimize(-O3) → JIT.
fn compile_mercury(src: &str, interner: &mut Interner) -> Option<MerModule> {
    let t = Instant::now();
    let (module, pd) = mercury_parser::parse_module(src, SourceId(0), interner);
    if pd.iter().any(|d| d.is_error()) {
        eprintln!("model: mercury parse error");
        return None;
    }
    let (sema, sd) = mercury_sema::check(&module, interner);
    if sd.iter().any(|d| d.is_error()) {
        eprintln!("model: mercury sema error: {sd:?}");
        return None;
    }
    let (mut program, ld) = mercury_mir_build::lower_program(&module, &sema, interner);
    if ld.iter().any(|d| d.is_error()) {
        eprintln!("model: mercury lower error: {ld:?}");
        return None;
    }
    mercury_opt::optimize(&mut program, 3);
    let mir = mercury_mir::print::print_program(&program, interner);
    let handle = match mercury_codegen_cranelift::jit_module(&program, interner) {
        Ok(h) => h,
        Err(e) => {
            eprintln!("model: mercury codegen error: {e}");
            return None;
        }
    };
    let compile = t.elapsed();
    Some(MerModule {
        handle,
        compile,
        mir,
    })
}

/// Compile + optimize only (no JIT) — the front half of the pipeline, for the interpreter oracle.
fn build_program(src: &str, interner: &mut Interner) -> Option<mercury_mir::Program> {
    let (module, pd) = mercury_parser::parse_module(src, SourceId(0), interner);
    if pd.iter().any(|d| d.is_error()) {
        return None;
    }
    let (sema, sd) = mercury_sema::check(&module, interner);
    if sd.iter().any(|d| d.is_error()) {
        return None;
    }
    let (mut program, ld) = mercury_mir_build::lower_program(&module, &sema, interner);
    if ld.iter().any(|d| d.is_error()) {
        return None;
    }
    mercury_opt::optimize(&mut program, 3);
    Some(program)
}

/// Scan optimized MIR for `mercury_*` runtime-kernel calls: the single-source-of-truth check that
/// the block really dispatches to the recognized kernels (recognizers are gate-blind — both
/// backends call the same symbol — so the bench *prints* what it runs).
fn kernel_calls(mir: &str) -> Vec<(String, usize)> {
    let mut counts: std::collections::BTreeMap<&str, usize> = std::collections::BTreeMap::new();
    for (i, _) in mir.match_indices("mercury_") {
        let rest = &mir[i..];
        let end = rest
            .find(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
            .unwrap_or(rest.len());
        *counts.entry(&rest[..end]).or_insert(0) += 1;
    }
    counts
        .into_iter()
        .map(|(k, v)| (k.to_string(), v))
        .collect()
}

struct CModel {
    _lib: libloading::Library,
    block: BlockFn,
    lnf: LnFn,
    compile: Duration,
}

fn compile_c_model(
    src: &str,
    dir: &Path,
    name: &str,
    cc: &str,
    args: &[&str],
) -> Option<CModel> {
    let src_path = dir.join(format!("{name}.c"));
    let dll = dir.join(format!("{name}_c.dll"));
    std::fs::write(&src_path, src).ok()?;
    let t = Instant::now();
    let status = Command::new(cc)
        .args(args)
        .arg("-o")
        .arg(&dll)
        .arg(&src_path)
        .status();
    let compile = t.elapsed();
    match status {
        Ok(s) if s.success() => {}
        Ok(_) => {
            eprintln!("{cc} failed to compile {name}.c");
            return None;
        }
        Err(_) => {
            eprintln!("could not run `{cc}` (skipping)");
            return None;
        }
    }
    unsafe {
        let lib = libloading::Library::new(&dll).ok()?;
        let block: BlockFn = *lib.get::<BlockFn>(b"kbench\0").ok()?;
        let lnf: LnFn = *lib.get::<LnFn>(b"kfinal\0").ok()?;
        Some(CModel {
            _lib: lib,
            block,
            lnf,
            compile,
        })
    }
}

// -------------------------------------------------------------------------------------------
// The forward driver (identical for every column) and timing
// -------------------------------------------------------------------------------------------

/// One full forward: restore the pristine input, run the block 12× with per-layer weights
/// (ping-ponging the two activation buffers), then the final LayerNorm into `y`.
#[allow(clippy::too_many_arguments)]
unsafe fn run_forward(
    block: BlockFn,
    lnf: LnFn,
    weights: &[LayerW],
    sc: &mut Scratch,
    x0: &[f32],
    xa: &mut [f32],
    xb: &mut [f32],
    lnf_g: &[f32],
    lnf_b: &[f32],
    y: &mut [f32],
) {
    xa.copy_from_slice(x0);
    for (l, w) in weights.iter().enumerate() {
        let (xi, xo) = if l % 2 == 0 {
            (xa.as_ptr(), xb.as_mut_ptr())
        } else {
            (xb.as_ptr(), xa.as_mut_ptr())
        };
        block(
            xi,
            w.ln1g.as_ptr(),
            w.ln1b.as_ptr(),
            w.wq.as_ptr(),
            w.wk.as_ptr(),
            w.wv.as_ptr(),
            w.wo.as_ptr(),
            w.ln2g.as_ptr(),
            w.ln2b.as_ptr(),
            w.w1.as_ptr(),
            w.w2.as_ptr(),
            sc.nrm.as_mut_ptr(),
            sc.q.as_mut_ptr(),
            sc.k.as_mut_ptr(),
            sc.v.as_mut_ptr(),
            sc.qh.as_mut_ptr(),
            sc.kh.as_mut_ptr(),
            sc.vt.as_mut_ptr(),
            sc.scores.as_mut_ptr(),
            sc.ah.as_mut_ptr(),
            sc.attn.as_mut_ptr(),
            sc.a.as_mut_ptr(),
            sc.ff1.as_mut_ptr(),
            xo,
        );
    }
    // LAYERS is even, so the last block wrote xa.
    lnf(xa.as_ptr(), lnf_g.as_ptr(), lnf_b.as_ptr(), y.as_mut_ptr());
}

/// Best-of-N timing sized for calls that cost 0.1–60 s (the whole-forward scale, where
/// `time_ns`'s 50 ms-batch protocol would take minutes per column): 1 warmup call, then keep
/// sampling until ≥4 s of measured time or 8 samples (at least 2 samples unless a single call
/// already exceeds 20 s). Reports the fastest sample — the same least-interfered "best observed"
/// methodology as `time_ns`.
fn time_forward(mut run: impl FnMut()) -> f64 {
    run(); // warm: page in weights, JIT/rayon warmup
    let mut best = f64::INFINITY;
    let mut total = 0.0f64;
    let mut n = 0usize;
    while n < 8 {
        if n >= 2 && total >= 4.0 {
            break;
        }
        if n >= 1 && total >= 20.0 {
            break;
        }
        let t = Instant::now();
        run();
        let e = t.elapsed().as_secs_f64();
        total += e;
        n += 1;
        if e < best {
            best = e;
        }
    }
    best * 1e9
}

#[derive(Clone)]
struct MeasureModel {
    compile: Duration,
    ns_per_fwd: f64,
    out: Vec<f32>,
}

// -------------------------------------------------------------------------------------------
// The PyTorch CPU peer (industry baseline)
// -------------------------------------------------------------------------------------------

/// The probed PyTorch environment: interpreter path, version/thread info, and whether the
/// (config-invariant) weight blob has already been dumped this run.
struct TorchCtx {
    py: String,
    version: String,
    threads: usize,
    weights_written: Cell<bool>,
}

/// Probe `python` (override with `PYTHON`) for an importable torch. Graceful: any failure —
/// no python on PATH, torch not installed — returns `None`; the bench prints a note and the
/// torch columns show `n/a`.
fn detect_torch() -> Option<TorchCtx> {
    let py = std::env::var("PYTHON").unwrap_or_else(|_| "python".to_string());
    let out = Command::new(&py)
        .args([
            "-c",
            "import torch; print(torch.__version__); print(torch.get_num_threads())",
        ])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let s = String::from_utf8_lossy(&out.stdout);
    let mut lines = s.lines();
    let version = lines.next()?.trim().to_string();
    let threads = lines.next()?.trim().parse().ok()?;
    Some(TorchCtx {
        py,
        version,
        threads,
        weights_written: Cell::new(false),
    })
}

/// Concatenate f32 slices into one little-endian binary file — the exact bytes the generated
/// Python reads back with `torch.frombuffer(dtype=torch.float32)`.
fn dump_f32_le(path: &Path, parts: &[&[f32]]) -> std::io::Result<()> {
    let f = std::fs::File::create(path)?;
    let mut w = std::io::BufWriter::with_capacity(1 << 20, f);
    for xs in parts {
        let mut buf = Vec::with_capacity(xs.len() * 4);
        for &v in *xs {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        w.write_all(&buf)?;
    }
    w.flush()
}

/// The self-contained eager-PyTorch peer script for one config. Layout facts it relies on
/// (verified against `mer_block`/`c_model`): `F.linear(x, W)` computes `x·Wᵀ` over a `[out,in]`
/// row-major `W` — exactly the `w[j*K+p]` layout Mercury and C dot against, so the dumped bytes
/// are used as-is; LayerNorm eps is 1e-5 in all three; the GELU is the tanh approximation
/// (√(2/π), 0.044715) — torch's `approximate="tanh"`; SDPA's default scale `1/√hd` equals
/// Mercury's inline `scale` (hd is a power of 4, exact in f32); Mercury's `-1e30` mask and
/// torch's `-inf`/`is_causal` agree after softmax (both underflow to exactly 0).
fn torch_script(cfg: Cfg, w_path: &Path, io_path: &Path, out_path: &Path) -> String {
    let (s, d, h, hd, dff) = (cfg.s, cfg.d, cfg.h, cfg.hd(), cfg.dff);
    format!(
        r#"# Auto-generated by mercury_xbench `model` — the PyTorch CPU peer for the 12-layer
# GPT-2-class forward. Reads the exact little-endian f32 weight/input bytes the Mercury and C
# columns use, rebuilds the identical eager float32 forward, and times it under
# torch.inference_mode(). Eager ONLY — torch.compile is not attempted (Windows).
import sys, time, array, statistics
import torch
import torch.nn.functional as F

S = {s}; D = {d}; H = {h}; HD = {hd}; DFF = {dff}; LAYERS = {layers}
W_PATH = r"{w}"
IO_PATH = r"{io}"
OUT_PATH = r"{out}"
WARMUP = 3
ITERS = 10
LNSHAPE = (D,)
DEFAULT_THREADS = torch.get_num_threads()

def load(path):
    with open(path, "rb") as f:
        data = f.read()
    return torch.frombuffer(bytearray(data), dtype=torch.float32)

wbuf = load(W_PATH)
sizes = [D, D, D * D, D * D, D * D, D * D, D, D, DFF * D, D * DFF]
shapes = [(D,), (D,), (D, D), (D, D), (D, D), (D, D), (D,), (D,), (DFF, D), (D, DFF)]
layers = []
off = 0
for _ in range(LAYERS):
    ws = []
    for n, sh in zip(sizes, shapes):
        ws.append(wbuf[off:off + n].clone().view(sh))
        off += n
    layers.append(ws)
assert off == wbuf.numel(), "weight blob size mismatch"
iobuf = load(IO_PATH)
assert iobuf.numel() == 2 * D + S * D, "io blob size mismatch"
lnf_g = iobuf[0:D].clone()
lnf_b = iobuf[D:2 * D].clone()
x0 = iobuf[2 * D:].clone().view(S, D)
del wbuf, iobuf

SCALE = HD ** -0.5
NEGINF = float("-inf")
MASK = torch.triu(torch.ones(S, S, dtype=torch.bool), diagonal=1)

# One pre-LN block. Linear weights are [out, in] row-major — torch's own x @ W.T layout, byte-
# identical to what Mercury/C dot against. GELU is the tanh approximation, matching Mercury's
# gelu() and the C column exactly (same flavor, so the cross-check needs no loosening).
def block(x, w, manual):
    ln1g, ln1b, wq, wk, wv, wo, ln2g, ln2b, w1, w2 = w
    nrm = F.layer_norm(x, LNSHAPE, ln1g, ln1b, 1e-5)
    q = F.linear(nrm, wq).view(S, H, HD).transpose(0, 1)
    k = F.linear(nrm, wk).view(S, H, HD).transpose(0, 1)
    v = F.linear(nrm, wv).view(S, H, HD).transpose(0, 1)
    if manual:
        sc = torch.matmul(q, k.transpose(-2, -1)) * SCALE
        sc = sc.masked_fill(MASK, NEGINF)
        att = torch.matmul(torch.softmax(sc, dim=-1), v)
    else:
        att = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    att = att.transpose(0, 1).reshape(S, D)
    a = x + F.linear(att, wo)
    nrm2 = F.layer_norm(a, LNSHAPE, ln2g, ln2b, 1e-5)
    return a + F.linear(F.gelu(F.linear(nrm2, w1), approximate="tanh"), w2)

def forward(manual):
    x = x0
    for w in layers:
        x = block(x, w, manual)
    return F.layer_norm(x, LNSHAPE, lnf_g, lnf_b, 1e-5)

# Warm WARMUP forwards, then time ITERS. The timed loop body is exactly one forward — no
# per-iteration allocation or IO beyond what eager torch itself does inside the forward.
def bench(manual):
    for _ in range(WARMUP):
        forward(manual)
    ts = [0.0] * ITERS
    for i in range(ITERS):
        t0 = time.perf_counter()
        forward(manual)
        ts[i] = time.perf_counter() - t0
    return min(ts) * 1e3, statistics.median(ts) * 1e3

with torch.inference_mode():
    print("TORCH_VERSION %s" % torch.__version__, flush=True)
    print("TORCH_THREADS %d" % DEFAULT_THREADS, flush=True)
    # Cross-check outputs first, at 1 thread (doubles as cache warmup for the timed 1t runs).
    torch.set_num_threads(1)
    y = forward(False)
    ym = forward(True)
    scale = max(y.abs().max().item(), 1e-6)
    print("TORCH_MANUAL_VS_SDPA %.3e" % ((ym - y).abs().max().item() / scale), flush=True)
    a = array.array("f", y.reshape(-1).tolist())
    if sys.byteorder != "little":
        a.byteswap()
    with open(OUT_PATH, "wb") as f:
        f.write(a.tobytes())
    mn, med = bench(False)
    print("TORCH1 %.3f %.3f" % (mn, med), flush=True)
    mn, med = bench(True)
    print("TORCH1M %.3f %.3f" % (mn, med), flush=True)
    # All-core variant LAST so its heat pollutes no single-thread torch number.
    torch.set_num_threads(DEFAULT_THREADS)
    mn, med = bench(False)
    print("TORCHN %.3f %.3f" % (mn, med), flush=True)
print("TORCH_OK", flush=True)
"#,
        s = s,
        d = d,
        h = h,
        hd = hd,
        dff = dff,
        layers = LAYERS,
        w = w_path.display(),
        io = io_path.display(),
        out = out_path.display(),
    )
}

/// Parsed torch timings — `(min, median)` ns/forward per variant — plus the SDPA output for the
/// cross-check and the script's own manual-vs-SDPA agreement figure.
struct TorchMeasure {
    sdpa_1t: Option<(f64, f64)>,
    manual_1t: Option<(f64, f64)>,
    sdpa_nt: Option<(f64, f64)>,
    manual_vs_sdpa: Option<f64>,
    out: Vec<f32>,
}

/// Dump the shared buffers, generate + run the peer script, parse its machine-readable lines.
fn bench_torch(
    ctx: &TorchCtx,
    dir: &Path,
    cfg: Cfg,
    weights: &[LayerW],
    x0: &[f32],
    lnf_g: &[f32],
    lnf_b: &[f32],
) -> Option<TorchMeasure> {
    // The 12-layer weights are config-invariant in this bench (same d/dff/LAYERS, and the
    // 0x0D15EA5E seed is consumed identically before x0 at every S), so the ~324 MiB blob is
    // dumped once per run; the small per-config blob carries lnf gamma/beta + the input.
    let w_path = dir.join("model_w.bin");
    if !ctx.weights_written.get() {
        let mut parts: Vec<&[f32]> = Vec::with_capacity(weights.len() * 10);
        for lw in weights {
            for t in [
                &lw.ln1g, &lw.ln1b, &lw.wq, &lw.wk, &lw.wv, &lw.wo, &lw.ln2g, &lw.ln2b, &lw.w1,
                &lw.w2,
            ] {
                parts.push(t);
            }
        }
        if let Err(e) = dump_f32_le(&w_path, &parts) {
            eprintln!("model: torch peer: cannot write weight blob: {e}");
            return None;
        }
        ctx.weights_written.set(true);
    }
    let io_path = dir.join(format!("model_io_s{}.bin", cfg.s));
    if let Err(e) = dump_f32_le(&io_path, &[lnf_g, lnf_b, x0]) {
        eprintln!("model: torch peer: cannot write io blob: {e}");
        return None;
    }
    let out_path = dir.join(format!("model_torch_out_s{}.bin", cfg.s));
    let script_path = dir.join(format!("model_torch_s{}.py", cfg.s));
    std::fs::write(&script_path, torch_script(cfg, &w_path, &io_path, &out_path)).ok()?;
    let run = Command::new(&ctx.py).arg(&script_path).output().ok()?;
    let stdout = String::from_utf8_lossy(&run.stdout);
    if !run.status.success() || !stdout.contains("TORCH_OK") {
        eprintln!(
            "model: torch peer script failed:\n{}",
            String::from_utf8_lossy(&run.stderr)
        );
        return None;
    }
    let grab = |tag: &str| -> Option<(f64, f64)> {
        for l in stdout.lines() {
            let mut it = l.split_whitespace();
            if it.next() == Some(tag) {
                let mn: f64 = it.next()?.parse().ok()?;
                let md: f64 = it.next()?.parse().ok()?;
                return Some((mn * 1e6, md * 1e6)); // ms -> ns
            }
        }
        None
    };
    let manual_vs_sdpa = stdout
        .lines()
        .find_map(|l| l.strip_prefix("TORCH_MANUAL_VS_SDPA ")?.trim().parse().ok());
    let bytes = std::fs::read(&out_path).ok()?;
    let out: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    if out.len() != cfg.s * cfg.d {
        eprintln!(
            "model: torch peer: output size mismatch ({} vs {})",
            out.len(),
            cfg.s * cfg.d
        );
        return None;
    }
    Some(TorchMeasure {
        sdpa_1t: grab("TORCH1"),
        manual_1t: grab("TORCH1M"),
        sdpa_nt: grab("TORCHN"),
        manual_vs_sdpa,
        out,
    })
}

// -------------------------------------------------------------------------------------------
// The benchmark
// -------------------------------------------------------------------------------------------

pub(crate) fn bench_model(cc: &str, dir: &Path) {
    println!("=== model: end-to-end GPT-2-class transformer stack (CPU inference) ===");
    println!(
        "  {LAYERS} pre-LN decoder blocks (MH causal attention + GELU MLP) + final LayerNorm; \
         d_model=768, heads=12, d_ff=3072."
    );
    println!(
        "  The layer loop lives in this harness for BOTH languages: one compiled block fn is \
         called {LAYERS}x per forward"
    );
    println!(
        "  with per-layer weight pointers; scratch is host-allocated and shared. C(gcc) = \
         -O3 -march=native -ffp-contract=fast"
    );
    println!(
        "  (the suite's standard basis); C(fast) = same source with -ffast-math (the llama2.c \
         -Ofast basis). Lower ms is better;"
    );
    println!("  absolute numbers are thermal-bound — the Mercury/C ratio is the stable metric.\n");

    let torch = detect_torch();
    match &torch {
        Some(t) => println!(
            "  PyTorch peer: torch {} — CPU EAGER float32 under torch.inference_mode() (NOT \
             torch.compile), {} threads available.\n  Same weights/inputs via little-endian f32 \
             blobs; T1 = torch.set_num_threads(1), Tn = default all threads;\n  sdpa = \
             F.scaled_dot_product_attention (fused industry path), man = manual matmul+softmax \
             attention.\n",
            t.version, t.threads
        ),
        None => println!(
            "  PyTorch peer: python + torch not importable on PATH — torch columns skipped.\n"
        ),
    }

    interp_gate();

    for s in [128usize, 512] {
        bench_model_size(
            cc,
            dir,
            Cfg {
                s,
                d: 768,
                h: 12,
                dff: 3072,
            },
            torch.as_ref(),
        );
        println!();
    }
}

/// The differential gate: the interpreter oracle runs the *identical* 12-layer forward (same MIR,
/// same weights, same harness layer loop) at a reduced config and must agree with the Cranelift
/// JIT bit-for-bit — the recognized kernels are marshalled identically by both backends, and the
/// glue (extractions, mask, scatter) executes the same vectorized MIR.
fn interp_gate() {
    let cfg = Cfg {
        s: 16,
        d: 64,
        h: 4,
        dff: 256,
    };
    let mut interner = Interner::new();
    let block_src = mer_block(cfg, false);
    let ln_src = mer_final_ln(cfg);
    let (Some(block_prog), Some(ln_prog)) = (
        build_program(&block_src, &mut interner),
        build_program(&ln_src, &mut interner),
    ) else {
        println!("  ! interp gate: reduced-config model failed to compile — skipping gate\n");
        return;
    };
    let entry = interner.intern("kbench");

    let mut seed = 0x5EED_CAFE_u64;
    let mut weights = make_weights(cfg, &mut seed);
    let sd = cfg.s * cfg.d;
    let x0 = fill(sd, 0.25, 0.0, &mut seed);
    let lnf_g = fill(cfg.d, 0.1, 1.0, &mut seed);
    let lnf_b = fill(cfg.d, 0.02, 0.0, &mut seed);

    // --- interpreter forward ---
    let mut sc = Scratch::new(cfg);
    let (mut xa, mut xb) = (vec![0.0f32; sd], vec![0.0f32; sd]);
    let mut y_interp = vec![0.0f32; sd];
    xa.copy_from_slice(&x0);
    let mut ok = true;
    for l in 0..LAYERS {
        let w = &mut weights[l];
        let (xi, xo) = if l % 2 == 0 {
            (&mut xa, &mut xb)
        } else {
            (&mut xb, &mut xa)
        };
        let mut bufs: [&mut [f32]; 24] = [
            xi,
            &mut w.ln1g,
            &mut w.ln1b,
            &mut w.wq,
            &mut w.wk,
            &mut w.wv,
            &mut w.wo,
            &mut w.ln2g,
            &mut w.ln2b,
            &mut w.w1,
            &mut w.w2,
            &mut sc.nrm,
            &mut sc.q,
            &mut sc.k,
            &mut sc.v,
            &mut sc.qh,
            &mut sc.kh,
            &mut sc.vt,
            &mut sc.scores,
            &mut sc.ah,
            &mut sc.attn,
            &mut sc.a,
            &mut sc.ff1,
            xo,
        ];
        if let Err(e) = mercury_interp::run_kernel_f32(&block_prog, entry, &mut bufs, &interner) {
            println!("  ! interp gate: interpreter error at layer {l}: {e}");
            ok = false;
            break;
        }
    }
    if ok {
        let (mut g, mut b) = (lnf_g.clone(), lnf_b.clone());
        let mut bufs: [&mut [f32]; 4] = [&mut xa, &mut g, &mut b, &mut y_interp];
        if let Err(e) = mercury_interp::run_kernel_f32(&ln_prog, entry, &mut bufs, &interner) {
            println!("  ! interp gate: interpreter error in final LN: {e}");
            ok = false;
        }
    }
    if !ok {
        return;
    }

    // --- native (Cranelift JIT) forward over the identical MIR + inputs ---
    let (Ok(block_jit), Ok(ln_jit)) = (
        mercury_codegen_cranelift::jit_module(&block_prog, &interner),
        mercury_codegen_cranelift::jit_module(&ln_prog, &interner),
    ) else {
        println!("  ! interp gate: JIT failed — skipping gate\n");
        return;
    };
    let (Some(bp), Some(lp)) = (block_jit.func_ptr(entry), ln_jit.func_ptr(entry)) else {
        println!("  ! interp gate: kbench symbol missing — skipping gate\n");
        return;
    };
    let block_fn: BlockFn = unsafe { std::mem::transmute(bp) };
    let ln_fn: LnFn = unsafe { std::mem::transmute(lp) };
    let mut sc2 = Scratch::new(cfg);
    let (mut xa2, mut xb2) = (vec![0.0f32; sd], vec![0.0f32; sd]);
    let mut y_native = vec![0.0f32; sd];
    unsafe {
        run_forward(
            block_fn, ln_fn, &weights, &mut sc2, &x0, &mut xa2, &mut xb2, &lnf_g, &lnf_b,
            &mut y_native,
        );
    }

    let (rel, at) = max_rel_err(&y_interp, &y_native);
    if rel == 0.0 {
        println!(
            "  interp gate (S={} D={} H={} Dff={}, {LAYERS} layers): interpreter == native \
             BIT-EXACT over all {sd} outputs\n",
            cfg.s, cfg.d, cfg.h, cfg.dff
        );
    } else {
        println!(
            "  ! interp gate: interpreter vs native max rel err {rel:.2e} at [{at}] \
             (interp={} native={})\n",
            y_interp[at], y_native[at]
        );
    }
}

fn bench_model_size(cc: &str, dir: &Path, cfg: Cfg, torch: Option<&TorchCtx>) {
    let flops = cfg.flops_per_layer() * LAYERS as f64;
    println!(
        "--- model S={} ({} tokens/forward, {:.1} GFLOP/forward) ---",
        cfg.s,
        cfg.s,
        flops / 1e9
    );

    // Shared inputs: every column reads the same weight/input buffers.
    let mut seed = 0x0D15EA5E_u64;
    let weights = make_weights(cfg, &mut seed);
    let sd = cfg.s * cfg.d;
    let x0 = fill(sd, 0.25, 0.0, &mut seed);
    let lnf_g = fill(cfg.d, 0.1, 1.0, &mut seed);
    let lnf_b = fill(cfg.d, 0.02, 0.0, &mut seed);
    let mut sc = Scratch::new(cfg);
    let (mut xa, mut xb) = (vec![0.0f32; sd], vec![0.0f32; sd]);
    let mut y = vec![0.0f32; sd];

    // --- Mercury (serial), through the real pipeline ---
    let mut interner = Interner::new();
    let mer = compile_mercury(&mer_block(cfg, false), &mut interner);
    let lnf_mod = compile_mercury(&mer_final_ln(cfg), &mut interner);
    let mer_m = match (&mer, &lnf_mod) {
        (Some(m), Some(l)) => {
            // Mechanism transparency: print exactly which recognized kernels the block dispatches.
            let calls = kernel_calls(&m.mir);
            let summary = calls
                .iter()
                .map(|(k, v)| format!("{v}x {}", k.trim_start_matches("mercury_")))
                .collect::<Vec<_>>()
                .join(", ");
            println!("  block dispatches (per layer, from optimized MIR): {summary}");
            for need in ["mercury_sgemm_nt", "mercury_sgemm_nt_epi", "mercury_norm_affine_f32"] {
                if !calls.iter().any(|(k, _)| k == need) {
                    println!("  ! WARNING: expected recognized kernel {need} did NOT dispatch — timing scalar loops");
                }
            }
            let (Some(bp), Some(lp)) = (m.func(&mut interner, "kbench"), l.func(&mut interner, "kbench"))
            else {
                println!("  ! mercury kbench symbol missing");
                return;
            };
            let block_fn: BlockFn = unsafe { std::mem::transmute(bp) };
            let ln_fn: LnFn = unsafe { std::mem::transmute(lp) };
            let mut run = || unsafe {
                run_forward(
                    block_fn, ln_fn, &weights, &mut sc, &x0, &mut xa, &mut xb, &lnf_g, &lnf_b,
                    &mut y,
                )
            };
            let ns = time_forward(&mut run);
            Some(MeasureModel {
                compile: m.compile + l.compile,
                ns_per_fwd: ns,
                out: y.clone(),
            })
        }
        _ => None,
    };

    // --- C (gcc, standard suite flags). A single naive-dot forward is tens of seconds at S=512
    // (the serial-FMA-chain regime), so like bench_matmul's naive-at-2048 rule it is skipped
    // there by default (XBENCH_MODEL_NAIVE forces it); correctness at S=512 is checked vs C(fast).
    // XBENCH_MODEL_MER_ONLY skips every external peer (C, C(fast), PyTorch) so a dev iterating on the
    // multicore kernels gets Mer(1c) vs Mer(par) + scaling in seconds instead of the ~2 min the peer
    // compilation + torch warmups cost. Not for reported numbers — the peers are the honesty bar.
    let mer_only = std::env::var("XBENCH_MODEL_MER_ONLY").is_ok();
    let run_c_slow = !mer_only && (cfg.s <= 128 || std::env::var("XBENCH_MODEL_NAIVE").is_ok());
    let c_src = c_model(cfg);
    let c_m = if run_c_slow {
        compile_c_model(
            &c_src,
            dir,
            &format!("model_s{}", cfg.s),
            cc,
            &["-O3", "-march=native", "-ffp-contract=fast", "-shared"],
        )
        .map(|cm| {
            let mut run = || unsafe {
                run_forward(
                    cm.block, cm.lnf, &weights, &mut sc, &x0, &mut xa, &mut xb, &lnf_g, &lnf_b,
                    &mut y,
                )
            };
            let ns = time_forward(&mut run);
            MeasureModel {
                compile: cm.compile,
                ns_per_fwd: ns,
                out: y.clone(),
            }
        })
    } else {
        println!(
            "  -> C(gcc) omitted at S={} (naive-dot forward is tens of seconds per call; its loss \
             is already shown at S=128). Set XBENCH_MODEL_NAIVE to force it.",
            cfg.s
        );
        None
    };

    // --- C(fast): identical source, -ffast-math ---
    let cfast_m = if mer_only { None } else { compile_c_model(
        &c_src,
        dir,
        &format!("model_s{}_fast", cfg.s),
        cc,
        &["-O3", "-march=native", "-ffast-math", "-shared"],
    )
    .map(|cm| {
        let mut run = || unsafe {
            run_forward(
                cm.block, cm.lnf, &weights, &mut sc, &x0, &mut xa, &mut xb, &lnf_g, &lnf_b,
                &mut y,
            )
        };
        let ns = time_forward(&mut run);
        MeasureModel {
            compile: cm.compile,
            ns_per_fwd: ns,
            out: y.clone(),
        }
    }) };

    // --- Mercury @parallel, measured LAST so its all-core heat pollutes no single-core column ---
    let mer_par_m = compile_mercury(&mer_block(cfg, true), &mut interner).and_then(|m| {
        // Same mechanism transparency as the serial column: print the @parallel dispatch set and
        // warn if the embedded GEMMs regressed to serial (they would still be *correct*, so only
        // this scan would catch it).
        let calls = kernel_calls(&m.mir);
        let summary = calls
            .iter()
            .map(|(k, v)| format!("{v}x {}", k.trim_start_matches("mercury_")))
            .collect::<Vec<_>>()
            .join(", ");
        println!("  @parallel block dispatches (per layer, from optimized MIR): {summary}");
        if !calls.iter().any(|(k, _)| k == "mercury_sgemm_nt_parallel") {
            println!(
                "  ! WARNING: @parallel block did NOT dispatch mercury_sgemm_nt_parallel — \
                 embedded GEMMs are running serial"
            );
        }
        let lnf = lnf_mod.as_ref()?;
        let (Some(bp), Some(lp)) = (
            m.func(&mut interner, "kbench"),
            lnf.func(&mut interner, "kbench"),
        ) else {
            return None;
        };
        let block_fn: BlockFn = unsafe { std::mem::transmute(bp) };
        let ln_fn: LnFn = unsafe { std::mem::transmute(lp) };
        let mut run = || unsafe {
            run_forward(
                block_fn, ln_fn, &weights, &mut sc, &x0, &mut xa, &mut xb, &lnf_g, &lnf_b, &mut y,
            )
        };
        let ns = time_forward(&mut run);
        Some(MeasureModel {
            compile: m.compile,
            ns_per_fwd: ns,
            out: y.clone(),
        })
    });

    // --- PyTorch CPU eager peer, timed immediately after the Mercury columns in the SAME bench
    // invocation (same-run adjacency). It runs in its own process over the just-dumped identical
    // buffers; inside the script the single-thread variants run first and the all-core variant
    // last, so multicore heat pollutes no single-thread torch number. The blob writes + torch
    // import + its own warmups sit between Mer(par)'s all-core burst and the first timed torch
    // iteration.
    let torch_m = if mer_only { None } else { torch }.and_then(|t| {
        println!(
            "  running PyTorch peer (torch {}, eager f32, inference_mode; warmup 3 + timed 10 \
             per variant)...",
            t.version
        );
        bench_torch(t, dir, cfg, &weights, &x0, &lnf_g, &lnf_b)
    });
    // Same-run Mer(1c) vs Mer(par) scaling — the reliable multicore instrument on this hybrid laptop
    // (both measured adjacently, same thermal state). Printed always; it's the number to move.
    if let (Some(s), Some(p)) = (&mer_m, &mer_par_m) {
        println!(
            "  Mer scaling: 1c {:.1} ms -> par {:.1} ms = {:.2}x",
            s.ns_per_fwd / 1e6,
            p.ns_per_fwd / 1e6,
            s.ns_per_fwd / p.ns_per_fwd
        );
    }

    // --- report ---
    // Torch columns are eager PyTorch (no compile step): T1(sdpa)/Tn(sdpa) = fused
    // F.scaled_dot_product_attention at 1/all threads, T1(man) = manual matmul+softmax at 1
    // thread. Column value is the min of the timed iterations (the suite's best-observed
    // convention); medians are printed below the table.
    let mk_torch = |v: Option<(f64, f64)>, out: &[f32]| {
        v.map(|(mn, _)| MeasureModel {
            compile: Duration::ZERO, // sentinel: eager, no compile step — printed as "eager"
            ns_per_fwd: mn,
            out: out.to_vec(),
        })
    };
    let (torch1_m, torchman_m, torchn_m) = match &torch_m {
        Some(t) => (
            mk_torch(t.sdpa_1t, &t.out),
            mk_torch(t.manual_1t, &[]),
            mk_torch(t.sdpa_nt, &[]),
        ),
        None => (None, None, None),
    };

    let cols: [(&str, &Option<MeasureModel>); 7] = [
        ("Mer(1c)", &mer_m),
        ("Mer(par)", &mer_par_m),
        ("C(gcc)", &c_m),
        ("C(fast)", &cfast_m),
        ("T1(sdpa)", &torch1_m),
        ("T1(man)", &torchman_m),
        ("Tn(sdpa)", &torchn_m),
    ];
    let row = |label: &str, f: &dyn Fn(&MeasureModel) -> String| {
        print!("  {:<22}", label);
        for (_, m) in &cols {
            print!(
                " {:>10}",
                m.as_ref().map(|x| f(x)).unwrap_or_else(|| "n/a".into())
            );
        }
        println!();
    };
    print!("  {:<22}", "");
    for (name, _) in &cols {
        print!(" {name:>10}");
    }
    println!();
    row("ms/forward", &|x| format!("{:.1}", x.ns_per_fwd / 1e6));
    row("ms/layer (incl. ln_f)", &|x| {
        format!("{:.2}", x.ns_per_fwd / 1e6 / LAYERS as f64)
    });
    row("tokens/sec", &|x| {
        format!("{:.0}", cfg.s as f64 / (x.ns_per_fwd / 1e9))
    });
    row("GFLOP/s (context)", &|x| format!("{:.1}", flops / x.ns_per_fwd));
    row("compile ms", &|x| {
        if x.compile == Duration::ZERO {
            "eager".into()
        } else {
            format!("{:.0}", x.compile.as_secs_f64() * 1e3)
        }
    });
    if let Some(t) = &torch_m {
        let fmt = |v: &Option<(f64, f64)>| {
            v.map(|(mn, md)| format!("min {:.1} / med {:.1} ms", mn / 1e6, md / 1e6))
                .unwrap_or_else(|| "n/a".into())
        };
        println!(
            "  torch detail: sdpa-1t {}; manual-1t {}; sdpa-all {}",
            fmt(&t.sdpa_1t),
            fmt(&t.manual_1t),
            fmt(&t.sdpa_nt)
        );
        if let Some(r) = t.manual_vs_sdpa {
            println!("  torch-internal manual-attn vs SDPA: max|Δ|/max|out| = {r:.2e}");
        }
    }

    let ratio_line = |mer: &Option<MeasureModel>, peer: &Option<MeasureModel>, who: &str, peer_name: &str| {
        if let (Some(m), Some(p)) = (mer, peer) {
            let r = p.ns_per_fwd / m.ns_per_fwd;
            println!(
                "  -> Mercury {who} is {:.2}x {} than {peer_name}",
                if r >= 1.0 { r } else { 1.0 / r },
                if r >= 1.0 { "faster" } else { "slower" }
            );
        }
    };
    ratio_line(&mer_m, &c_m, "(1 core)", "C (gcc -O3 -march=native)");
    ratio_line(&mer_m, &cfast_m, "(1 core)", "C(fast) (gcc -ffast-math)");
    ratio_line(&mer_par_m, &cfast_m, "@parallel", "C(fast) (single-threaded)");
    ratio_line(
        &mer_m,
        &torch1_m,
        "(1 core)",
        "PyTorch eager SDPA (1 thread)",
    );
    ratio_line(
        &mer_m,
        &torchman_m,
        "(1 core)",
        "PyTorch eager manual-attn (1 thread)",
    );
    ratio_line(
        &mer_m,
        &torchn_m,
        "(1 core)",
        "PyTorch eager SDPA (all threads)",
    );
    ratio_line(
        &mer_par_m,
        &torchn_m,
        "@parallel",
        "PyTorch eager SDPA (all threads)",
    );
    if mer_par_m.is_some() {
        println!(
            "     (thread disclosure: both C columns and the T1 torch columns are \
             single-threaded; Mer(par) and Tn(sdpa) are the two multicore columns — the \
             @parallel dispatch set is printed above)"
        );
    }

    // --- full-buffer cross-checks over the final [S, D] output ---
    // Mercury-vs-C: the GEMM/norm reductions reassociate and Mercury's ~1-ULP poly exp/tanh differ
    // from libm, compounding over 12 layers — and the final LayerNorm re-centers every row to mean
    // 0, so a *per-element* relative error is meaningless on the near-zero elements (catastrophic-
    // cancellation zeros). Use the suite's **magnitude-normalized** metric `max|Δ| / max|out|`
    // (the softmax_bwd / cumsum convention) at the usual 1e-3.
    let check = |a: &Option<MeasureModel>, b: &Option<MeasureModel>, who: &str, tol: f64| {
        if let (Some(x), Some(z)) = (a, b) {
            let maxabs = z.out.iter().fold(0.0f32, |m, &v| m.max(v.abs())).max(1e-6);
            let maxerr = x
                .out
                .iter()
                .zip(&z.out)
                .fold(0.0f32, |m, (&p, &q)| m.max((p - q).abs()));
            let rel = (maxerr / maxabs) as f64;
            if rel > tol {
                println!("  ! full-buffer mismatch {who}: max|Δ|/max|out| = {rel:.2e} (tol {tol:.0e})");
            } else {
                println!("  cross-check {who}: max|Δ|/max|out| = {rel:.2e} (tol {tol:.0e}) — PASS");
            }
        }
    };
    check(&mer_m, &c_m, "Mercury vs C", 1e-3);
    check(&mer_m, &cfast_m, "Mercury vs C(fast)", 1e-3);
    // Mercury vs torch: same GELU flavor (tanh approx) and eps, so the residual is the same
    // reassociation + poly-vs-libm class as vs C — expect ~1e-5-ish at the standard 1e-3.
    check(&mer_m, &torch1_m, "Mercury vs Torch (eager SDPA)", 1e-3);
    // Serial vs @parallel Mercury: every dispatched _parallel kernel is bit-identical to its serial
    // twin (fixed chunking / row-mapped) and the outlined glue loops are deterministic, so this one
    // stays the strict per-element check — it is expected EXACT.
    if let (Some(x), Some(z)) = (&mer_m, &mer_par_m) {
        let (rel, at) = max_rel_err(&x.out, &z.out);
        if rel > 0.0 {
            println!(
                "  ! Mercury serial vs @parallel differ at [{at}]: {} vs {} (rel {rel:.2e}) — \
                 expected bit-exact",
                x.out[at], z.out[at]
            );
        } else {
            println!("  cross-check Mercury serial vs @parallel: BIT-EXACT");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Pin the model block's recognized-kernel dispatch sets for BOTH variants. The recognizers
    /// are gate-blind (interp and native call the same symbol), so a silent regression to scalar
    /// loops — or to serial GEMMs under `@parallel` — passes every correctness gate; only this
    /// scan catches it. Uses the interp-gate config (small dims) to keep the test fast.
    #[test]
    fn block_dispatch_sets_pinned() {
        let cfg = Cfg { s: 16, d: 64, h: 4, dff: 256 };
        let mut interner = Interner::new();

        let serial = compile_mercury(&mer_block(cfg, false), &mut interner)
            .expect("serial block must compile");
        let calls = kernel_calls(&serial.mir);
        for need in [
            "mercury_sgemm_nt",
            "mercury_sgemm_nt_alpha",
            "mercury_sgemm_nt_epi",
            "mercury_norm_affine_f32",
            "mercury_norm_f32",
            "mercury_velem_f32",
        ] {
            assert!(
                calls.iter().any(|(k, _)| k == need),
                "serial block lost dispatch {need}; got {calls:?}"
            );
        }

        let par = compile_mercury(&mer_block(cfg, true), &mut interner)
            .expect("@parallel block must compile");
        let pcalls = kernel_calls(&par.mir);
        assert!(
            pcalls.iter().any(|(k, _)| k == "mercury_sgemm_nt_parallel"),
            "@parallel block's embedded GEMMs regressed to serial; got {pcalls:?}"
        );
        // No plain-serial NT GEMM may remain in the @parallel variant.
        assert!(
            !pcalls.iter().any(|(k, _)| k == "mercury_sgemm_nt"),
            "@parallel block still emits serial mercury_sgemm_nt calls; got {pcalls:?}"
        );
    }
}
