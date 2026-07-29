//! `bench_model` — an honest END-TO-END transformer-layer-stack CPU inference benchmark:
//! Wukong vs a strong idiomatic C implementation of the *same* 12-layer GPT-2-class forward.
//!
//! What is measured
//! ----------------
//! One full inference forward over a GPT-2 124M-shaped decoder stack: 12 pre-LayerNorm
//! transformer blocks (multi-head causal attention + GELU MLP, both with residuals) plus the
//! final LayerNorm, at d_model=768, heads=12, d_ff=3072, and seq lengths S=128 / S=512.
//! Reported as ms per forward, ms per layer, and tokens/sec (S tokens per forward). Absolute
//! numbers are clock/thermal-bound on this box — the Wukong-vs-C **ratio** is the stable metric.
//!
//! How the two columns are built (the honesty contract)
//! ----------------------------------------------------
//! * **Wukong** goes through the real pipeline: parse → sema → mir_build → optimize(-O3) →
//!   Cranelift JIT, exactly like every other kernel in this suite (`bench_wukong`). The block is
//!   ordinary Wukong source (adapted from `examples/gpt2.wk` to the full 768/3072 config): the
//!   LayerNorms / GEMM nests / softmax / GELU are spelled in the idiomatic recognized op-forms, so
//!   `mir_build` dispatches them to the tuned runtime kernels (`wukong_norm_affine_f32`,
//!   `wukong_sgemm_nt[_alpha|_epi]`, `wukong_norm_f32`, `wukong_velem_f32`). The optimized MIR
//!   is scanned and the dispatched kernel set is printed, so the mechanism is transparent — and if
//!   a recognizer ever regresses, the bench says so instead of silently timing scalar loops.
//! * **C** is the same computation, hand-written as a competent single-TU implementation
//!   (contiguous loops, per-head slice extraction, its own scaled-QKᵀ / masked-softmax / PV
//!   attention, two-pass LayerNorm, tanh-approx GELU matching Wukong's), compiled at runtime with
//!   `gcc -O3 -march=native -ffp-contract=fast` — the suite's standard basis (no `-ffast-math`, so
//!   its f32 dot reductions stay serial, the same disclosed convention as `bench_linear`/`dot`).
//!   **C(fast)** is the identical source at `-O3 -march=native -ffast-math` (the `llama2.c
//!   -Ofast` basis), which lets gcc reassociate + vectorize the dot products — the strongest
//!   flags-only C column.
//! * **PyTorch (the industry peer)** — when `python` + `torch` import (probed gracefully; a
//!   printed note + skipped columns otherwise), the harness dumps the *exact* weight/input
//!   buffers as little-endian f32 blobs and generates a self-contained PyTorch script that
//!   rebuilds the identical forward: `F.linear` computes `x·Wᵀ` over the same `[out,in]`
//!   row-major weights Wukong/C use (the layouts coincide — no transposition), with the Q/K/V
//!   weights concatenated into HuggingFace's fused `c_attn` `[3D, D]` form (see Known
//!   asymmetries — the peer is deliberately built the strong way), `F.layer_norm`
//!   at the same eps=1e-5, the same tanh-approx GELU via `F.gelu(approximate="tanh")`, and
//!   multi-head causal attention two ways — `F.scaled_dot_product_attention(is_causal=True)`
//!   (the fused industry path) *and* a manual matmul+softmax variant — under
//!   `torch.inference_mode()`, float32. Two regimes are timed: **eager** (`T1`/`Tn`) and
//!   **`torch.compile(..., mode="max-autotune", fullgraph=True)`** (TorchInductor, the `T1(comp)`/
//!   `Tn(comp)` columns) — the strongest fair PyTorch baseline: Inductor fuses the whole 12-layer
//!   forward + final LayerNorm into one graph and autotunes its CPU GEMM/reduction kernels. The
//!   compiled path is real on Windows: the harness locates the newest `vcvars64.bat`, captures the
//!   MSVC build environment (`cmd /c call vcvars64 && set`, parsed + cached), appends the chosen
//!   interpreter's `<base_prefix>\libs` to `LIB` (else Inductor's link step fails `LNK1104` on
//!   `pythonNNN.lib`), and spawns the peer under that env. If no `vcvars64.bat` is found the
//!   compiled columns degrade to `n/a (MSVC not found)` and eager runs exactly as before. Of the
//!   two candidate interpreters (`python` on PATH and `tools/torch-venv/Scripts/python.exe`;
//!   `PYTHON` env still wins) the one with the newest torch is chosen, and which/what-version is
//!   printed. Each timed variant warms then times ≥10 forwards (min + median; the timed loop body
//!   is one bare forward — no per-iteration allocation/IO beyond what torch does inside the
//!   forward), at `torch.set_num_threads(1)` and at the default all-threads. The compile
//!   wall-clock is a cold-start cost, printed separately (the `compile ms` row) and never mixed
//!   into a per-forward number. Torch runs in the same bench invocation immediately after the
//!   Wukong columns (same-run adjacency); both compiled variants are built first (untimed), then
//!   timing runs coolest-first — T1 eager → T1 compiled → Tn eager → Tn compiled — so multicore
//!   heat pollutes no single-thread torch number. Both the eager-SDPA and the compiled outputs are
//!   cross-checked against Wukong's with the suite's magnitude-normalized metric at the same 1e-3
//!   tolerance (the GELU flavor matches exactly, so no loosening is needed), plus a
//!   compiled-vs-eager max|Δ| line — a fast-but-wrong compiled path fails loudly and reports no
//!   time rather than a bogus win. Compile failures fall through a disclosed retry ladder: some
//!   torch builds' Inductor CPP GEMM template (`cpp_CppMicroGemmFP32Vec`) is broken on
//!   Windows/MSVC, so on a lowering failure the GEMM autotune backend is pinned to ATEN (= MKL,
//!   torch's strongest CPU GEMM — the strongest WORKING config, not a handicap) and fullgraph
//!   retried; a remaining `fullgraph=True` break falls back to `fullgraph=False` with the break
//!   count from `torch._dynamo.explain` disclosed.
//! * **The layer loop lives in this harness** for both languages: one JIT'd/compiled block
//!   function is called 12× per forward with per-layer weight pointers (ping-ponging two
//!   activation buffers), then the final LayerNorm — the way a real runtime drives a layer stack.
//!   Whole-layer scratch (nrm/q/k/v/attn/a/ff1) is host-allocated and passed as pointers to both
//!   languages. The *per-head* scratch (qh/kh/vt/scores/ah) differs by design: the Wukong source
//!   declares it inside the head loop — the natural spelling for independent iterations, which is
//!   what lets `@parallel` run heads concurrently (each iteration's scratch is private stack) —
//!   while C keeps the caller-scratch convention. Both spellings do the same per-head fills and
//!   compute; the Wukong one also zero-initializes its locals each head (~1% of the forward's
//!   work at these shapes — a cost Wukong pays, not hides). The Wukong calls run on a 64 MiB
//!   worker thread (spawned outside the timed region) because that loop-local scratch is
//!   ~1.5 MiB of stack frame at S=512.
//! * **Correctness**: the Wukong and C final outputs are cross-checked elementwise over the whole
//!   `[S, 768]` buffer with a magnitude-normalized relative tolerance (float reassociation and
//!   poly-vs-libm transcendentals differ, compounding over 12 layers). Additionally the
//!   interpreter oracle runs the identical 12-layer forward at a reduced config and must match the
//!   JIT bit-for-bit (the differential gate that keeps the recognizer dispatches honest).
//!
//! Known asymmetries (disclosed, not hidden)
//! -----------------------------------------
//! * Wukong's GEMMs go to the tuned AVX2/FMA microkernel; C's stay whatever gcc makes of the
//!   idiomatic nests. That *is* the product claim being measured (a shape-safe tensor language
//!   whose compiler lowers to tuned kernels), the same basis as `bench_matmul`/`bench_linear`.
//! * **The torch peer's Q/K/V projection is FUSED and Wukong's is not** — an asymmetry in the
//!   PEER's favour, on purpose. HuggingFace's `GPT2Attention` issues one `Conv1D` `c_attn` of
//!   shape `[D, 3D]` (a single `[S,D]x[D,3D]` GEMM), so the peer does too; the Wukong and C
//!   columns issue three separate `[S,D]x[D,D]` projections. A benchmark must run against the
//!   strongest honest peer, not the convenient weak one, and QKV fusion is a known open Wukong
//!   lever — so this ratio charges Wukong for not having it yet. The fusion is bit-exact with the
//!   three separate `F.linear` calls (same weights, same dot products), so the cross-check
//!   tolerance is unaffected. The disclosure is printed beside the ratio lines, not only here.
//! * The `@parallel` Wukong column is fully multicore, in two tiers: the whole-`[S,D]` ops
//!   outside the head loop (LayerNorms, Q/K/V/WO/down-proj GEMMs, fused-GELU FFN, residual adds)
//!   dispatch `_parallel` kernels (each bit-identical to its serial twin), and the per-head
//!   attention loop outlines into ONE `wukong_parallel_for` region — heads across cores, each
//!   head running the identical SERIAL kernel sequence (QKᵀ·α GEMM, batched softmax, PV GEMM) the
//!   serial column runs, with its loop-local scratch private per head. Head slices are disjoint
//!   (`attn[i*D + hh*HD + j]`), so serial == `@parallel` stays bit-exact. The run-time dispatch
//!   scan prints the `@parallel` variant's kernel set (and a unit test pins both the region and
//!   the per-function serial/parallel split), so a regression back to a serial head chain is
//!   visible, not silent. Both C columns are single-threaded idiomatic code, as everywhere in
//!   this suite; the torch `Tn(sdpa)` and `Tn(comp)` columns are PyTorch's own all-thread paths
//!   (eager and TorchInductor-compiled) — the other multicore columns, disclosed as such.

use std::cell::Cell;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::OnceLock;
use std::time::{Duration, Instant};

use wukong_span::{Interner, SourceId};

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
/// buffers, and the output activations. Wukong array params pass by base pointer, so the JIT'd
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

/// The Wukong block ABI: like [`BlockFn`] but WITHOUT the five per-head scratch pointers
/// (qh/kh/vt/scores/ah) — the Wukong source declares that scratch inside the head loop (private
/// per iteration, which is what lets `@parallel` run heads concurrently), so it never crosses the
/// ABI. The C implementation keeps the caller-scratch convention above.
type WukBlockFn = unsafe extern "C" fn(
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

/// One GPT-2 block in Wukong, adapted from `examples/gpt2.wk` to the full config, with every
/// sub-op spelled in its recognized form (verified by the dispatch scan printed at run time):
///  * batched affine LayerNorm  -> `wukong_norm_affine_f32`
///  * `nn.Linear` NT dot nests  -> `wukong_sgemm_nt`
///  * scaled scores `α·(Q·Kᵀ)`  -> `wukong_sgemm_nt_alpha`
///  * batched masked softmax    -> `wukong_norm_f32`
///  * GEMM + `gelu` epilogue    -> `wukong_sgemm_nt_epi`
///  * residual adds             -> `wukong_velem_f32`
/// Attention runs per head over contiguous slices (extract Qh/Kh, V transposed) so each head's
/// scores / PV products are plain NT GEMMs — the same structure the C implementation uses. The
/// head loop is spelled the natural way for independent iterations: its scratch (qh/kh/vt/scores/
/// ah) is declared INSIDE the loop body, private per head. Under `@parallel` the compiler outlines
/// the loop into a `wukong_parallel_for` region — heads across cores, each running the identical
/// serial per-head kernel sequence — instead of a serial chain of small multicore kernels.
/// (A (head, row-tile) div/mod respelling — h·4 units via the outliner's mixed-radix legality —
/// was measured 2026-07-11 in a same-state binary A/B: @parallel was a WASH at S=128 and S=512
/// (243.7 vs 244.7 ms) and the serial twin paid ~6% (1151.7 vs 1088.0 ms) for the per-tile kh/vt
/// re-packing, so the untiled spelling stays; the legality extension remains as compiler
/// infrastructure, exercised by tests/run/parallel_divmod_tiling.wk.)
fn wk_block(cfg: Cfg, parallel: bool) -> String {
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
    // 3. Multi-head causal attention (H = {h}, hd = {hd}). Iterations are independent — each head
    // reads/writes only its own hh-sliced column band and the scratch is loop-body-local (private
    // per head) — so under @parallel the compiler outlines this loop into ONE parallel region
    // (heads across cores, the identical serial kernel sequence inside each).
    for hh in 0..{h} {{
        let mut qh: [f32; {shd}] = [0.0; {shd}];
        let mut kh: [f32; {shd}] = [0.0; {shd}];
        let mut vt: [f32; {shd}] = [0.0; {shd}];
        let mut scores: [f32; {ss}] = [0.0; {ss}];
        let mut ah: [f32; {shd}] = [0.0; {shd}];
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

/// The final LayerNorm (`ln_f` in GPT-2) as its own Wukong module — the batched affine norm form,
/// dispatching one `wukong_norm_affine_f32` call.
fn wk_final_ln(cfg: Cfg) -> String {
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
/// Wukong's constants. Exported as `kbench` (block) and `kfinal` (final LayerNorm).
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
  /* 6. MLP up-projection + tanh-approx GELU (Wukong's gelu constants) */
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

struct WukModule {
    handle: wukong_codegen_cranelift::JitModuleHandle,
    compile: Duration,
    /// Optimized-MIR text, for the recognized-kernel dispatch scan.
    mir: String,
}

impl WukModule {
    fn func(&self, interner: &mut Interner, name: &str) -> Option<*const u8> {
        let sym = interner.intern(name);
        self.handle.func_ptr(sym)
    }
}

/// The real pipeline, same as `bench_wukong`: parse → sema → mir_build → optimize(-O3) → JIT.
fn compile_wukong(src: &str, interner: &mut Interner) -> Option<WukModule> {
    let t = Instant::now();
    let (module, pd) = wukong_parser::parse_module(src, SourceId(0), interner);
    if pd.iter().any(|d| d.is_error()) {
        eprintln!("model: wukong parse error");
        return None;
    }
    let (sema, sd) = wukong_sema::check(&module, interner);
    if sd.iter().any(|d| d.is_error()) {
        eprintln!("model: wukong sema error: {sd:?}");
        return None;
    }
    let (mut program, ld) = wukong_mir_build::lower_program(&module, &sema, interner);
    if ld.iter().any(|d| d.is_error()) {
        eprintln!("model: wukong lower error: {ld:?}");
        return None;
    }
    wukong_opt::optimize(&mut program, 3);
    let mir = wukong_mir::print::print_program(&program, interner);
    let handle = match wukong_codegen_cranelift::jit_module(&program, interner) {
        Ok(h) => h,
        Err(e) => {
            eprintln!("model: wukong codegen error: {e}");
            return None;
        }
    };
    let compile = t.elapsed();
    Some(WukModule {
        handle,
        compile,
        mir,
    })
}

/// Compile + optimize only (no JIT) — the front half of the pipeline, for the interpreter oracle.
fn build_program(src: &str, interner: &mut Interner) -> Option<wukong_mir::Program> {
    let (module, pd) = wukong_parser::parse_module(src, SourceId(0), interner);
    if pd.iter().any(|d| d.is_error()) {
        return None;
    }
    let (sema, sd) = wukong_sema::check(&module, interner);
    if sd.iter().any(|d| d.is_error()) {
        return None;
    }
    let (mut program, ld) = wukong_mir_build::lower_program(&module, &sema, interner);
    if ld.iter().any(|d| d.is_error()) {
        return None;
    }
    wukong_opt::optimize(&mut program, 3);
    Some(program)
}

/// Scan optimized MIR for `wukong_*` runtime-kernel calls: the single-source-of-truth check that
/// the block really dispatches to the recognized kernels (recognizers are gate-blind — both
/// backends call the same symbol — so the bench *prints* what it runs).
fn kernel_calls(mir: &str) -> Vec<(String, usize)> {
    let mut counts: std::collections::BTreeMap<&str, usize> = std::collections::BTreeMap::new();
    for (i, _) in mir.match_indices("wukong_") {
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

/// The suite's magnitude-normalized full-buffer metric `max|Δ| / max|out|`, or `Err(reason)` when
/// the two buffers **cannot be compared at all**.
///
/// The error arms are not pedantry — they are the difference between a cross-check and a rubber
/// stamp. `zip` truncates to the shorter buffer, so folding a full Wukong output against an EMPTY
/// peer output yields `max|Δ| = 0`, `max|out|` falls back to the `1e-6` floor, and the caller
/// prints `PASS` having compared zero elements — next to a fully-reported ms/forward column.
/// `f32::max` likewise *discards* NaN, so an all-NaN peer output also folds to 0 and passes. A
/// comparison that inspected nothing must be reported as not-run, never as agreement.
fn cross_check_rel(ours: &[f32], peer: &[f32]) -> Result<f64, String> {
    if peer.is_empty() {
        return Err(format!(
            "peer produced no output (0 of {} elements) — nothing to compare",
            ours.len()
        ));
    }
    if ours.len() != peer.len() {
        return Err(format!(
            "buffer lengths differ ({} ours vs {} peer) — a peer produced partial output",
            ours.len(),
            peer.len()
        ));
    }
    let nonfinite = ours.iter().chain(peer).filter(|v| !v.is_finite()).count();
    if nonfinite != 0 {
        return Err(format!(
            "{nonfinite} non-finite element(s) across the two buffers — max() discards NaN, so a \
             relative error would be meaningless"
        ));
    }
    let maxabs = peer.iter().fold(0.0f32, |m, &v| m.max(v.abs())).max(1e-6);
    let maxerr = ours
        .iter()
        .zip(peer)
        .fold(0.0f32, |m, (&p, &q)| m.max((p - q).abs()));
    Ok((maxerr / maxabs) as f64)
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
/// (ping-ponging the two activation buffers), then the final LayerNorm into `y`. This is the
/// C-ABI driver ([`BlockFn`], caller-provided per-head scratch); the Wukong columns use the
/// otherwise-identical [`run_forward_mer`].
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

/// [`run_forward`] for the Wukong ABI ([`WukBlockFn`]): identical layer loop, minus the five
/// per-head scratch pointers the Wukong block now owns as loop-body-locals.
#[allow(clippy::too_many_arguments)]
unsafe fn run_forward_mer(
    block: WukBlockFn,
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
            sc.attn.as_mut_ptr(),
            sc.a.as_mut_ptr(),
            sc.ff1.as_mut_ptr(),
            xo,
        );
    }
    // LAYERS is even, so the last block wrote xa.
    lnf(xa.as_ptr(), lnf_g.as_ptr(), lnf_b.as_ptr(), y.as_mut_ptr());
}

/// Run `f` on a worker thread with a large stack, re-raising any panic — the Wukong block's
/// per-head scratch is loop-body-local (stack allocas: ~1.5 MiB at S=512, probed by Cranelift's
/// inline stack probes), which does not fit the host main thread's default ~1 MiB Windows stack.
/// The serial column runs the whole frame on this one thread; the `@parallel` column's region
/// bodies run on the runtime pool's own 16 MiB-stack workers. The spawn sits OUTSIDE the timed
/// region (one spawn per measured column, not per iteration), so the instrument is unchanged.
fn on_big_stack<T: Send>(f: impl FnOnce() -> T + Send) -> T {
    std::thread::scope(|s| {
        let handle = std::thread::Builder::new()
            .name("xbench-model".into())
            .stack_size(64 * 1024 * 1024)
            .spawn_scoped(s, f)
            .expect("spawn model worker thread");
        match handle.join() {
            Ok(v) => v,
            Err(payload) => std::panic::resume_unwind(payload),
        }
    })
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

/// The probed PyTorch environment: interpreter path, version/thread info, the interpreter's
/// `sys.base_prefix` (whose `\libs` is appended to `LIB` so Inductor can link `pythonNNN.lib`),
/// and whether the (config-invariant) weight blob has already been dumped this run.
struct TorchCtx {
    py: String,
    version: String,
    /// Numeric version tuple (e.g. `[2, 12, 1]`) used to pick the newest torch across candidates.
    version_key: Vec<u32>,
    threads: usize,
    base_prefix: String,
    weights_written: Cell<bool>,
}

/// Probe one interpreter for an importable torch, returning its version / thread count / base
/// prefix. Any failure (no such interpreter, torch not installed) → `None`.
fn probe_python(py: &str) -> Option<TorchCtx> {
    let out = Command::new(py)
        .args([
            "-c",
            "import torch,sys; print(torch.__version__); \
             print(torch.get_num_threads()); print(sys.base_prefix)",
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
    let base_prefix = lines.next()?.trim().to_string();
    let version_key = version
        .split('+')
        .next()
        .unwrap_or(&version)
        .split('.')
        .map(|p| p.parse().unwrap_or(0))
        .collect();
    Some(TorchCtx {
        py: py.to_string(),
        version,
        version_key,
        threads,
        base_prefix,
        weights_written: Cell::new(false),
    })
}

/// Probe the candidate interpreters for an importable torch and pick the one with the NEWEST
/// version (a weak/old peer is forbidden — it is the honesty bar for the "beat PyTorch" claim).
/// `PYTHON` (if set) wins outright; otherwise `python` on PATH and the repo's
/// `tools/torch-venv/Scripts/python.exe` are both probed. Graceful: no importable torch anywhere
/// returns `None`; the bench prints a note and the torch columns show `n/a`.
fn detect_torch() -> Option<TorchCtx> {
    let candidates: Vec<String> = match std::env::var("PYTHON") {
        Ok(p) => vec![p],
        Err(_) => vec![
            "python".to_string(),
            "tools/torch-venv/Scripts/python.exe".to_string(),
        ],
    };
    candidates
        .iter()
        .filter_map(|py| probe_python(py))
        .max_by(|a, b| a.version_key.cmp(&b.version_key))
}

// -------------------------------------------------------------------------------------------
// MSVC environment bootstrap — TorchInductor's CPU backend shells out to `cl`/`link`, which only
// work under a vcvars64 environment. We locate the newest vcvars64.bat, run it in a throwaway cmd,
// snapshot the resulting environment, and spawn the peer python under it (plus the interpreter's
// `\libs` appended to LIB so the Inductor link step can find `pythonNNN.lib`). Cached once per run.
// -------------------------------------------------------------------------------------------

/// A parsed MSVC build environment: the full `KEY=VALUE` set emitted by `vcvars64.bat && set`.
type VcEnv = Vec<(String, String)>;

/// Parse an integer-dotted version like `14.50.35717` into a comparable tuple.
fn version_tuple(s: &str) -> Vec<u32> {
    s.trim()
        .split('.')
        .map(|p| p.parse().unwrap_or(0))
        .collect()
}

/// Glob for every `vcvars64.bat` under `C:\Program Files*\Microsoft Visual Studio\*\*\VC\...` and
/// return the one whose MSVC tools version (read from the sibling
/// `Microsoft.VCToolsVersion.default.txt`) is highest — the newest compiler, not the newest VS
/// product-folder name (which sorts wrong: "2017" > "18").
fn find_vcvars() -> Option<PathBuf> {
    let mut best: Option<(Vec<u32>, PathBuf)> = None;
    for pf in ["C:\\Program Files", "C:\\Program Files (x86)"] {
        let vs_root = Path::new(pf).join("Microsoft Visual Studio");
        let Ok(products) = std::fs::read_dir(&vs_root) else {
            continue;
        };
        for product in products.flatten() {
            let Ok(editions) = std::fs::read_dir(product.path()) else {
                continue;
            };
            for edition in editions.flatten() {
                let build = edition.path().join("VC").join("Auxiliary").join("Build");
                let vcvars = build.join("vcvars64.bat");
                if !vcvars.is_file() {
                    continue;
                }
                // Prefer the real MSVC tools version as the sort key; fall back to 0 if absent.
                let ver = std::fs::read_to_string(build.join("Microsoft.VCToolsVersion.default.txt"))
                    .map(|s| version_tuple(&s))
                    .unwrap_or_default();
                if best.as_ref().is_none_or(|(bv, _)| ver > *bv) {
                    best = Some((ver, vcvars));
                }
            }
        }
    }
    best.map(|(_, p)| p)
}

/// The cached MSVC build environment (or `None` if no vcvars64.bat is installed / could not be
/// captured). We drive it through a throwaway `.bat` (`call "<vcvars64>" >nul & set`) run under
/// `cmd /c <batfile>` — passing the complex quoted `call ... && set` string straight to `cmd /c`
/// hits cmd's quote-stripping and mangles the path, so the batfile indirection is the robust route.
/// The batfile silences vcvars' banner so only clean `KEY=VALUE` lines from `set` reach stdout.
fn vcvars_env() -> Option<&'static VcEnv> {
    static CACHE: OnceLock<Option<VcEnv>> = OnceLock::new();
    CACHE
        .get_or_init(|| {
            let vcvars = find_vcvars()?;
            let bat = std::env::temp_dir().join("wukong_xbench_vcvars.bat");
            std::fs::write(
                &bat,
                format!(
                    "@echo off\r\ncall \"{}\" >nul 2>&1\r\nset\r\n",
                    vcvars.display()
                ),
            )
            .ok()?;
            let out = Command::new("cmd").arg("/c").arg(&bat).output().ok()?;
            if !out.status.success() {
                return None;
            }
            let text = String::from_utf8_lossy(&out.stdout);
            let env: VcEnv = text
                .lines()
                .filter_map(|l| l.split_once('='))
                .map(|(k, v)| (k.trim_end().to_string(), v.trim_end().to_string()))
                .collect();
            // A valid vcvars environment always defines LIB/INCLUDE; if it did not parse, treat
            // it as unavailable rather than spawning a half-configured child.
            env.iter()
                .any(|(k, _)| k.eq_ignore_ascii_case("LIB"))
                .then_some(env)
        })
        .as_ref()
}

/// Build the peer-python `Command` spawning `script`. When an MSVC environment is available the
/// child runs under it with `<base_prefix>\libs` appended to `LIB` (so Inductor links
/// `pythonNNN.lib`); otherwise it inherits the parent environment (eager still runs; the script's
/// compile phase is disabled up front via `TRY_COMPILE`).
fn peer_command(ctx: &TorchCtx, script: &Path) -> Command {
    let mut cmd = Command::new(&ctx.py);
    cmd.arg(script);
    if let Some(env) = vcvars_env() {
        cmd.env_clear();
        let libs = format!("{}\\libs", ctx.base_prefix);
        let mut lib_set = false;
        for (k, v) in env {
            if k.eq_ignore_ascii_case("LIB") {
                cmd.env(k, format!("{v};{libs}"));
                lib_set = true;
            } else {
                cmd.env(k, v);
            }
        }
        if !lib_set {
            cmd.env("LIB", libs);
        }
    }
    cmd
}

// -------------------------------------------------------------------------------------------
// Power-state disclosure. This machine has THREE regimes, not two, and `tools/measure_gpt2.ps1`
// (the first-party instrument for the same workload) encodes them:
//   * BATTERY          — NON-REPORTABLE: single-core noisy, all-core meaningless (2-4x slow).
//   * AC + CHARGING    — ALL-CORE CAPPED: single core fine, all-core throttled ~25%.
//   * AC + full        — REPORTABLE: an all-core number is a real claim.
// Collapsing the middle state into "power: AC" prints a reportable-looking banner over multicore
// rows that are power-capped. Queried via the Win32 `GetSystemPowerStatus`, same `extern "system"`
// pattern as `pin_worker_to_cpu` in wukong_runtime/src/gemm.rs — `BatteryFlag & 0x08` is the
// charging bit, the Win32 equivalent of the `ChargeRate > 0` test measure_gpt2.ps1 uses.
// -------------------------------------------------------------------------------------------

/// One of the three states above (or `power: unknown`), for the bench headers.
#[cfg(windows)]
pub(crate) fn power_status_line() -> String {
    #[repr(C)]
    struct SystemPowerStatus {
        ac_line_status: u8,
        battery_flag: u8,
        battery_life_percent: u8,
        system_status_flag: u8,
        battery_life_time: u32,
        battery_full_life_time: u32,
    }
    #[link(name = "kernel32")]
    extern "system" {
        fn GetSystemPowerStatus(status: *mut SystemPowerStatus) -> i32;
    }
    let mut st = SystemPowerStatus {
        ac_line_status: 255,
        battery_flag: 255,
        battery_life_percent: 255,
        system_status_flag: 0,
        battery_life_time: 0,
        battery_full_life_time: 0,
    };
    let ok = unsafe { GetSystemPowerStatus(&mut st) };
    if ok == 0 {
        return "power: unknown".to_string();
    }
    // BatteryFlag: 1 high, 2 low, 4 critical, 8 CHARGING, 128 no system battery, 255 unknown.
    let charging = st.battery_flag != 255 && st.battery_flag & 0x08 != 0;
    let no_battery = st.battery_flag != 255 && st.battery_flag & 0x80 != 0;
    let pct = if st.battery_life_percent <= 100 {
        format!("{}%", st.battery_life_percent)
    } else {
        "?%".to_string()
    };
    match (st.ac_line_status, charging, no_battery) {
        (1, true, _) => format!(
            "power: AC+CHARGING ({pct}) — ALL-CORE CAPPED (~25%): single-core numbers are fine, \
             every multicore row below is DIRECTIONAL ONLY"
        ),
        (1, false, true) => "power: AC (desktop, no battery) — REPORTABLE".to_string(),
        (1, false, _) => format!("power: AC+full ({pct}) — REPORTABLE"),
        (0, _, _) => format!(
            "power: BATTERY ({pct}) — NON-REPORTABLE: single-core noisy, all-core meaningless \
             (2-4x slow); not comparable to AC runs"
        ),
        _ => "power: unknown".to_string(),
    }
}
#[cfg(not(windows))]
pub(crate) fn power_status_line() -> String {
    "power: n/a".to_string()
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

/// The self-contained PyTorch peer script for one config. Layout facts it relies on
/// (verified against `wk_block`/`c_model`): `F.linear(x, W)` computes `x·Wᵀ` over a `[out,in]`
/// row-major `W` — exactly the `w[j*K+p]` layout Wukong and C dot against, so the dumped bytes
/// are used as-is; LayerNorm eps is 1e-5 in all three; the GELU is the tanh approximation
/// (√(2/π), 0.044715) — torch's `approximate="tanh"`; SDPA's default scale `1/√hd` equals
/// Wukong's inline `scale` (hd is a power of 4, exact in f32); Wukong's `-1e30` mask and
/// torch's `-inf`/`is_causal` agree after softmax (both underflow to exactly 0).
///
/// Timed regimes: **eager** (`TORCH1`/`TORCH1M`/`TORCHN`) and, when `try_compile` (MSVC present),
/// **`torch.compile(..., mode="max-autotune", fullgraph=True)`** (`TORCH1C`/`TORCHNC`) — the whole
/// 12-layer SDPA forward + final LayerNorm as one Inductor-fused, autotuned graph. Both compiled
/// variants are built first (untimed; compile wall printed separately as `*_COMPILE`), then timing
/// runs coolest-first (T1 eager → T1 compiled → Tn eager → Tn compiled). The compiled output is
/// cross-checked against the eager reference (`TORCH_COMPILED_VS_EAGER_*`); if it diverges the
/// variant is marked `TORCH_COMPILED_BAD_*` and NOT timed. Compile failures fall through a
/// disclosed retry ladder (each tier printing `TORCH_COMPILE_DISCLOSE`): stock max-autotune
/// fullgraph → GEMM autotune backend pinned to ATEN/MKL (works around the Inductor CPP GEMM
/// template being broken on Windows/MSVC) → `fullgraph=False` (with the `torch._dynamo.explain`
/// break count); a total failure prints `TORCH_COMPILE_FAIL` and skips the variant.
fn torch_script(
    cfg: Cfg,
    w_path: &Path,
    io_path: &Path,
    out_path: &Path,
    cout_path: &Path,
    try_compile: bool,
) -> String {
    let (s, d, h, hd, dff) = (cfg.s, cfg.d, cfg.h, cfg.hd(), cfg.dff);
    format!(
        r#"# Auto-generated by wukong_xbench `model` — the PyTorch CPU peer for the 12-layer
# GPT-2-class forward. Reads the exact little-endian f32 weight/input bytes the Wukong and C
# columns use, rebuilds the identical float32 forward, and times it under torch.inference_mode()
# in two regimes: eager, and torch.compile(mode="max-autotune", fullgraph=True) (TorchInductor —
# the strongest fair CPU baseline, the whole forward fused + autotuned).
import sys, time, array, statistics
import torch
import torch.nn.functional as F

S = {s}; D = {d}; H = {h}; HD = {hd}; DFF = {dff}; LAYERS = {layers}
TRY_COMPILE = {try_compile}
W_PATH = r"{w}"
IO_PATH = r"{io}"
OUT_PATH = r"{out}"
COUT_PATH = r"{cout}"
WARMUP = 3
ITERS = 10
CWARMUP = 6          # >=5 steady calls after the first (compiling) call
CTOL = 1e-3          # compiled-vs-eager tolerance; above this the compiled path is not timed
LNSHAPE = (D,)
DEFAULT_THREADS = torch.get_num_threads()

def load(path):
    with open(path, "rb") as f:
        data = f.read()
    return torch.frombuffer(bytearray(data), dtype=torch.float32)

def dump(t, path):
    a = array.array("f", t.reshape(-1).tolist())
    if sys.byteorder != "little":
        a.byteswap()
    with open(path, "wb") as f:
        f.write(a.tobytes())

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
    # HuggingFace GPT2Attention issues the Q/K/V projection as ONE fused `Conv1D` c_attn of shape
    # [D, 3D] — a single [S,D]x[D,3D] GEMM, not three [S,D]x[D,D] ones. The peer must be the
    # STRONGEST honest implementation of this forward, so it is built the reference way. It is the
    # same arithmetic on the same bytes (each output column is the same dot product), verified
    # bit-exact against the three separate F.linear calls, so the vs-Wukong cross-check tolerance
    # is unchanged.
    ws[2:5] = [torch.cat(ws[2:5], dim=0)]     # wq|wk|wv  ->  wqkv [3D, D]
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
# identical to what Wukong/C dot against. GELU is the tanh approximation, matching Wukong's
# gelu() and the C column exactly (same flavor, so the cross-check needs no loosening).
# The Q/K/V projection is the FUSED HuggingFace `c_attn` form: one [S,D]x[D,3D] GEMM split into
# three [S,D] views (bit-exact with three separate F.linear calls). Wukong and C issue three
# separate [D,D] projections — a disclosed asymmetry in the PEER's favour, printed next to the
# ratio lines, because the peer's job is to be the strongest honest baseline.
def block(x, w, manual):
    ln1g, ln1b, wqkv, wo, ln2g, ln2b, w1, w2 = w
    nrm = F.layer_norm(x, LNSHAPE, ln1g, ln1b, 1e-5)
    q, k, v = F.linear(nrm, wqkv).split(D, dim=1)
    q = q.view(S, H, HD).transpose(0, 1)
    k = k.view(S, H, HD).transpose(0, 1)
    v = v.view(S, H, HD).transpose(0, 1)
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

# The whole forward via the fused SDPA path — the callable torch.compile specializes on.
def forward_sdpa():
    return forward(False)

# Warm WARMUP calls, then time ITERS. The timed loop body is exactly one forward — no per-iteration
# allocation or IO beyond what torch itself does inside the forward.
def bench_call(fn):
    for _ in range(WARMUP):
        fn()
    ts = [0.0] * ITERS
    for i in range(ITERS):
        t0 = time.perf_counter()
        fn()
        ts[i] = time.perf_counter() - t0
    return min(ts) * 1e3, statistics.median(ts) * 1e3

def try_build(fullgraph):
    fn = torch.compile(forward_sdpa, mode="max-autotune", fullgraph=fullgraph)
    t0 = time.perf_counter()
    out = fn()  # first call forces compilation; its wall time is the cold-start cost
    return fn, out, (time.perf_counter() - t0) * 1e3

# Whether the Inductor GEMM autotune backend has been pinned to ATEN. It is a GLOBAL inductor
# config, so once pinned it applies to every subsequent compile in this process (the disclosure
# line says so).
GEMM_PINNED = False

# Build one compiled variant, three tiers:
#  1. stock mode="max-autotune", fullgraph=True;
#  2. on failure, pin max_autotune_gemm_backends="ATEN" and retry fullgraph=True — some torch
#     builds' Inductor CPP GEMM template (cpp_CppMicroGemmFP32Vec) is broken on Windows/MSVC,
#     and ATen GEMM = MKL is torch's strongest CPU GEMM anyway, so this is the strongest
#     WORKING config, not a handicap (disclosed);
#  3. on a remaining fullgraph failure, disclose the torch._dynamo.explain break count and retry
#     fullgraph=False. Any exception past that propagates to the caller (total failure -> the
#     compiled columns stay n/a; eager unaffected).
def build_compiled(tag):
    global GEMM_PINNED
    try:
        return try_build(True)
    except Exception as e1:
        torch._dynamo.reset()
        if not GEMM_PINNED:
            try:
                import torch._inductor.config as icfg
                icfg.max_autotune_gemm_backends = "ATEN"
                GEMM_PINNED = True
                print("TORCH_COMPILE_DISCLOSE %s inductor CPP GEMM template broken on "
                      "Windows/MSVC -> max_autotune_gemm_backends pinned to ATEN (MKL) for all "
                      "compiled variants; first error: %s" % (tag, repr(str(e1))[:120]),
                      flush=True)
            except Exception:
                pass
    try:
        return try_build(True)
    except Exception as e2:
        gb = None
        try:
            gb = getattr(torch._dynamo.explain(forward_sdpa)(), "graph_break_count", None)
        except Exception:
            gb = None
        print("TORCH_COMPILE_DISCLOSE %s fullgraph-break gb=%s reason=%s"
              % (tag, gb, repr(str(e2))[:140]), flush=True)
        torch._dynamo.reset()
        return try_build(False)

with torch.inference_mode():
    print("TORCH_VERSION %s" % torch.__version__, flush=True)
    print("TORCH_THREADS %d" % DEFAULT_THREADS, flush=True)
    # Eager reference + cross-checks at 1 thread (doubles as cache warmup for the timed 1t runs).
    torch.set_num_threads(1)
    y = forward(False)
    ym = forward(True)
    scale = max(y.abs().max().item(), 1e-6)
    print("TORCH_MANUAL_VS_SDPA %.3e" % ((ym - y).abs().max().item() / scale), flush=True)
    dump(y, OUT_PATH)

    # --- compile phase: build BOTH variants first (untimed except the printed compile wall) ---
    comp1 = None
    compN = None
    if TRY_COMPILE:
        for tag, nth in (("1T", 1), ("NT", DEFAULT_THREADS)):
            wall_tag = "1C" if tag == "1T" else "NC"
            try:
                torch.set_num_threads(nth)
                fn, out, cw = build_compiled(tag)
                for _ in range(CWARMUP):
                    fn()
                delta = (out - y).abs().max().item() / scale
                print("TORCH%s_COMPILE %.1f" % (wall_tag, cw), flush=True)
                print("TORCH_COMPILED_VS_EAGER_%s %.3e" % (tag, delta), flush=True)
                if delta <= CTOL:
                    if tag == "1T":
                        # Dump BEFORE binding comp1: the compiled 1T variant is only timed once its
                        # output has actually landed for the vs-Wukong cross-check. Binding first
                        # left a dump failure (locked/full temp dir) with comp1 already set, so the
                        # variant was still timed and the Rust side cross-checked it against an
                        # empty buffer.
                        dump(out, COUT_PATH)   # the compiled output for the vs-Wukong cross-check
                        comp1 = fn
                    else:
                        compN = fn
                else:
                    # A fast-but-wrong compiled path must fail loudly and report NO time.
                    print("TORCH_COMPILED_BAD_%s %.3e" % (tag, delta), flush=True)
            except Exception as e:
                print("TORCH_COMPILE_FAIL %s reason=%s" % (tag, repr(str(e))[:160]), flush=True)
                try:
                    torch._dynamo.reset()
                except Exception:
                    pass
    else:
        print("TORCH_COMPILE_SKIP msvc-not-found", flush=True)

    # --- timing phase, coolest-first: T1 eager -> T1 compiled -> Tn eager -> Tn compiled ---
    torch.set_num_threads(1)
    mn, med = bench_call(lambda: forward(False))
    print("TORCH1 %.3f %.3f" % (mn, med), flush=True)
    mn, med = bench_call(lambda: forward(True))
    print("TORCH1M %.3f %.3f" % (mn, med), flush=True)
    if comp1 is not None:
        mn, med = bench_call(comp1)
        print("TORCH1C %.3f %.3f" % (mn, med), flush=True)
    # All-core variants LAST so their heat pollutes no single-thread torch number.
    torch.set_num_threads(DEFAULT_THREADS)
    mn, med = bench_call(lambda: forward(False))
    print("TORCHN %.3f %.3f" % (mn, med), flush=True)
    if compN is not None:
        mn, med = bench_call(compN)
        print("TORCHNC %.3f %.3f" % (mn, med), flush=True)
print("TORCH_OK", flush=True)
"#,
        s = s,
        d = d,
        h = h,
        hd = hd,
        dff = dff,
        layers = LAYERS,
        try_compile = if try_compile { "True" } else { "False" },
        w = w_path.display(),
        io = io_path.display(),
        out = out_path.display(),
        cout = cout_path.display(),
    )
}

/// Parsed torch timings — `(min, median)` ns/forward per variant — plus the SDPA + compiled outputs
/// for the cross-checks, the compile wall-clocks (ms; cold-start, kept out of per-forward numbers),
/// the compiled-vs-eager agreement figures, and any compile disclosures to echo.
struct TorchMeasure {
    sdpa_1t: Option<(f64, f64)>,
    manual_1t: Option<(f64, f64)>,
    sdpa_nt: Option<(f64, f64)>,
    comp_1t: Option<(f64, f64)>,
    comp_nt: Option<(f64, f64)>,
    compile_ms_1t: Option<f64>,
    compile_ms_nt: Option<f64>,
    compiled_vs_eager_1t: Option<f64>,
    compiled_vs_eager_nt: Option<f64>,
    manual_vs_sdpa: Option<f64>,
    /// Eager-SDPA final output (for the vs-Wukong cross-check).
    out: Vec<f32>,
    /// Compiled final output (for the vs-Wukong cross-check); empty if compilation was skipped/bad.
    comp_out: Vec<f32>,
    /// `TORCH_COMPILE_DISCLOSE` / `TORCH_COMPILE_FAIL` / `TORCH_COMPILED_BAD_*` lines to echo.
    disclosures: Vec<String>,
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
    let cout_path = dir.join(format!("model_torch_cout_s{}.bin", cfg.s));
    let script_path = dir.join(format!("model_torch_s{}.py", cfg.s));
    // The compiled variants are only attempted when the MSVC build environment could be captured;
    // without it Inductor's cl/link step can't run, so the script skips compilation entirely and
    // the compiled columns degrade to `n/a (MSVC not found)`.
    let try_compile = vcvars_env().is_some();
    // Stale outputs from a previous run must not masquerade as this run's results.
    let _ = std::fs::remove_file(&cout_path);
    std::fs::write(
        &script_path,
        torch_script(cfg, &w_path, &io_path, &out_path, &cout_path, try_compile),
    )
    .ok()?;
    // Spawn under the captured MSVC environment (with <base_prefix>\libs on LIB) so TorchInductor
    // can compile + link its CPU kernels; falls back to the inherited env when MSVC is absent.
    let run = peer_command(ctx, &script_path).output().ok()?;
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
    let grab1 = |tag: &str| -> Option<f64> {
        stdout
            .lines()
            .find_map(|l| l.strip_prefix(tag)?.trim().parse().ok())
    };
    let manual_vs_sdpa = grab1("TORCH_MANUAL_VS_SDPA ");
    // Disclosures the operator should see verbatim: graph breaks, total-failure reasons, and any
    // compiled-vs-eager divergence that suppressed a compiled timing.
    let disclosures: Vec<String> = stdout
        .lines()
        .filter(|l| {
            l.starts_with("TORCH_COMPILE_DISCLOSE")
                || l.starts_with("TORCH_COMPILE_FAIL")
                || l.starts_with("TORCH_COMPILE_SKIP")
                || l.starts_with("TORCH_COMPILED_BAD")
        })
        .map(|l| l.to_string())
        .collect();
    let read_f32 = |path: &Path| -> Vec<f32> {
        std::fs::read(path)
            .map(|bytes| {
                bytes
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect()
            })
            .unwrap_or_default()
    };
    let out = read_f32(&out_path);
    if out.len() != cfg.s * cfg.d {
        eprintln!(
            "model: torch peer: output size mismatch ({} vs {})",
            out.len(),
            cfg.s * cfg.d
        );
        return None;
    }
    // Only accept a compiled output of the right size (it exists only if the compiled path passed
    // its own vs-eager check and dumped before being bound for timing). Otherwise leave it empty —
    // and `bench_model_size` then suppresses the compiled column's time, so a variant that produced
    // no verifiable output never reports one.
    let comp_out = read_f32(&cout_path);
    let comp_out = if comp_out.len() == cfg.s * cfg.d {
        comp_out
    } else {
        Vec::new()
    };
    Some(TorchMeasure {
        sdpa_1t: grab("TORCH1"),
        manual_1t: grab("TORCH1M"),
        sdpa_nt: grab("TORCHN"),
        comp_1t: grab("TORCH1C"),
        comp_nt: grab("TORCHNC"),
        compile_ms_1t: grab1("TORCH1C_COMPILE "),
        compile_ms_nt: grab1("TORCHNC_COMPILE "),
        compiled_vs_eager_1t: grab1("TORCH_COMPILED_VS_EAGER_1T "),
        compiled_vs_eager_nt: grab1("TORCH_COMPILED_VS_EAGER_NT "),
        manual_vs_sdpa,
        out,
        comp_out,
        disclosures,
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
    println!("  absolute numbers are thermal-bound — the Wukong/C ratio is the stable metric.");
    // Power state: sustained-load timings taken on battery are not comparable to AC runs.
    println!("  {}\n", power_status_line());

    let torch = detect_torch();
    match &torch {
        Some(t) => {
            let compile_env = match vcvars_env() {
                Some(_) => "torch.compile ENABLED (vcvars64 MSVC env captured)".to_string(),
                None => "torch.compile disabled (MSVC vcvars64 env unavailable)".to_string(),
            };
            println!(
                "  PyTorch peer: torch {} via {} ({} threads) — CPU float32 under \
                 torch.inference_mode().\n  Regimes: EAGER (T1/Tn) and torch.compile(\
                 mode=\"max-autotune\", fullgraph=True) (T1c/Tnc). {compile_env}.\n  Same \
                 weights/inputs via little-endian f32 blobs; T1 = set_num_threads(1), Tn = default \
                 all threads;\n  sdpa = F.scaled_dot_product_attention (fused industry path), man = \
                 manual matmul+softmax; comp = TorchInductor-compiled whole forward.\n  Peer \
                 strength: Q/K/V is ONE fused [D,3D] projection (HuggingFace GPT2 `c_attn`); \
                 Wukong and C issue three [D,D] projections.\n",
                t.version, t.py, t.threads
            );
        }
        None => println!(
            "  PyTorch peer: no importable torch on `python` or tools/torch-venv (PYTHON overrides) \
             — torch columns skipped.\n"
        ),
    }

    if interp_gate() != Some(0.0) {
        println!(
            "  ! MODEL GATE NOT BIT-EXACT — see the gate messages above; the timings below are \
             not trustworthy until this is fixed"
        );
    }

    // XBENCH_MODEL_S=<n> restricts the sweep to a single seq length (a smoke/iteration knob — the
    // compiled peer's max-autotune warmup is minutes per S, so one S keeps a mechanism check fast).
    let sizes: Vec<usize> = match std::env::var("XBENCH_MODEL_S") {
        Ok(v) => v
            .split(',')
            .filter_map(|p| p.trim().parse().ok())
            .collect::<Vec<_>>(),
        Err(_) => vec![128, 512],
    };
    let sizes = if sizes.is_empty() { vec![128, 512] } else { sizes };
    for s in sizes {
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
/// glue (extractions, mask, scatter) executes the same vectorized MIR. The `@parallel` variant is
/// gated too: its native forward (multicore kernels + the outlined per-head region) must ALSO
/// equal the serial interpreter oracle bit-for-bit — heads write disjoint slices and each head
/// runs the identical serial kernel sequence, so any divergence is a real bug.
///
/// Returns the worst max-relative-error across the gated variants (`Some(0.0)` = bit-exact, the
/// only acceptable value — asserted by the `interp_gate_bit_exact` unit test), or `None` when a
/// stage failed to compile/run (reported on stdout).
fn interp_gate() -> Option<f64> {
    let cfg = Cfg {
        s: 16,
        d: 64,
        h: 4,
        dff: 256,
    };
    let mut interner = Interner::new();
    let block_src = wk_block(cfg, false);
    let ln_src = wk_final_ln(cfg);
    let (Some(block_prog), Some(ln_prog)) = (
        build_program(&block_src, &mut interner),
        build_program(&ln_src, &mut interner),
    ) else {
        println!("  ! interp gate: reduced-config model failed to compile — skipping gate\n");
        return None;
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
        let mut bufs: [&mut [f32]; 19] = [
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
            &mut sc.attn,
            &mut sc.a,
            &mut sc.ff1,
            xo,
        ];
        if let Err(e) = wukong_interp::run_kernel_f32(&block_prog, entry, &mut bufs, &interner) {
            println!("  ! interp gate: interpreter error at layer {l}: {e}");
            ok = false;
            break;
        }
    }
    if ok {
        let (mut g, mut b) = (lnf_g.clone(), lnf_b.clone());
        let mut bufs: [&mut [f32]; 4] = [&mut xa, &mut g, &mut b, &mut y_interp];
        if let Err(e) = wukong_interp::run_kernel_f32(&ln_prog, entry, &mut bufs, &interner) {
            println!("  ! interp gate: interpreter error in final LN: {e}");
            ok = false;
        }
    }
    if !ok {
        return None;
    }

    // --- native (Cranelift JIT) forward over the identical MIR + inputs ---
    let (Ok(block_jit), Ok(ln_jit)) = (
        wukong_codegen_cranelift::jit_module(&block_prog, &interner),
        wukong_codegen_cranelift::jit_module(&ln_prog, &interner),
    ) else {
        println!("  ! interp gate: JIT failed — skipping gate\n");
        return None;
    };
    let (Some(bp), Some(lp)) = (block_jit.func_ptr(entry), ln_jit.func_ptr(entry)) else {
        println!("  ! interp gate: kbench symbol missing — skipping gate\n");
        return None;
    };
    let block_fn: WukBlockFn = unsafe { std::mem::transmute(bp) };
    let ln_fn: LnFn = unsafe { std::mem::transmute(lp) };
    let mut sc2 = Scratch::new(cfg);
    let (mut xa2, mut xb2) = (vec![0.0f32; sd], vec![0.0f32; sd]);
    let mut y_native = vec![0.0f32; sd];
    unsafe {
        run_forward_mer(
            block_fn, ln_fn, &weights, &mut sc2, &x0, &mut xa2, &mut xb2, &lnf_g, &lnf_b,
            &mut y_native,
        );
    }

    let (rel, at) = max_rel_err(&y_interp, &y_native);
    if rel != 0.0 {
        println!(
            "  ! interp gate: interpreter vs native max rel err {rel:.2e} at [{at}] \
             (interp={} native={})\n",
            y_interp[at], y_native[at]
        );
    }

    // --- native @parallel forward (multicore kernels + the outlined per-head region) must ALSO
    // equal the serial interpreter oracle bit-for-bit: each head runs the identical serial kernel
    // sequence over a disjoint slice, so cross-head scheduling cannot change a single bit.
    let Some(par_prog) = build_program(&wk_block(cfg, true), &mut interner) else {
        println!("  ! interp gate: @parallel reduced-config model failed to compile\n");
        return None;
    };
    let Ok(par_jit) = wukong_codegen_cranelift::jit_module(&par_prog, &interner) else {
        println!("  ! interp gate: @parallel JIT failed — skipping gate\n");
        return None;
    };
    let Some(pp) = par_jit.func_ptr(entry) else {
        println!("  ! interp gate: @parallel kbench symbol missing — skipping gate\n");
        return None;
    };
    let par_fn: WukBlockFn = unsafe { std::mem::transmute(pp) };
    let mut sc3 = Scratch::new(cfg);
    let (mut xa3, mut xb3) = (vec![0.0f32; sd], vec![0.0f32; sd]);
    let mut y_par = vec![0.0f32; sd];
    unsafe {
        run_forward_mer(
            par_fn, ln_fn, &weights, &mut sc3, &x0, &mut xa3, &mut xb3, &lnf_g, &lnf_b, &mut y_par,
        );
    }
    let (prel, pat) = max_rel_err(&y_interp, &y_par);
    if prel != 0.0 {
        println!(
            "  ! interp gate: interpreter vs @parallel native max rel err {prel:.2e} at [{pat}] \
             (interp={} par={})\n",
            y_interp[pat], y_par[pat]
        );
    }
    let worst = rel.max(prel);
    if worst == 0.0 {
        println!(
            "  interp gate (S={} D={} H={} Dff={}, {LAYERS} layers): interpreter == native == \
             @parallel native BIT-EXACT over all {sd} outputs\n",
            cfg.s, cfg.d, cfg.h, cfg.dff
        );
    }
    Some(worst)
}

fn bench_model_size(cc: &str, dir: &Path, cfg: Cfg, torch: Option<&TorchCtx>) {
    let flops = cfg.flops_per_layer() * LAYERS as f64;
    println!(
        "--- model S={} ({} tokens/forward, {:.1} GFLOP/forward) ---",
        cfg.s,
        cfg.s,
        flops / 1e9
    );
    // Sample the power state BEFORE this size and again after it. A single sample at the top of
    // the run is not enough: a size takes minutes, and an all-core burst on this box can knock the
    // adapter off mid-sweep — after which every remaining column is battery-throttled with nothing
    // in the output to say so. `tools/measure_gpt2.ps1` samples per regime for exactly this reason.
    let power_before = power_status_line();
    println!("  {power_before}");

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

    // Sweep knobs (both purely additive; default behaviour unchanged):
    //  * XBENCH_MODEL_WUK_ONLY skips every external peer (C, C(fast), PyTorch) — a dev iterating on
    //    the multicore kernels gets Wuk(1c) vs Wuk(par) + scaling in seconds instead of the minutes
    //    the peer compilation + torch warmups cost. Not for reported numbers.
    //  * XBENCH_MODEL_TORCH_ONLY skips the Wukong and C columns and runs only the torch peer variants
    //    — for anchoring the peer's own machine state (e.g. an external thread sweep of the peer).
    //  They are mutually exclusive; WUK_ONLY (which also drops torch) wins if both are set.
    let wk_only = std::env::var("XBENCH_MODEL_WUK_ONLY").is_ok();
    let torch_only = !wk_only && std::env::var("XBENCH_MODEL_TORCH_ONLY").is_ok();

    // --- Wukong (serial), through the real pipeline ---
    let mut interner = Interner::new();
    let (wuk, lnf_mod) = if torch_only {
        (None, None)
    } else {
        (
            compile_wukong(&wk_block(cfg, false), &mut interner),
            compile_wukong(&wk_final_ln(cfg), &mut interner),
        )
    };
    let wk_m = match (&wuk, &lnf_mod) {
        (Some(m), Some(l)) => {
            // Mechanism transparency: print exactly which recognized kernels the block dispatches.
            let calls = kernel_calls(&m.mir);
            let summary = calls
                .iter()
                .map(|(k, v)| format!("{v}x {}", k.trim_start_matches("wukong_")))
                .collect::<Vec<_>>()
                .join(", ");
            println!("  block dispatches (per layer, from optimized MIR): {summary}");
            for need in ["wukong_sgemm_nt", "wukong_sgemm_nt_epi", "wukong_norm_affine_f32"] {
                if !calls.iter().any(|(k, _)| k == need) {
                    println!("  ! WARNING: expected recognized kernel {need} did NOT dispatch — timing scalar loops");
                }
            }
            let (Some(bp), Some(lp)) = (m.func(&mut interner, "kbench"), l.func(&mut interner, "kbench"))
            else {
                println!("  ! wukong kbench symbol missing");
                return;
            };
            let block_fn: WukBlockFn = unsafe { std::mem::transmute(bp) };
            let ln_fn: LnFn = unsafe { std::mem::transmute(lp) };
            // Big stack: the per-head scratch is loop-body-local in the Wukong block, so the
            // serial column runs the whole ~1.5 MiB (at S=512) frame on the calling thread.
            let ns = on_big_stack(|| {
                let mut run = || unsafe {
                    run_forward_mer(
                        block_fn, ln_fn, &weights, &mut sc, &x0, &mut xa, &mut xb, &lnf_g,
                        &lnf_b, &mut y,
                    )
                };
                time_forward(&mut run)
            });
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
    // The C columns are dropped entirely under WUK_ONLY (Wukong-only) or TORCH_ONLY (peer-only).
    let no_c = wk_only || torch_only;
    let run_c_slow = !no_c && (cfg.s <= 128 || std::env::var("XBENCH_MODEL_NAIVE").is_ok());
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
        // Only explain the naive-dot omission when C would otherwise have run; under WUK_ONLY /
        // TORCH_ONLY the C columns are dropped on purpose and need no per-size note.
        if !no_c {
            println!(
                "  -> C(gcc) omitted at S={} (naive-dot forward is tens of seconds per call; its \
                 loss is already shown at S=128). Set XBENCH_MODEL_NAIVE to force it.",
                cfg.s
            );
        }
        None
    };

    // --- C(fast): identical source, -ffast-math ---
    let cfast_m = if no_c { None } else { compile_c_model(
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

    // --- Wukong @parallel, measured LAST so its all-core heat pollutes no single-core column ---
    let wk_par_compiled = if torch_only {
        None
    } else {
        compile_wukong(&wk_block(cfg, true), &mut interner)
    };
    let wk_par_m = wk_par_compiled.and_then(|m| {
        // Same mechanism transparency as the serial column: print the @parallel dispatch set and
        // warn if the embedded GEMMs regressed to serial (they would still be *correct*, so only
        // this scan would catch it).
        let calls = kernel_calls(&m.mir);
        let summary = calls
            .iter()
            .map(|(k, v)| format!("{v}x {}", k.trim_start_matches("wukong_")))
            .collect::<Vec<_>>()
            .join(", ");
        println!("  @parallel block dispatches (per layer, from optimized MIR): {summary}");
        if !calls.iter().any(|(k, _)| k == "wukong_sgemm_nt_parallel") {
            println!(
                "  ! WARNING: @parallel block did NOT dispatch wukong_sgemm_nt_parallel — \
                 embedded GEMMs are running serial"
            );
        }
        // The head loop must have outlined into a parallel region (heads across cores). Without
        // it the block is still correct — the heads just run as the old serial chain — so only
        // this scan makes the regression visible.
        if !calls.iter().any(|(k, _)| k == "wukong_parallel_for") {
            println!(
                "  ! WARNING: @parallel block did NOT outline the head loop into a \
                 wukong_parallel_for region — heads are running serially"
            );
        }
        let lnf = lnf_mod.as_ref()?;
        let (Some(bp), Some(lp)) = (
            m.func(&mut interner, "kbench"),
            lnf.func(&mut interner, "kbench"),
        ) else {
            return None;
        };
        let block_fn: WukBlockFn = unsafe { std::mem::transmute(bp) };
        let ln_fn: LnFn = unsafe { std::mem::transmute(lp) };
        let ns = on_big_stack(|| {
            let mut run = || unsafe {
                run_forward_mer(
                    block_fn, ln_fn, &weights, &mut sc, &x0, &mut xa, &mut xb, &lnf_g, &lnf_b,
                    &mut y,
                )
            };
            time_forward(&mut run)
        });
        Some(MeasureModel {
            compile: m.compile,
            ns_per_fwd: ns,
            out: y.clone(),
        })
    });

    // --- PyTorch CPU eager peer, timed immediately after the Wukong columns in the SAME bench
    // invocation (same-run adjacency). It runs in its own process over the just-dumped identical
    // buffers; inside the script the single-thread variants run first and the all-core variant
    // last, so multicore heat pollutes no single-thread torch number. The blob writes + torch
    // import + its own warmups sit between Wuk(par)'s all-core burst and the first timed torch
    // iteration.
    let torch_m = if wk_only { None } else { torch }.and_then(|t| {
        println!(
            "  running PyTorch peer (torch {}, f32 inference_mode; eager + torch.compile \
             max-autotune; timed 10 per variant)...",
            t.version
        );
        bench_torch(t, dir, cfg, &weights, &x0, &lnf_g, &lnf_b)
    });
    // Same-run Wuk(1c) vs Wuk(par) scaling — the reliable multicore instrument on this hybrid laptop
    // (both measured adjacently, same thermal state). Printed always; it's the number to move.
    if let (Some(s), Some(p)) = (&wk_m, &wk_par_m) {
        println!(
            "  Wuk scaling: 1c {:.1} ms -> par {:.1} ms = {:.2}x",
            s.ns_per_fwd / 1e6,
            p.ns_per_fwd / 1e6,
            s.ns_per_fwd / p.ns_per_fwd
        );
    }

    // --- report ---
    // Torch eager columns (no compile step): T1(sdpa)/Tn(sdpa) = fused
    // F.scaled_dot_product_attention at 1/all threads, T1(man) = manual matmul+softmax at 1 thread.
    // Torch compiled columns: T1(comp)/Tn(comp) = torch.compile(max-autotune, fullgraph) of the
    // whole forward — their `compile ms` cell is the Inductor cold-start wall, kept OUT of the
    // per-forward numbers. Column value is the min of the timed iterations (the suite's
    // best-observed convention); medians are printed below the table.
    let mk_torch = |v: Option<(f64, f64)>, out: &[f32]| {
        v.map(|(mn, _)| MeasureModel {
            compile: Duration::ZERO, // sentinel: eager, no compile step — printed as "eager"
            ns_per_fwd: mn,
            out: out.to_vec(),
        })
    };
    // Compiled column: carries the Inductor compile wall in `compile` (printed in the compile-ms
    // row, separate from ns/forward). A missing wall falls back to a 1ns marker so the cell never
    // reads "eager" for a compiled variant.
    let mk_comp = |v: Option<(f64, f64)>, ms: Option<f64>, out: &[f32]| {
        v.map(|(mn, _)| MeasureModel {
            compile: Duration::from_secs_f64((ms.unwrap_or(0.0) / 1e3).max(1e-9)),
            ns_per_fwd: mn,
            out: out.to_vec(),
        })
    };
    // The compiled 1-thread variant is the one whose output is dumped for the vs-Wukong
    // cross-check. If that output did not land, its timing is SUPPRESSED rather than printed
    // beside an uncheckable column — the invariant `bench_torch` documents, now enforced on this
    // side too instead of only assumed of the peer script. (`Tn(comp)`, `T1(man)` and `Tn(sdpa)`
    // dump no output by design; they are timing-only columns and are never cross-checked.)
    if let Some(t) = &torch_m {
        if t.comp_1t.is_some() && t.comp_out.len() != sd {
            println!(
                "  ! T1(comp) SUPPRESSED — the compiled peer reported a time but dumped {} of {sd} \
                 output elements, so its result cannot be cross-checked",
                t.comp_out.len()
            );
        }
    }
    let (torch1_m, torchman_m, torchn_m, torch1c_m, torchnc_m) = match &torch_m {
        Some(t) => (
            mk_torch(t.sdpa_1t, &t.out),
            mk_torch(t.manual_1t, &[]),
            mk_torch(t.sdpa_nt, &[]),
            mk_comp(
                t.comp_1t.filter(|_| t.comp_out.len() == sd),
                t.compile_ms_1t,
                &t.comp_out,
            ),
            mk_comp(t.comp_nt, t.compile_ms_nt, &[]),
        ),
        None => (None, None, None, None, None),
    };

    let cols: [(&str, &Option<MeasureModel>); 9] = [
        ("Wuk(1c)", &wk_m),
        ("Wuk(par)", &wk_par_m),
        ("C(gcc)", &c_m),
        ("C(fast)", &cfast_m),
        ("T1(sdpa)", &torch1_m),
        ("T1(man)", &torchman_m),
        ("T1(comp)", &torch1c_m),
        ("Tn(sdpa)", &torchn_m),
        ("Tn(comp)", &torchnc_m),
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
            "  torch detail (eager): sdpa-1t {}; manual-1t {}; sdpa-all {}",
            fmt(&t.sdpa_1t),
            fmt(&t.manual_1t),
            fmt(&t.sdpa_nt)
        );
        // Compiled timings + the cold-start compile wall (printed separately so it never pollutes
        // any per-forward number).
        if t.comp_1t.is_some() || t.comp_nt.is_some() {
            let wall = |ms: &Option<f64>| ms.map(|m| format!("{m:.0} ms")).unwrap_or_else(|| "n/a".into());
            println!(
                "  torch detail (compiled, max-autotune): comp-1t {} (compile {}); comp-all {} (compile {})",
                fmt(&t.comp_1t),
                wall(&t.compile_ms_1t),
                fmt(&t.comp_nt),
                wall(&t.compile_ms_nt),
            );
        }
        if let Some(r) = t.manual_vs_sdpa {
            println!("  torch-internal manual-attn vs SDPA: max|Δ|/max|out| = {r:.2e}");
        }
        // Compiled-vs-eager agreement (the compiled path is only timed when this is within tol).
        if let Some(r) = t.compiled_vs_eager_1t {
            println!("  torch-internal compiled vs eager (1t): max|Δ|/max|out| = {r:.2e}");
        }
        if let Some(r) = t.compiled_vs_eager_nt {
            println!("  torch-internal compiled vs eager (all): max|Δ|/max|out| = {r:.2e}");
        }
        // Echo any compile disclosures verbatim (graph breaks, total failures, suppressed-timing
        // divergences, MSVC-absent skip) so the compiled regime's state is never silent.
        for d in &t.disclosures {
            if d.starts_with("TORCH_COMPILE_SKIP") {
                println!(
                    "  torch.compile skipped: MSVC build env not found — compiled columns n/a \
                     (eager unaffected)"
                );
            } else {
                println!("  torch.compile disclosure: {d}");
            }
        }
        // If compilation was attempted (MSVC present) yet a compiled variant produced no time, say
        // so plainly next to the n/a cells.
        if vcvars_env().is_some() && torch1c_m.is_none() && torchnc_m.is_none() && t.disclosures.is_empty() {
            println!("  note: T1(comp)/Tn(comp) n/a — torch.compile produced no timed result");
        }
    }

    let ratio_line = |wuk: &Option<MeasureModel>, peer: &Option<MeasureModel>, who: &str, peer_name: &str| {
        if let (Some(m), Some(p)) = (wuk, peer) {
            let r = p.ns_per_fwd / m.ns_per_fwd;
            println!(
                "  -> Wukong {who} is {:.2}x {} than {peer_name}",
                if r >= 1.0 { r } else { 1.0 / r },
                if r >= 1.0 { "faster" } else { "slower" }
            );
        }
    };
    ratio_line(&wk_m, &c_m, "(1 core)", "C (gcc -O3 -march=native)");
    ratio_line(&wk_m, &cfast_m, "(1 core)", "C(fast) (gcc -ffast-math)");
    ratio_line(&wk_par_m, &cfast_m, "@parallel", "C(fast) (single-threaded)");
    ratio_line(
        &wk_m,
        &torch1_m,
        "(1 core)",
        "PyTorch eager SDPA (1 thread)",
    );
    ratio_line(
        &wk_m,
        &torchman_m,
        "(1 core)",
        "PyTorch eager manual-attn (1 thread)",
    );
    ratio_line(
        &wk_m,
        &torchn_m,
        "(1 core)",
        "PyTorch eager SDPA (all threads)",
    );
    ratio_line(
        &wk_par_m,
        &torchn_m,
        "@parallel",
        "PyTorch eager SDPA (all threads)",
    );
    ratio_line(
        &wk_m,
        &torch1c_m,
        "(1 core)",
        "PyTorch compiled max-autotune (1 thread)",
    );
    ratio_line(
        &wk_par_m,
        &torchnc_m,
        "@parallel",
        "PyTorch compiled max-autotune (all threads)",
    );
    // Disclosed AT THE POINT THE NUMBER IS PRINTED, not only in the module header: every torch
    // ratio above is measured against the FUSED-c_attn peer, which is the strongest honest form of
    // this forward and stronger than what Wukong currently emits.
    if torch1_m.is_some() || torchn_m.is_some() || torch1c_m.is_some() || torchnc_m.is_some() {
        println!(
            "     (peer-strength disclosure: the torch columns use HuggingFace GPT2's FUSED \
             [D,3D] c_attn Q/K/V projection — ONE GEMM; Wukong and C issue THREE [D,D] \
             projections. QKV fusion is an open Wukong lever, so these ratios charge Wukong for \
             not having it.)"
        );
    }
    if wk_par_m.is_some() {
        println!(
            "     (thread disclosure: both C columns and the T1 torch columns are \
             single-threaded; Wuk(par), Tn(sdpa) and Tn(comp) are the multicore columns — the \
             @parallel dispatch set is printed above)"
        );
    }

    // --- full-buffer cross-checks over the final [S, D] output ---
    // Wukong-vs-C: the GEMM/norm reductions reassociate and Wukong's ~1-ULP poly exp/tanh differ
    // from libm, compounding over 12 layers — and the final LayerNorm re-centers every row to mean
    // 0, so a *per-element* relative error is meaningless on the near-zero elements (catastrophic-
    // cancellation zeros). Use the suite's **magnitude-normalized** metric `max|Δ| / max|out|`
    // (the softmax_bwd / cumsum convention) at the usual 1e-3.
    // Every non-PASS outcome is also collected so the run ends with ONE loud line: a `!` in the
    // middle of a long sweep scrolls past, and this bench's callers gate on the exit code, not on
    // stdout. A comparison that could not be made (`cross_check_rel` -> Err) is reported as NOT RUN
    // and counted here — it is never allowed to read as agreement.
    let unpassed = std::cell::RefCell::new(Vec::<String>::new());
    let check = |a: &Option<MeasureModel>, b: &Option<MeasureModel>, who: &str, tol: f64| {
        if let (Some(x), Some(z)) = (a, b) {
            match cross_check_rel(&x.out, &z.out) {
                Err(why) => {
                    println!("  ! cross-check {who}: NOT RUN — {why}");
                    unpassed.borrow_mut().push(format!("{who}: not run ({why})"));
                }
                Ok(rel) if rel > tol => {
                    println!("  ! full-buffer mismatch {who}: max|Δ|/max|out| = {rel:.2e} (tol {tol:.0e})");
                    unpassed
                        .borrow_mut()
                        .push(format!("{who}: max|Δ|/max|out| = {rel:.2e} > tol {tol:.0e}"));
                }
                Ok(rel) => {
                    println!("  cross-check {who}: max|Δ|/max|out| = {rel:.2e} (tol {tol:.0e}) — PASS")
                }
            }
        }
    };
    check(&wk_m, &c_m, "Wukong vs C", 1e-3);
    check(&wk_m, &cfast_m, "Wukong vs C(fast)", 1e-3);
    // Wukong vs torch: same GELU flavor (tanh approx) and eps, so the residual is the same
    // reassociation + poly-vs-libm class as vs C — expect ~1e-5-ish at the standard 1e-3.
    check(&wk_m, &torch1_m, "Wukong vs Torch (eager SDPA)", 1e-3);
    // The compiled output is cross-checked exactly like the eager SDPA output — vs Wukong at the
    // same 1e-3 magnitude-normalized tolerance (the script already gated it vs eager, so a
    // fast-but-wrong compiled path never reached a timed column in the first place).
    check(&wk_m, &torch1c_m, "Wukong vs Torch (compiled max-autotune)", 1e-3);
    // Serial vs @parallel Wukong: every dispatched _parallel kernel is bit-identical to its serial
    // twin (fixed chunking / row-mapped) and the outlined glue loops are deterministic, so this one
    // stays the strict per-element check — it is expected EXACT.
    if let (Some(x), Some(z)) = (&wk_m, &wk_par_m) {
        if x.out.len() != z.out.len() || x.out.is_empty() {
            println!(
                "  ! cross-check Wukong serial vs @parallel: NOT RUN — buffer lengths {} vs {}",
                x.out.len(),
                z.out.len()
            );
            unpassed
                .borrow_mut()
                .push("Wukong serial vs @parallel: not run (buffer length disagreement)".into());
        } else {
            let (rel, at) = max_rel_err(&x.out, &z.out);
            if rel > 0.0 {
                println!(
                    "  ! Wukong serial vs @parallel differ at [{at}]: {} vs {} (rel {rel:.2e}) — \
                     expected bit-exact",
                    x.out[at], z.out[at]
                );
                unpassed
                    .borrow_mut()
                    .push(format!("Wukong serial vs @parallel: rel {rel:.2e} at [{at}], expected bit-exact"));
            } else {
                println!("  cross-check Wukong serial vs @parallel: BIT-EXACT");
            }
        }
    }

    // The second power sample. A state change across the size means the columns above were taken
    // under two different machines, so none of them is comparable to any other.
    let power_after = power_status_line();
    if power_after == power_before {
        println!("  {power_after} (unchanged before -> after this size)");
    } else {
        println!(
            "  !!! POWER STATE CHANGED DURING S={} — every timing above is NON-REPORTABLE\n      \
             before: {power_before}\n      after:  {power_after}",
            cfg.s
        );
    }

    // One loud terminal line per size, so a failed or un-runnable cross-check cannot be mistaken
    // for a clean sweep by anyone skimming the table.
    let unpassed = unpassed.into_inner();
    if !unpassed.is_empty() {
        println!(
            "  !!! {} CROSS-CHECK(S) DID NOT PASS at S={} — the timings above are NOT trustworthy: {}",
            unpassed.len(),
            cfg.s,
            unpassed.join("; ")
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A cross-check that inspected no elements must never report agreement. The old closure
    /// `zip`ped the two buffers, so an EMPTY or truncated peer output folded to `max|Δ| = 0` and
    /// printed `PASS` beside a fully-reported ms/forward column; `f32::max` discarding NaN gave
    /// an all-NaN peer the same free pass. All three must come back as `Err`.
    #[test]
    fn cross_check_refuses_vacuous_comparisons() {
        let ours: Vec<f32> = (0..1024).map(|i| (i % 97) as f32 * 0.031 - 1.5).collect();

        // No peer output at all — the compiled-peer dump-failure path.
        assert!(cross_check_rel(&ours, &[]).is_err(), "empty peer must not pass");
        // Partial peer output: zip truncates, so this used to read as a perfect match.
        assert!(
            cross_check_rel(&ours, &ours[..4]).is_err(),
            "truncated peer must not pass"
        );
        // All-NaN peer: max() discards NaN, so maxerr folded to 0.0.
        let nans = vec![f32::NAN; ours.len()];
        assert!(cross_check_rel(&ours, &nans).is_err(), "NaN peer must not pass");

        // Real comparisons still produce the same magnitude-normalized number as before.
        assert_eq!(cross_check_rel(&ours, &ours), Ok(0.0));
        let off: Vec<f32> = ours.iter().map(|v| v + 1.0).collect();
        let maxabs = off.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
        let rel = cross_check_rel(&ours, &off).expect("equal-length finite buffers compare");
        assert!(
            (rel - (1.0 / maxabs) as f64).abs() < 1e-9,
            "magnitude-normalized metric changed: {rel}"
        );
    }

    /// Pin the model block's recognized-kernel dispatch sets for BOTH variants. The recognizers
    /// are gate-blind (interp and native call the same symbol), so a silent regression to scalar
    /// loops — or to serial GEMMs under `@parallel` — passes every correctness gate; only this
    /// scan catches it. Uses the interp-gate config (small dims) to keep the test fast.
    #[test]
    fn block_dispatch_sets_pinned() {
        let cfg = Cfg { s: 16, d: 64, h: 4, dff: 256 };
        let mut interner = Interner::new();

        let serial = compile_wukong(&wk_block(cfg, false), &mut interner)
            .expect("serial block must compile");
        let calls = kernel_calls(&serial.mir);
        for need in [
            "wukong_sgemm_nt",
            "wukong_sgemm_nt_alpha",
            "wukong_sgemm_nt_epi",
            "wukong_norm_affine_f32",
            "wukong_norm_f32",
            "wukong_velem_f32",
        ] {
            assert!(
                calls.iter().any(|(k, _)| k == need),
                "serial block lost dispatch {need}; got {calls:?}"
            );
        }
        // The natural per-head-scratch spelling must NOT parallelize without the attribute.
        assert!(
            !calls.iter().any(|(k, _)| k == "wukong_parallel_for"),
            "serial block must not outline a parallel region; got {calls:?}"
        );

        let par = compile_wukong(&wk_block(cfg, true), &mut interner)
            .expect("@parallel block must compile");
        let pcalls = kernel_calls(&par.mir);
        // The head loop outlines into ONE parallel region per layer call...
        assert!(
            pcalls.iter().any(|(k, _)| k == "wukong_parallel_for"),
            "@parallel block did not outline the head loop into a region; got {pcalls:?}"
        );
        // ...while the non-head GEMMs (Q/K/V, WO, FFN down-proj) stay multicore.
        assert!(
            pcalls.iter().any(|(k, _)| k == "wukong_sgemm_nt_parallel"),
            "@parallel block's embedded GEMMs regressed to serial; got {pcalls:?}"
        );
        // Split the MIR into the outlined region body vs everything else: per-head kernels run
        // SERIAL inside the region (the region supplies the threading — the identical op sequence
        // the serial spelling runs, which is the serial == @parallel bit-exactness argument), and
        // NO serial NT GEMM may appear outside it.
        let start = par
            .mir
            .find("fn wukong$par$")
            .expect("outlined head-region body missing from @parallel MIR");
        let after = &par.mir[start..];
        let end = after[3..].find("\nfn ").map(|i| i + 4).unwrap_or(after.len());
        let region = &after[..end];
        let rest = format!("{}{}", &par.mir[..start], &after[end..]);
        let rcalls = kernel_calls(region);
        for need in ["wukong_sgemm_nt", "wukong_sgemm_nt_alpha", "wukong_norm_f32"] {
            assert!(
                rcalls.iter().any(|(k, _)| k == need),
                "region body lost per-head serial dispatch {need}; got {rcalls:?}"
            );
        }
        assert!(
            !rcalls.iter().any(|(k, _)| k.ends_with("_parallel")),
            "region body must not nest multicore kernels; got {rcalls:?}"
        );
        let restc = kernel_calls(&rest);
        assert!(
            !restc.iter().any(|(k, _)| k == "wukong_sgemm_nt"),
            "@parallel block emits serial wukong_sgemm_nt outside the region; got {restc:?}"
        );
    }

    /// The peer must stay the STRONGEST honest implementation of this forward. The model bench
    /// previously compared against an unfused manual forward rather than the `Conv1D`-fused
    /// HuggingFace path, and the strongest-peer measurement of the same workload came out the
    /// other way round — so the fused `c_attn` form is pinned here, not left to review.
    #[test]
    fn torch_peer_uses_the_fused_hf_qkv_projection() {
        let cfg = Cfg { s: 128, d: 768, h: 12, dff: 3072 };
        let p = Path::new("unused.bin");
        let src = torch_script(cfg, p, p, p, p, true);
        assert!(
            src.contains("ws[2:5] = [torch.cat(ws[2:5], dim=0)]"),
            "peer no longer builds HuggingFace's fused [3D, D] c_attn weight"
        );
        assert!(
            src.contains("q, k, v = F.linear(nrm, wqkv).split(D, dim=1)"),
            "peer no longer issues the Q/K/V projection as ONE fused GEMM"
        );
        assert!(
            !src.contains("F.linear(nrm, wq)"),
            "peer regressed to three separate Q/K/V projections — that is the weaker peer the \
             band standard forbids"
        );
        // The asymmetry must be disclosed where the ratio is read, not only in a doc elsewhere.
        assert!(
            src.contains("HuggingFace"),
            "the fused-projection asymmetry lost its in-script disclosure"
        );
    }

    /// The reduced-config differential gate the bench prints must be BIT-exact, as a `cargo test`
    /// invariant (not only a bench-time printout): interpreter oracle == native serial == native
    /// `@parallel` (multicore kernels + the outlined per-head region) over the full 12-layer
    /// forward.
    #[test]
    fn interp_gate_bit_exact() {
        assert_eq!(
            interp_gate(),
            Some(0.0),
            "model interp/native/@parallel differential gate is not bit-exact"
        );
    }
}
