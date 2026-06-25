# Conv2d vs cuDNN — implicit-GEMM + Winograd, and a real cuDNN peer

Branch `perf/gpu-conv-2` (worktree `Mercury-conv2`). Target: mobile **RTX 4050, sm_89 (Ada)**, driver
592.27, `cudarc` 0.16.6, no CUDA toolkit (PTX driver-JIT + dlopen'd redist DLLs).

Goal: stand up a **real cuDNN peer** (the prior conv work measured only vs naive CUDA-C ≈6×, which
proves nothing about the true gap), then close — and beat — cuDNN on the common conv shapes via
implicit-GEMM, Winograd, and fused conv+bias+act.

## Status / plan

1. [done] cuDNN Tier-B peer bound (`baselines.rs::cudnn_conv2d_run` / `time_cudnn_conv2d`, takes
   `pad`/`stride`), NHWC fp16 tensor-core fast path, v7-heuristic-chosen engine disclosed. Benches
   `gpu.rs::{conv_vs_cudnn, conv_affine_vs_cudnn, conv_splitk_vs_cudnn, conv_winograd_vs_cudnn}`.
2. [done] Measured the true gap on 1×1/3×3/strided/padded — see tables below. The headline "~50% of
   cuDNN" from the first cut was a **measurement phantom** (single-shot cuDNN timing); fair `best_of`
   both → parity-to-win everywhere bar one soft shape.
3. [done] Implicit-GEMM improved: static-shape specialization + **split-K** (valid *and* affine) for
   the occupancy-starved deep-channel shapes. (cp.async/db-pipe tried, neutral here — L2-resident.)
4. [done] Winograd F(2×2,3×3) + F(4×4,3×3), batched α²-plane GEMM (`ptx_winograd.rs`), rel-Frobenius gate.
5. [done] Fused conv+bias+act epilogue (`conv2d_wmma_epi`) vs the unfused chain — see §Fused.
6. [done] Coverage: 1×1 (=GEMM), 3×3, **strided** (`conv2d_wmma_strided`), **zero-padded "same"**
   (`conv2d_wmma_padded` bounds-checked + `conv2d_wmma_padded_explicit` scatter + `conv2d_wmma_padded_auto`
   split-K). depthwise/grouped/dilated = the documented next bottleneck (§Coverage).

## The cuDNN peer (reproducible recipe)

cudarc 0.16.6 ships a full `cudnn` module (`safe`/`result`/`sys`) gated only on `driver`; `dynamic-
loading` dlopens `cudnn64_9.dll` automatically (the `_9` candidate is in `get_lib_name_candidates`, so
no rename needed). Added `"cudnn"` to the cudarc feature list in `mercury_codegen_gpu/Cargo.toml`.

Redist install (all four together — pip `--target` clobbers the shared `nvidia/` namespace if cuDNN
goes in separately):

```
pip install --target tools/cuda-redist --no-deps \
  nvidia-cuda-nvrtc-cu12==12.9.86 nvidia-cublas-cu12==12.9.2.10 \
  nvidia-nvjitlink-cu12==12.9.86 nvidia-cudnn-cu12==9.10.2.21
```

cuDNN 9 ships a thin loader `cudnn64_9.dll` + 7 sublibraries (`cudnn_graph/ops/cnn/adv/
engines_precompiled/engines_runtime_compiled/heuristic64_9.dll`); the legacy conv path also hard-needs
`cublasLt64_12.dll` (ships in the cublas wheel) and nvrtc/nvjitlink. All on PATH:

```
$r="<repo>/tools/cuda-redist/nvidia"
$env:PATH="$r/cudnn/bin;$r/cublas/bin;$r/cuda_nvrtc/bin;$r/nvjitlink/bin;"+$env:PATH
```

The legacy forward API (`cudnnConvolutionForward` + `cudnnGetConvolutionForwardAlgorithm_v7` +
`cudnnGetConvolutionForwardWorkspaceSize`) is **deprecated but present** in cuDNN 9. Fast fp16 path:
`CUDNN_TENSOR_NHWC` for x/w/y, fp16 data (C,K mult of 8), conv compute `f32`, math type
`CUDNN_TENSOR_OP_MATH`; the v7 heuristic typically returns `IMPLICIT_PRECOMP_GEMM` (3×3/1×1) or
`WINOGRAD_NONFUSED` (large 3×3). Mercury stores NCHW, so X/W are transposed to NHWC/KRSC once at setup
(outside the timed loop) and cuDNN's NHWC output is transposed back to `[K,P,Q]` for the same f64
cross-check Mercury faces.

Run: `cargo test -p mercury_codegen_gpu --features gpu --release -- --ignored --nocapture conv_vs_cudnn`.

## Measurements

All **same-run** (one process, shared clock), each kernel cross-checked vs the f64 oracle + checksum
before timing, each timed `best_of(ROUNDS)` (the fix below). RTX 4050 Laptop, fp16 tensor-core path.

### The measurement-integrity fix (this is load-bearing)

The first cut timed cuDNN **once** (a single 100-iter window) while Mercury got `best_of(ROUNDS)`. A
single thermal dip in cuDNN's lone window tanked its number and the gap swung **~7× run-to-run** — one
run read "Mercury 50% of cuDNN", the next "Mercury 343%". That 50% gap was a **phantom**. Wrapping
cuDNN in `best_of(ROUNDS)` (the same robustness Mercury and the GEMM scoreboard's cuBLAS peer already
get) is what makes the ratio trustworthy. Even so, the cuDNN/Mercury ratio still carries ±~25%
run-to-run clock noise (the two kernels respond differently to the mobile boost ramp), so the honest
unit is a **range over ≥4 reruns**, not a point.

### implicit-GEMM conv vs cuDNN (4 reruns, fair `best_of` both; cuDNN algo = IMPLICIT_PRECOMP_GEMM)

| shape (C,H,W,K,R,S)        | Mercury % of cuDNN (range) | × vs naive CUDA-C | verdict |
|----------------------------|----------------------------|-------------------|---------|
| C64 56² K64 **3×3**        | 93–103 %                   | 6.6–8.2×          | **parity** |
| C128 28² K128 **3×3**      | 95–121 %                   | 7.8–8.6×          | **parity / slight win** |
| C256 14² K256 **3×3** (sk8)| 92–107 %                   | 6.9–9.1×          | **parity** |
| C64 56² K64 **1×1**        | **352–564 %**              | 6.4–10.8×         | **decisive win** |
| C32 32² K32 **5×5**        | 93–114 %                   | 1.5–1.9×          | **parity** |
| C3 64² K64 **3×3** (1st)   | **450–564 %**              | 1.9–2.3×          | **decisive win** |

**Read:** Mercury's static-shape-specialized implicit-GEMM **matches** cuDNN's hand-tuned
IMPLICIT_PRECOMP_GEMM on the bread-and-butter deep-channel 3×3/5×5 (within clock noise), and **beats it
3.5–5.6×** on 1×1 (pointwise) and the C=3 first layer, where cuDNN's generic conv path carries launch /
generality overhead the baked-in static shape (fully-unrolled window, no dynamic bounds) avoids. Split-K
supplies the occupancy for the deep-channel shapes (C128→sk4, C256→sk8). Matching a shipping vendor
library from JIT-compiled PTX with no CUDA toolkit is the headline. **cuDNN caveat (disclosed):** the
peer is the legacy v7 forward API (`cudnnConvolutionForward` + v7 algo heuristic), the standard "use
cuDNN for conv" path; cuDNN's v8 graph API may pick a faster engine on some shapes — a fair Tier-B, not
a claim against cuDNN's absolute ceiling.

### Levers tried

* **Split-K** (shipped) — flips the deep-channel occupancy starvation; C256 14² base grid is only 12
  CTAs over 20 SMs, sk8 → ~96 CTAs ≈ one wave. Dispatch gated on a ~2-wave threshold so it never
  over-splits a near-full grid (measured slower).
* **Register double-buffered pipeline** (gated, NOT shipped) — **neutral** (same-run db-pipe A/B:
  0.84–1.22×, mean ~1.0; *worse* on the split-K shapes, where the prefetch registers cut occupancy).
  These convs are L2-resident, so the staging stall is L2- not HBM-latency, which a depth-1 register
  prefetch doesn't move. Honest negative result, kept as a documented A/B.

### Winograd F(4×4,3×3) — the 3×3 lever (shipped, gated, a win)

GPU Winograd (cuDNN's `WINOGRAD_NONFUSED` strategy): 4 resident phases — filter transform `U[α²,K,C]`,
input transform `V[α²,C,T]`, the **α² channel-reduction GEMMs `M[ξν]=U[ξν]·V[ξν]` batched into ONE
launch** (`gridDim.z=α²`), output transform → `O`. The three transforms are constant sparse linear maps
(unrolled FMA chains, f32 math / fp16 storage). Gated vs the f64 oracle by **relative-Frobenius**
(per-element rel is meaningless — the inverse transform makes many near-zero cancellation outputs):
**fro_rel ≈ 5.4e-4 (F(2,3)) / 2.8e-3 (F(4,3))**, stable across shapes, the ~5× ratio matching F(4,3)'s
amplification — ≫ a real-bug threshold (~1e-1).

**The batching lever (decisive).** The first cut launched the α² GEMMs in a loop: each fills only
`ceil(T/BN)·ceil(K/BM)` ≈ 4 CTAs and runs serially → the 20-SM GPU sits ~95% idle and Winograd was
**4–25× *slower* than the implicit-GEMM** (the very reason cuDNN's heuristic also picks
IMPLICIT_PRECOMP_GEMM, not Winograd, on these). Folding the α² planes into `gridDim.z` (one launch,
`α²·4` CTAs in flight) is a **20–50× speedup** and flips it to a win.

Same-run, large feature maps, F(4×4,3×3), 3 reruns. The **Winograd-vs-implicit-GEMM ratio is the
reliable number** (both are Mercury kernels in the same run → cancels the clock); cuDNN % is the noisier
cross-family ratio.

| shape (C,H,W,K) 3×3 | tiles T | Winograd ÷ implicit-GEMM (3 runs) | Winograd % of cuDNN |
|---------------------|---------|-----------------------------------|---------------------|
| C64 56²  K64        | 196     | 1.21 / 1.15 / 1.34 → **~1.2×**     | 82–102 %            |
| C32 64²  K64        | 256     | 0.78 / 0.76 / 0.76 → **0.77×**     | 77–89 %             |
| C64 112² K64        | 784     | 2.02 / 2.00 / 2.06 → **~2.0×**     | 76–77 %             |
| C128 28² K128       | 49      | 2.05 / 2.24 / 2.28 → **~2.2×**     | 112–176 %           |

**Read:** batched Winograd is **~1.2–2.2× Mercury's own (already cuDNN-parity) implicit-GEMM** on 3 of 4
large feature maps, and beats cuDNN outright on C64 56² and C128 28². It **loses only at low channel
count** (C32, 0.77×): the batched-GEMM reduction is just `GK=C=32` (≈2 WMMA k-steps, low intensity)
whereas the implicit-GEMM reduces over the full `GK=C·9=288`. So the win is channel-count-gated —
Winograd for ≥64-channel large-spatial 3×3, implicit-GEMM otherwise (a per-shape dispatch is the
follow-up). Absolute throughput up to **~9.7 TFLOP/s** (C64 112²). F(2,3) (lower amplification) and
F(4,3) (cuDNN-grade) both gated.

## Affine conv: strided + zero-padded "same" (shipped, gated, vs cuDNN at matching pad/stride)

The real CNN convs aren't valid-only — they downsample (`stride>1`) and pad ("same"). Both are threaded
through the *one* implicit-GEMM kernel (`conv_wmma_ptx_impl`, a `stride`+`pad` arg each):

* **Strided** (`conv2d_wmma_strided`) — output `P=⌊(H−R)/s⌋+1`; the hoisted im2col `xpart` scales by
  `stride`. `stride=1` is byte-identical to the dense kernel. Gated stride 2/3 over 3×3/5×5/1×1 + a
  non-divisible 31² — **bit-close** (max_abs 3.4–5.5e-3 vs f64).
* **Zero-padded** (`conv2d_wmma_padded`) — output `P=⌊(H+2·pad−R)/s⌋+1`; gather reads `(p·s+r−pad,
  q·s+s−pad)`, OOB→0. `pad>0` switches the gather from the linear `xpart` fold to a **single unsigned
  bounds compare per axis** (`0≤i<N ⟺ (u32)i<N`; the fold would row-wrap on OOB). Gated 3×3 pad-1 "same",
  ResNet 3×3 s2 p1 + 7×7 s2 p3 stem, 5×5 pad-2 — **bit-close** (max_abs 4.0–7.0e-3).
* Two padded implementations, measured **same-run A/B** (`conv_affine_vs_cudnn`): the single-kernel
  bounds-checked gather vs **explicit-pad** (`conv2d_wmma_padded_explicit`: scatter X into a zeroed
  `(H+2p)×(W+2p)` buffer, then the dense valid kernel — no per-tap predication). Reliable Mercury-vs-
  Mercury, stable across 3 reruns: **the bounds-checked single kernel is ~1.0–1.12× *faster*** — the
  scatter costs slightly more than the cheap bounds chain it removes (L2-resident convs). So the single
  kernel is the default; explicit-pad stays gated (and as the natural split-K seam).

### Split-K for affine — closes the deep-channel small-spatial gap (the one real residual)

The downsamples starve the SMs: C128 28²→14² is only **~8 CTAs** over 20 SMs. `conv_wmma_ptx_impl`
already composes `sk`+`pad` (the per-tap bounds check is K-slice-independent), so `conv2d_wmma_padded_auto`
exposes it — `conv_splitk_factor_affine` applies the same occupancy heuristic over the *downsampled*
output `P·Q`, and dispatches the single kernel or split-K + the deterministic reduce. **Reliable same-run
speedup over the single padded kernel** (auto ÷ explicit-single, stable across reruns):

| shape (C,H,W,K) 3×3, s/pad        | sk | auto ÷ single (same-run) | % of cuDNN (aligned clock) |
|-----------------------------------|----|--------------------------|----------------------------|
| C64 56² K64 **s1 p1** ("same")    | 1  | ~1.1× (=single)          | ~parity (51–134 %, noisy)  |
| C128 28² K128 **s1 p1** ("same")  | 3  | **~1.8×**                | 51–108 % (parity at clock) |
| C64 56² K64 **s2 p1** (downsample)| 6  | **~1.4–2.2×**            | 119–286 % (**win**)        |
| C128 28² K128 **s2 p1** (downsmp) | 8  | **~2.3–3.1×**            | 72–263 % (**parity-win**)  |
| C3 224² K64 **s2 p3** (stem)      | 1  | ~1.0× (=single)          | 96–118 % (**parity**)      |
| C64 56² K64 **1×1** (pointwise)   | 1  | ~1.0× (=single)          | 120–440 % (**decisive win**)|

**Read:** split-K delivers a clean **1.4–3.1× same-run** lift on the occupancy-starved downsamples and
moves the C128 s2p1 target from ~60 % to parity-to-win. The auto gate exercises sk 1/3/6/8, all bit-close
to the f64 oracle (max_abs ≤1.8e-2, the looser split-K-reduce tolerance). **Measurement caveat, reaffirmed
hard:** the cuDNN cross-family % is *clock-noise-dominated* — C128 s1p1 read **51 % then 108 % on byte-
identical code** between two back-to-back reruns (one caught cuDNN's window at high boost, Mercury's at
low). Only the **same-run Mercury-vs-Mercury** ratio (split-K ÷ single, cuDNN ÷ nothing) is trustworthy;
the % column is a range over reruns, never a point. The single genuinely-soft shape is **3×3 s1 p1 C64
"same"** at mid-size — cuDNN's IMPLICIT_PRECOMP_GEMM is strong there and the base grid (~49 CTAs) is
already full so split-K can't help; padded-Winograd (≥64-ch lever, currently valid-only) is the follow-up.

## Fused conv+bias+act epilogue (shipped, gated)

`conv2d_wmma_epi` folds a per-output-channel bias add + activation (ReLU/SiLU/GELU) into the implicit-GEMM
**store epilogue** — it runs on the f32 accumulators already in the smemC store-back
(`conv_wmma_epi_ptx`), racing cuDNN's `cudnnConvolutionBiasActivationForward`. Gated host-side (bias+act
applied to the plain-conv oracle). Same-run vs the **unfused chain** (plain conv + a separate `bias_relu`
pointwise pass = the extra `K·P·Q` HBM round-trip + launch that fusion elides), `conv_fused_epi_vs_unfused`:

| shape          | fused ÷ unfused | note |
|----------------|-----------------|------|
| 3×3 (compute-bound) | **1.0–1.03×** | conv dominates; the epilogue is ~free but the saved pass is a small fraction |
| 1×1 (memory-bound)  | **1.06–1.22×** | the elided HBM round-trip is a bigger share → the real win |

**Read:** the epilogue itself is **free** (a few instructions on registers already live), so fusion's
value is exactly the round-trip it removes — largest on the memory-bound 1×1 (where the separate pass is a
big fraction of the conv), marginal on the compute-bound 3×3. Honest, modest, in the right direction.

## Coverage & the next bottleneck

**Shipped & gated:** 1×1 (=GEMM, decisive win), 3×3 valid (parity-to-win), 5×5, 7×7 stem, **strided 2/3**,
**zero-padded "same"/downsample** (bounds-checked + explicit-pad + split-K), **fused bias+act**, **Winograd
F(2,3)+F(4,3)**, and a **`conv2d_best` per-shape dispatch** that auto-routes Winograd↔implicit↔split-K↔
affine (Winograd for valid 3×3 ≥64-ch large-spatial — the 1.2–2.2× lane — GEMM elsewhere; gate routes
through every lane, rel-Frobenius ≤8e-3). Every path bit-/Frobenius-close to the f64 oracle; `cargo test`
(no `gpu`) stays green (all GPU code behind `#[cfg(feature="gpu")]`).

**Next bottleneck, in priority order:**
1. **Dilated** — cheapest: the gather just scales the tap by `dilation` (`ih=p·s+r·d−pad`), a one-line
   change to the same kernel (atrous/segmentation convs).
2. **Depthwise / grouped** — a different kernel *structure* (per-group GEMM, no cross-channel reduction;
   depthwise has `GK=R·S` so it's bandwidth-bound, not tensor-core-bound) → a dedicated path, not a flag.
3. **Padded Winograd** — the lever for the soft 3×3 s1 p1 "same" mid-size shapes (Winograd is currently
   valid-only; padding it would let the ≥64-ch 3×3 win extend to "same" convs, and let `conv2d_best`
   route "same" 3×3 to Winograd too).
4. ~~Per-shape Winograd↔implicit dispatch~~ — **done** (`conv2d_best`).

## Winograd reference (Lavin & Gray 2016, wincnn convention — for `ptx_winograd.rs`)

**F(2×2,3×3)**, 4-pt tile, points {0,1,−1,∞}; `Y = Aᵀ[(G g Gᵀ) ⊙ (Bᵀ d B)]A`. 16 mults / 2×2 tile =
4/output vs 9 direct → **2.25× multiply reduction**.

```
Bᵀ = [ 1  0 -1  0 ;  0  1  1  0 ;  0 -1  1  0 ;  0 -1  0  1 ]   (4×4)
G  = [ 1  0  0 ;  1/2  1/2  1/2 ;  1/2 -1/2  1/2 ;  0  0  1 ]   (4×3)
Aᵀ = [ 1  1  1  0 ;  0  1 -1  1 ]                               (2×4)
```

**F(4×4,3×3)**, 6-pt tile, points {0,1,−1,2,−2,∞}; 6×6 tiles. 36 mults / 4×4 tile = 2.25/output vs 9
direct → **4× multiply reduction**. (The one cuDNN/most libs use.)

```
Bᵀ = [ 4  0 -5  0  1  0 ;  0 -4 -4  1  1  0 ;  0  4 -4 -1  1  0 ;
       0 -2 -1  2  1  0 ;  0  2 -1 -2  1  0 ;  0  4  0 -5  0  1 ]   (6×6, integers)
G  = [ 1/4 0 0 ; -1/6 -1/6 -1/6 ; -1/6 1/6 -1/6 ;
       1/24 1/12 1/6 ; 1/24 -1/12 1/6 ; 0 0 1 ]                    (6×3, fractions)
Aᵀ = [ 1  1  1  1  1  0 ;  0  1 -1  2 -2  0 ;
       0  1  1  4  4  0 ;  0  1 -1  8 -8  1 ]                       (4×6, integers)
```

**Sign-convention trap:** Bᵀ and Aᵀ come in self-consistent sets — do NOT mix the wincnn convention
above with the Lavin&Gray scalar-form (∞-row signs differ); G is identical either way.

**Batched-GEMM channel reduction:** transform all input tiles (V=BᵀdB, scatter into `V^(ξ,ν)[c,b]`) and
all filters (U=GgGᵀ, scatter into `U^(ξ,ν)[k,c]`), then the channel sum is **α² independent GEMMs**
`M^(ξ,ν)=U^(ξ,ν)·V^(ξ,ν)` of shape `[K×C]·[C×P]→[K×P]` (P = tiles); the tile-position (ξ,ν) is the batch
dim. 16 GEMMs for F(2,3), 36 for F(4,3). Each is a dense M=K,N=P,K-contract=C matmul → tensor cores.

**Tolerance (fp16-in / f32-accumulate F(4,3) vs f64 direct):** fp16 input quantization (~1e-3)
dominates — F(4,3) amplification (~4–7×) and f32 accumulation (negligible) sit under it. Gate
**rel-Frobenius ≤ 2e-3**; per-element backstop **rtol 1e-2 + atol ≈ 1e-2·max|Y|** (Winograd has lots of
cancellation). A real bug lands at 1e-1…1e0, far above the gate. Prefer F(4,3); standard-points F(6,3)
collapses in fp16.
