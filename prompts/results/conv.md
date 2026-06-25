# Conv2d vs cuDNN — implicit-GEMM + Winograd, and a real cuDNN peer

Branch `perf/gpu-conv-2` (worktree `Mercury-conv2`). Target: mobile **RTX 4050, sm_89 (Ada)**, driver
592.27, `cudarc` 0.16.6, no CUDA toolkit (PTX driver-JIT + dlopen'd redist DLLs).

Goal: stand up a **real cuDNN peer** (the prior conv work measured only vs naive CUDA-C ≈6×, which
proves nothing about the true gap), then close — and beat — cuDNN on the common conv shapes via
implicit-GEMM, Winograd, and fused conv+bias+act.

## Status / plan

1. [done] cuDNN Tier-B peer bound (`baselines.rs::cudnn_conv2d_run` / `time_cudnn_conv2d`), NHWC fp16
   tensor-core fast path, v7-heuristic-chosen engine disclosed. Bench `gpu.rs::conv_vs_cudnn`.
2. [in progress] Measure the true gap: existing `conv_wmma` implicit-GEMM vs cuDNN on 1×1/3×3/strided.
3. [todo] Improve the implicit-GEMM toward cuDNN (cp.async pipeline, mma.sync, raster, static unroll).
4. [todo] Winograd F(2×2,3×3) + F(4×4,3×3) for 3×3 stride-1 (`ptx_winograd.rs`), looser gate.
5. [todo] Fused conv+bias+act epilogue — beat the cuDNN unfused chain (and race cuDNN's *fused*
   `cudnnConvolutionBiasActivationForward`).
6. [todo] Coverage: 1×1 (=GEMM), strided, padded; depthwise/grouped/dilated as stretch.

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

TBD — same-run, checksum-cross-checked, %-of-cuDNN + ×-vs-naive, ≥3 reruns, cuDNN algo disclosed.

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
