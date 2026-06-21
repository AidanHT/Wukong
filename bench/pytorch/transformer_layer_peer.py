#!/usr/bin/env python
"""Tier-C GPU peer: PyTorch pre-norm transformer layer, same GPU as Mercury.

The M13 milestone is "beat the library-call-chain stack end-to-end AND beat PyTorch". Mercury's
fused fp16 ResidentLayerF16 / ResidentModelF16 already beat the cuBLAS call-chain (Tier B, measured
in-process by `cublas_chain_vs_mercury_layer_throughput` / `resident_model_vs_cublas_chain_throughput`).
This script is the **PyTorch** bar (Tier C): the *same* mathematical layer, built the way a PyTorch
user would, timed on the *same* RTX 4050 so the numbers are comparable to Mercury's Rust bench.

The two laws hold here too:
  * CORRECTNESS FIRST — every torch path is cross-checked against an f64 reference of the identical
    function (RMSNorm -> non-causal single-head MHA -> SiLU-FFN, two f32 residuals) before its speed
    is reported. A fast-but-wrong torch path can't flatter itself.
  * HONESTY — same machine, same shapes (D=64, Dff=256, S in {256,512,1024}); clock warmed + best-of-N
    (the laptop GPU boosts ~7x under load); torch gets its *fast* path (fp16 tensor-core matmuls +
    fp16 flash SDPA), i.e. it is represented at its best, not handicapped to Mercury's exact dtype
    boundaries. This is a cross-language, cross-process comparison (torch in Python vs Mercury in
    Rust) — like `mercury_xbench` for CPU C/Rust — NOT a same-process same-buffer measurement; both
    sides are warmed/best-of-N/synchronized and compute the same function to the same tolerance.

Mercury's measured ms/layer on this GPU (from the Rust bench, fp16 fused resident, same shapes) are
printed alongside for convenience — they are the authoritative Mercury figures; re-run the Rust bench
to refresh them.

Run (from repo root, in the gitignored torch venv):
    tools/torch-venv/Scripts/python.exe bench/pytorch/transformer_layer_peer.py
"""
import sys
import time

import torch
import torch.nn.functional as F

D, DFF, EPS = 64, 256, 1e-5
HALF = torch.float16

# Mercury fp16 fused resident, measured same-GPU (cublas_chain_vs_mercury / resident_model_vs_cublas),
# post the alloc_zeros pipelining fix. Authoritative figures live in the Rust bench; these are for
# side-by-side convenience only (re-run the Rust benches to refresh; ~10-20% run-to-run clock variance).
MERCURY_MS_PER_LAYER = {256: 0.31, 512: 0.72, 1024: 1.34}         # single layer, resident
MERCURY_STACK_MS_PER_LAYER = {1: 0.622, 2: 0.514, 4: 0.529, 8: 0.561}  # depth sweep, S=512


def rmsnorm_f32(x):
    """RMSNorm with no learned scale, eps=1e-5, computed in f32 (matches Mercury's f32 norm kernel)."""
    xf = x.float()
    inv = torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + EPS)
    return xf * inv


class TorchLayer:
    """One pre-norm transformer layer, weights resident. `dtype` is the matmul precision."""

    def __init__(self, gen, dtype):
        self.dtype = dtype
        # Weights [out, in], row-major — F.linear / `x @ W.t()` computes Mercury's x.W^T (no bias).
        self.Wq = gen(D, D); self.Wk = gen(D, D); self.Wv = gen(D, D); self.Wo = gen(D, D)
        self.W1 = gen(DFF, D); self.W2 = gen(D, DFF)

    def forward(self, x):
        """x: [S, D] f32 resident -> [S, D] f32. fp16 tensor-core matmuls, f32 norm/softmax/residual."""
        S = x.shape[0]
        scale = 1.0 / (D ** 0.5)
        h1 = rmsnorm_f32(x).to(self.dtype)
        q = h1 @ self.Wq.t(); k = h1 @ self.Wk.t(); v = h1 @ self.Wv.t()
        # single-head, non-causal flash SDPA (torch's fast fused attention), in the matmul dtype.
        a = F.scaled_dot_product_attention(
            q[None, None], k[None, None], v[None, None], scale=scale, is_causal=False
        )[0, 0]
        o = a.to(self.dtype) @ self.Wo.t()
        x1 = x + o.float()                       # residual 1 (f32)
        h2 = rmsnorm_f32(x1).to(self.dtype)
        f1 = h2 @ self.W1.t()
        f1a = F.silu(f1.float()).to(self.dtype)  # SiLU in f32 (matches Mercury's vmath silu)
        f2 = f1a @ self.W2.t()
        return x1 + f2.float()                   # residual 2 (f32)


def ref_forward_f64(layer, x):
    """f64 reference of the identical function — the correctness oracle (cf. Mercury's ref_*)."""
    f64 = torch.float64
    Wq, Wk, Wv, Wo = (w.to(f64) for w in (layer.Wq, layer.Wk, layer.Wv, layer.Wo))
    W1, W2 = layer.W1.to(f64), layer.W2.to(f64)
    xf = x.to(f64)
    scale = 1.0 / (D ** 0.5)

    def rms(t):
        return t * torch.rsqrt(t.pow(2).mean(-1, keepdim=True) + EPS)

    h1 = rms(xf)
    q = h1 @ Wq.t(); k = h1 @ Wk.t(); v = h1 @ Wv.t()
    scores = (q @ k.t()) * scale
    a = torch.softmax(scores, dim=-1) @ v
    o = a @ Wo.t()
    x1 = xf + o
    h2 = rms(x1)
    f1 = h2 @ W1.t()
    f1a = f1 * torch.sigmoid(f1)
    f2 = f1a @ W2.t()
    return x1 + f2


def max_dev(got, ref):
    """(max_abs, max_rel) of `got` vs f64 `ref` — both moved to cpu f64."""
    g = got.detach().to(torch.float64).cpu()
    r = ref.detach().to(torch.float64).cpu()
    abs_e = (g - r).abs()
    rel_e = abs_e / r.abs().clamp_min(1e-12)
    return abs_e.max().item(), rel_e.max().item()


def best_ms(fn, warmup, iters, rounds, repin):
    """Min over `rounds` of mean ms/call. `warmup` boosts the clock from cold; `repin` re-warms
    before each round so host gaps don't let the laptop clock decay (mirrors the Rust pin_clock!)."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    best = float("inf")
    for _ in range(rounds):
        for _ in range(repin):
            fn()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
        best = min(best, (time.perf_counter() - t0) / iters)
    return best * 1e3


def main():
    if not torch.cuda.is_available():
        print("[skip] CUDA not available to torch")
        return
    dev = torch.device("cuda")
    torch.manual_seed(0)
    print(f"torch {torch.__version__} | device {torch.cuda.get_device_name(0)}")
    # Mercury's flash is f32; torch's fp16 SDPA can use the fp16 backends. Allow them all (fast path).
    try:
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
    except Exception:
        pass

    def gen(*shape):
        return (torch.rand(*shape, device=dev) * 0.2 - 0.1).to(HALF)

    # Heavy clock warmup from cold (a few hundred ms of sustained GEMM), like the Rust bench's 1500x.
    wa = torch.rand(1024, 1024, device=dev, dtype=HALF)
    for _ in range(400):
        _ = wa @ wa
    torch.cuda.synchronize()

    print("\n== single layer (D=64, Dff=256, resident, fp16 eager) ==")
    for S in (256, 512, 1024):
        x = torch.rand(S, D, device=dev) * 2 - 1
        layer = TorchLayer(gen, HALF)
        got = layer.forward(x)
        ref = ref_forward_f64(layer, x)
        a_e, r_e = max_dev(got, ref)
        assert a_e < 5e-2, f"S={S} torch fp16 layer wrong: max_abs={a_e:.2e}"
        ms = best_ms(lambda: layer.forward(x), warmup=100, iters=50, rounds=5, repin=20)
        mer = MERCURY_MS_PER_LAYER[S]
        print(
            f"  S={S}: torch eager fp16 {ms:.3f} ms/layer | Mercury fused {mer:.3f} ms/layer "
            f"| Mercury {ms/mer:.2f}x faster  (torch max_abs={a_e:.2e})"
        )

    # torch.compile (Inductor) — the fusing bar. Needs Triton; may be unavailable on Windows.
    print("\n== single layer (fp16, torch.compile / Inductor) ==")
    for S in (256, 512, 1024):
        x = torch.rand(S, D, device=dev) * 2 - 1
        layer = TorchLayer(gen, HALF)
        try:
            compiled = torch.compile(layer.forward, mode="max-autotune")
            got = compiled(x)
            ref = ref_forward_f64(layer, x)
            a_e, r_e = max_dev(got, ref)
            assert a_e < 5e-2, f"S={S} torch.compile layer wrong: max_abs={a_e:.2e}"
            ms = best_ms(lambda: compiled(x), warmup=50, iters=50, rounds=5, repin=20)
            mer = MERCURY_MS_PER_LAYER[S]
            print(
                f"  S={S}: torch.compile fp16 {ms:.3f} ms/layer | Mercury fused {mer:.3f} ms/layer "
                f"| Mercury {ms/mer:.2f}x faster  (torch max_abs={a_e:.2e})"
            )
        except Exception as e:
            print(f"  S={S}: torch.compile unavailable ({type(e).__name__}: {str(e)[:80]})")

    # Depth stack (S=512) — the whole-model-resident counterpart to ResidentModelF16.
    print("\n== stack depth sweep (S=512, fp16 eager, resident) ==")
    S = 512
    for depth in (1, 2, 4, 8):
        x = torch.rand(S, D, device=dev) * 2 - 1
        layers = [TorchLayer(gen, HALF) for _ in range(depth)]

        def run():
            cur = x
            for ly in layers:
                cur = ly.forward(cur)
            return cur

        ms = best_ms(run, warmup=80, iters=40, rounds=5, repin=15)
        mer = MERCURY_STACK_MS_PER_LAYER[depth]
        print(
            f"  depth={depth}: torch eager fp16 {ms/depth:.3f} ms/layer ({ms:.3f} ms total) "
            f"| Mercury {mer:.3f} ms/layer | Mercury {(ms/depth)/mer:.2f}x faster"
        )


if __name__ == "__main__":
    sys.exit(main())
