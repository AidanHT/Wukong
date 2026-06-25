#!/usr/bin/env python3
"""Fused FlashAttention-2 peer for Mercury's GPU flash-attention benchmark.

Drives PyTorch's *fused* `scaled_dot_product_attention` backends — FLASH_ATTENTION
(FlashAttention-2), EFFICIENT_ATTENTION (cutlass mem-efficient fMHA), and
CUDNN_ATTENTION (cuDNN's fused flash) — each a genuinely fused FA-class kernel on
Ada sm_89, plus MATH (the unfused materialized softmax chain) as an in-process
anchor. It runs them over the SAME Q/K/V buffers Mercury's `flash_d64_mp` runs, so
the timing is a same-shape ratio and the output is cross-checked by the Rust caller
against the same CPU f64 reference Mercury's flash is gated against.

Q/K/V are read as raw little-endian float16, shape [B,H,S,D] row-major (head-major,
the layout Mercury's multi-head flash uses). O of the fastest successful *fused*
backend is written as raw little-endian float32. A JSON report (per-backend sec/iter
+ output checksum + the chosen fused backend) is printed to stdout.

This is the honest peer the mission demands: a *named, genuinely fused* FA2-class
kernel, not the pre-FlashAttention unfused cuBLAS chain.
"""
import argparse
import json
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--q", required=True)
    ap.add_argument("--k", required=True)
    ap.add_argument("--v", required=True)
    ap.add_argument("--o", required=True, help="output O path (f32) of the chosen fused backend")
    ap.add_argument("--report", default=None, help="optional flat key=value report file for the Rust caller")
    ap.add_argument("--B", type=int, default=1)
    ap.add_argument("--H", type=int, default=1)
    ap.add_argument("--S", type=int, required=True)
    ap.add_argument("--D", type=int, required=True)
    ap.add_argument("--scale", type=float, required=True)
    ap.add_argument("--causal", type=int, default=0)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--dtype", default="f16", choices=["f16", "bf16"])
    args = ap.parse_args()

    import numpy as np
    import torch
    import torch.nn.functional as F
    from torch.nn.attention import SDPBackend, sdpa_kernel

    if not torch.cuda.is_available():
        print(json.dumps({"error": "cuda not available", "torch": torch.__version__}))
        sys.exit(3)

    B, H, S, D = args.B, args.H, args.S, args.D
    n = B * H * S * D
    npdt = np.float16  # inputs are always dumped as f16 by the Rust caller
    tdt = torch.float16 if args.dtype == "f16" else torch.bfloat16

    def load(path):
        a = np.fromfile(path, dtype=npdt)
        if a.size != n:
            raise SystemExit(f"{path}: {a.size} elems != expected {n} ({B}x{H}x{S}x{D})")
        return torch.from_numpy(a.reshape(B, H, S, D).copy()).to(device="cuda", dtype=tdt)

    q, k, v = load(args.q), load(args.k), load(args.v)
    causal = bool(args.causal)
    scale = float(args.scale)

    def time_backend(backend):
        with sdpa_kernel(backend):
            o = None
            for _ in range(args.warmup):
                o = F.scaled_dot_product_attention(
                    q, k, v, attn_mask=None, dropout_p=0.0, is_causal=causal, scale=scale)
            torch.cuda.synchronize()
            best = float("inf")
            for _ in range(args.runs):
                s_ev = torch.cuda.Event(enable_timing=True)
                e_ev = torch.cuda.Event(enable_timing=True)
                s_ev.record()
                for _ in range(args.iters):
                    o = F.scaled_dot_product_attention(
                        q, k, v, attn_mask=None, dropout_p=0.0, is_causal=causal, scale=scale)
                e_ev.record()
                torch.cuda.synchronize()
                best = min(best, s_ev.elapsed_time(e_ev) / 1000.0 / args.iters)
            return best, o

    candidates = [
        ("flash", SDPBackend.FLASH_ATTENTION),
        ("efficient", SDPBackend.EFFICIENT_ATTENTION),
        ("cudnn", SDPBackend.CUDNN_ATTENTION),
        ("math", SDPBackend.MATH),  # unfused, in-process anchor (NOT a fused peer)
    ]
    report = {
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "dtype": args.dtype,
        "shape": {"B": B, "H": H, "S": S, "D": D},
        "causal": causal,
        "scale": scale,
        "backends": {},
    }
    best_fused = None  # (name, sec, O tensor)
    for name, be in candidates:
        try:
            sec, o = time_backend(be)
            report["backends"][name] = {
                "sec": sec,
                "checksum": float(o.float().sum().item()),
                "gflops": (4.0 * B * H * S * S * D) / sec / 1e9,
            }
            if name != "math" and (best_fused is None or sec < best_fused[1]):
                best_fused = (name, sec, o)
        except Exception as ex:  # noqa: BLE001 — report, don't crash; a backend may be unsupported
            report["backends"][name] = {"error": f"{type(ex).__name__}: {str(ex)[:240]}"}

    if best_fused is not None:
        report["chosen"] = best_fused[0]
        report["chosen_sec"] = best_fused[1]
        o = best_fused[2].float().contiguous().cpu().numpy().astype("<f4")
        o.tofile(args.o)
    else:
        report["chosen"] = None

    if args.report is not None:
        # Flat key=value report so the Rust caller parses without a JSON dependency.
        lines = [
            f"device={report['device']}",
            f"torch={report['torch']}",
            f"cuda={report['cuda']}",
            f"chosen={report['chosen']}",
        ]
        for name, _ in candidates:
            b = report["backends"].get(name, {})
            if "sec" in b:
                lines.append(f"{name}_sec={b['sec']:.9e}")
                lines.append(f"{name}_checksum={b['checksum']:.6f}")
                lines.append(f"{name}_gflops={b['gflops']:.3f}")
            else:
                lines.append(f"{name}_error={b.get('error', 'missing')}")
        with open(args.report, "w") as fh:
            fh.write("\n".join(lines) + "\n")

    print(json.dumps(report))


if __name__ == "__main__":
    main()
