#!/usr/bin/env python3
"""Fused FlashAttention-2 peer for Wukong's GPU flash-attention benchmark.

Drives PyTorch's *fused* `scaled_dot_product_attention` backends — FLASH_ATTENTION
(FlashAttention-2), EFFICIENT_ATTENTION (cutlass mem-efficient fMHA), and
CUDNN_ATTENTION (cuDNN's fused flash) — each a genuinely fused FA-class kernel on
Ada sm_89, plus MATH (the unfused materialized softmax chain) as an in-process
anchor. It runs them over the SAME Q/K/V buffers Wukong's `flash_d64_mp` runs, so
the timing is a same-shape ratio and the output is cross-checked by the Rust caller
against the same CPU f64 reference Wukong's flash is gated against.

Q/K/V are read as raw little-endian float16, shape [B,H,S,D] row-major (head-major,
the layout Wukong's multi-head flash uses). O of the fastest successful *fused*
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
    ap.add_argument("--rope", type=int, default=0,
                    help="if 1, the peer is the HONEST RoPE path: an optimized interleaved-RoPE kernel "
                         "over Q,K (what a model must run because the fused-attention library can't "
                         "absorb RoPE) + SDPA, timed as one pipeline. {backend}_sec then measures "
                         "rope+sdpa; {backend}_sdpa_sec the sdpa-only reference.")
    ap.add_argument("--cos", default=None, help="cos table [S, D/2] f32 (required with --rope)")
    ap.add_argument("--sin", default=None, help="sin table [S, D/2] f32 (required with --rope)")
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

    # Optional RoPE: the honest peer for the fused-RoPE comparison. The fused-attention library can't
    # absorb RoPE, so a model runs it as a separate elementwise pass over Q,K. We use the *fastest*
    # correct RoPE torch can produce (a contiguous interleaved rotation, plus torch.compile if it works),
    # so the comparison is fair to the peer — not the naive strided version.
    rope_enabled = bool(args.rope)
    rope_variants = []  # list of (label, fn); the timer takes the min over them
    if rope_enabled:
        if args.cos is None or args.sin is None:
            raise SystemExit("--rope needs --cos and --sin")
        half = D // 2
        cosv = np.fromfile(args.cos, dtype=np.float32).reshape(S, half)
        sinv = np.fromfile(args.sin, dtype=np.float32).reshape(S, half)
        cos_t = torch.from_numpy(cosv.copy()).to("cuda", tdt).view(1, 1, S, half)
        sin_t = torch.from_numpy(sinv.copy()).to("cuda", tdt).view(1, 1, S, half)

        def rope(x):
            xr = x.reshape(B, H, S, half, 2)
            x1 = xr[..., 0]
            x2 = xr[..., 1]
            o1 = x1 * cos_t - x2 * sin_t
            o2 = x1 * sin_t + x2 * cos_t
            return torch.stack((o1, o2), dim=-1).reshape(B, H, S, D)

        rope_variants.append(("eager", rope))

        # Complex-multiply interleaved RoPE — the *fastest* form torch can produce without triton: it
        # avoids the eager path's `stack` alloc+scatter, doing one complex elementwise multiply over
        # zero-copy `view_as_complex`/`view_as_real` views (rotation `cos+i·sin` precomputed once, since
        # it's position- not data-dependent). Taking the min over variants keeps the peer as strong as
        # possible — the conservative direction for any Wukong claim. Guarded: view_as_complex needs an
        # f32 last-dim-2 contiguous tensor, so we pay one f16→f32 cast in-loop (still fewer launches).
        try:
            rot = torch.view_as_complex(
                torch.stack((cos_t.float().view(S, half), sin_t.float().view(S, half)), dim=-1).contiguous()
            ).view(1, 1, S, half)

            def rope_cplx(x):
                xc = torch.view_as_complex(x.float().reshape(B, H, S, half, 2).contiguous())
                return torch.view_as_real(xc * rot).reshape(B, H, S, D).to(tdt)

            for _ in range(3):
                _ = rope_cplx(q)
            torch.cuda.synchronize()
            rope_variants.append(("complex", rope_cplx))
        except Exception:  # noqa: BLE001
            pass

        try:  # torch.compile may be unavailable on Windows (no triton/inductor backend); guard it
            rope_c = torch.compile(rope)
            for _ in range(3):
                _ = rope_c(q)
            torch.cuda.synchronize()
            rope_variants.append(("compiled", rope_c))
        except Exception:  # noqa: BLE001
            pass

    def time_backend(backend, ropef):
        with sdpa_kernel(backend):
            o = None
            for _ in range(args.warmup):
                qq = ropef(q) if ropef is not None else q
                kk = ropef(k) if ropef is not None else k
                o = F.scaled_dot_product_attention(
                    qq, kk, v, attn_mask=None, dropout_p=0.0, is_causal=causal, scale=scale)
            torch.cuda.synchronize()
            best = float("inf")
            for _ in range(args.runs):
                s_ev = torch.cuda.Event(enable_timing=True)
                e_ev = torch.cuda.Event(enable_timing=True)
                s_ev.record()
                for _ in range(args.iters):
                    qq = ropef(q) if ropef is not None else q
                    kk = ropef(k) if ropef is not None else k
                    o = F.scaled_dot_product_attention(
                        qq, kk, v, attn_mask=None, dropout_p=0.0, is_causal=causal, scale=scale)
                e_ev.record()
                torch.cuda.synchronize()
                best = min(best, s_ev.elapsed_time(e_ev) / 1000.0 / args.iters)
            return best, o

    def measure(backend):
        """Returns (pipeline_sec, sdpa_only_sec, O, rope_label). With RoPE, pipeline = best rope+sdpa over
        the rope variants (rope_label names the winner); sdpa_only is the no-rope reference. Without RoPE
        both secs are the same and rope_label is None."""
        if not rope_enabled:
            sec, o = time_backend(backend, None)
            return sec, sec, o, None
        sdpa_sec, _ = time_backend(backend, None)
        best_sec, best_o, best_label = float("inf"), None, None
        for label, fn in rope_variants:
            sec, o = time_backend(backend, fn)
            if sec < best_sec:
                best_sec, best_o, best_label = sec, o, label
        return best_sec, sdpa_sec, best_o, best_label

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
    report["rope"] = rope_enabled
    best_fused = None  # (name, sec, O tensor)
    for name, be in candidates:
        try:
            sec, sdpa_sec, o, rope_label = measure(be)
            report["backends"][name] = {
                "sec": sec,
                "sdpa_sec": sdpa_sec,
                "rope_variant": rope_label,
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
                lines.append(f"{name}_sdpa_sec={b.get('sdpa_sec', b['sec']):.9e}")
                lines.append(f"{name}_rope_variant={b.get('rope_variant') or 'none'}")
                lines.append(f"{name}_checksum={b['checksum']:.6f}")
                lines.append(f"{name}_gflops={b['gflops']:.3f}")
            else:
                lines.append(f"{name}_error={b.get('error', 'missing')}")
        lines.append(f"rope={1 if report['rope'] else 0}")
        with open(args.report, "w") as fh:
            fh.write("\n".join(lines) + "\n")

    print(json.dumps(report))


if __name__ == "__main__":
    main()
