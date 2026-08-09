"""The framework bar: `torch.compile` (Inductor + Triton), fairly configured.

GPU_RETARGET_PLAN.md section 0: "The framework bar is `torch.compile` with Inductor+Triton, which
works natively on Linux. **Eager PyTorch is not a bar; every claim measured against it is on
notice.**" This repo's published "beats PyTorch at every S" is an eager-only number whose stated
justification was that Triton does not install on Windows (BENCHMARKS.md:2182,2230). On Linux that
excuse expires, and the claim has to be re-earned here.

So this harness deliberately reports THREE columns per shape and makes the eager one the control
rather than the bar:

    eager      what the retracted claim was measured against; kept so the size of the old error is
               visible in the same run rather than argued about later
    compiled   torch.compile(mode="default") -- Inductor picks ATen for GEMM and fuses the tail
    autotune   torch.compile(mode="max-autotune") -- Triton templates benchmarked against ATen

`peer_sec` is the **fastest** of the three. Anything else would be picking a weak opponent.

Fairness rules applied here, each from D5 section 2.2:

  * dtype parity. Wukong's GEMM peers are f16-in / f32-accumulate, so the default here is fp16, not
    fp32-with-tf32. Feeding torch fp32 while Wukong runs fp16 would be a dtype-rigged comparison in
    Wukong's favour; running fp32 with tf32 OFF would be the same rigging with extra steps, so tf32
    is enabled whenever the dtype is fp32.
  * `dynamic=False` and `fullgraph=True`. Dynamic shapes make Inductor emit guarded, slower kernels;
    a graph break hands part of the work back to eager. Either would flatter Wukong.
  * compilation is never inside the timed region: every column is warmed until it stops compiling.
  * CUDA-event timing, back-to-back, no sleeps, best-of-`runs` over `iters` -- the same shape the
    Wukong side uses, so the ratio is apples-to-apples.
  * correctness first. Every column is checked against an fp32 reference before its time is
    reported; a fast wrong peer flatters Wukong exactly as much as a slow correct one handicaps it.

Ops, chosen to match what Wukong actually publishes:

    gemm         C = A @ B.T          the `nn.Linear` contract `baselines.rs` uses
    linear_gelu  gelu(x @ W.T + b)    the fused-epilogue bar (cuBLASLt epilogues / Wukong's FFN)
    sdpa         scaled dot product   attention, with the FLASH backend forced, plus FA4 if present

Output is flat `key=value` lines (the no-serde format the repo's other peer driver already uses),
plus an optional JSON dump. Pure ASCII: a POSIX-locale container makes a non-ASCII print an error.

    python torch_compile_peer.py --op gemm --shapes 4096x4096x4096,8192x8192x8192
    python torch_compile_peer.py --op sdpa --shapes 1x16x2048x128 --causal
"""

import argparse
import json
import statistics
import sys


def _events_time(fn, iters, runs, warmup):
    """best-of-`runs` mean-per-iter, in seconds, over CUDA-event-bracketed batches of `iters`.

    Best-of-N rather than a single timing because a rented container cannot lock clocks: the minimum
    is the sample least polluted by the clock ramp and by whatever else the host is doing. Both the
    minimum and the median are reported so a wide gap between them is visible instead of hidden.
    """
    import torch

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    per_run = []
    for _ in range(runs):
        ev0, ev1 = torch.cuda.Event(True), torch.cuda.Event(True)
        ev0.record()
        for _ in range(iters):
            fn()
        ev1.record()
        torch.cuda.synchronize()
        per_run.append(ev0.elapsed_time(ev1) / iters / 1e3)
    return min(per_run), statistics.median(per_run)


def _compile(fn, mode):
    import torch

    return torch.compile(fn, mode=mode, dynamic=False, fullgraph=True)


def _set_fp32_precision(mode):
    """Set the fp32 matmul/conv precision, tolerating both API generations.

    The old `allow_tf32` spelling is deprecated after torch 2.9 and mixing it with the new
    `fp32_precision` one is explicitly unsupported (D5 section 9 pitfall 13), so exactly one is used
    and the other is only a fallback for an older wheel.
    """
    import torch

    try:
        torch.backends.cuda.matmul.fp32_precision = mode
        torch.backends.cudnn.conv.fp32_precision = mode
    except (AttributeError, RuntimeError, ValueError):
        torch.backends.cuda.matmul.allow_tf32 = mode != "ieee"
        torch.backends.cudnn.allow_tf32 = mode != "ieee"


def _rel_err(got, want):
    """Max absolute error, and that error **relative to the reference's magnitude**.

    The relative figure is the one worth gating on. An fp16 GEMM at K=4096 over standard-normal
    inputs produces outputs of magnitude ~sqrt(K) = 64, so even a perfectly good tensor-core result
    differs from an fp32 reference by O(1e-1) in absolute terms. Gating on that absolute number would
    disqualify every correct peer column and silently report `peer=none` -- i.e. the harness would
    conclude "no peer" on a box where the peer was fine, which is exactly the kind of self-flattering
    failure this whole exercise exists to prevent.
    """
    d = (got.float() - want.float()).abs().max().item()
    scale = want.float().abs().max().item()
    return d, (d / scale if scale > 0 else d)


def op_gemm(args, dtype):
    """C = A @ B.T -- the `nn.Linear` contract, so the layouts match `baselines.rs` with no
    transpose fudge that would advantage either side."""
    import torch

    m, n, k = (int(x) for x in args.shape.split("x"))
    a = torch.randn(m, k, device="cuda", dtype=dtype)
    b = torch.randn(n, k, device="cuda", dtype=dtype)

    def f(a, b):
        return a @ b.t()

    ref = (a.float() @ b.float().t())
    return f, (a, b), ref, 2.0 * m * n * k


def op_linear_gelu(args, dtype):
    """gelu(x @ W.T + b) -- the fused-epilogue bar. Inductor fuses the bias+GELU tail into the
    matmul's epilogue, which is exactly what cuBLASLt's epilogues and Wukong's fused FFN do, so this
    is the honest peer for the fused path rather than for a bare GEMM."""
    import torch

    m, n, k = (int(x) for x in args.shape.split("x"))
    x = torch.randn(m, k, device="cuda", dtype=dtype)
    w = torch.randn(n, k, device="cuda", dtype=dtype)
    bias = torch.randn(n, device="cuda", dtype=dtype)

    def f(x, w, bias):
        return torch.nn.functional.gelu(torch.nn.functional.linear(x, w, bias))

    ref = torch.nn.functional.gelu(
        torch.nn.functional.linear(x.float(), w.float(), bias.float())
    )
    # 2*M*N*K for the GEMM; the epilogue is not counted, so the FLOP/s figure stays comparable to
    # the plain-GEMM column instead of being inflated by pointwise work.
    return f, (x, w, bias), ref, 2.0 * m * n * k


def op_sdpa(args, dtype):
    """Attention, head-major [B,H,S,D] -- the layout Wukong's flash uses.

    The FLASH backend is forced by name rather than left to SDPA's dispatcher, because "SDPA picked
    something" is not a bar: on the Linux wheels the FLASH backend IS FlashAttention-2 compiled into
    torch, and that is the kernel section 0 asks for.
    """
    import torch

    b, h, s, d = (int(x) for x in args.shape.split("x"))
    q = torch.randn(b, h, s, d, device="cuda", dtype=dtype)
    k = torch.randn(b, h, s, d, device="cuda", dtype=dtype)
    v = torch.randn(b, h, s, d, device="cuda", dtype=dtype)
    causal = args.causal

    def f(q, k, v):
        return torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=causal)

    ref = torch.nn.functional.scaled_dot_product_attention(
        q.float(), k.float(), v.float(), is_causal=causal
    )
    # 4*B*H*S*S*D for QK^T and PV; halved when causal, since half the score matrix is masked.
    flop = 4.0 * b * h * s * s * d * (0.5 if causal else 1.0)
    return f, (q, k, v), ref, flop


OPS = {"gemm": op_gemm, "linear_gelu": op_linear_gelu, "sdpa": op_sdpa}


def run_shape(args, dtype, out):
    import torch

    build = OPS[args.op]
    # Every reference is an fp32 computation, so it is affected by the tf32 switch too -- computing
    # it with tf32 ON would make the "reference" share the peer's own approximation and stop being an
    # independent check. Force IEEE for the reference, then restore whatever the run asked for.
    _set_fp32_precision("ieee")
    f, inputs, ref, flop = build(args, dtype)
    torch.cuda.synchronize()
    _set_fp32_precision("tf32" if dtype is torch.float32 else "ieee")
    tol = args.tol

    columns = {}

    def measure(name, fn, ctx=None):
        try:
            if ctx is not None:
                with ctx():
                    got = fn(*inputs)
            else:
                got = fn(*inputs)
            torch.cuda.synchronize()
        except Exception as e:  # noqa: BLE001 - a column that cannot run is reported, not fatal
            out["%s.%s.error" % (args.shape, name)] = repr(e)
            return
        abs_err, rel = _rel_err(got, ref)
        out["%s.%s.max_abs_err" % (args.shape, name)] = "%.3e" % abs_err
        out["%s.%s.max_rel_err" % (args.shape, name)] = "%.3e" % rel
        if not (rel < tol):
            # Not fatal for the run, but this column is disqualified: a wrong peer is not a peer.
            out["%s.%s.disqualified" % (args.shape, name)] = "rel %.3e >= tol %.3e" % (rel, tol)
            return
        if ctx is not None:
            def timed():
                with ctx():
                    fn(*inputs)
        else:
            def timed():
                fn(*inputs)
        best, med = _events_time(timed, args.iters, args.runs, args.warmup)
        columns[name] = best
        out["%s.%s.sec" % (args.shape, name)] = "%.9f" % best
        out["%s.%s.median_sec" % (args.shape, name)] = "%.9f" % med
        out["%s.%s.gflops" % (args.shape, name)] = "%.2f" % (flop / best / 1e9)

    if args.op == "sdpa":
        import torch.nn.attention as attn

        def flash_ctx():
            return attn.sdpa_kernel(attn.SDPBackend.FLASH_ATTENTION)

        measure("eager_flash", f, flash_ctx)
        measure("eager", f)
    else:
        measure("eager", f)

    measure("compiled", _compile(f, "default"))
    measure("autotune", _compile(f, "max-autotune"))

    if args.op == "sdpa" and args.fa4:
        # FlashAttention-4's own entry point, not through SDPA. Its layout is [B,S,H,D] while this
        # harness (and Wukong) are head-major [B,H,S,D], so the transposes are materialised ONCE,
        # outside the timed region -- timing a permute as if it were attention would handicap the
        # peer, which is the same sin as strawmanning it.
        try:
            from flash_attn.cute import flash_attn_func

            q, k, v = (t.transpose(1, 2).contiguous() for t in inputs)
            causal = args.causal

            def fa4(q, k, v):
                return flash_attn_func(q, k, v, causal=causal)

            got = fa4(q, k, v).transpose(1, 2)
            torch.cuda.synchronize()
            abs_err, rel = _rel_err(got, ref)
            out["%s.fa4.max_abs_err" % args.shape] = "%.3e" % abs_err
            out["%s.fa4.max_rel_err" % args.shape] = "%.3e" % rel
            if rel < tol:
                best, med = _events_time(lambda: fa4(q, k, v), args.iters, args.runs, args.warmup)
                columns["fa4"] = best
                out["%s.fa4.sec" % args.shape] = "%.9f" % best
                out["%s.fa4.median_sec" % args.shape] = "%.9f" % med
                out["%s.fa4.gflops" % args.shape] = "%.2f" % (flop / best / 1e9)
            else:
                out["%s.fa4.disqualified" % args.shape] = "rel %.3e >= tol %.3e" % (rel, tol)
        except Exception as e:  # noqa: BLE001
            out["%s.fa4.error" % args.shape] = repr(e)

    if not columns:
        out["%s.peer" % args.shape] = "none"
        return
    winner = min(columns, key=lambda c: columns[c])
    out["%s.peer" % args.shape] = winner
    out["%s.peer_sec" % args.shape] = "%.9f" % columns[winner]
    out["%s.peer_gflops" % args.shape] = "%.2f" % (flop / columns[winner] / 1e9)
    if "eager" in columns:
        # The number that decides whether the old eager-only claim survives: how much of the
        # published margin was the peer being weak rather than Wukong being fast.
        out["%s.eager_over_peer" % args.shape] = "%.4f" % (columns["eager"] / columns[winner])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--op", choices=sorted(OPS), default="gemm")
    ap.add_argument("--shapes", default="4096x4096x4096",
                    help="comma list; MxNxK for gemm/linear_gelu, BxHxSxD for sdpa")
    ap.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp16")
    ap.add_argument("--causal", action="store_true")
    ap.add_argument("--fa4", action="store_true", help="also time flash-attn-4 directly (sm_90+)")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--tol", type=float, default=2e-2,
                    help="max RELATIVE error (vs the reference's magnitude) a column may have and "
                         "still be timed; see _rel_err for why absolute would be wrong")
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    import torch

    if not torch.cuda.is_available():
        print("FAIL: no CUDA device")
        return 2
    try:
        import triton
    except Exception as e:  # noqa: BLE001
        print("FAIL: Triton does not import (%r). Without it torch.compile falls back to ATen and "
              "this would be an eager column wearing the framework bar's name." % (e,))
        return 2

    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]
    # fp32 without tf32 would be a strawman on any tensor-core part. `run_shape` flips this to
    # `ieee` while it computes each reference and back again, so the reference stays an independent
    # check rather than sharing the peer's own approximation.
    _set_fp32_precision("tf32" if dtype is torch.float32 else "ieee")

    p = torch.cuda.get_device_properties(0)
    out = {
        "device": p.name,
        "cc": "sm_%d%d" % (p.major, p.minor),
        "sms": p.multi_processor_count,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "triton": triton.__version__,
        "op": args.op,
        "dtype": args.dtype,
        "causal": int(args.causal),
        "iters": args.iters,
        "runs": args.runs,
    }
    for shape in args.shapes.split(","):
        args.shape = shape.strip()
        if not args.shape:
            continue
        run_shape(args, dtype, out)

    for k in sorted(out):
        print("%s=%s" % (k, out[k]))
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(out, fh, indent=2, sort_keys=True)
        print("report=%s" % args.json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
