"""Prove Inductor+Triton really compiled a kernel -- and pay the max-autotune cost exactly once.

Two jobs, both about honesty and both about money:

1. **Proof.** `torch.compile` is happy to fall back to ATen (= cuBLAS) and fuse only the pointwise
   tail. If that happens, the "torch.compile" column is cuBLAS with extra steps, and calling it the
   framework bar would be a lie -- a *weaker* peer than the one being claimed. So this asserts that a
   Triton kernel was actually generated, not that the call returned.
2. **Cost.** `mode="max-autotune"` benchmarks Triton templates against ATen on the FIRST call for
   each new shape, which takes minutes, and on a rented box those minutes are metered. Run this once
   on the cheapest SKU with `TORCHINDUCTOR_CACHE_DIR`/`TRITON_CACHE_DIR` pointed at the persistent
   Volume and every later round is a cache hit (D5 section 2.2 / section 9 pitfall 12).

Also correctness-checks the compiled result against eager, because a fast wrong peer flatters Wukong
exactly as much as a slow correct one handicaps it.

    TORCHINDUCTOR_CACHE_DIR=/persist/inductor-cache TRITON_CACHE_DIR=/persist/triton-cache \
        /opt/torch-venv/bin/python smoke_inductor.py

Pure-ASCII output: a POSIX-locale container turns a stray non-ASCII character into a
UnicodeEncodeError at print time.
"""

import argparse
import os
import sys
import time


def triton_kernel_was_generated(cache_dir, since):
    """Two independent ways to answer 'did Inductor emit a Triton kernel', because one of them is a
    private API.

    (a) `torch._inductor.codecache.PyCodeCache.modules` -- direct, and inherently scoped to THIS
        process, but private and free to move between torch releases. If it moves, the check must
        degrade rather than become a false failure.
    (b) the on-disk Inductor cache, which is a *documented* env var: any generated module containing
        `@triton.jit` is the same evidence and it survives an internals rename.

    `since` is what makes (b) honest. The whole point of pointing the cache at a persistent Volume is
    that it stays warm across runs -- so a plain walk would happily find last week's Triton kernel
    and certify a run that actually fell back to ATen. Only files written at or after this run's
    compile count. (A five-second slack absorbs clock granularity on a network filesystem.)
    """
    evidence = []
    try:
        from torch._inductor.codecache import PyCodeCache

        srcs = [getattr(m, "__file__", "") for m in PyCodeCache.modules]
        for s in srcs:
            if s and s.endswith(".py"):
                try:
                    with open(s) as fh:
                        body = fh.read()
                except OSError:
                    continue
                if "@triton.jit" in body or "triton_heuristics" in body:
                    evidence.append("PyCodeCache:" + os.path.basename(s))
                    break
    except Exception as e:  # noqa: BLE001 - a private-API move must not be a false failure
        evidence.append("[PyCodeCache unavailable: %r]" % (e,))

    stale = 0
    if cache_dir and os.path.isdir(cache_dir):
        found = False
        for root, _dirs, files in os.walk(cache_dir):
            for fn in files:
                if not fn.endswith(".py"):
                    continue
                path = os.path.join(root, fn)
                try:
                    fresh = os.path.getmtime(path) >= since - 5.0
                    with open(path) as fh:
                        body = fh.read()
                except OSError:
                    continue
                if "@triton.jit" not in body and "triton_heuristics" not in body:
                    continue
                if not fresh:
                    stale += 1
                    continue
                evidence.append("cache:" + fn)
                found = True
                break
            if found:
                break
    if stale:
        evidence.append("[%d older Triton kernels in the cache, not counted as evidence]" % stale)

    hard = [e for e in evidence if not e.startswith("[")]
    return bool(hard), evidence


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--m", type=int, default=4096)
    ap.add_argument("--n", type=int, default=4096)
    ap.add_argument("--k", type=int, default=4096)
    ap.add_argument("--mode", default="max-autotune",
                    help="the Inductor mode to prove; `default` picks ATen for GEMM and is NOT the "
                         "bar section 0 asks for")
    ap.add_argument("--iters", type=int, default=50)
    args = ap.parse_args()

    import torch

    if not torch.cuda.is_available():
        print("FAIL: torch.cuda.is_available() is False -- no device to compile for")
        return 2
    try:
        import triton
    except Exception as e:  # noqa: BLE001
        print("FAIL: Triton does not import (%r). torch.compile would fall back to ATen, i.e. to "
              "eager, and eager is not a bar." % (e,))
        return 2

    p = torch.cuda.get_device_properties(0)
    print("device=%s cc=sm_%d%d sms=%d torch=%s cuda=%s triton=%s"
          % (p.name, p.major, p.minor, p.multi_processor_count, torch.__version__,
             torch.version.cuda, triton.__version__))
    cache_dir = os.environ.get("TORCHINDUCTOR_CACHE_DIR", "")
    print("inductor_cache=%s triton_cache=%s"
          % (cache_dir or "(default, NOT persisted)",
             os.environ.get("TRITON_CACHE_DIR", "(default, NOT persisted)")))
    if not cache_dir:
        print("!! TORCHINDUCTOR_CACHE_DIR is unset: this max-autotune compile will be paid again "
              "on every future container, on metered hardware.")

    def f(a, b):
        return torch.softmax(a @ b, dim=-1)

    a = torch.randn(args.m, args.k, device="cuda", dtype=torch.float16)
    b = torch.randn(args.k, args.n, device="cuda", dtype=torch.float16)
    ref = f(a, b)

    # `dynamic=False`: dynamic-shape Inductor emits guarded, slower kernels, which would flatter
    # Wukong. `fullgraph=True`: a graph break would silently hand part of the work back to eager.
    compile_started = time.time()
    g = torch.compile(f, mode=args.mode, dynamic=False, fullgraph=True)
    out = g(a, b)
    torch.cuda.synchronize()
    print("compile_plus_first_call_sec=%.1f" % (time.time() - compile_started))

    ok, evidence = triton_kernel_was_generated(cache_dir, compile_started)
    print("triton_kernel_evidence=%s" % ("; ".join(evidence) or "none"))
    if not ok:
        print("FAIL: no generated Triton kernel found for THIS run. Either Inductor fell back to "
              "ATen -- in which case a 'torch.compile' column measured here would really be cuBLAS, "
              "a weaker peer than the one being claimed -- or torch's private PyCodeCache API moved "
              "AND the on-disk cache was warm, so nothing new was written to prove it either way. "
              "The evidence line above distinguishes them: `[PyCodeCache unavailable: ...]` plus "
              "`[N older Triton kernels ...]` is the second case, and pointing "
              "TORCHINDUCTOR_CACHE_DIR at an empty directory settles it.")
        return 1

    err = (out.float() - ref.float()).abs().max().item()
    print("max_abs_err_vs_eager=%.3e" % err)
    if not (err < 1e-2):
        print("FAIL: the compiled result does not match eager. A wrong peer is not a peer.")
        return 1

    ev0, ev1 = torch.cuda.Event(True), torch.cuda.Event(True)
    for _ in range(5):
        g(a, b)
    torch.cuda.synchronize()
    ev0.record()
    for _ in range(args.iters):
        g(a, b)
    ev1.record()
    torch.cuda.synchronize()
    print("compiled_ms_per_iter=%.4f" % (ev0.elapsed_time(ev1) / args.iters))
    print("OK: Inductor generated a Triton kernel and it matches eager")
    return 0


if __name__ == "__main__":
    sys.exit(main())
