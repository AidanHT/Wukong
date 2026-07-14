#!/usr/bin/env python
"""
bench_gpt2_torch.py — real PyTorch GPT-2 124M forward, timed, as the honest peer for the Wukong
`examples/gpt2_forward_bench.wk` program.

Loads HuggingFace GPT-2 ("gpt2", the SAME pretrained weights Wukong's gpt2_infer.wk matches to rel
~2e-6) and times the realistic autoregressive-inference forward: the transformer TRUNK over all S
positions (the context every token attends to) followed by the tied LM head on the LAST position only
(the single next-token distribution). Wukong's bench computes exactly this. Both use the SAME
deterministic ids (`id[s] = (s*1009 + 7) % 50000`); speed is token-value-independent, so the
last-position logits are directly cross-checkable against data/gpt2/gpt2_bench_lastrow.bin.

Regimes (min-of-N, warmup discarded, torch.inference_mode):
  * 1-thread eager   (torch.set_num_threads(1))
  * all-thread eager (torch default MKL threading)
  * all-thread torch.compile(max-autotune)  -- attempted; skipped with a note if Inductor cannot build
    (needs MSVC on PATH; run from a vcvars64 shell to enable).

Usage:  python tools/bench_gpt2_torch.py [S]   (default S=512)
"""

import os
import sys
import time

import numpy as np
import torch
from transformers import GPT2Model

S = int(sys.argv[1]) if len(sys.argv) > 1 else 512
NWARM = 2
NREPS = 5

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WUK_LASTROW = os.path.join(REPO, "data", "gpt2", "gpt2_bench_lastrow.bin")

torch.manual_seed(0)


def make_ids(s_len):
    return torch.tensor([[(s * 1009 + 7) % 50000 for s in range(s_len)]], dtype=torch.long)


def best_ms(fn, nwarm=NWARM, nreps=NREPS):
    with torch.inference_mode():
        for _ in range(nwarm):
            fn()
        best = float("inf")
        for _ in range(nreps):
            t0 = time.perf_counter()
            fn()
            best = min(best, (time.perf_counter() - t0) * 1e3)
    return best


def main():
    print(f"# real GPT-2 124M, fp32 CPU, S={S}, trunk(all pos) + LM head(last pos)  "
          f"(torch {torch.__version__}, {os.cpu_count()} logical CPUs)")
    trunk = GPT2Model.from_pretrained("gpt2").eval().float()
    wte = trunk.wte.weight  # [vocab, d], tied to the LM head
    ids = make_ids(S)

    def forward():
        h_last = trunk(ids).last_hidden_state[:, -1, :]   # [1, d] — final LN applied inside the trunk
        return h_last @ wte.t()                            # [1, vocab] — last-position logits only

    with torch.inference_mode():
        ref_logits = forward()[0].float().numpy()
    ref_argmax = int(ref_logits.argmax())

    results = {}
    torch.set_num_threads(1)
    results["eager 1-thread"] = best_ms(forward)
    torch.set_num_threads(os.cpu_count())
    results["eager all-thread"] = best_ms(forward)
    try:
        compiled = torch.compile(trunk, mode="max-autotune", fullgraph=False)

        def forward_c():
            h_last = compiled(ids).last_hidden_state[:, -1, :]
            return h_last @ wte.t()

        results["compiled all-thread"] = best_ms(forward_c, nwarm=1, nreps=3)
    except Exception as e:
        results["compiled all-thread"] = None
        print(f"# torch.compile skipped: {type(e).__name__}: {str(e)[:120]}")

    print(f"\n{'regime':<22} {'ms/forward':>12}")
    for name, ms in results.items():
        print(f"{name:<22} {('%.1f' % ms) if ms is not None else 'n/a':>12}")

    print(f"\ntorch argmax(last) = {ref_argmax}")
    if os.path.exists(WUK_LASTROW):
        wuk = np.fromfile(WUK_LASTROW, dtype=np.float32)
        if wuk.size == ref_logits.size:
            wuk_argmax = int(wuk.argmax())
            rel = float(np.max(np.abs(wuk - ref_logits)) / max(1e-9, np.max(np.abs(ref_logits))))
            agree = "MATCH" if wuk_argmax == ref_argmax and rel <= 1e-3 else "DIVERGE"
            print(f"wukong argmax(last) = {wuk_argmax}   rel = {rel:.3e}   -> {agree}")
        else:
            print(f"wukong last-row size {wuk.size} != {ref_logits.size}; skip cross-check")
    else:
        print(f"# {WUK_LASTROW} not found — run the .wk bench first for the cross-check")


if __name__ == "__main__":
    main()
