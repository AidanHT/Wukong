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
import json
import math
import struct
NWARM = 2
NREPS = 5

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WUK_LASTROW = os.path.join(REPO, "data", "gpt2", "gpt2_bench_lastrow.bin")

RESULT_PREFIX = "GPT2_PEER_RESULT "
VOCAB = 50257


def validate_logits(reference, candidate):
    """The existing last-row argmax/relative-error gate, with malformed rows rejected."""
    import numpy as np

    reference = np.asarray(reference, dtype=np.float32)
    candidate = np.asarray(candidate, dtype=np.float32)
    if len(reference) != VOCAB or len(candidate) != VOCAB:
        raise ValueError(f"last-row size must be {VOCAB}")
    if reference.ndim != 1 or candidate.ndim != 1:
        raise ValueError("last-row must be one-dimensional")
    if not np.isfinite(reference).all() or not np.isfinite(candidate).all():
        raise ValueError("last-row contains non-finite logits")
    ref_argmax = int(reference.argmax())
    argmax = int(candidate.argmax())
    # Preserve the original NumPy float32 subtraction, reduction, and division.
    rel = float(np.max(np.abs(candidate - reference)) / max(1e-9, np.max(np.abs(reference))))
    if ref_argmax != 338 or argmax != ref_argmax or rel > 1e-3:
        raise ValueError(f"DIVERGE: reference argmax={ref_argmax}, argmax={argmax}, rel={rel:.3e}")
    return rel


def load_reference(path, sequence_length):
    if sequence_length != 512:
        raise ValueError("the Wukong reference workload requires S=512")
    with open(path, "rb") as source:
        data = source.read()
    if len(data) != VOCAB * 4:
        raise ValueError(f"last-row byte size {len(data)} != {VOCAB * 4}")
    return struct.unpack(f"<{VOCAB}f", data)


def make_ids(s_len):
    return torch.tensor([[(s * 1009 + 7) % 50000 for s in range(s_len)]], dtype=torch.long)


def best_ms(fn, nwarm=NWARM, nreps=NREPS, validate=None, warmup_output=None):
    with torch.inference_mode():
        for _ in range(nwarm):
            warmup_output = fn()
        best = float("inf")
        for _ in range(nreps):
            t0 = time.perf_counter()
            output = fn()
            elapsed = (time.perf_counter() - t0) * 1e3
            if not math.isfinite(elapsed) or elapsed <= 0:
                raise ValueError("invalid forward duration")
            best = min(best, elapsed)
        if validate is not None:
            if nwarm or warmup_output is not None:
                validate(warmup_output)
            validate(output)
    return best


def best_compiled_ms(prepare, validate):
    try:
        forward = prepare()
        # The existing one warmup triggers lazy compilation; only preparation is optional.
        with torch.inference_mode():
            warmup = forward()
    except Exception as error:
        print(f"# torch.compile skipped: {type(error).__name__}: {str(error)[:120]}")
        return None
    # Timed execution and post-block correctness failures must propagate.
    return best_ms(forward, nwarm=0, nreps=3, validate=validate, warmup_output=warmup)


def main(sequence_length=512):
    # Reject missing/malformed references before importing or loading the model.
    wuk = load_reference(WUK_LASTROW, sequence_length) if sequence_length == 512 else None
    if wuk is not None:
        validate_logits(wuk, wuk)
    global torch
    import torch
    from transformers import GPT2Model

    torch.manual_seed(0)
    print(f"# real GPT-2 124M, fp32 CPU, S={sequence_length}, trunk(all pos) + LM head(last pos)  "
          f"(torch {torch.__version__}, {os.cpu_count()} logical CPUs)")
    print("# final warmup and final measured outputs checked after each timing block; intermediate outputs unchecked")
    trunk = GPT2Model.from_pretrained("gpt2", local_files_only=True).eval().float()
    wte = trunk.wte.weight  # [vocab, d], tied to the LM head
    ids = make_ids(sequence_length)

    def forward():
        h_last = trunk(ids).last_hidden_state[:, -1, :]   # [1, d] — final LN applied inside the trunk
        return h_last @ wte.t()                            # [1, vocab] — last-position logits only

    with torch.inference_mode():
        ref_logits = forward()[0].float().numpy()
    ref_argmax = int(ref_logits.argmax())
    rel = validate_logits(ref_logits, wuk) if wuk is not None else None

    def validate(output):
        row = output[0].float().numpy()
        if wuk is not None:
            validate_logits(ref_logits, row)
        elif len(row) != VOCAB or not all(math.isfinite(float(x)) for x in row):
            raise ValueError("invalid logits")

    results = {}
    torch.set_num_threads(1)
    results["eager 1-thread"] = best_ms(forward, validate=validate)
    torch.set_num_threads(os.cpu_count())
    results["eager all-thread"] = best_ms(forward, validate=validate)
    def prepare_compiled():
        compiled = torch.compile(trunk, mode="max-autotune", fullgraph=False)

        def forward_c():
            h_last = compiled(ids).last_hidden_state[:, -1, :]
            return h_last @ wte.t()

        return forward_c

    results["compiled all-thread"] = best_compiled_ms(prepare_compiled, validate)

    print(f"\n{'regime':<22} {'ms/forward':>12}")
    for name, ms in results.items():
        print(f"{name:<22} {('%.1f' % ms) if ms is not None else 'n/a':>12}")

    print(f"\ntorch argmax(last) = {ref_argmax}")
    if wuk is None:
        print("# no matching Wukong reference for this S; comparison INELIGIBLE")
        return {"schema": 1, "status": "ineligible", "sequence_length": sequence_length,
                "reason": "no matching Wukong reference", "results": results}
    print(f"wukong argmax(last) = 338   rel = {rel:.3e}   -> MATCH")
    return {"schema": 1, "status": "ok", "sequence_length": sequence_length,
            "vocab": VOCAB, "argmax": ref_argmax, "relative_error": rel,
            "results": results}


def cli(argv=None):
    try:
        args = sys.argv[1:] if argv is None else argv
        if len(args) > 1:
            raise ValueError("expected at most one sequence length argument")
        result = main(int(args[0]) if args else 512)
    except Exception as error:
        print(f"torch peer failed: {type(error).__name__}: {error}", file=sys.stderr)
        print(RESULT_PREFIX + json.dumps({"schema": 1, "status": "invalid", "error": str(error)}))
        return 1
    print(RESULT_PREFIX + json.dumps(result, allow_nan=False))
    return 0 if result["status"] == "ok" else 2


if __name__ == "__main__":
    sys.exit(cli())
