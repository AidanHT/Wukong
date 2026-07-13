#!/usr/bin/env python
"""
verify_gpt2.py — headline honesty gate for the full-scale GPT-2 run.

Compares the logits that examples/gpt2_infer.wk wrote (data/gpt2/gpt2_wuk_logits.bin) against the
authoritative HuggingFace reference (data/gpt2/gpt2_ref_logits.bin), both [SEQ, VOCAB] f32 LE.

Prints max|Δ|, the relative error rel = max|Δ| / max|ref|, and argmax(last) of each, then prints

    GPT2 VERIFY PASS

iff  rel <= 1e-3  AND  wuk argmax(last) == ref argmax(last) == 1757 (" John").  numpy only (no torch).

Data dir resolution (first that contains the wuk logits): $GPT2_DATA_DIR, then argv[1], then
"data/gpt2" under the CWD, then the main-repo absolute path (the blobs are gitignored and live in
the main checkout, not this worktree).
"""

import os
import sys

import numpy as np

SEQ = 5
VOCAB = 50257
EXPECT_ARGMAX = 1757          # " John"
REL_TOL = 1e-3


def find_data_dir():
    candidates = []
    if os.environ.get("GPT2_DATA_DIR"):
        candidates.append(os.environ["GPT2_DATA_DIR"])
    if len(sys.argv) > 1:
        candidates.append(sys.argv[1])
    candidates.append(os.path.join("data", "gpt2"))
    candidates.append(
        r"C:\Users\Quant\Documents\Programming\Projects\Compiler\Mercury\data\gpt2"
    )
    for d in candidates:
        if os.path.isfile(os.path.join(d, "gpt2_wuk_logits.bin")):
            return d
    # Fall back to the last candidate so the error message names a concrete path.
    return candidates[-1]


def load(path):
    a = np.fromfile(path, dtype="<f4")
    if a.size != SEQ * VOCAB:
        raise SystemExit(
            f"{path}: expected {SEQ * VOCAB} f32 ([{SEQ}, {VOCAB}]), got {a.size}"
        )
    return a.reshape(SEQ, VOCAB)


def main():
    data_dir = find_data_dir()
    wuk_path = os.path.join(data_dir, "gpt2_wuk_logits.bin")
    ref_path = os.path.join(data_dir, "gpt2_ref_logits.bin")
    for p in (wuk_path, ref_path):
        if not os.path.isfile(p):
            raise SystemExit(f"missing {p} (run examples/gpt2_infer.wk from the repo root first)")

    wuk = load(wuk_path)
    ref = load(ref_path)

    maxdiff = float(np.max(np.abs(wuk - ref)))
    maxref = float(np.max(np.abs(ref)))
    rel = maxdiff / maxref if maxref > 0 else float("inf")

    wuk_argmax_last = int(wuk[-1].argmax())
    ref_argmax_last = int(ref[-1].argmax())

    print(f"data dir             : {data_dir}")
    print(f"max|d|               : {maxdiff:.6g}")
    print(f"max|ref|             : {maxref:.6g}")
    print(f"rel = max|d|/max|ref| : {rel:.6g}  (tol {REL_TOL:g})")
    print(f"argmax(last)  wuk    : {wuk_argmax_last}")
    print(f"argmax(last)  ref    : {ref_argmax_last}  (expect {EXPECT_ARGMAX})")

    ok = (
        rel <= REL_TOL
        and wuk_argmax_last == ref_argmax_last == EXPECT_ARGMAX
    )
    if ok:
        print("GPT2 VERIFY PASS")
        return 0
    print("GPT2 VERIFY FAIL")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
