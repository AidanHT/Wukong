#!/usr/bin/env python
"""
verify_gpt2.py — headline honesty gate for the full-scale GPT-2 run.

Compares the logits that examples/gpt2_infer.wk wrote (data/gpt2/gpt2_wuk_logits.bin) against the
authoritative HuggingFace reference (data/gpt2/gpt2_ref_logits.bin), both [SEQ, VOCAB] f32 LE.

Prints max|Δ|, the relative error rel = max|Δ| / max|ref|, and argmax(last) of each, then prints

    GPT2 VERIFY PASS

iff  rel <= 1e-3  AND  wuk argmax(last) == ref argmax(last) == 1757 (" John").  numpy only (no torch).

FRESHNESS GATE. The verifier reads whatever `gpt2_wuk_logits.bin` is on disk; nothing in the file
says which run produced it. `examples/gpt2_infer.wk` exits 1 WITHOUT writing on every data-absent
path (wrong CWD prints -1, wrong blob size prints -2), so a missed nonzero exit followed by
`verify_gpt2.py` would re-assert the headline for a run that never happened. The artifact's age is
therefore part of the verdict: the file must be younger than --max-age-seconds (default 3600) or
this FAILS with the mtime named. Re-checking a deliberately archived artifact is still possible,
but only by saying so out loud with --no-age-check.

Data dir resolution (first that contains the wuk logits): $GPT2_DATA_DIR, then the positional
argument, then "data/gpt2" under the CWD, then the main-repo absolute path (the blobs are
gitignored and live in the main checkout, not this worktree).
"""

import argparse
import os
import time

import numpy as np

SEQ = 5
VOCAB = 50257
EXPECT_ARGMAX = 1757          # " John"
REL_TOL = 1e-3
MAX_AGE_S = 3600.0            # default freshness window for gpt2_wuk_logits.bin


def find_data_dir(cli_dir=None):
    candidates = []
    if os.environ.get("GPT2_DATA_DIR"):
        candidates.append(os.environ["GPT2_DATA_DIR"])
    if cli_dir:
        candidates.append(cli_dir)
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
    ap = argparse.ArgumentParser(description="GPT-2 headline honesty gate")
    ap.add_argument("data_dir", nargs="?", help="directory holding the gpt2_*.bin blobs")
    ap.add_argument(
        "--max-age-seconds",
        type=float,
        default=MAX_AGE_S,
        help=f"reject gpt2_wuk_logits.bin older than this (default {MAX_AGE_S:g}s)",
    )
    ap.add_argument(
        "--no-age-check",
        action="store_true",
        help="verify an archived artifact: skip the freshness gate (says so in the output)",
    )
    args = ap.parse_args()

    data_dir = find_data_dir(args.data_dir)
    wuk_path = os.path.join(data_dir, "gpt2_wuk_logits.bin")
    ref_path = os.path.join(data_dir, "gpt2_ref_logits.bin")
    for p in (wuk_path, ref_path):
        if not os.path.isfile(p):
            raise SystemExit(f"missing {p} (run examples/gpt2_infer.wk from the repo root first)")

    # Freshness: gpt2_infer.wk writes this file only on a successful full-scale run, so a stale
    # one means the run being verified never produced logits. Report the age either way.
    age = time.time() - os.path.getmtime(wuk_path)
    stamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(os.path.getmtime(wuk_path)))
    if args.no_age_check:
        print(f"wuk logits age       : {age:.0f}s (written {stamp})  [freshness gate DISABLED]")
    elif age > args.max_age_seconds:
        raise SystemExit(
            f"STALE ARTIFACT: {wuk_path} was written {stamp} ({age:.0f}s ago, limit "
            f"{args.max_age_seconds:.0f}s).\n"
            "It is not from this run — examples/gpt2_infer.wk exits 1 without writing when the "
            "data is unreachable.\nRe-run `wukongc --run --backend=native examples/gpt2_infer.wk` "
            "from the repo root, or pass --no-age-check to verify an archived artifact."
        )
    else:
        print(f"wuk logits age       : {age:.0f}s (written {stamp})")

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
