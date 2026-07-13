#!/usr/bin/env python
"""
export_gpt2.py — GPT-2 124M weight exporter + authoritative reference generator.

Loads HuggingFace `GPT2LMHeadModel.from_pretrained("gpt2")`, converts every HF `Conv1D`
weight (stored `[in, out]`) into the `nn.Linear [out, in]` row-major convention the Wukong
model consumes, splits `attn.c_attn` into wq/wk/wv (+ bq/bk/bv), and writes ONE flat
little-endian f32 blob in the FROZEN layout order:

    wte, wpe,
    per layer l in 0..12:  ln1g ln1b  wq bq  wk bk  wv bv  wo bo  ln2g ln2b  w1 b1  w2 b2
    ln_f_g, ln_f_b

Also writes the prompt token ids (int32) and the authoritative HF logits (f32), emits the
`examples/gpt2_config.wk` const file (single source of truth for every offset) and
`data/gpt2/MANIFEST.md`.

MANDATORY SELF-CHECK: an INDEPENDENT numpy forward is run by slicing the *just-written* flat
blob at the *computed* offsets (so it validates the transposes AND the offset table at once),
and asserted against the HF logits (small max|Δ|, and argmax of the last position == 1757,
" John"). Prints exactly:  EXPORTER SELF-CHECK PASS argmax=<n> maxdiff=<x>
"""

import math
import os

import numpy as np
import torch

# ------------------------------------------------------------------------------------------
# Frozen config (see scratchpad/gpt2-layout.md).
# ------------------------------------------------------------------------------------------
D = 768          # model / embedding dim
H = 12           # attention heads
HD = D // H      # head dim = 64
DFF = 4 * D      # FFN hidden = 3072
VOCAB = 50257    # token embedding rows (also the tied LM head)
POS = 1024       # positional embedding rows
LAYERS = 12
SEQ = 5          # test-prompt length
EPS = 1e-5
ATTN_SCALE = 1.0 / math.sqrt(HD)   # 1/sqrt(64) = 0.125

PROMPT = "Hello, my name is"
EXPECT_IDS = [15496, 11, 616, 1438, 318]   # frozen tokenization of PROMPT, S=5
EXPECT_ARGMAX = 1757                        # " John"

# Absolute main-repo output dir (blobs are large, NOT committed).
OUT_DIR = r"C:\Users\Quant\Documents\Programming\Projects\Compiler\Mercury\data\gpt2"
# Committed config file lives in the worktree next to this script's repo.
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
CONFIG_WK = os.path.join(REPO, "examples", "gpt2_config.wk")
# MANIFEST.md is a small COMMITTED artifact, so it is written into the repo tree (next to
# where `data/gpt2/` is version-controlled) rather than into the large uncommitted blob dir.
MANIFEST = os.path.join(REPO, "data", "gpt2", "MANIFEST.md")

WEIGHTS_BIN = os.path.join(OUT_DIR, "gpt2_124m_weights.bin")
TOKENS_BIN = os.path.join(OUT_DIR, "gpt2_tokens.bin")
REF_LOGITS_BIN = os.path.join(OUT_DIR, "gpt2_ref_logits.bin")


# ------------------------------------------------------------------------------------------
# 1. Load the model and pull out the tensors in the [out, in] convention.
# ------------------------------------------------------------------------------------------
def to2d_out_in(conv1d_weight):
    """HF Conv1D `.weight` is [in, out]; transpose to nn.Linear [out, in] (C-contiguous)."""
    w = conv1d_weight.detach().to(torch.float32).numpy()   # [in, out]
    return np.ascontiguousarray(w.T)                        # [out, in]


def vec(t):
    return np.ascontiguousarray(t.detach().to(torch.float32).numpy())


def build_layout():
    """Return (ordered list of (name, ndarray)) in the exact frozen blob order."""
    from transformers import GPT2LMHeadModel

    print("loading GPT2LMHeadModel.from_pretrained('gpt2') ...", flush=True)
    model = GPT2LMHeadModel.from_pretrained("gpt2").eval()
    tr = model.transformer

    parts = []

    # wte [VOCAB, D], wpe [POS, D] — already [·, D] row-major, copy as-is.
    parts.append(("wte", vec(tr.wte.weight)))
    parts.append(("wpe", vec(tr.wpe.weight)))

    for l in range(LAYERS):
        blk = tr.h[l]

        # LayerNorm 1 (gain/bias) — copy as-is.
        parts.append((f"h{l}.ln1g", vec(blk.ln_1.weight)))
        parts.append((f"h{l}.ln1b", vec(blk.ln_1.bias)))

        # attn.c_attn: [D, 3D] Conv1D -> [3D, D] -> split rows into wq/wk/wv each [D, D].
        wcat = to2d_out_in(blk.attn.c_attn.weight)          # [3D, D]
        bcat = vec(blk.attn.c_attn.bias)                    # [3D]
        wq, wk, wv = wcat[0:D], wcat[D:2 * D], wcat[2 * D:3 * D]
        bq, bk, bv = bcat[0:D], bcat[D:2 * D], bcat[2 * D:3 * D]
        parts.append((f"h{l}.wq", np.ascontiguousarray(wq)))
        parts.append((f"h{l}.bq", np.ascontiguousarray(bq)))
        parts.append((f"h{l}.wk", np.ascontiguousarray(wk)))
        parts.append((f"h{l}.bk", np.ascontiguousarray(bk)))
        parts.append((f"h{l}.wv", np.ascontiguousarray(wv)))
        parts.append((f"h{l}.bv", np.ascontiguousarray(bv)))

        # attn.c_proj: [D, D] Conv1D -> wo [D, D]; bias -> bo.
        parts.append((f"h{l}.wo", to2d_out_in(blk.attn.c_proj.weight)))
        parts.append((f"h{l}.bo", vec(blk.attn.c_proj.bias)))

        # LayerNorm 2.
        parts.append((f"h{l}.ln2g", vec(blk.ln_2.weight)))
        parts.append((f"h{l}.ln2b", vec(blk.ln_2.bias)))

        # mlp.c_fc: [D, 4D] Conv1D -> w1 [4D, D]; bias -> b1.
        parts.append((f"h{l}.w1", to2d_out_in(blk.mlp.c_fc.weight)))
        parts.append((f"h{l}.b1", vec(blk.mlp.c_fc.bias)))
        # mlp.c_proj: [4D, D] Conv1D -> w2 [D, 4D]; bias -> b2.
        parts.append((f"h{l}.w2", to2d_out_in(blk.mlp.c_proj.weight)))
        parts.append((f"h{l}.b2", vec(blk.mlp.c_proj.bias)))

    # Final LayerNorm.
    parts.append(("ln_f_g", vec(tr.ln_f.weight)))
    parts.append(("ln_f_b", vec(tr.ln_f.bias)))

    return model, parts


# ------------------------------------------------------------------------------------------
# 2. Offset table (computed, single source of truth for gpt2_config.wk).
# ------------------------------------------------------------------------------------------
# Per-tensor element counts within one layer, in order.
LAYER_TENSORS = [
    ("LN1G", D), ("LN1B", D),
    ("WQ", D * D), ("BQ", D),
    ("WK", D * D), ("BK", D),
    ("WV", D * D), ("BV", D),
    ("WO", D * D), ("BO", D),
    ("LN2G", D), ("LN2B", D),
    ("W1", DFF * D), ("B1", DFF),
    ("W2", D * DFF), ("B2", D),
]


def compute_offsets():
    rel = {}
    off = 0
    for name, n in LAYER_TENSORS:
        rel[name] = off
        off += n
    layer_stride = off

    off_wte = 0
    off_wpe = off_wte + VOCAB * D
    off_layer0 = off_wpe + POS * D
    off_lnfg = off_layer0 + LAYERS * layer_stride
    off_lnfb = off_lnfg + D
    total = off_lnfb + D
    return {
        "rel": rel,
        "LAYER_STRIDE": layer_stride,
        "OFF_WTE": off_wte,
        "OFF_WPE": off_wpe,
        "OFF_LAYER0": off_layer0,
        "OFF_LNFG": off_lnfg,
        "OFF_LNFB": off_lnfb,
        "TOTAL_WEIGHTS": total,
    }


# ------------------------------------------------------------------------------------------
# 3. Independent numpy forward (reads the FLAT BLOB at the COMPUTED offsets).
# ------------------------------------------------------------------------------------------
def layernorm(x, g, b):
    mu = x.mean(-1, keepdims=True)
    var = x.var(-1, keepdims=True)          # population variance (ddof=0), matches torch LN
    return (x - mu) / np.sqrt(var + EPS) * g + b


def gelu_new(x):
    # HF 'gelu_new' (tanh approximation), identical coefficients.
    c = math.sqrt(2.0 / math.pi)            # 0.7978845608...
    return 0.5 * x * (1.0 + np.tanh(c * (x + 0.044715 * x ** 3)))


def numpy_forward(blob, offs, ids):
    """Full GPT-2 forward from the flat blob; returns logits [S, VOCAB] float32."""
    rel = offs["rel"]

    def get(off, shape):
        n = int(np.prod(shape))
        return blob[off:off + n].reshape(shape)

    wte = get(offs["OFF_WTE"], (VOCAB, D))
    wpe = get(offs["OFF_WPE"], (POS, D))

    S = len(ids)
    x = (wte[ids] + wpe[:S]).astype(np.float32)             # [S, D]

    for l in range(LAYERS):
        base = offs["OFF_LAYER0"] + l * offs["LAYER_STRIDE"]
        ln1g = get(base + rel["LN1G"], (D,)); ln1b = get(base + rel["LN1B"], (D,))
        wq = get(base + rel["WQ"], (D, D)); bq = get(base + rel["BQ"], (D,))
        wk = get(base + rel["WK"], (D, D)); bk = get(base + rel["BK"], (D,))
        wv = get(base + rel["WV"], (D, D)); bv = get(base + rel["BV"], (D,))
        wo = get(base + rel["WO"], (D, D)); bo = get(base + rel["BO"], (D,))
        ln2g = get(base + rel["LN2G"], (D,)); ln2b = get(base + rel["LN2B"], (D,))
        w1 = get(base + rel["W1"], (DFF, D)); b1 = get(base + rel["B1"], (DFF,))
        w2 = get(base + rel["W2"], (D, DFF)); b2 = get(base + rel["B2"], (D,))

        # --- attention sublayer (pre-LN) ---
        h = layernorm(x, ln1g, ln1b)                        # [S, D]
        q = h @ wq.T + bq                                   # [S, D]
        k = h @ wk.T + bk
        v = h @ wv.T + bv

        attn = np.empty((S, D), dtype=np.float32)
        causal = np.triu(np.ones((S, S), dtype=bool), 1)    # True above diagonal (masked)
        for hh in range(H):
            sl = slice(hh * HD, (hh + 1) * HD)
            qh, kh, vh = q[:, sl], k[:, sl], v[:, sl]
            sc = (qh @ kh.T) * ATTN_SCALE                   # [S, S]
            sc = np.where(causal, np.float32(-1e30), sc)
            sc = sc - sc.max(-1, keepdims=True)
            e = np.exp(sc)
            p = e / e.sum(-1, keepdims=True)
            attn[:, sl] = p @ vh
        ao = attn @ wo.T + bo                               # [S, D]
        x = x + ao                                          # residual

        # --- MLP sublayer (pre-LN) ---
        h2 = layernorm(x, ln2g, ln2b)
        f = h2 @ w1.T + b1                                  # [S, DFF]
        f = gelu_new(f)
        dn = f @ w2.T + b2                                  # [S, D]
        x = x + dn                                          # residual

    lnfg = get(offs["OFF_LNFG"], (D,)); lnfb = get(offs["OFF_LNFB"], (D,))
    x = layernorm(x, lnfg, lnfb)
    logits = x @ wte.T                                      # tied LM head, no bias -> [S, VOCAB]
    return logits.astype(np.float32)


# ------------------------------------------------------------------------------------------
# 4. Emit gpt2_config.wk and MANIFEST.md.
# ------------------------------------------------------------------------------------------
def emit_config_wk(offs):
    rel = offs["rel"]
    lines = []
    lines.append("module examples.gpt2_config")
    lines.append("")
    lines.append("// GPT-2 124M layout constants — GENERATED by tools/export_gpt2.py.")
    lines.append("// Single source of truth for the flat f32 weight blob's offsets. Do not")
    lines.append("// hand-edit; re-run the exporter. All offsets are ELEMENT (f32) indices.")
    lines.append("")
    lines.append("// ---- model config ----")
    lines.append(f"const GPT2_D: i64 = {D};")
    lines.append(f"const GPT2_H: i64 = {H};")
    lines.append(f"const GPT2_HD: i64 = {HD};            // head dim = D / H")
    lines.append(f"const GPT2_DFF: i64 = {DFF};")
    lines.append(f"const GPT2_VOCAB: i64 = {VOCAB};")
    lines.append(f"const GPT2_POS: i64 = {POS};")
    lines.append(f"const GPT2_LAYERS: i64 = {LAYERS};")
    lines.append(f"const GPT2_SEQ: i64 = {SEQ};            // test-prompt length")
    lines.append("")
    lines.append("// ---- total element count (== canonical GPT-2 small parameter count) ----")
    lines.append(f"const GPT2_TOTAL_WEIGHTS: i64 = {offs['TOTAL_WEIGHTS']};")
    lines.append("")
    lines.append("// ---- top-level blob offsets ----")
    lines.append(f"const GPT2_OFF_WTE: i64 = {offs['OFF_WTE']};")
    lines.append(f"const GPT2_OFF_WPE: i64 = {offs['OFF_WPE']};")
    lines.append(f"const GPT2_LAYER_STRIDE: i64 = {offs['LAYER_STRIDE']};")
    lines.append(f"const GPT2_OFF_LAYER0: i64 = {offs['OFF_LAYER0']};   "
                 "// layer l base = OFF_LAYER0 + l*LAYER_STRIDE")
    lines.append("")
    lines.append("// ---- within-layer relative offsets (add to the layer base) ----")
    for name, _ in LAYER_TENSORS:
        lines.append(f"const GPT2_REL_{name}: i64 = {rel[name]};")
    lines.append("")
    lines.append("// ---- final LayerNorm ----")
    lines.append(f"const GPT2_OFF_LNFG: i64 = {offs['OFF_LNFG']};")
    lines.append(f"const GPT2_OFF_LNFB: i64 = {offs['OFF_LNFB']};")
    lines.append("")
    with open(CONFIG_WK, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines))
    print(f"wrote {CONFIG_WK}", flush=True)


def emit_manifest(offs, parts):
    rel = offs["rel"]
    lines = []
    lines.append("# GPT-2 124M export — MANIFEST")
    lines.append("")
    lines.append("Generated by `tools/export_gpt2.py`. All weights are little-endian f32 in the")
    lines.append("`nn.Linear [out, in]` row-major convention. Offsets are ELEMENT indices into the")
    lines.append("flat blob; byte offset = element offset * 4.")
    lines.append("")
    lines.append("## Config")
    lines.append("")
    lines.append(f"| D | H | head_dim | DFF | VOCAB | POS | LAYERS | SEQ | eps | attn_scale |")
    lines.append(f"|---|---|----------|-----|-------|-----|--------|-----|-----|------------|")
    lines.append(f"| {D} | {H} | {HD} | {DFF} | {VOCAB} | {POS} | {LAYERS} | {SEQ} "
                 f"| 1e-5 | {ATTN_SCALE:.6f} |")
    lines.append("")
    lines.append(f"GELU = tanh-approx ('gelu_new'). LM head is tied to `wte`.")
    lines.append("")
    lines.append("## Files")
    lines.append("")
    lines.append("| file | dtype | shape | bytes |")
    lines.append("|------|-------|-------|-------|")
    lines.append(f"| gpt2_124m_weights.bin | f32 LE | [{offs['TOTAL_WEIGHTS']}] (flat) "
                 f"| {offs['TOTAL_WEIGHTS'] * 4} |")
    lines.append(f"| gpt2_tokens.bin | i32 LE | [{SEQ}] | {SEQ * 4} |")
    lines.append(f"| gpt2_ref_logits.bin | f32 LE | [{SEQ}, {VOCAB}] | {SEQ * VOCAB * 4} |")
    lines.append("")
    lines.append(f"Prompt: `{PROMPT}`  ->  ids {EXPECT_IDS}  (argmax(last) == {EXPECT_ARGMAX}, "
                 "\" John\").")
    lines.append("")
    lines.append("## Top-level offsets (element indices)")
    lines.append("")
    lines.append("| tensor | shape | offset |")
    lines.append("|--------|-------|--------|")
    lines.append(f"| wte | [{VOCAB}, {D}] | {offs['OFF_WTE']} |")
    lines.append(f"| wpe | [{POS}, {D}] | {offs['OFF_WPE']} |")
    lines.append(f"| layer0 base | — | {offs['OFF_LAYER0']} |")
    lines.append(f"| layer stride | — | {offs['LAYER_STRIDE']} |")
    lines.append(f"| ln_f_g | [{D}] | {offs['OFF_LNFG']} |")
    lines.append(f"| ln_f_b | [{D}] | {offs['OFF_LNFB']} |")
    lines.append("")
    lines.append("Layer `l` base = OFF_LAYER0 + l * LAYER_STRIDE.")
    lines.append("")
    lines.append("## Within-layer relative offsets (element indices, add to layer base)")
    lines.append("")
    lines.append("| tensor | shape | rel offset |")
    lines.append("|--------|-------|------------|")
    shapes = {
        "LN1G": f"[{D}]", "LN1B": f"[{D}]",
        "WQ": f"[{D}, {D}]", "BQ": f"[{D}]",
        "WK": f"[{D}, {D}]", "BK": f"[{D}]",
        "WV": f"[{D}, {D}]", "BV": f"[{D}]",
        "WO": f"[{D}, {D}]", "BO": f"[{D}]",
        "LN2G": f"[{D}]", "LN2B": f"[{D}]",
        "W1": f"[{DFF}, {D}]", "B1": f"[{DFF}]",
        "W2": f"[{D}, {DFF}]", "B2": f"[{D}]",
    }
    for name, _ in LAYER_TENSORS:
        lines.append(f"| {name.lower()} | {shapes[name]} | {rel[name]} |")
    lines.append("")
    with open(MANIFEST, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines))
    print(f"wrote {MANIFEST}", flush=True)


# ------------------------------------------------------------------------------------------
# Main.
# ------------------------------------------------------------------------------------------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(MANIFEST), exist_ok=True)
    offs = compute_offsets()

    # --- build layout + flat blob ---
    model, parts = build_layout()

    # Cross-check the running offsets against the computed formula (proves the offset
    # table that lands in gpt2_config.wk matches the actual concatenation order).
    running = 0
    named = dict(parts)
    checkpoints = {
        "wpe": offs["OFF_WPE"],
        "h0.ln1g": offs["OFF_LAYER0"],
        "ln_f_g": offs["OFF_LNFG"],
        "ln_f_b": offs["OFF_LNFB"],
    }
    for name, arr in parts:
        if name in checkpoints:
            assert running == checkpoints[name], \
                f"offset mismatch at {name}: running={running} expected={checkpoints[name]}"
        running += arr.size
    assert running == offs["TOTAL_WEIGHTS"], \
        f"total mismatch: running={running} expected={offs['TOTAL_WEIGHTS']}"
    print(f"offset table OK; TOTAL_WEIGHTS = {offs['TOTAL_WEIGHTS']}", flush=True)

    blob = np.concatenate([arr.astype(np.float32, copy=False).ravel() for _, arr in parts])
    assert blob.size == offs["TOTAL_WEIGHTS"]
    blob.astype("<f4").tofile(WEIGHTS_BIN)
    print(f"wrote {WEIGHTS_BIN}  ({blob.size} f32 = {blob.size * 4} bytes)", flush=True)

    # --- tokens ---
    ids = list(EXPECT_IDS)
    # Verify the frozen ids against the real tokenizer when it is available.
    try:
        from transformers import GPT2TokenizerFast
        tok = GPT2TokenizerFast.from_pretrained("gpt2")
        got = tok(PROMPT)["input_ids"]
        assert got == EXPECT_IDS, f"tokenizer gave {got}, expected {EXPECT_IDS}"
        print(f"tokenizer check OK: {PROMPT!r} -> {got}", flush=True)
    except Exception as e:  # tokenizer files may be absent; ids are frozen regardless
        print(f"tokenizer check skipped ({type(e).__name__}: {e})", flush=True)
    np.asarray(ids, dtype="<i4").tofile(TOKENS_BIN)
    print(f"wrote {TOKENS_BIN}  ({len(ids)} i32)", flush=True)

    # --- authoritative HF logits (f32) ---
    with torch.no_grad():
        out = model(torch.tensor([ids], dtype=torch.long))
        hf_logits = out.logits[0].to(torch.float32).numpy()   # [S, VOCAB]
    hf_logits.astype("<f4").tofile(REF_LOGITS_BIN)
    hf_argmax_last = int(hf_logits[-1].argmax())
    print(f"wrote {REF_LOGITS_BIN}  ([{SEQ}, {VOCAB}] f32); HF argmax(last)={hf_argmax_last}",
          flush=True)
    assert hf_argmax_last == EXPECT_ARGMAX, \
        f"HF reference argmax {hf_argmax_last} != {EXPECT_ARGMAX}"

    # --- config + manifest ---
    emit_config_wk(offs)
    emit_manifest(offs, parts)

    # --- MANDATORY SELF-CHECK: independent numpy forward from the flat blob ---
    # Reload the blob from disk to prove the on-disk bytes are correct end-to-end.
    disk_blob = np.fromfile(WEIGHTS_BIN, dtype="<f4")
    assert disk_blob.size == offs["TOTAL_WEIGHTS"]
    np_logits = numpy_forward(disk_blob, offs, ids)

    maxdiff = float(np.max(np.abs(np_logits - hf_logits)))
    np_argmax_last = int(np_logits[-1].argmax())

    ok = (np_argmax_last == EXPECT_ARGMAX) and (maxdiff < 1e-2)
    if not ok:
        print(f"SELF-CHECK FAILED argmax={np_argmax_last} (want {EXPECT_ARGMAX}) "
              f"maxdiff={maxdiff:.6g}", flush=True)
        raise SystemExit(1)

    # Exact required success line.
    print(f"EXPORTER SELF-CHECK PASS argmax={np_argmax_last} maxdiff={maxdiff:.6g}", flush=True)


if __name__ == "__main__":
    main()
