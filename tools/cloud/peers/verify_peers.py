"""Resolve every STRONG peer and say plainly which ones are missing.

GPU_RETARGET_PLAN.md section 0 sets the bar Wukong's GPU numbers must be measured against:

    GEMM       cuBLAS/cuBLASLt with fused epilogues, and CUTLASS's profiler
    attention  cuDNN and a REAL FlashAttention build, not an unfused cuBLAS chain
    framework  torch.compile with Inductor+Triton -- eager PyTorch is NOT a bar
    int4/int8  Marlin/Machete-class kernels, not "no library peer exists"

The failure this script exists to prevent is not "a peer is missing". It is **a peer being missing
without anyone noticing**: a harness that politely skips a column and reports a green run publishes
a win over a bar it never raced. So `--require` is a declaration, and anything named there that does
not resolve makes this exit non-zero.

Runs under the torch venv's interpreter (see `WUKONG_TORCH_PYTHON`). Reads exactly the environment
variables `crates/wukong_codegen_gpu/src/baselines.rs` reads, so its verdict is the one a round will
actually get rather than a second opinion that can drift from it.

    python verify_peers.py --require torch-compile,flash-attn
    python verify_peers.py --require all --json /persist/peer-verify.json

Output is pure ASCII on purpose: these containers can run under a POSIX locale, where printing a
non-ASCII character to a pipe is a UnicodeEncodeError rather than a cosmetic problem.
"""

import argparse
import json
import os
import subprocess
import sys

# Canonical name -> the aliases `WUKONG_STRONG_PEERS` accepts. Kept in step with
# `StrongPeer::aliases` in baselines.rs: a name that one side accepts and the other rejects is a
# round that thinks it declared a bar and did not.
PEERS = {
    "torch-compile": ["torch-compile", "torch_compile", "torch", "inductor", "triton"],
    "flash-attn": ["flash-attn", "flash_attn", "flash", "fa", "fa2", "fa3", "fa4"],
    "cutlass": ["cutlass", "cutlass-profiler"],
    "marlin": ["marlin", "machete", "int4", "w4a16", "vllm"],
}

BAR = {
    "torch-compile": "the framework bar: torch.compile with Inductor+Triton (eager is NOT a bar)",
    "flash-attn": "the attention bar: a real FlashAttention build, not an unfused cuBLAS chain",
    "cutlass": "the GEMM bar: CUTLASS's profiler, best-of-many-kernels by exhaustive search",
    "marlin": "the int4 bar: Marlin/Machete-class kernels",
}

ENV_VAR = {
    "torch-compile": "WUKONG_TORCH_PYTHON",
    "flash-attn": "WUKONG_TORCH_PYTHON",
    "cutlass": "WUKONG_CUTLASS_PROFILER",
    "marlin": "WUKONG_VLLM_PYTHON",
}


def parse_require(raw):
    """`all` / `none` / a comma list of names or aliases. An unknown name is an ERROR.

    A typo that silently meant "require nothing" would hand back exactly the quiet skip this whole
    mechanism exists to remove, so it must never be tolerated.
    """
    out = []
    for tok in raw.replace(";", ",").replace(" ", ",").split(","):
        tok = tok.strip().lower()
        if not tok or tok in ("none", "0"):
            continue
        if tok in ("all", "1"):
            for name in PEERS:
                if name not in out:
                    out.append(name)
            continue
        hit = [name for name, al in PEERS.items() if tok in al]
        if not hit:
            sys.exit(
                "verify_peers: unknown peer %r. Known: %s (or `all` / `none`). Refusing to guess."
                % (tok, ", ".join(PEERS))
            )
        if hit[0] not in out:
            out.append(hit[0])
    return out


def device_line():
    try:
        import torch
    except Exception as e:  # noqa: BLE001 - any import failure is the same answer here
        return {"torch_err": type(e).__name__ + ": " + str(e)}
    info = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda or "none",
    }
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        info.update(
            device=p.name,
            cc="sm_%d%d" % (p.major, p.minor),
            sms=p.multi_processor_count,
            vram_mib=p.total_memory // (1024 * 1024),
        )
    else:
        info["device"] = "cpu-only (torch.cuda.is_available() is False)"
    return info


def check_torch_compile():
    """torch + Triton importable. Without Triton, `torch.compile` falls back to ATen and the peer
    quietly degrades to eager -- which section 0 says is not a bar at all."""
    try:
        import torch
    except Exception as e:  # noqa: BLE001
        return False, "torch does not import: %r" % (e,)
    try:
        import triton
    except Exception as e:  # noqa: BLE001
        return False, (
            "torch %s imports but Triton does not (%r); torch.compile would fall back to ATen, "
            "i.e. to eager, which is not a bar" % (torch.__version__, e)
        )
    return True, "torch %s (cuda %s) + triton %s" % (
        torch.__version__,
        torch.version.cuda,
        triton.__version__,
    )


def check_flash_attn():
    """A real fused FlashAttention kernel, by whichever of the three routes this box has.

    All three are honest (D5 section 3): FA4 is the Hopper/Blackwell CuTeDSL build, FA2 is the
    Ampere/Ada source build, and torch SDPA's FLASH_ATTENTION backend *is* FlashAttention-2 compiled
    into the torch wheel and forced by name -- which is what `tools/fa2_sdpa_peer.py` already drives.
    What would NOT be honest is calling an unfused cuBLAS chain the attention bar.
    """
    found = []
    cc = 0
    try:
        import torch

        if torch.cuda.is_available():
            d = torch.cuda.get_device_properties(0)
            cc = d.major * 10 + d.minor
    except Exception:  # noqa: BLE001
        pass

    try:
        import importlib.metadata as md

        import flash_attn.cute  # noqa: F401

        # FA4 targets SM90/SM100 only, but it is a pure-Python wheel that JITs through CuTeDSL, so
        # it imports perfectly happily on an Ada L4. "It imports" is therefore NOT evidence that this
        # box has an attention bar -- below sm_90 the line is informational and must not count, or
        # the bar would be marked satisfied on a device that cannot run the kernel. (Mirrors
        # `probe_flash_attn` in baselines.rs; the two verdicts have to agree.)
        line = "flash-attn-4 %s (CuTeDSL)" % md.version("flash-attn-4")
        found.append(line if cc >= 90 else "[%s: needs sm_90+, this device is sm_%d]" % (line, cc))
    except Exception:  # noqa: BLE001
        pass
    try:
        import flash_attn

        found.append("flash-attn %s" % getattr(flash_attn, "__version__", "unknown"))
    except Exception:  # noqa: BLE001
        pass
    try:
        import torch
        import torch.nn.attention as attn

        if torch.cuda.is_available():
            q = torch.randn(1, 2, 64, 64, device="cuda", dtype=torch.float16)
            with attn.sdpa_kernel(attn.SDPBackend.FLASH_ATTENTION):
                torch.nn.functional.scaled_dot_product_attention(q, q, q)
            found.append("torch SDPA FLASH_ATTENTION backend (this IS FA2, forced by name)")
    except Exception as e:  # noqa: BLE001
        found.append("[sdpa-flash unavailable: %r]" % (e,))
    real = [f for f in found if not f.startswith("[")]
    if not real:
        return False, "no FlashAttention route resolved: " + ("; ".join(found) or "nothing found")
    return True, " + ".join(found)


def check_cutlass():
    """The profiler must be *runnable here*, not merely present on the Volume.

    `--version` is the informative flag (it names the CUTLASS release for the round log) but it is a
    profiler CLI detail, not a contract this repo controls; `--help` is answered by every build. The
    fallback exists so a build without `--version` is not mistaken for a missing peer -- that false
    negative would fail an otherwise-good round on metered time. Mirrors `probe_cutlass` in
    baselines.rs: the two verdicts must agree, or the Python report and the Rust gate disagree about
    whether the round has a GEMM bar.

    Note what this cannot see: a profiler built for the wrong arch runs perfectly and silently omits
    the fastest kernels. That check is `modal_app.py::peers`, against the build-time manifest.
    """
    path = os.environ.get("WUKONG_CUTLASS_PROFILER", "")
    if not path:
        return False, "WUKONG_CUTLASS_PROFILER is unset"
    if not os.path.isfile(path):
        return False, "no cutlass_profiler at %s (build it with ::build_peers, on CPU)" % path
    why = []
    for flag in ("--version", "--help"):
        try:
            p = subprocess.run([path, flag], capture_output=True, text=True, timeout=120)
        except Exception as e:  # noqa: BLE001
            why.append("%s: %r" % (flag, e))
            continue
        if p.returncode != 0:
            why.append("%s exited %d: %s" % (flag, p.returncode, p.stderr.strip()[-200:]))
            continue
        first = next((ln.strip() for ln in p.stdout.splitlines() if ln.strip()), "cutlass_profiler")
        note = "" if flag == "--version" else " (no --version in this build; --help answered)"
        return True, "%s%s (%s)" % (first, note, path)
    return False, "%s is present but did not run -- %s" % (path, "; ".join(why))


def check_marlin():
    py = os.environ.get("WUKONG_VLLM_PYTHON", "")
    bench = os.environ.get("WUKONG_VLLM_BENCH_DIR", "")
    if not py or not os.path.isfile(py):
        return False, "no vLLM interpreter at %r (stage it with ::build_peers, on CPU)" % py
    for script in ("benchmark_marlin.py", "benchmark_machete.py"):
        if not bench or not os.path.isfile(os.path.join(bench, script)):
            return False, "%s missing from WUKONG_VLLM_BENCH_DIR=%r" % (script, bench)
    code = "import importlib.metadata as m, vllm._custom_ops; print(m.version('vllm'))"
    try:
        p = subprocess.run([py, "-c", code], capture_output=True, text=True, timeout=600)
    except Exception as e:  # noqa: BLE001
        return False, "%s could not be run: %r" % (py, e)
    if p.returncode != 0:
        return False, "%s cannot import vllm._custom_ops: %s" % (py, p.stderr.strip()[-400:])
    return True, "vllm %s + %s" % (p.stdout.strip(), bench)


CHECKS = {
    "torch-compile": check_torch_compile,
    "flash-attn": check_flash_attn,
    "cutlass": check_cutlass,
    "marlin": check_marlin,
}


def loader_path_warnings():
    """The one loader-path defect worth re-checking from inside the peer process.

    A CUDA `stubs` directory holds a link-time placeholder `libcuda.so` with no driver behind it, and
    `libcuda.so` is cudarc's FIRST driver candidate -- so a stubs dir on the path makes every driver
    call fail with a symptom that reads exactly like broken hardware. Several peer build recipes tell
    you to add it. `modal_app.py` refuses to start with it set and `baselines.rs` reports it too;
    this is the third place because it is the cheapest possible check and the most expensive miss.
    """
    var = "PATH" if os.name == "nt" else "LD_LIBRARY_PATH"
    sep = ";" if os.name == "nt" else ":"
    bad = [e for e in os.environ.get(var, "").split(sep) if e.rstrip("/\\").endswith("stubs")]
    return ["%s contains a CUDA stubs dir (%s): its stub libcuda.so shadows the real driver" % (var, e)
            for e in bad]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--require", default="none",
                    help="comma list of peers that MUST resolve (or `all` / `none`)")
    ap.add_argument("--json", default="", help="also write the report here")
    args = ap.parse_args()

    required = parse_require(args.require)
    prov = device_line()
    print("--- peer provenance ---")
    for k in sorted(prov):
        print("  %-12s %s" % (k, prov[k]))

    for w in loader_path_warnings():
        print("!! " + w)

    report = {"provenance": prov, "required": required, "peers": {}}
    missing = []
    print("--- strong peers ---")
    for name, fn in CHECKS.items():
        ok, detail = fn()
        need = "REQUIRED" if name in required else "optional"
        report["peers"][name] = {"ok": ok, "required": name in required, "detail": detail}
        print("  %-14s %-8s %-3s %s" % (name, need, "ok" if ok else "--", detail))
        if not ok and name in required:
            missing.append((name, detail))

    if args.json:
        try:
            with open(args.json, "w") as fh:
                json.dump(report, fh, indent=2, sort_keys=True)
            print("report -> %s" % args.json)
        except OSError as e:
            print("!! could not write %s: %r" % (args.json, e))

    if missing:
        print("")
        print("FAIL: %d of %d required peer(s) did not resolve. A round published now would be"
              % (len(missing), len(required)))
        print("      measured against a WEAKER bar than it claims (GPU_RETARGET_PLAN.md section 0).")
        for name, detail in missing:
            print("  %s -- %s" % (name, detail))
            print("      wanted for %s" % BAR[name])
            print("      point %s at it, or build it: ::build_peers" % ENV_VAR[name])
        return 1
    print("")
    print("OK: every required peer resolved (%s)" % (", ".join(required) or "none declared"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
