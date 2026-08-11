"""Exercise `modal_app.py`'s pure staging logic **without Modal, a container or a network**.

Everything `::build_peers` decides before it spends money -- which CUTLASS kernel families a filter
selects, whether a FlashAttention wheel actually carries the kernel the round will call, how many
build jobs the container's memory allows -- is ordinary Python, and every one of those decisions has
already been wrong once in a way that cost a metered run. But `modal_app.py` cannot simply be
imported off Modal (importing it builds an `Image` and an `App`), and none of this is reachable from
a Windows dev box otherwise. So this lifts the named top-level definitions out of the module with
`ast` and execs them against stub constants.

What it is for, concretely: a change to the kernel-name patterns, the census, the FA arch table or
the wheel audit can be checked here in a second, on the machine the change is written on, instead of
in a $1/hr container forty minutes later.

    python tools/cloud/peers/selftest_modal_helpers.py

Exit status is the verdict; every check prints PASS or FAIL. Pure ASCII, like every other script in
this directory, because these run under a POSIX locale where a stray non-ASCII byte on stdout is a
UnicodeEncodeError rather than a cosmetic problem.
"""

import ast
import os
import pathlib
import sys
import tempfile
import zipfile

DEFAULT_APP = pathlib.Path(__file__).resolve().parent.parent / "modal_app.py"

# The definitions this selftest needs. Named explicitly rather than "everything that parses", so a
# helper that grows a Modal dependency fails here loudly instead of being silently dropped.
WANT_FN = {
    "_parse_cutlass_dtypes", "_cutlass_patterns", "_cutlass_filter_matches",
    "_cutlass_generated_kernels", "_cutlass_census", "_dir_stats", "_verify_fa_wheel",
    "_build_jobs", "_clock_snapshot", "_clock_lock_attempt",
}
WANT_CONST = {
    "_CUTLASS_DTYPES", "_CUTLASS_KERNELS_3X", "_CUTLASS_KERNELS_2X", "_CUTLASS_FAMILY_TOKENS",
    "_CUTLASS_NAME_RE", "_CUTLASS_PROFILER_DTYPE", "_FA3_DISABLE_PRESETS", "_FA3_MIN_TIMEOUT",
    "_FA2_ARCH",
}

FAILS = []


def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name + (("  " + detail) if detail else ""))
    if not cond:
        FAILS.append(name)


def load(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    picked = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in WANT_FN:
            picked.append(node)
        elif isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id in WANT_CONST for t in node.targets
        ):
            picked.append(node)
    ns = {
        "os": os, "sys": sys, "re": __import__("re"), "json": __import__("json"),
        "time": __import__("time"), "subprocess": __import__("subprocess"), "pathlib": pathlib,
        # Stubs for the module constants the lifted code closes over. Values chosen to match the
        # defaults so the printed messages read the way they will in a real round.
        "WK_CUTLASS_TAG": "v4.6.1", "WK_PEER_TIMEOUT": 14400, "WK_PEER_MEM_MIB": 32768,
    }
    exec(compile(ast.Module(body=picked, type_ignores=[]), "<modal_app subset>", "exec"), ns)
    missing = sorted((WANT_FN | WANT_CONST) - set(ns))
    if missing:
        sys.exit("selftest: %s no longer defines %s at module level" % (path, missing))
    return ns


def test_filter_semantics(ns):
    """The port of CUTLASS's own matcher. If this drifts, the census counts a different set than
    the build compiles, which is worse than no census at all."""
    fm = ns["_cutlass_filter_matches"]
    check("filter: a trailing star is a prefix test",
          fm("cutlass3x_sm90_tensorop_gemm_e4m3_e4m3_f32_*",
             "cutlass3x_sm90_tensorop_gemm_e4m3_e4m3_f32_void_f16_128x128x64_1x1x1_0_tnn_align16"))
    check("filter: a different dtype is rejected",
          not fm("cutlass3x_sm90_tensorop_gemm_e4m3_e4m3_f32_*",
                 "cutlass3x_sm90_tensorop_gemm_f16_f16_f32_void_f16_128x128x64"))
    check("filter: substrings must appear IN ORDER (CUTLASS's rule, not fnmatch's)",
          fm("gemm*tf32*m128", "cutlass_tensorop_gemm_tf32_x_m128")
          and not fm("gemm*tf32*m128", "cutlass_tensorop_m128_gemm_tf32"))
    check("filter: a bare substring is unanchored",
          fm("_s8_s8_s32_", "cutlass3x_sm90_tensorop_gemm_s8_s8_s32_void_s32_128x128x128"))


def test_dtypes(ns):
    pd = ns["_parse_cutlass_dtypes"]
    check("dtypes: the default set", pd("f16,fp8,int8") == ["f16", "fp8", "int8"])
    check("dtypes: aliases fold onto families", pd("e4m3,s8") == ["fp8", "int8"])
    check("dtypes: `all`", pd("all") == list(ns["_CUTLASS_DTYPES"]))
    try:
        pd("fp4")
        check("dtypes: an unknown family is an error", False)
    except SystemExit as e:
        check("dtypes: an unknown family is an error", "fp4" in str(e))


def test_patterns(ns):
    cp = ns["_cutlass_patterns"]
    p90 = cp("90a", ["f16", "fp8", "int8"])
    check("patterns: sm90a covers all three families", len(p90) == 5, str(len(p90)))
    check("patterns: the fp8 pattern is the 3.x e4m3 prefix",
          "cutlass3x_sm90_tensorop_gemm_e4m3_e4m3_f32_*" in p90)
    check("patterns: int8 covers both signednesses",
          any("_s8_s8_s32_" in p for p in p90) and any("_u8_u8_s32_" in p for p in p90))
    try:
        cp("80", ["fp8"])
        check("patterns: fp8 below Ada is refused", False)
    except SystemExit as e:
        check("patterns: fp8 below Ada is refused", "no fp8 tensor-core matmul" in str(e))
    check("patterns: Blackwell retargets the arch token",
          all("_sm120_" in p for p in cp("120a", ["f16"])))
    check("patterns: sm_89 falls back to the 2.x names",
          all(p.startswith("cutlass_tensorop_") for p in cp("89", ["f16", "int8"])))
    return p90


def test_census(ns, p90, tmp):
    """The guard that turns a wrong pattern into five CPU minutes instead of a weak peer."""
    census, gen = ns["_cutlass_census"], ns["_cutlass_generated_kernels"]
    genroot = tmp / "tools" / "library" / "generated" / "gemm" / "90"
    genroot.mkdir(parents=True)
    f16 = ["cutlass3x_sm90_tensorop_gemm_f16_f16_f32_void_f16_%dx128x64_1x1x1_0_tnn_align8" % t
           for t in (64, 128, 256)]
    fp8 = ["cutlass3x_sm90_tensorop_gemm_e4m3_e4m3_f32_void_f16_%dx128x128_1x1x1_0_tnn_align16" % t
           for t in (64, 128)]
    s8 = ["cutlass3x_sm90_tensorop_gemm_s8_s8_s32_void_s32_128x128x128_1x1x1_0_tnn_align16"]
    body = "\n".join('  manifest.append(new Operation("%s"));' % n for n in f16 + fp8 + s8)
    (genroot / "all_sm90_tensorop_gemm_operations.cu").write_text(
        "#include <x>\nvoid init() {\n" + body + "\n}\n")
    check("census: reads the quoted operation names out of the generated sources",
          len(gen(str(tmp))) == 6)

    # The second source: a per-configuration .cu named after the kernel. Layout-independent by
    # design -- either source alone is enough to attribute a family.
    only_name = tmp / "byname" / "tools" / "library" / "generated" / "gemm" / "90"
    only_name.mkdir(parents=True)
    (only_name / (fp8[0] + ".cu")).write_text("// no quoted name in here at all\n")
    check("census: also reads kernel names off the generated FILE names",
          gen(str(tmp / "byname")) == [fp8[0]])

    counts = census(str(tmp), ["f16", "fp8", "int8"], p90, 800)
    check("census: per-family counts",
          counts["f16"] == 3 and counts["fp8"] == 2 and counts["int8"] == 1, str(counts))
    for dtypes, needle, label in (
        (["f16", "bf16"], "ZERO kernels", "a family that selected nothing aborts"),
        (["f16"], "cutlass-max-kernels", "the kernel ceiling aborts"),
    ):
        cap = 2 if needle == "cutlass-max-kernels" else 800
        pats = p90 if needle == "cutlass-max-kernels" else ns["_cutlass_patterns"]("90a", dtypes)
        try:
            census(str(tmp), dtypes, pats, cap)
            check("census: " + label, False)
        except SystemExit as e:
            check("census: " + label, needle in str(e))
    try:
        census(str(tmp / "absent"), ["f16"], p90, 0)
        check("census: an unreadable generated tree aborts", False)
    except SystemExit as e:
        check("census: an unreadable generated tree aborts", "no generated kernel names" in str(e))


def test_profiler_dtypes(ns):
    """`::cutlass --dtype s8` must ask for an s32 accumulator: f32 there selects no kernel, and a
    profiler that finds no kernel prints an empty column that reads as a missing library feature."""
    tbl = ns["_CUTLASS_PROFILER_DTYPE"]
    check("profiler dtype: int8 accumulates in s32", tbl["s8"] == ("int8", "s32", "s32"))
    check("profiler dtype: fp8 accumulates in f32", tbl["e4m3"][0] == "fp8" and tbl["e4m3"][2] == "f32")
    check("profiler dtype: f16 is unchanged from the pre-2026-08-10 behaviour",
          tbl["f16"] == ("f16", "f16", "f32"))
    check("profiler dtype: plain f32 claims no family (nothing to check the manifest against)",
          tbl["f32"][0] == "")


def test_flash_attention(ns, tmp):
    fa2 = ns["_FA2_ARCH"]
    check("fa2: Ada/Ampere-consumer map to 80, NOT to their own cc",
          fa2["sm_89"] == "80" and fa2["sm_86"] == "80")
    check("fa2: Hopper maps to 90", fa2["sm_90"] == "90")
    check("fa2: Turing is absent (FA2 needs Ampere+)", "sm_75" not in fa2)

    pre = ns["_FA3_DISABLE_PRESETS"]
    check("fa3: `full` disables nothing", pre["full"] == ())
    for k in ("PAGEDKV", "SPLIT", "PACKGQA", "VARLEN", "FP8"):
        check("fa3: `decode` KEEPS " + k, k not in pre["decode"])
        check("fa3: `minimal` still removes " + k + " (it reproduces the 2026-08 artifact)",
              k in pre["minimal"])
    check("fa3: `full` demands a longer timeout than `decode`",
          ns["_FA3_MIN_TIMEOUT"]["full"] > ns["_FA3_MIN_TIMEOUT"]["decode"])

    bj = ns["_build_jobs"]
    check("jobs: bounded by WK_PEER_MEM as well as by cores",
          bj(3.0) == min(os.cpu_count() or 1, 10), str(bj(3.0)))
    check("jobs: never zero", bj(1000.0) == 1)

    vw = ns["_verify_fa_wheel"]

    def wheel(path, iface, so_bytes, so="flash_attn_2_cuda.cpython-311.so"):
        with zipfile.ZipFile(path, "w") as zf:
            zf.writestr("flash_attn/flash_attn_interface.py", iface)
            if so_bytes is not None:
                zf.writestr(so, so_bytes)
        return str(path)

    KV = "def flash_attn_with_kvcache(q, k, v, block_table=None):\n    pass\n"
    NOKV = "def flash_attn_func(q, k, v):\n    pass\n"
    ELF = b"\x7fELF" + b"pad" * 400000
    good = wheel(tmp / "good.whl", KV, ELF + b"fwd_kvcache")
    info = vw(good, "flash-attn", want_kvcache=True)
    check("wheel: a complete FA2 wheel passes", info["kvcache_symbol"] and info["kvcache_api"])

    for path, iface, so_bytes, needle, label in (
        (tmp / "apionly.whl", KV, ELF, "built WITHOUT the kvcache entry point",
         "the python api is present but the kernel is not"),
        (tmp / "nokv.whl", NOKV, ELF, "built WITHOUT the kvcache entry point",
         "neither the api nor the kernel"),
        (tmp / "pyonly.whl", KV, None, "no compiled CUDA extension",
         "a python-only wheel (the CUDA build was skipped)"),
    ):
        p = wheel(path, iface, so_bytes)
        try:
            vw(p, "flash-attn", want_kvcache=True)
            check("wheel: rejects " + label, False)
        except SystemExit as e:
            check("wheel: rejects " + label, needle in str(e))


def test_cache_stats(ns, tmp):
    ds = ns["_dir_stats"]
    check("cache: an absent directory reports age -1", ds(str(tmp / "absent")) == (0, 0, -1.0))
    files, size, age = ds(str(tmp))
    check("cache: a populated directory reports files, bytes and age",
          files >= 3 and size > 0 and age >= 0, "%d files, %d B" % (files, size))


def test_clock_provenance(ns, path):
    """The clock lines are provenance, so their failure mode must be a labelled line, never an
    exception that kills a metered round -- and the wiring must actually bracket the cargo run,
    because a snapshot only before (or only after) cannot show a shift between arms."""
    snap = ns["_clock_snapshot"]
    lock = ns["_clock_lock_attempt"]
    absent = "wk-selftest-no-such-nvidia-smi"

    line = snap({}, "selftest", smi=absent)
    check("clock: snapshot without nvidia-smi returns a labelled line, no raise",
          isinstance(line, str) and line.startswith("[clock] selftest:")
          and "unavailable" in line, line)
    check("clock: lock attempt without nvidia-smi reports False, no raise",
          lock({}, smi=absent) is False)

    # Wiring: parse the real entry-point bodies and count the calls. `bench` is the timing path
    # (lock attempt + before/after); `test` records before/after only.
    calls = {"bench": [], "test": []}
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.FunctionDef) and node.name in calls:
            for sub in ast.walk(node):
                if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name):
                    if sub.func.id in ("_clock_snapshot", "_clock_lock_attempt"):
                        calls[node.name].append(sub.func.id)
    check("clock: bench attempts the lock once and snapshots before AND after",
          calls["bench"].count("_clock_lock_attempt") == 1
          and calls["bench"].count("_clock_snapshot") >= 2, repr(calls["bench"]))
    check("clock: test snapshots before AND after the device suite",
          calls["test"].count("_clock_snapshot") >= 2, repr(calls["test"]))


def main():
    path = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_APP
    print("selftest target: %s" % path)
    ns = load(path)
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="wk_modal_selftest_"))
    test_filter_semantics(ns)
    test_dtypes(ns)
    p90 = test_patterns(ns)
    test_census(ns, p90, tmp)
    test_profiler_dtypes(ns)
    test_flash_attention(ns, tmp)
    test_cache_stats(ns, tmp)
    test_clock_provenance(ns, path)
    print("")
    print("FAILED: %s" % (", ".join(FAILS) if FAILS else "nothing"))
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
