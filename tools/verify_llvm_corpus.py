"""Verify emitted LLVM modules with llvm-as or the installed llvmlite verifier.

Usage: python tools/verify_llvm_corpus.py --compiler target/debug/wukongc.exe --report out.json
Requires llvm-as on PATH (or --llvm-as), or llvmlite. Missing tooling is an error.
This checks IR structure, not executable semantics; the runtime ABI of every declaration is the
link-ABI law in wukong_codegen_llvm's closeout tests.

A module the compiler *declines* to emit (C0001, exit 2, an `LLVM emission [<reason>]` message
from wukong_codegen_llvm::DECLINED) is counted separately from a verifier failure: the report's
`declined` total is held to --max-declines (default 0, the ratchet crates/wukongc/tests/emit.rs
also pins), while any failure fails the run.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiler", type=Path, required=True)
    parser.add_argument("--llvm-as")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--max-declines", type=int, default=0)
    args = parser.parse_args()
    compiler = args.compiler.resolve(strict=True)
    assembler = args.llvm_as or shutil.which("llvm-as")
    if assembler:
        version = subprocess.run([assembler, "--version"], capture_output=True, text=True, check=True)
        verifier = version.stdout.strip()

        def verify(ir):
            result = subprocess.run(
                [assembler, "-o", os.devnull, "-"], input=ir,
                capture_output=True, text=True, timeout=30,
            )
            if result.returncode:
                raise ValueError(result.stderr.strip())
    else:
        # LLVM 15-backed llvmlite needs this before importing its binding module.
        os.environ["LLVMLITE_ENABLE_OPAQUE_POINTERS"] = "1"
        try:
            import llvmlite
            import llvmlite.binding as llvm
        except ImportError:
            parser.error("install llvm-as or llvmlite; no LLVM verifier is available")
        verifier = f"llvmlite {llvmlite.__version__}, LLVM {llvm.llvm_version_info}"

        def verify(ir):
            with llvm.parse_assembly(ir) as module:
                module.verify()

    env = os.environ.copy()
    for name in (
        "WUKONG_NO_RECOGNIZE", "WUKONG_NO_VECTORIZE", "WUKONG_NO_UNROLL",
        "WUKONG_NO_IV_RAMP", "WUKONG_P4_NO_256", "WUKONG_VEC_TRACE",
    ):
        env.pop(name, None)
    fixtures = sorted((args.root / "tests" / "run").glob("*.wk"))
    if not fixtures:
        parser.error("the run corpus is empty")
    rows = []
    for fixture in fixtures:
        for opt in (0, 2):
            row = {"fixture": fixture.name, "opt": opt, "passed": False}
            try:
                result = subprocess.run(
                    [str(compiler), "--emit=llvm-ir", f"-O{opt}", str(fixture.resolve())],
                    cwd=args.root, env=env, capture_output=True, text=True, timeout=60,
                )
                if result.returncode == 2 and "LLVM emission [" in result.stderr:
                    row["declined"] = True
                    row["error"] = result.stderr.strip()
                    print(f"DECLINED {fixture.name} -O{opt}: {row['error']}", file=sys.stderr)
                    rows.append(row)
                    continue
                if result.returncode:
                    raise ValueError(f"compiler exit {result.returncode}: {result.stderr.strip()}")
                if not result.stdout.strip():
                    raise ValueError("compiler emitted an empty module")
                verify(result.stdout)
                row["passed"] = True
            except (ValueError, RuntimeError, OSError, subprocess.TimeoutExpired) as error:
                row["error"] = str(error)
                print(f"FAIL {fixture.name} -O{opt}: {error}", file=sys.stderr)
            rows.append(row)
    report = {
        "compiler": str(compiler),
        "compiler_sha256": hashlib.sha256(compiler.read_bytes()).hexdigest(),
        "verifier": verifier,
        "passed": sum(row["passed"] for row in rows),
        "failed": sum(not row["passed"] and not row.get("declined") for row in rows),
        "declined": sum(bool(row.get("declined")) for row in rows),
        "max_declines": args.max_declines,
        "rows": rows,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        f"LLVM verification: {report['passed']} passed, {report['failed']} failed, "
        f"{report['declined']} declined (ratchet {args.max_declines}) ({verifier})"
    )
    return 1 if report["failed"] or report["declined"] > args.max_declines else 0


if __name__ == "__main__":
    sys.exit(main())
