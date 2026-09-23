"""Execute source-level math endpoints through LLVM, independently of Wukong's runtime.

Requires the installed llvmlite verifier/JIT. Keeps generated source, LLVM modules and a
compiler-hash report under --output. NaNs are checked by class; other results by f32 bits.
"""
import argparse
import ctypes
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import subprocess

os.environ["LLVMLITE_ENABLE_OPAQUE_POINTERS"] = "1"
import llvmlite.binding as llvm


def bits(value):
    return struct.pack("<f", value)


def matches(actual, expected):
    return math.isnan(actual) if math.isnan(expected) else bits(actual) == bits(expected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiler", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    compiler = args.compiler.resolve(strict=True)
    compiler_hash = hashlib.sha256(compiler.read_bytes()).hexdigest()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    inf, nan = math.inf, math.nan
    saturation = struct.unpack("<f", struct.pack("<I", 0x7F3504A4))[0]
    cases = {
        "exp": [(0.0, 1.0), (-0.0, 1.0), (inf, inf), (-inf, 0.0),
                (100.0, saturation), (-100.0, 0.0)],
        "log": [(0.0, -inf), (-0.0, -inf), (1.0, 0.0), (inf, inf),
                (-inf, nan), (-1.0, nan)],
        "expm1": [(0.0, 0.0), (-0.0, -0.0), (inf, inf), (-inf, -1.0), (-100.0, -1.0)],
        "log1p": [(0.0, 0.0), (-0.0, -0.0), (inf, inf), (-inf, nan), (-1.0, -inf)],
        "sigmoid": [(inf, 1.0), (-inf, 0.0), (0.0, 0.5), (-0.0, 0.5)],
        "softplus": [(inf, inf), (-inf, 0.0)],
        "logsigmoid": [(inf, -0.0), (-inf, -inf)],
        "asinh": [(inf, inf), (-inf, -inf), (0.0, 0.0), (-0.0, -0.0)],
        "cbrt": [(inf, inf), (-inf, -inf), (0.0, 0.0), (-0.0, -0.0), (1.0, 1.0), (-1.0, -1.0)],
        "mish": [(inf, inf), (-inf, nan), (0.0, 0.0), (-0.0, -0.0)],
        "silu": [(inf, inf), (-inf, nan), (0.0, 0.0), (-0.0, -0.0)],
        "sinh": [(inf, inf), (-inf, -inf), (0.0, 0.0)],
        "cosh": [(inf, inf), (-inf, inf), (0.0, 1.0)],
    }
    for values in cases.values():
        values.extend([(nan, nan), (-nan, nan)])
    n = 17  # Four SIMD groups and a scalar tail.
    source = []
    for name in cases:
        source.append(f"fn s_{name}(x:f32)->f32 {{ return {name}(x); }}")
        source.append(f"@generic fn v_{name}(x:[f32;{n}], mut out:[f32;{n}]) {{ "
                      f"for i in 0..{n} {{ out[i] = {name}(x[i]); }} }}")
    # Explicit disjoint call sites provide the frontend's alias proof. Never executed by the JIT.
    source.append(f"fn prove_disjoint() {{ let x:[f32;{n}]=[1.0;{n}]; "
                  f"let mut out:[f32;{n}]=[0.0;{n}]; "
                  + " ".join(f"v_{name}(x,out);" for name in cases) + " }")
    source_path = output / "math_specials.wk"
    source_path.write_text("\n".join(source), encoding="utf-8")
    env = os.environ.copy()
    for key in ("WUKONG_NO_RECOGNIZE", "WUKONG_NO_VECTORIZE", "WUKONG_NO_UNROLL",
                "WUKONG_NO_IV_RAMP", "WUKONG_P4_NO_256", "WUKONG_VEC_TRACE"):
        env.pop(key, None)
    llvm.initialize()
    llvm.initialize_native_target()
    llvm.initialize_native_asmprinter()
    checks = 0
    for opt in (0, 2):
        emitted = subprocess.run([str(compiler), "--emit=llvm-ir", f"-O{opt}", str(source_path)],
                                 capture_output=True, text=True, timeout=60, env=env)
        if emitted.returncode:
            raise RuntimeError(f"compiler failed at O{opt}: {emitted.stderr}")
        (output / f"math_specials_O{opt}.ll").write_text(emitted.stdout, encoding="utf-8")
        assert "<4 x float>" in emitted.stdout, "expected actual SIMD math in emitted LLVM"
        module = llvm.parse_assembly(emitted.stdout)
        module.verify()
        machine = llvm.Target.from_default_triple().create_target_machine(
            cpu=str(llvm.get_host_cpu_name()), features=llvm.get_host_cpu_features().flatten())
        engine = llvm.create_mcjit_compiler(module, machine)
        engine.finalize_object()
        for name, values in cases.items():
            assert "<4 x float>" in str(module.get_function(f"v_{name}")), (opt, name, "SIMD missing")
            scalar_address = engine.get_function_address(f"s_{name}")
            vector_address = engine.get_function_address(f"v_{name}")
            assert scalar_address and vector_address, name
            scalar = ctypes.CFUNCTYPE(ctypes.c_float, ctypes.c_float)(scalar_address)
            vector = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_void_p)(vector_address)
            for x, expected in values:
                actual = scalar(x)
                assert matches(actual, expected), (opt, name, x, actual, expected)
                checks += 1
            for shift in range(len(values)):
                rotated = [values[(i + shift) % len(values)] for i in range(n)]
                inputs = (ctypes.c_float * n)(*(x for x, _ in rotated))
                outputs = (ctypes.c_float * n)()
                vector(inputs, outputs)
                for actual, (x, expected) in zip(outputs, rotated):
                    assert matches(actual, expected), (opt, name, "vector/tail", x, actual, expected)
                    checks += 1
        engine.close()
    assert hashlib.sha256(compiler.read_bytes()).hexdigest() == compiler_hash, "compiler changed during verification"
    report = {"compiler": str(compiler), "compiler_sha256": compiler_hash,
              "llvm_version": llvm.llvm_version_info, "modules_verified": 2,
              "operations": len(cases), "checks_passed": checks, "nan_contract": "classification"}
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"LLVM source math: 2 modules verified; {checks} checks passed across {len(cases)} operations")


if __name__ == "__main__":
    main()
