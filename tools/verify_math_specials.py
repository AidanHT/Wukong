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
    # Uniform (input tuple, expected) representation also covers binary intrinsics.
    cases = {name: [((x,), expected) for x, expected in values]
             for name, values in cases.items()}
    axes = (0.0, -0.0, 1.0, -1.0, inf, -inf, nan, -nan)
    cases["atan2"] = [((y, x), math.atan2(y, x)) for y in axes for x in axes]
    tiny = struct.unpack("<f", struct.pack("<I", 1))[0]
    largest = struct.unpack("<f", struct.pack("<I", 0x7F7FFFFF))[0]
    cases["atan2"].extend([((-tiny, largest), -0.0), ((tiny, largest), 0.0)])
    cases["hypot"] = [((a, b), inf if math.isinf(a) or math.isinf(b)
                       else nan if math.isnan(a) or math.isnan(b) else 0.0)
                      for a in (0.0, -0.0, inf, -inf, nan, -nan)
                      for b in (0.0, -0.0, inf, -inf, nan, -nan)]
    cases["hypot"].extend([((3.0, 4.0), 5.0), ((largest, 0.0), largest), ((tiny, 0.0), tiny)])
    cases["pow"] = [((x, y), 1.0) for x in axes for y in (0.0, -0.0)]
    cases["pow"].extend([((1.0, y), 1.0) for y in axes])
    cases["pow"].extend([((nan, 2.0), nan), ((2.0, nan), nan), ((inf, 2.0), inf),
                         ((inf, -2.0), 0.0), ((0.0, 2.0), 0.0), ((0.0, -2.0), inf)])
    n = 17  # Four SIMD groups and a scalar tail.
    source = []
    arities = {name: len(values[0][0]) for name, values in cases.items()}
    for name, arity in arities.items():
        names = ["x", "y"][:arity]
        scalar_params = ",".join(f"{x}:f32" for x in names)
        vector_params = ",".join(f"{x}:[f32;{n}]" for x in names)
        vector_args = ",".join(f"{x}[i]" for x in names)
        source.append(f"fn s_{name}({scalar_params})->f32 {{ return {name}({','.join(names)}); }}")
        source.append(f"@generic fn v_{name}({vector_params}, mut out:[f32;{n}]) {{ "
                      f"for i in 0..{n} {{ out[i] = {name}({vector_args}); }} }}")
    # Explicit disjoint call sites provide the frontend's alias proof. Never executed by the JIT.
    source.append(f"fn prove_disjoint() {{ let x:[f32;{n}]=[1.0;{n}]; "
                  f"let y:[f32;{n}]=[2.0;{n}]; let mut out:[f32;{n}]=[0.0;{n}]; "
                  + " ".join(f"v_{name}({'x,y' if arity == 2 else 'x'},out);"
                             for name, arity in arities.items()) + " }")
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
            arity = arities[name]
            scalar = ctypes.CFUNCTYPE(ctypes.c_float, *([ctypes.c_float] * arity))(scalar_address)
            vector = ctypes.CFUNCTYPE(None, *([ctypes.c_void_p] * (arity + 1)))(vector_address)
            for inputs, expected in values:
                actual = scalar(*inputs)
                assert matches(actual, expected), (opt, name, inputs, actual, expected)
                checks += 1
            for shift in range(len(values)):
                rotated = [values[(i + shift) % len(values)] for i in range(n)]
                inputs = [(ctypes.c_float * n)(*(xs[j] for xs, _ in rotated))
                          for j in range(arity)]
                outputs = (ctypes.c_float * n)()
                vector(*inputs, outputs)
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
