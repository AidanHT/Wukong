"""Verify and execute the exact MIR regression modules exported by semantic_tests.

Usage: python crates/wukong_codegen_llvm/tools/verify_semantics.py OUTPUT_DIRECTORY
Requires an installed llvmlite; performs no installation or network access.
"""
import ctypes
import math
import os
from pathlib import Path
import struct
import sys

os.environ["LLVMLITE_ENABLE_OPAQUE_POINTERS"] = "1"
import llvmlite.binding as llvm

root = Path(sys.argv[1])
paths = sorted(root.glob("*.ll"))
assert len(paths) == 40, f"expected exactly 40 semantic regression modules, got {len(paths)}"
llvm.initialize()
llvm.initialize_native_target()
llvm.initialize_native_asmprinter()
machine = llvm.Target.from_default_triple().create_target_machine(
    cpu=str(llvm.get_host_cpu_name()), features=llvm.get_host_cpu_features().flatten()
)
engine = llvm.create_mcjit_compiler(llvm.parse_assembly(""), machine)
modules = []
for path in paths:
    module = llvm.parse_assembly(path.read_text())
    module.verify()
    engine.add_module(module)
    modules.append(module)
engine.finalize_object()

float_types = {"f32": ctypes.c_float, "f64": ctypes.c_double}
integer_types = {
    # MIR i1 is explicitly zero-extended to i8 by the test wrapper; unannotated LLVM i1
    # returns do not promise that the other seven C ABI return-byte bits are zero.
    ("i", 1): ctypes.c_uint8, ("u", 1): ctypes.c_uint8,
    ("i", 8): ctypes.c_int8, ("u", 8): ctypes.c_uint8,
    ("i", 16): ctypes.c_int16, ("u", 16): ctypes.c_uint16,
    ("i", 32): ctypes.c_int32, ("u", 32): ctypes.c_uint32,
    ("i", 64): ctypes.c_int64, ("u", 64): ctypes.c_uint64,
}


def saturation(value, sign, width):
    effective = 32 if width == 1 else width
    low = -(1 << (effective - 1)) if sign == "i" else 0
    high = (1 << (effective - (sign == "i"))) - 1
    if math.isnan(value):
        result = 0
    elif value <= low:
        result = low
    elif value >= high:
        result = high
    else:
        result = math.trunc(value)
    return result & 1 if width == 1 else result


def aligned_array(ctype, values):
    backing = ctypes.create_string_buffer(ctypes.sizeof(ctype) * len(values) + 63)
    array = (ctype * len(values)).from_address((ctypes.addressof(backing) + 63) & ~63)
    array[:] = values
    return backing, array


cast_checks = 0
for source, ft in float_types.items():
    for sign in ["i", "u"]:
        for width in [1, 8, 16, 32, 64]:
            it = integer_types[sign, width]
            name = f"sat_{source}_{sign}{width}"
            fn = ctypes.CFUNCTYPE(it, ft)(engine.get_function_address(name))
            samples = [math.nan, math.inf, -math.inf, -0.0, 0.0, 1.9, -1.9, 2.0, 3.0]
            for n in [7, 8, 15, 16, 31, 32, 63, 64]:
                for offset in [-1, 0, 1]:
                    value = float((1 << n) + offset)
                    samples.extend([value, -value])
            rounded = [ft(x).value for x in samples]
            for value in rounded:
                actual, want = fn(value), saturation(value, sign, width)
                assert actual == want, (name, value, actual, want)
                cast_checks += 1
            if width == 1:
                continue
            vector = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_void_p)(
                engine.get_function_address(f"sat_vec_{source}_{sign}{width}")
            )
            for start in range(0, len(rounded), 4):
                values = (rounded[start:start + 4] + [0.0] * 4)[:4]
                input_backing, inputs = aligned_array(ft, values)
                output_backing, outputs = aligned_array(it, [0] * 4)
                vector(inputs, outputs)
                for value, actual in zip(values, outputs):
                    want = saturation(value, sign, width)
                    assert actual == want, (name, "vector", value, actual, want)
                    cast_checks += 1


def same_float(actual, expected, source):
    if math.isnan(expected):
        return math.isnan(actual)
    fmt = "f" if source == "f32" else "d"
    return struct.pack(fmt, actual) == struct.pack(fmt, expected)


fma_checks = 0
for source, ft in float_types.items():
    precision = 23 if source == "f32" else 52
    delta = 2.0 ** -precision
    # First case has exact product 1 - delta**2. Separate multiplication rounds to 1,
    # producing zero; mandatory single rounding must return -delta**2 exactly.
    cases = [
        (1 + delta, 1 - delta, -1.0, -(delta * delta)),
        (2.0, 3.0, 4.0, 10.0),
        (-0.0, 2.0, -0.0, -0.0),
        (math.inf, 2.0, 1.0, math.inf),
        (math.inf, 0.0, 1.0, math.nan),
        (math.nan, 2.0, 1.0, math.nan),
        (1.0, 1.0, -1.0, 0.0),
        (-math.inf, 2.0, 1.0, -math.inf),
    ]
    fn = ctypes.CFUNCTYPE(ft, ft, ft, ft)(engine.get_function_address(f"fma_{source}"))
    for a, b, c, want in cases:
        actual = fn(a, b, c)
        assert same_float(actual, want, source), (source, a, b, c, actual, want)
        fma_checks += 1
    vector = ctypes.CFUNCTYPE(None, *([ctypes.c_void_p] * 4))(
        engine.get_function_address(f"fma_vec_{source}")
    )
    for start in [0, 4]:
        batch = cases[start:start + 4]
        ab, av = aligned_array(ft, [x[0] for x in batch])
        bb, bv = aligned_array(ft, [x[1] for x in batch])
        cb, cv = aligned_array(ft, [x[2] for x in batch])
        ob, ov = aligned_array(ft, [0] * 4)
        vector(av, bv, cv, ov)
        for actual, case in zip(ov, batch):
            assert same_float(actual, case[3], source), (source, "vector", case, actual)
            fma_checks += 1

print(f"LLVM {llvm.llvm_version_info}: {len(paths)} modules verified; "
      f"{cast_checks} saturating conversion and {fma_checks} exact FMA JIT checks passed")
