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
# The 46 modules executed below are the saturating-conversion, FMA and bf16-narrowing regressions;
# every other exported module (e.g. `lowp_fma`, which pins only its text) is parsed and verified.
executed = [p for p in paths if p.stem.startswith(("sat_", "fma_", "bf16_"))]
assert len(executed) == 46, f"expected exactly 46 executed regression modules, got {len(executed)}"
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


def adjacent(value, upward, source):
    """The immediate floating neighbor, with no f64 intermediate step for f32."""
    if source == "f64":
        return math.nextafter(value, math.inf if upward else -math.inf)
    assert value != 0.0 and math.isfinite(value)
    bits = struct.unpack("!I", struct.pack("!f", value))[0]
    bits += 1 if (value > 0.0) == upward else -1
    return struct.unpack("!f", struct.pack("!I", bits))[0]


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
                # Integer +/-1 collapses at large floating powers of two. Explicit neighbors
                # exercise the last representable in-range value and first out-of-range value.
                for boundary in [ft(2.0 ** n).value, ft(-(2.0 ** n)).value]:
                    below = adjacent(boundary, False, source)
                    above = adjacent(boundary, True, source)
                    assert below < boundary < above
                    assert ft(below).value == below and ft(above).value == above
                    samples.extend([below, boundary, above])
            rounded = [ft(x).value for x in samples]
            for value in rounded:
                actual, want = fn(value), saturation(value, sign, width)
                assert actual == want, (name, value, actual, want)
                cast_checks += 1
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
        # NaN classification is the contract here; payload bits are intentionally not compared.
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



def bf16_reference(bits):
    """`wukong_runtime::f32_to_bf16_bits`, transcribed."""
    if bits & 0x7FFFFFFF > 0x7F800000:
        return ((bits >> 16) | 0x0040) & 0xFFFF
    return ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16) & 0xFFFF


bf16_samples = [
    0x00000000, 0x80000000, 0x00000001, 0x807FFFFF, 0x3F800000, 0x3F808000, 0x3F818000,
    0x3F808001, 0x3F807FFF, 0x7F7FFFFF, 0x7F7F8000, 0x7F7F7FFF, 0xFF7FFFFF, 0x7F800000,
    0xFF800000, 0x7F800001, 0x7FBFFFFF, 0x7FC00000, 0xFFC12345, 0x7F80FFFF, 0xFF800001,
    0x0000FFFF, 0x00008000, 0x00018000, 0x3DCCCCCD, 0xBDCCCCCD,
]
state = 0x9E3779B9
for _ in range(4000):
    state = (state * 6364136223846793005 + 1442695040888963407) & (2**64 - 1)
    bf16_samples.append(state >> 32)
bf16_checks = 0
scalar = ctypes.CFUNCTYPE(ctypes.c_uint16, ctypes.c_void_p)(engine.get_function_address("bf16_bits"))
vector = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_void_p)(
    engine.get_function_address("bf16_vec_bits")
)
for start in range(0, len(bf16_samples), 4):
    lanes = (bf16_samples[start:start + 4] + [0] * 4)[:4]
    ib, iv = aligned_array(ctypes.c_uint32, lanes)
    ob, ov = aligned_array(ctypes.c_uint16, [0] * 4)
    vector(iv, ov)
    for k, x in enumerate(lanes):
        cell = (ctypes.c_uint32 * 1)(x)
        want = bf16_reference(x)
        assert scalar(cell) == want, ("bf16_bits", hex(x), hex(scalar(cell)), hex(want))
        assert ov[k] == want, ("bf16_vec_bits", hex(x), hex(ov[k]), hex(want))
        bf16_checks += 2

print(f"LLVM {llvm.llvm_version_info}: {len(paths)} modules verified ({len(executed)} executed); "
      f"{cast_checks} saturating conversion, {fma_checks} exact FMA and {bf16_checks} bf16 "
      "narrowing JIT checks passed")
