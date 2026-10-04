"""Execute exported trig LLVM through the actual Rust runtime, with independent f64 checks.

Consumes the artifacts exported by the trig_range integration test and an explicit runtime
rlib from that same build. Requires the installed llvmlite and Rust toolchain. No timing or
performance claim is made. All generated files and command results are kept under --output.
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


BRIDGE = """
#[unsafe(no_mangle)]
pub extern "C" fn test_trig(x: f32, op: i64) -> f32 {
    wukong_runtime::wukong_rt_trig_f32(x, op)
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn test_vmath(x: *const f32, y: *mut f32, n: i64, op: i64) {
    unsafe { wukong_runtime::wukong_vmath_f32(x, y, n, op) }
}
"""
OPS = {"sin": 15, "cos": 16, "tan": 32}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def float_bits(value):
    return struct.unpack("<I", struct.pack("<f", value))[0]


def as_float(value):
    return struct.unpack("<f", struct.pack("<I", value))[0]


def samples():
    values = [float_bits(x) for x in (0.0, -0.0, 0.125, -0.5, 1.0, -1.0)]
    for center in (float_bits(4096.0), 0x4F6E6B28, 0x7F7FFFFF):
        for delta in (-1, 0, 1):
            value = center + delta
            values.extend((value, value | 0x80000000))
    for exponent in range(140, 255, 11):
        value = (exponent << 23) | 0x00555555
        values.extend((value, value | 0x80000000))
    for k in (0, 1, 17, 511, 1303):
        for angle in ((k + 0.5) * math.pi, (k + 0.5) * math.pi / 2):
            center = float_bits(angle)
            for delta in (-1, 0, 1):
                value = center + delta
                values.extend((value, value | 0x80000000))
    values.extend((1, 0x80000001, 0x007FFFFF, 0x807FFFFF,
                   0x7F800000, 0xFF800000, 0x7FC12345, 0xFFC23456, 0x7FA12345))
    return values


def assert_accuracy(actual_bits, input_bits, name):
    actual, x = as_float(actual_bits), as_float(input_bits)
    if not math.isfinite(x):
        assert math.isnan(actual), (name, hex(input_bits), actual)
        return
    expected = getattr(math, name)(x)
    # Existing vmath_matches_libm and vmath_inverse_trig combined abs+relative bounds.
    tolerance = 5e-5 if name == "tan" else 1e-4
    assert math.isfinite(actual), (name, x, actual)
    assert abs(actual - expected) <= tolerance + tolerance * abs(expected), (
        name, x, actual, expected)
    if x == 0.0:
        assert actual_bits == (float_bits(1.0) if name == "cos" else input_bits)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--runtime-rlib", type=Path, required=True)
    parser.add_argument("--rustc", default="rustc")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    artifacts = args.artifacts.resolve(strict=True)
    runtime = args.runtime_rlib.resolve(strict=True)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    runtime_hash = digest(runtime)
    source = output / "bridge.rs"
    source.write_text(BRIDGE, encoding="utf-8")
    library = output / ("bridge.dll" if os.name == "nt" else "libbridge.so")
    command = [args.rustc, "--crate-type=cdylib", "--crate-name=wukong_trig_bridge",
               "--edition=2024", "-D", "warnings", "-L", f"dependency={runtime.parent}",
               "--extern", f"wukong_runtime={runtime}", str(source), "-o", str(library)]
    with (output / "build.stdout.log").open("w", encoding="utf-8") as stdout, (
            output / "build.stderr.log").open("w", encoding="utf-8") as stderr:
        built = subprocess.run(command, stdout=stdout, stderr=stderr, timeout=120)
    (output / "build.json").write_text(json.dumps({"command": command,
        "exit": built.returncode}, indent=2) + "\n", encoding="utf-8")
    assert built.returncode == 0, "Rust bridge build failed; see build.stderr.log"
    assert digest(runtime) == runtime_hash, "runtime changed during bridge build"
    bridge = ctypes.CDLL(str(library))
    runtime_array = bridge.test_vmath
    runtime_array.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64]
    runtime_array.restype = None
    llvm.initialize()
    llvm.initialize_native_target()
    llvm.initialize_native_asmprinter()
    llvm.add_symbol("wukong_rt_trig_f32", ctypes.cast(bridge.test_trig, ctypes.c_void_p).value)
    llvm.add_symbol("wukong_vmath_f32", ctypes.cast(runtime_array, ctypes.c_void_p).value)
    expected = {}
    values = samples()
    for name, opcode in OPS.items():
        for value in set(values + [float_bits((i - 8) / 8) for i in range(17)]):
            x, y = ctypes.c_uint32(value), ctypes.c_uint32(0)
            runtime_array(ctypes.byref(x), ctypes.byref(y), 1, opcode)
            assert x.value == value, "scalar runtime mutated its input"
            assert_accuracy(y.value, value, name)
            expected[name, value] = y.value

    checks, input_checks, guard_checks = 0, 0, 0
    modules = []
    for name in OPS:
        for recognized in (0, 1):
            for level in (0, 2):
                path = artifacts / f"{name}-rec{recognized}-O{level}.ll"
                text = path.read_text(encoding="utf-8")
                module = llvm.parse_assembly(text)
                module.verify()
                if not recognized:
                    body = str(module.get_function("entry"))
                    assert "<4 x float>" in body, "generic SIMD path missing"
                    assert "@wukong_rt_trig_f32" in body, "exceptional fallback missing"
                machine = llvm.Target.from_default_triple().create_target_machine(
                    cpu=str(llvm.get_host_cpu_name()), features=llvm.get_host_cpu_features().flatten())
                engine = llvm.create_mcjit_compiler(module, machine)
                engine.finalize_object()
                address = engine.get_function_address("entry")
                assert address, "entry symbol missing"
                entry = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_void_p)(address)
                for value in values:
                    for mixed in (False, True):
                        lanes = ([float_bits((i - 8) / 8) for i in range(17)]
                                 if mixed else [value] * 17)
                        if mixed:
                            lanes[0] = 0x80000000
                            for lane in (1, 7, 8, 15, 16):
                                lanes[lane] = value
                        inputs = (ctypes.c_uint32 * 17)(*lanes)
                        outputs = (ctypes.c_uint32 * 25)(*([0x4A123456] * 25))
                        entry(inputs, ctypes.byref(outputs, 16))
                        for lane, original in enumerate(lanes):
                            actual = outputs[lane + 4]
                            assert actual == expected[name, original], (
                                path.name, mixed, lane, hex(original), hex(actual),
                                hex(expected[name, original]))
                            assert_accuracy(actual, original, name)
                            assert inputs[lane] == original, "LLVM mutated its input"
                            checks += 1
                            input_checks += 1
                        for lane in (*range(4), *range(21, 25)):
                            assert outputs[lane] == 0x4A123456, "output guard overwritten"
                            guard_checks += 1
                modules.append({"file": str(path), "sha256": digest(path)})
                engine.close()
    assert digest(runtime) == runtime_hash, "runtime changed during execution"
    report = {"runtime_rlib": str(runtime), "runtime_sha256": runtime_hash,
              "bridge_sha256": digest(library), "bridge_source_sha256": digest(source),
              "llvm_version": llvm.llvm_version_info, "modules": modules,
              "output_checks": checks, "input_checks": input_checks, "guard_checks": guard_checks,
              "samples": len(values), "nan_contract": "raw-bit agreement with actual Rust runtime",
              "reference": "Python f64 libm; existing runtime combined absolute/relative bounds"}
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"LLVM trig: {len(modules)} modules; {checks} output, {input_checks} input, "
          f"{guard_checks} guard checks passed")


if __name__ == "__main__":
    main()
