"""Narrow GPT-2 gate tests (NumPy arithmetic only). Never import Torch or load a model."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np


SOURCE = Path(__file__).resolve().parents[1] / "bench_gpt2_torch.py"
spec = importlib.util.spec_from_file_location("gpt2_gate_under_test", SOURCE)
peer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(peer)


class GateTests(unittest.TestCase):
    def setUp(self):
        self.row = [0.0] * peer.VOCAB
        self.row[338] = 1.0

    def test_exact_and_existing_tolerance(self):
        self.assertEqual(peer.validate_logits(self.row, self.row), 0)
        candidate = self.row.copy()
        candidate[0] = 0.0009999999
        self.assertLessEqual(peer.validate_logits(self.row, candidate), 1e-3)
        candidate[0] = 1.00001e-3
        with self.assertRaisesRegex(ValueError, "DIVERGE"):
            peer.validate_logits(self.row, candidate)

    def test_original_float32_boundary_is_not_float64(self):
        self.row[338] = 11.0
        candidate = self.row.copy()
        candidate[0] = float(np.float32(0.011))
        self.assertLess(candidate[0] / 11.0, 1e-3)
        # Original float32 division rounds above 1e-3; float64 would pass.
        with self.assertRaisesRegex(ValueError, "DIVERGE"):
            peer.validate_logits(self.row, candidate)

    def test_wrong_size_argmax_and_nonfinite(self):
        with self.assertRaisesRegex(ValueError, "size"):
            peer.validate_logits(self.row, self.row[:-1])
        for bad in (float("nan"), float("inf"), -float("inf"), 2.0):
            candidate = self.row.copy()
            candidate[7] = bad
            with self.subTest(value=bad), self.assertRaises(ValueError):
                peer.validate_logits(self.row, candidate)
            with self.subTest(reference=bad), self.assertRaises(ValueError):
                peer.validate_logits(candidate, self.row)

    def test_reference_contract(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "row.bin"
            with self.assertRaises(FileNotFoundError):
                peer.load_reference(path, 512)
            path.write_bytes(b"bad")
            with self.assertRaisesRegex(ValueError, "byte size"):
                peer.load_reference(path, 512)
            path.write_bytes(struct.pack(f"<{peer.VOCAB}f", *self.row))
            self.assertEqual(peer.load_reference(path, 512)[338], 1)
            with self.assertRaisesRegex(ValueError, "S=512"):
                peer.load_reference(path, 128)

    def test_cli_failures_are_structured_and_nonzero(self):
        for argv, error in ((["wrong"], None), ([], ValueError("DIVERGE"))):
            out, err = io.StringIO(), io.StringIO()
            with patch.object(peer, "main", side_effect=error), contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                self.assertEqual(peer.cli(argv), 1)
            result = json.loads(out.getvalue().removeprefix(peer.RESULT_PREFIX))
            self.assertEqual(result["status"], "invalid")
            self.assertIn("failed:", err.getvalue())

    def test_non512_is_ineligible_not_success(self):
        with patch.object(peer, "main", return_value={"status": "ineligible"}) as main, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(peer.cli(["128"]), 2)
        main.assert_called_once_with(128)

    def test_post_block_validation_preserves_timer_and_call_counts(self):
        events = []
        checked = []
        def fn():
            events.append("forward")
            return events.count("forward")
        def clock():
            events.append("clock")
            return sum(event == "clock" for event in events)
        def validate(output):
            checked.append(output)
            events.append("validate")
        fake = types.SimpleNamespace(inference_mode=contextlib.nullcontext)
        with patch.object(peer, "torch", fake, create=True), patch.object(peer.time, "perf_counter", side_effect=clock):
            self.assertEqual(peer.best_ms(fn, validate=validate), 1000)
        self.assertEqual(events, ["forward"] * 2 + ["clock", "forward", "clock"] * 5 + ["validate"] * 2)
        self.assertEqual(checked, [2, 7])

    def test_compiled_prepared_warmup_checked_after_timing(self):
        checked = []
        fake = types.SimpleNamespace(inference_mode=contextlib.nullcontext)
        with patch.object(peer, "torch", fake, create=True), patch.object(peer.time, "perf_counter", side_effect=[0, 1, 2, 3, 4, 5]):
            peer.best_ms(lambda: 7, nwarm=0, nreps=3, warmup_output=3, validate=checked.append)
        self.assertEqual(checked, [3, 7])

    def test_compiled_preparation_only_is_optional(self):
        fake = types.SimpleNamespace(inference_mode=contextlib.nullcontext)
        prepare_error = unittest.mock.Mock(side_effect=RuntimeError("compiler unavailable"))
        warmup_error = lambda: unittest.mock.Mock(side_effect=RuntimeError("lazy compilation unavailable"))
        with patch.object(peer, "torch", fake, create=True):
            for prepare in (prepare_error, warmup_error):
                out = io.StringIO()
                with contextlib.redirect_stdout(out):
                    self.assertIsNone(peer.best_compiled_ms(prepare, lambda output: None))
                self.assertIn("torch.compile skipped", out.getvalue())
            timed_error = unittest.mock.Mock(side_effect=[1, RuntimeError("timed execution failed")])
            with self.assertRaisesRegex(RuntimeError, "timed execution failed"):
                peer.best_compiled_ms(lambda: timed_error, lambda output: None)
            numerical_error = unittest.mock.Mock(side_effect=ValueError("DIVERGE"))
            with patch.object(peer.time, "perf_counter", side_effect=[0, 1, 2, 3, 4, 5]), self.assertRaisesRegex(ValueError, "DIVERGE"):
                peer.best_compiled_ms(lambda: lambda: 1, numerical_error)

    def test_timer_and_later_repetition_failures_propagate(self):
        fake = types.SimpleNamespace(inference_mode=contextlib.nullcontext)
        with patch.object(peer, "torch", fake, create=True):
            for end in (0, -1, float("inf"), float("nan")):
                with self.subTest(end=end), patch.object(peer.time, "perf_counter", side_effect=[0, end]), self.assertRaises(ValueError):
                    peer.best_ms(lambda: None, nwarm=0, nreps=1)
            with patch.object(peer.time, "perf_counter", side_effect=[0, 1, 2, 3]), self.assertRaisesRegex(ValueError, "DIVERGE"):
                peer.best_ms(lambda: None, nwarm=1, nreps=2,
                             validate=unittest.mock.Mock(side_effect=[None, ValueError("DIVERGE")]))

    def test_real_cli_missing_reference_has_no_model_import(self):
        # A copied script derives a new, empty reference directory. Guard imports to
        # prove the failure runs without touching any installed ML packages.
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "tools" / SOURCE.name
            source.parent.mkdir()
            source.write_bytes(SOURCE.read_bytes())
            guard = Path(temp) / "guard.py"
            guard.write_text(
                "import builtins, runpy, sys\n"
                "original = builtins.__import__\n"
                "def guarded(name, *args, **kwargs):\n"
                "    if name.split('.')[0] in {'torch', 'transformers', 'numpy'}:\n"
                "        raise AssertionError('MODEL IMPORT FORBIDDEN')\n"
                "    return original(name, *args, **kwargs)\n"
                "builtins.__import__ = guarded\n"
                "sys.argv = [sys.argv[1]]\n"
                "runpy.run_path(sys.argv[0], run_name='__main__')\n")
            result = subprocess.run([sys.executable, "-B", str(guard), str(source)], capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 1)
        self.assertIn('"status": "invalid"', result.stdout)
        self.assertIn("FileNotFoundError", result.stderr)
        self.assertNotIn("MODEL IMPORT FORBIDDEN", result.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
