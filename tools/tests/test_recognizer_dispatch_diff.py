"""Synthetic/mocked dispatch-report tests; never invoke a compiler."""
import collections
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

REPO = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "dispatch_diff", REPO / "docs/sota-campaign-2026-09/recognizer-dispatch-diff.py")
diff = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(diff)


def source(*fields):
    return '\n'.join(['let gemm = GemmSyms {',
                      'recognize: opts.recognize,', 'vec_kernels: opts.vec_kernels,',
                      'alias_free: false,', *fields, '};'])


def row(base, candidate, base_error=None, candidate_error=None):
    return dict(base_calls=base, candidate_calls=candidate,
                base_error=base_error, cand_error=candidate_error)


class DispatchCensusTests(unittest.TestCase):
    def test_only_true_calls_count_and_multiplicity_survives(self):
        text = """fn wukong_a() -> void {
  bb0:
    call wukong_a(v0)
    v2 = call wukong_a(v1)
    global_addr wukong_b
    // call wukong_b(v0)
    const.str "call wukong_b()"
    call user_wukong_b(v0)
}
"""
        with mock.patch.object(diff.subprocess, "run", return_value=subprocess.CompletedProcess([], 0, text, "")):
            counts, error = diff.calls(Path("compiler"), Path("fixture"), REPO)
        self.assertEqual(counts, {"wukong_a": 2})
        self.assertIsNone(error)

    def test_equal_arms_still_report_absolute_missing_coverage(self):
        report = diff.summarize([("same", row({"wukong_a": 2}, {"wukong_a": 2})),
                                 ("empty", row({}, {}))], {"wukong_a", "wukong_b"})
        self.assertEqual(report["unchanged"], 2)
        self.assertEqual(report["fixture_results"]["same"]["candidate_calls"], {"wukong_a": 2})
        coverage = report["candidate_coverage"]
        self.assertEqual(coverage["missing_inventory"], ["wukong_b"])
        self.assertEqual(coverage["successful_zero_kernel_fixtures"], ["empty"])
        self.assertTrue(coverage["complete_fixture_scan"])

    def test_counter_differences_preserve_duplicate_changes(self):
        report = diff.summarize([("changed", row({"wukong_a": 3, "wukong_b": 1},
                                                {"wukong_a": 1, "wukong_c": 2}))], {"wukong_a"})
        self.assertEqual(report["changed"]["changed"], {
            "removed": {"wukong_a": 2, "wukong_b": 1}, "added": {"wukong_c": 2}})
        self.assertEqual(report["candidate_coverage"]["unexpected_symbols"], ["wukong_c"])

    def test_failed_candidate_calls_are_retained_but_not_coverage(self):
        report = diff.summarize([
            ("bad_candidate", row({"wukong_a": 1}, {"wukong_b": 2}, candidate_error="EXIT1")),
            ("bad_base", row({}, {"wukong_a": 1}, base_error="TIMEOUT")),
        ], {"wukong_a", "wukong_b"})
        coverage = report["candidate_coverage"]
        self.assertEqual(coverage["covered_inventory"], ["wukong_a"])
        self.assertEqual(coverage["partial_calls_from_errors"], {"wukong_b": 2})
        self.assertEqual(coverage["errored_fixtures"], ["bad_candidate"])
        self.assertFalse(coverage["complete_fixture_scan"])
        self.assertFalse(report["relative_comparison_complete"])
        self.assertEqual(report["fixture_results"]["bad_candidate"]["base_calls"], {"wukong_a": 1})
        base_only = diff.summarize([("f", row({}, {}, base_error="EXIT1"))], {"wukong_a"})
        self.assertTrue(base_only["candidate_coverage"]["complete_fixture_scan"])
        self.assertFalse(base_only["relative_comparison_complete"])

    def test_inventory_exclusions_are_explicit_and_duplicates_are_sets(self):
        inventory = diff.inventory(source(
            'a: interner.intern("wukong_a"),', 'b: interner.intern("wukong_a"),',
            'service: interner.intern("wukong_rt_trig_f32"),',
            'new_service: interner.intern("wukong_rt_future"),',
            'p: interner.intern("print_str"),'))
        self.assertEqual(inventory, {"wukong_a", "wukong_rt_future"})
        report = diff.summarize([("service", row({}, {"wukong_rt_trig_f32": 1}))], inventory)
        self.assertEqual(report["candidate_coverage"]["unexpected_symbols"], [])
        self.assertEqual(report["candidate_coverage"]["successful_zero_kernel_fixtures"], ["service"])

    def test_malformed_or_partial_inventory_fails_closed(self):
        good = source('a: interner.intern("wukong_a"),')
        for text in ["", source(), good + '\n' + good, good.replace('};', ''),
                     good.replace('"wukong_a"', 'name'),
                     good.replace('alias_free: false,', ''),
                     source('a: interner.intern("wukong_a"),', 'a: interner.intern("wukong_b"),'),
                     source('a: interner.intern("wukong_a"),', 'b: interner.intern("wukong_b),'),
                     source('a: interner.intern("unknown_service"),')]:
            with self.subTest(text=text), self.assertRaises(ValueError):
                diff.inventory(text)

    def test_launch_exit_and_timeout_errors_preserve_partial_calls(self):
        cases = [
            (subprocess.CompletedProcess([], 9, 'call wukong_a()\n', ''), 'EXIT9', {"wukong_a": 1}),
            (subprocess.TimeoutExpired('cc', 120, output=b'call wukong_a()\n'), 'TIMEOUT', {"wukong_a": 1}),
            (FileNotFoundError('missing compiler'), 'LAUNCH:', {}),
        ]
        for outcome, prefix, counts in cases:
            kwargs = {"side_effect": outcome} if isinstance(outcome, Exception) else {"return_value": outcome}
            with self.subTest(prefix=prefix), mock.patch.object(diff.subprocess, "run", **kwargs):
                actual, error = diff.calls(Path('cc'), Path('fixture'), REPO)
                self.assertEqual(actual, counts)
                self.assertTrue(error.startswith(prefix))

    def test_roots_and_template_substitutions_are_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            names = ['tests/run/a.wk', 'examples/b.wk', 'bench/kernels/c.wk',
                     'crates/wukong_xbench/wk/nested/d.wk', 'tests/run/lib/skip.wk']
            for name in names:
                path = repo / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('$S$ $SD$ $RC$ $QT$ $SCALE$')
            self.assertEqual(diff.fixtures(repo), sorted(repo / name for name in names[:-1]))
            self.assertEqual(diff.render(repo / names[3], repo), '256 65536 8388608 0.875 0.125')
            self.assertEqual(diff.render(repo / names[0], repo), '$S$ $SD$ $RC$ $QT$ $SCALE$')

    def test_template_is_rendered_once_identically_for_both_compilers(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            path = repo / 'crates/wukong_xbench/wk/a.wk'
            path.parent.mkdir(parents=True)
            path.write_text('$SD$')
            seen = []
            def fake_calls(exe, original, repo, rendered):
                seen.append((rendered, rendered.read_bytes()))
                return collections.Counter(), None
            with mock.patch.object(diff, 'calls', side_effect=fake_calls):
                diff.one(path, repo, Path('base'), Path('candidate'))
            self.assertEqual(seen, [(seen[0][0], b'65536')] * 2)

    def test_cli_missing_coverage_is_informational_but_errors_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            src = repo / 'inventory.rs'
            src.write_text(source('a: interner.intern("wukong_a"),'))
            paths = [repo / 'fixture.wk']
            for failed, code in [(False, 0), (True, 1)]:
                result = ('fixture.wk', row({}, {}, candidate_error='EXIT1' if failed else None))
                with mock.patch.object(diff, 'fixtures', return_value=paths), \
                     mock.patch.object(diff, 'one', return_value=result), \
                     contextlib.redirect_stdout(io.StringIO()) as stdout:
                    exit_code = diff.main(['--repo', str(repo), '--inventory-source', str(src),
                                           '--base', str(repo / 'base'), '--candidate', str(repo / 'cand')])
                self.assertEqual(exit_code, code)
                output = repo / 'target/sota-campaign-logs/recognizer-dispatch-diff.json'
                report = json.loads(output.read_text())
                self.assertEqual(report['candidate_coverage']['missing_inventory'], ['wukong_a'])
                self.assertEqual(report['candidate_coverage']['complete_fixture_scan'], not failed)
                self.assertIn('missing inventory:', stdout.getvalue())
                self.assertEqual(report['base'], str(repo / 'base'))

    def test_cli_invalid_inventory_and_empty_fixture_set_fail_before_compile(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            src = repo / 'inventory.rs'
            for content in [None, '', source('a: interner.intern("wukong_a"),')]:
                if content is not None:
                    src.write_text(content)
                with mock.patch.object(diff, 'calls') as calls, contextlib.redirect_stderr(io.StringIO()):
                    self.assertEqual(diff.main(['--repo', str(repo), '--inventory-source', str(src)]), 1)
                    calls.assert_not_called()


if __name__ == '__main__':
    unittest.main()
