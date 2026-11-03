from contextlib import redirect_stdout, redirect_stderr
from io import StringIO
from pathlib import Path
import tempfile
import unittest
from tools.task_source_diff import compare, inventory, main


class SourceDiffTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.old, self.new = self.root / 'old', self.root / 'new'
        for root in (self.old, self.new):
            root.mkdir()
            (root / '__init__.py').write_text('')
            (root / 'nfsp_training.py').write_text('from .task_training import run\n')
            (root / 'task_training.py').write_text('from .shared import value\ndef run():\n return value\n')
            (root / 'shared.py').write_text('value=1\n')

    def diff(self):
        return compare(self.old, self.new, expected_reference=inventory(self.old)[2], expected_candidate=inventory(self.new)[2])

    def test_exact_dependency_equality_is_inventory_not_authority(self):
        r = self.diff()
        self.assertEqual(r['classification'], 'dependency_bytes_unchanged')
        self.assertEqual(len(r['unchanged_dependency_modules']), 4)
        self.assertFalse(r['behavioral_compatibility_verified'])
        self.assertFalse(r['a2_evidence_reusable'])
        self.assertFalse(r['execution_authorized'])

    def test_cycle_and_import_inside_function_are_followed(self):
        (self.new / 'shared.py').write_text('value=1\ndef f():\n from . import extra\n')
        (self.new / 'extra.py').write_text('from .shared import value\n')
        r = self.diff()
        self.assertIn('extra', r['candidate_dependencies']['modules'])
        self.assertIsNone(r['changed_dependency_modules']['extra']['reference_sha256'])
        self.assertIn('f', r['changed_dependency_modules']['shared']['changed_definitions'])

    def test_same_ast_with_different_bytes_stays_explicit(self):
        (self.new / 'shared.py').write_text('# comment\nvalue = 1\n')
        r = self.diff()
        self.assertTrue(r['changed_dependency_modules']['shared']['ast_identical'])
        self.assertFalse(r['a2_evidence_reusable'])

    def test_unused_change_is_separate_but_in_package_fingerprint(self):
        (self.new / 'unused.py').write_text('raise RuntimeError("must never execute")\n')
        r = self.diff()
        self.assertNotEqual(r['reference_package_sha256'], r['candidate_package_sha256'])
        self.assertEqual(r['changed_files_outside_dependency_closure'], ['unused.py'])
        self.assertNotIn('unused', r['changed_dependency_modules'])

    def test_dynamic_imports_stay_visible_and_local_literal_is_followed(self):
        (self.new / 'shared.py').write_text('value=1\n__import__("graygo.extra")\n__import__(name)\n')
        (self.new / 'extra.py').write_text('value=2\n')
        r = self.diff()
        self.assertEqual(len(r['candidate_dependencies']['dynamic_sites']), 2)
        self.assertIn('extra', r['candidate_dependencies']['modules'])

    def test_changed_member_and_external_dependency_are_recorded(self):
        (self.new / 'shared.py').write_text('import numpy\nvalue=1\nclass Actor:\n def forward(self): return 2\n')
        r = self.diff()
        self.assertIn('Actor.forward', r['changed_dependency_modules']['shared']['changed_definitions'])
        self.assertIn('numpy', r['candidate_dependencies']['external_imports'])

    def test_pin_missing_import_and_symlink_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'fingerprint'):
            compare(self.old, self.new, expected_reference='0'*64, expected_candidate=inventory(self.new)[2])
        (self.new / 'shared.py').write_text('from .missing import x\n')
        with self.assertRaisesRegex(ValueError, 'missing local'):
            self.diff()
        (self.new / 'shared.py').unlink()
        (self.new / 'shared.py').symlink_to(self.old / 'shared.py')
        with self.assertRaisesRegex(ValueError, 'symlinks'):
            self.diff()

    def test_cli_preserves_inputs_existing_output_and_rejects_in_package_output(self):
        old, new = inventory(self.old), inventory(self.new)
        output = self.root / 'report.json'
        args = ['--reference', str(self.old), '--candidate', str(self.new), '--expected-reference', old[2],
                '--expected-candidate', new[2], '--output', str(output)]
        with redirect_stdout(StringIO()):
            self.assertEqual(main(args), 0)
        saved = output.read_bytes()
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
            main(args)
        self.assertEqual(output.read_bytes(), saved)
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
            main(args[:-1] + [str(self.new / 'report.json')])
        self.assertEqual(inventory(self.old), old)
        self.assertEqual(inventory(self.new), new)
