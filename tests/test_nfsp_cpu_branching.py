import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from graygo.task_training import NFSPTaskConfig

spec = importlib.util.spec_from_file_location('branching_runner', Path(__file__).parents[1] / 'tools/nfsp_cpu_branching.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class BranchingAdmissionTests(unittest.TestCase):
    def setUp(self):
        self.identity = {'host': 'fixed', 'runtime': {'device': 'cpu', 'torch_threads': 1}}
        self.price = {'task': 'branching', 'learner': 'neural', 'q_reuse': 4, 'device': 'cpu',
                      'device_hours': '0', 'runtime_margin': '2', 'formal_record_cap': 15000000,
                      'original_seed_scope': [0, 1, 2], 'cpu_process_ceiling_seconds': '15300',
                      'source_sha256': 'a' * 64, 'raw_budget_sha256': 'b' * 64}
        self.proof = {'version': runner.VERSION, 'status': 'qualified_cpu_branching_neural_price',
                      'execution_authorized': False, 'runtime_identity': self.identity, 'price': self.price}
        self.allocation = {'seeds': [0, 1, 2], 'task': 'branching', 'learner': 'neural',
                           'q_reuse': 4, 'target': 'one-step', 'additional_control_reservation_seconds': '0',
                           'new_accelerator_jobs': 0, 'per_case_process_ceiling_seconds': 15300,
                           'aggregate_neural_cpu_reservation_seconds': '45900',
                           'learner_source_sha256': 'a' * 64,
                           'case_roots': {str(s): f'runs/original-branching/seed{s}' for s in range(3)},
                           'case_configs': {str(s): NFSPTaskConfig(task='branching', seed=s, q_reuse=4,
                               time_cap_seconds=15300, budget_profile_sha256='b' * 64).plan() for s in range(3)}}

    def test_accepts_original_full_cohort_without_execution_grant(self):
        for seed in range(3):
            config = runner.case_settings(self.allocation, seed, self.proof, self.identity)
            self.assertTrue(config.formal_protocol())
            self.assertEqual((config.task, config.seed, config.records), ('branching', seed, 15000000))
        self.assertFalse(self.proof['execution_authorized'])

    def test_rejects_changed_other_seed_recipe_before_any_seed_starts(self):
        for key, value in [('seed', 3), ('task', 'delayed'), ('records', 100000), ('batch', 128),
                           ('capacity', 10000), ('hidden', 16), ('q_reuse', 1), ('target_every', 3200),
                           ('device', 'cuda'), ('storage', 'rule-state'), ('time_cap_seconds', 18000),
                           ('reservoir_open', 1000000), ('budget_profile_sha256', 'c' * 64)]:
            with self.subTest(key=key):
                allocation = copy.deepcopy(self.allocation)
                allocation['case_configs']['2'][key] = value
                with self.assertRaises(ValueError):
                    runner.case_settings(allocation, 0, self.proof, self.identity)

    def test_rejects_seed_filtering_aliases_traversal_and_nested_roots(self):
        for roots in [dict(self.allocation['case_roots'], **{'2': 'runs/original-branching/seed0'}),
                      dict(self.allocation['case_roots'], **{'2': 'runs/original-branching/./seed0'}),
                      dict(self.allocation['case_roots'], **{'2': 'runs/../outside'}),
                      dict(self.allocation['case_roots'], **{'2': '/tmp/case'}),
                      dict(self.allocation['case_roots'], **{'2': 'runs/original-branching/seed0/subcase'})]:
            allocation = copy.deepcopy(self.allocation); allocation['case_roots'] = roots
            with self.assertRaises(ValueError):
                runner.case_settings(allocation, 0, self.proof, self.identity)
        allocation = copy.deepcopy(self.allocation); allocation['seeds'] = [0, 1]
        with self.assertRaises(ValueError):
            runner.case_settings(allocation, 0, self.proof, self.identity)
        with self.assertRaises(ValueError):
            runner.case_settings(self.allocation, True, self.proof, self.identity)

    def test_rejects_resource_source_and_runtime_changes(self):
        for key, value in [('additional_control_reservation_seconds', '1'), ('new_accelerator_jobs', 1),
                           ('per_case_process_ceiling_seconds', 18000),
                           ('aggregate_neural_cpu_reservation_seconds', '15300'),
                           ('learner_source_sha256', 'c' * 64), ('target', 'retrace')]:
            allocation = copy.deepcopy(self.allocation); allocation[key] = value
            with self.assertRaises(ValueError):
                runner.case_settings(allocation, 0, self.proof, self.identity)
        with self.assertRaises(ValueError):
            runner.case_settings(self.allocation, 0, self.proof, {'host': 'different'})

    def test_fresh_price_uses_neural_corpus_without_approving_controls(self):
        proposal = {'selected': False, 'owner_approved': False, 'audited_cpu_prices': [self.price],
                    'runtime_identity': self.identity, 'evidence_files_sha256': {}}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'proposal.json'; path.write_text(json.dumps(proposal))
            evidence = Path(tmp) / 'evidence.json'; evidence.write_text('{}')
            with patch.object(runner, 'build_proposal', return_value=proposal):
                proof = runner.qualify(tmp, path, evidence)
            self.assertFalse(proof['control_scope_approved'])
            self.assertFalse(proof['execution_authorized'])
            self.assertEqual(proof['price']['cpu_process_ceiling_seconds'], '15300')
            changed = dict(proposal, selected=True)
            with patch.object(runner, 'build_proposal', return_value=changed):
                with self.assertRaises(ValueError):
                    runner.qualify(tmp, path, evidence)


if __name__ == '__main__':
    unittest.main()
