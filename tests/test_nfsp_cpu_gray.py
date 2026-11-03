import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from graygo.task_training import NFSPTaskConfig

spec = importlib.util.spec_from_file_location('gray_cpu_runner', Path(__file__).parents[1] / 'tools/nfsp_cpu_gray.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class GrayCPUCaseTests(unittest.TestCase):
    def setUp(self):
        self.identity = {'hostname': 'fixed', 'runtime': {'device': 'cpu', 'workers': 1, 'torch_threads': 1}}
        caps = {'gray-pure': 110700, 'gray-mixed': 132300}
        self.prices = {task: {'task': task, 'learner': 'neural', 'q_reuse': 4, 'device_hours': '0',
                             'cpu_process_ceiling_seconds': str(cap), 'source_sha256': 'a' * 64,
                             'raw_budget_sha256': ('b' if task == 'gray-pure' else 'c') * 64}
                       for task, cap in caps.items()}
        self.proof = {'version': runner.VERSION, 'status': 'qualified_cpu_gray_neural_prices',
                      'execution_authorized': False, 'learner_source_sha256': 'a' * 64,
                      'runtime_identity': self.identity, 'prices': self.prices}
        self.allocation = {'tasks': list(runner.TASKS), 'seeds': [0, 1, 2], 'learner': 'neural',
                           'q_reuse': 4, 'target': 'one-step', 'learner_source_sha256': 'a' * 64,
                           'additional_control_reservation_seconds': '0', 'new_accelerator_jobs': 0,
                           'per_case_process_ceiling_seconds': caps,
                           'aggregate_neural_cpu_reservation_seconds': '729000',
                           'case_roots': {t + '/' + str(s): f'runs/original-gray/{t}/seed{s}'
                                          for t in runner.TASKS for s in range(3)},
                           'case_configs': {t + '/' + str(s): NFSPTaskConfig(task=t, seed=s, q_reuse=4,
                               time_cap_seconds=caps[t], budget_profile_sha256=self.prices[t]['raw_budget_sha256']).plan()
                                            for t in runner.TASKS for s in range(3)}}

    def test_all_six_cases_use_full_original_board_model_and_individual_prices(self):
        for task in runner.TASKS:
            for seed in range(3):
                config = runner.case_settings(self.allocation, task, seed, self.proof, self.identity)
                self.assertTrue(config.formal_protocol())
                self.assertEqual((config.task, config.seed, config.width, config.blocks, config.batch, config.capacity),
                                 (task, seed, 64, 8, 512, 500000))
                self.assertEqual(config.budget_profile_sha256, self.prices[task]['raw_budget_sha256'])

    def test_a_changed_case_in_the_other_family_prevents_any_launch(self):
        for key, value in [('seed', 7), ('task', 'branching'), ('learner', 'tabular'), ('records', 100000),
                           ('width', 16), ('blocks', 2), ('norm_groups', 4), ('batch', 128), ('capacity', 1000),
                           ('q_reuse', 1), ('target_every', 3200), ('device', 'cuda'), ('storage', 'rule-state'),
                           ('reservoir_open', 0), ('handoff_end', 8000000), ('epsilon_decay', 1000000),
                           ('budget_profile_sha256', 'd' * 64), ('time_cap_seconds', 150000)]:
            with self.subTest(key=key):
                allocation = copy.deepcopy(self.allocation); allocation['case_configs']['gray-mixed/2'][key] = value
                with self.assertRaises(ValueError):
                    runner.case_settings(allocation, 'gray-pure', 0, self.proof, self.identity)

    def test_missing_seed_or_family_cannot_redefine_the_selected_cohort(self):
        for key, value in [('tasks', ['gray-pure']), ('seeds', [0, 1]), ('learner', 'tabular'), ('target', 'retrace')]:
            allocation = copy.deepcopy(self.allocation); allocation[key] = value
            with self.assertRaises(ValueError):
                runner.case_settings(allocation, 'gray-pure', 0, self.proof, self.identity)
        allocation = copy.deepcopy(self.allocation); del allocation['case_configs']['gray-mixed/2']
        with self.assertRaises(ValueError):
            runner.case_settings(allocation, 'gray-pure', 0, self.proof, self.identity)
        with self.assertRaises(ValueError):
            runner.case_settings(self.allocation, 'gray-pure', True, self.proof, self.identity)

    def test_case_roots_cannot_alias_escape_or_contain_each_other(self):
        for root in ['runs/original-gray/gray-pure/seed0', 'runs/original-gray/gray-pure/./seed0',
                     'runs/original-gray/gray-pure/seed0/subcase', 'runs/../outside', '/tmp/outside']:
            allocation = copy.deepcopy(self.allocation); allocation['case_roots']['gray-mixed/2'] = root
            with self.assertRaises(ValueError):
                runner.case_settings(allocation, 'gray-pure', 0, self.proof, self.identity)

    def test_source_backend_runtime_and_all_six_cpu_caps_must_remain_funded(self):
        for key, value in [('learner_source_sha256', 'd' * 64), ('new_accelerator_jobs', 1),
                           ('additional_control_reservation_seconds', '1'),
                           ('aggregate_neural_cpu_reservation_seconds', '243000'),
                           ('per_case_process_ceiling_seconds', {'gray-pure': 132300, 'gray-mixed': 132300})]:
            allocation = copy.deepcopy(self.allocation); allocation[key] = value
            with self.assertRaises(ValueError):
                runner.case_settings(allocation, 'gray-pure', 0, self.proof, self.identity)
        with self.assertRaises(ValueError):
            runner.case_settings(self.allocation, 'gray-pure', 0, self.proof, {'hostname': 'different'})

    def test_prices_cannot_be_transferred_between_families_sources_or_devices(self):
        for key, value in [('task', 'gray-pure'), ('source_sha256', 'd' * 64),
                           ('device_hours', '1'), ('learner', 'tabular'), ('q_reuse', 1)]:
            proof = copy.deepcopy(self.proof); proof['prices']['gray-mixed'][key] = value
            with self.assertRaises(ValueError):
                runner.case_settings(self.allocation, 'gray-pure', 0, proof, self.identity)

    def test_reading_receipt_rechecks_source_and_runtime_evidence(self):
        proof = copy.deepcopy(self.proof)
        proof.update(verifier_sha256=runner.digest(runner.__file__), qualification_seconds=1.,
                     evidence={'repo': '/repo', 'specification': '/spec'})
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'receipt.json'; path.write_text(json.dumps(proof))
            fresh = copy.deepcopy(proof); fresh['qualification_seconds'] = 2.
            with patch.object(runner, 'qualify', return_value=fresh) as check:
                self.assertEqual(runner.read_qualified(path), proof)
                check.assert_called_once_with('/repo', '/spec')
            fresh['runtime_identity']['hostname'] = 'changed'
            with patch.object(runner, 'qualify', return_value=fresh):
                with self.assertRaises(ValueError):
                    runner.read_qualified(path)


if __name__ == '__main__':
    unittest.main()
