import ast
import copy
import inspect
import unittest
from graygo.task_training import NFSPTaskConfig
from tools import delayed_r1_admission as admission, task_pipeline_parity as original, nfsp_cpu_r1_comparison as runner
from tools.nfsp_delayed_cpu_prices import VERSION as PRICE_VERSION


class DelayedR1ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.identity = {'hostname': 'host', 'cpu': 'cpu', 'python': 'python', 'platform': 'platform',
                         'dependencies': {}, 'runtime': {'device': 'cpu', 'torch_threads': 1, 'workers': 1}}
        prices = {'version': PRICE_VERSION, 'status': 'qualified_prospective_cpu_delayed_prices',
                  'target_runtime_source_sha256': 'a' * 64, 'runtime_identity': self.identity,
                  'prices': {'C1': {'q_reuse': 1, 'device_hours': '0', 'cpu_process_ceiling_seconds': '16200', 'raw_budget_sha256': 'b' * 64}}}
        self.proof = {'version': admission.VERSION, 'status': 'qualified_cpu_delayed_R1_comparison_source',
                      'primary_A2_slots_fillable': False, 'execution_authorized': False,
                      'candidate_source_sha256': 'a' * 64, 'cpu_prices': prices}
        self.allocation = {'seeds': [1, 2], 'task': 'delayed', 'learner': 'neural', 'q_reuse': 1, 'target': 'one-step',
                           'cohort_role': 'one-step-reuse-comparison', 'primary_A2_slots_fillable': False,
                           'learner_source_sha256': 'a' * 64, 'additional_control_reservation_seconds': '0', 'new_accelerator_jobs': 0,
                           'per_case_process_ceiling_seconds': 16200, 'aggregate_neural_cpu_reservation_seconds': '32400',
                           'case_roots': {str(s): f'runs/original-r1/seed{s}' for s in (1, 2)},
                           'case_configs': {str(s): NFSPTaskConfig(seed=s, q_reuse=1, time_cap_seconds=16200,
                               budget_profile_sha256='b' * 64).plan() for s in (1, 2)}}

    def test_entire_pipeline_comparison_only_changes_the_declared_reuse_recipe(self):
        expected = inspect.getsource(original.compare_case).replace("config['q_reuse'] == 4", "config['q_reuse'] == 1 and config['learner'] == 'neural'")
        self.assertEqual(ast.dump(ast.parse(expected)), ast.dump(ast.parse(inspect.getsource(admission.compare_case))))

    def test_both_original_remaining_seeds_keep_full_nonprimary_c1_settings(self):
        for seed in (1, 2):
            config = runner.case_settings(self.allocation, seed, self.proof, self.identity)
            self.assertTrue(config.formal_protocol())
            self.assertEqual((config.records, config.seed, config.q_reuse), (15000000, seed, 1))
            result = admission.check_case(dict(self.identity, source_sha256='a' * 64, config=config.plan()), self.proof)
            self.assertFalse(result['primary_A2_slots_fillable'])
            self.assertEqual(result['cohort_role'], 'one-step-reuse-comparison')

    def test_other_original_seed_cannot_change_recipe_or_backend(self):
        for key, value in [('records', 5000000), ('q_reuse', 4), ('hidden', 32), ('batch', 128), ('capacity', 1000),
                           ('target_every', 3200), ('reservoir_open', 0), ('handoff_end', 8000000),
                           ('device', 'cuda'), ('storage', 'rule-state'), ('time_cap_seconds', 19800),
                           ('budget_profile_sha256', 'c' * 64), ('seed', 0)]:
            with self.subTest(key=key):
                allocation = copy.deepcopy(self.allocation); allocation['case_configs']['2'][key] = value
                with self.assertRaises(ValueError):
                    runner.case_settings(allocation, 1, self.proof, self.identity)

    def test_comparison_never_becomes_a_primary_case_or_replaces_seed_zero(self):
        for key, value in [('primary_A2_slots_fillable', True), ('cohort_role', 'primary'), ('seeds', [0, 1, 2]),
                           ('additional_control_reservation_seconds', '1'), ('new_accelerator_jobs', 1),
                           ('aggregate_neural_cpu_reservation_seconds', '16200')]:
            allocation = copy.deepcopy(self.allocation); allocation[key] = value
            with self.assertRaises(ValueError):
                runner.case_settings(allocation, 1, self.proof, self.identity)
        for seed in (0, True, 3):
            with self.assertRaises(ValueError):
                runner.case_settings(self.allocation, seed, self.proof, self.identity)
        proof = copy.deepcopy(self.proof); proof['primary_A2_slots_fillable'] = True
        with self.assertRaises(ValueError):
            runner.case_settings(self.allocation, 1, proof, self.identity)

    def test_source_host_price_and_unique_original_roots_remain_bound(self):
        for root in ('runs/original-r1/seed1', 'runs/original-r1/./seed1', 'runs/original-r1/seed1/child', '/tmp/case', 'runs/../case'):
            allocation = copy.deepcopy(self.allocation); allocation['case_roots']['2'] = root
            with self.assertRaises(ValueError):
                runner.case_settings(allocation, 1, self.proof, self.identity)
        for key, value in [('learner_source_sha256', 'c' * 64), ('per_case_process_ceiling_seconds', 18000)]:
            allocation = copy.deepcopy(self.allocation); allocation[key] = value
            with self.assertRaises(ValueError):
                runner.case_settings(allocation, 1, self.proof, self.identity)
        with self.assertRaises(ValueError):
            runner.case_settings(self.allocation, 1, self.proof, self.identity | {'hostname': 'different'})


if __name__ == '__main__':
    unittest.main()
