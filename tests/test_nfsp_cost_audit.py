"""Adversarial checks against the retained real full-model cost measurement."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from tools.audit_nfsp_cost import load, production_shape, recompute, sha, _seal


class CostAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parents[1]/'docs/verification/nfsp-r4-full-cost'
        cls.original = json.loads((root/'report.json').read_text())
        cls.pipeline_original = json.loads((root/'pipeline.json').read_text())

    def setUp(self):
        self.report = deepcopy(self.original)
        self.pipeline = deepcopy(self.pipeline_original)

    def reject(self):
        with self.assertRaises(ValueError):
            recompute(self.report, self.pipeline)

    def test_real_complete_measurement_recalculates_all_components(self):
        result = recompute(self.report, self.pipeline)
        self.assertTrue(production_shape(self.report))
        self.assertEqual(self.report['fresh_profile_records'], 65792)
        self.assertEqual(self.report['forecast']['per_seed']['pilot']['q_updates'], 62472)
        self.assertEqual(self.report['forecast']['per_seed']['pilot']['target_refreshes'], 625)
        self.assertAlmostEqual(result['per_seed_allocated_process_hours']['pilot'], 52.74488929064047)
        self.assertGreater(result['scientific_and_probe_allocated_process_hours'], 16)

    def test_undercounted_q_reuse_and_target_refresh_are_rejected(self):
        for field, value in [('q_updates', 15618), ('q_loss_targets', 32768), ('target_refreshes', 1), ('fresh_q', 32767)]:
            with self.subTest(field=field):
                self.pipeline = deepcopy(self.pipeline_original)
                self.pipeline['phases']['closed']['counters'][field] = value
                self.reject()

    def test_missing_actor_snapshot_or_averaging_work_is_rejected(self):
        for field in ('actor_per_record', 'snapshot', 'supervised_update'):
            with self.subTest(field=field):
                self.report = deepcopy(self.original)
                self.report['rates_seconds'][field] *= .5
                self.reject()
        self.report = deepcopy(self.original)
        del self.report['rates_seconds']['target_copy']
        self.reject()

    def test_collection_cannot_double_subtract_overlapping_timing_scopes(self):
        self.report['reference_pipeline_timing_observers'][0]['snapshot_copy_seconds'] = 100
        self.reject()

    def test_version_diverse_stress_and_all_measured_samples_are_required(self):
        self.report['actors']['distinct_mixed_groups'] = 2
        self.reject()
        self.report = deepcopy(self.original)
        self.report['actors']['samples_seconds']['distinct_mixed'].pop()
        self.reject()
        self.report = deepcopy(self.original)
        self.report['kernels']['q_seconds'].pop()
        self.reject()

    def test_offline_rows_do_not_become_fresh_learning_records(self):
        self.report['fresh_profile_records'] += 500000
        self.reject()
        self.report = deepcopy(self.original)
        self.report['kernels']['new_fresh_learner_records'] = 32768
        self.reject()

    def test_evaluation_color_pair_rounds_and_total_area_are_checked(self):
        self.report['evaluation']['matches'][0]['physical_rounds'] -= 1
        self.reject()
        self.report = deepcopy(self.original)
        self.report['evaluation']['matches'][0]['pair']['subject'][0]['black_score_units'] += 1
        self.reject()

    def test_forecast_cannot_omit_checkpoint_or_integrity_costs(self):
        for field in ('checkpoints', 'evaluation_policy_integrity'):
            with self.subTest(field=field):
                self.report = deepcopy(self.original)
                del self.report['forecast']['per_seed']['pilot']['components_seconds'][field]
                self.reject()

    def test_lower_ceiling_or_scientific_price_is_rejected(self):
        for field in ('H_P', 'H_D', 'H_N', 'scientific_and_probe_device_hours'):
            with self.subTest(field=field):
                self.report = deepcopy(self.original)
                self.report['forecast'][field] *= .5
                self.reject()

    def test_screening_cannot_remove_its_pilot_prefix(self):
        for container in (self.report['plan']['evaluation_game_budgets_per_seed'], self.report['evaluation_game_budgets_per_seed']):
            container['screening_cumulative'] = 2000
        self.reject()

    def test_cpu_and_reduced_or_changed_schedule_never_have_production_shape(self):
        for field, value in [('device', 'cpu'), ('capacity', 16), ('width', 8), ('reservoir_open', 0), ('handoff_end', 9000000), ('uniform_comparison', True)]:
            with self.subTest(field=field):
                report = deepcopy(self.original); report['plan']['runtime'][field] = value
                self.assertFalse(production_shape(report))
        self.report['plan']['diagnostic'] = True
        self.assertFalse(production_shape(self.report))

    def test_checkpoint_seal_checks_actual_files_not_just_manifest_reference(self):
        from graygo.storage import CheckpointStore
        with tempfile.TemporaryDirectory() as directory:
            store = CheckpointStore(Path(directory)/'store', reserve_bytes=0)
            bundle = store.save(0, lambda p: (p/'cost-payload.pt').write_bytes(b'real bytes'), kind='weights', metadata={'test': True}, estimated_peak_bytes=1024)
            inventory = {}; _seal(bundle, inventory)
            self.assertEqual(inventory[str(bundle/'cost-payload.pt')], sha(bundle/'cost-payload.pt'))
            (bundle/'cost-payload.pt').write_bytes(b'changed bytes')
            with self.assertRaises(ValueError): _seal(bundle, {})

    def test_duplicate_keys_and_nonfinite_json_cannot_hide_changed_numbers(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'data.json'
            for value in ('{"H_P":24,"H_P":52}', '{"H_P":NaN}', '{"H_P":Infinity}'):
                path.write_text(value)
                with self.subTest(value=value), self.assertRaises(ValueError): load(path)


if __name__ == '__main__':
    unittest.main()
