import ast
from copy import deepcopy
from dataclasses import replace
import inspect
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from graygo import board_profile
from graygo.board_runtime import BoardRuntimeConfig
from tests import test_nfsp_cost_profile as fixtures
from tools import nfsp_uniform_pipeline as pipeline, nfsp_uniform_costs as producer, audit_nfsp_uniform_costs as auditor


class UniformReferenceCostsTests(unittest.TestCase):
    def test_real_closed_dense_collection_uses_a_local_opening_view_without_mutating_offline_labels(self):
        fixture = fixtures.CostProfileTests(methodName='runTest'); torch.set_num_threads(1)
        replay, reservoir, _ = producer.core.materialize(fixture.source())
        config = replace(fixture.config(), uniform_comparison=True)
        backend, _ = producer.core.kernel_measurements(config, replay, reservoir, iterations=1, warmup=0)
        states = [replay.spec.codec.unpack(replay.arrays['spatial'][i])[0] for i in (0, 2)]
        before = {key: value.copy() for key, value in reservoir.arrays.items()}; seen = reservoir.seen
        with patch.object(producer, 'FrozenForwardPool', return_value=None):
            result = producer.dense_uniform(config, backend, replay, reservoir, states, records=8)
        self.assertEqual(result['new_profile_records'], 8)
        self.assertGreater(result['acting_counts']['uniform'], 0)
        self.assertEqual(result['acting_counts']['average'], 0)
        self.assertEqual(reservoir.opening_record, 0)
        self.assertEqual(reservoir.seen, seen)
        for key, value in before.items():np.testing.assert_array_equal(value, reservoir.arrays[key])
        view = producer.ClosedReservoirSink(reservoir, 16, config.reservoir_open)
        self.assertEqual(view.opening_record, 5000000)
        with self.assertRaises(ValueError):view.offer_index(None)

    def test_real_pipeline_body_only_changes_the_declared_uniform_recipe_guard(self):
        expected = inspect.getsource(board_profile.profile_board)
        expected = expected.replace('config.handoff_end != 7_000_000 or config.uniform_comparison',
                                    'config.handoff_end != 7_000_000 or not config.uniform_comparison')
        expected = expected.replace('base runtime must retain the primary production schedule',
                                    'uniform reference must retain its declared opening and hand-off schedule')
        self.assertEqual(ast.dump(ast.parse(expected)), ast.dump(ast.parse(inspect.getsource(pipeline.profile_board))))

    def test_reference_config_changes_only_uniform_fallback_and_preserves_common_components(self):
        base = BoardRuntimeConfig(q_reuse=4, device='cuda', cuda_graphs=True)
        changed = producer.compatible_config(base.plan())
        self.assertEqual(changed, replace(base, uniform_comparison=True))
        self.assertEqual(changed.backend_config(), base.backend_config())
        self.assertEqual(changed.model_config(), base.model_config())
        self.assertEqual(changed.schedule(), base.schedule())
        for config in (replace(base, q_reuse=1), replace(base, cuda_graphs=False), replace(base, uniform_comparison=True)):
            with self.assertRaises(ValueError):producer.compatible_config(config.plan())

    def test_primary_measurement_cannot_be_rebranded_as_uniform_reference(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'uniform'
            with self.assertRaises(ValueError):
                pipeline.profile_board(BoardRuntimeConfig(q_reuse=4), root, records=4096, time_cap_seconds=10)
            self.assertFalse(root.exists())

    def fixtures(self):
        parts = {'collection': 8., 'private_actors': 8., 'frozen_snapshots': 15.626,
                 'frozen_graph_captures': 16.4, 'frozen_parameter_packs': 313.2, 'setup': 2.,
                 'Q': 100., 'supervised_average': 20., 'target_refresh': 1., 'checkpoints': 10.,
                 'one_restore': 1., 'public_archives': 1., 'evaluation_groups': 10.,
                 'evaluation_endpoint_metadata': 1., 'evaluation_policy_integrity': 1.,
                 'evaluation_model_loads': 1., 'cold_process_startups': 1.}
        base = {'core_report': {'rates_seconds': {'collection_per_record': .000001}},
                'actors': {'steady_actor_seconds_per_record': .000001, 'snapshot_copy_seconds': [.001]},
                'graph_setup': {'capture_max_seconds': .1, 'weight_pack_max_seconds': .01},
                'forecast': {'per_seed': {'pilot': {'records': 8000000, 'evaluation_games': 36000, 'evaluation_endpoints': 18,
                    'snapshot_publication_upper': 15626, 'graph_capture_upper': 164, 'weight_pack_upper': 31320,
                    'components_seconds': parts}}}}
        phases = {'closed': {'records': 32768, 'timings': {'collection': .5}, 'setup_seconds': .8},
                  'early_open_diagnostic': {'records': 32768, 'timings': {'collection': .6}, 'setup_seconds': .9}}
        observers = [{'phase': 'closed', 'records': 32768, 'actor_seconds': .1, 'snapshot_copy_seconds': .1,
                      'resolver_storage_bookkeeping_seconds': .3},
                     {'phase': 'early_open_diagnostic', 'records': 32768, 'actor_seconds': .2, 'snapshot_copy_seconds': .1,
                      'resolver_storage_bookkeeping_seconds': .3}]
        cases = {}; index = 0
        for name in ('shared_uniform', 'distinct_q_uniform'):
            cases[name] = []
            for _ in range(4):
                cases[name].append({'raw_seconds': .003, 'graph_setup_seconds': .001, 'steady_seconds': .002,
                                    'graph_event_span': [index, index + 1]}); index += 1
        setup = {'events': [{'kind': 'capture', 'seconds': .001} for _ in range(8)] + [{'kind': 'weight_pack', 'seconds': .0002}],
                 'capture_max_seconds': .001, 'weight_pack_max_seconds': .0002, 'pools': {'0': {'eager_fallbacks': 0}}}
        measured = {'pipeline': {'phases': phases}, 'pipeline_observers': observers,
                    'dense': {'new_profile_records': 256, 'total_seconds': .5, 'actor_seconds': .1, 'resolver_storage_bookkeeping_seconds': .4},
                    'actors': {'cases': cases, 'requests_per_sample': 32, 'snapshot_copy_seconds': [.002]}}
        return base, measured, setup

    def test_actual_retained_common_report_schema_is_accepted_without_gpu_execution(self):
        path = Path(__file__).parents[1] / 'docs/verification/nfsp-graph-costs/full-cost/report.json'
        actual = json.loads(path.read_text())
        _, measured, setup = self.fixtures()
        produced = producer.reference_forecast(actual, measured, setup)
        independent = auditor.calculate(actual, measured, setup)
        self.assertEqual(produced['components_seconds_per_seed'], independent['components_seconds_per_seed'])
        self.assertGreater(produced['B_U_board_device_hours'], 0)

    def test_independent_reference_equation_keeps_common_costs_and_larger_measured_rates(self):
        base, measured, setup = self.fixtures()
        result = producer.reference_forecast(base, measured, setup)
        independent = auditor.calculate(base, measured, setup)
        self.assertEqual(result['components_seconds_per_seed'], independent['components_seconds_per_seed'])
        self.assertAlmostEqual(result['components_seconds_per_seed']['collection'], 12500)
        self.assertAlmostEqual(result['components_seconds_per_seed']['private_actors'], 500)
        self.assertEqual(result['components_seconds_per_seed']['Q'], 100)
        self.assertEqual(result['components_seconds_per_seed']['evaluation_groups'], 10)
        self.assertAlmostEqual(result['per_seed_ceiling_seconds'], 2 * sum(result['components_seconds_per_seed'].values()))
        self.assertAlmostEqual(result['B_U_board_device_hours'], 3 * result['H_U_device_hours_per_seed'])
        self.assertFalse(result['A2_uniform_subset_CPU_price_qualified'])
        self.assertFalse(result['full_B_U_scope_qualified']); self.assertFalse(result['execution_authorized'])

    def test_cold_setup_cannot_be_subtracted_twice_or_omitted(self):
        base, measured, setup = self.fixtures()
        duplicate = deepcopy(measured)
        duplicate['actors']['cases']['distinct_q_uniform'][0]['graph_event_span'] = [0, 1]
        with self.assertRaises(ValueError):auditor.calculate(base, duplicate, setup)
        bad = deepcopy(measured); bad['actors']['cases']['shared_uniform'][0]['graph_setup_seconds'] = 0
        with self.assertRaises(ValueError):auditor.calculate(base, bad, setup)
        bad_setup = deepcopy(setup); bad_setup['pools']['0']['eager_fallbacks'] = 1
        with self.assertRaises(ValueError):auditor.calculate(base, measured, bad_setup)

    def test_missing_actual_coverage_scope_or_changed_seed_set_is_rejected(self):
        base, measured, setup = self.fixtures()
        with self.assertRaises(ValueError):producer.reference_forecast(base, measured, setup, seeds=(101, 102, 103))
        wrong = deepcopy(base); wrong['forecast']['per_seed']['pilot']['records'] = 5000000
        with self.assertRaises(ValueError):producer.reference_forecast(wrong, measured, setup)
        missing = deepcopy(measured); missing['pipeline_observers'].pop()
        with self.assertRaises(ValueError):auditor.calculate(base, missing, setup)
        missing = deepcopy(measured); missing['actors']['cases'].pop('shared_uniform')
        with self.assertRaises(ValueError):auditor.calculate(base, missing, setup)


if __name__ == '__main__':unittest.main()
