from copy import deepcopy
from dataclasses import replace
from decimal import Decimal
import hashlib
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch
from tests import test_board_execution as board_fixtures
from tools.nfsp_resources import NFSPResourcePlan, TERMS, canonical
from tools import nfsp_development_scope as scope


class DevelopmentScopeTests(unittest.TestCase):
    def plan(self, **changes):
        terms = {key: None for key in TERMS}
        terms.update(P='2', U='1', C1='0', C4='0', B_U='5', H_P='2', H_D='3', H_N='4')
        fields = dict(name='scope-fixture', learner_source_sha256='a' * 64, selected_reuse=4,
                      selected_target='one-step', term_device_hours=terms,
                      evidence_sha256={key: 'b' * 64 for key, value in terms.items() if value is not None},
                      r4_selected=True, other_unreused_device_hours=Decimal('.5'))
        fields.update(changes)
        return NFSPResourcePlan(**fields)

    def decision(self, plan, mode='limited-pilot', **changes):
        record = 'Owner approves exactly the declared ' + mode + ' scope in this unit fixture.'
        value = {'version': scope.VERSION, 'owner_approved': True,
                 'resource_plan_sha256': canonical(plan.to_dict()), 'mode': mode,
                 'project_device_hour_ceiling': '200', 'uniform_board_comparison_selected': mode == 'full-study',
                 'decision_record': record, 'decision_record_sha256': hashlib.sha256(record.encode()).hexdigest()}
        value.update(changes)
        return value

    def selection(self, plan, mode='limited-pilot', **changes):
        return scope.ScopeSelection.from_decision(plan, self.decision(plan, mode, **changes))

    def test_filled_costs_do_not_select_or_authorize_a_campaign(self):
        report = scope.scope_review(self.plan())
        self.assertEqual(report['status'], 'owner_decision_missing')
        self.assertFalse(report['selected_scope_costed_and_within_ceiling'])
        self.assertFalse(report['execution_authorized'])
        self.assertFalse(report['runtime_prices_independently_qualified'])

    def test_selected_prefix_costs_keep_hd_cumulative_and_optional_uniform_explicit(self):
        plan = self.plan(); selection = self.selection(plan)
        report = scope.scope_review(plan, selection)
        self.assertEqual(report['scope_ceiling_device_hours'],
                         {'stop': '0', 'limited-pilot': '105.5', 'limited-screening': '108.5', 'full-study': '177.5'})
        self.assertTrue(report['selected_scope_costed_and_within_ceiling'])
        with_uniform = scope.scope_review(plan, self.selection(plan, uniform_board_comparison_selected=True))
        self.assertEqual(with_uniform['scope_ceiling_device_hours']['limited-pilot'], '110.5')
        self.assertEqual(with_uniform['scope_ceiling_device_hours']['limited-screening'], '113.5')
        self.assertEqual(scope.scope_review(plan, self.selection(plan, project_device_hour_ceiling='106'))['status'],
                         'scope_numerically_affordable')
        self.assertFalse(scope.scope_review(plan, self.selection(plan, 'full-study', project_device_hour_ceiling='106'))['selected_scope_costed_and_within_ceiling'])

    def test_decision_must_bind_exact_plan_answer_and_existing_ceiling(self):
        plan = self.plan()
        for change in ({'owner_approved': False}, {'resource_plan_sha256': 'c' * 64},
                       {'version': 'graygo/control-completion-budget-proposal/1'},
                       {'decision_record': 'Changed answer'}, {'decision_record_sha256': 'c' * 64},
                       {'mode': 'automatic'}, {'project_device_hour_ceiling': True}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                scope.ScopeSelection.from_decision(plan, self.decision(plan, **change))
        fixed = self.plan(project_device_hour_ceiling='200')
        with self.assertRaises(ValueError):
            self.selection(fixed, project_device_hour_ceiling='201')
        with self.assertRaises(ValueError):
            self.selection(plan, 'full-study', uniform_board_comparison_selected=False)

    def test_missing_prices_references_or_component_headroom_cannot_fund_a_scope(self):
        original = self.plan()
        for key in ('H_P', 'H_D', 'H_N', 'C1', 'C4'):
            terms = dict(original.term_device_hours); terms[key] = None
            plan = self.plan(term_device_hours=terms)
            self.assertFalse(scope.scope_review(plan, self.selection(plan))['selected_scope_costed_and_within_ceiling'])
        references = dict(original.evidence_sha256); references.pop('C1')
        plan = self.plan(evidence_sha256=references)
        self.assertFalse(scope.scope_review(plan, self.selection(plan))['selected_scope_costed_and_within_ceiling'])
        terms = dict(original.term_device_hours); terms.update(H_P='25', H_D='26')
        plan = self.plan(term_device_hours=terms)
        self.assertFalse(scope.scope_review(plan, self.selection(plan))['selected_scope_costed_and_within_ceiling'])
        self.assertEqual(scope.scope_review(plan, self.selection(plan, 'stop', project_device_hour_ceiling='0'))['status'], 'stop')

    def test_unpriced_uniform_comparison_blocks_full_study_without_selecting_it_for_a_limited_pilot(self):
        for value in (None, '0'):
            terms = dict(self.plan().term_device_hours); terms['B_U'] = value
            plan = self.plan(term_device_hours=terms)
            pilot = scope.scope_review(plan, self.selection(plan))
            self.assertTrue(pilot['selected_scope_costed_and_within_ceiling'])
            self.assertIsNone(pilot['scope_ceiling_device_hours']['full-study'])
            self.assertTrue(pilot['full_study_price_gaps'])
            self.assertFalse(scope.scope_review(plan, self.selection(plan, uniform_board_comparison_selected=True))['selected_scope_costed_and_within_ceiling'])
            self.assertFalse(scope.scope_review(plan, self.selection(plan, 'full-study'))['selected_scope_costed_and_within_ceiling'])

    def cohort(self, *, status='pass', source='a' * 64, reuse=4, device='cuda:0'):
        return SimpleNamespace(sha256='c' * 64,
            body={'source_sha256': source, 'training_runs': {str(s): {} for s in (100, 101, 102)},
                  'shared_config_without_seed': {'q_reuse': reuse, 'device': device}},
            report=Mock(return_value={'body': {'status': status, 'pilot': {'status': status}}}))

    def test_stage_endpoints_keep_full_horizon_source_and_all_registered_seeds(self):
        plan = self.plan(); selected = self.selection(plan)
        for seed in (100, 101, 102):
            initial = scope.plan_development_stage(plan, selected, 'initialize', seed)
            self.assertEqual(initial['endpoint_records'], 0)
            self.assertEqual(initial['unchanged_development_record_horizon'], 20000000)
        with patch.object(scope.DevelopmentCohort, 'load', return_value=self.cohort()):
            pre = scope.plan_development_stage(plan, selected, 'pilot-pre', 100, cohort_root='/registered')
            post = scope.plan_development_stage(plan, selected, 'pilot-post', 100, cohort_root='/registered')
            self.assertEqual((pre['endpoint_records'], post['endpoint_records']), (5000000, 8000000))
            self.assertEqual(Decimal(pre['cumulative_stage_time_ceiling_seconds']), 7200)
            self.assertEqual(Decimal(pre['unchanged_execution_contract_time_ceiling_seconds']), 10800)
            self.assertTrue(post['fixed_opening_evidence_required_before_post_collection'])
            self.assertFalse(post['execution_authorized'])
        for seed in (0, 200, True):
            with self.assertRaises(ValueError):
                scope.plan_development_stage(plan, selected, 'initialize', seed)
        with self.assertRaises(ValueError):
            scope.plan_development_stage(plan, selected, 'pilot-pre', 100)
        for changed in (self.cohort(source='d' * 64), self.cohort(reuse=1), self.cohort(device='cpu')):
            with patch.object(scope.DevelopmentCohort, 'load', return_value=changed), self.assertRaises(ValueError):
                scope.plan_development_stage(plan, selected, 'pilot-pre', 100, cohort_root='/registered')
        terms = dict(plan.term_device_hours); terms.update(H_P='0', H_D='0')
        cpu_plan = self.plan(term_device_hours=terms)
        with self.assertRaises(ValueError):
            scope.plan_development_stage(cpu_plan, self.selection(cpu_plan), 'initialize', 100)

    def test_screening_requires_separate_scope_and_actual_complete_cohort_pass(self):
        plan = self.plan()
        with patch.object(scope.DevelopmentCohort, 'load', return_value=self.cohort()), self.assertRaises(ValueError):
            scope.plan_development_stage(plan, self.selection(plan), 'screening', 100, cohort_root='/registered')
        for status in ('incomplete', 'no_go'):
            with patch.object(scope.DevelopmentCohort, 'load', return_value=self.cohort(status=status)), self.assertRaises(ValueError):
                scope.plan_development_stage(plan, self.selection(plan, 'limited-screening'), 'screening', 100, cohort_root='/registered')
        actual_reader = self.cohort()
        with patch.object(scope.DevelopmentCohort, 'load', return_value=actual_reader):
            stage = scope.plan_development_stage(plan, self.selection(plan, 'limited-screening'), 'screening', 100, cohort_root='/registered')
            self.assertEqual(stage['endpoint_records'], 20000000)
            actual_reader.report.assert_called_once_with('pilot')
        with self.assertRaises(ValueError):
            scope.plan_development_stage(plan, self.selection(plan, 'full-study'), 'confirmation', 100)


class CumulativeStageCapTests(unittest.TestCase):
    def setUp(self):
        self.fixture = board_fixtures.BoardExecutionTests(methodName='runTest')

    def test_real_executor_stage_budget_survives_resume_without_contract_or_counter_reset(self):
        clock = board_fixtures.FakeClock()
        with tempfile.TemporaryDirectory() as directory:
            with self.fixture.executor(directory, clock=clock) as run:
                original_contract = deepcopy(run.contract); advance = run.runtime.advance
                def step(*args, **kwargs):
                    advance(*args, **kwargs); clock.now += 2
                with patch.object(run.runtime, 'advance', side_effect=step):
                    result = scope.run_with_cumulative_stage_cap(run, 32, 3)
                self.assertEqual(result['status'], 'stage_budget_limited')
                self.assertEqual(result['executor_report']['status'], 'interrupted')
                self.assertEqual(result['executor_report']['records'], 16)
                self.assertEqual(run.contract, original_contract)
                self.assertEqual(run.runtime.backend.counters.fresh_records, 16)
            clock.now = 100
            with self.fixture.executor(directory, resume=True, clock=clock) as resumed:
                same = scope.run_with_cumulative_stage_cap(resumed, 32, 3)
                self.assertEqual(same['status'], 'stage_budget_limited')
                self.assertEqual(same['executor_report']['records'], 16)
                advance = resumed.runtime.advance
                def next_step(*args, **kwargs):
                    advance(*args, **kwargs); clock.now += 2
                with patch.object(resumed.runtime, 'advance', side_effect=next_step):
                    continued = scope.run_with_cumulative_stage_cap(resumed, 32, 8)
                self.assertEqual(continued['executor_report']['records'], 32)
                self.assertEqual(resumed.contract, original_contract)
                self.assertFalse(continued['execution_authority_granted'])

    def test_unhit_stage_guard_preserves_actual_learning_rng_and_buffer_state(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.fixture.executor(Path(directory) / 'plain') as plain:
                plain.run(32); expected = plain.runtime.state_dict(); expected.pop('timings')
                expected_arrays = {name: {key: value.copy() for key, value in buffer.arrays.items()}
                    for name, buffer in (('replay', plain.runtime.replay), ('reservoir', plain.runtime.reservoir))}
            with self.fixture.executor(Path(directory) / 'guarded') as guarded:
                scope.run_with_cumulative_stage_cap(guarded, 32, 1000)
                actual = guarded.runtime.state_dict(); actual.pop('timings')
                board_fixtures.exact(expected, actual)
                for name, arrays in expected_arrays.items():
                    for key, value in arrays.items():
                        board_fixtures.np.testing.assert_array_equal(value, getattr(guarded.runtime, name).arrays[key])

    def test_stage_cap_cannot_widen_contract_or_mask_storage_failure(self):
        clock = board_fixtures.FakeClock()
        with tempfile.TemporaryDirectory() as directory:
            with self.fixture.executor(directory, clock=clock) as run:
                for cap in (0, -1, True, float('nan'), float('inf'), 1001):
                    with self.subTest(cap=cap), self.assertRaises(ValueError):
                        scope.run_with_cumulative_stage_cap(run, 32, cap)
                self.assertEqual(run.runtime.records, 0)
        raw = {'status': 'storage_limited', 'contract_sha256': 'a' * 64, 'budget_boundary': 'paired chunks'}
        fake = SimpleNamespace(active=True, plan=SimpleNamespace(time_ceiling_seconds=100),
                               contract={'version': scope.EXECUTION_VERSION}, _spent=lambda: 50)
        def stopped_run(until, *, stop_requested):
            self.assertTrue(stop_requested()); return raw
        fake.run = stopped_run
        self.assertEqual(scope.run_with_cumulative_stage_cap(fake, 32, 20)['status'], 'storage_limited')


if __name__ == '__main__':
    unittest.main()
