"""Financial gates must reject incomplete diagnostics before formal collection."""
from copy import deepcopy
import unittest
from tools.run_nfsp_matrix_stage import stage_settings,review_funding,measured_setup


def costs(seconds=8.):
    return [{'phase':phase,'learner':learner,'status':'complete','records':100000,'seconds_including_setup':seconds,
             'counters':{'q_updates':750,'target_refreshes':7,'supervised_updates':0 if phase=='pre' else 750},
             'acting_policy_counts':{'average':0 if phase=='pre' else 50000}}
            for phase in ('pre','post') for learner in ('tabular','neural')]


class MatrixStageFundingTests(unittest.TestCase):
    def test_old_defaults_and_explicit_scoped_limits_preserve_record_recipe(self):
        profiles,trials=stage_settings('asymmetric')
        self.assertEqual([t['time_cap_seconds'] for t in trials],[5400.,5400.])
        self.assertEqual([p['config']['time_cap_seconds'] for p in profiles],[1800]*4)
        profiles,trials=stage_settings('rps',tabular_trial_hours=5/6,neural_trial_hours=1.,profile_seconds=65)
        self.assertEqual([t['time_cap_seconds'] for t in trials],[3000.,3600.])
        self.assertTrue(all(t['records']==15000000 and t['eval_every']==500000 and t['q_reuse']==1 for t in trials))
        self.assertTrue(all(p['config']['records']==100000 and p['config']['time_cap_seconds']==65 for p in profiles))

    def test_explicit_invalid_limits_do_not_fall_back_to_defaults(self):
        for changes in ({'profile_seconds':0},{'profile_seconds':float('nan')},{'tabular_trial_hours':0},
                        {'neural_trial_hours':-1},{'trial_hours':True},{'neural_trial_hours':float('inf')}):
            with self.subTest(changes=changes),self.assertRaises(ValueError):stage_settings('rps',**changes)

    def test_original_seed_propagates_without_changing_record_or_update_recipe(self):
        profiles,trials=stage_settings('rps',seed=1,profile_seconds=65,tabular_trial_hours=1/3,neural_trial_hours=.5)
        self.assertTrue(all(row['config']['seed']==1 for row in profiles))
        self.assertTrue(all(row['seed']==1 and row['records']==15000000 and row['q_reuse']==1 for row in trials))
        self.assertEqual([row['time_cap_seconds'] for row in trials],[1200.,1800.])
        for seed in (-1,True):
            with self.assertRaises(ValueError):stage_settings('rps',seed=seed)

    def test_complete_pre_post_including_setup_uses_original_twofold_gate(self):
        _,trials=stage_settings('rps',tabular_trial_hours=5/6,neural_trial_hours=1.)
        rows=review_funding(costs(8.),trials)
        self.assertTrue(all(row['twofold_margin_passed'] for row in rows))
        self.assertEqual([row['phase_weighted_15M_seconds'] for row in rows],[1200.,1200.])
        expensive=costs(11.);rows=review_funding(expensive,trials)
        self.assertFalse(rows[0]['twofold_margin_passed']);self.assertTrue(rows[1]['twofold_margin_passed'])

    def test_missing_target_average_records_or_phase_cannot_fund_formal_trials(self):
        _,trials=stage_settings('asymmetric')
        for mutate in (lambda r:r[0].update(status='budget_limited'),lambda r:r[1].update(records=90000),
                       lambda r:r[0]['counters'].update(target_refreshes=0),lambda r:r[2]['counters'].update(supervised_updates=0),
                       lambda r:r[2]['acting_policy_counts'].update(average=0),lambda r:r[0]['acting_policy_counts'].update(average=1)):
            rows=costs();mutate(rows);self.assertFalse(all(v['twofold_margin_passed'] for v in review_funding(rows,trials)))
        with self.assertRaises(ValueError):review_funding(costs()[:-1],trials)
        rows=costs();rows[0]['seconds_including_setup']=float('nan')
        with self.assertRaises(ValueError):review_funding(rows,trials)

    def test_resume_requires_and_accumulates_original_constructor_cost(self):
        self.assertEqual(measured_setup(None,.2,resuming=False),.2)
        self.assertAlmostEqual(measured_setup(.2,.3,resuming=True),.5)
        with self.assertRaises(ValueError):measured_setup(None,.3,resuming=True)
        with self.assertRaises(ValueError):measured_setup(.2,-1,resuming=True)
