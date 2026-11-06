from copy import deepcopy
import unittest
from tools.adaptive_gate_schedule import initial_state, observe_gate, observe_screen, screen_due


def screen(score, *, history=.9, incumbent='champion', records=100):
    def row(value, identity):
        return {'games': 20, 'win_score': value, 'opponent_sha256': identity, 'status': 'complete'}
    return {'status': 'complete', 'average_eligible': score > .5, 'candidate_records': records,
            'heads': {'average': {'opponents': {'champion': row(score, incumbent),
                                               'historical-0': row(history, 'old-model')}}}}


class AdaptiveScheduleTests(unittest.TestCase):
    def test_strong_screen_starts_gate_immediately_without_waiting_for_timer(self):
        original = initial_state(1000)
        state, action = observe_screen(original, screen(.9), 1000, 'champion')
        self.assertTrue(action['run_gate'])
        self.assertEqual(state['interval_seconds'], 1800)
        self.assertEqual(original, initial_state(1000))

    def test_close_improvement_gets_time_to_train_then_can_gate_at_timer(self):
        state, action = observe_screen(initial_state(0), screen(.55), 0, 'champion')
        self.assertFalse(action['run_gate'])
        state, action = observe_screen(state, screen(.65), 3600, 'champion')
        self.assertTrue(action['run_gate'])
        self.assertEqual(state['interval_seconds'], 3600)

    def test_repeated_flat_results_back_off_to_four_hours(self):
        state = initial_state(0)
        for now in range(0, 3600*8, 3600):
            state, action = observe_screen(state, screen(.4), now, 'champion')
            self.assertFalse(action['run_gate'])
            self.assertGreaterEqual(state['interval_seconds'], 1800)
            self.assertLessEqual(state['interval_seconds'], 14400)
        self.assertEqual(state['interval_seconds'], 14400)
        self.assertEqual(state['next_screen_learning_seconds'], now+3600)

    def test_strong_recovery_shortens_a_long_training_interval(self):
        state = initial_state(0)
        for now in (0, 3600, 7200, 10800, 14400):
            state, _ = observe_screen(state, screen(.4), now, 'champion')
        before = state['interval_seconds']
        state, action = observe_screen(state, screen(.9), 18000, 'champion')
        self.assertTrue(action['run_gate'])
        self.assertLess(state['interval_seconds'], before)

    def test_historical_regression_blocks_acceleration(self):
        state, _ = observe_screen(initial_state(0), screen(.6, history=1.), 0, 'champion')
        state, action = observe_screen(state, screen(.9, history=.6), 3600, 'champion')
        self.assertFalse(action['run_gate'])
        self.assertEqual(state['last_signal'], 'history_regression')

    def test_incomplete_screen_does_not_count_as_slow_learning_or_start_gate(self):
        state = initial_state(0)
        partial = screen(.9)
        partial['status'] = 'budget_limited'
        state, action = observe_screen(state, partial, 0, 'champion')
        self.assertFalse(action['run_gate'])
        self.assertEqual(state['interval_seconds'], 3600)
        self.assertFalse(screen_due(state, 100))

    def test_full_gate_not_passed_backs_off_and_does_not_retrigger_on_same_score(self):
        state, _ = observe_screen(initial_state(0), screen(.9), 0, 'champion')
        state = observe_gate(state, {'status': 'complete'}, 0, 'champion')
        self.assertEqual(state['interval_seconds'], 3600)
        state, action = observe_screen(state, screen(.9), 1800, 'champion')
        self.assertFalse(action['run_gate'])
        state, action = observe_screen(state, screen(1.), 1800, 'champion')
        self.assertTrue(action['run_gate'])

    def test_plateau_cannot_postpone_eligible_gate_beyond_maximum_training_gap(self):
        state, _ = observe_screen(initial_state(0), screen(.55), 0, 'champion')
        state = observe_gate(state, {'status': 'complete'}, 0, 'champion')
        for now in (3600, 7200, 10800):
            state, _ = observe_screen(state, screen(.55), now, 'champion')
        state, action = observe_screen(state, screen(.55), 14400, 'champion')
        self.assertTrue(action['run_gate'])

    def test_promotion_resets_reference_but_preserves_cumulative_gate_count(self):
        state, _ = observe_screen(initial_state(0), screen(.9), 0, 'champion')
        state = observe_gate(state, {'status': 'complete'}, 0, 'champion', promoted=True)
        self.assertEqual(state['gate_count'], 1)
        self.assertEqual(state['interval_seconds'], 1800)
        state, action = observe_screen(state, screen(.5, incumbent='new-champion'), 1800, 'new-champion')
        self.assertFalse(action['run_gate'])
        self.assertNotEqual(state['last_signal'], 'flat_or_behind')

    def test_unfinished_gate_does_not_change_schedule(self):
        state = initial_state(0)
        before = deepcopy(state)
        with self.assertRaises(ValueError):
            observe_gate(state, {'status': 'budget_limited'}, 1000, 'champion')
        self.assertEqual(state, before)


if __name__ == '__main__':
    unittest.main()
