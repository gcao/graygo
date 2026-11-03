import unittest
import numpy as np
from graygo.engine import BLACK, WHITE
from graygo.task_adapter import TaskAdapter
from graygo.task_training import NFSPTaskConfig
from graygo.public_game import markov_gaps
from tools.branch_native_response import native_value, enumerated_response, diagnose


class NativeBranchResponseTests(unittest.TestCase):
    def setUp(self):
        self.adapter = TaskAdapter(NFSPTaskConfig(task='branching', learner='neural', purpose='profile',
            records=32, eval_every=16, batch=8, learning_start=8, capacity=32, q_reuse=4, time_cap_seconds=30))

    def table(self, probability):
        result = np.zeros((len(self.adapter.states), 2, self.adapter.actions))
        for state, index in self.adapter.index.items():
            for role in (BLACK, WHITE):
                legal = self.adapter.game.actions(state, role)
                if len(legal) == 1:result[index, role - 1, legal[0]] = 1
                else:result[index, role - 1, :2] = [probability, 1 - probability]
        return result

    def test_hidden_opponent_type_is_mixed_before_best_response_not_revealed_by_an_oracle(self):
        root = self.adapter.game.root(BLACK); components = (self.table(1.), self.table(0.)); weights = np.array([.5, .5])
        best = native_value(self.adapter, root, BLACK, components, weights)
        oracle = sum(weight * native_value(self.adapter, root, BLACK, (component,), np.array([1.]))
                     for weight, component in zip(weights, components))
        self.assertAlmostEqual(best, 0.)
        self.assertAlmostEqual(oracle, .75)
        self.assertGreater(oracle, best)
        self.assertAlmostEqual(native_value(self.adapter, root, BLACK, components, weights, self.table(1.)), 0.)

    def test_actual_public_root_branch_can_update_the_opponent_component_belief(self):
        root = self.adapter.game.root(BLACK)
        components = (self.table(1.), self.table(0.)); weights = np.array([.5, .5])
        # Opponent black's public root choice reveals which component acted;
        # white may condition its later response on that observed branch.
        best = native_value(self.adapter, root, WHITE, components, weights)
        enumerated, actual, count = enumerated_response(self.adapter, root, WHITE, components, weights, self.table(1.))
        self.assertAlmostEqual(best, 1.)
        self.assertAlmostEqual(enumerated, best); self.assertAlmostEqual(actual, 0.)
        self.assertEqual(count, 4)

    def test_best_response_quality_is_distinct_from_the_greedy_equilibrium_gap(self):
        opponent = self.table(.5)
        for state, index in self.adapter.index.items():
            if state.depth == 0:
                opponent[index, state.controller - 1] = [1., 0.]
            else:
                p = 3 / 7 if state.branch == 0 else 5 / 11
                opponent[index, :, 0] = p; opponent[index, :, 1] = 1 - p
        greedy = self.table(1.)
        roots = diagnose(self.adapter, (opponent,), np.array([1.]), greedy)
        self.assertLess(max(role['greedy_native_best_response_gap'] for root in roots for role in root['roles']), 1e-12)
        root = self.adapter.game.root(BLACK)
        def policy(role):return lambda state:greedy[self.adapter.index[state], role - 1, list(self.adapter.game.actions(state, role))]
        legacy = markov_gaps(self.adapter.game, root, policy(BLACK), policy(WHITE), -1 / 7)
        self.assertGreater(max(legacy), .05)

    def test_belief_recursion_and_independent_complete_strategies_agree_for_random_native_policies(self):
        rng = np.random.default_rng(829)
        for _ in range(12):
            components = []
            for _ in range(3):
                table = self.table(.5)
                for state, index in self.adapter.index.items():
                    for role in (BLACK, WHITE):
                        if len(self.adapter.game.actions(state, role)) == 2:table[index, role - 1] = rng.dirichlet([1., 1.])
                components.append(table)
            weights = rng.dirichlet([1., 1., 1.]); own = self.table(.5)
            for root in self.adapter.roots:
                for role in (BLACK, WHITE):
                    best, actual, _ = enumerated_response(self.adapter, root, role, components, weights, own)
                    self.assertAlmostEqual(best, native_value(self.adapter, root, role, components, weights), places=12)
                    self.assertAlmostEqual(actual, native_value(self.adapter, root, role, components, weights, own), places=12)

    def test_unobservable_or_invalid_probability_data_cannot_be_accepted(self):
        root = self.adapter.game.root(BLACK)
        with self.assertRaises(ValueError):native_value(self.adapter, root, BLACK, (self.table(.5),), np.array([.5]))
        invalid = self.table(.5); invalid[self.adapter.index[root], WHITE - 1] = [0., 0.]
        with self.assertRaises(ValueError):native_value(self.adapter, root, BLACK, (invalid,), np.array([1.]))


if __name__ == '__main__':unittest.main()
