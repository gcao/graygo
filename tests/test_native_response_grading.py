from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from graygo.engine import BLACK, WHITE
from graygo.native_response import branching_response
from graygo.solving import SolveBudgetExceeded
from graygo.task_adapter import TaskAdapter
from graygo.task_evaluation import EVALUATION_VERSION
from graygo.task_training import NFSPTaskConfig, NFSPTaskTrainer
from tools.branch_native_response import enumerated_response


class NativeResponseGradingTests(unittest.TestCase):
    def config(self, learner='tabular', records=80):
        return NFSPTaskConfig(time_cap_seconds=300, task='branching', learner=learner,
            purpose='profile', records=records, eval_every=16, batch=4, learning_start=4,
            capacity=128, target_every=8, reservoir_open=0, handoff_end=8,
            epsilon_decay=40, hidden=8, width=8, blocks=1, norm_groups=2)

    def table(self, adapter, probability):
        table = np.zeros((len(adapter.states), 2, adapter.actions))
        for state, index in adapter.index.items():
            for role in (BLACK, WHITE):
                legal = adapter.game.actions(state, role)
                table[index, role - 1, list(legal)] = ([1.] if len(legal) == 1
                                                       else [probability, 1 - probability])
        return table

    def test_hidden_type_is_mixed_before_maximization_and_public_branch_updates_belief(self):
        adapter = TaskAdapter(self.config())
        components = (self.table(adapter, 1.), self.table(adapter, 0.))
        own = self.table(adapter, 1.)
        root = adapter.game.root(BLACK)
        hidden = branching_response(adapter, root, BLACK, components, [.5, .5], own)
        oracle = sum(.5 * branching_response(adapter, root, BLACK, (c,), [1.], own)['best_response_value']
                     for c in components)
        self.assertAlmostEqual(hidden['best_response_value'], 0.)
        self.assertAlmostEqual(oracle, .75)
        observed = branching_response(adapter, root, WHITE, components, [.5, .5], own)
        self.assertAlmostEqual(observed['best_response_value'], 1.)
        self.assertAlmostEqual(observed['greedy_value'], 0.)

    def test_independent_complete_strategy_enumeration_matches_every_root_and_role(self):
        adapter = TaskAdapter(self.config())
        rng = np.random.default_rng(1927)
        for _ in range(8):
            tables = []
            for _ in range(4):
                table = self.table(adapter, .5)
                for state, index in adapter.index.items():
                    for role in (BLACK, WHITE):
                        if len(adapter.game.actions(state, role)) == 2:
                            table[index, role - 1] = rng.dirichlet([1., 1.])
                tables.append(table)
            components, own = tables[:3], tables[3]
            weights = rng.dirichlet([1., 1., 1.])
            for root in adapter.roots:
                for role in (BLACK, WHITE):
                    expected_best, expected_actual, _ = enumerated_response(
                        adapter, root, role, components, weights, own)
                    row = branching_response(adapter, root, role, components, weights, own)
                    self.assertAlmostEqual(row['best_response_value'], expected_best, places=12)
                    self.assertAlmostEqual(row['greedy_value'], expected_actual, places=12)
                    self.assertAlmostEqual(row['gap'], expected_best - expected_actual, places=12)

    def calibrate(self, trainer, records):
        # Exact fixture, not a training result. Seeing the controller take
        # branch one reveals its Q component: the average always takes zero.
        # Account for that posterior instead of using unconditional weights.
        trainer.backend.counters.fresh_records = records
        trainer.backend.counters.fresh_q = records
        trainer.backend.counters.supervised_updates = 1
        epsilon = trainer.schedule.epsilon(records)
        values = (-1 / 7, -7 / 22)
        bad_branch = .1 * epsilon / 2
        q_weight_after_zero = .1 * (1 - epsilon / 2) / (1 - bad_branch)
        with torch.no_grad():
            for state in trainer.adapter.states:
                for role in (BLACK, WHITE):
                    token = int(trainer.adapter.game.observation(state, role).spatial.argmax())
                    q = trainer.backend.q.values[token]
                    average = trainer.backend.average.probabilities[token]
                    q.zero_(); average.zero_()
                    controller = role == state.controller
                    if state.depth == 0:
                        if controller:
                            q[:] = torch.tensor(values)
                        else:
                            branch_one_target_value = .375 - .125 * epsilon / 2
                            q[0] = (1 - bad_branch) / 7 + bad_branch * branch_one_target_value
                        average[0] = 1.
                    elif state.branch == 0:
                        q[:] = values[0] if controller else -values[0]
                        weight = q_weight_after_zero if controller else .1
                        probability = (3 / 7 - weight * .5) / (1 - weight)
                        average[:] = torch.tensor([probability, 1 - probability], dtype=torch.float64)
                    else:
                        if controller:
                            q[:] = values[1]
                            probability = 5 / 11
                        else:
                            # Conditional opponent Q is uniform; these exact
                            # action values make the response choose action one.
                            q[:] = torch.tensor([.25, .375])
                            probability = (5 / 11 - .1 * epsilon / 2) / .9
                        average[:] = torch.tensor([probability, 1 - probability], dtype=torch.float64)

    def test_actual_evaluator_and_trainer_accept_exact_native_response_with_non_nash_greedy_q(self):
        with tempfile.TemporaryDirectory() as directory:
            trainer = NFSPTaskTrainer(self.config(), directory, reserve_bytes=0)
            for records in (16, 32, 48, 64, 80):
                self.calibrate(trainer, records)
                trainer.evaluate()
            result = trainer.result('complete')
            self.assertEqual(result['evaluation_protocol_version'], EVALUATION_VERSION)
            self.assertEqual(trainer.manifest['evaluation_protocol_version'], EVALUATION_VERSION)
            self.assertEqual(result['q_credit_metric'], 'native_best_response_gap')
            self.assertLess(result['final_five_max_native_response_gap'], 1e-7)
            self.assertLess(result['final_five_max_early_error'], 1e-7)
            self.assertGreater(result['final_five_max_greedy_gap'], .05)
            self.assertLess(result['final_five_max_average_gap'], .05)
            self.assertTrue(result['q_credit_passed'])
            self.assertEqual(result['average_readiness'], 'short_game_pass')
            self.assertFalse(result['formal_case_passed'])  # Tiny analytical fixture.
            self.assertFalse(result['a2_complete'])
            # Missing v2 evidence must fail, never fall back to the old metric.
            saved = deepcopy(trainer.evaluations)
            del trainer.evaluations[0]['roots'][0]['native_response']
            with self.assertRaises(KeyError):
                trainer.result('complete')
            trainer.evaluations = saved
            # With the same deployed average, actual bad greedy actions against
            # the native opponent are detected and deny Q credit.
            with torch.no_grad():
                for state in trainer.adapter.states:
                    if state.depth == 1:
                        for role in (BLACK, WHITE):
                            token = int(trainer.adapter.game.observation(state, role).spatial.argmax())
                            trainer.backend.q.values[token] = torch.tensor([1., -1.])
            trainer.evaluations.pop()
            trainer.evaluate()
            bad = trainer.result('complete')
            self.assertGreater(bad['final_five_max_native_response_gap'], .05)
            self.assertFalse(bad['q_credit_passed'])
            self.assertLess(bad['final_five_max_average_gap'], .05)

    def test_mid_episode_neural_resume_retains_learning_rng_and_new_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            config = self.config('neural', 32)
            full = NFSPTaskTrainer(config, Path(directory) / 'full', reserve_bytes=0)
            expected = full.run(verbose=False)
            part = NFSPTaskTrainer(config, Path(directory) / 'part', reserve_bytes=0)
            part.run(until_records=2, verbose=False)
            self.assertIsNotNone(part.active)
            resumed = NFSPTaskTrainer(config, Path(directory) / 'part', resume=True, reserve_bytes=0)
            actual = resumed.run(verbose=False)
            for left, right in zip(expected['evaluations'], actual['evaluations']):
                self.assertEqual({k: v for k, v in left.items() if k != 'elapsed_seconds'},
                                 {k: v for k, v in right.items() if k != 'elapsed_seconds'})
            self.assertEqual(expected['counters'], actual['counters'])
            for name in ('q', 'target', 'average'):
                for key, value in getattr(full.backend, name).state_dict().items():
                    torch.testing.assert_close(value, getattr(resumed.backend, name).state_dict()[key], rtol=0, atol=0)
            for name, rng in full.rng.items():
                self.assertEqual(rng.bit_generator.state, resumed.rng[name].bit_generator.state)
            self.assertEqual((Path(directory) / 'full/episodes.bin').read_bytes(),
                             (Path(directory) / 'part/episodes.bin').read_bytes())

    def test_invalid_probabilities_scope_and_exhausted_node_budget_fail_closed(self):
        adapter = TaskAdapter(self.config())
        table = self.table(adapter, .5)
        root = adapter.roots[0]
        for weights in ([.5], [float('nan')], [-1.], [1., 0.]):
            with self.assertRaises(ValueError):
                branching_response(adapter, root, BLACK, (table,), weights, table)
        invalid = table.copy(); invalid[adapter.index[root], WHITE - 1] = 0
        with self.assertRaises(ValueError):
            branching_response(adapter, root, BLACK, (invalid,), [1.], table)
        with self.assertRaises(ValueError):
            branching_response(adapter, root, BLACK, (table[:-1],), [1.], table)
        limited = TaskAdapter(self.config())
        limited.config = replace(limited.config, max_nodes=1)
        with self.assertRaises(SolveBudgetExceeded):
            branching_response(limited, limited.roots[0], BLACK, (self.table(limited, .5),), [1.], self.table(limited, .5))
        other = TaskAdapter(replace(self.config(), task='delayed', horizons=(2,)))
        with self.assertRaises(ValueError):
            branching_response(other, other.roots[0], BLACK, (table,), [1.], table)


if __name__ == '__main__':
    unittest.main()
