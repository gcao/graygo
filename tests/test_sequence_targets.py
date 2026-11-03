from copy import deepcopy
from dataclasses import replace
from fractions import Fraction
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from torch import nn

from graygo.buffers import ReplayBuffer
from graygo.engine import BLACK, WHITE, Rules
from graygo.nfsp import NFSPConfig, OneStepBatch, one_step_targets
from graygo.sequence_targets import SequenceBatch, retrace_targets, update_retrace_q, TraceDiagnostics
from graygo.task_training import NFSPTaskConfig, NFSPTaskTrainer
from graygo.task_adapter import TaskAdapter
from graygo.board_runtime import BoardRuntimeConfig, BoardNFSPRuntime


class Table(nn.Module):
    def __init__(self, values):
        super().__init__()
        self.values = nn.Parameter(torch.tensor(values, dtype=torch.float32))
        self.calls = []

    def forward(self, spatial, globals):
        if not bool((spatial.sum(-1) == 1).all()):
            raise AssertionError('padding or terminal placeholder reached a network')
        self.calls.append((torch.is_grad_enabled(), len(spatial)))
        return spatial @ self.values


def data_for(actions=(0, 1, 0), behavior=(.9, .9, .2), terminal=False, padding=0):
    length = len(actions); total = length + padding; tokens = length + 1
    valid = np.zeros((1, total), dtype=bool); valid[0, :length] = True
    spatial = np.zeros((1, total, tokens), dtype=np.float32)
    following = spatial.copy()
    legal = np.zeros((1, total, 2), dtype=bool)
    next_legal = legal.copy()
    for i in range(length):
        spatial[0, i, i] = 1; following[0, i, i + 1] = 1
        legal[0, i] = True; next_legal[0, i] = True
    terminated = ~valid.copy()
    if terminal:
        terminated[0, length - 1] = True
        next_legal[0, length - 1] = False
        following[0, length - 1] = 0
    data = {'spatial': spatial, 'globals': np.empty((1, total, 0), dtype=np.float32), 'legal': legal,
            'next_spatial': following, 'next_globals': np.empty((1, total, 0), dtype=np.float32),
            'next_legal': next_legal, 'valid': valid, 'terminated': terminated,
            'action': np.array([list(actions) + [0] * padding], dtype=np.int64),
            'reward': np.zeros((1, total), dtype=np.float32),
            'behavior_prob': np.array([list(behavior) + [1.] * padding], dtype=np.float32),
            'value_mask': valid.copy(), 'mode': np.zeros((1, total), dtype=np.uint8),
            'supplemental': np.zeros((1, total), dtype=bool)}
    for name in ('stream', 'opponent_version', 'policy_version'):
        data[name] = np.zeros((1, total), dtype=np.uint64)
    data['step'] = np.arange(total, dtype=np.uint32)[None]
    data['ordinal'] = np.arange(total, dtype=np.uint64)[None]
    return data


class SequenceTargetTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1); torch.manual_seed(7)

    def test_off_policy_indexing_matches_independent_fraction_calculation_and_ignores_c_at_start(self):
        data = data_for()
        online = Table([[.1, .8], [.8, .1], [.8, .1], [.8, .1]])
        target = Table([[.25, -.5], [.5, -.25], [-.5, .75], [.125, .5]])
        batch = SequenceBatch.from_replay(data, 'cpu')
        labels, stats, details = retrace_targets(online, target, batch, .2, target_batch=2)
        # Current online pi chooses the target distribution; frozen Q values it.
        # c0=.95*(.1/.9) is deliberately small and MUST NOT weight delta0.
        c1 = Fraction(19, 20) * Fraction(1, 9)
        c2 = Fraction(19, 20)
        d0 = Fraction(17, 40) - Fraction(1, 4)
        d1 = -Fraction(3, 8) + Fraction(1, 4)
        d2 = Fraction(13, 80) + Fraction(1, 2)
        expected = Fraction(1, 4) + d0 + c1 * d1 + c1 * c2 * d2
        self.assertAlmostEqual(float(labels[0]), float(expected), places=6)
        self.assertAlmostEqual(details['trace_mass'][0], float(1 + c1 + c1 * c2), places=6)
        self.assertGreater(details['trace_mass'][0], 1.)
        self.assertFalse(labels.requires_grad)
        self.assertIsNone(target.values.grad); self.assertIsNone(online.values.grad)
        self.assertTrue(all(not grad and size <= 2 for grad, size in online.calls + target.calls))
        self.assertEqual(stats['targets'], 1); self.assertEqual(stats['context_records'], 2)
        self.assertEqual(stats['nonterminal_boundaries'], 1)
        # Behavior at the conditioned start affects diagnostic overlap only.
        data['behavior_prob'][0, 0] = .05
        other, _, _ = retrace_targets(online, target, SequenceBatch.from_replay(data, 'cpu'), .2)
        torch.testing.assert_close(labels, other, rtol=0, atol=0)

    def test_one_step_reduction_terminal_padding_and_continuing_boundary_bootstrap(self):
        for terminal in (False, True):
            data = data_for((0,), (.5,), terminal=terminal, padding=4)
            data['spatial'][0, 1:] = np.nan  # Must never forward padding.
            if terminal: data['reward'][0, 0] = -1.
            online = Table([[.1, .2], [.8, .1]])
            target = Table([[.25, -.5], [.5, -.25]])
            batch = SequenceBatch.from_replay(data, 'cpu')
            result, stats, details = retrace_targets(online, target, batch, .2)
            expected, _ = one_step_targets(online, target, batch.starts, .2)
            torch.testing.assert_close(result, expected, rtol=0, atol=1e-7)
            self.assertEqual(details['trace_mass'][0], 1.)
            self.assertEqual(stats['nonterminal_boundaries'], int(not terminal))
        # lambda=0 reduces a longer retained sequence to the same one-step target.
        data = data_for()
        online = Table([[.1, .8], [.8, .1], [.8, .1], [.8, .1]])
        target = Table([[.25, -.5], [.5, -.25], [-.5, .75], [.125, .5]])
        batch = SequenceBatch.from_replay(data, 'cpu')
        actual, _, _ = retrace_targets(online, target, batch, .2, trace_lambda=0)
        expected, _ = one_step_targets(online, target, batch.starts, .2)
        torch.testing.assert_close(actual, expected, rtol=0, atol=1e-7)

    def test_on_policy_telescoping_and_projection_apply_only_to_loss_target(self):
        data = data_for((0, 0, 0), (1., 1., 1.), terminal=True)
        data['reward'][0, 2] = 1.
        online = Table([[.8, .1]] * 4)
        target = Table([[.2, .1], [-.2, .1], [.4, .1], [.1, .1]])
        labels, _, details = retrace_targets(online, target, SequenceBatch.from_replay(data, 'cpu'), 0., trace_lambda=1.)
        torch.testing.assert_close(labels, torch.tensor([1.]), atol=1e-7, rtol=0)
        self.assertEqual(details['trace_mass'][0], 3.)
        data = data_for((0, 1), (1., .01), terminal=True)
        data['reward'][0, 1] = 1.
        online = Table([[.8, .1]] * 3)
        target = Table([[0., 0.], [1., -1.], [0., 0.]])
        labels, stats, _ = retrace_targets(online, target, SequenceBatch.from_replay(data, 'cpu'), .02)
        self.assertEqual(float(labels[0]), 1.)
        self.assertAlmostEqual(stats['raw_target_max'], 2.88, places=5)
        self.assertEqual(stats['projected_targets'], 1)

    def test_loss_updates_only_starts_not_sequence_context(self):
        data = data_for()
        online = Table([[.1, .8], [.8, .1], [.8, .1], [.8, .1]])
        target = Table([[.25, -.5], [.5, -.25], [-.5, .75], [.125, .5]])
        before = online.values.detach().clone(); frozen = target.values.detach().clone()
        result = update_retrace_q(online, target, torch.optim.SGD(online.parameters(), lr=.1),
                                 data, device='cpu', epsilon=.2, target_batch=2)
        self.assertNotEqual(float(online.values[0, 0].detach()), float(before[0, 0]))
        torch.testing.assert_close(online.values[1:], before[1:], rtol=0, atol=0)
        self.assertEqual(float(online.values[0, 1].detach()), float(before[0, 1]))
        torch.testing.assert_close(target.values, frozen, rtol=0, atol=0)
        self.assertIsNone(target.values.grad)
        self.assertEqual(result['targets'], 1)
        self.assertEqual([size for grad, size in online.calls if grad], [1])

    def test_real_meaningful_branch_replay_with_low_overlap_differs_from_uncorrected_return(self):
        config = NFSPTaskConfig(task='branching', learner='tabular', purpose='profile',
            records=32, eval_every=16, batch=4, learning_start=4, capacity=32, time_cap_seconds=60)
        adapter = TaskAdapter(config); game = adapter.game
        root = game.root(BLACK); leaf = game.transition(root, 0, 0); terminal = game.transition(leaf, 1, 0)
        self.assertEqual(game.utility(terminal), -1.)
        replay = ReplayBuffer(8, adapter.spec)
        replay.add(game.observation(root, BLACK), 0, 0., game.observation(leaf, BLACK), False,
                   behavior_prob=.5, stream=0, step=0, opponent_version=2, policy_version=2)
        replay.add(game.observation(leaf, BLACK), 1, -1., game.observation(terminal, BLACK), True,
                   behavior_prob=.9, stream=0, step=1, opponent_version=2, policy_version=2)
        class FirstStart:
            def integers(self, low, high, size): return np.zeros(size, dtype=np.int64)
        data = replay.sample_sequences(1, 32, FirstStart(), q_only=True)
        values = np.zeros((adapter.spec.spatial_shape[0], 2))
        root_id = game.observation(root, BLACK).spatial.argmax()
        leaf_id = game.observation(leaf, BLACK).spatial.argmax()
        online = Table(values); target = Table(values)
        with torch.no_grad():
            online.values[leaf_id] = torch.tensor([.8, .1])
            target.values[leaf_id] = torch.tensor([.5, -.5])
            target.values[root_id, 0] = .1
        labels, _, details = retrace_targets(online, target, SequenceBatch.from_replay(data, 'cpu'), .02)
        expected = .49 + .95 * (.01 / .9) * (-.5)
        self.assertAlmostEqual(float(labels[0]), expected, places=6)
        self.assertGreater(details['trace_mass'][0], 1.)
        self.assertLess(details['trace_mass'][0], 1.02)
        self.assertGreater(abs(float(labels[0]) - game.utility(terminal)), 1.)

    def test_staleness_decodes_the_frozen_actor_policy_bit(self):
        trace = TraceDiagnostics()
        data = data_for((0,), (.5,))
        starts = {name: value[:, 0] for name, value in data.items()}
        starts['opponent_version'][0] = 7  # generation 3, average-policy bit 1.
        trace.observe(starts, {'trace_mass': np.array([1.]), 'sequence_records': np.array([1]),
                              'start_overlap': np.array([.5]), 'continuation_overlap_sum': np.array([0.])}, 1, 3)
        self.assertEqual(float(trace.sums[..., 3].sum()), 0.)
        starts['opponent_version'][0] = 9
        with self.assertRaises(ValueError):
            trace.observe(starts, {'trace_mass': np.array([1.]), 'sequence_records': np.array([1]),
                                  'start_overlap': np.array([.5]), 'continuation_overlap_sum': np.array([0.])}, 1, 3)

    def test_malformed_sequences_cannot_cross_boundaries_or_enter_ineligible_starts(self):
        for field, value in [('stream', 1), ('opponent_version', 1), ('policy_version', 1),
                             ('mode', 1), ('supplemental', True), ('step', 8),
                             ('ordinal', 0), ('terminated', True), ('value_mask', False),
                             ('behavior_prob', 0.)]:
            with self.subTest(field=field):
                data = data_for()
                data[field][0, 1] = value
                with self.assertRaises(ValueError): SequenceBatch.from_replay(data, 'cpu')
        data = data_for(); data['next_spatial'][0, 0] = 0
        with self.assertRaises(ValueError): SequenceBatch.from_replay(data, 'cpu')
        data = data_for(padding=2); data['valid'][0, 1] = False
        with self.assertRaises(ValueError): SequenceBatch.from_replay(data, 'cpu')
        for kwargs in ({'q_target':'unknown'}, {'trace_length':33}, {'trace_lambda':float('nan')},
                       {'trace_lambda':1.1}, {'target_batch':0}):
            with self.assertRaises(ValueError): NFSPConfig(**kwargs)


class SequenceRuntimeTests(unittest.TestCase):
    def setUp(self): torch.set_num_threads(1)

    def config(self, task='delayed'):
        return NFSPTaskConfig(task=task, learner='neural', purpose='profile', records=80,
            eval_every=16, batch=4, learning_start=4, capacity=128, target_every=8,
            q_reuse=4, q_target='retrace', target_batch=3, horizons=(5,), equivalent_actions=3,
            reservoir_open=0, handoff_end=8, epsilon_decay=40, hidden=8, width=8, blocks=1,
            norm_groups=2, time_cap_seconds=300)

    def test_task_exact_mid_episode_resume_and_context_does_not_increase_reuse(self):
        for task in ('delayed', 'branching'):
            with self.subTest(task=task), tempfile.TemporaryDirectory() as directory:
                config = self.config(task)
                full = NFSPTaskTrainer(config, Path(directory)/'full', reserve_bytes=0)
                expected = full.run(verbose=False)
                part = NFSPTaskTrainer(config, Path(directory)/'part', reserve_bytes=0)
                part.run(until_records=2, verbose=False)
                resumed = NFSPTaskTrainer(config, Path(directory)/'part', resume=True, reserve_bytes=0)
                actual = resumed.run(verbose=False)
                for name in ('q','target','average'):
                    for key, value in getattr(full.backend,name).state_dict().items():
                        torch.testing.assert_close(value,getattr(resumed.backend,name).state_dict()[key],rtol=0,atol=0)
                for name, rng in full.rng.items():
                    self.assertEqual(rng.bit_generator.state,resumed.rng[name].bit_generator.state)
                self.assertEqual(expected['counters'],actual['counters'])
                self.assertEqual(expected['trace_diagnostics'],actual['trace_diagnostics'])
                for a,b in zip(expected['evaluations'],actual['evaluations']): self.assertEqual(a['roots'],b['roots'])
                self.assertEqual(expected['trace_diagnostics']['loss_starts'],expected['counters']['q_loss_targets'])
                sums=np.asarray(expected['trace_diagnostics']['sums']).sum(axis=(0,1,2,3))
                self.assertGreater(sums[1],expected['counters']['q_loss_targets'])
                self.assertEqual(expected['counters']['target_refreshes'],expected['counters']['fresh_q']//8)
                self.assertEqual(expected['counters']['q_loss_targets'],expected['counters']['q_updates']*4)
                self.assertFalse(expected['formal_case_passed'])
                wrong=replace(config,q_target='one-step')
                with self.assertRaises(ValueError): NFSPTaskTrainer(wrong,Path(directory)/'part',resume=True,reserve_bytes=0)
                broken=deepcopy(full.backend.state_dict()); broken['trace_diagnostics']['counts'].zero_()
                with self.assertRaises(ValueError): full.backend.load_state_dict(broken)

    def test_augmented_board_runtime_exact_resume_with_sequence_diagnostics(self):
        config = BoardRuntimeConfig(rules=Rules(3,5),envs=4,batch=8,learning_start=8,capacity=64,
            target_every=16,width=8,blocks=1,norm_groups=2,reservoir_open=0,handoff_end=8,
            q_target='retrace',q_reuse=4,target_batch=4)
        full=BoardNFSPRuntime(config); expected=full.run_profile(64,time_cap_seconds=120)
        part=BoardNFSPRuntime(config); part.run_profile(24,time_cap_seconds=120)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint=part.save(directory,reserve_bytes=0)
            resumed=BoardNFSPRuntime(config); resumed.restore(checkpoint)
            actual=resumed.run_profile(64,time_cap_seconds=120)
        for name in ('q','target','average'):
            for key,value in getattr(full.backend,name).state_dict().items():
                torch.testing.assert_close(value,getattr(resumed.backend,name).state_dict()[key],rtol=0,atol=0)
        self.assertEqual(expected['counters'],actual['counters'])
        self.assertEqual(expected['trace_diagnostics'],actual['trace_diagnostics'])
        self.assertEqual(expected['metrics'],actual['metrics'])
        for name, rng in full.rng.items():
            self.assertEqual(rng.bit_generator.state,resumed.rng[name].bit_generator.state)
        self.assertEqual(expected['trace_diagnostics']['loss_starts'],expected['counters']['q_loss_targets'])

    def test_formal_retrace_recipe_never_satisfies_a_legacy_one_step_plan(self):
        original=NFSPTaskConfig(time_cap_seconds=3600,budget_profile_sha256='0'*64,q_reuse=4)
        self.assertTrue(original.formal_protocol());self.assertFalse(original.formal_retrace_protocol())
        candidate=replace(original,q_target='retrace')
        self.assertFalse(candidate.formal_protocol());self.assertTrue(candidate.formal_retrace_protocol())
        for changed in (replace(candidate,trace_length=16),replace(candidate,trace_lambda=.9),
                        replace(candidate,records=1000000),replace(candidate,reservoir_open=0)):
            self.assertFalse(changed.formal_protocol());self.assertFalse(changed.formal_retrace_protocol())

    def test_command_rejects_one_step_or_mismatched_trace_prices_before_creating_a_learner(self):
        import json
        from types import SimpleNamespace
        from graygo.cli import train_nfsp_task
        from graygo.manifest import source_digest
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)/'trial';price=Path(directory)/'price.json'
            budget={'source_sha256':source_digest(),'task':'delayed','learner':'neural','q_reuse':4,
                    'storage':'features','complete_pipeline':True,'phases':['pre','post'],'native_library_sha256':None}
            args=SimpleNamespace(budget_profile=str(price),task='delayed',learner='neural',q_reuse=4,
                                 storage='features',native_library=None,q_target='retrace',trace_length=32,
                                 trace_lambda=.95,target_batch=64,run_dir=str(root))
            price.write_text(json.dumps(budget))
            with self.assertRaisesRegex(ValueError,'different Q target'): train_nfsp_task(args)
            self.assertFalse(root.exists())
            budget.update(q_target='retrace',trace_length=32,trace_lambda=.9,target_batch=64)
            price.write_text(json.dumps(budget))
            with self.assertRaisesRegex(ValueError,'Retrace budget profile'): train_nfsp_task(args)
            self.assertFalse(root.exists())

    def test_real_full_shape_board_loss_has_finite_gradients_and_bounded_sequence_forwards(self):
        config=BoardRuntimeConfig(envs=1,batch=2,learning_start=4,capacity=16,target_every=8,
            q_target='retrace',target_batch=2,augmentation=True)
        runtime=BoardNFSPRuntime(config)
        result=runtime.run_profile(8,time_cap_seconds=120)
        self.assertEqual(result['status'],'complete')
        self.assertEqual(result['counters']['q_loss_targets'],6)
        self.assertEqual(result['counters']['target_refreshes'],1)
        self.assertEqual(result['trace_diagnostics']['loss_starts'],6)
        sums=np.asarray(result['trace_diagnostics']['sums']).sum(axis=(0,1,2,3))
        self.assertGreater(sums[1],6)
        self.assertTrue(all(bool(torch.isfinite(p).all()) for p in runtime.backend.q.parameters()))
        self.assertTrue(all(p.grad is None for p in runtime.backend.target.parameters()))
        self.assertFalse(result['formal_learning_run'])


if __name__ == '__main__': unittest.main()
