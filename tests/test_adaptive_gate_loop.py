from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch, PropertyMock
import torch
from graygo.board_execution import BoardExecutor, BoardExecutionPlan
from graygo.board_journal import BoardJournalPlan
from graygo.confidence import canonical_sha
from graygo.nfsp import NFSPBackend
from tools.champion_gate import GatePlan
from tools.champion_loop import ChampionLoop, LoopPlan
from tools.champion_screen import ScreenPlan
from tools import adaptive_gate_schedule as adaptive
from test_continuous_training import config
from test_learning_loop import exact, normalized


class Screen:
    plan = ScreenPlan(4, 30)
    sha256 = canonical_sha({'fixture': 'adaptive-screen'})

    def __init__(self):
        self.calls = 0

    def evaluate(self, archive, **kwargs):
        self.calls += 1
        return {'status': 'complete', 'average_eligible': True,
                'candidate_records': kwargs['candidate_records'], 'heads': {'average': {'opponents': {
                    name: {'games': 20, 'win_score': .9, 'status': 'complete', 'opponent_sha256': opponent.sha256}
                    for name, opponent in kwargs['reference_opponents'].items()}}}}


class Gate:
    plan = GatePlan(1, (4,))
    sha256 = canonical_sha({'fixture': 'adaptive-gate'})
    protected_authorizer = None

    def __init__(self, executor, *, deferred=False):
        self.executor = executor
        self.deferred = deferred
        self.calls = []

    def evaluate(self, candidate, champion, pool, root, **kwargs):
        self.calls.append((candidate.sha256, champion.sha256, tuple(pool)))
        self.executor.base_observed += 1000
        if self.deferred and len(self.calls) == 1:
            return {'status': 'budget_limited', 'numeric_pass': False, 'promotion_permitted': False}
        return {'status': 'complete', 'numeric_pass': False, 'promotion_permitted': False,
                'experimental_promotion_permitted': False, 'fixture_only': True}


class AdaptiveLoopTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def execute(self, root):
        return BoardExecutor(root, config(), BoardJournalPlan('loop', 16, (0, 16)),
                             BoardExecutionPlan(None), reserve_bytes=0)

    def test_strong_candidate_runs_full_gate_despite_old_evaluation_debt(self):
        with tempfile.TemporaryDirectory() as directory, self.execute(directory) as executor:
            executor.base_observed += 60000
            gate, screen = Gate(executor), Screen()
            with ChampionLoop(executor, LoopPlan((16,), adaptive_schedule=True), gate, screen=screen) as loop:
                loop.state['evaluation_seconds'] = 20000
                loop.state['adaptive_scheduler'] = adaptive.initial_state(loop._learning_seconds())
                before = deepcopy(executor.runtime.state_dict())
                candidate = {'policy_sha256': 'candidate', 'weights_sha256': 'new'}
                pending = {'index': 0, 'target': 16, 'status': 'training', 'history': [], 'champion': loop.state['champion']}
                self.assertEqual(loop._evaluation_remaining(), 0)
                with patch('tools.champion_loop.policy', side_effect=lambda descriptor: SimpleNamespace(sha256=descriptor['policy_sha256'])), patch.object(NFSPBackend, 'average_ready', new_callable=PropertyMock, return_value=True):
                    result = loop._evaluate('fixture', candidate, pending, lambda: False)
                self.assertEqual(result['status'], 'complete')
                self.assertEqual(len(gate.calls), 1)
                self.assertEqual(screen.calls, 1)
                self.assertEqual(loop.state['adaptive_scheduler']['last_signal'], 'gate_not_passed')
                self.assertGreater(loop.state['adaptive_scheduler']['next_gate_learning_seconds']-loop._learning_seconds(), 3590)
                exact(normalized(before), normalized(executor.runtime.state_dict()))

    def test_inconclusive_gate_resumes_same_candidate_without_screen_or_training_reset(self):
        with tempfile.TemporaryDirectory() as directory, self.execute(directory) as executor:
            gate, screen = Gate(executor, deferred=True), Screen()
            with ChampionLoop(executor, LoopPlan((16,), adaptive_schedule=True), gate, screen=screen) as loop:
                candidate = {'policy_sha256': 'candidate', 'weights_sha256': 'new'}
                pending = {'index': 0, 'target': 16, 'status': 'training', 'history': [], 'champion': loop.state['champion']}
                with patch('tools.champion_loop.policy', side_effect=lambda descriptor: SimpleNamespace(sha256=descriptor['policy_sha256'])), patch.object(NFSPBackend, 'average_ready', new_callable=PropertyMock, return_value=True):
                    first = loop._evaluate('fixture', candidate, pending, lambda: False)
                    self.assertEqual(first['status'], 'gate_pending')
                    saved_candidate = deepcopy(pending['candidate'])
                    second = loop._evaluate('fixture', candidate, pending, lambda: False)
                self.assertEqual(second['status'], 'complete')
                self.assertEqual(screen.calls, 1)
                self.assertEqual(gate.calls[0], gate.calls[1])
                self.assertEqual(saved_candidate, pending['candidate'])
                self.assertEqual(loop.state['adaptive_scheduler']['gate_count'], 1)
                self.assertEqual(executor.runtime.records, 0)

    def test_waiting_interval_skips_model_loading_and_full_gate(self):
        with tempfile.TemporaryDirectory() as directory, self.execute(directory) as executor:
            gate, screen = Gate(executor), Screen()
            with ChampionLoop(executor, LoopPlan((16,), adaptive_schedule=True), gate, screen=screen) as loop:
                loop.state['adaptive_scheduler']['next_screen_learning_seconds'] = loop._learning_seconds()+3600
                candidate = {'policy_sha256': 'candidate', 'weights_sha256': 'new'}
                pending = {'index': 0, 'target': 16, 'status': 'training', 'history': [], 'champion': loop.state['champion']}
                with patch.object(NFSPBackend, 'average_ready', new_callable=PropertyMock, return_value=True):
                    result = loop._evaluate('fixture', candidate, pending, lambda: False)
                self.assertEqual(result['reason'], 'adaptive_training_interval')
                self.assertEqual((screen.calls, len(gate.calls)), (0, 0))


if __name__ == '__main__':
    unittest.main()
