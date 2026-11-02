from dataclasses import replace
from contextlib import redirect_stderr
import io
import json
from pathlib import Path
import tempfile
import subprocess
import sys
import unittest
from unittest.mock import patch
import numpy as np
import torch
from graygo.board_execution import BoardExecutor, BoardExecutionPlan
from graygo.board_journal import BoardJournalPlan
from graygo.board_runtime import BoardRuntimeConfig
from graygo.confidence import canonical_sha
from graygo.engine import Rules
from graygo.storage import CheckpointStore, CorruptCheckpoint, DiskBudgetError


class FakeClock:
    def __init__(self):
        self.now = 0.
    def monotonic(self):
        return self.now
    def wall(self):
        return 1800000000. + self.now


def rewrite_clock(root, change):
    path = Path(root) / 'execution.json'
    wrapper = json.loads(path.read_text())
    change(wrapper['body']['clock'])
    wrapper['sha256'] = canonical_sha(wrapper['body'])
    path.write_text(json.dumps(wrapper))


def exact(a, b):
    if isinstance(a, torch.Tensor):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            exact(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert type(a) == type(b) and len(a) == len(b)
        for x, y in zip(a, b):
            exact(x, y)
    else:
        assert a == b


class BoardExecutionTests(unittest.TestCase):
    def config(self, device='cpu'):
        return BoardRuntimeConfig(rules=Rules(3, 5), envs=4, seed=5, batch=8, learning_start=8,
                                  capacity=256, target_every=16, width=8, blocks=1, norm_groups=2,
                                  reservoir_open=64, handoff_end=96, device=device)
    def journal(self):
        return BoardJournalPlan('diagnostic', 256, (0, 32, 48, 64, 96, 128, 256), 83)
    def plan(self):
        return BoardExecutionPlan(1000., checkpoint_every=32, heartbeat_seconds=10)
    def executor(self, path, *, resume=False, clock=None, device='cpu', plan=None, config=None):
        timing = {} if clock is None else {'monotonic': clock.monotonic, 'wall_time': clock.wall}
        return BoardExecutor(path, config or self.config(device), self.journal(), plan or self.plan(),
                             reserve_bytes=0, resume=resume, **timing)

    def test_exact_chunk_boundaries_opening_pause_and_retention(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.executor(directory) as run:
                result = run.run()
                self.assertEqual((result['status'], result['records']), ('opening_pending', 64))
                self.assertEqual(run.runtime.reservoir.seen, 0)
                self.assertEqual(result['archived_records'], [0, 32, 48, 64])
                self.assertFalse(result['formal_scope_approved'])
                run.journal.diagnostic_opening('pass')
                self.assertEqual(run.run()['status'], 'complete')
                self.assertGreater(run.runtime.reservoir.seen, 0)
                self.assertTrue((run.runtime.reservoir.arrays['record'][:run.runtime.reservoir.size] >= 64).all())
                self.assertEqual(len(list((Path(directory) / 'resume').glob('[0-9]*'))), 1)
                self.assertEqual(len(list((Path(directory) / 'weights').glob('[0-9]*'))), 7)
            clock = json.loads((Path(directory) / 'execution.json').read_text())['body']['clock']
            self.assertIsNone(clock['inflight_since'])
            self.assertEqual(clock['last_checkpoint']['records'], 256)

    def continuation(self, device):
        with tempfile.TemporaryDirectory() as directory:
            with self.executor(Path(directory) / 'full', device=device) as full:
                full.run()
                full.journal.diagnostic_opening('pass')
                full.run()
                expected = full.runtime.state_dict()
                expected.pop('timings')
                arrays = {name: {key: value.copy() for key, value in buffer.arrays.items()}
                          for name, buffer in (('replay', full.runtime.replay), ('reservoir', full.runtime.reservoir))}
            root = Path(directory) / 'part'
            with self.executor(root, device=device) as part:
                self.assertEqual(part.run(32)['records'], 32)
            with self.executor(root, resume=True, device=device) as part:
                self.assertEqual(part.runtime.records, 32)
                self.assertEqual(part.run()['status'], 'opening_pending')
                part.journal.diagnostic_opening('pass')
                # Exit before another heavy checkpoint: opening overlay must
                # survive while the clock still points at the pending bundle.
            with self.executor(root, resume=True, device=device) as part:
                self.assertTrue(part.journal.label_insertion_allowed())
                self.assertEqual(part.run()['status'], 'complete')
                actual = part.runtime.state_dict()
                actual.pop('timings')
                exact(expected, actual)
                for name in arrays:
                    for key, value in arrays[name].items():
                        np.testing.assert_array_equal(value, getattr(part.runtime, name).arrays[key])

    def test_exact_complete_cpu_resume_across_opening_and_clock_sessions(self):
        self.continuation('cpu')

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_exact_complete_cuda_resume_across_opening_and_clock_sessions(self):
        self.continuation('cuda')

    def test_execution_lock_covers_initialization_running_and_pending_decision(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.executor(directory) as run:
                with self.assertRaises(BlockingIOError):
                    with self.executor(directory, resume=True):
                        pass
                run.run()
                with self.assertRaises(BlockingIOError):
                    with self.executor(directory, resume=True):
                        pass
            with self.executor(directory, resume=True) as run:
                self.assertEqual(run.runtime.records, 64)

    def test_cumulative_time_cap_cannot_be_refreshed_by_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            clock = FakeClock()
            plan = replace(self.plan(), time_ceiling_seconds=3.)
            with self.executor(directory, clock=clock, plan=plan) as run:
                def elapsed():
                    if run.runtime.records >= 16:
                        clock.now = 3.
                    return False
                result = run.run(stop_requested=elapsed)
                self.assertEqual((result['status'], result['records']), ('budget_limited', 16))
            clock.now = 100.
            with self.executor(directory, resume=True, clock=clock, plan=plan) as run:
                self.assertEqual(run.run()['records'], 16)
                self.assertEqual(run.clock['recovery_charge_seconds'], 0.)
            original = (Path(directory) / 'execution.json').read_bytes()
            with self.assertRaisesRegex(ValueError, 'plan'):
                with self.executor(directory, resume=True, clock=clock, plan=replace(plan, time_ceiling_seconds=4.)):
                    pass
            self.assertEqual((Path(directory) / 'execution.json').read_bytes(), original)

    def test_unclean_gap_and_clock_reversal_conservatively_spend_remaining_budget(self):
        for delta in (-100., 100.):
            with self.subTest(delta=delta), tempfile.TemporaryDirectory() as directory:
                clock = FakeClock()
                with self.executor(directory, clock=clock) as run:
                    run.run(32)
                rewrite_clock(directory, lambda c: c.update(inflight_since=clock.wall() - delta))
                with self.executor(directory, resume=True, clock=clock) as run:
                    self.assertEqual(run.clock['unclean_recoveries'], 1)
                    self.assertEqual(run.clock['recovery_charge_seconds'], 1000. if delta < 0 else 100.)
                    if delta < 0:
                        self.assertEqual(run.run()['status'], 'budget_limited')

    def test_partial_budget_signal_stop_is_not_terminal_and_preserves_two_generations(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.executor(directory) as run:
                result = run.run(stop_requested=lambda: run.runtime.records >= 32)
                self.assertEqual((result['status'], result['records']), ('interrupted', 32))
                self.assertEqual(len(list((Path(directory) / 'resume').glob('[0-9]*'))), 2)
            with self.executor(directory, resume=True) as run:
                self.assertEqual(run.run()['status'], 'opening_pending')

    def test_no_go_cannot_be_replaced_or_resume_into_label_collection(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.executor(directory) as run:
                run.run()
                run.journal.diagnostic_opening('no_go')
                self.assertEqual(run.run()['status'], 'opening_no_go')
            with self.executor(directory, resume=True) as run:
                self.assertEqual(run.run()['status'], 'opening_no_go')
                self.assertEqual(run.runtime.reservoir.seen, 0)
                with self.assertRaisesRegex(ValueError, 'immutable'):
                    run.journal.diagnostic_opening('pass')

    def test_bundle_commit_before_clock_write_is_adopted_without_clock_rollback(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.executor(directory) as run:
                original = json.loads((Path(directory) / 'execution.json').read_text())
                run.run(32)
            # Simulate death after the next committed bundle but before the
            # separate execution clock update; record zero is still retained.
            original['body']['clock']['inflight_since'] = 1800000000.
            original['sha256'] = canonical_sha(original['body'])
            (Path(directory) / 'execution.json').write_text(json.dumps(original))
            with self.executor(directory, resume=True) as run:
                self.assertEqual(run.runtime.records, 32)
                self.assertEqual(run.clock['last_checkpoint']['records'], 32)

    def test_clock_source_and_bundle_corruption_reject_without_resetting_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.executor(directory) as run:
                run.run(32)
            path = Path(directory) / 'execution.json'
            original = path.read_bytes()
            wrapper = json.loads(original)
            wrapper['body']['clock']['observed_seconds'] = -1
            path.write_text(json.dumps(wrapper))
            rejected = path.read_bytes()
            with self.assertRaisesRegex(ValueError, 'checksum'):
                with self.executor(directory, resume=True):
                    pass
            self.assertEqual(path.read_bytes(), rejected)
            path.write_bytes(original)
            with patch('graygo.board_execution.source_digest', return_value='0' * 64):
                with self.assertRaisesRegex(ValueError, 'source'):
                    with self.executor(directory, resume=True):
                        pass
            self.assertEqual(path.read_bytes(), original)
            bundle = CheckpointStore(directory, reserve_bytes=0).latest()
            (bundle / 'controller.json').write_text('{}')
            with self.assertRaises(CorruptCheckpoint):
                with self.executor(directory, resume=True):
                    pass

    def test_unexpected_failure_keeps_original_durable_state_and_charges_work(self):
        with tempfile.TemporaryDirectory() as directory:
            clock = FakeClock()
            with self.assertRaisesRegex(RuntimeError, 'numerical failure'):
                with self.executor(directory, clock=clock) as run:
                    run.run(32)
                    clock.now += 12.
                    with patch.object(run.runtime, 'advance', side_effect=RuntimeError('numerical failure')):
                        run.run()
            with self.executor(directory, resume=True, clock=clock) as run:
                self.assertEqual(run.runtime.records, 32)
                self.assertGreaterEqual(run.clock['observed_seconds'], 12.)

    def test_storage_stop_retains_durable_bundle_and_does_not_lower_reserve(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.executor(directory) as run:
                run.run(32)
                expected = run.clock['last_checkpoint'].copy()
                with patch.object(run.store, 'ensure_space', side_effect=DiskBudgetError('disk reserve')):
                    result = run.run()
                self.assertEqual(result['status'], 'storage_limited')
                self.assertEqual(result['last_durable_checkpoint'], expected)
                self.assertEqual(result['unsaved_records'], 0)
                self.assertEqual(run.contract['reserve_bytes'], 0)

    def test_missing_clock_or_manifest_cannot_reset_allocation_or_create_new_run(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.executor(directory) as run:
                run.run(32)
            manifest = Path(directory) / 'run.json'
            saved = manifest.read_bytes()
            manifest.unlink()
            with self.assertRaisesRegex(ValueError, 'manifest is missing'):
                with self.executor(directory, resume=True):
                    pass
            self.assertFalse(manifest.exists())
            manifest.write_bytes(saved)
            (Path(directory) / 'execution.json').unlink()
            for resume in (False, True):
                with self.assertRaisesRegex(ValueError, 'clock is missing'):
                    with self.executor(directory, resume=resume):
                        pass

    def test_plan_limits_and_context_are_required(self):
        for value in (0, -1, float('inf'), float('nan'), True):
            with self.assertRaises(ValueError):
                BoardExecutionPlan(value)
        with self.assertRaises(ValueError):
            BoardExecutionPlan(1., checkpoint_every=3)
        with self.assertRaises(ValueError):
            BoardExecutionPlan(1., heartbeat_seconds=61)
        with tempfile.TemporaryDirectory() as directory:
            run = self.executor(directory)
            with self.assertRaisesRegex(RuntimeError, 'context'):
                run.run()
            with self.assertRaises(ValueError):
                self.executor(directory, plan=replace(self.plan(), time_ceiling_seconds=3601))

    def test_cpu_identity_does_not_create_an_accelerator_context(self):
        check = subprocess.run([sys.executable, '-c',
            'import torch; from graygo.board_execution import _runtime_identity; '
            'from graygo.board_runtime import BoardRuntimeConfig; '
            'assert not torch.cuda.is_initialized(); _runtime_identity(BoardRuntimeConfig()); '
            'assert not torch.cuda.is_initialized()'], capture_output=True, text=True)
        self.assertEqual(check.returncode, 0, check.stderr)

    def test_pre_entry_setup_is_charged_again_on_every_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            clock = FakeClock()
            plan = replace(self.plan(), time_ceiling_seconds=3.)
            arguments = dict(reserve_bytes=0, monotonic=clock.monotonic, wall_time=clock.wall)
            with BoardExecutor(directory, self.config(), self.journal(), plan, prior_setup_seconds=2., **arguments) as run:
                run.run(32)
            with BoardExecutor(directory, self.config(), self.journal(), plan, resume=True,
                               prior_setup_seconds=1., **arguments) as run:
                self.assertEqual((run.run()['status'], run.runtime.records), ('budget_limited', 32))
                self.assertEqual(run.clock['observed_seconds'], 3.)

    def test_public_command_rejects_formal_sized_work_before_creating_a_run(self):
        from graygo.cli import main
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'not-created'
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                main(['run-board-diagnostic', '--run-dir', str(root), '--records', '80000000',
                      '--time-cap-hours', '.01'])
            self.assertEqual(error.exception.code, 1)
            self.assertFalse(root.exists())

    def test_full_capacity_write_headroom_cannot_be_lowered_by_a_small_declaration(self):
        with tempfile.TemporaryDirectory() as directory:
            plan = replace(self.plan(), checkpoint_headroom_bytes=65536)
            with self.executor(directory, plan=plan) as run:
                self.assertGreater(run.write_headroom, plan.checkpoint_headroom_bytes)
                self.assertGreater(run.write_headroom, sum(a.nbytes for b in (run.runtime.replay, run.runtime.reservoir)
                                                           for a in b.arrays.values()))


if __name__ == '__main__':
    unittest.main()
