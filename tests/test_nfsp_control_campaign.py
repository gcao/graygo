from copy import deepcopy
from decimal import Decimal
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
from tools import nfsp_control_campaign as control
from tools.nfsp_resources import CPU_LIMITS

REPO = Path(__file__).resolve().parents[1]


class FixedControlTests(unittest.TestCase):
    def setUp(self):
        self.proposal = json.loads((REPO / 'docs/experiments/nfsp-control-completion-scope-proposal.json').read_text())

    def decision(self, **changes):
        text = 'Synthetic unit-test approval of exactly the named thirteen controls and 72 aggregate CPU hours.'
        value = {'version': control.DECISION_VERSION, 'approved': True,
                 'proposal_sha256': control.canonical(self.proposal),
                 'control_ceiling_seconds': self.proposal['proposed_control_ceiling_seconds'],
                 'case_ids': [case['id'] for case in self.proposal['cases']],
                 'decision_record': text, 'decision_record_sha256': hashlib.sha256(text.encode()).hexdigest()}
        value.update(changes); return value

    def allocation(self):
        return {'campaign_root': 'runs/fixed-controls', 'allocation_sha256': 'a' * 64,
                'cases': {case['id']: {'root': 'runs/fixed-controls/cases/' + case['id']} for case in self.proposal['cases']}}

    def identity(self):
        value = control.process_identity(os.getpid())
        return dict(value, process_group=os.getpid())

    def test_exact_full_original_thirteen_cases_and_original_defaults(self):
        configs = control.validate_cases(self.proposal)
        self.assertEqual(len(configs), 13)
        self.assertEqual(sum(c.records for c in configs.values()), 195000000)
        self.assertEqual(sum('/R1' in key for key in configs), 2)
        self.assertTrue(all(c.formal_protocol() and c.learner == 'tabular' for c in configs.values()))
        self.assertEqual(control.decision_limit(self.proposal, self.decision()), Decimal(259200))
        self.assertEqual(CPU_LIMITS['control'], Decimal(43200))
        projected = Decimal(self.proposal['current_control_commitment_seconds']) + Decimal(213300) + control.CONTROLLER_RESERVE
        self.assertLess(projected, Decimal(259200))

    def test_filtered_replaced_shortened_or_changed_controls_are_rejected(self):
        for key, value in [('records', 5000000), ('learner', 'neural'), ('q_reuse', 1), ('batch', 128),
                           ('capacity', 10000), ('target_every', 3200), ('device', 'cuda'),
                           ('uniform_comparison', True), ('time_cap_seconds', 19800), ('seed', 3)]:
            proposal = deepcopy(self.proposal); proposal['cases'][0]['config'][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                control.validate_cases(proposal)
        for cases in (self.proposal['cases'][:-1], self.proposal['cases'][:-1] + [self.proposal['cases'][0]]):
            proposal = deepcopy(self.proposal); proposal['cases'] = cases
            with self.assertRaises(ValueError): control.validate_cases(proposal)

    def test_missing_wrong_or_reinterpreted_owner_approval_cannot_raise_ceiling(self):
        values = [None, self.decision(approved=False), self.decision(proposal_sha256='c' * 64),
                  self.decision(control_ceiling_seconds='300000'), self.decision(case_ids=[]),
                  self.decision(version='graygo/nfsp-development-scope/1'),
                  self.decision(decision_record='different owner answer')]
        for decision in values:
            with self.subTest(decision=decision), self.assertRaises(ValueError):
                control.decision_limit(self.proposal, decision)

    def test_actual_cli_rejects_before_torch_locks_or_model_directories_without_approval(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); proposal_path = root / 'proposal.json'; proposal_path.write_text(json.dumps(self.proposal))
            allocation = root / 'allocation.json'; allocation.write_text(json.dumps({'proposal': 'proposal.json', 'cases': {}}))
            decision = root / 'decision.json'; decision.write_text(json.dumps(self.decision(approved=False)))
            before = {p.name: p.read_bytes() for p in root.iterdir()}
            command = [sys.executable, '-S', str(REPO / 'tools/nfsp_control_campaign.py'), 'run-case',
                       '--repo', str(root), '--allocation', str(allocation), '--allocation-sha', control.sha(allocation),
                       '--decision', str(decision), '--decision-sha', control.sha(decision),
                       '--case-id', self.proposal['cases'][0]['id']]
            result = subprocess.run(command, text=True, capture_output=True, timeout=10)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('explicit matching owner approval', result.stderr)
            self.assertEqual({p.name: p.read_bytes() for p in root.iterdir()}, before)
            self.assertFalse((root / 'runs').exists())

    def test_claim_reserves_every_cap_and_blocks_duplicate_campaigns_and_workers(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = self.allocation(); first = self.proposal['cases'][0]['id']; second = self.proposal['cases'][1]['id']
            with patch.object(control, 'process_identity', return_value=self.identity()):
                root, identity = control.claim_case(tmp, a, self.proposal, self.decision(), 'b' * 64, first)
                state = control.load_campaign(root / 'campaign.json')
                self.assertEqual(len(state['cases']), 13)
                self.assertEqual(state['cases'][first]['original_worker'], identity)
                self.assertEqual(sum(Decimal(row['complete_process_ceiling_seconds']) for row in state['cases'].values()), Decimal(213300))
                with self.assertRaises(ValueError):control.claim_case(tmp, a, self.proposal, self.decision(), 'b' * 64, first)
                with self.assertRaises(ValueError):control.claim_case(tmp, a, self.proposal, self.decision(), 'b' * 64, second)
                clone = deepcopy(a); clone.update(campaign_root='runs/duplicate', allocation_sha256='c' * 64)
                with self.assertRaises(ValueError):control.claim_case(tmp, clone, self.proposal, self.decision(), 'b' * 64, second)
            self.assertFalse(any((Path(tmp) / entry['root']).exists() for entry in a['cases'].values()))

    def test_missing_approval_creates_no_campaign_and_corrupt_ledger_cannot_reset_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = self.allocation(); identifier = self.proposal['cases'][0]['id']
            with self.assertRaises(ValueError):control.claim_case(tmp, a, self.proposal, None, 'b' * 64, identifier)
            self.assertFalse((Path(tmp) / 'runs').exists())
            with patch.object(control, 'process_identity', return_value=self.identity()):
                root, _ = control.claim_case(tmp, a, self.proposal, self.decision(), 'b' * 64, identifier)
            path = root / 'campaign.json'; wrapper = control.load(path)
            wrapper['body']['cases'][identifier]['status'] = 'reserved'; path.write_text(json.dumps(wrapper))
            with self.assertRaises(ValueError):control.load_campaign(path)

    def test_settlement_retains_actual_overrun_and_prevents_further_launches(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = self.allocation(); identifier = self.proposal['cases'][0]['id']
            with patch.object(control, 'process_identity', return_value=self.identity()):
                root, identity = control.claim_case(tmp, a, self.proposal, self.decision(), 'b' * 64, identifier)
            cap = float(self.proposal['cases'][0]['complete_process_ceiling_seconds'])
            control.write(root / 'guards' / (control.canonical(identifier) + '.json'),
                {'status': 'original_worker_terminal', 'original_worker': identity,
                 'observed_process_elapsed_upper_seconds': cap + 2, 'hard_budget_stop': False, 'active_monitor_seconds': .1})
            with self.assertRaises(ValueError):control.settle_case(tmp, a, identifier, 'b' * 64)
            state = control.load_campaign(root / 'campaign.json')
            self.assertEqual(state['cases'][identifier]['measured_complete_process_upper_seconds'], cap + 2)
            self.assertTrue(state['budget_violations'])
            with patch.object(control, 'process_identity', return_value=self.identity()), self.assertRaises(ValueError):
                control.claim_case(tmp, a, self.proposal, self.decision(), 'b' * 64, self.proposal['cases'][1]['id'])

    def test_failed_atomic_write_preserves_previous_data_and_removes_only_its_temporary_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'ledger.json'; control.write(path, {'old': True}); before = path.read_bytes()
            with patch.object(control.os, 'fsync', side_effect=OSError('simulated failed save')):
                with self.assertRaises(OSError):control.write(path, {'new': True})
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual([p.name for p in Path(tmp).iterdir()], ['ledger.json'])


class IndependentWatchdogTests(unittest.TestCase):
    def victim(self):
        return subprocess.Popen([sys.executable, '-S', '-c', 'import signal,time; signal.signal(signal.SIGUSR1,signal.SIG_IGN); time.sleep(60)'], start_new_session=True)

    def ready_identity(self, child):
        for _ in range(100):
            identity = control.process_identity(child.pid)
            if identity is not None and identity['cmdline_sha256'] != hashlib.sha256(b'').hexdigest():return identity
            time.sleep(.01)
        self.fail('own test child did not exec')

    def test_owned_original_process_receives_graceful_signal_then_hard_deadline(self):
        child = self.victim()
        try:
            identity = self.ready_identity(child)
            with tempfile.TemporaryDirectory() as tmp:
                state = control.guard(identity, 1.8, Path(tmp) / 'guard.json', graceful_headroom=1., hard_headroom=.6, interval=.01)
                self.assertEqual(state['status'], 'original_worker_terminal')
                self.assertTrue(state['hard_budget_stop']); self.assertTrue(state['graceful_budget_signal'])
                self.assertLessEqual(state['observed_process_elapsed_upper_seconds'], 1.8)
            self.assertEqual(child.wait(timeout=5), -9)
        finally:
            if child.poll() is None:child.kill(); child.wait()

    def test_changed_identity_or_reset_birth_is_never_signalled(self):
        child = self.victim()
        try:
            identity = self.ready_identity(child)
            with tempfile.TemporaryDirectory() as tmp:
                wrong = dict(identity, cmdline_sha256='f' * 64)
                state = control.guard(wrong, 2, Path(tmp) / 'guard.json', graceful_headroom=1., hard_headroom=.5, interval=.01)
                self.assertEqual(state['status'], 'original_identity_changed_no_signal')
                self.assertIsNone(child.poll())
                wrong = dict(identity, birth_boottime_seconds=identity['birth_boottime_seconds'] + 100)
                with self.assertRaises(ValueError):
                    control.guard(wrong, 2, Path(tmp) / 'other.json', graceful_headroom=1., hard_headroom=.5)
        finally:
            if child.poll() is None:child.kill(); child.wait()

    def test_lost_logging_does_not_disable_the_deadline(self):
        child = self.victim()
        try:
            identity = self.ready_identity(child)
            with patch.object(control, 'write', side_effect=OSError('simulated full log filesystem')):
                state = control.guard(identity, 1.8, '/unused', graceful_headroom=1., hard_headroom=.6, interval=.01)
            self.assertTrue(state['hard_budget_stop']); self.assertIn('last_logging_error', state)
            self.assertEqual(child.wait(timeout=5), -9)
        finally:
            if child.poll() is None:child.kill(); child.wait()

    def test_standalone_stdlib_watchdog_survives_its_launcher_exit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); module = REPO / 'tools/nfsp_control_campaign.py'
            launcher = root / 'launcher.py'
            launcher.write_text('''import importlib.util,json,os,subprocess,sys,time
from pathlib import Path
module=Path(sys.argv[1]);root=Path(sys.argv[2])
spec=importlib.util.spec_from_file_location("guard_module",module);control=importlib.util.module_from_spec(spec);spec.loader.exec_module(control)
victim=subprocess.Popen([sys.executable,"-S","-c","import signal,sys,time; signal.signal(signal.SIGUSR1,lambda s,f:sys.exit(0)); time.sleep(60)"],start_new_session=True)
for attempt in range(100):
 identity=control.process_identity(victim.pid)
 if identity and identity["cmdline_sha256"]!=control.hashlib.sha256(b"").hexdigest():break
 time.sleep(.01)
control.write(root/"identity.json",identity)
guard=subprocess.Popen([sys.executable,"-S",str(module),"guard","--identity",str(root/"identity.json"),"--cap","31","--output",str(root/"guard.json")],start_new_session=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
for attempt in range(100):
 guard_identity=control.process_identity(guard.pid)
 if guard_identity and guard_identity["cmdline_sha256"]!=control.hashlib.sha256(b"").hexdigest():break
 time.sleep(.01)
control.write(root/"owned-pids.json",{"victim":identity,"guard":guard_identity})
os._exit(0)
''')
            result = subprocess.run([sys.executable, '-S', str(launcher), str(module), str(root)], timeout=5)
            self.assertEqual(result.returncode, 0)
            owned = control.load(root / 'owned-pids.json')
            try:
                deadline = time.monotonic() + 8
                state = None
                while time.monotonic() < deadline:
                    if (root / 'guard.json').exists():
                        state = control.load(root / 'guard.json')
                        if state['status'] == 'original_worker_terminal':break
                    time.sleep(.05)
                self.assertIsNotNone(state)
                self.assertEqual(state['status'], 'original_worker_terminal')
                self.assertTrue(state['graceful_budget_signal'])
                self.assertEqual(state['original_worker'], owned['victim'])
                self.assertIsNone(control.process_identity(owned['victim']['pid']))
            finally:
                for expected in (owned['victim'], owned['guard']):
                    pid = expected['pid']
                    current = control.process_identity(pid)
                    if current is not None and current['process_group'] == pid and current['start_ticks'] == expected['start_ticks']:
                        os.killpg(pid, 9)


if __name__ == '__main__':
    unittest.main()
