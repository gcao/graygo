import json
import os
from pathlib import Path
import sqlite3
import tempfile
import unittest
from tools.training_benchmarks import benchmark_fields, latest_benchmark
from tools.log_training_status import summary
from tools.adaptive_gate_schedule import initial_state


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def evidence(path, utilities):
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as db:
        db.execute('CREATE TABLE units(idx INTEGER, payload TEXT)')
        for index in range(0, len(utilities), 2):
            db.execute('INSERT INTO units VALUES (?,?)', (index//2, json.dumps({
                'subject': [{'subject_utility': value} for value in utilities[index:index+2]]})))


class BenchmarkLogTests(unittest.TestCase):
    def test_cached_gate_from_previous_segment_names_actual_opponent_and_age(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)/'stages/000000'
            league = root/'learner/league'
            write(league/'state.json', {'body': {'rounds': [{'index': 0, 'records': 100,
                  'incumbent': {'label': 'old-champion'}}], 'history': []}})
            gate = league/'evaluations/0000'
            write(gate/'gate-plan.json', {'pool': {'random': 'old-random', 'historical-0': 'old'}})
            evidence(gate/'head/evidence.sqlite', [1, 1, 0, -1])
            evidence(gate/'random/evidence.sqlite', [1, 1, 1, 1])
            os.utime(gate/'head/evidence.sqlite', (1000, 1000))
            fields = benchmark_fields({'campaign_dir': str(root.parents[1])}, root.parent/'000001', 500, now=8200)
            text = ' | '.join(fields)
            self.assertIn('benchmark_records=100', text)
            self.assertIn('benchmark_age=2.0h lag_records=400', text)
            self.assertIn('average_vs_champion(old-champion)=2W/1D/1L', text)
            self.assertNotIn('random', text)
            self.assertNotIn('tactical', text)

    def test_newer_screen_takes_precedence_over_old_gate_and_shows_histories(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            league = root/'learner/league'
            report = {'games': 4, 'wins': 3, 'draws': 0, 'losses': 1, 'win_score': .75,
                      'opponent_label': 'champion-100','opponent_sha256':'known-100'}
            write(league/'state.json',{'body':{'history':[{'policy_sha256':'known-50','label':'champion-50'}]}})
            path = league/'screens/0001/champion-first-v2/screen.json'
            write(path, {'candidate_records': 200, 'status': 'complete', 'completed_at':'2026-11-06T00:00:00Z', 'heads': {'average': {
                'opponents': {'champion': report, 'historical-0': dict(report, opponent_label='historical-0',opponent_sha256='known-50')}}}})
            actual = latest_benchmark([root])
            self.assertEqual(actual['records'], 200)
            fields = ' | '.join(benchmark_fields({}, root, 400))
            self.assertIn('source=screen status=complete', fields)
            self.assertIn('average_vs_historical-0(champion-50)', fields)
            self.assertIn('lag_records=200', fields)

    def test_hourly_summary_keeps_champion_even_when_no_new_benchmark_exists(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory); root = project/'run'
            write(project/'runs/latest-training-loop.json', {'run_dir': str(root), 'unlimited': True})
            write(root/'status.json', {'status': 'running'})
            write(root/'learner/league/state.json', {'body': {'prior_promotions': 1}})
            write(root/'learner/league/champion.json', {'body': {'champion': {'label': 'champion-100'}}})
            write(root/'learner/league/screens/0000/screen.json', {
                'heads': {'average': {'opponents': {'random': {'games': 20}}}}})
            text = ' | '.join(summary(project))
            self.assertIn('promotions=1 champion=champion-100', text)
            self.assertIn('benchmark=not_measured_against_learned_opponents', text)
            self.assertNotIn('average_vs_random', text)

    def test_hourly_log_exposes_adaptive_timer_without_charging_evaluation_as_training(self):
        with tempfile.TemporaryDirectory() as directory:
            project=Path(directory);root=project/'run'
            write(project/'runs/latest-training-loop.json',{'run_dir':str(root),'unlimited':True})
            write(root/'status.json',{'status':'running'})
            write(root/'learner/execution.json',{'body':{'clock':{'observed_seconds':10000.,'recovery_charge_seconds':0.}}})
            write(root/'learner/league/state.json',{'body':{'evaluation_seconds':2000.,
                'adaptive_scheduler':initial_state(8000.)}})
            text=' | '.join(summary(project))
            self.assertIn('gate_schedule=adaptive interval=60m',text)
            self.assertIn('screen_due_in=0m gate_timer_in=60m',text)
            self.assertIn('trigger=strong_screen_or_timer',text)


if __name__ == '__main__':
    unittest.main()
