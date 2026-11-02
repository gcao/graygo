import json
import subprocess
import sys
import tempfile
from pathlib import Path
import unittest


class CommandTests(unittest.TestCase):
    def run_cli(self,*args):
        return subprocess.run([sys.executable,"-m","graygo",*args],capture_output=True,text=True)

    def test_simulate_reproducible_results_and_recorded_rules(self):
        args=("simulate","--size","3","--limit","4","--games","3","--seed","18")
        a,b=self.run_cli(*args),self.run_cli(*args)
        self.assertEqual(a.returncode,0,a.stderr)
        ra,rb=json.loads(a.stdout),json.loads(b.stdout)
        for k in ("black_wins","white_wins","draws","collisions","distinct_trajectories"):
            self.assertEqual(ra["result"][k],rb["result"][k])
        self.assertEqual(ra["manifest"]["rules"]["size"],3)
        self.assertEqual(ra["result"]["games"],3)
        self.assertLessEqual(ra["result"]["max_rounds"],4)

    def test_solve_fixture_uses_its_actual_rules_and_reports_gap(self):
        fixture=json.loads((Path(__file__).parent/"fixtures"/"mixed_subgame.json").read_text())
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/"state.json";path.write_text(json.dumps(fixture["state"]))
            result=self.run_cli("solve","--state",str(path))
        self.assertEqual(result.returncode,0,result.stderr)
        report=json.loads(result.stdout)
        self.assertEqual(report["manifest"]["rules"]["size"],3)
        self.assertAlmostEqual(report["result"]["value_black"],-.25)
        self.assertLess(report["result"]["numerical_value_error_bound"],1e-7)
        budget=self.run_cli("solve","--size","3","--limit","2","--max-nodes","1")
        self.assertEqual(budget.returncode,1)
        self.assertIn("exhausted",budget.stderr)
