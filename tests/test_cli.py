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


    def test_buffer_profile_routes_arguments_and_verifies_roundtrip(self):
        with tempfile.TemporaryDirectory() as root:
            result=self.run_cli('profile-buffers','--size','3','--limit','4','--capacity','16',
                                '--records','32','--iterations','1','--batch','8','--buffer-dir',root)
        self.assertEqual(result.returncode,0,result.stderr)
        report=json.loads(result.stdout)
        self.assertEqual(report['result']['filled_rows'],16)
        self.assertEqual(report['result']['fresh_records'],32)
        self.assertTrue(report['result']['roundtrip_verified'])

    def test_nfsp_cli_writes_report_and_resumes_completed_run(self):
        with tempfile.TemporaryDirectory() as folder:
            run=Path(folder)/'run';output=Path(folder)/'report.json'
            args=('train-nfsp','--task','asymmetric','--learner','tabular','--records','128',
                  '--run-dir',str(run),'--output',str(output))
            result=self.run_cli(*args)
            self.assertEqual(result.returncode,0,result.stderr)
            report=json.loads(output.read_text())
            self.assertEqual(report['result']['per_role_records'],[64,64])
            self.assertFalse(report['result']['formal_gate_passed'])
            resumed=self.run_cli(*args,'--resume')
            self.assertEqual(resumed.returncode,0,resumed.stderr)
            other=json.loads(output.read_text())
            self.assertEqual(other['result']['counters'],report['result']['counters'])
            self.assertIsNone(other['manifest']['rules'])
            self.assertEqual(other['manifest']['encoding_version'],'graygo/nfsp-matrix-role/1')

    def test_multiround_cli_preserves_actual_task_rules_and_active_episode(self):
        with tempfile.TemporaryDirectory() as folder:
            output=Path(folder)/'report.json';run=Path(folder)/'run'
            args=('train-nfsp-task','--task','gray-mixed','--learner','tabular','--records','2',
                  '--purpose','profile','--time-cap-hours','.05','--run-dir',str(run),'--output',str(output))
            a=self.run_cli(*args);self.assertEqual(a.returncode,0,a.stderr)
            report=json.loads(output.read_text())
            self.assertEqual(report['manifest']['rules']['size'],3)
            self.assertEqual(report['manifest']['rules']['max_rounds'],6)
            self.assertEqual(report['result']['active_episode_steps'],1)
            self.assertFalse(report['result']['formal_case_passed'])
            b=self.run_cli(*args,'--resume');self.assertEqual(b.returncode,0,b.stderr)
            self.assertEqual(json.loads(output.read_text())['result']['counters'],report['result']['counters'])

    def test_rule_buffer_profile_writes_schema_and_checks_checkpoint_restore(self):
        with tempfile.TemporaryDirectory() as root:
            result=self.run_cli('profile-buffers','--storage','rule-state','--size','3','--limit','4',
                '--capacity','16','--records','32','--iterations','1','--batch','8','--buffer-dir',root)
        self.assertEqual(result.returncode,0,result.stderr)
        report=json.loads(result.stdout)['result']
        self.assertEqual(report['schema']['storage'],'rule-state')
        self.assertTrue(report['roundtrip_verified'])
        self.assertGreaterEqual(report['restore_validation_seconds'],0.)

    @unittest.skipIf(__import__('shutil').which('c++') is None,'C++17 compiler unavailable')
    def test_native_build_and_compact_profile_cli(self):
        with tempfile.TemporaryDirectory() as root:
            build=self.run_cli('build-native','--output-dir',str(Path(root)/'native'))
            self.assertEqual(build.returncode,0,build.stderr)
            report=json.loads(build.stdout);library=report['result']['library']
            self.assertIsNone(report['manifest']['rules'])
            result=self.run_cli('profile-buffers','--storage','rule-state','--native-library',library,
                '--size','3','--limit','4','--capacity','16','--records','32','--iterations','1','--batch','8',
                '--buffer-dir',str(Path(root)/'profile'))
            self.assertEqual(result.returncode,0,result.stderr)
            checked=json.loads(result.stdout)['result']
            self.assertTrue(checked['roundtrip_verified'])
            self.assertEqual(checked['schema']['native_decoder']['sha256'],report['result']['library_sha256'])

    def test_frozen_paired_evaluation_cli_resumes_exact_planned_evidence(self):
        from graygo.confidence import ConfidenceFamily
        with tempfile.TemporaryDirectory() as root:
            family=ConfidenceFamily('cli-eval',('utility',),(4,8));plan=Path(root)/'confidence.json';plan.write_text(json.dumps(family.to_dict()))
            output=Path(root)/'report.json'
            args=('evaluate','--size','3','--limit','3','--subject','random','--opponent','random','--incumbent','random',
                '--name','test','--subject-label','subject','--endpoint','utility','--confidence-plan',str(plan),
                '--run-dir',str(Path(root)/'run'),'--time-cap-hours','.05','--output',str(output))
            a=self.run_cli(*args,'--games','4');self.assertEqual(a.returncode,0,a.stderr)
            first=json.loads(output.read_text())['result'];self.assertFalse(first['promotion_permitted'])
            b=self.run_cli(*args,'--games','8','--resume');self.assertEqual(b.returncode,0,b.stderr)
            final=json.loads(output.read_text())['result']
            self.assertEqual(final['collection']['committed_pairs'],4)
            self.assertEqual(final['confidence']['mean'],0.)
            self.assertEqual(final['confidence']['independent_units'],4)

    def test_calibration_cli_preserves_underpowered_pilot_and_resumes(self):
        from graygo.confidence import ConfidenceFamily
        from graygo.engine import Rules
        from graygo.evaluation import BaselinePolicy,EvaluationPlan,EvaluationRun
        from graygo.promotion import ProtectedPool,PromotionProtocol
        with tempfile.TemporaryDirectory() as root:
            subject=BaselinePolicy('tactical');incumbent=BaselinePolicy('random');opponents=[BaselinePolicy('random'),BaselinePolicy('tactical')]
            rules=Rules(3,2);family=ConfidenceFamily('cli-pilot',('delta/0','delta/1'),(4,8))
            pool=ProtectedPool('cli',1,tuple((str(j),p.sha256) for j,p in enumerate(opponents)),(.5,.5),rules)
            protocol=PromotionProtocol(pool);path=Path(root)/'protocol.json';path.write_text(json.dumps(protocol.to_dict()))
            args=['calibrate-promotion','--protocol',str(path),'--run-dir',str(Path(root)/'calibration'),'--simulations','2','--seed','7','--time-cap-hours','.01']
            for j,opponent in enumerate(opponents):
                directory=Path(root)/str(j);plan=EvaluationPlan('pilot','pilot','delta/'+str(j),rules,3,family.looks,confidence_sha256=family.sha256,comparison='improvement')
                run=EvaluationRun(plan,directory,subject,opponent,incumbent=incumbent,reserve_bytes=0);run.run(8,time_cap_seconds=30)
                args.extend(['--pilot',str(j)+'='+str(directory)])
            a=self.run_cli(*args);self.assertEqual(a.returncode,0,a.stderr)
            report=json.loads(a.stdout)['result'];self.assertFalse(report['passed']);self.assertEqual(report['status'],'not_calibrated')
            self.assertIn('insufficient separately designated pilot units',report['reasons'])
            b=self.run_cli(*args,'--resume');self.assertEqual(b.returncode,0,b.stderr)
            self.assertEqual(json.loads(b.stdout)['result']['scenarios'],report['scenarios'])
            progress=Path(root)/'calibration/progress.json';bad=json.loads(progress.read_text());bad['state']['elapsed_seconds']=0.;progress.write_text(json.dumps(bad))
            c=self.run_cli(*args,'--resume');self.assertEqual(c.returncode,1);self.assertIn('checksum',c.stderr)
