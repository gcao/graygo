from pathlib import Path
import json
import sys
import tempfile
import time
import unittest
import torch
from graygo.board_execution import BoardExecutor,BoardExecutionPlan
from graygo.board_journal import BoardJournalPlan
from graygo.board_runtime import BoardRuntimeConfig
from graygo.engine import Rules
from tools.training_comparison import ComparisonPlan,prepare,resume_policies,run_limited


class TrainingComparisonTests(unittest.TestCase):
    def test_one_total_budget_is_split_between_both_arms_and_common_evaluation(self):
        plan=ComparisonPlan(3600)
        self.assertEqual(plan.arm_seconds,1620)
        self.assertEqual(2*plan.arm_seconds+plan.total_seconds*plan.evaluation_fraction,plan.total_seconds)
        for seconds in (float('nan'),float('inf'),0,7200):
            with self.assertRaises(ValueError):ComparisonPlan(seconds)

    def test_prepare_freezes_only_training_libraries_and_never_launches(self):
        with tempfile.TemporaryDirectory() as directory:
            parent=Path(directory);repo=parent/'old';repo.mkdir()
            files=('model.py','train.py','selfplay.py','selfplay_cpp.py','export_model.py','engine.py')
            for name in (*files,'run.py','secrets.py'):(repo/name).write_text('# fixture\n')
            reference=parent/'reference.pt';reference.write_bytes(b'fixture')
            root=parent/'prepared'
            prepare(root,ComparisonPlan(60,device='cpu'),repo,reference)
            self.assertEqual({p.name for p in (root/'legacy-source').iterdir()},set(files))
            self.assertEqual(json.loads((root/'status.json').read_text()),{'status':'prepared','training_started':False})
            self.assertFalse((root/'new').exists());self.assertFalse((root/'legacy').exists())
            (repo/'model.py').write_text('# user edit after preparation\n')
            self.assertEqual((root/'legacy-source/model.py').read_text(),'# fixture\n')
            with self.assertRaises(FileExistsError):prepare(root,ComparisonPlan(60),repo,reference)

    @unittest.skipUnless(sys.platform=='linux','POSIX group deadline evidence uses Linux /proc')
    def test_watchdog_kills_native_style_unresponsive_child_and_descendants(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);pid=root/'grandchild.pid'
            script=('import signal,subprocess,sys,time;signal.signal(signal.SIGTERM,signal.SIG_IGN);'
                    'p=subprocess.Popen([sys.executable,"-c","import time;time.sleep(60)"]);'
                    'open(sys.argv[1],"w").write(str(p.pid));time.sleep(60)')
            result=run_limited([sys.executable,'-c',script,str(pid)],cwd=root,log=root/'child.log',seconds=1.)
            self.assertTrue(result['soft_stop']);self.assertTrue(result['hard_stop'])
            self.assertLess(result['wall_seconds'],3.)
            process=Path('/proc')/pid.read_text()/'stat'
            for attempt in range(20):
                try:state=process.read_text().split()[2]
                except (FileNotFoundError,ProcessLookupError):break
                if state=='Z':break
                time.sleep(.01)
            else:self.fail('grandchild survived process-group deadline')

    def test_completed_worker_preserves_exit_status(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            result=run_limited([sys.executable,'-c','print("completed")'],cwd=root,log=root/'child.log',seconds=5.)
            self.assertEqual(result['returncode'],0);self.assertFalse(result['soft_stop'])
            self.assertIn('completed',(root/'child.log').read_text())

    def test_practice_checkpoint_exposes_q_and_reports_uninitialized_average(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as directory:
            config=BoardRuntimeConfig(rules=Rules(3,5),envs=2,capacity=256,batch=4,learning_start=4,
                target_every=8,width=8,blocks=1,norm_groups=2)
            with BoardExecutor(directory,config,BoardJournalPlan('practice',16,(0,16)),
                               BoardExecutionPlan(30),reserve_bytes=0) as learner:
                result=learner.run();self.assertEqual(result['records'],16)
                self.assertEqual(learner.runtime.reservoir.size,0)
            policies,report=resume_policies(directory)
            self.assertEqual(set(policies),{'new-greedy_q'})
            self.assertFalse(report['average_ready']);self.assertGreater(report['counters']['q_updates'],0)


if __name__=='__main__':unittest.main()
