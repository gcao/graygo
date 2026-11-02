import json
from pathlib import Path
import runpy
import tempfile
import unittest
import torch
from graygo.storage import CheckpointStore,digest
from graygo.task_training import NFSPTaskConfig,NFSPTaskTrainer

AUDIT=runpy.run_path(str(Path(__file__).resolve().parents[1]/'tools/audit_nfsp_delayed.py'))['audit']


class DelayedAuditTests(unittest.TestCase):
    def fixture(self,root,learner='tabular'):
        config=NFSPTaskConfig(time_cap_seconds=60,task='delayed',learner=learner,seed=5,records=320,eval_every=64,
            batch=8,learning_start=8,capacity=64,target_every=16,reservoir_open=32,handoff_end=64,
            epsilon_decay=160,horizons=(3,4),equivalent_actions=4,hidden=8,purpose='profile')
        trainer=NFSPTaskTrainer(config,root,reserve_bytes=0)
        result=trainer.run(verbose=False)
        self.assertEqual(result['status'],'complete')
        return trainer

    def reseal(self,root,change):
        path=CheckpointStore(root,reserve_bytes=0).latest()
        state=torch.load(path/'trainer.pt',weights_only=True,map_location='cpu');change(state)
        torch.save(state,path/'trainer.pt')
        manifest=json.loads((path/'manifest.json').read_text())
        manifest['files']['trainer.pt']={'bytes':(path/'trainer.pt').stat().st_size,'sha256':digest(path/'trainer.pt')}
        (path/'manifest.json').write_text(json.dumps(manifest,sort_keys=True))
        (path/'manifest.sha256').write_text(digest(path/'manifest.json')+'\n')
        CheckpointStore.verify(path)

    def test_tabular_and_neural_complete_diagnostics_pass_read_only_audit(self):
        for learner in ('tabular','neural'):
            with self.subTest(learner=learner),tempfile.TemporaryDirectory() as directory:
                self.fixture(directory,learner)
                before={p.relative_to(directory):digest(p) for p in Path(directory).rglob('*') if p.is_file()}
                result=AUDIT(directory,require_formal=False)
                self.assertTrue(result['verified']);self.assertEqual(result['records'],320)
                self.assertEqual(result['eligibility_version'],2)
                self.assertEqual(result['q_eligible_records']+result['excluded_passive_root_records'],320)
                self.assertGreater(result['excluded_passive_root_records'],0)
                self.assertFalse(result['formal_protocol']);self.assertFalse(result['formal_case_passed'])
                after={p.relative_to(directory):digest(p) for p in Path(directory).rglob('*') if p.is_file()}
                self.assertEqual(before,after)
                with self.assertRaisesRegex(ValueError,'formal primary'):AUDIT(directory)

    def test_validly_sealed_wrong_native_opponent_value_is_detected(self):
        with tempfile.TemporaryDirectory() as directory:
            self.fixture(directory)
            def change(state):state['evaluations'][-1]['roots'][0]['root_action_values'][2]['expected']+=.25
            self.reseal(directory,change)
            with self.assertRaisesRegex(ValueError,'numerical result differs'):AUDIT(directory,require_formal=False)

    def test_validly_sealed_private_rng_drift_is_detected(self):
        with tempfile.TemporaryDirectory() as directory:
            self.fixture(directory)
            def change(state):state['rng']['action_black']['state']['state']+=1
            self.reseal(directory,change)
            with self.assertRaisesRegex(ValueError,'private draw count'):AUDIT(directory,require_formal=False)

    def test_validly_sealed_wrong_eligible_clock_is_independently_detected(self):
        with tempfile.TemporaryDirectory() as directory:
            self.fixture(directory)
            def change(state):state['backend']['counters']['fresh_q']+=1
            self.reseal(directory,change)
            with self.assertRaisesRegex(ValueError,'eligible Q budget accounting'):
                AUDIT(directory,require_formal=False)

if __name__=='__main__':unittest.main()
