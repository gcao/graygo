from pathlib import Path
import json
import tempfile
import unittest
from unittest.mock import patch
from graygo.manifest import source_digest
from graygo.task_training import NFSPTaskConfig, NFSPTaskTrainer
from tools.a2_cohort import digest, canonical, input_snapshot
from tools.branch_case_v2 import collect, AUDITOR


class BranchCaseV2Tests(unittest.TestCase):
    def fixture(self, root, target='one-step'):
        config=NFSPTaskConfig(120,task='branching',learner='tabular',seed=0,records=64,eval_every=16,
            batch=8,learning_start=8,capacity=64,target_every=16,reservoir_open=8,handoff_end=32,
            epsilon_decay=32,hidden=8,width=8,blocks=1,norm_groups=2,purpose='profile',q_reuse=4,q_target=target)
        trainer=NFSPTaskTrainer(config,root,reserve_bytes=0)
        trainer.run(verbose=False)
        return trainer

    def test_current_branching_collected_with_v2_independent_response_and_unchanged_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)/'trial';self.fixture(root);before=input_snapshot(root)
            output=Path(directory)/'case.json'
            with patch('graygo.task_evaluation.evaluate_task',side_effect=AssertionError('producer grader forbidden')):
                wrapper=collect(root,output,expected_source_sha256=source_digest(),
                    expected_auditor_sha256=digest(AUDITOR),allow_diagnostic=True)
            body=wrapper['body']
            self.assertEqual(before,input_snapshot(root));self.assertEqual(canonical(body),wrapper['sha256'])
            self.assertEqual(body['version'],'graygo/nfsp-branching-case/2')
            self.assertEqual(body['independent_audit']['audit_semantics'],'graygo/independent-bounded-task-audit/2')
            self.assertIn('max_native_response_gap',body['independent_audit']['scheduled_evaluations'][0])
            self.assertFalse(body['primary_recipe']);self.assertFalse(body['a2_complete'])
            self.assertFalse(body['source_admission_qualified'])
            with self.assertRaisesRegex(ValueError,'preserve'):
                collect(root,output,expected_source_sha256=source_digest(),
                    expected_auditor_sha256=digest(AUDITOR),allow_diagnostic=True)
            with self.assertRaises(ValueError):
                collect(root,Path(directory)/'wrong.json',expected_source_sha256='0'*64,
                    expected_auditor_sha256=digest(AUDITOR),allow_diagnostic=True)

    def test_one_step_auditor_cannot_issue_retrace_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)/'trial';self.fixture(root,'retrace')
            with self.assertRaisesRegex(ValueError,'one-step'):
                collect(root,Path(directory)/'case.json',expected_source_sha256=source_digest(),
                    expected_auditor_sha256=digest(AUDITOR),allow_diagnostic=True)


if __name__=='__main__':unittest.main()
