from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from tools.continuous_training import create,run
from tools.learning_loop import worker
from tools.upgrade_gate_scheduler import apply
from test_continuous_training import parent_run


class SchedulerUpgradeTests(unittest.TestCase):
    def test_offline_scheduler_upgrade_preserves_checkpoints_evidence_and_can_resume(self):
        checkout=Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory);parent,publication=parent_run(base);campaign=base/'campaign'
            create(campaign,parent,checkout,publication,project=base,segment_records=16,block=8,reserve_bytes=0)
            def interrupted(stage,**kwargs):
                kwargs['stop_state'].append(True);return worker(stage,**kwargs)
            with patch('tools.continuous_training.worker',side_effect=interrupted):run(campaign,max_segments=1,reserve_bytes=0)
            stage=campaign/'stages/000000';checkpoint=stage/'learner/resume/000000000008/manifest.json'
            original=checkpoint.read_bytes()
            evidence=stage/'learner/league/evaluations/0000/retained-evidence.txt'
            evidence.parent.mkdir(parents=True);evidence.write_text('fixture preserved')
            receipt=apply(campaign,reserve_bytes=0)
            self.assertFalse(receipt['numeric_promotion_policy_changed'])
            self.assertEqual(checkpoint.read_bytes(),original)
            self.assertEqual(evidence.read_text(),'fixture preserved')
            self.assertTrue((Path(receipt['archive'])/'league-state.json').exists())
            result=run(campaign,max_segments=1,reserve_bytes=0)
            self.assertEqual(result['completed_segments'],1)


if __name__=='__main__':unittest.main()
