import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from tools.continuous_training import create,run
from tools.learning_loop import worker
from tools.champion_loop import ChampionLoop
from tools.upgrade_gate_scheduler import apply
from graygo.confidence import canonical_sha
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
            receipt=apply(campaign,reserve_bytes=0,evaluation_schedule=True)
            self.assertTrue(receipt['evaluation_schedule_changed'])
            self.assertEqual(receipt['opponents'],'champion-and-frozen-model-history')
            self.assertFalse(receipt['numeric_promotion_policy_changed'])
            self.assertEqual(checkpoint.read_bytes(),original)
            self.assertEqual(evidence.read_text(),'fixture preserved')
            self.assertTrue((Path(receipt['archive'])/'league-state.json').exists())
            result=run(campaign,max_segments=1,reserve_bytes=0)
            self.assertEqual(result['completed_segments'],1)

    def test_legacy_allowance_schedule_can_upgrade_and_resume_without_changing_checkpoint(self):
        checkout=Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory);parent,publication=parent_run(base);campaign=base/'campaign'
            create(campaign,parent,checkout,publication,project=base,segment_records=16,block=8,reserve_bytes=0)
            def legacy_worker(stage,**kwargs):
                stage=Path(stage);path=stage/'plan.json';wrapper=json.loads(path.read_text())
                wrapper['body']['adaptive_gate_schedule']=False
                wrapper['body']['tools'].pop('adaptive_gate_schedule.py')
                wrapper['sha256']=canonical_sha(wrapper['body']);path.write_text(json.dumps(wrapper))
                reservation=Path(wrapper['body']['initialization']['parent_run'])/'handoff.json'
                record=json.loads(reservation.read_text());record['plan_sha256']=wrapper['sha256']
                reservation.write_text(json.dumps(record))
                kwargs['stop_state'].append(True)
                return worker(stage,**kwargs)
            with patch('tools.continuous_training.worker',side_effect=legacy_worker):
                run(campaign,max_segments=1,reserve_bytes=0)
            stage=campaign/'stages/000000';checkpoint=stage/'learner/resume/000000000008/manifest.json'
            saved=checkpoint.read_bytes()
            state_path=stage/'learner/league/state.json'
            self.assertNotIn('adaptive_scheduler',json.loads(state_path.read_text())['body'])
            receipt=apply(campaign,reserve_bytes=0,adaptive_schedule=True)
            state=json.loads(state_path.read_text())['body']
            self.assertTrue(receipt['adaptive_schedule_enabled'])
            self.assertEqual(saved,checkpoint.read_bytes())
            self.assertTrue(state['contract']['loop_plan']['adaptive_schedule'])
            self.assertIn('adaptive_scheduler',state)
            self.assertEqual(run(campaign,max_segments=1,reserve_bytes=0)['completed_segments'],1)

    def test_adaptive_upgrade_preserves_started_gate_and_saved_state(self):
        checkout=Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory);parent,publication=parent_run(base);campaign=base/'campaign'
            create(campaign,parent,checkout,publication,project=base,segment_records=16,block=8,reserve_bytes=0)
            def stop_at_gate(loop,archive,candidate,pending,stop_requested):
                pending.update(status='evaluating',candidate=candidate)
                loop._save()
                return {'status':'interrupted','numeric_pass':False,'promotion_permitted':False}
            with patch.object(ChampionLoop,'_evaluate',stop_at_gate):run(campaign,max_segments=1,reserve_bytes=0)
            stage=campaign/'stages/000000';state_path=stage/'learner/league/state.json'
            before=json.loads(state_path.read_text())['body']
            receipt=apply(campaign,reserve_bytes=0,adaptive_schedule=True)
            after=json.loads(state_path.read_text())['body']
            self.assertTrue(receipt['adaptive_schedule_enabled'])
            self.assertEqual(before['pending'],after['pending'])
            self.assertEqual(before['adaptive_scheduler'],after['adaptive_scheduler'])
            self.assertEqual(before['evaluation_seconds'],after['evaluation_seconds'])
            self.assertEqual(before['champion'],after['champion'])

    def test_old_incomplete_gate_is_retired_without_reusing_evidence_or_changing_model(self):
        checkout=Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory);parent,publication=parent_run(base);campaign=base/'campaign'
            create(campaign,parent,checkout,publication,project=base,segment_records=16,block=8,reserve_bytes=0)
            def stop_at_gate(loop,archive,candidate,pending,stop_requested):
                pending.update(status='evaluating',candidate=candidate)
                loop._save()
                return {'status':'interrupted','numeric_pass':False,'promotion_permitted':False}
            with patch.object(ChampionLoop,'_evaluate',stop_at_gate):
                first=run(campaign,max_segments=1,reserve_bytes=0)
            self.assertEqual(first['status'],'interrupted')
            stage=campaign/'stages/000000';checkpoint=stage/'learner/resume/000000000016/manifest.json'
            original=checkpoint.read_bytes()
            before=json.loads(publication.read_text())['body']['champion']['policy_sha256']
            apply(campaign,reserve_bytes=0,evaluation_schedule=True)
            state=json.loads((stage/'learner/league/state.json').read_text())['body']
            self.assertIsNone(state['pending'])
            self.assertEqual(state['rounds'][-1]['decision']['reason'],'evaluation_schedule_superseded')
            self.assertFalse(state['rounds'][-1]['promoted'])
            self.assertEqual(checkpoint.read_bytes(),original)
            self.assertEqual(json.loads(publication.read_text())['body']['champion']['policy_sha256'],before)
            self.assertEqual(run(campaign,max_segments=1,reserve_bytes=0)['completed_segments'],1)


if __name__=='__main__':unittest.main()
