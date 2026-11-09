import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from tools.continuous_training import create,run
from tools.learning_loop import worker
from tools.champion_loop import ChampionLoop
from tools.upgrade_gate_scheduler import apply,screen_defaults_only
from tools.champion_screen import IndependentScreen,ScreenPlan
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


    def screen_size_fixture(self,base,status):
        checkout=Path(__file__).resolve().parents[1]
        parent,publication=parent_run(base);campaign=base/'campaign'
        create(campaign,parent,checkout,publication,project=base,segment_records=16,block=8,reserve_bytes=0)
        def stop_during_evaluation(loop,archive,candidate,pending,stop_requested):
            pending.update(status=status,candidate=candidate,screen={'fixture':'old 20-game screen'})
            loop._save()
            return {'status':'interrupted','numeric_pass':False,'promotion_permitted':False}
        def old_screen(plan,**kwargs):return IndependentScreen(ScreenPlan(20,120.),**kwargs)
        with patch('tools.learning_loop.IndependentScreen',side_effect=old_screen),patch.object(ChampionLoop,'_evaluate',stop_during_evaluation):
            run(campaign,max_segments=1,reserve_bytes=0)
        stage=campaign/'stages/000000';path=stage/'learner/league/state.json';wrapper=json.loads(path.read_text())
        wrapper['body']['adaptive_scheduler'].update(previous_score=.675,last_gate_score=.825,
            history_scores={'old-20-game-baseline':.825},next_gate_learning_seconds=99999.)
        wrapper['sha256']=canonical_sha(wrapper['body']);path.write_text(json.dumps(wrapper))
        return campaign,stage,publication

    def test_screen_size_upgrade_preserves_frozen_gate_and_full_learner_payload(self):
        with tempfile.TemporaryDirectory() as directory:
            campaign,stage,publication=self.screen_size_fixture(Path(directory),'evaluating')
            path=stage/'learner/league/state.json';before=json.loads(path.read_text())['body']
            checkpoint=stage/'learner/resume/000000000016/runtime.pt';payload=checkpoint.read_bytes()
            evidence=stage/'learner/league/evaluations/0000/frozen-fixture.txt'
            evidence.parent.mkdir(parents=True,exist_ok=True);evidence.write_text('retain the same candidate, seeds and evidence')
            receipt=apply(campaign,reserve_bytes=0,screen_size=True);after=json.loads(path.read_text())['body']
            self.assertTrue(receipt['inflight_gate_preserved'])
            self.assertEqual(before['pending'],after['pending'])
            self.assertEqual(before['champion'],after['champion'])
            self.assertEqual(before['contract']['gate_sha256'],after['contract']['gate_sha256'])
            self.assertEqual(before['contract']['gate_looks'],after['contract']['gate_looks'])
            self.assertEqual(payload,checkpoint.read_bytes())
            self.assertEqual(evidence.read_text(),'retain the same candidate, seeds and evidence')
            self.assertEqual(after['contract']['screen_sha256'],IndependentScreen().sha256)
            self.assertEqual(after['adaptive_scheduler']['history_scores'],{})
            self.assertIsNone(after['adaptive_scheduler']['previous_score'])
            self.assertIsNone(after['adaptive_scheduler']['last_gate_score'])
            self.assertEqual(after['adaptive_scheduler']['last_gate_learning_seconds'],before['adaptive_scheduler']['last_gate_learning_seconds'])
            self.assertEqual(after['adaptive_scheduler']['next_gate_learning_seconds'],99999.)
            def resume_same_gate(loop,archive,candidate,pending,stop_requested):
                self.assertEqual(pending,before['pending'])
                return {'status':'interrupted','numeric_pass':False,'promotion_permitted':False}
            with patch.object(ChampionLoop,'_evaluate',resume_same_gate):
                self.assertEqual(run(campaign,max_segments=1,reserve_bytes=0)['status'],'interrupted')

    def test_screen_size_upgrade_retires_only_an_incomplete_old_screen(self):
        with tempfile.TemporaryDirectory() as directory:
            campaign,stage,publication=self.screen_size_fixture(Path(directory),'screening')
            checkpoint=stage/'learner/resume/000000000016/runtime.pt';payload=checkpoint.read_bytes()
            receipt=apply(campaign,reserve_bytes=0,screen_size=True)
            after=json.loads((stage/'learner/league/state.json').read_text())['body']
            self.assertFalse(receipt['inflight_gate_preserved']);self.assertIsNone(after['pending'])
            self.assertEqual(after['rounds'][-1]['decision']['reason'],'screen_size_changed')
            self.assertEqual(after['rounds'][-1]['screen'],{'fixture':'old 20-game screen'})
            self.assertEqual(payload,checkpoint.read_bytes())
            self.assertEqual(run(campaign,max_segments=1,reserve_bytes=0)['completed_segments'],1)

    def test_screen_size_source_admission_rejects_an_unrelated_behavior_change(self):
        checkout=Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            old=Path(directory)/'old.py';new=Path(directory)/'new.py'
            source=(checkout/'tools/champion_screen.py').read_text()
            old.write_text(source.replace('games: int = 100','games: int = 20').replace('seconds: float = 600.','seconds: float = 120.'))
            new.write_text(source);screen_defaults_only(old,new,'champion_screen.py')
            new.write_text(source.replace("mean_utility'] > 0","mean_utility'] > -.1"))
            with self.assertRaisesRegex(ValueError,'other code'):screen_defaults_only(old,new,'champion_screen.py')


if __name__=='__main__':unittest.main()
