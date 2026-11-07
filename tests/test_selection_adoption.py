from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from graygo.confidence import canonical_sha
from graygo.evaluation import FrozenNeuralPolicy
from graygo.models import ModelConfig
from tools.policy_selection import selection_spec,SelectedAveragePolicy
from tools.adopt_selection import apply,assess
from graygo.manifest import source_digest
from graygo.storage import digest
from tools.continuous_training import create,run
from tools.learning_loop import worker
from tools.champion_loop import ChampionLoop,LoopPlan
from tools.champion_gate import GatePlan
from test_continuous_training import parent_run,config
from graygo.board_execution import BoardExecutor,BoardExecutionPlan
from graygo.board_journal import BoardJournalPlan


class SelectionAdoptionTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1)

    def test_selected_candidate_export_round_trip(self):
        from tools.champion_loop import policy
        class FixtureGate:
            plan=GatePlan(1,(4,));sha256=canonical_sha({'fixture':'selection'});protected_authorizer=None
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            with BoardExecutor(root,config(),BoardJournalPlan('loop',16,(0,16)),BoardExecutionPlan(None),reserve_bytes=0) as executor:
                with ChampionLoop(executor,LoopPlan((16,),release_temperature=0.,release_placement_only=True),FixtureGate()) as loop:
                    descriptor=loop._export(loop._candidate(root/'weights/000000000000','fixture'))
                    self.assertEqual(descriptor['selection'],selection_spec())
                    self.assertTrue(Path(descriptor['bundle']).with_name('policy_selection.py').exists())
                    self.assertEqual(policy(descriptor).sha256,descriptor['policy_sha256'])
                    bundle=torch.load(descriptor['bundle'],map_location='cpu',weights_only=True)
                    self.assertEqual(bundle['selection'],descriptor['selection'])

    def test_assessor_rejects_spent_changed_or_double_allocated_confidence_slots(self):
        for condition,message in (('spent','already used'),('changed','already used'),
                                  ('alpha','per-slot alpha'),('ordinary','ordinary gate also used')):
            with self.subTest(condition=condition),tempfile.TemporaryDirectory() as directory:
                trial=Path(directory)/'trial';stage=Path(directory)/'stage';trial.mkdir()
                def save(path,body):
                    path.parent.mkdir(parents=True,exist_ok=True)
                    path.write_text(json.dumps({'body':body,'sha256':canonical_sha(body)}))
                row={'index':0,'decision':{'reason':'adaptive_training_interval'},'promoted':False}
                if condition=='spent':row['decision']['criteria']={}
                stage_plan={'gate_alpha':.05};save(trial/'allocation-stage-plan.json',stage_plan)
                (trial/'parallel_gate.py').write_text('# frozen runner fixture\n')
                gate=GatePlan(1,(4,))
                body={'version':'graygo/selection-release-trial/1','allocation_stage':str(stage),
                    'parallel_runner_sha256':digest(trial/'parallel_gate.py'),
                    'numeric_gate_sha256':digest(Path(__file__).resolve().parents[1]/'tools/champion_gate.py'),
                    'stage_plan_sha256':canonical_sha(stage_plan),'source_sha256':source_digest(),
                    'round_index':0,'original_row_sha256':canonical_sha(row),'gate_plan':asdict(gate),
                    'allocated_alpha':.1 if condition=='alpha' else .05}
                save(trial/'plan.json',body)
                if condition=='changed':row['decision']['reason']='changed'
                save(stage/'learner/league/state.json',{'rounds':[row],
                    'contract':{'loop_plan':{'round_ends':[10]}}})
                authority=stage/'learner/league/evaluations/0000/external-selection-allocation.json'
                authority.parent.mkdir(parents=True);authority.write_text(json.dumps({'plan_sha256':canonical_sha(body)}))
                if condition=='ordinary':(authority.parent/'evidence.sqlite').touch()
                with self.assertRaisesRegex(ValueError,message):assess(trial)

    def test_adoption_preserves_learner_and_continues_selected_parallel_gates(self):
        checkout=Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);parent,publication=parent_run(root);campaign=root/'campaign'
            create(campaign,parent,checkout,publication,project=root,segment_records=16,block=8,reserve_bytes=0)
            def stopped(stage,**kwargs):kwargs['stop_state'].append(True);return worker(stage,**kwargs)
            with patch('tools.continuous_training.worker',side_effect=stopped):run(campaign,max_segments=1,reserve_bytes=0)
            stage=campaign/'stages/000000';state_path=stage/'learner/league/state.json';before=json.loads(state_path.read_text())['body']
            archive=stage/'learner/weights/000000000008';base=FrozenNeuralPolicy.from_archive(archive,label='selected-fixture');selection=selection_spec();selected=SelectedAveragePolicy(base,selection)
            candidate={'kind':'archive','archive':str(archive),'label':'selected-fixture','rules':base.identity['origin']['rules'],
                'records':8,'policy_sha256':selected.sha256,'weights_sha256':base.identity['weights_sha256'],'selection':selection}
            trial=root/'trial';trial.mkdir();stage_plan_sha=json.loads((stage/'plan.json').read_text())['sha256']
            trial_body={'allocation_stage':str(stage),'stage_plan_sha256':stage_plan_sha}
            (trial/'plan.json').write_text(json.dumps({'body':trial_body,'sha256':canonical_sha(trial_body)}))
            decision={'status':'complete','numeric_pass':True,'head_win_score':{'mean':.75,'lower':.6}}
            assessment={'trial':str(trial),'records':8,'candidate':candidate,'incumbent':before['champion'],'pool':{},'decision':decision,
                'round_index':0,'evaluation_root':str(trial/'evaluation'),'allocation':str(trial/'allocation.json')}
            checkpoint=stage/'learner/resume/000000000008/runtime.pt';payload=checkpoint.read_bytes()
            with patch('tools.adopt_selection.assess',return_value=assessment):
                receipt=apply(campaign,trial,reserve_bytes=0)
                with self.assertRaisesRegex(ValueError,'stage plan changed|champion changed'):apply(campaign,trial,reserve_bytes=0)
            after=json.loads(state_path.read_text())['body']
            self.assertEqual(checkpoint.read_bytes(),payload)
            self.assertEqual(after['history'][-1],before['champion'])
            self.assertEqual(after['contract']['loop_plan']['release_temperature'],0.)
            self.assertEqual(after['contract']['gate_workers'],8)
            self.assertEqual(after['prior_promotions'],before.get('prior_promotions',0)+1)
            self.assertEqual(after['external_evaluations'][-1]['candidate'],candidate)
            result=run(campaign,max_segments=1,reserve_bytes=0)
            self.assertEqual(result['completed_segments'],1)
