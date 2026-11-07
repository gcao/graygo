from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from graygo.confidence import canonical_sha
from graygo.evaluation import FrozenNeuralPolicy
from tools.champion_gate import GatePlan,HEAD_TO_HEAD_HISTORY,POSITIVE_POOL_GAIN,LEGACY_POOL_GAIN,gate_root
from tools.adopt_head_to_head_policy import reassess,apply
from tools.continuous_training import create,run
from tools.learning_loop import worker
from tools.champion_loop import ChampionLoop
from test_continuous_training import parent_run


class HeadToHeadPolicyTests(unittest.TestCase):
    def test_saturated_historical_pool_cannot_veto_confident_head_to_head_improvement(self):
        head=np.full(1000,.5);pool={'historical-0':np.ones(1000)}
        current=GatePlan(1);old=GatePlan(1,pool_gain_policy=POSITIVE_POOL_GAIN)
        decision=current.numeric(0,head,pool,pool,2000);original=old.numeric(0,head,pool,pool,2000)
        self.assertTrue(decision['numeric_pass']);self.assertFalse(original['numeric_pass'])
        self.assertNotIn('pool_gain',decision['criteria'])
        self.assertFalse(decision['pool_gain_required'])
        self.assertFalse(decision['diagnostics']['pool_gain'])
        for field in ('head','aggregate','opponents','family_sha256'):
            self.assertEqual(decision[field],original[field])
        self.assertEqual(gate_root('gate',current),Path('gate/head-to-head-history-v3'))

    def test_losing_to_champion_historical_regression_and_low_strength_each_block_promotion(self):
        n=1000;good=np.ones(n)
        cases=[(-good,good,good,'head_to_head_win_score_lower_above_half'),
               (good,np.full(n,.4),good,'no_historical_regression'),
               (good,-good,-good,'absolute_floor')]
        for head,candidate,incumbent,criterion in cases:
            decision=GatePlan(1).numeric(0,head,{'historical-0':candidate},{'historical-0':incumbent},2000)
            self.assertFalse(decision['numeric_pass'])
            self.assertFalse(decision['criteria'][criterion])

    def test_small_negative_pool_gain_is_not_a_futility_veto_with_stable_history(self):
        n=100000;head=np.ones(n);pool={'historical-0':np.full(n,.95)};incumbent={'historical-0':np.ones(n)}
        decision=GatePlan(1,looks=(2*n,)).numeric(0,head,pool,incumbent,2*n)
        self.assertLess(decision['aggregate']['mean']+decision['aggregate']['radius'],0)
        self.assertTrue(decision['criteria']['no_historical_regression'])
        self.assertTrue(decision['numeric_pass']);self.assertFalse(decision['stop_for_futility'])

    def test_reassessment_preserves_exact_statistics_and_original_decision(self):
        old=GatePlan(1,pool_gain_policy=POSITIVE_POOL_GAIN)
        values=(np.full(1000,.5),{'historical-0':np.ones(1000)},{'historical-0':np.ones(1000)})
        decision=old.numeric(0,*values,2000)
        with tempfile.TemporaryDirectory() as directory:
            stage=Path(directory);root=stage/'learner/league/evaluations/0000/positive-pool-gain-v2';root.mkdir(parents=True)
            candidate={'policy_sha256':'candidate'};incumbent={'policy_sha256':'incumbent'}
            row={'index':0,'records':100,'candidate':candidate,'incumbent':incumbent,'decision':decision}
            body={'rounds':[row],'contract':{'loop_plan':{'round_ends':[100]}}}
            config={'gate_alpha':.05,'gate_seed':73}
            (stage/'plan.json').write_text(json.dumps({'body':config,'sha256':canonical_sha(config)}))
            (root/'gate-plan.json').write_text(json.dumps({'family':old.family(0,('historical-0',)).to_dict(),'pool':{'historical-0':'old-model'}}))
            evidence={}
            for name in ('head','historical-0'):
                rows=([{'subject':[{'subject_utility':1},{'subject_utility':0}]} for _ in range(1000)] if name=='head'
                      else [{'utility':1.,'incumbent_utility':1.} for _ in range(1000)])
                evidence[name]=SimpleNamespace(sha256='evidence-'+name,available_games=2000,rows=rows,
                    contract={'subject_sha256':'candidate','opponent_sha256':'incumbent' if name=='head' else 'old-model',
                              'incumbent_sha256':'incumbent'},assert_current=lambda:None)
            decision.update(status='complete',evidence={name:item.sha256 for name,item in evidence.items()})
            (stage/'learner/league/state.json').write_text(json.dumps({'body':body,'sha256':canonical_sha(body)}))
            (root/'decision.json').write_text(json.dumps(decision));original=(root/'decision.json').read_bytes()
            with patch('tools.adopt_head_to_head_policy.FrozenEvaluationEvidence.load',side_effect=lambda p:evidence[p.name]),patch('tools.adopt_head_to_head_policy.policy',return_value=SimpleNamespace(assert_frozen=lambda:None)):
                report=reassess(stage,100)
                self.assertTrue(report['decision']['numeric_pass'])
                self.assertEqual(report['decision']['head'],decision['head'])
                self.assertEqual((root/'decision.json').read_bytes(),original)
                evidence['head'].available_games=1998
                with self.assertRaisesRegex(ValueError,'incomplete'):reassess(stage,100)

    def fixture_campaign(self,base,*,at_gate=False):
        checkout=Path(__file__).resolve().parents[1];parent,publication=parent_run(base);campaign=base/'campaign'
        create(campaign,parent,checkout,publication,project=base,segment_records=16,block=8,reserve_bytes=0)
        if at_gate:
            def stop_at_gate(loop,archive,candidate,pending,stop_requested):
                pending.update(status='evaluating',candidate=candidate);loop._save()
                return {'status':'interrupted','numeric_pass':False,'promotion_permitted':False}
            with patch.object(ChampionLoop,'_evaluate',stop_at_gate):run(campaign,max_segments=1,reserve_bytes=0)
        else:
            def interrupted(stage,**kwargs):kwargs['stop_state'].append(True);return worker(stage,**kwargs)
            with patch('tools.continuous_training.worker',side_effect=interrupted):run(campaign,max_segments=1,reserve_bytes=0)
        stage=campaign/'stages/000000';path=stage/'learner/league/state.json';wrapper=json.loads(path.read_text());state=wrapper['body'];incumbent=deepcopy(state['champion'])
        records=16 if at_gate else 8
        frozen=FrozenNeuralPolicy.from_archive(stage/'learner/weights'/f'{records:012d}',label='approved-model')
        candidate={'kind':'archive','archive':str(stage/'learner/weights'/f'{records:012d}'),'label':'approved-model',
                   'rules':incumbent['rules'],'records':records,'policy_sha256':frozen.sha256,'weights_sha256':frozen.identity['weights_sha256']}
        assessment={'version':'graygo/promotion-policy-assessment/3','promotion_policy':HEAD_TO_HEAD_HISTORY,
                    'records':records,'candidate':candidate,'incumbent':incumbent,'decision':{'status':'complete','numeric_pass':True,
                        'head_win_score':{'mean':.75,'lower':.6},'aggregate':{'lower':-.12}}}
        return campaign,stage,publication,assessment

    def test_publication_preserves_checkpoint_and_adaptive_state_and_can_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            campaign,stage,publication,assessment=self.fixture_campaign(Path(directory))
            checkpoint=stage/'learner/resume/000000000008/manifest.json';saved=checkpoint.read_bytes()
            before=json.loads((stage/'learner/league/state.json').read_text())['body']
            with patch('tools.adopt_head_to_head_policy.reassess',return_value=assessment):
                receipt=apply(campaign,8,reserve_bytes=0)
                with self.assertRaisesRegex(ValueError,'different current champion'):
                    apply(campaign,8,reserve_bytes=0)
            after=json.loads((stage/'learner/league/state.json').read_text())['body']
            self.assertEqual(checkpoint.read_bytes(),saved)
            self.assertEqual(after['history'][-1],before['champion'])
            self.assertEqual(after['adaptive_scheduler']['gate_count'],before['adaptive_scheduler']['gate_count'])
            self.assertEqual(after['adaptive_scheduler']['interval_seconds'],1800)
            self.assertEqual(json.loads(publication.read_text())['body']['champion'],assessment['candidate'])
            self.assertFalse(receipt['pool_gain_required'])
            self.assertEqual(run(campaign,max_segments=1,reserve_bytes=0)['completed_segments'],1)

    def test_changing_incumbent_retires_an_old_gate_without_reusing_its_slot(self):
        with tempfile.TemporaryDirectory() as directory:
            campaign,stage,publication,assessment=self.fixture_campaign(Path(directory),at_gate=True)
            before=json.loads((stage/'learner/league/state.json').read_text())['body']
            with patch('tools.adopt_head_to_head_policy.reassess',return_value=assessment):apply(campaign,16,reserve_bytes=0)
            after=json.loads((stage/'learner/league/state.json').read_text())['body']
            self.assertIsNone(after['pending'])
            self.assertEqual(after['round'],before['round']+1)
            self.assertEqual(after['rounds'][-1]['decision']['reason'],'incumbent_changed_on_policy_adoption')
            self.assertFalse(after['rounds'][-1]['promoted'])
            self.assertEqual(run(campaign,max_segments=1,reserve_bytes=0)['completed_segments'],1)


if __name__=='__main__':unittest.main()
