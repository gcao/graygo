from pathlib import Path
from types import SimpleNamespace
import json
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from graygo.confidence import canonical_sha
from tools.champion_gate import GatePlan,POSITIVE_POOL_GAIN,LEGACY_POOL_GAIN,pool_gain_passes
from tools.adopt_positive_gain_policy import reassess,apply
from tools.continuous_training import create,run
from tools.learning_loop import worker
from graygo.evaluation import FrozenNeuralPolicy
from test_continuous_training import parent_run


def inputs():
    n=1000;head=np.ones(n);head[:74]=0
    random=np.ones(n);old_random=np.ones(n);old_random[:12]=0;old_random[12]=.5
    tactical=np.ones(n);tactical[:9]=0
    old_tactical=np.ones(n);old_tactical[:287]=0;old_tactical[287]=.5
    return head,{'random':random,'tactical':tactical},{'random':old_random,'tactical':old_tactical}


class PositiveGainPolicyTests(unittest.TestCase):
    def test_positive_supported_improvement_passes_without_relaxing_other_guards(self):
        values=inputs();new=GatePlan(74,alpha=.025,pool_gain_policy=POSITIVE_POOL_GAIN)
        old=GatePlan(74,alpha=.025,improvement=.03,pool_gain_policy=LEGACY_POOL_GAIN)
        current=new.numeric(24,*values,2000);original=old.numeric(24,*values,2000)
        self.assertGreater(current['aggregate']['lower'],0);self.assertLess(current['aggregate']['lower'],.03)
        self.assertTrue(current['numeric_pass']);self.assertFalse(original['criteria']['pool_gain'])
        for field in ('head','aggregate','opponents'):self.assertEqual(current[field],original[field])
        self.assertEqual(current['family_sha256'],original['family_sha256'])
        self.assertEqual(new.regression,old.regression);self.assertEqual(new.floor,old.floor)

    def test_zero_and_negative_pool_lower_bounds_still_fail(self):
        plan=GatePlan(1,pool_gain_policy=POSITIVE_POOL_GAIN)
        self.assertFalse(pool_gain_passes(0,plan));self.assertFalse(pool_gain_passes(-.001,plan))
        self.assertTrue(pool_gain_passes(.00001,plan))

    def test_reassessment_reproduces_old_statistics_and_preserves_original_decision(self):
        values=inputs();old=GatePlan(74,alpha=.025,improvement=.03,pool_gain_policy=LEGACY_POOL_GAIN)
        decision=old.numeric(24,*values,2000)
        with tempfile.TemporaryDirectory() as directory:
            stage=Path(directory);root=stage/'learner/league/evaluations/0024';root.mkdir(parents=True)
            candidate={'policy_sha256':'candidate'};incumbent={'policy_sha256':'incumbent'}
            row={'index':24,'records':7340032,'candidate':candidate,'incumbent':incumbent,'decision':decision}
            body={'rounds':[row],'contract':{'loop_plan':{'round_ends':list(range(74))}}}
            (stage/'learner/league/state.json').write_text(json.dumps({'body':body,'sha256':canonical_sha(body)}))
            config={'gate_alpha':.025,'gate_seed':73};(stage/'plan.json').write_text(json.dumps({'body':config,'sha256':canonical_sha(config)}))
            family=old.family(24,('random','tactical'))
            (root/'gate-plan.json').write_text(json.dumps({'family':family.to_dict()}))
            evidence={}
            for name in ('head','random','tactical'):
                if name=='head':rows=[{'subject':[{'subject_utility':1},{'subject_utility':2*x-1}]} for x in values[0]]
                else:rows=[{'utility':float(x),'incumbent_utility':float(y)} for x,y in zip(values[1][name],values[2][name])]
                evidence[name]=SimpleNamespace(sha256='evidence-'+name,available_games=2000,rows=rows,
                    contract={'subject_sha256':'candidate','opponent_sha256':'incumbent','incumbent_sha256':'incumbent'},assert_current=lambda:None)
            decision.update(status='complete',evidence={name:e.sha256 for name,e in evidence.items()})
            (stage/'learner/league/state.json').write_text(json.dumps({'body':body,'sha256':canonical_sha(body)}))
            (root/'decision.json').write_text(json.dumps(decision));original=(root/'decision.json').read_bytes()
            with patch('tools.adopt_positive_gain_policy.FrozenEvaluationEvidence.load',side_effect=lambda p:evidence[p.name]),patch('tools.adopt_positive_gain_policy.policy',return_value=SimpleNamespace(assert_frozen=lambda:None)):
                report=reassess(stage,7340032)
                self.assertTrue(report['decision']['numeric_pass']);self.assertFalse(report['owner_approved'])
                self.assertEqual((root/'decision.json').read_bytes(),original)
                evidence['head'].available_games=1998
                with self.assertRaisesRegex(ValueError,'incomplete'):reassess(stage,7340032)

    def test_approved_publication_updates_history_and_keeps_the_current_learner(self):
        checkout=Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory);parent,publication=parent_run(base);campaign=base/'campaign'
            create(campaign,parent,checkout,publication,project=base,segment_records=16,block=8,reserve_bytes=0)
            def interrupted(stage,**kwargs):kwargs['stop_state'].append(True);return worker(stage,**kwargs)
            with patch('tools.continuous_training.worker',side_effect=interrupted):run(campaign,max_segments=1,reserve_bytes=0)
            stage=campaign/'stages/000000';state_path=stage/'learner/league/state.json'
            wrapped=json.loads(state_path.read_text());state=wrapped['body'];old=state['champion']
            frozen=FrozenNeuralPolicy.from_archive(stage/'learner/weights/000000000008',label='approved-fixture')
            candidate={'kind':'archive','archive':str(stage/'learner/weights/000000000008'),'label':'approved-fixture',
                       'rules':old['rules'],'records':8,'policy_sha256':frozen.sha256,'weights_sha256':frozen.identity['weights_sha256']}
            state['pending']={'index':0,'target':16,'status':'training','champion':old,'history':[]}
            wrapped['sha256']=canonical_sha(state);state_path.write_text(json.dumps(wrapped))
            checkpoint=stage/'learner/resume/000000000008/manifest.json';original=checkpoint.read_bytes()
            assessment={'version':'graygo/promotion-policy-assessment/2','promotion_policy':POSITIVE_POOL_GAIN,'records':8,
                'candidate':candidate,'incumbent':old,'decision':{'numeric_pass':True,'head_win_score':{'mean':1.,'lower':.9},'aggregate':{'lower':.01}}}
            with patch('tools.adopt_positive_gain_policy.reassess',return_value=assessment):receipt=apply(campaign,8,reserve_bytes=0)
            self.assertEqual(checkpoint.read_bytes(),original)
            state=json.loads(state_path.read_text())['body'];published=json.loads(publication.read_text())['body']
            self.assertEqual(published['champion']['policy_sha256'],candidate['policy_sha256'])
            self.assertEqual(state['pending']['champion']['policy_sha256'],candidate['policy_sha256'])
            self.assertEqual(state['prior_promotions'],1);self.assertEqual(state['history'][-1],old)
            self.assertTrue(Path(receipt['assessment']).exists())
            self.assertEqual(run(campaign,max_segments=1,reserve_bytes=0)['completed_segments'],1)


if __name__=='__main__':unittest.main()
