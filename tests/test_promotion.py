from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from graygo.confidence import ConfidenceFamily,canonical_sha
from graygo.engine import Rules
from graygo.evaluation import BaselinePolicy,EmptyStarts,EvaluationPlan,EvaluationRun
from graygo.promotion import ProtectedPool,PromotionProtocol,PairedEvidence,aligned_evidence,decision_from_evidence,agent_signature
from graygo.promotion_calibration import PilotPopulation,calibrate,qualified_report,scenario_definitions


class PromotionTests(unittest.TestCase):
    def protocol(self,looks=(500,2000)):
        return PromotionProtocol(ProtectedPool('protected',1,(('a','a'*64),('b','b'*64)),(.5,.5)),looks)

    def test_all_numerical_requirements_are_checked_jointly(self):
        p=self.protocol();n=1000
        passed=p.numerical_gate(np.full((n,2),.9),np.full((n,2),.1),2000)
        self.assertTrue(passed['numerical_pass'])
        full=PromotionProtocol(ProtectedPool('ten',1,tuple((str(j),f'{j:064x}') for j in range(10)),tuple([.1]*10)))
        extreme=full.numerical_gate(np.ones((1000,10)),-np.ones((1000,10)),2000)
        self.assertTrue(extreme['numerical_pass'])
        zero=p.numerical_gate(np.full((n,2),.4),np.full((n,2),.4),2000)
        self.assertFalse(zero['criteria']['aggregate_improvement'])
        c=np.tile([.1,.9],(n,1));i=np.tile([.3,0.],(n,1))
        regression=p.numerical_gate(c,i,2000)
        self.assertTrue(regression['criteria']['aggregate_improvement']);self.assertFalse(regression['criteria']['per_opponent_regression']);self.assertTrue(regression['criteria']['absolute_floor'])
        floor=p.numerical_gate(np.tile([-.2,.8],(n,1)),np.tile([-.4,0.],(n,1)),2000)
        self.assertTrue(floor['criteria']['aggregate_improvement']);self.assertTrue(floor['criteria']['per_opponent_regression']);self.assertFalse(floor['criteria']['absolute_floor'])

    def test_weights_and_variance_use_aligned_pair_differences(self):
        p=self.protocol();n=1000
        c=np.tile([[.8,.0],[.0,.8]],(n//2,1));i=np.zeros_like(c)
        result=p.numerical_gate(c,i,2000)
        self.assertLess(result['aggregate']['sample_variance'],1e-30)
        self.assertGreater(result['opponents'][0]['improvement']['sample_variance'],0.)
        self.assertTrue(result['numerical_pass'])

    def test_policy_display_label_does_not_rebrand_same_weights(self):
        import torch
        from graygo.models import QNet,ModelConfig
        from graygo.evaluation import FrozenNeuralPolicy
        torch.manual_seed(3);model=QNet(ModelConfig(8,1,2))
        a=FrozenNeuralPolicy(model,representation='greedy_q',label='pilot')
        b=FrozenNeuralPolicy(model,representation='greedy_q',label='candidate')
        self.assertEqual(a.sha256,b.sha256)
        self.assertEqual(agent_signature(a.identity),agent_signature(b.identity))

    def test_protocol_is_frozen_and_pool_changes_require_new_identity(self):
        p=self.protocol();self.assertEqual(PromotionProtocol.from_dict(p.to_dict()),p)
        bad=deepcopy(p.to_dict());bad['regression_tolerance']=-.2
        with self.assertRaises(ValueError):PromotionProtocol.from_dict(bad)
        bad=deepcopy(p.to_dict());bad['confidence']['looks']=[500,2000,4000]
        with self.assertRaises(ValueError):PromotionProtocol.from_dict(bad)
        with self.assertRaises(ValueError):ProtectedPool('too-many',1,tuple((str(i),f'{i:064x}') for i in range(11)),tuple([1/11]*11))
        other=ProtectedPool('protected',2,p.pool.opponents,(.7,.3))
        self.assertNotEqual(other.sha256,p.pool.sha256)

    def make_evidence(self,root,*,candidate_family=False):
        subject=BaselinePolicy('tactical');incumbent=BaselinePolicy('random');opponents=[BaselinePolicy('random'),BaselinePolicy('tactical')]
        rules=Rules(3,2);pool=ProtectedPool('protected-real',1,tuple((str(j),p.sha256) for j,p in enumerate(opponents)),(.5,.5),rules)
        protocol=PromotionProtocol(pool)
        family=protocol.family if candidate_family else ConfidenceFamily('separate-pilot',('delta/0','delta/1'),(4,8))
        result={}
        for j,opponent in enumerate(opponents):
            plan=EvaluationPlan('pilot','pilot','delta/'+str(j),rules,5,family.looks,confidence_sha256=family.sha256,comparison='improvement',
                decision_protocol_sha256=protocol.sha256 if candidate_family else None,calibration_sha256='0'*64 if candidate_family else None)
            run=EvaluationRun(plan,Path(root)/str(j),subject,opponent,incumbent=incumbent,reserve_bytes=0)
            run.run(500 if candidate_family else 8,time_cap_seconds=30)
            result[str(j)]=PairedEvidence.load(Path(root)/str(j))
        return protocol,result

    def test_actual_sql_pilot_alignment_and_source_binding(self):
        with tempfile.TemporaryDirectory() as root:
            p,evidence=self.make_evidence(root)
            c,i,subject,incumbent=aligned_evidence(p,evidence,for_calibration=True)
            self.assertEqual(c.shape,(4,2));self.assertNotEqual(subject,incumbent)
            pilot=PilotPopulation.from_evidence(p,evidence);self.assertTrue(pilot.usable_for_promotion)
            with self.assertRaises(ValueError):aligned_evidence(p,evidence)
            bad=dict(evidence);bad['0']=deepcopy(bad['0']);bad['0'].rows[0]['utility']=1.
            with self.assertRaises(ValueError):aligned_evidence(p,bad,for_calibration=True)

    def test_numeric_reports_do_not_promote_without_calibration_and_readiness(self):
        with tempfile.TemporaryDirectory() as root:
            p,evidence=self.make_evidence(root,candidate_family=True)
            result=decision_from_evidence(p,evidence,500)
            self.assertFalse(result['promotion_permitted']);self.assertFalse(result['calibration_qualified']);self.assertFalse(result['prerequisites_proven'])
            self.assertEqual(set(result['evidence']),{'0','1'})


class CalibrationTests(unittest.TestCase):
    def protocol(self):return PromotionProtocol(ProtectedPool('cal',1,(('a','a'*64),('b','b'*64)),(.5,.5)),(500,2000,8000))
    def test_calibration_is_reproducible_resumable_and_never_qualifies_synthetic_pilot(self):
        p=self.protocol();pilot=PilotPopulation.synthetic(np.zeros((20,2)),np.zeros((20,2)))
        first,progress=calibrate(p,pilot,simulations=12,seed=9,time_cap_seconds=1e-12)
        self.assertEqual(first['status'],'not_calibrated');self.assertFalse(first['complete'])
        resumed,_=calibrate(p,pilot,simulations=12,seed=9,progress=progress,time_cap_seconds=30)
        direct,_=calibrate(p,pilot,simulations=12,seed=9,time_cap_seconds=30)
        self.assertEqual(resumed['scenarios'],direct['scenarios']);self.assertFalse(resumed['passed']);self.assertFalse(qualified_report(p,resumed))
        names={s['name'] for s in resumed['scenarios']}
        self.assertIn('uniform-plus-0.03',names);self.assertIn('nontransitive',names)
        self.assertIn('missing-a2',names);self.assertIn('missing-board-smoke',names)
        self.assertIn('regression-at/a',names);self.assertIn('regression-beyond/b',names);self.assertIn('floor-violation/a',names)
        with self.assertRaises(ValueError):calibrate(p,pilot,simulations=13,seed=9,progress=progress)

    def test_infeasible_useful_scenario_is_not_replaced_with_an_easier_one(self):
        p=self.protocol();pilot=PilotPopulation.synthetic(np.ones((20,2)),np.ones((20,2)))
        report,_=calibrate(p,pilot,simulations=2,seed=1,time_cap_seconds=30)
        useful=next(s for s in report['scenarios'] if s['name']=='uniform-plus-0.03')
        self.assertEqual(useful['status'],'infeasible');self.assertFalse(report['passed'])
        self.assertIn('incomplete or infeasible required calibration scenario',report['reasons'])

    def test_certificate_tampering_and_missing_data_fail_closed(self):
        p=self.protocol();self.assertFalse(qualified_report(p,None))
        pilot=PilotPopulation.synthetic(np.zeros((4,2)),np.zeros((4,2)))
        report,_=calibrate(p,pilot,simulations=2,seed=1,time_cap_seconds=30)
        bad=deepcopy(report);bad['passed']=True
        self.assertFalse(qualified_report(p,bad))
        self.assertEqual(report['pilot_pairs'],4);self.assertFalse(report['usable_for_promotion'])

if __name__=='__main__':unittest.main()
