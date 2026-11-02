from copy import deepcopy
from fractions import Fraction as F
import json
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
import torch
from graygo.engine import Rules,State,transform,transform_action,BLACK,WHITE
from graygo.evaluation import FrozenNeuralPolicy,EpisodeMixture
from graygo.models import QNet,PolicyNet,ModelConfig
from graygo.probe_benchmark import ProbeBenchmark,ProbeFamily,assess_marginals,construct_benchmark,frozen_marginals
from graygo.probe_certification import CertifiedRoot,certify_one_round
from graygo.probe_setup import replay_setup

FIXTURE=Path(__file__).parent/'fixtures/collision-benchmark-v1.json'


class ProbeBenchmarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.benchmark=ProbeBenchmark.from_dict(json.loads(FIXTURE.read_text()))
    def policies(self):
        return {seed:{f.identity:{'before':f.before.certificate.equilibrium,'after':f.after.certificate.equilibrium} for f in self.benchmark.families} for seed in (200,201,202)}

    def test_catalog_has_32_balanced_instances_64_roots_but_one_strategic_family(self):
        b=self.benchmark
        self.assertEqual(len(b.families),32);self.assertEqual(sum(f.direction==1 for f in b.families),16)
        self.assertEqual(len({r for f in b.families for r in f.root_ids}),64)
        self.assertEqual(ProbeBenchmark.from_dict(b.to_dict()).sha256,b.sha256)
        for family in b.families:
            self.assertEqual(family.before.state.rules,Rules());self.assertEqual(family.after.state.rules,Rules())
            self.assertEqual(family.before.state.rounds_played,199);self.assertEqual(family.after.state.rounds_played,199)
            self.assertEqual(replay_setup(family.before_setup),family.before.state);self.assertEqual(replay_setup(family.after_setup),family.after.state)
            self.assertEqual(len(family.before_setup),199);self.assertEqual(len(family.after_setup),199)
            self.assertTrue(family.to_dict()['collision_rule_sensitive'])
            self.assertEqual(sum(x!=y for x,y in zip(family.before.state.board,family.after.state.board)),1)
            a,b=family.before.certificate,family.after.certificate
            separation=b.lower-a.upper if family.direction==1 else a.lower-b.upper
            self.assertGreaterEqual(separation,F(1,10))

    def test_certificates_verify_without_any_floating_solver(self):
        with patch('graygo.probe_certification.solve_matrix',side_effect=AssertionError('floating LP must not verify proof')):
            result=ProbeBenchmark.from_dict(self.benchmark.to_dict())
        self.assertEqual(result.sha256,self.benchmark.sha256)

    def test_incomplete_tampered_and_symmetry_duplicate_benchmarks_are_rejected(self):
        b=self.benchmark;data=b.to_dict();data['cases'].pop()
        with self.assertRaisesRegex(ValueError,'32-case'):ProbeBenchmark.from_dict(data)
        data=b.to_dict();data['required_passes_per_seed']=23
        with self.assertRaisesRegex(ValueError,'requirements'):ProbeBenchmark.from_dict(data)
        data=b.to_dict();data['cases'][0]['direction']*=-1
        with self.assertRaisesRegex(ValueError,'direction'):ProbeBenchmark.from_dict(data)
        first=b.families[0];params={'rotation':1,'reflect':True,'shift':(1,2),'swap':True}
        before=certify_one_round(transform(first.before.state,**params));after=certify_one_round(transform(first.after.state,**params))
        point=transform_action(first.changed_point,first.before.state.rules,rotation=1,reflect=True,shift=(1,2))
        def moved_trace(trace):
            result=[]
            for black,white in trace:
                def move(action):return transform_action(action,first.before.state.rules,rotation=1,reflect=True,shift=(1,2))
                result.append((move(white),move(black)))
            return tuple(result)
        copied=ProbeFamily(before,after,point,b.families[1].attempt,moved_trace(first.before_setup),moved_trace(first.after_setup))
        families=list(b.families);families[1]=copied
        with self.assertRaisesRegex(ValueError,'symmetry copies'):ProbeBenchmark(b.seed,tuple(families))

    def test_instance_counts_cannot_satisfy_independent_family_gate(self):
        policies=self.policies();initial=assess_marginals(self.benchmark,policies)
        self.assertTrue(initial['all_seed_instance_thresholds_passed']);self.assertFalse(initial['all_seed_criterion_passed'])
        self.assertEqual(initial['distinct_strategic_families'],1);self.assertFalse(initial['family_coverage_complete'])
        self.assertEqual([r['passed_instances'] for r in initial['seeds']],[32,32,32])
        for seed,failures in ((201,8),(202,9)):
            for family in self.benchmark.families[:failures]:
                pair={}
                for name,root in (('before',family.before),('after',family.after)):
                    values=[]
                    for actions in (root.certificate.black_actions,root.certificate.white_actions):
                        vector=[F(0)]*len(actions);vector[actions.index(root.state.rules.pass_action)]=F(1);values.append(tuple(vector))
                    pair[name]=tuple(values)
                policies[seed][family.identity]=pair
        result=assess_marginals(self.benchmark,policies)
        self.assertEqual([r['passed_instances'] for r in result['seeds']],[32,24,23])
        self.assertEqual([r['instance_threshold_passed'] for r in result['seeds']],[True,True,False])
        self.assertFalse(result['all_seed_criterion_passed']);self.assertEqual([r['passed_families'] for r in result['seeds']],[1,0,0]);self.assertFalse(result['gray_specialization_demonstrated'])
        self.assertFalse(result['trained_policy_provenance_verified'])

    def test_forged_setup_trace_cannot_masquerade_as_a_reachable_root(self):
        data=self.benchmark.to_dict();data['cases'][0]['before_setup'][-1]=[81,81]
        with self.assertRaisesRegex(ValueError,'setup trace'):ProbeBenchmark.from_dict(data)
        data=self.benchmark.to_dict();data['cases'][0]['collision_rule_sensitive']=False
        with self.assertRaisesRegex(ValueError,'proof, direction'):ProbeBenchmark.from_dict(data)

    def test_missing_seed_or_family_cannot_be_dropped(self):
        policies=self.policies();del policies[202]
        with self.assertRaisesRegex(ValueError,'three fixed'):assess_marginals(self.benchmark,policies)
        policies=self.policies();del policies[202][self.benchmark.families[-1].identity]
        with self.assertRaisesRegex(ValueError,'every original'):assess_marginals(self.benchmark,policies)

    def test_bounded_constructor_preserves_incomplete_and_interrupted_status(self):
        benchmark,report=construct_benchmark(473927,attempt_cap=20,time_cap_seconds=30)
        self.assertIsNone(benchmark);self.assertFalse(report['benchmark_complete']);self.assertEqual(report['status'],'attempt_limited')
        self.assertEqual(report['attempts'],20)
        benchmark,report=construct_benchmark(473927,attempt_cap=20,time_cap_seconds=30,stop_requested=lambda:True)
        self.assertIsNone(benchmark);self.assertEqual(report['status'],'interrupted');self.assertEqual(report['attempts'],0)

    def test_private_episode_mixture_marginals_use_public_inputs_without_resampling(self):
        torch.set_num_threads(1);q=QNet(ModelConfig(8,1,2));average=PolicyNet(ModelConfig(8,1,2))
        with torch.no_grad():
            for model in (q,average):
                for parameter in model.parameters():parameter.zero_()
            q.head.pass_head[-1].bias.fill_(.5)
        a=FrozenNeuralPolicy(q,representation='greedy_q');b=FrozenNeuralPolicy(average)
        mixture=EpisodeMixture((a,b),(.25,.75));root=self.benchmark.families[0].before.state
        torch_rng=torch.get_rng_state().clone();p=mixture.initial_distribution(root,1,fresh_episode=True)
        legal=root.legal_actions(1);expected=np.zeros(82);expected[list(legal)]=.75/len(legal);expected[81]+=.25
        np.testing.assert_allclose(p,expected,rtol=0,atol=1e-15)
        torch.testing.assert_close(torch.get_rng_state(),torch_rng,rtol=0,atol=0)
        self.assertFalse(p[[a for a in range(82) if a not in legal]].any())
        values,bindings=frozen_marginals(self.benchmark,{200:mixture,201:mixture,202:mixture})
        self.assertEqual(set(values),{200,201,202});self.assertEqual(len(values[200]),32);self.assertEqual(bindings[200]['sha256'],mixture.sha256)
        prefix=State(Rules(3,3))
        from graygo.engine import resolve
        prefix=resolve(prefix,0,1).state
        with self.assertRaisesRegex(ValueError,'prefix-conditioned'):mixture.initial_distribution(prefix,1)

if __name__=='__main__':unittest.main()
