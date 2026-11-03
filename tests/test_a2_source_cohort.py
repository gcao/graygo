"""Synthetic grading tests establish rules, never real source admission."""
from copy import deepcopy
from pathlib import Path
import json
import ast
import inspect
import tempfile
import unittest
from tests import test_a2_cohort as legacy_fixtures
from tools.a2_cohort import NFSPA2Plan,assess,canonical,FAMILIES
from tools.a2_source_cohort import (assess_sources,protocol_key,path_for,read_registry,assess_registry_files,REGISTRY_VERSION)
from tools import task_source_bridge,terminal_batch_admission
from tools import a2_source_cohort


class SourceAwareGradingTests(unittest.TestCase):
    def setUp(self):
        self.fixture=legacy_fixtures.A2ProtocolTests();self.a=self.fixture.plan();self.b=NFSPA2Plan('fast','c'*64,self.a.collector_sha256,dict(self.a.auditors),self.a.multiround_reuse)
        self.plans={p.sha256:p for p in (self.a,self.b)};self.allowed={self.a.learner_source_sha256:set(FAMILIES),self.b.learner_source_sha256:{'asymmetric','rps'}}

    def test_same_source_all_legacy_numerical_fields_and_inputs_are_unchanged(self):
        cases=self.fixture.cases(self.a);before=deepcopy(cases);old=assess(self.a,cases);new=assess_sources(self.a,self.plans,self.allowed,cases)
        for key,value in old.items():
            if key not in ('version','plan_sha256','learner_source_sha256'):self.assertEqual(new[key],value,key)
        self.assertEqual(cases,before);self.assertFalse(new['a2_complete']);self.assertTrue(new['source_identities_preserved'])

    def test_aggregation_code_differs_only_in_receiving_checked_original_cases(self):
        expected=ast.parse(inspect.getsource(assess)).body[0];actual=ast.parse(inspect.getsource(a2_source_cohort._aggregate_verified)).body[0]
        expected.name='_aggregate_verified';expected.args.args[1].arg='verified'
        loop=next(n for n in expected.body if isinstance(n,ast.For));self.assertEqual(ast.unparse(loop.body[0].value),'checked_case(plan, wrapper)')
        loop.body.pop(0);loop.target=ast.Tuple(elts=[ast.Name(id=n,ctx=ast.Store()) for n in ('key','body','summary')],ctx=ast.Store());loop.iter=ast.Name(id='verified',ctx=ast.Load())
        self.assertEqual(ast.dump(expected,include_attributes=False),ast.dump(actual,include_attributes=False))

    def mixed(self):
        cases=self.fixture.cases(self.a)
        for index,case in enumerate(cases):
            family,learner,seed=case['body']['case']
            if family in ('asymmetric','rps') and seed in (1,2):cases[index]=self.fixture.case(self.b,family,learner,seed)
        return cases

    def test_mixed_source_pass_preserves_every_original_plan_source_and_checksum(self):
        cases=self.mixed();before=deepcopy(cases);result=assess_sources(self.a,self.plans,self.allowed,cases)
        self.assertTrue(result['neural']['numeric_passed']);self.assertTrue(result['controls']['numeric_passed'])
        self.assertEqual(cases,before)
        for case in cases:
            key='/'.join(map(str,case['body']['case']));origin=result['case_provenance'][key]
            self.assertEqual(origin['original_case_sha256'],case['sha256']);self.assertEqual(origin['original_plan_sha256'],case['body']['plan_sha256'])
            self.assertEqual(origin['original_source_sha256'],case['body']['provenance']['learner_source_sha256'])

    def test_extension_keeps_original_hashes_across_implementations(self):
        cases=self.mixed();case=self.fixture.find(cases,'rps',seed=2);self.fixture.mutate(case,lambda b:b['evaluations'][-5].update(max_average_gap=.06))
        result=assess_sources(self.a,self.plans,self.allowed,cases);e=result['marginal_extension'];self.assertTrue(e['numeric_eligible']);self.assertFalse(e['selected'])
        self.assertEqual(e['original_case_sha256'],[self.fixture.find(cases,'rps',seed=s)['sha256'] for s in (0,1,2)])
        self.fixture.mutate(case,lambda b:b['evaluations'][-1].update(max_average_gap=.051))
        self.assertFalse(assess_sources(self.a,self.plans,self.allowed,cases)['marginal_extension']['numeric_eligible'])

    def test_duplicate_seed_or_root_cannot_be_replaced_by_another_source(self):
        cases=[self.fixture.case(self.a,'rps','neural',1),self.fixture.case(self.b,'rps','neural',1)]
        with self.assertRaises(ValueError):assess_sources(self.a,self.plans,self.allowed,cases)
        cases=[self.fixture.case(self.a,'rps','neural',0),self.fixture.case(self.b,'asymmetric','neural',1)]
        self.fixture.mutate(cases[1],lambda b:b.update(root=cases[0]['body']['root']))
        with self.assertRaises(ValueError):assess_sources(self.a,self.plans,self.allowed,cases)

    def test_unregistered_source_family_recipe_or_diagnostic_is_rejected(self):
        with self.assertRaises(ValueError):assess_sources(self.a,{self.a.sha256:self.a},self.allowed,[self.fixture.case(self.b,'rps','neural',1)])
        with self.assertRaises(ValueError):assess_sources(self.a,self.plans,self.allowed,[self.fixture.case(self.b,'delayed','neural',1)])
        other=NFSPA2Plan('changed','d'*64,self.a.collector_sha256,dict(self.a.auditors),4)
        with self.assertRaises(ValueError):assess_sources(self.a,{self.a.sha256:self.a,other.sha256:other},self.allowed,[])
        case=self.fixture.case(self.a,'rps','neural',0);self.fixture.mutate(case,lambda b:b.update(primary_recipe=False))
        with self.assertRaises(ValueError):assess_sources(self.a,self.plans,self.allowed,[case])

    def test_budget_limited_case_does_not_become_a_convergence_miss(self):
        case=self.fixture.case(self.b,'rps','neural',1);self.fixture.mutate(case,lambda b:b.update(status='budget_limited',records=10000000,fresh_audit_verified=False,evaluations=b['evaluations'][:20]))
        result=assess_sources(self.a,self.plans,self.allowed,[case]);self.assertEqual(result['cases']['rps/neural/1']['status'],'budget_limited');self.assertEqual(result['neural']['case_misses'],[])

    def test_admission_paths_cannot_cross_task_scopes(self):
        edges=[{'left':'a','right':'b','families':{'rps'}},{'left':'b','right':'c','families':{'asymmetric'}}]
        self.assertEqual(path_for(edges,'b','a','rps'),[0]);self.assertIsNone(path_for(edges,'c','a','rps'))

    def test_archived_registry_can_reanalyze_without_claiming_current_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);runtime='e'*64
            baseproof={'version':task_source_bridge.VERSION,'status':'qualified_cpu_features_A2_source_bridge','source_compatibility_qualified':True,
                'families':list(FAMILIES),'reference_source_sha256':self.a.learner_source_sha256,'candidate_source_sha256':runtime,'verifier_sha256':'f'*64,'execution_authorized':False}
            fastproof={'version':terminal_batch_admission.VERSION,'status':'qualified_cpu_matrix_terminal_batch','collection_implementation':'terminal-batch',
                'families':['asymmetric','rps'],'reference_source_sha256':runtime,'candidate_source_sha256':self.b.learner_source_sha256,'verifier_sha256':'f'*64,'execution_authorized':False}
            body={'version':REGISTRY_VERSION,'canonical_plan_sha256':self.a.sha256,'protocol_sha256':protocol_key(self.a),'target_runtime_source_sha256':runtime,
                'plans':[{'path':str(root/('absent-'+str(i))),'file_sha256':'f'*64,'plan_sha256':p.sha256,'plan':p.to_dict()} for i,p in enumerate((self.a,self.b))],
                'bridges':[{'path':str(root/('bridge-'+str(i))),'file_sha256':'f'*64,'proof':p} for i,p in enumerate((baseproof,fastproof))],
                'original_case_and_source_identity_rewriting_allowed':False,'a2_complete':False,'execution_authorized':False}
            registry=root/'registry.json';registry.write_text(json.dumps({'body':body,'sha256':canonical(body)}));case=root/'case.json';case.write_text(json.dumps(self.fixture.case(self.b,'rps','neural',1)))
            report=assess_registry_files(registry,[case],repo=tmp,current=False);self.assertEqual(report['present_cases'],1)
            self.assertFalse(report['current_input_snapshots_verified']);self.assertFalse(report['current_source_admission_inputs_verified']);self.assertFalse(report['a2_complete'])
            with self.assertRaises(OSError):read_registry(registry,repo=tmp,current=True)
