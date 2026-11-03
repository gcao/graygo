from copy import deepcopy
from contextlib import redirect_stdout,redirect_stderr
import io
import json
from pathlib import Path
import tempfile
import unittest
import torch
from graygo.manifest import source_digest
from graygo.nfsp_training import NFSPMatrixConfig,NFSPMatrixTrainer
from graygo.task_training import NFSPTaskConfig,NFSPTaskTrainer
from tools.a2_cohort import NFSPA2Plan,assess,audit_case,assess_files,canonical,digest,FAMILIES,SEEDS,LEARNERS,AUDIT_FILES,main

TOOL=Path(__file__).resolve().parents[1]/'tools/a2_cohort.py'


class A2ProtocolTests(unittest.TestCase):
    def plan(self):return NFSPA2Plan('synthetic','a'*64,'b'*64,{k:str(i+1)*64 for i,k in enumerate(sorted(AUDIT_FILES))})
    def case(self,plan,family,learner,seed):
        key=[family,learner,seed];kind='matrix' if family in ('asymmetric','rps') else ('delayed' if family=='delayed' else 'bounded')
        snapshot={'files':{'run.json':canonical(key)},'directories':[]}
        body={'version':'graygo/nfsp-a2-case/1','case':key,'root':'/synthetic/'+family+'/'+learner+'/'+str(seed),
            'plan_sha256':plan.sha256,'status':'complete','records':15000000,'configured_records':15000000,'evaluation_interval':500000,
            'primary_recipe':True,'fresh_audit_verified':True,'input_snapshot':snapshot,
            'provenance':{'learner_source_sha256':plan.learner_source_sha256,'collector_sha256':plan.collector_sha256,
                'auditor_sha256':dict(plan.auditors)[kind],'shared_helper_sha256':dict(plan.auditors)['delayed'] if kind=='bounded' else None,
                'training_manifest_sha256':canonical(key),'final_checkpoint_manifest_sha256':canonical([key,'checkpoint']),
                'input_snapshot_sha256':canonical(snapshot)},
            'evaluations':[{'records':n,'average_ready':n>5000000,'max_average_gap':.01,'max_greedy_gap':.01,
                            'max_early_error':None if family in ('asymmetric','rps') else .01} for n in range(500000,15000001,500000)]}
        return {'body':body,'sha256':canonical(body)}
    def cases(self,plan):return [self.case(plan,f,l,s) for f in FAMILIES for l in LEARNERS for s in SEEDS]
    def mutate(self,case,fn):fn(case['body']);case['sha256']=canonical(case['body'])
    def find(self,cases,family='delayed',learner='neural',seed=0):return next(c for c in cases if c['body']['case']==[family,learner,seed])

    def test_strict_plan_roundtrip_cannot_redefine_families_seeds_or_thresholds(self):
        p=self.plan();self.assertEqual(NFSPA2Plan.from_dict(p.to_dict()),p)
        for field,value in (('records',5000000),('seeds',[1,2,3]),('families',['asymmetric']),('target','uncorrected')):
            data=p.to_dict();data[field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):NFSPA2Plan.from_dict(data)
        with self.assertRaises(ValueError):NFSPA2Plan('bad','a'*64,'b'*64,dict(p.auditors),8)
        data=p.to_dict();data['extension']['maximum_relative_miss']=1.3
        with self.assertRaises(ValueError):NFSPA2Plan.from_dict(data)

    def test_synthetic_full_numeric_pass_never_grants_a2_or_training_authority(self):
        p=self.plan();r=assess(p,self.cases(p));self.assertEqual(r['required_cases'],36)
        self.assertTrue(r['neural']['numeric_passed']);self.assertTrue(r['controls']['numeric_passed'])
        self.assertEqual(r['neural_deployment_band'],'all_neural_numeric_gates_pass')
        for key in ('a2_complete','pilot_authorized','extension_authorized','resource_scope_verified'):self.assertFalse(r[key])

    def test_missing_controls_or_original_seed_remain_incomplete_without_seed_filtering(self):
        p=self.plan();cases=self.cases(p);r=assess(p,[c for c in cases if c['body']['case'][1]=='neural'])
        self.assertTrue(r['neural']['numeric_passed']);self.assertFalse(r['controls']['complete']);self.assertEqual(len(r['missing_cases']),18)
        r=assess(p,[c for c in cases if c['body']['case']!=['rps','neural',2]])
        self.assertFalse(r['neural']['complete']);self.assertIn('rps/neural/2',r['missing_cases']);self.assertFalse(r['marginal_extension']['numeric_eligible'])

    def test_average_lag_is_separate_from_credit_and_cannot_sacrifice_a_mixed_gate(self):
        p=self.plan();cases=self.cases(p)
        for seed in SEEDS:
            self.mutate(self.find(cases,seed=seed),lambda b:[row.update(max_average_gap=.25) for row in b['evaluations'][-5:]])
        r=assess(p,cases);self.assertEqual(r['neural_deployment_band'],'averaging_lag')
        self.assertTrue(r['cases']['delayed/neural/0']['q_credit_passed'])
        self.mutate(self.find(cases,'rps'),lambda b:b['evaluations'][-1].update(max_average_gap=.051))
        self.assertEqual(assess(p,cases)['neural_deployment_band'],'not_ready')
        self.mutate(self.find(cases),lambda b:b['evaluations'][-1].update(max_average_gap=.251))
        self.assertEqual(assess(p,cases)['cases']['delayed/neural/0']['average_band'],'above_0.25')

    def test_good_average_does_not_erase_failed_delayed_credit(self):
        p=self.plan();cases=self.cases(p);self.mutate(self.find(cases),lambda b:b['evaluations'][-1].update(max_early_error=.322))
        r=assess(p,cases);self.assertFalse(r['cases']['delayed/neural/0']['q_credit_passed'])
        self.assertEqual(r['cases']['delayed/neural/0']['average_band'],'at_most_0.10');self.assertFalse(r['marginal_extension']['numeric_eligible'])

    def test_exact_twenty_percent_miss_can_only_propose_all_three_original_seeds(self):
        p=self.plan();cases=self.cases(p);self.mutate(self.find(cases,'rps'),lambda b:b['evaluations'][-5].update(max_average_gap=.06))
        e=assess(p,cases)['marginal_extension'];self.assertTrue(e['numeric_eligible']);self.assertEqual(e['family'],'rps')
        self.assertEqual(e['paired_seeds'],[0,1,2]);self.assertEqual(e['required_endpoint'],17000000)
        self.assertEqual(len(e['original_case_sha256']),3);self.assertFalse(e['selected']);self.assertFalse(e['resource_authorization_verified'])

    def test_late_failure_large_excursion_multiple_seeds_or_families_cannot_extend(self):
        p=self.plan()
        for variant in ('latest','large','two-seeds','two-families','uninitialized'):
            cases=self.cases(p);case=self.find(cases,'rps');self.mutate(case,lambda b:b['evaluations'][-5].update(max_average_gap=.06))
            if variant=='latest':self.mutate(case,lambda b:b['evaluations'][-1].update(max_average_gap=.051))
            elif variant=='large':self.mutate(case,lambda b:b['evaluations'][-5].update(max_average_gap=.0601))
            elif variant=='two-seeds':self.mutate(self.find(cases,'rps',seed=1),lambda b:b['evaluations'][-5].update(max_average_gap=.06))
            elif variant=='two-families':self.mutate(self.find(cases,'asymmetric'),lambda b:b['evaluations'][-5].update(max_average_gap=.06))
            else:self.mutate(case,lambda b:b['evaluations'][-5].update(average_ready=False))
            with self.subTest(variant=variant):self.assertFalse(assess(p,cases)['marginal_extension']['numeric_eligible'])

    def test_control_miss_is_reported_separately_from_candidate_neural_requirements(self):
        p=self.plan();cases=self.cases(p);self.mutate(self.find(cases,learner='tabular'),lambda b:b['evaluations'][-1].update(max_early_error=1.1))
        r=assess(p,cases);self.assertTrue(r['neural']['numeric_passed']);self.assertFalse(r['controls']['numeric_passed'])
        self.assertTrue(r['control_interpretation_required']);self.assertFalse(r['a2_complete'])

    def test_wrong_source_aliases_and_hidden_evaluation_inventory_are_rejected(self):
        p=self.plan()
        for variant in ('source','auditor','alias','missing-look','later-look','extension'):
            cases=self.cases(p);case=cases[0]
            if variant=='source':self.mutate(case,lambda b:b['provenance'].update(learner_source_sha256='c'*64))
            elif variant=='auditor':self.mutate(case,lambda b:b['provenance'].update(auditor_sha256='c'*64))
            elif variant=='alias':self.mutate(cases[1],lambda b:b.update(root=case['body']['root']))
            elif variant=='missing-look':self.mutate(case,lambda b:b['evaluations'].pop(-2))
            elif variant=='later-look':self.mutate(case,lambda b:b['evaluations'].append(dict(b['evaluations'][-1],records=15500000)))
            else:self.mutate(case,lambda b:b.update(records=17000000))
            with self.subTest(variant=variant),self.assertRaises(ValueError):assess(p,cases)
        with self.assertRaises(ValueError):assess(p,[self.cases(p)[0]]*2)

    def test_time_capped_or_diagnostic_case_is_incomplete_not_a_convergence_miss(self):
        p=self.plan();case=self.case(p,'delayed','neural',0)
        self.mutate(case,lambda b:b.update(status='budget_limited',records=10000000,fresh_audit_verified=False,evaluations=b['evaluations'][:20]))
        r=assess(p,[case]);self.assertEqual(r['cases']['delayed/neural/0']['status'],'budget_limited')
        self.assertEqual(r['neural']['case_misses'],[]);self.assertFalse(r['neural']['complete'])
        self.mutate(case,lambda b:b.update(primary_recipe=False))
        self.assertEqual(assess(p,[case])['cases']['delayed/neural/0']['status'],'diagnostic_only')

    def test_primary_average_cannot_be_rebranded_as_initialized_before_opening(self):
        p=self.plan();case=self.case(p,'rps','neural',0)
        self.mutate(case,lambda b:b['evaluations'][0].update(average_ready=True))
        with self.assertRaisesRegex(ValueError,'fixed opening'):assess(p,[case])


class A2SnapshotTests(unittest.TestCase):
    def plan(self):return NFSPA2Plan('actual-diagnostic',source_digest(),digest(TOOL),{k:digest(TOOL.with_name(v)) for k,v in AUDIT_FILES.items()})
    def fixture(self,root,family='asymmetric',pause=False):
        common=dict(task=family,learner='tabular',seed=0,records=64,eval_every=16,batch=8,learning_start=8,capacity=16,
                    target_every=16,reservoir_open=8,handoff_end=32,epsilon_decay=32)
        if family in ('asymmetric','rps'):t=NFSPMatrixTrainer(NFSPMatrixConfig(**common,hidden=8),root,reserve_bytes=0)
        else:t=NFSPTaskTrainer(NFSPTaskConfig(60,**common,hidden=8,width=8,blocks=1,norm_groups=2,
                                           horizons=(3,4),equivalent_actions=4,purpose='profile'),root,reserve_bytes=0)
        t.run(until_records=32 if pause else None,verbose=False);return t
    def snapshot(self,root):return {p.relative_to(root):digest(p) for p in root.rglob('*') if p.is_file()}

    def test_fresh_all_three_auditors_make_source_bound_read_only_diagnostic_snapshots(self):
        for family in ('asymmetric','delayed'):
            with self.subTest(family=family),tempfile.TemporaryDirectory() as directory:
                root=Path(directory)/'run';self.fixture(root,family);before=self.snapshot(root);output=Path(directory)/'case.json';p=self.plan()
                state=torch.get_rng_state().clone();case=audit_case(p,root,output,allow_diagnostic=True,reserve_bytes=0)
                self.assertTrue(case['body']['fresh_audit_verified']);self.assertFalse(case['body']['primary_recipe'])
                self.assertEqual(before,self.snapshot(root));self.assertTrue(torch.equal(state,torch.get_rng_state()))
                report=assess_files(p,[output]);self.assertEqual(report['present_cases'],1);self.assertEqual(len(report['missing_cases']),35)
                self.assertTrue(report['current_input_snapshots_verified']);self.assertFalse(report['a2_complete'])
                with self.assertRaisesRegex(ValueError,'full fixed primary'):audit_case(p,root,Path(directory)/'primary.json',reserve_bytes=0)

    def test_version_one_collector_rejects_version_two_branching_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)/'run';self.fixture(root,'branching');before=self.snapshot(root)
            with self.assertRaises(ValueError):
                audit_case(self.plan(),root,Path(directory)/'case.json',allow_diagnostic=True,reserve_bytes=0)
            self.assertEqual(before,self.snapshot(root))

    def test_interrupted_snapshot_does_not_invent_a_fresh_policy_audit(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)/'run';self.fixture(root,pause=True)
            case=audit_case(self.plan(),root,Path(directory)/'case.json',allow_diagnostic=True,reserve_bytes=0)
            self.assertFalse(case['body']['fresh_audit_verified']);self.assertEqual(case['body']['records'],32)

    def test_current_input_drift_output_alias_and_active_writer_are_rejected(self):
        import fcntl
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)/'run';self.fixture(root);p=self.plan();output=Path(directory)/'case.json'
            with self.assertRaisesRegex(ValueError,'outside'):audit_case(p,root,root/'result.json',allow_diagnostic=True,reserve_bytes=0)
            with (root/'.trainer.lock').open('rb') as lock:
                fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                try:
                    with self.assertRaises(BlockingIOError):audit_case(p,root,output,allow_diagnostic=True,reserve_bytes=0)
                finally:fcntl.flock(lock,fcntl.LOCK_UN)
            audit_case(p,root,output,allow_diagnostic=True,reserve_bytes=0)
            with self.assertRaisesRegex(ValueError,'outside'):assess_files(p,[output],output=root/'result.json',reserve_bytes=0)
            path=root/'result.json';data=json.loads(path.read_text());data['unrecorded_edit']=True;path.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError,'stale'):assess_files(p,[output])

    def test_plan_case_and_assess_cli_preserve_the_original_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)/'run';self.fixture(root);before=self.snapshot(root);plan=Path(directory)/'plan.json';case=Path(directory)/'case.json'
            with redirect_stdout(io.StringIO()):
                self.assertEqual(main(['plan','--name','cli','--learner-source-sha',source_digest(),'--reuse','1','--output',str(plan)]),0)
                self.assertEqual(main(['audit-case','--plan',str(plan),'--run-dir',str(root),'--allow-diagnostic','--time-cap-seconds','60','--output',str(case)]),0)
                self.assertEqual(main(['assess','--plan',str(plan),'--case',str(case)]),0)
            self.assertEqual(before,self.snapshot(root))
            with redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):main(['plan','--name','again','--learner-source-sha',source_digest(),'--reuse','1','--output',str(plan)])

    def test_archived_reanalysis_never_claims_current_training_inputs_are_valid(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)/'run';self.fixture(root);p=self.plan();output=Path(directory)/'case.json'
            audit_case(p,root,output,allow_diagnostic=True,reserve_bytes=0)
            data=json.loads((root/'result.json').read_text());data['unrecorded_edit']=True;(root/'result.json').write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError,'stale'):assess_files(p,[output])
            result=assess_files(p,[output],verify_current_inputs=False)
            self.assertEqual(result['analysis_mode'],'archived_snapshots');self.assertFalse(result['current_input_snapshots_verified'])
            self.assertFalse(result['a2_complete']);self.assertFalse(result['pilot_authorized'])

    def test_completed_or_incomplete_case_snapshots_are_never_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)/'run';self.fixture(root,pause=True);p=self.plan();output=Path(directory)/'case.json'
            audit_case(p,root,output,allow_diagnostic=True,reserve_bytes=0);before=output.read_bytes()
            with self.assertRaisesRegex(ValueError,'preserve it'):audit_case(p,root,output,allow_diagnostic=True,reserve_bytes=0)
            self.assertEqual(output.read_bytes(),before)


if __name__=='__main__':unittest.main()
