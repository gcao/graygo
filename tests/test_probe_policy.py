from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import sqlite3
import tempfile
import time
import unittest
from unittest.mock import patch
import numpy as np
import torch
from graygo.confidence import canonical_sha
from graygo.evaluation import FrozenNeuralPolicy
from graygo.manifest import source_digest
from graygo.models import ModelConfig,PolicyNet,QNet
from graygo.probe_benchmark import ProbeBenchmark
from graygo.probe_policy_protocol import ProbePolicyProtocol
from graygo.probe_policy_evaluation import ProbePolicyRun,ProbePolicyEvidence,_meta_read,_meta_write
from graygo.science_protocol import ScienceProtocol,policy_keys
from graygo.storage import CheckpointStore,digest,DiskBudgetError

ROOT=Path(__file__).resolve().parents[1]
BENCHMARK=ROOT/'docs/verification/probe-multicell-v2-construction/benchmark.json'
FREEZE=ROOT/'docs/experiments/collision-benchmark-v2-freeze.json'


class FixturePolicy:
    def __init__(self,benchmark,seed,*,solver=False):
        self.identity={'kind':'solver-fixture' if solver else 'uniform-fixture','seed':seed}
        self.sha256=canonical_sha(self.identity);self.table={}
        if solver:
            for family in benchmark.families:
                for root in (family.before,family.after):
                    for role in (1,2):
                        values=np.zeros(82);actions=(root.certificate.black_actions,root.certificate.white_actions)[role-1]
                        values[list(actions)]=[float(x) for x in root.certificate.equilibrium[role-1]]
                        self.table[root.state.canonical_json(),role]=values
    def assert_frozen(self):
        if canonical_sha(self.identity)!=self.sha256:raise ValueError('fixture policy changed')
    def initial_distribution(self,state,role,*,fresh_episode=False):
        if not fresh_episode:raise ValueError('public roots must be declared fresh probe episodes')
        if self.table:return self.table[state.canonical_json(),role].copy()
        result=np.zeros(82);legal=state.legal_actions(role);result[list(legal)]=1/len(legal);return result


class ProbePolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.benchmark=ProbeBenchmark.from_dict(json.loads(BENCHMARK.read_text()));cls.freeze=json.loads(FREEZE.read_text())

    def policies(self,solver=False):return {s:FixturePolicy(self.benchmark,s,solver=solver) for s in (200,201,202)}

    def protocol(self,policies,cap=120,**kwargs):
        return ProbePolicyProtocol('fixture',self.benchmark.sha256,canonical_sha(self.freeze),
            {s:p.sha256 for s,p in policies.items()},cap,purpose='diagnostic',representation=next(iter(policies.values())).identity['kind'],**kwargs)

    def build(self,path,*,policies=None,resume=False,cap=120):
        policies=policies or self.policies()
        return ProbePolicyRun(self.protocol(policies,cap),path,self.benchmark,self.freeze,policies,resume=resume,reserve_bytes=0)

    def test_strict_protocol_roundtrip_and_confirmation_registry_binding(self):
        policies=self.policies();p=self.protocol(policies)
        self.assertEqual(ProbePolicyProtocol.from_dict(p.to_dict()),p)
        for name,value in (('families',31),('required_passes_per_seed',23),('individual_gap','1/10'),('seeds',[200,201])):
            data=p.to_dict();data[name]=value
            with self.assertRaises(ValueError):ProbePolicyProtocol.from_dict(data)
        for registry in ([(200,)],[(200,'a'*64)]*3,None):
            with self.assertRaises(ValueError):ProbePolicyProtocol('bad','a'*64,'b'*64,registry,60,purpose='diagnostic')
        ids={k:canonical_sha(list(k)) for k in policy_keys()};science=ScienceProtocol('science','b'*64,ids,60)
        selected={s:ids['exposure','gray',s] for s in (200,201,202)}
        p=ProbePolicyProtocol('formal','a'*64,'c'*64,selected,60,catalog_sha256='b'*64,science=science)
        self.assertEqual(ProbePolicyProtocol.from_dict(p.to_dict()),p)
        selected[202]=ids['empty','gray',202]
        with self.assertRaisesRegex(ValueError,'base-rule arm'):
            ProbePolicyProtocol('bad','a'*64,'c'*64,selected,60,catalog_sha256='b'*64,science=science)

    def test_complete_solver_fixture_passes_numerically_but_is_never_confirmation(self):
        with tempfile.TemporaryDirectory() as directory:
            run=self.build(directory,policies=self.policies(solver=True));result=run.run()
            self.assertEqual(result['status'],'complete');self.assertEqual(len(run.records()),32)
            self.assertTrue(run.estimate()['numerical_behavior_pass']);self.assertFalse(run.estimate()['frozen_inference_reproduced'])
            self.assertTrue(run.reproduce()['verified']);report=run.estimate()
            self.assertEqual([row['passed_families'] for row in report['grade']['seeds']],[32,32,32])
            self.assertTrue(report['within_inference_budget']);self.assertTrue(report['frozen_inference_reproduced'])
            for key in ('archive_policy_binding_verified','confirmation_behavioral_criterion_met','a2_complete','gray_specialization_demonstrated'):
                self.assertFalse(report[key])
            self.assertEqual(run.clock()['reproduced_families'],32)

    def test_atomic_all_seed_families_resume_exactly_and_report_is_read_only(self):
        with tempfile.TemporaryDirectory() as directory:
            full=self.build(Path(directory)/'full');full.run(5)
            part=self.build(Path(directory)/'part');part.run(2);elapsed=part.clock()['observed_seconds']
            resumed=self.build(Path(directory)/'part',resume=True);resumed.run(5)
            self.assertEqual(full.records(),resumed.records());self.assertGreater(resumed.clock()['observed_seconds'],elapsed)
            before={p.relative_to(directory):digest(p) for p in Path(directory).rglob('*') if p.is_file()}
            report=ProbePolicyEvidence(resumed.root).estimate()
            self.assertEqual(before,{p.relative_to(directory):digest(p) for p in Path(directory).rglob('*') if p.is_file()})
            self.assertEqual(report['completed_families'],5);self.assertIsNone(report['grade'])
            self.assertFalse(report['confirmation_behavioral_criterion_met'])
            with self.assertRaisesRegex(ValueError,'hide already committed'):resumed.run(2)

    def test_partial_family_failure_and_transaction_failure_commit_no_subset(self):
        import graygo.probe_policy_evaluation as module
        with tempfile.TemporaryDirectory() as directory:
            run=self.build(directory);original=module._distribution;calls=[0]
            def fail(*args):
                calls[0]+=1
                if calls[0]==7:raise RuntimeError('mid-family')
                return original(*args)
            with patch.object(module,'_distribution',side_effect=fail),self.assertRaisesRegex(RuntimeError,'mid-family'):run.run(1)
            self.assertEqual(run.records(),[]);self.assertEqual(run.clock()['committed_families'],0)
            original_write=module._meta_write
            def fail_write(db,key,value):
                if key=='clock' and value['committed_families']==1:raise RuntimeError('transaction failure')
                original_write(db,key,value)
            with patch.object(module,'_meta_write',side_effect=fail_write),self.assertRaisesRegex(RuntimeError,'transaction'):run.run(1)
            self.assertEqual(run.records(),[]);self.assertEqual(run.clock()['committed_families'],0)
            run.run(1);self.assertEqual(len(run.records()),1)

    def test_resume_cannot_change_policy_source_ceiling_benchmark_or_runtime(self):
        with tempfile.TemporaryDirectory() as directory:
            run=self.build(directory)
            with self.assertRaisesRegex(ValueError,'changed on resume'):self.build(directory,resume=True,cap=121)
            policies=self.policies();policies[202].identity['variant']=2;policies[202].sha256=canonical_sha(policies[202].identity)
            with self.assertRaisesRegex(ValueError,'changed on resume'):self.build(directory,resume=True,policies=policies)
            old=torch.get_num_threads()
            try:
                torch.set_num_threads(old+1)
                with self.assertRaisesRegex(ValueError,'changed on resume'):self.build(directory,resume=True)
            finally:torch.set_num_threads(old)
            with patch('graygo.probe_policy_evaluation.source_digest',return_value='a'*64),self.assertRaisesRegex(ValueError,'runtime changed'):run.run(1)
            wrong=deepcopy(self.freeze);wrong['benchmark_sha256']='a'*64
            with self.assertRaisesRegex(ValueError,'frozen benchmark'):
                ProbePolicyRun(self.protocol(self.policies()),Path(directory)/'wrong',self.benchmark,wrong,self.policies(),reserve_bytes=0)

    def test_unclean_resume_and_clock_reversal_exhaust_original_budget(self):
        for reversal in (False,True):
            with self.subTest(reversal=reversal),tempfile.TemporaryDirectory() as directory:
                run=self.build(directory,cap=10);db=sqlite3.connect(run.database);clock=_meta_read(db,'clock')
                clock['inflight_since']=time.time()+100 if reversal else time.time()-100
                with db:_meta_write(db,'clock',clock)
                db.close();resumed=self.build(directory,cap=10,resume=True);result=resumed.run(1)
                self.assertEqual(result['status'],'budget_limited');self.assertEqual(result['committed_families'],0)
                self.assertGreaterEqual(result['clock']['observed_seconds']+result['clock']['recovery_charge_seconds'],10)
                self.assertFalse(resumed.estimate()['within_inference_budget'])
                db=sqlite3.connect(run.database)
                with db:db.execute("UPDATE meta SET payload='{}' WHERE key='clock'")
                db.close()
                with self.assertRaisesRegex(ValueError,'checksum'):ProbePolicyEvidence(directory).estimate()

    def test_marginal_inventory_illegal_weights_and_resealed_root_tampering_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            run=self.build(directory);run.run(1);original=run.records()[0]
            for mutation in ('seed','root','role','weights'):
                row=deepcopy(original)
                if mutation=='seed':del row['marginals']['202']
                elif mutation=='root':del row['marginals']['200']['after']
                elif mutation=='role':row['marginals']['200']['before'].pop()
                else:row['marginals']['200']['before'][0][0]=-1
                with self.subTest(mutation=mutation),self.assertRaises(ValueError):run.validate_family(row,0)
            policy=run.policies[200]
            with patch.object(policy,'initial_distribution',return_value=np.ones(82)/82),self.assertRaisesRegex(ValueError,'illegal action'):run.run(2)
            self.assertEqual(len(run.records()),1)

    def test_resealed_legal_marginals_require_actual_frozen_forward_reproduction(self):
        with tempfile.TemporaryDirectory() as directory:
            run=self.build(directory);run.run(1);run.reproduce();row=run.records()[0]
            p=row['marginals']['200']['before'][0];p[:]=[0.]*len(p);p[0]=1.
            raw=json.dumps(row,sort_keys=True,separators=(',',':'))
            db=sqlite3.connect(run.database)
            with db:db.execute('UPDATE families SET payload=?,sha256=? WHERE idx=0',(raw,hashlib.sha256(raw.encode()).hexdigest()))
            db.close()
            self.assertFalse(run.estimate()['frozen_inference_reproduced'])
            with self.assertRaisesRegex(ValueError,'actual frozen-model forward pass'):run.reproduce()

    def test_inference_and_audit_share_one_persistent_cap(self):
        with tempfile.TemporaryDirectory() as directory:
            run=self.build(directory);run.run(1);db=sqlite3.connect(run.database);clock=run.clock()
            clock['observed_seconds']=120
            with db:_meta_write(db,'clock',clock)
            db.close();before=clock['observed_seconds'];result=run.reproduce()
            self.assertEqual(result['status'],'budget_limited');self.assertEqual(result['checked_families'],0)
            self.assertGreater(run.clock()['observed_seconds'],before);self.assertFalse(run.estimate()['frozen_inference_reproduced'])

    def test_disk_floor_and_concurrent_writer_are_enforced(self):
        from collections import namedtuple
        from graygo.evaluation import _EvaluationLock
        Usage=namedtuple('Usage','total used free')
        with tempfile.TemporaryDirectory() as directory:
            run=self.build(directory);before=digest(run.database)
            with patch('graygo.probe_policy_evaluation.shutil.disk_usage',return_value=Usage(100,99,1)),self.assertRaises(DiskBudgetError):run.run(1)
            self.assertEqual(before,digest(run.database))
            with _EvaluationLock(run.root/'.probe.lock'),self.assertRaises(BlockingIOError):run.run(1)

    def archive(self,root,seed):
        root.mkdir();config={'width':8,'blocks':1,'norm_groups':2,'seed':seed,'records':2,'scope':'diagnostic'}
        manifest={'config':config,'source_sha256':source_digest(),'rules':{'size':9,'max_rounds':200,'collision':'gray'}}
        (root/'run.json').write_text(json.dumps(manifest,sort_keys=True));torch.manual_seed(seed)
        raw=io.BytesIO();torch.save({'config':config,'q':QNet(ModelConfig(8,1,2)).state_dict(),'average':PolicyNet(ModelConfig(8,1,2)).state_dict()},raw)
        data=raw.getvalue();store=CheckpointStore(root,reserve_bytes=0)
        return store.save(0,lambda p:(p/'models.pt').write_bytes(data),metadata={'source_sha256':source_digest()},estimated_peak_bytes=len(data)+65536,kind='weights')

    def test_real_untrained_archives_resume_and_replay_but_cannot_be_confirmation(self):
        from graygo.curriculum import create_catalog
        with tempfile.TemporaryDirectory() as directory:
            paths={s:self.archive(Path(directory)/str(s),s) for s in (200,201,202)}
            policies={s:FrozenNeuralPolicy.from_archive(p) for s,p in paths.items()}
            protocol=self.protocol(policies)
            run=ProbePolicyRun(protocol,Path(directory)/'probe',self.benchmark,self.freeze,policies,archive_paths=paths,reserve_bytes=0)
            run.run(2);self.assertTrue(run.reproduce()['verified']);report=run.estimate()
            self.assertFalse(report['archive_policy_binding_verified']);self.assertFalse(report['confirmation_behavioral_criterion_met'])
            catalog=create_catalog(443,train_families=6,heldout_families=6);ids={k:canonical_sha(list(k)) for k in policy_keys()}
            for s,p in policies.items():ids['exposure','gray',s]=p.sha256
            science=ScienceProtocol('negative',catalog.sha256,ids,60)
            formal=ProbePolicyProtocol('negative',self.benchmark.sha256,canonical_sha(self.freeze),{s:p.sha256 for s,p in policies.items()},
                60,catalog_sha256=catalog.sha256,science=science)
            with self.assertRaisesRegex(ValueError,'milestone provenance differs'):
                ProbePolicyRun(formal,Path(directory)/'formal',self.benchmark,self.freeze,policies,catalog=catalog,archive_paths=paths)
            self.assertFalse((Path(directory)/'formal/plan.json').exists())

    def test_cli_real_archive_collection_read_only_report_and_budgeted_audit(self):
        import contextlib
        from graygo.cli import main
        with tempfile.TemporaryDirectory() as directory:
            paths={s:self.archive(Path(directory)/str(s),s) for s in (200,201,202)}
            root=Path(directory)/'probe'
            args=['probe-evaluate','--run-dir',str(root),'--benchmark',str(BENCHMARK),'--freeze-receipt',str(FREEZE),
                  '--name','cli-fixture','--purpose','diagnostic','--families','2','--time-cap-hours','.025']
            for seed,path in paths.items():args+=['--policy',str(seed)+'='+str(path)]
            def invoke(args):
                result=io.StringIO()
                with contextlib.redirect_stdout(result):self.assertEqual(main(args),0)
                return json.loads(result.getvalue())['result']
            result=invoke(args);self.assertEqual(result['collection']['committed_families'],2)
            invoke(args+['--resume'])
            before={p.relative_to(directory):digest(p) for p in Path(directory).rglob('*') if p.is_file()}
            report=invoke(['probe-report','--run-dir',str(root)])
            self.assertEqual(before,{p.relative_to(directory):digest(p) for p in Path(directory).rglob('*') if p.is_file()})
            self.assertFalse(report['inference_complete']);self.assertFalse(report['confirmation_behavioral_criterion_met'])
            audit=invoke(['probe-audit','--run-dir',str(root)])
            self.assertTrue(audit['audit']['verified']);self.assertFalse(audit['summary']['confirmation_behavioral_criterion_met'])
            with contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):
                main(['probe-report','--run-dir',str(root),'--output',str(root/'plan.json')])
            changed=args.copy();changed[changed.index('.025')]='.03'
            with contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):main(changed+['--resume'])

    def test_shared_confirmation_archive_contract_checks_optimizer_journal_and_seal(self):
        # Synthetic 80M metadata checks the loader contract, never a learned result.
        from graygo.board_journal import BoardJournalPlan
        from graygo.curriculum import create_catalog,ExposureStarts
        from graygo.science_evaluation import _archive_provenance
        catalog=create_catalog(443,train_families=6,heldout_families=6);key=('exposure','gray',200)
        journal=BoardJournalPlan('confirmation',80000000,(0,4500000,4750000,5000000,6000000,7000000,8000000,80000000))
        base={'width':64,'blocks':8,'norm_groups':8,'seed':200,'records':80000000,'scope':'confirmation',
              'batch':512,'learning_start':4096,'capacity':500000,'target_every':12800,'learning_rate':3e-4,'max_grad_norm':.5,
              'reservoir_open':5000000,'handoff_end':7000000,'uniform_comparison':False,'augmentation':True,'deterministic':True,
              'journal_plan':journal.to_dict()}
        torch.manual_seed(200);shape=ModelConfig(64,8,8);q=QNet(shape).state_dict();avg=PolicyNet(shape).state_dict()
        with tempfile.TemporaryDirectory() as directory:
            for variant in ('optimizer','journal','seal','valid-container'):
                with self.subTest(variant=variant):
                    config=deepcopy(base)
                    if variant=='optimizer':config['learning_rate']=.01
                    elif variant=='journal':config['journal_plan']['scope']='diagnostic'
                    root=Path(directory)/variant;root.mkdir();manifest={'config':config,'source_sha256':source_digest(),
                        'rules':{'size':9,'max_rounds':200,'collision':'gray'},'training_starts':ExposureStarts(catalog).identity}
                    (root/'run.json').write_text(json.dumps(manifest,sort_keys=True));raw=io.BytesIO()
                    torch.save({'config':config,'q':q,'average':avg},raw);data=raw.getvalue();store=CheckpointStore(root,reserve_bytes=0)
                    seal={'source_sha256':source_digest(),'training_run_sha256':digest(root/'run.json'),
                          'journal_plan_sha256':canonical_sha(config['journal_plan'])}
                    if variant=='seal':seal['training_run_sha256']='a'*64
                    path=store.save(80000000,lambda p:(p/'models.pt').write_bytes(data),metadata=seal,
                                    estimated_peak_bytes=len(data)+65536,kind='weights')
                    policy=FrozenNeuralPolicy.from_archive(path);ids={k:canonical_sha(list(k)) for k in policy_keys()};ids[key]=policy.sha256
                    science=ScienceProtocol('synthetic-contract',catalog.sha256,ids,60)
                    if variant=='valid-container':
                        refs=_archive_provenance(science,{key:policy},{key:path},catalog,expected_keys=(key,))
                        self.assertFalse(refs[key]['scope_approval_independently_verified'])
                    else:
                        with self.assertRaises(ValueError):_archive_provenance(science,{key:policy},{key:path},catalog,expected_keys=(key,))


if __name__=='__main__':unittest.main()
