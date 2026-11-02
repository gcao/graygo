from copy import deepcopy
import json
from pathlib import Path
import sqlite3
import tempfile
import time
import io
import torch
import unittest
from unittest.mock import patch
import numpy as np
from graygo.confidence import canonical_sha
from graygo.storage import digest,CheckpointStore,DiskBudgetError
from graygo.models import QNet,PolicyNet,ModelConfig
from graygo.evaluation import FrozenNeuralPolicy,EmptyStarts
from graygo.curriculum import ExposureStarts
from graygo.manifest import source_digest
from graygo.curriculum import create_catalog
from graygo.engine import BLACK,WHITE,Rules,State
from graygo.science_protocol import ScienceProtocol,policy_keys,endpoint,GAMES_PER_BLOCK
from graygo.science_evaluation import ScientificRun,ScientificEvidence,_meta_read,_meta_write


class PassActor:
    component=None
    def action(self,state,player):return state.rules.pass_action


class PassPolicy:
    identity={'kind':'test-pass','version':1}
    sha256=canonical_sha(identity)
    def assert_frozen(self):
        if self.sha256!=canonical_sha(self.identity):raise ValueError('changed pass policy')
    def begin_episode(self,seed):return PassActor()


class ScienceProtocolTests(unittest.TestCase):
    def protocol(self,looks=(500,2000)):
        return ScienceProtocol('test','b'*64,{key:'a'*64 for key in policy_keys()},60,block_looks=looks,purpose='diagnostic')
    def values(self,n=2000):
        x=np.zeros((n,2,2,3,2));x[...,0]=.5;x[...,1]=-.5;return x

    def test_budget_units_simultaneous_coverage_and_strict_protocol_roundtrip(self):
        p=self.protocol();self.assertEqual(p.physical_looks,(24000,96000));self.assertEqual(len(p.endpoints),48)
        self.assertEqual(p.delta,.05/96);self.assertEqual(ScienceProtocol.from_dict(p.to_dict()),p)
        for key,value in (('games_per_block',2),('seeds',[200,201]),('method','other')):
            data=p.to_dict();data[key]=value
            with self.assertRaises(ValueError):ScienceProtocol.from_dict(data)
        ids={key:canonical_sha(list(key)) for key in policy_keys()}
        with self.assertRaises(ValueError):ScienceProtocol('formal','b'*64,ids,60,block_looks=(2,4))

    def test_primary_uses_paired_interaction_and_conditional_equal_seed_average(self):
        p=self.protocol();x=self.values();report=p.summarize(x,96000)
        self.assertTrue(report['numerical_performance_pass']);self.assertFalse(report['gray_specialization_demonstrated'])
        u=report['primary']['aggregate_bounds']['u_G'];interaction=report['primary']['aggregate_bounds']['interaction']
        self.assertEqual(u['independent_units'],2000);self.assertEqual(interaction['support'],[-2.,2.])
        self.assertEqual(u['mean'],.5);self.assertEqual(interaction['mean'],1.)
        z=np.zeros_like(x);z[::2,...,0]=1;z[1::2,...,0]=-1;z[...,1]=z[...,0]
        repeated=p.summarize(z,96000)
        paired=repeated['estimates'][endpoint('official-empty','exposure','interaction','mean3')]
        self.assertEqual(paired['mean'],0);self.assertEqual(paired['sample_variance'],0)
        self.assertFalse(repeated['numerical_performance_pass'])

    def test_one_negative_seed_or_only_secondary_success_cannot_pass_primary(self):
        p=self.protocol();x=self.values();x[:,0,1,2,0]=-.5;x[:,0,1,2,1]=-1
        report=p.summarize(x,96000)
        self.assertGreater(report['primary']['aggregate_bounds']['u_G']['lower'],.05)
        self.assertGreater(report['primary']['aggregate_bounds']['interaction']['lower'],.05)
        self.assertFalse(report['primary']['all_seed_points_positive']);self.assertFalse(report['numerical_performance_pass'])
        x=self.values();x[:,0,1]=0;report=p.summarize(x,96000)
        self.assertTrue(report['secondary']['numerical_pass']);self.assertFalse(report['numerical_performance_pass'])

    def test_missing_data_impossible_pair_values_and_hidden_later_looks_are_rejected(self):
        p=self.protocol();x=self.values()
        for bad in (x[:499],x[:,:,:,:2],np.full_like(x,.25),np.full_like(x,np.nan),np.concatenate((x,x[:1]))):
            with self.assertRaises(ValueError):p.summarize(bad,96000)
        with self.assertRaisesRegex(ValueError,'later completed'):p.summarize(x,24000)
        self.assertTrue(p.summarize(x[:500],24000)['numerical_performance_pass'])


class ScientificEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.catalog=create_catalog(443,train_families=6,heldout_families=6)
    def build(self,path,*,resume=False,cap=60):
        policies={key:PassPolicy() for key in policy_keys()}
        p=ScienceProtocol('diag',self.catalog.sha256,{key:policy.sha256 for key,policy in policies.items()},cap,seed=13,block_looks=(2,4),purpose='diagnostic')
        return ScientificRun(p,path,policies,self.catalog,resume=resume,reserve_bytes=0)

    def test_actual_both_rule_context_seed_blocks_resume_exactly_and_remain_diagnostic(self):
        with tempfile.TemporaryDirectory() as directory:
            full=self.build(Path(directory)/'full');full.run(192)
            part=self.build(Path(directory)/'part');part.run(96)
            before=part.clock()['observed_seconds'];resumed=self.build(Path(directory)/'part',resume=True);resumed.run(192)
            self.assertEqual(full.records(),resumed.records());self.assertGreaterEqual(resumed.clock()['observed_seconds'],before)
            self.assertEqual(len(resumed.records()),4)
            for row in resumed.records():
                self.assertEqual(len(row['cells']),2)
                for context in row['cells'].values():
                    for curriculum in context.values():
                        for rules in curriculum.values():self.assertEqual(rules['gray']['utility'],rules['remove']['utility'])
            snapshots={p.relative_to(directory):digest(p) for p in (Path(directory)/'part').rglob('*') if p.is_file()}
            report=ScientificEvidence(Path(directory)/'part').estimate(192)
            self.assertEqual(snapshots,{p.relative_to(directory):digest(p) for p in (Path(directory)/'part').rglob('*') if p.is_file()})
            self.assertEqual(report['blocks'],4);self.assertEqual(report['physical_games'],192)
            self.assertFalse(report['gray_specialization_demonstrated']);self.assertFalse(report['numerical_performance_pass'])
            with self.assertRaisesRegex(ValueError,'later completed'):resumed.estimate(96)

    def test_resume_cannot_change_numerical_inference_runtime(self):
        with tempfile.TemporaryDirectory() as directory:
            run=self.build(directory);old=torch.get_num_threads()
            try:
                torch.set_num_threads(old+1)
                with self.assertRaisesRegex(ValueError,'runtime changed'):self.build(directory,resume=True)
            finally:torch.set_num_threads(old)
            self.assertEqual(run.records(),[])

    def test_failed_partial_block_commits_no_subset_and_replays_same_seeds(self):
        with tempfile.TemporaryDirectory() as directory:
            run=self.build(directory)
            import graygo.science_evaluation as module
            original=module.evaluate_pair;calls=[0]
            def fail(*args,**kwargs):
                calls[0]+=1
                if calls[0]==7:raise RuntimeError('injected mid-block failure')
                return original(*args,**kwargs)
            with patch.object(module,'evaluate_pair',side_effect=fail):
                with self.assertRaisesRegex(RuntimeError,'mid-block'):run.run(96)
            self.assertEqual(run.records(),[]);self.assertEqual(run.clock()['committed_blocks'],0)
            self.assertIsNone(run.clock()['inflight_since']);self.assertGreater(run.clock()['observed_seconds'],0)
            resumed=self.build(directory,resume=True);resumed.run(96)
            self.assertEqual(len(resumed.records()),2)

    def test_sql_transaction_failure_does_not_advance_the_budget_prefix(self):
        with tempfile.TemporaryDirectory() as directory:
            run=self.build(directory)
            import graygo.science_evaluation as module
            original=module._meta_write
            def fail(db,key,value):
                if key=='clock' and value['committed_blocks']==1:raise RuntimeError('injected transaction failure')
                return original(db,key,value)
            with patch.object(module,'_meta_write',side_effect=fail):
                with self.assertRaisesRegex(RuntimeError,'transaction failure'):run.run(96)
            self.assertEqual(run.records(),[]);self.assertEqual(run.clock()['committed_blocks'],0)
            run.run(96);self.assertEqual(len(run.records()),2)

    def test_clock_corruption_rollback_and_unclean_resume_cannot_reset_the_ceiling(self):
        with tempfile.TemporaryDirectory() as directory:
            run=self.build(directory,cap=1)
            db=sqlite3.connect(run.database);clock=_meta_read(db,'clock');clock['inflight_since']=time.time()-5
            with db:_meta_write(db,'clock',clock)
            db.close();resumed=self.build(directory,resume=True,cap=1);result=resumed.run(96)
            self.assertEqual(result['status'],'budget_limited');self.assertEqual(result['committed_blocks'],0)
            self.assertGreaterEqual(result['clock']['recovery_charge_seconds']+result['clock']['observed_seconds'],1)
            with self.assertRaises(ValueError):self.build(directory,resume=True,cap=2)
            db=sqlite3.connect(run.database)
            with db:db.execute("UPDATE meta SET payload='{}' WHERE key='clock'")
            db.close()
            with self.assertRaisesRegex(ValueError,'checksum'):resumed.clock()

    def test_heldout_seed_rule_or_game_tampering_is_rejected_even_if_resealed(self):
        with tempfile.TemporaryDirectory() as directory:
            run=self.build(directory);run.run(96);rows=run.records()
            for mutate in ('seed','rule','score','opening'):
                row=deepcopy(rows[0])
                if mutate=='seed':del row['cells']['official-empty']['exposure']['202']
                elif mutate=='rule':del row['cells']['official-empty']['exposure']['200']['remove']
                elif mutate=='score':row['cells']['official-empty']['exposure']['200']['gray']['subject'][0]['black_score_units']+=1
                else:row['openings']['heldout-gray']['remove']['root_sha256']='a'*64
                with self.subTest(mutate=mutate),self.assertRaises(ValueError):run.validate_block(row,0)

    def test_disk_reserve_stops_before_any_clock_or_evidence_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            run=self.build(directory);before=digest(run.database)
            from collections import namedtuple
            Usage=namedtuple('Usage','total used free')
            with patch('graygo.science_evaluation.shutil.disk_usage',return_value=Usage(100,99,1)):
                with self.assertRaises(DiskBudgetError):run.run(96)
            self.assertEqual(digest(run.database),before);self.assertEqual(run.records(),[])

    def test_verified_containers_still_cannot_relabel_development_as_confirmation(self):
        # Synthetic weight containers exercise metadata guards, not training.
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as directory:
            paths={};policies={}
            for curriculum,rule,seed in policy_keys():
                key=(curriculum,rule,seed);root=Path(directory)/f'{curriculum}-{rule}-{seed}';root.mkdir()
                config={'width':8,'blocks':1,'norm_groups':2,'seed':seed,'records':80000000,
                        'scope':'development' if key==policy_keys()[0] else 'confirmation'}
                starts=EmptyStarts().identity if curriculum=='empty' else ExposureStarts(self.catalog).identity
                manifest={'config':config,'rules':{'size':9,'max_rounds':200,'collision':rule},'source_sha256':source_digest(),'training_starts':starts}
                (root/'run.json').write_text(json.dumps(manifest,sort_keys=True))
                q,average=QNet(ModelConfig(8,1,2)),PolicyNet(ModelConfig(8,1,2))
                raw=io.BytesIO();torch.save({'config':config,'q':q.state_dict(),'average':average.state_dict()},raw)
                payload=raw.getvalue();store=CheckpointStore(root,reserve_bytes=0)
                path=store.save(80000000,lambda p:(p/'models.pt').write_bytes(payload),metadata={'source_sha256':source_digest()},estimated_peak_bytes=len(payload)+65536,kind='weights')
                paths[key]=path;policies[key]=FrozenNeuralPolicy.from_archive(path)
            protocol=ScienceProtocol('metadata-check',self.catalog.sha256,{key:p.sha256 for key,p in policies.items()},60)
            with self.assertRaisesRegex(ValueError,'development/gate agents'):
                ScientificRun(protocol,Path(directory)/'study',policies,self.catalog,archive_paths=paths)
            self.assertFalse((Path(directory)/'study/plan.json').exists())

    def test_diagnostic_agents_cannot_be_used_as_confirmation_models(self):
        policies={key:PassPolicy() for key in policy_keys()}
        with tempfile.TemporaryDirectory() as directory:
            ids={key:canonical_sha(list(key)) for key in policy_keys()}
            p=ScienceProtocol('formal',self.catalog.sha256,ids,60)
            with self.assertRaises(ValueError):ScientificRun(p,directory,policies,self.catalog)
            self.assertFalse((Path(directory)/'plan.json').exists())

if __name__=='__main__':unittest.main()
