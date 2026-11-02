"""Synthetic archive/score fixtures exercise gates, never training strength."""
from copy import deepcopy
from contextlib import redirect_stderr
from dataclasses import asdict
import fcntl
import hashlib
import json
import io
import os
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch
import torch
from graygo.board_journal import BoardJournalPlan,BoardMilestoneJournal
from graygo.board_runtime import BoardNFSPRuntime,BoardRuntimeConfig,configure_runtime
from graygo.cohort_evidence import DevelopmentCohort
from graygo.confidence import canonical_sha
from graygo.engine import Rules,State
from graygo.evaluation import BaselinePolicy,EvaluationPlan,EvaluationRun,FrozenNeuralPolicy,seed_for
from graygo.evaluation_evidence import FrozenEvaluationEvidence
from graygo.manifest import source_digest
from graygo.nfsp import NFSPCadence
from graygo.pilot_gates import DEVELOPMENT_SEEDS,PRE,POST,OPPONENTS
from graygo.storage import CheckpointStore,digest


def hashes(root):return {str(p.relative_to(root)):digest(p) for p in Path(root).rglob('*') if p.is_file()}


class CohortFixture:
    def __init__(self,root,*,complete=False):
        self.root=Path(root);self.runs={};self.journals={}
        self.plan=BoardJournalPlan('development',20000000,(0,*PRE,*POST,20000000))
        for seed in DEVELOPMENT_SEEDS:
            runtime=BoardNFSPRuntime(BoardRuntimeConfig(seed=seed))
            directory=self.root/str(seed);journal=BoardMilestoneJournal(directory,runtime,self.plan)
            journal.checkpoint();self.runs[seed]=directory;self.journals[seed]=journal
        self.cohort=DevelopmentCohort.register(self.root/'cohort','fixture',self.runs)
        if complete:self.populate()

    def archive(self,seed,record):
        directory=self.runs[seed];store=CheckpointStore(directory)
        initial=directory/'weights'/'000000000000';seal=store.verify(initial);meta=deepcopy(seal['metadata'])
        reuse=self.journals[seed].runtime.config.q_reuse
        updates=max(0,record//512-8+1)*reuse;sl=int(record>5000000)
        meta['counters']=asdict(NFSPCadence(fresh_records=record,fresh_q=record,fresh_ordinary=record,q_updates=updates,q_loss_targets=updates*512,
                             supervised_updates=sl,supervised_targets=sl*512,target_refreshes=record//12800))
        path=directory/'weights'/f'{record:012d}'
        if not path.exists():store.save(record,lambda p:os.link(initial/'models.pt',p/'models.pt'),kind='weights',metadata=meta,
                                        estimated_peak_bytes=(initial/'models.pt').stat().st_size+65536)
        return path

    def scores(self,seed,record,representation,opponent,*,games=500,bad_tail=False,wrong_seed=False):
        archive=self.archive(seed,record);plan,family,_,path=self.cohort.plan_for(seed,record,representation,opponent)
        if wrong_seed:
            values=asdict(plan);values['rules']=plan.rules;values['seed']+=1;plan=EvaluationPlan(**values)
        subject=FrozenNeuralPolicy.from_archive(archive,representation=representation,label=plan.subject_label)
        run=EvaluationRun(plan,path,subject,BaselinePolicy(opponent),resume=(path/'plan.json').exists(),reserve_bytes=0)
        db=run._connect()
        try:
            start=db.execute('SELECT COUNT(*) FROM units').fetchone()[0];rows=[]
            for index in range(start,games//2):
                unit=seed_for(plan.seed,'unit',index);value=-1. if bad_tail and index>=250 else 1.
                scored=[]
                for color in range(2):
                    black=value if color==0 else -value
                    scored.append({'subject_utility':value,'black_utility':black,'rounds':200,'termination':'round_limit',
                        'black_score_units':324 if black>0 else 0,'white_score_units':0 if black>0 else 324,
                        'trajectory_sha256':canonical_sha({'fixture':plan.endpoint,'color':color}),
                        'subject_component':None,'opponent_component':None})
                row={'index':index,'uid':str(unit),'root_sha256':hashlib.sha256(State().canonical_json().encode()).hexdigest(),
                     'subject':scored,'incumbent':[],'utility':value,'incumbent_utility':None,'improvement':None}
                payload=json.dumps(row,sort_keys=True,separators=(',',':'))
                rows.append((index,str(unit),payload,hashlib.sha256(payload.encode()).hexdigest()))
            with db:db.executemany('INSERT INTO units VALUES (?,?,?,?)',rows)
        finally:db.close()
        return run

    def populate(self):
        for seed in DEVELOPMENT_SEEDS:
            for record in (*PRE,*POST,20000000):
                representations=('average',) if record==20000000 else ('greedy_q',) if record in PRE else ('greedy_q','average')
                for representation in representations:
                    for opponent in OPPONENTS:self.scores(seed,record,representation,opponent)


class CohortEvidenceTests(unittest.TestCase):
    def test_real_fresh_initializations_register_once_without_issuing_authority(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture=CohortFixture(directory);cohort=fixture.cohort
            before=hashes(directory);state=torch.get_rng_state().clone()
            report=cohort.report('pilot');body=report['body']
            self.assertEqual(body['status'],'incomplete');self.assertEqual(len(body['missing']),54)
            self.assertFalse(body['formal_scope_approved']);self.assertFalse(body['a2_complete'])
            self.assertFalse(body['training_trajectory_independently_verified'])
            torch.testing.assert_close(torch.get_rng_state(),state,rtol=0,atol=0)
            self.assertEqual(hashes(directory),before)
            with self.assertRaises(FileExistsError):DevelopmentCohort.register(cohort.root,'fixture',fixture.runs)

    def test_complete_synthetic_scores_pass_only_conditional_numerical_cohort_gates(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture=CohortFixture(directory,complete=True);before=hashes(directory)
            pilot=fixture.cohort.report('pilot')['body'];screen=fixture.cohort.report('screening')['body']
            self.assertEqual(pilot['status'],'pass');self.assertEqual(screen['status'],'pass')
            self.assertEqual(len(pilot['reads']['pilot']),54);self.assertEqual(len(screen['reads']['screening']),6)
            self.assertTrue(all(row['repeated_trajectory_fraction']>.9 for row in pilot['reads']['pilot']))
            self.assertFalse(screen['formal_scope_approved']);self.assertFalse(screen['training_trajectory_independently_verified'])
            self.assertEqual(hashes(directory),before)

    def test_missing_average_diagnostic_prevents_pass_and_largest_completed_look_is_used(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture=CohortFixture(directory,complete=True);cohort=fixture.cohort
            missing=cohort.location(102,POST[-1],'average','tactical');hidden=missing.with_name('temporarily-missing')
            missing.rename(hidden)
            self.assertEqual(cohort.report('pilot')['body']['status'],'incomplete')
            hidden.rename(missing)
            for record in PRE[:2]:
                for opponent in OPPONENTS:fixture.scores(100,record,'greedy_q',opponent,games=2000,bad_tail=True)
            body=cohort.report('screening')['body']
            self.assertEqual(body['pilot']['status'],'no_go');self.assertEqual(body['status'],'no_go')
            changed=[row for row in body['reads']['pilot'] if row['endpoint'].startswith('seed100/') and row['subject_origin']['records'] in PRE[:2]]
            self.assertTrue(all(row['games']==2000 for row in changed))

    def test_wrong_evaluation_seed_or_policy_cannot_satisfy_an_original_slot(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture=CohortFixture(directory);fixture.scores(100,PRE[0],'greedy_q','random',wrong_seed=True)
            with self.assertRaisesRegex(ValueError,'another policy, plan'):fixture.cohort.report('pilot')
        with tempfile.TemporaryDirectory() as directory:
            fixture=CohortFixture(directory)
            archive=fixture.archive(100,PRE[0]);plan,family,_,path=fixture.cohort.plan_for(100,PRE[0],'greedy_q','random')
            subject=FrozenNeuralPolicy.from_archive(archive,representation='average',label=plan.subject_label)
            EvaluationRun(plan,path,subject,BaselinePolicy('random'),reserve_bytes=0)
            with self.assertRaisesRegex(ValueError,'another policy, plan'):fixture.cohort.report('pilot')

    def test_registry_or_archive_accounting_tampering_fails_before_a_gate_read(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture=CohortFixture(directory);path=fixture.cohort.root/'cohort.json';data=json.loads(path.read_text())
            data['body']['training_runs']['100']['training_run_sha256']='0'*64;path.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError,'integrity'):DevelopmentCohort.load(fixture.cohort.root)
        with tempfile.TemporaryDirectory() as directory:
            fixture=CohortFixture(directory);run=fixture.scores(100,PRE[0],'greedy_q','random')
            archive=fixture.runs[100]/'weights'/f'{PRE[0]:012d}';path=archive/'manifest.json';data=json.loads(path.read_text())
            data['metadata']['counters']['q_updates']+=1
            path.write_text(json.dumps(data,sort_keys=True,indent=2)+'\n');(archive/'manifest.sha256').write_text(digest(path)+'\n')
            with self.assertRaisesRegex(ValueError,'accounting'):fixture.cohort.report('pilot')

    def test_collectors_cannot_change_a_present_endpoint_during_cohort_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture=CohortFixture(directory);run=fixture.scores(100,PRE[0],'greedy_q','random',games=4)
            with fixture.cohort.snapshot('pilot') as report:
                self.assertEqual(report['body']['status'],'incomplete')
                with self.assertRaises(BlockingIOError):
                    with run._lock():pass
            with run._lock():pass

    def test_register_rejects_wrong_cohort_or_post_initialization_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture=CohortFixture(directory);fixture.archive(100,PRE[0])
            with self.assertRaisesRegex(ValueError,'before training'):
                DevelopmentCohort.register(Path(directory)/'retrospective','late',fixture.runs)
            with self.assertRaisesRegex(ValueError,'three-seed'):
                DevelopmentCohort.register(Path(directory)/'missing','missing',{100:fixture.runs[100]})
            with self.assertRaises(ValueError):fixture.cohort.plan_for(200,PRE[0],'greedy_q','random')
            with self.assertRaises(ValueError):fixture.cohort.plan_for(100,PRE[0],'average','random')

    def test_fixed_opening_loader_uses_read_only_scores_and_keeps_input_locks_until_commit(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture=CohortFixture(directory);journal=fixture.journals[100];runtime=journal.runtime
            for record in PRE:fixture.archive(100,record);journal.archived[str(record)]=digest(fixture.runs[100]/'weights'/f'{record:012d}'/'manifest.json')
            paths={}
            for record in PRE:
                for opponent in OPPONENTS:
                    fixture.scores(100,record,'greedy_q',opponent)
                    paths[record,opponent]=fixture.cohort.location(100,record,'greedy_q',opponent)
            runtime.backend.counters=NFSPCadence(fresh_records=5000000,fresh_q=5000000,fresh_ordinary=5000000,
                q_updates=5000000//512-8+1,q_loss_targets=(5000000//512-8+1)*512,target_refreshes=5000000//12800)
            before={str(path):hashes(path) for path in paths.values()};state=torch.get_rng_state().clone()
            original=journal._write_decision
            def locked_commit(details):
                for path in paths.values():
                    with (path/'.evaluation.lock').open('rb') as lock:
                        with self.assertRaises(BlockingIOError):fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                return original(details)
            with patch.object(journal,'_write_decision',side_effect=locked_commit):
                decision=journal.record_pilot_opening(fixture.cohort.pilot,paths)
            self.assertEqual(decision['status'],'pass');self.assertEqual(runtime.reservoir.seen,0)
            torch.testing.assert_close(torch.get_rng_state(),state,rtol=0,atol=0)
            self.assertEqual({str(path):hashes(path) for path in paths.values()},before)
            with self.assertRaisesRegex(ValueError,'original fixed pre-window'):
                journal.record_pilot_opening(fixture.cohort.pilot,{(POST[0],'random'):next(iter(paths.values()))})

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA unavailable')
    def test_public_archive_load_uses_cpu_initialization_even_under_default_cuda_context(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture=CohortFixture(directory);configure_runtime('cuda',True)
            cpu=torch.get_rng_state().clone();cuda=[value.clone() for value in torch.cuda.get_rng_state_all()]
            with torch.device('cuda'):
                policy=FrozenNeuralPolicy.from_archive(fixture.runs[100]/'weights'/'000000000000',device='cpu')
            self.assertEqual(policy.device.type,'cpu');torch.testing.assert_close(torch.get_rng_state(),cpu,rtol=0,atol=0)
            for before,after in zip(cuda,torch.cuda.get_rng_state_all()):torch.testing.assert_close(before,after,rtol=0,atol=0)

    def test_resealed_changed_record_zero_weights_cannot_be_registered_as_a_fresh_seed(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture=CohortFixture(directory);archive=fixture.runs[100]/'weights'/'000000000000'
            data=torch.load(archive/'models.pt',weights_only=True,map_location='cpu')
            first=next(iter(data['q']));data['q'][first].view(-1)[0]+=.01;torch.save(data,archive/'models.pt')
            path=archive/'manifest.json';seal=json.loads(path.read_text())
            seal['files']['models.pt']={'bytes':(archive/'models.pt').stat().st_size,'sha256':digest(archive/'models.pt')}
            path.write_text(json.dumps(seal,sort_keys=True,indent=2)+'\n');(archive/'manifest.sha256').write_text(digest(path)+'\n')
            with self.assertRaisesRegex(ValueError,'fresh seeded initialization'):
                DevelopmentCohort.register(Path(directory)/'changed','changed',fixture.runs)

    def test_report_cli_cannot_overwrite_the_frozen_registry_or_training_inputs(self):
        from graygo.cli import main
        with tempfile.TemporaryDirectory() as directory:
            fixture=CohortFixture(directory);before=hashes(directory)
            for output in (fixture.cohort.root/'cohort.json',fixture.runs[100]/'run.json'):
                with redirect_stderr(io.StringIO()),self.assertRaises(SystemExit) as result:
                    main(['cohort-report','--run-dir',str(fixture.cohort.root),'--output',str(output)])
                self.assertEqual(result.exception.code,1)
            self.assertEqual(hashes(directory),before)


if __name__=='__main__':unittest.main()
