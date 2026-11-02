import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from graygo.buffers import ReplayBuffer,ReservoirBuffer
from graygo.nfsp_training import NFSPMatrixConfig,NFSPMatrixTrainer
from graygo.storage import CheckpointStore,digest
from tools.audit_nfsp_matrix import audit

AUDITOR=Path(__file__).resolve().parents[1]/'tools/audit_nfsp_matrix.py'


class MatrixAuditReceiptTests(unittest.TestCase):
    def fixture(self,root,*,task='pure82',learner='tabular',records=254,label_rule='executed',capacity=16):
        c=NFSPMatrixConfig(task=task,learner=learner,label_rule=label_rule,seed=2,records=records,eval_every=32,batch=8,
            learning_start=8,capacity=capacity,target_every=16,hidden=8,reservoir_open=0,handoff_end=32,epsilon_decay=128)
        t=NFSPMatrixTrainer(c,root,reserve_bytes=0);self.assertEqual(t.run(verbose=False)['status'],'complete');return t

    def snapshot(self,root):return {p.relative_to(root):digest(p) for p in Path(root).rglob('*') if p.is_file()}

    def reseal(self,root,mutate):
        path=CheckpointStore(root,reserve_bytes=0).latest();mutate(path)
        manifest=json.loads((path/'manifest.json').read_text())
        for name in manifest['files']:
            p=path/name;manifest['files'][name]={'bytes':p.stat().st_size,'sha256':digest(p)}
        (path/'manifest.json').write_text(json.dumps(manifest,sort_keys=True));(path/'manifest.sha256').write_text(digest(path/'manifest.json')+'\n')
        CheckpointStore.verify(path)

    def test_valid_unpublished_executed_label_tail_recovers_the_actual_publication(self):
        with tempfile.TemporaryDirectory() as directory:
            t=self.fixture(directory);before=self.snapshot(directory)
            with patch.object(NFSPMatrixTrainer,'__init__',side_effect=AssertionError('audit must not initialize a trainer')):
                report=audit(directory,require_formal=False)
            self.assertTrue(report['verified']);self.assertEqual(report['format_version'],2)
            self.assertEqual(report['last_supervised_record'],248);self.assertEqual(report['eligible_labels'],15)
            self.assertEqual(report['published_label_counts_per_role'],[9,5])
            self.assertTrue(report['published_tabular_mean_verified']);self.assertEqual(report['unresolved_published_greedy_labels'],0)
            self.assertEqual(report['retained_executed_label_targets_verified'],15)
            self.assertEqual(report['exact_published_mean_roles'],[1,2]);self.assertFalse(report['formal_protocol']);self.assertFalse(report['a2_complete'])
            x,_,_=t.spec.unpack(t.reservoir.arrays['spatial'][:t.reservoir.size],t.reservoir.arrays['globals'][:t.reservoir.size],t.reservoir.arrays['legal'][:t.reservoir.size])
            labels=np.unpackbits(t.reservoir.arrays['greedy'][:t.reservoir.size],axis=-1,count=t.actions,bitorder='little').astype(float)
            latest=labels[x[:,0]>0].mean(0)
            self.assertGreater(np.max(np.abs(latest-t.backend.average.probabilities[0].numpy())),.05)
            self.assertEqual(before,self.snapshot(directory))

    def test_receipt_binds_every_input_and_all_scheduled_archive_grades(self):
        with tempfile.TemporaryDirectory() as directory:
            self.fixture(directory,task='asymmetric',learner='neural',label_rule='greedy',records=256)
            report=audit(directory,require_formal=False);root=Path(directory)
            self.assertEqual(report['result_sha256'],digest(root/'result.json'));self.assertEqual(report['auditor_sha256'],digest(AUDITOR))
            self.assertEqual(report['checkpoint_manifest_sha256'],digest(root/'resume/000000000256/manifest.json'))
            self.assertEqual(set(report['archive_manifests_sha256']),{str(n) for n in range(32,257,32)})
            for record,sha in report['archive_manifests_sha256'].items():self.assertEqual(sha,digest(root/'weights'/f'{int(record):012d}'/'manifest.json'))
            self.assertEqual([row['records'] for row in report['scheduled_evaluations']],list(range(32,257,32)))
            self.assertGreater(report['archive_boundary_actions_reconstructed'],14)

    def test_executed_labels_cannot_be_replaced_by_another_legal_one_hot_target(self):
        with tempfile.TemporaryDirectory() as directory:
            self.fixture(directory)
            def mutate(path):
                r=ReservoirBuffer.load(path/'reservoir.npz');mask=np.unpackbits(r.arrays['greedy'][0],count=r.spec.actions,bitorder='little')
                action=int(mask.argmax());mask[:]=0;mask[(action+1)%r.spec.actions]=1;r.arrays['greedy'][0]=np.packbits(mask,bitorder='little');r.save(path/'reservoir.npz')
            self.reseal(directory,mutate)
            with self.assertRaises((AssertionError,ValueError)):audit(directory,require_formal=False)

    def test_final_archive_and_final_learner_must_describe_the_same_parameters(self):
        with tempfile.TemporaryDirectory() as directory:
            self.fixture(directory,learner='neural')
            def mutate(path):
                data=torch.load(path/'trainer.pt',weights_only=True,map_location='cpu');key=next(iter(data['backend']['q']))
                data['backend']['q'][key].add_(.1);torch.save(data,path/'trainer.pt')
            self.reseal(directory,mutate)
            with self.assertRaisesRegex(ValueError,'final public parameters'):audit(directory,require_formal=False)

    def test_duplicate_retained_ordinal_is_rejected_even_in_a_valid_container(self):
        with tempfile.TemporaryDirectory() as directory:
            self.fixture(directory)
            def mutate(path):
                r=ReplayBuffer.load(path/'replay.npz');r.arrays['ordinal'][0]=r.arrays['ordinal'][1];r.save(path/'replay.npz')
            self.reseal(directory,mutate)
            with self.assertRaises((AssertionError,ValueError)):audit(directory,require_formal=False)

    def test_noncanonical_reservoir_padding_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            self.fixture(directory)
            def mutate(path):
                r=ReservoirBuffer.load(path/'reservoir.npz');r.arrays['spatial'][0,0]|=128;r.save(path/'reservoir.npz')
            self.reseal(directory,mutate)
            with self.assertRaises((AssertionError,ValueError)):audit(directory,require_formal=False)

    def test_no_supervised_publication_keeps_the_initial_uniform_table(self):
        with tempfile.TemporaryDirectory() as directory:
            self.fixture(directory,task='asymmetric',label_rule='greedy',records=8)
            report=audit(directory,require_formal=False);self.assertIsNone(report['last_supervised_record'])
            self.assertTrue(report['published_tabular_mean_verified']);self.assertEqual(report['published_label_counts_per_role'],[0,0])

    def test_rng_threads_and_ambient_cuda_are_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            self.fixture(directory,task='rps',learner='neural',label_rule='greedy',records=96)
            cpu=torch.get_rng_state().clone();cuda=[s.clone() for s in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else []
            old=torch.get_num_threads()
            try:
                torch.set_num_threads(2)
                if cuda:
                    with torch.device('cuda'):audit(directory,require_formal=False)
                else:audit(directory,require_formal=False)
                self.assertEqual(torch.get_num_threads(),2);self.assertTrue(torch.equal(cpu,torch.get_rng_state()))
                self.assertTrue(all(torch.equal(a,b) for a,b in zip(cuda,torch.cuda.get_rng_state_all())))
            finally:torch.set_num_threads(old)

    def test_writer_lock_and_dependency_mismatch_prevent_reading_stale_evidence(self):
        import fcntl
        with tempfile.TemporaryDirectory() as directory:
            self.fixture(directory);root=Path(directory)
            with (root/'.trainer.lock').open('rb') as lock:
                fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                try:
                    with self.assertRaises(BlockingIOError):audit(root,require_formal=False)
                finally:fcntl.flock(lock,fcntl.LOCK_UN)
            manifest=json.loads((root/'run.json').read_text());manifest['dependencies']['numpy']='0.0';(root/'run.json').write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError,'pinned dependency'):audit(root,require_formal=False)

    def test_cli_outputs_receipt_outside_inputs_and_rejects_overwriting_results(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)/'run';self.fixture(root);before=self.snapshot(root);output=Path(directory)/'audit.json'
            command=[sys.executable,str(AUDITOR),str(root),'--allow-diagnostic','--time-cap-seconds','60','--output',str(output)]
            result=subprocess.run(command,text=True,capture_output=True,timeout=90);self.assertEqual(result.returncode,0,result.stderr)
            self.assertTrue(json.loads(output.read_text())['verified']);self.assertEqual(before,self.snapshot(root))
            command[-1]=str(root/'result.json');result=subprocess.run(command,text=True,capture_output=True,timeout=90)
            self.assertNotEqual(result.returncode,0);self.assertIn('outside',result.stderr);self.assertEqual(before,self.snapshot(root))

    def test_replaced_unpublished_greedy_target_keeps_its_explicit_bounds(self):
        with tempfile.TemporaryDirectory() as directory:
            self.fixture(directory,task='asymmetric',label_rule='greedy',capacity=8)
            report=audit(directory,require_formal=False)
            self.assertEqual(report['last_supervised_record'],248)
            self.assertEqual(report['unresolved_published_greedy_labels'],1)
            self.assertTrue(report['published_tabular_mean_interval_checked']);self.assertFalse(report['published_tabular_mean_verified'])
            self.assertEqual(report['exact_published_mean_roles'],[2])

    def test_validly_resealed_scheduled_clock_drift_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            self.fixture(directory,task='rps',learner='neural',label_rule='greedy')
            def mutate(path):
                data=torch.load(path/'trainer.pt',weights_only=True,map_location='cpu')
                data['evaluations'][-1]['counters']['fresh_q']+=2;torch.save(data,path/'trainer.pt')
            self.reseal(directory,mutate)
            with self.assertRaisesRegex(ValueError,'scheduled own-side clock'):audit(directory,require_formal=False)


if __name__=='__main__':unittest.main()
